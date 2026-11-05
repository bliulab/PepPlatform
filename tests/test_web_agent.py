import asyncio
import json

import pytest

fastapi = pytest.importorskip("fastapi")

from pepplatform.analysis import save_analysis
from pepplatform.web.app import create_app
from pepplatform.web.ai import AiStreamInterruptedError


class _Response:
    def __init__(self):
        self.cookies = {}

    def set_cookie(self, key, value, **_kwargs):
        self.cookies[key] = value

    def delete_cookie(self, key, **_kwargs):
        self.cookies.pop(key, None)


class _Request:
    def __init__(self, cookies=None):
        self.cookies = cookies or {}
        self.headers = {}
        self.client = None


class _FakeAgentAiClient:
    def __init__(self, answer):
        self.answers = list(answer) if isinstance(answer, list) else [answer]
        self.calls = []

    def chat(
        self,
        provider_config,
        api_key,
        messages,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
    ):
        answer = self.answers[min(len(self.calls), len(self.answers) - 1)]
        self.calls.append(
            {
                "provider_config": dict(provider_config),
                "api_key": api_key,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": response_format,
            }
        )
        return {
            "answer": answer,
            "provider": provider_config["provider"],
            "model": provider_config["model"],
            "usage": {"total_tokens": 88},
        }

    def stream_chat(
        self,
        provider_config,
        api_key,
        messages,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
    ):
        answer = self.answers[min(len(self.calls), len(self.answers) - 1)]
        self.calls.append(
            {
                "provider_config": dict(provider_config),
                "api_key": api_key,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "response_format": response_format,
            }
        )
        midpoint = max(1, len(answer) // 2)
        for content in [answer[:midpoint], answer[midpoint:]]:
            if content:
                yield {"type": "delta", "content": content}
        yield {
            "type": "done",
            "provider": provider_config["provider"],
            "model": provider_config["model"],
            "finish_reason": "stop",
            "usage": {"total_tokens": 12},
        }


class _CancelableAgentAiClient(_FakeAgentAiClient):
    def __init__(self, answer):
        super().__init__(answer)
        self.closed = False

    def stream_chat(
        self,
        provider_config,
        api_key,
        messages,
        max_tokens=1200,
        temperature=0.2,
        response_format=None,
    ):
        self.calls.append(
            {
                "provider_config": dict(provider_config),
                "api_key": api_key,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "response_format": response_format,
            }
        )
        try:
            yield {"type": "delta", "content": '{"actions":['}
            yield {"type": "heartbeat"}
        finally:
            self.closed = True


class _FakeWebSearchClient:
    def __init__(self, available=True, reason=""):
        self.available = available
        self.reason = reason
        self.queries = []

    def status(self):
        return {
            "configured": True,
            "available": self.available,
            "reason": self.reason,
            "remaining": 8 if self.available else 0,
            "limit": 1000,
            "resets_on": "2026-08-01",
        }

    def search(self, query):
        self.queries.append(query)
        return {
            "requested": True,
            "status": "complete",
            "query": query,
            "provider": "tavily",
            "credits": 1,
            "reason": "",
            "sources": [
                {
                    "index": 1,
                    "title": "Official current source",
                    "url": "https://example.test/current",
                    "domain": "example.test",
                    "snippet": "Current verified evidence.",
                    "favicon": "https://example.test/favicon.ico",
                }
            ],
        }


def _endpoint(app, path, method):
    for route in app.routes:
        if getattr(route, "path", "") == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"{method} {path} endpoint was not registered")


def _streaming_response_iterator(response):
    frame = getattr(response.body_iterator, "ag_frame", None)
    iterator = frame.f_locals.get("iterator") if frame else None
    if iterator is None:
        raise AssertionError("StreamingResponse did not retain its synchronous iterator")
    return iterator


def _auth_app(tmp_path, monkeypatch, answer):
    monkeypatch.setenv("PEP_AUTH_ENABLED", "1")
    monkeypatch.setenv("PEP_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("PEP_LLM_KEY_ENCRYPTION_SECRET", "test-llm-encryption-secret")
    project_root = tmp_path / "project"
    project_root.mkdir()
    app = create_app(project_root=project_root, runtime_root=tmp_path / "runtime")
    app.state.ai_client = _FakeAgentAiClient(answer)
    return app


def _register_verified(app, email):
    register = _endpoint(app, "/api/auth/register", "POST")
    verify = _endpoint(app, "/api/auth/verify-email", "POST")
    response = register({"email": email, "password": "password123"})
    verify_response = _Response()
    verified = verify({"email": email, "code": response["debug_code"]}, verify_response)
    return verified["user"], verify_response.cookies


def _create_provider(app, cookies):
    create_provider = _endpoint(app, "/api/ai/providers", "POST")
    return create_provider(
        {
            "provider": "deepseek",
            "display_name": "DeepSeek",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key": "sk-agent-123456",
            "enabled": True,
        },
        _Request(cookies),
    )


def _create_sequence_dataset(app, user):
    root = app.state.paths.datasets / f"agent-{user['id']}"
    for split, label in [("train", "1"), ("val", "0")]:
        split_dir = root / split
        split_dir.mkdir(parents=True)
        (split_dir / "seqs.fasta").write_text(f">{split}1\nACDEFG\n", encoding="utf-8")
        (split_dir / "labels.csv").write_text(f"active\n{label}\n", encoding="utf-8")
    return app.state.store.create_dataset(
        "agent demo",
        "pep_function",
        "upload",
        "sequence_dir",
        root,
        metadata={"splits": {"train": 1, "val": 1}},
        owner_user_id=user["id"],
    )


def _create_agent_dataset(app, user, task_type):
    if task_type == "pep_function":
        return _create_sequence_dataset(app, user)
    root = app.state.paths.datasets / f"agent-{task_type}-{user['id']}"
    root.mkdir(parents=True)
    return app.state.store.create_dataset(
        f"agent {task_type}",
        task_type,
        "upload",
        "pair_split_dir",
        root,
        metadata={"splits": {"train": 10, "val": 2}},
        owner_user_id=user["id"],
    )


def _agent_answer(extra_config=None):
    extra_config = extra_config or {}
    return json.dumps(
        {
            "candidates": [
                {
                    "lane": "conservative",
                    "title": "CNN baseline",
                    "rationale": "Keep the current sequence route simple.",
                    "expected_tradeoffs": ["Fast", "Lower capacity"],
                    "objective_alignment": "It preserves a simple baseline for comparison.",
                    "baseline_changes": ["Uses the lightweight CNN path."],
                    "risk_level": "low",
                    "verification_steps": ["Check validation F1 after training."],
                    "config": {
                        "tokenizer": "kmer",
                        "encoder": "index",
                        "model": "cnn",
                        **extra_config,
                    },
                },
                {
                    "lane": "balanced",
                    "title": "One-hot RF",
                    "rationale": "Try a classical model on dense sequence features.",
                    "expected_tradeoffs": ["Reusable inference", "Less sequence context"],
                    "objective_alignment": "It keeps inference reusable while changing the feature family.",
                    "baseline_changes": [
                        "Switches from sequence training to a classical classifier."
                    ],
                    "risk_level": "medium",
                    "verification_steps": ["Compare validation and test F1 against the baseline."],
                    "config": {
                        "tokenizer": "kmer",
                        "encoder": "onehot",
                        "model": "rf",
                        "classifier": "rf",
                    },
                },
                {
                    "lane": "exploratory",
                    "title": "Pretrained MLP",
                    "rationale": "Use pretrained representations for a stronger feature extractor.",
                    "expected_tradeoffs": ["Potentially stronger", "May need model cache"],
                    "objective_alignment": "It explores a higher-capacity representation route.",
                    "baseline_changes": ["Uses a pretrained tokenizer and neural classifier."],
                    "risk_level": "high",
                    "verification_steps": ["Confirm pretrained model source availability."],
                    "config": {
                        "tokenizer": "esm2",
                        "encoder": "index",
                        "model": "mlp",
                        "classifier": "mlp",
                    },
                },
            ]
        }
    )


def _pair_agent_answer():
    payload = json.loads(_agent_answer())
    configs = [
        {"tokenizer": "esm2", "encoder": "ppi_fusion", "model": "mlp"},
        {
            "tokenizer": "kmer",
            "encoder": "tfidf",
            "model": "rf",
            "classifier": "rf",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "classifier",
        },
        {
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "model": "cross_attn",
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
        },
    ]
    for candidate, config in zip(payload["candidates"], configs):
        candidate["config"] = config
    return json.dumps(payload)


def _single_campaign_action_answer():
    return json.dumps({"actions": [json.loads(_agent_answer())["candidates"][0]]})


def _agent_answer_for_strategy(strategy):
    payload = json.loads(_agent_answer())
    risk_bands = {
        "conservative": ["low", "low", "medium"],
        "balanced": ["low", "medium", "high"],
        "aggressive": ["medium", "high", "high"],
    }
    for candidate, risk_level in zip(payload["candidates"], risk_bands[strategy]):
        candidate["risk_level"] = risk_level
    return json.dumps(payload)


def _create_finished_experiment(app, dataset, user, name="finished baseline", f1=0.72):
    root = app.state.paths.experiments / name.replace(" ", "-")
    root.mkdir(parents=True, exist_ok=True)
    experiment = app.state.store.create_experiment(
        name,
        dataset["id"],
        dataset["task_type"],
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        root / "train.log",
        root / "artifact",
        owner_user_id=user["id"],
    )
    return app.state.store.update_experiment(
        experiment["id"],
        status="succeeded",
        metrics_json={"events": [], "summary": {"best_val_f1": f1}},
        finished_at="2026-01-01T00:00:00+00:00",
    )


def _finish_first_created_experiment(app, plan, user, f1=0.82):
    experiment_id = plan["execution"]["created_experiments"][0]["experiment_id"]
    experiment = app.state.store.get_experiment(experiment_id, user=user)
    save_analysis(
        experiment["artifact_dir"],
        {
            "warnings": [],
            "feature_importance": {
                "available": True,
                "items": [{"feature": "kmer:ACD", "importance": 0.42}],
            },
        },
    )
    app.state.store.update_experiment(
        experiment_id,
        status="succeeded",
        metrics_json={"events": [], "summary": {"best_val_f1": f1}},
        finished_at="2026-01-01T00:00:00+00:00",
    )
    return experiment_id


def test_agent_experiment_design_returns_preflighted_candidates_and_drops_unknown_keys(
    tmp_path, monkeypatch
):
    app = _auth_app(
        tmp_path, monkeypatch, [_agent_answer({"unknown_knob": "drop-me"}), _agent_answer()]
    )
    try:
        user, cookies = _register_verified(app, "agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        design = _endpoint(app, "/api/agent/experiment-design", "POST")

        response = design(
            {
                "provider_config_id": provider["id"],
                "task_type": "pep_function",
                "dataset_id": dataset["id"],
                "objective": "Find a stronger baseline.",
                "language": "en",
            },
            _Request(cookies),
        )

        assert response["provider"] == "deepseek"
        assert response["model"] == "deepseek-v4-flash"
        assert len(response["candidates"]) == 3
        assert response["candidates"][0]["valid"] is True
        assert response["candidates"][0]["objective_alignment"]
        assert response["candidates"][0]["risk_level"] == "low"
        assert response["repair_attempted"] is True
        assert response["repair_succeeded"] is True
        assert response["context_summary"]["recent_experiment_count"] == 0
        assert "unknown_knob" not in response["candidates"][0]["config"]
        assert "unknown_knob" not in response["candidates"][0]["preflight"]["normalized_config"]
        assert response["candidates"][0]["preflight"]["normalized_config"]["batch_size"] == 32
        assert response["candidates"][0]["preflight"]["normalized_config"]["lr"] == 1e-3
        assert response["candidates"][0]["preflight"]["normalized_config"]["epochs"] == 30
        assert response["candidates"][0]["preflight"]["parameter_sources"]["epochs"] == "default"
        assert response["candidates"][1]["preflight"]["normalized_config"]["rf_estimators"] == 300
        assert "epochs" not in response["candidates"][1]["preflight"]["normalized_config"]
        assert response["warnings"] == []

        sent_message = "\n".join(
            message["content"] for message in app.state.ai_client.calls[-1]["messages"]
        )
        assert "sk-agent-123456" not in sent_message
        assert "allowed_config_keys" in sent_message
        assert (
            "Every neural candidate must explicitly include batch_size, lr, and epochs"
            in sent_message
        )
        assert all(
            call["response_format"] == {"type": "json_object"} for call in app.state.ai_client.calls
        )

        events = {item["event_type"] for item in app.state.store.list_audit_logs(limit=20)["items"]}
        assert "ai.experiment_design" in events
    finally:
        app.state.runner.stop()


def test_agent_experiment_design_repairs_short_ai_response(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, [json.dumps({"candidates": []}), _agent_answer()])
    try:
        user, cookies = _register_verified(app, "short-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        design = _endpoint(app, "/api/agent/experiment-design", "POST")

        response = design(
            {
                "provider_config_id": provider["id"],
                "dataset_id": dataset["id"],
            },
            _Request(cookies),
        )

        assert len(response["candidates"]) == 3
        assert response["repair_attempted"] is True
        assert response["repair_succeeded"] is True
        assert len(app.state.ai_client.calls) == 2
    finally:
        app.state.runner.stop()


@pytest.mark.parametrize(
    ("task_type", "answer"),
    [
        ("pep_function", _agent_answer()),
        ("pep_protein_interaction", _pair_agent_answer()),
        ("pep_hlaii", _pair_agent_answer()),
        ("pep_hlai", _pair_agent_answer()),
    ],
)
def test_agent_experiment_design_is_fully_preflighted_for_every_task_type(
    tmp_path,
    monkeypatch,
    task_type,
    answer,
):
    app = _auth_app(tmp_path, monkeypatch, answer)
    try:
        user, cookies = _register_verified(app, f"design-{task_type}@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_agent_dataset(app, user, task_type)
        design = _endpoint(app, "/api/agent/experiment-design", "POST")

        response = design(
            {
                "provider_config_id": provider["id"],
                "dataset_id": dataset["id"],
                "task_type": task_type,
            },
            _Request(cookies),
        )

        assert len(response["candidates"]) == 3
        assert all(candidate["valid"] for candidate in response["candidates"])
        assert response["attempts"] == 1
    finally:
        app.state.runner.stop()


def test_agent_experiment_design_repairs_invalid_preflight_candidate(tmp_path, monkeypatch):
    invalid_answer = _agent_answer({"model": "rf", "classifier": "rf"})
    app = _auth_app(tmp_path, monkeypatch, [invalid_answer, _agent_answer()])
    try:
        user, cookies = _register_verified(app, "invalid-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        design = _endpoint(app, "/api/agent/experiment-design", "POST")

        response = design(
            {
                "provider_config_id": provider["id"],
                "dataset_id": dataset["id"],
            },
            _Request(cookies),
        )

        assert response["repair_attempted"] is True
        assert response["repair_succeeded"] is True
        assert response["candidates"][0]["valid"] is True
        repair_prompt = app.state.ai_client.calls[1]["messages"][-1]["content"]
        assert "classical_index_unsupported" in repair_prompt
    finally:
        app.state.runner.stop()


def test_agent_experiment_design_includes_leaderboard_context(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "leaderboard-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        baseline = _create_finished_experiment(app, dataset, user)
        design = _endpoint(app, "/api/agent/experiment-design", "POST")

        response = design(
            {
                "provider_config_id": provider["id"],
                "dataset_id": dataset["id"],
                "baseline_experiment_ids": [baseline["id"]],
            },
            _Request(cookies),
        )

        assert response["context_summary"] == {
            "recent_experiment_count": 1,
            "baseline_experiment_count": 1,
            "leaderboard_count": 1,
        }
        sent_message = app.state.ai_client.calls[-1]["messages"][-1]["content"]
        assert '"leaderboard"' in sent_message
        assert baseline["name"] in sent_message
    finally:
        app.state.runner.stop()


def test_agent_experiment_design_rejects_unparseable_repair_response(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, ["not json", "still not json"])
    try:
        user, cookies = _register_verified(app, "unparseable-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        design = _endpoint(app, "/api/agent/experiment-design", "POST")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            design(
                {
                    "provider_config_id": provider["id"],
                    "dataset_id": dataset["id"],
                },
                _Request(cookies),
            )
        assert exc_info.value.status_code == 502
        assert exc_info.value.detail["code"] == "agent_preflight_exhausted"
        assert exc_info.value.detail["attempts"] == 3
        assert "valid experiment-design JSON" in exc_info.value.detail["issues"][0]
        assert len(app.state.ai_client.calls) == 3
    finally:
        app.state.runner.stop()


def test_agent_experiment_design_is_owner_scoped(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        owner, owner_cookies = _register_verified(app, "agent-owner@example.com")
        _other, other_cookies = _register_verified(app, "agent-other@example.com")
        provider = _create_provider(app, owner_cookies)
        dataset = _create_sequence_dataset(app, owner)
        design = _endpoint(app, "/api/agent/experiment-design", "POST")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            design(
                {
                    "provider_config_id": provider["id"],
                    "dataset_id": dataset["id"],
                },
                _Request(other_cookies),
            )
        assert exc_info.value.status_code == 404
        assert not app.state.ai_client.calls
    finally:
        app.state.runner.stop()


def test_agent_experiment_batch_creates_selected_candidates_and_audits(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "batch-agent@example.com")
        dataset = _create_sequence_dataset(app, user)
        app.state.store.set_user_quota(user["id"], {"daily_concurrent_experiment_limit": 3})
        app.state.runner.stop()
        batch = _endpoint(app, "/api/agent/experiment-batches", "POST")
        http_response = fastapi.Response()

        response = batch(
            {
                "dataset_id": dataset["id"],
                "task_type": "pep_function",
                "objective": "Create two agent experiments.",
                "candidates": [
                    {
                        "id": "conservative-1",
                        "lane": "conservative",
                        "title": "CNN baseline",
                        "config": {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
                    },
                    {
                        "id": "balanced-2",
                        "lane": "balanced",
                        "title": "RF one-hot",
                        "config": {
                            "tokenizer": "kmer",
                            "encoder": "onehot",
                            "model": "rf",
                            "classifier": "rf",
                        },
                    },
                ],
            },
            _Request(cookies),
            http_response,
        )

        assert response["batch_id"]
        assert len(response["created_experiments"]) == 2
        assert (
            response["created_experiments"][0]["experiment"]["name"]
            == "Agent conservative: CNN baseline"
        )
        assert response["created_experiments"][0]["experiment"]["generated_name"] == {
            "kind": "config",
            "qualifier": "conservative",
        }
        assert response["created_experiments"][0]["experiment"]["status"] == "queued"
        assert http_response.headers["deprecation"] == "true"
        assert "Experiment Wizard" in http_response.headers["warning"]
        deprecated_paths = {
            route.path for route in app.routes if getattr(route, "deprecated", False)
        }
        assert "/api/agent/experiment-batches" in deprecated_paths
        assert "/api/agent/execution-plans/{plan_id}/execute" in deprecated_paths
        assert app.state.store.get_quota_usage(user["id"])["experiments_created"] == 2
        events = {item["event_type"] for item in app.state.store.list_audit_logs(limit=20)["items"]}
        assert "ai.experiment_batch_create" in events
    finally:
        app.state.runner.stop()


def test_agent_experiment_batch_rejects_invalid_candidate_without_partial_create(
    tmp_path, monkeypatch
):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "batch-invalid-agent@example.com")
        dataset = _create_sequence_dataset(app, user)
        app.state.runner.stop()
        batch = _endpoint(app, "/api/agent/experiment-batches", "POST")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            batch(
                {
                    "dataset_id": dataset["id"],
                    "candidates": [
                        {
                            "id": "valid-1",
                            "lane": "conservative",
                            "title": "Valid CNN",
                            "config": {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
                        },
                        {
                            "id": "invalid-2",
                            "lane": "balanced",
                            "title": "Invalid RF index",
                            "config": {
                                "tokenizer": "kmer",
                                "encoder": "index",
                                "model": "rf",
                                "classifier": "rf",
                            },
                        },
                    ],
                },
                _Request(cookies),
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == "agent_batch_preflight_failed"
        assert (
            "classical_index_unsupported"
            in exc_info.value.detail["candidates"][0]["preflight"]["error_codes"]
        )
        assert app.state.store.list_experiments(user=user) == []
    finally:
        app.state.runner.stop()


def test_agent_experiment_batch_is_owner_scoped(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        owner, _owner_cookies = _register_verified(app, "batch-owner@example.com")
        _other, other_cookies = _register_verified(app, "batch-other@example.com")
        dataset = _create_sequence_dataset(app, owner)
        app.state.runner.stop()
        batch = _endpoint(app, "/api/agent/experiment-batches", "POST")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            batch(
                {
                    "dataset_id": dataset["id"],
                    "candidates": [
                        {
                            "lane": "conservative",
                            "title": "CNN",
                            "config": {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
                        }
                    ],
                },
                _Request(other_cookies),
            )
        assert exc_info.value.status_code == 403
        assert app.state.store.list_experiments(user=owner) == []
    finally:
        app.state.runner.stop()


def test_agent_experiment_batch_checks_quota_for_requested_count(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "batch-quota-agent@example.com")
        dataset = _create_sequence_dataset(app, user)
        app.state.store.set_user_quota(user["id"], {"daily_experiment_limit": 1})
        app.state.runner.stop()
        batch = _endpoint(app, "/api/agent/experiment-batches", "POST")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            batch(
                {
                    "dataset_id": dataset["id"],
                    "candidates": [
                        {
                            "lane": "conservative",
                            "title": "CNN",
                            "config": {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
                        },
                        {
                            "lane": "balanced",
                            "title": "One-hot RF",
                            "config": {
                                "tokenizer": "kmer",
                                "encoder": "onehot",
                                "model": "rf",
                                "classifier": "rf",
                            },
                        },
                    ],
                },
                _Request(cookies),
            )
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["quota_type"] == "daily_experiment_limit"
        assert exc_info.value.detail["requested"] == 2
        assert app.state.store.list_experiments(user=user) == []
    finally:
        app.state.runner.stop()


def _review_answer():
    return json.dumps(
        {
            "summary": "The approved plan created queued experiments.",
            "evidence": ["Two experiments are queued."],
            "action_outcomes": [
                {
                    "action_id": "create-conservative-1",
                    "experiment_id": "experiment-1",
                    "assessment": "Ready to monitor.",
                }
            ],
            "next_recommendations": ["Wait for training metrics before selecting the next route."],
            "risks": ["No completed metrics are available yet."],
        }
    )


def test_agent_execution_plan_creates_persisted_preflighted_actions(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "plan-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")

        plan = create_plan(
            {
                "provider_config_id": provider["id"],
                "dataset_id": dataset["id"],
                "task_type": "pep_function",
                "objective": "Plan approved experiments.",
                "language": "en",
            },
            _Request(cookies),
        )

        assert plan["id"]
        assert plan["status"] == "draft"
        assert len(plan["actions"]) == 3
        assert plan["actions"][0]["type"] == "create_experiment"
        assert plan["actions"][0]["preflight"]["valid"] is True
        assert plan["actions"][0]["status"] == "pending"
        assert plan["repair_attempted"] is False
        assert plan["trace"]["strategy_preference"] == "balanced"
        assert plan["trace"]["attempts"] == 1
        assert all(action["valid"] for action in plan["actions"])
        assert "no_test_split" in plan["actions"][0]["preflight"]["warning_codes"]
        stored = app.state.store.get_agent_execution_plan(plan["id"], user=user)
        assert stored["actions"][0]["id"] == plan["actions"][0]["id"]
        assert app.state.ai_client.calls[-1]["response_format"] == {"type": "json_object"}
        events = {item["event_type"] for item in app.state.store.list_audit_logs(limit=20)["items"]}
        assert "ai.agent_plan_created" in events
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_stream_emits_stages_and_persists_only_the_final_plan(
    tmp_path, monkeypatch
):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "plan-stream-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        stream_plan = _endpoint(app, "/api/agent/execution-plans/stream", "POST")

        response = stream_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"], "language": "en"},
            _Request(cookies),
        )
        body = _consume_streaming_response(response)

        assert '"stage":"generating"' in body
        assert '"stage":"validating"' in body
        assert '"stage":"saving"' in body
        assert "event: done" in body
        plans, total = app.state.store.list_agent_execution_plans(user=user)
        assert total == 1
        assert plans[0]["status"] == "draft"
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_stream_close_stops_upstream_without_persisting(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    cancelable_client = _CancelableAgentAiClient(_agent_answer())
    app.state.ai_client = cancelable_client
    try:
        user, cookies = _register_verified(app, "plan-stream-stop-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        stream_plan = _endpoint(app, "/api/agent/execution-plans/stream", "POST")

        response = stream_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"], "language": "en"},
            _Request(cookies),
        )
        iterator = _streaming_response_iterator(response)
        assert "event: stage" in next(iterator)
        assert next(iterator) == ": ping\n\n"
        iterator.close()

        assert cancelable_client.closed is True
        assert app.state.store.list_agent_execution_plans(user=user)[1] == 0
        audit_items = app.state.store.list_audit_logs(event_type="ai.agent_plan_created", limit=20)[
            "items"
        ]
        assert any(item["metadata"].get("status") == "stopped" for item in audit_items)
    finally:
        app.state.runner.stop()


@pytest.mark.parametrize(
    ("task_type", "answer"),
    [
        ("pep_function", _agent_answer()),
        ("pep_protein_interaction", _pair_agent_answer()),
        ("pep_hlaii", _pair_agent_answer()),
        ("pep_hlai", _pair_agent_answer()),
    ],
)
def test_agent_execution_plan_is_fully_preflighted_for_every_task_type(
    tmp_path,
    monkeypatch,
    task_type,
    answer,
):
    app = _auth_app(tmp_path, monkeypatch, answer)
    try:
        user, cookies = _register_verified(app, f"plan-{task_type}@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_agent_dataset(app, user, task_type)
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")

        plan = create_plan(
            {
                "provider_config_id": provider["id"],
                "dataset_id": dataset["id"],
                "task_type": task_type,
            },
            _Request(cookies),
        )

        assert len(plan["actions"]) == 3
        assert all(action["valid"] and action["preflight"]["valid"] for action in plan["actions"])
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_repairs_invalid_response_before_persisting(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, [json.dumps({"actions": []}), _agent_answer()])
    try:
        user, cookies = _register_verified(app, "plan-repair@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")

        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )

        assert plan["repair_attempted"] is True
        assert plan["repair_succeeded"] is True
        assert plan["trace"]["attempts"] == 2
        assert all(action["valid"] for action in plan["actions"])
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_exhaustion_returns_502_without_persisting(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, json.dumps({"actions": []}))
    try:
        user, cookies = _register_verified(app, "plan-exhausted@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            create_plan(
                {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
                _Request(cookies),
            )

        assert exc_info.value.status_code == 502
        assert exc_info.value.detail["code"] == "agent_preflight_exhausted"
        assert exc_info.value.detail["attempts"] == 3
        assert app.state.store.list_agent_execution_plans(user=user)[0] == []
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_enforces_strategy_risk_bands(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer_for_strategy("aggressive"))
    try:
        user, cookies = _register_verified(app, "plan-aggressive@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")

        plan = create_plan(
            {
                "provider_config_id": provider["id"],
                "dataset_id": dataset["id"],
                "strategy_preference": "aggressive",
            },
            _Request(cookies),
        )

        assert [action["risk_level"] for action in plan["actions"]] == ["medium", "high", "high"]
        assert plan["trace"]["strategy_preference"] == "aggressive"
        sent_prompt = "\n".join(
            message["content"] for message in app.state.ai_client.calls[0]["messages"]
        )
        assert "conservative=medium, balanced=high, exploratory=high" in sent_prompt
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_rejects_unknown_strategy_preference(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "plan-strategy-invalid@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            create_plan(
                {
                    "provider_config_id": provider["id"],
                    "dataset_id": dataset["id"],
                    "strategy_preference": "reckless",
                },
                _Request(cookies),
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == "agent_strategy_preference_invalid"
        assert app.state.ai_client.calls == []
    finally:
        app.state.runner.stop()


def test_agent_campaign_round_accepts_one_fully_valid_action(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _single_campaign_action_answer())
    try:
        user, cookies = _register_verified(app, "campaign-one-action@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        create_campaign = _endpoint(app, "/api/agent/campaigns", "POST")
        create_round = _endpoint(app, "/api/agent/campaigns/{campaign_id}/rounds", "POST")
        campaign = create_campaign(
            {
                "dataset_id": dataset["id"],
                "task_type": "pep_function",
                "objective": "Iterate safely.",
                "max_rounds": 2,
            },
            _Request(cookies),
        )

        plan = create_round(
            campaign["id"],
            {"provider_config_id": provider["id"]},
            _Request(cookies),
        )

        assert len(plan["actions"]) == 1
        assert plan["actions"][0]["valid"] is True
        assert plan["trace"]["attempts"] == 1
    finally:
        app.state.runner.stop()


def test_agent_campaign_round_stream_persists_and_advances_after_done(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _single_campaign_action_answer())
    try:
        user, cookies = _register_verified(app, "campaign-stream-round@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        create_campaign = _endpoint(app, "/api/agent/campaigns", "POST")
        stream_round = _endpoint(app, "/api/agent/campaigns/{campaign_id}/rounds/stream", "POST")
        campaign = create_campaign(
            {
                "dataset_id": dataset["id"],
                "task_type": "pep_function",
                "objective": "Iterate with cancellation support.",
                "max_rounds": 2,
            },
            _Request(cookies),
        )

        response = stream_round(
            campaign["id"],
            {"provider_config_id": provider["id"]},
            _Request(cookies),
        )
        body = _consume_streaming_response(response)
        updated_campaign = app.state.store.get_agent_campaign(
            campaign["id"], user=user, include_plans=True
        )

        assert "event: done" in body
        assert updated_campaign["current_round_index"] == 1
        assert len(updated_campaign["plans"]) == 1
    finally:
        app.state.runner.stop()


def test_agent_campaign_rounds_require_review_and_include_learning_context(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, [_agent_answer(), _review_answer(), _agent_answer()])
    try:
        user, cookies = _register_verified(app, "campaign-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.store.set_user_quota(user["id"], {"daily_concurrent_experiment_limit": 3})
        app.state.runner.stop()
        create_campaign = _endpoint(app, "/api/agent/campaigns", "POST")
        create_round = _endpoint(app, "/api/agent/campaigns/{campaign_id}/rounds", "POST")
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")
        review_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/review", "POST")
        get_campaign = _endpoint(app, "/api/agent/campaigns/{campaign_id}", "GET")

        campaign = create_campaign(
            {
                "dataset_id": dataset["id"],
                "task_type": "pep_function",
                "objective": "Iterate toward stronger reusable inference.",
                "max_rounds": 3,
            },
            _Request(cookies),
        )
        first = create_round(
            campaign["id"],
            {"provider_config_id": provider["id"], "language": "en"},
            _Request(cookies),
        )
        executed = execute_plan(
            first["id"],
            {"approved_action_ids": [first["actions"][0]["id"]]},
            _Request(cookies),
        )
        _finish_first_created_experiment(app, executed, user)
        reviewed = review_plan(
            executed["id"],
            {"provider_config_id": provider["id"], "language": "en"},
            _Request(cookies),
        )
        second = create_round(
            campaign["id"],
            {"provider_config_id": provider["id"], "language": "en"},
            _Request(cookies),
        )

        assert first["campaign_id"] == campaign["id"]
        assert first["round_index"] == 1
        assert reviewed["status"] == "reported"
        assert second["round_index"] == 2
        assert second["context_summary"]["previous_round_count"] == 1
        assert second["context_summary"]["review_count"] == 1
        assert second["context_summary"]["recommendation_count"] == 1
        assert second["context_summary"]["avoided_config_count"] == 3
        sent_message = app.state.ai_client.calls[-1]["messages"][-1]["content"]
        assert '"campaign"' in sent_message
        assert first["id"] in sent_message
        assert "The approved plan created queued experiments." in sent_message
        assert "Wait for training metrics before selecting the next route." in sent_message
        assert '"do_not_repeat_configs"' in sent_message
        detail = get_campaign(campaign["id"], _Request(cookies))
        assert len(detail["plans"]) == 2
        assert detail["current_round_index"] == 2
    finally:
        app.state.runner.stop()


def test_agent_campaign_next_round_rejects_unfinished_previous_round(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, [_agent_answer(), _agent_answer()])
    try:
        user, cookies = _register_verified(app, "campaign-unfinished-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.store.set_user_quota(user["id"], {"daily_concurrent_experiment_limit": 3})
        app.state.runner.stop()
        create_campaign = _endpoint(app, "/api/agent/campaigns", "POST")
        create_round = _endpoint(app, "/api/agent/campaigns/{campaign_id}/rounds", "POST")
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")

        campaign = create_campaign(
            {
                "dataset_id": dataset["id"],
                "task_type": "pep_function",
                "objective": "Iterate toward stronger reusable inference.",
                "max_rounds": 3,
            },
            _Request(cookies),
        )
        first = create_round(
            campaign["id"],
            {"provider_config_id": provider["id"], "language": "en"},
            _Request(cookies),
        )
        execute_plan(
            first["id"], {"approved_action_ids": [first["actions"][0]["id"]]}, _Request(cookies)
        )

        with pytest.raises(fastapi.HTTPException) as exc_info:
            create_round(
                campaign["id"],
                {"provider_config_id": provider["id"], "language": "en"},
                _Request(cookies),
            )

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "agent_campaign_previous_round_unfinished"
        assert len(app.state.ai_client.calls) == 1
    finally:
        app.state.runner.stop()


def test_agent_campaign_next_round_requires_previous_review(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, [_agent_answer(), _agent_answer()])
    try:
        user, cookies = _register_verified(app, "campaign-needs-review-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.store.set_user_quota(user["id"], {"daily_concurrent_experiment_limit": 3})
        app.state.runner.stop()
        create_campaign = _endpoint(app, "/api/agent/campaigns", "POST")
        create_round = _endpoint(app, "/api/agent/campaigns/{campaign_id}/rounds", "POST")
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")

        campaign = create_campaign(
            {
                "dataset_id": dataset["id"],
                "task_type": "pep_function",
                "objective": "Iterate toward stronger reusable inference.",
                "max_rounds": 3,
            },
            _Request(cookies),
        )
        first = create_round(
            campaign["id"],
            {"provider_config_id": provider["id"], "language": "en"},
            _Request(cookies),
        )
        executed = execute_plan(
            first["id"], {"approved_action_ids": [first["actions"][0]["id"]]}, _Request(cookies)
        )
        _finish_first_created_experiment(app, executed, user)

        with pytest.raises(fastapi.HTTPException) as exc_info:
            create_round(
                campaign["id"],
                {"provider_config_id": provider["id"], "language": "en"},
                _Request(cookies),
            )

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "agent_campaign_previous_round_needs_review"
        assert len(app.state.ai_client.calls) == 1
    finally:
        app.state.runner.stop()


def test_agent_campaign_completes_only_after_final_round_review(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, [_agent_answer(), _review_answer()])
    try:
        user, cookies = _register_verified(app, "campaign-complete-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.store.set_user_quota(user["id"], {"daily_concurrent_experiment_limit": 3})
        app.state.runner.stop()
        create_campaign = _endpoint(app, "/api/agent/campaigns", "POST")
        create_round = _endpoint(app, "/api/agent/campaigns/{campaign_id}/rounds", "POST")
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")
        review_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/review", "POST")
        get_campaign = _endpoint(app, "/api/agent/campaigns/{campaign_id}", "GET")

        campaign = create_campaign(
            {
                "dataset_id": dataset["id"],
                "task_type": "pep_function",
                "objective": "Run one closed-loop round.",
                "max_rounds": 1,
            },
            _Request(cookies),
        )
        first = create_round(
            campaign["id"],
            {"provider_config_id": provider["id"], "language": "en"},
            _Request(cookies),
        )
        after_draft = get_campaign(campaign["id"], _Request(cookies))
        assert after_draft["status"] == "active"
        assert after_draft["current_round_index"] == 1

        executed = execute_plan(
            first["id"], {"approved_action_ids": [first["actions"][0]["id"]]}, _Request(cookies)
        )
        _finish_first_created_experiment(app, executed, user)
        review_plan(
            executed["id"],
            {"provider_config_id": provider["id"], "language": "en"},
            _Request(cookies),
        )

        completed = get_campaign(campaign["id"], _Request(cookies))
        assert completed["status"] == "completed"
        assert completed["current_round_index"] == 1
    finally:
        app.state.runner.stop()


def test_agent_execution_action_edit_repreflights_and_blocks_invalid_execution(
    tmp_path, monkeypatch
):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "plan-edit-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        update_action = _endpoint(
            app, "/api/agent/execution-plans/{plan_id}/actions/{action_id}", "PATCH"
        )
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")
        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )
        action = plan["actions"][0]

        edited = update_action(
            plan["id"],
            action["id"],
            {
                "config": {
                    "tokenizer": "kmer",
                    "encoder": "index",
                    "model": "rf",
                    "classifier": "rf",
                }
            },
            _Request(cookies),
        )

        edited_action = next(item for item in edited["actions"] if item["id"] == action["id"])
        assert edited_action["edited"] is True
        assert edited_action["valid"] is False
        assert "classical_index_unsupported" in edited_action["preflight"]["error_codes"]
        assert edited["trace"]["edited_action_count"] == 1

        with pytest.raises(fastapi.HTTPException) as exc_info:
            execute_plan(plan["id"], {"approved_action_ids": [action["id"]]}, _Request(cookies))
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == "agent_plan_action_validation_failed"
        assert app.state.store.list_experiments(user=user) == []
    finally:
        app.state.runner.stop()


def test_agent_execution_action_confirmation_creates_and_links_one_experiment(
    tmp_path, monkeypatch
):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "plan-confirm-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.store.set_user_quota(user["id"], {"daily_concurrent_experiment_limit": 3})
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        confirm_action = _endpoint(
            app,
            "/api/agent/execution-plans/{plan_id}/actions/{action_id}/confirm",
            "POST",
        )
        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )
        action = plan["actions"][0]

        experiment = confirm_action(
            plan["id"],
            action["id"],
            {
                "name": "Reviewed CNN baseline",
                "task_type": plan["task_type"],
                "dataset_id": plan["dataset_id"],
                "config": {**action["config"], "epochs": 12},
                "generated_name": {"kind": "config"},
            },
            _Request(cookies),
        )

        stored = app.state.store.get_agent_execution_plan(plan["id"], user=user)
        stored_action = next(item for item in stored["actions"] if item["id"] == action["id"])
        assert experiment["name"] == "Reviewed CNN baseline"
        assert experiment["generated_name"] == {
            "kind": "config",
            "qualifier": "conservative",
        }
        assert experiment["config"]["epochs"] == 12
        assert stored["status"] == "running"
        assert stored_action["status"] == "executed"
        assert stored_action["edited"] is True
        assert stored_action["result"] == {
            "experiment_id": experiment["id"],
            "status": "queued",
            "confirmed_via": "experiment_wizard",
        }
        assert stored["execution"]["created_experiments"][0]["experiment_id"] == experiment["id"]
        assert any(
            item["type"] == "cancel_experiment" and item["source_action_id"] == action["id"]
            for item in stored["actions"]
        )
        assert app.state.store.get_quota_usage(user["id"])["experiments_created"] == 1
        events = {item["event_type"] for item in app.state.store.list_audit_logs(limit=30)["items"]}
        assert "ai.agent_action_confirmed" in events
    finally:
        app.state.runner.stop()


def test_agent_execution_action_confirmation_repreflights_and_has_no_invalid_side_effects(
    tmp_path, monkeypatch
):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "plan-confirm-invalid@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        confirm_action = _endpoint(
            app,
            "/api/agent/execution-plans/{plan_id}/actions/{action_id}/confirm",
            "POST",
        )
        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )
        action = plan["actions"][0]

        with pytest.raises(fastapi.HTTPException) as exc_info:
            confirm_action(
                plan["id"],
                action["id"],
                {
                    "task_type": plan["task_type"],
                    "dataset_id": plan["dataset_id"],
                    "config": {
                        "tokenizer": "kmer",
                        "encoder": "index",
                        "model": "rf",
                        "classifier": "rf",
                    },
                },
                _Request(cookies),
            )

        stored = app.state.store.get_agent_execution_plan(plan["id"], user=user)
        stored_action = next(item for item in stored["actions"] if item["id"] == action["id"])
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["code"] == "agent_action_preflight_failed"
        assert "classical_index_unsupported" in exc_info.value.detail["preflight"]["error_codes"]
        assert stored_action["status"] == "pending"
        assert stored["execution"] == {}
        assert app.state.store.list_experiments(user=user) == []
        assert app.state.store.get_quota_usage(user["id"])["experiments_created"] == 0
    finally:
        app.state.runner.stop()


def test_agent_execution_action_confirmation_is_idempotent_and_context_bound(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "plan-confirm-idempotent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.store.set_user_quota(user["id"], {"daily_concurrent_experiment_limit": 3})
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        confirm_action = _endpoint(
            app,
            "/api/agent/execution-plans/{plan_id}/actions/{action_id}/confirm",
            "POST",
        )
        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )
        action = plan["actions"][0]
        payload = {
            "name": "Confirmed once",
            "task_type": plan["task_type"],
            "dataset_id": plan["dataset_id"],
            "config": action["config"],
        }

        with pytest.raises(fastapi.HTTPException) as exc_info:
            confirm_action(
                plan["id"],
                action["id"],
                {**payload, "dataset_id": "different-dataset"},
                _Request(cookies),
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "agent_action_context_mismatch"

        first = confirm_action(plan["id"], action["id"], payload, _Request(cookies))
        second = confirm_action(plan["id"], action["id"], payload, _Request(cookies))

        assert second["id"] == first["id"]
        assert len(app.state.store.list_experiments(user=user)) == 1
        assert app.state.store.get_quota_usage(user["id"])["experiments_created"] == 1
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_executes_selected_create_actions_and_appends_cancel_actions(
    tmp_path, monkeypatch
):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "plan-execute-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.store.set_user_quota(user["id"], {"daily_concurrent_experiment_limit": 3})
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")
        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )
        approved_ids = [action["id"] for action in plan["actions"][:2]]

        executed = execute_plan(
            plan["id"], {"approved_action_ids": approved_ids}, _Request(cookies)
        )

        assert executed["status"] == "running"
        assert executed["execution"]["progress"]["queued"] == 2
        assert executed["execution"]["report_ready"] is False
        assert len(executed["execution"]["created_experiments"]) == 2
        assert app.state.store.get_quota_usage(user["id"])["experiments_created"] == 2
        assert all(
            action_id in executed["execution"]["executed_action_ids"] for action_id in approved_ids
        )
        cancel_actions = [
            action for action in executed["actions"] if action["type"] == "cancel_experiment"
        ]
        assert len(cancel_actions) == 2
        assert cancel_actions[0]["status"] == "pending"
        stored_experiments = app.state.store.list_experiments(user=user)
        assert len(stored_experiments) == 2
        events = {item["event_type"] for item in app.state.store.list_audit_logs(limit=30)["items"]}
        assert "ai.agent_plan_executed" in events
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_rejects_create_actions_over_concurrent_quota(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "plan-concurrent-quota-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")
        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )

        with pytest.raises(fastapi.HTTPException) as exc_info:
            execute_plan(
                plan["id"],
                {"approved_action_ids": [action["id"] for action in plan["actions"]]},
                _Request(cookies),
            )

        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["quota_type"] == "daily_concurrent_experiment_limit"
        assert exc_info.value.detail["limit"] == 1
        assert exc_info.value.detail["used"] == 0
        assert exc_info.value.detail["requested"] == 3
        assert app.state.store.list_experiments(user=user) == []
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_cancel_action_only_targets_plan_created_experiment(
    tmp_path, monkeypatch
):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "plan-cancel-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")
        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )
        created = execute_plan(
            plan["id"],
            {"approved_action_ids": [plan["actions"][0]["id"]]},
            _Request(cookies),
        )
        cancel_action = next(
            action for action in created["actions"] if action["type"] == "cancel_experiment"
        )

        canceled = execute_plan(
            plan["id"], {"approved_action_ids": [cancel_action["id"]]}, _Request(cookies)
        )

        experiment_id = created["execution"]["created_experiments"][0]["experiment"]["id"]
        assert app.state.store.get_experiment(experiment_id, user=user)["status"] == "canceled"
        assert canceled["execution"]["canceled_experiments"][0]["experiment_id"] == experiment_id
        assert (
            next(action for action in canceled["actions"] if action["id"] == cancel_action["id"])[
                "status"
            ]
            == "executed"
        )
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_refresh_marks_report_ready_after_experiments_finish(
    tmp_path, monkeypatch
):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "plan-refresh-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")
        refresh_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/refresh", "POST")
        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )
        executed = execute_plan(
            plan["id"],
            {"approved_action_ids": [plan["actions"][0]["id"]]},
            _Request(cookies),
        )
        experiment_id = executed["execution"]["created_experiments"][0]["experiment_id"]
        app.state.store.update_experiment(
            experiment_id,
            status="succeeded",
            metrics_json={"events": [], "summary": {"best_val_f1": 0.81}},
            finished_at="2026-01-01T00:00:00+00:00",
        )

        refreshed = refresh_plan(plan["id"], _Request(cookies))

        assert refreshed["status"] == "ready_for_report"
        assert refreshed["execution"]["progress"]["succeeded"] == 1
        assert refreshed["execution"]["report_ready"] is True
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_review_rejects_unfinished_experiments(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, [_agent_answer(), _review_answer()])
    try:
        user, cookies = _register_verified(app, "plan-unfinished-review-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")
        review_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/review", "POST")
        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )
        executed = execute_plan(
            plan["id"],
            {"approved_action_ids": [plan["actions"][0]["id"]]},
            _Request(cookies),
        )

        with pytest.raises(fastapi.HTTPException) as exc_info:
            review_plan(
                executed["id"],
                {"provider_config_id": provider["id"], "language": "en"},
                _Request(cookies),
            )

        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "agent_plan_experiments_unfinished"
        assert len(app.state.ai_client.calls) == 1
    finally:
        app.state.runner.stop()


def test_agent_execution_plan_review_stores_structured_review(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, [_agent_answer(), _review_answer()])
    try:
        user, cookies = _register_verified(app, "plan-review-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")
        review_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/review/stream", "POST")
        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )
        executed = execute_plan(
            plan["id"],
            {"approved_action_ids": [plan["actions"][0]["id"]]},
            _Request(cookies),
        )
        experiment_id = executed["execution"]["created_experiments"][0]["experiment_id"]
        experiment = app.state.store.get_experiment(experiment_id, user=user)
        save_analysis(
            experiment["artifact_dir"],
            {
                "warnings": ["analysis warning"],
                "feature_importance": {
                    "available": True,
                    "items": [{"feature": "kmer:ACD", "importance": 0.42}],
                },
            },
        )
        app.state.store.update_experiment(
            experiment_id,
            status="succeeded",
            metrics_json={"events": [], "summary": {"best_val_f1": 0.82}},
            finished_at="2026-01-01T00:00:00+00:00",
        )

        response = review_plan(
            executed["id"],
            {"provider_config_id": provider["id"], "language": "en"},
            _Request(cookies),
        )
        body = _consume_streaming_response(response)
        reviewed = app.state.store.get_agent_execution_plan(executed["id"], user=user)

        assert '"stage":"generating"' in body
        assert '"stage":"saving"' in body
        assert "event: done" in body
        assert reviewed["status"] == "reported"
        assert reviewed["review"]["summary"] == "The approved plan created queued experiments."
        assert reviewed["reviewed_at"]
        sent_message = app.state.ai_client.calls[-1]["messages"][-1]["content"]
        assert '"created_experiments"' in sent_message
        assert '"analysis"' in sent_message
        assert "sk-agent-123456" not in sent_message
        events = {item["event_type"] for item in app.state.store.list_audit_logs(limit=40)["items"]}
        assert "ai.agent_plan_reviewed" in events
    finally:
        app.state.runner.stop()


def test_list_agent_execution_plans_returns_summaries_and_filters(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "list-agent@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        list_plans = _endpoint(app, "/api/agent/execution-plans", "GET")

        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )

        response = list_plans(_Request(cookies))

        assert response["total"] == 1
        assert response["limit"] == 20
        assert response["offset"] == 0
        summary = response["plans"][0]
        assert summary["id"] == plan["id"]
        assert summary["status"] == "draft"
        assert summary["task_type"] == "pep_function"
        assert summary["dataset_id"] == dataset["id"]
        assert summary["action_count"] == 3
        assert summary["created_experiment_count"] == 0
        assert summary["canceled_experiment_count"] == 0
        assert summary["has_review"] is False
        assert "actions" not in summary
        assert "execution" not in summary

        filtered = list_plans(
            _Request(cookies),
            task_type="pep_function",
            status="draft",
            limit="10",
            offset="0",
        )
        assert filtered["total"] == 1
        assert filtered["plans"][0]["id"] == plan["id"]

        empty = list_plans(_Request(cookies), status="reported")
        assert empty["total"] == 0
        assert empty["plans"] == []
    finally:
        app.state.runner.stop()


def test_list_agent_execution_plans_is_owner_scoped(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        owner, owner_cookies = _register_verified(app, "list-owner@example.com")
        _other, other_cookies = _register_verified(app, "list-other@example.com")
        provider = _create_provider(app, owner_cookies)
        dataset = _create_sequence_dataset(app, owner)
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        list_plans = _endpoint(app, "/api/agent/execution-plans", "GET")

        create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(owner_cookies),
        )

        owner_response = list_plans(_Request(owner_cookies))
        other_response = list_plans(_Request(other_cookies))

        assert owner_response["total"] == 1
        assert other_response["total"] == 0
    finally:
        app.state.runner.stop()


def test_delete_agent_execution_plan_succeeds_for_owner(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "delete-owner@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        delete_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}", "DELETE")
        get_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}", "GET")

        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )

        result = delete_plan(plan["id"], _Request(cookies))
        assert result == {"plan_id": plan["id"], "deleted": True}

        with pytest.raises(fastapi.HTTPException) as exc_info:
            get_plan(plan["id"], _Request(cookies))
        assert exc_info.value.status_code == 404

        events = {item["event_type"] for item in app.state.store.list_audit_logs(limit=20)["items"]}
        assert "ai.agent_plan_deleted" in events
    finally:
        app.state.runner.stop()


def test_delete_agent_execution_plan_rejects_non_owner(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        owner, owner_cookies = _register_verified(app, "delete-owner-reject@example.com")
        _other, other_cookies = _register_verified(app, "delete-other@example.com")
        provider = _create_provider(app, owner_cookies)
        dataset = _create_sequence_dataset(app, owner)
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        delete_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}", "DELETE")

        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(owner_cookies),
        )

        with pytest.raises(fastapi.HTTPException) as exc_info:
            delete_plan(plan["id"], _Request(other_cookies))
        assert exc_info.value.status_code == 404
    finally:
        app.state.runner.stop()


def test_delete_agent_execution_plan_rejects_unfinished_running(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, _agent_answer())
    try:
        user, cookies = _register_verified(app, "delete-running@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        app.state.store.set_user_quota(user["id"], {"daily_concurrent_experiment_limit": 3})
        app.state.runner.stop()
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")
        execute_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}/execute", "POST")
        delete_plan = _endpoint(app, "/api/agent/execution-plans/{plan_id}", "DELETE")

        plan = create_plan(
            {"provider_config_id": provider["id"], "dataset_id": dataset["id"]},
            _Request(cookies),
        )
        executed = execute_plan(
            plan["id"],
            {"approved_action_ids": [plan["actions"][0]["id"]]},
            _Request(cookies),
        )

        with pytest.raises(fastapi.HTTPException) as exc_info:
            delete_plan(executed["id"], _Request(cookies))
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "agent_plan_has_unfinished_experiments"
    finally:
        app.state.runner.stop()


def test_legacy_agent_chat_returns_normalized_answer(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, "The normalized answer is visible.")
    try:
        _user, cookies = _register_verified(app, "legacy-chat@example.com")
        provider = _create_provider(app, cookies)
        chat_endpoint = _endpoint(app, "/api/agent/chat", "POST")

        response = chat_endpoint(
            {
                "provider_config_id": provider["id"],
                "messages": [{"role": "user", "content": "Why was the old reply blank?"}],
                "language": "en",
            },
            _Request(cookies),
        )

        assert response["reply"] == "The normalized answer is visible."
        assert response["provider"] == "deepseek"
        assert response["model"] == "deepseek-v4-flash"
    finally:
        app.state.runner.stop()


def test_legacy_agent_chat_can_search_and_returns_sources(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, "The current answer is supported [1].")
    app.state.web_search_client = _FakeWebSearchClient()
    try:
        _user, cookies = _register_verified(app, "legacy-search-chat@example.com")
        provider = _create_provider(app, cookies)
        chat_endpoint = _endpoint(app, "/api/agent/chat", "POST")

        response = chat_endpoint(
            {
                "provider_config_id": provider["id"],
                "messages": [{"role": "user", "content": "Find the current answer"}],
                "language": "en",
                "web_search": True,
            },
            _Request(cookies),
        )

        assert response["reply"] == "The current answer is supported [1]."
        assert response["web_search"]["status"] == "complete"
        assert response["web_search"]["sources"][0]["url"] == "https://example.test/current"
        assert app.state.web_search_client.queries == ["Find the current answer"]
        assert "Official current source" in app.state.ai_client.calls[-1]["messages"][-1]["content"]
    finally:
        app.state.runner.stop()


def test_agent_chat_stream_persists_delta_and_emits_terminal_message(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, "A streamed answer.")
    try:
        user, cookies = _register_verified(app, "stream-chat@example.com")
        provider = _create_provider(app, cookies)
        stream_endpoint = _endpoint(app, "/api/agent/chat/stream", "POST")

        response = stream_endpoint(
            {
                "provider_config_id": provider["id"],
                "content": "Stream this response",
                "language": "en",
            },
            _Request(cookies),
        )

        async def consume_body():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

        body = asyncio.run(consume_body())
        assert "event: meta" in body
        assert "event: delta" in body
        assert "event: done" in body
        assert "A streamed answer." in body

        listing = app.state.store.list_agent_chat_conversations(user=user)
        assert listing["total"] == 1
        detail = app.state.store.list_agent_chat_messages(listing["items"][0]["id"], user=user)
        assert [message["content"] for message in detail["messages"]] == [
            "Stream this response",
            "A streamed answer.",
        ]
        assert detail["messages"][1]["status"] == "complete"
        assert detail["messages"][1]["usage"] == {"total_tokens": 12}
    finally:
        app.state.runner.stop()


def test_agent_chat_stream_preserves_partial_answer_with_friendly_interruption(
    tmp_path, monkeypatch
):
    app = _auth_app(tmp_path, monkeypatch, "unused")

    class _InterruptedAiClient(_FakeAgentAiClient):
        def stream_chat(self, provider_config, api_key, messages, max_tokens=1200, temperature=0.2):
            yield {"type": "delta", "content": "Partial answer"}
            raise AiStreamInterruptedError(partial=True)

    app.state.ai_client = _InterruptedAiClient("unused")
    try:
        user, cookies = _register_verified(app, "stream-interrupted@example.com")
        provider = _create_provider(app, cookies)
        stream_endpoint = _endpoint(app, "/api/agent/chat/stream", "POST")
        response = stream_endpoint(
            {
                "provider_config_id": provider["id"],
                "content": "Stream this response",
                "language": "en",
            },
            _Request(cookies),
        )

        async def consume_body():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

        body = asyncio.run(consume_body())
        assert "event: error" in body
        assert '"code":"ai_stream_interrupted"' in body
        assert "generated content has been preserved" in body
        assert "Errno" not in body

        conversation = app.state.store.list_agent_chat_conversations(user=user)["items"][0]
        assistant = app.state.store.list_agent_chat_messages(conversation["id"], user=user)[
            "messages"
        ][-1]
        assert assistant["content"] == "Partial answer"
        assert assistant["status"] == "error"
        assert assistant["finish_reason"] == "interrupted"
        assert "generated content has been preserved" in assistant["error"]
    finally:
        app.state.runner.stop()


def test_agent_chat_stream_searches_on_request_and_persists_sources(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, "A current answer [1].")
    app.state.web_search_client = _FakeWebSearchClient()
    try:
        user, cookies = _register_verified(app, "web-search-chat@example.com")
        provider = _create_provider(app, cookies)
        stream_endpoint = _endpoint(app, "/api/agent/chat/stream", "POST")

        response = stream_endpoint(
            {
                "provider_config_id": provider["id"],
                "content": "Please search the web and answer with current sources",
                "language": "en",
                "web_search": True,
            },
            _Request(cookies),
        )

        async def consume_body():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

        body = asyncio.run(consume_body())
        assert "event: search" in body
        assert '"status":"searching"' in body
        assert '"status":"complete"' in body
        assert app.state.web_search_client.queries == [
            "Please search the web and answer with current sources"
        ]
        assert "Official current source" in app.state.ai_client.calls[-1]["messages"][-1]["content"]

        conversation = app.state.store.list_agent_chat_conversations(user=user)["items"][0]
        messages = app.state.store.list_agent_chat_messages(conversation["id"], user=user)[
            "messages"
        ]
        assistant = messages[-1]
        assert assistant["web_search"]["credits"] == 1
        assert assistant["web_search"]["sources"][0]["url"] == "https://example.test/current"

        follow_up = app.state.store.get_agent_chat_context(
            conversation["id"],
            assistant["ordinal"] + 1,
            user=user,
        )
        assert "Sources used for this answer" in follow_up[-1]["content"]
        assert "https://example.test/current" in follow_up[-1]["content"]
    finally:
        app.state.runner.stop()


def test_agent_chat_search_quota_failure_falls_back_to_normal_answer(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, "Offline fallback answer.")
    app.state.web_search_client = _FakeWebSearchClient(available=False, reason="quota_exhausted")
    try:
        user, cookies = _register_verified(app, "web-search-quota@example.com")
        provider = _create_provider(app, cookies)
        stream_endpoint = _endpoint(app, "/api/agent/chat/stream", "POST")
        response = stream_endpoint(
            {
                "provider_config_id": provider["id"],
                "content": "Use current information",
                "language": "en",
                "web_search": True,
            },
            _Request(cookies),
        )

        async def consume_body():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
            return "".join(chunks)

        body = asyncio.run(consume_body())
        assert "Offline fallback answer." in body
        assert '"status":"unavailable"' in body
        assert "quota_exhausted" in body
        assert app.state.web_search_client.queries == []
        assert (
            "could not be verified online"
            in app.state.ai_client.calls[-1]["messages"][-1]["content"]
        )

        conversation = app.state.store.list_agent_chat_conversations(user=user)["items"][0]
        assistant = app.state.store.list_agent_chat_messages(conversation["id"], user=user)[
            "messages"
        ][-1]
        assert assistant["status"] == "complete"
        assert assistant["web_search"]["reason"] == "quota_exhausted"
    finally:
        app.state.runner.stop()


def test_agent_chat_history_is_strictly_owner_scoped_and_keeps_model_snapshots(
    tmp_path, monkeypatch
):
    app = _auth_app(tmp_path, monkeypatch, "unused")
    try:
        owner, _owner_cookies = _register_verified(app, "chat-owner@example.com")
        other, _other_cookies = _register_verified(app, "chat-admin@example.com")
        admin = app.state.store.update_user_admin_fields(other["id"], role="admin")
        first_provider = app.state.store.create_ai_provider_config(
            owner["id"],
            "openai",
            "OpenAI A",
            "https://example.test/v1",
            "model-a",
            "cipher-a",
            "er-a",
        )
        second_provider = app.state.store.create_ai_provider_config(
            owner["id"],
            "openai_compatible",
            "Model B",
            "https://example.test/v1",
            "model-b",
            "cipher-b",
            "er-b",
        )
        first_turn = app.state.store.begin_agent_chat_turn(
            owner["id"],
            first_provider["id"],
            "openai",
            "model-a",
            "first question",
            user=owner,
        )
        app.state.store.update_agent_chat_message(
            first_turn["assistant_message"]["id"],
            user=owner,
            content="first answer",
            status="complete",
        )
        second_turn = app.state.store.begin_agent_chat_turn(
            owner["id"],
            second_provider["id"],
            "openai_compatible",
            "model-b",
            "second question",
            conversation_id=first_turn["conversation"]["id"],
            user=owner,
        )
        context = app.state.store.get_agent_chat_context(
            first_turn["conversation"]["id"],
            second_turn["assistant_message"]["ordinal"],
            user=owner,
        )

        assert [message["content"] for message in context] == [
            "first question",
            "first answer",
            "second question",
        ]
        assert app.state.store.list_agent_chat_conversations(user=admin)["total"] == 0
        app.state.store.delete_ai_provider_config(first_provider["id"], user=owner)
        messages = app.state.store.list_agent_chat_messages(
            first_turn["conversation"]["id"], user=owner
        )["messages"]
        first_assistant = next(
            message
            for message in messages
            if message["id"] == first_turn["assistant_message"]["id"]
        )
        second_assistant = next(
            message
            for message in messages
            if message["id"] == second_turn["assistant_message"]["id"]
        )
        assert first_assistant["provider_config_id"] is None
        assert first_assistant["model"] == "model-a"
        assert second_assistant["model"] == "model-b"
    finally:
        app.state.runner.stop()


def _experiment_analysis_answer():
    return json.dumps(
        {
            "summary": "Validation is stable, but a classical baseline would improve the comparison.",
            "evidence": [
                {
                    "label": "Validation F1",
                    "value": "0.72",
                    "interpretation": "The current run has a usable provisional checkpoint.",
                    "source_section": "experiment",
                }
            ],
            "findings": [
                {
                    "severity": "warning",
                    "title": "Baseline coverage is narrow",
                    "detail": "Only the neural route has been evaluated.",
                    "source_sections": ["experiment", "analysis"],
                }
            ],
            "recommendations": [
                {
                    "title": "Run a random-forest baseline",
                    "rationale": "It provides a reusable classical comparison.",
                    "priority": "high",
                    "success_criteria": [
                        "The comparison separates representation gains from classifier gains."
                    ],
                }
            ],
            "limitations": ["The running experiment does not have final test metrics yet."],
        }
    )


def _create_analysis_experiment(app, dataset, user, experiment_id, status):
    root = app.state.paths.experiments / experiment_id
    artifact_dir = root / "artifact"
    artifact_dir.mkdir(parents=True)
    (root / "train.log").write_text("epoch 1 val_f1=0.72\n", encoding="utf-8")
    experiment = app.state.store.create_experiment(
        "diagnostic run",
        dataset["id"],
        dataset["task_type"],
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn", "epochs": 8},
        root / "train.log",
        artifact_dir,
        experiment_id=experiment_id,
        owner_user_id=user["id"],
    )
    if status != "queued":
        experiment = app.state.store.update_experiment(
            experiment["id"],
            status=status,
            metrics_json={"events": [], "summary": {"best_val_f1": 0.72}},
        )
    return experiment


def _consume_streaming_response(response):
    return "".join(
        chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        for chunk in _streaming_response_iterator(response)
    )


def test_experiment_analysis_repairs_report_and_persists_followup(
    tmp_path,
    monkeypatch,
):
    app = _auth_app(
        tmp_path,
        monkeypatch,
        [
            "not diagnostic json",
            _experiment_analysis_answer(),
            "The follow-up uses the saved report.",
        ],
    )
    try:
        user, cookies = _register_verified(app, "experiment-analysis@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        experiment = _create_analysis_experiment(app, dataset, user, "analysis-running", "running")
        stream_endpoint = _endpoint(app, "/api/experiments/{experiment_id}/analysis/stream", "POST")

        response = stream_endpoint(
            experiment["id"],
            {
                "provider_config_id": provider["id"],
                "content": "Diagnose this experiment.",
                "language": "en",
            },
            _Request(cookies),
        )
        body = _consume_streaming_response(response)

        assert '"stage":"repairing_report"' in body
        assert "event: done" in body
        assert len(app.state.ai_client.calls) == 2
        assert all(
            call["response_format"] == {"type": "json_object"} for call in app.state.ai_client.calls
        )

        listing = app.state.store.list_agent_chat_conversations(
            user=user,
            scope="experiment_analysis",
            experiment_id=experiment["id"],
        )
        assert listing["total"] == 1
        conversation = listing["items"][0]
        detail = app.state.store.list_agent_chat_messages(
            conversation["id"],
            user=user,
            scope="experiment_analysis",
            experiment_id=experiment["id"],
        )
        report_message = detail["messages"][-1]
        report = report_message["payload"]["report"]
        recommendation = report["recommendations"][0]
        assert report_message["kind"] == "report"
        assert report_message["status"] == "complete"
        assert report_message["payload"]["repair_attempted"] is True
        assert report["status"] == "provisional"
        assert recommendation["success_criteria"] == [
            "The comparison separates representation gains from classifier gains."
        ]
        assert "proposed_config" not in recommendation
        assert "preflight" not in recommendation
        assert report_message["payload"]["warnings"] == []
        assert report_message["context_manifest"]["context_hash"]

        followup_response = stream_endpoint(
            experiment["id"],
            {
                "conversation_id": conversation["id"],
                "provider_config_id": provider["id"],
                "content": "Why use a classical baseline?",
                "language": "en",
            },
            _Request(cookies),
        )
        followup_body = _consume_streaming_response(followup_response)
        assert "event: delta" in followup_body
        assert "The follow-up uses the saved report." in followup_body

        updated_detail = app.state.store.list_agent_chat_messages(
            conversation["id"],
            user=user,
            scope="experiment_analysis",
            experiment_id=experiment["id"],
        )
        assert [message["kind"] for message in updated_detail["messages"]] == [
            "chat",
            "report",
            "chat",
            "chat",
        ]
        assert updated_detail["messages"][-1]["content"] == "The follow-up uses the saved report."
    finally:
        app.state.runner.stop()


def test_experiment_diagnostic_report_seeds_agent_planning_context(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, [_experiment_analysis_answer(), _agent_answer()])
    try:
        user, cookies = _register_verified(app, "diagnostic-planning@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        experiment = _create_analysis_experiment(app, dataset, user, "analysis-source", "succeeded")
        stream_endpoint = _endpoint(app, "/api/experiments/{experiment_id}/analysis/stream", "POST")
        get_message = _endpoint(
            app, "/api/experiments/{experiment_id}/analysis/messages/{message_id}", "GET"
        )
        create_plan = _endpoint(app, "/api/agent/execution-plans", "POST")

        response = stream_endpoint(
            experiment["id"],
            {
                "provider_config_id": provider["id"],
                "content": "Diagnose this completed experiment.",
                "language": "en",
            },
            _Request(cookies),
        )
        _consume_streaming_response(response)
        conversation = app.state.store.list_agent_chat_conversations(
            user=user,
            scope="experiment_analysis",
            experiment_id=experiment["id"],
        )["items"][0]
        messages = app.state.store.list_agent_chat_messages(
            conversation["id"],
            user=user,
            scope="experiment_analysis",
            experiment_id=experiment["id"],
        )["messages"]
        report_message = messages[-1]

        loaded_message = get_message(experiment["id"], report_message["id"], _Request(cookies))
        assert loaded_message["payload"]["report"]["summary"].startswith("Validation is stable")

        plan = create_plan(
            {
                "provider_config_id": provider["id"],
                "dataset_id": dataset["id"],
                "task_type": dataset["task_type"],
                "objective": "Plan from the attached diagnosis.",
                "diagnostic_source": {
                    "experiment_id": experiment["id"],
                    "message_id": report_message["id"],
                },
                "language": "en",
            },
            _Request(cookies),
        )

        assert plan["trace"]["diagnostic_source"] == {
            "experiment_id": experiment["id"],
            "message_id": report_message["id"],
            "conversation_id": conversation["id"],
        }
        assert plan["context_summary"]["baseline_experiment_count"] == 1
        planning_prompt = app.state.ai_client.calls[-1]["messages"][-1]["content"]
        assert '"diagnostic_source"' in planning_prompt
        assert "Validation is stable" in planning_prompt

        with pytest.raises(fastapi.HTTPException) as exc_info:
            create_plan(
                {
                    "provider_config_id": provider["id"],
                    "dataset_id": dataset["id"],
                    "task_type": "pep_hlai",
                    "diagnostic_source": {
                        "experiment_id": experiment["id"],
                        "message_id": report_message["id"],
                    },
                },
                _Request(cookies),
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "agent_diagnostic_source_context_mismatch"
    finally:
        app.state.runner.stop()


def test_experiment_analysis_rejects_queued_runs_and_scopes_history_to_experiment(
    tmp_path,
    monkeypatch,
):
    app = _auth_app(tmp_path, monkeypatch, _experiment_analysis_answer())
    try:
        user, cookies = _register_verified(app, "experiment-analysis-scope@example.com")
        provider = _create_provider(app, cookies)
        dataset = _create_sequence_dataset(app, user)
        queued = _create_analysis_experiment(app, dataset, user, "analysis-queued", "queued")
        running = _create_analysis_experiment(app, dataset, user, "analysis-visible", "running")
        stream_endpoint = _endpoint(app, "/api/experiments/{experiment_id}/analysis/stream", "POST")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            stream_endpoint(
                queued["id"],
                {"provider_config_id": provider["id"], "content": "Analyze now."},
                _Request(cookies),
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["code"] == "experiment_analysis_not_ready"

        turn = app.state.store.begin_agent_chat_turn(
            user["id"],
            provider["id"],
            provider["provider"],
            provider["model"],
            "Experiment-only question",
            user=user,
            scope="experiment_analysis",
            experiment_id=running["id"],
            assistant_kind="report",
        )
        conversation_id = turn["conversation"]["id"]
        assert app.state.store.list_agent_chat_conversations(user=user)["total"] == 0
        assert (
            app.state.store.list_agent_chat_conversations(
                user=user,
                scope="experiment_analysis",
                experiment_id=queued["id"],
            )["total"]
            == 0
        )
        assert (
            app.state.store.get_agent_chat_conversation(
                conversation_id,
                user=user,
                scope="experiment_analysis",
                experiment_id=queued["id"],
            )
            is None
        )
        assert (
            app.state.store.get_agent_chat_conversation(
                conversation_id,
                user=user,
                scope="experiment_analysis",
                experiment_id=running["id"],
            )["id"]
            == conversation_id
        )
    finally:
        app.state.runner.stop()
