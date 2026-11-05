"""Web contracts: numerical parity, provenance, frozen Test gate and isolation."""

import json
from pathlib import Path

import numpy as np
import pytest

from pepplatform.web.ensemble_data import (
    PROVENANCE_FILE,
    file_digest,
    read_json,
    read_split,
    response_scores,
    tree_identity,
    validation_identity,
    write_json,
)
from pepplatform.web.ensemble_math import EnsembleError, fuse, normalize_definition, select
from pepplatform.web.ensemble_service import EnsembleService, SnapshotStore, WebModelService
from pepplatform.web.ensemble_worker import assemble
from pepplatform.web.inference import ModelService
from pepplatform.web.paths import RuntimePaths
from pepplatform.web.store import MetadataStore


@pytest.mark.parametrize(
    "method,expected",
    [
        ("probability_mean", [[0.65, 0.35], [0.35, 0.65]]),
        ("logit_mean", [[2 / 3, 1 / 3], [1 / 3, 2 / 3]]),
        ("rank_mean", [[0.75, 0.25], [0.25, 0.75]]),
    ],
)
def test_fusion_matches_known_weighted_scores(method, expected):
    members = [np.array([[0.2, 0.8], [0.8, 0.2]]), np.array([[0.8, 0.2], [0.2, 0.8]])]
    np.testing.assert_allclose(fuse(members, [0.25, 0.75], method), expected)


@pytest.mark.parametrize("task", ["pep_hlai", "pep_hlaii", "pep_protein_interaction"])
@pytest.mark.parametrize("method", ["probability_mean", "logit_mean", "rank_mean"])
def test_binary_selection_finds_the_perfect_member(task, method):
    y = np.array([0, 1, 0, 1])
    members = [np.array([0.8, 0.2, 0.8, 0.2]), np.array([0.2, 0.8, 0.2, 0.8])]
    definition = normalize_definition(
        {
            "task_type": task,
            "member_ids": ["a", "b"],
            "method": method,
            "weight_source": "validation",
            "grid_step": 0.5,
        }
    )
    weights, point, _ = select(members, y, ["positive"], definition)
    np.testing.assert_array_equal(weights, [0, 1])
    from pepplatform.evaluators import Evaluator

    metrics = Evaluator.score_metrics(y, fuse(members, weights, method), point)
    assert metrics["auc"] == metrics["f1"] == 1


def test_sequence_selection_and_input_errors():
    from pepplatform.ensemble import select_weights

    y = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    scores = [np.array([[0.1, 0.6], [0.7, 0.2], [0.8, 0.9], [0.3, 0.4]]), np.full((4, 2), 0.4)]
    definition = normalize_definition(
        {
            "task_type": "pep_function",
            "member_ids": ["a", "b"],
            "weight_source": "validation",
            "grid_step": 0.5,
        }
    )
    weights, point, _ = select(scores, y, ["A", "B"], definition)
    expected = select_weights(scores, y, label_names=["A", "B"], grid_step=0.5)
    np.testing.assert_array_equal(weights, expected["weights"])
    assert point == expected["operating_point"]
    for weights in ([0, 0], [1, -1], [np.nan, 1], [1]):
        with pytest.raises(EnsembleError):
            fuse(scores, weights)
    with pytest.raises(EnsembleError, match="complete batch"):
        fuse([np.array([0.2]), np.array([0.3])], [0.5, 0.5], "rank_mean")
    with pytest.raises(EnsembleError):
        fuse([np.array([np.nan, 0.2]), np.array([0.1, 0.3])], [0.5, 0.5], "rank_mean")


def test_new_sequence_ensembles_default_to_f1_and_require_dual_parameters():
    from pepplatform.evaluators import Evaluator
    from pepplatform.objectives import ObjectiveSpec

    request = {
        "task_type": "pep_function",
        "member_ids": ["a", "b"],
        "weight_source": "validation",
        "grid_step": 0.5,
    }
    definition = normalize_definition(request)
    assert definition["schema_version"] == 2
    assert definition["objective"] == definition["threshold_objective"] == "macro_f1"
    with pytest.raises(EnsembleError, match="parameters"):
        normalize_definition({**request, "objective": "dual_f1_sacc"})
    refs = {"reference_f1": 0.4, "reference_sacc": 0.6}
    definition = normalize_definition(
        {**request, "objective": "dual_f1_sacc", "objective_params": refs}
    )
    # A dual weight objective does not implicitly change F1 threshold fitting.
    assert definition["threshold_objective"] == "macro_f1"
    rng = np.random.default_rng(13)
    y = rng.integers(0, 2, (41, 3))
    scores = [rng.random(y.shape), rng.random(y.shape)]
    weights, point, _ = select(scores, y, ["A", "B", "C"], definition)
    expected = Evaluator.fit_operating_point(
        y,
        fuse(scores, weights),
        objective=ObjectiveSpec.parse("macro_f1"),
        label_names=["A", "B", "C"],
    )
    assert point == expected


def test_custom_weights_keep_original_values_across_runner_normalization():
    definition = normalize_definition(
        {
            "task_type": "pep_hlai",
            "member_ids": ["a", "b"],
            "weight_source": "custom",
            "weights": [2, 3],
        }
    )
    assert normalize_definition(definition) == definition
    assert definition["requested_weights"] == [2, 3]


def dataset_files(root, task):
    for split, values in {
        "train": ["AAAA", "CCCC", "DDDD", "EEEE"],
        "val": ["FFFF", "GGGG", "HHHH", "IIII"],
        "test": ["KKKK", "LLLL", "MMMM", "NNNN"],
    }.items():
        directory = root / split
        directory.mkdir(parents=True)
        if task == "pep_function":
            (directory / "seqs.fasta").write_text(
                "".join(f">{i}\n{x}\n" for i, x in enumerate(values))
            )
            (directory / "labels.csv").write_text("A,B\n0,1\n1,0\n0,1\n1,0\n")
        else:
            (directory / "pairs.csv").write_text(
                "sample_id,peptide,protein,label\n"
                + "".join(f"{i},{x},ACDE,{i % 2}\n" for i, x in enumerate(values))
            )
            (directory / "labels.csv").write_text("label\n0\n1\n0\n1\n")


def make_request(tmp_path, task="pep_function", method="probability_mean"):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    store = MetadataStore(paths.database)
    root = paths.datasets / "demo"
    dataset_files(root, task)
    dataset = store.create_dataset(
        "demo", task, "upload", "sequence_dir" if task == "pep_function" else "pair_split_dir", root
    )
    labels = ["A", "B"] if task == "pep_function" else ["interaction"]
    members = []
    for index in range(2):
        bundle = paths.models / f"member{index}"
        bundle.mkdir()
        metadata = {
            "evaluation_protocol_version": 3,
            "feature_fit_scope": "train_only",
            "bundle_type": "sequence_nn" if task == "pep_function" else "pair_vector_nn",
            "label_names": labels,
            "args": {"task_type": task},
        }
        write_json(bundle / "metadata.json", metadata)
        write_json(bundle / PROVENANCE_FILE, validation_identity(dataset))
        experiment = store.create_experiment(
            f"member{index}", dataset["id"], task, {}, bundle / "log", bundle
        )
        members.append(
            store.create_model(experiment["id"], f"member{index}", task, bundle, True, metadata)
        )
    payload = {
        "task_type": task,
        "dataset_id": dataset["id"],
        "member_ids": [x["id"] for x in members],
        "method": method,
    }
    service = EnsembleService(store, paths)
    return service.prepare(payload), service, payload, paths


def fake_prediction(self, model_id, sequences=None, pairs=None):
    record = self.store.get_model(model_id)
    metadata = read_json(Path(record["bundle_path"]) / "metadata.json")
    values = sequences if sequences is not None else [x["peptide"] for x in pairs]
    count = len(metadata["label_names"])
    scores = [
        [((sum(map(ord, x)) + column * 7) % 19 + 1) / 21 for column in range(count)] for x in values
    ]
    return {
        "model_id": model_id,
        "labels": metadata["label_names"],
        "score_kind": "probability",
        "predictions": [{"scores": row, "score_kind": "probability"} for row in scores],
    }


@pytest.mark.parametrize(
    "task", ["pep_function", "pep_hlai", "pep_hlaii", "pep_protein_interaction"]
)
@pytest.mark.parametrize("method", ["probability_mean", "logit_mean", "rank_mean"])
@pytest.mark.parametrize("weight_source", ["equal", "custom", "validation"])
def test_assembly_reload_and_source_deletion(tmp_path, monkeypatch, task, method, weight_source):
    request, service, payload, paths = make_request(tmp_path, task, method)
    request = service.prepare(
        {**payload, "weight_source": weight_source, "weights": [2, 3], "grid_step": 0.5}
    )
    before = [tree_identity(x["bundle_path"]) for x in request["members"]]
    monkeypatch.setattr(ModelService, "_predict", fake_prediction)
    output = paths.models / "ensemble"
    metadata = assemble(request, output, report=lambda _: None)
    assert read_json(output / "audit.json")["passed"]
    assert read_json(output / "analysis.json")["curves"]["validation"]["roc"]["available"]
    assert not read_json(output / "evaluation.json")["test_splits"]
    assert before == [tree_identity(x["bundle_path"]) for x in request["members"]]
    record = {
        "id": "ensemble",
        "bundle_path": str(output),
        "metadata": metadata,
        "supports_inference": True,
    }
    predictor = WebModelService(SnapshotStore([record]))
    val = read_split(request["dataset"], "val")
    response = predictor.predict("ensemble", **val["inputs"])
    with np.load(output / "evaluation_scores.npz") as scores:
        result = np.asarray([row["scores"] for row in response["predictions"]])
        np.testing.assert_array_equal(
            result if task == "pep_function" else result[:, 0], scores["validation_scores"]
        )
    # No source records are available in this store: all inference uses snapshots.
    assert response["ensemble"]["batch_dependent"] == (method == "rank_mean")
    from pepplatform.web.app import _prediction_csv
    import csv

    rows = list(csv.DictReader(_prediction_csv(response).splitlines()))
    assert len(rows) == len(val["y"])
    assert all(
        row["score_kind"] == ("rank" if method == "rank_mean" else "probability") for row in rows
    )
    assert all(row["batch_digest"] == response["ensemble"]["batch_digest"] for row in rows)
    assert all(
        row["ensemble_method"] == method and int(row["batch_input_count"]) == len(rows)
        for row in rows
    )
    (output / "members/0/metadata.json").write_text("{}")
    with pytest.raises(EnsembleError, match="missing or changed"):
        predictor.predict("ensemble", **val["inputs"])


@pytest.mark.parametrize("mode", ["valid", "row_order", "missing_ids", "labels", "scores_changed"])
def test_cached_validation_requires_score_and_sample_identity(tmp_path, monkeypatch, mode):
    request, service, payload, paths = make_request(tmp_path)
    val = read_split(request["dataset"], "val")
    for member in request["members"]:
        bundle = Path(member["bundle_path"])
        ids = np.asarray(val["identity"]["sample_ids"])
        arrays = {
            "validation_labels": 1 - val["y"] if mode == "labels" else val["y"],
            "validation_scores": np.full(val["y"].shape, 0.4),
        }
        if mode != "missing_ids":
            arrays["validation_sample_ids"] = ids[::-1] if mode == "row_order" else ids
        np.savez_compressed(bundle / "evaluation_scores.npz", **arrays)
        proof = read_json(bundle / PROVENANCE_FILE)
        proof.update(
            evaluation_scores_sha256=file_digest(bundle / "evaluation_scores.npz"),
            score_kind="probability",
        )
        write_json(bundle / PROVENANCE_FILE, proof)
        if mode == "scores_changed":
            np.savez_compressed(bundle / "evaluation_scores.npz", **{**arrays, "extra": [1]})
    calls = []

    def predict(self, model_id, **inputs):
        calls.append(model_id)
        return fake_prediction(self, model_id, **inputs)

    monkeypatch.setattr(ModelService, "_predict", predict)
    request = service.prepare(payload)
    output = paths.models / "cache-ensemble"
    if mode == "labels":
        with pytest.raises(EnsembleError, match="Cached validation labels"):
            assemble(request, output, report=lambda _: None)
        return
    assemble(request, output, report=lambda _: None)
    expected = "verified_cached_validation" if mode == "valid" else "fresh_frozen_member_inference"
    assert read_json(output / "ensemble.json")["validation_score_sources"] == [expected] * 2
    assert len(calls) == (0 if mode == "valid" else 2)


@pytest.mark.parametrize(
    "change,code",
    [
        ("member_labels", "ensemble_label_mismatch"),
        ("dataset_labels", "ensemble_label_mismatch"),
        ("nested", "ensemble_nested_unsupported"),
        ("duplicate", "ensemble_member_duplicate"),
        ("paper_link", "ensemble_paper_source_forbidden"),
    ],
)
def test_preflight_rejects_invalid_members_and_columns(tmp_path, change, code):
    request, service, payload, _ = make_request(tmp_path)
    bundle = Path(request["members"][0]["bundle_path"])
    metadata = read_json(bundle / "metadata.json")
    if change == "member_labels":
        write_json(bundle / "metadata.json", {**metadata, "label_names": ["B", "A"]})
    elif change == "dataset_labels":
        path = Path(request["dataset"]["root_path"]) / "val/labels.csv"
        path.write_text(path.read_text().replace("A,B", "B,A"))
    elif change == "nested":
        write_json(bundle / "metadata.json", {**metadata, "bundle_type": "ensemble"})
    elif change == "duplicate":
        payload["member_ids"] = [payload["member_ids"][0]] * 2
    else:
        paper_file = tmp_path / "experiment_paper/frozen.json"
        write_json(paper_file, {"sealed": True})
        (bundle / "borrowed.json").symlink_to(paper_file)
    assert service.preflight(payload)["error_codes"] == [code]


def test_member_mutation_between_preparation_and_snapshot_is_rejected(tmp_path, monkeypatch):
    request, _, _, paths = make_request(tmp_path)
    write_json(Path(request["members"][0]["bundle_path"]) / "new-parameters.json", {"value": 1})
    monkeypatch.setattr(
        ModelService, "_predict", lambda *a, **kw: pytest.fail("must reject before inference")
    )
    with pytest.raises(EnsembleError, match="changed after preparation"):
        assemble(request, paths.models / "changed", report=lambda _: None)


@pytest.mark.parametrize(
    "kind,values,method",
    [
        ("margin", [0.2, 0.3], "probability_mean"),
        ("margin", [0.2, 0.3], "logit_mean"),
        ("probability", [float("nan"), 0.3], "rank_mean"),
        ("probability", [-0.1, 1.2], "probability_mean"),
    ],
)
def test_invalid_member_scores_fail_closed(kind, values, method):
    response = {
        "labels": ["interaction"],
        "score_kind": kind,
        "predictions": [{"scores": [value]} for value in values],
    }
    with pytest.raises(EnsembleError):
        scores = response_scores(response, ["interaction"], 2, "pep_hlai", method)
        fuse([scores, scores], [0.5, 0.5], method)


def test_pair_binary_float_labels_are_supported_without_truncation(tmp_path):
    request, _, _, _ = make_request(tmp_path, "pep_hlai")
    directory = Path(request["dataset"]["root_path"]) / "val"
    path = directory / "pairs.csv"
    original = path.read_text()
    path.write_text(original.replace(",0\n", ",0.0\n").replace(",1\n", ",1.0\n"))
    np.testing.assert_array_equal(read_split(request["dataset"], "val")["y"], [0, 1, 0, 1])
    path.write_text(original.replace(",1\n", ",1.9\n"))
    with pytest.raises(EnsembleError):
        read_split(request["dataset"], "val")


def test_named_independent_tests_are_separate_and_follow_freeze(tmp_path, monkeypatch):
    request, _, _, paths = make_request(tmp_path, "pep_hlaii")
    dataset = request["dataset"]
    root = Path(dataset["root_path"])
    (root / "test").rename(root / "test_warm")
    cold = root / "test_cold"
    cold.mkdir()
    (cold / "pairs.csv").write_text(
        "sample_id,peptide,protein,label\n1,PPPP,ACDE,0\n2,QQQQ,ACDE,1\n"
    )
    dataset["metadata"] = {
        **dataset.get("metadata", {}),
        "evaluation_splits": ["test_warm", "test_cold"],
    }
    request["definition"]["evaluate_test"] = True
    monkeypatch.setattr(ModelService, "_predict", fake_prediction)
    output = paths.models / "named"

    def report(event):
        if event["stage"] == "definition_frozen":
            assert (output / "ensemble.json").is_file()
            assert not (output / "evaluations").exists()

    assemble(request, output, report=report)
    evaluation = read_json(output / "evaluation.json")
    assert set(evaluation["test_splits"]) == {"test_warm", "test_cold"}
    assert "test" not in evaluation
    for name in evaluation["test_splits"]:
        assert (output / "evaluations" / name / "scores.npz").is_file()


def test_snapshot_digest_ignores_only_generated_profile_caches(tmp_path):
    write_json(tmp_path / "metadata.json", {"model": "frozen"})
    before = tree_identity(tmp_path)
    write_json(
        tmp_path / "profile_generation/inference_cache/batch/scores.json", {"generated": True}
    )
    write_json(tmp_path / "profile_generation/blastdb/db.json", {"generated": True})
    assert tree_identity(tmp_path) == before
    write_json(tmp_path / "profile_generation/subject.json", {"training": True})
    assert tree_identity(tmp_path) != before


def test_missing_proof_and_row_identity_fail_closed(tmp_path):
    request, service, payload, paths = make_request(tmp_path)
    dataset = request["dataset"]
    root = Path(dataset["root_path"])
    fasta = root / "val/seqs.fasta"
    # Labels are identical; only sequence identities change.
    fasta.write_text(fasta.read_text().replace("FFFF", "YYYY"))
    assert service.preflight(payload)["error_codes"] == ["ensemble_dataset_changed"]
    (Path(request["members"][0]["bundle_path"]) / PROVENANCE_FILE).unlink()
    assert service.preflight(payload)["error_codes"] == ["ensemble_evidence_missing"]


def test_test_access_is_after_freeze_and_cannot_affect_selection(tmp_path, monkeypatch):
    import pepplatform.web.ensemble_worker as worker

    request, _, _, paths = make_request(tmp_path, "pep_hlai")
    request["definition"]["evaluate_test"] = True
    request["definition"]["weight_source"] = "validation"
    request["definition"]["grid_step"] = 0.5
    monkeypatch.setattr(ModelService, "_predict", fake_prediction)
    original = worker.read_split
    output = paths.models / "first"
    accesses = []

    def guarded(dataset, split):
        if split == "test":
            assert (output / "ensemble.json").is_file()
            accesses.append(split)
        return original(dataset, split)

    monkeypatch.setattr(worker, "read_split", guarded)
    assemble(request, output, report=lambda _: None)
    first = read_json(output / "ensemble.json")
    assert accesses
    root = Path(request["dataset"]["root_path"]) / "test"
    (root / "labels.csv").write_text("label\n1\n0\n1\n0\n")
    lines = (root / "pairs.csv").read_text().splitlines()
    (root / "pairs.csv").write_text(
        lines[0] + "\n" + "\n".join(line[:-1] + str(1 - int(line[-1])) for line in lines[1:]) + "\n"
    )
    output = paths.models / "second"
    assemble(request, output, report=lambda _: None)
    second = read_json(output / "ensemble.json")
    assert first["weights"] == second["weights"]
    assert first["operating_point"] == second["operating_point"]


def test_legacy_dispatch_does_not_close_shared_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ModelService,
        "_predict",
        lambda self, model_id, **kwargs: calls.append(model_id) or {"predictions": []},
    )
    monkeypatch.setattr(
        ModelService, "close", lambda: pytest.fail("request closed the shared runtime")
    )
    service = WebModelService(
        SnapshotStore([{"id": "old", "metadata": {"bundle_type": "ensemble"}}])
    )
    assert service._predict("old")["predictions"] == []
    assert calls == ["old"]


def test_ensemble_readiness_preserves_warnings_and_error_schema(monkeypatch):
    import pepplatform.web.ensemble_service as adapter

    record = {"id": "ensemble", "metadata": {"bundle_type": "web_ensemble"}}
    service = WebModelService(SnapshotStore([record]))
    monkeypatch.setattr(
        adapter,
        "load_bundle",
        lambda _: ({}, {"definition": {"method": "rank_mean"}}, [{"id": "member"}]),
    )
    warning = {"code": "profile_cache_pending", "message": "Cache is generated on demand"}
    monkeypatch.setattr(
        ModelService,
        "readiness",
        lambda self, _: {
            "ready": True,
            "profile_required": True,
            "errors": [],
            "warnings": [warning],
        },
    )
    result = service.readiness("ensemble")
    assert result["ready"] and result["profile_required"]
    assert result["status"] == "warning"
    assert result["warnings"] == [warning]

    def unavailable(_):
        raise EnsembleError("ensemble_snapshot_changed", "Changed snapshot")

    monkeypatch.setattr(adapter, "load_bundle", unavailable)
    blocked = service.readiness("ensemble")
    assert not blocked["ready"] and blocked["warnings"] == []
    assert blocked["errors"][0]["code"] == "ensemble_snapshot_changed"


def endpoint(app, path, method):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == path and method in route.methods
    )


def test_api_gate_creation_and_legacy_endpoint_cannot_bypass(tmp_path, monkeypatch):
    from pepplatform.web.app import create_app
    from fastapi import HTTPException

    request, _, payload, paths = make_request(tmp_path)
    monkeypatch.setenv("PEP_AUTH_ENABLED", "0")
    monkeypatch.delenv("PEP_DATABASE_URL", raising=False)
    app = create_app(project_root=paths.project_root, runtime_root=paths.root)
    try:
        monkeypatch.setattr(app.state.runner, "submit", lambda _: None)
        create = endpoint(app, "/api/ensembles", "POST")
        monkeypatch.setenv("PEP_WEB_ENSEMBLES_ENABLED", "0")
        with pytest.raises(HTTPException) as exc:
            create(payload)
        assert exc.value.status_code == 409
        assert endpoint(app, "/api/ensembles/preflight", "POST")(payload)["valid"]
        with pytest.raises(HTTPException):
            endpoint(app, "/api/experiments", "POST")(
                {"dataset_id": payload["dataset_id"], "config": {"web_ensemble": payload}}
            )
        monkeypatch.setenv("PEP_WEB_ENSEMBLES_ENABLED", "1")
        record = create(payload)
        assert record["config"]["web_ensemble"]["member_ids"] == payload["member_ids"]
        command = app.state.runner.build_command(record, request["dataset"])
        assert command[1:3] == ["-m", "pepplatform.web.ensemble_worker"]
        assert all("experiment_paper" not in arg for arg in command)
        missing = {**payload, "member_ids": [payload["member_ids"][0], "not-visible"]}
        assert endpoint(app, "/api/ensembles/preflight", "POST")(missing)["error_codes"] == [
            "ensemble_member_unavailable"
        ]
    finally:
        app.state.runner.stop()
        app.state.models.close()


def test_ensemble_creation_is_enabled_by_default_and_can_be_disabled(tmp_path, monkeypatch):
    from pepplatform.web.app import create_app
    from fastapi import HTTPException

    request, _, payload, paths = make_request(tmp_path)
    monkeypatch.setenv("PEP_AUTH_ENABLED", "0")
    monkeypatch.delenv("PEP_DATABASE_URL", raising=False)
    monkeypatch.delenv("PEP_WEB_ENSEMBLES_ENABLED", raising=False)
    app = create_app(project_root=paths.project_root, runtime_root=paths.root)
    try:
        monkeypatch.setattr(app.state.runner, "submit", lambda _: None)
        capabilities = endpoint(app, "/api/ensembles/capabilities", "GET")()
        assert capabilities["enabled"] is True
        record = endpoint(app, "/api/ensembles", "POST")(payload)
        assert record["config"]["web_ensemble"]["member_ids"] == payload["member_ids"]

        monkeypatch.setenv("PEP_WEB_ENSEMBLES_ENABLED", "0")
        with pytest.raises(HTTPException) as exc:
            endpoint(app, "/api/ensembles", "POST")(payload)
        assert exc.value.status_code == 409
        assert exc.value.detail["code"] == "ensemble_creation_disabled"
    finally:
        app.state.runner.stop()
        app.state.models.close()


def test_member_permissions_are_checked_before_loading_bundle(tmp_path):
    request, service, payload, _ = make_request(tmp_path)

    class RestrictedStore:
        def get_dataset(self, dataset_id, user=None):
            assert user == {"id": "owner"}
            return request["dataset"]

        def get_model(self, member_id, user=None):
            assert user == {"id": "owner"}
            return None

    service.store = RestrictedStore()
    assert service.preflight(payload, user={"id": "owner"})["error_codes"] == [
        "ensemble_member_unavailable"
    ]


@pytest.mark.parametrize("method", ["probability_mean", "logit_mean"])
def test_fusion_is_independent_of_compute_chunks(method):
    scores = [np.array([0.1, 0.3, 0.9, 0.7]), np.array([0.4, 0.2, 0.3, 0.8])]
    whole = fuse(scores, [0.3, 0.7], method)
    chunked = np.concatenate(
        [fuse([x[i : i + 2] for x in scores], [0.3, 0.7], method) for i in (0, 2)]
    )
    np.testing.assert_array_equal(whole, chunked)


def test_rank_preserves_the_full_batch_semantics_and_duplicate_rows():
    scores = [np.array([0.2, 0.2, 0.8, 0.5]), np.array([0.4, 0.4, 0.3, 0.6])]
    whole = fuse(scores, [0.5, 0.5], "rank_mean")
    assert len(whole) == 4
    assert not np.array_equal(whole[:2], fuse([x[:2] for x in scores], [0.5, 0.5], "rank_mean"))


@pytest.mark.parametrize("method", ["probability_mean", "logit_mean", "rank_mean"])
def test_real_cpu_pair_members_and_api_history(tmp_path, monkeypatch, method):
    """Fit real RF members on Train; exercise unchanged feature/inference code."""
    import asyncio
    import joblib
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from pepplatform.web.app import create_app

    request, service, payload, paths = make_request(tmp_path, "pep_protein_interaction", method)
    train = read_split(request["dataset"], "train")
    for index, member in enumerate(request["members"]):
        bundle = Path(member["bundle_path"])
        glove = bundle / "glove.vec"
        glove.write_text(
            "".join(
                f"{letter} {i / 20} {(20 - i) / 20}\n"
                for i, letter in enumerate("ACDEFGHIKLMNPQRSTVWY")
            )
            + "<UNK> 0 0\n"
        )
        metadata = {
            **read_json(bundle / "metadata.json"),
            "bundle_type": "pair_vector_classical",
            "threshold": 0.5,
            "classic": {
                "tokenizer": "kmer",
                "encoder": "glove",
                "max_length": 4,
                "k": 1,
                "glove_path": str(glove),
            },
            "pair": {"fusion_mode": "concat_all"},
            "args": {
                "task_type": "pep_protein_interaction",
                "tokenizer": "kmer",
                "encoder": "glove",
                "model": "rf",
                "classifier": "rf",
            },
        }
        features = ModelService._extract_pair_vector_features(
            metadata, train["inputs"]["pairs"], torch.device("cpu"), bundle=bundle
        ).numpy()
        estimator = RandomForestClassifier(n_estimators=7, random_state=42 + index, n_jobs=1)
        estimator.fit(features, train["y"])
        joblib.dump({"estimator_kind": "rf", "estimator": estimator}, bundle / "model.joblib")
        write_json(bundle / "metadata.json", metadata)
    request = service.prepare(payload)
    output = paths.models / "real-pair"
    metadata = assemble(request, output, report=lambda _: None)
    member_experiment = service.store.get_experiment(request["members"][0]["experiment_id"])
    record = service.store.create_model(
        member_experiment["id"], "real-pair", "pep_protein_interaction", output, True, metadata
    )
    monkeypatch.setenv("PEP_AUTH_ENABLED", "0")
    monkeypatch.setenv("PEP_WEB_ENSEMBLES_ENABLED", "0")
    monkeypatch.delenv("PEP_DATABASE_URL", raising=False)
    app = create_app(project_root=paths.project_root, runtime_root=paths.root)
    try:
        pairs = read_split(request["dataset"], "val")["inputs"]["pairs"]
        inputs = {"pairs": [pairs[2], pairs[0], pairs[2], pairs[1]]}
        # Both prediction endpoints must recreate a released model service.
        app.state.models.close()
        app.state.models = None
        normal = endpoint(app, "/api/models/{model_id}/predict", "POST")(record["id"], inputs)
        app.state.models.close()
        app.state.models = None
        streaming = endpoint(app, "/api/models/{model_id}/predict/stream", "POST")(
            record["id"], inputs
        )

        async def collect():
            return [chunk async for chunk in streaming.body_iterator]

        events = []
        for chunk in asyncio.run(collect()):
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            lines = text.strip().splitlines()
            events.append(
                (lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
            )
        assert [name for name, _ in events if name != "stage"] == ["done"]
        streamed = events[-1][1]
        assert normal["predictions"] == streamed["predictions"]
        assert normal["ensemble"] == streamed["ensemble"]
        assert normal["score_kind"] == ("rank" if method == "rank_mean" else "probability")
        assert [row["peptide"] for row in normal["predictions"]] == [
            pair["peptide"] for pair in inputs["pairs"]
        ]
        runs = endpoint(app, "/api/models/{model_id}/inference-runs", "GET")(record["id"])
        assert len(runs) == 2
        assert all(run["request"] == inputs for run in runs)
        csv = endpoint(app, "/api/inference-runs/{run_id}/csv", "GET")(
            normal["run_id"]
        ).body.decode()
        assert len(csv.strip().splitlines()) == 5
        if method == "rank_mean":
            assert "rank" in csv
        # Manifest mutation is detected against the immutable registered digest.
        manifest = read_json(output / "ensemble.json")
        write_json(output / "ensemble.json", {**manifest, "weights": [1, 0]})
        assert not app.state.models.readiness(record["id"])["ready"]
    finally:
        app.state.runner.stop()
        app.state.models.close()


def test_runner_stamps_only_training_time_unchanged_identity(tmp_path, monkeypatch):
    from pepplatform.web.ensemble_runner import WebExperimentRunner
    from pepplatform.web.runner import ExperimentRunner

    request, service, _, paths = make_request(tmp_path)
    experiment = service.store.get_experiment(request["members"][0]["experiment_id"])
    bundle = Path(experiment["artifact_dir"])
    (bundle / PROVENANCE_FILE).unlink()
    runner = object.__new__(WebExperimentRunner)
    runner.store, runner.paths = service.store, paths
    monkeypatch.setattr(
        ExperimentRunner, "_register_model_if_present", lambda *args: {"supports_inference": True}
    )
    runner._register_model_if_present(experiment["id"], experiment)
    assert not (bundle / PROVENANCE_FILE).exists()
    runner._web_training_identity = validation_identity(request["dataset"])
    runner._register_model_if_present(experiment["id"], experiment)
    assert read_json(bundle / PROVENANCE_FILE) == runner._web_training_identity
    (bundle / PROVENANCE_FILE).unlink()
    path = Path(request["dataset"]["root_path"]) / "val/seqs.fasta"
    path.write_text(path.read_text().replace("FFFF", "YYYY"))
    runner._register_model_if_present(experiment["id"], experiment)
    assert not (bundle / PROVENANCE_FILE).exists()
