from pathlib import Path
import sqlite3

from pepplatform.web.store import MetadataStore, generated_agent_experiment_name_records


def _leaderboard_experiment(
    store, tmp_path, dataset, name, metrics, status="succeeded", supports_inference=True
):
    experiment = store.create_experiment(
        name,
        dataset["id"],
        dataset["task_type"],
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn", "epochs": 3},
        tmp_path / f"{name}.log",
        tmp_path / name,
    )
    for event in metrics:
        store.append_metric_event(experiment["id"], event)
    model = store.create_model(
        experiment["id"],
        f"{name} model",
        dataset["task_type"],
        tmp_path / name,
        supports_inference,
        {"bundle_type": "sequence_nn"},
    )
    store.update_experiment(
        experiment["id"],
        status=status,
        model_id=model["id"],
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:02:00+00:00",
    )
    return store.get_experiment(experiment["id"]), model


def test_metadata_store_round_trip(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "demo run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    store.append_metric_event(experiment["id"], {"kind": "validation", "val_f1": 0.45})
    store.append_metric_event(experiment["id"], {"kind": "best", "best_val_f1": 0.5})
    model = store.create_model(
        experiment["id"],
        "demo model",
        "pep_function",
        tmp_path / "artifact",
        True,
        {"bundle_type": "sequence_nn"},
    )
    updated = store.update_experiment(experiment["id"], status="succeeded", model_id=model["id"])

    assert updated["status"] == "succeeded"
    assert updated["metrics"]["summary"]["val_f1"] == 0.45
    assert updated["metrics"]["summary"]["best_val_f1"] == 0.5
    assert store.get_model(model["id"])["supports_inference"] is True
    assert Path(store.get_dataset(dataset["id"])["root_path"]) == tmp_path / "dataset"


def test_metadata_store_compacts_structured_run_stage_snapshots(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "progress run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    store.append_metric_event(
        experiment["id"],
        {
            "kind": "stage",
            "stage": "classifier_fit",
            "status": "running",
            "current": 1,
            "total": 10,
        },
    )
    store.append_metric_event(
        experiment["id"],
        {
            "kind": "stage",
            "stage": "classifier_fit",
            "status": "running",
            "current": 7,
            "total": 10,
        },
    )
    store.append_metric_event(
        experiment["id"],
        {"kind": "stage", "stage": "finalization", "status": "running", "current": 1, "total": 3},
    )

    events = store.get_experiment(experiment["id"])["metrics"]["events"]
    assert [event["stage"] for event in events] == ["classifier_fit", "finalization"]
    assert events[0]["current"] == 7


def test_metadata_store_summarizes_neural_validation_metrics_at_best_epoch(tmp_path):
    from pepplatform.pr_metrics import metric_definition
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "cnn run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "pssm", "model": "cnn"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    store.append_metric_event(
        experiment["id"],
        {**metric_definition(), "kind": "epoch", "epoch": 1, "val_f1": 0.5, "val_auc": 0.9, "val_aupr": 0.7},
    )
    store.append_metric_event(
        experiment["id"],
        {**metric_definition(), "kind": "epoch", "epoch": 2, "val_f1": 0.6, "val_auc": 0.8, "val_aupr": 0.65},
    )
    store.append_metric_event(
        experiment["id"],
        {"kind": "best", "best_epoch": 2, "best_val_f1": 0.6},
    )
    store.append_metric_event(experiment["id"], {"kind": "test", "test_f1": 0.55})

    summary = store.get_experiment(experiment["id"])["metrics"]["summary"]

    assert summary["val_f1"] == 0.6
    assert summary["val_auc"] == 0.8
    assert summary["val_aupr"] == 0.65
    assert summary["test_f1"] == 0.55
    assert summary["kind"] == "test"


def test_metadata_store_uses_auc_for_provisional_pair_checkpoint_summary(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "pair", "pep_hlai", "upload", "pair_split_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "pair run",
        dataset["id"],
        "pep_hlai",
        {"tokenizer": "esm2", "encoder": "ppi_fusion", "model": "mlp"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    store.append_metric_event(
        experiment["id"],
        {"kind": "epoch", "epoch": 1, "val_f1": 0.9, "val_auc": 0.7},
    )
    store.append_metric_event(
        experiment["id"],
        {"kind": "epoch", "epoch": 2, "val_f1": 0.8, "val_auc": 0.95},
    )

    summary = store.get_experiment(experiment["id"])["metrics"]["summary"]

    assert summary["best_epoch"] == 2
    assert summary["best_val_auc"] == 0.95
    assert summary["val_f1"] == 0.8


def test_metadata_store_refreshes_historical_metric_summaries_idempotently(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "historical run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "esm2", "model": "mlp"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    events = [
        {"kind": "epoch", "epoch": 1, "val_f1": 0.7, "val_auc": 0.91, "val_aupr": 0.8},
        {
            "kind": "epoch",
            "epoch": 2,
            "val_f1": 0.8,
            "val_auc": 0.85,
            "val_aupr": 0.75,
            "val_precision": 0.72,
        },
        {"kind": "best", "best_epoch": 2, "best_val_f1": 0.8},
        {"kind": "test", "test_f1": 0.76, "test_auc": 0.88},
    ]
    store.update_experiment(
        experiment["id"],
        metrics_json={
            "events": events,
            "summary": {
                "kind": "test",
                "best_epoch": 2,
                "best_val_f1": 0.8,
                "test_f1": 0.76,
                "test_auc": 0.88,
            },
        },
    )

    first = store.refresh_metric_summaries()
    summary = store.get_experiment(experiment["id"])["metrics"]["summary"]
    second = store.refresh_metric_summaries()

    assert first == {"scanned": 1, "refreshed": 1, "unchanged": 0, "skipped": []}
    assert summary["val_f1"] == 0.8
    assert summary["val_auc"] == 0.85
    assert summary["val_aupr"] is None
    assert store.raw_experiment_metrics(experiment["id"])["events"][1]["val_aupr"] == 0.75
    assert summary["val_precision"] == 0.72
    assert summary["test_f1"] == 0.76
    assert second == {"scanned": 1, "refreshed": 0, "unchanged": 1, "skipped": []}


def test_metadata_store_user_profile_round_trip(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    user = store.create_user("User@Example.com", "hash")

    assert user["display_name"] == "user"
    assert user["preferred_locale"] == "en"
    localized = store.update_user_locale(user["id"], "zh")
    assert localized["preferred_locale"] == "zh"
    updated = store.update_user_profile(user["id"], "Research Lead")
    assert updated["display_name"] == "Research Lead"


def test_metadata_store_adds_profile_columns_to_legacy_sqlite(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            """
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                email_verified INTEGER NOT NULL,
                role TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO users (
                id, email, password_hash, email_verified, role, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("legacy", "legacy@example.com", "hash", 1, "user", "active", "now", "now"),
        )
        connection.commit()
    finally:
        connection.close()

    store = MetadataStore(database)
    user = store.get_user("legacy")

    assert user["display_name"] == "legacy"
    assert "avatar_object_key" not in user
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT avatar_object_key, avatar_content_type, avatar_updated_at FROM users"
        ).fetchone()
    assert row == (None, None, None)


def test_metadata_store_upgrades_legacy_agent_chat_for_experiment_analysis(tmp_path):
    database = tmp_path / "legacy-agent-chat.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            CREATE TABLE agent_chat_conversations (
                id TEXT PRIMARY KEY,
                owner_user_id TEXT,
                title TEXT NOT NULL,
                pinned_at TEXT,
                last_provider_config_id TEXT,
                last_provider TEXT NOT NULL,
                last_model TEXT NOT NULL,
                last_message_preview TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE agent_chat_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                provider_config_id TEXT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                finish_reason TEXT NOT NULL,
                error TEXT NOT NULL,
                usage_json TEXT NOT NULL,
                web_search_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO agent_chat_conversations (
                id, owner_user_id, title, pinned_at, last_provider_config_id,
                last_provider, last_model, last_message_preview, created_at, updated_at
            ) VALUES (
                'legacy-conversation', NULL, 'Legacy chat', NULL, NULL,
                'openai', 'legacy-model', 'Legacy answer', 'now', 'now'
            );
            INSERT INTO agent_chat_messages (
                id, conversation_id, ordinal, role, content, status,
                provider_config_id, provider, model, finish_reason, error,
                usage_json, web_search_json, created_at, updated_at
            ) VALUES (
                'legacy-message', 'legacy-conversation', 1, 'assistant',
                'Legacy answer', 'complete', NULL, 'openai', 'legacy-model',
                'stop', '', '{}', '{}', 'now', 'now'
            );
            """
        )
        connection.commit()
    finally:
        connection.close()

    store = MetadataStore(database)

    legacy_listing = store.list_agent_chat_conversations()
    legacy_detail = store.list_agent_chat_messages("legacy-conversation")
    assert legacy_listing["total"] == 1
    assert legacy_listing["items"][0]["scope"] == "agent"
    assert legacy_detail["messages"][0]["kind"] == "chat"
    assert legacy_detail["messages"][0]["payload"] == {}
    assert legacy_detail["messages"][0]["context_manifest"] == {}

    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "demo run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    provider = store.create_ai_provider_config(
        None,
        "openai",
        "OpenAI",
        "https://example.test/v1",
        "model-a",
        "ciphertext",
        "text",
    )
    turn = store.begin_agent_chat_turn(
        None,
        provider["id"],
        provider["provider"],
        provider["model"],
        "Diagnose this experiment",
        scope="experiment_analysis",
        experiment_id=experiment["id"],
        assistant_kind="report",
    )

    analysis_listing = store.list_agent_chat_conversations(
        scope="experiment_analysis",
        experiment_id=experiment["id"],
    )
    assert analysis_listing["total"] == 1
    assert analysis_listing["items"][0]["id"] == turn["conversation"]["id"]
    assert store.list_agent_chat_conversations()["total"] == 1

    connection = sqlite3.connect(database)
    try:
        conversation_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(agent_chat_conversations)")
        }
        message_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(agent_chat_messages)")
        }
        indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(agent_chat_conversations)")
        }
    finally:
        connection.close()
    assert {"scope", "experiment_id"}.issubset(conversation_columns)
    assert {"kind", "payload_json", "context_manifest_json"}.issubset(message_columns)
    assert "ix_agent_chat_conversations_owner_scope_experiment_updated" in indexes

    store.delete_experiment(experiment["id"])
    assert (
        store.get_agent_chat_conversation(
            turn["conversation"]["id"],
            scope="experiment_analysis",
            experiment_id=experiment["id"],
        )
        is None
    )


def test_metadata_store_masks_historical_profile_models_as_not_inference_ready(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "demo run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "pssm", "model": "rf"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    model = store.create_model(
        experiment["id"],
        "demo model",
        "pep_function",
        tmp_path / "artifact",
        True,
        {
            "bundle_type": "sequence_classical",
            "supports_inference": True,
            "args": {"tokenizer": "kmer", "encoder": "pssm", "model": "rf"},
        },
    )

    stored = store.get_model(model["id"])
    assert stored["supports_inference"] is False
    assert stored["metadata"]["reason_code"] == "profile_features_require_sample_id"


def test_metadata_store_delete_model_clears_experiment_reference(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "demo run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    model = store.create_model(
        experiment["id"], "demo model", "pep_function", tmp_path / "artifact", True
    )
    store.update_experiment(experiment["id"], model_id=model["id"])

    deleted = store.delete_model(model["id"])

    assert deleted["id"] == model["id"]
    assert store.get_model(model["id"]) is None
    assert store.get_experiment(experiment["id"])["model_id"] is None


def test_metadata_store_records_and_deletes_inference_runs(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "demo run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    model = store.create_model(
        experiment["id"], "demo model", "pep_function", tmp_path / "artifact", True
    )
    run = store.create_inference_run(
        model["id"],
        {"sequences": ["ACD"]},
        {
            "model_id": model["id"],
            "labels": ["active"],
            "predictions": [{"sequence": "ACD", "scores": [0.8]}],
        },
    )

    runs = store.list_inference_runs(model["id"])
    assert runs[0]["id"] == run["id"]
    assert runs[0]["input_kind"] == "sequences"
    assert runs[0]["input_count"] == 1
    assert runs[0]["prediction_count"] == 1

    store.delete_model(model["id"])

    assert store.get_inference_run(run["id"]) is None


def test_metadata_store_deletes_single_inference_run(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "demo run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    model = store.create_model(
        experiment["id"], "demo model", "pep_function", tmp_path / "artifact", True
    )
    run = store.create_inference_run(
        model["id"],
        {"sequences": ["ACD"]},
        {
            "model_id": model["id"],
            "labels": ["active"],
            "predictions": [{"sequence": "ACD", "scores": [0.8]}],
        },
    )

    deleted = store.delete_inference_run(run["id"])

    assert deleted["id"] == run["id"]
    assert store.get_inference_run(run["id"]) is None
    assert store.delete_inference_run(run["id"]) is None


def test_metadata_store_counts_and_deletes_datasets(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )

    assert store.count_experiments_for_dataset(dataset["id"]) == 0
    deleted = store.delete_dataset(dataset["id"])

    assert deleted["id"] == dataset["id"]
    assert store.get_dataset(dataset["id"]) is None


def test_metadata_store_counts_dataset_references(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    store.create_experiment(
        "demo run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )

    assert store.count_experiments_for_dataset(dataset["id"]) == 1
    assert store.count_experiments_for_dataset("missing") == 0


def test_metadata_store_persists_structured_experiment_failure(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo",
        "pep_function",
        "upload",
        "sequence_dir",
        tmp_path / "dataset",
    )
    experiment = store.create_experiment(
        "failed run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "psfm", "model": "cnn"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    failure = {
        "schema_version": 1,
        "code": "profile_no_nonself_hits",
        "category": "profile",
        "stage": "profile_generation",
        "message": "no usable homologs",
        "context": {"feature": "psfm", "sample_id": "Train5"},
        "exit_code": 1,
    }

    stored = store.update_experiment(
        experiment["id"],
        status="failed",
        error="no usable homologs",
        failure=failure,
    )

    assert stored["failure"] == failure
    assert stored["error"] == "no usable homologs"


def test_metadata_store_adds_failure_column_to_existing_database(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    store = MetadataStore(database)
    store.engine.dispose()
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE experiments DROP COLUMN failure_json")

    migrated = MetadataStore(database)
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(experiments)")}

    assert "failure_json" in columns
    assert migrated.list_experiments() == []


def test_metadata_store_updates_model_name_without_changing_experiment(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "demo run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    model = store.create_model(
        experiment["id"], "demo model", "pep_function", tmp_path / "artifact", True
    )

    updated = store.update_model_name(model["id"], "GRU kmer/pssm")

    assert updated["name"] == "GRU kmer/pssm"
    assert updated["name_is_custom"] is True
    assert store.get_experiment(experiment["id"])["name"] == "demo run"


def test_metadata_store_updates_experiment_name_and_only_inherited_models(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "MLP + ESM2 正则化增强",
        dataset["id"],
        "pep_function",
        {"tokenizer": "esm2", "encoder": "index", "model": "mlp"},
        tmp_path / "train.log",
        tmp_path / "artifact",
        name_kind="config",
        name_qualifier="balanced",
    )
    inherited = store.create_model(
        experiment["id"], experiment["name"], "pep_function", tmp_path / "inherited", True
    )
    custom = store.create_model(
        experiment["id"], experiment["name"], "pep_function", tmp_path / "custom", True
    )
    store.update_model_name(custom["id"], "Publication model")
    original_updated_at = experiment["updated_at"]

    renamed = store.update_experiment_name(experiment["id"], "Regularized MLP")

    assert renamed["name"] == "Regularized MLP"
    assert renamed["generated_name"] is None
    assert renamed["updated_at"] == original_updated_at
    assert store.get_model(inherited["id"])["name"] == "Regularized MLP"
    assert store.get_model(inherited["id"])["name_is_custom"] is False
    assert store.get_model(custom["id"])["name"] == "Publication model"
    assert store.get_model(custom["id"])["name_is_custom"] is True


def test_generated_agent_experiment_name_records_require_execution_provenance():
    records = generated_agent_experiment_name_records(
        [
            {
                "id": "action-1",
                "type": "create_experiment",
                "lane": "balanced",
                "title": "MLP + ESM2 正则化增强",
                "result": {"experiment_id": "experiment-1"},
            },
            {
                "id": "action-2",
                "type": "create_experiment",
                "lane": "custom-lane",
                "title": "Unlinked title",
            },
        ],
        {},
    )

    assert records == {
        "experiment-1": {
            "names": {
                "MLP + ESM2 正则化增强",
                "Agent balanced: MLP + ESM2 正则化增强",
            },
            "qualifier": "balanced",
        }
    }


def test_metadata_store_round_trips_generated_rerun_metadata(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    source = store.create_experiment(
        "MLP baseline",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        tmp_path / "source.log",
        tmp_path / "source",
    )
    rerun = store.create_experiment(
        "MLP baseline · 重跑",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        tmp_path / "rerun.log",
        tmp_path / "rerun",
        name_kind="rerun",
        name_base="MLP baseline",
        name_base_kind="literal",
        rerun_count=1,
        rerun_source_experiment_id=source["id"],
    )
    model = store.create_model(
        rerun["id"],
        rerun["name"],
        "pep_function",
        tmp_path / "rerun",
        True,
    )

    assert rerun["generated_name"] == {
        "kind": "rerun",
        "base_name": "MLP baseline",
        "base_kind": "literal",
        "count": 1,
        "source_experiment_id": source["id"],
    }
    assert model["name_is_custom"] is False


def test_metadata_store_delete_experiment_removes_linked_models(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    experiment = store.create_experiment(
        "demo run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
        tmp_path / "train.log",
        tmp_path / "artifact",
    )
    model = store.create_model(
        experiment["id"], "demo model", "pep_function", tmp_path / "artifact", True
    )
    store.update_experiment(experiment["id"], status="succeeded", model_id=model["id"])

    deleted = store.delete_experiment(experiment["id"])

    assert deleted["id"] == experiment["id"]
    assert store.get_experiment(experiment["id"]) is None
    assert store.get_model(model["id"]) is None
    assert store.get_dataset(dataset["id"]) is not None


def test_metadata_store_leaderboard_uses_test_metric_before_validation_fallback(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    _leaderboard_experiment(
        store,
        tmp_path,
        dataset,
        "test-backed",
        [
            {"kind": "validation", "val_f1": 0.95},
            {"kind": "test", "test_f1": 0.70, "test_auc": 0.80},
        ],
    )
    _leaderboard_experiment(
        store,
        tmp_path,
        dataset,
        "validation-only",
        [{"kind": "validation", "val_f1": 0.90, "val_auc": 0.77}],
    )
    _leaderboard_experiment(
        store,
        tmp_path,
        dataset,
        "best-only",
        [{"kind": "best", "best_val_f1": 0.65}],
    )

    rows = store.list_leaderboard(metric="f1", split="auto")

    assert [row["experiment"]["name"] for row in rows] == [
        "validation-only",
        "test-backed",
        "best-only",
    ]
    assert rows[0]["score"] == 0.90
    assert rows[0]["score_source"] == "val_f1"
    assert rows[1]["score"] == 0.70
    assert rows[1]["score_source"] == "test_f1"
    assert rows[2]["score_source"] == "best_val_f1"
    assert rows[1]["metrics"]["validation"]["f1"] == 0.95
    assert rows[1]["duration_seconds"] == 120


def test_metadata_store_leaderboard_filters_and_places_missing_scores_last(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    other_dataset = store.create_dataset(
        "ppi", "pep_protein_interaction", "upload", "pair_split_dir", tmp_path / "ppi"
    )
    _leaderboard_experiment(
        store,
        tmp_path,
        dataset,
        "ready",
        [{"kind": "test", "test_auc": 0.88}],
        supports_inference=True,
    )
    _leaderboard_experiment(
        store,
        tmp_path,
        dataset,
        "metrics-only",
        [{"kind": "test", "test_auc": 0.91}],
        supports_inference=False,
    )
    _leaderboard_experiment(
        store,
        tmp_path,
        dataset,
        "missing-score",
        [{"kind": "validation", "val_f1": 0.72}],
        supports_inference=True,
    )
    _leaderboard_experiment(
        store,
        tmp_path,
        other_dataset,
        "other-task",
        [{"kind": "test", "test_auc": 0.99}],
        supports_inference=True,
    )

    ready_rows = store.list_leaderboard(
        task_type="pep_function",
        dataset_id=dataset["id"],
        metric="auc",
        split="test",
        supports_inference=True,
    )
    metrics_only_rows = store.list_leaderboard(metric="auc", supports_inference=False)

    assert [row["experiment"]["name"] for row in ready_rows] == ["ready", "missing-score"]
    assert ready_rows[0]["score"] == 0.88
    assert ready_rows[1]["score"] is None
    assert metrics_only_rows[0]["experiment"]["name"] == "metrics-only"


def test_metadata_store_leaderboard_rejects_unknown_metric(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")

    try:
        store.list_leaderboard(metric="loss")
    except ValueError as exc:
        assert "unsupported leaderboard metric" in str(exc)
    else:
        raise AssertionError("expected unsupported leaderboard metric to fail")
