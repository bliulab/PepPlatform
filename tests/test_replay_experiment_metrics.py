from scripts.replay_experiment_metrics import (
    replay_all_missing_validation_metrics,
    replay_missing_validation_metrics,
)

from pepplatform.web.metrics import parse_metric_line
from pepplatform.web.store import MetadataStore


def test_replay_repairs_neural_summary_without_retraining(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    log_path = tmp_path / "train.log"
    lines = [
        "Epoch 1/2 | Train Loss: 0.3000 | Val Loss: 0.2500 | Val Acc: 0.4000 | "
        "Val SAcc: 0.5000 | Val F1: 0.5000 | Val AUC: 0.9000 | Val AUPR: 0.7000",
        "Epoch 2/2 | Train Loss: 0.2000 | Val Loss: 0.2400 | Val Acc: 0.4081 | "
        "Val SAcc: 0.5532 | Val F1: 0.5514 | Val AUC: 0.8620 | "
        "Val Balanced Accuracy: 0.7452 | Val Precision: 0.6064 | "
        "Val Recall: 0.5453 | Val Specificity: 0.9452 | Val MCC: 0.5171 | Val AUPR: 0.6533",
        "Finished. Best Val F1: 0.5514 (Epoch 2)",
        "Test | Acc: 0.3577 | SAcc: 0.5100 | F1: 0.5050 | AUC: 0.8602 | AUPR: 0.6200",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    experiment = store.create_experiment(
        "cnn run",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "encoder": "pssm", "model": "cnn"},
        log_path,
        tmp_path / "artifact",
    )
    events = [parse_metric_line(line) for line in lines]
    store.update_experiment(
        experiment["id"],
        metrics_json={
            "events": events,
            "summary": {
                "kind": "test",
                "best_epoch": 2,
                "best_val_f1": 0.5514,
                "test_f1": 0.505,
                "test_auc": 0.8602,
            },
        },
    )

    result = replay_missing_validation_metrics(store, experiment["id"])
    summary = store.get_experiment(experiment["id"])["metrics"]["summary"]

    assert result["status"] == "summary_refreshed"
    assert result["added_metric_events"] == 0
    assert summary["best_epoch"] == 2
    assert summary["val_f1"] == 0.5514
    assert summary["val_auc"] == 0.862
    # A historical log number is AP; no curve can be reconstructed from it.
    assert summary["val_aupr"] is None
    assert summary["val_balanced_accuracy"] == 0.7452
    assert summary["test_f1"] == 0.505
    assert summary["kind"] == "test"


def test_replay_uses_auc_fallback_for_incomplete_pair_log(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "pair", "pep_hlaii", "upload", "pair_split_dir", tmp_path / "dataset"
    )
    log_path = tmp_path / "pair.log"
    lines = [
        "Epoch 1/2 | Train Loss: 0.3000 | Val Loss: 0.2500 | Val Acc: 0.7000 | "
        "Val SAcc: 0.7000 | Val F1: 0.9000 | Val AUC: 0.7000",
        "Epoch 2/2 | Train Loss: 0.2000 | Val Loss: 0.2400 | Val Acc: 0.7500 | "
        "Val SAcc: 0.7500 | Val F1: 0.8000 | Val AUC: 0.9500",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    experiment = store.create_experiment(
        "incomplete pair",
        dataset["id"],
        "pep_hlaii",
        {"tokenizer": "esm2", "encoder": "ppi_fusion", "model": "mlp"},
        log_path,
        tmp_path / "artifact",
    )

    result = replay_missing_validation_metrics(store, experiment["id"])
    summary = store.get_experiment(experiment["id"])["metrics"]["summary"]

    assert result["status"] == "updated"
    assert summary["best_epoch"] == 2
    assert summary["best_val_auc"] == 0.95
    assert summary["val_f1"] == 0.8


def test_replay_all_supports_dry_run_and_continues_when_a_log_is_missing(tmp_path):
    store = MetadataStore(tmp_path / "meta.sqlite3")
    dataset = store.create_dataset(
        "demo", "pep_function", "upload", "sequence_dir", tmp_path / "dataset"
    )
    log_path = tmp_path / "train.log"
    lines = [
        "Epoch 1/1 | Train Loss: 0.2000 | Val Loss: 0.2400 | Val Acc: 0.4081 | "
        "Val SAcc: 0.5532 | Val F1: 0.5514 | Val AUC: 0.8620 | Val AUPR: 0.6533",
        "Finished. Best Val F1: 0.5514 (Epoch 1)",
        "Test | Acc: 0.3577 | SAcc: 0.5100 | F1: 0.5050 | AUC: 0.8602 | AUPR: 0.6200",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    repairable = store.create_experiment(
        "repairable",
        dataset["id"],
        "pep_function",
        {"tokenizer": "esm2", "model": "mlp"},
        log_path,
        tmp_path / "artifact-repairable",
    )
    store.update_experiment(
        repairable["id"],
        status="succeeded",
        metrics_json={
            "events": [parse_metric_line(line) for line in lines],
            "summary": {"best_epoch": 1, "best_val_f1": 0.5514, "test_f1": 0.505},
        },
    )
    missing_log = store.create_experiment(
        "missing log",
        dataset["id"],
        "pep_function",
        {"tokenizer": "kmer", "model": "cnn"},
        tmp_path / "missing.log",
        tmp_path / "artifact-missing",
    )
    store.update_experiment(missing_log["id"], status="failed")

    dry_run = replay_all_missing_validation_metrics(store, dry_run=True)
    dry_run_statuses = {result["experiment_id"]: result["status"] for result in dry_run["results"]}

    assert dry_run["experiment_count"] == 2
    assert dry_run_statuses[repairable["id"]] == "summary_refreshed"
    assert dry_run_statuses[missing_log["id"]] == "experiment_log_not_found"
    assert "val_auc" not in store.get_experiment(repairable["id"])["metrics"]["summary"]

    updated = replay_all_missing_validation_metrics(store)
    updated_statuses = {result["experiment_id"]: result["status"] for result in updated["results"]}
    summary = store.get_experiment(repairable["id"])["metrics"]["summary"]

    assert updated_statuses[repairable["id"]] == "summary_refreshed"
    assert updated_statuses[missing_log["id"]] == "experiment_log_not_found"
    assert summary["val_f1"] == 0.5514
    assert summary["val_auc"] == 0.862
    assert summary["val_aupr"] is None

    repeated = replay_all_missing_validation_metrics(store)
    repeated_statuses = {
        result["experiment_id"]: result["status"] for result in repeated["results"]
    }
    assert repeated_statuses[repairable["id"]] == "validation_summary_already_present"
