import json

import numpy as np
import pytest

from pepplatform.analysis import load_analysis
from pepplatform.evaluators import Evaluator, OperatingPoint
from pepplatform.metric_migration import file_sha256, load_metric_reference, recompute_artifact
from pepplatform.pr_metrics import has_current_aupr
from pepplatform.web.store import MetadataStore
from scripts.recompute_aupr_metrics import migrate_experiment


def fixture_experiment(tmp_path, *, scores=True, status="succeeded"):
    store = MetadataStore(tmp_path / "metadata.sqlite3")
    dataset = store.create_dataset("demo", "pep_hlaii", "upload", "pair_split_dir", tmp_path / "data")
    root = tmp_path / "artifact"
    root.mkdir()
    point = OperatingPoint(values=(0.5,)).to_dict()
    old = {"aupr": 5 / 6, "auc": 0.5, "f1": 0.8, "averaging": "binary"}
    (root / "metadata.json").write_text(json.dumps({"operating_point": point, "evaluation_protocol_version": 3}))
    (root / "evaluation.json").write_text(json.dumps({"operating_point": point, "validation": old, "test": old}))
    (root / "model.pt").write_bytes(b"frozen model bytes")
    if scores:
        labels, values = np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7])
        np.savez_compressed(root / "evaluation_scores.npz", validation_labels=labels,
                            validation_scores=values, test_labels=labels, test_scores=values)
    experiment = store.create_experiment("old", dataset["id"], "pep_hlaii", {}, tmp_path / "run.log", root)
    store.update_experiment(experiment["id"], status=status, metrics_json={
        "events": [], "summary": {"test_aupr": 5 / 6, "val_aupr": 5 / 6, "test_auc": 0.5},
    })
    return store, experiment["id"], root


def test_dry_run_then_migration_preserves_frozen_files_and_is_idempotent(tmp_path):
    store, identity, root = fixture_experiment(tmp_path)
    historical = store.raw_experiment_metrics(identity)
    historical["events"] = [
        {"kind": "epoch", "epoch": 1, "val_f1": 0.8, "val_aupr": 5 / 6},
        {"kind": "best", "best_epoch": 1, "best_val_f1": 0.8},
        {"kind": "test", "test_aupr": 5 / 6},
    ]
    historical["summary"]["best_epoch"] = 1
    store.update_experiment(identity, metrics_json=historical)
    originals = {p.name: file_sha256(p) for p in root.iterdir()}
    before = store.raw_experiment_metrics(identity)
    assert store.get_experiment(identity)["metrics"]["summary"]["test_aupr"] is None
    assert migrate_experiment(store, identity, dry_run=True)["status"] == "ready"
    assert not (root / "metrics_v2.json").exists()
    assert store.raw_experiment_metrics(identity) == before
    assert migrate_experiment(store, identity, dry_run=False)["status"] == "updated"
    current = store.get_experiment(identity)["metrics"]
    assert has_current_aupr(current)
    assert current["summary"]["test_aupr"] == pytest.approx(19 / 24)
    assert current["summary"]["test_auc"] == 0.5
    assert json.loads((root / "metric_migration_v2_original.json").read_text())["metrics"] == before
    assert {name: file_sha256(root / name) for name in originals} == originals
    assert load_analysis(root)["curves"]["test"]["pr"]["aupr"] == pytest.approx(19 / 24)
    assert migrate_experiment(store, identity, dry_run=False)["status"] == "already_current"
    store.refresh_metric_summaries()
    summary = store.get_experiment(identity)["metrics"]["summary"]
    assert summary["test_aupr"] == pytest.approx(19 / 24)
    assert summary["val_aupr"] == pytest.approx(19 / 24)
    assert "aupr_unavailable_reason" not in summary
    assert store.refresh_metric_summaries()["refreshed"] == 0


def test_missing_scores_never_promote_historical_ap(tmp_path):
    store, identity, root = fixture_experiment(tmp_path, scores=False)
    result = migrate_experiment(store, identity, dry_run=False)
    assert result["status"] == "scores_unavailable"
    assert store.get_experiment(identity)["metrics"]["summary"]["test_aupr"] is None
    assert store.raw_experiment_metrics(identity)["summary"]["test_auc"] == 0.5
    assert load_analysis(root)["curves"] == {}


def test_active_experiment_is_not_touched(tmp_path):
    store, identity, root = fixture_experiment(tmp_path, status="running")
    before = store.raw_experiment_metrics(identity)
    assert migrate_experiment(store, identity, dry_run=False)["status"] == "active_experiment_skipped"
    assert store.raw_experiment_metrics(identity) == before
    assert not (root / "metrics_v2.json").exists()


def test_reference_rejects_changed_score_evidence(tmp_path):
    _, _, root = fixture_experiment(tmp_path)
    recompute_artifact(root, write=True)
    with (root / "evaluation_scores.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        load_metric_reference(root)


def test_named_web_splits_use_declared_scores_and_same_definition(tmp_path):
    _, _, root = fixture_experiment(tmp_path)
    evaluation = json.loads((root / "evaluation.json").read_text())
    evaluation["test_splits"] = {"warm": {"aupr": 5 / 6, "auc": 0.5}}
    (root / "evaluation.json").write_text(json.dumps(evaluation))
    directory = root / "evaluations" / "warm"
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "scores.npz", labels=[1, 0, 1], scores=[0.9, 0.8, 0.7])
    reference = recompute_artifact(root)
    assert reference["splits"]["warm"]["metrics"]["aupr"] == pytest.approx(19 / 24)
    assert reference["splits"]["warm"]["metrics"]["auc"] == 0.5
    assert has_current_aupr(reference["splits"]["warm"]["curves"]["pr"])


def test_new_log_lines_identify_definition_and_old_lines_do_not():
    from pepplatform.web.metrics import parse_metric_line
    metrics = Evaluator.binary_score_metrics([1, 0, 1], [0.9, 0.8, 0.7])
    line = "RF Val Acc: 0.6667 | SAcc: 0.6667 | F1: 0.8 | AUC: 0.5"
    line += Evaluator.format_extra_metrics(metrics) + " | AUPR: 0.7917"
    assert has_current_aupr(parse_metric_line(line))
    assert not has_current_aupr(parse_metric_line(line.split(" | AUPRMethod")[0]))


def test_auditor_requires_hash_bound_reference_for_historical_pr_metrics(tmp_path):
    from pepplatform.evaluation_audit import audit_artifact
    root = tmp_path / "sealed"
    root.mkdir()
    y, scores = np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7])
    point = Evaluator.fit_operating_point(y, scores)
    old_metrics = Evaluator.score_metrics(y, scores, point)
    old_metrics.update(aupr=5 / 6, aupr_macro=5 / 6, aupr_micro=5 / 6, aupr_by_label=[5 / 6])
    old_metrics.pop("metric_namespace_version")
    old_metrics.pop("aupr_method")
    protocol = {"evaluation_protocol_version": 3, "checkpoint_metric": "validation_roc_auc",
                "feature_fit_scope": "train_only", "evaluation_mode": "independent_iid",
                "operating_point": point.to_dict(), "threshold": point.threshold,
                "threshold_source": point.to_dict()["threshold_source"]}
    (root / "metadata.json").write_text(json.dumps(protocol))
    (root / "evaluation.json").write_text(json.dumps({**protocol, "validation": old_metrics}))
    np.savez_compressed(root / "evaluation_scores.npz", validation_labels=y, validation_scores=scores)
    assert any("aupr" in error for error in audit_artifact(root / "evaluation.json"))
    reference = recompute_artifact(root)
    assert audit_artifact(root / "evaluation.json", metric_reference=reference) == []
    (root / "metadata.json").write_text(json.dumps({**protocol, "feature_fit_scope": "changed"}))
    assert any("identity mismatch" in e for e in audit_artifact(root / "evaluation.json", metric_reference=reference))
