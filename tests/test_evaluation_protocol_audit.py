import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from pepplatform.evaluators import EVALUATION_PROTOCOL_VERSION, Evaluator, OperatingPoint
from tools.audit_evaluation_protocol import audit_artifact


def _write_metadata(artifact_dir, evaluation_payload):
    point = OperatingPoint.from_dict(evaluation_payload["operating_point"])
    metadata = {
        key: evaluation_payload[key]
        for key in [
            "evaluation_protocol_version",
            "checkpoint_metric",
            "feature_fit_scope",
            "evaluation_mode",
        ]
    }
    metadata.update(
        {
            "bundle_type": "pair_vector_nn" if point.is_binary else "sequence_nn",
            "operating_point": point.to_dict(),
            "threshold_source": point.to_dict()["threshold_source"],
        }
    )
    if point.is_binary:
        metadata["threshold"] = point.threshold
    else:
        metadata["thresholds"] = list(point.values)
        metadata["label_names"] = list(point.label_names)
        if point.label_names:
            metadata["threshold_by_label"] = dict(zip(point.label_names, point.values))
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def test_protocol_audit_cli_starts_from_repository_root():
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/audit_evaluation_protocol.py", "--help"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert f"saved evaluation files (format {EVALUATION_PROTOCOL_VERSION})" in result.stdout


def test_protocol_audit_recomputes_validation_and_test_metrics(tmp_path):
    val_y = np.array([[0], [0], [1], [1]])
    val_scores = np.array([[0.1], [0.2], [0.8], [0.9]])
    test_y = np.array([[0], [1]])
    test_scores = np.array([[0.3], [0.7]])
    point = Evaluator.fit_operating_point(val_y, val_scores)
    payload = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_metric": "validation_roc_auc",
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": point.to_dict(),
        "validation": Evaluator.score_metrics(val_y, val_scores, point),
        "test": Evaluator.score_metrics(test_y, test_scores, point),
    }
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_metadata(tmp_path, payload)
    np.savez_compressed(
        tmp_path / "evaluation_scores.npz",
        validation_labels=val_y,
        validation_scores=val_scores,
        test_labels=test_y,
        test_scores=test_scores,
    )

    assert audit_artifact(evaluation_path) == []


def test_protocol_audit_refits_dual_primary_operating_point_with_recorded_objective(
    tmp_path,
):
    val_y = np.array(
        [[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]],
        dtype=np.int32,
    )
    val_scores = np.array(
        [
            [0.72, 0.48, 0.22],
            [0.65, 0.58, 0.31],
            [0.42, 0.68, 0.61],
            [0.30, 0.41, 0.76],
            [0.57, 0.49, 0.54],
        ],
        dtype=np.float64,
    )
    point = Evaluator.fit_operating_point(
        val_y,
        val_scores,
        objective="dual_f1_sacc",
        objective_params={"reference_f1": 0.4, "reference_sacc": 0.6},
        label_names=["a", "b", "c"],
    )
    payload = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_metric": "validation_dual_f1_sacc",
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": point.to_dict(),
        "validation": Evaluator.score_metrics(val_y, val_scores, point),
    }
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_metadata(tmp_path, payload)
    np.savez_compressed(
        tmp_path / "evaluation_scores.npz",
        validation_labels=val_y,
        validation_scores=val_scores,
    )

    assert audit_artifact(evaluation_path) == []


def test_protocol_audit_detects_changed_metric(tmp_path):
    y = np.array([[0], [1]])
    scores = np.array([[0.1], [0.9]])
    point = Evaluator.fit_operating_point(y, scores)
    metrics = Evaluator.score_metrics(y, scores, point)
    metrics["f1"] = 0.0
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
                "checkpoint_metric": "validation_roc_auc",
                "feature_fit_scope": "train_only",
                "evaluation_mode": "independent_iid",
                "operating_point": point.to_dict(),
                "validation": metrics,
            }
        ),
        encoding="utf-8",
    )
    _write_metadata(
        tmp_path,
        {
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "checkpoint_metric": "validation_roc_auc",
            "feature_fit_scope": "train_only",
            "evaluation_mode": "independent_iid",
            "operating_point": point.to_dict(),
        },
    )
    np.savez_compressed(
        tmp_path / "evaluation_scores.npz",
        validation_labels=y,
        validation_scores=scores,
    )

    assert any("validation.f1 mismatch" in error for error in audit_artifact(evaluation_path))


def test_protocol_audit_rejects_operating_point_not_fitted_on_validation(tmp_path):
    y = np.array([[0], [0], [1], [1]])
    scores = np.array([[0.1], [0.2], [0.8], [0.9]])
    point = Evaluator.fit_operating_point(y, scores)
    forged = point.to_dict()
    forged["values"] = [0.5]
    forged["threshold"] = 0.5
    payload = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_metric": "validation_roc_auc",
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": forged,
        "validation": Evaluator.score_metrics(y, scores, OperatingPoint.from_dict(forged)),
    }
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_metadata(tmp_path, payload)
    np.savez_compressed(
        tmp_path / "evaluation_scores.npz",
        validation_labels=y,
        validation_scores=scores,
    )

    assert any(
        "thresholds do not match validation-fitted thresholds" in error
        for error in audit_artifact(evaluation_path)
    )


def test_protocol_audit_rejects_metadata_only_threshold_change(tmp_path):
    y = np.array([[0], [0], [1], [1]])
    scores = np.array([[0.1], [0.2], [0.8], [0.9]])
    point = Evaluator.fit_operating_point(y, scores)
    payload = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_metric": "validation_roc_auc",
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": point.to_dict(),
        "validation": Evaluator.score_metrics(y, scores, point),
    }
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    metadata = _write_metadata(tmp_path, payload)
    metadata["threshold"] = 0.5
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez_compressed(
        tmp_path / "evaluation_scores.npz",
        validation_labels=y,
        validation_scores=scores,
    )

    assert "metadata.threshold does not match operating_point" in audit_artifact(evaluation_path)


def test_protocol_audit_rejects_metadata_protocol_mismatch(tmp_path):
    y = np.array([[0], [1]])
    scores = np.array([[0.1], [0.9]])
    point = Evaluator.fit_operating_point(y, scores)
    payload = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_metric": "validation_roc_auc",
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": point.to_dict(),
        "validation": Evaluator.score_metrics(y, scores, point),
    }
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    metadata = _write_metadata(tmp_path, payload)
    metadata["evaluation_mode"] = "inductive"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez_compressed(
        tmp_path / "evaluation_scores.npz",
        validation_labels=y,
        validation_scores=scores,
    )

    assert "metadata/evaluation evaluation_mode mismatch" in audit_artifact(evaluation_path)


def test_protocol_audit_rejects_non_finite_scores(tmp_path):
    y = np.array([[0], [1]])
    fitted_scores = np.array([[0.1], [0.9]])
    point = Evaluator.fit_operating_point(y, fitted_scores)
    payload = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_metric": "validation_roc_auc",
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": point.to_dict(),
        "validation": Evaluator.score_metrics(y, fitted_scores, point),
    }
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_metadata(tmp_path, payload)
    np.savez_compressed(
        tmp_path / "evaluation_scores.npz",
        validation_labels=y,
        validation_scores=np.array([[0.1], [np.nan]]),
    )

    assert "validation_scores contains a non-finite value" in audit_artifact(evaluation_path)


def test_protocol_audit_checkpoint_allowlist_is_call_scoped(tmp_path):
    y = np.array([[0], [1]])
    scores = np.array([[0.1], [0.9]])
    point = Evaluator.fit_operating_point(y, scores)
    payload = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_metric": "validation_loss",
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": point.to_dict(),
        "validation": Evaluator.score_metrics(y, scores, point),
    }
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_metadata(tmp_path, payload)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    metadata["checkpoint_metric"] = "validation_loss"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.savez_compressed(
        tmp_path / "evaluation_scores.npz",
        validation_labels=y,
        validation_scores=scores,
    )

    assert "checkpoint_metric is missing or not approved" in " ".join(
        audit_artifact(evaluation_path)
    )
    assert (
        audit_artifact(
            evaluation_path,
            allowed_checkpoint_metrics={"validation_loss"},
        )
        == []
    )
