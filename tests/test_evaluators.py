import numpy as np
import pytest

from pepplatform.evaluators import (
    EVALUATION_PROTOCOL_VERSION,
    Evaluator,
    OperatingPoint,
    evaluation_protocol_errors,
)


def test_binary_score_metrics_include_paper_metrics():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.7, 0.9])

    metrics = Evaluator.binary_score_metrics(y_true, scores, threshold=0.5)

    assert metrics["acc"] == 0.75
    assert metrics["sacc"] == 0.75
    assert metrics["f1"] == 0.8
    assert metrics["auc"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 0.5
    assert round(metrics["mcc"], 6) == 0.577350


def test_multilabel_score_metrics_keep_sample_accuracy_and_macro_metrics():
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    scores = np.array([[0.8, 0.5], [0.4, 0.5], [0.9, 0.7]])

    metrics = Evaluator.nn_score_metrics(y_true, scores)

    assert metrics["acc"] == 2 / 3
    assert metrics["sacc"] == pytest.approx(5 / 6)
    assert metrics["precision"] == pytest.approx(5 / 6)
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["mcc"] == 0.5
    assert metrics["averaging"] == "macro"
    assert metrics["auc"] == metrics["auc_macro"]
    assert metrics["aupr"] == metrics["aupr_macro"]
    assert metrics["auc_micro"] is not None
    assert metrics["aupr_micro"] is not None


def test_test_metrics_use_frozen_validation_thresholds():
    val_y = np.array([0, 0, 1, 1])
    val_scores = np.array([0.10, 0.20, 0.70, 0.80])
    test_y = np.array([0, 0, 1, 1])
    test_scores = np.array([0.60, 0.65, 0.66, 0.67])

    operating_point = Evaluator.fit_operating_point(val_y, val_scores)
    frozen = Evaluator.score_metrics(test_y, test_scores, operating_point)
    leaked, leaked_point = Evaluator.fit_and_score(test_y, test_scores)

    assert operating_point.threshold != leaked_point.threshold
    assert frozen["f1"] < leaked["f1"]
    assert frozen["operating_point"]["threshold_source"] == "validation_f1"
    assert frozen["evaluation_protocol_version"] == EVALUATION_PROTOCOL_VERSION


def test_multilabel_operating_point_has_one_validation_threshold_per_label():
    y_true = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    scores = np.array([[0.1, 0.8], [0.2, 0.7], [0.8, 0.3], [0.9, 0.2]])

    point = Evaluator.fit_operating_point(y_true, scores, label_names=["a", "b"])
    metrics = Evaluator.score_metrics(y_true, scores, point)

    assert len(point.values) == 2
    assert point.to_dict()["threshold_by_label"] == {"a": point.values[0], "b": point.values[1]}
    assert metrics["f1"] == 1.0


def test_constant_validation_label_records_threshold_fallback():
    y_true = np.zeros((3, 1), dtype=np.int32)
    scores = np.array([[0.1], [0.2], [0.3]])

    point = Evaluator.fit_operating_point(y_true, scores)

    assert point.threshold == 0.5
    assert point.fallback_labels == (0,)
    assert Evaluator.score_metrics(y_true, scores, point)["auc"] is None


def test_protocol_metadata_requires_explicit_v3_operating_point_fields():
    payload = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_metric": "validation_roc_auc",
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": OperatingPoint(values=(0.4,)).to_dict(),
    }

    assert (
        evaluation_protocol_errors(
            payload,
            allowed_checkpoint_metrics={"validation_roc_auc"},
            allowed_evaluation_modes={"independent_iid"},
            require_binary_operating_point=True,
        )
        == []
    )

    missing_checkpoint = dict(payload)
    missing_checkpoint.pop("checkpoint_metric")
    assert "checkpoint_metric is missing or not approved" in evaluation_protocol_errors(
        missing_checkpoint,
        allowed_checkpoint_metrics={"validation_roc_auc"},
    )

    missing_objective = dict(payload)
    missing_objective["operating_point"] = dict(payload["operating_point"])
    missing_objective["operating_point"].pop("objective")
    assert "operating_point.objective is missing" in evaluation_protocol_errors(
        missing_objective,
        allowed_checkpoint_metrics={"validation_roc_auc"},
    )

    wrong_objective = dict(payload)
    wrong_objective["operating_point"] = dict(payload["operating_point"])
    wrong_objective["operating_point"]["objective"] = "accuracy"
    assert "operating_point.objective is not approved" in evaluation_protocol_errors(
        wrong_objective,
        allowed_checkpoint_metrics={"validation_roc_auc"},
    )

    mismatched_threshold = dict(payload)
    mismatched_threshold["operating_point"] = dict(payload["operating_point"])
    mismatched_threshold["operating_point"]["threshold"] = 0.9
    assert "operating_point.threshold does not match values" in evaluation_protocol_errors(
        mismatched_threshold,
        allowed_checkpoint_metrics={"validation_roc_auc"},
    )
