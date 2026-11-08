import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pepplatform.analysis import (
    build_classical_analysis,
    build_curve_bundle,
    build_score_analysis,
    load_analysis,
    save_analysis,
)
from pepplatform.evaluators import OperatingPoint


def test_build_curve_bundle_binary_outputs_roc_and_pr():
    bundle = build_curve_bundle(
        np.array([[0], [0], [1], [1]]), np.array([[0.1], [0.3], [0.7], [0.9]])
    )

    assert bundle["average"] == "binary"
    assert bundle["roc"]["available"] is True
    assert bundle["pr"]["available"] is True
    assert bundle["roc"]["auc"] == 1.0
    assert bundle["pr"]["aupr"] == 1.0
    assert bundle["roc"]["points"][0] == {"fpr": 0.0, "tpr": 0.0}
    assert bundle["confusion_matrix"]["available"] is True
    assert bundle["confusion_matrix"]["tp"] == 2
    assert bundle["confusion_matrix"]["tn"] == 2
    assert bundle["threshold_sweep"]["available"] is True
    assert (
        bundle["threshold_sweep"]["selected_threshold"] == bundle["confusion_matrix"]["threshold"]
    )
    assert {"precision", "recall", "f1", "specificity", "mcc"} <= set(
        bundle["threshold_sweep"]["points"][0]
    )
    assert bundle["score_distribution"]["available"] is True
    assert bundle["score_distribution"]["positive"]["count"] == 2
    assert bundle["score_distribution"]["negative"]["count"] == 2
    assert bundle["confusion_matrix"]["threshold"] == 0.5
    assert bundle["confusion_matrix"]["threshold_source"] == "fixed_default"


def test_build_curve_bundle_large_counts_do_not_overflow_mcc():
    negatives = 60000
    positives = 60000
    y_true = np.r_[np.zeros(negatives, dtype=np.int32), np.ones(positives, dtype=np.int32)]
    scores = np.r_[
        np.full(negatives // 2, 0.1, dtype=np.float64),
        np.full(negatives // 2, 0.9, dtype=np.float64),
        np.full(positives // 2, 0.1, dtype=np.float64),
        np.full(positives // 2, 0.9, dtype=np.float64),
    ]

    bundle = build_curve_bundle(y_true, scores, operating_threshold=0.5)

    assert bundle["confusion_matrix"]["available"] is True
    assert bundle["confusion_matrix"]["tp"] == positives // 2
    assert bundle["confusion_matrix"]["fp"] == negatives // 2
    assert np.isfinite(bundle["confusion_matrix"]["mcc"])
    assert bundle["threshold_sweep"]["available"] is True


def test_build_curve_bundle_single_class_is_unavailable():
    bundle = build_curve_bundle(np.array([[1], [1]]), np.array([[0.8], [0.9]]))

    assert bundle["roc"]["available"] is False
    assert bundle["pr"]["available"] is False
    assert "one class" in bundle["roc"]["reason"]
    assert bundle["confusion_matrix"]["available"] is True
    assert bundle["score_distribution"]["available"] is True
    assert bundle["score_distribution"]["positive"]["count"] == 2


def test_build_score_analysis_applies_validation_threshold_to_test_diagnostics():
    analysis = build_score_analysis(
        np.array([[0], [0], [1], [1]]),
        np.array([[0.1], [0.2], [0.7], [0.9]]),
        np.array([[0], [1]]),
        np.array([[0.4], [0.6]]),
    )

    validation = analysis["curves"]["validation"]
    test = analysis["curves"]["test"]
    assert validation["confusion_matrix"]["threshold_source"] == "validation_f1"
    assert test["confusion_matrix"]["threshold_source"] == "validation_f1"
    assert test["confusion_matrix"]["threshold"] == validation["confusion_matrix"]["threshold"]
    assert (
        test["threshold_sweep"]["selected_threshold"]
        == validation["threshold_sweep"]["selected_threshold"]
    )


def test_build_score_analysis_uses_per_label_thresholds_for_multilabel_diagnostics():
    point = OperatingPoint(
        values=(0.8, 0.6),
        label_names=("alpha", "beta"),
        source="validation",
        objective="f1",
    )
    analysis = build_score_analysis(
        np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int32),
        np.array([[0.9, 0.7], [0.8, 0.65], [0.75, 0.55]], dtype=np.float64),
        np.array([[1, 0], [0, 1]], dtype=np.int32),
        np.array([[0.85, 0.7], [0.75, 0.65]], dtype=np.float64),
        operating_point=point,
    )

    validation = analysis["curves"]["validation"]
    test = analysis["curves"]["test"]
    assert validation["sample_count"] == 3
    assert validation["label_count"] == 2
    assert validation["decision_count"] == 6
    assert validation["confusion_matrix"] == {
        "available": True,
        "average": "micro",
        "threshold_mode": "per_label",
        "threshold_count": 2,
        "threshold_source": "validation_f1",
        "tp": 2,
        "fp": 2,
        "fn": 2,
        "tn": 0,
        "total": 6,
        "accuracy": pytest.approx(1 / 3),
        "balanced_accuracy": 0.25,
        "precision": 0.5,
        "recall": 0.5,
        "specificity": 0.0,
        "f1": 0.5,
        "mcc": pytest.approx(-0.5),
    }
    assert "threshold" not in validation["confusion_matrix"]
    assert validation["threshold_sweep"]["available"] is False
    assert "per-label" in validation["threshold_sweep"]["reason"]
    assert test["confusion_matrix"]["threshold_mode"] == "per_label"
    assert test["confusion_matrix"]["threshold_source"] == "validation_f1"


def test_load_analysis_upgrades_stale_multilabel_diagnostics_from_saved_scores(tmp_path):
    point = OperatingPoint(values=(0.8, 0.6), label_names=("alpha", "beta"))
    payload = {
        "schema_version": 4,
        "curves": {
            "validation": {
                "average": "micro",
                "confusion_matrix": {
                    "available": True,
                    "threshold": 0.71,
                    "threshold_source": "validation_micro_f1",
                },
                "threshold_sweep": {"available": True, "selected_threshold": 0.71, "points": []},
            }
        },
        "metrics": {"operating_point": point.to_dict()},
        "warnings": [],
    }
    (tmp_path / "analysis.json").write_text(json.dumps(payload), encoding="utf-8")
    np.savez_compressed(
        tmp_path / "evaluation_scores.npz",
        validation_labels=np.array([[1, 0], [0, 1]], dtype=np.int32),
        validation_scores=np.array([[0.9, 0.7], [0.75, 0.65]], dtype=np.float32),
    )

    loaded = load_analysis(tmp_path)

    assert loaded["schema_version"] == 6
    assert loaded["curves"]["validation"]["confusion_matrix"]["threshold_mode"] == "per_label"
    assert "threshold" not in loaded["curves"]["validation"]["confusion_matrix"]
    assert loaded["curves"]["validation"]["threshold_sweep"]["available"] is False


def test_load_analysis_hides_stale_multilabel_diagnostics_without_saved_scores(tmp_path):
    point = OperatingPoint(values=(0.8, 0.6), label_names=("alpha", "beta"))
    payload = {
        "schema_version": 4,
        "curves": {
            "validation": {
                "average": "micro",
                "confusion_matrix": {"available": True, "threshold": 0.71},
                "threshold_sweep": {"available": True, "selected_threshold": 0.71, "points": []},
            }
        },
        "metrics": {"operating_point": point.to_dict()},
        "warnings": [],
    }
    (tmp_path / "analysis.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_analysis(tmp_path)

    assert loaded["curves"]["validation"]["confusion_matrix"]["available"] is False
    assert loaded["curves"]["validation"]["threshold_sweep"]["available"] is False
    assert any("saved evaluation scores" in warning for warning in loaded["warnings"])


def test_build_classical_analysis_includes_projection_and_importance(tmp_path):
    train_x = np.array([[0, 0, 0], [0, 1, 0.2], [1, 0, 0.8], [1, 1, 1]], dtype=float)
    train_y = np.array([[0], [0], [1], [1]], dtype=int)
    val_x = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    val_y = np.array([[0], [1]], dtype=int)
    clf = LogisticRegression(solver="liblinear").fit(train_x, train_y.ravel())
    val_scores = clf.predict_proba(val_x)[:, 1][:, None]

    analysis = build_classical_analysis(
        train_x,
        train_y,
        val_x,
        val_y,
        val_scores,
        estimator_bundle={
            "estimator_kind": "probability_ensemble",
            "estimators": [{"kind": "sklearn", "value": clf}],
        },
        feature_names=["left", "right", "depth"],
    )
    save_analysis(tmp_path, analysis)

    loaded = load_analysis(tmp_path)
    assert loaded["curves"]["validation"]["roc"]["available"] is True
    assert loaded["projection_2d"]["available"] is True
    assert loaded["projection_3d"]["available"] is True
    assert "z" in loaded["projection_3d"]["points"][0]
    assert loaded["clusters"]["available"] is True
    assert loaded["feature_importance"]["available"] is True
    assert loaded["feature_importance"]["method"] == "built_in"
    assert {item["name"] for item in loaded["feature_importance"]["items"]} <= {
        "left",
        "right",
        "depth",
    }
    assert loaded["split_importance"]["validation"]["available"] is True
    assert loaded["split_importance"]["validation"]["method"] == "permutation"
    assert loaded["split_importance"]["validation"]["method_label"] == "Permutation importance"
    assert {item["name"] for item in loaded["split_importance"]["validation"]["items"]} <= {
        "left",
        "right",
        "depth",
    }
    assert loaded["shap_importance"] == {}
    assert (
        json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))["schema_version"] == 6
    )


def test_build_classical_analysis_handles_low_dimensional_projection_inputs():
    train_x = np.array([[0], [1], [2], [3]], dtype=float)
    train_y = np.array([[0], [0], [1], [1]], dtype=int)
    val_x = np.array([[0], [3]], dtype=float)
    val_y = np.array([[0], [1]], dtype=int)
    val_scores = np.array([[0.1], [0.9]], dtype=float)

    analysis = build_classical_analysis(train_x, train_y, val_x, val_y, val_scores)

    assert analysis["projection_2d"]["available"] is True
    assert analysis["projection_3d"]["available"] is True
    assert analysis["projection_3d"]["points"][0]["z"] == 0.0
    assert analysis["projection_3d"]["method_details"]["padded_dimensions"] == 2


def test_build_classical_analysis_marks_rank_limited_pca_projection():
    train_x = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float)
    train_y = np.array([[0], [1]], dtype=int)
    val_x = np.array([[0.5, 1.0, 1.5]], dtype=float)
    val_y = np.array([[1]], dtype=int)
    val_scores = np.array([[0.8]], dtype=float)

    analysis = build_classical_analysis(train_x, train_y, val_x, val_y, val_scores)

    projection = analysis["projection_3d"]
    assert projection["available"] is True
    assert projection["method"] == "pca_rank_limited"
    assert projection["method_details"] == {
        "fit_scope": "train_only",
        "requested_dimensions": 3,
        "fitted_dimensions": 1,
        "padded_dimensions": 2,
        "centering": "train_feature_means",
        "implementation": "sklearn.decomposition.PCA",
    }


def test_build_classical_analysis_rejects_nonfinite_projection_features():
    train_x = np.array([[0.0, 1.0], [1.0, np.nan], [2.0, 3.0]], dtype=float)
    train_y = np.array([[0], [1], [1]], dtype=int)
    val_x = np.array([[0.5, 1.0]], dtype=float)
    val_y = np.array([[1]], dtype=int)
    val_scores = np.array([[0.8]], dtype=float)

    analysis = build_classical_analysis(train_x, train_y, val_x, val_y, val_scores)

    assert analysis["projection_2d"]["available"] is False
    assert "non-finite" in analysis["projection_2d"]["reason"]


def test_load_analysis_normalizes_legacy_importance_method_labels(tmp_path):
    legacy = {
        "schema_version": 1,
        "feature_importance": {
            "available": True,
            "method": "built_in",
            "method_label": "value",
            "items": [{"name": "left", "importance": 0.2, "rank": 1}],
        },
        "split_importance": {
            "validation": {
                "available": True,
                "method": "permutation",
                "method_label": "value",
                "metric": "roc_auc",
                "items": [{"name": "right", "importance": 0.1, "rank": 1}],
            }
        },
        "shap_importance": {
            "test": {
                "available": True,
                "method_label": "value",
                "items": [{"name": "right", "importance": 0.3, "rank": 1}],
            }
        },
    }
    (tmp_path / "analysis.json").write_text(json.dumps(legacy), encoding="utf-8")

    loaded = load_analysis(tmp_path)

    assert loaded["schema_version"] == 1
    assert loaded["projection_3d"]["available"] is False
    assert loaded["feature_importance"]["method_label"] == "Built-in importance"
    assert loaded["split_importance"]["validation"]["method_label"] == "Permutation importance"
    assert loaded["split_importance"]["test"]["method"] == "shap_mean_abs"
    assert loaded["split_importance"]["test"]["method_label"] == "SHAP mean(|value|)"
    assert loaded["shap_importance"]["test"]["method_label"] == "SHAP mean(|value|)"


def test_build_classical_analysis_includes_rf_permutation_split_importance():
    train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=float)
    train_y = np.array([[0], [0], [0], [1], [1], [1]], dtype=int)
    val_x = np.array([[0.0, 0.0], [2.0, 1.0], [0.0, 1.0], [2.0, 0.0]], dtype=float)
    val_y = np.array([[0], [1], [0], [1]], dtype=int)
    rf = RandomForestClassifier(n_estimators=8, random_state=42).fit(train_x, train_y.ravel())
    val_scores = rf.predict_proba(val_x)[:, 1][:, None]

    analysis = build_classical_analysis(
        train_x,
        train_y,
        val_x,
        val_y,
        val_scores,
        estimator_bundle={"estimator_kind": "rf", "estimator": rf},
        feature_names=["left", "right"],
    )

    assert analysis["split_importance"]["validation"]["available"] is True
    assert analysis["split_importance"]["validation"]["method"] == "permutation"
    assert analysis["split_importance"]["validation"]["metric"] == "roc_auc"
    assert analysis["split_importance"]["validation"]["sample_count"] == 4
    assert {item["name"] for item in analysis["split_importance"]["validation"]["items"]} <= {
        "left",
        "right",
    }


def test_build_classical_analysis_includes_xgboost_shap_for_validation_and_test():
    xgboost = pytest.importorskip("xgboost")

    train_x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    train_y = np.array([[0], [0], [0], [1], [1], [1]], dtype=int)
    val_x = np.array([[0.0, 0.0], [2.0, 1.0]], dtype=float)
    val_y = np.array([[0], [1]], dtype=int)
    test_x = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=float)
    test_y = np.array([[0], [1]], dtype=int)
    clf = xgboost.XGBClassifier(
        n_estimators=8,
        max_depth=2,
        learning_rate=0.5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
    )
    clf.fit(train_x, train_y.ravel())
    val_scores = clf.predict_proba(val_x)[:, 1][:, None]
    test_scores = clf.predict_proba(test_x)[:, 1][:, None]

    analysis = build_classical_analysis(
        train_x,
        train_y,
        val_x,
        val_y,
        val_scores,
        test_x,
        test_y,
        test_scores,
        estimator_bundle={
            "estimator_kind": "probability_ensemble",
            "estimators": [{"kind": "sklearn", "value": clf}],
        },
        feature_names=["left", "right"],
    )

    assert analysis["feature_importance"]["method"] == "built_in"
    assert analysis["split_importance"]["validation"]["available"] is True
    assert analysis["split_importance"]["validation"]["method"] == "shap_mean_abs"
    assert analysis["split_importance"]["validation"]["method_label"] == "SHAP mean(|value|)"
    assert analysis["split_importance"]["test"]["available"] is True
    assert {item["name"] for item in analysis["split_importance"]["test"]["items"]} <= {
        "left",
        "right",
    }
    assert analysis["shap_importance"]["validation"]["method"] == "shap_mean_abs"


def test_build_classical_analysis_includes_lightgbm_shap_for_validation_and_test():
    lightgbm = pytest.importorskip("lightgbm")

    train_x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )
    train_y = np.array([[0], [0], [0], [1], [1], [1]], dtype=int)
    val_x = np.array([[0.0, 0.0], [2.0, 1.0]], dtype=float)
    val_y = np.array([[0], [1]], dtype=int)
    test_x = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=float)
    test_y = np.array([[0], [1]], dtype=int)
    clf = lightgbm.LGBMClassifier(
        n_estimators=8,
        max_depth=2,
        learning_rate=0.5,
        objective="binary",
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        min_child_samples=1,
        min_data_in_bin=1,
    )
    clf.fit(train_x, train_y.ravel())
    val_scores = clf.predict_proba(val_x)[:, 1][:, None]
    test_scores = clf.predict_proba(test_x)[:, 1][:, None]

    analysis = build_classical_analysis(
        train_x,
        train_y,
        val_x,
        val_y,
        val_scores,
        test_x,
        test_y,
        test_scores,
        estimator_bundle={
            "estimator_kind": "probability_ensemble",
            "estimators": [{"kind": "sklearn", "value": clf}],
        },
        feature_names=["left", "right"],
    )

    assert analysis["split_importance"]["validation"]["available"] is True
    assert analysis["split_importance"]["validation"]["method"] == "shap_mean_abs"
    assert analysis["split_importance"]["validation"]["method_label"] == "SHAP mean(|value|)"
    assert analysis["split_importance"]["test"]["available"] is True
    assert {item["name"] for item in analysis["split_importance"]["test"]["items"]} <= {
        "left",
        "right",
    }
