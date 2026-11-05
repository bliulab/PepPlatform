import json
import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from pepplatform.evaluators import Evaluator, OperatingPoint
from pepplatform.pr_metrics import binary_aupr, metric_definition, without_legacy_aupr


SCHEMA_VERSION = 6
MAX_CURVE_POINTS = 300
MAX_THRESHOLD_POINTS = 101
SCORE_DISTRIBUTION_BINS = 20
MAX_PROJECTION_POINTS = 1000
MAX_IMPORTANCE_ITEMS = 20
MAX_PERMUTATION_FEATURES = 200
MAX_PERMUTATION_ROWS = 512
PERMUTATION_REPEATS = 3
IMPORTANCE_METHOD_LABELS = {
    "built_in": "Built-in importance",
    "permutation": "Permutation importance",
    "shap_mean_abs": "SHAP mean(|value|)",
}
GENERIC_IMPORTANCE_LABELS = {"", "value", "importance"}


def empty_analysis():
    return {
        "schema_version": SCHEMA_VERSION,
        "curves": {},
        "metrics": {},
        "projection_2d": {"available": False, "reason": "not generated", "points": []},
        "projection_3d": {"available": False, "reason": "not generated", "points": []},
        "clusters": {"available": False, "reason": "not generated"},
        "feature_importance": {"available": False, "reason": "not generated", "items": []},
        "split_importance": {},
        "shap_importance": {},
        "warnings": [],
    }


def analysis_path(artifact_dir):
    return Path(artifact_dir) / "analysis.json"


def _normalize_importance_payload(payload, default_method=None):
    if not isinstance(payload, dict):
        return payload
    payload = dict(payload)
    method = str(payload.get("method") or default_method or "").strip()
    method_label = str(payload.get("method_label") or "").strip()
    method_label_lower = method_label.lower()
    if not method:
        if method_label_lower.startswith("shap"):
            method = "shap_mean_abs"
        elif method_label_lower == "permutation importance":
            method = "permutation"
        elif "built" in method_label_lower:
            method = "built_in"
    if method:
        payload["method"] = method
    expected_label = IMPORTANCE_METHOD_LABELS.get(method)
    if expected_label and method_label_lower in GENERIC_IMPORTANCE_LABELS:
        payload["method_label"] = expected_label
    return payload


def _normalize_importance_map(mapping, default_method=None):
    if not isinstance(mapping, dict):
        return {}
    return {
        split: _normalize_importance_payload(payload, default_method=default_method)
        for split, payload in mapping.items()
    }


def _normalize_loaded_analysis(payload):
    base = empty_analysis()
    base.update(payload or {})
    base["feature_importance"] = _normalize_importance_payload(base.get("feature_importance"))
    base["shap_importance"] = _normalize_importance_map(
        base.get("shap_importance"),
        default_method="shap_mean_abs",
    )
    base["split_importance"] = _normalize_importance_map(base.get("split_importance"))
    if base.get("shap_importance"):
        if not base.get("split_importance"):
            base["split_importance"] = dict(base["shap_importance"])
        else:
            for split, shap_payload in base["shap_importance"].items():
                split_payload = base["split_importance"].get(split)
                if isinstance(split_payload, dict):
                    method = str(split_payload.get("method") or "").strip()
                    label = str(split_payload.get("method_label") or "").strip().lower()
                    if not method and label in GENERIC_IMPORTANCE_LABELS:
                        split_payload = _normalize_importance_payload(
                            split_payload,
                            default_method="shap_mean_abs",
                        )
                        base["split_importance"][split] = split_payload
                else:
                    base["split_importance"][split] = shap_payload
    return base


def _multilabel_operating_point(payload, artifact_dir):
    point_payload = (payload.get("metrics") or {}).get("operating_point")
    if point_payload is None:
        evaluation_path = Path(artifact_dir) / "evaluation.json"
        if evaluation_path.exists():
            with evaluation_path.open("r", encoding="utf-8") as handle:
                point_payload = (json.load(handle) or {}).get("operating_point")
    if point_payload is None:
        return None
    point = OperatingPoint.from_dict(point_payload)
    return point if not point.is_binary else None


def _hide_stale_multilabel_diagnostics(payload, reason):
    for split_payload in (payload.get("curves") or {}).values():
        if not isinstance(split_payload, dict):
            continue
        split_payload["confusion_matrix"] = {"available": False, "reason": reason}
        split_payload["threshold_sweep"] = {"available": False, "reason": reason, "points": []}
    warnings = list(payload.get("warnings") or [])
    if reason not in warnings:
        warnings.append(reason)
    payload["warnings"] = warnings
    return payload


def _upgrade_multilabel_analysis(artifact_dir, payload):
    if int(payload.get("schema_version") or 0) >= SCHEMA_VERSION:
        return payload
    point = _multilabel_operating_point(payload, artifact_dir)
    if point is None:
        return payload
    metrics = dict(payload.get("metrics") or {})
    metrics.setdefault("operating_point", point.to_dict())
    payload["metrics"] = metrics
    score_path = Path(artifact_dir) / "evaluation_scores.npz"
    if not score_path.exists():
        return _hide_stale_multilabel_diagnostics(
            payload,
            "multi-label diagnostics require saved evaluation scores",
        )
    try:
        with np.load(score_path) as arrays:
            if "validation_labels" not in arrays or "validation_scores" not in arrays:
                raise ValueError("validation scores are missing")
            test_y = arrays["test_labels"] if "test_labels" in arrays else None
            test_scores = arrays["test_scores"] if "test_scores" in arrays else None
            payload["curves"] = build_curves(
                arrays["validation_labels"],
                arrays["validation_scores"],
                test_y,
                test_scores,
                operating_point=point,
            )
        payload["schema_version"] = SCHEMA_VERSION
        return payload
    except (OSError, TypeError, ValueError) as exc:
        return _hide_stale_multilabel_diagnostics(
            payload,
            f"multi-label diagnostics could not be rebuilt: {exc}",
        )


def load_analysis(artifact_dir):
    path = analysis_path(artifact_dir)
    if not path.exists():
        payload = empty_analysis()
    else:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    payload = _normalize_loaded_analysis(payload)
    payload = _upgrade_multilabel_analysis(artifact_dir, payload)
    from pepplatform.metric_migration import analysis_with_reference, load_metric_reference, recompute_artifact
    try:
        reference = load_metric_reference(artifact_dir)
        if reference is None and (Path(artifact_dir) / "evaluation.json").is_file():
            reference = recompute_artifact(artifact_dir)
        if reference is not None:
            return analysis_with_reference(payload, reference)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        payload.setdefault("warnings", []).append(f"AUPR recalculation unavailable: {exc}")
    return without_legacy_aupr(payload)


def save_analysis(artifact_dir, analysis):
    artifact_dir = str(artifact_dir or "").strip()
    if not artifact_dir:
        return
    os.makedirs(artifact_dir, exist_ok=True)
    payload = empty_analysis()
    payload.update(analysis or {})
    with analysis_path(artifact_dir).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def collect_nn_scores(model, loader, device):
    import torch

    model.eval()
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            scores = torch.sigmoid(model(inputs))
            all_scores.append(scores.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
    if not all_scores:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.vstack(all_targets), np.vstack(all_scores)


def _as_2d(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _downsample_indices(length, max_points):
    if length <= max_points:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, max_points).astype(int))


def _curve_points(x, y, x_key, y_key, max_points=MAX_CURVE_POINTS):
    idx = _downsample_indices(len(x), max_points)
    return [
        {x_key: float(x[i]), y_key: float(y[i])}
        for i in idx
        if np.isfinite(x[i]) and np.isfinite(y[i])
    ]


def _flatten_binary_targets(y_true, scores):
    y_true = _as_2d(y_true).astype(np.int32)
    scores = _as_2d(scores).astype(np.float64)
    if y_true.shape != scores.shape:
        min_rows = min(y_true.shape[0], scores.shape[0])
        min_cols = min(y_true.shape[1], scores.shape[1])
        y_true = y_true[:min_rows, :min_cols]
        scores = scores[:min_rows, :min_cols]
    if y_true.size == 0 or scores.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64), "empty"
    average = "binary" if y_true.shape[1] == 1 else "micro"
    y_flat = y_true.reshape(-1)
    s_flat = scores.reshape(-1)
    mask = np.isfinite(s_flat)
    return y_flat[mask], s_flat[mask], average


def _binary_arrays(y_true, scores):
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    limit = min(y.shape[0], s.shape[0])
    if limit <= 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    y = y[:limit]
    s = s[:limit]
    mask = np.isfinite(y) & np.isfinite(s)
    return (y[mask] >= 0.5).astype(np.int32), s[mask].astype(np.float64)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _metrics_at_threshold(y_true, scores, threshold):
    y, s = _binary_arrays(y_true, scores)
    threshold = float(threshold)
    pred = (s >= threshold).astype(np.int32)
    payload = _metrics_from_predictions(y, pred)
    payload["threshold"] = threshold
    return payload


def _metrics_from_predictions(y_true, predictions):
    y, pred = _binary_arrays(y_true, predictions)
    pred = (pred >= 0.5).astype(np.int32)
    positive = y == 1
    negative = ~positive
    predicted_positive = pred == 1
    predicted_negative = ~predicted_positive

    tp = int(np.sum(positive & predicted_positive))
    fp = int(np.sum(negative & predicted_positive))
    fn = int(np.sum(positive & predicted_negative))
    tn = int(np.sum(negative & predicted_negative))
    total = int(y.shape[0])
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    f1 = _ratio(2 * precision * recall, precision + recall)
    accuracy = _ratio(tp + tn, total)
    balanced_accuracy = (recall + specificity) / 2.0
    mcc_denominator = float(
        np.sqrt(np.prod(np.asarray([tp + fp, tp + fn, tn + fp, tn + fn], dtype=np.float64)))
    )
    mcc = _ratio(tp * tn - fp * fn, mcc_denominator)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
    }


def _threshold_candidates(scores, selected_threshold=None):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.array([], dtype=np.float64)
    score_min = float(np.min(finite))
    score_max = float(np.max(finite))
    if score_min >= 0.0 and score_max <= 1.0:
        lower, upper = 0.0, 1.0
    else:
        lower, upper = score_min, score_max
        if lower == upper:
            padding = max(0.5, abs(lower) * 0.05)
            lower -= padding
            upper += padding
    base = np.linspace(lower, upper, MAX_THRESHOLD_POINTS)
    extras = [0.5]
    if selected_threshold is not None:
        try:
            selected = float(selected_threshold)
            if np.isfinite(selected):
                extras.append(selected)
        except (TypeError, ValueError):
            pass
    return np.unique(np.concatenate([base, np.asarray(extras, dtype=np.float64)]))


def _best_threshold_by_f1(y_true, scores):
    y, s = _binary_arrays(y_true, scores)
    if y.size == 0 or np.unique(y).size < 2:
        return 0.5
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in _threshold_candidates(s):
        score = _metrics_at_threshold(y, s, threshold)["f1"]
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def _unavailable_diagnostics(reason):
    reason = str(reason)
    return {
        "confusion_matrix": {"available": False, "reason": reason},
        "threshold_sweep": {"available": False, "reason": reason, "points": []},
        "score_distribution": {"available": False, "reason": reason, "bins": []},
    }


def _score_summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def _build_score_distribution(y_true, scores):
    y, s = _binary_arrays(y_true, scores)
    if s.size == 0:
        return {"available": False, "reason": "no scores available", "bins": []}
    score_min = float(np.min(s))
    score_max = float(np.max(s))
    if score_min >= 0.0 and score_max <= 1.0:
        lower, upper = 0.0, 1.0
    else:
        lower, upper = score_min, score_max
        if lower == upper:
            padding = max(0.5, abs(lower) * 0.05)
            lower -= padding
            upper += padding
    edges = np.linspace(lower, upper, SCORE_DISTRIBUTION_BINS + 1)
    positive_scores = s[y == 1]
    negative_scores = s[y == 0]
    positive_counts, _ = np.histogram(positive_scores, bins=edges)
    negative_counts, _ = np.histogram(negative_scores, bins=edges)
    bins = [
        {
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "positive_count": int(positive_counts[index]),
            "negative_count": int(negative_counts[index]),
        }
        for index in range(SCORE_DISTRIBUTION_BINS)
    ]
    return {
        "available": True,
        "bin_count": SCORE_DISTRIBUTION_BINS,
        "score_min": float(lower),
        "score_max": float(upper),
        "positive": _score_summary(positive_scores),
        "negative": _score_summary(negative_scores),
        "bins": bins,
    }


def _build_confusion_matrix(y_true, scores, threshold, threshold_source):
    y, s = _binary_arrays(y_true, scores)
    if s.size == 0:
        return {"available": False, "reason": "no scores available"}
    payload = _metrics_at_threshold(y, s, threshold)
    payload.update({"available": True, "threshold_source": str(threshold_source or "split_f1")})
    return payload


def _build_threshold_sweep(y_true, scores, selected_threshold, threshold_source):
    y, s = _binary_arrays(y_true, scores)
    if s.size == 0:
        return {"available": False, "reason": "no scores available", "points": []}
    points = [
        {
            key: value
            for key, value in _metrics_at_threshold(y, s, threshold).items()
            if key in {"threshold", "accuracy", "precision", "recall", "specificity", "f1", "mcc"}
        }
        for threshold in _threshold_candidates(s, selected_threshold=selected_threshold)
    ]
    return {
        "available": bool(points),
        "selected_threshold": float(selected_threshold),
        "threshold_source": str(threshold_source or "split_f1"),
        "points": points,
    }


def _build_binary_diagnostics(y_true, scores, operating_threshold=None, threshold_source=None):
    y, s = _binary_arrays(y_true, scores)
    if s.size == 0:
        return _unavailable_diagnostics("no scores available")
    if operating_threshold is None:
        operating_threshold = 0.5
        threshold_source = threshold_source or "fixed_default"
    else:
        threshold_source = threshold_source or "provided"
    return {
        "confusion_matrix": _build_confusion_matrix(y, s, operating_threshold, threshold_source),
        "threshold_sweep": _build_threshold_sweep(y, s, operating_threshold, threshold_source),
        "score_distribution": _build_score_distribution(y, s),
    }


def _aligned_label_score_matrices(y_true, scores):
    y_matrix = _as_2d(y_true).astype(np.int32)
    score_matrix = _as_2d(scores).astype(np.float64)
    min_rows = min(y_matrix.shape[0], score_matrix.shape[0])
    min_cols = min(y_matrix.shape[1], score_matrix.shape[1])
    return y_matrix[:min_rows, :min_cols], score_matrix[:min_rows, :min_cols]


def _build_multilabel_diagnostics(y_true, scores, operating_point):
    y_matrix, score_matrix = _aligned_label_score_matrices(y_true, scores)
    point = OperatingPoint.from_dict(operating_point)
    if y_matrix.size == 0 or score_matrix.size == 0:
        return _unavailable_diagnostics("no scores available")
    if len(point.values) != y_matrix.shape[1]:
        raise ValueError(
            f"operating point has {len(point.values)} thresholds for {y_matrix.shape[1]} labels"
        )
    thresholds = np.asarray(point.values, dtype=np.float64)[None, :]
    finite = np.isfinite(score_matrix)
    y_flat = y_matrix[finite]
    pred_flat = (score_matrix >= thresholds)[finite].astype(np.int32)
    confusion = _metrics_from_predictions(y_flat, pred_flat)
    confusion.update(
        {
            "available": True,
            "average": "micro",
            "threshold_mode": "per_label",
            "threshold_count": len(point.values),
            "threshold_source": point.to_dict()["threshold_source"],
        }
    )
    reason = "multi-label operating point uses per-label thresholds"
    return {
        "confusion_matrix": confusion,
        "threshold_sweep": {"available": False, "reason": reason, "points": []},
        "score_distribution": _build_score_distribution(y_matrix, score_matrix),
    }


def build_curve_bundle(
    y_true,
    scores,
    operating_threshold=None,
    threshold_source=None,
    operating_point=None,
):
    y_matrix, score_matrix = _aligned_label_score_matrices(y_true, scores)
    y_flat, s_flat, average = _flatten_binary_targets(y_true, scores)
    point = OperatingPoint.from_dict(operating_point) if operating_point is not None else None
    if point is not None and not point.is_binary:
        diagnostics = _build_multilabel_diagnostics(y_matrix, score_matrix, point)
    else:
        if point is not None:
            operating_threshold = point.threshold
            threshold_source = point.to_dict()["threshold_source"]
        diagnostics = _build_binary_diagnostics(
            y_flat,
            s_flat,
            operating_threshold=operating_threshold,
            threshold_source=threshold_source,
        )
    counts = {
        "sample_count": int(y_matrix.shape[0]),
        "label_count": int(y_matrix.shape[1]) if y_matrix.ndim == 2 else 0,
        "decision_count": int(y_flat.size),
    }
    if y_flat.size == 0:
        reason = "no scores available"
        return {
            "average": average,
            **counts,
            "roc": {"available": False, "reason": reason, "points": []},
            "pr": {"available": False, "reason": reason, "points": []},
            **diagnostics,
        }
    if np.unique(y_flat).size < 2:
        reason = "split contains only one class"
        return {
            "average": average,
            "positive_count": int(y_flat.sum()),
            **counts,
            "roc": {"available": False, "reason": reason, "points": []},
            "pr": {"available": False, "reason": reason, "points": []},
            **diagnostics,
        }

    fpr, tpr, _ = roc_curve(y_flat, s_flat)
    precision, recall, _ = precision_recall_curve(y_flat, s_flat)
    roc_auc = float(roc_auc_score(y_flat, s_flat))
    aupr = binary_aupr(y_flat, s_flat)
    return {
        **metric_definition(),
        "average": average,
        "positive_count": int(y_flat.sum()),
        **counts,
        "roc": {
            "available": True,
            "auc": roc_auc,
            "points": _curve_points(fpr, tpr, "fpr", "tpr"),
        },
        "pr": {
            **metric_definition(),
            "available": True,
            "aupr": aupr,
            "points": _curve_points(recall[::-1], precision[::-1], "recall", "precision"),
        },
        **diagnostics,
    }


def build_curves(val_y=None, val_scores=None, test_y=None, test_scores=None, operating_point=None):
    curves = {}
    point = OperatingPoint.from_dict(operating_point) if operating_point is not None else None
    if val_y is not None and val_scores is not None:
        if point is None and _as_2d(val_scores).size > 0:
            point = Evaluator.fit_operating_point(val_y, val_scores, score_kind="probability")
        curves["validation"] = build_curve_bundle(
            val_y,
            val_scores,
            operating_point=point,
        )
    if test_y is not None and test_scores is not None:
        curves["test"] = build_curve_bundle(
            test_y,
            test_scores,
            operating_point=point,
        )
    return curves


def build_metric_summaries(val_y, val_scores, test_y=None, test_scores=None, operating_point=None):
    if operating_point is None:
        operating_point = Evaluator.fit_operating_point(val_y, val_scores, score_kind="probability")
    else:
        operating_point = OperatingPoint.from_dict(operating_point)
    result = {
        "operating_point": operating_point.to_dict(),
        "validation": Evaluator.score_metrics(val_y, val_scores, operating_point),
    }
    if test_y is not None and test_scores is not None:
        result["test"] = Evaluator.score_metrics(test_y, test_scores, operating_point)
    return result


def _split_label(row, label_names=None):
    row = np.asarray(row).reshape(-1)
    if row.size == 1:
        return str(int(row[0] >= 0.5))
    names = list(label_names or [])
    positives = []
    for idx, value in enumerate(row):
        if value >= 0.5:
            positives.append(names[idx] if idx < len(names) else f"label_{idx + 1}")
    if not positives:
        return "none"
    if len(positives) <= 2:
        return ",".join(positives)
    return f"{positives[0]},+{len(positives) - 1}"


def _safe_2d_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    return matrix


def _projection_transform(train_x, fallback_x, dimensions):
    if train_x.shape[1] >= 2 and train_x.shape[0] >= 2:
        n_components = min(dimensions, train_x.shape[1], train_x.shape[0] - 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reducer = PCA(n_components=n_components, random_state=42)
            reducer.fit(train_x)
        singular_values = np.asarray(reducer.singular_values_, dtype=np.float64)
        tolerance = (
            np.finfo(np.float64).eps
            * max(train_x.shape)
            * float(singular_values[0] if singular_values.size else 0.0)
        )
        fitted_dimensions = int(np.count_nonzero(singular_values > tolerance))
        if caught or fitted_dimensions <= 0:
            return (
                None,
                "unavailable",
                {
                    "reason": "PCA projection training matrix is numerically degenerate",
                    "fit_scope": "train_only",
                    "requested_dimensions": int(dimensions),
                    "fitted_dimensions": 0,
                    "padded_dimensions": int(dimensions),
                },
            )

        def transform(x):
            z = reducer.transform(x)[:, :fitted_dimensions]
            if fitted_dimensions < dimensions:
                z = np.column_stack([z, np.zeros((z.shape[0], dimensions - fitted_dimensions))])
            return z

        return (
            transform,
            ("pca" if fitted_dimensions == dimensions else "pca_rank_limited"),
            {
                "fit_scope": "train_only",
                "requested_dimensions": int(dimensions),
                "fitted_dimensions": fitted_dimensions,
                "padded_dimensions": int(dimensions - fitted_dimensions),
                "centering": "train_feature_means",
                "implementation": "sklearn.decomposition.PCA",
            },
        )

    if fallback_x.shape[1] >= dimensions:
        return (
            lambda x: x[:, :dimensions],
            f"first_{dimensions}_features",
            {
                "requested_dimensions": int(dimensions),
                "fitted_dimensions": 0,
                "padded_dimensions": 0,
            },
        )
    if fallback_x.shape[1] >= 2:

        def transform(x):
            z = x[:, :2]
            if dimensions > 2:
                z = np.column_stack([z, np.zeros((z.shape[0], dimensions - 2))])
            return z

        return (
            transform,
            "first_two_features",
            {
                "requested_dimensions": int(dimensions),
                "fitted_dimensions": 0,
                "padded_dimensions": max(0, int(dimensions - 2)),
            },
        )
    return (
        lambda x: np.column_stack([x[:, 0], np.zeros((x.shape[0], dimensions - 1))]),
        "single_feature",
        {
            "requested_dimensions": int(dimensions),
            "fitted_dimensions": 0,
            "padded_dimensions": max(0, int(dimensions - 1)),
        },
    )


def _project_rows(train_x, val_x, val_y, test_x=None, test_y=None, dimensions=2):
    train_x = _safe_2d_matrix(train_x)
    val_x = _safe_2d_matrix(val_x)
    test_x = _safe_2d_matrix(test_x) if test_x is not None else None
    if val_x.shape[0] == 0:
        return {
            "available": False,
            "reason": "no validation features",
            "coords": np.empty((0, dimensions)),
            "metadata": [],
        }
    matrices = [train_x, val_x] + ([test_x] if test_x is not None else [])
    if any(not np.isfinite(matrix).all() for matrix in matrices):
        return {
            "available": False,
            "reason": "projection features contain non-finite values",
            "coords": np.empty((0, dimensions)),
            "metadata": [],
        }

    eval_parts = [("validation", val_x, _as_2d(val_y))]
    if test_x is not None and test_x.shape[0] > 0 and test_y is not None:
        eval_parts.append(("test", test_x, _as_2d(test_y)))

    transform, method, method_details = _projection_transform(train_x, val_x, int(dimensions))
    if transform is None:
        return {
            "available": False,
            "reason": str(method_details.get("reason") or "PCA projection unavailable"),
            "coords": np.empty((0, dimensions)),
            "metadata": [],
            "method": method,
            "method_details": method_details,
        }

    coords = []
    metadata = []
    for split, x, y in eval_parts:
        z = transform(x)
        limit = min(z.shape[0], y.shape[0])
        for row_idx in range(limit):
            coords.append(z[row_idx])
            metadata.append((split, y[row_idx]))

    if not coords:
        return {
            "available": False,
            "reason": "no projected rows",
            "coords": np.empty((0, dimensions)),
            "metadata": [],
        }

    coords = np.asarray(coords, dtype=np.float64)
    return {
        "available": True,
        "method": method,
        "method_details": method_details,
        "coords": coords,
        "metadata": metadata,
    }


def _build_cluster_assignments(coords):
    clusters = None
    cluster_info = {"available": False, "reason": "not enough projected rows"}
    if coords.shape[0] >= 2:
        n_clusters = min(5, coords.shape[0])
        try:
            clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(coords)
            cluster_info = {
                "available": True,
                "method": "kmeans",
                "n_clusters": int(n_clusters),
                "source": "projection_2d",
                "dimension": 2,
                "sample_count": int(coords.shape[0]),
            }
        except Exception as exc:
            cluster_info = {"available": False, "reason": str(exc)}
    return clusters, cluster_info


def _projection_payload(projected, dimensions, label_names=None, clusters=None):
    if not projected.get("available"):
        return {
            "available": False,
            "reason": str(projected.get("reason") or "no projection"),
            "points": [],
        }
    coords = np.asarray(projected.get("coords"), dtype=np.float64)
    metadata = list(projected.get("metadata") or [])
    idx = _downsample_indices(coords.shape[0], MAX_PROJECTION_POINTS)
    points = []
    for out_index in idx:
        split, labels = metadata[int(out_index)]
        point = {
            "x": float(coords[out_index, 0]),
            "y": float(coords[out_index, 1]),
            "split": split,
            "label": _split_label(labels, label_names),
            "positive_count": int(np.asarray(labels).sum()),
        }
        if dimensions >= 3:
            point["z"] = float(coords[out_index, 2])
        if clusters is not None:
            point["cluster"] = int(clusters[out_index])
        points.append(point)

    return {
        "available": True,
        "method": projected.get("method"),
        "method_details": dict(projected.get("method_details") or {}),
        "points": points,
    }


def build_projections(train_x, val_x, val_y, test_x=None, test_y=None, label_names=None):
    projection_2d = _project_rows(train_x, val_x, val_y, test_x, test_y, dimensions=2)
    projection_3d = _project_rows(train_x, val_x, val_y, test_x, test_y, dimensions=3)
    clusters = None
    cluster_info = {"available": False, "reason": "no projection"}
    if projection_2d.get("available"):
        clusters, cluster_info = _build_cluster_assignments(
            np.asarray(projection_2d.get("coords"), dtype=np.float64)
        )
    elif projection_2d.get("reason"):
        cluster_info = {"available": False, "reason": str(projection_2d.get("reason"))}
    return (
        _projection_payload(projection_2d, 2, label_names, clusters),
        _projection_payload(projection_3d, 3, label_names, clusters),
        cluster_info,
    )


def build_projection_2d(train_x, val_x, val_y, test_x=None, test_y=None, label_names=None):
    projection, _projection_3d, clusters = build_projections(
        train_x, val_x, val_y, test_x, test_y, label_names
    )
    return projection, clusters


def _importance_from_estimator(estimator):
    if estimator is None:
        return None
    if isinstance(estimator, dict):
        if estimator.get("kind") == "sklearn":
            estimator = estimator.get("value")
        else:
            return None
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=np.float64).reshape(-1)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=np.float64)
        return np.abs(coef).mean(axis=0).reshape(-1)
    return None


def _importance_unavailable(reason, method=None, method_label=None, source=None):
    payload = {"available": False, "reason": str(reason), "items": []}
    if method:
        payload["method"] = method
    if method_label:
        payload["method_label"] = method_label
    if source:
        payload["source"] = source
    return payload


def _rank_importance_values(
    values,
    feature_count,
    feature_names=None,
    source="transformed_feature_space",
    method="built_in",
    method_label="Built-in importance",
):
    width = min(int(feature_count or 0), len(values))
    if width <= 0:
        return _importance_unavailable(
            "no feature dimensions", method=method, method_label=method_label, source=source
        )
    values = np.asarray(values[:width], dtype=np.float64)
    names = list(feature_names or [])
    if len(names) < width:
        names.extend(f"feature_{idx + 1}" for idx in range(len(names), width))
    ranked = np.argsort(values)[::-1][:MAX_IMPORTANCE_ITEMS]
    items = [
        {"name": str(names[idx]), "importance": float(values[idx]), "rank": rank}
        for rank, idx in enumerate(ranked, start=1)
        if np.isfinite(values[idx])
    ]
    return {
        "available": bool(items),
        "source": source,
        "method": method,
        "method_label": method_label,
        "items": items,
    }


def build_feature_importance(bundle, feature_count, feature_names=None):
    bundle = bundle or {}
    importances = []
    if bundle.get("estimator_kind") in {"rf", "tree_ensemble"}:
        value = _importance_from_estimator(bundle.get("estimator"))
        if value is not None:
            importances.append(value)
    elif bundle.get("estimator_kind") in {"svm", "probability_ensemble"}:
        for estimator in bundle.get("estimators", []):
            value = _importance_from_estimator(estimator)
            if value is not None:
                importances.append(value)

    if not importances:
        return _importance_unavailable(
            "classifier does not expose feature importances",
            method="built_in",
            method_label="Built-in importance",
            source="transformed_feature_space",
        )

    width = min(int(feature_count or 0), *(len(value) for value in importances))
    values = np.vstack([value[:width] for value in importances]).mean(axis=0)
    return _rank_importance_values(
        values,
        width,
        feature_names,
        source="transformed_feature_space",
        method="built_in",
        method_label="Built-in importance",
    )


def _iter_sklearn_estimators(bundle):
    bundle = bundle or {}
    if bundle.get("estimator_kind") in {"rf", "tree_ensemble"}:
        estimator = bundle.get("estimator")
        if estimator is not None:
            yield estimator
    elif bundle.get("estimator_kind") in {"svm", "probability_ensemble"}:
        for estimator in bundle.get("estimators", []):
            if isinstance(estimator, dict):
                if estimator.get("kind") == "sklearn":
                    value = estimator.get("value")
                    if value is not None:
                        yield value
            elif estimator is not None:
                yield estimator


def _is_xgboost_estimator(estimator):
    module = str(getattr(estimator.__class__, "__module__", ""))
    return hasattr(estimator, "get_booster") and "xgboost" in module.lower()


def _is_lightgbm_estimator(estimator):
    module = str(getattr(estimator.__class__, "__module__", ""))
    return hasattr(estimator, "predict") and "lightgbm" in module.lower()


def _mean_abs_contributions(contributions):
    if isinstance(contributions, list):
        parts = []
        for item in contributions:
            arr = np.asarray(item, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] <= 1:
                continue
            parts.append(arr[:, :-1])
        if not parts:
            return None
        values = np.stack(parts, axis=1)
    else:
        contributions = np.asarray(contributions, dtype=np.float64)
        if contributions.ndim == 2:
            if contributions.shape[1] <= 1:
                return None
            values = contributions[:, :-1]
        elif contributions.ndim == 3:
            if contributions.shape[2] <= 1:
                return None
            values = contributions[:, :, :-1]
        else:
            return None
    return np.abs(values).mean(axis=tuple(range(values.ndim - 1))).reshape(-1)


def _xgboost_mean_abs_shap(estimator, features):
    features = _safe_2d_matrix(features)
    if features.shape[0] == 0 or features.shape[1] == 0:
        return None
    import xgboost as xgb

    contributions = estimator.get_booster().predict(xgb.DMatrix(features), pred_contribs=True)
    return _mean_abs_contributions(contributions)


def _lightgbm_mean_abs_shap(estimator, features):
    features = _safe_2d_matrix(features)
    if features.shape[0] == 0 or features.shape[1] == 0:
        return None
    contributions = estimator.predict(features, pred_contrib=True)
    return _mean_abs_contributions(contributions)


def _is_native_shap_estimator(estimator):
    return _is_xgboost_estimator(estimator) or _is_lightgbm_estimator(estimator)


def _native_mean_abs_shap(estimator, features):
    if _is_xgboost_estimator(estimator):
        return _xgboost_mean_abs_shap(estimator, features)
    if _is_lightgbm_estimator(estimator):
        return _lightgbm_mean_abs_shap(estimator, features)
    return None


def _bundle_has_native_shap_estimator(bundle):
    return any(
        _is_native_shap_estimator(estimator) for estimator in _iter_sklearn_estimators(bundle)
    )


def _build_native_shap_importance(bundle, feature_count, feature_names, features, split_name):
    features = _safe_2d_matrix(features)
    if features.shape[0] == 0:
        return _importance_unavailable(
            f"no {split_name} features",
            method="shap_mean_abs",
            method_label="SHAP mean(|value|)",
            source="transformed_feature_space",
        )
    importances = []
    for estimator in _iter_sklearn_estimators(bundle):
        if not _is_native_shap_estimator(estimator):
            continue
        try:
            value = _native_mean_abs_shap(estimator, features)
        except Exception:
            value = None
        if value is not None:
            importances.append(value)
    if not importances:
        return _importance_unavailable(
            "classifier does not support native SHAP contributions",
            method="shap_mean_abs",
            method_label="SHAP mean(|value|)",
            source="transformed_feature_space",
        )
    width = min(int(feature_count or 0), features.shape[1], *(len(value) for value in importances))
    values = np.vstack([value[:width] for value in importances]).mean(axis=0)
    return _rank_importance_values(
        values,
        width,
        feature_names,
        source="transformed_feature_space",
        method="shap_mean_abs",
        method_label="SHAP mean(|value|)",
    )


def _positive_proba_column(proba, classes):
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        return proba.reshape(-1)
    if proba.shape[1] == 0:
        return np.zeros(proba.shape[0], dtype=np.float64)
    classes_list = list(classes) if classes is not None else []
    if 1 in classes_list:
        return proba[:, classes_list.index(1)]
    if proba.shape[1] > 1:
        return proba[:, 1]
    return np.zeros(proba.shape[0], dtype=np.float64)


def _score_single_estimator(estimator, features):
    if isinstance(estimator, dict):
        if estimator.get("kind") == "constant":
            return np.full(features.shape[0], float(estimator.get("value", 0.0)), dtype=np.float64)
        if estimator.get("kind") == "sklearn":
            estimator = estimator.get("value")
        else:
            return None
    if estimator is None:
        return None
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(features)
        if isinstance(proba, list):
            classes = getattr(estimator, "classes_", [None] * len(proba))
            return np.stack(
                [_positive_proba_column(item, classes[index]) for index, item in enumerate(proba)],
                axis=1,
            )
        return _positive_proba_column(proba, getattr(estimator, "classes_", None))
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(features), dtype=np.float64)
        return scores
    if hasattr(estimator, "predict"):
        return np.asarray(estimator.predict(features), dtype=np.float64)
    return None


def _predict_bundle_scores(bundle, features):
    bundle = bundle or {}
    features = _safe_2d_matrix(features)
    kind = bundle.get("estimator_kind")
    if kind in {"rf", "tree_ensemble"}:
        scores = _score_single_estimator(bundle.get("estimator"), features)
        if scores is None:
            return None
        scores = np.asarray(scores, dtype=np.float64)
        return scores if scores.ndim == 2 else scores[:, None]
    if kind in {"svm", "calibrated_svm", "probability_ensemble"}:
        columns = []
        for estimator in bundle.get("estimators", []):
            scores = _score_single_estimator(estimator, features)
            if scores is None:
                continue
            scores = np.asarray(scores, dtype=np.float64)
            if scores.ndim == 2:
                columns.extend(scores[:, idx] for idx in range(scores.shape[1]))
            else:
                columns.append(scores.reshape(-1))
        if columns:
            return np.stack(columns, axis=1)
    return None


def _permutation_metric(y_true, scores):
    if y_true is None or scores is None:
        return None
    y_flat, s_flat, _ = _flatten_binary_targets(y_true, scores)
    if y_flat.size == 0 or np.unique(y_flat).size < 2:
        return None
    try:
        return float(roc_auc_score(y_flat, s_flat))
    except Exception:
        return None


def _feature_name_at(index, feature_names=None):
    names = list(feature_names or [])
    return str(names[index]) if index < len(names) else f"feature_{index + 1}"


def _permutation_candidate_indices(bundle, features, feature_count):
    width = min(int(feature_count or 0), features.shape[1])
    if width <= 0:
        return np.array([], dtype=np.int64)
    if width <= MAX_PERMUTATION_FEATURES:
        return np.arange(width, dtype=np.int64)

    importances = []
    for estimator in _iter_sklearn_estimators(bundle):
        value = _importance_from_estimator(estimator)
        if value is not None and len(value) >= width:
            importances.append(value[:width])
    if importances:
        values = np.vstack(importances).mean(axis=0)
    else:
        values = np.nanvar(features[:, :width], axis=0)
    ranked = np.argsort(np.asarray(values, dtype=np.float64))[::-1]
    return ranked[:MAX_PERMUTATION_FEATURES].astype(np.int64)


def _build_permutation_importance(
    bundle, feature_count, feature_names, features, targets, split_name
):
    features = _safe_2d_matrix(features)
    targets = _as_2d(targets) if targets is not None else None
    if features.shape[0] == 0:
        return _importance_unavailable(
            f"no {split_name} features",
            method="permutation",
            method_label="Permutation importance",
            source="transformed_feature_space",
        )
    if targets is None or targets.shape[0] == 0:
        return _importance_unavailable(
            f"no {split_name} labels",
            method="permutation",
            method_label="Permutation importance",
            source="transformed_feature_space",
        )
    limit = min(features.shape[0], targets.shape[0])
    features = features[:limit]
    targets = targets[:limit]
    if limit < 2:
        return _importance_unavailable(
            f"not enough {split_name} rows",
            method="permutation",
            method_label="Permutation importance",
            source="transformed_feature_space",
        )
    row_indices = _downsample_indices(limit, MAX_PERMUTATION_ROWS)
    features = features[row_indices]
    targets = targets[row_indices]

    baseline_scores = _predict_bundle_scores(bundle, features)
    baseline = _permutation_metric(targets, baseline_scores)
    if baseline is None:
        return _importance_unavailable(
            f"{split_name} labels cannot produce ROC AUC permutation scores",
            method="permutation",
            method_label="Permutation importance",
            source="transformed_feature_space",
        )

    candidate_indices = _permutation_candidate_indices(bundle, features, feature_count)
    if candidate_indices.size == 0:
        return _importance_unavailable(
            "no feature dimensions",
            method="permutation",
            method_label="Permutation importance",
            source="transformed_feature_space",
        )

    rng = np.random.default_rng(42)
    values = []
    for feature_idx in candidate_indices:
        drops = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = features.copy()
            order = rng.permutation(shuffled.shape[0])
            if shuffled.shape[0] > 1 and np.array_equal(order, np.arange(shuffled.shape[0])):
                order = np.roll(order, 1)
            shuffled[:, feature_idx] = shuffled[order, feature_idx]
            score = _permutation_metric(targets, _predict_bundle_scores(bundle, shuffled))
            if score is not None:
                drops.append(max(0.0, baseline - score))
        values.append(float(np.mean(drops)) if drops else 0.0)

    candidate_names = [_feature_name_at(int(index), feature_names) for index in candidate_indices]
    payload = _rank_importance_values(
        values,
        len(values),
        candidate_names,
        source="transformed_feature_space",
        method="permutation",
        method_label="Permutation importance",
    )
    payload.update(
        {
            "metric": "roc_auc",
            "baseline_score": float(baseline),
            "sample_count": int(features.shape[0]),
            "candidate_count": int(candidate_indices.size),
            "feature_count": int(min(int(feature_count or 0), features.shape[1])),
        }
    )
    return payload


def build_split_importance(
    bundle, feature_count, feature_names=None, val_x=None, val_y=None, test_x=None, test_y=None
):
    def build_split(features, targets, split_name):
        if _bundle_has_native_shap_estimator(bundle):
            return _build_native_shap_importance(
                bundle, feature_count, feature_names, features, split_name
            )
        return _build_permutation_importance(
            bundle, feature_count, feature_names, features, targets, split_name
        )

    result = {}
    if val_x is not None:
        result["validation"] = build_split(val_x, val_y, "validation")
    if test_x is not None:
        result["test"] = build_split(test_x, test_y, "test")
    return result


def _legacy_shap_importance(split_importance):
    return {
        split: payload
        for split, payload in (split_importance or {}).items()
        if payload.get("method") == "shap_mean_abs"
    }


def build_classical_analysis(
    train_x,
    train_y,
    val_x,
    val_y,
    val_scores,
    test_x=None,
    test_y=None,
    test_scores=None,
    estimator_bundle=None,
    feature_names=None,
    warnings=None,
    operating_point=None,
):
    projection_2d, projection_3d, clusters = build_projections(
        train_x, val_x, val_y, test_x, test_y
    )
    feature_count = _safe_2d_matrix(train_x).shape[1]
    split_importance = build_split_importance(
        estimator_bundle,
        feature_count,
        feature_names,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "curves": build_curves(
            val_y, val_scores, test_y, test_scores, operating_point=operating_point
        ),
        "metrics": build_metric_summaries(
            val_y, val_scores, test_y, test_scores, operating_point=operating_point
        ),
        "projection_2d": projection_2d,
        "projection_3d": projection_3d,
        "clusters": clusters,
        "feature_importance": build_feature_importance(
            estimator_bundle, feature_count, feature_names
        ),
        "split_importance": split_importance,
        "shap_importance": _legacy_shap_importance(split_importance),
        "warnings": list(warnings or []),
    }


def build_score_analysis(
    val_y, val_scores, test_y=None, test_scores=None, warnings=None, operating_point=None
):
    return {
        "schema_version": SCHEMA_VERSION,
        "curves": build_curves(
            val_y, val_scores, test_y, test_scores, operating_point=operating_point
        ),
        "metrics": build_metric_summaries(
            val_y, val_scores, test_y, test_scores, operating_point=operating_point
        ),
        "warnings": list(warnings or []),
    }
