"""The single PR-area definition used by current training and evaluation.

Metric namespace 1 called average precision ``aupr``. Namespace 2 integrates
all empirical PR thresholds with the trapezoidal rule. Model serialization and
frozen operating points retain their existing independent format.
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

METRIC_NAMESPACE_VERSION = 2
AUPR_METHOD = "trapezoidal_pr_auc"


def metric_definition():
    return {"metric_namespace_version": METRIC_NAMESPACE_VERSION, "aupr_method": AUPR_METHOD}


def has_current_aupr(payload):
    return (
        isinstance(payload, dict)
        and payload.get("metric_namespace_version") == METRIC_NAMESPACE_VERSION
        and payload.get("aupr_method") == AUPR_METHOD
    )


def binary_aupr(y_true, scores, *, sample_weight=None):
    """Full-threshold trapezoidal PR area, or None without both classes.

    Do not integrate the decimated display curve. Repeated recall coordinates
    and tied scores are handled by sklearn's threshold grouping; no extra
    endpoints, precision envelope, score rounding or smoothing are applied.
    """
    labels = np.asarray(y_true).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != values.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("AUPR requires aligned binary labels and scores")
    if not np.isfinite(values).all():
        raise ValueError("AUPR scores must be finite")
    effective = labels
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if (
            sample_weight.shape != labels.shape
            or not np.isfinite(sample_weight).all()
            or np.any(sample_weight < 0)
        ):
            raise ValueError("AUPR sample weights must be aligned, finite and nonnegative")
        effective = labels[sample_weight > 0]
    if np.unique(effective).size != 2:
        return None
    precision, recall, _ = precision_recall_curve(
        labels, values, sample_weight=sample_weight, drop_intermediate=False
    )
    return float(auc(recall, precision))


def weighted_aupr(y_true, scores, weights):
    """Evaluate bootstrap weights with the same definition as ordinary AUPR."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("bootstrap weights must be a two-dimensional matrix")
    return np.asarray(
        [np.nan if (value := binary_aupr(y_true, scores, sample_weight=w)) is None else value
         for w in weights],
        dtype=np.float64,
    )


def is_aupr_key(key):
    return isinstance(key, str) and key not in {"aupr_method", "aupr_unavailable_reason", "report_aupr"} and (
        key == "aupr" or key.startswith("aupr_") or key.endswith("_aupr") or "_aupr_" in key
    )


def preserve_legacy_summary(metrics):
    """Keep the original summary before a derived/current summary replaces it."""
    summary = metrics.get("summary") or {}
    if not has_current_aupr(summary) and any(
        is_aupr_key(k) and v is not None for k, v in summary.items()
    ):
        metrics.setdefault("legacy_metric_summary", copy.deepcopy(summary))
    return metrics


def without_legacy_aupr(payload, *, current=False):
    """Return a public view that cannot label unversioned historical AP AUPR.

    This is definition recognition, not an AP implementation. Originals remain
    on disk/in storage. Raw metric events must carry their own definition,
    because one experiment may have both historical and current events.
    """
    if isinstance(payload, list):
        return [without_legacy_aupr(value, current=current) for value in payload]
    if not isinstance(payload, dict):
        return copy.deepcopy(payload)
    if "metric_namespace_version" in payload or "aupr_method" in payload:
        current = has_current_aupr(payload)
    result = {}
    unavailable = False
    for key, value in payload.items():
        if is_aupr_key(key) and not current:
            result[key] = None
            unavailable = True
        elif key == "legacy_metric_summary":
            result[key] = without_legacy_aupr(value)
        elif key == "events" and isinstance(value, list):
            result[key] = [without_legacy_aupr(event) for event in value]
        else:
            result[key] = without_legacy_aupr(value, current=current)
    if unavailable:
        result["aupr_unavailable_reason"] = "historical_metric_requires_score_recalculation"
    return result
