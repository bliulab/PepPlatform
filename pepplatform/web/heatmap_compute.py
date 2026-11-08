"""Deterministic aggregation of frozen scores; no fitting or inference."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from pepplatform.evaluators import Evaluator, OperatingPoint
from pepplatform.pr_metrics import AUPR_METHOD, metric_definition
from .heatmap_data import VERSION, load_evidence, strata


def support(y):
    positive = int(np.sum(y))
    count = int(len(y))
    return {
        "count": count,
        "positive": positive,
        "negative": count - positive,
        "prevalence": positive / count if count else None,
    }


def metrics(y, scores, threshold):
    y, scores = y.reshape(-1, 1), scores.reshape(-1, 1)
    ranking = Evaluator._ranking_metrics(y, scores)
    values = {"auc": ranking["auc"], "aupr": ranking["aupr"]}
    if threshold is not None:
        point = OperatingPoint(values=(float(threshold),), source="validation", objective="frozen")
        values.update(
            {
                k: v
                for k, v in Evaluator.score_metrics(y, scores, point).items()
                if k in ("acc", "f1")
            }
        )
    return values


def compute(context, params):
    y, scores, labels, point, ids, evaluation = load_evidence(context, params)
    cells = []
    result = {
        **metric_definition(),
        "version": VERSION,
        "params": params,
        "sample_count": len(y),
        "experiment_id": context["experiment"]["id"],
        "task_type": context["experiment"]["task_type"],
        "operating_point": point.to_dict() if point else None,
        "definitions": {
            "aupr": AUPR_METHOD,
            "acc": "per_label_binary_accuracy",
            "prediction": "score >= frozen validation threshold",
            "f1_zero_division": 0,
            "ranking_single_class": "unavailable",
            "post_hoc": params["split"] != "validation",
        },
    }
    if params["kind"] == "labels":
        rows, columns = labels, ["acc", "f1", "auc", "aupr"]
        for i, label in enumerate(labels):
            stats = support(y[:, i])
            values = metrics(y[:, i], scores[:, i], point.values[i] if point else None)
            for metric in columns:
                value = values.get(metric)
                reason = (
                    "threshold_missing"
                    if metric in ("acc", "f1") and point is None
                    else "single_class"
                )
                cells.append(
                    {
                        "row": label,
                        "column": metric,
                        "value": value,
                        **stats,
                        "reason": "" if value is not None else reason,
                    }
                )
    elif params["kind"] == "patterns":
        rows = columns = labels
        event = y.astype(bool)
        if params["pattern"] != "truth":
            pred = scores >= np.asarray(point.values)[None, :]
            event = (
                pred
                if params["pattern"] == "prediction"
                else (
                    (y == 1) & ~pred if params["pattern"] == "false_negative" else (y == 0) & pred
                )
            )
        counts = event.astype(np.int64).T @ event.astype(np.int64)
        totals = np.sum(event, axis=0)
        for i, left in enumerate(labels):
            for j, right in enumerate(labels):
                intersection = int(counts[i, j])
                union = int(totals[i] + totals[j] - intersection)
                value = (
                    intersection
                    if params["scale"] == "count"
                    else (intersection / union if union else None)
                )
                cells.append(
                    {
                        "row": left,
                        "column": right,
                        "value": value,
                        "count": len(y),
                        "intersection": intersection,
                        "union": union,
                        "row_support": int(totals[i]),
                        "column_support": int(totals[j]),
                        "reason": "" if value is not None else "empty_union",
                    }
                )
    else:
        groups = defaultdict(list)
        row_counts = Counter()
        for index, (group, length) in enumerate(strata(context, params, y, ids, evaluation)):
            groups[(group, str(length))].append(index)
            row_counts[group] += 1
        rows = sorted(row_counts, key=lambda key: (-row_counts[key], key))
        columns = sorted({key[1] for key in groups}, key=int)
        for (group, length), indices in groups.items():
            target = y[indices, 0]
            stats = support(target)
            values = (
                {}
                if params["metric"] in ("count", "prevalence")
                else metrics(target, scores[indices, 0], point.values[0] if point else None)
            )
            values.update(count=stats["count"], prevalence=stats["prevalence"])
            value = values.get(params["metric"])
            cells.append(
                {
                    "row": group,
                    "column": length,
                    "value": value,
                    **stats,
                    "reason": "" if value is not None else "single_class",
                    "low_support": stats["count"] < 20,
                    "imbalanced_support": min(stats["positive"], stats["negative"]) < 5,
                }
            )
    is_count = (params["kind"] == "patterns" and params["scale"] == "count") or params.get(
        "metric"
    ) == "count"
    result.update(
        rows=rows,
        columns=columns,
        cells=cells,
        color_max=max((c["value"] or 0 for c in cells), default=1) or 1 if is_count else 1,
        count_scale=is_count,
    )
    return result
