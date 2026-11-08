"""Versioned Web fusion with explicit selection and threshold definitions."""

from __future__ import annotations
from pepplatform.task_compat import normalize_task_type


import numpy as np

from pepplatform.evaluators import Evaluator
from pepplatform.pr_metrics import metric_definition
from pepplatform.objectives import ObjectiveSpec, ObjectiveError

TASKS = ("pep_function", "pep_hlai", "pep_hlaii", "pep_protein_interaction")
METHODS = ("probability_mean", "logit_mean", "rank_mean")
WEIGHT_SOURCES = ("equal", "custom", "validation")
CONTRACT_VERSION = 2


class EnsembleError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

    def detail(self):
        return {"code": self.code, "message": str(self)}


def normalize_definition(payload):
    if payload.get("schema_version", CONTRACT_VERSION) != CONTRACT_VERSION:
        raise EnsembleError("ensemble_version_invalid", "Unsupported ensemble definition version")
    task = normalize_task_type(payload.get("task_type"))
    if normalize_task_type(task) not in TASKS:
        raise EnsembleError("ensemble_task_invalid", "Unsupported ensemble task")
    ids = payload.get("member_ids")
    if (
        not isinstance(ids, list)
        or len(ids) < 2
        or any(not isinstance(x, str) or not x.strip() for x in ids)
    ):
        raise EnsembleError("ensemble_members_required", "Select at least two registered models")
    if len(set(ids)) != len(ids):
        raise EnsembleError("ensemble_member_duplicate", "Each member must be selected once")
    method = payload.get("method", "probability_mean")
    source = payload.get("weight_source", "equal")
    if method not in METHODS or source not in WEIGHT_SOURCES:
        raise EnsembleError("ensemble_method_invalid", "Unsupported method or weight source")
    try:
        step = float(payload.get("grid_step", 0.05))
        seed = int(payload.get("seed", 42))
    except (ValueError, TypeError, OverflowError) as exc:
        raise EnsembleError("ensemble_parameters_invalid", "Invalid grid step or seed") from exc
    # An integral grid partitions the entire simplex, including both endpoints.
    if (
        not np.isfinite(step)
        or not 0.01 <= step <= 0.5
        or not np.isclose(1 / step, round(1 / step))
    ):
        raise EnsembleError(
            "ensemble_grid_invalid", "Grid step must divide 1 exactly and lie between 0.01 and 0.5"
        )
    objective = payload.get("objective") or (
        "macro_f1" if normalize_task_type(task) == "pep_function" else "auc"
    )
    if objective not in (
        ("macro_f1", "sacc", "dual_f1_sacc", "auc", "aupr")
        if normalize_task_type(task) == "pep_function"
        else ("auc", "aupr", "f1")
    ):
        raise EnsembleError("ensemble_objective_invalid", "Invalid validation selection objective")
    threshold_id = payload.get("threshold_objective") or (
        "macro_f1" if normalize_task_type(task) == "pep_function" else "f1"
    )
    try:
        spec = ObjectiveSpec.parse(
            objective,
            payload.get("objective_params"),
            multilabel=normalize_task_type(task) == "pep_function",
        )
        threshold_spec = ObjectiveSpec.parse(
            threshold_id,
            payload.get("threshold_objective_params"),
            multilabel=normalize_task_type(task) == "pep_function",
        ).validate_threshold()
    except (ObjectiveError, TypeError, ValueError) as exc:
        raise EnsembleError("ensemble_objective_invalid", str(exc)) from exc
    evaluate = payload.get("evaluate_test", False)
    if not isinstance(evaluate, bool):
        raise EnsembleError("ensemble_test_flag_invalid", "evaluate_test must be a boolean")
    definition = {
        **metric_definition(),
        "schema_version": CONTRACT_VERSION,
        "task_type": task,
        "member_ids": list(ids),
        "method": method,
        "weight_source": source,
        "grid_step": step,
        "objective": objective,
        "seed": seed,
        "evaluate_test": evaluate,
        "objective_params": dict(spec.params),
        "objective_definition": spec.to_dict(),
        "threshold_objective": threshold_spec.id,
        "threshold_objective_params": dict(threshold_spec.params),
        "threshold_definition": threshold_spec.to_dict(),
    }
    if source == "custom":
        raw = payload.get("weights")
        normalized = normalized_weights(raw, len(ids))
        requested = payload.get("requested_weights", raw)
        if not np.allclose(
            normalized_weights(requested, len(ids)), normalized, rtol=1e-12, atol=1e-12
        ):
            raise EnsembleError(
                "ensemble_weights_invalid", "Requested and normalized weights disagree"
            )
        definition["weights"] = normalized.tolist()
        definition["requested_weights"] = [float(x) for x in requested]
    return definition


def normalized_weights(weights, count):
    try:
        values = np.asarray(weights, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise EnsembleError(
            "ensemble_weights_invalid", "Weights must be finite non-negative numbers"
        ) from exc
    if values.shape != (count,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise EnsembleError(
            "ensemble_weights_invalid",
            "Weights must match the members and be finite and non-negative",
        )
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise EnsembleError("ensemble_weights_invalid", "At least one weight must be positive")
    return values / total


def transform_scores(scores, method):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim not in (1, 2) or not len(values) or not np.all(np.isfinite(values)):
        raise EnsembleError(
            "ensemble_scores_invalid", "Member scores must be nonempty finite vectors or matrices"
        )
    if method in ("probability_mean", "logit_mean") and (np.any(values < 0) or np.any(values > 1)):
        raise EnsembleError(
            "ensemble_probability_required", "This fusion requires probability scores in [0, 1]"
        )
    if method == "rank_mean":
        if len(values) < 2:
            raise EnsembleError(
                "ensemble_rank_batch_required",
                "Rank fusion requires at least two rows in a complete batch",
            )
        # Per-column double argsort, exactly as the frozen HLA-I implementation.
        return np.argsort(np.argsort(values, axis=0), axis=0) / (len(values) - 1)
    if method == "logit_mean":
        clipped = np.clip(values, 1e-7, 1 - 1e-7)
        return np.log(clipped) - np.log1p(-clipped)
    if method != "probability_mean":
        raise EnsembleError("ensemble_method_invalid", "Unsupported fusion method")
    return values


def fuse(member_scores, weights, method="probability_mean"):
    if len(member_scores) < 2:
        raise EnsembleError("ensemble_members_required", "At least two members are required")
    matrices = [transform_scores(x, method) for x in member_scores]
    if any(x.shape != matrices[0].shape for x in matrices):
        raise EnsembleError("ensemble_shape_mismatch", "Member score shapes differ")
    weights = normalized_weights(weights, len(matrices))
    result = sum(float(w) * x for w, x in zip(weights, matrices))
    if method == "logit_mean":
        result = 1 / (1 + np.exp(-np.clip(result, -50, 50)))
    return result


def select(member_scores, y, labels, definition, progress=lambda event: None):
    if definition.get("schema_version") != CONTRACT_VERSION:
        raise EnsembleError("ensemble_version_invalid", "Unsupported ensemble definition version")
    task, method = definition["task_type"], definition["method"]
    source, count = definition["weight_source"], len(member_scores)
    score_kind = "rank" if method == "rank_mean" else "probability"
    objective = ObjectiveSpec.parse(definition["threshold_definition"])
    weight_spec = ObjectiveSpec.parse(definition["objective_definition"])
    y = np.asarray(y, dtype=np.int32)
    if any(np.asarray(x).shape != y.shape for x in member_scores) or not np.all(np.isin(y, [0, 1])):
        raise EnsembleError("ensemble_labels_invalid", "Scores must match binary validation labels")
    if normalize_task_type(task) != "pep_function" and len(np.unique(y)) != 2:
        raise EnsembleError(
            "ensemble_auc_unavailable", "Binary validation must contain both classes"
        )

    def point(scores):
        return Evaluator.fit_operating_point(
            y, scores, label_names=labels, score_kind=score_kind, objective=objective
        )

    def key(weights):
        scores = fuse(member_scores, weights, method)
        if normalize_task_type(task) == "pep_function":
            metrics = Evaluator.score_metrics(y, scores, point(scores))
            return weight_spec.key(metrics)
        ranking = Evaluator.ranking_metrics(y, scores)
        auc, aupr = ranking["auc"], ranking["aupr"]
        if definition["objective"] == "f1":
            return weight_spec.key(Evaluator.score_metrics(y, scores, point(scores)))
        return (aupr, auc) if definition["objective"] == "aupr" else (auc, aupr)

    weights = (
        normalized_weights(definition["weights"], count)
        if source == "custom"
        else np.full(count, 1 / count)
    )
    search = []
    if source == "validation":
        grid = np.arange(0, 1 + definition["grid_step"] / 2, definition["grid_step"])
        for sweep in range(5):
            changed = False
            for index in range(count):
                selected, selected_key = weights.copy(), key(weights)
                others = [i for i in range(count) if i != index]
                for alpha in grid:
                    trial = weights.copy()
                    trial[index] = alpha
                    total = weights[others].sum()
                    trial[others] = weights[others] * (1 - alpha) / total if total > 1e-12 else 0
                    if np.any(trial < 0) or not np.isclose(trial.sum(), 1, rtol=1e-9, atol=1e-9):
                        continue
                    trial_key = key(trial)
                    search.append(
                        {
                            "sweep": sweep,
                            "member_index": index,
                            "weights": trial.tolist(),
                            "selection_key": list(trial_key),
                        }
                    )
                    if trial_key > selected_key:
                        selected, selected_key = trial, trial_key
                if not np.allclose(selected, weights, rtol=1e-9, atol=1e-9):
                    changed = True
                    weights = selected
                progress(
                    {
                        "stage": "selecting_weights",
                        "completed": sweep * count + index + 1,
                        "total": 5 * count,
                    }
                )
            if not changed:
                break
        weights = normalized_weights(np.round(weights, 8), count)
    return tuple(float(x) for x in weights), point(fuse(member_scores, weights, method)), search
