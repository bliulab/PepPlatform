"""Generic probability-mean ensemble support for sequence tasks.

An ensemble artifact is assembled from already-trained member artifacts.  Weights
and the multi-label operating point are selected on the validation split only,
then the fused artifact is sealed for a one-time independent-test evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from pepplatform.objectives import ObjectiveSpec, selection_fields
from pepplatform.analysis import save_analysis
from pepplatform.evaluators import (
    EVALUATION_PROTOCOL_VERSION,
    Evaluator,
    OperatingPoint,
)


ENSEMBLE_METHODS = frozenset({"probability_mean"})
ENSEMBLE_WEIGHT_SOURCES = frozenset({"equal", "validation"})
DEFAULT_GRID_STEP = 0.05


class EnsembleConfigError(ValueError):
    """Raised when an ensemble definition or member artifact is invalid."""


@dataclass(frozen=True)
class ProbabilityMeanEnsemble:
    """Serializable probability-mean ensemble used for inference."""

    members: tuple[dict[str, Any], ...]
    weights: tuple[float, ...]
    method: str = "probability_mean"
    label_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise EnsembleConfigError("ensemble requires at least two members")
        if len(self.members) != len(self.weights):
            raise EnsembleConfigError("member/weight count mismatch")
        if not np.allclose(sum(self.weights), 1.0, rtol=1e-9, atol=1e-9):
            raise EnsembleConfigError("ensemble weights must sum to 1")
        if any(weight < 0 for weight in self.weights):
            raise EnsembleConfigError("ensemble weights must be non-negative")

    def fuse(self, score_matrices: Sequence[np.ndarray]) -> np.ndarray:
        if len(score_matrices) != len(self.weights):
            raise EnsembleConfigError("score matrix count mismatch")
        arrays = [np.asarray(scores, dtype=np.float64) for scores in score_matrices]
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays[1:]):
            raise EnsembleConfigError("member score shapes must match")
        fused = np.zeros(shape, dtype=np.float64)
        for scores, weight in zip(arrays, self.weights):
            fused += weight * scores
        return fused


def _weighted_mean(score_matrices: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    fused = np.zeros_like(np.asarray(score_matrices[0], dtype=np.float64))
    for scores, weight in zip(score_matrices, weights):
        fused += float(weight) * np.asarray(scores, dtype=np.float64)
    return fused


def select_weights(
    member_scores,
    y,
    *,
    label_names,
    method="probability_mean",
    weight_source="equal",
    grid_step=DEFAULT_GRID_STEP,
    objective="macro_f1",
    objective_params=None,
    threshold_objective="macro_f1",
    threshold_objective_params=None,
    search_profile=None,
):
    """New assemblies use one explicit threshold solver for every weight trial."""
    if method not in ENSEMBLE_METHODS or weight_source not in ENSEMBLE_WEIGHT_SOURCES:
        raise EnsembleConfigError("unsupported ensemble method or weight source")
    target = np.asarray(y, dtype=np.int32)
    if target.ndim != 2:
        raise EnsembleConfigError("ensemble labels must be a matrix")
    matrices = [np.asarray(x, dtype=np.float64) for x in member_scores]
    if len(matrices) < 2 or any(x.shape != target.shape for x in matrices):
        raise EnsembleConfigError("member validation labels/shape mismatch")
    if any(not np.all(np.isfinite(x)) for x in matrices):
        raise EnsembleConfigError("member scores must be finite")
    multi = target.shape[1] > 1
    selection = ObjectiveSpec.parse(objective, objective_params, multilabel=multi)
    threshold = ObjectiveSpec.parse(
        threshold_objective, threshold_objective_params, multilabel=multi
    ).validate_threshold()

    def evaluate(weights):
        fused = _weighted_mean(matrices, weights)
        point = Evaluator.fit_operating_point(
            target,
            fused,
            label_names=label_names,
            objective=threshold,
            search_profile=search_profile,
        )
        metrics = Evaluator.score_metrics(target, fused, point)
        return point, metrics, selection.key(metrics)

    grid = []
    if weight_source == "equal":
        weights = tuple(1 / len(matrices) for _ in matrices)
    else:
        if len(matrices) != 2:
            raise EnsembleConfigError(
                "validation weight search currently requires exactly two members"
            )
        step = float(grid_step)
        if not np.isfinite(step) or not 0 < step < 1:
            raise EnsembleConfigError("ensemble_grid_step must be in (0, 1)")
        best = None
        weights = None
        for alpha in np.round(np.arange(0, 1 + step / 2, step), 8):
            trial = (float(alpha), float(1 - alpha))
            point, metrics, key = evaluate(trial)
            grid.append(
                {
                    "alpha_member_0": trial[0],
                    "alpha_member_1": trial[1],
                    "val_f1": metrics["f1"],
                    "val_sacc": metrics["sacc"],
                    "selection_key": list(key),
                    "thresholds": list(point.values),
                }
            )
            if best is None or key > best:
                best, weights = key, trial
    point, metrics, _ = evaluate(weights)
    return {
        "weights": weights,
        "operating_point": point,
        "validation_metrics": metrics,
        "grid": grid,
        "weight_objective": selection.to_dict(),
    }


def ensemble_selection_fields(weight_objective, point):
    weight = ObjectiveSpec.parse(weight_objective)
    threshold = ObjectiveSpec.parse(point.objective_definition)
    config = {
        "task_type": "pep_function" if not point.is_binary else "pep_hlai",
        "model": "rf",
        "checkpoint_objective": weight.id,
        "checkpoint_objective_params": dict(weight.params),
        "threshold_objective": threshold.id,
        "threshold_objective_params": dict(threshold.params),
        "tuning_objective": weight.id,
        "tuning_objective_params": dict(weight.params),
        "threshold_search_profile": point.search_config["profile"],
    }
    fields = selection_fields(config)
    fields["selection_contract"]["weight_selection"] = weight.to_dict()
    return fields


def build_ensemble_artifact(
    *,
    members: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    y,
    fused_scores,
    label_names: Sequence[str],
    artifact_dir: str | Path,
    operating_point: OperatingPoint,
    method: str = "probability_mean",
    weight_source: str = "equal",
    seed: int = 42,
    report_aupr: bool = True,
    test_y=None,
    test_scores=None,
    weight_objective=None,
) -> dict[str, Any]:
    """Write an ensemble model bundle with evaluation metadata."""

    operating_point = OperatingPoint.from_dict(operating_point)
    if operating_point.objective_definition is None or operating_point.search_config is None:
        raise EnsembleConfigError(
            "ensemble artifacts require an explicit objective definition and search config"
        )
    weight_objective = weight_objective or operating_point.objective_definition
    fields = ensemble_selection_fields(weight_objective, operating_point)
    checkpoint_metric = ObjectiveSpec.parse(weight_objective).checkpoint_metric
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.int32)
    fused_scores = np.asarray(fused_scores, dtype=np.float64)
    validation_metrics = Evaluator.score_metrics(y, fused_scores, operating_point)
    evaluation_payload = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_metric": checkpoint_metric,
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": operating_point.to_dict(),
        "validation": validation_metrics,
    }
    evaluation_payload.update(fields)
    arrays = {
        "validation_labels": y,
        "validation_scores": fused_scores,
    }
    test_metrics = None
    if test_y is not None and test_scores is not None:
        test_metrics = Evaluator.score_metrics(test_y, test_scores, operating_point)
        evaluation_payload["test"] = test_metrics
        arrays["test_labels"] = np.asarray(test_y, dtype=np.int32)
        arrays["test_scores"] = np.asarray(test_scores, dtype=np.float64)

    (artifact_dir / "evaluation.json").write_text(
        json.dumps(evaluation_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(artifact_dir / "evaluation_scores.npz", **arrays)

    operating_dict = operating_point.to_dict()
    metadata = {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_metric": checkpoint_metric,
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": operating_dict,
        "args": {
            "task_type": "pep_function",
            "seed": int(seed),
            "ensemble_method": method,
            "ensemble_weight_source": weight_source,
            "report_aupr": bool(report_aupr),
        },
        "bundle_type": "ensemble",
        "label_names": list(label_names),
        "thresholds": list(operating_dict["thresholds"]),
        "threshold_by_label": operating_dict.get("threshold_by_label", {}),
        "threshold_source": operating_dict["threshold_source"],
        "supports_inference": True,
    }
    metadata.update(fields)
    (artifact_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    member_payload = []
    for index, member in enumerate(members):
        member_payload.append(
            {
                "name": str(member.get("name") or f"member_{index + 1}"),
                "artifact_dir": str(member.get("artifact_dir") or ""),
                "model_id": str(member.get("model_id") or ""),
                "weight": float(weights[index]),
            }
        )
    ensemble_payload = {
        "schema_version": 1,
        "method": method,
        "members": member_payload,
        "weights": [float(value) for value in weights],
        "label_names": list(label_names),
    }
    (artifact_dir / "ensemble.json").write_text(
        json.dumps(ensemble_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    model = ProbabilityMeanEnsemble(
        members=tuple(dict(member) for member in member_payload),
        weights=tuple(float(value) for value in weights),
        method=method,
        label_names=tuple(str(value) for value in label_names),
    )
    joblib.dump(model, artifact_dir / "model.joblib")
    save_analysis(
        artifact_dir,
        {
            "metrics": {"operating_point": operating_dict},
            "warnings": ["ensemble artifact; member-level analysis remains in member bundles"],
        },
    )
    return {
        "artifact_dir": str(artifact_dir),
        "operating_point": operating_dict,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }


def evaluate_ensemble_test(
    member_test_scores: Sequence[np.ndarray],
    weights: Sequence[float],
    y_test,
    operating_point: OperatingPoint,
):
    model = ProbabilityMeanEnsemble(
        members=tuple({"artifact_dir": ""} for _ in weights),
        weights=tuple(float(value) for value in weights),
        label_names=tuple(operating_point.label_names),
    )
    fused = model.fuse(member_test_scores)
    return Evaluator.score_metrics(y_test, fused, operating_point), fused
