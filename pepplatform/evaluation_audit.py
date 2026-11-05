"""Fail-closed consistency checks for saved evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pepplatform.objectives import ObjectiveSpec, search_definition, checkpoint_improved

from pepplatform.evaluators import (
    Evaluator,
    OperatingPoint,
    evaluation_protocol_errors,
)


METRIC_KEYS = [
    "acc",
    "sacc",
    "f1",
    "dual_score",
    "auc",
    "aupr",
    "auc_macro",
    "auc_micro",
    "aupr_macro",
    "aupr_micro",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "mcc",
]
PROTOCOL_KEYS = (
    "evaluation_protocol_version",
    "checkpoint_metric",
    "feature_fit_scope",
    "evaluation_mode",
)


def _same_value(left: Any, right: Any, tolerance: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return bool(np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance))
    except (TypeError, ValueError, OverflowError):
        return False


def _read_json(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(f"{label} is missing")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        errors.append(f"{label} cannot be read: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label} is not a mapping")
        return None
    return payload


def _operating_point_errors(
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
    tolerance: float,
) -> tuple[list[str], OperatingPoint | None]:
    errors: list[str] = []
    try:
        metadata_point = OperatingPoint.from_dict(metadata.get("operating_point") or {})
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"metadata operating_point is invalid: {exc}")
        metadata_point = None
    try:
        evaluation_point = OperatingPoint.from_dict(evaluation.get("operating_point") or {})
    except (TypeError, ValueError, OverflowError) as exc:
        errors.append(f"evaluation operating_point is invalid: {exc}")
        evaluation_point = None
    if metadata_point is None or evaluation_point is None:
        return errors, evaluation_point

    exact_fields = (
        "source",
        "objective",
        "score_kind",
        "label_names",
        "fallback_labels",
        "protocol_version",
        "objective_definition",
        "search_config",
    )
    for field in exact_fields:
        if getattr(metadata_point, field) != getattr(evaluation_point, field):
            errors.append(f"metadata/evaluation operating_point.{field} mismatch")
    if len(metadata_point.values) != len(evaluation_point.values) or not np.allclose(
        metadata_point.values,
        evaluation_point.values,
        rtol=tolerance,
        atol=tolerance,
    ):
        errors.append("metadata/evaluation operating_point.values mismatch")

    expected_threshold_source = metadata_point.to_dict()["threshold_source"]
    if metadata.get("threshold_source") != expected_threshold_source:
        errors.append("metadata.threshold_source does not match the operating-point objective")
    if metadata_point.is_binary:
        try:
            runtime_threshold = float(metadata["threshold"])
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append("metadata.threshold is missing or invalid")
        else:
            if not np.isfinite(runtime_threshold):
                errors.append("metadata.threshold is not finite")
            elif not np.isclose(
                runtime_threshold,
                metadata_point.threshold,
                rtol=tolerance,
                atol=tolerance,
            ):
                errors.append("metadata.threshold does not match operating_point")
    else:
        try:
            runtime_thresholds = tuple(float(value) for value in metadata["thresholds"])
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append("metadata.thresholds is missing or invalid")
        else:
            if not np.all(np.isfinite(runtime_thresholds)):
                errors.append("metadata.thresholds contains a non-finite value")
            elif len(runtime_thresholds) != len(metadata_point.values) or not np.allclose(
                runtime_thresholds,
                metadata_point.values,
                rtol=tolerance,
                atol=tolerance,
            ):
                errors.append("metadata.thresholds does not match operating_point")
        if metadata_point.label_names:
            expected = dict(zip(metadata_point.label_names, metadata_point.values))
            actual = metadata.get("threshold_by_label")
            if not isinstance(actual, dict) or set(actual) != set(expected):
                errors.append("metadata.threshold_by_label does not match operating_point labels")
            else:
                for label, value in expected.items():
                    if not _same_value(actual.get(label), value, tolerance):
                        errors.append(
                            "metadata.threshold_by_label does not match operating_point values"
                        )
                        break
            if (
                tuple(str(value) for value in metadata.get("label_names", []))
                != metadata_point.label_names
            ):
                errors.append("metadata.label_names does not match operating_point.label_names")
    return errors, evaluation_point


def _validated_arrays(
    arrays: Any,
    split: str,
    errors: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    labels_key, scores_key = f"{split}_labels", f"{split}_scores"
    if labels_key not in arrays or scores_key not in arrays:
        errors.append(f"{split} score arrays are missing or incomplete")
        return None
    try:
        labels = np.asarray(arrays[labels_key])
        scores = np.asarray(arrays[scores_key], dtype=np.float64)
    except (OSError, EOFError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"{split} score arrays are invalid: {exc}")
        return None
    try:
        labels_finite = bool(np.all(np.isfinite(labels.astype(np.float64))))
    except (TypeError, ValueError, OverflowError):
        labels_finite = False
    if not labels_finite:
        errors.append(f"{split}_labels contains a non-finite or non-numeric value")
        return None
    if not np.all(np.isfinite(scores)):
        errors.append(f"{split}_scores contains a non-finite value")
        return None
    return labels, scores


def audit_artifact(
    evaluation_path: str | Path,
    tolerance: float = 1e-7,
    *,
    allowed_checkpoint_metrics: set[str] | frozenset[str] | None = None,
    allowed_evaluation_modes: set[str] | frozenset[str] | None = None,
    allowed_threshold_objectives: set[str] | frozenset[str] | None = None,
    metric_reference: dict | None = None,
) -> list[str]:
    """Audit metadata, recorded metrics, and persisted score arrays as one artifact.

    Optional allow-lists are call-scoped. Threshold refitting uses only the
    explicit objective and search definitions persisted in the artifact.
    """
    evaluation_path = Path(evaluation_path)
    metadata_path = evaluation_path.with_name("metadata.json")
    score_path = evaluation_path.with_name("evaluation_scores.npz")
    errors: list[str] = []
    if metric_reference is not None:
        from pepplatform.metric_migration import file_sha256
        from pepplatform.pr_metrics import has_current_aupr
        if not has_current_aupr(metric_reference):
            return ["metric reference has an unsupported PR-area definition"]
        sources = metric_reference.get("source_files", {})
        for path in (evaluation_path, metadata_path, score_path):
            try:
                if sources.get(str(path.resolve())) != file_sha256(path):
                    errors.append(f"metric reference input identity mismatch: {path}")
            except OSError as exc:
                errors.append(f"metric reference input unavailable: {exc}")
        if errors:
            return errors

    evaluation = _read_json(evaluation_path, "evaluation.json", errors)
    metadata = _read_json(metadata_path, "metadata.json", errors)
    protocol_options = {
        "allowed_checkpoint_metrics": allowed_checkpoint_metrics,
        "allowed_evaluation_modes": allowed_evaluation_modes,
        "allowed_threshold_objectives": allowed_threshold_objectives,
    }
    if evaluation is not None:
        errors.extend(
            f"evaluation: {error}"
            for error in evaluation_protocol_errors(evaluation, **protocol_options)
        )
    if metadata is not None:
        errors.extend(
            f"metadata: {error}"
            for error in evaluation_protocol_errors(metadata, **protocol_options)
        )

    operating_point = None
    if evaluation is not None and metadata is not None:
        for key in PROTOCOL_KEYS:
            if metadata.get(key) != evaluation.get(key):
                errors.append(f"metadata/evaluation {key} mismatch")
        for key in ("selection_contract_version", "selection_contract"):
            if metadata.get(key) != evaluation.get(key):
                errors.append(f"metadata/evaluation {key} mismatch")
        if metadata.get("selection_contract_version") is not None:
            try:
                if metadata["selection_contract_version"] != 1:
                    raise ValueError("unknown selection contract version")
                contract = metadata["selection_contract"]
                if contract["version"] != 1:
                    raise ValueError("unknown selection definition version")
                for stage in ("checkpoint", "threshold", "tuning", "checkpoint_threshold"):
                    spec = ObjectiveSpec.parse(contract[stage])
                    if stage in {"threshold", "checkpoint_threshold"}:
                        spec.validate_threshold()
                if contract["threshold"] != (metadata.get("operating_point") or {}).get(
                    "objective_definition"
                ):
                    raise ValueError("threshold definition differs from selection contract")
                search = contract["search"]
                if search != search_definition(search["profile"], search["max_sweeps"]):
                    raise ValueError("threshold search definition mismatch")
                if search != (metadata.get("operating_point") or {}).get("search_config"):
                    raise ValueError("operating point search differs from selection contract")
                checkpoint_search = contract["checkpoint_search"]
                if checkpoint_search != search_definition(
                    checkpoint_search["profile"], checkpoint_search["max_sweeps"]
                ):
                    raise ValueError("checkpoint threshold search definition mismatch")
            except (ValueError, TypeError, KeyError) as exc:
                errors.append(f"invalid selection contract: {exc}")
        point_errors, operating_point = _operating_point_errors(metadata, evaluation, tolerance)
        errors.extend(point_errors)
    elif evaluation is not None:
        try:
            operating_point = OperatingPoint.from_dict(evaluation.get("operating_point") or {})
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f"evaluation operating_point is invalid: {exc}")

    if not score_path.exists():
        errors.append("evaluation_scores.npz is missing")
        return errors
    history_path = evaluation_path.with_name("selection_history.json")
    if (
        metadata
        and metadata.get("selection_contract_version") == 1
        and metadata.get("selection_contract", {}).get("checkpoint_scope")
        in {"encoder", "final_model"}
        and not history_path.is_file()
    ):
        errors.append("selection_history.json is missing for a trained checkpoint")
    if metadata and metadata.get("selection_contract_version") == 1 and history_path.is_file():
        try:
            history = json.loads(history_path.read_text())
            if history.get("selection_contract") != metadata.get("selection_contract"):
                raise ValueError("selection history definition differs from metadata")
            spec = ObjectiveSpec.parse(metadata["selection_contract"]["checkpoint"])
            best, best_epoch = None, None
            for index, row in enumerate(history["epochs"], 1):
                if row["epoch"] != index:
                    raise ValueError("selection history epochs are not consecutive")
                key = spec.key(row["metrics"])
                stored = row["selection_key"]
                if len(key) != len(stored) or not np.allclose(key, stored, atol=tolerance, rtol=0):
                    raise ValueError("selection history key differs from metrics")
                improved = checkpoint_improved(key, best, history["early_stop_min_delta"])
                if "improved" in row and bool(row["improved"]) != improved:
                    raise ValueError("selection history decision differs from comparison rule")
                if improved:
                    best, best_epoch = key, index
            if not best_epoch or metadata.get("selected_epoch") != best_epoch:
                raise ValueError("selected epoch differs from selection history")
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(f"invalid selection history: {exc}")
    try:
        arrays_context = np.load(score_path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        errors.append(f"evaluation_scores.npz cannot be read: {exc}")
        return errors

    with arrays_context as arrays:
        validation = _validated_arrays(arrays, "validation", errors)
        if validation is not None and operating_point is not None:
            validation_labels, validation_scores = validation
            try:
                search = operating_point.search_config
                if search is not None and search != search_definition(
                    search["profile"], search["max_sweeps"]
                ):
                    raise ValueError("threshold search definition mismatch")
                if operating_point.objective_definition is None:
                    raise ValueError(
                        "operating_point.objective_definition is required for threshold audit"
                    )
                if search is None:
                    raise ValueError(
                        "operating_point.search_config is required for threshold audit"
                    )
                fitted_point = Evaluator.fit_operating_point(
                    validation_labels,
                    validation_scores,
                    score_kind=operating_point.score_kind,
                    label_names=operating_point.label_names,
                    objective=operating_point.objective_definition,
                    search_profile=search["profile"],
                    max_sweeps=search["max_sweeps"],
                )
                if len(fitted_point.values) != len(operating_point.values) or not np.allclose(
                    fitted_point.values,
                    operating_point.values,
                    rtol=tolerance,
                    atol=tolerance,
                ):
                    errors.append(
                        "operating_point thresholds do not match validation-fitted thresholds"
                    )
                if fitted_point.fallback_labels != operating_point.fallback_labels:
                    errors.append("operating_point fallback_labels do not match validation fitting")
            except (TypeError, ValueError, OverflowError, KeyError, OSError) as exc:
                errors.append(f"cannot fit validation operating point: {exc}")

        splits = ["validation"]
        test_array_present = any(key in arrays for key in ["test_labels", "test_scores"])
        if evaluation is not None and ("test" in evaluation or test_array_present):
            splits.append("test")
        for split in splits:
            values = (
                validation if split == "validation" else _validated_arrays(arrays, split, errors)
            )
            if values is None or operating_point is None or evaluation is None:
                continue
            if not isinstance(evaluation.get(split), dict):
                errors.append(f"{split} metrics are missing")
                continue
            try:
                recomputed = Evaluator.score_metrics(values[0], values[1], operating_point)
            except (TypeError, ValueError, OverflowError) as exc:
                errors.append(f"cannot recompute {split} metrics: {exc}")
                continue
            recorded = evaluation.get(split) or {}
            if metric_reference is not None:
                from pepplatform.pr_metrics import is_aupr_key
                replacement = (metric_reference.get("splits", {}).get(split) or {}).get("metrics")
                if replacement is None:
                    errors.append(f"metric reference does not contain {split}")
                    continue
                recorded = {**recorded, **{k: v for k, v in replacement.items() if is_aupr_key(k)}}
            if recorded.get("averaging") != recomputed.get("averaging"):
                errors.append(
                    f"{split}.averaging mismatch: recorded={recorded.get('averaging')!r}, "
                    f"recomputed={recomputed.get('averaging')!r}"
                )
            for key in METRIC_KEYS:
                if not _same_value(recorded.get(key), recomputed.get(key), tolerance):
                    errors.append(
                        f"{split}.{key} mismatch: recorded={recorded.get(key)!r}, "
                        f"recomputed={recomputed.get(key)!r}"
                    )
    return errors
