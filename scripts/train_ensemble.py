"""Assemble a generic probability-mean ensemble from trained member artifacts.

Weights and the multi-label operating point are selected on the validation split
only; the independent test split is evaluated once after the definition is
frozen.  Member artifacts must contain ``evaluation_scores.npz`` with
``validation_labels``/``validation_scores`` (the standard platform artifact
format).  Optional test evaluation reuses each member's stored
``evaluations/test.npz`` when present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pepplatform.ensemble import (
    ENSEMBLE_METHODS,
    ENSEMBLE_WEIGHT_SOURCES,
    build_ensemble_artifact,
    select_weights,
)
from pepplatform.objectives import REGISTRY


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_validation(artifact_dir: Path):
    scores_path = artifact_dir / "evaluation_scores.npz"
    if not scores_path.is_file():
        raise SystemExit(f"member artifact is missing evaluation_scores.npz: {artifact_dir}")
    with np.load(scores_path) as arrays:
        if "validation_labels" not in arrays or "validation_scores" not in arrays:
            raise SystemExit(f"member artifact validation scores are missing: {artifact_dir}")
        return (
            np.asarray(arrays["validation_labels"], dtype=np.int32),
            np.asarray(arrays["validation_scores"], dtype=np.float64),
        )


def _load_test(artifact_dir: Path):
    candidates = [
        artifact_dir.parent / "evaluations" / "test.npz",
        artifact_dir / "evaluations" / "test.npz",
        artifact_dir / "test_evaluation_scores.npz",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        with np.load(path) as arrays:
            if "labels" in arrays and "scores" in arrays:
                return (
                    np.asarray(arrays["labels"], dtype=np.int32),
                    np.asarray(arrays["scores"], dtype=np.float64),
                )
            if "test_labels" in arrays and "test_scores" in arrays:
                return (
                    np.asarray(arrays["test_labels"], dtype=np.int32),
                    np.asarray(arrays["test_scores"], dtype=np.float64),
                )
    return None


def _split_members(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_type", default="pep_function")
    parser.add_argument(
        "--members", required=True, help="comma-separated member artifact directories"
    )
    parser.add_argument(
        "--member-model-ids", default="", help="comma-separated registered model ids (Web)"
    )
    parser.add_argument(
        "--ensemble-method",
        choices=sorted(ENSEMBLE_METHODS),
        default="probability_mean",
    )
    parser.add_argument(
        "--ensemble-weight-source",
        choices=sorted(ENSEMBLE_WEIGHT_SOURCES),
        default="validation",
    )
    parser.add_argument("--ensemble-grid-step", type=float, default=0.05)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument(
        "--data-dir", default="datasets/task_peptide_function/data_tppred_benchmark_public"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_objective", choices=list(REGISTRY), default="macro_f1")
    parser.add_argument("--weight_objective_params", default=None)
    parser.add_argument("--threshold_objective", choices=list(REGISTRY), default="macro_f1")
    parser.add_argument("--threshold_objective_params", default=None)
    parser.add_argument("--threshold_search_profile", default=None)
    parser.add_argument("--report-aupr", action="store_true", default=True)
    parser.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="evaluate the independent test split when member test scores exist (default: auto)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.task_type != "pep_function":
        raise SystemExit("ensemble v1 supports task_type=pep_function only")
    member_dirs = [Path(path).resolve() for path in _split_members(args.members)]
    if len(member_dirs) < 2:
        raise SystemExit("at least two member artifacts are required")
    for path in member_dirs:
        if not path.is_dir():
            raise SystemExit(f"member artifact directory does not exist: {path}")

    members = []
    validation = []
    test = []
    label_names = None
    for index, path in enumerate(member_dirs):
        metadata = _read_json(path / "metadata.json")
        member_label_names = metadata.get("label_names") or []
        if member_label_names:
            member_label_names = [str(value) for value in member_label_names]
            if label_names is None:
                label_names = member_label_names
            elif member_label_names != label_names:
                raise SystemExit(f"member label_names mismatch: {path}")
        y_val, scores_val = _load_validation(path)
        if index == 0:
            expected_shape = scores_val.shape
            if label_names is None:
                label_names = [f"label_{i + 1}" for i in range(scores_val.shape[1])]
        elif scores_val.shape != expected_shape or not np.array_equal(y_val, validation[0][0]):
            raise SystemExit(f"member validation labels/shape mismatch: {path}")
        validation.append((y_val, scores_val))
        members.append(
            {
                "name": str(metadata.get("label", path.name)),
                "artifact_dir": str(path),
            }
        )
        test.append(_load_test(path))

    model_ids = _split_members(args.member_model_ids)
    if model_ids and len(model_ids) != len(members):
        raise SystemExit("--member-model-ids count must match --members count")
    for index, model_id in enumerate(model_ids):
        members[index]["model_id"] = model_id

    selected = select_weights(
        [scores for _, scores in validation],
        validation[0][0],
        label_names=label_names,
        method=args.ensemble_method,
        weight_source=args.ensemble_weight_source,
        grid_step=args.ensemble_grid_step,
        objective=args.weight_objective,
        objective_params=args.weight_objective_params,
        threshold_objective=args.threshold_objective,
        threshold_objective_params=args.threshold_objective_params,
        search_profile=args.threshold_search_profile,
    )
    weights = selected["weights"]
    operating_point = selected["operating_point"]
    fused_scores = sum(float(weight) * scores for weight, (_, scores) in zip(weights, validation))

    test_y = None
    test_scores = None
    if args.evaluate_test is False:
        test = [None] * len(test)
    elif all(value is not None for value in test):
        first_test = test[0]
        if any(
            value[0].shape != first_test[0].shape or not np.array_equal(value[0], first_test[0])
            for value in test[1:]
        ):
            raise SystemExit("member test labels/shape mismatch")
        test_y = first_test[0]
        test_scores = sum(float(weight) * scores for weight, (_, scores) in zip(weights, test))
    elif args.evaluate_test is True:
        raise SystemExit("--evaluate-test requested but member test score files are missing")

    result = build_ensemble_artifact(
        members=members,
        weights=weights,
        y=validation[0][0],
        fused_scores=fused_scores,
        label_names=label_names,
        artifact_dir=args.artifact_dir,
        operating_point=operating_point,
        method=args.ensemble_method,
        weight_source=args.ensemble_weight_source,
        seed=args.seed,
        report_aupr=args.report_aupr,
        test_y=test_y,
        test_scores=test_scores,
        weight_objective=selected["weight_objective"],
    )
    print(
        json.dumps(
            {
                "artifact_dir": str(result["artifact_dir"]),
                "weights": list(weights),
                "validation_sacc": result["validation_metrics"]["sacc"],
                "validation_f1": result["validation_metrics"]["f1"],
                "test_sacc": (result["test_metrics"] or {}).get("sacc"),
                "test_f1": (result["test_metrics"] or {}).get("f1"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
