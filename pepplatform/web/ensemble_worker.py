"""Subprocess entry point for Web assemblies, independent of paper launchers."""

from __future__ import annotations
from pepplatform.task_compat import normalize_task_type


import argparse
import json
import platform
from pathlib import Path

import numpy as np

from pepplatform.analysis import build_score_analysis, save_analysis
from pepplatform.pr_metrics import metric_definition
from pepplatform.evaluation_audit import audit_artifact
from pepplatform.evaluators import (
    EVALUATION_CHECKPOINT_METRICS,
    EVALUATION_PROTOCOL_VERSION,
    Evaluator,
)
from pepplatform.objectives import ObjectiveSpec
from pepplatform.ensemble import ensemble_selection_fields
from .ensemble_data import (
    PROVENANCE_FILE,
    digest_json,
    file_digest,
    read_json,
    read_split,
    response_scores,
    snapshot_member,
    tree_identity,
    validation_identity,
    verify_provenance,
    write_json,
)
from .ensemble_math import CONTRACT_VERSION, EnsembleError, fuse, select
from .ensemble_service import SnapshotStore
from .inference import ModelService


def progress(event):
    from pepplatform.run_progress import emit_run_progress

    stage = event["stage"]
    mapped = (
        "representation"
        if stage in {"snapshotting_members", "validation_inference"}
        else "classifier_fit"
        if stage == "selecting_weights"
        else "finalization"
    )
    current, total = event.get("completed", 1), event.get("total", 1)
    emit_run_progress(
        mapped, "complete" if stage == "completed" else "running", current, total, feature=stage
    )
    print(json.dumps({"web_ensemble_progress": event}), flush=True)


def assemble(request, artifact_dir, report=progress):
    output = Path(artifact_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise EnsembleError(
            "ensemble_output_exists", "An ensemble must use an empty artifact directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    definition = request["definition"]
    dataset = request["dataset"]
    task, method = normalize_task_type(definition["task_type"]), definition["method"]
    current_identity = validation_identity(dataset)
    if current_identity != request["dataset_identity"]:
        raise EnsembleError("ensemble_dataset_changed", "Dataset changed after preflight")
    val = read_split(dataset, "val")
    labels = request["label_names"]
    if normalize_task_type(task) == "pep_function" and val["labels"] != labels:
        raise EnsembleError(
            "ensemble_label_mismatch", "Dataset label columns differ from member label names"
        )
    if method == "rank_mean" and len(val["y"]) < 2:
        raise EnsembleError(
            "ensemble_rank_batch_required", "Rank fusion needs at least two validation samples"
        )
    members, records = [], []
    for index, record in enumerate(request["members"]):
        verify_provenance(record["bundle_path"], dataset, current_identity)
        if tree_identity(record["bundle_path"]) != record["bundle_files"]:
            raise EnsembleError("ensemble_member_changed", "Member changed after preparation")
        destination = output / "members" / str(index)
        member = snapshot_member(record, destination)
        member["snapshot_path"] = f"members/{index}"
        members.append(member)
        records.append({**record, "bundle_path": str(destination)})
        report(
            {
                "stage": "snapshotting_members",
                "completed": index + 1,
                "total": len(request["members"]),
            }
        )

    service = ModelService(SnapshotStore(records))
    score_sources = []
    try:
        matrices = []
        for index, record in enumerate(records):
            bundle = Path(record["bundle_path"])
            evidence = read_json(bundle / PROVENANCE_FILE)
            scores_path = bundle / "evaluation_scores.npz"
            scores = None
            if (
                evidence.get("evaluation_scores_sha256")
                and scores_path.is_file()
                and file_digest(scores_path) == evidence["evaluation_scores_sha256"]
            ):
                with np.load(scores_path, allow_pickle=False) as arrays:
                    if "validation_sample_ids" in arrays and np.array_equal(
                        arrays["validation_sample_ids"], val["identity"]["sample_ids"]
                    ):
                        if not np.array_equal(arrays["validation_labels"], val["y"]):
                            raise EnsembleError(
                                "ensemble_label_mismatch",
                                "Cached validation labels differ from the verified dataset",
                            )
                        # Access validation arrays only. Test arrays remain unopened.
                        scores = np.asarray(arrays["validation_scores"], dtype=np.float64)
                        kind = evidence.get("score_kind")
                        if kind is None or (method != "rank_mean" and kind != "probability"):
                            scores = None
            if scores is None:
                response = service.predict(record["id"], **val["inputs"])
                scores = response_scores(response, labels, len(val["y"]), task, method)
                score_sources.append("fresh_frozen_member_inference")
            else:
                score_sources.append("verified_cached_validation")
            matrices.append(scores)
            report({"stage": "validation_inference", "completed": index + 1, "total": len(records)})
        if validation_identity(dataset) != current_identity:
            raise EnsembleError(
                "ensemble_dataset_changed", "Dataset changed during validation inference"
            )
        weights, point, search = select(matrices, val["y"], labels, definition, report)
        validation_scores = fuse(matrices, weights, method)
        metrics = Evaluator.score_metrics(val["y"], validation_scores, point)
        manifest = {
            "schema_version": CONTRACT_VERSION,
            "definition": definition,
            "members": members,
            "weights": list(weights),
            "label_names": labels,
            "operating_point": point.to_dict(),
            "dataset_identity": current_identity,
            "validation_score_sources": score_sources,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "rank_rule": "double_argsort_axis0_full_batch" if method == "rank_mean" else None,
        }
        # This durable definition is committed before opening any Test split.
        write_json(output / "ensemble.json", manifest)
        write_json(output / "weight_search.json", {"definition": definition, "trials": search})
        report({"stage": "definition_frozen"})
        checkpoint_metric = ObjectiveSpec.parse(
            definition["objective_definition"]
        ).checkpoint_metric
        protocol = {
            **metric_definition(),
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "checkpoint_metric": checkpoint_metric,
            "feature_fit_scope": "train_only",
            "evaluation_mode": "independent_iid",
            "operating_point": point.to_dict(),
        }
        protocol.update(ensemble_selection_fields(definition["objective_definition"], point))
        evaluation = {**protocol, "validation": metrics}
        arrays = {
            "validation_labels": val["y"],
            "validation_scores": validation_scores,
            "validation_sample_ids": np.asarray(val["identity"]["sample_ids"]),
        }
        test_metrics = {}
        if definition["evaluate_test"]:
            registered = (dataset.get("metadata") or {}).get("evaluation_splits")
            if registered is None:
                registered = [
                    path.name
                    for path in sorted(Path(dataset["root_path"]).iterdir())
                    if path.is_dir() and (path.name == "test" or path.name.startswith("test_"))
                ]
            if (
                not isinstance(registered, list)
                or not registered
                or len(set(registered)) != len(registered)
            ):
                raise EnsembleError(
                    "ensemble_test_missing", "No valid named independent test splits are registered"
                )
            seen = read_split(dataset, "train")["keys"] | val["keys"]
            for split_name in registered:
                if split_name in {"train", "val", "validation"}:
                    raise EnsembleError(
                        "ensemble_test_invalid",
                        "Training and validation cannot be selected as tests",
                    )
                split = read_split(dataset, split_name)
                if normalize_task_type(task) == "pep_function" and split["labels"] != labels:
                    raise EnsembleError(
                        "ensemble_label_mismatch",
                        "Test label columns differ from member label names",
                    )
                if seen & split["keys"]:
                    raise EnsembleError(
                        "ensemble_test_overlap", "Independent splits have overlapping inputs"
                    )
                seen |= split["keys"]
                split_matrices = [
                    response_scores(
                        service.predict(record["id"], **split["inputs"]),
                        labels,
                        len(split["y"]),
                        task,
                        method,
                    )
                    for record in records
                ]
                fused = fuse(split_matrices, weights, method)
                if read_split(dataset, split_name)["identity"] != split["identity"]:
                    raise EnsembleError(
                        "ensemble_dataset_changed", "A test split changed during inference"
                    )
                test_metrics[split_name] = Evaluator.score_metrics(split["y"], fused, point)
                split_dir = output / "evaluations" / split_name
                split_dir.mkdir(parents=True)
                np.savez_compressed(
                    split_dir / "scores.npz",
                    labels=split["y"],
                    scores=fused,
                    sample_ids=np.asarray(split["identity"]["sample_ids"]),
                )
                write_json(
                    split_dir / "evaluation.json",
                    {
                        **protocol,
                        "metrics": test_metrics[split_name],
                        "data_identity": split["identity"],
                    },
                )
                if split_name == "test":
                    evaluation["test"] = test_metrics[split_name]
                    arrays.update(test_labels=split["y"], test_scores=fused)
                report({"stage": "test_evaluation", "split": split_name})
        evaluation["test_splits"] = test_metrics
        metadata = {
            **protocol,
            "bundle_type": "web_ensemble",
            "supports_inference": True,
            "input_schema": {
                "pep_function": "sequences",
                "pep_hlai": "peptide_hlai_pairs",
                "pep_hlaii": "peptide_hla_pairs",
                "pep_protein_interaction": "peptide_protein_pairs",
            }[task],
            "args": {"task_type": task, "seed": definition["seed"]},
            "label_names": labels,
            "ensemble_manifest_digest": digest_json(manifest),
            "web_ensemble": {
                "method": method,
                "weight_source": definition["weight_source"],
                "weights": list(weights),
                "objective": definition["objective"],
                "members": [
                    {"id": member["model_id"], "name": member["name"]} for member in members
                ],
                "batch_dependent": method == "rank_mean",
                "test_splits": test_metrics,
            },
            "threshold_source": point.to_dict()["threshold_source"],
            "effective_route": {
                "schema_version": 5,
                "route_id": f"{task}.web_ensemble",
                "task_type": task,
                "family": "ensemble",
                "model": "ensemble",
                "tokenizer": "ensemble",
                "encoder": method,
            },
            "route_schema_version": 5,
        }
        if normalize_task_type(task) == "pep_function":
            metadata["thresholds"] = list(point.values)
            metadata["threshold_by_label"] = dict(zip(labels, point.values))
        else:
            metadata["threshold"] = point.threshold
        write_json(output / "metadata.json", metadata)
        write_json(output / "evaluation.json", evaluation)
        np.savez_compressed(output / "evaluation_scores.npz", **arrays)
        errors = audit_artifact(
            output / "evaluation.json",
            allowed_checkpoint_metrics=EVALUATION_CHECKPOINT_METRICS | {"validation_aupr"},
        )
        if errors:
            raise EnsembleError("ensemble_artifact_audit_failed", "; ".join(errors))
        analysis = build_score_analysis(
            val["y"],
            validation_scores,
            arrays.get("test_labels"),
            arrays.get("test_scores"),
            operating_point=point,
            warnings=[
                "Ensemble analysis is based on fused scores; member explanations remain separate."
            ],
        )
        save_analysis(output, analysis)
        write_json(
            output / "audit.json",
            {"passed": True, "errors": [], "manifest_digest": digest_json(manifest)},
        )
        report({"stage": "completed", "validation": metrics, "test_splits": test_metrics})
        return metadata
    finally:
        service.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--artifact-dir", required=True)
    args = parser.parse_args()
    assemble(read_json(args.request), args.artifact_dir)


if __name__ == "__main__":
    main()
