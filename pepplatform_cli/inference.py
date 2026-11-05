"""Inference adapters over isolated copies of existing reusable bundles."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pepplatform.evaluators import EVALUATION_PROTOCOL_VERSION
from pepplatform.task_compat import normalize_task_type
from pepplatform.web.ensemble_math import CONTRACT_VERSION, fuse, normalized_weights
from pepplatform.web.ensemble_data import response_scores
from pepplatform.web.ensemble_service import SnapshotStore
from pepplatform.web.inference import ModelService
from .ensemble import runtime_identity
from .storage import CLIError, copy_bundle, digest_json, inventory, read_json


def load_ensemble(bundle, metadata):
    manifest = read_json(bundle / "ensemble.json")
    if metadata["bundle_type"] in {"cli_ensemble", "web_ensemble"}:
        if manifest.get("schema_version") != CONTRACT_VERSION or digest_json(manifest) != metadata.get("ensemble_manifest_digest"):
            raise CLIError("manifest_changed", "Ensemble definition is missing, changed or unsupported")
        if metadata.get("operating_point") != manifest.get("operating_point"):
            raise CLIError("threshold_changed", "Ensemble operating point differs from its frozen definition")
        if manifest["definition"]["method"] == "rank_mean" and manifest.get("numpy_version") != np.__version__:
            raise CLIError("rank_runtime_changed", "Rank inference requires the recorded NumPy version")
        if manifest.get("runtime_sources") is not None and manifest["runtime_sources"] != runtime_identity():
            raise CLIError("runtime_changed", "CLI ensemble runtime source identity changed; use its recorded version")
        for member in manifest["members"]:
            path = (bundle / member["snapshot_path"]).resolve()
            if not path.is_relative_to(bundle / "members") or inventory(path) != member["snapshot_files"]:
                raise CLIError("snapshot_changed", "Ensemble member snapshot is missing or changed")
        return manifest
    if manifest.get("schema_version") != 1 or manifest.get("method", "probability_mean") != "probability_mean":
        raise CLIError("legacy_invalid", "Unsupported legacy ensemble definition")
    members = manifest.get("members", [])
    if len(members) < 2 or len(members) != len(manifest.get("weights", [])):
        raise CLIError("legacy_invalid", "Legacy member/weight count mismatch")
    normalized_weights(manifest["weights"], len(members))
    return manifest


def predict_artifact(artifact_dir, inputs, runtime_dir):
    """Runtime writes, including profile generation, stay inside runtime_dir."""
    bundle = Path(artifact_dir).expanduser().resolve()
    metadata = read_json(bundle / "metadata.json")
    task = normalize_task_type(metadata.get("args", {}).get("task_type") or metadata.get("task_type"))
    expected = "sequences" if task == "pep_function" else "pairs"
    if set(inputs) != {expected} or not inputs[expected]:
        raise CLIError("input_invalid", f"This model requires nonempty {expected}")
    kind = metadata.get("bundle_type")
    runtime = Path(runtime_dir).resolve()
    if kind not in {"cli_ensemble", "web_ensemble", "ensemble"}:
        destination = runtime / "model"
        copy_bundle(bundle, destination)
        record = {"id": "local", "name": bundle.name, "task_type": task,
                  "bundle_path": str(destination), "supports_inference": metadata.get("supports_inference", True),
                  "metadata": read_json(destination / "metadata.json")}
        service = ModelService(SnapshotStore([record]))
        response = service.predict("local", **inputs)
        response["provenance_status"] = "not_checked"
        return response

    manifest = load_ensemble(bundle, metadata)
    legacy = kind == "ensemble"
    definition = {"method": "probability_mean"} if legacy else manifest["definition"]
    method = definition["method"]
    count = len(inputs[expected])
    if method == "rank_mean" and count < 2:
        raise CLIError("rank_batch_required", "Rank fusion requires at least two rows in the complete request")
    labels = manifest.get("label_names") or metadata.get("label_names")
    if not labels:
        raise CLIError("labels_missing", "Ensemble requires explicit output label names")
    records = []
    seen = set()
    for index, member in enumerate(manifest["members"]):
        if legacy:
            raw_path = member.get("artifact_dir")
            if not raw_path:
                raise CLIError("legacy_member_missing", "Local legacy inference requires member artifact_dir values")
            source = Path(raw_path).expanduser()
            source = source.resolve() if source.is_absolute() else (bundle / source).resolve()
        else:
            source = (bundle / member["snapshot_path"]).resolve()
        if source in seen:
            raise CLIError("member_duplicate", "Duplicate member in frozen ensemble")
        seen.add(source)
        member_metadata = read_json(source / "metadata.json")
        if member_metadata.get("bundle_type") in {"ensemble", "web_ensemble", "cli_ensemble"}:
            raise CLIError("nested_unsupported", "Nested ensembles are unsupported")
        member_task = normalize_task_type(member_metadata.get("args", {}).get("task_type"))
        if member_task != task or member_metadata.get("label_names") != labels:
            raise CLIError("member_incompatible", "Member task or output-label order differs")
        destination = runtime / "members" / str(index)
        copy_bundle(source, destination, None if legacy else member["snapshot_files"])
        records.append({"id": f"member_{index}", "name": member.get("name", str(index)),
                        "task_type": task, "bundle_path": str(destination), "supports_inference": True,
                        "metadata": read_json(destination / "metadata.json")})
    service = ModelService(SnapshotStore(records))
    matrices = [response_scores(service.predict(record["id"], **inputs), labels, count, task, method)
                for record in records]
    scores = fuse(matrices, manifest["weights"], method)
    point = manifest.get("operating_point") or metadata.get("operating_point") or {}
    thresholds = point.get("values") or metadata.get("thresholds") or [metadata.get("threshold", 0.5)]
    if len(thresholds) != len(labels):
        raise CLIError("threshold_invalid", "Frozen threshold count differs from output labels")
    score_kind = "rank" if method == "rank_mean" else "probability"
    response = {"model_id": "local", "labels": labels, "thresholds": thresholds,
                "score_kind": score_kind, "evaluation_protocol_version": metadata.get("evaluation_protocol_version", EVALUATION_PROTOCOL_VERSION),
                "provenance_status": manifest.get("provenance_status", "unverified" if legacy else "verified"),
                "ensemble": {"method": method, "batch_dependent": method == "rank_mean",
                             "batch_digest": digest_json(inputs), "input_count": count,
                             "numpy_version": np.__version__, "contract_version": 1 if legacy else CONTRACT_VERSION}}
    if task == "pep_function":
        response["predictions"] = [{"sequence": sequence, "scores": row.tolist(),
            "predicted_labels": [label for label, value, threshold in zip(labels, row, thresholds) if value >= threshold]}
            for sequence, row in zip(inputs["sequences"], scores)]
    else:
        response["predictions"] = [service._pair_prediction_record(pair, [value], thresholds[0], score_kind, [0, 1])
                                   for pair, value in zip(inputs["pairs"], scores)]
    return response
