"""Local ensemble orchestration; all numerical rules come from existing code."""
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np

from pepplatform.task_compat import normalize_task_type
from pepplatform.web.ensemble_math import CONTRACT_VERSION, normalize_definition, select, fuse
from pepplatform.web.ensemble_data import response_scores
from . import data as local_data
from .storage import (CLIError, ROOT, copy_bundle, digest_json, file_digest, guard_output,
                      inventory, read_json, write_json)


def runtime_identity():
    return {str(path.relative_to(ROOT)): file_digest(path)
            for name in ("pepplatform", "scripts", "pepplatform_cli")
            for path in sorted((ROOT / name).rglob("*.py")) if "__pycache__" not in path.parts}


def source_evidence(bundle, metadata, identity, files, specified=None):
    candidates = ([Path(specified).expanduser().resolve()] if specified else [
        bundle.parent / "training_provenance.json", bundle / "web_ensemble_provenance.json"])
    proof_path = next((p for p in candidates if p.is_file()), None)
    if proof_path is None:
        if specified:
            raise CLIError("provenance_missing", f"Explicit provenance file is missing: {specified}")
        return {"status": "unverified", "reason": "No training-time provenance"}
    proof = read_json(proof_path)
    if proof.get("schema_version") != 1:
        raise CLIError("provenance_invalid", f"Unsupported provenance: {proof_path}")
    expected = proof.get("dataset_identity", proof)
    try:
        matches = local_data.same_data(expected, identity)
    except (KeyError, TypeError, ValueError) as exc:
        raise CLIError("provenance_invalid", f"Invalid data evidence: {proof_path}") from exc
    if not matches:
        raise CLIError("provenance_data_changed", f"Training/Validation identity differs: {bundle}")
    if proof.get("kind") == "cli_training":
        if proof.get("status") != "verified" or proof.get("artifact_files") != files:
            raise CLIError("provenance_artifact_changed", f"Training artifact differs from receipt: {bundle}")
    elif proof_path.name != "web_ensemble_provenance.json":
        raise CLIError("provenance_invalid", "Expected a CLI training receipt or original Web provenance")
    return {"status": "verified", "path": str(proof_path), "sha256": file_digest(proof_path),
            "evidence": proof}


def prepare(config, *, check_resources=True):
    from pepplatform.evaluators import EVALUATION_PROTOCOL_VERSION
    from pepplatform.web.inference import ModelService
    from pepplatform.web.ensemble_service import SnapshotStore

    raw = config.get("members")
    if not isinstance(raw, list) or len(raw) < 2:
        raise CLIError("members_required", "Select at least two local member artifacts")
    specs = [dict(value) if isinstance(value, dict) else {"artifact_dir": value} for value in raw]
    paths = [Path(x["artifact_dir"]).expanduser().resolve() for x in specs]
    if len(set(paths)) != len(paths):
        raise CLIError("member_duplicate", "Each member artifact must be selected once")
    ids = ["local_" + digest_json(str(path))[:24] for path in paths]
    definition = normalize_definition({**config, "member_ids": ids})
    allow = config.get("allow_unverified_provenance", False)
    if not isinstance(allow, bool):
        raise CLIError("provenance_flag_invalid", "allow_unverified_provenance must be a boolean")
    if "test_splits" in config and (not isinstance(config["test_splits"], list) or
            not config["test_splits"] or any(not isinstance(x, str) or not x or Path(x).name != x or
            x.startswith(".") or x in {"train", "val", "validation"} for x in config["test_splits"]) or
            len(set(config["test_splits"])) != len(config["test_splits"])):
        raise CLIError("test_splits_invalid", "Test splits must be unique independent split names")
    data = local_data.dataset({**config, "task_type": definition["task_type"]})
    identity = local_data.validation_identity(data)
    val = local_data.read_split(data, "val")
    records, labels, warnings = [], None, []
    for spec, path, member_id in zip(specs, paths, ids):
        metadata = read_json(path / "metadata.json")
        task = normalize_task_type((metadata.get("args") or {}).get("task_type") or metadata.get("task_type"))
        if task != definition["task_type"] or metadata.get("supports_inference") is False:
            raise CLIError("member_incompatible", f"Member must support inference for the same task: {path}")
        if metadata.get("bundle_type") in {"ensemble", "web_ensemble", "cli_ensemble"}:
            raise CLIError("nested_unsupported", "Nested ensembles are unsupported")
        if (
            metadata.get("evaluation_protocol_version") != EVALUATION_PROTOCOL_VERSION
            or metadata.get("feature_fit_scope") != "train_only"
        ):
            raise CLIError(
                "protocol_required",
                f"Member requires supported evaluation metadata and train-only feature provenance: {path}",
            )
        member_labels = metadata.get("label_names")
        if (not isinstance(member_labels, list) or not member_labels or
                any(not isinstance(x, str) or not x.strip() for x in member_labels) or
                len(set(member_labels)) != len(member_labels)):
            raise CLIError("labels_invalid", f"Member needs explicit unique output labels: {path}")
        if labels is not None and member_labels != labels:
            raise CLIError("label_mismatch", "Member labels or order differ")
        if definition["task_type"] == "pep_function" and member_labels != val["labels"]:
            raise CLIError("label_mismatch", "Member and dataset label columns differ")
        if definition["task_type"] != "pep_function" and len(member_labels) != 1:
            raise CLIError("label_mismatch", "Pair members must have one output label")
        labels = member_labels
        files = inventory(path)
        evidence = source_evidence(path, metadata, identity, files, spec.get("provenance"))
        if evidence["status"] != "verified":
            if not allow:
                raise CLIError("provenance_required", f"Training-time provenance required: {path}. "
                               "Use --allow-unverified-provenance explicitly for legacy compatibility.")
            warnings.append(f"Unverified training provenance: {path}")
        records.append({"id": member_id, "name": str(metadata.get("label") or path.name),
                        "task_type": task, "bundle_path": str(path), "supports_inference": True,
                        "metadata": metadata, "bundle_files": files, "provenance": evidence})
    if definition["method"] == "rank_mean" and len(val["y"]) < 2:
        raise CLIError("rank_batch_required", "Rank fusion requires at least two validation rows")
    if check_resources:
        service = ModelService(SnapshotStore(records))
        for record in records:
            report = service.readiness(record["id"])
            warnings.extend(str(item) for item in report.get("warnings", []))
            if not report.get("ready"):
                raise CLIError("member_not_ready", f"{record['name']}: {report.get('errors', [])}")
    status = "verified" if all(x["provenance"]["status"] == "verified" for x in records) else "unverified"
    return {"definition": definition, "dataset": data, "dataset_identity": identity,
            "members": records, "label_names": labels, "provenance_status": status,
            "allow_unverified_provenance": allow, "warnings": warnings,
            **({"test_splits": config["test_splits"]} if "test_splits" in config else {})}


def preflight(config):
    try:
        request = prepare(config)
        return {"valid": True, "errors": [], "error_codes": [], "definition": request["definition"],
                "label_names": request["label_names"], "provenance_status": request["provenance_status"],
                "warnings": request["warnings"],
                "validation_count": request["dataset_identity"]["validation"]["count"],
                "members": [{"id": x["id"], "name": x["name"],
                             "provenance_status": x["provenance"]["status"]} for x in request["members"]]}
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return {"valid": False, "errors": [str(exc)],
                "error_codes": [getattr(exc, "code", "preflight_failed")], "warnings": []}


def snapshot_records(request, output):
    members, records = [], []
    for index, record in enumerate(request["members"]):
        destination = output / "members" / str(index)
        proof = record["provenance"]
        if proof.get("path") and file_digest(proof["path"]) != proof["sha256"]:
            raise CLIError("provenance_changed", "Member provenance changed after preflight")
        frozen = copy_bundle(record["bundle_path"], destination, record["bundle_files"])
        members.append({**frozen, "snapshot_path": f"members/{index}", "model_id": record["id"],
                        "name": record["name"], "task_type": record["task_type"], "provenance": proof})
        records.append({**record, "bundle_path": str(destination)})
    return members, records


def cached_validation(record, val, labels, definition):
    """Only training-time hash-bound cache evidence can avoid a fresh forward."""
    proof = record["provenance"]
    if proof["status"] != "verified":
        return None
    evidence = proof["evidence"]
    path = Path(record["bundle_path"]) / "evaluation_scores.npz"
    if not path.is_file() or evidence.get("evaluation_scores_sha256") != file_digest(path):
        return None
    kind = evidence.get("score_kind")
    if kind is None or (definition["method"] != "rank_mean" and kind != "probability"):
        return None
    with np.load(path, allow_pickle=False) as arrays:
        if "validation_sample_ids" not in arrays or not np.array_equal(
                arrays["validation_sample_ids"], val["identity"]["sample_ids"]):
            return None
        if not np.array_equal(arrays["validation_labels"], val["y"]):
            raise CLIError("cache_label_mismatch", "Verified validation cache labels disagree")
        scores = np.asarray(arrays["validation_scores"], dtype=np.float64)
    if scores.shape != val["y"].shape or not np.all(np.isfinite(scores)):
        raise CLIError("cache_invalid", "Verified validation cache shape or values disagree")
    return scores


def assemble(request, artifact_dir, report=lambda event: None):
    from pepplatform.analysis import build_score_analysis, save_analysis
    from pepplatform.ensemble import ensemble_selection_fields
    from pepplatform.evaluation_audit import audit_artifact
    from pepplatform.evaluators import (
        EVALUATION_CHECKPOINT_METRICS,
        EVALUATION_PROTOCOL_VERSION,
        Evaluator,
    )
    from pepplatform.objectives import ObjectiveSpec
    from pepplatform.pr_metrics import metric_definition
    from pepplatform.web.inference import ModelService
    from pepplatform.web.ensemble_service import SnapshotStore

    data, definition = request["dataset"], request["definition"]
    inputs = [data["root_path"], *[r["bundle_path"] for r in request["members"]]]
    output = guard_output(artifact_dir, inputs)
    output.mkdir(parents=True, exist_ok=False)
    if local_data.validation_identity(data) != request["dataset_identity"]:
        raise CLIError("dataset_changed", "Dataset changed after preflight")
    members, records = snapshot_records(request, output)
    report({"stage": "members_frozen", "count": len(records)})
    service = ModelService(SnapshotStore(records))
    val = local_data.read_split(data, "val")
    labels, task, method = request["label_names"], definition["task_type"], definition["method"]
    matrices, sources = [], []
    for record in records:
        scores = cached_validation(record, val, labels, definition)
        if scores is None:
            scores = response_scores(service.predict(record["id"], **val["inputs"]),
                                     labels, len(val["y"]), task, method)
            sources.append("fresh_frozen_member_inference")
        else:
            sources.append("verified_cached_validation")
        matrices.append(scores)
        report({"stage": "validation_inference", "completed": len(matrices), "total": len(records)})
    if local_data.validation_identity(data) != request["dataset_identity"]:
        raise CLIError("dataset_changed", "Dataset changed during validation inference")
    weights, point, search = select(matrices, val["y"], labels, definition, report)
    validation_scores = fuse(matrices, weights, method)
    manifest = {"schema_version": CONTRACT_VERSION, "definition": definition, "members": members,
                "weights": list(weights), "label_names": labels, "operating_point": point.to_dict(),
                "dataset_identity": request["dataset_identity"], "validation_score_sources": sources,
                "provenance_status": request["provenance_status"], "warnings": request["warnings"],
                "numpy_version": np.__version__, "python_version": platform.python_version(),
                "runtime_sources": runtime_identity(),
                "rank_rule": "double_argsort_axis0_full_batch" if method == "rank_mean" else None}
    write_json(output / "ensemble.json", manifest)
    write_json(output / "weight_search.json", {"definition": definition, "trials": search})
    report({"stage": "definition_frozen"})
    protocol = {**metric_definition(), "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
                "checkpoint_metric": ObjectiveSpec.parse(definition["objective_definition"]).checkpoint_metric,
                "feature_fit_scope": "train_only", "evaluation_mode": "independent_iid",
                "operating_point": point.to_dict(), "provenance_status": request["provenance_status"]}
    protocol.update(ensemble_selection_fields(definition["objective_definition"], point))
    evaluation = {**protocol, "validation": Evaluator.score_metrics(val["y"], validation_scores, point),
                  "test_splits": {}}
    arrays = {"validation_labels": val["y"], "validation_scores": validation_scores,
              "validation_sample_ids": np.asarray(val["identity"]["sample_ids"])}
    if definition["evaluate_test"]:
        splits = request.get("test_splits") or [p.name for p in sorted(Path(data["root_path"]).iterdir())
                                                if p.is_dir() and (p.name == "test" or p.name.startswith("test_"))]
        if not splits or len(set(splits)) != len(splits):
            raise CLIError("test_missing", "No unique independent test splits available")
        seen = local_data.read_split(data, "train")["keys"] | val["keys"]
        for name in splits:
            if name in {"train", "val", "validation"}:
                raise CLIError("test_invalid", "Train/Validation cannot be used as independent Test")
            split = local_data.read_split(data, name)
            if (task == "pep_function" and split["labels"] != labels) or seen & split["keys"]:
                raise CLIError("test_overlap_or_labels", "Independent split overlap or label mismatch")
            seen |= split["keys"]
            scores = fuse([response_scores(service.predict(r["id"], **split["inputs"]), labels,
                                             len(split["y"]), task, method) for r in records], weights, method)
            if local_data.read_split(data, name)["identity"] != split["identity"]:
                raise CLIError("dataset_changed", "Test data changed during inference")
            metrics = Evaluator.score_metrics(split["y"], scores, point)
            evaluation["test_splits"][name] = metrics
            directory = output / "evaluations" / name
            directory.mkdir(parents=True)
            np.savez_compressed(directory / "scores.npz", labels=split["y"], scores=scores,
                                sample_ids=np.asarray(split["identity"]["sample_ids"]))
            write_json(directory / "evaluation.json", {**protocol, "metrics": metrics,
                                                       "data_identity": split["identity"]})
            if name == "test":
                evaluation["test"] = metrics
                arrays.update(test_labels=split["y"], test_scores=scores)
    metadata = {**protocol, "bundle_type": "cli_ensemble", "supports_inference": True,
                "args": {"task_type": task, "seed": definition["seed"]}, "label_names": labels,
                "ensemble_manifest_digest": digest_json(manifest),
                "threshold_source": point.to_dict()["threshold_source"]}
    if task == "pep_function":
        metadata.update(thresholds=list(point.values), threshold_by_label=dict(zip(labels, point.values)))
    else:
        metadata["threshold"] = point.threshold
    write_json(output / "metadata.json", metadata)
    write_json(output / "evaluation.json", evaluation)
    np.savez_compressed(output / "evaluation_scores.npz", **arrays)
    errors = audit_artifact(output / "evaluation.json",
                            allowed_checkpoint_metrics=EVALUATION_CHECKPOINT_METRICS | {"validation_aupr"})
    if errors:
        raise CLIError("artifact_audit_failed", "; ".join(errors))
    save_analysis(output, build_score_analysis(val["y"], validation_scores, arrays.get("test_labels"),
                  arrays.get("test_scores"), operating_point=point, warnings=request["warnings"]))
    write_json(output / "audit.json", {"passed": True, "errors": [],
                                       "manifest_digest": digest_json(manifest)})
    report({"stage": "completed"})
    return {"artifact_dir": str(output), "weights": list(weights), "metadata": metadata,
            "validation": evaluation["validation"], "test_splits": evaluation["test_splits"],
            "provenance_status": request["provenance_status"]}
