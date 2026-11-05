"""Authenticated preparation and inference adapters for versioned Web ensembles."""

from __future__ import annotations
from pepplatform.task_compat import normalize_task_type


import os
from pathlib import Path

import numpy as np

from pepplatform.evaluators import EVALUATION_PROTOCOL_VERSION
from .ensemble_data import (
    digest_json,
    read_json,
    read_split,
    response_scores,
    safe_root,
    tree_identity,
    validation_identity,
    verify_provenance,
)
from .ensemble_math import (
    CONTRACT_VERSION,
    METHODS,
    TASKS,
    WEIGHT_SOURCES,
    EnsembleError,
    fuse,
    normalize_definition,
)
from .inference import ModelService


def creation_enabled():
    # Ensemble creation is a supported platform capability by default.  Keep
    # the environment variable as an emergency/maintenance kill switch for
    # operators that need to pause new creations without disabling reads,
    # inference, or history for existing ensembles.
    return os.environ.get("PEP_WEB_ENSEMBLES_ENABLED", "1").lower() in {"1", "true", "yes", "on"}


class EnsembleService:
    def __init__(self, store, paths):
        self.store, self.paths = store, paths

    def capabilities(self):
        from pepplatform.objectives import objective_capabilities

        return {
            "enabled": creation_enabled(),
            "schema_version": CONTRACT_VERSION,
            "objectives": objective_capabilities(),
            "tasks": list(TASKS),
            "methods": list(METHODS),
            "weight_sources": list(WEIGHT_SOURCES),
            "defaults": {
                "method": "probability_mean",
                "weight_source": "equal",
                "grid_step": 0.05,
                "seed": 42,
                "evaluate_test": False,
            },
        }

    def prepare(self, payload, user=None):
        definition = normalize_definition(payload)
        dataset = self.store.get_dataset(payload.get("dataset_id"), user=user)
        if not dataset or normalize_task_type(dataset["task_type"]) != normalize_task_type(
            definition["task_type"]
        ):
            raise EnsembleError(
                "ensemble_dataset_invalid", "Select an accessible dataset with the same task"
            )
        safe_root(dataset["root_path"])
        identity = validation_identity(dataset)
        validation_labels = read_split(dataset, "val")["labels"]
        records, labels = [], None
        for member_id in definition["member_ids"]:
            record = self.store.get_model(member_id, user=user)
            if not record:
                raise EnsembleError(
                    "ensemble_member_unavailable", "A selected model is unavailable"
                )
            if (
                normalize_task_type(record["task_type"])
                != normalize_task_type(definition["task_type"])
                or not record["supports_inference"]
            ):
                raise EnsembleError(
                    "ensemble_member_incompatible",
                    "Members must support inference for the selected task",
                )
            bundle = safe_root(record["bundle_path"])
            if not bundle.is_relative_to(self.paths.root.resolve()):
                raise EnsembleError(
                    "ensemble_member_unmanaged", "Only models in the Web runtime can be used"
                )
            metadata = read_json(bundle / "metadata.json")
            if metadata.get("bundle_type") in {"ensemble", "web_ensemble"}:
                raise EnsembleError(
                    "ensemble_nested_unsupported", "Nested ensembles are not supported"
                )
            if (
                metadata.get("evaluation_protocol_version") != EVALUATION_PROTOCOL_VERSION
                or metadata.get("feature_fit_scope") != "train_only"
            ):
                raise EnsembleError(
                    "ensemble_protocol_required",
                    "Members require current evaluation metadata and features fitted on training data only",
                )
            member_labels = metadata.get("label_names")
            if (
                not isinstance(member_labels, list)
                or not member_labels
                or any(not isinstance(label, str) or not label.strip() for label in member_labels)
                or len(set(member_labels)) != len(member_labels)
            ):
                raise EnsembleError(
                    "ensemble_labels_invalid", "Members require explicit, unique label names"
                )
            if (
                normalize_task_type(definition["task_type"]) != "pep_function"
                and len(member_labels) != 1
            ):
                raise EnsembleError(
                    "ensemble_labels_invalid", "Pair members must have one output label"
                )
            if labels is not None and member_labels != labels:
                raise EnsembleError("ensemble_label_mismatch", "Member label names/order differ")
            if (
                normalize_task_type(definition["task_type"]) == "pep_function"
                and member_labels != validation_labels
            ):
                raise EnsembleError(
                    "ensemble_label_mismatch",
                    "Dataset label columns differ from member label names",
                )
            labels = member_labels
            verify_provenance(bundle, dataset, identity)
            records.append({**record, "bundle_files": tree_identity(bundle)})
        return {
            "definition": definition,
            "dataset": dataset,
            "members": records,
            "label_names": labels,
            "dataset_identity": identity,
        }

    def preflight(self, payload, user=None):
        try:
            request = self.prepare(payload, user=user)
            return {
                "valid": True,
                "errors": [],
                "error_codes": [],
                "definition": request["definition"],
                "label_names": request["label_names"],
                "validation_count": request["dataset_identity"]["validation"]["count"],
                "members": [{"id": x["id"], "name": x["name"]} for x in request["members"]],
                "validation_policy": "verified_identity_then_cached_or_fresh_inference",
            }
        except EnsembleError as exc:
            return {"valid": False, "errors": [str(exc)], "error_codes": [exc.code], **exc.detail()}


class SnapshotStore:
    """Per-request records, never a registration or mutation of the live store."""

    def __init__(self, records):
        self.records = {record["id"]: record for record in records}

    def get_model(self, model_id):
        return self.records.get(model_id)


def snapshot_records(bundle, manifest):
    records = []
    for member in manifest["members"]:
        path = (Path(bundle) / member["snapshot_path"]).resolve()
        if not path.is_relative_to(Path(bundle).resolve() / "members"):
            raise EnsembleError(
                "ensemble_snapshot_invalid", "Member snapshot escapes the ensemble bundle"
            )
        if tree_identity(path) != member["snapshot_files"]:
            raise EnsembleError(
                "ensemble_snapshot_changed", "An ensemble member snapshot is missing or changed"
            )
        records.append(
            {
                "id": member["model_id"],
                "name": member["name"],
                "task_type": member["task_type"],
                "bundle_path": str(path),
                "supports_inference": True,
            }
        )
    return records


def load_bundle(record):
    bundle = Path(record["bundle_path"])
    metadata = read_json(bundle / "metadata.json")
    manifest = read_json(bundle / "ensemble.json")
    expected = (record.get("metadata") or {}).get("ensemble_manifest_digest")
    if (
        manifest.get("schema_version") != CONTRACT_VERSION
        or not expected
        or digest_json(manifest) != expected
    ):
        raise EnsembleError(
            "ensemble_manifest_changed", "Ensemble manifest is missing, changed, or unsupported"
        )
    if metadata.get("operating_point") != manifest.get("operating_point"):
        raise EnsembleError("ensemble_threshold_changed", "Ensemble operating point changed")
    if (
        manifest["definition"]["method"] == "rank_mean"
        and manifest.get("numpy_version") != np.__version__
    ):
        raise EnsembleError(
            "ensemble_rank_runtime_changed",
            "Rank fusion requires the NumPy version recorded at creation",
        )
    return metadata, manifest, snapshot_records(bundle, manifest)


class WebModelService(ModelService):
    """Only dispatch the new bundle type; all legacy models use ModelService."""

    def _predict(self, model_id, sequences=None, pairs=None):
        record = self.store.get_model(model_id)
        if not record or (record.get("metadata") or {}).get("bundle_type") != "web_ensemble":
            return super()._predict(model_id, sequences=sequences, pairs=pairs)
        metadata, manifest, records = load_bundle(record)
        definition = manifest["definition"]
        task, method = definition["task_type"], definition["method"]
        if normalize_task_type(task) == "pep_function":
            if pairs is not None or not sequences:
                raise EnsembleError("ensemble_input_invalid", "This ensemble requires sequences")
            inputs = {"sequences": sequences}
            count = len(sequences)
        else:
            if sequences is not None or not pairs:
                raise EnsembleError(
                    "ensemble_input_invalid", "This ensemble requires peptide/partner pairs"
                )
            inputs = {"pairs": pairs}
            count = len(pairs)
        if method == "rank_mean" and count < 2:
            raise EnsembleError(
                "ensemble_rank_batch_required",
                "Rank fusion requires at least two rows in a complete batch",
            )
        labels = manifest["label_names"]
        member_service = ModelService(SnapshotStore(records))
        from .inference import _report_inference_progress

        matrices = []
        # The application owns the shared ESM3 pool. This per-request store
        # view must not close it.
        for index, member in enumerate(records):
            response = member_service._predict(member["id"], **inputs)
            matrices.append(response_scores(response, labels, count, task, method))
            _report_inference_progress("ensemble_members", completed=index + 1, total=len(records))
        scores = fuse(matrices, manifest["weights"], method)
        point = manifest["operating_point"]
        thresholds = point["values"]
        kind = "rank" if method == "rank_mean" else "probability"
        result = {
            "model_id": model_id,
            "labels": labels,
            "thresholds": thresholds,
            "score_kind": kind,
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "legacy_evaluation_protocol": False,
            "ensemble": {
                "method": method,
                "batch_dependent": method == "rank_mean",
                "batch_digest": digest_json(inputs),
                "input_count": count,
                "numpy_version": np.__version__,
                "contract_version": CONTRACT_VERSION,
            },
        }
        if normalize_task_type(task) == "pep_function":
            result["predictions"] = [
                {
                    "sequence": sequence,
                    "scores": row.tolist(),
                    "predicted_labels": [
                        label
                        for label, value, threshold in zip(labels, row, thresholds)
                        if value >= threshold
                    ],
                }
                for sequence, row in zip(sequences, scores)
            ]
        else:
            result["predictions"] = [
                self._pair_prediction_record(pair, [value], thresholds[0], kind, [0, 1])
                for pair, value in zip(pairs, scores)
            ]
        return result

    def readiness(self, model_id):
        record = self.store.get_model(model_id)
        if not record or (record.get("metadata") or {}).get("bundle_type") != "web_ensemble":
            return super().readiness(model_id)
        try:
            _, manifest, records = load_bundle(record)
            service = ModelService(SnapshotStore(records))
            reports = [service.readiness(member["id"]) for member in records]
            errors = [error for report in reports for error in report.get("errors", [])]
            warnings = [warning for report in reports for warning in report.get("warnings", [])]
            ready = all(report.get("ready") for report in reports)
            return {
                "model_id": model_id,
                "ready": ready,
                "status": ("warning" if warnings else "ready") if ready else "blocked",
                "profile_required": any(x.get("profile_required") for x in reports),
                "mode": "ensemble",
                "checks": [],
                "errors": errors,
                "warnings": warnings,
                "members": reports,
                "batch_dependent": manifest["definition"]["method"] == "rank_mean",
            }
        except (EnsembleError, OSError, ValueError) as exc:
            detail = (
                exc.detail()
                if isinstance(exc, EnsembleError)
                else {"code": "ensemble_unavailable", "message": str(exc)}
            )
            return {
                "model_id": model_id,
                "ready": False,
                "status": "blocked",
                "mode": "ensemble",
                "profile_required": False,
                "checks": [],
                "errors": [detail],
                "warnings": [],
            }
