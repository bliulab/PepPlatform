"""Derive current PR metrics from saved scores without changing sealed bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from pepplatform.evaluators import Evaluator
from pepplatform.pr_metrics import has_current_aupr, is_aupr_key, metric_definition, without_legacy_aupr

REFERENCE_NAME = "metrics_v2.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_metric_reference(artifact_dir):
    path = Path(artifact_dir) / REFERENCE_NAME
    if not path.is_file():
        return None
    reference = read_json(path)
    if not has_current_aupr(reference):
        raise ValueError("Saved metric reference has an unsupported definition")
    if not reference.get("source_files") and reference.get("splits"):
        raise ValueError("Saved metric reference has no score provenance")
    for source, expected in reference["source_files"].items():
        if file_sha256(source) != expected:
            raise ValueError(f"Metric reference source changed: {source}")
    return reference


def recompute_artifact(artifact_dir, *, write=False):
    """Re-evaluate PR areas only; never fit thresholds or load model parameters.

    Missing split scores make that split unavailable. Malformed or mismatched
    available evidence fails instead of silently keeping its old AP value.
    """
    from pepplatform.analysis import build_curve_bundle

    root = Path(artifact_dir).resolve()
    source_files = {}

    def track(path):
        path = Path(path).resolve()
        source_files[str(path)] = file_sha256(path)
        return path

    evaluation_path, metadata_path = root / "evaluation.json", root / "metadata.json"
    evaluation = read_json(track(evaluation_path)) if evaluation_path.is_file() else {}
    metadata = read_json(track(metadata_path)) if metadata_path.is_file() else {}
    manifest_path = root / "artifact_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(track(manifest_path))
        for name in ("metadata.json", "evaluation.json", "evaluation_scores.npz"):
            expected = manifest.get("files", {}).get(name)
            if expected is not None and file_sha256(root / name) != expected:
                raise ValueError(f"Sealed evaluation input changed: {root / name}")
    point = evaluation.get("operating_point") or metadata.get("operating_point")
    reference = {**metric_definition(), "schema_version": 1, "splits": {}, "unavailable": {}}

    def evaluate(split, path, prefix, old):
        if not path.is_file():
            reference["unavailable"][split] = "saved_scores_missing"
            return
        with np.load(track(path), allow_pickle=False) as arrays:
            label_key, score_key = prefix + "labels", prefix + "scores"
            if label_key not in arrays or score_key not in arrays:
                reference["unavailable"][split] = "saved_labels_or_scores_missing"
                return
            labels, scores = arrays[label_key], arrays[score_key]
        ranking = Evaluator.ranking_metrics(labels, scores)
        metrics = without_legacy_aupr(old or {})
        metrics.update({k: v for k, v in ranking.items() if is_aupr_key(k)})
        metrics.pop("aupr_unavailable_reason", None)
        metrics.update(metric_definition())
        reference["splits"][split] = {
            "metrics": metrics,
            "curves": build_curve_bundle(labels, scores, operating_point=point),
            "scores_path": str(path.resolve()),
            "prefix": prefix,
        }

    main_scores = root / "evaluation_scores.npz"
    for split in ("validation", "test"):
        if split in evaluation:
            evaluate(split, main_scores, split + "_", evaluation[split])
    for split, old in (evaluation.get("test_splits") or {}).items():
        if split in reference["splits"]:
            continue
        # The Web ensemble writer declares these names in evaluation.json.
        # Names must not escape the bundle; no latest-file/directory fallback.
        if Path(split).name != split or split in {".", ".."}:
            raise ValueError("Invalid saved evaluation split name")
        evaluate(split, root / "evaluations" / split / "scores.npz", "", old)
    if not evaluation:
        reference["unavailable"]["validation"] = "saved_evaluation_missing"
    reference["source_files"] = source_files
    reference["operating_point"] = copy.deepcopy(point)
    if write:
        write_json_atomic(root / REFERENCE_NAME, reference)
    return reference


def summary_from_reference(reference):
    summary = {}
    for split, entry in reference["splits"].items():
        prefix = "val" if split == "validation" else "test" if split == "test" else f"test_{split}"
        summary.update({f"{prefix}_{k}": v for k, v in entry["metrics"].items() if is_aupr_key(k)})
    return {**summary, **metric_definition()}


def analysis_with_reference(payload, reference):
    result = without_legacy_aupr(payload)
    result.setdefault("metrics", {})
    result.setdefault("curves", {})
    for split, entry in reference["splits"].items():
        old_metrics = result["metrics"].get(split) or {}
        result["metrics"][split] = {**old_metrics, **entry["metrics"]}
        result["metrics"][split].pop("aupr_unavailable_reason", None)
        result["curves"][split] = entry["curves"]
    result.update(metric_definition())
    return result
