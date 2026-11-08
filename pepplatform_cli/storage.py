"""Local run storage and independent, content-checked model copies."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "output" / "cli"
RUNTIME_DIRS = {"profile_generation/inference_cache", "profile_generation/blastdb"}


class CLIError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

    def detail(self):
        return {"code": self.code, "message": str(self)}


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CLIError("evidence_missing", f"Cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CLIError("evidence_invalid", f"Expected a JSON object: {path}")
    return value


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    # Reuse the existing finite-number/NumPy serialization policy.
    from pepplatform.web.ensemble_data import write_json as write
    write(path, value)


def guard_output(path, inputs=()):
    path = Path(path).expanduser().resolve()
    protected = [ROOT / name for name in (
        "pepplatform", "scripts", "experiment_paper", "datasets", "models", "docs",
        "docs-site", "frontend", ".pepplatform_web")]
    protected += [Path(p).expanduser().resolve() for p in inputs]
    if "experiment_paper" in path.parts or any(p.startswith("final_roster_") for p in path.parts):
        raise CLIError("output_protected", f"Protected experiment output path: {path}")
    if any(path == p or path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
        raise CLIError("output_overlap", f"Output overlaps protected data or source: {path}")
    return path


def inventory(root):
    """Hash source bytes without writing caches or following directory symlinks."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise CLIError("artifact_missing", f"Artifact directory does not exist: {root}")
    result = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(dirs):
            candidate = base / name
            relative = candidate.relative_to(root).as_posix()
            if name == "__pycache__" or relative in RUNTIME_DIRS:
                dirs.remove(name)
            elif candidate.is_symlink():
                raise CLIError("artifact_symlink", f"Directory symlinks are unsupported: {candidate}")
        for name in sorted(names):
            candidate = base / name
            if candidate.is_file():
                result[candidate.relative_to(root).as_posix()] = file_digest(candidate)
    return dict(sorted(result.items()))


def copy_bundle(source, destination, expected=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    guard_output(destination, [source])
    before = inventory(source)
    if expected is not None and before != expected:
        raise CLIError("member_changed", f"Member changed after preflight: {source}")

    def ignore(directory, names):
        return [name for name in names if name == "__pycache__" or
                (Path(directory) / name).relative_to(source).as_posix() in RUNTIME_DIRS]

    shutil.copytree(source, destination, symlinks=False, ignore=ignore)
    if inventory(destination) != before or inventory(source) != before:
        raise CLIError("member_changed", f"Member changed during copying: {source}")

    def relocate(value):
        if isinstance(value, str) and value.startswith(str(source) + os.sep):
            return str(destination) + value[len(str(source)):]
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        return value

    for path in destination.rglob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        updated = relocate(value)
        if value != updated:
            write_json(path, updated)
    return {"source_path": str(source), "source_files": before,
            "snapshot_files": inventory(destination)}


def new_run(workspace, kind, *, inputs=(), artifact_dir=None):
    workspace = guard_output(workspace)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    directory = workspace / "runs" / run_id
    guard_output(directory, inputs)
    artifact = guard_output(artifact_dir, inputs) if artifact_dir else directory / "artifact"
    if artifact.exists():
        raise CLIError("output_exists", f"Choose a new artifact directory: {artifact}")
    directory.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "run_id": run_id, "kind": kind, "status": "running",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "directory": str(directory), "artifact_dir": str(artifact)}
    write_json(directory / "run.json", record)
    return record


def finish_run(run, **fields):
    run.update(fields)
    write_json(Path(run["directory"]) / "run.json", run)
    return read_json(Path(run["directory"]) / "run.json")


def history(workspace, run_id=None):
    root = Path(workspace).expanduser().resolve() / "runs"
    if run_id is not None:
        if Path(run_id).name != run_id or run_id.startswith("."):
            raise CLIError("run_id_invalid", "Invalid run ID")
        return read_json(root / run_id / "run.json")
    return [read_json(path) for path in sorted(root.glob("*/run.json"), reverse=True)]


def export_csv(path, response):
    labels = response.get("labels", [])
    ensemble = response.get("ensemble", {})
    rows = []
    for index, prediction in enumerate(response.get("predictions", [])):
        row = {"row": index, "sequence": prediction.get("sequence", ""),
               "peptide": prediction.get("peptide", ""), "protein": prediction.get("protein", ""),
               "score_kind": prediction.get("score_kind", response.get("score_kind", "probability")),
               "provenance_status": response.get("provenance_status", "not_checked"),
               "ensemble_method": ensemble.get("method", ""),
               "batch_digest": ensemble.get("batch_digest", ""),
               "batch_input_count": ensemble.get("input_count", len(response.get("predictions", []))),
               "contract_version": ensemble.get("contract_version", ""),
               "numpy_version": ensemble.get("numpy_version", ""),
               "predicted_labels": "|".join(prediction.get("predicted_labels", [])),
               "is_interaction": prediction.get("is_interaction", "")}
        for column, score in enumerate(prediction.get("scores", [])):
            label = labels[column] if column < len(labels) else str(column)
            row[f"score:{label}"] = score
        rows.append(row)
    if not rows:
        raise CLIError("predictions_empty", "No predictions to export")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
