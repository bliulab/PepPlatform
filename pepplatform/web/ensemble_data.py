"""Read-only dataset identity and isolated artifact snapshots for Web ensembles."""

from __future__ import annotations
from pepplatform.task_compat import normalize_task_type


import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .ensemble_math import EnsembleError

PROVENANCE_FILE = "web_ensemble_provenance.json"
# The unchanged profile inference service builds these from the frozen,
# separately hashed subject FASTA. They are disposable runtime state, not
# trained parameters or source evidence.
RUNTIME_DIRS = ("profile_generation/inference_cache", "profile_generation/blastdb")


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EnsembleError(
            "ensemble_evidence_missing",
            f"Required evidence is missing or invalid: {Path(path).name}",
        ) from exc
    if not isinstance(value, dict):
        raise EnsembleError("ensemble_evidence_invalid", "Evidence must be a JSON object")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [clean(x) for x in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_root(path):
    root = Path(path).resolve()
    if "experiment_paper" in root.parts:
        raise EnsembleError(
            "ensemble_paper_source_forbidden",
            "Paper experiment directories cannot be used by Web ensembles",
        )
    return root


def dataset_file(path, root):
    resolved = safe_root(path)
    if not resolved.is_relative_to(root):
        raise EnsembleError("ensemble_dataset_invalid", "Dataset file escapes its registered root")
    return resolved


def binary_labels(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isin(values, [0, 1])):
        raise ValueError("Labels must be binary")
    return values.astype(np.int32)


def read_split(dataset, split):
    """Read exactly one explicitly named split, never probe/read other splits."""
    if not split or Path(split).name != split or split.startswith("."):
        raise EnsembleError("ensemble_split_invalid", "Invalid dataset split")
    root = safe_root(dataset["root_path"])
    directory = (root / split).resolve()
    if not directory.is_relative_to(root):
        raise EnsembleError("ensemble_split_invalid", "Dataset split escapes its root")
    task = dataset["task_type"]
    labels_path = dataset_file(directory / "labels.csv", root)
    paths = []
    if normalize_task_type(task) == "pep_function":
        fasta = dataset_file(directory / "seqs.fasta", root)
        try:
            sequences, ids, current = [], [], []
            for line in fasta.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(">"):
                    if ids:
                        sequences.append("".join(current).upper())
                    ids.append(line[1:])
                    current = []
                elif line:
                    if not ids:
                        raise ValueError("FASTA sequence has no header")
                    current.append(line)
            if ids:
                sequences.append("".join(current).upper())
            with labels_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            labels = rows[0]
            y = binary_labels(rows[1:])
        except (OSError, ValueError, IndexError) as exc:
            raise EnsembleError(
                "ensemble_dataset_invalid", f"Invalid sequence dataset split: {split}"
            ) from exc
        inputs = {"sequences": sequences}
        paths = [fasta, labels_path]
        keys = sequences
        if y.shape != (len(sequences), len(labels)):
            raise EnsembleError(
                "ensemble_alignment_invalid", "FASTA rows and label rows do not match"
            )
    else:
        from .datasets import _pair_label, _pair_peptide, _pair_partner, _load_hlai_pseudoseqs

        pairs_path = dataset_file(directory / "pairs.csv", root)
        try:
            with pairs_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            pseudo_root = safe_root(
                (dataset.get("metadata") or {}).get("source_root") or root.parent
            )
            if (
                normalize_task_type(task) == "pep_hlai"
                and (pseudo_root / "pseudoseqs_decoded.csv").is_file()
            ):
                dataset_file(pseudo_root / "pseudoseqs_decoded.csv", pseudo_root)
            pseudo_map = (
                _load_hlai_pseudoseqs(pseudo_root)
                if normalize_task_type(task) == "pep_hlai"
                else {}
            )
            pairs = [
                {
                    "peptide": _pair_peptide(row),
                    "protein": _pair_partner(
                        row,
                        pseudo_map,
                        source_path=pairs_path,
                        row_number=index,
                        pseudoseq_path=pseudo_root / "pseudoseqs_decoded.csv",
                    ),
                }
                for index, row in enumerate(rows, start=1)
            ]
            ids = [
                str(row.get("sample_id") or row.get("id") or index)
                for index, row in enumerate(rows)
            ]
            y = binary_labels([_pair_label(row) for row in rows])
            if labels_path.exists():
                with labels_path.open(encoding="utf-8", newline="") as handle:
                    csv_labels = list(csv.reader(handle))[1:]
                if not np.array_equal(binary_labels(csv_labels).reshape(-1), y):
                    raise ValueError("pairs.csv and labels.csv disagree")
                paths.append(labels_path)
        except (OSError, ValueError, IndexError) as exc:
            raise EnsembleError(
                "ensemble_dataset_invalid", f"Invalid Pair dataset split: {split}"
            ) from exc
        if any(not p["peptide"] or not p["protein"] or not p["protein"].isalpha() for p in pairs):
            raise EnsembleError(
                "ensemble_partner_unresolved",
                "Pair inference requires resolved partner sequences, not allele identifiers",
            )
        labels = ["function" if normalize_task_type(task) == "pep_function" else "interaction"]
        inputs = {"pairs": pairs}
        keys = [(p["peptide"], p["protein"]) for p in pairs]
        paths.append(pairs_path)
        if pseudo_map:
            paths.append(pseudo_root / "pseudoseqs_decoded.csv")
    if not keys or any(not x for x in keys) or not np.all(np.isin(y, [0, 1])):
        raise EnsembleError(
            "ensemble_dataset_invalid", "Splits require nonempty inputs and binary labels"
        )
    sample_ids = [
        f"{index}:{identifier}:{digest_json(key)}"
        for index, (identifier, key) in enumerate(zip(ids, keys))
    ]
    identity = {
        "split": split,
        "count": len(keys),
        "sample_ids": sample_ids,
        "input_digest": digest_json(keys),
        "label_digest": digest_json(y.tolist()),
        "files": {str(path.resolve()): file_digest(path) for path in paths},
    }
    return {
        "inputs": inputs,
        "labels": labels,
        "y": y,
        "identity": identity,
        "keys": {digest_json(key) for key in keys},
    }


def validation_identity(dataset):
    train, val = read_split(dataset, "train"), read_split(dataset, "val")
    if train["labels"] != val["labels"]:
        raise EnsembleError("ensemble_label_mismatch", "Train and validation label columns differ")
    if train["keys"] & val["keys"]:
        raise EnsembleError(
            "ensemble_train_validation_overlap", "Train and validation contain overlapping inputs"
        )
    return {
        "schema_version": 1,
        "dataset_id": dataset["id"],
        "task_type": dataset["task_type"],
        "train": train["identity"],
        "validation": val["identity"],
    }


def verify_provenance(bundle, dataset, current):
    evidence = read_json(Path(bundle) / PROVENANCE_FILE)
    if evidence.get("schema_version") != 1 or evidence.get("dataset_id") != dataset["id"]:
        raise EnsembleError(
            "ensemble_provenance_invalid", "Member dataset provenance is incompatible"
        )
    if normalize_task_type(evidence.get("task_type")) != normalize_task_type(
        dataset["task_type"]
    ) or any(evidence.get(key) != current[key] for key in ("train", "validation")):
        raise EnsembleError(
            "ensemble_dataset_changed",
            "Member training/validation identity does not match the registered dataset",
        )
    return evidence


def tree_identity(root):
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(relative == name or relative.startswith(name + "/") for name in RUNTIME_DIRS):
            continue
        if path.is_symlink() and path.is_dir():
            raise EnsembleError(
                "ensemble_symlink_directory", "Artifact directory symlinks are not supported"
            )
        if path.is_file():
            safe_root(path)
            result[path.relative_to(root).as_posix()] = file_digest(path)
    return result


def snapshot_member(record, destination):
    source = safe_root(record["bundle_path"])
    before = tree_identity(source)

    def ignore_runtime(directory, names):
        return {
            name
            for name in names
            if (Path(directory) / name).relative_to(source).as_posix() in RUNTIME_DIRS
        }

    shutil.copytree(source, destination, symlinks=False, ignore=ignore_runtime)
    if tree_identity(destination) != before or tree_identity(source) != before:
        raise EnsembleError(
            "ensemble_member_changed", "Member changed while its snapshot was being created"
        )

    # JSON runtime references to files inside the original bundle follow the
    # independent snapshot. External pretrained identities are checked by the
    # unchanged member inference implementation.
    def relocate(value):
        if isinstance(value, str) and value.startswith(str(source) + "/"):
            return str(destination) + value[len(str(source)) :]
        if isinstance(value, list):
            return [relocate(x) for x in value]
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        return value

    for path in Path(destination).rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated = relocate(payload)
        if updated != payload:
            write_json(path, updated)
    return {
        "model_id": record["id"],
        "name": record["name"],
        "task_type": record["task_type"],
        "source_files": before,
        "snapshot_files": tree_identity(destination),
    }


def response_scores(response, labels, count, task, method):
    task = normalize_task_type(task)
    predictions = response.get("predictions") or []
    if response.get("labels") != labels or len(predictions) != count:
        raise EnsembleError(
            "ensemble_prediction_alignment", "Member prediction labels or row count changed"
        )
    kinds = {
        row.get("score_kind", response.get("score_kind", "probability")) for row in predictions
    }
    if method != "rank_mean" and kinds != {"probability"}:
        raise EnsembleError(
            "ensemble_probability_required",
            "Probability/logit fusion requires calibrated probability members",
        )
    scores = np.asarray([row.get("scores") for row in predictions], dtype=np.float64)
    expected = (count, len(labels))
    if scores.shape != expected or not np.all(np.isfinite(scores)):
        raise EnsembleError(
            "ensemble_prediction_invalid", "Member returned invalid score dimensions or values"
        )
    return scores if normalize_task_type(task) == "pep_function" else scores[:, 0]
