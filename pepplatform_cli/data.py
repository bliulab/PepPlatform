"""Read-only local data identities, with the existing peptide/partner rules."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pepplatform.task_compat import normalize_task_type
from .storage import CLIError, digest_json, file_digest, read_json


def fasta(path):
    sequences, identifiers, current = [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(">"):
            if identifiers:
                sequences.append("".join(current).upper())
            identifiers.append(line[1:])
            current = []
        elif line:
            if not identifiers:
                raise CLIError("fasta_invalid", "FASTA sequence has no header")
            current.append(line)
    if identifiers:
        sequences.append("".join(current).upper())
    if not sequences or any(not sequence or not sequence.isalpha() for sequence in sequences):
        raise CLIError("fasta_invalid", "FASTA requires nonempty amino-acid sequences")
    return identifiers, sequences


def binary(values):
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isin(array, [0, 1])):
        raise CLIError("labels_invalid", "Labels must be binary")
    return array.astype(np.int32)


def pair_rows(path, task, source_root):
    from pepplatform.web.datasets import _pair_peptide, _pair_partner, _load_hlai_pseudoseqs
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping_path = Path(source_root) / "pseudoseqs_decoded.csv"
    mapping = _load_hlai_pseudoseqs(Path(source_root)) if normalize_task_type(task) == "pep_hlai" else {}
    pairs = [{"peptide": _pair_peptide(row), "protein": _pair_partner(
        row, mapping, source_path=Path(path), row_number=i + 1, pseudoseq_path=mapping_path)}
        for i, row in enumerate(rows)]
    if not pairs or any(not p["peptide"] or not p["peptide"].isalpha() or
                        not p["protein"] or not p["protein"].isalpha() for p in pairs):
        raise CLIError("pairs_invalid", "Pairs require peptide and resolved partner sequences")
    return rows, pairs, [mapping_path] if mapping else []


def dataset(config):
    task = normalize_task_type(config.get("task_type", "pep_function"))
    root = Path(config["data_dir"]).expanduser().resolve()
    if not root.is_dir():
        raise CLIError("dataset_missing", f"Dataset directory does not exist: {root}")
    return {"task_type": task, "root_path": str(root),
            "source_root": str(Path(config.get("source_root") or root.parent).expanduser().resolve())}


def read_split(data, name):
    if not name or Path(name).name != name or name.startswith("."):
        raise CLIError("split_invalid", "Invalid split name")
    root = Path(data["root_path"]).resolve()
    directory = (root / name).resolve()
    if not directory.is_relative_to(root):
        raise CLIError("split_invalid", "Split escapes data directory")

    def checked(filename):
        path = (directory / filename).resolve()
        if not path.is_relative_to(root):
            raise CLIError("dataset_path_invalid", f"Dataset file escapes root: {path}")
        return path

    task = data["task_type"]
    if task == "pep_function":
        sequences_path, labels_path = checked("seqs.fasta"), checked("labels.csv")
        ids, sequences = fasta(sequences_path)
        with labels_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        labels, y = rows[0], binary(rows[1:])
        inputs, keys, paths = {"sequences": sequences}, sequences, [sequences_path, labels_path]
        if y.shape != (len(sequences), len(labels)):
            raise CLIError("alignment_invalid", "FASTA and label rows do not match")
    else:
        from pepplatform.web.datasets import _pair_label
        pairs_path = checked("pairs.csv")
        rows, pairs, extra = pair_rows(pairs_path, task, data["source_root"])
        ids = [str(row.get("sample_id") or row.get("id") or i) for i, row in enumerate(rows)]
        y = binary([_pair_label(row) for row in rows])
        labels, inputs = ["interaction"], {"pairs": pairs}
        keys = [(p["peptide"], p["protein"]) for p in pairs]
        paths = [pairs_path, *extra]
        labels_path = checked("labels.csv")
        if labels_path.exists():
            with labels_path.open(encoding="utf-8", newline="") as stream:
                values = list(csv.reader(stream))[1:]
            if not np.array_equal(binary(values).reshape(-1), y):
                raise CLIError("alignment_invalid", "pairs.csv and labels.csv disagree")
            paths.append(labels_path)
    if not labels or len(set(labels)) != len(labels) or any(not str(x).strip() for x in labels):
        raise CLIError("labels_invalid", "Output labels must be explicit and unique")
    sample_ids = [f"{i}:{identifier}:{digest_json(key)}" for i, (identifier, key) in enumerate(zip(ids, keys))]
    identity = {"split": name, "count": len(keys), "sample_ids": sample_ids,
                "input_digest": digest_json(keys), "label_digest": digest_json(y.tolist()),
                "files": {str(p): file_digest(p) for p in paths}}
    return {"inputs": inputs, "labels": labels, "y": y, "identity": identity,
            "keys": {digest_json(key) for key in keys}}


def validation_identity(data):
    train, val = read_split(data, "train"), read_split(data, "val")
    if train["labels"] != val["labels"]:
        raise CLIError("label_mismatch", "Train/Validation label columns differ")
    if train["keys"] & val["keys"]:
        raise CLIError("train_validation_overlap", "Train and Validation contain overlapping inputs")
    return {"schema_version": 1, "task_type": data["task_type"],
            "train": train["identity"], "validation": val["identity"]}


def same_data(left, right):
    """Relocated byte-identical inputs retain identity; paths are not evidence."""
    def logical(identity):
        return {**{k: v for k, v in identity.items() if k != "files"},
                "file_hashes": sorted(identity["files"].values())}
    return normalize_task_type(left.get("task_type")) == normalize_task_type(right.get("task_type")) and all(
        logical(left[k]) == logical(right[k]) for k in ("train", "validation"))


def prediction_inputs(path, task, source_root=None):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".json":
        inputs = read_json(path)
        expected = "sequences" if task == "pep_function" else "pairs"
        if set(inputs) != {expected} or not isinstance(inputs[expected], list) or not inputs[expected]:
            raise CLIError("input_invalid", f"Expected a nonempty {expected} list")
        if expected == "sequences":
            inputs[expected] = [str(x).strip().upper() for x in inputs[expected]]
            if any(not x or not x.isalpha() for x in inputs[expected]):
                raise CLIError("input_invalid", "Sequences must contain amino-acid letters")
        elif any(not isinstance(p, dict) or not p.get("peptide") or not p.get("protein")
                 or not str(p["protein"]).isalpha() for p in inputs[expected]):
            raise CLIError("input_invalid", "Pairs require peptide and resolved protein sequences")
        return inputs
    if task == "pep_function":
        return {"sequences": fasta(path)[1]}
    return {"pairs": pair_rows(path, task, source_root or path.parent)[1]}
