from pepplatform.hlai_input import canonical_hlai_columns
from pepplatform.task_compat import LEGACY_HLAI_COLUMN
from pepplatform.task_compat import normalize_task_type

import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pepplatform.sequence_lengths import (
    dataset_role_max_location,
    length_stats_are_current,
    scan_dataset_length_stats,
)
from pepplatform.top_ngram import configured_top_ngram_max_residues


SPLITS = ["train", "val", "test"]
AA_CHARS = set("ACDEFGHIKLMNPQRSTVWYXBZJUO")


def _issue(severity, code, message, split="", file="", params=None):
    payload = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if split:
        payload["split"] = split
    if file:
        payload["file"] = file
    if params:
        payload["params"] = params
    return payload


def _relative(root, path):
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return Path(path).name


def _stats(values):
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
    }


def _truncate(text, limit=80):
    text = str(text or "")
    return text if len(text) <= limit else f"{text[:limit]}..."


def _read_fasta(path, root, split, issues, preview_limit):
    records = []
    ids = []
    lengths = []
    duplicate_ids = set()
    seen_ids = set()
    invalid_count = 0
    current_id = None
    current_lines = []

    def flush():
        nonlocal invalid_count
        if current_id is None and not current_lines:
            return
        sequence = "".join(current_lines).strip().upper()
        sample_id = current_id or str(len(ids) + 1)
        if sample_id in seen_ids:
            duplicate_ids.add(sample_id)
        seen_ids.add(sample_id)
        ids.append(sample_id)
        lengths.append(len(sequence))
        invalid = sorted({char for char in sequence if char not in AA_CHARS})
        if invalid:
            invalid_count += 1
        if len(records) < preview_limit:
            records.append(
                {
                    "sample_id": sample_id,
                    "sequence": _truncate(sequence),
                    "sequence_length": len(sequence),
                    "invalid_residues": invalid,
                }
            )

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_id = line[1:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        flush()

    if duplicate_ids:
        issues.append(
            _issue(
                "warning",
                "duplicate_sample_id",
                f"{len(duplicate_ids)} duplicate sample id(s) found",
                split,
                _relative(root, path),
                {"count": len(duplicate_ids)},
            )
        )
    if invalid_count:
        issues.append(
            _issue(
                "warning",
                "noncanonical_residue",
                f"{invalid_count} sequence(s) contain non-standard residue characters",
                split,
                _relative(root, path),
                {"count": invalid_count},
            )
        )
    return {
        "count": len(ids),
        "ids": ids,
        "preview": records,
        "sequence_lengths": _stats(lengths),
    }


def _parse_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _label_summary_from_rows(header, rows, root, path, split, issues):
    label_columns = list(header or [])
    mode = (
        "multi_label"
        if len(label_columns) > 1
        else ("single_label" if len(label_columns) == 1 else "unknown")
    )
    positives = Counter()
    distribution = Counter()
    parse_errors = 0
    preview = []
    for row in rows:
        if mode == "multi_label":
            active = []
            for column, value in zip(label_columns, row):
                numeric = _parse_number(value)
                if numeric is None:
                    parse_errors += 1
                    continue
                if numeric > 0:
                    positives[column] += 1
                    active.append(column)
            if len(preview) < 20:
                preview.append(active)
        elif mode == "single_label":
            value = row[0] if row else ""
            if value == "":
                parse_errors += 1
            else:
                distribution[str(value)] += 1
            if len(preview) < 20:
                preview.append([str(value)] if value != "" else [])
    if parse_errors:
        issues.append(
            _issue(
                "warning",
                "label_parse_error",
                f"{parse_errors} label value(s) could not be parsed",
                split,
                _relative(root, path),
                {"count": parse_errors},
            )
        )
    return {
        "mode": mode,
        "columns": label_columns,
        "row_count": len(rows),
        "positive_counts": dict(positives),
        "value_counts": dict(distribution),
        "preview_labels": preview,
    }


def _read_labels(path, root, split, issues):
    with open(path, "r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        issues.append(
            _issue("error", "empty_labels", "labels.csv is empty", split, _relative(root, path))
        )
        return {
            "mode": "unknown",
            "columns": [],
            "row_count": 0,
            "positive_counts": {},
            "value_counts": {},
            "preview_labels": [],
        }
    return _label_summary_from_rows(rows[0], rows[1:], root, path, split, issues)


def _pair_value(row, names):
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value:
            return value
    return ""


def _read_pair_csv(path, root, split, issues, preview_limit):
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    column_set = set(columns)
    required_groups = {
        "peptide": {"peptide", "pep"},
        "partner": {"protein", "hla", LEGACY_HLAI_COLUMN, "pair_pseudo", "hla_pair"},
        "label": {"label", "tgt", "bind"},
    }
    for group, choices in required_groups.items():
        if not (column_set & choices):
            issues.append(
                _issue(
                    "error",
                    "missing_column",
                    f"pairs CSV is missing a {group} column",
                    split,
                    _relative(root, path),
                    {"column": group},
                )
            )
    duplicate_ids = set()
    seen_ids = set()
    label_counts = Counter()
    peptide_lengths = []
    partner_lengths = []
    preview = []
    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", "") or "").strip()
        if sample_id:
            if sample_id in seen_ids:
                duplicate_ids.add(sample_id)
            seen_ids.add(sample_id)
        peptide = _pair_value(row, ["peptide", "pep"]).upper()
        partner = _pair_value(
            canonical_hlai_columns(row), ["protein", "hla", "pair_pseudo", "hla_pair"]
        ).upper()
        label = _pair_value(row, ["label", "tgt", "bind"])
        if label != "":
            label_counts[label] += 1
        peptide_lengths.append(len(peptide))
        partner_lengths.append(len(partner))
        if len(preview) < preview_limit:
            preview.append(
                {
                    "sample_id": sample_id or str(index),
                    "peptide": _truncate(peptide),
                    "partner": _truncate(partner),
                    "peptide_length": len(peptide),
                    "partner_length": len(partner),
                    "label": label,
                }
            )
    if duplicate_ids:
        issues.append(
            _issue(
                "warning",
                "duplicate_sample_id",
                f"{len(duplicate_ids)} duplicate sample id(s) found",
                split,
                _relative(root, path),
                {"count": len(duplicate_ids)},
            )
        )
    return {
        "columns": columns,
        "row_count": len(rows),
        "label_counts": dict(label_counts),
        "peptide_lengths": _stats(peptide_lengths),
        "partner_lengths": _stats(partner_lengths),
        "preview": preview,
    }


def _merge_sequence_preview(sequence_preview, label_preview):
    rows = []
    for index, item in enumerate(sequence_preview):
        row = dict(item)
        labels = label_preview[index] if index < len(label_preview) else []
        row["labels"] = labels
        rows.append(row)
    return rows


def _missing_required(root, path, split, issues):
    issues.append(
        _issue(
            "error",
            "missing_required_file",
            "required file is missing",
            split,
            _relative(root, path),
        )
    )


def _sequence_split_report(root, split, require_pairs, preview_limit, issues):
    split_dir = root / split
    seq_path = split_dir / "seqs.fasta"
    label_path = split_dir / "labels.csv"
    pair_path = split_dir / "pairs.csv"
    files = {
        "seqs.fasta": seq_path.exists(),
        "labels.csv": label_path.exists(),
        "pairs.csv": pair_path.exists(),
    }
    if not seq_path.exists():
        _missing_required(root, seq_path, split, issues)
    if not label_path.exists():
        _missing_required(root, label_path, split, issues)
    if require_pairs and not pair_path.exists():
        _missing_required(root, pair_path, split, issues)

    sequence_info = {"count": 0, "preview": [], "sequence_lengths": _stats([])}
    label_info = {
        "mode": "unknown",
        "columns": [],
        "row_count": 0,
        "positive_counts": {},
        "value_counts": {},
        "preview_labels": [],
    }
    pair_info = None
    if seq_path.exists():
        sequence_info = _read_fasta(seq_path, root, split, issues, preview_limit)
    if label_path.exists():
        label_info = _read_labels(label_path, root, split, issues)
    if pair_path.exists():
        pair_info = _read_pair_csv(pair_path, root, split, issues, preview_limit)
    if (
        seq_path.exists()
        and label_path.exists()
        and sequence_info["count"] != label_info["row_count"]
    ):
        issues.append(
            _issue(
                "error",
                "row_count_mismatch",
                "sequence count does not match label row count",
                split,
                _relative(root, label_path),
                {"variant": "sequence_label"},
            )
        )
    if sequence_info["count"] == 0 and (seq_path.exists() or label_path.exists()):
        issues.append(
            _issue(
                "error",
                "empty_split",
                "split contains no sequences",
                split,
                params={"variant": "sequences"},
            )
        )
    preview = _merge_sequence_preview(sequence_info["preview"], label_info["preview_labels"])
    return {
        "name": split,
        "sample_count": sequence_info["count"],
        "files": files,
        "columns": {
            "labels": label_info["columns"],
            "pairs": pair_info["columns"] if pair_info else [],
        },
        "label_mode": label_info["mode"],
        "label_counts": {
            "positive": label_info["positive_counts"],
            "values": label_info["value_counts"],
        },
        "sequence_lengths": sequence_info["sequence_lengths"],
        "peptide_lengths": pair_info["peptide_lengths"] if pair_info else _stats([]),
        "partner_lengths": pair_info["partner_lengths"] if pair_info else _stats([]),
        "preview": preview,
    }


def _pair_split_report(root, split, preview_limit, issues):
    split_dir = root / split
    pair_path = split_dir / "pairs.csv"
    label_path = split_dir / "labels.csv"
    sequence_path = split_dir / "seqs.fasta"
    files = {
        "pairs.csv": pair_path.exists(),
        "labels.csv": label_path.exists(),
        "seqs.fasta": sequence_path.exists(),
    }
    if not pair_path.exists():
        _missing_required(root, pair_path, split, issues)
    if not label_path.exists():
        _missing_required(root, label_path, split, issues)
    pair_info = {
        "columns": [],
        "row_count": 0,
        "label_counts": {},
        "peptide_lengths": _stats([]),
        "partner_lengths": _stats([]),
        "preview": [],
    }
    label_info = {
        "mode": "unknown",
        "columns": [],
        "row_count": 0,
        "positive_counts": {},
        "value_counts": {},
        "preview_labels": [],
    }
    sequence_count = None
    sequence_lengths = _stats([])
    if pair_path.exists():
        pair_info = _read_pair_csv(pair_path, root, split, issues, preview_limit)
    if label_path.exists():
        label_info = _read_labels(label_path, root, split, issues)
    if sequence_path.exists():
        sequence_info = _read_fasta(sequence_path, root, split, issues, preview_limit)
        sequence_count = sequence_info["count"]
        sequence_lengths = sequence_info["sequence_lengths"]
        if sequence_count != pair_info["row_count"]:
            issues.append(
                _issue(
                    "error",
                    "row_count_mismatch",
                    "sequence count does not match pair row count",
                    split,
                    _relative(root, sequence_path),
                    {"variant": "sequence_pair"},
                )
            )
    if (
        pair_path.exists()
        and label_path.exists()
        and pair_info["row_count"] != label_info["row_count"]
    ):
        issues.append(
            _issue(
                "error",
                "row_count_mismatch",
                "pair row count does not match label row count",
                split,
                _relative(root, label_path),
                {"variant": "pair_label"},
            )
        )
    if pair_info["row_count"] == 0 and (pair_path.exists() or label_path.exists()):
        issues.append(
            _issue(
                "error",
                "empty_split",
                "split contains no pairs",
                split,
                params={"variant": "pairs"},
            )
        )
    return {
        "name": split,
        "sample_count": pair_info["row_count"],
        "files": files,
        "columns": {
            "pairs": pair_info["columns"],
            "labels": label_info["columns"],
        },
        "label_mode": label_info["mode"] if label_info["mode"] != "unknown" else "single_label",
        "label_counts": {
            "positive": label_info["positive_counts"],
            "values": label_info["value_counts"] or pair_info["label_counts"],
        },
        "sequence_count": sequence_count,
        "sequence_lengths": sequence_lengths,
        "peptide_lengths": pair_info["peptide_lengths"],
        "partner_lengths": pair_info["partner_lengths"],
        "preview": pair_info["preview"],
    }


def _pair_csv_bundle_split_report(root, split, preview_limit, issues):
    path = root / f"{split}.csv"
    files = {f"{split}.csv": path.exists()}
    if not path.exists():
        _missing_required(root, path, split, issues)
        return {
            "name": split,
            "sample_count": 0,
            "files": files,
            "columns": {"pairs": [], "labels": []},
            "label_mode": "single_label",
            "label_counts": {"positive": {}, "values": {}},
            "sequence_lengths": _stats([]),
            "peptide_lengths": _stats([]),
            "partner_lengths": _stats([]),
            "preview": [],
        }
    pair_info = _read_pair_csv(path, root, split, issues, preview_limit)
    if pair_info["row_count"] == 0:
        issues.append(
            _issue(
                "error",
                "empty_split",
                "split contains no pairs",
                split,
                _relative(root, path),
                {"variant": "pairs"},
            )
        )
    return {
        "name": split,
        "sample_count": pair_info["row_count"],
        "files": files,
        "columns": {"pairs": pair_info["columns"], "labels": []},
        "label_mode": "single_label",
        "label_counts": {"positive": {}, "values": pair_info["label_counts"]},
        "sequence_lengths": _stats([]),
        "peptide_lengths": pair_info["peptide_lengths"],
        "partner_lengths": pair_info["partner_lengths"],
        "preview": pair_info["preview"],
    }


def build_dataset_report(dataset, preview_limit=5):
    try:
        preview_limit = max(1, min(int(preview_limit), 20))
    except (TypeError, ValueError):
        preview_limit = 5
    root = Path(dataset["root_path"])
    issues = []
    split_reports = []
    metadata = dataset.get("metadata") or {}
    if metadata.get("validation_error"):
        issues.append(
            _issue("error", "existing_validation_error", str(metadata["validation_error"]))
        )
    if not root.exists():
        issues.append(_issue("error", "dataset_root_missing", "dataset root path does not exist"))
    else:
        fmt = dataset.get("format")
        task_type = dataset.get("task_type")
        if fmt == "sequence_dir":
            require_pairs = normalize_task_type(task_type) == "pep_hlaii"
            for split in SPLITS:
                if split == "test" and not (root / split).exists():
                    issues.append(
                        _issue(
                            "warning",
                            "missing_optional_test_split",
                            "optional test split is missing",
                            split,
                        )
                    )
                    continue
                split_reports.append(
                    _sequence_split_report(root, split, require_pairs, preview_limit, issues)
                )
        elif fmt == "pair_split_dir":
            for split in SPLITS:
                if split == "test" and not (root / split).exists():
                    issues.append(
                        _issue(
                            "warning",
                            "missing_optional_test_split",
                            "optional test split is missing",
                            split,
                        )
                    )
                    continue
                split_reports.append(_pair_split_report(root, split, preview_limit, issues))
        elif fmt == "pair_csv_bundle":
            for split in SPLITS:
                if split == "test" and not (root / "test.csv").exists():
                    issues.append(
                        _issue(
                            "warning",
                            "missing_optional_test_split",
                            "optional test.csv is missing",
                            split,
                        )
                    )
                    continue
                split_reports.append(
                    _pair_csv_bundle_split_report(root, split, preview_limit, issues)
                )
        elif fmt == "pair_root":
            issues.append(
                _issue(
                    "warning",
                    "legacy_pair_root",
                    "legacy pair_root datasets should be normalized before preview",
                )
            )
        else:
            issues.append(
                _issue(
                    "error",
                    "unsupported_format",
                    f"unsupported dataset format: {fmt}",
                    params={"format": fmt or ""},
                )
            )

    has_errors = any(issue["severity"] == "error" for issue in issues)
    has_warnings = any(issue["severity"] == "warning" for issue in issues)
    label_columns = []
    label_mode = "unknown"
    for split in split_reports:
        columns = split.get("columns", {}).get("labels") or []
        if columns:
            label_columns = columns
            label_mode = split.get("label_mode") or label_mode
            break
    total_samples = sum(int(split.get("sample_count") or 0) for split in split_reports)
    length_stats = metadata.get("length_stats")
    if not length_stats_are_current(length_stats) and root.exists():
        length_stats = scan_dataset_length_stats(
            root,
            dataset.get("format", ""),
            source_root=metadata.get("source_root"),
        )
    top_ngram_limit = configured_top_ngram_max_residues()
    roles = (
        ("sequence",)
        if normalize_task_type(dataset.get("task_type")) == "pep_function"
        else ("peptide", "partner")
    )
    for role in roles:
        maximum, split = dataset_role_max_location(length_stats, role)
        if maximum > top_ngram_limit:
            issues.append(
                _issue(
                    "warning",
                    "top_ngram_sequence_limit_exceeded",
                    f"{role} sequence length {maximum} exceeds the Top-n-gram "
                    f"platform limit of {top_ngram_limit}",
                    split,
                    params={
                        "role": role,
                        "length": maximum,
                        "limit": top_ngram_limit,
                    },
                )
            )
    has_errors = any(issue["severity"] == "error" for issue in issues)
    has_warnings = any(issue["severity"] == "warning" for issue in issues)
    return {
        "dataset_id": dataset["id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "valid": not has_errors,
        "status": "error" if has_errors else ("warning" if has_warnings else "valid"),
        "task_type": dataset["task_type"],
        "format": dataset["format"],
        "summary": {
            "total_samples": total_samples,
            "splits": {
                split["name"]: int(split.get("sample_count") or 0) for split in split_reports
            },
            "has_test_split": any(split["name"] == "test" for split in split_reports),
            "label_mode": label_mode,
            "label_columns": label_columns,
        },
        "splits": split_reports,
        "issues": issues,
    }
