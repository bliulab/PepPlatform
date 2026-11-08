from pepplatform.task_compat import normalize_task_type, install_legacy_exports, LEGACY_HLAI_COLUMN

import csv
import hashlib
import random
import shutil
import uuid
from pathlib import Path

from pepplatform.hlai_input import HLAIInputError, hla_allele, require_hlai_partner_sequence

from pepplatform.sequence_lengths import (
    length_stats_are_current,
    scan_dataset_length_stats as _scan_dataset_length_stats,
)

from .capabilities import builtin_dataset_specs
from .dataset_uploads import (
    DatasetUploadLimits,
    DatasetValidationError,
    extract_dataset_archive,
)


def _dataset_hla_allele(row, default=""):
    try:
        return hla_allele(row, default)
    except HLAIInputError as exc:
        raise DatasetValidationError(str(exc)) from exc


def scan_dataset_length_stats(*args, **kwargs):
    try:
        return _scan_dataset_length_stats(*args, **kwargs)
    except HLAIInputError as exc:
        raise DatasetValidationError(str(exc)) from exc


PAIR_SPLIT_DIR = "pair_split_dir"
CONCAT_LINKER = "GGG"


def seed_builtin_datasets(store, paths):
    project_root = paths.project_root
    seeded = []
    for spec in builtin_dataset_specs(project_root):
        existing = store.get_dataset(spec["id"])
        if existing and existing["id"] != spec["id"]:
            seeded.append(existing)
            continue
        root_path = Path(spec["root_path"])
        if not root_path.exists():
            continue
        metadata = {"managed": False}
        fmt = spec["format"]
        if fmt == "pair_root" and normalize_task_type(spec["task_type"]) in {
            "pep_protein_interaction",
            "pep_hlai",
        }:
            try:
                root_path, metadata = normalize_builtin_pair_root(spec, paths)
                fmt = PAIR_SPLIT_DIR
            except DatasetValidationError as exc:
                metadata["validation_error"] = str(exc)
        elif fmt == PAIR_SPLIT_DIR:
            try:
                metadata.update(
                    validate_pair_split_dir(
                        root_path,
                        require_sequence_view=normalize_task_type(spec["task_type"]) == "pep_hlaii",
                    )
                )
                if normalize_task_type(spec["task_type"]) in {
                    "pep_protein_interaction",
                    "pep_hlai",
                }:
                    source_root = root_path.parent
                    metadata["source_root"] = str(source_root)
                    metadata["sequence_view"] = "peptide_GGG_partner"
                    if spec.get("feature_dir"):
                        metadata["feature_dir"] = str(Path(spec["feature_dir"]))
                    metadata["length_stats"] = scan_dataset_length_stats(
                        root_path,
                        fmt,
                        source_root=source_root,
                    )
            except DatasetValidationError as exc:
                metadata["validation_error"] = str(exc)
        elif fmt == "sequence_dir":
            try:
                metadata.update(
                    validate_sequence_dir(
                        root_path,
                        require_pairs=normalize_task_type(spec["task_type"]) == "pep_hlaii",
                    )
                )
            except DatasetValidationError:
                pass
        seeded.append(
            store.upsert_dataset(
                spec["id"],
                spec["name"],
                spec["task_type"],
                "builtin",
                fmt,
                root_path,
                metadata,
            )
        )
    return seeded


def _unwrap_single_directory(root):
    entries = [entry for entry in root.iterdir() if entry.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return root


def _normalize_uploaded_tsv_files(root):
    """Adapt uploaded tables to the CSV layout consumed by existing readers."""
    for split in ("train", "val", "test"):
        for relative in (f"{split}.tsv", f"{split}/labels.tsv", f"{split}/pairs.tsv"):
            source = root / relative
            target = source.with_suffix(".csv")
            # CSV always wins, including when the unused TSV is malformed.
            if target.exists() or not source.is_file():
                continue
            try:
                with (
                    source.open("r", encoding="utf-8", newline="") as incoming,
                    target.open("x", encoding="utf-8", newline="") as outgoing,
                ):
                    reader = csv.reader(incoming, delimiter="\t", strict=True)
                    csv.writer(outgoing).writerows(reader)
            except (csv.Error, UnicodeError) as exc:
                raise DatasetValidationError(f"invalid TSV file {relative}: {exc}") from exc


def _txt_table_delimiter(incoming):
    """Inspect only the header, ignoring delimiters inside quoted fields."""
    delimiters = set()
    quoted = False
    field_start = True
    for line in incoming:
        index = 0
        while index < len(line):
            char = line[index]
            if quoted:
                if char == '"':
                    if index + 1 < len(line) and line[index + 1] == '"':
                        index += 1
                    else:
                        quoted = False
            elif char == '"' and field_start:
                quoted = True
                field_start = False
            elif char in {",", "\t"}:
                delimiters.add(char)
                field_start = True
            elif char in {"\r", "\n"}:
                break
            else:
                field_start = False
            index += 1
        if not quoted:
            break
    incoming.seek(0)
    if len(delimiters) > 1:
        raise DatasetValidationError(
            "ambiguous header: use only commas or tabs outside quoted fields"
        )
    return next(iter(delimiters), ",")


def _convert_txt_table(source, target):
    with (
        source.open("r", encoding="utf-8-sig", newline="") as incoming,
        target.open("x", encoding="utf-8", newline="") as outgoing,
    ):
        delimiter = _txt_table_delimiter(incoming)
        reader = csv.reader(incoming, delimiter=delimiter, strict=True)
        writer = csv.writer(outgoing)
        header = next(reader, None)
        if not header or any(not cell.strip() for cell in header):
            raise DatasetValidationError("must contain a nonempty header")
        writer.writerow(header)
        count = 0
        for row in reader:
            if len(row) != len(header):
                raise DatasetValidationError(
                    f"line {reader.line_num}: cell count does not match header"
                )
            writer.writerow(row)
            count += 1
        if not count:
            raise DatasetValidationError("must contain a header and at least one row")
    return delimiter


def _convert_txt_sequences(source, target):
    with (
        source.open("r", encoding="utf-8-sig", newline=None) as incoming,
        target.open("x", encoding="utf-8", newline="\n") as outgoing,
    ):
        count = 0
        for line_number, line in enumerate(incoming, start=1):
            sequence = line.strip()
            if not sequence or not sequence.isascii() or not sequence.isalpha():
                raise DatasetValidationError(
                    f"line {line_number}: expected one nonempty sequence of ASCII letters; "
                    "use .fasta for FASTA records"
                )
            count += 1
            outgoing.write(f">s{count}\n{sequence}\n")
        if not count:
            raise DatasetValidationError("must contain at least one sequence")


def _upload_file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_uploaded_txt_files(root, archive_root):
    """Convert new uploads only; leave canonical files and their readers intact."""
    conversions = []
    for split in ("train", "val", "test"):
        for relative in (
            f"{split}.txt",
            f"{split}/labels.txt",
            f"{split}/pairs.txt",
            f"{split}/seqs.txt",
        ):
            source = root / relative
            sequence_file = source.name == "seqs.txt"
            target = source.with_suffix(".fasta" if sequence_file else ".csv")
            # Called after TSV conversion: CSV > TSV > TXT, FASTA > TXT.
            if target.exists() or not source.is_file():
                continue
            record = {"kind": "sequences" if sequence_file else "table"}
            try:
                if sequence_file:
                    _convert_txt_sequences(source, target)
                else:
                    record["delimiter"] = _convert_txt_table(source, target)
            except (csv.Error, UnicodeError, DatasetValidationError) as exc:
                raise DatasetValidationError(f"invalid TXT file {relative}: {exc}") from exc
            record.update(
                {
                    "source": source.relative_to(archive_root).as_posix(),
                    "target": target.relative_to(archive_root).as_posix(),
                    "source_sha256": _upload_file_sha256(source),
                    "target_sha256": _upload_file_sha256(target),
                }
            )
            conversions.append(record)
    return conversions


def _count_fasta_records(path):
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def _read_label_count(path):
    with open(path, "r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) <= 1:
        raise DatasetValidationError(f"{path} must contain a header and at least one row")
    return len(rows) - 1


def _pair_label(row):
    return row.get("label", row.get("tgt", row.get("bind", "")))


def _pair_peptide(row):
    return str(row.get("peptide", row.get("pep", ""))).strip().upper()


def _pair_partner(
    row, pseudo_map=None, *, source_path="<pair row>", row_number=None, pseudoseq_path=None
):
    if row.get("pair_pseudo"):
        partner = str(row.get("pair_pseudo", "")).strip().upper()
        resolved = True
    else:
        raw = str(
            row.get("protein", _dataset_hla_allele(row, default=row.get("hla_pair", "")))
        ).strip()
        mapped = bool(pseudo_map and raw in pseudo_map)
        partner = pseudo_map[raw] if mapped else raw.upper()
        resolved = bool(mapped or (row.get("protein") and raw == str(row["protein"]).strip()))
    hla = str(_dataset_hla_allele(row) or "").strip()
    if hla:
        try:
            require_hlai_partner_sequence(
                partner,
                allele=hla,
                resolved=resolved,
                source_path=source_path,
                row_number=row_number,
                pseudoseq_path=pseudoseq_path,
            )
        except HLAIInputError as exc:
            raise DatasetValidationError(str(exc)) from exc
    return partner


def _combined_sequence(row, pseudo_map=None):
    if row.get("combined_sequence"):
        return str(row.get("combined_sequence", "")).strip().upper()
    peptide = _pair_peptide(row)
    partner = _pair_partner(row, pseudo_map)
    if not peptide or not partner:
        return ""
    return f"{peptide}{CONCAT_LINKER}{partner}"


def _read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_labels(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"])
        for row in rows:
            writer.writerow([_pair_label(row)])


def _write_sequence_view(path, split, rows, pseudo_map=None):
    with open(path, "w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            sequence = _combined_sequence(row, pseudo_map)
            if not sequence:
                continue
            sample_id = str(row.get("sample_id", "")).strip() or f"{split}_{index}"
            handle.write(f">{sample_id}\n{sequence}\n")


def _write_pair_split(root, split, fieldnames, rows, pseudo_map=None):
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    with open(split_dir / "pairs.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_labels(split_dir / "labels.csv", rows)
    _write_sequence_view(split_dir / "seqs.fasta", split, rows, pseudo_map)


def _split_pair_rows(rows, val_ratio=0.1, seed=42):
    positives = [row for row in rows if int(float(_pair_label(row))) == 1]
    negatives = [row for row in rows if int(float(_pair_label(row))) == 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positive_val_count = int(len(positives) * val_ratio)
    negative_val_count = int(len(negatives) * val_ratio)
    val_rows = positives[:positive_val_count] + negatives[:negative_val_count]
    train_rows = positives[positive_val_count:] + negatives[negative_val_count:]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def _load_hlai_pseudoseqs(root):
    path = root / "pseudoseqs_decoded.csv"
    if not path.exists():
        return {}
    mapping = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            hla = str(_dataset_hla_allele(row)).strip()
            pseudo = str(row.get("pseudoseq", "")).strip().upper()
            status = str(row.get("status", "ok")).strip().lower()
            if hla and pseudo and status in {"", "ok"}:
                mapping[hla] = pseudo
    return mapping


def _builtin_pair_sources(spec):
    root = Path(spec["root_path"])
    if normalize_task_type(spec["task_type"]) == "pep_protein_interaction":
        return {
            "root": root,
            "train": root / "processed" / "train.csv",
            "val": None,
            "test": root / "processed" / "test_1440.csv",
            "pseudo_map": {},
        }
    base = root / "processed"
    return {
        "root": root,
        "train": base / "train.csv",
        "val": base / "val.csv",
        "test": base / "test.csv",
        "pseudo_map": _load_hlai_pseudoseqs(root),
    }


def normalize_builtin_pair_root(spec, paths, target_root=None, source_root=None):
    sources = _builtin_pair_sources(spec)
    if source_root is not None:
        sources["root"] = Path(source_root)
    if not sources["train"].exists():
        raise DatasetValidationError(f"missing built-in train csv: {sources['train']}")
    if not sources["test"].exists():
        raise DatasetValidationError(f"missing built-in test csv: {sources['test']}")

    target = Path(target_root) if target_root is not None else paths.datasets / spec["id"]
    if (target / "train" / "pairs.csv").exists() and (target / "val" / "pairs.csv").exists():
        metadata = validate_pair_split_dir(target)
    else:
        shutil.rmtree(target, ignore_errors=True)
        train_fieldnames, train_rows = _read_csv_rows(sources["train"])
        if not train_fieldnames or not train_rows:
            raise DatasetValidationError(f"{sources['train']} must contain pair rows")
        if sources["val"] and sources["val"].exists():
            val_fieldnames, val_rows = _read_csv_rows(sources["val"])
        else:
            val_fieldnames = train_fieldnames
            train_rows, val_rows = _split_pair_rows(train_rows)
        test_fieldnames, test_rows = _read_csv_rows(sources["test"])
        _write_pair_split(target, "train", train_fieldnames, train_rows, sources["pseudo_map"])
        _write_pair_split(
            target, "val", val_fieldnames or train_fieldnames, val_rows, sources["pseudo_map"]
        )
        _write_pair_split(
            target, "test", test_fieldnames or train_fieldnames, test_rows, sources["pseudo_map"]
        )
        metadata = validate_pair_split_dir(target)
    metadata.update(
        {
            "managed": False,
            "normalized_from": str(sources["root"]),
            "source_root": str(sources["root"]),
            "sequence_view": "peptide_GGG_partner",
        }
    )
    metadata["length_stats"] = scan_dataset_length_stats(
        target,
        PAIR_SPLIT_DIR,
        source_root=sources["root"],
    )
    return target, metadata


def _validate_hlaii_pairs(path, expected_count):
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        rows = list(reader)
    required = {"peptide", "pair_pseudo", "label"}
    if not required.issubset(header):
        raise DatasetValidationError(
            f"{path} must include peptide, pair_pseudo, and label columns for pep_hlaii"
        )
    if len(rows) != expected_count:
        raise DatasetValidationError(f"{path} row count does not match sequence count")


def validate_sequence_dir(root, require_pairs=False):
    root = Path(root)
    metadata = {"splits": {}}
    for split in ["train", "val"]:
        seq_path = root / split / "seqs.fasta"
        label_path = root / split / "labels.csv"
        if not seq_path.exists() or not label_path.exists():
            raise DatasetValidationError(f"missing {split}/seqs.fasta or {split}/labels.csv")
        seq_count = _count_fasta_records(seq_path)
        with open(label_path, "r", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if len(rows) <= 1:
            raise DatasetValidationError(
                f"{split}/labels.csv must contain a header and at least one row"
            )
        if seq_count != len(rows) - 1:
            raise DatasetValidationError(f"{split} sequence count does not match label count")
        if require_pairs:
            pair_path = root / split / "pairs.csv"
            if not pair_path.exists():
                raise DatasetValidationError(f"missing {split}/pairs.csv")
            _validate_hlaii_pairs(pair_path, seq_count)
        metadata["splits"][split] = seq_count
    test_seq = root / "test" / "seqs.fasta"
    test_labels = root / "test" / "labels.csv"
    if test_seq.exists() and test_labels.exists():
        test_count = _count_fasta_records(test_seq)
        if require_pairs:
            pair_path = root / "test" / "pairs.csv"
            if not pair_path.exists():
                raise DatasetValidationError("missing test/pairs.csv")
            _validate_hlaii_pairs(pair_path, test_count)
        metadata["splits"]["test"] = test_count
    metadata["length_stats"] = scan_dataset_length_stats(root, "sequence_dir")
    return metadata


def _read_pair_header(path):
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        rows = list(reader)
    peptide_ok = bool({"peptide", "pep"} & header)
    partner_ok = bool({"protein", "hla", "pair_pseudo", "hla_pair", LEGACY_HLAI_COLUMN} & header)
    label_ok = bool({"label", "tgt", "bind"} & header)
    if not peptide_ok or not partner_ok or not label_ok:
        raise DatasetValidationError(
            f"{path.name} must include peptide/pep, protein/hla/pair_pseudo/hla_pair, and label/tgt/bind columns"
        )
    for row in rows:
        _dataset_hla_allele(row)
    return rows


def validate_pair_dir(root):
    root = Path(root)
    metadata = {"splits": {}}
    for split in ["train", "val"]:
        path = root / f"{split}.csv"
        if not path.exists():
            raise DatasetValidationError(f"missing {split}.csv")
        metadata["splits"][split] = len(_read_pair_header(path))
    test_path = root / "test.csv"
    if test_path.exists():
        metadata["splits"]["test"] = len(_read_pair_header(test_path))
    metadata["length_stats"] = scan_dataset_length_stats(root, "pair_csv_bundle")
    return metadata


def validate_pair_split_dir(root, require_sequence_view=False):
    root = Path(root)
    metadata = {"splits": {}, "has_sequence_view": True}
    for split in ["train", "val"]:
        _validate_pair_split(
            root, split, require_sequence_view=require_sequence_view, metadata=metadata
        )
    test_dir = root / "test"
    if test_dir.exists():
        _validate_pair_split(
            root, "test", require_sequence_view=require_sequence_view, metadata=metadata
        )
    metadata["length_stats"] = scan_dataset_length_stats(root, PAIR_SPLIT_DIR)
    return metadata


def _validate_pair_split(root, split, require_sequence_view, metadata):
    split_dir = root / split
    pair_path = split_dir / "pairs.csv"
    label_path = split_dir / "labels.csv"
    if not pair_path.exists() or not label_path.exists():
        raise DatasetValidationError(f"missing {split}/pairs.csv or {split}/labels.csv")
    pair_rows = _read_pair_header(pair_path)
    label_count = _read_label_count(label_path)
    if len(pair_rows) != label_count:
        raise DatasetValidationError(f"{split} pair count does not match label count")
    sequence_path = split_dir / "seqs.fasta"
    if require_sequence_view and not sequence_path.exists():
        raise DatasetValidationError(f"missing {split}/seqs.fasta")
    if sequence_path.exists():
        sequence_count = _count_fasta_records(sequence_path)
        if sequence_count != len(pair_rows):
            raise DatasetValidationError(f"{split} sequence count does not match pair count")
    else:
        metadata["has_sequence_view"] = False
    metadata["splits"][split] = len(pair_rows)


class DatasetService:
    def __init__(self, store, paths, upload_limits=None):
        self.store = store
        self.paths = paths
        self.upload_limits = (
            upload_limits if upload_limits is not None else DatasetUploadLimits.from_env()
        )

    def ensure_length_stats(self, dataset):
        metadata = dict(dataset.get("metadata") or {})
        if length_stats_are_current(metadata.get("length_stats")):
            return dataset
        metadata["length_stats"] = scan_dataset_length_stats(
            dataset["root_path"],
            dataset["format"],
            source_root=metadata.get("source_root"),
        )
        return self.store.upsert_dataset(
            dataset["id"],
            dataset["name"],
            dataset["task_type"],
            dataset["source"],
            dataset["format"],
            dataset["root_path"],
            metadata,
            owner_user_id=dataset.get("owner_user_id"),
        )

    def upload_archive(self, task_type, name, filename, content, owner_user_id=None):
        task_type = normalize_task_type(task_type)
        if not filename.lower().endswith(".zip"):
            raise DatasetValidationError("dataset upload must be a .zip archive")
        if normalize_task_type(task_type) not in {
            "pep_function",
            "pep_protein_interaction",
            "pep_hlaii",
            "pep_hlai",
        }:
            raise DatasetValidationError(f"unsupported task_type={task_type}")
        dataset_id = uuid.uuid4().hex
        target_dir = self.paths.datasets / dataset_id
        target_dir.mkdir(parents=True, exist_ok=False)
        try:
            extract_dataset_archive(content, target_dir, self.upload_limits)
            root = _unwrap_single_directory(target_dir)
            _normalize_uploaded_tsv_files(root)
            txt_conversions = _normalize_uploaded_txt_files(root, target_dir)
            if normalize_task_type(task_type) == "pep_function":
                fmt = "sequence_dir"
                metadata = validate_sequence_dir(root)
            elif normalize_task_type(task_type) == "pep_hlaii":
                fmt = PAIR_SPLIT_DIR
                metadata = validate_pair_split_dir(root, require_sequence_view=True)
            else:
                fmt = PAIR_SPLIT_DIR
                if (root / "train" / "pairs.csv").exists():
                    metadata = validate_pair_split_dir(root)
                else:
                    metadata = normalize_uploaded_pair_bundle(root)
                    root = root / "normalized"
            metadata["managed"] = True
            metadata["original_filename"] = filename
            if txt_conversions:
                metadata["upload_conversion"] = {
                    "version": 1,
                    "path_base": "archive_root",
                    "files": txt_conversions,
                }
            return self.store.upsert_dataset(
                dataset_id,
                name,
                task_type,
                "upload",
                fmt,
                root,
                metadata,
                owner_user_id=owner_user_id,
            )
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise


def normalize_uploaded_pair_bundle(root):
    root = Path(root)
    metadata = validate_pair_dir(root)
    target = root / "normalized"
    shutil.rmtree(target, ignore_errors=True)
    for split in metadata["splits"]:
        source = root / f"{split}.csv"
        fieldnames, rows = _read_csv_rows(source)
        _write_pair_split(target, split, fieldnames, rows)
    return validate_pair_split_dir(target)


install_legacy_exports(globals())
