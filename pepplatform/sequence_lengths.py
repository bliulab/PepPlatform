from pepplatform.task_compat import legacy_keyword, install_legacy_exports
import csv
from pathlib import Path
from typing import Iterable, Mapping

from pepplatform.hlai_input import hla_allele, require_hlai_partner_sequence


LENGTH_STATS_VERSION = 1
LENGTH_ROLES = ("sequence", "peptide", "partner")
DATASET_SPLITS = ("train", "val", "test")


def _normalized_sequence(value) -> str:
    return "".join(str(value or "").split()).upper()


def _length_stats(values: Iterable[int]) -> dict:
    normalized = [int(value) for value in values]
    if not normalized:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(normalized),
        "min": min(normalized),
        "max": max(normalized),
        "mean": round(sum(normalized) / len(normalized), 2),
    }


def _empty_role_values() -> dict[str, list[int]]:
    return {role: [] for role in LENGTH_ROLES}


def _fasta_lengths(path: Path) -> list[int]:
    lengths = []
    current = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    lengths.append(len(_normalized_sequence("".join(current))))
                current = []
            else:
                current.append(line)
    if current:
        lengths.append(len(_normalized_sequence("".join(current))))
    return lengths


def load_hlai_pseudosequence_map(root) -> dict[str, str]:
    path = Path(root) / "pseudoseqs_decoded.csv"
    if not path.is_file():
        return {}
    mapping = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            hla = str(hla_allele(row) or "").strip()
            pseudosequence = _normalized_sequence(row.get("pseudoseq", ""))
            status = str(row.get("status", "ok") or "").strip().lower()
            if hla and pseudosequence and status in {"", "ok"}:
                mapping[hla] = pseudosequence
    return mapping


def pair_sequences(
    row: Mapping[str, object],
    *,
    hlai_pseudosequences: Mapping[str, str] | None = None,
    source_path="<pair row>",
    row_number=None,
    pseudoseq_path=None,
    **legacy,
) -> tuple[str, str]:
    hlai_pseudosequences = legacy_keyword(legacy, "hlai_pseudosequences", hlai_pseudosequences)
    if legacy:
        raise TypeError(f"unexpected keywords: {sorted(legacy)}")
    peptide = _normalized_sequence(row.get("peptide", row.get("pep", "")))
    protein = str(row.get("protein", "") or "").strip()
    hla = str(hla_allele(row) or "").strip()
    pair_pseudo = _normalized_sequence(row.get("pair_pseudo", ""))
    hla_pair = str(row.get("hla_pair", "") or "").strip()
    partner = pair_pseudo or _normalized_sequence(protein or hla or hla_pair)
    if protein:
        partner = _normalized_sequence(protein)
    elif hlai_pseudosequences and hla in hlai_pseudosequences:
        partner = _normalized_sequence(hlai_pseudosequences[hla])
    if hla:
        require_hlai_partner_sequence(
            partner,
            allele=hla,
            resolved=bool(
                protein or pair_pseudo or (hlai_pseudosequences and hla in hlai_pseudosequences)
            ),
            source_path=source_path,
            row_number=row_number,
            pseudoseq_path=pseudoseq_path,
        )
    return peptide, partner


def _pair_lengths(
    path: Path,
    *,
    hlai_pseudosequences: Mapping[str, str] | None = None,
    pseudoseq_path=None,
) -> tuple[list[int], list[int]]:
    peptide_lengths = []
    partner_lengths = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            peptide, partner = pair_sequences(
                row,
                hlai_pseudosequences=hlai_pseudosequences,
                source_path=path,
                row_number=index,
                pseudoseq_path=pseudoseq_path,
            )
            peptide_lengths.append(len(peptide))
            partner_lengths.append(len(partner))
    return peptide_lengths, partner_lengths


def scan_dataset_length_stats(
    root,
    fmt: str,
    *,
    source_root=None,
) -> dict:
    root = Path(root)
    pseudosequence_root = Path(source_root) if source_root else root
    if (
        not source_root
        and not (root / "pseudoseqs_decoded.csv").is_file()
        and (root.parent / "pseudoseqs_decoded.csv").is_file()
    ):
        pseudosequence_root = root.parent
    pseudosequences = load_hlai_pseudosequence_map(pseudosequence_root)
    by_split = {}
    overall_values = _empty_role_values()

    for split in DATASET_SPLITS:
        role_values = _empty_role_values()
        if fmt in {"sequence_dir", "pair_split_dir"}:
            split_root = root / split
            fasta_path = split_root / "seqs.fasta"
            pair_path = split_root / "pairs.csv"
        elif fmt == "pair_csv_bundle":
            fasta_path = None
            pair_path = root / f"{split}.csv"
        else:
            continue

        if fasta_path is not None and fasta_path.is_file():
            role_values["sequence"] = _fasta_lengths(fasta_path)
        if pair_path.is_file():
            peptide, partner = _pair_lengths(
                pair_path,
                hlai_pseudosequences=pseudosequences,
                pseudoseq_path=pseudosequence_root / "pseudoseqs_decoded.csv",
            )
            role_values["peptide"] = peptide
            role_values["partner"] = partner
        if not any(role_values.values()):
            continue

        by_split[split] = {role: _length_stats(role_values[role]) for role in LENGTH_ROLES}
        for role in LENGTH_ROLES:
            overall_values[role].extend(role_values[role])

    return {
        "version": LENGTH_STATS_VERSION,
        "by_split": by_split,
        "overall": {role: _length_stats(overall_values[role]) for role in LENGTH_ROLES},
    }


def length_stats_are_current(value) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("version") == LENGTH_STATS_VERSION
        and isinstance(value.get("by_split"), Mapping)
        and isinstance(value.get("overall"), Mapping)
    )


def dataset_role_max(length_stats, role: str) -> int:
    if role not in LENGTH_ROLES or not length_stats_are_current(length_stats):
        return 0
    stats = length_stats.get("overall", {}).get(role, {})
    try:
        return max(0, int(stats.get("max") or 0))
    except (AttributeError, TypeError, ValueError):
        return 0


def dataset_role_min(length_stats, role: str) -> int:
    if role not in LENGTH_ROLES or not length_stats_are_current(length_stats):
        return 0
    stats = length_stats.get("overall", {}).get(role, {})
    try:
        return max(0, int(stats.get("min") or 0))
    except (AttributeError, TypeError, ValueError):
        return 0


def dataset_role_max_location(length_stats, role: str) -> tuple[int, str]:
    maximum = dataset_role_max(length_stats, role)
    if maximum <= 0:
        return 0, ""
    for split, values in length_stats.get("by_split", {}).items():
        try:
            split_maximum = int(values.get(role, {}).get("max") or 0)
        except (AttributeError, TypeError, ValueError):
            continue
        if split_maximum == maximum:
            return maximum, str(split)
    return maximum, ""


install_legacy_exports(globals())
