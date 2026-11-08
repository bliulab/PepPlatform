#!/usr/bin/env python3
"""Build role-aware, leakage-audited pair-task sequence profiles."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pepplatform.profile_generation import (  # noqa: E402
    ProfileGenerationConfig,
    build_profile_caches,
    index_subject_fasta,
    write_canonical_subject_fasta,
)
from pepplatform.sequence_profiles import (  # noqa: E402
    AA_ORDER as _AA_ORDER,
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
    FEATURE_AC_PSSM_BLOSUM62_HYBRID,
    FEATURE_CC_PSSM_BLOSUM62_HYBRID,
    FEATURE_PSFM,
    FEATURE_PSFM_BLOSUM62_HYBRID,
    FEATURE_PSSM,
    FEATURE_PSSM_BLOSUM62_HYBRID,
    FEATURE_TARGET_FREQUENCY,
    blosum62_score_profile,
    normalize_profile_sequence,
    parse_ascii_pssm as _strict_parse_ascii_pssm,
    sha256_file,
)
from pepplatform.top_ngram import (  # noqa: E402
    TOP_N_GRAM_FALLBACK_POLICY,
    TOP_N_GRAM_STRICT_PROFILE_POLICY,
    normalize_top_ngram_sequence,
)
from pepplatform.training_failures import run_with_failure_reporting  # noqa: E402
from pepplatform.task_compat import install_legacy_exports, legacy_keyword
from pepplatform.hlai_input import hla_allele, require_hlai_partner_sequence  # noqa: E402


AA_ORDER = list(_AA_ORDER)


def resolve_path(path: str) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else PROJECT_ROOT / value)


def load_hlai_pseudoseq_map(task_root: str):
    path = Path(task_root) / "pseudoseqs_decoded.csv"
    mapping = {}
    if not path.is_file():
        return mapping
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            hla = str(hla_allele(row)).strip()
            pseudo = str(row.get("pseudoseq", "")).strip().upper()
            status = str(row.get("status", "ok")).strip().lower()
            if hla and pseudo and status in {"", "ok"}:
                mapping[hla] = pseudo
    return mapping


def read_ppi_csv(path: str, hla_pseudo_map=None, *, pseudoseq_path=None, **legacy):
    hla_pseudo_map = legacy_keyword(legacy, "hla_pseudo_map", hla_pseudo_map)
    if legacy:
        raise TypeError(f"unexpected keywords: {sorted(legacy)}")
    rows = []
    source = Path(path)
    if not path or not source.is_file():
        return rows
    with source.open("r", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            sample_id = str(row.get("sample_id", "")).strip() or (f"{source.stem}_{index}")
            peptide = str(row.get("peptide", row.get("pep", ""))).strip().upper()
            raw_partner = str(
                row.get(
                    "protein",
                    hla_allele(row, default=row.get("pair_pseudo", row.get("hla_pair", ""))),
                )
            ).strip()
            partner = (
                hla_pseudo_map.get(raw_partner, raw_partner).upper()
                if hla_pseudo_map
                else raw_partner.upper()
            )
            mapped = bool(hla_pseudo_map and raw_partner in hla_pseudo_map)
            explicit = bool(
                (row.get("protein") and raw_partner == str(row["protein"]).strip())
                or (row.get("pair_pseudo") and raw_partner == str(row["pair_pseudo"]).strip())
            )
            if not mapped and not explicit and row.get("pair_pseudo"):
                partner = str(row["pair_pseudo"]).strip().upper()
                explicit = True
            if sample_id and peptide and partner:
                hla = str(hla_allele(row) or "").strip()
                if hla:
                    require_hlai_partner_sequence(
                        partner,
                        allele=hla,
                        resolved=mapped or explicit,
                        source_path=path,
                        row_number=index,
                        pseudoseq_path=pseudoseq_path,
                    )
                rows.append((sample_id, peptide, partner))
    return rows


def collect_role_sequences(rows, query_role="all"):
    mapping = {}
    for sample_id, peptide, partner in rows:
        for key, sequence, role in (
            (f"{sample_id}::pep", peptide, "peptide"),
            (f"{sample_id}::pro", partner, "partner"),
        ):
            if query_role != "all" and role != query_role:
                continue
            if key in mapping:
                raise ValueError(f"duplicate pair sample role ID in profile input: {key!r}")
            mapping[key] = sequence
    return mapping


def build_subject_fasta(rows, out_fasta: str):
    records = [
        (f"{sample_id}::{role}", sequence)
        for sample_id, peptide, partner in rows
        for role, sequence in (("pep", peptide), ("pro", partner))
    ]
    return write_canonical_subject_fasta(records, out_fasta)


def parse_ascii_pssm(path: str, sequence: str | None = None):
    if sequence is None:
        residues = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts and parts[0].isdigit() and len(parts) >= 2:
                residues.append(parts[1])
        sequence = "".join(residues)
    return torch.from_numpy(_strict_parse_ascii_pssm(path, sequence)).float()


def build_fallback(sequence: str, mode: str, as_psfm: bool = False):
    if as_psfm:
        raise ValueError("PSFM does not permit score-derived fallback")
    if mode not in {"blosum62", "blosum62_hybrid"}:
        raise ValueError("only the explicit pssm_blosum62_hybrid feature permits fallback")
    return torch.from_numpy(blosum62_score_profile(sequence)).float()


def pad_or_truncate(matrix: torch.Tensor, max_length: int):
    if matrix.shape[0] >= max_length:
        return matrix[:max_length]
    output = torch.zeros((max_length, matrix.shape[1]), dtype=matrix.dtype)
    output[: matrix.shape[0]] = matrix
    return output


def _selected_features(value: str) -> tuple[str, ...]:
    aliases = {
        "both": (FEATURE_PSSM, FEATURE_PSFM),
        "all": (FEATURE_PSSM, FEATURE_PSFM, FEATURE_TARGET_FREQUENCY),
    }
    selected: list[str] = []
    for token in str(value).split(","):
        token = token.strip()
        selected.extend(aliases.get(token, (token,)))
    allowed = {
        FEATURE_PSSM,
        FEATURE_PSSM_BLOSUM62_HYBRID,
        FEATURE_AC_PSSM_BLOSUM62_HYBRID,
        FEATURE_CC_PSSM_BLOSUM62_HYBRID,
        FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
        FEATURE_PSFM,
        FEATURE_PSFM_BLOSUM62_HYBRID,
        FEATURE_TARGET_FREQUENCY,
    }
    if not selected or set(selected).difference(allowed):
        raise ValueError(f"unsupported profile feature selection: {value!r}")
    return tuple(dict.fromkeys(selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task_root",
        default="datasets/task_peptide_protein_interaction",
    )
    parser.add_argument("--train_csv", default="")
    parser.add_argument("--val_csv", default="")
    parser.add_argument("--test_csv", default="")
    parser.add_argument("--db_scope", choices=["train_only"], default="train_only")
    parser.add_argument("--db_fasta", default="")
    parser.add_argument(
        "--feature",
        default="both",
        help="Feature name, legacy both/all alias, or comma-separated names",
    )
    parser.add_argument(
        "--query_role",
        choices=["all", "peptide", "partner"],
        default="all",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Top-n-gram safety limit; v3 matrices are stored at full length",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--evalue", type=float, default=0.001)
    parser.add_argument(
        "--inclusion-evalue",
        "--inclusion_evalue",
        dest="inclusion_evalue",
        type=float,
        default=0.001,
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--timeout-seconds",
        "--timeout_seconds",
        dest="timeout_seconds",
        type=int,
        default=600,
    )
    parser.add_argument(
        "--on_missing",
        choices=[TOP_N_GRAM_STRICT_PROFILE_POLICY, TOP_N_GRAM_FALLBACK_POLICY],
        default=TOP_N_GRAM_STRICT_PROFILE_POLICY,
        help=(
            "Applies only to target_frequency; PSSM/PSFM fallback uses an explicit hybrid feature"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_length < 1:
        raise SystemExit("max_length must be a positive integer")
    task_root = Path(resolve_path(args.task_root))
    train_csv = Path(
        resolve_path(args.train_csv) if args.train_csv else task_root / "processed" / "train.csv"
    )
    val_csv = Path(
        resolve_path(args.val_csv) if args.val_csv else task_root / "processed" / "val.csv"
    )
    test_csv = Path(
        resolve_path(args.test_csv) if args.test_csv else task_root / "processed" / "test_1440.csv"
    )
    hla_map = load_hlai_pseudoseq_map(str(task_root))
    pseudo_path = task_root / "pseudoseqs_decoded.csv"
    train_rows = read_ppi_csv(str(train_csv), hla_pseudo_map=hla_map, pseudoseq_path=pseudo_path)
    val_rows = read_ppi_csv(str(val_csv), hla_pseudo_map=hla_map, pseudoseq_path=pseudo_path)
    test_rows = read_ppi_csv(str(test_csv), hla_pseudo_map=hla_map, pseudoseq_path=pseudo_path)
    if not train_rows:
        raise SystemExit(f"No rows found in train CSV: {train_csv}")
    role_sequences = collect_role_sequences(
        train_rows + val_rows + test_rows,
        query_role=args.query_role,
    )
    features = _selected_features(args.feature)
    if FEATURE_TARGET_FREQUENCY in features:
        for role_id, sequence in role_sequences.items():
            normalize_top_ngram_sequence(
                sequence,
                max_length=args.max_length,
                context=f"Top-n-gram profile sequence {role_id!r}",
            )

    feature_directories = {}
    for feature in features:
        if feature == FEATURE_PSSM:
            feature_directories[feature] = task_root / "pssm_cache" / "ppi"
        elif feature == FEATURE_PSSM_BLOSUM62_HYBRID:
            feature_directories[feature] = task_root / "pssm_hybrid_cache" / "ppi"
        elif feature in {
            FEATURE_AC_PSSM_BLOSUM62_HYBRID,
            FEATURE_CC_PSSM_BLOSUM62_HYBRID,
            FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
        }:
            feature_directories[feature] = task_root / f"{feature}_cache" / "ppi"
        elif feature == FEATURE_PSFM:
            feature_directories[feature] = task_root / "psfm_cache" / "ppi"
        elif feature == FEATURE_PSFM_BLOSUM62_HYBRID:
            feature_directories[feature] = task_root / "psfm_hybrid_cache" / "ppi"
        else:
            frequency_root = task_root / "frequency_profile_cache" / "ppi"
            if args.query_role != "all":
                frequency_root /= args.query_role
            feature_directories[feature] = frequency_root

    config = ProfileGenerationConfig(
        iterations=args.iterations,
        evalue=args.evalue,
        inclusion_evalue=args.inclusion_evalue,
        threads=args.threads,
        timeout_seconds=args.timeout_seconds,
    )
    with tempfile.TemporaryDirectory(prefix="ppi_profile_subject_") as temporary:
        if args.db_fasta:
            subject_fasta = Path(resolve_path(args.db_fasta))
            if not subject_fasta.is_file():
                raise SystemExit(f"db_fasta not found: {subject_fasta}")
            sequence_subjects = index_subject_fasta(subject_fasta)
            database_scope = "external_frozen_fasta"
        else:
            subject_fasta = Path(temporary) / "train_subjects.fasta"
            sequence_subjects = build_subject_fasta(
                train_rows,
                str(subject_fasta),
            )
            database_scope = "train_only"
        indexes = build_profile_caches(
            samples=role_sequences.items(),
            feature_directories=feature_directories,
            subject_fasta=subject_fasta,
            sequence_subject_ids=sequence_subjects,
            config=config,
            database_scope=database_scope,
            database_provenance={
                "train_csv": str(train_csv.resolve()),
                "train_csv_sha256": sha256_file(train_csv),
                "query_role": args.query_role,
                "split_counts": {
                    "train": len(train_rows),
                    "val": len(val_rows),
                    "test": len(test_rows),
                },
                "external_fasta": bool(args.db_fasta),
            },
            target_frequency_fallback=(args.on_missing == TOP_N_GRAM_FALLBACK_POLICY),
            force=args.force,
        )
    for feature, index_path in indexes.items():
        print(f"{feature} index: {index_path}")


install_legacy_exports(globals())


if __name__ == "__main__":
    run_with_failure_reporting(main, default_stage="profile_generation")
