#!/usr/bin/env python3
"""Build leakage-audited sequence PSSM, PSFM, and target-frequency caches."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pepplatform.profile_generation import (  # noqa: E402
    ProfileGenerationConfig,
    build_profile_caches,
    build_psiblast_command,
    write_canonical_subject_fasta,
)
from pepplatform.sequence_profiles import (  # noqa: E402
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
    FEATURE_AC_PSSM_BLOSUM62_HYBRID,
    FEATURE_CC_PSSM_BLOSUM62_HYBRID,
    FEATURE_PSFM,
    FEATURE_PSFM_BLOSUM62_HYBRID,
    FEATURE_PSSM,
    FEATURE_PSSM_BLOSUM62_HYBRID,
    FEATURE_TARGET_FREQUENCY,
    normalize_profile_sequence,
    sha256_file,
)
from pepplatform.top_ngram import (  # noqa: E402
    TOP_N_GRAM_FALLBACK_POLICY,
    TOP_N_GRAM_STRICT_PROFILE_POLICY,
    normalize_top_ngram_sequence,
)
from pepplatform.training_failures import run_with_failure_reporting  # noqa: E402


def read_fasta(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    header = ""
    chunks: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header:
                rows.append((header, "".join(chunks).upper()))
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)
    if header:
        rows.append((header, "".join(chunks).upper()))
    return rows


def build_train_subject_fasta(
    train_fasta: Path,
    output_path: Path,
) -> dict[str, list[str]]:
    return write_canonical_subject_fasta(read_fasta(train_fasta), output_path)


def run_query(
    sequence: str,
    query_id: str,
    db_prefix: Path,
    work_dir: Path,
    excluded_subjects: list[str],
    iterations: int,
    evalue: float,
    threads: int,
) -> tuple[Path, Path, Path]:
    """Compatibility wrapper used by focused command-construction tests."""

    query_path = work_dir / f"{query_id}.fasta"
    pssm_path = work_dir / f"{query_id}.pssm"
    checkpoint_path = work_dir / f"{query_id}.checkpoint"
    hits_path = work_dir / f"{query_id}.hits.tsv"
    query_path.write_text(
        f">{query_id}\n{normalize_profile_sequence(sequence)}\n",
        encoding="ascii",
    )
    excluded_path = None
    if excluded_subjects:
        excluded_path = work_dir / f"{query_id}.exclude"
        excluded_path.write_text(
            "\n".join(excluded_subjects) + "\n",
            encoding="utf-8",
        )
    command = build_psiblast_command(
        query_fasta=query_path,
        database_prefix=db_prefix,
        ascii_pssm=pssm_path,
        checkpoint=checkpoint_path,
        hits=hits_path,
        config=ProfileGenerationConfig(
            iterations=iterations,
            evalue=evalue,
            inclusion_evalue=evalue,
            threads=threads,
        ),
        negative_seqidlist=excluded_path,
    )
    subprocess.run(command, check=True, capture_output=True, text=True)
    return pssm_path, checkpoint_path, hits_path


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Top-n-gram safety limit; v3 matrices are stored at full length",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--evalue", type=float, default=0.001)
    parser.add_argument("--inclusion-evalue", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument(
        "--feature",
        default="both",
        help="Feature name, legacy both/all alias, or comma-separated names",
    )
    parser.add_argument(
        "--on-missing",
        choices=[TOP_N_GRAM_STRICT_PROFILE_POLICY, TOP_N_GRAM_FALLBACK_POLICY],
        default=TOP_N_GRAM_STRICT_PROFILE_POLICY,
        help=(
            "Applies only to target_frequency; PSSM/PSFM fallback uses an explicit hybrid feature"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_length < 1:
        raise SystemExit("max-length must be a positive integer")
    data_dir = Path(args.data_dir).resolve()
    output_root = Path(args.out_root).resolve()
    train_fasta = data_dir / "train" / "seqs.fasta"
    if not train_fasta.is_file():
        raise SystemExit(f"missing training FASTA: {train_fasta}")

    records: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_fasta = data_dir / split / "seqs.fasta"
        split_rows = read_fasta(split_fasta) if split_fasta.is_file() else []
        split_counts[split] = len(split_rows)
        for sample_id, sequence in split_rows:
            previous = seen.get(sample_id)
            if previous is not None:
                raise SystemExit(f"duplicate FASTA sample ID across splits: {sample_id!r}")
            seen[sample_id] = sequence
            records.append((sample_id, sequence))

    features = _selected_features(args.feature)
    if FEATURE_TARGET_FREQUENCY in features:
        for sample_id, sequence in records:
            normalize_top_ngram_sequence(
                sequence,
                max_length=args.max_length,
                context=f"Top-n-gram profile sequence {sample_id!r}",
            )
    config = ProfileGenerationConfig(
        iterations=args.iterations,
        evalue=args.evalue,
        inclusion_evalue=args.inclusion_evalue,
        threads=args.threads,
        timeout_seconds=args.timeout_seconds,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sequence_profile_subject_") as temporary:
        subject_fasta = Path(temporary) / "train_subjects.fasta"
        sequence_subjects = build_train_subject_fasta(
            train_fasta,
            subject_fasta,
        )
        indexes = build_profile_caches(
            samples=records,
            feature_directories={feature: output_root / feature for feature in features},
            subject_fasta=subject_fasta,
            sequence_subject_ids=sequence_subjects,
            config=config,
            database_scope="train_only",
            database_provenance={
                "train_fasta": str(train_fasta.resolve()),
                "train_fasta_sha256": sha256_file(train_fasta),
                "split_counts": split_counts,
            },
            target_frequency_fallback=(args.on_missing == TOP_N_GRAM_FALLBACK_POLICY),
            force=args.force,
        )
    for feature, index_path in indexes.items():
        print(f"{feature} index: {index_path}")


if __name__ == "__main__":
    run_with_failure_reporting(main, default_stage="profile_generation")
