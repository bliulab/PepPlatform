import argparse
import sys
import tempfile
from pathlib import Path


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
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
    FEATURE_AC_PSSM_BLOSUM62_HYBRID,
    FEATURE_CC_PSSM_BLOSUM62_HYBRID,
    FEATURE_PSSM,
    FEATURE_PSSM_BLOSUM62_HYBRID,
    normalize_profile_sequence,
    sha256_file,
)


def resolve_path(path: str) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else PROJECT_ROOT / value)


def read_fasta(path: str):
    sequences = []
    header = None
    chunks = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences.append((header, "".join(chunks).upper()))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            sequences.append((header, "".join(chunks).upper()))
    return sequences


def build_subject_fasta(train_fa: str, out_fa: str):
    """Write the canonical duplicate-preserving training-only subject FASTA."""

    return write_canonical_subject_fasta(read_fasta(train_fa), out_fa)


def main():
    parser = argparse.ArgumentParser(description="Build a strict v3 PSI-BLAST PSSM cache")
    parser.add_argument(
        "--data_dir",
        default="datasets/task_peptide_function/data",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--out_dir",
        default="datasets/task_peptide_function/pssm_cache/train",
    )
    parser.add_argument(
        "--db_fasta",
        default="",
        help="Optional frozen external subject FASTA; default is train-only",
    )
    parser.add_argument(
        "--feature",
        choices=[
            FEATURE_PSSM,
            FEATURE_PSSM_BLOSUM62_HYBRID,
            FEATURE_AC_PSSM_BLOSUM62_HYBRID,
            FEATURE_CC_PSSM_BLOSUM62_HYBRID,
            FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
        ],
        default=FEATURE_PSSM,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=60,
        help="Deprecated compatibility option; v3 stores full-length matrices",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--evalue", type=float, default=0.001)
    parser.add_argument("--inclusion_evalue", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_length < 1:
        raise SystemExit("max_length must be a positive integer")
    data_dir = Path(resolve_path(args.data_dir))
    output_directory = Path(resolve_path(args.out_dir))
    split_fasta = data_dir / args.split / "seqs.fasta"
    train_fasta = data_dir / "train" / "seqs.fasta"
    if not split_fasta.is_file():
        raise SystemExit(f"Missing split FASTA: {split_fasta}")
    if not train_fasta.is_file():
        raise SystemExit(f"Missing train FASTA for database: {train_fasta}")

    config = ProfileGenerationConfig(
        iterations=args.iterations,
        evalue=args.evalue,
        inclusion_evalue=args.inclusion_evalue,
        threads=args.threads,
        timeout_seconds=args.timeout_seconds,
    )
    with tempfile.TemporaryDirectory(prefix="pssm_subject_") as temporary:
        if args.db_fasta:
            subject_fasta = Path(resolve_path(args.db_fasta))
            if not subject_fasta.is_file():
                raise SystemExit(f"db_fasta not found: {subject_fasta}")
            sequence_subject_ids = index_subject_fasta(subject_fasta)
            database_scope = "external_frozen_fasta"
        else:
            subject_fasta = Path(temporary) / "subject.fasta"
            sequence_subject_ids = build_subject_fasta(
                str(train_fasta),
                str(subject_fasta),
            )
            database_scope = "train_only"

        indexes = build_profile_caches(
            samples=read_fasta(str(split_fasta)),
            feature_directories={args.feature: output_directory},
            subject_fasta=subject_fasta,
            sequence_subject_ids=sequence_subject_ids,
            config=config,
            database_scope=database_scope,
            database_provenance={
                "train_fasta": str(train_fasta.resolve()),
                "train_fasta_sha256": sha256_file(train_fasta),
                "split": args.split,
                "external_fasta": bool(args.db_fasta),
            },
            force=args.force,
        )
    print(f"{args.feature} index: {indexes[args.feature]}")


if __name__ == "__main__":
    main()
