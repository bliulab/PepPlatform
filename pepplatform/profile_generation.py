"""Unified, reproducible PSI-BLAST profile generation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from pepplatform.profile_progress import ProfileProgressReporter
from pepplatform.sequence_profiles import (
    AA_ORDER,
    BLOSUM62_BACKGROUND_SHA256,
    BLOSUM62_MATRIX_SHA256,
    PROFILE_SCIENTIFIC_REFERENCES,
    PROFILE_CONTRACT_VERSION,
    PROFILE_MATRIX_KEY,
    PROFILE_HIT_SUMMARY_FIELDS,
    PROFILE_QUERY_COVERAGE_FORMULA,
    PSSM_HYBRID_SEMANTICS,
    PSFM_NAMING_REFERENCE,
    PSFM_HYBRID_SEMANTICS,
    PSFM_SEMANTICS,
    TARGET_FREQUENCY_HYBRID_SEMANTICS,
    TARGET_FREQUENCY_STRICT_SEMANTICS,
    FEATURE_PSFM,
    FEATURE_PSFM_BLOSUM62_HYBRID,
    FEATURE_PSSM,
    FEATURE_PSSM_BLOSUM62_HYBRID,
    FEATURE_TARGET_FREQUENCY,
    PSSM_HYBRID_FEATURE_TYPES,
    PSSM_SCORE_FEATURE_TYPES,
    PSFM_HYBRID_FEATURE_TYPES,
    PROFILE_FEATURE_TYPES,
    ProfileStore,
    SOURCE_BLOSUM62_FALLBACK,
    SOURCE_BLOSUM62_PSFM_FALLBACK,
    SOURCE_BLOSUM62_TARGET_FREQUENCY,
    SOURCE_PSI_BLAST_PSSM,
    SOURCE_PSI_BLAST_TARGET_FREQUENCY,
    SOURCE_PSI_BLAST_WEIGHTED_OBSERVED,
    SequenceProfileError,
    atomic_save_profile_matrix,
    atomic_write_json,
    blosum62_score_profile,
    blosum62_target_frequency_profile,
    new_profile_manifest,
    normalize_profile_sequence,
    normalize_psfm_sequence,
    parse_ascii_pssm,
    parse_psiblast_psfm,
    parse_psiblast_target_frequencies,
    profile_audit_summary,
    profile_sample_metadata,
    read_json_strict,
    sha256_file,
)


PROFILE_FALLBACK_NONE = "error"
PROFILE_FALLBACK_BLOSUM62_HYBRID = "blosum62_hybrid"
TARGET_FREQUENCY_FALLBACK_BLOSUM62 = "blosum62_target_frequency"
HIT_SUMMARY_FIELDS = PROFILE_HIT_SUMMARY_FIELDS
HIT_SUMMARY_OUTFMT = "7 " + " ".join(HIT_SUMMARY_FIELDS)


@dataclass(frozen=True)
class ProfileGenerationConfig:
    iterations: int = 3
    evalue: float = 0.001
    inclusion_evalue: float = 0.001
    threads: int = 1
    timeout_seconds: int = 600
    matrix: str = "BLOSUM62"
    gap_open: int = 11
    gap_extend: int = 1
    word_size: int = 3
    threshold: float = 11.0
    max_target_sequences: int = 500
    pseudocount: int = 0
    composition_based_stats: str = "1"
    seg: str = "no"
    soft_masking: bool = False
    save_after_last_round: bool = True

    def __post_init__(self):
        for name in (
            "iterations",
            "threads",
            "timeout_seconds",
            "gap_open",
            "gap_extend",
            "word_size",
            "max_target_sequences",
            "pseudocount",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise SequenceProfileError(
                    f"{name} must be an integer",
                    code="profile_parameter_invalid",
                )
        if self.iterations < 2:
            raise SequenceProfileError(
                "iterations must be >= 2 so PSI-BLAST can construct a PSSM",
                code="profile_iterations_too_low",
            )
        if self.threads < 1 or self.timeout_seconds < 1:
            raise SequenceProfileError(
                "threads and timeout_seconds must be positive",
                code="profile_parameter_invalid",
            )
        if self.word_size < 2 or self.max_target_sequences < 1:
            raise SequenceProfileError(
                "word_size and max_target_sequences are out of range",
                code="profile_parameter_invalid",
            )
        if self.gap_open < 0 or self.gap_extend < 0 or self.pseudocount < 0:
            raise SequenceProfileError(
                "gap penalties and pseudocount must be non-negative",
                code="profile_parameter_invalid",
            )
        if (
            not np.isfinite(self.evalue)
            or self.evalue <= 0.0
            or not np.isfinite(self.inclusion_evalue)
            or self.inclusion_evalue <= 0.0
            or not np.isfinite(self.threshold)
            or self.threshold < 0.0
        ):
            raise SequenceProfileError(
                "E-values must be positive and threshold must be non-negative",
                code="profile_parameter_invalid",
            )
        if self.matrix != "BLOSUM62":
            raise SequenceProfileError(
                f"the v{PROFILE_CONTRACT_VERSION} profile contract fixes matrix=BLOSUM62",
                code="profile_parameter_invalid",
            )
        if self.composition_based_stats != "1":
            raise SequenceProfileError(
                f"the v{PROFILE_CONTRACT_VERSION} profile contract fixes composition_based_stats=1",
                code="profile_parameter_invalid",
            )
        if self.seg != "no" or self.soft_masking:
            raise SequenceProfileError(
                f"the v{PROFILE_CONTRACT_VERSION} profile contract fixes SEG and soft masking off",
                code="profile_parameter_invalid",
            )
        if not self.save_after_last_round:
            raise SequenceProfileError(
                f"the v{PROFILE_CONTRACT_VERSION} profile contract saves the PSSM after the last round",
                code="profile_parameter_invalid",
            )

    def scientific_parameters(self) -> dict[str, object]:
        return {
            "iterations": self.iterations,
            "evalue": self.evalue,
            "inclusion_evalue": self.inclusion_evalue,
            "matrix": self.matrix,
            "gap_open": self.gap_open,
            "gap_extend": self.gap_extend,
            "word_size": self.word_size,
            "threshold": self.threshold,
            "max_target_sequences": self.max_target_sequences,
            "pseudocount": self.pseudocount,
            "composition_based_stats": self.composition_based_stats,
            "seg": self.seg,
            "soft_masking": self.soft_masking,
            "save_after_last_round": self.save_after_last_round,
        }

    def runtime_parameters(self) -> dict[str, object]:
        return {
            "threads": self.threads,
            "timeout_seconds": self.timeout_seconds,
        }


def require_blast_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SequenceProfileError(
            f"required NCBI BLAST+ binary is unavailable: {name}",
            code="profile_binary_missing",
        )
    return path


def blast_binary_version(name: str) -> str:
    executable = require_blast_binary(name)
    try:
        completed = subprocess.run(
            [executable, "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SequenceProfileError(
            f"cannot determine {name} version: {exc}",
            code="profile_binary_version_failed",
        ) from exc
    output = (completed.stdout or completed.stderr).strip()
    if not output:
        raise SequenceProfileError(
            f"{name} returned an empty version string",
            code="profile_binary_version_failed",
        )
    return output.splitlines()[0]


def psiblast_version() -> str:
    return blast_binary_version("psiblast")


def makeblastdb_version() -> str:
    return blast_binary_version("makeblastdb")


@dataclass(frozen=True)
class BlastRunResult:
    warnings: tuple[str, ...]
    stderr: str


def build_psiblast_command(
    *,
    query_fasta: os.PathLike[str] | str,
    database_prefix: os.PathLike[str] | str,
    ascii_pssm: os.PathLike[str] | str,
    checkpoint: os.PathLike[str] | str,
    hits: os.PathLike[str] | str,
    config: ProfileGenerationConfig,
    negative_seqidlist: os.PathLike[str] | str | None = None,
) -> list[str]:
    command = [
        require_blast_binary("psiblast"),
        "-query",
        os.fspath(query_fasta),
        "-db",
        os.fspath(database_prefix),
        "-num_iterations",
        str(config.iterations),
        "-evalue",
        str(config.evalue),
        "-inclusion_ethresh",
        str(config.inclusion_evalue),
        "-matrix",
        config.matrix,
        "-gapopen",
        str(config.gap_open),
        "-gapextend",
        str(config.gap_extend),
        "-word_size",
        str(config.word_size),
        "-threshold",
        str(config.threshold),
        "-max_target_seqs",
        str(config.max_target_sequences),
        "-pseudocount",
        str(config.pseudocount),
        "-comp_based_stats",
        config.composition_based_stats,
        "-seg",
        config.seg,
        "-soft_masking",
        "false",
        "-num_threads",
        str(config.threads),
        "-out_ascii_pssm",
        os.fspath(ascii_pssm),
        "-out_pssm",
        os.fspath(checkpoint),
        "-out",
        os.fspath(hits),
        "-outfmt",
        HIT_SUMMARY_OUTFMT,
        "-save_pssm_after_last_round",
    ]
    if negative_seqidlist is not None:
        command.extend(["-negative_seqidlist", os.fspath(negative_seqidlist)])
    return command


def make_blast_database(
    subject_fasta: os.PathLike[str] | str,
    database_prefix: os.PathLike[str] | str,
    *,
    timeout_seconds: int,
) -> None:
    command = [
        require_blast_binary("makeblastdb"),
        "-in",
        os.fspath(subject_fasta),
        "-dbtype",
        "prot",
        "-out",
        os.fspath(database_prefix),
        "-parse_seqids",
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise SequenceProfileError(
            f"makeblastdb timed out after {timeout_seconds}s",
            code="profile_database_timeout",
        ) from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise SequenceProfileError(
            f"makeblastdb failed: {stderr.strip() or exc}",
            code="profile_database_failed",
        ) from exc


def run_psiblast(
    command: list[str],
    *,
    timeout_seconds: int,
) -> BlastRunResult:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise SequenceProfileError(
            f"PSI-BLAST timed out after {timeout_seconds}s",
            code="profile_search_timeout",
        ) from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise SequenceProfileError(
            f"PSI-BLAST failed: {stderr.strip() or exc}",
            code="profile_search_failed",
        ) from exc
    stderr = (completed.stderr or "").strip()
    warnings = tuple(line.strip() for line in stderr.splitlines() if line.strip())
    return BlastRunResult(warnings=warnings, stderr=stderr)


def parse_hit_summary(
    path: os.PathLike[str] | str,
    *,
    expected_query_id: str | None = None,
    expected_query_length: int | None = None,
    excluded_subject_ids: Iterable[str] = (),
) -> dict[str, object]:
    try:
        lines = (
            Path(path)
            .read_text(
                encoding="utf-8",
                errors="strict",
            )
            .splitlines()
        )
    except (OSError, UnicodeError) as exc:
        raise SequenceProfileError(
            f"cannot read PSI-BLAST hit output {os.fspath(path)!r}: {exc}",
            code="profile_hits_unreadable",
        ) from exc
    iterations: dict[int, list[tuple[str, float, float, float]]] = {}
    current_iteration: int | None = None
    excluded = {str(value) for value in excluded_subject_ids}
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "Search has CONVERGED!":
            continue
        if stripped.startswith("#"):
            if stripped.startswith("# Iteration:"):
                try:
                    current_iteration = int(stripped.split(":", maxsplit=1)[1].strip())
                except ValueError as exc:
                    raise SequenceProfileError(
                        f"invalid PSI-BLAST iteration marker on line {line_number}",
                        code="profile_hits_invalid",
                    ) from exc
                if current_iteration < 1 or current_iteration in iterations:
                    raise SequenceProfileError(
                        f"invalid or duplicate PSI-BLAST iteration {current_iteration}",
                        code="profile_hits_invalid",
                    )
                iterations[current_iteration] = []
            continue
        if current_iteration is None:
            raise SequenceProfileError(
                "PSI-BLAST hit output is missing outfmt 7 iteration markers",
                code="profile_hits_iteration_missing",
            )
        fields = line.split("\t")
        if len(fields) != len(HIT_SUMMARY_FIELDS):
            raise SequenceProfileError(
                f"PSI-BLAST hit row {line_number} has {len(fields)} fields; "
                f"expected {len(HIT_SUMMARY_FIELDS)}",
                code="profile_hits_invalid",
            )
        query_id = fields[0].strip()
        subject_id = fields[1].strip()
        if not query_id or not subject_id:
            raise SequenceProfileError(
                f"PSI-BLAST hit row {line_number} has an empty sequence ID",
                code="profile_hits_invalid",
            )
        if expected_query_id is not None and query_id != expected_query_id:
            raise SequenceProfileError(
                f"PSI-BLAST hit row {line_number} has query ID "
                f"{query_id!r}; expected {expected_query_id!r}",
                code="profile_hits_query_mismatch",
            )
        if subject_id in excluded:
            raise SequenceProfileError(
                f"PSI-BLAST returned excluded subject ID {subject_id!r}",
                code="profile_self_exclusion_failed",
            )
        try:
            identity = float(fields[2])
            query_start = int(fields[3])
            query_end = int(fields[4])
            query_length = int(fields[5])
            evalue = float(fields[6])
            bit_score = float(fields[7])
        except (TypeError, ValueError) as exc:
            raise SequenceProfileError(
                f"PSI-BLAST hit row {line_number} contains a non-numeric field",
                code="profile_hits_invalid",
            ) from exc
        values = np.asarray([identity, evalue, bit_score], dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or not 0.0 <= identity <= 100.0
            or query_length < 1
            or (expected_query_length is not None and query_length != int(expected_query_length))
            or not 1 <= query_start <= query_end <= query_length
            or evalue < 0.0
        ):
            raise SequenceProfileError(
                f"PSI-BLAST hit row {line_number} contains invalid values",
                code="profile_hits_invalid",
            )
        coverage = (query_end - query_start + 1) / query_length
        iterations[current_iteration].append((subject_id, identity, coverage, evalue))
    if not iterations:
        raise SequenceProfileError(
            "PSI-BLAST hit output contains no completed iteration markers",
            code="profile_hits_iteration_missing",
        )
    final_iteration = max(iterations)
    final_rows = iterations[final_iteration]
    subjects: dict[str, tuple[float, float, float]] = {}
    for subject_id, identity, coverage, evalue in final_rows:
        previous = subjects.get(subject_id)
        if previous is None:
            subjects[subject_id] = (identity, coverage, evalue)
        else:
            subjects[subject_id] = (
                max(previous[0], identity),
                max(previous[1], coverage),
                min(previous[2], evalue),
            )
    hits = list(subjects.values())
    return {
        "nonself_hit_count": len(hits),
        "max_query_coverage": max((row[1] for row in hits), default=0.0),
        "max_identity": max((row[0] for row in hits), default=0.0),
        "min_evalue": min((row[2] for row in hits), default=None),
        "zero_nonself_hits": not hits,
        "final_iteration": final_iteration,
        "final_iteration_hit_rows": len(final_rows),
    }


def index_subject_fasta(
    path: os.PathLike[str] | str,
) -> dict[str, list[str]]:
    """Map normalized subject sequences to every parseable FASTA identifier."""

    try:
        lines = (
            Path(path)
            .read_text(
                encoding="utf-8",
                errors="strict",
            )
            .splitlines()
        )
    except (OSError, UnicodeError) as exc:
        raise SequenceProfileError(
            f"cannot read profile subject FASTA {os.fspath(path)!r}: {exc}",
            code="profile_database_input_unreadable",
        ) from exc
    mapping: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    current_id = ""
    chunks: list[str] = []

    def commit() -> None:
        if not current_id:
            return
        if not chunks:
            raise SequenceProfileError(
                f"profile subject {current_id!r} has an empty sequence",
                code="profile_database_input_invalid",
            )
        sequence = normalize_profile_sequence("".join(chunks))
        mapping.setdefault(sequence, []).append(current_id)

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            commit()
            identifier = line[1:].split(maxsplit=1)[0] if line[1:] else ""
            if not identifier:
                raise SequenceProfileError(
                    f"profile subject FASTA line {line_number} has no ID",
                    code="profile_database_input_invalid",
                )
            if identifier in seen_ids:
                raise SequenceProfileError(
                    f"duplicate profile subject ID {identifier!r}",
                    code="profile_database_input_invalid",
                )
            seen_ids.add(identifier)
            current_id = identifier
            chunks = []
        else:
            if not current_id:
                raise SequenceProfileError(
                    "profile subject FASTA contains sequence data before a header",
                    code="profile_database_input_invalid",
                )
            chunks.append(line)
    commit()
    if not mapping:
        raise SequenceProfileError(
            "profile subject FASTA contains no sequences",
            code="profile_database_input_invalid",
        )
    return mapping


def write_canonical_subject_fasta(
    records: Iterable[tuple[object, object]],
    output_path: os.PathLike[str] | str,
) -> dict[str, list[str]]:
    """Write the single canonical train-subject representation used by all tools."""

    lines: list[str] = []
    sequence_subject_ids: dict[str, list[str]] = {}
    count = 0
    for index, (_raw_identifier, raw_sequence) in enumerate(records):
        sequence = normalize_profile_sequence(raw_sequence)
        subject_id = f"TRAIN_{index:06d}"
        lines.extend((f">{subject_id}", sequence))
        sequence_subject_ids.setdefault(sequence, []).append(subject_id)
        count += 1
    if count == 0:
        raise SequenceProfileError(
            "profile subject FASTA requires at least one record",
            code="profile_database_input_invalid",
        )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="ascii") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return sequence_subject_ids


def _normalize_samples(
    samples: Iterable[tuple[object, object]],
) -> dict[str, tuple[str, str]]:
    normalized: dict[str, tuple[str, str]] = {}
    for raw_identifier, raw_sequence in samples:
        identifier = str(raw_identifier or "")
        if not identifier:
            raise SequenceProfileError(
                "profile sample IDs must be non-empty",
                code="profile_sample_id_missing",
            )
        input_sequence = "".join(str(raw_sequence or "").split()).upper()
        profile_sequence = normalize_profile_sequence(raw_sequence)
        previous = normalized.get(identifier)
        if previous is not None and previous != (input_sequence, profile_sequence):
            raise SequenceProfileError(
                f"duplicate sample ID {identifier!r} refers to different sequences",
                code="profile_sample_id_collision",
            )
        normalized[identifier] = (input_sequence, profile_sequence)
    if not normalized:
        raise SequenceProfileError(
            "profile generation requires at least one sample",
            code="profile_samples_empty",
        )
    return normalized


def _generation_fingerprint(provenance: Mapping[str, object]) -> str:
    serialized = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _sample_filename(identifier: str, sequence: str) -> str:
    digest = hashlib.sha256(
        identifier.encode("utf-8") + b"\0" + hashlib.sha256(sequence.encode("ascii")).digest()
    ).hexdigest()
    return f"{digest[:32]}.npz"


def _content_addressed_matrix_path(
    directory: Path,
    matrix: np.ndarray,
) -> tuple[Path, str]:
    contiguous = np.ascontiguousarray(matrix)
    content_digest = hashlib.sha256()
    content_digest.update(str(contiguous.dtype).encode("ascii"))
    content_digest.update(b"\0")
    content_digest.update(",".join(str(value) for value in contiguous.shape).encode("ascii"))
    content_digest.update(b"\0")
    content_digest.update(contiguous.tobytes(order="C"))
    relative_path = Path("matrices") / f"{content_digest.hexdigest()}.npz"
    target = directory / relative_path
    if target.is_file():
        try:
            with np.load(target, allow_pickle=False) as payload:
                if payload.files != [PROFILE_MATRIX_KEY]:
                    raise SequenceProfileError(
                        f"content-addressed profile matrix is malformed: {target}",
                        code="profile_matrix_archive_invalid",
                    )
                stored = np.asarray(payload[PROFILE_MATRIX_KEY])
        except SequenceProfileError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise SequenceProfileError(
                f"cannot verify content-addressed profile matrix {target}: {exc}",
                code="profile_matrix_unreadable",
            ) from exc
        if (
            stored.dtype != contiguous.dtype
            or stored.shape != contiguous.shape
            or not np.array_equal(stored, contiguous)
        ):
            raise SequenceProfileError(
                f"content-addressed profile matrix collision at {target}",
                code="profile_matrix_content_collision",
            )
        return relative_path, sha256_file(target)
    return relative_path, atomic_save_profile_matrix(target, contiguous)


def _persist_subject_fasta(
    source: Path,
    directory: Path,
    digest: str,
) -> Path:
    relative = Path("subjects") / f"{digest}.fasta"
    target = directory / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if sha256_file(target) != digest:
            raise SequenceProfileError(
                f"cached subject FASTA digest mismatch: {target}",
                code="profile_database_digest_mismatch",
            )
        return relative
    atomic_copy_file(source, target, expected_sha256=digest)
    return relative


def atomic_copy_file(
    source: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Atomically copy an artifact and optionally pin its expected digest."""

    source_path = Path(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with (
            os.fdopen(file_descriptor, "wb") as output,
            open(
                source_path,
                "rb",
            ) as input_file,
        ):
            shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        digest = sha256_file(temporary_name)
        if expected_sha256 is not None and digest != expected_sha256:
            raise SequenceProfileError(
                f"artifact digest changed while copying {source_path}",
                code="profile_database_digest_mismatch",
            )
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return digest


def _existing_sample(
    *,
    index_path: Path,
    feature_type: str,
    sample_id: str,
    sequence: str,
    generation_fingerprint: str,
) -> Mapping[str, object] | None:
    if not index_path.is_file():
        return None
    try:
        store = ProfileStore(index_path, feature_type=feature_type)
        if (
            store.manifest.get("provenance", {}).get("generation_fingerprint")
            != generation_fingerprint
        ):
            return None
        store.load(sample_id, sequence)
        return store.samples[sample_id]
    except (OSError, SequenceProfileError):
        return None


def build_profile_caches(
    *,
    samples: Iterable[tuple[object, object]],
    feature_directories: Mapping[str, os.PathLike[str] | str],
    subject_fasta: os.PathLike[str] | str,
    sequence_subject_ids: Mapping[str, Iterable[str]],
    config: ProfileGenerationConfig,
    database_scope: str,
    database_provenance: Mapping[str, object],
    target_frequency_fallback: bool = False,
    force: bool = False,
    lock_path: os.PathLike[str] | str | None = None,
) -> dict[str, Path]:
    requested = tuple(dict.fromkeys(str(value) for value in feature_directories))
    if not requested:
        raise SequenceProfileError(
            "profile generation requires at least one feature type",
            code="profile_feature_type_invalid",
        )
    invalid_features = set(requested).difference(PROFILE_FEATURE_TYPES)
    if invalid_features:
        raise SequenceProfileError(
            f"unsupported profile feature types: {sorted(invalid_features)}",
            code="profile_feature_type_invalid",
        )
    normalized_samples = _normalize_samples(samples)
    if FEATURE_PSFM in requested:
        for sample_id, (input_sequence, _profile_sequence) in normalized_samples.items():
            try:
                normalize_psfm_sequence(input_sequence)
            except SequenceProfileError as exc:
                raise SequenceProfileError(
                    f"PSFM sample_id={sample_id!r} is invalid: {exc}",
                    code=exc.code,
                ) from exc
    subject_path = Path(subject_fasta)
    if not subject_path.is_file():
        raise SequenceProfileError(
            f"profile subject FASTA is missing: {subject_path}",
            code="profile_database_input_missing",
        )
    indexed_subjects = index_subject_fasta(subject_path)
    supplied_subjects: dict[str, set[str]] = {}
    for raw_sequence, raw_identifiers in sequence_subject_ids.items():
        sequence = normalize_profile_sequence(raw_sequence)
        supplied_subjects.setdefault(sequence, set()).update(
            str(identifier) for identifier in raw_identifiers if str(identifier)
        )
    expected_subjects = {
        sequence: set(identifiers) for sequence, identifiers in indexed_subjects.items()
    }
    if supplied_subjects != expected_subjects:
        raise SequenceProfileError(
            "sequence_subject_ids does not exactly match the subject FASTA; "
            "complete self-exclusion cannot be guaranteed",
            code="profile_self_exclusion_index_mismatch",
        )
    blast_version = psiblast_version()
    database_binary_version = makeblastdb_version()
    subject_fasta_digest = sha256_file(subject_path)
    subject_artifact = Path("subjects") / f"{subject_fasta_digest}.fasta"
    base_provenance: dict[str, object] = {
        "profile_contract_version": PROFILE_CONTRACT_VERSION,
        "algorithm": "NCBI PSI-BLAST",
        "algorithm_lineage": "Altschul_et_al_1997",
        "implementation": "NCBI BLAST+",
        "blast_version": blast_version,
        "makeblastdb_version": database_binary_version,
        "scientific_parameters": config.scientific_parameters(),
        "runtime_parameters": config.runtime_parameters(),
        "database_scope": str(database_scope),
        "subject_fasta_sha256": subject_fasta_digest,
        "subject_fasta_artifact": subject_artifact.as_posix(),
        "database": dict(database_provenance),
        "self_exclusion": "all subject IDs with an identical normalized query sequence",
        "blosum62_matrix_sha256": BLOSUM62_MATRIX_SHA256,
        "blosum62_background_sha256": BLOSUM62_BACKGROUND_SHA256,
        "scientific_references": [dict(reference) for reference in PROFILE_SCIENTIFIC_REFERENCES],
        "quality_summary": {
            "outfmt_fields": list(HIT_SUMMARY_FIELDS),
            "query_coverage": PROFILE_QUERY_COVERAGE_FORMULA,
            "iteration_scope": "final_completed_iteration",
            "hit_count_unit": "unique_subject",
        },
    }
    fingerprint = _generation_fingerprint(base_provenance)
    base_provenance["generation_fingerprint"] = fingerprint

    feature_paths = {
        feature_type: Path(feature_directories[feature_type]) for feature_type in requested
    }
    for directory in feature_paths.values():
        (directory / "matrices").mkdir(parents=True, exist_ok=True)
        _persist_subject_fasta(subject_path, directory, subject_fasta_digest)
    if lock_path is None:
        common_parent = Path(
            os.path.commonpath([str(path.resolve()) for path in feature_paths.values()])
        )
        lock_file = common_parent / ".profile-cache.lock"
    else:
        lock_file = Path(lock_path)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    manifest_samples: dict[str, dict[str, Mapping[str, object]]] = {
        feature_type: {} for feature_type in requested
    }
    audit_rows: list[dict[str, object]] = []
    index_paths = {
        feature_type: directory / "index.json" for feature_type, directory in feature_paths.items()
    }
    progress_feature = "+".join(requested)

    with open(lock_file, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(prefix="pep_profile_build_") as temporary:
            temporary_root = Path(temporary)
            database_prefix = temporary_root / "profile_db"
            database_progress = ProfileProgressReporter(
                1,
                progress_feature,
                stage="database",
            )
            database_progress.start()
            make_blast_database(
                subject_path,
                database_prefix,
                timeout_seconds=config.timeout_seconds,
            )
            database_progress.complete()
            profile_progress = ProfileProgressReporter(
                len(normalized_samples),
                progress_feature,
            )
            profile_progress.start()
            processed_samples = 0
            for sample_id, (input_sequence, sequence) in normalized_samples.items():
                current: dict[str, Mapping[str, object]] = {}
                if not force:
                    for feature_type, index_path in index_paths.items():
                        existing = _existing_sample(
                            index_path=index_path,
                            feature_type=feature_type,
                            sample_id=sample_id,
                            sequence=input_sequence,
                            generation_fingerprint=fingerprint,
                        )
                        if existing is not None:
                            current[feature_type] = existing
                if len(current) == len(requested):
                    for feature_type, metadata in current.items():
                        manifest_samples[feature_type][sample_id] = metadata
                    representative = next(iter(current.values()))
                    audit_rows.append(
                        {
                            "sample_id": sample_id,
                            "sequence_length": len(sequence),
                            "excluded_subject_count": len(indexed_subjects.get(sequence, ())),
                            **dict(representative.get("quality", {})),
                            "zero_observation_positions": list(
                                representative.get(
                                    "zero_observation_positions",
                                    [],
                                )
                            ),
                            "cache_reused": True,
                            "cache_reused_by_feature": {
                                feature_type: True for feature_type in requested
                            },
                        }
                    )
                    processed_samples += 1
                    profile_progress.update(processed_samples)
                    continue

                sample_digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20]
                query_fasta = temporary_root / f"{sample_digest}.fa"
                ascii_pssm = temporary_root / f"{sample_digest}.pssm"
                checkpoint = temporary_root / f"{sample_digest}.asn1"
                hits = temporary_root / f"{sample_digest}.hits.tsv"
                query_fasta.write_text(
                    f">query_{sample_digest}\n{sequence}\n",
                    encoding="ascii",
                )
                excluded = sorted(
                    {
                        str(identifier)
                        for identifier in indexed_subjects.get(sequence, ())
                        if str(identifier)
                    }
                )
                negative_seqidlist = None
                if excluded:
                    negative_seqidlist = temporary_root / f"{sample_digest}.exclude"
                    negative_seqidlist.write_text(
                        "\n".join(excluded) + "\n",
                        encoding="utf-8",
                    )
                command = build_psiblast_command(
                    query_fasta=query_fasta,
                    database_prefix=database_prefix,
                    ascii_pssm=ascii_pssm,
                    checkpoint=checkpoint,
                    hits=hits,
                    config=config,
                    negative_seqidlist=negative_seqidlist,
                )
                run_result = run_psiblast(
                    command,
                    timeout_seconds=config.timeout_seconds,
                )
                warnings = (
                    list(run_result.warnings) if isinstance(run_result, BlastRunResult) else []
                )
                query_id = f"query_{sample_digest}"
                quality = parse_hit_summary(
                    hits,
                    expected_query_id=query_id,
                    expected_query_length=len(sequence),
                    excluded_subject_ids=excluded,
                )
                quality["blast_warnings"] = warnings

                parsed_pssm: np.ndarray | None = None
                pssm_parse_error: SequenceProfileError | None = None
                if ascii_pssm.is_file():
                    try:
                        parsed_pssm = parse_ascii_pssm(ascii_pssm, sequence)
                    except SequenceProfileError as exc:
                        pssm_parse_error = exc
                zero_hits = bool(quality["zero_nonself_hits"])
                profile_context = (
                    f"sample_id={sample_id!r}, sequence_length={len(sequence)}, "
                    f"nonself_hit_count={quality['nonself_hit_count']}, "
                    f"zero_nonself_hits={zero_hits}"
                )

                for feature_type in requested:
                    if feature_type in current:
                        manifest_samples[feature_type][sample_id] = current[feature_type]
                        continue
                    fallback = False
                    fallback_reason = ""
                    zero_positions: tuple[int, ...] = ()
                    if feature_type == FEATURE_PSSM:
                        if pssm_parse_error is not None:
                            raise pssm_parse_error
                        if parsed_pssm is None:
                            if zero_hits:
                                raise SequenceProfileError(
                                    "strict PSSM cannot represent this sample because "
                                    "PSI-BLAST found no usable non-self homologs and "
                                    f"did not produce an ASCII PSSM ({profile_context})",
                                    code="profile_no_nonself_hits",
                                    details={
                                        "feature": FEATURE_PSSM,
                                        "sample_id": sample_id,
                                        "sequence_length": len(sequence),
                                        "nonself_hit_count": quality["nonself_hit_count"],
                                        "zero_nonself_hits": True,
                                    },
                                )
                            raise SequenceProfileError(
                                "strict PSSM cannot represent this sample because "
                                "PSI-BLAST completed without producing an ASCII "
                                f"PSSM ({profile_context}); use "
                                "pssm_blosum62_hybrid when an explicit BLOSUM62 "
                                "score fallback is scientifically acceptable",
                                code="profile_output_missing",
                            )
                        matrix = parsed_pssm
                        source = SOURCE_PSI_BLAST_PSSM
                    elif feature_type in PSSM_HYBRID_FEATURE_TYPES:
                        if parsed_pssm is not None:
                            matrix = parsed_pssm
                            source = SOURCE_PSI_BLAST_PSSM
                        else:
                            matrix = blosum62_score_profile(sequence)
                            source = SOURCE_BLOSUM62_FALLBACK
                            fallback = True
                            fallback_reason = (
                                pssm_parse_error.code
                                if pssm_parse_error is not None
                                else "profile_output_missing"
                            )
                    elif feature_type == FEATURE_PSFM:
                        if not checkpoint.is_file() or not ascii_pssm.is_file():
                            if zero_hits:
                                raise SequenceProfileError(
                                    "strict PSFM cannot represent this sample because "
                                    "PSI-BLAST found no usable non-self homologs and "
                                    "did not produce both required PSFM outputs "
                                    f"({profile_context})",
                                    code="profile_no_nonself_hits",
                                    details={
                                        "feature": FEATURE_PSFM,
                                        "sample_id": sample_id,
                                        "sequence_length": len(sequence),
                                        "nonself_hit_count": quality["nonself_hit_count"],
                                        "zero_nonself_hits": True,
                                    },
                                )
                            raise SequenceProfileError(
                                "PSI-BLAST completed without producing both the "
                                "ASN.1 checkpoint and ASCII PSSM "
                                f"for PSFM ({profile_context})",
                                code="profile_output_missing",
                            )
                        matrix, zero_positions = parse_psiblast_psfm(
                            checkpoint,
                            expected_sequence=sequence,
                            ascii_pssm=ascii_pssm,
                        )
                        source = SOURCE_PSI_BLAST_WEIGHTED_OBSERVED
                    elif feature_type in PSFM_HYBRID_FEATURE_TYPES:
                        if checkpoint.is_file() and ascii_pssm.is_file():
                            try:
                                matrix, zero_positions = parse_psiblast_psfm(
                                    checkpoint,
                                    expected_sequence=sequence,
                                    ascii_pssm=ascii_pssm,
                                )
                                source = SOURCE_PSI_BLAST_WEIGHTED_OBSERVED
                            except SequenceProfileError as exc:
                                matrix = blosum62_target_frequency_profile(input_sequence)
                                source = SOURCE_BLOSUM62_PSFM_FALLBACK
                                fallback = True
                                fallback_reason = exc.code
                                zero_positions = ()
                        else:
                            matrix = blosum62_target_frequency_profile(sequence)
                            source = SOURCE_BLOSUM62_PSFM_FALLBACK
                            fallback = True
                            fallback_reason = "profile_output_missing"
                    else:
                        if checkpoint.is_file():
                            try:
                                matrix = parse_psiblast_target_frequencies(
                                    checkpoint,
                                    expected_length=len(sequence),
                                    expected_sequence=sequence,
                                )
                                source = SOURCE_PSI_BLAST_TARGET_FREQUENCY
                            except SequenceProfileError as exc:
                                if target_frequency_fallback:
                                    matrix = blosum62_target_frequency_profile(sequence)
                                    source = SOURCE_BLOSUM62_TARGET_FREQUENCY
                                    fallback = True
                                    fallback_reason = exc.code
                                else:
                                    raise
                        elif target_frequency_fallback:
                            matrix = blosum62_target_frequency_profile(sequence)
                            source = SOURCE_BLOSUM62_TARGET_FREQUENCY
                            fallback = True
                            fallback_reason = "profile_output_missing"
                        else:
                            raise SequenceProfileError(
                                "PSI-BLAST completed without producing a checkpoint "
                                f"for target_frequency ({profile_context})",
                                code="profile_output_missing",
                            )

                    if feature_type in PSSM_SCORE_FEATURE_TYPES:
                        matrix = np.asarray(matrix, dtype=np.int32)
                    else:
                        matrix = np.asarray(matrix, dtype=np.float32)
                    relative_path, matrix_sha = _content_addressed_matrix_path(
                        feature_paths[feature_type],
                        matrix,
                    )
                    manifest_samples[feature_type][sample_id] = profile_sample_metadata(
                        path=relative_path,
                        matrix=matrix,
                        sequence=input_sequence,
                        matrix_sha256=matrix_sha,
                        profile_source=source,
                        fallback=fallback,
                        fallback_reason=fallback_reason,
                        zero_observation_positions=zero_positions,
                        quality=quality,
                    )
                audit_rows.append(
                    {
                        "sample_id": sample_id,
                        "sequence_length": len(sequence),
                        "excluded_subject_count": len(excluded),
                        **quality,
                        "zero_observation_positions": list(zero_positions),
                        "cache_reused": False,
                        "cache_reused_by_feature": {
                            feature_type: feature_type in current for feature_type in requested
                        },
                        "command": command,
                    }
                )
                processed_samples += 1
                profile_progress.update(processed_samples)

        created_at = datetime.now(timezone.utc).isoformat()
        for feature_type in requested:
            feature_audit_rows: list[dict[str, object]] = []
            for row in audit_rows:
                sample_id = str(row["sample_id"])
                metadata = manifest_samples[feature_type][sample_id]
                feature_row = dict(row)
                reused_by_feature = feature_row.pop(
                    "cache_reused_by_feature",
                    {},
                )
                feature_row["cache_reused"] = bool(
                    reused_by_feature.get(
                        feature_type,
                        feature_row.get("cache_reused", False),
                    )
                )
                feature_audit_rows.append(
                    {
                        **feature_row,
                        "zero_observation_positions": list(
                            metadata.get("zero_observation_positions", [])
                        ),
                        "profile_source": metadata.get("profile_source"),
                        "fallback": metadata.get("fallback"),
                        "fallback_reason": metadata.get("fallback_reason"),
                    }
                )
            audit_payload = {
                "feature_type": feature_type,
                "generation_fingerprint": fingerprint,
                "summary": profile_audit_summary(manifest_samples[feature_type]),
                "samples": feature_audit_rows,
            }
            audit_key = _generation_fingerprint(audit_payload)
            audit_relative_path = Path("audits") / f"{audit_key}.json"
            audit_path = feature_paths[feature_type] / audit_relative_path
            if audit_path.is_file():
                if read_json_strict(audit_path) != audit_payload:
                    raise SequenceProfileError(
                        f"content-addressed audit collision at {audit_path}",
                        code="profile_audit_content_collision",
                    )
            else:
                atomic_write_json(audit_path, audit_payload)
            provenance = {
                **base_provenance,
                "feature_type": feature_type,
                "created_at": created_at,
                "audit_artifact": audit_relative_path.as_posix(),
                "audit_sha256": sha256_file(audit_path),
                "audit_summary": dict(audit_payload["summary"]),
                "sequence_normalization": (
                    {
                        "uppercase": True,
                        "strip_whitespace": True,
                        "alphabet": AA_ORDER,
                        "ambiguous_residues": "error",
                        "non_letters": "error",
                    }
                    if feature_type == FEATURE_PSFM
                    else {
                        "uppercase": True,
                        "strip_whitespace": True,
                        "J_U_O_to_X": True,
                        "B_Z_X_preserved": True,
                        "non_letters": "error",
                    }
                ),
                "profile_semantics": (
                    {
                        **PSFM_SEMANTICS,
                        "aa_order": AA_ORDER,
                        "scientific_reference": dict(PSFM_NAMING_REFERENCE),
                    }
                    if feature_type == FEATURE_PSFM
                    else {
                        **(
                            {
                                **PSFM_HYBRID_SEMANTICS,
                                "aa_order": AA_ORDER,
                                "scientific_reference": dict(PSFM_NAMING_REFERENCE),
                            }
                            if feature_type in PSFM_HYBRID_FEATURE_TYPES
                            else PSSM_HYBRID_SEMANTICS[feature_type]
                            if feature_type in PSSM_HYBRID_FEATURE_TYPES
                            else TARGET_FREQUENCY_HYBRID_SEMANTICS
                            if (
                                feature_type == FEATURE_TARGET_FREQUENCY
                                and target_frequency_fallback
                            )
                            else TARGET_FREQUENCY_STRICT_SEMANTICS
                            if feature_type == FEATURE_TARGET_FREQUENCY
                            else {
                                "method_name": feature_type,
                                "matrix_layout": ("query_position_by_amino_acid"),
                            }
                        ),
                    }
                ),
                "fallback_policy": (
                    PROFILE_FALLBACK_BLOSUM62_HYBRID
                    if feature_type in (PSSM_HYBRID_FEATURE_TYPES | PSFM_HYBRID_FEATURE_TYPES)
                    else TARGET_FREQUENCY_FALLBACK_BLOSUM62
                    if (feature_type == FEATURE_TARGET_FREQUENCY and target_frequency_fallback)
                    else PROFILE_FALLBACK_NONE
                ),
            }
            manifest = new_profile_manifest(
                feature_type=feature_type,
                samples=manifest_samples[feature_type],
                provenance=provenance,
            )
            atomic_write_json(index_paths[feature_type], manifest)
            atomic_write_json(
                feature_paths[feature_type] / "profile_audit.json",
                audit_payload,
            )
        profile_progress.complete()
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return index_paths
