"""Scientific contracts and storage for BLOSUM62 and PSI-BLAST profiles.

The module deliberately separates three related representations:

* BLOSUM62: context-free half-bit substitution score rows.
* PSSM: position-specific integer scores emitted by PSI-BLAST.
* PSFM: PSI-BLAST weighted observed residue frequencies.

Target frequencies used by Top-n-gram are also parsed here, but remain a
distinct feature type rather than an alias for PSFM.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_ALPHABET = tuple(AA_ORDER)
AA_INDEX = {amino_acid: index for index, amino_acid in enumerate(AA_ORDER)}
PROFILE_INPUT_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
PROFILE_X_ALIASES = frozenset({"J", "U", "O"})

PROFILE_CONTRACT_VERSION = 4
PROFILE_SCHEMA = "pepplatform.sequence_profile"
PROFILE_MATRIX_KEY = "matrix"
PROFILE_MATRIX_ARCHIVE_MEMBER = f"{PROFILE_MATRIX_KEY}.npy"

FEATURE_PSSM = "pssm"
FEATURE_PSSM_BLOSUM62_HYBRID = "pssm_blosum62_hybrid"
FEATURE_AC_PSSM_BLOSUM62_HYBRID = "ac_pssm_blosum62_hybrid"
FEATURE_CC_PSSM_BLOSUM62_HYBRID = "cc_pssm_blosum62_hybrid"
FEATURE_ACC_PSSM_BLOSUM62_HYBRID = "acc_pssm_blosum62_hybrid"
FEATURE_PSFM = "psfm"
FEATURE_PSFM_BLOSUM62_HYBRID = "psfm_blosum62_hybrid"
FEATURE_TARGET_FREQUENCY = "target_frequency"
PSSM_HYBRID_FEATURE_TYPES = frozenset(
    {
        FEATURE_PSSM_BLOSUM62_HYBRID,
        FEATURE_AC_PSSM_BLOSUM62_HYBRID,
        FEATURE_CC_PSSM_BLOSUM62_HYBRID,
        FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
    }
)
PROFILE_MATRIX_LAYOUT = "query_position_by_amino_acid"
PSSM_SCORE_FEATURE_TYPES = frozenset({FEATURE_PSSM, *PSSM_HYBRID_FEATURE_TYPES})
PSFM_HYBRID_FEATURE_TYPES = frozenset({FEATURE_PSFM_BLOSUM62_HYBRID})
PSFM_FREQUENCY_FEATURE_TYPES = frozenset({FEATURE_PSFM, *PSFM_HYBRID_FEATURE_TYPES})
PSSM_HYBRID_DOWNSTREAM_TRANSFORMS = {
    FEATURE_PSSM_BLOSUM62_HYBRID: "identity",
    FEATURE_AC_PSSM_BLOSUM62_HYBRID: "ac_pssm",
    FEATURE_CC_PSSM_BLOSUM62_HYBRID: "cc_pssm",
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID: "acc_pssm",
}
PROFILE_HYBRID_FALLBACK_TRIGGER = "strict_profile_unavailable_after_successful_search"
PSSM_HYBRID_SEMANTICS = {
    feature_type: {
        "method_name": feature_type,
        "matrix_layout": PROFILE_MATRIX_LAYOUT,
        "matrix_semantics": "psi_blast_pssm_or_blosum62_score_fallback",
        "downstream_transform": downstream_transform,
        "fallback_trigger": PROFILE_HYBRID_FALLBACK_TRIGGER,
    }
    for feature_type, downstream_transform in (PSSM_HYBRID_DOWNSTREAM_TRANSFORMS.items())
}
PROFILE_FEATURE_TYPES = frozenset(
    {
        *PSSM_SCORE_FEATURE_TYPES,
        *PSFM_FREQUENCY_FEATURE_TYPES,
        FEATURE_TARGET_FREQUENCY,
    }
)

SOURCE_PSI_BLAST_PSSM = "psi_blast_pssm"
SOURCE_PSI_BLAST_WEIGHTED_OBSERVED = "psi_blast_weighted_observed"
SOURCE_PSI_BLAST_TARGET_FREQUENCY = "psi_blast_target_frequency"
SOURCE_BLOSUM62_FALLBACK = "blosum62_score_fallback"
SOURCE_BLOSUM62_PSFM_FALLBACK = "blosum62_target_frequency_psfm_fallback"
SOURCE_BLOSUM62_TARGET_FREQUENCY = "blosum62_target_frequency_fallback"

PROFILE_ROW_SUM_ATOL = 1e-5
PSFM_ASCII_CROSSCHECK_TOLERANCE_PERCENTAGE_POINTS = 1
PROFILE_HIT_SUMMARY_FIELDS = (
    "qseqid",
    "sseqid",
    "pident",
    "qstart",
    "qend",
    "qlen",
    "evalue",
    "bitscore",
)
PROFILE_QUERY_COVERAGE_FORMULA = "(qend-qstart+1)/qlen"
PSFM_SEMANTICS = {
    "method_name": "PSFM",
    "qualified_name": ("PSI-BLAST weighted-observed position-specific frequency matrix"),
    "checkpoint_field": "weightedResFreqsPerPos",
    "frequency_stage": "weighted_observed_before_pseudocounts",
    "pseudocounts_included": False,
    "zero_observation_rows": "preserved",
    "ascii_frequency_crosscheck_tolerance_percentage_points": (
        PSFM_ASCII_CROSSCHECK_TOLERANCE_PERCENTAGE_POINTS
    ),
}
PSFM_HYBRID_SEMANTICS = {
    "method_name": FEATURE_PSFM_BLOSUM62_HYBRID,
    "qualified_name": (
        "PSI-BLAST weighted-observed PSFM with BLOSUM62-derived target-frequency fallback"
    ),
    "matrix_layout": PROFILE_MATRIX_LAYOUT,
    "primary_source": SOURCE_PSI_BLAST_WEIGHTED_OBSERVED,
    "fallback_source": SOURCE_BLOSUM62_PSFM_FALLBACK,
    "fallback_matrix": "blosum62_target_frequency_profile",
    "fallback_trigger": PROFILE_HYBRID_FALLBACK_TRIGGER,
    "zero_observation_rows": "preserved_for_psi_blast_profiles",
    "ascii_frequency_crosscheck_tolerance_percentage_points": (
        PSFM_ASCII_CROSSCHECK_TOLERANCE_PERCENTAGE_POINTS
    ),
}
TARGET_FREQUENCY_STRICT_SEMANTICS = {
    "method_name": FEATURE_TARGET_FREQUENCY,
    "matrix_layout": PROFILE_MATRIX_LAYOUT,
    "matrix_semantics": "psi_blast_target_frequency",
}
TARGET_FREQUENCY_HYBRID_SEMANTICS = {
    "method_name": FEATURE_TARGET_FREQUENCY,
    "matrix_layout": PROFILE_MATRIX_LAYOUT,
    "matrix_semantics": ("psi_blast_target_frequency_or_blosum62_target_frequency_fallback"),
    "fallback_trigger": PROFILE_HYBRID_FALLBACK_TRIGGER,
}
PROFILE_FALLBACK_REASON_CODES = frozenset(
    {
        "profile_output_missing",
        "profile_output_unreadable",
        "profile_header_invalid",
        "profile_row_invalid",
        "profile_value_invalid",
        "profile_shape_invalid",
        "profile_length_mismatch",
        "profile_position_invalid",
        "profile_sequence_mismatch",
        "profile_frequency_unavailable",
        "profile_frequency_not_normalized",
        "profile_frequency_crosscheck_failed",
        "profile_checkpoint_field_missing",
        "profile_checkpoint_duplicate_field",
        "profile_checkpoint_invalid",
        "profile_checkpoint_shape_invalid",
        "profile_checkpoint_value_invalid",
        "profile_checkpoint_query_missing",
        "profile_checkpoint_query_invalid",
        "profile_checkpoint_unreadable",
        "profile_checkpoint_noncanonical_mass",
        "psfm_query_residue_invalid",
    }
)


# Standard BLOSUM62 score rows in half-bit units. Columns use AA_ORDER.
BLOSUM62_HALF_BIT_SCORES = np.asarray(
    [
        [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
        [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
        [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
        [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
        [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
        [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
        [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
        [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
        [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
        [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
        [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
        [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
        [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
        [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
        [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
        [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
        [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
        [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
    ],
    dtype=np.int8,
)

# Standard extended rows distributed with BLOSUM62.
BLOSUM62_AMBIGUOUS_ROWS = {
    "B": np.asarray(
        [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3],
        dtype=np.int8,
    ),
    "Z": np.asarray(
        [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2],
        dtype=np.int8,
    ),
    "X": np.asarray(
        [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1],
        dtype=np.int8,
    ),
}

# NCBI BLOSUM62 background probabilities in AA_ORDER.
BLOSUM62_BACKGROUND = np.asarray(
    [
        0.074216205067993410,
        0.051614486141284632,
        0.044645808512758175,
        0.053626000835514163,
        0.024687457167944848,
        0.034259650591366467,
        0.054311925684587504,
        0.074146941452064999,
        0.026212984805262735,
        0.067917367618953758,
        0.098907868497150946,
        0.058155682303079400,
        0.024990197579643109,
        0.047418459742284756,
        0.038538003320306206,
        0.057229029180032963,
        0.050891364550728126,
        0.013029956129972818,
        0.032281512313581263,
        0.072919098205619245,
    ],
    dtype=np.float64,
)
BLOSUM62_BACKGROUND /= BLOSUM62_BACKGROUND.sum()

BLOSUM62_MATRIX_SHA256 = hashlib.sha256(
    AA_ORDER.encode("ascii") + BLOSUM62_HALF_BIT_SCORES.tobytes(order="C")
).hexdigest()
BLOSUM62_BACKGROUND_SHA256 = hashlib.sha256(
    AA_ORDER.encode("ascii") + BLOSUM62_BACKGROUND.tobytes(order="C")
).hexdigest()

BLOSUM62_REFERENCE = {
    "method": "BLOSUM62",
    "authors": "Henikoff and Henikoff",
    "year": 1992,
    "doi": "10.1073/pnas.89.22.10915",
}
PSI_BLAST_REFERENCE = {
    "method": "PSI-BLAST",
    "authors": "Altschul et al.",
    "year": 1997,
    "doi": "10.1093/nar/25.17.3389",
}
PSI_BLAST_COMPOSITION_REFERENCE = {
    "method": "composition-based statistics for PSI-BLAST",
    "authors": "Schäffer et al.",
    "year": 2001,
    "doi": "10.1093/nar/29.14.2994",
}
BLAST_PLUS_REFERENCE = {
    "method": "NCBI BLAST+",
    "authors": "Camacho et al.",
    "year": 2009,
    "doi": "10.1186/1471-2105-10-421",
}
PSFM_NAMING_REFERENCE = {
    "method": "weighted-observation-frequency PSFM",
    "authors": "Martin et al.",
    "year": 2007,
    "doi": "10.1109/TCBB.2007.70256",
}
PROFILE_SCIENTIFIC_REFERENCES = (
    BLOSUM62_REFERENCE,
    PSI_BLAST_REFERENCE,
    PSI_BLAST_COMPOSITION_REFERENCE,
    BLAST_PLUS_REFERENCE,
)

# NCBIstdaa numeric codes for AA_ORDER.
NCBISTDAA_CODES = np.asarray(
    [1, 16, 13, 4, 3, 15, 5, 7, 8, 9, 11, 10, 12, 6, 14, 17, 18, 20, 22, 19],
    dtype=np.int64,
)
NCBISTDAA_ALPHABET = "-ABCDEFGHIKLMNPQRSTVWXYZU*OJ"
NCBISTDAA_CODE_TO_RESIDUE = {index: residue for index, residue in enumerate(NCBISTDAA_ALPHABET)}
NCBISTDAA_CANONICAL_CODE_SET = frozenset(int(value) for value in NCBISTDAA_CODES)


class SequenceProfileError(ValueError):
    """Raised when a profile violates the scientific or storage contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "sequence_profile_invalid",
        details: Mapping[str, object] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def blosum62_contract_metadata(k: int = 1) -> dict[str, Any]:
    """Return an auditable description of this BLOSUM62 encoder route."""

    k_value = int(k)
    if k_value < 1:
        raise SequenceProfileError(
            "BLOSUM62 k must be at least 1",
            code="blosum62_k_invalid",
        )
    direct = k_value == 1
    return {
        "method_name": ("BLOSUM62" if direct else "BLOSUM62-derived k-mer mean"),
        "canonical_blosum62_per_residue": direct,
        "matrix": "BLOSUM62",
        "aa_order": AA_ORDER,
        "matrix_sha256": BLOSUM62_MATRIX_SHA256,
        "score_semantics": "integer_log_odds",
        "score_unit": "half_bit",
        "ambiguous_rows": ["B", "Z", "X"],
        "aliases_to_x": ["J", "U", "O"],
        "k": k_value,
        "kmer_aggregation": (
            "direct_residue_score_row" if direct else "arithmetic_mean_of_residue_score_rows"
        ),
        "scientific_reference": dict(BLOSUM62_REFERENCE),
    }


def _normalized_input_letters(sequence: object) -> str:
    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise SequenceProfileError(
            "profile sequence is empty after removing whitespace",
            code="profile_sequence_empty",
        )
    invalid = sorted(set(normalized).difference(PROFILE_INPUT_ALPHABET))
    if invalid:
        rendered = ", ".join(repr(character) for character in invalid)
        raise SequenceProfileError(
            f"profile sequence contains unsupported character(s): {rendered}",
            code="profile_sequence_character_invalid",
        )
    return normalized


def normalize_profile_sequence(sequence: object) -> str:
    """Normalize a sequence for BLAST/profile generation."""

    normalized = _normalized_input_letters(sequence)
    return "".join("X" if residue in PROFILE_X_ALIASES else residue for residue in normalized)


def normalize_psfm_sequence(sequence: object) -> str:
    """Validate the canonical 20-residue query alphabet required by PSFM v3."""

    normalized = _normalized_input_letters(sequence)
    unsupported = sorted(set(normalized).difference(AA_ORDER))
    if unsupported:
        rendered = ", ".join(repr(residue) for residue in unsupported)
        raise SequenceProfileError(
            f"PSFM requires the 20 canonical amino acids; unsupported residue(s): {rendered}",
            code="psfm_query_residue_invalid",
        )
    return normalized


def input_sequence_sha256(sequence: object) -> str:
    return hashlib.sha256(_normalized_input_letters(sequence).encode("ascii")).hexdigest()


def profile_sequence_sha256(sequence: object) -> str:
    return hashlib.sha256(normalize_profile_sequence(sequence).encode("ascii")).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blosum62_row(residue: object) -> np.ndarray:
    value = str(residue or "").upper()
    if len(value) != 1 or value not in PROFILE_INPUT_ALPHABET:
        raise SequenceProfileError(
            f"BLOSUM62 residue must be one ASCII letter, got {residue!r}",
            code="blosum62_residue_invalid",
        )
    if value in PROFILE_X_ALIASES:
        value = "X"
    index = AA_INDEX.get(value)
    if index is not None:
        return BLOSUM62_HALF_BIT_SCORES[index]
    row = BLOSUM62_AMBIGUOUS_ROWS.get(value)
    if row is None:
        raise SequenceProfileError(
            f"BLOSUM62 has no supported row for residue {value!r}",
            code="blosum62_residue_invalid",
        )
    return row


def blosum62_token_mean(token: object) -> np.ndarray:
    value = str(token or "")
    if not value:
        raise SequenceProfileError(
            "BLOSUM62 token must be a non-empty string",
            code="blosum62_token_invalid",
        )
    rows = np.stack([blosum62_row(residue) for residue in value], axis=0)
    return rows.astype(np.float32).mean(axis=0)


def blosum62_score_profile(sequence: object) -> np.ndarray:
    normalized = _normalized_input_letters(sequence)
    return np.stack([blosum62_row(residue) for residue in normalized], axis=0).astype(np.int32)


def blosum62_target_frequency_profile(sequence: object) -> np.ndarray:
    normalized = _normalized_input_letters(sequence)
    rows = np.empty((len(normalized), len(AA_ORDER)), dtype=np.float64)
    for position, residue in enumerate(normalized):
        weights = BLOSUM62_BACKGROUND * np.exp2(blosum62_row(residue).astype(np.float64) / 2.0)
        rows[position] = weights / weights.sum()
    return validate_frequency_matrix(
        rows.astype(np.float32),
        context="BLOSUM62 target-frequency profile",
        allow_zero_rows=False,
    )[0]


def validate_pssm_matrix(
    matrix: object,
    *,
    expected_length: int | None = None,
    context: str = "PSSM",
) -> np.ndarray:
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[1] != len(AA_ORDER):
        raise SequenceProfileError(
            f"{context} must have shape [L,20], got {tuple(value.shape)}",
            code="profile_shape_invalid",
        )
    if value.shape[0] < 1:
        raise SequenceProfileError(
            f"{context} must contain at least one row",
            code="profile_shape_invalid",
        )
    if expected_length is not None and value.shape[0] != int(expected_length):
        raise SequenceProfileError(
            f"{context} has {value.shape[0]} rows; expected {int(expected_length)}",
            code="profile_length_mismatch",
        )
    if not np.isfinite(value).all():
        raise SequenceProfileError(
            f"{context} contains non-finite values",
            code="profile_value_invalid",
        )
    if not np.equal(value, np.rint(value)).all():
        raise SequenceProfileError(
            f"{context} must contain integer PSI-BLAST scores",
            code="profile_value_invalid",
        )
    info = np.iinfo(np.int32)
    if (value < info.min).any() or (value > info.max).any():
        raise SequenceProfileError(
            f"{context} contains scores outside int32 range",
            code="profile_value_invalid",
        )
    return value.astype(np.int32, copy=False)


def validate_frequency_matrix(
    matrix: object,
    *,
    expected_length: int | None = None,
    context: str,
    allow_zero_rows: bool,
) -> tuple[np.ndarray, tuple[int, ...]]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(AA_ORDER):
        raise SequenceProfileError(
            f"{context} must have shape [L,20], got {tuple(value.shape)}",
            code="profile_shape_invalid",
        )
    if value.shape[0] < 1:
        raise SequenceProfileError(
            f"{context} must contain at least one row",
            code="profile_shape_invalid",
        )
    if expected_length is not None and value.shape[0] != int(expected_length):
        raise SequenceProfileError(
            f"{context} has {value.shape[0]} rows; expected {int(expected_length)}",
            code="profile_length_mismatch",
        )
    if not np.isfinite(value).all() or (value < 0).any():
        raise SequenceProfileError(
            f"{context} contains non-finite or negative values",
            code="profile_value_invalid",
        )
    row_sums = value.sum(axis=1)
    zero_rows = tuple(int(index) for index in np.flatnonzero(row_sums == 0.0))
    if zero_rows and not allow_zero_rows:
        raise SequenceProfileError(
            f"{context} contains zero-sum rows at {list(zero_rows[:8])}",
            code="profile_frequency_unavailable",
        )
    nonzero = row_sums > 0.0
    if nonzero.any():
        deviations = np.abs(row_sums[nonzero] - 1.0)
        if (deviations > PROFILE_ROW_SUM_ATOL).any():
            raise SequenceProfileError(
                f"{context} non-zero rows must sum to 1 within {PROFILE_ROW_SUM_ATOL}",
                code="profile_frequency_not_normalized",
            )
    return value.astype(np.float32), zero_rows


def parse_ascii_pssm(
    path: os.PathLike[str] | str,
    query_sequence: object,
) -> np.ndarray:
    """Parse and validate NCBI's ASCII PSSM score section."""

    query = normalize_profile_sequence(query_sequence)
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SequenceProfileError(
            f"cannot read ASCII PSSM {os.fspath(path)!r}: {exc}",
            code="profile_output_unreadable",
        ) from exc

    header: list[str] | None = None
    rows: list[list[int]] = []
    residues: list[str] = []
    positions: list[int] = []
    for line in lines:
        parts = line.strip().split()
        if header is None and len(parts) >= 20:
            candidate = [token.upper() for token in parts[:20]]
            if len(set(candidate)) == 20 and set(candidate) == set(AA_ORDER):
                header = candidate
                continue
        if not parts or not parts[0].isdigit():
            continue
        if header is None:
            raise SequenceProfileError(
                "ASCII PSSM contains data rows before a canonical amino-acid header",
                code="profile_header_invalid",
            )
        if len(parts) < 22 or len(parts[1]) != 1:
            raise SequenceProfileError(
                f"malformed ASCII PSSM row: {line!r}",
                code="profile_row_invalid",
            )
        try:
            position = int(parts[0])
            values = [int(token) for token in parts[2:22]]
        except ValueError as exc:
            raise SequenceProfileError(
                f"ASCII PSSM row {parts[0]!r} contains non-integer scores",
                code="profile_value_invalid",
            ) from exc
        positions.append(position)
        residues.append(parts[1].upper())
        rows.append(values)

    if header is None:
        raise SequenceProfileError(
            "ASCII PSSM is missing the canonical amino-acid header",
            code="profile_header_invalid",
        )
    if len(rows) != len(query):
        raise SequenceProfileError(
            f"ASCII PSSM has {len(rows)} rows; expected {len(query)}",
            code="profile_length_mismatch",
        )
    expected_positions = list(range(1, len(query) + 1))
    if positions != expected_positions:
        raise SequenceProfileError(
            "ASCII PSSM positions must be continuous and one-based",
            code="profile_position_invalid",
        )
    mismatches = [
        index + 1
        for index, (observed, expected) in enumerate(zip(residues, query))
        if observed != expected
    ]
    if mismatches:
        raise SequenceProfileError(
            f"ASCII PSSM query residues differ at positions {mismatches[:8]}",
            code="profile_sequence_mismatch",
        )

    order = [header.index(amino_acid) for amino_acid in AA_ORDER]
    matrix = np.asarray(rows, dtype=np.int32)[:, order]
    return validate_pssm_matrix(
        matrix,
        expected_length=len(query),
        context="ASCII PSSM",
    )


def parse_ascii_pssm_frequencies(
    path: os.PathLike[str] | str,
    query_sequence: object,
) -> np.ndarray:
    """Parse the integer weighted-observation percentages in an ASCII PSSM."""

    query = normalize_profile_sequence(query_sequence)
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SequenceProfileError(
            f"cannot read ASCII PSSM {os.fspath(path)!r}: {exc}",
            code="profile_output_unreadable",
        ) from exc

    header: list[str] | None = None
    rows: list[list[int]] = []
    residues: list[str] = []
    positions: list[int] = []
    for line in lines:
        parts = line.strip().split()
        if header is None and len(parts) >= 20:
            candidate = [token.upper() for token in parts[:20]]
            if len(set(candidate)) == 20 and set(candidate) == set(AA_ORDER):
                header = candidate
                continue
        if not parts or not parts[0].isdigit():
            continue
        if header is None or len(parts) < 42 or len(parts[1]) != 1:
            raise SequenceProfileError(
                f"malformed ASCII PSSM frequency row: {line!r}",
                code="profile_row_invalid",
            )
        try:
            position = int(parts[0])
            frequencies = [int(token) for token in parts[22:42]]
        except ValueError as exc:
            raise SequenceProfileError(
                f"ASCII PSSM row {parts[0]!r} contains non-integer frequencies",
                code="profile_value_invalid",
            ) from exc
        if any(value < 0 or value > 100 for value in frequencies):
            raise SequenceProfileError(
                f"ASCII PSSM row {position} has a frequency outside [0,100]",
                code="profile_value_invalid",
            )
        positions.append(position)
        residues.append(parts[1].upper())
        rows.append(frequencies)

    if header is None:
        raise SequenceProfileError(
            "ASCII PSSM is missing the canonical amino-acid header",
            code="profile_header_invalid",
        )
    if len(rows) != len(query) or positions != list(range(1, len(query) + 1)):
        raise SequenceProfileError(
            "ASCII PSSM frequency positions do not match the query",
            code="profile_length_mismatch",
        )
    mismatches = [
        index + 1
        for index, (observed, expected) in enumerate(zip(residues, query))
        if observed != expected
    ]
    if mismatches:
        raise SequenceProfileError(
            f"ASCII PSSM query residues differ at positions {mismatches[:8]}",
            code="profile_sequence_mismatch",
        )
    order = [header.index(amino_acid) for amino_acid in AA_ORDER]
    percentages = np.asarray(rows, dtype=np.int16)[:, order]
    sums = percentages.sum(axis=1)
    invalid_sums = np.flatnonzero((sums != 0) & ((sums < 95) | (sums > 105)))
    if invalid_sums.size:
        raise SequenceProfileError(
            "ASCII PSSM weighted-observation percentages have implausible "
            f"row sums at {invalid_sums[:8].tolist()}",
            code="profile_frequency_not_normalized",
        )
    return percentages


_ASN_REAL = re.compile(r"\{\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\}")


def _extract_named_brace_block(text: str, field: str) -> str:
    matches = list(re.finditer(rf"\b{re.escape(field)}\b", text))
    if not matches:
        raise SequenceProfileError(
            f"PSI-BLAST checkpoint is missing {field}",
            code="profile_checkpoint_field_missing",
        )
    if len(matches) != 1:
        raise SequenceProfileError(
            f"PSI-BLAST checkpoint contains duplicate {field} fields",
            code="profile_checkpoint_duplicate_field",
        )
    match = matches[0]
    start = text.find("{", match.end())
    if start < 0:
        raise SequenceProfileError(
            f"PSI-BLAST checkpoint has no value block for {field}",
            code="profile_checkpoint_invalid",
        )
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise SequenceProfileError(
        f"PSI-BLAST checkpoint contains an unterminated {field} block",
        code="profile_checkpoint_invalid",
    )


@dataclass(frozen=True)
class _CheckpointDimensions:
    num_rows: int
    num_columns: int
    by_row: bool


def _checkpoint_dimensions(text: str) -> _CheckpointDimensions:
    def integer_field(name: str) -> int:
        matches = list(re.finditer(rf"\b{re.escape(name)}\s+(\d+)\s*,", text))
        if not matches:
            raise SequenceProfileError(
                f"PSI-BLAST checkpoint is missing {name}",
                code="profile_checkpoint_field_missing",
            )
        if len(matches) != 1:
            raise SequenceProfileError(
                f"PSI-BLAST checkpoint contains duplicate {name} fields",
                code="profile_checkpoint_duplicate_field",
            )
        return int(matches[0].group(1))

    protein_matches = list(re.finditer(r"\bisProtein\s+(TRUE|FALSE)\s*,", text))
    if len(protein_matches) != 1 or protein_matches[0].group(1) != "TRUE":
        raise SequenceProfileError(
            "PSI-BLAST checkpoint must uniquely declare isProtein TRUE",
            code="profile_checkpoint_field_missing",
        )
    by_row_matches = list(re.finditer(r"\bbyRow\s+(TRUE|FALSE)\s*,", text))
    if len(by_row_matches) != 1:
        raise SequenceProfileError(
            "PSI-BLAST checkpoint must uniquely declare byRow",
            code="profile_checkpoint_field_missing",
        )
    dimensions = _CheckpointDimensions(
        num_rows=integer_field("numRows"),
        num_columns=integer_field("numColumns"),
        by_row=by_row_matches[0].group(1) == "TRUE",
    )
    if dimensions.num_rows != len(NCBISTDAA_ALPHABET) or dimensions.num_columns < 1:
        raise SequenceProfileError(
            "unsupported PSI-BLAST checkpoint dimensions "
            f"({dimensions.num_rows}, {dimensions.num_columns})",
            code="profile_checkpoint_shape_invalid",
        )
    return dimensions


def _parse_checkpoint_real_matrix(
    text: str,
    *,
    field: str,
    dimensions: _CheckpointDimensions,
) -> np.ndarray:
    block = _extract_named_brace_block(text, field)
    triples = _ASN_REAL.findall(block)
    remainder = _ASN_REAL.sub("", block)
    if re.sub(r"[\s,]", "", remainder):
        raise SequenceProfileError(
            f"PSI-BLAST checkpoint {field} contains malformed REAL values",
            code="profile_checkpoint_value_invalid",
        )
    expected = dimensions.num_rows * dimensions.num_columns
    if len(triples) != expected:
        raise SequenceProfileError(
            f"PSI-BLAST checkpoint {field} contains {len(triples)} values; expected {expected}",
            code="profile_checkpoint_shape_invalid",
        )
    values = np.empty(expected, dtype=np.float64)
    for index, (mantissa_text, base_text, exponent_text) in enumerate(triples):
        mantissa = int(mantissa_text)
        base = int(base_text)
        exponent = int(exponent_text)
        if base not in {2, 10}:
            raise SequenceProfileError(
                f"PSI-BLAST checkpoint has invalid real-number base {base}",
                code="profile_checkpoint_value_invalid",
            )
        try:
            values[index] = float(mantissa) * math.pow(float(base), exponent)
        except (OverflowError, ValueError) as exc:
            raise SequenceProfileError(
                f"PSI-BLAST checkpoint {field} contains an invalid real number",
                code="profile_checkpoint_value_invalid",
            ) from exc
    if not np.isfinite(values).all():
        raise SequenceProfileError(
            f"PSI-BLAST checkpoint {field} contains a non-finite real number",
            code="profile_checkpoint_value_invalid",
        )
    if dimensions.by_row:
        return values.reshape(dimensions.num_rows, dimensions.num_columns).T
    return values.reshape(dimensions.num_columns, dimensions.num_rows)


def _checkpoint_query_sequence(text: str) -> str:
    matches = list(
        re.finditer(
            r"\bseq-data\s+ncbistdaa\s+'([0-9A-Fa-f\s]+)'H",
            text,
        )
    )
    if len(matches) != 1:
        raise SequenceProfileError(
            "PSI-BLAST checkpoint must contain exactly one ncbistdaa query",
            code="profile_checkpoint_query_missing",
        )
    compact = re.sub(r"\s", "", matches[0].group(1))
    if not compact or len(compact) % 2:
        raise SequenceProfileError(
            "PSI-BLAST checkpoint query has invalid hexadecimal encoding",
            code="profile_checkpoint_query_invalid",
        )
    try:
        codes = bytes.fromhex(compact)
    except ValueError as exc:
        raise SequenceProfileError(
            "PSI-BLAST checkpoint query has invalid hexadecimal encoding",
            code="profile_checkpoint_query_invalid",
        ) from exc
    residues: list[str] = []
    for position, code in enumerate(codes, start=1):
        residue = NCBISTDAA_CODE_TO_RESIDUE.get(code)
        if residue is None:
            raise SequenceProfileError(
                f"PSI-BLAST checkpoint query has invalid code {code} at position {position}",
                code="profile_checkpoint_query_invalid",
            )
        residues.append(residue)
    return "".join(residues)


def _read_checkpoint(
    path: os.PathLike[str] | str,
    *,
    require_query: bool,
) -> tuple[str, _CheckpointDimensions, str | None]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SequenceProfileError(
            f"cannot read PSI-BLAST checkpoint {os.fspath(path)!r}: {exc}",
            code="profile_checkpoint_unreadable",
        ) from exc
    if len(re.findall(r"\bPssmWithParameters\s*::=", text)) != 1:
        raise SequenceProfileError(
            "PSI-BLAST checkpoint is not an ASN.1 PssmWithParameters document",
            code="profile_checkpoint_invalid",
        )
    pssm = _extract_named_brace_block(text, "pssm")
    query = _checkpoint_query_sequence(text) if require_query else None
    return text, _checkpoint_dimensions(pssm), query


def parse_psiblast_psfm(
    checkpoint: os.PathLike[str] | str,
    *,
    expected_sequence: object,
    ascii_pssm: os.PathLike[str] | str | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    expected = normalize_psfm_sequence(expected_sequence)
    text, dimensions, checkpoint_query = _read_checkpoint(
        checkpoint,
        require_query=True,
    )
    if checkpoint_query != expected:
        raise SequenceProfileError(
            "PSI-BLAST checkpoint query sequence does not match the requested PSFM sequence",
            code="profile_sequence_mismatch",
        )
    if dimensions.num_columns != len(expected):
        raise SequenceProfileError(
            f"PSI-BLAST checkpoint has {dimensions.num_columns} columns; expected {len(expected)}",
            code="profile_length_mismatch",
        )
    serialized = _parse_checkpoint_real_matrix(
        text,
        field="weightedResFreqsPerPos",
        dimensions=dimensions,
    )
    if not np.isfinite(serialized).all() or (serialized < 0.0).any():
        raise SequenceProfileError(
            "PSI-BLAST weighted observed frequencies contain non-finite or negative values",
            code="profile_checkpoint_value_invalid",
        )
    noncanonical_codes = sorted(
        set(range(dimensions.num_rows)).difference(NCBISTDAA_CANONICAL_CODE_SET)
    )
    noncanonical_mass = serialized[:, noncanonical_codes].sum(axis=1)
    invalid_positions = np.flatnonzero(noncanonical_mass > PROFILE_ROW_SUM_ATOL)
    if invalid_positions.size:
        raise SequenceProfileError(
            "PSI-BLAST weighted observed frequencies place mass in "
            "non-canonical residue channels at positions "
            f"{invalid_positions[:8].tolist()}",
            code="profile_checkpoint_noncanonical_mass",
        )
    matrix = serialized[:, NCBISTDAA_CODES]
    validated, zero_rows = validate_frequency_matrix(
        matrix,
        expected_length=len(expected),
        context="PSI-BLAST weighted observed frequencies",
        allow_zero_rows=True,
    )
    if ascii_pssm is not None:
        percentages = parse_ascii_pssm_frequencies(ascii_pssm, expected)
        # BLAST renders non-negative percentages with conventional half-up
        # integer rounding, whereas numpy.rint uses ties-to-even.
        expected_percentages = np.floor(validated.astype(np.float64) * 100.0 + 0.5).astype(np.int16)
        differences = np.abs(percentages - expected_percentages)
        mismatches = np.argwhere(differences > PSFM_ASCII_CROSSCHECK_TOLERANCE_PERCENTAGE_POINTS)
        if mismatches.size:
            position, residue_index = mismatches[0].tolist()
            raise SequenceProfileError(
                "ASN.1 weighted frequencies disagree with ASCII PSSM "
                f"percentages at position {position + 1}, residue "
                f"{AA_ORDER[residue_index]} by "
                f"{int(differences[position, residue_index])} percentage "
                "points",
                code="profile_frequency_crosscheck_failed",
                details={
                    "position": position + 1,
                    "residue": AA_ORDER[residue_index],
                    "asn1_rounded_percentage": int(expected_percentages[position, residue_index]),
                    "ascii_percentage": int(percentages[position, residue_index]),
                    "difference_percentage_points": int(differences[position, residue_index]),
                    "tolerance_percentage_points": (
                        PSFM_ASCII_CROSSCHECK_TOLERANCE_PERCENTAGE_POINTS
                    ),
                },
            )
    return validated, zero_rows


def parse_psiblast_target_frequencies(
    checkpoint: os.PathLike[str] | str,
    *,
    expected_length: int | None = None,
    expected_sequence: object | None = None,
) -> np.ndarray:
    text, dimensions, checkpoint_query = _read_checkpoint(
        checkpoint,
        require_query=expected_sequence is not None,
    )
    if expected_sequence is not None:
        expected = normalize_profile_sequence(expected_sequence)
        if checkpoint_query != expected:
            raise SequenceProfileError(
                "PSI-BLAST checkpoint query sequence does not match the "
                "requested target-frequency sequence",
                code="profile_sequence_mismatch",
            )
        if expected_length is not None and len(expected) != int(expected_length):
            raise SequenceProfileError(
                "expected target-frequency length and sequence disagree",
                code="profile_length_mismatch",
            )
        expected_length = len(expected)
    if (
        expected_length is not None
        and checkpoint_query is not None
        and len(checkpoint_query) != int(expected_length)
    ):
        raise SequenceProfileError(
            f"PSI-BLAST checkpoint query has length {len(checkpoint_query)}; "
            f"expected {int(expected_length)}",
            code="profile_length_mismatch",
        )
    if expected_length is not None and dimensions.num_columns != int(expected_length):
        raise SequenceProfileError(
            f"PSI-BLAST checkpoint has {dimensions.num_columns} columns; "
            f"expected {int(expected_length)}",
            code="profile_length_mismatch",
        )
    serialized = _parse_checkpoint_real_matrix(
        text,
        field="freqRatios",
        dimensions=dimensions,
    )
    # NCBI keeps the historical freqRatios member name, but the PSI engine
    # stores normalized target-frequency numerators Q_i in this array and
    # divides by standard probabilities P_i when constructing scores.
    matrix = serialized[:, NCBISTDAA_CODES]
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise SequenceProfileError(
            "PSI-BLAST target frequencies contain non-finite or negative values",
            code="profile_checkpoint_value_invalid",
        )
    row_sums = matrix.sum(axis=1)
    if (row_sums <= 0.0).any():
        positions = np.flatnonzero(row_sums <= 0.0).tolist()
        raise SequenceProfileError(
            f"PSI-BLAST target frequencies contain zero-sum rows at {positions[:8]}",
            code="profile_frequency_unavailable",
        )
    matrix = matrix / row_sums[:, np.newaxis]
    validated, _ = validate_frequency_matrix(
        matrix,
        expected_length=expected_length,
        context="PSI-BLAST target frequencies",
        allow_zero_rows=False,
    )
    return validated


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SequenceProfileError(
                f"profile manifest contains duplicate key {key!r}",
                code="profile_manifest_duplicate_key",
            )
        result[key] = value
    return result


def read_json_strict(path: os.PathLike[str] | str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    except SequenceProfileError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SequenceProfileError(
            f"cannot read profile manifest {os.fspath(path)!r}: {exc}",
            code="profile_manifest_unreadable",
        ) from exc


def _safe_manifest_path(root: Path, relative_path: object) -> Path:
    value = str(relative_path or "")
    candidate = Path(value)
    if not value or candidate.is_absolute():
        raise SequenceProfileError(
            f"profile matrix path must be relative, got {value!r}",
            code="profile_manifest_path_invalid",
        )
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise SequenceProfileError(
            f"profile matrix path escapes manifest root: {value!r}",
            code="profile_manifest_path_escape",
        ) from exc
    return resolved


def _expected_sources(feature_type: str) -> frozenset[str]:
    if feature_type == FEATURE_PSSM:
        return frozenset({SOURCE_PSI_BLAST_PSSM})
    if feature_type in PSSM_HYBRID_FEATURE_TYPES:
        return frozenset({SOURCE_PSI_BLAST_PSSM, SOURCE_BLOSUM62_FALLBACK})
    if feature_type == FEATURE_PSFM:
        return frozenset({SOURCE_PSI_BLAST_WEIGHTED_OBSERVED})
    if feature_type in PSFM_HYBRID_FEATURE_TYPES:
        return frozenset(
            {
                SOURCE_PSI_BLAST_WEIGHTED_OBSERVED,
                SOURCE_BLOSUM62_PSFM_FALLBACK,
            }
        )
    return frozenset({SOURCE_PSI_BLAST_TARGET_FREQUENCY, SOURCE_BLOSUM62_TARGET_FREQUENCY})


def profile_audit_summary(
    samples: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the complete v4 audit summary from sample metadata."""

    zero_count = sum(len(metadata["zero_observation_positions"]) for metadata in samples.values())
    residue_count = sum(int(metadata["sequence_length"]) for metadata in samples.values())
    fallback_reasons = Counter(
        str(metadata.get("fallback_reason") or "")
        for metadata in samples.values()
        if metadata.get("fallback") is True
    )
    fallback_count = sum(fallback_reasons.values())
    return {
        "sample_count": len(samples),
        "samples_with_zero_observations": sum(
            bool(metadata["zero_observation_positions"]) for metadata in samples.values()
        ),
        "zero_observation_position_count": zero_count,
        "zero_observation_fraction": (zero_count / residue_count if residue_count else 0.0),
        "final_iteration_unique_subject_count": sum(
            int(metadata["quality"]["nonself_hit_count"]) for metadata in samples.values()
        ),
        "fallback_sample_count": fallback_count,
        "fallback_fraction": (fallback_count / len(samples) if samples else 0.0),
        "fallback_reason_counts": dict(sorted(fallback_reasons.items())),
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _positive_finite(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric) and numeric > 0.0


def _nonnegative_finite(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric) and numeric >= 0.0


def _validate_manifest(
    manifest: object,
    *,
    expected_feature_type: str,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise SequenceProfileError(
            "profile manifest must be a JSON object",
            code="profile_manifest_invalid",
        )
    if manifest.get("schema") != PROFILE_SCHEMA:
        raise SequenceProfileError(
            "legacy or unsupported profile manifest schema; rebuild the cache",
            code="profile_manifest_legacy",
        )
    if manifest.get("contract_version") != PROFILE_CONTRACT_VERSION:
        raise SequenceProfileError(
            "legacy or unsupported profile contract version; rebuild the cache",
            code="profile_manifest_legacy",
        )
    feature_type = str(manifest.get("feature_type") or "")
    if feature_type != expected_feature_type:
        raise SequenceProfileError(
            f"profile manifest feature_type={feature_type!r}; expected {expected_feature_type!r}",
            code="profile_manifest_feature_mismatch",
        )
    if manifest.get("aa_order") != AA_ORDER:
        raise SequenceProfileError(
            f"profile manifest aa_order must be {AA_ORDER!r}",
            code="profile_manifest_aa_order_invalid",
        )
    if manifest.get("matrix_layout") != PROFILE_MATRIX_LAYOUT:
        raise SequenceProfileError(
            f"profile manifest matrix_layout must be {PROFILE_MATRIX_LAYOUT!r}",
            code="profile_manifest_layout_invalid",
        )
    if not isinstance(manifest.get("provenance"), dict):
        raise SequenceProfileError(
            "profile manifest requires provenance",
            code="profile_manifest_provenance_missing",
        )
    provenance = manifest["provenance"]
    if provenance.get("profile_contract_version") != PROFILE_CONTRACT_VERSION:
        raise SequenceProfileError(
            "profile provenance contract version does not match the manifest",
            code="profile_manifest_provenance_invalid",
        )
    required_text = {
        "algorithm": "NCBI PSI-BLAST",
        "implementation": "NCBI BLAST+",
        "self_exclusion": ("all subject IDs with an identical normalized query sequence"),
    }
    for key, expected in required_text.items():
        if provenance.get(key) != expected:
            raise SequenceProfileError(
                f"profile provenance {key!r} must be {expected!r}",
                code="profile_manifest_provenance_invalid",
            )
    if provenance.get("algorithm_lineage") != "Altschul_et_al_1997":
        raise SequenceProfileError(
            "profile provenance has an unsupported algorithm lineage",
            code="profile_manifest_provenance_invalid",
        )
    for key in ("blast_version", "makeblastdb_version", "database_scope"):
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise SequenceProfileError(
                f"profile provenance requires non-empty {key!r}",
                code="profile_manifest_provenance_invalid",
            )
    references = provenance.get("scientific_references")
    expected_dois = {str(reference["doi"]) for reference in PROFILE_SCIENTIFIC_REFERENCES}
    if (
        not isinstance(references, list)
        or {
            str(reference.get("doi") or "")
            for reference in references
            if isinstance(reference, dict)
        }
        != expected_dois
    ):
        raise SequenceProfileError(
            "profile provenance requires the pinned BLOSUM62, PSI-BLAST, "
            "composition-statistics, and BLAST+ references",
            code="profile_manifest_provenance_invalid",
        )
    if not _is_sha256(provenance.get("subject_fasta_sha256")):
        raise SequenceProfileError(
            "profile provenance requires subject_fasta_sha256",
            code="profile_manifest_provenance_invalid",
        )
    subject_artifact = provenance.get("subject_fasta_artifact")
    if (
        not isinstance(subject_artifact, str)
        or not subject_artifact
        or Path(subject_artifact).is_absolute()
    ):
        raise SequenceProfileError(
            "profile provenance requires a relative subject_fasta_artifact",
            code="profile_manifest_provenance_invalid",
        )
    if not _is_sha256(provenance.get("generation_fingerprint")):
        raise SequenceProfileError(
            "profile provenance requires generation_fingerprint",
            code="profile_manifest_provenance_invalid",
        )
    audit_artifact = provenance.get("audit_artifact")
    if (
        not isinstance(audit_artifact, str)
        or not audit_artifact
        or Path(audit_artifact).is_absolute()
        or not _is_sha256(provenance.get("audit_sha256"))
        or not isinstance(provenance.get("audit_summary"), dict)
    ):
        raise SequenceProfileError(
            "profile provenance requires a pinned full audit artifact",
            code="profile_manifest_provenance_invalid",
        )
    if provenance.get("blosum62_matrix_sha256") != BLOSUM62_MATRIX_SHA256:
        raise SequenceProfileError(
            "profile provenance BLOSUM62 matrix digest does not match",
            code="profile_manifest_provenance_invalid",
        )
    if provenance.get("blosum62_background_sha256") != BLOSUM62_BACKGROUND_SHA256:
        raise SequenceProfileError(
            "profile provenance BLOSUM62 background digest does not match",
            code="profile_manifest_provenance_invalid",
        )
    scientific_parameters = provenance.get("scientific_parameters")
    required_scientific_parameters = {
        "iterations",
        "evalue",
        "inclusion_evalue",
        "matrix",
        "gap_open",
        "gap_extend",
        "word_size",
        "threshold",
        "max_target_sequences",
        "pseudocount",
        "composition_based_stats",
        "seg",
        "soft_masking",
        "save_after_last_round",
    }
    if (
        not isinstance(scientific_parameters, dict)
        or set(scientific_parameters) != required_scientific_parameters
        or scientific_parameters.get("matrix") != "BLOSUM62"
        or isinstance(scientific_parameters.get("iterations"), bool)
        or not isinstance(scientific_parameters.get("iterations"), int)
        or scientific_parameters["iterations"] < 2
        or not _positive_finite(scientific_parameters.get("evalue"))
        or not _positive_finite(scientific_parameters.get("inclusion_evalue"))
        or not _nonnegative_int(scientific_parameters.get("gap_open"))
        or not _nonnegative_int(scientific_parameters.get("gap_extend"))
        or not _positive_int(scientific_parameters.get("word_size"))
        or scientific_parameters["word_size"] < 2
        or not _nonnegative_finite(scientific_parameters.get("threshold"))
        or not _positive_int(scientific_parameters.get("max_target_sequences"))
        or not _nonnegative_int(scientific_parameters.get("pseudocount"))
        or scientific_parameters.get("composition_based_stats") != "1"
        or scientific_parameters.get("seg") != "no"
        or scientific_parameters.get("soft_masking") is not False
        or scientific_parameters.get("save_after_last_round") is not True
    ):
        raise SequenceProfileError(
            "profile provenance has incomplete scientific parameters",
            code="profile_manifest_provenance_invalid",
        )
    runtime_parameters = provenance.get("runtime_parameters")
    if (
        not isinstance(runtime_parameters, dict)
        or set(runtime_parameters) != {"threads", "timeout_seconds"}
        or not _positive_int(runtime_parameters.get("threads"))
        or not _positive_int(runtime_parameters.get("timeout_seconds"))
    ):
        raise SequenceProfileError(
            "profile provenance requires valid runtime_parameters",
            code="profile_manifest_provenance_invalid",
        )
    if not isinstance(provenance.get("database"), dict) or not provenance["database"]:
        raise SequenceProfileError(
            "profile provenance requires non-empty database metadata",
            code="profile_manifest_provenance_invalid",
        )
    normalization = provenance.get("sequence_normalization")
    expected_normalization = (
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
    )
    if normalization != expected_normalization:
        raise SequenceProfileError(
            "profile provenance has an unsupported sequence normalization",
            code="profile_manifest_provenance_invalid",
        )
    if provenance.get("quality_summary") != {
        "outfmt_fields": list(PROFILE_HIT_SUMMARY_FIELDS),
        "query_coverage": PROFILE_QUERY_COVERAGE_FORMULA,
        "iteration_scope": "final_completed_iteration",
        "hit_count_unit": "unique_subject",
    }:
        raise SequenceProfileError(
            "profile provenance has an unsupported quality-summary contract",
            code="profile_manifest_provenance_invalid",
        )
    semantics = provenance.get("profile_semantics")
    if feature_type == FEATURE_PSFM:
        expected_semantics = {
            **PSFM_SEMANTICS,
            "aa_order": AA_ORDER,
            "scientific_reference": dict(PSFM_NAMING_REFERENCE),
        }
        if semantics != expected_semantics:
            raise SequenceProfileError(
                "PSFM provenance does not identify weighted-observed frequency semantics",
                code="profile_manifest_provenance_invalid",
            )
    elif feature_type in PSFM_HYBRID_FEATURE_TYPES:
        expected_semantics = {
            **PSFM_HYBRID_SEMANTICS,
            "aa_order": AA_ORDER,
            "scientific_reference": dict(PSFM_NAMING_REFERENCE),
        }
        if semantics != expected_semantics:
            raise SequenceProfileError(
                "hybrid PSFM provenance does not identify its weighted-observed "
                "and BLOSUM62-derived frequency semantics",
                code="profile_manifest_provenance_invalid",
            )
    elif feature_type in PSSM_HYBRID_FEATURE_TYPES:
        expected_semantics = PSSM_HYBRID_SEMANTICS[feature_type]
        if semantics != expected_semantics:
            raise SequenceProfileError(
                "hybrid PSSM provenance does not match its feature-specific downstream transform",
                code="profile_manifest_provenance_invalid",
            )
    elif feature_type == FEATURE_TARGET_FREQUENCY:
        expected_semantics = (
            TARGET_FREQUENCY_HYBRID_SEMANTICS
            if provenance.get("fallback_policy") == "blosum62_target_frequency"
            else TARGET_FREQUENCY_STRICT_SEMANTICS
        )
        if semantics != expected_semantics:
            raise SequenceProfileError(
                "target-frequency provenance does not match its fallback policy",
                code="profile_manifest_provenance_invalid",
            )
    elif not isinstance(semantics, dict) or not semantics:
        raise SequenceProfileError(
            "profile provenance requires feature semantics",
            code="profile_manifest_provenance_invalid",
        )
    if provenance.get("feature_type") != feature_type:
        raise SequenceProfileError(
            "profile provenance feature_type does not match the manifest",
            code="profile_manifest_provenance_invalid",
        )
    fallback_policy = provenance.get("fallback_policy")
    allowed_fallback_policies = {
        FEATURE_PSSM: {"error"},
        **{hybrid_feature: {"blosum62_hybrid"} for hybrid_feature in PSSM_HYBRID_FEATURE_TYPES},
        FEATURE_PSFM: {"error"},
        **{hybrid_feature: {"blosum62_hybrid"} for hybrid_feature in PSFM_HYBRID_FEATURE_TYPES},
        FEATURE_TARGET_FREQUENCY: {
            "error",
            "blosum62_target_frequency",
        },
    }
    if fallback_policy not in allowed_fallback_policies[feature_type]:
        raise SequenceProfileError(
            "profile provenance fallback policy is invalid for its feature",
            code="profile_manifest_provenance_invalid",
        )
    samples = manifest.get("samples")
    if not isinstance(samples, dict):
        raise SequenceProfileError(
            "profile manifest samples must be an object",
            code="profile_manifest_samples_invalid",
        )
    return manifest


@dataclass(frozen=True)
class LoadedProfile:
    matrix: torch.Tensor
    metadata: Mapping[str, Any]


class ProfileStore:
    """Validated access to an immutable profile manifest and its matrices."""

    def __init__(self, index_path: os.PathLike[str] | str, *, feature_type: str):
        if feature_type not in PROFILE_FEATURE_TYPES:
            raise SequenceProfileError(
                f"unsupported profile feature type {feature_type!r}",
                code="profile_feature_type_invalid",
            )
        self.index_path = Path(index_path)
        self.root = self.index_path.parent
        self.feature_type = feature_type
        self.manifest_sha256 = sha256_file(self.index_path)
        self.manifest = _validate_manifest(
            read_json_strict(self.index_path),
            expected_feature_type=feature_type,
        )
        self.samples = self.manifest["samples"]
        provenance = self.manifest["provenance"]
        self.fallback_policy = str(provenance.get("fallback_policy") or "")
        subject_path = _safe_manifest_path(
            self.root,
            provenance["subject_fasta_artifact"],
        )
        if not subject_path.is_file():
            raise SequenceProfileError(
                f"profile subject FASTA artifact is missing: {subject_path}",
                code="profile_database_input_missing",
            )
        if sha256_file(subject_path) != provenance["subject_fasta_sha256"]:
            raise SequenceProfileError(
                "profile subject FASTA artifact digest does not match provenance",
                code="profile_database_digest_mismatch",
            )
        audit_path = _safe_manifest_path(
            self.root,
            provenance["audit_artifact"],
        )
        if not audit_path.is_file() or sha256_file(audit_path) != provenance["audit_sha256"]:
            raise SequenceProfileError(
                "profile audit artifact is missing or has a digest mismatch",
                code="profile_audit_digest_mismatch",
            )
        audit = read_json_strict(audit_path)
        if (
            not isinstance(audit, dict)
            or audit.get("feature_type") != self.feature_type
            or audit.get("generation_fingerprint") != provenance["generation_fingerprint"]
            or audit.get("summary") != provenance["audit_summary"]
            or not isinstance(audit.get("samples"), list)
            or len(audit["samples"]) != len(self.samples)
        ):
            raise SequenceProfileError(
                "profile audit artifact does not match the manifest",
                code="profile_audit_invalid",
            )
        audit_rows = audit["samples"]
        audit_by_id: dict[str, Mapping[str, Any]] = {}
        for row in audit_rows:
            sample_id = row.get("sample_id") if isinstance(row, dict) else None
            if not isinstance(sample_id, str) or not sample_id or sample_id in audit_by_id:
                raise SequenceProfileError(
                    "profile audit contains an invalid or duplicate sample ID",
                    code="profile_audit_invalid",
                )
            audit_by_id[sample_id] = row
        if set(audit_by_id) != set(self.samples):
            raise SequenceProfileError(
                "profile audit sample IDs do not match the manifest",
                code="profile_audit_invalid",
            )
        if any(
            not isinstance(metadata, dict) or not isinstance(metadata.get("quality"), dict)
            for metadata in self.samples.values()
        ):
            raise SequenceProfileError(
                "profile manifest samples cannot support the audit contract",
                code="profile_audit_invalid",
            )
        try:
            expected_summary = profile_audit_summary(self.samples)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise SequenceProfileError(
                "profile manifest cannot support its audit summary",
                code="profile_audit_invalid",
            ) from exc
        if audit["summary"] != expected_summary:
            raise SequenceProfileError(
                "profile audit summary does not match manifest samples",
                code="profile_audit_invalid",
            )
        for sample_id, metadata in self.samples.items():
            row = audit_by_id[sample_id]
            if not isinstance(row.get("cache_reused"), bool):
                raise SequenceProfileError(
                    f"profile audit row has invalid cache reuse state for {sample_id!r}",
                    code="profile_audit_invalid",
                )
            expected_fields = {
                "sequence_length": metadata.get("sequence_length"),
                "zero_observation_positions": metadata.get("zero_observation_positions"),
                "profile_source": metadata.get("profile_source"),
                "fallback": metadata.get("fallback"),
                "fallback_reason": metadata.get("fallback_reason"),
                **dict(metadata.get("quality") or {}),
            }
            if any(row.get(key) != value for key, value in expected_fields.items()):
                raise SequenceProfileError(
                    f"profile audit row does not match sample {sample_id!r}",
                    code="profile_audit_invalid",
                )

    def load(self, sample_id: object, sequence: object) -> LoadedProfile:
        identifier = str(sample_id or "")
        if not identifier:
            raise SequenceProfileError(
                "profile encoding requires a non-empty sample_id",
                code="profile_sample_id_missing",
            )
        metadata = self.samples.get(identifier)
        if not isinstance(metadata, dict):
            raise SequenceProfileError(
                f"profile is missing for sample_id={identifier!r}",
                code="profile_sample_missing",
            )
        normalized_input = _normalized_input_letters(sequence)
        normalized_profile = (
            normalize_psfm_sequence(sequence)
            if self.feature_type == FEATURE_PSFM
            else normalize_profile_sequence(sequence)
        )
        expected_input_sha = hashlib.sha256(normalized_input.encode("ascii")).hexdigest()
        expected_profile_sha = hashlib.sha256(normalized_profile.encode("ascii")).hexdigest()
        if metadata.get("sequence_sha256") != expected_input_sha:
            raise SequenceProfileError(
                f"profile input sequence digest mismatch for sample_id={identifier!r}",
                code="profile_sequence_mismatch",
            )
        if metadata.get("profile_sequence_sha256") != expected_profile_sha:
            raise SequenceProfileError(
                f"profile normalized sequence digest mismatch for sample_id={identifier!r}",
                code="profile_sequence_mismatch",
            )
        if metadata.get("sequence_length") != len(normalized_profile):
            raise SequenceProfileError(
                f"profile sequence length mismatch for sample_id={identifier!r}",
                code="profile_length_mismatch",
            )
        recorded_zero_rows = metadata.get("zero_observation_positions")
        if (
            not isinstance(recorded_zero_rows, list)
            or any(
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
                or position >= len(normalized_profile)
                for position in recorded_zero_rows
            )
            or recorded_zero_rows != sorted(set(recorded_zero_rows))
            or metadata.get("zero_observation_count") != len(recorded_zero_rows)
            or metadata.get("zero_observation_fraction")
            != len(recorded_zero_rows) / len(normalized_profile)
        ):
            raise SequenceProfileError(
                "profile zero-observation metadata is invalid",
                code="profile_manifest_sample_invalid",
            )

        source = str(metadata.get("profile_source") or "")
        if source not in _expected_sources(self.feature_type):
            raise SequenceProfileError(
                f"profile source {source!r} is invalid for {self.feature_type!r}",
                code="profile_source_invalid",
            )
        fallback = metadata.get("fallback")
        if not isinstance(fallback, bool):
            raise SequenceProfileError(
                "profile sample metadata requires a boolean fallback field",
                code="profile_manifest_sample_invalid",
            )
        if (
            self.feature_type not in PSSM_HYBRID_FEATURE_TYPES
            and self.feature_type not in PSFM_HYBRID_FEATURE_TYPES
            and fallback
        ):
            if not (
                self.feature_type == FEATURE_TARGET_FREQUENCY
                and source == SOURCE_BLOSUM62_TARGET_FREQUENCY
            ):
                raise SequenceProfileError(
                    f"{self.feature_type!r} does not permit fallback samples",
                    code="profile_fallback_forbidden",
                )
        fallback_reason = metadata.get("fallback_reason")
        if not isinstance(fallback_reason, str):
            raise SequenceProfileError(
                "profile sample metadata requires a fallback_reason string",
                code="profile_manifest_sample_invalid",
            )
        if fallback and not fallback_reason:
            raise SequenceProfileError(
                "fallback profile samples require a recorded reason",
                code="profile_manifest_sample_invalid",
            )
        if not fallback and fallback_reason:
            raise SequenceProfileError(
                "non-fallback profile samples cannot record a fallback reason",
                code="profile_manifest_sample_invalid",
            )
        if self.feature_type in PSSM_HYBRID_FEATURE_TYPES:
            expected_fallback = source == SOURCE_BLOSUM62_FALLBACK
            if fallback != expected_fallback:
                raise SequenceProfileError(
                    "hybrid profile fallback flag does not match its source",
                    code="profile_manifest_sample_invalid",
                )
            if fallback and fallback_reason not in PROFILE_FALLBACK_REASON_CODES:
                raise SequenceProfileError(
                    "hybrid profile has an unsupported fallback reason",
                    code="profile_manifest_sample_invalid",
                )
        if self.feature_type in PSFM_HYBRID_FEATURE_TYPES:
            expected_fallback = source == SOURCE_BLOSUM62_PSFM_FALLBACK
            if fallback != expected_fallback:
                raise SequenceProfileError(
                    "hybrid PSFM fallback flag does not match its source",
                    code="profile_manifest_sample_invalid",
                )
            if fallback and fallback_reason not in PROFILE_FALLBACK_REASON_CODES:
                raise SequenceProfileError(
                    "hybrid PSFM has an unsupported fallback reason",
                    code="profile_manifest_sample_invalid",
                )
        if self.feature_type == FEATURE_TARGET_FREQUENCY:
            expected_fallback = source == SOURCE_BLOSUM62_TARGET_FREQUENCY
            if fallback != expected_fallback:
                raise SequenceProfileError(
                    "target-frequency fallback flag does not match its source",
                    code="profile_manifest_sample_invalid",
                )
            if fallback and self.fallback_policy != "blosum62_target_frequency":
                raise SequenceProfileError(
                    "target-frequency fallback is not authorized by provenance",
                    code="profile_fallback_forbidden",
                )
            if fallback and fallback_reason not in PROFILE_FALLBACK_REASON_CODES:
                raise SequenceProfileError(
                    "target-frequency profile has an unsupported fallback reason",
                    code="profile_manifest_sample_invalid",
                )

        quality = metadata.get("quality")
        if not isinstance(quality, dict):
            raise SequenceProfileError(
                "profile sample metadata requires quality metrics",
                code="profile_manifest_sample_invalid",
            )
        hit_count = quality.get("nonself_hit_count")
        zero_hits = quality.get("zero_nonself_hits")
        coverage = quality.get("max_query_coverage")
        identity = quality.get("max_identity")
        min_evalue = quality.get("min_evalue")
        final_iteration = quality.get("final_iteration")
        hit_rows = quality.get("final_iteration_hit_rows")
        blast_warnings = quality.get("blast_warnings", [])
        if (
            isinstance(hit_count, bool)
            or not isinstance(hit_count, int)
            or hit_count < 0
            or not isinstance(zero_hits, bool)
            or zero_hits != (hit_count == 0)
            or not _positive_int(final_iteration)
            or not _nonnegative_int(hit_rows)
            or hit_rows < hit_count
            or not isinstance(blast_warnings, list)
            or any(not isinstance(warning, str) or not warning for warning in blast_warnings)
        ):
            raise SequenceProfileError(
                "profile hit-count quality metadata is invalid",
                code="profile_manifest_sample_invalid",
            )
        try:
            quality_values = np.asarray(
                [float(coverage), float(identity)],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise SequenceProfileError(
                "profile quality coverage/identity values are invalid",
                code="profile_manifest_sample_invalid",
            ) from exc
        if (
            not np.isfinite(quality_values).all()
            or not 0.0 <= quality_values[0] <= 1.0
            or not 0.0 <= quality_values[1] <= 100.0
        ):
            raise SequenceProfileError(
                "profile quality coverage/identity values are out of range",
                code="profile_manifest_sample_invalid",
            )
        if min_evalue is not None:
            try:
                numeric_evalue = float(min_evalue)
            except (TypeError, ValueError) as exc:
                raise SequenceProfileError(
                    "profile minimum E-value is invalid",
                    code="profile_manifest_sample_invalid",
                ) from exc
            if not np.isfinite(numeric_evalue) or numeric_evalue < 0.0:
                raise SequenceProfileError(
                    "profile minimum E-value is invalid",
                    code="profile_manifest_sample_invalid",
                )
        if (
            zero_hits
            and (quality_values[0] != 0.0 or quality_values[1] != 0.0 or min_evalue is not None)
        ) or (not zero_hits and (quality_values[0] <= 0.0 or min_evalue is None)):
            raise SequenceProfileError(
                "profile quality metrics are inconsistent with the hit count",
                code="profile_manifest_sample_invalid",
            )
        matrix_path = _safe_manifest_path(self.root, metadata.get("path"))
        if not matrix_path.is_file():
            raise SequenceProfileError(
                f"profile matrix is missing: {matrix_path}",
                code="profile_matrix_missing",
            )
        expected_sha = str(metadata.get("matrix_sha256") or "")
        if not _is_sha256(expected_sha) or sha256_file(matrix_path) != expected_sha:
            raise SequenceProfileError(
                f"profile matrix digest mismatch: {matrix_path}",
                code="profile_matrix_digest_mismatch",
            )
        expected_shape = metadata.get("matrix_shape")
        if expected_shape != [len(normalized_profile), len(AA_ORDER)]:
            raise SequenceProfileError(
                f"profile manifest matrix_shape is invalid for sample_id={identifier!r}",
                code="profile_manifest_sample_invalid",
            )
        expected_dtype = "int32" if self.feature_type in PSSM_SCORE_FEATURE_TYPES else "float32"
        if metadata.get("matrix_dtype") != expected_dtype:
            raise SequenceProfileError(
                f"profile matrix dtype must be {expected_dtype} for {self.feature_type}",
                code="profile_dtype_invalid",
            )
        try:
            with zipfile.ZipFile(matrix_path, "r") as archive:
                members = archive.infolist()
                if (
                    len(members) != 1
                    or members[0].filename != PROFILE_MATRIX_ARCHIVE_MEMBER
                    or members[0].file_size > len(normalized_profile) * len(AA_ORDER) * 8 + 4096
                ):
                    raise SequenceProfileError(
                        "profile archive has an unexpected or oversized member",
                        code="profile_matrix_archive_invalid",
                    )
            with np.load(matrix_path, allow_pickle=False) as payload:
                if payload.files != [PROFILE_MATRIX_KEY]:
                    raise SequenceProfileError(
                        f"profile archive must contain only {PROFILE_MATRIX_KEY!r}",
                        code="profile_matrix_archive_invalid",
                    )
                matrix = np.asarray(payload[PROFILE_MATRIX_KEY])
        except SequenceProfileError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise SequenceProfileError(
                f"cannot load profile matrix {matrix_path}: {exc}",
                code="profile_matrix_unreadable",
            ) from exc

        if list(matrix.shape) != expected_shape:
            raise SequenceProfileError(
                f"profile matrix shape mismatch for sample_id={identifier!r}",
                code="profile_shape_invalid",
            )
        if str(matrix.dtype) != expected_dtype:
            raise SequenceProfileError(
                f"profile matrix dtype mismatch for sample_id={identifier!r}",
                code="profile_dtype_invalid",
            )

        if self.feature_type in PSSM_SCORE_FEATURE_TYPES:
            validated = validate_pssm_matrix(
                matrix,
                expected_length=len(normalized_profile),
            )
        else:
            validated, zero_rows = validate_frequency_matrix(
                matrix,
                expected_length=len(normalized_profile),
                context=self.feature_type,
                allow_zero_rows=(self.feature_type in PSFM_FREQUENCY_FEATURE_TYPES),
            )
            if list(zero_rows) != recorded_zero_rows:
                raise SequenceProfileError(
                    f"profile zero-row metadata mismatch for sample_id={identifier!r}",
                    code="profile_manifest_sample_invalid",
                )
        return LoadedProfile(
            matrix=torch.from_numpy(np.array(validated, copy=True)).float(),
            metadata=metadata,
        )


def atomic_save_profile_matrix(
    path: os.PathLike[str] | str,
    matrix: object,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            np.savez_compressed(handle, **{PROFILE_MATRIX_KEY: np.asarray(matrix)})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return sha256_file(target)


def atomic_write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def profile_sample_metadata(
    *,
    path: os.PathLike[str] | str,
    matrix: np.ndarray,
    sequence: object,
    matrix_sha256: str,
    profile_source: str,
    fallback: bool,
    fallback_reason: str = "",
    zero_observation_positions: tuple[int, ...] = (),
    quality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_input = _normalized_input_letters(sequence)
    normalized_profile = normalize_profile_sequence(sequence)
    metadata: dict[str, Any] = {
        "path": Path(path).as_posix(),
        "sequence_sha256": hashlib.sha256(normalized_input.encode("ascii")).hexdigest(),
        "profile_sequence_sha256": hashlib.sha256(normalized_profile.encode("ascii")).hexdigest(),
        "sequence_length": len(normalized_profile),
        "matrix_sha256": matrix_sha256,
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
        "profile_source": profile_source,
        "fallback": bool(fallback),
        "fallback_reason": str(fallback_reason or ""),
        "zero_observation_positions": list(zero_observation_positions),
        "zero_observation_count": len(zero_observation_positions),
        "zero_observation_fraction": (len(zero_observation_positions) / len(normalized_profile)),
    }
    if quality:
        metadata["quality"] = dict(quality)
    return metadata


def new_profile_manifest(
    *,
    feature_type: str,
    samples: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if feature_type not in PROFILE_FEATURE_TYPES:
        raise SequenceProfileError(
            f"unsupported profile feature type {feature_type!r}",
            code="profile_feature_type_invalid",
        )
    manifest = {
        "schema": PROFILE_SCHEMA,
        "contract_version": PROFILE_CONTRACT_VERSION,
        "feature_type": feature_type,
        "aa_order": AA_ORDER,
        "matrix_layout": PROFILE_MATRIX_LAYOUT,
        "samples": dict(samples),
        "provenance": dict(provenance),
    }
    return _validate_manifest(manifest, expected_feature_type=feature_type)
