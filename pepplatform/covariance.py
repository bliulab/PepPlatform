"""Scientific contract for sequence AC, CC, and ACC encoders.

The physicochemical properties are from Guo et al. (2008), Supplementary
Table S1 (doi:10.1093/nar/gkn159). AC/CC follow the auto-cross covariance
decomposition introduced by Wold et al. (1993).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Integral

import numpy as np


COVARIANCE_ENCODERS = frozenset({"ac", "cc", "acc"})
COVARIANCE_ENCODER_CONTRACT_VERSION = 1
COVARIANCE_DEFAULT_MAX_RAW_LENGTH = 100_000
COVARIANCE_MAX_RAW_LENGTH = 100_000

GUO_AMINO_ACID_ORDER = tuple("ACDEFGHIKLMNPQRSTVWY")
GUO_PROPERTY_NAMES = ("H1", "H2", "V", "P1", "P2", "SASA", "NCI")
GUO_SOURCE_DOI = "10.1093/nar/gkn159"
GUO_SUPPLEMENTARY_DOC_SHA256 = "f98f4b5fa182ce6053e5d46364c5a16fcfde4c2f75fedcb45503d50e7b647a59"

# Rows follow GUO_AMINO_ACID_ORDER; columns follow GUO_PROPERTY_NAMES.
GUO_RAW_PROPERTY_TABLE = np.asarray(
    [
        [0.62, -0.5, 27.5, 8.1, 0.046, 1.181, 0.007187],
        [0.29, -1.0, 44.6, 5.5, 0.128, 1.461, -0.03661],
        [-0.9, 3.0, 40.0, 13.0, 0.105, 1.587, -0.02382],
        [-0.74, 3.0, 62.0, 12.3, 0.151, 1.862, 0.006802],
        [1.19, -2.5, 115.5, 5.2, 0.29, 2.228, 0.037552],
        [0.48, 0.0, 0.0, 9.0, 0.0, 0.881, 0.179052],
        [-0.4, -0.5, 79.0, 10.4, 0.23, 2.025, -0.01069],
        [1.38, -1.8, 93.5, 5.2, 0.186, 1.81, 0.021631],
        [-1.5, 3.0, 100.0, 11.3, 0.219, 2.258, 0.017708],
        [1.06, -1.8, 93.5, 4.9, 0.186, 1.931, 0.051672],
        [0.64, -1.3, 94.1, 5.7, 0.221, 2.034, 0.002683],
        [-0.78, 2.0, 58.7, 11.6, 0.134, 1.655, 0.005392],
        [0.12, 0.0, 41.9, 8.0, 0.131, 1.468, 0.239531],
        [-0.85, 0.2, 80.7, 10.5, 0.18, 1.932, 0.049211],
        [-2.53, 3.0, 105.0, 10.5, 0.291, 2.56, 0.043587],
        [-0.18, 0.3, 29.3, 9.2, 0.062, 1.298, 0.004627],
        [-0.05, -0.4, 51.3, 8.6, 0.108, 1.525, 0.003352],
        [1.08, -1.5, 71.5, 5.9, 0.14, 1.645, 0.057004],
        [0.81, -3.4, 145.5, 5.4, 0.409, 2.663, 0.037977],
        [0.26, -2.3, 117.3, 6.2, 0.298, 2.368, 0.023599],
    ],
    dtype=np.float64,
)


def validate_covariance_lag(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("covariance lag must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError("covariance lag must be a positive integer")
    return value


def validate_covariance_max_raw_length(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("covariance_max_raw_length must be a positive integer")
    value = int(value)
    if value < 1 or value > COVARIANCE_MAX_RAW_LENGTH:
        raise ValueError(
            f"covariance_max_raw_length must be in [1,{COVARIANCE_MAX_RAW_LENGTH}], got {value}"
        )
    return value


def parse_covariance_max_raw_length(value: str) -> int:
    """Parse the CLI string form while keeping the domain validator strict."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("covariance_max_raw_length must be an integer") from exc
    try:
        return validate_covariance_max_raw_length(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def standardize_property_table(raw_table) -> np.ndarray:
    """Z-score each property across the 20 amino acids using population SD."""

    try:
        table = np.asarray(raw_table, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("property table must be numeric") from exc
    expected_shape = (len(GUO_AMINO_ACID_ORDER), len(GUO_PROPERTY_NAMES))
    if table.shape != expected_shape:
        raise ValueError(f"property table must have shape {expected_shape}, got {table.shape}")
    if not np.isfinite(table).all():
        raise ValueError("property table must contain only finite values")
    standard_deviation = table.std(axis=0, ddof=0)
    if np.any(standard_deviation == 0.0):
        raise ValueError("property table columns must have non-zero variance")
    return (table - table.mean(axis=0)) / standard_deviation


GUO_NORMALIZED_PROPERTY_TABLE = standardize_property_table(GUO_RAW_PROPERTY_TABLE)
GUO_RAW_PROPERTY_TABLE.setflags(write=False)
GUO_NORMALIZED_PROPERTY_TABLE.setflags(write=False)


def _property_table_sha256(table: np.ndarray, *, normalized: bool) -> str:
    payload = {
        "amino_acid_order": list(GUO_AMINO_ACID_ORDER),
        "property_names": list(GUO_PROPERTY_NAMES),
        "normalized": bool(normalized),
        "values": np.asarray(table, dtype=np.float64).tolist(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


GUO_RAW_PROPERTY_TABLE_SHA256 = _property_table_sha256(
    GUO_RAW_PROPERTY_TABLE,
    normalized=False,
)
GUO_NORMALIZED_PROPERTY_TABLE_SHA256 = _property_table_sha256(
    GUO_NORMALIZED_PROPERTY_TABLE,
    normalized=True,
)

_AA_TO_INDEX = {aa: index for index, aa in enumerate(GUO_AMINO_ACID_ORDER)}


def normalize_covariance_sequence(sequence, *, max_raw_length: int) -> str:
    if not isinstance(sequence, str):
        raise ValueError(
            f"covariance encoders require a sequence string, got {type(sequence).__name__}"
        )
    limit = validate_covariance_max_raw_length(max_raw_length)
    normalized = sequence.strip().upper()
    if not normalized:
        raise ValueError("covariance encoders require a non-empty sequence")
    if any(character.isspace() for character in normalized):
        raise ValueError("covariance encoder sequence must not contain internal whitespace")
    if len(normalized) > limit:
        raise ValueError(
            f"sequence length {len(normalized)} exceeds covariance_max_raw_length={limit}"
        )
    invalid = sorted(set(normalized).difference(_AA_TO_INDEX))
    if invalid:
        raise ValueError(
            "covariance encoder sequence contains non-standard amino acid "
            f"residue(s): {', '.join(invalid)}"
        )
    return normalized


def covariance_feature_names(encoder: str, lag: int) -> tuple[str, ...]:
    normalized_encoder = str(encoder).strip().lower()
    if normalized_encoder not in COVARIANCE_ENCODERS:
        raise ValueError(f"unsupported covariance encoder: {encoder!r}")
    lag = validate_covariance_lag(lag)
    ac_names = tuple(
        f"ac_lag{distance}_{property_name}"
        for distance in range(1, lag + 1)
        for property_name in GUO_PROPERTY_NAMES
    )
    cc_names = tuple(
        f"cc_lag{distance}_{source}_to_{target}"
        for distance in range(1, lag + 1)
        for source in GUO_PROPERTY_NAMES
        for target in GUO_PROPERTY_NAMES
        if source != target
    )
    if normalized_encoder == "ac":
        return ac_names
    if normalized_encoder == "cc":
        return cc_names
    return ac_names + cc_names


def covariance_feature_dimension(encoder: str, lag: int) -> int:
    return len(covariance_feature_names(encoder, lag))


def compute_covariance_features(
    sequence,
    *,
    encoder: str,
    lag: int,
    max_raw_length: int,
    property_table=GUO_NORMALIZED_PROPERTY_TABLE,
) -> np.ndarray:
    normalized_encoder = str(encoder).strip().lower()
    if normalized_encoder not in COVARIANCE_ENCODERS:
        raise ValueError(f"unsupported covariance encoder: {encoder!r}")
    lag = validate_covariance_lag(lag)
    sequence = normalize_covariance_sequence(
        sequence,
        max_raw_length=max_raw_length,
    )
    if len(sequence) <= lag:
        raise ValueError(
            f"covariance encoder requires sequence length > lag; "
            f"got length={len(sequence)}, lag={lag}"
        )
    table = np.asarray(property_table, dtype=np.float64)
    expected_shape = (len(GUO_AMINO_ACID_ORDER), len(GUO_PROPERTY_NAMES))
    if table.shape != expected_shape:
        raise ValueError(
            f"normalized property table must have shape {expected_shape}, got {table.shape}"
        )
    if not np.isfinite(table).all():
        raise ValueError("normalized property table must contain only finite values")
    if np.any(table.std(axis=0, ddof=0) == 0.0):
        raise ValueError("normalized property table columns must have non-zero variance")

    indices = np.fromiter(
        (_AA_TO_INDEX[amino_acid] for amino_acid in sequence),
        dtype=np.intp,
        count=len(sequence),
    )
    properties = table[indices]
    centered = properties - properties.mean(axis=0, dtype=np.float64)
    ac_blocks = []
    cc_blocks = []
    off_diagonal = ~np.eye(len(GUO_PROPERTY_NAMES), dtype=bool)
    for distance in range(1, lag + 1):
        left = centered[:-distance]
        right = centered[distance:]
        denominator = len(sequence) - distance
        if normalized_encoder in {"ac", "acc"}:
            ac_blocks.append(np.sum(left * right, axis=0, dtype=np.float64) / denominator)
        if normalized_encoder in {"cc", "acc"}:
            directed = (left.T @ right) / denominator
            cc_blocks.append(directed[off_diagonal])

    ac = np.concatenate(ac_blocks) if ac_blocks else np.empty(0, dtype=np.float64)
    cc = np.concatenate(cc_blocks) if cc_blocks else np.empty(0, dtype=np.float64)
    if normalized_encoder == "ac":
        return ac
    if normalized_encoder == "cc":
        return cc
    return np.concatenate((ac, cc))


def build_covariance_encoder_contract(
    *,
    encoder: str,
    lag: int,
    max_raw_length: int,
) -> dict:
    normalized_encoder = str(encoder).strip().lower()
    lag = validate_covariance_lag(lag)
    max_raw_length = validate_covariance_max_raw_length(max_raw_length)
    output_dimension = covariance_feature_dimension(normalized_encoder, lag)
    return {
        "contract_version": COVARIANCE_ENCODER_CONTRACT_VERSION,
        "encoder": normalized_encoder,
        "lag": lag,
        "amino_acid_order": list(GUO_AMINO_ACID_ORDER),
        "property_names": list(GUO_PROPERTY_NAMES),
        "property_source_doi": GUO_SOURCE_DOI,
        "property_source_document_sha256": GUO_SUPPLEMENTARY_DOC_SHA256,
        "raw_property_table_sha256": GUO_RAW_PROPERTY_TABLE_SHA256,
        "normalized_property_table_sha256": GUO_NORMALIZED_PROPERTY_TABLE_SHA256,
        "property_normalization": {
            "formula": "(P[a,j]-mean_over_20_AA[j])/population_sd_over_20_AA[j]",
            "ddof": 0,
            "dtype": "float64",
            "pre_rounding": "none",
        },
        "sequence_centering": "complete_sequence_per_property_mean",
        "denominator": "n-lag",
        "cc_direction": "property_at_i_to_property_at_i_plus_lag",
        "feature_ordering": (
            "lag_major_property_order"
            if normalized_encoder == "ac"
            else (
                "lag_major_source_property_then_target_property_excluding_diagonal"
                if normalized_encoder == "cc"
                else "ac_block_then_cc_block_each_in_native_lag_major_order"
            )
        ),
        "full_length_policy": "complete_raw_sequence_no_tokenizer_truncation",
        "covariance_max_raw_length": max_raw_length,
        "calculation_dtype": "float64",
        "output_dtype": "float32",
        "output_dimension": output_dimension,
    }


def validate_covariance_encoder_contract(
    contract,
    *,
    encoder: str,
    lag: int,
    max_raw_length: int,
) -> dict:
    if not isinstance(contract, dict):
        raise ValueError("covariance_encoder_contract must be an object")
    expected = build_covariance_encoder_contract(
        encoder=encoder,
        lag=lag,
        max_raw_length=max_raw_length,
    )
    if contract != expected:
        raise ValueError(
            "covariance_encoder_contract does not match the supported Guo-7 v1 contract"
        )
    return expected
