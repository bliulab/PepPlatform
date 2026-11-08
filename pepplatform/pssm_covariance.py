"""Scientific contract for AC-PSSM, CC-PSSM, and ACC-PSSM.

The covariance formulas follow Dong, Zhou, and Guan (2009),
doi:10.1093/bioinformatics/btp500.  Feature ordering follows the pinned
BioSeq-BLM reference implementation documented below.
"""

from __future__ import annotations

import argparse
from numbers import Integral

import numpy as np

from pepplatform.sequence_profiles import (
    AA_ORDER,
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
    FEATURE_AC_PSSM_BLOSUM62_HYBRID,
    FEATURE_CC_PSSM_BLOSUM62_HYBRID,
    FEATURE_PSSM,
    PROFILE_CONTRACT_VERSION,
    PROFILE_HYBRID_FALLBACK_TRIGGER,
    validate_pssm_matrix,
)


STRICT_PSSM_COVARIANCE_ENCODERS = frozenset({"ac_pssm", "cc_pssm", "acc_pssm"})
HYBRID_PSSM_COVARIANCE_ENCODERS = frozenset(
    {
        FEATURE_AC_PSSM_BLOSUM62_HYBRID,
        FEATURE_CC_PSSM_BLOSUM62_HYBRID,
        FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
    }
)
PSSM_COVARIANCE_ENCODERS = frozenset(
    {*STRICT_PSSM_COVARIANCE_ENCODERS, *HYBRID_PSSM_COVARIANCE_ENCODERS}
)
PSSM_COVARIANCE_BASE_ENCODERS = {
    "ac_pssm": "ac_pssm",
    "cc_pssm": "cc_pssm",
    "acc_pssm": "acc_pssm",
    FEATURE_AC_PSSM_BLOSUM62_HYBRID: "ac_pssm",
    FEATURE_CC_PSSM_BLOSUM62_HYBRID: "cc_pssm",
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID: "acc_pssm",
}
PSSM_COVARIANCE_PROFILE_FEATURE_TYPES = {
    "ac_pssm": FEATURE_PSSM,
    "cc_pssm": FEATURE_PSSM,
    "acc_pssm": FEATURE_PSSM,
    FEATURE_AC_PSSM_BLOSUM62_HYBRID: FEATURE_AC_PSSM_BLOSUM62_HYBRID,
    FEATURE_CC_PSSM_BLOSUM62_HYBRID: FEATURE_CC_PSSM_BLOSUM62_HYBRID,
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID: FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
}
PSSM_COVARIANCE_LAG_KEYS = {encoder: f"{encoder}_lag" for encoder in PSSM_COVARIANCE_ENCODERS}
PSSM_COVARIANCE_ENCODER_CONTRACT_VERSION = 1
PSSM_COVARIANCE_DEFAULT_LAG = 10
PSSM_COVARIANCE_SOURCE_DOI = "10.1093/bioinformatics/btp500"
BIOSEQ_BLM_SOURCE_DOI = "10.1093/nar/gkab829"
BIOSEQ_BLM_REFERENCE_REPOSITORY = "https://github.com/Zimiao1025/BioSeq-BLM.git"
BIOSEQ_BLM_REFERENCE_COMMIT = "51bc6d1c802d93f6542cc2998d5b8e4ee9402361"
BIOSEQ_BLM_REFERENCE_PATH = "code/FeatureExtractionMode/SR/profile.py"


def normalize_pssm_covariance_encoder(encoder: str) -> str:
    normalized = str(encoder).strip().lower()
    if normalized not in PSSM_COVARIANCE_ENCODERS:
        raise ValueError(f"unsupported PSSM covariance encoder: {encoder!r}")
    return normalized


def pssm_covariance_base_encoder(encoder: str) -> str:
    return PSSM_COVARIANCE_BASE_ENCODERS[normalize_pssm_covariance_encoder(encoder)]


def pssm_covariance_lag_key(encoder: str) -> str:
    return PSSM_COVARIANCE_LAG_KEYS[normalize_pssm_covariance_encoder(encoder)]


def pssm_covariance_profile_feature_type(encoder: str) -> str:
    return PSSM_COVARIANCE_PROFILE_FEATURE_TYPES[normalize_pssm_covariance_encoder(encoder)]


def validate_pssm_covariance_lag(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("PSSM covariance lag must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError("PSSM covariance lag must be a positive integer")
    return value


def parse_pssm_covariance_lag(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("PSSM covariance lag must be an integer") from exc
    try:
        return validate_pssm_covariance_lag(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def pssm_covariance_feature_names(
    encoder: str,
    lag: int,
) -> tuple[str, ...]:
    normalized_encoder = normalize_pssm_covariance_encoder(encoder)
    base_encoder = pssm_covariance_base_encoder(normalized_encoder)
    hybrid = normalized_encoder in HYBRID_PSSM_COVARIANCE_ENCODERS
    ac_prefix = "ac_pssm_blosum62_hybrid" if hybrid else "ac_pssm"
    cc_prefix = "cc_pssm_blosum62_hybrid" if hybrid else "cc_pssm"
    lag = validate_pssm_covariance_lag(lag)
    result = []
    for distance in range(1, lag + 1):
        ac_names = [f"{ac_prefix}_lag{distance}_{amino_acid}" for amino_acid in AA_ORDER]
        cc_names = [
            f"{cc_prefix}_lag{distance}_{source}_to_{target}"
            for source in AA_ORDER
            for target in AA_ORDER
            if source != target
        ]
        if base_encoder in {"ac_pssm", "acc_pssm"}:
            result.extend(ac_names)
        if base_encoder in {"cc_pssm", "acc_pssm"}:
            result.extend(cc_names)
    return tuple(result)


def pssm_covariance_feature_dimension(encoder: str, lag: int) -> int:
    return len(pssm_covariance_feature_names(encoder, lag))


def compute_pssm_covariance_features(
    matrix,
    *,
    encoder: str,
    lag: int,
) -> np.ndarray:
    normalized_encoder = normalize_pssm_covariance_encoder(encoder)
    base_encoder = pssm_covariance_base_encoder(normalized_encoder)
    lag = validate_pssm_covariance_lag(lag)
    pssm = validate_pssm_matrix(
        matrix,
        context="PSSM covariance input",
    ).astype(np.float64, copy=False)
    length = int(pssm.shape[0])
    if length <= lag:
        raise ValueError(
            f"PSSM covariance requires profile length > lag; got length={length}, lag={lag}"
        )

    centered = pssm - pssm.mean(axis=0, dtype=np.float64)
    off_diagonal = ~np.eye(len(AA_ORDER), dtype=bool)
    output = []
    for distance in range(1, lag + 1):
        left = centered[:-distance]
        right = centered[distance:]
        denominator = length - distance
        if base_encoder in {"ac_pssm", "acc_pssm"}:
            output.append(np.sum(left * right, axis=0, dtype=np.float64) / denominator)
        if base_encoder in {"cc_pssm", "acc_pssm"}:
            directed = (left.T @ right) / denominator
            output.append(directed[off_diagonal])
    return np.concatenate(output).astype(np.float64, copy=False)


def build_pssm_covariance_encoder_contract(
    *,
    encoder: str,
    lag: int,
) -> dict:
    normalized_encoder = normalize_pssm_covariance_encoder(encoder)
    base_encoder = pssm_covariance_base_encoder(normalized_encoder)
    lag = validate_pssm_covariance_lag(lag)
    output_dimension = pssm_covariance_feature_dimension(
        normalized_encoder,
        lag,
    )
    ordering = {
        "ac_pssm": "lag_major_profile_column_order",
        "cc_pssm": ("lag_major_source_column_then_target_column_excluding_diagonal"),
        "acc_pssm": "lag_major_ac_20_then_cc_380_per_lag",
    }[base_encoder]
    contract = {
        "contract_version": PSSM_COVARIANCE_ENCODER_CONTRACT_VERSION,
        "encoder": normalized_encoder,
        "lag": lag,
        "method_source_doi": PSSM_COVARIANCE_SOURCE_DOI,
        "profile_contract_version": PROFILE_CONTRACT_VERSION,
        "profile_type": "psi_blast_pssm_integer_scores",
        "amino_acid_order": list(AA_ORDER),
        "sequence_centering": "complete_profile_per_column_mean",
        "column_scaling": "none",
        "denominator": "profile_length-distance",
        "cc_direction": "column_at_j_to_column_at_j_plus_distance",
        "cc_diagonal": "excluded",
        "feature_ordering": ordering,
        "full_length_policy": ("complete_profile_no_tokenizer_or_max_length_truncation"),
        "short_profile_policy": "error_when_profile_length_le_lag",
        "calculation_dtype": "float64",
        "output_dtype": "float32",
        "output_dimension": output_dimension,
        "reference_implementation": {
            "source_doi": BIOSEQ_BLM_SOURCE_DOI,
            "repository": BIOSEQ_BLM_REFERENCE_REPOSITORY,
            "commit": BIOSEQ_BLM_REFERENCE_COMMIT,
            "path": BIOSEQ_BLM_REFERENCE_PATH,
            "scope": "method_names_and_feature_ordering_only",
        },
    }
    if normalized_encoder in HYBRID_PSSM_COVARIANCE_ENCODERS:
        contract.update(
            {
                "profile_feature_type": pssm_covariance_profile_feature_type(normalized_encoder),
                "profile_type": ("psi_blast_pssm_or_blosum62_score_fallback"),
                "fallback_policy": "blosum62_hybrid",
                "fallback_condition": PROFILE_HYBRID_FALLBACK_TRIGGER,
            }
        )
    return contract


def validate_pssm_covariance_encoder_contract(
    contract,
    *,
    encoder: str,
    lag: int,
) -> dict:
    if not isinstance(contract, dict):
        raise ValueError("pssm_covariance_encoder_contract must be an object")
    expected = build_pssm_covariance_encoder_contract(
        encoder=encoder,
        lag=lag,
    )
    if contract != expected:
        raise ValueError(
            "pssm_covariance_encoder_contract does not match the supported Dong 2009 v1 contract"
        )
    return expected
