"""Strict contract for the platform's Distance-based Top-n-gram representation."""

from __future__ import annotations

import math
import re
from numbers import Integral, Real

from pepplatform.top_ngram import (
    TOP_N_GRAM_AA_ORDER,
    TOP_N_GRAM_CONTRACT_VERSION,
    TOP_N_GRAM_PROFILE_FEATURE_TYPE,
    TOP_N_GRAM_TIE_POLICY,
)


DT_CONTRACT_VERSION = 1
DT_VARIANT = "distance_based_top_1_gram_2014_plus_platform_extensions"
DT_ALPHABET = tuple("ARDCQEHIGNLKMFPSTWYV")
DT_DEFAULT_MAX_DISTANCE = 5
DT_MAX_DISTANCE = 200
DT_DEFAULT_DENSE_MEMORY_LIMIT_GIB = 4.0
DT_TOKEN_ORDER = "source_position_then_distance"
DT_FEATURE_ORDER = "distance_0_then_ordered_pairs_by_distance"
DT_MAX_LENGTH_UNIT = "raw_residues_hard_limit"

_DT_INDEX = {residue: index for index, residue in enumerate(DT_ALPHABET)}
_DT_PAIR_TOKEN = re.compile(r"^([A-Z])_([0-9]+)_([A-Z])$")


class DTContractError(ValueError):
    """Raised when DT configuration or artifacts violate the v1 contract."""

    def __init__(self, message: str, *, code: str = "dt_contract_invalid"):
        super().__init__(message)
        self.code = code


def validate_dt_max_distance(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DTContractError(
            f"DT max_distance must be an integer in [0, {DT_MAX_DISTANCE}], got {value!r}",
            code="dt_distance_invalid",
        )
    normalized = int(value)
    if not 0 <= normalized <= DT_MAX_DISTANCE:
        raise DTContractError(
            f"DT max_distance must be an integer in [0, {DT_MAX_DISTANCE}], got {value!r}",
            code="dt_distance_invalid",
        )
    return normalized


def parse_dt_max_distance(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DTContractError(
            f"DT max_distance must be an integer in [0, {DT_MAX_DISTANCE}], got {value!r}",
            code="dt_distance_invalid",
        ) from exc
    return validate_dt_max_distance(parsed)


def validate_dt_dense_memory_limit_gib(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DTContractError(
            f"DT dense memory limit must be a positive number of GiB, got {value!r}",
            code="dt_memory_limit_invalid",
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise DTContractError(
            f"DT dense memory limit must be a positive number of GiB, got {value!r}",
            code="dt_memory_limit_invalid",
        )
    return normalized


def parse_dt_dense_memory_limit_gib(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DTContractError(
            f"DT dense memory limit must be a positive number of GiB, got {value!r}",
            code="dt_memory_limit_invalid",
        ) from exc
    return validate_dt_dense_memory_limit_gib(parsed)


def dt_feature_dim(max_distance: object) -> int:
    distance = validate_dt_max_distance(max_distance)
    alphabet_size = len(DT_ALPHABET)
    return alphabet_size + alphabet_size * alphabet_size * distance


def dt_feature_index(token: object, *, max_distance: object) -> int:
    distance_limit = validate_dt_max_distance(max_distance)
    if isinstance(token, str) and len(token) == 1 and token in _DT_INDEX:
        return _DT_INDEX[token]
    if not isinstance(token, str):
        raise DTContractError(
            f"invalid DT token {token!r}",
            code="dt_token_invalid",
        )
    match = _DT_PAIR_TOKEN.fullmatch(token)
    if match is None:
        raise DTContractError(
            f"invalid DT token {token!r}",
            code="dt_token_invalid",
        )
    first, raw_distance, second = match.groups()
    distance = int(raw_distance)
    if first not in _DT_INDEX or second not in _DT_INDEX or not 1 <= distance <= distance_limit:
        raise DTContractError(
            f"invalid DT token {token!r}",
            code="dt_token_invalid",
        )
    alphabet_size = len(DT_ALPHABET)
    return (
        alphabet_size
        + (distance - 1) * alphabet_size * alphabet_size
        + _DT_INDEX[first] * alphabet_size
        + _DT_INDEX[second]
    )


def dt_contract_metadata(*, max_distance: object) -> dict[str, object]:
    distance = validate_dt_max_distance(max_distance)
    return {
        "contract_version": DT_CONTRACT_VERSION,
        "variant": DT_VARIANT,
        "top_n": 1,
        "profile_contract_version": TOP_N_GRAM_CONTRACT_VERSION,
        "profile_type": TOP_N_GRAM_PROFILE_FEATURE_TYPE,
        "profile_aa_order": TOP_N_GRAM_AA_ORDER,
        "tie_policy": TOP_N_GRAM_TIE_POLICY,
        "alphabet": "".join(DT_ALPHABET),
        "max_distance": distance,
        "token_order": DT_TOKEN_ORDER,
        "feature_order": DT_FEATURE_ORDER,
        "feature_dim": dt_feature_dim(distance),
        "distance_zero": "top_1_gram_occurrence",
        "pair_direction": "left_to_right",
        "bow_normalization": "none",
        "full_length": True,
    }


def estimate_dt_dense_bytes(
    sample_count: object,
    *,
    max_distance: object,
    element_size: int = 4,
) -> int:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or int(sample_count) < 0
    ):
        raise DTContractError(
            f"DT sample_count must be a non-negative integer, got {sample_count!r}",
            code="dt_memory_estimate_invalid",
        )
    if (
        isinstance(element_size, bool)
        or not isinstance(element_size, Integral)
        or int(element_size) < 1
    ):
        raise DTContractError(
            f"DT element_size must be a positive integer, got {element_size!r}",
            code="dt_memory_estimate_invalid",
        )
    return int(sample_count) * dt_feature_dim(max_distance) * int(element_size)


def ensure_dt_dense_memory_budget(
    estimated_bytes: object,
    *,
    limit_gib: object = DT_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    context: str = "DT feature allocation",
) -> int:
    if (
        isinstance(estimated_bytes, bool)
        or not isinstance(estimated_bytes, Integral)
        or int(estimated_bytes) < 0
    ):
        raise DTContractError(
            f"DT estimated_bytes must be a non-negative integer, got {estimated_bytes!r}",
            code="dt_memory_estimate_invalid",
        )
    normalized_limit = validate_dt_dense_memory_limit_gib(limit_gib)
    requested = int(estimated_bytes)
    budget = int(normalized_limit * (1024**3))
    if requested > budget:
        requested_gib = requested / float(1024**3)
        raise DTContractError(
            f"{context} is estimated to require {requested_gib:.3f} GiB, "
            f"exceeding dt_dense_memory_limit_gib={normalized_limit:g}; "
            "reduce the data size or dt_max_distance",
            code="dt_dense_memory_limit_exceeded",
        )
    return requested
