"""Strict contract for the platform's Distance-based Residue representation."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Iterable, Iterator


DR_CONTRACT_VERSION = 2
DR_VARIANT = "distance_based_residue_2014_plus_bioseq_blm_extensions"
DR_ALPHABET = tuple("ARDCQEHIGNLKMFPSTWYV")
DR_DEFAULT_MAX_DISTANCE = 5
DR_MAX_DISTANCE = 200
DR_DEFAULT_MAX_RAW_LENGTH = 50_000
DR_ABSOLUTE_MAX_RAW_LENGTH = 100_000
DR_DEFAULT_DENSE_MEMORY_LIMIT_GIB = 4.0
DR_DEFAULT_UNKNOWN_RESIDUE_POLICY = "error"
DR_UNKNOWN_RESIDUE_POLICIES = ("error", "zero")
DR_ZERO_RESIDUE = "X"
DR_TOKEN_ORDER = "source_position_then_distance"
DR_FEATURE_ORDER = "distance_0_then_ordered_pairs_by_distance"
DR_GLOBAL_MAX_LENGTH_UNIT = "not_applicable_full_normalized_sequence"
DR_ORDERED_MAX_LENGTH_UNIT = "dr_tokens"


class DRContractError(ValueError):
    """Raised when a DR parameter or sequence violates the v2 contract."""

    def __init__(self, message: str, *, code: str = "dr_contract_invalid"):
        super().__init__(message)
        self.code = code


def validate_dr_max_distance(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DRContractError(
            f"DR max_distance must be an integer in [0, {DR_MAX_DISTANCE}], got {value!r}",
            code="dr_distance_invalid",
        )
    normalized = int(value)
    if not 0 <= normalized <= DR_MAX_DISTANCE:
        raise DRContractError(
            f"DR max_distance must be an integer in [0, {DR_MAX_DISTANCE}], got {value!r}",
            code="dr_distance_invalid",
        )
    return normalized


def validate_dr_max_raw_length(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DRContractError(
            "DR max_raw_length must be an integer in "
            f"[1, {DR_ABSOLUTE_MAX_RAW_LENGTH}], got {value!r}",
            code="dr_length_invalid",
        )
    normalized = int(value)
    if not 1 <= normalized <= DR_ABSOLUTE_MAX_RAW_LENGTH:
        raise DRContractError(
            "DR max_raw_length must be an integer in "
            f"[1, {DR_ABSOLUTE_MAX_RAW_LENGTH}], got {value!r}",
            code="dr_length_invalid",
        )
    return normalized


def validate_dr_dense_memory_limit_gib(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DRContractError(
            f"DR dense memory limit must be a positive number of GiB, got {value!r}",
            code="dr_memory_limit_invalid",
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise DRContractError(
            f"DR dense memory limit must be a positive number of GiB, got {value!r}",
            code="dr_memory_limit_invalid",
        )
    return normalized


def parse_dr_max_distance(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DRContractError(
            f"DR max_distance must be an integer in [0, {DR_MAX_DISTANCE}], got {value!r}",
            code="dr_distance_invalid",
        ) from exc
    return validate_dr_max_distance(parsed)


def parse_dr_max_raw_length(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DRContractError(
            "DR max_raw_length must be an integer in "
            f"[1, {DR_ABSOLUTE_MAX_RAW_LENGTH}], got {value!r}",
            code="dr_length_invalid",
        ) from exc
    return validate_dr_max_raw_length(parsed)


def parse_dr_dense_memory_limit_gib(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DRContractError(
            f"DR dense memory limit must be a positive number of GiB, got {value!r}",
            code="dr_memory_limit_invalid",
        ) from exc
    return validate_dr_dense_memory_limit_gib(parsed)


def validate_dr_unknown_residue_policy(value: object) -> str:
    normalized = str(value or DR_DEFAULT_UNKNOWN_RESIDUE_POLICY).strip().lower()
    if normalized not in DR_UNKNOWN_RESIDUE_POLICIES:
        raise DRContractError(
            "DR unknown-residue policy must be one of "
            f"{DR_UNKNOWN_RESIDUE_POLICIES}, got {value!r}",
            code="dr_unknown_residue_policy_invalid",
        )
    return normalized


def normalize_dr_sequence(
    sequence: object,
    *,
    max_raw_length: int = DR_DEFAULT_MAX_RAW_LENGTH,
    unknown_residue_policy: str = DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
    context: str = "DR sequence",
) -> str:
    limit = validate_dr_max_raw_length(max_raw_length)
    policy = validate_dr_unknown_residue_policy(unknown_residue_policy)
    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise DRContractError(
            f"{context} is empty after removing whitespace",
            code="dr_sequence_invalid",
        )
    accepted = set(DR_ALPHABET)
    if policy == "zero":
        accepted.add(DR_ZERO_RESIDUE)
        # Under the zero policy, IUPAC ambiguity codes (B=Asx, J=Xle,
        # Z=Glx, U=Sec, O=Pyl) are treated as unknown residues, matching
        # the existing DR_ZERO_RESIDUE ("X") semantics: monomer and pair
        # contributions involving the residue are zeroed.  The "error"
        # policy (default) keeps rejecting them, and the feature alphabet /
        # dimension (20 + 20*20*max_distance) is unchanged.
        normalized = normalized.translate(str.maketrans("BJZUO", DR_ZERO_RESIDUE * 5))
    invalid = sorted(set(normalized).difference(accepted))
    if invalid:
        rendered = ", ".join(repr(character) for character in invalid)
        raise DRContractError(
            f"{context} contains unsupported character(s): {rendered}; "
            f"allowed alphabet is {''.join(DR_ALPHABET)}",
            code="dr_sequence_invalid",
        )
    if len(normalized) > limit:
        raise DRContractError(
            f"{context} contains {len(normalized)} residues, exceeding configured "
            f"dr_max_raw_length={limit}",
            code="dr_sequence_too_long",
        )
    return normalized


def dr_feature_dim(max_distance: object) -> int:
    distance = validate_dr_max_distance(max_distance)
    alphabet_size = len(DR_ALPHABET)
    return alphabet_size + alphabet_size * alphabet_size * distance


def dr_feature_tokens(max_distance: object) -> tuple[str, ...]:
    distance = validate_dr_max_distance(max_distance)
    tokens = list(DR_ALPHABET)
    for offset in range(1, distance + 1):
        for first in DR_ALPHABET:
            for second in DR_ALPHABET:
                tokens.append(f"{first}_{offset}_{second}")
    return tuple(tokens)


def iter_dr_tokens(
    normalized_sequence: str,
    max_distance: object,
    *,
    unknown_residue_policy: str = DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
) -> Iterator[str]:
    """Yield DR words in source-position-major order.

    Distance-zero residue words are followed by all valid ordered pairs
    starting at that source position.
    """

    distance = validate_dr_max_distance(max_distance)
    policy = validate_dr_unknown_residue_policy(unknown_residue_policy)
    length = len(normalized_sequence)
    for index, first in enumerate(normalized_sequence):
        if first != DR_ZERO_RESIDUE or policy != "zero":
            yield first
        upper = min(distance, length - index - 1)
        for offset in range(1, upper + 1):
            second = normalized_sequence[index + offset]
            if policy == "zero" and DR_ZERO_RESIDUE in {first, second}:
                continue
            yield f"{first}_{offset}_{second}"


def iter_normalized_dr_tokens(
    sequence: object,
    *,
    max_distance: object = DR_DEFAULT_MAX_DISTANCE,
    max_raw_length: object = DR_DEFAULT_MAX_RAW_LENGTH,
    unknown_residue_policy: str = DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
    context: str = "DR sequence",
) -> Iterator[str]:
    normalized = normalize_dr_sequence(
        sequence,
        max_raw_length=validate_dr_max_raw_length(max_raw_length),
        unknown_residue_policy=unknown_residue_policy,
        context=context,
    )
    return iter_dr_tokens(
        normalized,
        max_distance,
        unknown_residue_policy=unknown_residue_policy,
    )


def normalize_dr_sequences(
    sequences: Iterable[object],
    *,
    max_raw_length: int = DR_DEFAULT_MAX_RAW_LENGTH,
    unknown_residue_policy: str = DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
    context: str = "DR training sequence",
) -> list[str]:
    normalized = [
        normalize_dr_sequence(
            sequence,
            max_raw_length=max_raw_length,
            unknown_residue_policy=unknown_residue_policy,
            context=f"{context} #{index}",
        )
        for index, sequence in enumerate(sequences, start=1)
    ]
    if not normalized:
        raise DRContractError(
            "DR training corpus is empty",
            code="dr_sequence_invalid",
        )
    return normalized


def dr_contract_metadata(
    *,
    max_distance: object,
    max_raw_length: object,
    unknown_residue_policy: str = DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
) -> dict[str, object]:
    distance = validate_dr_max_distance(max_distance)
    raw_limit = validate_dr_max_raw_length(max_raw_length)
    policy = validate_dr_unknown_residue_policy(unknown_residue_policy)
    metadata = {
        "contract_version": DR_CONTRACT_VERSION,
        "variant": DR_VARIANT,
        "alphabet": "".join(DR_ALPHABET),
        "max_distance": distance,
        "max_raw_length": raw_limit,
        "token_order": DR_TOKEN_ORDER,
        "feature_order": DR_FEATURE_ORDER,
        "feature_dim": dr_feature_dim(distance),
        "distance_zero": "single_residue_occurrence",
        "pair_direction": "left_to_right",
        "bow_normalization": "none",
    }
    if policy != DR_DEFAULT_UNKNOWN_RESIDUE_POLICY:
        metadata.update(
            {
                "unknown_residue_policy": policy,
                "unknown_residue": DR_ZERO_RESIDUE,
                "unknown_contribution": "zero_for_monomer_and_pairs_with_unknown_endpoint",
                "distance_coordinates": "original_normalized_sequence_positions",
            }
        )
    return metadata


def dr_dense_budget_bytes(limit_gib: object) -> int:
    return int(validate_dr_dense_memory_limit_gib(limit_gib) * (1024**3))


def estimate_dr_dense_tensor_bytes(
    sample_count: object,
    *,
    encoder: str,
    max_distance: object,
    max_length: object,
    embedding_dim: object | None = None,
) -> int:
    """Estimate one materialized dense DR feature tensor.

    The estimate follows the actual platform dtypes: index tensors are int64
    and all other DR encodings are float32.
    """

    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or int(sample_count) < 0
    ):
        raise DRContractError(
            f"DR sample_count must be a non-negative integer, got {sample_count!r}",
            code="dr_memory_estimate_invalid",
        )
    if isinstance(max_length, bool) or not isinstance(max_length, Integral) or int(max_length) < 1:
        raise DRContractError(
            f"DR max_length must be a positive integer, got {max_length!r}",
            code="dr_memory_estimate_invalid",
        )
    normalized_encoder = str(encoder or "").strip().lower()
    feature_dim = dr_feature_dim(max_distance)
    if normalized_encoder == "index":
        elements_per_sample = int(max_length)
        element_size = 8
    elif normalized_encoder == "onehot":
        elements_per_sample = int(max_length) * (feature_dim + 2)
        element_size = 4
    elif normalized_encoder in {"bow", "tfidf", "textrank"}:
        elements_per_sample = feature_dim
        element_size = 4
    elif normalized_encoder in {"word2vec", "glove", "fasttext"}:
        if (
            isinstance(embedding_dim, bool)
            or not isinstance(embedding_dim, Integral)
            or int(embedding_dim) < 1
        ):
            raise DRContractError(
                "DR static-embedding memory estimation requires a positive "
                f"integer embedding_dim, got {embedding_dim!r}",
                code="dr_memory_estimate_invalid",
            )
        elements_per_sample = int(max_length) * int(embedding_dim)
        element_size = 4
    else:
        raise DRContractError(
            f"unsupported DR encoder for dense memory estimation: {encoder!r}",
            code="dr_memory_estimate_invalid",
        )
    return int(sample_count) * elements_per_sample * element_size


def ensure_dr_dense_memory_budget(
    estimated_bytes: object,
    *,
    limit_gib: object = DR_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    context: str = "DR feature allocation",
) -> int:
    if (
        isinstance(estimated_bytes, bool)
        or not isinstance(estimated_bytes, Integral)
        or int(estimated_bytes) < 0
    ):
        raise DRContractError(
            f"DR estimated_bytes must be a non-negative integer, got {estimated_bytes!r}",
            code="dr_memory_estimate_invalid",
        )
    budget = dr_dense_budget_bytes(limit_gib)
    requested = int(estimated_bytes)
    if requested > budget:
        requested_gib = requested / float(1024**3)
        raise DRContractError(
            f"{context} is estimated to require {requested_gib:.3f} GiB, "
            f"exceeding dr_dense_memory_limit_gib="
            f"{validate_dr_dense_memory_limit_gib(limit_gib):g}; "
            "reduce data size, max_length, max_distance, or use a compact "
            "encoder",
            code="dr_dense_memory_limit_exceeded",
        )
    return requested
