"""Top-n-gram v3 contract and target-frequency profile storage.

The 2008 Top-n-gram representation is defined over PSI-BLAST target
frequencies, not over the integer log-odds scores printed in an ASCII PSSM.
This module keeps that distinction explicit and centralizes validation shared
by profile builders, tokenizers, training, and inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
import warnings
import zipfile
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from pepplatform.sequence_profiles import (
    AA_ORDER,
    FEATURE_TARGET_FREQUENCY,
    NCBISTDAA_CODES,
    PROFILE_SCHEMA,
    ProfileStore,
    SOURCE_BLOSUM62_TARGET_FREQUENCY,
    SOURCE_PSI_BLAST_TARGET_FREQUENCY,
    SequenceProfileError,
    blosum62_target_frequency_profile as _shared_blosum62_target_frequency_profile,
    parse_psiblast_target_frequencies as _shared_parse_target_frequencies,
)

TOP_N_GRAM_CONTRACT_VERSION = 4
TOP_N_GRAM_VARIANT = "psi_blast_target_frequency_2008"
TOP_N_GRAM_ALPHABET = tuple(AA_ORDER)
TOP_N_GRAM_AA_ORDER = "".join(TOP_N_GRAM_ALPHABET)
TOP_N_GRAM_MIN_N = 1
TOP_N_GRAM_MAX_N = 3
TOP_N_GRAM_DEFAULT_DENSE_MEMORY_LIMIT_GIB = 4.0
TOP_N_GRAM_DEFAULT_MAX_RESIDUES = 10_000
TOP_N_GRAM_MAX_RESIDUES_ENV = "PEP_TOP_NGRAM_MAX_RESIDUES"
TOP_N_GRAM_TIE_POLICY = "frequency_desc_then_canonical_aa_order"
TOP_N_GRAM_MAX_LENGTH_UNIT = "raw_residues_hard_limit"
TOP_N_GRAM_PROFILE_FEATURE_TYPE = "target_frequency"
TOP_N_GRAM_PROFILE_MATRIX_KEY = "matrix"
TOP_N_GRAM_ROW_SUM_ATOL = 1e-5
TOP_N_GRAM_STRICT_PROFILE_POLICY = "error"
TOP_N_GRAM_FALLBACK_POLICY = "blosum62_target_frequency"
TOP_N_GRAM_FALLBACK_POLICY_VERSION = "blosum62_target_frequency_v2"
TOP_N_GRAM_PROFILE_SOURCE_PSI_BLAST = SOURCE_PSI_BLAST_TARGET_FREQUENCY
TOP_N_GRAM_PROFILE_SOURCE_FALLBACK = SOURCE_BLOSUM62_TARGET_FREQUENCY


class TopNGramContractError(ValueError):
    """Raised when Top-n-gram configuration or artifacts violate v3."""

    def __init__(self, message: str, *, code: str = "top_ngram_contract_invalid"):
        super().__init__(message)
        self.code = code


def parse_top_ngram_max_residues(value: object) -> int:
    if isinstance(value, bool):
        raise TopNGramContractError(
            "Top-n-gram platform residue limit must be a positive integer",
            code="top_ngram_max_residues_invalid",
        )
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise TopNGramContractError(
            "Top-n-gram platform residue limit must be a positive integer",
            code="top_ngram_max_residues_invalid",
        ) from exc
    if normalized < 1 or str(value).strip() != str(normalized):
        raise TopNGramContractError(
            "Top-n-gram platform residue limit must be a positive integer",
            code="top_ngram_max_residues_invalid",
        )
    return normalized


def configured_top_ngram_max_residues(environ=None) -> int:
    source = os.environ if environ is None else environ
    raw = source.get(
        TOP_N_GRAM_MAX_RESIDUES_ENV,
        str(TOP_N_GRAM_DEFAULT_MAX_RESIDUES),
    )
    return parse_top_ngram_max_residues(raw)


def resolve_top_ngram_max_length(
    configured: object,
    required: object,
    *,
    platform_limit: object | None = None,
    context: str = "Top-n-gram sequence",
) -> tuple[int, bool]:
    if isinstance(configured, bool) or not isinstance(configured, Integral) or int(configured) < 1:
        raise TopNGramContractError(
            f"Top-n-gram max_length must be a positive integer, got {configured!r}",
            code="top_ngram_length_invalid",
        )
    if isinstance(required, bool) or not isinstance(required, Integral) or int(required) < 0:
        raise TopNGramContractError(
            f"Top-n-gram required sequence length is invalid: {required!r}",
            code="top_ngram_length_invalid",
        )
    limit = (
        configured_top_ngram_max_residues()
        if platform_limit is None
        else parse_top_ngram_max_residues(platform_limit)
    )
    requested = int(configured)
    minimum = int(required)
    if requested > limit:
        raise TopNGramContractError(
            f"{context} configured max_length={requested} exceeds the platform "
            f"Top-n-gram limit of {limit} residues",
            code="top_ngram_sequence_limit_exceeded",
        )
    if minimum > limit:
        raise TopNGramContractError(
            f"{context} requires {minimum} residues, exceeding the platform "
            f"Top-n-gram limit of {limit}",
            code="top_ngram_sequence_limit_exceeded",
        )
    effective = max(requested, minimum)
    return effective, effective != requested


class TopNGramProfileQualityWarning(UserWarning):
    """A valid profile was generated without non-self homologous hits."""


def blosum62_target_frequency_profile(sequence: object) -> np.ndarray:
    """Build an explicit substitution-prior profile for unavailable BLAST profiles."""
    try:
        return _shared_blosum62_target_frequency_profile(sequence)
    except SequenceProfileError as exc:
        raise TopNGramContractError(
            str(exc),
            code="top_ngram_sequence_invalid",
        ) from exc


def validate_top_ngram_n(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TopNGramContractError(
            "top_ngram_n must be an integer in "
            f"[{TOP_N_GRAM_MIN_N}, {TOP_N_GRAM_MAX_N}], got {value!r}",
            code="top_ngram_n_invalid",
        )
    normalized = int(value)
    if not TOP_N_GRAM_MIN_N <= normalized <= TOP_N_GRAM_MAX_N:
        raise TopNGramContractError(
            "top_ngram_n must be an integer in "
            f"[{TOP_N_GRAM_MIN_N}, {TOP_N_GRAM_MAX_N}], got {value!r}",
            code="top_ngram_n_invalid",
        )
    return normalized


def parse_top_ngram_n(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TopNGramContractError(
            "top_ngram_n must be an integer in "
            f"[{TOP_N_GRAM_MIN_N}, {TOP_N_GRAM_MAX_N}], got {value!r}",
            code="top_ngram_n_invalid",
        ) from exc
    return validate_top_ngram_n(parsed)


def validate_top_ngram_dense_memory_limit_gib(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TopNGramContractError(
            f"Top-n-gram dense memory limit must be a positive number of GiB, got {value!r}",
            code="top_ngram_memory_limit_invalid",
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise TopNGramContractError(
            f"Top-n-gram dense memory limit must be a positive number of GiB, got {value!r}",
            code="top_ngram_memory_limit_invalid",
        )
    return normalized


def parse_top_ngram_dense_memory_limit_gib(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TopNGramContractError(
            f"Top-n-gram dense memory limit must be a positive number of GiB, got {value!r}",
            code="top_ngram_memory_limit_invalid",
        ) from exc
    return validate_top_ngram_dense_memory_limit_gib(parsed)


def top_ngram_feature_dim(top_n: object) -> int:
    return len(TOP_N_GRAM_ALPHABET) ** validate_top_ngram_n(top_n)


def estimate_top_ngram_dense_bytes(
    sample_count: object,
    top_n: object,
    *,
    element_size: int = 4,
) -> int:
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or int(sample_count) < 0
    ):
        raise TopNGramContractError(
            f"Top-n-gram sample_count must be a non-negative integer, got {sample_count!r}",
            code="top_ngram_memory_estimate_invalid",
        )
    if (
        isinstance(element_size, bool)
        or not isinstance(element_size, Integral)
        or int(element_size) < 1
    ):
        raise TopNGramContractError(
            f"Top-n-gram element_size must be a positive integer, got {element_size!r}",
            code="top_ngram_memory_estimate_invalid",
        )
    return int(sample_count) * top_ngram_feature_dim(top_n) * int(element_size)


def ensure_top_ngram_dense_memory_budget(
    estimated_bytes: object,
    *,
    limit_gib: object = TOP_N_GRAM_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    context: str = "Top-n-gram feature allocation",
) -> int:
    if (
        isinstance(estimated_bytes, bool)
        or not isinstance(estimated_bytes, Integral)
        or int(estimated_bytes) < 0
    ):
        raise TopNGramContractError(
            f"Top-n-gram estimated_bytes must be a non-negative integer, got {estimated_bytes!r}",
            code="top_ngram_memory_estimate_invalid",
        )
    normalized_limit = validate_top_ngram_dense_memory_limit_gib(limit_gib)
    requested = int(estimated_bytes)
    budget = int(normalized_limit * (1024**3))
    if requested > budget:
        requested_gib = requested / float(1024**3)
        raise TopNGramContractError(
            f"{context} is estimated to require {requested_gib:.3f} GiB, "
            f"exceeding topngram_dense_memory_limit_gib={normalized_limit:g}; "
            "reduce the data size or top_ngram_n",
            code="top_ngram_dense_memory_limit_exceeded",
        )
    return requested


def normalize_top_ngram_sequence(
    sequence: object,
    *,
    max_length: object,
    context: str = "Top-n-gram sequence",
    platform_limit: object | None = None,
) -> str:
    if isinstance(max_length, bool) or not isinstance(max_length, Integral) or int(max_length) < 1:
        raise TopNGramContractError(
            f"Top-n-gram max_length must be a positive integer, got {max_length!r}",
            code="top_ngram_length_invalid",
        )
    if platform_limit is None:
        platform_limit = configured_top_ngram_max_residues()
    elif (
        isinstance(platform_limit, bool)
        or not isinstance(platform_limit, Integral)
        or int(platform_limit) < 1
    ):
        raise TopNGramContractError(
            f"Top-n-gram platform limit must be a positive integer, got {platform_limit!r}",
            code="top_ngram_length_invalid",
        )
    platform_limit = int(platform_limit)
    if int(max_length) > platform_limit:
        raise TopNGramContractError(
            f"{context} configured max_length={int(max_length)} exceeds the "
            f"platform Top-n-gram limit of {platform_limit} residues",
            code="top_ngram_sequence_limit_exceeded",
        )
    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise TopNGramContractError(
            f"{context} is empty after removing whitespace",
            code="top_ngram_sequence_invalid",
        )
    invalid = sorted(set(normalized).difference("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    if invalid:
        rendered = ", ".join(repr(character) for character in invalid)
        raise TopNGramContractError(
            f"{context} contains unsupported character(s): {rendered}",
            code="top_ngram_sequence_invalid",
        )
    if len(normalized) > int(max_length):
        raise TopNGramContractError(
            f"{context} contains {len(normalized)} residues, exceeding "
            f"configured max_length={int(max_length)}; Top-n-gram v3 never "
            "silently truncates target-frequency profiles",
            code="top_ngram_sequence_too_long",
        )
    return normalized


def sequence_sha256(sequence: object) -> str:
    normalized = "".join(str(sequence or "").split()).upper()
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TopNGramContractError(
                f"duplicate JSON key in frequency-profile manifest: {key!r}",
                code="top_ngram_manifest_duplicate_key",
            )
        result[key] = value
    return result


def load_json_without_duplicate_keys(path: os.PathLike[str] | str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_json_no_duplicate_keys)
    except TopNGramContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TopNGramContractError(
            f"cannot read frequency-profile manifest {os.fspath(path)!r}: {exc}",
            code="top_ngram_manifest_unreadable",
        ) from exc


def parse_psiblast_target_frequencies(
    checkpoint: os.PathLike[str] | str,
) -> np.ndarray:
    """Recover PSI-BLAST target frequencies from an ASN.1 text checkpoint."""
    try:
        return _shared_parse_target_frequencies(checkpoint)
    except SequenceProfileError as exc:
        code_map = {
            "profile_checkpoint_unreadable": "top_ngram_checkpoint_unreadable",
            "profile_checkpoint_field_missing": "top_ngram_checkpoint_field_missing",
            "profile_checkpoint_shape_invalid": "top_ngram_checkpoint_shape_invalid",
            "profile_checkpoint_value_invalid": "top_ngram_checkpoint_value_invalid",
            "profile_frequency_unavailable": ("top_ngram_checkpoint_target_frequency_unavailable"),
        }
        raise TopNGramContractError(
            str(exc),
            code=code_map.get(exc.code, "top_ngram_checkpoint_invalid"),
        ) from exc


def load_target_frequency_with_fallback(
    checkpoint: os.PathLike[str] | str,
    sequence: object,
    *,
    fallback_policy: str,
) -> tuple[np.ndarray, bool, str]:
    policy = str(fallback_policy or "").strip().lower()
    if policy not in {
        TOP_N_GRAM_STRICT_PROFILE_POLICY,
        TOP_N_GRAM_FALLBACK_POLICY,
    }:
        raise TopNGramContractError(
            f"unsupported Top-n-gram profile fallback policy: {fallback_policy!r}",
            code="top_ngram_fallback_policy_invalid",
        )
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        if policy == TOP_N_GRAM_FALLBACK_POLICY:
            return (
                blosum62_target_frequency_profile(sequence),
                True,
                "checkpoint_output_missing",
            )
        raise TopNGramContractError(
            f"PSI-BLAST completed without producing target-frequency checkpoint "
            f"{os.fspath(checkpoint)!r}",
            code="top_ngram_target_frequency_output_missing",
        )
    try:
        return parse_psiblast_target_frequencies(checkpoint_path), False, ""
    except TopNGramContractError as exc:
        if (
            policy == TOP_N_GRAM_FALLBACK_POLICY
            and exc.code == "top_ngram_checkpoint_target_frequency_unavailable"
        ):
            return (
                blosum62_target_frequency_profile(sequence),
                True,
                exc.code,
            )
        raise


def validate_target_frequency_matrix(
    matrix: object,
    *,
    expected_length: int | None = None,
    context: str = "target-frequency matrix",
) -> np.ndarray:
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[1] != len(TOP_N_GRAM_ALPHABET):
        raise TopNGramContractError(
            f"{context} must have shape [L,20], got {tuple(value.shape)}",
            code="top_ngram_profile_shape_invalid",
        )
    if expected_length is not None and value.shape[0] != int(expected_length):
        raise TopNGramContractError(
            f"{context} has {value.shape[0]} rows; expected {int(expected_length)}",
            code="top_ngram_profile_length_mismatch",
        )
    if value.shape[0] < 1:
        raise TopNGramContractError(
            f"{context} must contain at least one residue row",
            code="top_ngram_profile_shape_invalid",
        )
    numeric = value.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise TopNGramContractError(
            f"{context} contains non-finite values",
            code="top_ngram_profile_value_invalid",
        )
    if (numeric < 0).any():
        raise TopNGramContractError(
            f"{context} contains negative values",
            code="top_ngram_profile_value_invalid",
        )
    row_sums = numeric.sum(axis=1)
    if (row_sums <= 0).any():
        raise TopNGramContractError(
            f"{context} contains zero-sum rows",
            code="top_ngram_profile_value_invalid",
        )
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=TOP_N_GRAM_ROW_SUM_ATOL):
        maximum_error = float(np.max(np.abs(row_sums - 1.0)))
        raise TopNGramContractError(
            f"{context} rows must sum to 1 (maximum error {maximum_error:.3g})",
            code="top_ngram_profile_value_invalid",
        )
    return value.astype(np.float32, copy=False)


def save_target_frequency_matrix(
    path: os.PathLike[str] | str,
    matrix: object,
) -> None:
    validated = validate_target_frequency_matrix(matrix)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            **{TOP_N_GRAM_PROFILE_MATRIX_KEY: validated},
        )
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class TargetFrequencySample:
    sample_id: str
    path: Path
    matrix_sha256: str
    sequence_length: int
    sequence_sha256: str
    zero_nonself_hits: bool
    fallback: bool
    profile_source: str
    fallback_reason: str


class TargetFrequencyProfileStore:
    """Validated reader for a Top-n-gram target-frequency manifest."""

    def __init__(
        self,
        index_path: os.PathLike[str] | str,
        *,
        max_length: object,
        platform_limit: object | None = None,
    ):
        self.index_path = Path(index_path).resolve()
        if not self.index_path.is_file():
            raise TopNGramContractError(
                f"Top-n-gram frequency-profile index not found: {index_path}",
                code="top_ngram_manifest_missing",
            )
        self.max_length = int(max_length) if isinstance(max_length, Integral) else max_length
        # Reuse the strict max-length validation without requiring a sequence.
        if (
            isinstance(self.max_length, bool)
            or not isinstance(self.max_length, Integral)
            or int(self.max_length) < 1
        ):
            raise TopNGramContractError(
                f"Top-n-gram max_length must be a positive integer, got {max_length!r}",
                code="top_ngram_length_invalid",
            )
        self.max_length = int(self.max_length)
        if platform_limit is None:
            platform_limit = configured_top_ngram_max_residues()
        if (
            isinstance(platform_limit, bool)
            or not isinstance(platform_limit, Integral)
            or int(platform_limit) < 1
        ):
            raise TopNGramContractError(
                f"Top-n-gram platform limit must be a positive integer, got {platform_limit!r}",
                code="top_ngram_length_invalid",
            )
        self.platform_limit = int(platform_limit)
        if self.max_length > self.platform_limit:
            raise TopNGramContractError(
                f"Top-n-gram max_length={self.max_length} exceeds the platform "
                f"limit of {self.platform_limit} residues",
                code="top_ngram_sequence_limit_exceeded",
            )
        raw = load_json_without_duplicate_keys(self.index_path)
        if not isinstance(raw, Mapping):
            raise TopNGramContractError(
                "Top-n-gram frequency-profile index must be a JSON object",
                code="top_ngram_manifest_invalid",
            )
        self._shared_store = None
        if raw.get("schema") == PROFILE_SCHEMA:
            try:
                shared_store = ProfileStore(
                    self.index_path,
                    feature_type=FEATURE_TARGET_FREQUENCY,
                )
            except SequenceProfileError as exc:
                raise TopNGramContractError(
                    str(exc),
                    code="top_ngram_legacy_profile_rejected"
                    if exc.code == "profile_manifest_legacy"
                    else "top_ngram_manifest_invalid",
                ) from exc
            samples: dict[str, TargetFrequencySample] = {}
            for sample_id, entry in shared_store.samples.items():
                quality = entry.get("quality")
                if not isinstance(quality, Mapping):
                    raise TopNGramContractError(
                        f"missing profile quality metadata for sample {sample_id!r}",
                        code="top_ngram_manifest_invalid",
                    )
                samples[sample_id] = TargetFrequencySample(
                    sample_id=sample_id,
                    path=(shared_store.root / str(entry.get("path") or "")).resolve(),
                    matrix_sha256=str(entry.get("matrix_sha256") or ""),
                    sequence_length=int(entry.get("sequence_length", 0)),
                    sequence_sha256=str(entry.get("sequence_sha256") or ""),
                    zero_nonself_hits=bool(quality.get("zero_nonself_hits")),
                    fallback=bool(entry.get("fallback")),
                    profile_source=str(entry.get("profile_source") or ""),
                    fallback_reason=str(entry.get("fallback_reason") or ""),
                )
            if not samples:
                raise TopNGramContractError(
                    "Top-n-gram target-frequency manifest must contain samples",
                    code="top_ngram_manifest_invalid",
                )
            fallback_policy = raw.get("provenance", {}).get("fallback_policy")
            self.manifest = {
                **dict(raw),
                "contract_version": TOP_N_GRAM_CONTRACT_VERSION,
                "tie_policy": TOP_N_GRAM_TIE_POLICY,
                "provenance": {
                    **dict(raw.get("provenance") or {}),
                    "fallback_policy": (
                        TOP_N_GRAM_FALLBACK_POLICY_VERSION
                        if fallback_policy == TOP_N_GRAM_FALLBACK_POLICY
                        else TOP_N_GRAM_STRICT_PROFILE_POLICY
                    ),
                },
            }
            self.manifest_sha256 = sha256_file(self.index_path)
            self.samples = samples
            self._shared_store = shared_store
            self._warned_zero_hit_ids = set()
            return
        if raw.get("contract_version") != TOP_N_GRAM_CONTRACT_VERSION:
            raise TopNGramContractError(
                "Top-n-gram requires a target-frequency profile v4 manifest; "
                "legacy PSSM/PSFM indexes are incompatible and must be rebuilt",
                code="top_ngram_legacy_profile_rejected",
            )
        if raw.get("feature_type") != TOP_N_GRAM_PROFILE_FEATURE_TYPE:
            raise TopNGramContractError(
                "Top-n-gram v3 requires feature_type='target_frequency', got "
                f"{raw.get('feature_type')!r}",
                code="top_ngram_profile_type_invalid",
            )
        if raw.get("aa_order") != TOP_N_GRAM_AA_ORDER:
            raise TopNGramContractError(
                f"Top-n-gram v3 frequency-profile aa_order must be {TOP_N_GRAM_AA_ORDER!r}",
                code="top_ngram_profile_aa_order_invalid",
            )
        if raw.get("matrix_layout") != "residue_by_amino_acid":
            raise TopNGramContractError(
                "Top-n-gram v3 frequency-profile matrix_layout must be 'residue_by_amino_acid'",
                code="top_ngram_profile_layout_invalid",
            )
        if raw.get("tie_policy") != TOP_N_GRAM_TIE_POLICY:
            raise TopNGramContractError(
                f"Top-n-gram v3 frequency-profile tie_policy must be {TOP_N_GRAM_TIE_POLICY!r}",
                code="top_ngram_tie_policy_invalid",
            )
        if not isinstance(raw.get("provenance"), Mapping):
            raise TopNGramContractError(
                "Top-n-gram v3 frequency-profile manifest must contain a provenance object",
                code="top_ngram_manifest_invalid",
            )
        fallback_policy = raw["provenance"].get("fallback_policy")
        if fallback_policy not in {
            TOP_N_GRAM_STRICT_PROFILE_POLICY,
            TOP_N_GRAM_FALLBACK_POLICY_VERSION,
        }:
            raise TopNGramContractError(
                "Top-n-gram v3 manifest has an unsupported fallback policy",
                code="top_ngram_manifest_invalid",
            )
        raw_samples = raw.get("samples")
        if not isinstance(raw_samples, Mapping) or not raw_samples:
            raise TopNGramContractError(
                "Top-n-gram v3 frequency-profile manifest must contain samples",
                code="top_ngram_manifest_invalid",
            )
        self.manifest = dict(raw)
        self.manifest_sha256 = sha256_file(self.index_path)
        root = self.index_path.parent.resolve()
        samples: dict[str, TargetFrequencySample] = {}
        for sample_id, entry in raw_samples.items():
            if not isinstance(sample_id, str) or not sample_id:
                raise TopNGramContractError(
                    "Top-n-gram profile sample IDs must be non-empty strings",
                    code="top_ngram_manifest_invalid",
                )
            if not isinstance(entry, Mapping):
                raise TopNGramContractError(
                    f"Top-n-gram profile entry for {sample_id!r} must be an object",
                    code="top_ngram_manifest_invalid",
                )
            relative = entry.get("path")
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                raise TopNGramContractError(
                    f"Top-n-gram profile path for {sample_id!r} must be relative",
                    code="top_ngram_profile_path_invalid",
                )
            resolved = (root / relative).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise TopNGramContractError(
                    f"Top-n-gram profile path for {sample_id!r} escapes the cache root",
                    code="top_ngram_profile_path_invalid",
                ) from exc
            length = entry.get("sequence_length")
            if isinstance(length, bool) or not isinstance(length, Integral) or int(length) < 1:
                raise TopNGramContractError(
                    f"invalid sequence_length for Top-n-gram sample {sample_id!r}",
                    code="top_ngram_manifest_invalid",
                )
            digest = entry.get("sequence_sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise TopNGramContractError(
                    f"invalid sequence_sha256 for Top-n-gram sample {sample_id!r}",
                    code="top_ngram_manifest_invalid",
                )
            fallback = entry.get("fallback")
            if not isinstance(fallback, bool):
                raise TopNGramContractError(
                    f"fallback must be boolean for sample {sample_id!r}",
                    code="top_ngram_manifest_invalid",
                )
            profile_source = entry.get("profile_source")
            if profile_source not in {
                TOP_N_GRAM_PROFILE_SOURCE_PSI_BLAST,
                TOP_N_GRAM_PROFILE_SOURCE_FALLBACK,
            }:
                raise TopNGramContractError(
                    f"invalid profile_source for Top-n-gram sample {sample_id!r}",
                    code="top_ngram_manifest_invalid",
                )
            expected_fallback = profile_source == TOP_N_GRAM_PROFILE_SOURCE_FALLBACK
            if fallback is not expected_fallback:
                raise TopNGramContractError(
                    f"fallback and profile_source disagree for sample {sample_id!r}",
                    code="top_ngram_manifest_invalid",
                )
            fallback_reason = entry.get("fallback_reason", "")
            if not isinstance(fallback_reason, str) or (fallback and not fallback_reason.strip()):
                raise TopNGramContractError(
                    f"invalid fallback_reason for Top-n-gram sample {sample_id!r}",
                    code="top_ngram_manifest_invalid",
                )
            if fallback and fallback_policy != TOP_N_GRAM_FALLBACK_POLICY_VERSION:
                raise TopNGramContractError(
                    f"Top-n-gram fallback profile for sample {sample_id!r} "
                    "is not authorized by the manifest policy",
                    code="top_ngram_fallback_profile_rejected",
                )
            matrix_digest = entry.get("matrix_sha256")
            if (
                not isinstance(matrix_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", matrix_digest) is None
            ):
                raise TopNGramContractError(
                    f"invalid matrix_sha256 for Top-n-gram sample {sample_id!r}",
                    code="top_ngram_manifest_invalid",
                )
            zero_hits = entry.get("zero_nonself_hits")
            if not isinstance(zero_hits, bool):
                raise TopNGramContractError(
                    f"zero_nonself_hits must be boolean for sample {sample_id!r}",
                    code="top_ngram_manifest_invalid",
                )
            samples[sample_id] = TargetFrequencySample(
                sample_id=sample_id,
                path=resolved,
                matrix_sha256=matrix_digest,
                sequence_length=int(length),
                sequence_sha256=digest,
                zero_nonself_hits=zero_hits,
                fallback=fallback,
                profile_source=profile_source,
                fallback_reason=fallback_reason.strip(),
            )
        self.samples = samples
        self._warned_zero_hit_ids: set[str] = set()

    def sample_metadata(self, sample_id: object) -> TargetFrequencySample:
        if not isinstance(sample_id, str) or not sample_id:
            raise TopNGramContractError(
                "Top-n-gram v3 requires a non-empty sample_id",
                code="top_ngram_sample_id_missing",
            )
        sample = self.samples.get(sample_id)
        if sample is None:
            raise TopNGramContractError(
                f"target-frequency profile missing for sample_id={sample_id!r}",
                code="top_ngram_profile_missing",
            )
        return sample

    def load(self, sample_id: object, sequence: object) -> torch.Tensor:
        sample = self.sample_metadata(sample_id)
        normalized = normalize_top_ngram_sequence(
            sequence,
            max_length=self.max_length,
            context=f"Top-n-gram sequence for sample_id={sample.sample_id!r}",
            platform_limit=self.platform_limit,
        )
        if len(normalized) != sample.sequence_length:
            raise TopNGramContractError(
                f"Top-n-gram sequence length for sample_id={sample.sample_id!r} "
                f"is {len(normalized)}; profile expects {sample.sequence_length}",
                code="top_ngram_profile_length_mismatch",
            )
        digest = sequence_sha256(normalized)
        if digest != sample.sequence_sha256:
            raise TopNGramContractError(
                f"Top-n-gram sequence digest does not match profile for "
                f"sample_id={sample.sample_id!r}",
                code="top_ngram_profile_sequence_mismatch",
            )
        if self._shared_store is not None:
            try:
                matrix = self._shared_store.load(
                    sample.sample_id,
                    normalized,
                ).matrix
            except SequenceProfileError as exc:
                raise TopNGramContractError(
                    str(exc),
                    code="top_ngram_profile_unreadable",
                ) from exc
            if sample.fallback and sample.sample_id not in self._warned_zero_hit_ids:
                warnings.warn(
                    f"Top-n-gram profile for sample_id={sample.sample_id!r} uses "
                    "the explicit BLOSUM62 target-frequency prior because "
                    "PSI-BLAST did not provide a usable profile "
                    f"({sample.fallback_reason})",
                    TopNGramProfileQualityWarning,
                    stacklevel=2,
                )
                self._warned_zero_hit_ids.add(sample.sample_id)
            if sample.zero_nonself_hits and sample.sample_id not in self._warned_zero_hit_ids:
                warnings.warn(
                    f"Top-n-gram profile for sample_id={sample.sample_id!r} has "
                    "no non-self PSI-BLAST hits; the valid target-frequency "
                    "profile is retained but carries weaker evolutionary evidence",
                    TopNGramProfileQualityWarning,
                    stacklevel=2,
                )
                self._warned_zero_hit_ids.add(sample.sample_id)
            return matrix
        if not sample.path.is_file():
            raise TopNGramContractError(
                f"target-frequency matrix missing for sample_id={sample.sample_id!r}: "
                f"{sample.path}",
                code="top_ngram_profile_missing",
            )
        actual_matrix_digest = sha256_file(sample.path)
        if actual_matrix_digest != sample.matrix_sha256:
            raise TopNGramContractError(
                f"target-frequency matrix digest does not match manifest for "
                f"sample_id={sample.sample_id!r}",
                code="top_ngram_profile_digest_mismatch",
            )
        try:
            with zipfile.ZipFile(sample.path, "r") as archive_file:
                members = archive_file.infolist()
                expected_member = f"{TOP_N_GRAM_PROFILE_MATRIX_KEY}.npy"
                if len(members) != 1 or members[0].filename != expected_member:
                    raise TopNGramContractError(
                        f"target-frequency archive for "
                        f"sample_id={sample.sample_id!r} must contain exactly "
                        f"{expected_member!r}",
                        code="top_ngram_profile_archive_invalid",
                    )
                maximum_member_bytes = (
                    sample.sequence_length
                    * len(TOP_N_GRAM_ALPHABET)
                    * np.dtype(np.float32).itemsize
                    + 65536
                )
                if members[0].file_size > maximum_member_bytes:
                    raise TopNGramContractError(
                        f"target-frequency archive for "
                        f"sample_id={sample.sample_id!r} exceeds the bounded "
                        "float32 matrix size",
                        code="top_ngram_profile_archive_invalid",
                    )
            with np.load(sample.path, allow_pickle=False) as archive:
                if set(archive.files) != {TOP_N_GRAM_PROFILE_MATRIX_KEY}:
                    raise TopNGramContractError(
                        f"target-frequency archive for sample_id={sample.sample_id!r} "
                        f"must contain only {TOP_N_GRAM_PROFILE_MATRIX_KEY!r}",
                        code="top_ngram_profile_archive_invalid",
                    )
                raw_matrix = archive[TOP_N_GRAM_PROFILE_MATRIX_KEY]
                if raw_matrix.dtype != np.dtype(np.float32):
                    raise TopNGramContractError(
                        f"target-frequency matrix for "
                        f"sample_id={sample.sample_id!r} must use float32, "
                        f"got {raw_matrix.dtype}",
                        code="top_ngram_profile_archive_invalid",
                    )
                matrix = np.array(raw_matrix, copy=True)
        except TopNGramContractError:
            raise
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise TopNGramContractError(
                f"cannot read target-frequency matrix for sample_id={sample.sample_id!r}: {exc}",
                code="top_ngram_profile_unreadable",
            ) from exc
        validated = validate_target_frequency_matrix(
            matrix,
            expected_length=sample.sequence_length,
            context=f"target-frequency matrix for sample_id={sample.sample_id!r}",
        )
        if sample.fallback and sample.sample_id not in self._warned_zero_hit_ids:
            warnings.warn(
                f"Top-n-gram profile for sample_id={sample.sample_id!r} uses "
                "the explicit BLOSUM62 target-frequency prior because PSI-BLAST "
                f"did not provide a usable profile ({sample.fallback_reason})",
                TopNGramProfileQualityWarning,
                stacklevel=2,
            )
            self._warned_zero_hit_ids.add(sample.sample_id)
        if sample.zero_nonself_hits and sample.sample_id not in self._warned_zero_hit_ids:
            warnings.warn(
                f"Top-n-gram profile for sample_id={sample.sample_id!r} has no "
                "non-self PSI-BLAST hits; the valid target-frequency profile "
                "is retained but carries weaker evolutionary evidence",
                TopNGramProfileQualityWarning,
                stacklevel=2,
            )
            self._warned_zero_hit_ids.add(sample.sample_id)
        return torch.from_numpy(validated.copy())


def stable_top_residue_indices(matrix: object, top_n: object) -> np.ndarray:
    n = validate_top_ngram_n(top_n)
    validated = validate_target_frequency_matrix(matrix)
    # Stable sorting preserves the canonical column order for exact ties.
    return np.argsort(-validated, axis=1, kind="stable")[:, :n]


def top_residue_ids_to_feature_ids(indices: object) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 2 or values.shape[1] < 1:
        raise TopNGramContractError(
            f"Top-n-gram residue indices must have shape [L,n], got {tuple(values.shape)}",
            code="top_ngram_token_shape_invalid",
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise TopNGramContractError(
            "Top-n-gram residue indices must be integers",
            code="top_ngram_token_value_invalid",
        )
    if (values < 0).any() or (values >= len(TOP_N_GRAM_ALPHABET)).any():
        raise TopNGramContractError(
            "Top-n-gram residue indices are outside the canonical alphabet",
            code="top_ngram_token_value_invalid",
        )
    feature_ids = np.zeros(values.shape[0], dtype=np.int64)
    for column in range(values.shape[1]):
        feature_ids = feature_ids * len(TOP_N_GRAM_ALPHABET) + values[:, column]
    return feature_ids
