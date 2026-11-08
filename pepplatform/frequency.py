"""Shared contract helpers for global BoW and TF-IDF encoders."""

from __future__ import annotations

import argparse
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Iterable

import torch


FREQUENCY_ENCODER_CONTRACT_VERSION = 2
FREQUENCY_LEGACY_CONTRACT_VERSION = 1
FREQUENCY_DEFAULT_MAX_RAW_LENGTH = 50_000
FREQUENCY_MAX_RAW_LENGTH = 100_000
FREQUENCY_TF_FORMULA = "c_j / sum_k(c_k)"
FREQUENCY_IDF_FORMULA = "ln((N + 1) / (df_j + 1)) + 1"


def validate_frequency_contract_version(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("frequency_contract_version must be integer 1 or 2")
    value = int(value)
    if value not in {
        FREQUENCY_LEGACY_CONTRACT_VERSION,
        FREQUENCY_ENCODER_CONTRACT_VERSION,
    }:
        raise ValueError("frequency_contract_version must be 1 or 2")
    return value


def validate_frequency_max_raw_length(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("frequency_max_raw_length must be a positive integer")
    value = int(value)
    if value < 1 or value > FREQUENCY_MAX_RAW_LENGTH:
        raise ValueError(
            f"frequency_max_raw_length must be in [1,{FREQUENCY_MAX_RAW_LENGTH}], got {value}"
        )
    return value


def parse_frequency_max_raw_length(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("frequency_max_raw_length must be an integer") from exc
    try:
        return validate_frequency_max_raw_length(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def normalize_frequency_sequence(sequence, *, max_raw_length: int) -> str:
    if not isinstance(sequence, str):
        raise ValueError(
            f"frequency encoders require sequence strings, got {type(sequence).__name__}"
        )
    limit = validate_frequency_max_raw_length(max_raw_length)
    if len(sequence) > limit:
        raise ValueError(
            f"sequence length {len(sequence)} exceeds frequency_max_raw_length={limit}"
        )
    return sequence.upper()


def _strict_feature_id(value, *, feature_dim: int, context: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or value.dtype == torch.bool:
            raise ValueError(f"{context} must contain scalar integer feature ids")
        if value.dtype.is_floating_point or value.dtype.is_complex:
            raise ValueError(f"{context} must contain integer feature ids")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{context} must contain integer feature ids, got {value!r}")
    index = int(value)
    if index < 0 or index >= feature_dim:
        raise ValueError(f"{context} feature id {index} is outside [0,{feature_dim})")
    return index


def count_feature_ids(
    ids: Iterable,
    *,
    feature_dim: int,
    padding_id: int | None = None,
    context: str = "frequency encoder",
) -> torch.Tensor:
    if (
        isinstance(feature_dim, bool)
        or not isinstance(feature_dim, Integral)
        or int(feature_dim) < 1
    ):
        raise ValueError("feature_dim must be a positive integer")
    feature_dim = int(feature_dim)
    if padding_id is not None:
        padding_id = _strict_feature_id(
            padding_id,
            feature_dim=feature_dim,
            context=f"{context} padding",
        )
    counts = torch.zeros(feature_dim, dtype=torch.float)
    for raw_id in ids:
        index = _strict_feature_id(
            raw_id,
            feature_dim=feature_dim,
            context=context,
        )
        if index != padding_id:
            counts[index] += 1.0
    return counts


def validate_idf_vector(idf, *, feature_dim: int, context: str) -> torch.Tensor:
    if idf is None:
        raise ValueError(f"{context} requires a fitted or persisted IDF vector")
    try:
        tensor = torch.as_tensor(idf, dtype=torch.float).detach().clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{context} IDF vector is not numeric") from exc
    if tensor.ndim != 1 or tensor.shape[0] != int(feature_dim):
        raise ValueError(
            f"{context} IDF vector width must match feature dimension "
            f"{int(feature_dim)}, got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all() or (tensor <= 0).any():
        raise ValueError(f"{context} IDF vector must contain finite positive values")
    return tensor


def fit_smoothed_idf(
    documents: Iterable[Iterable],
    *,
    feature_dim: int,
    padding_id: int | None = None,
    context: str = "TF-IDF",
) -> tuple[torch.Tensor, int]:
    df = torch.zeros(int(feature_dim), dtype=torch.float)
    document_count = 0
    for ids in documents:
        counts = count_feature_ids(
            ids,
            feature_dim=feature_dim,
            padding_id=padding_id,
            context=f"{context} training document",
        )
        df[counts > 0] += 1.0
        document_count += 1
    if document_count < 1:
        raise ValueError(f"{context} training corpus is empty")
    return torch.log((document_count + 1) / (df + 1)) + 1.0, document_count


def vocabulary_sha256(tokenizer) -> str:
    if hasattr(tokenizer, "kmer_vocab"):
        payload = [
            token
            for token, _index in sorted(
                tokenizer.kmer_vocab.items(),
                key=lambda item: item[1],
            )
        ]
    elif hasattr(tokenizer, "feature_tokens"):
        payload = list(tokenizer.feature_tokens)
    else:
        payload = {
            "type": type(tokenizer).__name__,
            "feature_dim": int(tokenizer.feature_dim),
            "top_n": getattr(tokenizer, "top_n", None),
            "max_distance": getattr(tokenizer, "max_distance", None),
        }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_frequency_encoder_contract(
    *,
    tokenizer,
    encoder: str,
    contract_version: int,
    max_raw_length: int,
    training_document_count: int | None = None,
    corpus_sha256: str | None = None,
    fit_scope: str | None = None,
) -> dict:
    contract_version = validate_frequency_contract_version(contract_version)
    normalized_encoder = str(encoder).strip().lower()
    if normalized_encoder in {"tf-idf", "tf_idf"}:
        normalized_encoder = "tfidf"
    raw_feature_dim = getattr(tokenizer, "feature_dim", None)
    if raw_feature_dim is None:
        raw_feature_dim = getattr(tokenizer, "vocab_size")
    feature_dim = int(raw_feature_dim)
    is_v2 = contract_version == FREQUENCY_ENCODER_CONTRACT_VERSION
    metadata = {
        "contract_version": contract_version,
        "stream_policy": "full_token_stream" if is_v2 else "legacy_token_stream",
        "bow_formula": "x_j = c_j" if is_v2 else "legacy_tokenizer_specific",
        "tf_formula": FREQUENCY_TF_FORMULA,
        "idf_formula": FREQUENCY_IDF_FORMULA,
        "post_normalization": "none" if is_v2 else "legacy_tokenizer_specific",
        "frequency_max_raw_length": validate_frequency_max_raw_length(max_raw_length),
        "feature_dimension": feature_dim,
        "vocabulary_sha256": vocabulary_sha256(tokenizer),
        "encoder": normalized_encoder,
        "training_document_count": (
            int(training_document_count)
            if training_document_count is not None
            else (0 if normalized_encoder == "bow" else None)
        ),
        "training_corpus_sha256": (str(corpus_sha256) if corpus_sha256 is not None else None),
        "fit_scope": (
            str(fit_scope)
            if fit_scope is not None
            else ("not_applicable" if normalized_encoder == "bow" else None)
        ),
    }
    return metadata


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_idf_state(
    idf,
    *,
    artifact_dir,
    filename: str,
) -> dict:
    tensor = torch.as_tensor(idf, dtype=torch.float).detach().cpu().clone()
    if tensor.ndim != 1 or tensor.numel() < 1:
        raise ValueError(f"TF-IDF state must be a nonempty vector, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all() or (tensor <= 0).any():
        raise ValueError("TF-IDF state must contain finite positive values")
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_root = artifact_root.resolve()
    path = (artifact_root / filename).resolve()
    try:
        path.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError("TF-IDF state filename must remain inside artifact_dir") from exc
    torch.save(tensor, path)
    return {
        "path": path.relative_to(artifact_root).as_posix(),
        "sha256": sha256_file(path),
        "dtype": "float32",
        "shape": [int(tensor.numel())],
    }
