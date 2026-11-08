"""Scientific and artifact contracts for trainable static token embeddings."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from numbers import Integral, Real


STATIC_EMBEDDING_CONTRACT_VERSION = 2
STATIC_EMBEDDING_DEFAULT_MEMORY_LIMIT_GIB = 4.0
FASTTEXT_DEFAULT_BUCKET = 2_000_000
FASTTEXT_MIN_N = 3
FASTTEXT_MAX_N = 6

WORD2VEC_ALGORITHM = "word2vec_skipgram_negative_sampling"
FASTTEXT_ALGORITHM = "fasttext_subword_skipgram_negative_sampling"
GENSIM_IMPLEMENTATION = "gensim"
GENSIM_IMPLEMENTATION_VERSION = 1


def stable_token_hash(value: object) -> int:
    """Return a process-independent hash for gensim vector initialization."""
    digest = hashlib.blake2b(
        str(value).encode("utf-8"),
        digest_size=8,
        person=b"pep-w2v",
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def gensim_version() -> str:
    try:
        return importlib.metadata.version("gensim")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def validate_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def validate_seed(value: object) -> int:
    normalized = validate_nonnegative_int(value, name="seed")
    if normalized > 2**32 - 1:
        raise ValueError(f"seed must be <= {2**32 - 1}, got {value!r}")
    return normalized


def validate_positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return normalized


def validate_static_embedding_memory_limit_gib(value: object) -> float:
    return validate_positive_finite(value, name="static_embedding_memory_limit_gib")


def estimate_fasttext_training_bytes(
    *,
    bucket: object,
    vector_size: object,
    vocab_size: object,
) -> int:
    """Conservatively estimate dense fastText parameter memory.

    Gensim allocates the complete n-gram bucket matrix plus combined, vocabulary,
    and negative-sampling word matrices. A 25% margin covers temporary buffers
    and allocator overhead without pretending to estimate Python object memory.
    """
    bucket_count = validate_positive_int(bucket, name="fasttext_bucket")
    width = validate_positive_int(vector_size, name="embedding_vector_size")
    vocabulary = validate_positive_int(vocab_size, name="fastText vocab_size")
    dense_float_count = bucket_count * width + 3 * vocabulary * width
    return math.ceil(dense_float_count * 4 * 1.25)


def ensure_fasttext_memory_budget(
    *,
    bucket: object,
    vector_size: object,
    vocab_size: object,
    limit_gib: object = STATIC_EMBEDDING_DEFAULT_MEMORY_LIMIT_GIB,
) -> int:
    estimated = estimate_fasttext_training_bytes(
        bucket=bucket,
        vector_size=vector_size,
        vocab_size=vocab_size,
    )
    limit = validate_static_embedding_memory_limit_gib(limit_gib)
    budget = int(limit * 1024**3)
    if estimated > budget:
        requested = estimated / float(1024**3)
        raise ValueError(
            "fastText training is estimated to require "
            f"{requested:.3f} GiB, exceeding static_embedding_memory_limit_gib="
            f"{limit:g}; reduce --fasttext_bucket or --embedding_vector_size, "
            "or explicitly raise the memory limit"
        )
    return estimated


def word2vec_cache_signature(*, workers: int) -> dict[str, object]:
    worker_count = validate_positive_int(workers, name="embedding_workers")
    return {
        "contract_version": STATIC_EMBEDDING_CONTRACT_VERSION,
        "algorithm": WORD2VEC_ALGORITHM,
        "implementation": GENSIM_IMPLEMENTATION,
        "implementation_version": GENSIM_IMPLEMENTATION_VERSION,
        "gensim_version": gensim_version(),
        "architecture": "skipgram",
        "objective": "negative_sampling",
        "hierarchical_softmax": False,
        "negative_samples": 5,
        "negative_sampling_exponent": 0.75,
        "frequent_token_subsampling": 0.001,
        "initial_learning_rate": 0.025,
        "minimum_learning_rate": 0.0001,
        "dynamic_window": True,
        "initializer_hash": "blake2b-64-pep-w2v",
        "workers": worker_count,
        "deterministic": worker_count == 1,
        "artifact_format": "word2vec_text",
        "oov_policy": "provided_unk_else_null_vector",
    }


def fasttext_cache_signature(*, workers: int, bucket: int) -> dict[str, object]:
    worker_count = validate_positive_int(workers, name="embedding_workers")
    bucket_count = validate_positive_int(bucket, name="fasttext_bucket")
    return {
        "contract_version": STATIC_EMBEDDING_CONTRACT_VERSION,
        "algorithm": FASTTEXT_ALGORITHM,
        "implementation": GENSIM_IMPLEMENTATION,
        "implementation_version": GENSIM_IMPLEMENTATION_VERSION,
        "gensim_version": gensim_version(),
        "architecture": "skipgram",
        "objective": "negative_sampling",
        "hierarchical_softmax": False,
        "negative_samples": 5,
        "negative_sampling_exponent": 0.5,
        "frequent_token_subsampling": 0.0001,
        "initial_learning_rate": 0.05,
        "minimum_learning_rate": 0.0,
        "dynamic_window": True,
        "min_n": FASTTEXT_MIN_N,
        "max_n": FASTTEXT_MAX_N,
        "bucket": bucket_count,
        "ngram_hash": "fnv1a",
        "initializer_hash": "blake2b-64-pep-w2v",
        "workers": worker_count,
        "deterministic": worker_count == 1,
        "artifact_format": "facebook_fasttext_binary",
        "oov_policy": "character_ngram_synthesis",
    }
