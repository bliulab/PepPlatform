"""Contracted latent semantic analysis for classic global sequence features."""

from __future__ import annotations
from pepplatform.task_compat import normalize_task_type


import hashlib
import json
import math
import warnings
from typing import Any

import numpy as np
import sklearn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from pepplatform.truncated_svd import (
    TRUNCATED_SVD_PARAMS,
    new_truncated_svd,
    truncated_svd_params,
)


LSA_CONTRACT_VERSION = 1
LSA_VARIANT = "train_only_uncentered_randomized_truncated_svd"
LSA_SUPPORTED_ENCODERS = frozenset({"bow", "tfidf", "textrank"})
LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB = 4.0
LSA_FIT_SCOPE = "train_only"
LSA_PRIMITIVE_PARAMS = dict(TRUNCATED_SVD_PARAMS)
_PAIR_TASK_TYPES = frozenset({"pep_protein_interaction", "pep_hlaii", "pep_hlai"})


class LSAContractError(ValueError):
    """Raised when an LSA route, fit, transform, or artifact violates contract v1."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def normalize_lsa_encoder(encoder: object) -> str:
    value = str(encoder or "").strip().lower()
    return "tfidf" if value in {"tf-idf", "tf_idf"} else value


def validate_lsa_dense_memory_limit_gib(value: object) -> float:
    try:
        limit = float(value)
    except (TypeError, ValueError) as exc:
        raise LSAContractError(
            "lsa_memory_limit_invalid",
            "lsa_dense_memory_limit_gib must be a positive finite number",
        ) from exc
    if not math.isfinite(limit) or limit <= 0.0:
        raise LSAContractError(
            "lsa_memory_limit_invalid",
            "lsa_dense_memory_limit_gib must be a positive finite number",
        )
    return limit


def validate_lsa_dimension(value: object) -> int:
    if isinstance(value, bool):
        raise LSAContractError("lsa_dimension_invalid", "lsa_dim must be a positive integer")
    try:
        dimension = int(value)
    except (TypeError, ValueError) as exc:
        raise LSAContractError(
            "lsa_dimension_invalid", "lsa_dim must be a positive integer"
        ) from exc
    if dimension <= 0 or str(value).strip() not in {str(dimension), f"{dimension}.0"}:
        raise LSAContractError("lsa_dimension_invalid", "lsa_dim must be a positive integer")
    return dimension


def validate_lsa_route(source: object, *, pair: bool | None = None) -> None:
    def read(key: str, default: object = "") -> object:
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    reduction = str(read("reduction", "none") or "none").strip().lower()
    if reduction != "lsa":
        return
    task_type = normalize_task_type(read("task_type", "") or "")
    is_pair = normalize_task_type(task_type) in _PAIR_TASK_TYPES if pair is None else bool(pair)
    if is_pair:
        raise LSAContractError(
            "lsa_pair_route_unsupported",
            "reduction=lsa is not supported for Pair workflows; use ml_reduction=tsvd after Pair feature fusion",
        )
    encoder = normalize_lsa_encoder(read("encoder", ""))
    if encoder not in LSA_SUPPORTED_ENCODERS:
        raise LSAContractError(
            "lsa_encoder_unsupported",
            "reduction=lsa requires a classic global encoder in {bow, tfidf, textrank}",
            details={"encoder": encoder},
        )
    validate_lsa_dimension(read("lsa_dim", 128))
    validate_lsa_dense_memory_limit_gib(
        read("lsa_dense_memory_limit_gib", LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB)
    )


def canonical_contract_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _matrix_shape(matrix: object, *, split: str) -> tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise LSAContractError(
            "lsa_matrix_shape_invalid",
            f"LSA {split} matrix must be two-dimensional",
            details={"split": split, "shape": list(shape or [])},
        )
    rows, columns = int(shape[0]), int(shape[1])
    if rows <= 0 or columns <= 0:
        raise LSAContractError(
            "lsa_matrix_empty",
            f"LSA {split} matrix must contain at least one row and one feature",
            details={"split": split, "shape": [rows, columns]},
        )
    return rows, columns


def _matrix_dtype(matrix: object) -> np.dtype:
    try:
        return np.dtype(getattr(matrix, "dtype", np.float64))
    except TypeError:
        return np.dtype(np.float64)


def _validate_matrix(
    matrix: object,
    *,
    split: str,
    expected_features: int | None = None,
    dense_memory_limit_gib: float,
) -> tuple[int, int]:
    rows, columns = _matrix_shape(matrix, split=split)
    if expected_features is not None and columns != expected_features:
        raise LSAContractError(
            "lsa_feature_width_mismatch",
            f"LSA {split} matrix has {columns} features; expected {expected_features}",
            details={
                "split": split,
                "expected_features": expected_features,
                "actual_features": columns,
            },
        )
    dtype = _matrix_dtype(matrix)
    if not np.issubdtype(dtype, np.number):
        raise LSAContractError(
            "lsa_matrix_dtype_invalid",
            f"LSA {split} matrix must be numeric",
            details={"split": split, "dtype": str(dtype)},
        )
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    try:
        finite = bool(np.isfinite(values).all())
    except TypeError as exc:
        raise LSAContractError(
            "lsa_matrix_dtype_invalid",
            f"LSA {split} matrix must be numeric",
            details={"split": split, "dtype": str(dtype)},
        ) from exc
    if not finite:
        raise LSAContractError(
            "lsa_matrix_nonfinite",
            f"LSA {split} matrix contains NaN or infinite values",
            details={"split": split},
        )
    dense_bytes = rows * columns * max(1, dtype.itemsize)
    dense_gib = dense_bytes / float(1024**3)
    if dense_gib > dense_memory_limit_gib:
        raise LSAContractError(
            "lsa_dense_memory_limit_exceeded",
            f"LSA {split} dense matrix estimate is {dense_gib:.3f} GiB, exceeding lsa_dense_memory_limit_gib={dense_memory_limit_gib:g}",
            details={
                "split": split,
                "estimated_dense_gib": dense_gib,
                "limit_gib": dense_memory_limit_gib,
            },
        )
    return rows, columns


def _validate_requested_rank(n_components: int, rows: int, columns: int) -> None:
    if n_components > rows:
        raise LSAContractError(
            "lsa_dimension_exceeds_samples",
            f"lsa_dim={n_components} exceeds the {rows} training samples",
            details={"lsa_dim": n_components, "train_sample_count": rows},
        )
    if n_components >= columns:
        raise LSAContractError(
            "lsa_dimension_not_below_features",
            f"lsa_dim={n_components} must be smaller than the {columns} input features",
            details={"lsa_dim": n_components, "input_feature_count": columns},
        )


def lsa_primitive_params(seed: int) -> dict[str, Any]:
    return truncated_svd_params(seed)


def _new_estimator(n_components: int, seed: int) -> TruncatedSVD:
    return new_truncated_svd(n_components, seed)


def _diagnostics(estimator: TruncatedSVD, n_components: int) -> dict[str, Any]:
    singular_values = np.asarray(estimator.singular_values_, dtype=np.float64)
    explained_variance = np.asarray(estimator.explained_variance_, dtype=np.float64)
    explained_ratio = np.asarray(estimator.explained_variance_ratio_, dtype=np.float64)
    for name, values in {
        "singular_values": singular_values,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_ratio,
    }.items():
        if values.shape != (n_components,) or not np.isfinite(values).all():
            raise LSAContractError(
                "lsa_diagnostics_invalid",
                f"LSA produced invalid {name} diagnostics",
                details={"diagnostic": name, "shape": list(values.shape)},
            )
    tolerance = (
        np.finfo(singular_values.dtype).eps
        * max(estimator.components_.shape)
        * float(singular_values[0] if singular_values.size else 0.0)
    )
    effective_rank = int(np.count_nonzero(singular_values > tolerance))
    if effective_rank < n_components:
        raise LSAContractError(
            "lsa_requested_rank_degenerate",
            f"training matrix rank {effective_rank} is below requested lsa_dim={n_components}",
            details={"effective_rank": effective_rank, "lsa_dim": n_components},
        )
    return {
        "effective_rank": effective_rank,
        "rank_tolerance": float(tolerance),
        "singular_values": singular_values.tolist(),
        "explained_variance": explained_variance.tolist(),
        "explained_variance_ratio": explained_ratio.tolist(),
        "explained_variance_ratio_sum": float(explained_ratio.sum()),
    }


def build_lsa_contract(
    estimator: TruncatedSVD,
    *,
    encoder: str,
    train_sample_count: int,
    input_feature_count: int,
    n_components: int,
    seed: int,
    dense_memory_limit_gib: float,
    upstream_feature_contract: object,
) -> dict[str, Any]:
    diagnostics = _diagnostics(estimator, n_components)
    upstream_digest = canonical_contract_digest(upstream_feature_contract)
    return {
        "contract_version": LSA_CONTRACT_VERSION,
        "algorithm": "latent_semantic_analysis",
        "public_name": "LSA",
        "variant": LSA_VARIANT,
        "status": "verified_platform_variant",
        "route": "single_sequence_classic_global",
        "supported_encoder": encoder,
        "input_orientation": "sequence_by_term",
        "input_weighting": encoder,
        "centering": False,
        "decomposition": "truncated_svd",
        "implementation": "sklearn.decomposition.TruncatedSVD",
        "implementation_version": sklearn.__version__,
        "feature_fit_scope": LSA_FIT_SCOPE,
        "held_out_semantics": "frozen_linear_transform",
        "train_sample_count": int(train_sample_count),
        "input_feature_count": int(input_feature_count),
        "output_dimension": int(n_components),
        "seed": int(seed),
        "estimator_params": {
            "n_components": int(n_components),
            **lsa_primitive_params(seed),
        },
        "dense_memory_limit_gib": float(dense_memory_limit_gib),
        "upstream_feature_contract_sha256": upstream_digest,
        "diagnostics": diagnostics,
        "serialization": {
            "format": "joblib",
            "requires_matching_sklearn_version": True,
        },
    }


def validate_lsa_contract(
    contract: object,
    *,
    estimator: object | None = None,
    require_current_sklearn: bool = False,
) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise LSAContractError("lsa_contract_missing", "LSA artifact contract is missing")
    params = contract.get("estimator_params")
    diagnostics = contract.get("diagnostics")
    serialization = contract.get("serialization")
    dimension = contract.get("output_dimension")
    input_features = contract.get("input_feature_count")
    train_samples = contract.get("train_sample_count")
    expected_params = (
        {"n_components": dimension, **lsa_primitive_params(contract.get("seed", -1))}
        if isinstance(contract.get("seed"), int) and not isinstance(contract.get("seed"), bool)
        else None
    )
    digest = contract.get("upstream_feature_contract_sha256")
    valid_digest = (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )
    valid_diagnostics = isinstance(diagnostics, dict) and all(
        isinstance(diagnostics.get(key), list)
        and len(diagnostics[key]) == dimension
        and all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in diagnostics[key]
        )
        for key in ("singular_values", "explained_variance", "explained_variance_ratio")
    )
    if (
        contract.get("contract_version") != LSA_CONTRACT_VERSION
        or contract.get("algorithm") != "latent_semantic_analysis"
        or contract.get("public_name") != "LSA"
        or contract.get("variant") != LSA_VARIANT
        or contract.get("status") != "verified_platform_variant"
        or contract.get("route") != "single_sequence_classic_global"
        or contract.get("supported_encoder") not in LSA_SUPPORTED_ENCODERS
        or contract.get("input_orientation") != "sequence_by_term"
        or contract.get("centering") is not False
        or contract.get("decomposition") != "truncated_svd"
        or contract.get("implementation") != "sklearn.decomposition.TruncatedSVD"
        or contract.get("feature_fit_scope") != LSA_FIT_SCOPE
        or contract.get("held_out_semantics") != "frozen_linear_transform"
        or not isinstance(train_samples, int)
        or isinstance(train_samples, bool)
        or train_samples < 1
        or not isinstance(input_features, int)
        or isinstance(input_features, bool)
        or input_features < 2
        or not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 1
        or dimension > train_samples
        or dimension >= input_features
        or params != expected_params
        or not valid_digest
        or not valid_diagnostics
        or diagnostics.get("effective_rank", 0) < dimension
        or not isinstance(serialization, dict)
        or serialization.get("format") != "joblib"
        or serialization.get("requires_matching_sklearn_version") is not True
    ):
        raise LSAContractError(
            "lsa_contract_invalid", "LSA artifact contract is incomplete or inconsistent"
        )
    if require_current_sklearn and contract.get("implementation_version") != sklearn.__version__:
        raise LSAContractError(
            "lsa_runtime_version_mismatch",
            "LSA artifact was trained with a different scikit-learn version",
            details={
                "trained_version": contract.get("implementation_version"),
                "runtime_version": sklearn.__version__,
            },
        )
    if estimator is not None:
        if not isinstance(estimator, TruncatedSVD):
            raise LSAContractError(
                "lsa_reducer_type_invalid",
                "LSA artifact reducer must be sklearn.decomposition.TruncatedSVD",
            )
        actual = estimator.get_params(deep=False)
        if any(actual.get(key) != value for key, value in expected_params.items()):
            raise LSAContractError(
                "lsa_reducer_contract_mismatch",
                "LSA reducer parameters do not match the artifact contract",
            )
        if np.asarray(estimator.components_).shape != (dimension, input_features):
            raise LSAContractError(
                "lsa_reducer_contract_mismatch",
                "LSA reducer component shape does not match the artifact contract",
            )
        _diagnostics(estimator, dimension)
    return contract


def transform_lsa_held_out(
    estimator: TruncatedSVD,
    matrix: object,
    *,
    split: str,
    input_feature_count: int,
    output_dimension: int,
    dense_memory_limit_gib: float,
) -> np.ndarray:
    rows, _ = _validate_matrix(
        matrix,
        split=split,
        expected_features=input_feature_count,
        dense_memory_limit_gib=dense_memory_limit_gib,
    )
    try:
        transformed = estimator.transform(matrix)
    except Exception as exc:
        raise LSAContractError(
            "lsa_transform_failed",
            f"LSA failed to transform the {split} matrix: {exc}",
            details={"split": split},
        ) from exc
    values = np.asarray(transformed)
    if values.shape != (rows, output_dimension) or not np.isfinite(values).all():
        raise LSAContractError(
            "lsa_transform_output_invalid",
            f"LSA produced invalid {split} output",
            details={"split": split, "shape": list(values.shape)},
        )
    return values


def fit_lsa_transform(
    train_matrix: object,
    validation_matrix: object,
    test_matrix: object | None = None,
    *,
    encoder: object,
    n_components: object,
    seed: int,
    dense_memory_limit_gib: object = LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    upstream_feature_contract: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, TruncatedSVD, dict[str, Any]]:
    normalized_encoder = normalize_lsa_encoder(encoder)
    if normalized_encoder not in LSA_SUPPORTED_ENCODERS:
        raise LSAContractError(
            "lsa_encoder_unsupported",
            "LSA requires a classic global encoder in {bow, tfidf, textrank}",
            details={"encoder": normalized_encoder},
        )
    dimension = validate_lsa_dimension(n_components)
    limit = validate_lsa_dense_memory_limit_gib(dense_memory_limit_gib)
    train_rows, input_features = _validate_matrix(
        train_matrix,
        split="train",
        dense_memory_limit_gib=limit,
    )
    _validate_requested_rank(dimension, train_rows, input_features)
    _validate_matrix(
        validation_matrix,
        split="validation",
        expected_features=input_features,
        dense_memory_limit_gib=limit,
    )
    if test_matrix is not None:
        _validate_matrix(
            test_matrix,
            split="test",
            expected_features=input_features,
            dense_memory_limit_gib=limit,
        )
    estimator = _new_estimator(dimension, int(seed))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            train_output = estimator.fit_transform(train_matrix)
        except Exception as exc:
            raise LSAContractError("lsa_fit_failed", f"LSA fitting failed: {exc}") from exc
    if caught:
        raise LSAContractError(
            "lsa_fit_warning",
            "LSA fitting emitted a numerical warning and the fitted state was rejected",
            details={"warnings": [str(item.message) for item in caught]},
        )
    train_values = np.asarray(train_output)
    if train_values.shape != (train_rows, dimension) or not np.isfinite(train_values).all():
        raise LSAContractError("lsa_fit_output_invalid", "LSA produced invalid training output")
    upstream = (
        upstream_feature_contract
        if upstream_feature_contract is not None
        else {
            "encoder": normalized_encoder,
            "input_feature_count": input_features,
        }
    )
    contract = build_lsa_contract(
        estimator,
        encoder=normalized_encoder,
        train_sample_count=train_rows,
        input_feature_count=input_features,
        n_components=dimension,
        seed=int(seed),
        dense_memory_limit_gib=limit,
        upstream_feature_contract=upstream,
    )
    validation_output = transform_lsa_held_out(
        estimator,
        validation_matrix,
        split="validation",
        input_feature_count=input_features,
        output_dimension=dimension,
        dense_memory_limit_gib=limit,
    )
    test_output = None
    if test_matrix is not None:
        test_output = transform_lsa_held_out(
            estimator,
            test_matrix,
            split="test",
            input_feature_count=input_features,
            output_dimension=dimension,
            dense_memory_limit_gib=limit,
        )
    validate_lsa_contract(contract, estimator=estimator)
    return train_values, validation_output, test_output, estimator, contract
