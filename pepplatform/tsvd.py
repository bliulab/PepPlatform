"""Contracted generic TruncatedSVD transformation for classical ML routes."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import sklearn
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from pepplatform.truncated_svd import new_truncated_svd, truncated_svd_params


TSVD_CONTRACT_VERSION = 1
TSVD_VARIANT = "train_only_uncentered_randomized_truncated_svd"
TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB = 4.0
TSVD_FIT_SCOPE = "train_only"


class TSVDContractError(ValueError):
    """Raised when a generic TSVD route or artifact violates contract v1."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def validate_tsvd_dense_memory_limit_gib(value: object) -> float:
    try:
        limit = float(value)
    except (TypeError, ValueError) as exc:
        raise TSVDContractError(
            "tsvd_memory_limit_invalid",
            "tsvd_dense_memory_limit_gib must be a positive finite number",
        ) from exc
    if not math.isfinite(limit) or limit <= 0.0:
        raise TSVDContractError(
            "tsvd_memory_limit_invalid",
            "tsvd_dense_memory_limit_gib must be a positive finite number",
        )
    return limit


def validate_tsvd_dimension(value: object) -> int:
    if isinstance(value, bool):
        raise TSVDContractError(
            "tsvd_dimension_invalid", "ml_reduction_dim must be a positive integer"
        )
    try:
        dimension = int(value)
    except (TypeError, ValueError) as exc:
        raise TSVDContractError(
            "tsvd_dimension_invalid", "ml_reduction_dim must be a positive integer"
        ) from exc
    if dimension <= 0 or str(value).strip() not in {str(dimension), f"{dimension}.0"}:
        raise TSVDContractError(
            "tsvd_dimension_invalid", "ml_reduction_dim must be a positive integer"
        )
    return dimension


def _validate_seed(value: object) -> int:
    if isinstance(value, bool):
        raise TSVDContractError("tsvd_seed_invalid", "TSVD seed must be a non-negative integer")
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise TSVDContractError(
            "tsvd_seed_invalid", "TSVD seed must be a non-negative integer"
        ) from exc
    if seed < 0 or str(value).strip() not in {str(seed), f"{seed}.0"}:
        raise TSVDContractError("tsvd_seed_invalid", "TSVD seed must be a non-negative integer")
    return seed


def _matrix_shape(matrix: object, *, split: str) -> tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TSVDContractError(
            "tsvd_matrix_shape_invalid",
            f"TSVD {split} matrix must be two-dimensional",
            details={"split": split, "shape": list(shape or [])},
        )
    rows, columns = int(shape[0]), int(shape[1])
    if rows <= 0 or columns <= 0:
        raise TSVDContractError(
            "tsvd_matrix_empty",
            f"TSVD {split} matrix must contain at least one row and one feature",
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
        raise TSVDContractError(
            "tsvd_feature_width_mismatch",
            f"TSVD {split} matrix has {columns} features; expected {expected_features}",
            details={
                "split": split,
                "expected_features": expected_features,
                "actual_features": columns,
            },
        )
    dtype = _matrix_dtype(matrix)
    if not np.issubdtype(dtype, np.number):
        raise TSVDContractError(
            "tsvd_matrix_dtype_invalid",
            f"TSVD {split} matrix must be numeric",
            details={"split": split, "dtype": str(dtype)},
        )
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    try:
        finite = bool(np.isfinite(values).all())
    except TypeError as exc:
        raise TSVDContractError(
            "tsvd_matrix_dtype_invalid",
            f"TSVD {split} matrix must be numeric",
            details={"split": split, "dtype": str(dtype)},
        ) from exc
    if not finite:
        raise TSVDContractError(
            "tsvd_matrix_nonfinite",
            f"TSVD {split} matrix contains NaN or infinite values",
            details={"split": split},
        )
    dense_bytes = rows * columns * max(1, dtype.itemsize)
    dense_gib = dense_bytes / float(1024**3)
    if dense_gib > dense_memory_limit_gib:
        raise TSVDContractError(
            "tsvd_dense_memory_limit_exceeded",
            f"TSVD {split} dense matrix estimate is {dense_gib:.3f} GiB, exceeding "
            f"tsvd_dense_memory_limit_gib={dense_memory_limit_gib:g}",
            details={
                "split": split,
                "estimated_dense_gib": dense_gib,
                "limit_gib": dense_memory_limit_gib,
            },
        )
    return rows, columns


def _validate_requested_rank(n_components: int, rows: int, columns: int) -> None:
    if n_components > rows:
        raise TSVDContractError(
            "tsvd_dimension_exceeds_samples",
            f"ml_reduction_dim={n_components} exceeds the {rows} training samples",
            details={"ml_reduction_dim": n_components, "train_sample_count": rows},
        )
    if n_components > columns:
        raise TSVDContractError(
            "tsvd_dimension_exceeds_features",
            f"ml_reduction_dim={n_components} exceeds the {columns} input features",
            details={"ml_reduction_dim": n_components, "input_feature_count": columns},
        )


def _diagnostics(estimator: TruncatedSVD, n_components: int) -> dict[str, Any]:
    components = np.asarray(estimator.components_)
    if components.shape != (
        n_components,
        int(getattr(estimator, "n_features_in_", components.shape[-1])),
    ):
        raise TSVDContractError(
            "tsvd_components_invalid",
            "TSVD component shape is inconsistent with the fitted estimator",
        )
    if not np.isfinite(components).all():
        raise TSVDContractError(
            "tsvd_components_invalid", "TSVD components contain non-finite values"
        )
    diagnostics = {}
    for name in ("singular_values_", "explained_variance_", "explained_variance_ratio_"):
        values = np.asarray(getattr(estimator, name, []), dtype=np.float64)
        if values.shape != (n_components,) or not np.isfinite(values).all():
            raise TSVDContractError(
                "tsvd_diagnostics_invalid",
                f"TSVD produced invalid {name[:-1]} diagnostics",
                details={"diagnostic": name, "shape": list(values.shape)},
            )
        diagnostics[name[:-1]] = values.tolist()
    singular_values = np.asarray(estimator.singular_values_, dtype=np.float64)
    tolerance = (
        np.finfo(singular_values.dtype).eps
        * max(components.shape)
        * float(singular_values[0] if singular_values.size else 0.0)
    )
    diagnostics["effective_rank"] = int(np.count_nonzero(singular_values > tolerance))
    diagnostics["rank_tolerance"] = float(tolerance)
    diagnostics["explained_variance_ratio_sum"] = float(
        np.asarray(estimator.explained_variance_ratio_, dtype=np.float64).sum()
    )
    return diagnostics


def build_tsvd_contract(
    estimator: TruncatedSVD,
    *,
    train_sample_count: int,
    input_feature_count: int,
    n_components: int,
    seed: int,
    dense_memory_limit_gib: float,
    input_dtype: object,
    input_format: str,
) -> dict[str, Any]:
    return {
        "contract_version": TSVD_CONTRACT_VERSION,
        "algorithm": "truncated_svd",
        "public_name": "TSVD",
        "variant": TSVD_VARIANT,
        "status": "verified_platform_variant",
        "route": "generic_numeric_reduction",
        "input_orientation": "samples_by_features",
        "input_semantics": "generic_numeric",
        "input_dtype": str(np.dtype(input_dtype)),
        "input_format": str(input_format),
        "centering": False,
        "decomposition": "truncated_svd",
        "implementation": "sklearn.decomposition.TruncatedSVD",
        "implementation_version": sklearn.__version__,
        "feature_fit_scope": TSVD_FIT_SCOPE,
        "held_out_semantics": "frozen_linear_transform",
        "train_sample_count": int(train_sample_count),
        "input_feature_count": int(input_feature_count),
        "output_dimension": int(n_components),
        "seed": int(seed),
        "estimator_params": {
            "n_components": int(n_components),
            **truncated_svd_params(seed),
        },
        "dense_memory_limit_gib": float(dense_memory_limit_gib),
        "diagnostics": _diagnostics(estimator, n_components),
        "serialization": {
            "format": "joblib",
            "requires_matching_sklearn_version": True,
        },
    }


def validate_tsvd_contract(
    contract: object,
    *,
    estimator: object | None = None,
    require_current_sklearn: bool = False,
) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise TSVDContractError("tsvd_contract_missing", "TSVD artifact contract is missing")
    dimension = contract.get("output_dimension")
    input_features = contract.get("input_feature_count")
    train_samples = contract.get("train_sample_count")
    seed = contract.get("seed")
    expected_params = (
        {"n_components": dimension, **truncated_svd_params(seed)}
        if isinstance(seed, int) and not isinstance(seed, bool)
        else None
    )
    diagnostics = contract.get("diagnostics")
    diagnostic_names = ("singular_values", "explained_variance", "explained_variance_ratio")
    valid_diagnostics = (
        isinstance(diagnostics, dict)
        and isinstance(diagnostics.get("effective_rank"), int)
        and not isinstance(diagnostics.get("effective_rank"), bool)
        and diagnostics.get("effective_rank") >= 0
        and all(
            isinstance(diagnostics.get(key), list)
            and isinstance(dimension, int)
            and len(diagnostics[key]) == dimension
            and all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in diagnostics[key]
            )
            for key in diagnostic_names
        )
    )
    try:
        limit = validate_tsvd_dense_memory_limit_gib(contract.get("dense_memory_limit_gib"))
    except TSVDContractError:
        limit = None
    valid_dimensions = (
        isinstance(train_samples, int)
        and not isinstance(train_samples, bool)
        and train_samples >= 1
        and isinstance(input_features, int)
        and not isinstance(input_features, bool)
        and input_features >= 1
        and isinstance(dimension, int)
        and not isinstance(dimension, bool)
        and 1 <= dimension <= min(train_samples, input_features)
    )
    if (
        contract.get("contract_version") != TSVD_CONTRACT_VERSION
        or contract.get("algorithm") != "truncated_svd"
        or contract.get("public_name") != "TSVD"
        or contract.get("variant") != TSVD_VARIANT
        or contract.get("status") != "verified_platform_variant"
        or contract.get("route") != "generic_numeric_reduction"
        or contract.get("input_orientation") != "samples_by_features"
        or contract.get("input_semantics") != "generic_numeric"
        or not isinstance(contract.get("input_dtype"), str)
        or contract.get("input_format") not in {"dense", "sparse"}
        or not isinstance(contract.get("implementation_version"), str)
        or contract.get("centering") is not False
        or contract.get("decomposition") != "truncated_svd"
        or contract.get("implementation") != "sklearn.decomposition.TruncatedSVD"
        or contract.get("feature_fit_scope") != TSVD_FIT_SCOPE
        or contract.get("held_out_semantics") != "frozen_linear_transform"
        or not valid_dimensions
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or expected_params is None
        or contract.get("estimator_params") != expected_params
        or limit is None
        or not valid_diagnostics
        or not isinstance(contract.get("serialization"), dict)
        or contract["serialization"].get("format") != "joblib"
        or contract["serialization"].get("requires_matching_sklearn_version") is not True
    ):
        raise TSVDContractError(
            "tsvd_contract_invalid", "TSVD artifact contract is incomplete or inconsistent"
        )
    if require_current_sklearn and contract.get("implementation_version") != sklearn.__version__:
        raise TSVDContractError(
            "tsvd_runtime_version_mismatch",
            "TSVD artifact was trained with a different scikit-learn version",
            details={
                "trained_version": contract.get("implementation_version"),
                "runtime_version": sklearn.__version__,
            },
        )
    if estimator is not None:
        if not isinstance(estimator, TruncatedSVD):
            raise TSVDContractError(
                "tsvd_reducer_type_invalid",
                "TSVD artifact reducer must be sklearn.decomposition.TruncatedSVD",
            )
        actual = estimator.get_params(deep=False)
        if any(actual.get(key) != value for key, value in expected_params.items()):
            raise TSVDContractError(
                "tsvd_reducer_contract_mismatch",
                "TSVD reducer parameters do not match the artifact contract",
            )
        if np.asarray(getattr(estimator, "components_", [])).shape != (
            dimension,
            input_features,
        ):
            raise TSVDContractError(
                "tsvd_reducer_contract_mismatch",
                "TSVD reducer component shape does not match the artifact contract",
            )
        actual_diagnostics = _diagnostics(estimator, dimension)
        for key in diagnostic_names:
            if not np.allclose(actual_diagnostics[key], diagnostics[key], rtol=0.0, atol=0.0):
                raise TSVDContractError(
                    "tsvd_reducer_contract_mismatch",
                    f"TSVD reducer {key} diagnostics do not match the artifact contract",
                )
    return contract


def transform_tsvd_held_out(
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
        raise TSVDContractError(
            "tsvd_transform_failed",
            f"TSVD failed to transform the {split} matrix: {exc}",
            details={"split": split},
        ) from exc
    values = np.asarray(transformed)
    if values.shape != (rows, output_dimension) or not np.isfinite(values).all():
        raise TSVDContractError(
            "tsvd_transform_output_invalid",
            f"TSVD produced invalid {split} output",
            details={"split": split, "shape": list(values.shape)},
        )
    return values


def fit_tsvd_transform(
    train_matrix: object,
    validation_matrix: object,
    test_matrix: object | None = None,
    *,
    n_components: object,
    seed: int,
    dense_memory_limit_gib: object = TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, TruncatedSVD, dict[str, Any]]:
    dimension = validate_tsvd_dimension(n_components)
    normalized_seed = _validate_seed(seed)
    limit = validate_tsvd_dense_memory_limit_gib(dense_memory_limit_gib)
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
    estimator = new_truncated_svd(dimension, normalized_seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            train_output = estimator.fit_transform(train_matrix)
        except Exception as exc:
            raise TSVDContractError("tsvd_fit_failed", f"TSVD fitting failed: {exc}") from exc
    if caught:
        raise TSVDContractError(
            "tsvd_fit_warning",
            "TSVD fitting emitted a numerical warning and the fitted state was rejected",
            details={"warnings": [str(item.message) for item in caught]},
        )
    train_values = np.asarray(train_output)
    if train_values.shape != (train_rows, dimension) or not np.isfinite(train_values).all():
        raise TSVDContractError(
            "tsvd_fit_output_invalid",
            "TSVD produced invalid training output",
            details={"shape": list(train_values.shape)},
        )
    contract = build_tsvd_contract(
        estimator,
        train_sample_count=train_rows,
        input_feature_count=input_features,
        n_components=dimension,
        seed=normalized_seed,
        dense_memory_limit_gib=limit,
        input_dtype=_matrix_dtype(train_matrix),
        input_format="sparse" if sparse.issparse(train_matrix) else "dense",
    )
    validation_output = transform_tsvd_held_out(
        estimator,
        validation_matrix,
        split="validation",
        input_feature_count=input_features,
        output_dimension=dimension,
        dense_memory_limit_gib=limit,
    )
    test_output = None
    if test_matrix is not None:
        test_output = transform_tsvd_held_out(
            estimator,
            test_matrix,
            split="test",
            input_feature_count=input_features,
            output_dimension=dimension,
            dense_memory_limit_gib=limit,
        )
    validate_tsvd_contract(contract, estimator=estimator)
    return train_values, validation_output, test_output, estimator, contract
