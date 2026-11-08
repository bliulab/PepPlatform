"""Contracted Train-fitted Principal Component Analysis for ML routes."""

from __future__ import annotations

import hashlib
import math
import warnings
from typing import Any

import numpy as np
import sklearn
from sklearn.decomposition import PCA


PCA_CONTRACT_VERSION = 1
PCA_VARIANT = "train_only_centered_pca"
PCA_DEFAULT_DENSE_MEMORY_LIMIT_GIB = 4.0
PCA_FIT_SCOPE = "train_only"


class PCAContractError(ValueError):
    """Raised when a PCA route or artifact violates contract v1."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def validate_pca_dense_memory_limit_gib(value: object) -> float:
    if isinstance(value, bool):
        raise PCAContractError(
            "pca_memory_limit_invalid",
            "pca_dense_memory_limit_gib must be a positive finite number",
        )
    try:
        limit = float(value)
    except (TypeError, ValueError) as exc:
        raise PCAContractError(
            "pca_memory_limit_invalid",
            "pca_dense_memory_limit_gib must be a positive finite number",
        ) from exc
    if not math.isfinite(limit) or limit <= 0.0:
        raise PCAContractError(
            "pca_memory_limit_invalid",
            "pca_dense_memory_limit_gib must be a positive finite number",
        )
    return limit


def validate_pca_dimension(value: object) -> int:
    if isinstance(value, bool):
        raise PCAContractError(
            "pca_dimension_invalid", "ml_reduction_dim must be a positive integer"
        )
    try:
        dimension = int(value)
    except (TypeError, ValueError) as exc:
        raise PCAContractError(
            "pca_dimension_invalid", "ml_reduction_dim must be a positive integer"
        ) from exc
    if dimension <= 0 or str(value).strip() not in {str(dimension), f"{dimension}.0"}:
        raise PCAContractError(
            "pca_dimension_invalid", "ml_reduction_dim must be a positive integer"
        )
    return dimension


def _validate_seed(value: object) -> int:
    if isinstance(value, bool):
        raise PCAContractError("pca_seed_invalid", "PCA seed must be a non-negative integer")
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise PCAContractError(
            "pca_seed_invalid", "PCA seed must be a non-negative integer"
        ) from exc
    if seed < 0 or str(value).strip() not in {str(seed), f"{seed}.0"}:
        raise PCAContractError("pca_seed_invalid", "PCA seed must be a non-negative integer")
    return seed


def _as_dense_matrix(
    matrix: object,
    *,
    split: str,
    expected_features: int | None = None,
) -> np.ndarray:
    if hasattr(matrix, "tocsr"):
        raise PCAContractError(
            "pca_matrix_format_invalid",
            f"PCA {split} matrix must be dense",
            details={"split": split},
        )
    try:
        values = np.asarray(matrix)
    except Exception as exc:
        raise PCAContractError(
            "pca_matrix_invalid",
            f"PCA {split} matrix could not be converted to an array",
            details={"split": split},
        ) from exc
    if values.ndim != 2:
        raise PCAContractError(
            "pca_matrix_shape_invalid",
            f"PCA {split} matrix must be two-dimensional",
            details={"split": split, "shape": list(values.shape)},
        )
    rows, columns = map(int, values.shape)
    if rows <= 0 or columns <= 0:
        raise PCAContractError(
            "pca_matrix_empty",
            f"PCA {split} matrix must contain at least one row and one feature",
            details={"split": split, "shape": [rows, columns]},
        )
    if expected_features is not None and columns != expected_features:
        raise PCAContractError(
            "pca_feature_width_mismatch",
            f"PCA {split} matrix has {columns} features; expected {expected_features}",
            details={"split": split, "expected": expected_features, "actual": columns},
        )
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise PCAContractError(
            "pca_matrix_dtype_invalid",
            f"PCA {split} matrix must contain numeric values",
            details={"split": split, "dtype": str(values.dtype)},
        )
    if not np.isfinite(values).all():
        raise PCAContractError(
            "pca_matrix_nonfinite",
            f"PCA {split} matrix contains non-finite values",
            details={"split": split},
        )
    return np.asarray(values)


def _resolve_solver(rows: int, columns: int, dimension: int) -> str:
    if columns <= 1_000 and rows >= 10 * columns:
        return "covariance_eigh"
    if max(rows, columns) <= 500:
        return "full"
    if 1 <= dimension < 0.8 * min(rows, columns):
        return "randomized"
    return "full"


def _estimate_fit_memory_gib(
    rows: int,
    columns: int,
    dimension: int,
    dtype: object,
) -> float:
    itemsize = max(int(np.dtype(dtype).itemsize), 8)
    rank = min(rows, columns)
    solver = _resolve_solver(rows, columns, dimension)
    work = 2 * rows * columns
    if solver == "covariance_eigh":
        work += columns * columns
    elif solver == "randomized":
        sketch = min(columns, dimension + 10)
        work += rows * sketch + columns * sketch + sketch * sketch
    else:
        work += rows * rank + rank * columns
    return float(2.0 * work * itemsize / (1024**3))


def _estimate_transform_memory_gib(
    rows: int,
    columns: int,
    dimension: int,
    dtype: object,
) -> float:
    itemsize = max(int(np.dtype(dtype).itemsize), 8)
    work = 2 * rows * columns + rows * dimension
    return float(2.0 * work * itemsize / (1024**3))


def _validate_memory(estimate: float, limit: float, split: str) -> float:
    if estimate > limit:
        raise PCAContractError(
            "pca_dense_memory_limit_exceeded",
            f"PCA {split} matrix requires approximately {estimate:.3g} GiB, above the {limit:g} GiB limit",
            details={"split": split, "estimated_gib": estimate, "limit_gib": limit},
        )
    return estimate


def _array_digest(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _real_numeric_dtype(value: object) -> np.dtype | None:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError):
        return None
    if not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.complexfloating):
        return None
    return dtype


def _diagnostics(estimator: PCA, dimension: int) -> dict[str, Any]:
    components = np.asarray(estimator.components_)
    if (
        components.shape != (dimension, int(estimator.n_features_in_))
        or not np.isfinite(components).all()
    ):
        raise PCAContractError("pca_components_invalid", "PCA components are invalid")
    diagnostics: dict[str, Any] = {}
    for name in ("singular_values_", "explained_variance_", "explained_variance_ratio_"):
        values = np.asarray(getattr(estimator, name), dtype=np.float64)
        if values.shape != (dimension,) or not np.isfinite(values).all():
            raise PCAContractError(
                "pca_diagnostics_invalid",
                f"PCA produced invalid {name[:-1]} diagnostics",
                details={"diagnostic": name, "shape": list(values.shape)},
            )
        diagnostics[name[:-1]] = values.tolist()
    singular_values = np.asarray(estimator.singular_values_, dtype=np.float64)
    tolerance = np.finfo(np.float64).eps * max(components.shape) * float(singular_values[0])
    effective_rank = int(np.count_nonzero(singular_values > tolerance))
    if effective_rank < dimension:
        raise PCAContractError(
            "pca_rank_insufficient",
            f"PCA training matrix has effective rank {effective_rank}, below requested {dimension}",
            details={"effective_rank": effective_rank, "requested": dimension},
        )
    diagnostics.update(
        {
            "effective_rank": effective_rank,
            "rank_tolerance": float(tolerance),
            "explained_variance_ratio_sum": float(np.sum(estimator.explained_variance_ratio_)),
        }
    )
    return diagnostics


def build_pca_contract(
    estimator: PCA,
    *,
    train_sample_count: int,
    input_feature_count: int,
    n_components: int,
    seed: int,
    dense_memory_limit_gib: float,
    input_dtype: object,
    estimated_memory_gib: float,
) -> dict[str, Any]:
    diagnostics = _diagnostics(estimator, n_components)
    return {
        "contract_version": PCA_CONTRACT_VERSION,
        "algorithm": "pca",
        "public_name": "PCA",
        "variant": PCA_VARIANT,
        "status": "verified_platform_variant",
        "route": "generic_numeric_reduction",
        "input_orientation": "samples_by_features",
        "input_semantics": "generic_numeric",
        "input_dtype": str(np.dtype(input_dtype)),
        "input_format": "dense",
        "centering": True,
        "decomposition": "singular_value_decomposition",
        "implementation": "sklearn.decomposition.PCA",
        "implementation_version": sklearn.__version__,
        "fit_solver": str(getattr(estimator, "_fit_svd_solver", estimator.svd_solver)),
        "feature_fit_scope": PCA_FIT_SCOPE,
        "held_out_semantics": "frozen_centered_linear_transform",
        "train_sample_count": int(train_sample_count),
        "input_feature_count": int(input_feature_count),
        "output_dimension": int(n_components),
        "seed": int(seed),
        "estimator_params": {
            "n_components": int(n_components),
            "svd_solver": "auto",
            "whiten": False,
            "copy": True,
            "random_state": int(seed),
        },
        "dense_memory_limit_gib": float(dense_memory_limit_gib),
        "estimated_memory_gib": float(estimated_memory_gib),
        "mean_sha256": _array_digest(estimator.mean_),
        "components_sha256": _array_digest(estimator.components_),
        "diagnostics": diagnostics,
        "serialization": {"format": "joblib", "requires_matching_sklearn_version": True},
    }


def validate_pca_contract(
    contract: object,
    *,
    estimator: object | None = None,
    require_current_sklearn: bool = False,
) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise PCAContractError("pca_contract_missing", "PCA artifact contract is missing")
    dimension = contract.get("output_dimension")
    features = contract.get("input_feature_count")
    samples = contract.get("train_sample_count")
    seed = contract.get("seed")
    valid_dimensions = (
        isinstance(dimension, int)
        and not isinstance(dimension, bool)
        and dimension >= 1
        and isinstance(features, int)
        and not isinstance(features, bool)
        and features >= 2
        and isinstance(samples, int)
        and not isinstance(samples, bool)
        and samples >= 2
        and dimension < features
        and dimension <= samples - 1
    )
    diagnostics = contract.get("diagnostics")
    names = ("singular_values", "explained_variance", "explained_variance_ratio")
    input_dtype = _real_numeric_dtype(contract.get("input_dtype"))
    singular_values = diagnostics.get("singular_values") if isinstance(diagnostics, dict) else None
    explained_variance = (
        diagnostics.get("explained_variance") if isinstance(diagnostics, dict) else None
    )
    explained_ratio = (
        diagnostics.get("explained_variance_ratio") if isinstance(diagnostics, dict) else None
    )
    expected_tolerance = None
    if (
        isinstance(dimension, int)
        and isinstance(features, int)
        and isinstance(singular_values, list)
        and singular_values
    ):
        try:
            expected_tolerance = (
                np.finfo(np.float64).eps * max(dimension, features) * float(singular_values[0])
            )
        except (TypeError, ValueError):
            expected_tolerance = None
    valid_diagnostics = bool(
        isinstance(diagnostics, dict)
        and isinstance(dimension, int)
        and isinstance(diagnostics.get("effective_rank"), int)
        and not isinstance(diagnostics.get("effective_rank"), bool)
        and diagnostics.get("effective_rank") == dimension
        and all(
            isinstance(diagnostics.get(name), list)
            and len(diagnostics[name]) == dimension
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in diagnostics[name]
            )
            for name in names
        )
        and expected_tolerance is not None
        and all(value > expected_tolerance for value in singular_values)
        and all(value >= 0.0 for value in explained_variance)
        and all(value >= 0.0 for value in explained_ratio)
        and isinstance(diagnostics.get("rank_tolerance"), (int, float))
        and not isinstance(diagnostics.get("rank_tolerance"), bool)
        and math.isfinite(diagnostics["rank_tolerance"])
        and diagnostics["rank_tolerance"] == expected_tolerance
        and isinstance(diagnostics.get("explained_variance_ratio_sum"), (int, float))
        and not isinstance(diagnostics.get("explained_variance_ratio_sum"), bool)
        and math.isfinite(diagnostics["explained_variance_ratio_sum"])
        and 0.0 <= diagnostics["explained_variance_ratio_sum"] <= 1.0 + 1e-12
        and math.isclose(
            diagnostics["explained_variance_ratio_sum"],
            sum(explained_ratio),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    )
    try:
        limit = validate_pca_dense_memory_limit_gib(contract.get("dense_memory_limit_gib"))
    except PCAContractError:
        limit = None
    estimated = contract.get("estimated_memory_gib")
    expected_estimated = (
        _estimate_fit_memory_gib(samples, features, dimension, input_dtype)
        if valid_dimensions and input_dtype is not None
        else None
    )
    expected_solver = _resolve_solver(samples, features, dimension) if valid_dimensions else None
    valid_memory = (
        isinstance(estimated, (int, float))
        and not isinstance(estimated, bool)
        and math.isfinite(estimated)
        and estimated >= 0.0
        and limit is not None
        and estimated <= limit
        and estimated == expected_estimated
    )
    valid = (
        contract.get("contract_version") == PCA_CONTRACT_VERSION
        and contract.get("algorithm") == "pca"
        and contract.get("public_name") == "PCA"
        and contract.get("variant") == PCA_VARIANT
        and contract.get("status") == "verified_platform_variant"
        and contract.get("route") == "generic_numeric_reduction"
        and contract.get("input_orientation") == "samples_by_features"
        and contract.get("input_semantics") == "generic_numeric"
        and input_dtype is not None
        and contract.get("input_format") == "dense"
        and contract.get("centering") is True
        and contract.get("decomposition") == "singular_value_decomposition"
        and contract.get("implementation") == "sklearn.decomposition.PCA"
        and isinstance(contract.get("implementation_version"), str)
        and contract.get("fit_solver") == expected_solver
        and contract.get("feature_fit_scope") == PCA_FIT_SCOPE
        and contract.get("held_out_semantics") == "frozen_centered_linear_transform"
        and valid_dimensions
        and isinstance(seed, int)
        and not isinstance(seed, bool)
        and seed >= 0
        and contract.get("estimator_params")
        == {
            "n_components": dimension,
            "svd_solver": "auto",
            "whiten": False,
            "copy": True,
            "random_state": seed,
        }
        and valid_memory
        and _is_sha256(contract.get("mean_sha256"))
        and _is_sha256(contract.get("components_sha256"))
        and valid_diagnostics
        and isinstance(contract.get("serialization"), dict)
        and contract["serialization"].get("format") == "joblib"
        and contract["serialization"].get("requires_matching_sklearn_version") is True
    )
    if not valid:
        raise PCAContractError(
            "pca_contract_invalid", "PCA artifact contract is incomplete or inconsistent"
        )
    if require_current_sklearn and contract["implementation_version"] != sklearn.__version__:
        raise PCAContractError(
            "pca_runtime_version_mismatch",
            "PCA artifact was trained with a different scikit-learn version",
            details={
                "trained_version": contract["implementation_version"],
                "runtime_version": sklearn.__version__,
            },
        )
    if estimator is not None:
        if not isinstance(estimator, PCA):
            raise PCAContractError(
                "pca_reducer_type_invalid", "PCA artifact reducer must be sklearn.decomposition.PCA"
            )
        expected_params = contract["estimator_params"]
        actual = estimator.get_params(deep=False)
        if any(actual.get(key) != value for key, value in expected_params.items()):
            raise PCAContractError(
                "pca_reducer_contract_mismatch",
                "PCA reducer parameters do not match the artifact contract",
            )
        if np.asarray(getattr(estimator, "components_", [])).shape != (dimension, features):
            raise PCAContractError(
                "pca_reducer_contract_mismatch",
                "PCA component shape does not match the artifact contract",
            )
        if np.asarray(getattr(estimator, "mean_", [])).shape != (features,):
            raise PCAContractError(
                "pca_reducer_contract_mismatch",
                "PCA mean shape does not match the artifact contract",
            )
        if str(getattr(estimator, "_fit_svd_solver", "")) != contract["fit_solver"]:
            raise PCAContractError(
                "pca_reducer_contract_mismatch",
                "PCA resolved solver does not match the artifact contract",
            )
        if (
            _array_digest(estimator.mean_) != contract["mean_sha256"]
            or _array_digest(estimator.components_) != contract["components_sha256"]
        ):
            raise PCAContractError(
                "pca_reducer_contract_mismatch",
                "PCA fitted state digest does not match the artifact contract",
            )
        actual_diagnostics = _diagnostics(estimator, dimension)
        for name in names:
            if not np.allclose(
                actual_diagnostics[name], contract["diagnostics"][name], rtol=0.0, atol=0.0
            ):
                raise PCAContractError(
                    "pca_reducer_contract_mismatch",
                    f"PCA {name} diagnostics do not match the artifact contract",
                )
    return contract


def transform_pca_held_out(
    estimator: PCA,
    matrix: object,
    *,
    split: str,
    input_feature_count: int,
    output_dimension: int,
    dense_memory_limit_gib: float,
) -> np.ndarray:
    values = _as_dense_matrix(matrix, split=split, expected_features=input_feature_count)
    estimate = _estimate_transform_memory_gib(
        values.shape[0],
        values.shape[1],
        output_dimension,
        values.dtype,
    )
    _validate_memory(estimate, dense_memory_limit_gib, split)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transformed = estimator.transform(values)
    except Exception as exc:
        raise PCAContractError(
            "pca_transform_failed",
            f"PCA failed to transform the {split} matrix: {exc}",
            details={"split": split},
        ) from exc
    if caught:
        raise PCAContractError(
            "pca_transform_warning",
            f"PCA emitted a warning while transforming the {split} matrix",
            details={"split": split, "warnings": [str(w.message) for w in caught]},
        )
    result = np.asarray(transformed)
    if result.shape != (values.shape[0], output_dimension) or not np.isfinite(result).all():
        raise PCAContractError(
            "pca_transform_output_invalid",
            f"PCA produced invalid {split} output",
            details={"split": split, "shape": list(result.shape)},
        )
    return result


def fit_pca_transform(
    train_matrix: object,
    validation_matrix: object,
    test_matrix: object | None = None,
    *,
    n_components: object,
    seed: object,
    dense_memory_limit_gib: object = PCA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, PCA, dict[str, Any]]:
    dimension = validate_pca_dimension(n_components)
    normalized_seed = _validate_seed(seed)
    limit = validate_pca_dense_memory_limit_gib(dense_memory_limit_gib)
    train = _as_dense_matrix(train_matrix, split="train")
    validation = _as_dense_matrix(
        validation_matrix, split="validation", expected_features=train.shape[1]
    )
    test = (
        None
        if test_matrix is None
        else _as_dense_matrix(test_matrix, split="test", expected_features=train.shape[1])
    )
    if train.shape[0] < 2 or train.shape[1] < 2:
        raise PCAContractError(
            "pca_matrix_too_small", "PCA requires at least two training samples and two features"
        )
    if dimension >= train.shape[1]:
        raise PCAContractError(
            "pca_dimension_exceeds_features",
            f"ml_reduction_dim={dimension} must be below the {train.shape[1]} input features",
        )
    if dimension > train.shape[0] - 1:
        raise PCAContractError(
            "pca_dimension_exceeds_samples",
            f"ml_reduction_dim={dimension} exceeds the centered rank bound of {train.shape[0] - 1} training samples",
        )
    estimated = _validate_memory(
        _estimate_fit_memory_gib(train.shape[0], train.shape[1], dimension, train.dtype),
        limit,
        "train",
    )
    _validate_memory(
        _estimate_transform_memory_gib(
            validation.shape[0], validation.shape[1], dimension, validation.dtype
        ),
        limit,
        "validation",
    )
    if test is not None:
        _validate_memory(
            _estimate_transform_memory_gib(test.shape[0], test.shape[1], dimension, test.dtype),
            limit,
            "test",
        )
    estimator = PCA(
        n_components=dimension,
        svd_solver="auto",
        whiten=False,
        copy=True,
        random_state=normalized_seed,
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train_output = estimator.fit_transform(train)
            validation_output = estimator.transform(validation)
            test_output = None if test is None else estimator.transform(test)
    except PCAContractError:
        raise
    except Exception as exc:
        raise PCAContractError("pca_fit_failed", f"PCA fitting failed: {exc}") from exc
    if caught:
        raise PCAContractError(
            "pca_fit_warning",
            "PCA emitted a warning during fitting",
            details={"warnings": [str(w.message) for w in caught]},
        )
    contract = build_pca_contract(
        estimator,
        train_sample_count=train.shape[0],
        input_feature_count=train.shape[1],
        n_components=dimension,
        seed=normalized_seed,
        dense_memory_limit_gib=limit,
        input_dtype=train.dtype,
        estimated_memory_gib=estimated,
    )
    for split, values in (
        ("train", train_output),
        ("validation", validation_output),
        ("test", test_output),
    ):
        if values is None:
            continue
        array = np.asarray(values)
        expected_rows = (
            train.shape[0]
            if split == "train"
            else validation.shape[0]
            if split == "validation"
            else test.shape[0]
        )
        if array.shape != (expected_rows, dimension) or not np.isfinite(array).all():
            raise PCAContractError(
                "pca_fit_output_invalid",
                f"PCA produced invalid {split} output",
                details={"split": split, "shape": list(array.shape)},
            )
    return (
        np.asarray(train_output),
        np.asarray(validation_output),
        None if test_output is None else np.asarray(test_output),
        estimator,
        contract,
    )
