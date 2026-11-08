"""Contracted Train-fitted Kernel PCA for dense classical ML routes."""

from __future__ import annotations

import hashlib
import math
import warnings
from typing import Any

import numpy as np
import sklearn
from sklearn.decomposition import KernelPCA


KPCA_CONTRACT_VERSION = 1
KPCA_VARIANT = "train_only_centered_psd_kernel_pca"
KPCA_DEFAULT_MEMORY_LIMIT_GIB = 4.0
KPCA_FIT_SCOPE = "train_only"
KPCA_KERNELS = frozenset({"rbf", "linear", "poly", "cosine"})
KPCA_GAMMA_KERNELS = frozenset({"rbf", "poly"})


class KPCAContractError(ValueError):
    """Raised when a KPCA route or artifact violates contract v1."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def validate_kpca_memory_limit_gib(value: object) -> float:
    if isinstance(value, bool):
        raise KPCAContractError(
            "kpca_memory_limit_invalid", "kpca_memory_limit_gib must be a positive finite number"
        )
    try:
        limit = float(value)
    except (TypeError, ValueError) as exc:
        raise KPCAContractError(
            "kpca_memory_limit_invalid", "kpca_memory_limit_gib must be a positive finite number"
        ) from exc
    if not math.isfinite(limit) or limit <= 0.0:
        raise KPCAContractError(
            "kpca_memory_limit_invalid", "kpca_memory_limit_gib must be a positive finite number"
        )
    return limit


def validate_kpca_kernel(value: object) -> str:
    kernel = str(value or "").strip().lower()
    if kernel not in KPCA_KERNELS:
        raise KPCAContractError(
            "kpca_kernel_invalid", f"kpca_kernel must be one of {sorted(KPCA_KERNELS)}"
        )
    return kernel


def validate_kpca_dimension(value: object) -> int:
    if isinstance(value, bool):
        raise KPCAContractError(
            "kpca_dimension_invalid", "ml_reduction_dim must be a positive integer"
        )
    try:
        dimension = int(value)
    except (TypeError, ValueError) as exc:
        raise KPCAContractError(
            "kpca_dimension_invalid", "ml_reduction_dim must be a positive integer"
        ) from exc
    if dimension <= 0 or str(value).strip() not in {str(dimension), f"{dimension}.0"}:
        raise KPCAContractError(
            "kpca_dimension_invalid", "ml_reduction_dim must be a positive integer"
        )
    return dimension


def _validate_seed(value: object) -> int:
    if isinstance(value, bool):
        raise KPCAContractError("kpca_seed_invalid", "KPCA seed must be a non-negative integer")
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise KPCAContractError(
            "kpca_seed_invalid", "KPCA seed must be a non-negative integer"
        ) from exc
    if seed < 0 or str(value).strip() not in {str(seed), f"{seed}.0"}:
        raise KPCAContractError("kpca_seed_invalid", "KPCA seed must be a non-negative integer")
    return seed


def _as_dense_matrix(
    matrix: object, *, split: str, expected_features: int | None = None
) -> np.ndarray:
    if hasattr(matrix, "tocsr"):
        raise KPCAContractError("kpca_matrix_format_invalid", f"KPCA {split} matrix must be dense")
    values = np.asarray(matrix)
    if values.ndim != 2:
        raise KPCAContractError(
            "kpca_matrix_shape_invalid",
            f"KPCA {split} matrix must be two-dimensional",
            details={"shape": list(values.shape)},
        )
    rows, columns = map(int, values.shape)
    if rows <= 0 or columns <= 0:
        raise KPCAContractError(
            "kpca_matrix_empty",
            f"KPCA {split} matrix must contain at least one row and one feature",
        )
    if expected_features is not None and columns != expected_features:
        raise KPCAContractError(
            "kpca_feature_width_mismatch",
            f"KPCA {split} matrix has {columns} features; expected {expected_features}",
        )
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise KPCAContractError(
            "kpca_matrix_dtype_invalid", f"KPCA {split} matrix must contain numeric values"
        )
    if not np.isfinite(values).all():
        raise KPCAContractError(
            "kpca_matrix_nonfinite", f"KPCA {split} matrix contains non-finite values"
        )
    return np.asarray(values)


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


def _estimate_memory_gib(
    rows: int, columns: int, dimension: int, *, dtype: object, train_rows: int | None = None
) -> float:
    reference_rows = rows if train_rows is None else train_rows
    itemsize = max(int(np.dtype(dtype).itemsize), 8)
    if train_rows is None:
        work = rows * columns + 2 * rows * rows + 2 * rows * dimension
    else:
        work = 2 * rows * reference_rows + rows * dimension
    return float(2.0 * work * itemsize / (1024**3))


def _effective_solver(train_rows: int, dimension: int) -> str:
    return "arpack" if train_rows > 200 and dimension < 10 else "dense"


def validate_kpca_gamma(kernel: object, value: object) -> float | None:
    normalized_kernel = validate_kpca_kernel(kernel)
    omitted = value is None or (isinstance(value, str) and not value.strip())
    if normalized_kernel not in KPCA_GAMMA_KERNELS:
        if not omitted:
            raise KPCAContractError(
                "kpca_gamma_not_applicable",
                f"kpca_gamma is not applicable to the {normalized_kernel} kernel",
            )
        return None
    if isinstance(value, bool):
        raise KPCAContractError(
            "kpca_gamma_invalid", "kpca_gamma must be a positive finite number for this kernel"
        )
    if omitted:
        return 0.1
    try:
        gamma = float(value)
    except (TypeError, ValueError) as exc:
        raise KPCAContractError(
            "kpca_gamma_invalid", "kpca_gamma must be a positive finite number for this kernel"
        ) from exc
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise KPCAContractError(
            "kpca_gamma_invalid", "kpca_gamma must be a positive finite number for this kernel"
        )
    return gamma


def _diagnostics(estimator: KernelPCA, dimension: int) -> dict[str, Any]:
    eigenvalues = np.asarray(estimator.eigenvalues_, dtype=np.float64)
    eigenvectors = np.asarray(estimator.eigenvectors_)
    if eigenvalues.shape != (dimension,) or eigenvectors.shape != (
        estimator.X_fit_.shape[0],
        dimension,
    ):
        raise KPCAContractError(
            "kpca_diagnostics_invalid", "KPCA eigenvalue/eigenvector shapes are invalid"
        )
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise KPCAContractError(
            "kpca_diagnostics_invalid", "KPCA eigenvalue/eigenvector diagnostics are non-finite"
        )
    tolerance = np.finfo(np.float64).eps * max(eigenvectors.shape) * max(float(eigenvalues[0]), 1.0)
    effective_rank = int(np.count_nonzero(eigenvalues > tolerance))
    if effective_rank < dimension:
        raise KPCAContractError(
            "kpca_rank_insufficient",
            f"KPCA kernel matrix has effective rank {effective_rank}, below requested {dimension}",
        )
    return {
        "eigenvalues": eigenvalues.tolist(),
        "effective_rank": effective_rank,
        "rank_tolerance": float(tolerance),
        "eigenvectors_sha256": _array_digest(eigenvectors),
    }


def build_kpca_contract(
    estimator: KernelPCA,
    *,
    train_sample_count: int,
    input_feature_count: int,
    n_components: int,
    seed: int,
    memory_limit_gib: float,
    estimated_memory_gib: float,
    input_dtype: object,
    kernel: str,
    gamma: float | None,
) -> dict[str, Any]:
    diagnostics = _diagnostics(estimator, n_components)
    return {
        "contract_version": KPCA_CONTRACT_VERSION,
        "algorithm": "kpca",
        "public_name": "KPCA",
        "variant": KPCA_VARIANT,
        "status": "verified_platform_variant",
        "route": "generic_numeric_reduction",
        "input_orientation": "samples_by_features",
        "input_semantics": "generic_numeric",
        "input_dtype": str(np.dtype(input_dtype)),
        "input_format": "dense",
        "centering": "kernel_centerer",
        "decomposition": "centered_kernel_eigendecomposition",
        "implementation": "sklearn.decomposition.KernelPCA",
        "implementation_version": sklearn.__version__,
        "effective_eigen_solver": _effective_solver(train_sample_count, n_components),
        "feature_fit_scope": KPCA_FIT_SCOPE,
        "held_out_semantics": "frozen_centered_kernel_transform",
        "train_sample_count": int(train_sample_count),
        "input_feature_count": int(input_feature_count),
        "output_dimension": int(n_components),
        "seed": int(seed),
        "kernel": kernel,
        "kernel_params": {
            "gamma": gamma,
            "degree": 3,
            "coef0": 1,
            "fit_inverse_transform": False,
            "remove_zero_eig": False,
        },
        "estimator_params": {
            "n_components": int(n_components),
            "kernel": kernel,
            "gamma": gamma,
            "degree": 3,
            "coef0": 1,
            "alpha": 1.0,
            "fit_inverse_transform": False,
            "eigen_solver": "auto",
            "remove_zero_eig": False,
            "copy_X": True,
            "n_jobs": None,
            "random_state": int(seed),
        },
        "memory_limit_gib": float(memory_limit_gib),
        "estimated_memory_gib": float(estimated_memory_gib),
        "train_matrix_sha256": _array_digest(estimator.X_fit_),
        "diagnostics": diagnostics,
        "serialization": {"format": "joblib", "requires_matching_sklearn_version": True},
    }


def validate_kpca_contract(
    contract: object, *, estimator: object | None = None, require_current_sklearn: bool = False
) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise KPCAContractError("kpca_contract_missing", "KPCA artifact contract is missing")
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
        and dimension < min(features, samples)
    )
    kernel = contract.get("kernel")
    params = contract.get("kernel_params")
    diagnostics = contract.get("diagnostics")
    input_dtype = _real_numeric_dtype(contract.get("input_dtype"))
    eigenvalues = diagnostics.get("eigenvalues") if isinstance(diagnostics, dict) else None
    expected_tolerance = None
    if (
        isinstance(samples, int)
        and isinstance(dimension, int)
        and isinstance(eigenvalues, list)
        and eigenvalues
    ):
        try:
            expected_tolerance = (
                np.finfo(np.float64).eps * max(samples, dimension) * max(float(eigenvalues[0]), 1.0)
            )
        except (TypeError, ValueError):
            expected_tolerance = None
    valid_diagnostics = bool(
        isinstance(diagnostics, dict)
        and isinstance(dimension, int)
        and isinstance(diagnostics.get("effective_rank"), int)
        and not isinstance(diagnostics.get("effective_rank"), bool)
        and diagnostics.get("effective_rank") == dimension
        and isinstance(diagnostics.get("eigenvalues"), list)
        and len(diagnostics["eigenvalues"]) == dimension
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in diagnostics["eigenvalues"]
        )
        and expected_tolerance is not None
        and all(value > expected_tolerance for value in diagnostics["eigenvalues"])
        and isinstance(diagnostics.get("rank_tolerance"), (int, float))
        and not isinstance(diagnostics.get("rank_tolerance"), bool)
        and math.isfinite(diagnostics["rank_tolerance"])
        and diagnostics["rank_tolerance"] == expected_tolerance
        and _is_sha256(diagnostics.get("eigenvectors_sha256"))
    )
    try:
        limit = validate_kpca_memory_limit_gib(contract.get("memory_limit_gib"))
    except KPCAContractError:
        limit = None
    estimated = contract.get("estimated_memory_gib")
    expected_estimated = (
        _estimate_memory_gib(samples, features, dimension, dtype=input_dtype)
        if valid_dimensions and input_dtype is not None
        else None
    )
    valid_memory = (
        isinstance(estimated, (int, float))
        and not isinstance(estimated, bool)
        and math.isfinite(estimated)
        and estimated >= 0.0
        and limit is not None
        and estimated <= limit
        and estimated == expected_estimated
    )
    expected_solver = _effective_solver(samples, dimension) if valid_dimensions else None
    valid = (
        contract.get("contract_version") == KPCA_CONTRACT_VERSION
        and contract.get("algorithm") == "kpca"
        and contract.get("public_name") == "KPCA"
        and contract.get("variant") == KPCA_VARIANT
        and contract.get("status") == "verified_platform_variant"
        and contract.get("route") == "generic_numeric_reduction"
        and contract.get("input_orientation") == "samples_by_features"
        and contract.get("input_semantics") == "generic_numeric"
        and input_dtype is not None
        and contract.get("input_format") == "dense"
        and contract.get("centering") == "kernel_centerer"
        and contract.get("decomposition") == "centered_kernel_eigendecomposition"
        and contract.get("implementation") == "sklearn.decomposition.KernelPCA"
        and isinstance(contract.get("implementation_version"), str)
        and contract.get("effective_eigen_solver") == expected_solver
        and contract.get("feature_fit_scope") == KPCA_FIT_SCOPE
        and contract.get("held_out_semantics") == "frozen_centered_kernel_transform"
        and valid_dimensions
        and isinstance(kernel, str)
        and kernel in KPCA_KERNELS
        and isinstance(seed, int)
        and not isinstance(seed, bool)
        and seed >= 0
        and isinstance(params, dict)
        and params.get("degree") == 3
        and params.get("coef0") == 1
        and params.get("fit_inverse_transform") is False
        and params.get("remove_zero_eig") is False
        and (
            (
                kernel in KPCA_GAMMA_KERNELS
                and isinstance(params.get("gamma"), (int, float))
                and not isinstance(params.get("gamma"), bool)
                and math.isfinite(float(params["gamma"]))
                and float(params["gamma"]) > 0.0
            )
            or (kernel not in KPCA_GAMMA_KERNELS and params.get("gamma") is None)
        )
        and contract.get("estimator_params")
        == {
            "n_components": dimension,
            "kernel": kernel,
            "gamma": params.get("gamma"),
            "degree": 3,
            "coef0": 1,
            "alpha": 1.0,
            "fit_inverse_transform": False,
            "eigen_solver": "auto",
            "remove_zero_eig": False,
            "copy_X": True,
            "n_jobs": None,
            "random_state": seed,
        }
        and valid_memory
        and _is_sha256(contract.get("train_matrix_sha256"))
        and valid_diagnostics
        and isinstance(contract.get("serialization"), dict)
        and contract["serialization"].get("format") == "joblib"
        and contract["serialization"].get("requires_matching_sklearn_version") is True
    )
    if not valid:
        raise KPCAContractError(
            "kpca_contract_invalid", "KPCA artifact contract is incomplete or inconsistent"
        )
    if require_current_sklearn and contract["implementation_version"] != sklearn.__version__:
        raise KPCAContractError(
            "kpca_runtime_version_mismatch",
            "KPCA artifact was trained with a different scikit-learn version",
            details={
                "trained_version": contract["implementation_version"],
                "runtime_version": sklearn.__version__,
            },
        )
    if estimator is not None:
        if not isinstance(estimator, KernelPCA):
            raise KPCAContractError(
                "kpca_reducer_type_invalid",
                "KPCA artifact reducer must be sklearn.decomposition.KernelPCA",
            )
        actual_params = estimator.get_params(deep=False)
        if any(
            actual_params.get(key) != value for key, value in contract["estimator_params"].items()
        ):
            raise KPCAContractError(
                "kpca_reducer_contract_mismatch",
                "KPCA reducer parameters do not match the artifact contract",
            )
        if np.asarray(getattr(estimator, "eigenvalues_", [])).shape != (dimension,) or np.asarray(
            getattr(estimator, "eigenvectors_", [])
        ).shape != (samples, dimension):
            raise KPCAContractError(
                "kpca_reducer_contract_mismatch",
                "KPCA eigen component shapes do not match the artifact contract",
            )
        if _array_digest(estimator.X_fit_) != contract["train_matrix_sha256"]:
            raise KPCAContractError(
                "kpca_reducer_contract_mismatch",
                "KPCA fitted training matrix does not match the artifact contract",
            )
        actual = _diagnostics(estimator, dimension)
        if actual["eigenvectors_sha256"] != diagnostics["eigenvectors_sha256"] or not np.allclose(
            actual["eigenvalues"], diagnostics["eigenvalues"], rtol=0.0, atol=0.0
        ):
            raise KPCAContractError(
                "kpca_reducer_contract_mismatch",
                "KPCA spectral diagnostics do not match the artifact contract",
            )
    return contract


def transform_kpca_held_out(
    estimator: KernelPCA, matrix: object, *, split: str, contract: dict[str, Any]
) -> np.ndarray:
    values = _as_dense_matrix(
        matrix, split=split, expected_features=int(contract["input_feature_count"])
    )
    limit = validate_kpca_memory_limit_gib(contract["memory_limit_gib"])
    estimate = _estimate_memory_gib(
        values.shape[0],
        values.shape[1],
        int(contract["output_dimension"]),
        dtype=values.dtype,
        train_rows=int(contract["train_sample_count"]),
    )
    if estimate > limit:
        raise KPCAContractError(
            "kpca_memory_limit_exceeded",
            f"KPCA {split} transform requires approximately {estimate:.3g} GiB, above the {limit:g} GiB limit",
        )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transformed = estimator.transform(values)
    except Exception as exc:
        raise KPCAContractError(
            "kpca_transform_failed", f"KPCA failed to transform the {split} matrix: {exc}"
        ) from exc
    if caught:
        raise KPCAContractError(
            "kpca_transform_warning",
            f"KPCA emitted a warning while transforming the {split} matrix",
        )
    result = np.asarray(transformed)
    if (
        result.shape != (values.shape[0], int(contract["output_dimension"]))
        or not np.isfinite(result).all()
    ):
        raise KPCAContractError(
            "kpca_transform_output_invalid",
            f"KPCA produced invalid {split} output",
            details={"shape": list(result.shape)},
        )
    return result


def fit_kpca_transform(
    train_matrix: object,
    validation_matrix: object,
    test_matrix: object | None = None,
    *,
    n_components: object,
    kernel: object = "rbf",
    gamma: object = None,
    seed: object = 42,
    memory_limit_gib: object = KPCA_DEFAULT_MEMORY_LIMIT_GIB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, KernelPCA, dict[str, Any]]:
    dimension = validate_kpca_dimension(n_components)
    normalized_seed = _validate_seed(seed)
    normalized_kernel = validate_kpca_kernel(kernel)
    effective_gamma = validate_kpca_gamma(normalized_kernel, gamma)
    limit = validate_kpca_memory_limit_gib(memory_limit_gib)
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
        raise KPCAContractError(
            "kpca_matrix_too_small", "KPCA requires at least two training samples and two features"
        )
    if dimension >= min(train.shape[0], train.shape[1]):
        raise KPCAContractError(
            "kpca_dimension_exceeds_rank_bound",
            f"ml_reduction_dim={dimension} must be below both training samples and input features",
        )
    estimated = _estimate_memory_gib(train.shape[0], train.shape[1], dimension, dtype=train.dtype)
    if estimated > limit:
        raise KPCAContractError(
            "kpca_memory_limit_exceeded",
            f"KPCA train matrix requires approximately {estimated:.3g} GiB, above the {limit:g} GiB limit",
        )
    for split, values in (("validation", validation), ("test", test)):
        if values is not None:
            heldout_estimate = _estimate_memory_gib(
                values.shape[0],
                values.shape[1],
                dimension,
                dtype=values.dtype,
                train_rows=train.shape[0],
            )
            if heldout_estimate > limit:
                raise KPCAContractError(
                    "kpca_memory_limit_exceeded",
                    f"KPCA {split} transform requires approximately {heldout_estimate:.3g} GiB, above the {limit:g} GiB limit",
                )
    estimator = KernelPCA(
        n_components=dimension,
        kernel=normalized_kernel,
        gamma=effective_gamma,
        degree=3,
        coef0=1,
        alpha=1.0,
        fit_inverse_transform=False,
        eigen_solver="auto",
        remove_zero_eig=False,
        copy_X=True,
        n_jobs=None,
        random_state=normalized_seed,
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train_output = estimator.fit_transform(train)
            validation_output = estimator.transform(validation)
            test_output = None if test is None else estimator.transform(test)
    except Exception as exc:
        raise KPCAContractError("kpca_fit_failed", f"KPCA fitting failed: {exc}") from exc
    if caught:
        raise KPCAContractError(
            "kpca_fit_warning",
            "KPCA emitted a warning during fitting",
            details={"warnings": [str(w.message) for w in caught]},
        )
    contract = build_kpca_contract(
        estimator,
        train_sample_count=train.shape[0],
        input_feature_count=train.shape[1],
        n_components=dimension,
        seed=normalized_seed,
        memory_limit_gib=limit,
        estimated_memory_gib=estimated,
        input_dtype=train.dtype,
        kernel=normalized_kernel,
        gamma=effective_gamma,
    )
    for split, values, expected_rows in (
        ("train", train_output, train.shape[0]),
        ("validation", validation_output, validation.shape[0]),
        ("test", test_output, None if test is None else test.shape[0]),
    ):
        if values is None:
            continue
        array = np.asarray(values)
        if array.shape != (expected_rows, dimension) or not np.isfinite(array).all():
            raise KPCAContractError(
                "kpca_fit_output_invalid",
                f"KPCA produced invalid {split} output",
                details={"shape": list(array.shape)},
            )
    return (
        np.asarray(train_output),
        np.asarray(validation_output),
        None if test_output is None else np.asarray(test_output),
        estimator,
        contract,
    )
