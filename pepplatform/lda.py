"""Contracted latent Dirichlet allocation for classic BoW sequence counts."""

from __future__ import annotations

from pepplatform.stream_compat import is_pair_bow_route
from pepplatform.task_compat import normalize_task_type


import hashlib
import json
import math
import warnings
from typing import Any

import numpy as np
import sklearn
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation
from pepplatform.pair_bow import (
    PAIR_BOW_ROUTE,
    PairBoWContractError,
    validate_pair_bow_joint_contract,
    validate_pair_bow_matrix,
    validate_pair_bow_route,
)


LDA_CONTRACT_VERSION = 1
LDA_VARIANT = "train_only_batch_variational_bayes_bow_counts"
LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB = 4.0
LDA_FIT_SCOPE = "train_only"
LDA_MAX_ITER = 100
LDA_PERP_TOL = 0.1
LDA_MEAN_CHANGE_TOL = 0.001
LDA_MAX_DOC_UPDATE_ITER = 100
_PAIR_TASK_TYPES = frozenset({"pep_protein_interaction", "pep_hlaii", "pep_hlai"})


class LDAContractError(ValueError):
    """Raised when an LDA route, fit, transform, or artifact violates contract v1."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def validate_lda_dense_memory_limit_gib(value: object) -> float:
    try:
        limit = float(value)
    except (TypeError, ValueError) as exc:
        raise LDAContractError(
            "lda_memory_limit_invalid",
            "lda_dense_memory_limit_gib must be a positive finite number",
        ) from exc
    if not math.isfinite(limit) or limit <= 0.0:
        raise LDAContractError(
            "lda_memory_limit_invalid",
            "lda_dense_memory_limit_gib must be a positive finite number",
        )
    return limit


def validate_lda_topics(value: object) -> int:
    if isinstance(value, bool):
        raise LDAContractError("lda_topics_invalid", "lda_topics must be a positive integer")
    try:
        topics = int(value)
    except (TypeError, ValueError) as exc:
        raise LDAContractError(
            "lda_topics_invalid", "lda_topics must be a positive integer"
        ) from exc
    if topics <= 0 or str(value).strip() not in {str(topics), f"{topics}.0"}:
        raise LDAContractError("lda_topics_invalid", "lda_topics must be a positive integer")
    return topics


def validate_lda_route(source: object, *, pair: bool | None = None) -> None:
    def read(key: str, default: object = "") -> object:
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    reduction = str(read("reduction", "none") or "none").strip().lower()
    if reduction != "lda":
        return
    task_type = normalize_task_type(read("task_type", "") or "")
    is_pair = normalize_task_type(task_type) in _PAIR_TASK_TYPES if pair is None else bool(pair)
    if is_pair:
        # Keep the historical error for incomplete/legacy Pair argument
        # namespaces; a complete strict two-stream BoW route is now allowed.
        if (
            not str(read("classifier", "") or "").strip()
            or not str(read("pair_strategy", "") or "").strip()
        ):
            raise LDAContractError(
                "lda_pair_route_unsupported",
                "reduction=lda is not supported for incomplete Pair workflows because fused vectors are not a verified BoW route; provide the strict Pair Two-stream BoW route",
            )
        try:
            validate_pair_bow_route(source, pair=True)
        except PairBoWContractError as exc:
            raise LDAContractError(exc.code, str(exc), details=exc.details) from exc
        validate_lda_topics(read("lda_topics", 50))
        validate_lda_dense_memory_limit_gib(
            read("lda_dense_memory_limit_gib", LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB)
        )
        return
    encoder = str(read("encoder", "") or "").strip().lower()
    if encoder != "bow":
        raise LDAContractError(
            "lda_encoder_unsupported",
            "reduction=lda requires encoder=bow with the current raw-count contract",
            details={"encoder": encoder},
        )
    validate_lda_topics(read("lda_topics", 50))
    validate_lda_dense_memory_limit_gib(
        read("lda_dense_memory_limit_gib", LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB)
    )


def canonical_lda_contract_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_lda_upstream_contract(
    upstream_contract: object,
    *,
    input_feature_count: int | None = None,
) -> dict[str, Any]:
    if not isinstance(upstream_contract, dict):
        raise LDAContractError(
            "lda_upstream_contract_missing",
            "LDA requires the persisted classic BoW v2 upstream contract",
        )
    if is_pair_bow_route(upstream_contract.get("route")):
        try:
            return validate_pair_bow_joint_contract(
                upstream_contract, input_feature_count=input_feature_count
            )
        except PairBoWContractError as exc:
            raise LDAContractError(exc.code, str(exc), details=exc.details) from exc
    encoder = str(upstream_contract.get("encoder") or "").strip().lower()
    tokenizer = str(upstream_contract.get("tokenizer") or "").strip().lower()
    frequency = upstream_contract.get("frequency_encoder_contract")
    if (
        encoder != "bow"
        or tokenizer not in {"kmer", "dr", "top-n-gram", "dt"}
        or not isinstance(frequency, dict)
        or frequency.get("contract_version") != 2
        or frequency.get("encoder") != "bow"
        or frequency.get("stream_policy") != "full_token_stream"
        or frequency.get("bow_formula") != "x_j = c_j"
        or frequency.get("post_normalization") != "none"
        or frequency.get("fit_scope") != "sequence_train_split"
    ):
        raise LDAContractError(
            "lda_upstream_contract_invalid",
            "LDA requires a Train-scoped BoW v2 raw-count feature contract",
        )
    feature_dimension = frequency.get("feature_dimension")
    if (
        not isinstance(feature_dimension, int)
        or isinstance(feature_dimension, bool)
        or feature_dimension < 1
        or (input_feature_count is not None and feature_dimension != int(input_feature_count))
    ):
        raise LDAContractError(
            "lda_upstream_feature_width_mismatch",
            "LDA input width does not match the BoW v2 feature contract",
            details={
                "contract_feature_count": feature_dimension,
                "input_feature_count": input_feature_count,
            },
        )
    vocabulary_digest = frequency.get("vocabulary_sha256")
    corpus_digest = frequency.get("training_corpus_sha256")
    valid_vocabulary_digest = (
        isinstance(vocabulary_digest, str)
        and len(vocabulary_digest) == 64
        and all(character in "0123456789abcdef" for character in vocabulary_digest)
    )
    valid_corpus_digest = (
        isinstance(corpus_digest, str)
        and len(corpus_digest) == 64
        and all(character in "0123456789abcdef" for character in corpus_digest)
    )
    if not valid_vocabulary_digest or not valid_corpus_digest:
        raise LDAContractError(
            "lda_upstream_contract_invalid",
            "LDA BoW v2 contract is missing a valid vocabulary or training-corpus digest",
        )
    return upstream_contract


def _matrix_shape(matrix: object, *, split: str) -> tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise LDAContractError(
            "lda_matrix_shape_invalid",
            f"LDA {split} matrix must be two-dimensional",
            details={"split": split, "shape": list(shape or [])},
        )
    rows, columns = int(shape[0]), int(shape[1])
    if rows <= 0 or columns <= 0:
        raise LDAContractError(
            "lda_matrix_empty",
            f"LDA {split} matrix must contain at least one row and one feature",
            details={"split": split, "shape": [rows, columns]},
        )
    return rows, columns


def _matrix_dtype(matrix: object) -> np.dtype:
    try:
        return np.dtype(getattr(matrix, "dtype", np.float64))
    except TypeError:
        return np.dtype(np.float64)


def _estimated_working_set_gib(
    rows: int,
    columns: int,
    topics: int,
    dtype: np.dtype,
) -> float:
    input_bytes = rows * columns * max(1, dtype.itemsize)
    topic_state_bytes = 3 * topics * columns * np.dtype(np.float64).itemsize
    document_state_bytes = 2 * rows * topics * np.dtype(np.float64).itemsize
    return (input_bytes + topic_state_bytes + document_state_bytes) / float(1024**3)


def _validate_matrix(
    matrix: object,
    *,
    split: str,
    topics: int,
    dense_memory_limit_gib: float,
    expected_features: int | None = None,
    require_total_count: bool = False,
) -> tuple[int, int]:
    rows, columns = _matrix_shape(matrix, split=split)
    if expected_features is not None and columns != expected_features:
        raise LDAContractError(
            "lda_feature_width_mismatch",
            f"LDA {split} matrix has {columns} features; expected {expected_features}",
            details={
                "split": split,
                "expected_features": expected_features,
                "actual_features": columns,
            },
        )
    dtype = _matrix_dtype(matrix)
    if not np.issubdtype(dtype, np.number):
        raise LDAContractError(
            "lda_matrix_dtype_invalid",
            f"LDA {split} matrix must contain numeric BoW counts",
            details={"split": split, "dtype": str(dtype)},
        )
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    try:
        finite = bool(np.isfinite(values).all())
        nonnegative = bool((values >= 0).all())
        integral = bool(np.equal(values, np.rint(values)).all())
    except TypeError as exc:
        raise LDAContractError(
            "lda_matrix_dtype_invalid",
            f"LDA {split} matrix must contain numeric BoW counts",
            details={"split": split, "dtype": str(dtype)},
        ) from exc
    if not finite:
        raise LDAContractError(
            "lda_matrix_nonfinite",
            f"LDA {split} matrix contains NaN or infinite values",
            details={"split": split},
        )
    if not nonnegative:
        raise LDAContractError(
            "lda_matrix_negative",
            f"LDA {split} matrix contains negative values",
            details={"split": split},
        )
    if not integral:
        raise LDAContractError(
            "lda_matrix_not_counts",
            f"LDA {split} matrix must contain integer-valued BoW counts",
            details={"split": split},
        )
    if require_total_count and float(values.sum()) <= 0.0:
        raise LDAContractError(
            "lda_training_counts_empty",
            "LDA training matrix must contain at least one observed token count",
        )
    estimate = _estimated_working_set_gib(rows, columns, topics, dtype)
    if estimate > dense_memory_limit_gib:
        raise LDAContractError(
            "lda_dense_memory_limit_exceeded",
            f"LDA {split} working-set estimate is {estimate:.3f} GiB, exceeding lda_dense_memory_limit_gib={dense_memory_limit_gib:g}",
            details={
                "split": split,
                "estimated_working_set_gib": estimate,
                "limit_gib": dense_memory_limit_gib,
            },
        )
    return rows, columns


def lda_primitive_params(n_components: int, seed: int) -> dict[str, Any]:
    prior = 1.0 / int(n_components)
    return {
        "n_components": int(n_components),
        "doc_topic_prior": prior,
        "topic_word_prior": prior,
        "learning_method": "batch",
        "learning_decay": 0.7,
        "learning_offset": 10.0,
        "max_iter": LDA_MAX_ITER,
        "batch_size": 128,
        "evaluate_every": 1,
        "total_samples": 1_000_000.0,
        "perp_tol": LDA_PERP_TOL,
        "mean_change_tol": LDA_MEAN_CHANGE_TOL,
        "max_doc_update_iter": LDA_MAX_DOC_UPDATE_ITER,
        "n_jobs": 1,
        "verbose": 0,
        "random_state": int(seed),
    }


def _new_estimator(n_components: int, seed: int) -> LatentDirichletAllocation:
    return LatentDirichletAllocation(**lda_primitive_params(n_components, seed))


def _diagnostics(
    estimator: LatentDirichletAllocation,
    *,
    n_components: int,
    input_feature_count: int,
) -> dict[str, Any]:
    components = np.asarray(estimator.components_, dtype=np.float64)
    exp_components = np.asarray(estimator.exp_dirichlet_component_, dtype=np.float64)
    if (
        components.shape != (n_components, input_feature_count)
        or exp_components.shape != components.shape
        or not np.isfinite(components).all()
        or not np.isfinite(exp_components).all()
        or (components <= 0.0).any()
        or (exp_components <= 0.0).any()
    ):
        raise LDAContractError(
            "lda_components_invalid",
            "LDA produced invalid topic-word variational state",
        )
    iterations = int(estimator.n_iter_)
    bound = float(estimator.bound_)
    if iterations < 1 or iterations >= LDA_MAX_ITER:
        raise LDAContractError(
            "lda_not_converged",
            f"LDA did not satisfy the perplexity stopping rule within {LDA_MAX_ITER} iterations",
            details={"n_iter": iterations, "max_iter": LDA_MAX_ITER},
        )
    if not math.isfinite(bound) or bound <= 0.0:
        raise LDAContractError("lda_bound_invalid", "LDA produced a non-finite training perplexity")
    return {
        "converged": True,
        "n_iter": iterations,
        "max_iter": LDA_MAX_ITER,
        "bound_perplexity": bound,
        "component_min": float(components.min()),
        "component_max": float(components.max()),
        "component_sum": float(components.sum()),
    }


def build_lda_contract(
    estimator: LatentDirichletAllocation,
    *,
    train_sample_count: int,
    input_feature_count: int,
    n_components: int,
    seed: int,
    dense_memory_limit_gib: float,
    upstream_feature_contract: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = _diagnostics(
        estimator,
        n_components=n_components,
        input_feature_count=input_feature_count,
    )
    pair_route = is_pair_bow_route(upstream_feature_contract.get("route"))
    return {
        "contract_version": LDA_CONTRACT_VERSION,
        "algorithm": "latent_dirichlet_allocation",
        "public_name": "LDA",
        "variant": LDA_VARIANT,
        "status": "verified",
        "route": PAIR_BOW_ROUTE if pair_route else "single_sequence_classic_global_bow_v2",
        "input_orientation": (
            "sample_by_peptide_term_then_partner_term" if pair_route else "sequence_by_term"
        ),
        "input_semantics": "raw_nonnegative_integer_bow_counts",
        "implementation": "sklearn.decomposition.LatentDirichletAllocation",
        "implementation_version": sklearn.__version__,
        "inference_method": "variational_bayes",
        "feature_fit_scope": LDA_FIT_SCOPE,
        "held_out_semantics": "frozen_normalized_document_topic_posterior",
        "output_normalization": "probability_simplex",
        "zero_document_semantics": "symmetric_doc_topic_prior",
        "train_sample_count": int(train_sample_count),
        "input_feature_count": int(input_feature_count),
        "output_dimension": int(n_components),
        "seed": int(seed),
        "estimator_params": lda_primitive_params(n_components, seed),
        "effective_priors": {
            "doc_topic_prior": float(estimator.doc_topic_prior_),
            "topic_word_prior": float(estimator.topic_word_prior_),
        },
        "dense_memory_limit_gib": float(dense_memory_limit_gib),
        "upstream_feature_contract_sha256": canonical_lda_contract_digest(
            upstream_feature_contract
        ),
        "diagnostics": diagnostics,
        "serialization": {
            "format": "joblib",
            "requires_matching_sklearn_version": True,
        },
    }


def validate_lda_contract(
    contract: object,
    *,
    estimator: object | None = None,
    upstream_feature_contract: object | None = None,
    require_current_sklearn: bool = False,
) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise LDAContractError("lda_contract_missing", "LDA artifact contract is missing")
    dimension = contract.get("output_dimension")
    input_features = contract.get("input_feature_count")
    train_samples = contract.get("train_sample_count")
    seed = contract.get("seed")
    diagnostics = contract.get("diagnostics")
    serialization = contract.get("serialization")
    priors = contract.get("effective_priors")
    expected_params = (
        lda_primitive_params(dimension, seed)
        if isinstance(dimension, int)
        and not isinstance(dimension, bool)
        and dimension > 0
        and isinstance(seed, int)
        and not isinstance(seed, bool)
        else None
    )
    digest = contract.get("upstream_feature_contract_sha256")
    valid_digest = (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )
    valid_diagnostics = (
        isinstance(diagnostics, dict)
        and diagnostics.get("converged") is True
        and isinstance(diagnostics.get("n_iter"), int)
        and 1 <= diagnostics["n_iter"] < LDA_MAX_ITER
        and diagnostics.get("max_iter") == LDA_MAX_ITER
        and all(
            isinstance(diagnostics.get(key), (int, float))
            and not isinstance(diagnostics.get(key), bool)
            and math.isfinite(float(diagnostics[key]))
            and float(diagnostics[key]) > 0.0
            for key in (
                "bound_perplexity",
                "component_min",
                "component_max",
                "component_sum",
            )
        )
    )
    expected_prior = 1.0 / dimension if isinstance(dimension, int) and dimension > 0 else None
    valid_priors = (
        isinstance(priors, dict)
        and all(
            isinstance(priors.get(key), (int, float)) and not isinstance(priors.get(key), bool)
            for key in ("doc_topic_prior", "topic_word_prior")
        )
        and priors.get("doc_topic_prior") == expected_prior
        and priors.get("topic_word_prior") == expected_prior
    )
    raw_memory_limit = contract.get("dense_memory_limit_gib")
    try:
        memory_limit = validate_lda_dense_memory_limit_gib(raw_memory_limit)
    except LDAContractError:
        memory_limit = None
    if (
        contract.get("contract_version") != LDA_CONTRACT_VERSION
        or contract.get("algorithm") != "latent_dirichlet_allocation"
        or contract.get("public_name") != "LDA"
        or contract.get("variant") != LDA_VARIANT
        or contract.get("status") != "verified"
        or (contract.get("route") != "single_sequence_classic_global_bow_v2"
            and not is_pair_bow_route(contract.get("route")))
        or contract.get("input_orientation")
        not in {"sequence_by_term", "sample_by_peptide_term_then_partner_term"}
        or (
            is_pair_bow_route(contract.get("route"))
            and contract.get("input_orientation") != "sample_by_peptide_term_then_partner_term"
        )
        or (
            contract.get("route") == "single_sequence_classic_global_bow_v2"
            and contract.get("input_orientation") != "sequence_by_term"
        )
        or contract.get("input_semantics") != "raw_nonnegative_integer_bow_counts"
        or contract.get("implementation") != "sklearn.decomposition.LatentDirichletAllocation"
        or not isinstance(contract.get("implementation_version"), str)
        or not contract.get("implementation_version")
        or contract.get("inference_method") != "variational_bayes"
        or contract.get("feature_fit_scope") != LDA_FIT_SCOPE
        or contract.get("held_out_semantics") != "frozen_normalized_document_topic_posterior"
        or contract.get("output_normalization") != "probability_simplex"
        or contract.get("zero_document_semantics") != "symmetric_doc_topic_prior"
        or not isinstance(train_samples, int)
        or isinstance(train_samples, bool)
        or train_samples < 1
        or not isinstance(input_features, int)
        or isinstance(input_features, bool)
        or input_features < 1
        or expected_params is None
        or contract.get("estimator_params") != expected_params
        or not valid_priors
        or not valid_digest
        or not valid_diagnostics
        or memory_limit is None
        or not isinstance(raw_memory_limit, (int, float))
        or isinstance(raw_memory_limit, bool)
        or not isinstance(serialization, dict)
        or serialization.get("format") != "joblib"
        or serialization.get("requires_matching_sklearn_version") is not True
    ):
        raise LDAContractError(
            "lda_contract_invalid", "LDA artifact contract is incomplete or inconsistent"
        )
    if upstream_feature_contract is not None:
        validate_lda_upstream_contract(
            upstream_feature_contract,
            input_feature_count=input_features,
        )
        if canonical_lda_contract_digest(upstream_feature_contract) != digest:
            raise LDAContractError(
                "lda_upstream_contract_mismatch",
                "LDA upstream BoW contract does not match the fitted artifact",
            )
    elif is_pair_bow_route(contract.get("route")):
        raise LDAContractError(
            "lda_upstream_contract_missing",
            "Pair LDA artifacts require the persisted Two-stream BoW joint contract",
        )
    if require_current_sklearn and contract.get("implementation_version") != sklearn.__version__:
        raise LDAContractError(
            "lda_runtime_version_mismatch",
            "LDA artifact was trained with a different scikit-learn version",
            details={
                "trained_version": contract.get("implementation_version"),
                "runtime_version": sklearn.__version__,
            },
        )
    if estimator is not None:
        if not isinstance(estimator, LatentDirichletAllocation):
            raise LDAContractError(
                "lda_reducer_type_invalid",
                "LDA artifact reducer must be sklearn.decomposition.LatentDirichletAllocation",
            )
        if estimator.get_params(deep=False) != expected_params:
            raise LDAContractError(
                "lda_reducer_contract_mismatch",
                "LDA reducer parameters do not match the artifact contract",
            )
        actual_diagnostics = _diagnostics(
            estimator,
            n_components=dimension,
            input_feature_count=input_features,
        )
        for key, expected in diagnostics.items():
            actual = actual_diagnostics.get(key)
            if isinstance(expected, float):
                if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise LDAContractError(
                        "lda_reducer_contract_mismatch",
                        "LDA reducer diagnostics do not match the artifact contract",
                    )
            elif actual != expected:
                raise LDAContractError(
                    "lda_reducer_contract_mismatch",
                    "LDA reducer diagnostics do not match the artifact contract",
                )
    return contract


def transform_lda_held_out(
    estimator: LatentDirichletAllocation,
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
        topics=output_dimension,
        expected_features=input_feature_count,
        dense_memory_limit_gib=dense_memory_limit_gib,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            transformed = estimator.transform(matrix)
        except Exception as exc:
            raise LDAContractError(
                "lda_transform_failed",
                f"LDA failed to transform the {split} matrix: {exc}",
                details={"split": split},
            ) from exc
    if caught:
        raise LDAContractError(
            "lda_transform_warning",
            f"LDA emitted a warning while transforming the {split} matrix",
            details={"split": split, "warnings": [str(item.message) for item in caught]},
        )
    values = np.asarray(transformed)
    if (
        values.shape != (rows, output_dimension)
        or not np.isfinite(values).all()
        or (values < 0.0).any()
        or not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)
    ):
        raise LDAContractError(
            "lda_transform_output_invalid",
            f"LDA produced an invalid {split} document-topic distribution",
            details={"split": split, "shape": list(values.shape)},
        )
    return values


def fit_lda_transform(
    train_matrix: object,
    validation_matrix: object,
    test_matrix: object | None = None,
    *,
    n_components: object,
    seed: int,
    dense_memory_limit_gib: object = LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    upstream_feature_contract: object,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    LatentDirichletAllocation,
    dict[str, Any],
]:
    dimension = validate_lda_topics(n_components)
    limit = validate_lda_dense_memory_limit_gib(dense_memory_limit_gib)
    train_rows, input_features = _validate_matrix(
        train_matrix,
        split="train",
        topics=dimension,
        dense_memory_limit_gib=limit,
        require_total_count=True,
    )
    upstream = validate_lda_upstream_contract(
        upstream_feature_contract,
        input_feature_count=input_features,
    )
    if is_pair_bow_route(upstream.get("route")):
        for matrix, split in ((train_matrix, "train"), (validation_matrix, "validation")):
            try:
                validate_pair_bow_matrix(
                    matrix,
                    split=split,
                    input_feature_count=input_features,
                    expected_sample_count=(
                        upstream["train_sample_count"] if split == "train" else None
                    ),
                )
            except PairBoWContractError as exc:
                raise LDAContractError(exc.code, str(exc), details=exc.details) from exc
        if test_matrix is not None:
            try:
                validate_pair_bow_matrix(
                    test_matrix,
                    split="test",
                    input_feature_count=input_features,
                )
            except PairBoWContractError as exc:
                raise LDAContractError(exc.code, str(exc), details=exc.details) from exc
    _validate_matrix(
        validation_matrix,
        split="validation",
        topics=dimension,
        expected_features=input_features,
        dense_memory_limit_gib=limit,
    )
    if test_matrix is not None:
        _validate_matrix(
            test_matrix,
            split="test",
            topics=dimension,
            expected_features=input_features,
            dense_memory_limit_gib=limit,
        )
    estimator = _new_estimator(dimension, int(seed))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            estimator.fit(train_matrix)
        except Exception as exc:
            raise LDAContractError("lda_fit_failed", f"LDA fitting failed: {exc}") from exc
    if caught:
        raise LDAContractError(
            "lda_fit_warning",
            "LDA fitting emitted a warning and the fitted state was rejected",
            details={"warnings": [str(item.message) for item in caught]},
        )
    contract = build_lda_contract(
        estimator,
        train_sample_count=train_rows,
        input_feature_count=input_features,
        n_components=dimension,
        seed=int(seed),
        dense_memory_limit_gib=limit,
        upstream_feature_contract=upstream,
    )
    train_output = transform_lda_held_out(
        estimator,
        train_matrix,
        split="train",
        input_feature_count=input_features,
        output_dimension=dimension,
        dense_memory_limit_gib=limit,
    )
    validation_output = transform_lda_held_out(
        estimator,
        validation_matrix,
        split="validation",
        input_feature_count=input_features,
        output_dimension=dimension,
        dense_memory_limit_gib=limit,
    )
    test_output = None
    if test_matrix is not None:
        test_output = transform_lda_held_out(
            estimator,
            test_matrix,
            split="test",
            input_feature_count=input_features,
            output_dimension=dimension,
            dense_memory_limit_gib=limit,
        )
    validate_lda_contract(
        contract,
        estimator=estimator,
        upstream_feature_contract=upstream,
    )
    return train_output, validation_output, test_output, estimator, contract
