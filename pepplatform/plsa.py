"""Contracted probabilistic latent semantic analysis for BoW counts."""

from __future__ import annotations

from pepplatform.stream_compat import is_pair_bow_route
from pepplatform.task_compat import normalize_task_type


import hashlib
import json
import math
from typing import Any

import numpy as np
import scipy
from scipy import sparse
from pepplatform.pair_bow import (
    PAIR_BOW_ROUTE,
    PairBoWContractError,
    validate_pair_bow_joint_contract,
    validate_pair_bow_matrix,
    validate_pair_bow_route,
)


PLSA_CONTRACT_VERSION = 2
PLSA_VARIANT = "train_only_em_bow_counts"
PLSA_DEFAULT_MEMORY_LIMIT_GIB = 4.0
PLSA_FIT_SCOPE = "train_only"
PLSA_MAX_ITER = 400
PLSA_TOL = 1e-5
PLSA_NNZ_CHUNK_SIZE = 16_384
_PAIR_TASK_TYPES = frozenset({"pep_protein_interaction", "pep_hlaii", "pep_hlai"})


class PLSAContractError(ValueError):
    """Raised when a PLSA route, fit, transform, or artifact is invalid."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def validate_plsa_topics(value: object) -> int:
    if isinstance(value, bool):
        raise PLSAContractError("plsa_topics_invalid", "plsa_topics must be a positive integer")
    try:
        topics = int(value)
    except (TypeError, ValueError) as exc:
        raise PLSAContractError(
            "plsa_topics_invalid", "plsa_topics must be a positive integer"
        ) from exc
    if topics <= 0 or str(value).strip() not in {str(topics), f"{topics}.0"}:
        raise PLSAContractError("plsa_topics_invalid", "plsa_topics must be a positive integer")
    return topics


def validate_plsa_memory_limit_gib(value: object) -> float:
    try:
        limit = float(value)
    except (TypeError, ValueError) as exc:
        raise PLSAContractError(
            "plsa_memory_limit_invalid",
            "plsa_memory_limit_gib must be a positive finite number",
        ) from exc
    if not math.isfinite(limit) or limit <= 0.0:
        raise PLSAContractError(
            "plsa_memory_limit_invalid",
            "plsa_memory_limit_gib must be a positive finite number",
        )
    return limit


def validate_plsa_route(source: object, *, pair: bool | None = None) -> None:
    def read(key: str, default: object = "") -> object:
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    reduction = str(read("reduction", "none") or "none").strip().lower()
    if reduction != "plsa":
        return
    task_type = normalize_task_type(read("task_type", "") or "")
    is_pair = normalize_task_type(task_type) in _PAIR_TASK_TYPES if pair is None else bool(pair)
    if is_pair:
        if (
            not str(read("classifier", "") or "").strip()
            or not str(read("pair_strategy", "") or "").strip()
        ):
            raise PLSAContractError(
                "plsa_pair_route_unsupported",
                "reduction=plsa is not supported for incomplete Pair workflows because fused vectors are not a verified BoW route; provide the strict Pair Two-stream BoW route",
            )
        try:
            validate_pair_bow_route(source, pair=True)
        except PairBoWContractError as exc:
            raise PLSAContractError(exc.code, str(exc), details=exc.details) from exc
        validate_plsa_topics(read("plsa_topics", 50))
        validate_plsa_memory_limit_gib(read("plsa_memory_limit_gib", PLSA_DEFAULT_MEMORY_LIMIT_GIB))
        return
    encoder = str(read("encoder", "") or "").strip().lower()
    if encoder != "bow":
        raise PLSAContractError(
            "plsa_encoder_unsupported",
            "reduction=plsa requires encoder=bow with the current raw-count contract",
            details={"encoder": encoder},
        )
    validate_plsa_topics(read("plsa_topics", 50))
    validate_plsa_memory_limit_gib(read("plsa_memory_limit_gib", PLSA_DEFAULT_MEMORY_LIMIT_GIB))


def canonical_plsa_contract_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_plsa_upstream_contract(
    upstream_contract: object,
    *,
    input_feature_count: int | None = None,
) -> dict[str, Any]:
    if not isinstance(upstream_contract, dict):
        raise PLSAContractError(
            "plsa_upstream_contract_missing",
            "PLSA requires the persisted classic BoW v2 upstream contract",
        )
    if is_pair_bow_route(upstream_contract.get("route")):
        try:
            return validate_pair_bow_joint_contract(
                upstream_contract, input_feature_count=input_feature_count
            )
        except PairBoWContractError as exc:
            raise PLSAContractError(exc.code, str(exc), details=exc.details) from exc
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
        raise PLSAContractError(
            "plsa_upstream_contract_invalid",
            "PLSA requires a Train-scoped BoW v2 raw-count feature contract",
        )
    feature_dimension = frequency.get("feature_dimension")
    if (
        not isinstance(feature_dimension, int)
        or isinstance(feature_dimension, bool)
        or feature_dimension < 1
        or (input_feature_count is not None and feature_dimension != int(input_feature_count))
    ):
        raise PLSAContractError(
            "plsa_upstream_feature_width_mismatch",
            "PLSA input width does not match the BoW v2 feature contract",
            details={
                "contract_feature_count": feature_dimension,
                "input_feature_count": input_feature_count,
            },
        )
    for key in ("vocabulary_sha256", "training_corpus_sha256"):
        value = frequency.get(key)
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            raise PLSAContractError(
                "plsa_upstream_contract_invalid",
                "PLSA BoW v2 contract is missing a valid vocabulary or training-corpus digest",
            )
    return upstream_contract


def _matrix_shape(matrix: object, *, split: str) -> tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise PLSAContractError(
            "plsa_matrix_shape_invalid",
            f"PLSA {split} matrix must be two-dimensional",
            details={"split": split, "shape": list(shape or [])},
        )
    rows, columns = int(shape[0]), int(shape[1])
    if rows <= 0 or columns <= 0:
        raise PLSAContractError(
            "plsa_matrix_empty",
            f"PLSA {split} matrix must contain at least one row and one feature",
            details={"split": split, "shape": [rows, columns]},
        )
    return rows, columns


def _estimated_working_set_gib(
    matrix: object,
    *,
    rows: int,
    columns: int,
    topics: int,
) -> float:
    if sparse.issparse(matrix):
        nnz = int(matrix.nnz)
        input_bytes = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    else:
        values = np.asarray(matrix)
        nnz = int(np.count_nonzero(values))
        input_bytes = int(values.nbytes)
    csr_copy_bytes = nnz * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize)
    csr_copy_bytes += (rows + 1) * np.dtype(np.int32).itemsize
    state_bytes = 2 * (rows * topics + topics * columns) * np.dtype(np.float64).itemsize
    chunk = min(max(nnz, 1), PLSA_NNZ_CHUNK_SIZE)
    responsibility_bytes = 3 * chunk * topics * np.dtype(np.float64).itemsize
    return (input_bytes + csr_copy_bytes + state_bytes + responsibility_bytes) / float(1024**3)


def _validate_matrix(
    matrix: object,
    *,
    split: str,
    topics: int,
    memory_limit_gib: float,
    expected_features: int | None = None,
    require_total_count: bool = False,
) -> tuple[sparse.csr_matrix, dict[str, int | float]]:
    rows, columns = _matrix_shape(matrix, split=split)
    if expected_features is not None and columns != expected_features:
        raise PLSAContractError(
            "plsa_feature_width_mismatch",
            f"PLSA {split} matrix has {columns} features; expected {expected_features}",
            details={
                "split": split,
                "expected_features": expected_features,
                "actual_features": columns,
            },
        )
    try:
        dtype = np.dtype(getattr(matrix, "dtype", np.float64))
    except TypeError:
        dtype = np.dtype(np.float64)
    if not np.issubdtype(dtype, np.number):
        raise PLSAContractError(
            "plsa_matrix_dtype_invalid",
            f"PLSA {split} matrix must contain numeric BoW counts",
        )
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    try:
        finite = bool(np.isfinite(values).all())
        nonnegative = bool((values >= 0).all())
        integral = bool(np.equal(values, np.rint(values)).all())
    except TypeError as exc:
        raise PLSAContractError(
            "plsa_matrix_dtype_invalid",
            f"PLSA {split} matrix must contain numeric BoW counts",
        ) from exc
    if not finite:
        raise PLSAContractError(
            "plsa_matrix_nonfinite", f"PLSA {split} matrix contains NaN or infinity"
        )
    if not nonnegative:
        raise PLSAContractError(
            "plsa_matrix_negative", f"PLSA {split} matrix contains negative values"
        )
    if not integral:
        raise PLSAContractError(
            "plsa_matrix_not_counts",
            f"PLSA {split} matrix must contain integer-valued BoW counts",
        )
    total = float(values.sum())
    if require_total_count and total <= 0.0:
        raise PLSAContractError(
            "plsa_training_counts_empty",
            "PLSA training matrix must contain at least one observed token count",
        )
    estimate = _estimated_working_set_gib(matrix, rows=rows, columns=columns, topics=topics)
    if estimate > memory_limit_gib:
        raise PLSAContractError(
            "plsa_memory_limit_exceeded",
            f"PLSA {split} working-set estimate is {estimate:.3f} GiB, exceeding plsa_memory_limit_gib={memory_limit_gib:g}",
            details={
                "split": split,
                "estimated_working_set_gib": estimate,
                "limit_gib": memory_limit_gib,
            },
        )
    try:
        csr = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise PLSAContractError(
            "plsa_matrix_dtype_invalid",
            f"PLSA {split} matrix could not be converted to float64 CSR counts",
        ) from exc
    csr.sum_duplicates()
    csr.eliminate_zeros()
    csr.sort_indices()
    row_counts = np.asarray(csr.sum(axis=1), dtype=np.float64).reshape(-1)
    column_counts = np.asarray(csr.sum(axis=0), dtype=np.float64).reshape(-1)
    return csr, {
        "rows": rows,
        "columns": columns,
        "nnz": int(csr.nnz),
        "zero_documents": int(np.count_nonzero(row_counts == 0.0)),
        "zero_terms": int(np.count_nonzero(column_counts == 0.0)),
        "estimated_working_set_gib": float(estimate),
    }


class PLSAEstimator:
    """Maximum-likelihood PLSA with EM training and frozen-topic fold-in."""

    def __init__(
        self,
        n_components: int,
        *,
        max_iter: int = PLSA_MAX_ITER,
        tol: float = PLSA_TOL,
        random_state: int = 42,
        memory_limit_gib: float = PLSA_DEFAULT_MEMORY_LIMIT_GIB,
        chunk_size: int = PLSA_NNZ_CHUNK_SIZE,
    ):
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.memory_limit_gib = float(memory_limit_gib)
        self.chunk_size = int(chunk_size)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "memory_limit_gib": self.memory_limit_gib,
            "chunk_size": self.chunk_size,
        }

    @staticmethod
    def _log_likelihood(
        matrix: sparse.csr_matrix,
        doc_topic: np.ndarray,
        topic_word: np.ndarray,
        *,
        split: str,
        chunk_size: int,
    ) -> float:
        coo = matrix.tocoo(copy=False)
        total = 0.0
        for start in range(0, coo.nnz, chunk_size):
            stop = min(start + chunk_size, coo.nnz)
            rows = coo.row[start:stop]
            columns = coo.col[start:stop]
            probabilities = np.sum(
                doc_topic[rows] * topic_word[:, columns].T,
                axis=1,
            )
            if not np.isfinite(probabilities).all() or (probabilities <= 0.0).any():
                raise PLSAContractError(
                    "plsa_probability_invalid",
                    f"PLSA produced zero or non-finite probability for an observed {split} term",
                    details={"split": split},
                )
            total += float(np.dot(coo.data[start:stop], np.log(probabilities)))
        if not math.isfinite(total):
            raise PLSAContractError(
                "plsa_likelihood_nonfinite",
                f"PLSA produced a non-finite {split} log-likelihood",
            )
        return total

    def _expectation_counts(
        self,
        matrix: sparse.csr_matrix,
        doc_topic: np.ndarray,
        topic_word: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        doc_expected = np.zeros_like(doc_topic, dtype=np.float64)
        topic_expected = np.zeros_like(topic_word, dtype=np.float64)
        coo = matrix.tocoo(copy=False)
        for start in range(0, coo.nnz, self.chunk_size):
            stop = min(start + self.chunk_size, coo.nnz)
            rows = coo.row[start:stop]
            columns = coo.col[start:stop]
            weights = doc_topic[rows] * topic_word[:, columns].T
            denominators = weights.sum(axis=1)
            if not np.isfinite(denominators).all() or (denominators <= 0.0).any():
                raise PLSAContractError(
                    "plsa_probability_invalid",
                    "PLSA E-step produced zero or non-finite observed-term probability",
                )
            responsibilities = weights / denominators[:, None]
            expected = coo.data[start:stop, None] * responsibilities
            np.add.at(doc_expected, rows, expected)
            for topic in range(self.n_components):
                np.add.at(topic_expected[topic], columns, expected[:, topic])
        return doc_expected, topic_expected

    def fit(self, matrix: object) -> "PLSAEstimator":
        counts, matrix_info = _validate_matrix(
            matrix,
            split="train",
            topics=self.n_components,
            memory_limit_gib=self.memory_limit_gib,
            require_total_count=True,
        )
        rows, columns = counts.shape
        rng = np.random.default_rng(self.random_state)
        doc_topic = rng.dirichlet(np.ones(self.n_components), size=rows)
        topic_word = rng.dirichlet(np.ones(columns), size=self.n_components)
        row_totals = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
        zero_rows = row_totals == 0.0
        doc_topic[zero_rows] = 1.0 / self.n_components
        history = [
            self._log_likelihood(
                counts,
                doc_topic,
                topic_word,
                split="train",
                chunk_size=self.chunk_size,
            )
        ]
        converged = False
        for iteration in range(1, self.max_iter + 1):
            doc_expected, topic_expected = self._expectation_counts(counts, doc_topic, topic_word)
            topic_totals = topic_expected.sum(axis=1)
            if not np.isfinite(topic_totals).all() or (topic_totals <= 0.0).any():
                raise PLSAContractError(
                    "plsa_topic_mass_invalid",
                    "PLSA M-step produced an empty or non-finite topic",
                )
            next_doc_topic = np.full_like(doc_topic, 1.0 / self.n_components)
            positive_rows = ~zero_rows
            next_doc_topic[positive_rows] = (
                doc_expected[positive_rows] / row_totals[positive_rows, None]
            )
            next_topic_word = topic_expected / topic_totals[:, None]
            likelihood = self._log_likelihood(
                counts,
                next_doc_topic,
                next_topic_word,
                split="train",
                chunk_size=self.chunk_size,
            )
            previous = history[-1]
            decrease_tolerance = 1e-10 * max(1.0, abs(previous))
            if likelihood < previous - decrease_tolerance:
                raise PLSAContractError(
                    "plsa_likelihood_decreased",
                    "PLSA EM decreased the conditional log-likelihood",
                    details={"previous": previous, "current": likelihood},
                )
            history.append(likelihood)
            doc_topic, topic_word = next_doc_topic, next_topic_word
            relative_scale = max(1.0, abs(previous))
            if likelihood - previous <= self.tol * relative_scale:
                converged = True
                break
        if not converged:
            raise PLSAContractError(
                "plsa_not_converged",
                f"PLSA did not satisfy its likelihood stopping rule within {self.max_iter} iterations",
                details={"max_iter": self.max_iter, "final_log_likelihood": history[-1]},
            )
        self.components_ = topic_word
        self.doc_topic_ = doc_topic
        self.n_features_in_ = int(columns)
        self.n_iter_ = int(iteration)
        self.converged_ = True
        self.log_likelihood_history_ = np.asarray(history, dtype=np.float64)
        self.training_log_likelihood_ = float(history[-1])
        self.training_matrix_info_ = dict(matrix_info)
        return self

    def _fold_in_row(
        self,
        indices: np.ndarray,
        data: np.ndarray,
        *,
        split: str,
        row_index: int,
    ) -> np.ndarray:
        if data.size == 0:
            return np.full(self.n_components, 1.0 / self.n_components)
        # A fixed Train-scoped BoW vocabulary can contain terms with no count in
        # the current fitting split (the full Train split or an inner-CV fold).
        # Such terms have zero probability under every fitted topic and are
        # therefore fold-local OOV, not evidence from which a document-topic
        # posterior can be inferred. Ignore them without smoothing or fitting
        # held-out data. A row containing only these terms follows the existing
        # zero-document contract.
        supported = np.any(self.components_[:, indices] > 0.0, axis=0)
        if not supported.all():
            indices = indices[supported]
            data = data[supported]
            if data.size == 0:
                return np.full(self.n_components, 1.0 / self.n_components)
        doc_topic = np.full(self.n_components, 1.0 / self.n_components)
        previous = None
        total_count = float(data.sum())
        for _iteration in range(1, self.max_iter + 1):
            expected = np.zeros(self.n_components, dtype=np.float64)
            likelihood = 0.0
            for start in range(0, data.size, self.chunk_size):
                stop = min(start + self.chunk_size, data.size)
                columns = indices[start:stop]
                probabilities = doc_topic @ self.components_[:, columns]
                if not np.isfinite(probabilities).all() or (probabilities <= 0.0).any():
                    raise PLSAContractError(
                        "plsa_unseen_term_probability",
                        f"PLSA cannot fold in {split} row {row_index} because an observed term has zero topic probability",
                        details={"split": split, "row_index": row_index},
                    )
                responsibilities = (
                    doc_topic[:, None] * self.components_[:, columns]
                ) / probabilities[None, :]
                expected += responsibilities @ data[start:stop]
                likelihood += float(np.dot(data[start:stop], np.log(probabilities)))
            next_doc_topic = expected / total_count
            if not np.isfinite(next_doc_topic).all() or (next_doc_topic < 0.0).any():
                raise PLSAContractError(
                    "plsa_transform_output_invalid",
                    f"PLSA produced an invalid {split} document-topic distribution",
                )
            if previous is not None:
                decrease_tolerance = 1e-10 * max(1.0, abs(previous))
                if likelihood < previous - decrease_tolerance:
                    raise PLSAContractError(
                        "plsa_fold_in_likelihood_decreased",
                        f"PLSA fold-in decreased likelihood for {split} row {row_index}",
                    )
                if likelihood - previous <= self.tol * max(1.0, abs(previous)):
                    return next_doc_topic
            previous = likelihood
            doc_topic = next_doc_topic
        raise PLSAContractError(
            "plsa_fold_in_not_converged",
            f"PLSA fold-in did not converge for {split} row {row_index} within {self.max_iter} iterations",
            details={"split": split, "row_index": row_index},
        )

    def transform(self, matrix: object, *, split: str = "inference") -> np.ndarray:
        if not getattr(self, "converged_", False):
            raise PLSAContractError("plsa_estimator_not_fitted", "PLSA estimator is not fitted")
        counts, _ = _validate_matrix(
            matrix,
            split=split,
            topics=self.n_components,
            memory_limit_gib=self.memory_limit_gib,
            expected_features=self.n_features_in_,
        )
        output = np.empty((counts.shape[0], self.n_components), dtype=np.float64)
        for row_index in range(counts.shape[0]):
            start, stop = counts.indptr[row_index : row_index + 2]
            output[row_index] = self._fold_in_row(
                counts.indices[start:stop],
                counts.data[start:stop],
                split=split,
                row_index=row_index,
            )
        _validate_output(output, rows=counts.shape[0], topics=self.n_components, split=split)
        return output

    def fit_transform(self, matrix: object) -> np.ndarray:
        self.fit(matrix)
        return np.asarray(self.doc_topic_, dtype=np.float64).copy()


def _validate_output(
    output: object,
    *,
    rows: int,
    topics: int,
    split: str,
) -> np.ndarray:
    values = np.asarray(output, dtype=np.float64)
    if (
        values.shape != (rows, topics)
        or not np.isfinite(values).all()
        or (values < 0.0).any()
        or not np.allclose(values.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9)
    ):
        raise PLSAContractError(
            "plsa_transform_output_invalid",
            f"PLSA produced an invalid {split} document-topic distribution",
            details={"split": split, "shape": list(values.shape)},
        )
    return values


def _estimator_params(n_components: int, seed: int, memory_limit_gib: float) -> dict[str, Any]:
    return {
        "n_components": int(n_components),
        "max_iter": PLSA_MAX_ITER,
        "tol": PLSA_TOL,
        "random_state": int(seed),
        "memory_limit_gib": float(memory_limit_gib),
        "chunk_size": PLSA_NNZ_CHUNK_SIZE,
    }


def build_plsa_contract(
    estimator: PLSAEstimator,
    *,
    train_sample_count: int,
    input_feature_count: int,
    n_components: int,
    seed: int,
    memory_limit_gib: float,
    upstream_feature_contract: dict[str, Any],
) -> dict[str, Any]:
    history = np.asarray(estimator.log_likelihood_history_, dtype=np.float64)
    diagnostics = {
        "converged": bool(estimator.converged_),
        "n_iter": int(estimator.n_iter_),
        "max_iter": PLSA_MAX_ITER,
        "initial_log_likelihood": float(history[0]),
        "final_log_likelihood": float(history[-1]),
        "minimum_likelihood_improvement": float(np.diff(history).min()),
        "topic_word_min": float(estimator.components_.min()),
        "topic_word_max": float(estimator.components_.max()),
        "topic_word_simplex_max_error": float(
            np.max(np.abs(estimator.components_.sum(axis=1) - 1.0))
        ),
        "document_topic_simplex_max_error": float(
            np.max(np.abs(estimator.doc_topic_.sum(axis=1) - 1.0))
        ),
        "zero_training_documents": int(estimator.training_matrix_info_["zero_documents"]),
        "zero_training_terms": int(estimator.training_matrix_info_["zero_terms"]),
    }
    pair_route = is_pair_bow_route(upstream_feature_contract.get("route"))
    return {
        "contract_version": PLSA_CONTRACT_VERSION,
        "algorithm": "probabilistic_latent_semantic_analysis",
        "public_name": "PLSA",
        "variant": PLSA_VARIANT,
        "status": "verified",
        "route": PAIR_BOW_ROUTE if pair_route else "single_sequence_classic_global_bow_v2",
        "input_orientation": (
            "sample_by_peptide_term_then_partner_term" if pair_route else "sequence_by_term"
        ),
        "input_semantics": "raw_nonnegative_integer_bow_counts",
        "implementation": "pepplatform.plsa.PLSAEstimator",
        "implementation_version": PLSA_CONTRACT_VERSION,
        "library_versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "objective": "conditional_multinomial_log_likelihood",
        "optimization": "maximum_likelihood_expectation_maximization",
        "initialization": "seeded_symmetric_dirichlet_alpha_1",
        "inference_method": "frozen_topic_word_em_fold_in",
        "feature_fit_scope": PLSA_FIT_SCOPE,
        "held_out_semantics": "frozen_normalized_document_topic_fold_in",
        "held_out_unseen_term_semantics": "ignore_zero_training_probability_terms",
        "output_normalization": "probability_simplex",
        "zero_document_semantics": "uniform_topic_distribution",
        "train_sample_count": int(train_sample_count),
        "input_feature_count": int(input_feature_count),
        "output_dimension": int(n_components),
        "seed": int(seed),
        "estimator_params": _estimator_params(n_components, seed, memory_limit_gib),
        "memory_limit_gib": float(memory_limit_gib),
        "upstream_feature_contract_sha256": canonical_plsa_contract_digest(
            upstream_feature_contract
        ),
        "topic_word_sha256": _array_sha256(estimator.components_),
        "diagnostics": diagnostics,
        "serialization": {
            "format": "joblib",
            "publication": "atomic_replace",
            "requires_matching_numpy_scipy_versions": True,
        },
    }


def validate_plsa_contract(
    contract: object,
    *,
    estimator: object | None = None,
    upstream_feature_contract: object | None = None,
    require_current_versions: bool = False,
) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise PLSAContractError("plsa_contract_missing", "PLSA artifact contract is missing")
    topics = contract.get("output_dimension")
    seed = contract.get("seed")
    memory_limit = contract.get("memory_limit_gib")
    expected_params = None
    try:
        valid_topics = validate_plsa_topics(topics)
        valid_memory_limit = validate_plsa_memory_limit_gib(memory_limit)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError
        expected_params = _estimator_params(valid_topics, seed, valid_memory_limit)
    except (PLSAContractError, ValueError, TypeError):
        valid_topics = None
    diagnostics = contract.get("diagnostics")
    serialization = contract.get("serialization")
    versions = contract.get("library_versions")
    digest = contract.get("upstream_feature_contract_sha256")
    topic_digest = contract.get("topic_word_sha256")
    valid_hashes = all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in (digest, topic_digest)
    )
    valid_diagnostics = (
        isinstance(diagnostics, dict)
        and diagnostics.get("converged") is True
        and isinstance(diagnostics.get("n_iter"), int)
        and 1 <= diagnostics["n_iter"] < PLSA_MAX_ITER
        and diagnostics.get("max_iter") == PLSA_MAX_ITER
        and all(
            isinstance(diagnostics.get(key), (int, float))
            and not isinstance(diagnostics.get(key), bool)
            and math.isfinite(float(diagnostics[key]))
            for key in (
                "initial_log_likelihood",
                "final_log_likelihood",
                "minimum_likelihood_improvement",
                "topic_word_min",
                "topic_word_max",
                "topic_word_simplex_max_error",
                "document_topic_simplex_max_error",
            )
        )
        and diagnostics.get("minimum_likelihood_improvement", -1.0) >= -1e-8
        and diagnostics.get("topic_word_min", -1.0) >= 0.0
        and diagnostics.get("topic_word_max", 0.0) > 0.0
        and diagnostics.get("topic_word_simplex_max_error", 1.0) <= 1e-8
        and diagnostics.get("document_topic_simplex_max_error", 1.0) <= 1e-8
        and isinstance(diagnostics.get("zero_training_documents"), int)
        and diagnostics["zero_training_documents"] >= 0
        and isinstance(diagnostics.get("zero_training_terms"), int)
        and diagnostics["zero_training_terms"] >= 0
    )
    if (
        contract.get("contract_version") != PLSA_CONTRACT_VERSION
        or contract.get("algorithm") != "probabilistic_latent_semantic_analysis"
        or contract.get("public_name") != "PLSA"
        or contract.get("variant") != PLSA_VARIANT
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
        or contract.get("implementation") != "pepplatform.plsa.PLSAEstimator"
        or contract.get("implementation_version") != PLSA_CONTRACT_VERSION
        or contract.get("objective") != "conditional_multinomial_log_likelihood"
        or contract.get("optimization") != "maximum_likelihood_expectation_maximization"
        or contract.get("initialization") != "seeded_symmetric_dirichlet_alpha_1"
        or contract.get("inference_method") != "frozen_topic_word_em_fold_in"
        or contract.get("feature_fit_scope") != PLSA_FIT_SCOPE
        or contract.get("held_out_semantics") != "frozen_normalized_document_topic_fold_in"
        or contract.get("held_out_unseen_term_semantics")
        != "ignore_zero_training_probability_terms"
        or contract.get("output_normalization") != "probability_simplex"
        or contract.get("zero_document_semantics") != "uniform_topic_distribution"
        or not isinstance(contract.get("train_sample_count"), int)
        or isinstance(contract.get("train_sample_count"), bool)
        or contract["train_sample_count"] < 1
        or not isinstance(contract.get("input_feature_count"), int)
        or isinstance(contract.get("input_feature_count"), bool)
        or contract["input_feature_count"] < 1
        or valid_topics is None
        or contract.get("estimator_params") != expected_params
        or not valid_hashes
        or not valid_diagnostics
        or not isinstance(versions, dict)
        or not isinstance(versions.get("numpy"), str)
        or not isinstance(versions.get("scipy"), str)
        or not isinstance(serialization, dict)
        or serialization.get("format") != "joblib"
        or serialization.get("publication") != "atomic_replace"
        or serialization.get("requires_matching_numpy_scipy_versions") is not True
    ):
        raise PLSAContractError(
            "plsa_contract_invalid",
            "PLSA artifact contract is incomplete or inconsistent",
        )
    if require_current_versions and (
        versions.get("numpy") != np.__version__ or versions.get("scipy") != scipy.__version__
    ):
        raise PLSAContractError(
            "plsa_runtime_version_mismatch",
            "PLSA artifact requires the recorded NumPy and SciPy versions",
            details={
                "trained": versions,
                "runtime": {"numpy": np.__version__, "scipy": scipy.__version__},
            },
        )
    if upstream_feature_contract is not None:
        validate_plsa_upstream_contract(
            upstream_feature_contract,
            input_feature_count=contract["input_feature_count"],
        )
        if canonical_plsa_contract_digest(upstream_feature_contract) != digest:
            raise PLSAContractError(
                "plsa_upstream_contract_mismatch",
                "PLSA upstream BoW contract does not match the fitted artifact",
            )
    elif is_pair_bow_route(contract.get("route")):
        raise PLSAContractError(
            "plsa_upstream_contract_missing",
            "Pair PLSA artifacts require the persisted Two-stream BoW joint contract",
        )
    if estimator is not None:
        if not isinstance(estimator, PLSAEstimator):
            raise PLSAContractError(
                "plsa_estimator_type_invalid",
                "PLSA artifact does not contain a PLSAEstimator",
            )
        if (
            estimator.get_params() != expected_params
            or not getattr(estimator, "converged_", False)
            or getattr(estimator, "n_features_in_", None) != contract["input_feature_count"]
            or np.asarray(getattr(estimator, "components_", [])).shape
            != (valid_topics, contract["input_feature_count"])
            or _array_sha256(estimator.components_) != topic_digest
        ):
            raise PLSAContractError(
                "plsa_estimator_contract_mismatch",
                "PLSA estimator state does not match its artifact contract",
            )
    return contract


def fit_plsa_transform(
    train_matrix: object,
    validation_matrix: object,
    test_matrix: object | None = None,
    *,
    n_components: object,
    seed: int,
    memory_limit_gib: object = PLSA_DEFAULT_MEMORY_LIMIT_GIB,
    upstream_feature_contract: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, PLSAEstimator, dict[str, Any]]:
    topics = validate_plsa_topics(n_components)
    limit = validate_plsa_memory_limit_gib(memory_limit_gib)
    train_counts, train_info = _validate_matrix(
        train_matrix,
        split="train",
        topics=topics,
        memory_limit_gib=limit,
        require_total_count=True,
    )
    upstream = validate_plsa_upstream_contract(
        upstream_feature_contract,
        input_feature_count=int(train_info["columns"]),
    )
    if is_pair_bow_route(upstream.get("route")):
        for matrix, split in ((train_matrix, "train"), (validation_matrix, "validation")):
            try:
                validate_pair_bow_matrix(
                    matrix,
                    split=split,
                    input_feature_count=int(train_info["columns"]),
                    expected_sample_count=(
                        upstream["train_sample_count"] if split == "train" else None
                    ),
                )
            except PairBoWContractError as exc:
                raise PLSAContractError(exc.code, str(exc), details=exc.details) from exc
        if test_matrix is not None:
            try:
                validate_pair_bow_matrix(
                    test_matrix,
                    split="test",
                    input_feature_count=int(train_info["columns"]),
                )
            except PairBoWContractError as exc:
                raise PLSAContractError(exc.code, str(exc), details=exc.details) from exc
    validation_counts, _ = _validate_matrix(
        validation_matrix,
        split="validation",
        topics=topics,
        memory_limit_gib=limit,
        expected_features=int(train_info["columns"]),
    )
    test_counts = None
    if test_matrix is not None:
        test_counts, _ = _validate_matrix(
            test_matrix,
            split="test",
            topics=topics,
            memory_limit_gib=limit,
            expected_features=int(train_info["columns"]),
        )
    estimator = PLSAEstimator(
        topics,
        random_state=int(seed),
        memory_limit_gib=limit,
    ).fit(train_counts)
    train_output = _validate_output(
        estimator.doc_topic_, rows=train_counts.shape[0], topics=topics, split="train"
    ).copy()
    validation_output = estimator.transform(validation_counts, split="validation")
    test_output = (
        estimator.transform(test_counts, split="test") if test_counts is not None else None
    )
    contract = build_plsa_contract(
        estimator,
        train_sample_count=train_counts.shape[0],
        input_feature_count=train_counts.shape[1],
        n_components=topics,
        seed=int(seed),
        memory_limit_gib=limit,
        upstream_feature_contract=upstream,
    )
    validate_plsa_contract(
        contract,
        estimator=estimator,
        upstream_feature_contract=upstream,
    )
    return train_output, validation_output, test_output, estimator, contract
