"""Contract and matrix validation for the strict Pair BoW topic route."""

from __future__ import annotations

from pepplatform.stream_compat import normalize_stream_fields, normalize_stream_namespace, is_legacy_pair_bow_route
from pepplatform.task_compat import normalize_task_type


import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
from scipy import sparse


PAIR_BOW_CONTRACT_VERSION = 2
PAIR_BOW_ROUTE = "pair_two_stream_classic_bow_v2"
PAIR_BOW_ROLE_ORDER = ("peptide", "partner")
PAIR_BOW_FUSION_MODE = "concat"
PAIR_BOW_CLASSIFIERS = frozenset({"svm", "rf", "extratrees", "histgb", "xgb", "lgb", "logreg"})
PAIR_TASK_TYPES = frozenset({"pep_protein_interaction", "pep_hlaii", "pep_hlai"})


class PairBoWContractError(ValueError):
    """Raised when the strict Pair topic route is not contract-valid."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})


def pair_bow_contract_digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read(source: object, key: str, default: object = "") -> object:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def validate_pair_bow_route(source: object, *, pair: bool | None = None) -> None:
    """Validate only the static route shape; data contracts are checked later."""
    reduction = str(_read(source, "reduction", "none") or "none").strip().lower()
    if reduction not in {"lda", "plsa"}:
        return
    task_type = normalize_task_type(_read(source, "task_type", "") or "")
    is_pair = normalize_task_type(task_type) in PAIR_TASK_TYPES if pair is None else bool(pair)
    if not is_pair:
        return
    encoder = str(_read(source, "encoder", "") or "").strip().lower()
    streams = _read(source, "streams", None)
    stream_encoders = []
    if isinstance(streams, Mapping):
        stream_encoders = [
            str((streams.get(role) or {}).get("encoder") or "").strip().lower()
            for role in PAIR_BOW_ROLE_ORDER
        ]
    if (stream_encoders and stream_encoders != ["bow", "bow"]) or (
        not stream_encoders and encoder != "bow"
    ):
        raise PairBoWContractError(
            "pair_bow_encoder_unsupported",
            "Pair LDA/PLSA requires encoder=bow on both streams",
            details={"encoder": encoder, "stream_encoders": stream_encoders},
        )
    strategy = str(_read(source, "pair_strategy", "two_stream") or "").strip().lower()
    interaction = str(_read(source, "interaction_level", "vector") or "").strip().lower()
    predictor = str(_read(source, "predictor_type", "classifier") or "").strip().lower()
    classifier = (
        str(_read(source, "classifier", "") or _read(source, "model", "") or "").strip().lower()
    )
    fusion = str(_read(source, "fusion_mode", "concat") or "concat").strip().lower()
    if strategy != "two_stream" or interaction != "vector" or predictor != "classifier":
        raise PairBoWContractError(
            "pair_bow_route_shape_unsupported",
            "Pair LDA/PLSA requires two_stream vector interaction with a classical classifier",
        )
    if fusion != PAIR_BOW_FUSION_MODE:
        raise PairBoWContractError(
            "pair_bow_fusion_unsupported",
            "Pair LDA/PLSA requires fusion_mode=concat",
            details={"fusion_mode": fusion},
        )
    if classifier not in PAIR_BOW_CLASSIFIERS:
        raise PairBoWContractError(
            "pair_bow_classifier_unsupported",
            "Pair LDA/PLSA supports only classical classifiers",
            details={"classifier": classifier},
        )


def _validate_role_contract(role: str, contract: object) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise PairBoWContractError(
            "pair_bow_role_contract_missing",
            f"Pair {role} stream is missing its BoW-v2 contract",
        )
    spec = contract.get("spec") if isinstance(contract.get("spec"), Mapping) else {}
    encoder = str(contract.get("encoder") or spec.get("encoder") or "").strip().lower()
    tokenizer = str(contract.get("tokenizer") or spec.get("tokenizer") or "").strip().lower()
    frequency = contract.get("frequency_encoder_contract")
    if encoder != "bow" or tokenizer not in {"kmer", "dr", "top-n-gram", "dt"}:
        raise PairBoWContractError(
            "pair_bow_role_encoder_invalid",
            f"Pair {role} stream must use a classic BoW encoder",
            details={"role": role, "encoder": encoder, "tokenizer": tokenizer},
        )
    if not isinstance(frequency, Mapping):
        raise PairBoWContractError(
            "pair_bow_role_contract_invalid",
            f"Pair {role} stream is missing frequency_encoder_contract",
        )
    checks = {
        "contract_version": 2,
        "encoder": "bow",
        "stream_policy": "full_token_stream",
        "bow_formula": "x_j = c_j",
        "post_normalization": "none",
    }
    if any(frequency.get(key) != expected for key, expected in checks.items()):
        raise PairBoWContractError(
            "pair_bow_role_contract_invalid",
            f"Pair {role} stream must use a Train-scoped raw-count BoW-v2 contract",
            details={"role": role},
        )
    allowed_scopes = {
        "peptide": {"peptide_role", "combined_pair_roles"},
        "partner": {"partner_role", "protein_role", "combined_pair_roles"},
    }
    if frequency.get("fit_scope") not in allowed_scopes[role]:
        raise PairBoWContractError(
            "pair_bow_role_contract_invalid",
            f"Pair {role} stream must be fitted on its training role split",
            details={"role": role, "fit_scope": frequency.get("fit_scope")},
        )
    dimension = frequency.get("feature_dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise PairBoWContractError(
            "pair_bow_role_width_invalid",
            f"Pair {role} stream has an invalid BoW feature dimension",
        )
    if not _is_digest(frequency.get("vocabulary_sha256")) or not _is_digest(
        frequency.get("training_corpus_sha256")
    ):
        raise PairBoWContractError(
            "pair_bow_role_digest_invalid",
            f"Pair {role} stream is missing a vocabulary or corpus digest",
        )
    document_count = frequency.get("training_document_count")
    if (
        not isinstance(document_count, int)
        or isinstance(document_count, bool)
        or document_count < 1
    ):
        raise PairBoWContractError(
            "pair_bow_role_document_count_mismatch",
            f"Pair {role} stream must record its positive training document count",
            details={"role": role, "training_document_count": document_count},
        )
    return dict(contract)


def build_pair_bow_joint_contract(
    peptide_contract: Mapping[str, Any],
    partner_contract: Mapping[str, Any],
    *,
    train_sample_count: int,
    ordered_pair_train_sha256: str,
    fused_feature_dimension: int | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(train_sample_count, int)
        or isinstance(train_sample_count, bool)
        or train_sample_count < 1
    ):
        raise PairBoWContractError(
            "pair_bow_sample_count_invalid",
            "train_sample_count must be a positive integer",
        )
    if not _is_digest(ordered_pair_train_sha256):
        raise PairBoWContractError(
            "pair_bow_pair_digest_invalid",
            "ordered_pair_train_sha256 must be a lowercase SHA-256 digest",
        )
    roles = {
        "peptide": _validate_role_contract("peptide", peptide_contract),
        "partner": _validate_role_contract("partner", partner_contract),
    }
    scopes = {
        role: roles[role]["frequency_encoder_contract"]["fit_scope"] for role in PAIR_BOW_ROLE_ORDER
    }
    combined_scope = [scope == "combined_pair_roles" for scope in scopes.values()]
    if any(combined_scope) and not all(combined_scope):
        raise PairBoWContractError(
            "pair_bow_role_contract_invalid",
            "Pair BoW streams must both use role-specific fitting or the same combined-role fitting",
            details={"fit_scopes": scopes},
        )
    if all(combined_scope) and roles["peptide"] != roles["partner"]:
        raise PairBoWContractError(
            "pair_bow_role_contract_invalid",
            "Combined-role Pair BoW streams must share the same fitted feature contract",
        )
    expected_document_count = train_sample_count * (2 if all(combined_scope) else 1)
    for role in PAIR_BOW_ROLE_ORDER:
        actual_document_count = roles[role]["frequency_encoder_contract"]["training_document_count"]
        if actual_document_count != expected_document_count:
            raise PairBoWContractError(
                "pair_bow_role_document_count_mismatch",
                f"Pair {role} stream training document count does not match the joint Train split",
                details={
                    "role": role,
                    "expected": expected_document_count,
                    "actual": actual_document_count,
                },
            )
    dimensions = {
        role: int(roles[role]["frequency_encoder_contract"]["feature_dimension"])
        for role in PAIR_BOW_ROLE_ORDER
    }
    expected_width = dimensions["peptide"] + dimensions["partner"]
    if fused_feature_dimension is not None and (
        not isinstance(fused_feature_dimension, int) or isinstance(fused_feature_dimension, bool)
    ):
        raise PairBoWContractError(
            "pair_bow_fused_width_mismatch",
            "Pair concat width must be an integer",
        )
    width = expected_width if fused_feature_dimension is None else fused_feature_dimension
    if width != expected_width:
        raise PairBoWContractError(
            "pair_bow_fused_width_mismatch",
            "Pair concat width does not equal the two role BoW widths",
            details={"expected": expected_width, "actual": width},
        )
    contract: dict[str, Any] = {
        "contract_version": PAIR_BOW_CONTRACT_VERSION,
        "route": PAIR_BOW_ROUTE,
        "roles": list(PAIR_BOW_ROLE_ORDER),
        "fusion_mode": PAIR_BOW_FUSION_MODE,
        "input_orientation": "sample_by_peptide_term_then_partner_term",
        "input_semantics": "raw_nonnegative_integer_bow_counts",
        "role_contracts": roles,
        "role_feature_dimensions": dimensions,
        "fused_feature_dimension": width,
        "role_contract_sha256": {
            role: pair_bow_contract_digest(roles[role]) for role in PAIR_BOW_ROLE_ORDER
        },
        "train_sample_count": train_sample_count,
        "ordered_pair_train_sha256": ordered_pair_train_sha256,
    }
    return contract


def validate_pair_bow_joint_contract(
    contract: object, *, input_feature_count: int | None = None
) -> dict[str, Any]:
    if isinstance(contract, Mapping) and contract.get("contract_version") == 1:
        from pepplatform._legacy_stream.pair_bow import validate_pair_bow_joint_contract as validate_legacy
        from pepplatform._legacy_stream.pair_bow import PairBoWContractError as LegacyError
        try:
            return validate_legacy(contract, input_feature_count=input_feature_count)
        except LegacyError as exc:
            raise PairBoWContractError(exc.code, str(exc), details=exc.details) from exc
    if not isinstance(contract, Mapping):
        raise PairBoWContractError(
            "pair_bow_contract_missing", "Pair BoW joint contract is missing"
        )
    if (
        contract.get("contract_version") != PAIR_BOW_CONTRACT_VERSION
        or contract.get("route") != PAIR_BOW_ROUTE
    ):
        raise PairBoWContractError(
            "pair_bow_contract_invalid", "Pair BoW joint contract version or route is invalid"
        )
    if (
        tuple(contract.get("roles") or ()) != PAIR_BOW_ROLE_ORDER
        or contract.get("fusion_mode") != PAIR_BOW_FUSION_MODE
    ):
        raise PairBoWContractError(
            "pair_bow_contract_invalid",
            "Pair BoW contract must preserve peptide, partner role order and concat fusion",
        )
    dimensions = contract.get("role_feature_dimensions")
    roles = contract.get("role_contracts")
    if not isinstance(dimensions, Mapping) or not isinstance(roles, Mapping):
        raise PairBoWContractError(
            "pair_bow_contract_invalid", "Pair BoW contract is missing role contracts"
        )
    train_sample_count = contract.get("train_sample_count")
    if (
        not isinstance(train_sample_count, int)
        or isinstance(train_sample_count, bool)
        or train_sample_count < 1
    ):
        raise PairBoWContractError(
            "pair_bow_sample_count_invalid",
            "Pair BoW joint contract must record a positive training sample count",
        )
    ordered_digest = contract.get("ordered_pair_train_sha256")
    if not _is_digest(ordered_digest):
        raise PairBoWContractError(
            "pair_bow_pair_digest_invalid",
            "Pair BoW joint contract must record the ordered Pair Train digest",
        )
    rebuilt = build_pair_bow_joint_contract(
        roles.get("peptide"),
        roles.get("partner"),
        train_sample_count=train_sample_count,
        ordered_pair_train_sha256=ordered_digest,
        fused_feature_dimension=contract.get("fused_feature_dimension"),
    )
    if dict(contract) != rebuilt:
        raise PairBoWContractError(
            "pair_bow_contract_invalid", "Pair BoW joint contract is incomplete or inconsistent"
        )
    if (
        input_feature_count is not None
        and int(input_feature_count) != rebuilt["fused_feature_dimension"]
    ):
        raise PairBoWContractError(
            "pair_bow_fused_width_mismatch",
            "Pair BoW input width does not match its joint contract",
        )
    return dict(contract)


def validate_pair_bow_matrix(
    matrix: object,
    *,
    split: str,
    input_feature_count: int,
    expected_sample_count: int | None = None,
) -> tuple[int, int]:
    shape = getattr(matrix, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise PairBoWContractError(
            "pair_bow_matrix_shape_invalid", f"Pair BoW {split} matrix must be two-dimensional"
        )
    rows, columns = int(shape[0]), int(shape[1])
    if rows <= 0 or columns != int(input_feature_count):
        raise PairBoWContractError(
            "pair_bow_matrix_width_mismatch",
            f"Pair BoW {split} matrix width does not match its contract",
        )
    if expected_sample_count is not None and rows != int(expected_sample_count):
        raise PairBoWContractError(
            "pair_bow_sample_count_mismatch",
            f"Pair BoW {split} matrix row count does not match its joint contract",
            details={"expected": int(expected_sample_count), "actual": rows},
        )
    dtype = np.dtype(getattr(matrix, "dtype", np.float64))
    if not np.issubdtype(dtype, np.number):
        raise PairBoWContractError(
            "pair_bow_matrix_dtype_invalid", f"Pair BoW {split} matrix must be numeric"
        )
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    try:
        valid = (
            bool(np.isfinite(values).all())
            and bool((values >= 0).all())
            and bool(np.equal(values, np.rint(values)).all())
        )
    except TypeError as exc:
        raise PairBoWContractError(
            "pair_bow_matrix_dtype_invalid", f"Pair BoW {split} matrix must contain counts"
        ) from exc
    if not valid:
        raise PairBoWContractError(
            "pair_bow_matrix_not_counts",
            f"Pair BoW {split} matrix must contain finite nonnegative integer counts",
        )
    return rows, columns
