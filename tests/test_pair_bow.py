import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pepplatform.pair_bow import (
    PairBoWContractError,
    build_pair_bow_joint_contract,
    validate_pair_bow_joint_contract,
    validate_pair_bow_matrix,
    validate_pair_bow_route,
)
from pepplatform.lda import LDAContractError, fit_lda_transform, validate_lda_contract
from pepplatform.plsa import (
    PLSA_CONTRACT_VERSION,
    PLSAContractError,
    fit_plsa_transform,
    validate_plsa_contract,
)
from pepplatform.web.inference import ModelService
from scripts.train_pair_pipeline import _pair_vector_artifact_info


def _role(dimension, scope, document_count=4):
    return {
        "spec": {"tokenizer": "kmer", "encoder": "bow"},
        "frequency_encoder_contract": {
            "contract_version": 2,
            "encoder": "bow",
            "stream_policy": "full_token_stream",
            "bow_formula": "x_j = c_j",
            "post_normalization": "none",
            "frequency_max_raw_length": 50_000,
            "fit_scope": scope,
            "feature_dimension": dimension,
            "training_document_count": document_count,
            "vocabulary_sha256": "a" * 64,
            "training_corpus_sha256": "b" * 64,
        },
    }


def _contract(train_sample_count=4, legacy=False):
    builder = build_pair_bow_joint_contract
    if legacy:
        from pepplatform._legacy_stream.pair_bow import build_pair_bow_joint_contract as builder
    return builder(
        _role(3, "peptide_role", train_sample_count),
        _role(2, "partner_role", train_sample_count),
        train_sample_count=train_sample_count,
        ordered_pair_train_sha256="c" * 64,
        fused_feature_dimension=5,
    )


def _matrix():
    return np.asarray(
        [
            [2, 0, 1, 0, 0],
            [0, 3, 0, 1, 0],
            [1, 1, 2, 0, 1],
            [0, 1, 1, 3, 0],
            [3, 0, 0, 1, 1],
            [1, 2, 0, 2, 0],
        ],
        dtype=np.float64,
    )


def test_pair_bow_contract_preserves_roles_and_concat_width():
    contract = _contract()
    assert contract["route"] == "pair_two_stream_classic_bow_v2"
    assert contract["roles"] == ["peptide", "partner"]
    assert contract["fused_feature_dimension"] == 5
    assert validate_pair_bow_joint_contract(contract)["role_feature_dimensions"] == {
        "peptide": 3,
        "partner": 2,
    }


def test_pair_bow_contract_rejects_tampered_digest_and_width():
    contract = _contract()
    contract["ordered_pair_train_sha256"] = "not-a-digest"
    with pytest.raises(PairBoWContractError):
        validate_pair_bow_joint_contract(contract)
    contract = _contract()
    contract["role_contracts"]["peptide"]["frequency_encoder_contract"]["feature_dimension"] = 4
    with pytest.raises(PairBoWContractError):
        validate_pair_bow_joint_contract(contract)
    contract = _contract()
    with pytest.raises(PairBoWContractError):
        build_pair_bow_joint_contract(
            _role(3, "peptide_role"),
            _role(2, "partner_role"),
            train_sample_count=4,
            ordered_pair_train_sha256="c" * 64,
            fused_feature_dimension=4,
        )


@pytest.mark.parametrize(
    ("role", "scope"),
    [("peptide", "partner_role"), ("partner", "peptide_role")],
)
def test_pair_bow_contract_binds_role_scopes(role, scope):
    peptide = _role(3, "peptide_role")
    partner = _role(2, "partner_role")
    (peptide if role == "peptide" else partner)["frequency_encoder_contract"]["fit_scope"] = scope
    with pytest.raises(PairBoWContractError):
        build_pair_bow_joint_contract(
            peptide,
            partner,
            train_sample_count=4,
            ordered_pair_train_sha256="c" * 64,
            fused_feature_dimension=5,
        )


def test_pair_bow_contract_binds_role_training_document_counts():
    peptide = _role(3, "peptide_role")
    partner = _role(2, "partner_role")
    partner["frequency_encoder_contract"]["training_document_count"] = 3
    with pytest.raises(PairBoWContractError):
        build_pair_bow_joint_contract(
            peptide,
            partner,
            train_sample_count=4,
            ordered_pair_train_sha256="c" * 64,
            fused_feature_dimension=5,
        )


def test_pair_vector_artifact_builds_a_complete_combined_role_contract():
    shared = _role(3, "combined_pair_roles", 8)["frequency_encoder_contract"]
    args = SimpleNamespace(
        task_type="pep_protein_interaction",
        tokenizer="kmer",
        encoder="bow",
        model="rf",
        classifier="rf",
        pair_strategy="two_stream",
        interaction_level="vector",
        predictor_type="classifier",
        fusion_mode="concat",
        max_length=60,
        k=2,
        _frequency_encoder_contract=shared,
    )
    info = _pair_vector_artifact_info(
        args,
        torch.zeros((4, 3)),
        torch.zeros((4, 3)),
        torch.zeros((4, 6)),
        profile_rows=[(str(index), "AA", "BB", index % 2) for index in range(4)],
    )

    contract = validate_pair_bow_joint_contract(
        info["pair_bow_joint_contract"], input_feature_count=6
    )
    assert contract["role_contracts"]["peptide"] == contract["role_contracts"]["partner"]


def test_pair_vector_artifact_propagates_shared_contract_to_explicit_streams():
    shared = _role(3, "combined_pair_roles", 8)["frequency_encoder_contract"]
    stream = {
        "feature_route": "classic",
        "tokenizer": "kmer",
        "encoder": "bow",
        "max_length": 60,
        "params": {"k": 2},
    }
    args = SimpleNamespace(
        task_type="pep_protein_interaction",
        tokenizer="kmer",
        encoder="bow",
        model="rf",
        classifier="rf",
        pair_strategy="two_stream",
        interaction_level="vector",
        predictor_type="classifier",
        fusion_mode="concat",
        max_length=60,
        k=2,
        streams={"peptide": stream, "partner": stream},
        stream_mode="homogeneous",
        _frequency_encoder_contract=shared,
    )

    info = _pair_vector_artifact_info(
        args,
        torch.zeros((4, 3)),
        torch.zeros((4, 3)),
        torch.zeros((4, 6)),
        profile_rows=[(str(index), "AA", "BB", index % 2) for index in range(4)],
    )

    contract = validate_pair_bow_joint_contract(
        info["pair_bow_joint_contract"], input_feature_count=6
    )
    assert info["streams"]["peptide"]["frequency_encoder_contract"] == shared
    assert info["streams"]["partner"]["frequency_encoder_contract"] == shared
    assert contract["role_contracts"]["peptide"] == contract["role_contracts"]["partner"]


def _artifact_descriptor(path):
    return {
        "path": "model.joblib",
        "publication": "atomic_replace",
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_pair_lda_fits_and_binds_the_joint_contract_digest(tmp_path, legacy):
    joint = _contract(train_sample_count=6, legacy=legacy)
    train, validation, _, estimator, contract = fit_lda_transform(
        _matrix(),
        _matrix()[:2],
        n_components=2,
        seed=17,
        upstream_feature_contract=joint,
    )

    assert train.shape == (6, 2)
    assert validation.shape == (2, 2)
    assert contract["route"] == "pair_two_stream_classic_bow_v2"
    assert contract["input_orientation"] == "sample_by_peptide_term_then_partner_term"
    if legacy:
        contract["route"] = joint["route"]
    original = copy.deepcopy(contract)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"pair-lda")
    assert ModelService._validate_lda_artifact(
        {
            "task_type": "pep_protein_interaction",
            "args": {"reduction": "lda"},
            "pair_bow_joint_contract": joint,
            "lda_contract": contract,
            "model_artifact": _artifact_descriptor(model_path),
        },
        tmp_path,
    )
    assert contract == original
    tampered = copy.deepcopy(joint)
    tampered["ordered_pair_train_sha256"] = "d" * 64
    with pytest.raises(LDAContractError) as exc_info:
        validate_lda_contract(
            contract,
            estimator=estimator,
            upstream_feature_contract=tampered,
        )
    assert exc_info.value.code == "lda_upstream_contract_mismatch"


@pytest.mark.parametrize("legacy", [False, True])
def test_pair_plsa_fits_and_binds_the_joint_contract_digest(tmp_path, legacy):
    joint = _contract(train_sample_count=6, legacy=legacy)
    train, validation, _, estimator, contract = fit_plsa_transform(
        _matrix(),
        _matrix()[:2],
        n_components=2,
        seed=17,
        upstream_feature_contract=joint,
    )

    np.testing.assert_allclose(train.sum(axis=1), 1.0)
    np.testing.assert_allclose(validation.sum(axis=1), 1.0)
    assert contract["route"] == "pair_two_stream_classic_bow_v2"
    assert contract["input_orientation"] == "sample_by_peptide_term_then_partner_term"
    if legacy:
        contract["route"] = joint["route"]
    original = copy.deepcopy(contract)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"pair-plsa")
    assert ModelService._validate_plsa_artifact(
        {
            "task_type": "pep_protein_interaction",
            "args": {
                "reduction": "plsa",
                "plsa_contract_version": PLSA_CONTRACT_VERSION,
            },
            "pair_bow_joint_contract": joint,
            "plsa_contract": contract,
            "model_artifact": _artifact_descriptor(model_path),
        },
        tmp_path,
    )
    assert contract == original
    tampered = copy.deepcopy(joint)
    tampered["ordered_pair_train_sha256"] = "d" * 64
    with pytest.raises(PLSAContractError) as exc_info:
        validate_plsa_contract(
            contract,
            estimator=estimator,
            upstream_feature_contract=tampered,
        )
    assert exc_info.value.code == "plsa_upstream_contract_mismatch"


def test_pair_topic_fit_rejects_train_rows_from_a_different_joint_contract():
    with pytest.raises(LDAContractError) as exc_info:
        fit_lda_transform(
            _matrix()[:5],
            _matrix()[:2],
            n_components=2,
            seed=17,
            upstream_feature_contract=_contract(train_sample_count=6),
        )
    assert exc_info.value.code == "pair_bow_sample_count_mismatch"


def test_pair_bow_matrix_requires_integer_nonnegative_counts():
    validate_pair_bow_matrix(
        np.zeros((2, 5), dtype=np.float32), split="train", input_feature_count=5
    )
    with pytest.raises(PairBoWContractError) as exc_info:
        validate_pair_bow_matrix(
            np.zeros((2, 5), dtype=np.float32),
            split="train",
            input_feature_count=5,
            expected_sample_count=4,
        )
    assert exc_info.value.code == "pair_bow_sample_count_mismatch"
    with pytest.raises(PairBoWContractError):
        validate_pair_bow_matrix(np.array([[0.5] * 5]), split="train", input_feature_count=5)


def test_pair_topic_route_requires_concat_and_classical_classifier():
    validate_pair_bow_route(
        SimpleNamespace(
            reduction="lda",
            task_type="pep_hlaii",
            encoder="bow",
            pair_strategy="two_stream",
            interaction_level="vector",
            predictor_type="classifier",
            classifier="svm",
            fusion_mode="concat",
        ),
        pair=True,
    )
    with pytest.raises(PairBoWContractError):
        validate_pair_bow_route(
            {
                "reduction": "plsa",
                "task_type": "pep_hlaii",
                "encoder": "bow",
                "pair_strategy": "two_stream",
                "interaction_level": "vector",
                "predictor_type": "classifier",
                "classifier": "mlp",
                "fusion_mode": "concat",
            },
            pair=True,
        )
