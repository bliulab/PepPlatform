import hashlib
import json
from argparse import Namespace

import joblib
import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import NMF
import torch
from torch.utils.data import TensorDataset

from pepplatform.plsa import (
    PLSA_CONTRACT_VERSION,
    PLSA_DEFAULT_MEMORY_LIMIT_GIB,
    PLSA_MAX_ITER,
    PLSA_VARIANT,
    PLSAContractError,
    PLSAEstimator,
    fit_plsa_transform,
    validate_plsa_contract,
    validate_plsa_route,
)
from pepplatform.ml_trainers import ClassicalMLTrainer
from pepplatform.web.inference import InferenceUnsupportedError, ModelService


def _matrix():
    return np.asarray(
        [
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
            [1.0, 1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0, 3.0],
            [3.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )


def _upstream(feature_dimension=4):
    return {
        "tokenizer": "kmer",
        "encoder": "bow",
        "k": 1,
        "frequency_encoder_contract": {
            "contract_version": 2,
            "stream_policy": "full_token_stream",
            "bow_formula": "x_j = c_j",
            "tf_formula": "c_j / sum_k(c_k)",
            "idf_formula": "ln((N + 1) / (df_j + 1)) + 1",
            "post_normalization": "none",
            "frequency_max_raw_length": 50_000,
            "feature_dimension": feature_dimension,
            "vocabulary_sha256": hashlib.sha256(b"test-vocabulary").hexdigest(),
            "encoder": "bow",
            "training_document_count": 6,
            "training_corpus_sha256": hashlib.sha256(b"train-corpus").hexdigest(),
            "fit_scope": "sequence_train_split",
        },
    }


def _fit(validation=None, *, topics=2, limit=PLSA_DEFAULT_MEMORY_LIMIT_GIB):
    train = _matrix()
    validation = train[:2] if validation is None else validation
    return fit_plsa_transform(
        train,
        validation,
        n_components=topics,
        seed=17,
        memory_limit_gib=limit,
        upstream_feature_contract=_upstream(),
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plsa_is_deterministic_train_only_em_and_normalized():
    train_a, val_a, _, estimator_a, contract_a = _fit()
    train_b, val_b, _, estimator_b, contract_b = _fit(_matrix()[2:5] * 100.0)

    np.testing.assert_allclose(train_a, train_b)
    np.testing.assert_allclose(estimator_a.components_, estimator_b.components_)
    np.testing.assert_allclose(train_a.sum(axis=1), 1.0)
    np.testing.assert_allclose(val_a.sum(axis=1), 1.0)
    np.testing.assert_allclose(val_b.sum(axis=1), 1.0)
    assert np.all(train_a >= 0.0)
    assert contract_a == contract_b
    assert contract_a["variant"] == PLSA_VARIANT
    assert contract_a["feature_fit_scope"] == "train_only"
    assert contract_a["input_semantics"] == "raw_nonnegative_integer_bow_counts"
    assert contract_a["contract_version"] == PLSA_CONTRACT_VERSION
    assert contract_a["held_out_unseen_term_semantics"] == "ignore_zero_training_probability_terms"
    assert contract_a["diagnostics"]["converged"] is True
    assert contract_a["diagnostics"]["n_iter"] < PLSA_MAX_ITER
    assert contract_a["diagnostics"]["minimum_likelihood_improvement"] >= -1e-8


def test_plsa_one_topic_matches_corpus_term_distribution():
    estimator = PLSAEstimator(1, random_state=3).fit(_matrix())
    expected = _matrix().sum(axis=0) / _matrix().sum()
    np.testing.assert_allclose(estimator.components_[0], expected)
    np.testing.assert_allclose(estimator.doc_topic_, 1.0)


def test_plsa_fold_in_is_batch_invariant_reload_stable_and_handles_zero_document(tmp_path):
    _, combined, _, estimator, contract = _fit(_matrix()[1:5])
    individual = np.vstack(
        [estimator.transform(row[None, :], split="validation") for row in _matrix()[1:5]]
    )
    np.testing.assert_allclose(combined, individual)

    zero = estimator.transform(np.zeros((1, 4)), split="validation")
    np.testing.assert_allclose(zero, [[0.5, 0.5]])

    path = tmp_path / "plsa.joblib"
    joblib.dump({"estimator": estimator, "contract": contract}, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored["estimator"].transform(_matrix()[1:5]), combined)
    validate_plsa_contract(restored["contract"], estimator=restored["estimator"])


@pytest.mark.parametrize("sparse_input", [False, True])
def test_plsa_held_out_ignores_terms_with_zero_training_probability(sparse_input):
    train = np.pad(_matrix(), ((0, 0), (0, 1)))
    validation = np.asarray(
        [
            [2.0, 0.0, 1.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 3.0],
        ]
    )
    train_input = sparse.csr_matrix(train) if sparse_input else train
    validation_input = sparse.csr_matrix(validation) if sparse_input else validation

    _, transformed, _, estimator, contract = fit_plsa_transform(
        train_input,
        validation_input,
        n_components=2,
        seed=17,
        upstream_feature_contract=_upstream(feature_dimension=5),
    )

    supported_only = validation.copy()
    supported_only[:, 4] = 0.0
    expected = estimator.transform(supported_only, split="validation")
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0)
    np.testing.assert_allclose(transformed[1], [0.5, 0.5])
    assert contract["held_out_unseen_term_semantics"] == "ignore_zero_training_probability_terms"


def test_plsa_dense_and_sparse_inputs_match():
    dense = PLSAEstimator(2, random_state=11).fit(_matrix())
    csr = PLSAEstimator(2, random_state=11).fit(sparse.csr_matrix(_matrix()))
    np.testing.assert_allclose(dense.components_, csr.components_)
    np.testing.assert_allclose(dense.doc_topic_, csr.doc_topic_)


@pytest.mark.parametrize(
    ("matrix", "code"),
    [
        (np.asarray([[1.0, -1.0], [0.0, 2.0]]), "plsa_matrix_negative"),
        (np.asarray([[1.0, 0.5], [0.0, 2.0]]), "plsa_matrix_not_counts"),
        (np.asarray([[1.0, np.nan], [0.0, 2.0]]), "plsa_matrix_nonfinite"),
        (np.zeros((2, 2)), "plsa_training_counts_empty"),
    ],
)
def test_plsa_rejects_invalid_count_matrices(matrix, code):
    with pytest.raises(PLSAContractError) as exc_info:
        PLSAEstimator(2).fit(matrix)
    assert exc_info.value.code == code


def test_plsa_topics_are_not_silently_skipped_or_bounded_by_feature_width():
    train, validation, _, estimator, _ = _fit(topics=5)
    assert train.shape == (6, 5)
    assert validation.shape == (2, 5)
    assert estimator.n_components == 5


@pytest.mark.parametrize("encoder", ["tfidf", "textrank", "onehot"])
def test_plsa_route_rejects_non_bow_encoders(encoder):
    with pytest.raises(PLSAContractError) as exc_info:
        validate_plsa_route({"reduction": "plsa", "encoder": encoder, "plsa_topics": 2})
    assert exc_info.value.code == "plsa_encoder_unsupported"


def test_plsa_route_rejects_incomplete_pair_vectors():
    with pytest.raises(PLSAContractError) as exc_info:
        validate_plsa_route(
            {
                "task_type": "pep_protein_interaction",
                "reduction": "plsa",
                "encoder": "bow",
                "plsa_topics": 2,
            }
        )
    assert exc_info.value.code == "plsa_pair_route_unsupported"


def test_plsa_requires_exact_bow_v2_upstream_contract():
    upstream = _upstream()
    upstream["frequency_encoder_contract"]["post_normalization"] = "l2"
    with pytest.raises(PLSAContractError) as exc_info:
        fit_plsa_transform(
            _matrix(),
            _matrix()[:2],
            n_components=2,
            seed=17,
            upstream_feature_contract=upstream,
        )
    assert exc_info.value.code == "plsa_upstream_contract_invalid"


def test_plsa_contract_detects_estimator_and_upstream_tampering():
    _, _, _, estimator, contract = _fit()
    estimator.components_[0, 0] += 0.1
    with pytest.raises(PLSAContractError) as exc_info:
        validate_plsa_contract(contract, estimator=estimator)
    assert exc_info.value.code == "plsa_estimator_contract_mismatch"

    _, _, _, estimator, contract = _fit()
    upstream = _upstream()
    upstream["frequency_encoder_contract"]["vocabulary_sha256"] = "1" * 64
    with pytest.raises(PLSAContractError) as exc_info:
        validate_plsa_contract(contract, estimator=estimator, upstream_feature_contract=upstream)
    assert exc_info.value.code == "plsa_upstream_contract_mismatch"


def test_plsa_memory_guard_fails_before_fit():
    with pytest.raises(PLSAContractError) as exc_info:
        _fit(limit=1e-12)
    assert exc_info.value.code == "plsa_memory_limit_exceeded"


def test_plsa_rejects_partial_convergence():
    with pytest.raises(PLSAContractError) as exc_info:
        PLSAEstimator(2, max_iter=1, tol=0.0, random_state=17).fit(_matrix())
    assert exc_info.value.code == "plsa_not_converged"


def test_plsa_artifact_contract_and_model_digest_round_trip(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="bow",
        reduction="plsa",
        plsa_topics=2,
        plsa_memory_limit_gib=4.0,
        plsa_contract_version=PLSA_CONTRACT_VERSION,
        ml_scaler="none",
        ml_reduction="none",
        logreg_c=1.0,
        logreg_max_iter=100,
        logreg_class_weight="none",
        logreg_convergence_policy="strict",
        report_aupr=False,
        seed=42,
        _artifact_info={"classic": _upstream(), "label_names": ["active"]},
    )
    trainer = ClassicalMLTrainer(args)
    features = torch.tensor(_matrix(), dtype=torch.float32)
    labels = torch.tensor([[0], [1], [0], [1], [0], [1]], dtype=torch.float32)
    train = TensorDataset(features, labels)
    train.label_names = ["active"]
    validation = TensorDataset(features[:2], labels[:2])

    trainer.run_logreg(train, validation)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    payload = joblib.load(tmp_path / "model.joblib")
    assert metadata["plsa_contract"] == payload["plsa_contract"]
    assert payload["task_reducer"]["kind"] == "pepplatform_plsa"
    validate_plsa_contract(
        metadata["plsa_contract"],
        estimator=payload["task_reducer"]["value"],
        upstream_feature_contract=metadata["classic"],
    )
    assert metadata["model_artifact"] == {
        "path": "model.joblib",
        "publication": "atomic_replace",
        "sha256": _sha256(tmp_path / "model.joblib"),
        "size_bytes": (tmp_path / "model.joblib").stat().st_size,
    }
    ModelService._validate_plsa_artifact(metadata, tmp_path)
    ModelService._load_classical_payload(tmp_path, metadata)


def test_web_plsa_validation_rejects_tampering_upstream_and_version_mismatch(tmp_path):
    _, _, _, estimator, contract = _fit()
    payload = {
        "task_reducer": {"kind": "pepplatform_plsa", "value": estimator},
        "plsa_contract": contract,
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(payload, model_path)
    metadata = {
        "task_type": "pep_function",
        "args": {
            "reduction": "plsa",
            "plsa_contract_version": PLSA_CONTRACT_VERSION,
        },
        "classic": _upstream(),
        "plsa_contract": contract,
        "model_artifact": {
            "path": "model.joblib",
            "publication": "atomic_replace",
            "sha256": _sha256(model_path),
            "size_bytes": model_path.stat().st_size,
        },
    }

    metadata["model_artifact"]["sha256"] = "0" * 64
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_plsa_artifact(metadata, tmp_path)
    assert exc_info.value.code == "plsa_model_integrity_failed"

    metadata["model_artifact"]["sha256"] = _sha256(model_path)
    metadata["classic"]["frequency_encoder_contract"]["vocabulary_sha256"] = "1" * 64
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_plsa_artifact(metadata, tmp_path)
    assert exc_info.value.code == "plsa_upstream_contract_mismatch"

    metadata["classic"] = _upstream()
    metadata["plsa_contract"] = {
        **contract,
        "library_versions": {**contract["library_versions"], "numpy": "0.invalid"},
    }
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_plsa_artifact(metadata, tmp_path)
    assert exc_info.value.code == "plsa_runtime_version_mismatch"


def test_web_legacy_kl_nmf_warns_and_preserves_historical_transform(tmp_path):
    legacy = NMF(
        n_components=2,
        init="nndsvda",
        random_state=17,
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=400,
    ).fit(_matrix())
    payload = {
        "task_reducer": {"kind": "nonnegative_sklearn", "value": legacy},
        "ml_scaler": None,
        "ml_reducer": None,
    }
    joblib.dump(payload, tmp_path / "model.joblib")
    metadata = {
        "task_type": "pep_function",
        "args": {"reduction": "plsa"},
    }
    with pytest.warns(RuntimeWarning, match="KL-NMF.*legacy_only"):
        assert ModelService._validate_plsa_artifact(metadata, tmp_path) is False
    ModelService._load_classical_payload(tmp_path, metadata)

    features = _matrix()[:2].copy()
    features[0, 1] = -3.0
    expected = legacy.transform(np.maximum(features, 0.0))
    transformed = ModelService._transform_classical_features(payload, features)
    np.testing.assert_allclose(transformed, expected)


def test_web_plsa_rejects_missing_current_contract_and_invalid_legacy_payload(tmp_path):
    current_missing = {
        "task_type": "pep_function",
        "args": {
            "reduction": "plsa",
            "plsa_contract_version": PLSA_CONTRACT_VERSION,
        },
    }
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_plsa_artifact(current_missing, tmp_path)
    assert exc_info.value.code == "plsa_contract_missing"

    metadata = {"task_type": "pep_function", "args": {"reduction": "plsa"}}
    joblib.dump(
        {"task_reducer": {"kind": "pepplatform_plsa", "value": object()}},
        tmp_path / "model.joblib",
    )
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._load_classical_payload(tmp_path, metadata)
    assert exc_info.value.code == "plsa_legacy_payload_invalid"


def test_plsa_tuning_preprocessing_restores_persisted_state():
    args = Namespace(
        encoder="bow",
        reduction="plsa",
        plsa_topics=2,
        plsa_memory_limit_gib=4.0,
        ml_scaler="none",
        ml_reduction="none",
        seed=17,
        _artifact_info={"classic": _upstream()},
    )
    trainer = ClassicalMLTrainer(args)
    original_reducer = object()
    original_contract = {"sentinel": True}
    trainer._task_reducer = original_reducer
    trainer._plsa_contract = original_contract
    original_info = args._artifact_info

    train, validation = trainer._preprocess_tuning_split(
        _matrix(),
        np.asarray([0, 1, 0, 1, 0, 1]),
        _matrix()[:2],
    )

    assert train.shape == (6, 2)
    assert validation.shape == (2, 2)
    assert trainer._task_reducer is original_reducer
    assert trainer._plsa_contract is original_contract
    assert args._artifact_info is original_info


@pytest.mark.parametrize("sparse_input", [False, True])
def test_plsa_tuning_ignores_terms_unseen_in_fold_training(sparse_input):
    args = Namespace(
        encoder="bow",
        reduction="plsa",
        plsa_topics=2,
        plsa_memory_limit_gib=4.0,
        ml_scaler="none",
        ml_reduction="none",
        seed=17,
        _artifact_info={"classic": _upstream(feature_dimension=5)},
    )
    trainer = ClassicalMLTrainer(args)
    train = np.pad(_matrix(), ((0, 0), (0, 1)))
    validation = np.asarray(
        [
            [2.0, 0.0, 1.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 3.0],
        ]
    )
    if sparse_input:
        train = sparse.csr_matrix(train)
        validation = sparse.csr_matrix(validation)

    transformed_train, transformed_validation = trainer._preprocess_tuning_split(
        train,
        np.asarray([0, 1, 0, 1, 0, 1]),
        validation,
    )

    assert transformed_train.shape == (6, 2)
    assert transformed_validation.shape == (2, 2)
    np.testing.assert_allclose(transformed_validation.sum(axis=1), 1.0)
    np.testing.assert_allclose(transformed_validation[1], [0.5, 0.5])
