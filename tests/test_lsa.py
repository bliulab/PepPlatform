import hashlib
import json
from argparse import Namespace

import joblib
import numpy as np
import pytest
import sklearn
import torch
from torch.utils.data import TensorDataset

from pepplatform.lsa import (
    LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    LSA_VARIANT,
    LSAContractError,
    fit_lsa_transform,
    transform_lsa_held_out,
    validate_lsa_contract,
    validate_lsa_route,
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


def _fit(validation=None, *, dimension=2, limit=LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB):
    train = _matrix()
    validation = train[:2] if validation is None else validation
    return fit_lsa_transform(
        train,
        validation,
        encoder="tfidf",
        n_components=dimension,
        seed=17,
        dense_memory_limit_gib=limit,
        upstream_feature_contract={"tokenizer": "kmer", "encoder": "tfidf", "k": 2},
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lsa_is_deterministic_train_only_uncentered_svd():
    train_a, val_a, _, estimator_a, contract_a = _fit()
    train_b, val_b, _, estimator_b, contract_b = _fit(_matrix()[2:5] * 100.0)

    np.testing.assert_allclose(train_a, train_b)
    np.testing.assert_allclose(estimator_a.components_, estimator_b.components_)
    np.testing.assert_allclose(train_a, _matrix() @ estimator_a.components_.T)
    assert not hasattr(estimator_a, "mean_")
    assert val_a.shape == (2, 2)
    assert val_b.shape == (3, 2)
    assert contract_a == contract_b
    assert contract_a["variant"] == LSA_VARIANT
    assert contract_a["feature_fit_scope"] == "train_only"
    assert contract_a["centering"] is False
    assert contract_a["estimator_params"] == {
        "n_components": 2,
        "algorithm": "randomized",
        "n_iter": 5,
        "n_oversamples": 10,
        "power_iteration_normalizer": "auto",
        "tol": 0.0,
        "random_state": 17,
    }


def test_generic_pair_tsvd_matches_lsa_operator_for_same_matrix():
    train = _matrix()
    validation = train[:2]
    lsa_train, lsa_validation, _, _, _ = fit_lsa_transform(
        train,
        validation,
        encoder="bow",
        n_components=2,
        seed=17,
        upstream_feature_contract={"tokenizer": "kmer", "encoder": "bow"},
    )

    trainer = ClassicalMLTrainer(Namespace(ml_reduction="tsvd", ml_reduction_dim=2, seed=17))
    tsvd_train, tsvd_validation, _ = trainer.apply_ml_reduction(
        train,
        validation,
    )

    np.testing.assert_allclose(tsvd_train, lsa_train)
    np.testing.assert_allclose(tsvd_validation, lsa_validation)
    assert trainer._ml_reducer is not None


def test_lsa_held_out_transform_is_batch_invariant_and_reload_stable(tmp_path):
    _, combined, _, estimator, contract = _fit(_matrix()[1:5])
    individual = np.vstack(
        [
            transform_lsa_held_out(
                estimator,
                row[None, :],
                split="validation",
                input_feature_count=4,
                output_dimension=2,
                dense_memory_limit_gib=4.0,
            )
            for row in _matrix()[1:5]
        ]
    )
    np.testing.assert_allclose(combined, individual)

    path = tmp_path / "lsa.joblib"
    joblib.dump({"estimator": estimator, "contract": contract}, path)
    restored = joblib.load(path)
    validate_lsa_contract(restored["contract"], estimator=restored["estimator"])
    np.testing.assert_allclose(
        combined,
        restored["estimator"].transform(_matrix()[1:5]),
    )


@pytest.mark.parametrize("encoder", ["index", "onehot", "esm2", ""])
def test_lsa_rejects_unsupported_encoders(encoder):
    with pytest.raises(LSAContractError) as exc_info:
        validate_lsa_route(Namespace(reduction="lsa", encoder=encoder, lsa_dim=2), pair=False)
    assert exc_info.value.code == "lsa_encoder_unsupported"


@pytest.mark.parametrize("task_type", ["pep_protein_interaction", "pep_hlaii", "pep_hlai"])
def test_lsa_rejects_every_pair_route(task_type):
    with pytest.raises(LSAContractError) as exc_info:
        validate_lsa_route(
            Namespace(
                task_type=task_type,
                reduction="lsa",
                encoder="bow",
                lsa_dim=2,
            )
        )
    assert exc_info.value.code == "lsa_pair_route_unsupported"
    assert "ml_reduction=tsvd" in str(exc_info.value)


@pytest.mark.parametrize(
    ("dimension", "code"),
    [
        (0, "lsa_dimension_invalid"),
        (7, "lsa_dimension_exceeds_samples"),
        (4, "lsa_dimension_not_below_features"),
    ],
)
def test_lsa_rejects_invalid_dimensions(dimension, code):
    with pytest.raises(LSAContractError) as exc_info:
        _fit(dimension=dimension)
    assert exc_info.value.code == code


@pytest.mark.parametrize(
    ("matrix", "code"),
    [
        (np.zeros((4, 3), dtype=np.float64), "lsa_fit_warning"),
        (np.ones((4, 3), dtype=np.float64), "lsa_fit_warning"),
        (np.asarray([[1.0, np.nan, 0.0], [0.0, 1.0, 2.0]]), "lsa_matrix_nonfinite"),
        (np.asarray([[1.0, np.inf, 0.0], [0.0, 1.0, 2.0]]), "lsa_matrix_nonfinite"),
    ],
)
def test_lsa_rejects_degenerate_or_nonfinite_training_matrices(matrix, code):
    with pytest.raises(LSAContractError) as exc_info:
        fit_lsa_transform(
            matrix,
            matrix[:1],
            encoder="bow",
            n_components=1 if matrix.shape[0] < 3 else 2,
            seed=42,
        )
    assert exc_info.value.code == code


def test_lsa_allows_zero_rows_and_columns_when_requested_rank_is_valid():
    train = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    transformed, _, _, _, contract = fit_lsa_transform(
        train,
        train[:2],
        encoder="bow",
        n_components=2,
        seed=42,
    )
    assert transformed.shape == (4, 2)
    assert contract["diagnostics"]["effective_rank"] == 2


def test_lsa_memory_guard_fails_before_fit():
    with pytest.raises(LSAContractError) as exc_info:
        _fit(limit=1e-12)
    assert exc_info.value.code == "lsa_dense_memory_limit_exceeded"


def test_lsa_artifact_contract_and_model_digest_round_trip(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="bow",
        reduction="lsa",
        lsa_dim=2,
        lsa_dense_memory_limit_gib=4.0,
        ml_scaler="none",
        ml_reduction="none",
        logreg_c=1.0,
        logreg_max_iter=100,
        logreg_class_weight="none",
        logreg_convergence_policy="strict",
        report_aupr=False,
        seed=42,
        _artifact_info={
            "classic": {"tokenizer": "kmer", "encoder": "bow", "k": 1},
            "label_names": ["active"],
        },
    )
    trainer = ClassicalMLTrainer(args)
    features = torch.tensor(_matrix(), dtype=torch.float32)
    labels = torch.tensor([[0], [1], [0], [1], [0], [1]], dtype=torch.float32)
    train = TensorDataset(features, labels)
    train.label_names = ["active"]
    val = TensorDataset(features[:2], labels[:2])

    trainer.run_logreg(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    payload = joblib.load(tmp_path / "model.joblib")
    assert metadata["lsa_contract"] == payload["lsa_contract"]
    validate_lsa_contract(metadata["lsa_contract"], estimator=payload["task_reducer"]["value"])
    assert metadata["model_artifact"] == {
        "path": "model.joblib",
        "publication": "atomic_replace",
        "sha256": _sha256(tmp_path / "model.joblib"),
        "size_bytes": (tmp_path / "model.joblib").stat().st_size,
    }
    assert metadata["lsa_contract"]["implementation_version"] == sklearn.__version__
    ModelService._validate_lsa_artifact(metadata, tmp_path)
    ModelService._load_classical_payload(tmp_path, metadata)


def test_web_lsa_validation_rejects_tampering_and_version_mismatch(tmp_path):
    _, _, _, estimator, contract = _fit()
    payload = {
        "task_reducer": {"kind": "sklearn", "value": estimator},
        "lsa_contract": contract,
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(payload, model_path)
    metadata = {
        "task_type": "pep_function",
        "args": {"reduction": "lsa"},
        "lsa_contract": contract,
        "model_artifact": {
            "path": "model.joblib",
            "publication": "atomic_replace",
            "sha256": _sha256(model_path),
            "size_bytes": model_path.stat().st_size,
        },
    }

    metadata["model_artifact"]["sha256"] = "0" * 64
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_lsa_artifact(metadata, tmp_path)
    assert exc_info.value.code == "lsa_model_integrity_failed"

    metadata["model_artifact"]["sha256"] = _sha256(model_path)
    metadata["lsa_contract"] = {**contract, "implementation_version": "0.invalid"}
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_lsa_artifact(metadata, tmp_path)
    assert exc_info.value.code == "lsa_runtime_version_mismatch"


def test_web_legacy_lsa_warns_and_preserves_missing_reducer_behavior():
    metadata = {
        "task_type": "pep_protein_interaction",
        "args": {"reduction": "lsa"},
    }
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        assert ModelService._validate_lsa_artifact(metadata, ".") is False

    features = np.asarray([[1.0, 2.0]], dtype=np.float32)
    transformed = ModelService._transform_classical_features(
        {"task_reducer": None, "ml_scaler": None, "ml_reducer": None},
        features,
    )
    np.testing.assert_array_equal(transformed, features)
