import hashlib
import json
from argparse import Namespace

import joblib
import numpy as np
import pytest
import sklearn
import torch
from scipy import sparse
from torch.utils.data import TensorDataset

from pepplatform.ml_trainers import ClassicalMLTrainer
from pepplatform.tsvd import (
    TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    TSVDContractError,
    fit_tsvd_transform,
    transform_tsvd_held_out,
    validate_tsvd_contract,
)
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


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit(validation=None, *, dimension=2, limit=TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB):
    train = _matrix()
    validation = train[:2] if validation is None else validation
    return fit_tsvd_transform(
        train,
        validation,
        n_components=dimension,
        seed=17,
        dense_memory_limit_gib=limit,
    )


def test_tsvd_is_deterministic_and_fits_train_only():
    train_a, val_a, _, estimator_a, contract_a = _fit()
    train_b, val_b, _, estimator_b, contract_b = _fit(_matrix()[2:5] * 100.0)

    np.testing.assert_allclose(train_a, train_b)
    np.testing.assert_allclose(estimator_a.components_, estimator_b.components_)
    np.testing.assert_allclose(train_a, _matrix() @ estimator_a.components_.T)
    assert not hasattr(estimator_a, "mean_")
    assert val_a.shape == (2, 2)
    assert val_b.shape == (3, 2)
    assert contract_a == contract_b
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


def test_tsvd_held_out_transform_is_batch_invariant_and_reload_stable(tmp_path):
    _, combined, _, estimator, contract = _fit(_matrix()[1:5])
    individual = np.vstack(
        [
            transform_tsvd_held_out(
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

    path = tmp_path / "tsvd.joblib"
    joblib.dump({"estimator": estimator, "contract": contract}, path)
    restored = joblib.load(path)
    validate_tsvd_contract(restored["contract"], estimator=restored["estimator"])
    np.testing.assert_allclose(combined, restored["estimator"].transform(_matrix()[1:5]))


def test_tsvd_supports_sparse_train_and_held_out_matrices():
    train = sparse.csr_matrix(_matrix())
    validation = train[:2]
    train_output, validation_output, _, estimator, contract = fit_tsvd_transform(
        train,
        validation,
        n_components=2,
        seed=17,
    )

    assert contract["input_format"] == "sparse"
    np.testing.assert_allclose(train_output, estimator.transform(train))
    np.testing.assert_allclose(validation_output, estimator.transform(validation))
    assert train_output.shape == (train.shape[0], 2)
    assert validation_output.shape == (validation.shape[0], 2)
    assert np.isfinite(train_output).all()
    assert np.isfinite(validation_output).all()


@pytest.mark.parametrize(
    ("dimension", "code"),
    [
        (0, "tsvd_dimension_invalid"),
        (7, "tsvd_dimension_exceeds_samples"),
        (5, "tsvd_dimension_exceeds_features"),
        ("1.5", "tsvd_dimension_invalid"),
    ],
)
def test_tsvd_rejects_invalid_dimensions(dimension, code):
    with pytest.raises(TSVDContractError) as exc_info:
        _fit(dimension=dimension)
    assert exc_info.value.code == code


@pytest.mark.parametrize(
    ("matrix", "code"),
    [
        (np.zeros((4, 3), dtype=np.float64), "tsvd_fit_warning"),
        (np.ones((4, 3), dtype=np.float64), "tsvd_fit_warning"),
        (np.asarray([[1.0, np.nan, 0.0], [0.0, 1.0, 2.0]]), "tsvd_matrix_nonfinite"),
        (np.asarray([[1.0, np.inf, 0.0], [0.0, 1.0, 2.0]]), "tsvd_matrix_nonfinite"),
    ],
)
def test_tsvd_rejects_degenerate_or_nonfinite_training_matrices(matrix, code):
    with pytest.raises(TSVDContractError) as exc_info:
        fit_tsvd_transform(matrix, matrix[:1], n_components=1, seed=42)
    assert exc_info.value.code == code


def test_tsvd_memory_guard_fails_before_fit():
    with pytest.raises(TSVDContractError) as exc_info:
        _fit(limit=1e-12)
    assert exc_info.value.code == "tsvd_dense_memory_limit_exceeded"


def test_trainer_rejects_invalid_tsvd_instead_of_silent_noop():
    trainer = ClassicalMLTrainer(
        Namespace(
            ml_reduction="tsvd",
            ml_reduction_dim=7,
            tsvd_dense_memory_limit_gib=4.0,
            seed=17,
        )
    )
    with pytest.raises(TSVDContractError, match="exceeds"):
        trainer.apply_ml_reduction(_matrix(), _matrix()[:2])


def test_tsvd_artifact_contract_and_model_digest_round_trip(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="bow",
        reduction="none",
        ml_scaler="none",
        ml_reduction="tsvd",
        ml_reduction_dim=2,
        tsvd_dense_memory_limit_gib=4.0,
        logreg_c=1.0,
        logreg_max_iter=100,
        logreg_class_weight="none",
        report_aupr=False,
        seed=42,
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
    assert metadata["tsvd_contract"] == payload["tsvd_contract"]
    validate_tsvd_contract(
        metadata["tsvd_contract"],
        estimator=payload["ml_reducer"],
    )
    assert metadata["model_artifact"] == {
        "path": "model.joblib",
        "publication": "atomic_replace",
        "sha256": _sha256(tmp_path / "model.joblib"),
        "size_bytes": (tmp_path / "model.joblib").stat().st_size,
    }
    assert metadata["tsvd_contract"]["implementation_version"] == sklearn.__version__
    ModelService._validate_tsvd_artifact(metadata, tmp_path)
    ModelService._load_classical_payload(tmp_path, metadata)


def test_web_tsvd_validation_rejects_tampering_and_version_mismatch(tmp_path):
    _, _, _, estimator, contract = _fit()
    payload = {"ml_reducer": estimator, "tsvd_contract": contract}
    model_path = tmp_path / "model.joblib"
    joblib.dump(payload, model_path)
    metadata = {
        "task_type": "pep_function",
        "args": {"ml_reduction": "tsvd"},
        "tsvd_contract": contract,
        "model_artifact": {
            "path": "model.joblib",
            "publication": "atomic_replace",
            "sha256": _sha256(model_path),
            "size_bytes": model_path.stat().st_size,
        },
    }

    metadata["model_artifact"]["sha256"] = "0" * 64
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_tsvd_artifact(metadata, tmp_path)
    assert exc_info.value.code == "tsvd_model_integrity_failed"

    metadata["model_artifact"]["sha256"] = _sha256(model_path)
    metadata["tsvd_contract"] = {**contract, "implementation_version": "0.invalid"}
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_tsvd_artifact(metadata, tmp_path)
    assert exc_info.value.code == "tsvd_runtime_version_mismatch"


def test_web_legacy_tsvd_warns_and_preserves_missing_contract_behavior():
    metadata = {"task_type": "pep_protein_interaction", "args": {"ml_reduction": "tsvd"}}
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        assert ModelService._validate_tsvd_artifact(metadata, ".") is False
