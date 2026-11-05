import hashlib
import json
from argparse import Namespace

import joblib
import numpy as np
import pytest
import sklearn
import torch
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset

from pepplatform.ml_trainers import ClassicalMLTrainer
from pepplatform.pca import (
    PCAContractError,
    fit_pca_transform,
    transform_pca_held_out,
    validate_pca_contract,
)
from pepplatform.web.inference import InferenceUnsupportedError, ModelService


def _matrix():
    return np.asarray(
        [
            [2.0, 0.0, 1.0, 4.0],
            [0.0, 3.0, 0.0, 1.0],
            [1.0, 1.0, 2.0, 0.0],
            [0.0, 1.0, 1.0, 3.0],
            [3.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 2.0],
            [4.0, 1.0, 3.0, 0.0],
            [2.0, 4.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def _fit(validation=None, *, dimension=2, limit=4.0):
    train = _matrix()
    validation = train[:3] if validation is None else validation
    return fit_pca_transform(
        train,
        validation,
        n_components=dimension,
        seed=17,
        dense_memory_limit_gib=limit,
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pca_matches_sklearn_and_fits_train_only():
    train_out, validation_out, _, estimator, contract = _fit()
    direct = PCA(n_components=2, svd_solver="auto", random_state=17).fit(_matrix())

    np.testing.assert_allclose(train_out, direct.transform(_matrix()))
    np.testing.assert_allclose(validation_out, direct.transform(_matrix()[:3]))
    np.testing.assert_allclose(estimator.mean_, _matrix().mean(axis=0))
    assert contract["centering"] is True
    assert contract["feature_fit_scope"] == "train_only"
    assert contract["fit_solver"] == direct._fit_svd_solver

    _, changed_validation, _, changed_estimator, changed_contract = _fit(_matrix()[:3] * 100.0)
    np.testing.assert_allclose(estimator.components_, changed_estimator.components_)
    assert contract == changed_contract
    assert not np.allclose(validation_out, changed_validation)


def test_pca_held_out_transform_is_batch_invariant_and_reload_stable(tmp_path):
    _, combined, _, estimator, contract = _fit(_matrix()[1:6])
    individual = np.vstack(
        [
            transform_pca_held_out(
                estimator,
                row[None, :],
                split="validation",
                input_feature_count=4,
                output_dimension=2,
                dense_memory_limit_gib=4.0,
            )
            for row in _matrix()[1:6]
        ]
    )
    np.testing.assert_allclose(combined, individual)

    path = tmp_path / "pca.joblib"
    joblib.dump({"estimator": estimator, "contract": contract}, path)
    restored = joblib.load(path)
    validate_pca_contract(restored["contract"], estimator=restored["estimator"])
    np.testing.assert_allclose(combined, restored["estimator"].transform(_matrix()[1:6]))


@pytest.mark.parametrize(
    ("dimension", "code"),
    [
        (0, "pca_dimension_invalid"),
        (4, "pca_dimension_exceeds_features"),
        ("1.5", "pca_dimension_invalid"),
    ],
)
def test_pca_rejects_invalid_dimensions(dimension, code):
    with pytest.raises(PCAContractError) as exc_info:
        _fit(dimension=dimension)
    assert exc_info.value.code == code


def test_pca_rejects_centered_sample_rank_overflow():
    with pytest.raises(PCAContractError) as exc_info:
        fit_pca_transform(_matrix()[:3], _matrix()[3:5], n_components=3, seed=17)
    assert exc_info.value.code == "pca_dimension_exceeds_samples"


@pytest.mark.parametrize(
    ("matrix", "code"),
    [
        (np.ones((6, 4), dtype=np.float64), "pca_fit_warning"),
        (np.asarray([[1.0, np.nan, 0.0], [0.0, 1.0, 2.0]]), "pca_matrix_nonfinite"),
        (np.asarray([[1.0, np.inf, 0.0], [0.0, 1.0, 2.0]]), "pca_matrix_nonfinite"),
    ],
)
def test_pca_rejects_degenerate_or_nonfinite_training_matrices(matrix, code):
    with pytest.raises(PCAContractError) as exc_info:
        fit_pca_transform(matrix, matrix[:1], n_components=1, seed=42)
    assert exc_info.value.code == code


def test_pca_memory_guard_fails_before_fit():
    with pytest.raises(PCAContractError) as exc_info:
        _fit(limit=1e-12)
    assert exc_info.value.code == "pca_dense_memory_limit_exceeded"


def test_pca_held_out_guard_estimates_transform_not_a_second_svd():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(120, 100))
    held_out = rng.normal(size=(100, 100))
    estimator = PCA(n_components=2, random_state=17).fit(train)

    transformed = transform_pca_held_out(
        estimator,
        held_out,
        split="validation",
        input_feature_count=100,
        output_dimension=2,
        dense_memory_limit_gib=0.0004,
    )

    assert transformed.shape == (100, 2)


def test_pca_contract_validation_rejects_malformed_bound_state():
    _, _, _, _, contract = _fit()
    malformed = []

    invalid_hash = json.loads(json.dumps(contract))
    invalid_hash["components_sha256"] = "not-a-digest"
    malformed.append(invalid_hash)

    invalid_solver = json.loads(json.dumps(contract))
    invalid_solver["fit_solver"] = "randomized"
    malformed.append(invalid_solver)

    invalid_memory = json.loads(json.dumps(contract))
    invalid_memory["estimated_memory_gib"] = 0.0
    malformed.append(invalid_memory)

    invalid_spectrum = json.loads(json.dumps(contract))
    invalid_spectrum["diagnostics"]["singular_values"][0] = "invalid"
    malformed.append(invalid_spectrum)

    for candidate in malformed:
        with pytest.raises(PCAContractError) as exc_info:
            validate_pca_contract(candidate)
        assert exc_info.value.code == "pca_contract_invalid"


def test_pca_rejects_complex_input_explicitly():
    matrix = _matrix().astype(np.complex128)
    with pytest.raises(PCAContractError) as exc_info:
        fit_pca_transform(matrix, matrix[:2], n_components=2, seed=17)
    assert exc_info.value.code == "pca_matrix_dtype_invalid"


def test_trainer_rejects_invalid_pca_instead_of_silent_noop():
    trainer = ClassicalMLTrainer(
        Namespace(
            ml_reduction="pca",
            ml_reduction_dim=4,
            pca_dense_memory_limit_gib=4.0,
            seed=17,
        )
    )
    with pytest.raises(PCAContractError):
        trainer.apply_ml_reduction(_matrix(), _matrix()[:2])


def test_pca_artifact_contract_and_model_digest_round_trip(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="bow",
        reduction="none",
        ml_scaler="none",
        ml_reduction="pca",
        ml_reduction_dim=2,
        pca_dense_memory_limit_gib=4.0,
        logreg_c=1.0,
        logreg_max_iter=100,
        logreg_class_weight="none",
        report_aupr=False,
        seed=42,
    )
    trainer = ClassicalMLTrainer(args)
    features = torch.tensor(_matrix(), dtype=torch.float32)
    labels = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=torch.float32)
    train = TensorDataset(features, labels)
    train.label_names = ["active"]
    val = TensorDataset(features[:3], labels[:3])

    trainer.run_logreg(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    payload = joblib.load(tmp_path / "model.joblib")
    assert metadata["pca_contract"] == payload["pca_contract"]
    validate_pca_contract(metadata["pca_contract"], estimator=payload["ml_reducer"])
    assert metadata["model_artifact"]["sha256"] == _sha256(tmp_path / "model.joblib")
    assert metadata["pca_contract"]["implementation_version"] == sklearn.__version__
    ModelService._validate_pca_artifact(metadata, tmp_path)
    ModelService._load_classical_payload(tmp_path, metadata)


def test_web_pca_validation_rejects_tampering_and_version_mismatch(tmp_path):
    _, _, _, estimator, contract = _fit()
    model_path = tmp_path / "model.joblib"
    joblib.dump({"ml_reducer": estimator, "pca_contract": contract}, model_path)
    metadata = {
        "args": {"ml_reduction": "pca"},
        "pca_contract": contract,
        "model_artifact": {
            "path": "model.joblib",
            "publication": "atomic_replace",
            "sha256": _sha256(model_path),
            "size_bytes": model_path.stat().st_size,
        },
    }
    metadata["model_artifact"]["sha256"] = "0" * 64
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_pca_artifact(metadata, tmp_path)
    assert exc_info.value.code == "pca_model_integrity_failed"

    metadata["model_artifact"]["sha256"] = _sha256(model_path)
    metadata["pca_contract"] = {**contract, "implementation_version": "0.invalid"}
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_pca_artifact(metadata, tmp_path)
    assert exc_info.value.code == "pca_runtime_version_mismatch"


def test_web_legacy_pca_warns_and_preserves_missing_contract_behavior():
    metadata = {"args": {"ml_reduction": "pca"}}
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        assert ModelService._validate_pca_artifact(metadata, ".") is False
