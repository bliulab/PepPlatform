import hashlib
import json
from argparse import Namespace

import joblib
import numpy as np
import pytest
import sklearn
import torch
from sklearn.decomposition import KernelPCA
from torch.utils.data import TensorDataset

from pepplatform.kpca import (
    KPCAContractError,
    fit_kpca_transform,
    transform_kpca_held_out,
    validate_kpca_contract,
)
from pepplatform.ml_trainers import ClassicalMLTrainer
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


def _fit(validation=None, *, dimension=2, kernel="rbf", gamma=0.1, limit=4.0):
    train = _matrix()
    validation = train[:3] if validation is None else validation
    return fit_kpca_transform(
        train,
        validation,
        n_components=dimension,
        kernel=kernel,
        gamma=gamma,
        seed=17,
        memory_limit_gib=limit,
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_kpca_matches_sklearn_and_fits_train_only():
    train_out, validation_out, _, estimator, contract = _fit()
    direct = KernelPCA(
        n_components=2,
        kernel="rbf",
        gamma=0.1,
        degree=3,
        coef0=1,
        alpha=1.0,
        fit_inverse_transform=False,
        eigen_solver="auto",
        remove_zero_eig=False,
        copy_X=True,
        n_jobs=None,
        random_state=17,
    ).fit(_matrix())
    np.testing.assert_allclose(train_out, direct.transform(_matrix()))
    np.testing.assert_allclose(validation_out, direct.transform(_matrix()[:3]))
    assert contract["centering"] == "kernel_centerer"
    assert contract["feature_fit_scope"] == "train_only"

    _, changed_validation, _, changed_estimator, changed_contract = _fit(_matrix()[:3] * 100.0)
    np.testing.assert_allclose(estimator.eigenvalues_, changed_estimator.eigenvalues_)
    assert contract == changed_contract
    assert not np.allclose(validation_out, changed_validation)


def test_kpca_held_out_transform_is_batch_invariant_and_reload_stable(tmp_path):
    _, combined, _, estimator, contract = _fit(_matrix()[1:6])
    individual = np.vstack(
        [
            transform_kpca_held_out(estimator, row[None, :], split="validation", contract=contract)
            for row in _matrix()[1:6]
        ]
    )
    np.testing.assert_allclose(combined, individual)
    path = tmp_path / "kpca.joblib"
    joblib.dump({"estimator": estimator, "contract": contract}, path)
    restored = joblib.load(path)
    validate_kpca_contract(restored["contract"], estimator=restored["estimator"])
    np.testing.assert_allclose(combined, restored["estimator"].transform(_matrix()[1:6]))


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"dimension": 0}, "kpca_dimension_invalid"),
        ({"dimension": 4}, "kpca_dimension_exceeds_rank_bound"),
        ({"kernel": "sigmoid"}, "kpca_kernel_invalid"),
        ({"gamma": 0.0}, "kpca_gamma_invalid"),
        ({"gamma": float("nan")}, "kpca_gamma_invalid"),
    ],
)
def test_kpca_rejects_invalid_configuration(kwargs, code):
    with pytest.raises(KPCAContractError) as exc_info:
        _fit(**kwargs)
    assert exc_info.value.code == code


def test_kpca_linear_and_cosine_do_not_claim_gamma():
    for kernel in ("linear", "cosine"):
        _, _, _, estimator, contract = _fit(kernel=kernel, gamma=None)
        assert estimator.gamma is None
        assert contract["kernel_params"]["gamma"] is None


@pytest.mark.parametrize("kernel", ["linear", "cosine"])
def test_kpca_rejects_gamma_for_kernels_that_do_not_use_it(kernel):
    with pytest.raises(KPCAContractError) as exc_info:
        _fit(kernel=kernel, gamma=0.1)
    assert exc_info.value.code == "kpca_gamma_not_applicable"


@pytest.mark.parametrize(
    ("matrix", "code"),
    [
        (np.ones((6, 4), dtype=np.float64), "kpca_rank_insufficient"),
        (
            np.asarray([[1.0, np.nan, 0.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
            "kpca_matrix_nonfinite",
        ),
    ],
)
def test_kpca_rejects_degenerate_or_nonfinite_training_matrices(matrix, code):
    with pytest.raises(KPCAContractError) as exc_info:
        fit_kpca_transform(matrix, matrix[:1], n_components=1, kernel="rbf", gamma=0.1)
    assert exc_info.value.code == code


def test_kpca_memory_guard_fails_before_fit():
    with pytest.raises(KPCAContractError) as exc_info:
        _fit(limit=1e-12)
    assert exc_info.value.code == "kpca_memory_limit_exceeded"


def test_kpca_contract_validation_rejects_malformed_bound_state():
    _, _, _, _, contract = _fit()
    malformed = []

    invalid_hash = json.loads(json.dumps(contract))
    invalid_hash["train_matrix_sha256"] = "not-a-digest"
    malformed.append(invalid_hash)

    invalid_solver = json.loads(json.dumps(contract))
    invalid_solver["effective_eigen_solver"] = "arpack"
    malformed.append(invalid_solver)

    invalid_memory = json.loads(json.dumps(contract))
    invalid_memory["estimated_memory_gib"] = 0.0
    malformed.append(invalid_memory)

    invalid_spectrum = json.loads(json.dumps(contract))
    invalid_spectrum["diagnostics"]["eigenvalues"][0] = "invalid"
    malformed.append(invalid_spectrum)

    for candidate in malformed:
        with pytest.raises(KPCAContractError) as exc_info:
            validate_kpca_contract(candidate)
        assert exc_info.value.code == "kpca_contract_invalid"


def test_kpca_rejects_complex_input_explicitly():
    matrix = _matrix().astype(np.complex128)
    with pytest.raises(KPCAContractError) as exc_info:
        fit_kpca_transform(
            matrix,
            matrix[:2],
            n_components=2,
            kernel="rbf",
            gamma=0.1,
        )
    assert exc_info.value.code == "kpca_matrix_dtype_invalid"


def test_trainer_rejects_invalid_kpca_instead_of_silent_noop():
    trainer = ClassicalMLTrainer(
        Namespace(
            ml_reduction="kpca",
            ml_reduction_dim=4,
            kpca_kernel="rbf",
            kpca_gamma=0.1,
            kpca_memory_limit_gib=4.0,
            seed=17,
        )
    )
    with pytest.raises(KPCAContractError):
        trainer.apply_ml_reduction(_matrix(), _matrix()[:2])


def test_kpca_artifact_contract_and_model_digest_round_trip(tmp_path):
    args = Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="bow",
        reduction="none",
        ml_scaler="none",
        ml_reduction="kpca",
        ml_reduction_dim=2,
        kpca_kernel="rbf",
        kpca_gamma=0.1,
        kpca_memory_limit_gib=4.0,
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
    assert metadata["kpca_contract"] == payload["kpca_contract"]
    validate_kpca_contract(metadata["kpca_contract"], estimator=payload["ml_reducer"])
    assert metadata["model_artifact"]["sha256"] == _sha256(tmp_path / "model.joblib")
    assert metadata["kpca_contract"]["implementation_version"] == sklearn.__version__
    ModelService._validate_kpca_artifact(metadata, tmp_path)
    ModelService._load_classical_payload(tmp_path, metadata)


def test_web_kpca_validation_rejects_tampering_and_version_mismatch(tmp_path):
    _, _, _, estimator, contract = _fit()
    model_path = tmp_path / "model.joblib"
    joblib.dump({"ml_reducer": estimator, "kpca_contract": contract}, model_path)
    metadata = {
        "args": {"ml_reduction": "kpca"},
        "kpca_contract": contract,
        "model_artifact": {
            "path": "model.joblib",
            "publication": "atomic_replace",
            "sha256": _sha256(model_path),
            "size_bytes": model_path.stat().st_size,
        },
    }
    metadata["model_artifact"]["sha256"] = "0" * 64
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_kpca_artifact(metadata, tmp_path)
    assert exc_info.value.code == "kpca_model_integrity_failed"

    metadata["model_artifact"]["sha256"] = _sha256(model_path)
    metadata["kpca_contract"] = {**contract, "implementation_version": "0.invalid"}
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_kpca_artifact(metadata, tmp_path)
    assert exc_info.value.code == "kpca_runtime_version_mismatch"


def test_web_legacy_kpca_warns_and_preserves_historical_sigmoid_behavior():
    metadata = {"args": {"ml_reduction": "kpca", "kpca_kernel": "sigmoid"}}
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        assert ModelService._validate_kpca_artifact(metadata, ".") is False
