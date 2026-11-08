from argparse import Namespace
import json

import joblib
import numpy as np
import pytest
import sklearn
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import TensorDataset

from pepplatform.ml_trainers import (
    CALIBRATED_SVM_PREDICTOR_CONTRACT_VERSION,
    ClassicalMLTrainer,
    SVMConvergenceError,
    SVMProbabilityContractError,
    calibrated_svm_predictor_contract,
)
from pepplatform.web.inference import InferenceUnsupportedError, ModelService


def _args(**overrides):
    values = {
        "seed": 17,
        "classifier": "svm",
        "model": "svm",
        "svm_calibration": "sigmoid",
        "svm_calibration_cv_folds": 3,
        "svm_c": 1.0,
        "svm_class_weight": "none",
        "svm_kernel": "linear",
        "svm_gamma": "scale",
        "svm_max_iter": 10_000,
        "svm_tol": 1e-4,
        "svm_convergence_policy": "strict",
    }
    values.update(overrides)
    return Namespace(**values)


def _multioutput_data():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(72, 6)).astype(np.float32)
    first = (features[:, 0] + 0.25 * features[:, 1] > 0).astype(int)
    second = (features[:, 2] - features[:, 3] > 0).astype(int)
    return features, np.column_stack([first, second])


def _fit_contract(kernel="linear"):
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args(svm_kernel=kernel, svm_gamma=0.2))
    estimators = trainer._fit_calibrated_svm_estimators(features, labels)
    probabilities = trainer._probability_ensemble_scores(estimators, features[:18])
    quality = trainer._svm_probability_quality(labels[:18], probabilities)
    contract = calibrated_svm_predictor_contract(
        estimators,
        convergence=trainer._last_svm_convergence,
        probability_quality=quality,
        input_feature_count=features.shape[1],
    )
    return trainer, features, labels, estimators, probabilities, contract


@pytest.mark.parametrize(
    ("kernel", "base_type", "base_name"),
    [
        ("linear", LinearSVC, "sklearn.svm.LinearSVC"),
        ("rbf", SVC, "sklearn.svm.SVC"),
    ],
)
def test_calibrated_svm_identity_probability_and_fixed_cv(kernel, base_type, base_name):
    _, _, _, estimators, probabilities, contract = _fit_contract(kernel)

    assert probabilities.shape == (18, 2)
    assert np.isfinite(probabilities).all()
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
    assert all(isinstance(item["value"], CalibratedClassifierCV) for item in estimators)
    for item in estimators:
        estimator = item["value"]
        assert estimator.method == "sigmoid"
        assert estimator.ensemble is True
        assert estimator.cv.n_splits == 3
        assert estimator.cv.shuffle is True
        assert estimator.cv.random_state == 17
        assert len(estimator.calibrated_classifiers_) == 3
        assert all(
            isinstance(calibrated.estimator, base_type)
            for calibrated in estimator.calibrated_classifiers_
        )

    assert contract["contract_version"] == CALIBRATED_SVM_PREDICTOR_CONTRACT_VERSION
    assert contract["implementation"] == "sklearn.calibration.CalibratedClassifierCV"
    assert contract["base_implementation"] == base_name
    assert contract["implementation_version"] == sklearn.__version__
    assert contract["status"] == "verified_platform_variant"
    assert contract["score_kind"] == "probability"
    assert contract["calibrated"] is True
    assert contract["calibration"]["fit_scope"] == "train_only_cross_validation"
    assert contract["probability_quality"]["split"] == "validation"


def test_calibrated_svm_is_deterministic_for_fixed_seed():
    features, labels = _multioutput_data()
    first = ClassicalMLTrainer(_args())
    second = ClassicalMLTrainer(_args())

    first_estimators = first._fit_calibrated_svm_estimators(features, labels)
    second_estimators = second._fit_calibrated_svm_estimators(features, labels)

    np.testing.assert_array_equal(
        first._probability_ensemble_scores(first_estimators, features),
        second._probability_ensemble_scores(second_estimators, features),
    )


def test_calibrated_svm_probability_quality_records_each_output_and_macro():
    trainer, _, labels, _, probabilities, _ = _fit_contract()

    quality = trainer._svm_probability_quality(labels[:18], probabilities)

    assert quality["status"] == "validated"
    assert quality["split"] == "validation"
    assert quality["ece_bins"] == 10
    assert len(quality["outputs"]) == 2
    for output in quality["outputs"]:
        assert output["sample_count"] == 18
        assert set(output) >= {"brier", "log_loss", "ece", "probability_min", "probability_max"}
        assert all(np.isfinite(output[key]) for key in ("brier", "log_loss", "ece"))
    for key in ("brier", "log_loss", "ece"):
        assert quality["macro"][key] == pytest.approx(
            np.mean([output[key] for output in quality["outputs"]])
        )


@pytest.mark.parametrize(
    ("features", "labels", "message"),
    [
        (np.ones((6, 2), dtype=np.float32), np.zeros((6, 1), dtype=int), "both binary classes"),
        (
            np.arange(12, dtype=np.float32).reshape(6, 2),
            np.asarray([[0], [0], [1], [1], [1], [1]]),
            "at least 3 training samples",
        ),
        (
            np.asarray([[0.0, np.nan], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]),
            np.asarray([[0], [0], [0], [1], [1], [1]]),
            "finite",
        ),
    ],
)
def test_calibrated_svm_rejects_invalid_training_data(features, labels, message):
    trainer = ClassicalMLTrainer(_args())

    with pytest.raises((ValueError, SVMProbabilityContractError), match=message):
        trainer._fit_calibrated_svm_estimators(features, labels)


def test_calibrated_svm_rejects_nonconverged_base_estimators():
    rng = np.random.default_rng(19)
    features = rng.normal(size=(90, 24)).astype(np.float32)
    labels = (rng.normal(size=90) > 0).astype(int)[:, None]
    trainer = ClassicalMLTrainer(_args(svm_c=10_000.0, svm_max_iter=1, svm_tol=1e-12))

    with pytest.raises(SVMConvergenceError, match="did not converge"):
        trainer._fit_calibrated_svm_estimators(features, labels)


def test_calibrated_svm_joblib_reload_preserves_probabilities(tmp_path):
    trainer, features, _, estimators, expected, _ = _fit_contract()
    path = tmp_path / "calibrated_svm.joblib"

    joblib.dump(estimators, path)
    restored = joblib.load(path)

    np.testing.assert_array_equal(
        expected,
        trainer._probability_ensemble_scores(restored, features[:18]),
    )


def test_web_accepts_current_calibrated_contract_and_scores_probability():
    _, features, _, estimators, expected, contract = _fit_contract()
    metadata = {
        "args": {
            "model": "svm",
            "classifier": "svm",
            "svm_calibration": "sigmoid",
        },
        "predictor_contract": contract,
    }

    ModelService._validate_svm_predictor_contract(metadata)
    actual = ModelService._predict_classical_scores(
        {
            "classifier": "svm",
            "estimator_kind": "calibrated_svm",
            "estimators": estimators,
            "input_feature_count": features.shape[1],
        },
        features[:18],
    )

    assert ModelService._classical_score_kind({"estimator_kind": "calibrated_svm"}) == "probability"
    np.testing.assert_array_equal(expected, actual)


def test_web_rejects_incomplete_current_calibrated_contract():
    _, _, _, _, _, contract = _fit_contract()
    contract["probability_quality"]["ece_bins"] = 9

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_svm_predictor_contract(
            {
                "args": {
                    "model": "svm",
                    "classifier": "svm",
                    "svm_calibration": "sigmoid",
                },
                "predictor_contract": contract,
            }
        )

    assert exc_info.value.code == "calibrated_svm_predictor_contract_unsupported"


def test_calibrated_svm_artifact_records_probability_contract(tmp_path):
    features, labels = _multioutput_data()
    train = TensorDataset(torch.from_numpy(features[:54]), torch.from_numpy(labels[:54]).float())
    val = TensorDataset(torch.from_numpy(features[54:]), torch.from_numpy(labels[54:]).float())
    args = _args(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="bow",
        reduction="none",
        ml_scaler="standard",
        ml_reduction="none",
        ml_auto_tune=False,
        report_aupr=False,
    )

    ClassicalMLTrainer(args).run_svm(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    evaluation = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    payload = joblib.load(tmp_path / "model.joblib")
    contract = metadata["predictor_contract"]
    assert payload["classifier"] == "svm"
    assert payload["estimator_kind"] == "calibrated_svm"
    assert metadata["args"]["classifier"] == "svm"
    assert metadata["args"]["svm_calibration"] == "sigmoid"
    assert metadata["operating_point"]["score_kind"] == "probability"
    assert contract["calibrated"] is True
    assert contract["probability_quality"] == evaluation["probability_quality"]
    ModelService._validate_svm_predictor_contract(metadata)


def test_legacy_calibrated_svm_artifact_warns_without_claiming_current_contract():
    metadata = {"args": {"model": "calibrated_svm", "classifier": "calibrated_svm"}}

    with pytest.warns(RuntimeWarning, match="legacy_only"):
        ModelService._validate_svm_predictor_contract(metadata)
