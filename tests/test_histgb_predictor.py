from argparse import Namespace
import json

import joblib
import numpy as np
import pytest
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import TensorDataset

from pepplatform.ml_trainers import (
    HISTGB_PREDICTOR_CONTRACT_VERSION,
    ClassicalMLTrainer,
    HistGBArtifactContractError,
    HistGBTrainingDataError,
    histgb_predictor_contract,
)
from pepplatform.training_failures import failure_from_exception
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from pepplatform.web.preflight import preflight_experiment


def _args(**overrides):
    values = {
        "task_type": "pep_function",
        "tokenizer": "kmer",
        "encoder": "bow",
        "model": "histgb",
        "classifier": "histgb",
        "artifact_dir": "",
        "reduction": "none",
        "ml_scaler": "none",
        "ml_reduction": "none",
        "ml_auto_tune": False,
        "report_aupr": False,
        "histgb_max_iter": 8,
        "histgb_max_leaf_nodes": 7,
        "histgb_lr": 0.1,
        "histgb_l2_regularization": 0.0,
        "seed": 17,
        "_artifact_source": "sequence_features",
        "_artifact_info": {"label_names": ["alpha", "beta"]},
    }
    values.update(overrides)
    return Namespace(**values)


def _multioutput_data(sample_count=80):
    rng = np.random.default_rng(31)
    features = rng.normal(size=(sample_count, 8)).astype(np.float32)
    first = (features[:, 0] + 0.2 * features[:, 1] > 0).astype(int)
    second = (features[:, 2] - 0.4 * features[:, 3] > 0).astype(int)
    return features, np.column_stack([first, second])


def _dataset_record():
    return {
        "id": "toy",
        "name": "toy",
        "task_type": "pep_function",
        "format": "sequence_dir",
        "metadata": {"splits": {"train": 40, "val": 20}},
    }


def test_histgb_identity_contract_probability_nan_and_batch_invariance():
    features, labels = _multioutput_data()
    features[0, 0] = np.nan
    trainer = ClassicalMLTrainer(_args())

    estimators = trainer._fit_histgb_estimators(features, labels)
    together = trainer._probability_ensemble_scores(estimators, features[:9])
    separately = np.concatenate(
        [trainer._probability_ensemble_scores(estimators, row[None, :]) for row in features[:9]],
        axis=0,
    )
    contract = histgb_predictor_contract(
        estimators,
        input_feature_count=features.shape[1],
    )

    assert all(isinstance(item["value"], HistGradientBoostingClassifier) for item in estimators)
    assert together.shape == (9, 2)
    assert np.isfinite(together).all()
    assert ((0.0 <= together) & (together <= 1.0)).all()
    np.testing.assert_array_equal(together, separately)
    assert contract["contract_version"] == HISTGB_PREDICTOR_CONTRACT_VERSION
    assert contract["algorithm"] == "histogram_gradient_boosting"
    assert contract["implementation_version"] == sklearn.__version__
    assert contract["status"] == "verified_platform_variant"
    assert contract["loss"] == "log_loss"
    assert contract["inverse_link"] == "logistic_sigmoid"
    assert contract["native_missing_values"] is True
    assert contract["early_stopping"] is False
    assert contract["fitted_iterations_by_output"] == [8, 8]
    assert contract["score_kind"] == "probability"
    assert contract["calibrated"] is False
    assert contract["estimator_params"]["max_bins"] == 255
    assert contract["estimator_params"]["categorical_features"] == "from_dtype"


@pytest.mark.parametrize(
    ("features_transform", "labels_transform", "message"),
    [
        (lambda x: x[:0], lambda y: y[:0], "at least one training sample"),
        (None, lambda y: np.zeros_like(y), "both binary classes"),
        (None, lambda y: np.full_like(y, 2), "both binary classes"),
        (
            lambda x: np.concatenate(
                [np.full((1, x.shape[1]), np.inf, dtype=x.dtype), x[1:]], axis=0
            ),
            None,
            "not infinity",
        ),
        (lambda x: x.astype(str), None, "numeric"),
    ],
)
def test_histgb_rejects_invalid_or_degenerate_training_data(
    features_transform, labels_transform, message
):
    features, labels = _multioutput_data()
    if features_transform is not None:
        features = features_transform(features)
    if labels_transform is not None:
        labels = labels_transform(labels)

    with pytest.raises(HistGBTrainingDataError, match=message) as exc_info:
        ClassicalMLTrainer(_args())._fit_histgb_estimators(features, labels)

    assert exc_info.value.code == "histgb_training_data_invalid"


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("histgb_max_iter", 0, "at least 1"),
        ("histgb_max_leaf_nodes", 1, r"0 \(unlimited\) or at least 2"),
        ("histgb_lr", 0.0, "greater than 0"),
        ("histgb_lr", np.inf, "finite"),
        ("histgb_l2_regularization", -1.0, "greater than or equal to 0"),
    ],
)
def test_histgb_rejects_invalid_parameters(parameter, value, message):
    features, labels = _multioutput_data()

    with pytest.raises(HistGBTrainingDataError, match=message):
        ClassicalMLTrainer(_args(**{parameter: value}))._fit_histgb_estimators(features, labels)


def test_histgb_disables_sklearn_auto_early_stopping_above_10000_samples():
    features, labels = _multioutput_data(sample_count=10_001)
    trainer = ClassicalMLTrainer(_args(histgb_max_iter=3))

    estimators = trainer._fit_histgb_estimators(features, labels)

    assert all(item["value"].early_stopping is False for item in estimators)
    assert all(item["value"].do_early_stopping_ is False for item in estimators)
    assert all(item["value"].n_iter_ == 3 for item in estimators)
    assert all(item["value"].validation_score_.size == 0 for item in estimators)


def test_histgb_contract_rejects_constant_and_non_identity_outputs():
    with pytest.raises(TypeError, match="without constant fallbacks"):
        histgb_predictor_contract(
            [{"kind": "constant", "value": 0.0}],
            input_feature_count=4,
        )

    features, labels = _multioutput_data()
    estimators = ClassicalMLTrainer(_args())._fit_histgb_estimators(features, labels)
    estimators[0]["value"].set_params(early_stopping="auto")
    with pytest.raises(ValueError, match="fixed log-loss"):
        histgb_predictor_contract(estimators[:1], input_feature_count=8)


def test_histgb_joblib_reload_preserves_probabilities(tmp_path):
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args())
    estimators = trainer._fit_histgb_estimators(features, labels)
    expected = trainer._probability_ensemble_scores(estimators, features)
    path = tmp_path / "histgb.joblib"

    joblib.dump(estimators, path)
    restored = joblib.load(path)

    np.testing.assert_array_equal(
        expected,
        trainer._probability_ensemble_scores(restored, features),
    )


def test_histgb_training_entry_emits_current_contract(tmp_path):
    features, labels = _multioutput_data()
    train = TensorDataset(torch.from_numpy(features[:64]), torch.from_numpy(labels[:64]).float())
    validation = TensorDataset(
        torch.from_numpy(features[64:]), torch.from_numpy(labels[64:]).float()
    )
    trainer = ClassicalMLTrainer(_args(artifact_dir=str(tmp_path)))

    result = trainer.run_histgb(train, validation)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    payload = joblib.load(tmp_path / "model.joblib")
    contract = metadata["predictor_contract"]
    assert np.isfinite(result)
    assert contract["algorithm"] == "histogram_gradient_boosting"
    assert contract["input_feature_count"] == features.shape[1]
    assert contract["fitted_iterations_by_output"] == [8, 8]
    assert metadata["operating_point"]["score_kind"] == "probability"
    assert payload["classifier"] == "histgb"
    assert payload["input_feature_count"] == features.shape[1]
    ModelService._validate_histgb_predictor_contract(metadata)


def test_histgb_artifact_and_operating_point_are_test_label_independent(tmp_path):
    features, labels = _multioutput_data()
    outputs = []
    for name, test_labels in [
        ("original", labels[72:]),
        ("flipped", 1 - labels[72:]),
    ]:
        artifact_dir = tmp_path / name
        trainer = ClassicalMLTrainer(_args(artifact_dir=str(artifact_dir)))
        train = TensorDataset(torch.from_numpy(features[:56]), torch.from_numpy(labels[:56]))
        validation = TensorDataset(
            torch.from_numpy(features[56:72]), torch.from_numpy(labels[56:72])
        )
        test = TensorDataset(torch.from_numpy(features[72:]), torch.from_numpy(test_labels))

        trainer.run_histgb(train, validation, test)
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        evaluation = json.loads((artifact_dir / "evaluation.json").read_text(encoding="utf-8"))
        payload = joblib.load(artifact_dir / "model.joblib")
        outputs.append(
            (
                metadata,
                evaluation,
                ModelService._predict_classical_scores(payload, features[72:]),
            )
        )

    original, flipped = outputs
    assert original[0]["predictor_contract"] == flipped[0]["predictor_contract"]
    assert original[0]["operating_point"] == flipped[0]["operating_point"]
    assert original[1]["validation"] == flipped[1]["validation"]
    np.testing.assert_array_equal(original[2], flipped[2])


def test_histgb_legacy_and_current_web_contract_boundaries():
    legacy = {"args": {"classifier": "histgb"}, "label_names": ["label"]}
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        ModelService._validate_histgb_predictor_contract(legacy)

    features, labels = _multioutput_data()
    estimators = ClassicalMLTrainer(_args())._fit_histgb_estimators(features, labels)
    contract = histgb_predictor_contract(estimators, input_feature_count=8)
    ModelService._validate_histgb_predictor_contract(
        {"args": {"classifier": "histgb"}, "predictor_contract": contract}
    )

    contract["implementation_version"] = "0.0.invalid"
    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_histgb_predictor_contract(
            {"args": {"classifier": "histgb"}, "predictor_contract": contract}
        )
    assert exc_info.value.code == "histgb_runtime_version_mismatch"


def test_histgb_artifact_contract_failure_has_stable_runtime_code(tmp_path):
    trainer = ClassicalMLTrainer(_args(artifact_dir=str(tmp_path)))
    bundle = {
        "classifier": "histgb",
        "estimator_kind": "probability_ensemble",
        "estimators": [{"kind": "constant", "value": 0.0}],
        "input_feature_count": 4,
    }

    with pytest.raises(HistGBArtifactContractError) as exc_info:
        trainer._maybe_save_artifact(bundle, object())

    failure = failure_from_exception(exc_info.value)
    assert exc_info.value.code == "histgb_artifact_contract_failed"
    assert failure["category"] == "runtime"
    assert failure["stage"] == "artifact_export"


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("histgb_lr", 0.0),
        ("histgb_lr", float("nan")),
        ("histgb_max_leaf_nodes", 1),
        ("histgb_l2_regularization", -0.1),
    ],
)
def test_histgb_preflight_rejects_invalid_parameters(parameter, value):
    config = {
        "tokenizer": "kmer",
        "encoder": "bow",
        "model": "histgb",
        "classifier": "histgb",
        parameter: value,
    }

    report = preflight_experiment(_dataset_record(), "pep_function", config)

    assert not report["valid"]
    assert "invalid_numeric_parameter" in report["error_codes"]


def test_histgb_preflight_accepts_unlimited_leaf_sentinel():
    report = preflight_experiment(
        _dataset_record(),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "bow",
            "model": "histgb",
            "classifier": "histgb",
            "histgb_max_leaf_nodes": 0,
            "histgb_lr": 0.1,
        },
    )

    assert report["valid"]
