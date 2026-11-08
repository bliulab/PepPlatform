from argparse import Namespace
import json

import joblib
import numpy as np
import pytest
import sklearn
import torch
from torch.utils.data import TensorDataset

from pepplatform.ml_trainers import (
    EXTRATREES_PREDICTOR_CONTRACT_VERSION,
    ClassicalMLTrainer,
    extratrees_predictor_contract,
)
from pepplatform.web.inference import InferenceUnsupportedError, ModelService


def _args(**overrides):
    values = {
        "task_type": "pep_function",
        "tokenizer": "kmer",
        "encoder": "bow",
        "model": "extratrees",
        "classifier": "extratrees",
        "artifact_dir": "",
        "reduction": "none",
        "ml_scaler": "none",
        "ml_reduction": "none",
        "ml_auto_tune": False,
        "report_aupr": False,
        "et_estimators": 30,
        "et_max_depth": 0,
        "et_min_samples_leaf": 1,
        "seed": 17,
    }
    values.update(overrides)
    return Namespace(**values)


def _binary_data():
    rng = np.random.RandomState(7)
    features = rng.normal(size=(40, 6)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(int)[:, None]
    return features, labels


def _tensor_splits(test_labels):
    features, labels = _binary_data()
    train = TensorDataset(torch.from_numpy(features[:28]), torch.from_numpy(labels[:28]))
    val = TensorDataset(torch.from_numpy(features[28:34]), torch.from_numpy(labels[28:34]))
    test = TensorDataset(
        torch.from_numpy(features[34:]), torch.as_tensor(test_labels, dtype=torch.long)
    )
    return train, val, test, features[34:]


def _web_contract(version=None):
    return {
        "contract_version": EXTRATREES_PREDICTOR_CONTRACT_VERSION,
        "algorithm": "extra_trees",
        "implementation": "sklearn.ensemble.ExtraTreesClassifier",
        "implementation_version": version or sklearn.__version__,
        "status": "verified",
        "architecture": "whole_sample_extremely_randomized_tree_ensemble",
        "split_rule": "random_threshold_per_candidate_feature_then_best_impurity",
        "aggregation": "mean_class_probability",
        "score_kind": "probability",
        "calibrated": False,
        "multioutput": "native_sklearn_multi_output",
    }


def test_extratrees_identity_contract_and_progress_fit_match_one_shot():
    features, labels = _binary_data()
    trainer = ClassicalMLTrainer(_args())

    one_shot = trainer._fit_extra_trees_model(features, labels, progress=False)
    progressive = trainer._fit_extra_trees_model(features, labels, progress=True)

    np.testing.assert_array_equal(
        trainer._rf_scores(one_shot, features),
        trainer._rf_scores(progressive, features),
    )
    assert progressive.estimator_.splitter == "random"
    contract = extratrees_predictor_contract(progressive)
    assert contract == {
        "contract_version": EXTRATREES_PREDICTOR_CONTRACT_VERSION,
        "algorithm": "extra_trees",
        "implementation": "sklearn.ensemble.ExtraTreesClassifier",
        "implementation_version": sklearn.__version__,
        "status": "verified",
        "input_contract": "fixed_width_global_vector",
        "input_feature_count": 6,
        "architecture": "whole_sample_extremely_randomized_tree_ensemble",
        "split_rule": "random_threshold_per_candidate_feature_then_best_impurity",
        "aggregation": "mean_class_probability",
        "score_kind": "probability",
        "calibrated": False,
        "multioutput": "native_sklearn_multi_output",
        "estimator_params": {
            "n_estimators": 30,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": False,
            "random_state": 17,
            "n_jobs": -1,
            "warm_start": True,
        },
        "serialization": {
            "format": "joblib",
            "requires_matching_sklearn_version": True,
        },
    }


def test_extratrees_contract_rejects_non_identity_configuration():
    features, labels = _binary_data()
    trainer = ClassicalMLTrainer(_args())
    estimator = trainer._fit_extra_trees_model(features, labels, progress=False)
    estimator.set_params(bootstrap=True)

    with pytest.raises(ValueError, match="without bootstrap"):
        extratrees_predictor_contract(estimator)


def test_extratrees_multioutput_probabilities_are_finite_and_batch_invariant():
    features, labels = _binary_data()
    multi_labels = np.column_stack(
        [
            np.zeros(features.shape[0], dtype=int),
            np.ones(features.shape[0], dtype=int),
            labels[:, 0],
        ]
    )
    trainer = ClassicalMLTrainer(_args())
    estimator = trainer._fit_extra_trees_model(features, multi_labels, progress=False)

    together = trainer._rf_scores(estimator, features[:5])
    separately = np.concatenate(
        [trainer._rf_scores(estimator, row[None, :]) for row in features[:5]],
        axis=0,
    )

    assert together.shape == (5, 3)
    assert np.isfinite(together).all()
    assert ((0.0 <= together) & (together <= 1.0)).all()
    np.testing.assert_array_equal(together[:, 0], np.zeros(5))
    np.testing.assert_array_equal(together[:, 1], np.ones(5))
    np.testing.assert_array_equal(together, separately)


def test_extratrees_tuning_uses_probability_fallback_for_single_negative_class(
    monkeypatch,
):
    trainer = ClassicalMLTrainer(_args())
    train_x = np.zeros((4, 2), dtype=np.float32)
    train_y = np.zeros((4, 1), dtype=int)
    val_x = np.zeros((3, 2), dtype=np.float32)
    val_y = np.zeros((3, 1), dtype=int)
    monkeypatch.setattr(trainer, "_fit_tuning_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        trainer,
        "_tuning_scores",
        lambda *args, **kwargs: np.zeros((3, 1), dtype=np.float32),
    )

    result = trainer._score_candidate_holdout("extratrees", train_x, train_y, val_x, val_y, {})

    assert result["metrics"]["acc"] == 1.0
    assert result["metrics"]["specificity"] == 1.0


def test_extratrees_joblib_reload_preserves_probability_scores(tmp_path):
    features, labels = _binary_data()
    trainer = ClassicalMLTrainer(_args())
    estimator = trainer._fit_extra_trees_model(features, labels, progress=False)
    expected = trainer._rf_scores(estimator, features)
    path = tmp_path / "extratrees.joblib"

    joblib.dump(estimator, path)
    restored = joblib.load(path)

    np.testing.assert_array_equal(expected, trainer._rf_scores(restored, features))


def test_extratrees_rejects_empty_training_data():
    trainer = ClassicalMLTrainer(_args())

    with pytest.raises(ValueError):
        trainer._fit_extra_trees_model(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 1), dtype=int),
            progress=False,
        )


def test_extratrees_artifact_and_operating_point_are_test_label_independent(tmp_path):
    outputs = []
    for name, test_labels in [
        ("original", [[0], [0], [1], [1], [0], [1]]),
        ("flipped", [[1], [1], [0], [0], [1], [0]]),
    ]:
        artifact_dir = tmp_path / name
        train, val, test, test_features = _tensor_splits(test_labels)
        trainer = ClassicalMLTrainer(_args(artifact_dir=str(artifact_dir), et_estimators=12))
        trainer.run_extratrees(train, val, test)
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        evaluation = json.loads((artifact_dir / "evaluation.json").read_text(encoding="utf-8"))
        payload = joblib.load(artifact_dir / "model.joblib")
        outputs.append(
            (
                metadata,
                evaluation,
                ModelService._predict_classical_scores(payload, test_features),
            )
        )

    original, flipped = outputs
    assert original[0]["predictor_contract"]["algorithm"] == "extra_trees"
    assert original[0]["predictor_contract"]["input_feature_count"] == 6
    assert original[1]["feature_fit_scope"] == "train_only"
    assert original[1]["operating_point"]["threshold_source"] == "validation_macro_f1"
    assert original[0]["operating_point"] == flipped[0]["operating_point"]
    assert original[1]["validation"] == flipped[1]["validation"]
    np.testing.assert_array_equal(original[2], flipped[2])


def test_legacy_extratrees_contract_warns_but_remains_loadable():
    metadata = {"args": {"model": "extratrees", "classifier": "extratrees"}}

    with pytest.warns(RuntimeWarning, match="legacy_only"):
        ModelService._validate_extratrees_predictor_contract(metadata)


def test_current_extratrees_contract_accepts_matching_sklearn_version():
    metadata = {
        "args": {"model": "extratrees", "classifier": "extratrees"},
        "predictor_contract": _web_contract(),
    }

    ModelService._validate_extratrees_predictor_contract(metadata)


def test_current_extratrees_contract_rejects_incomplete_contract():
    contract = _web_contract()
    contract.pop("split_rule")
    metadata = {
        "args": {"model": "extratrees", "classifier": "extratrees"},
        "predictor_contract": contract,
    }

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_extratrees_predictor_contract(metadata)

    assert exc_info.value.code == "extratrees_predictor_contract_unsupported"
    assert exc_info.value.stage == "bundle"


def test_current_extratrees_contract_rejects_sklearn_version_mismatch():
    metadata = {
        "args": {"model": "extratrees", "classifier": "extratrees"},
        "predictor_contract": _web_contract("0.0.invalid"),
    }

    with pytest.raises(InferenceUnsupportedError) as exc_info:
        ModelService._validate_extratrees_predictor_contract(metadata)

    assert exc_info.value.code == "extratrees_runtime_version_mismatch"
    assert exc_info.value.stage == "bundle"
    assert "0.0.invalid" in exc_info.value.technical_detail
