from argparse import Namespace

import joblib
import numpy as np
import pytest
import sklearn

from pepplatform.ml_trainers import (
    RF_PREDICTOR_CONTRACT_VERSION,
    ClassicalMLTrainer,
    rf_predictor_contract,
)


def _args(**overrides):
    values = {
        "rf_estimators": 30,
        "rf_max_depth": 0,
        "rf_min_samples_leaf": 1,
        "seed": 17,
    }
    values.update(overrides)
    return Namespace(**values)


def _binary_data():
    rng = np.random.RandomState(7)
    features = rng.normal(size=(40, 6)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(int)[:, None]
    return features, labels


def test_rf_identity_contract_and_progress_fit_match_one_shot():
    features, labels = _binary_data()
    trainer = ClassicalMLTrainer(_args())

    one_shot = trainer._fit_rf_model(features, labels, progress=False)
    progressive = trainer._fit_rf_model(features, labels, progress=True)

    np.testing.assert_array_equal(
        trainer._rf_scores(one_shot, features),
        trainer._rf_scores(progressive, features),
    )
    contract = rf_predictor_contract(progressive)
    assert contract == {
        "contract_version": RF_PREDICTOR_CONTRACT_VERSION,
        "algorithm": "random_forest",
        "implementation": "sklearn.ensemble.RandomForestClassifier",
        "implementation_version": sklearn.__version__,
        "status": "verified",
        "input_contract": "fixed_width_global_vector",
        "architecture": "bootstrap_tree_ensemble_with_random_feature_subsampling",
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
            "bootstrap": True,
            "random_state": 17,
            "n_jobs": -1,
            "warm_start": True,
        },
        "serialization": {
            "format": "joblib",
            "requires_matching_sklearn_version": True,
        },
    }


def test_rf_multioutput_probabilities_are_finite_and_batch_invariant():
    features, labels = _binary_data()
    multi_labels = np.column_stack([np.zeros(features.shape[0], dtype=int), labels[:, 0]])
    trainer = ClassicalMLTrainer(_args())
    estimator = trainer._fit_rf_model(features, multi_labels, progress=False)

    together = trainer._rf_scores(estimator, features[:5])
    separately = np.concatenate(
        [trainer._rf_scores(estimator, row[None, :]) for row in features[:5]],
        axis=0,
    )

    assert together.shape == (5, 2)
    assert np.isfinite(together).all()
    assert ((0.0 <= together) & (together <= 1.0)).all()
    np.testing.assert_array_equal(together[:, 0], np.zeros(5))
    np.testing.assert_array_equal(together, separately)


def test_rf_tuning_uses_probability_fallback_for_single_negative_class(monkeypatch):
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

    result = trainer._score_candidate_holdout("rf", train_x, train_y, val_x, val_y, {})

    assert result["metrics"]["acc"] == 1.0
    assert result["metrics"]["specificity"] == 1.0


def test_rf_joblib_reload_preserves_probability_scores(tmp_path):
    features, labels = _binary_data()
    trainer = ClassicalMLTrainer(_args())
    estimator = trainer._fit_rf_model(features, labels, progress=False)
    expected = trainer._rf_scores(estimator, features)
    path = tmp_path / "rf.joblib"

    joblib.dump(estimator, path)
    restored = joblib.load(path)

    np.testing.assert_array_equal(expected, trainer._rf_scores(restored, features))


def test_rf_rejects_empty_training_data():
    trainer = ClassicalMLTrainer(_args())

    with pytest.raises(ValueError):
        trainer._fit_rf_model(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 1), dtype=int),
            progress=False,
        )
