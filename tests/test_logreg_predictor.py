from argparse import Namespace
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import TensorDataset

from pepplatform.ml_trainers import (
    LOGREG_PREDICTOR_CONTRACT_VERSION,
    ClassicalMLTrainer,
    LogRegConvergenceError,
    logreg_predictor_contract,
)
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from scripts.train_pair_pipeline import build_argument_parser as build_pair_parser
from scripts.train_sequence_pipeline import build_argument_parser as build_sequence_parser


def _args(**overrides):
    values = {
        "task_type": "pep_function",
        "model": "logreg",
        "classifier": "logreg",
        "tokenizer": "kmer",
        "encoder": "bow",
        "seed": 17,
        "logreg_c": 1.0,
        "logreg_class_weight": "none",
        "logreg_max_iter": 2_000,
        "logreg_convergence_policy": "strict",
        "artifact_dir": "",
    }
    values.update(overrides)
    return Namespace(**values)


def _multioutput_data():
    rng = np.random.default_rng(31)
    features = rng.normal(size=(80, 8)).astype(np.float32)
    first = (features[:, 0] + 0.2 * features[:, 1] > 0).astype(int)
    second = (features[:, 2] - 0.4 * features[:, 3] > 0).astype(int)
    return features, np.column_stack([first, second])


def test_logreg_identity_contract_probability_and_batch_invariance():
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args())

    estimators = trainer._fit_logreg_estimators(features, labels)
    together = trainer._probability_ensemble_scores(estimators, features[:9])
    separately = np.concatenate(
        [trainer._probability_ensemble_scores(estimators, row[None, :]) for row in features[:9]],
        axis=0,
    )
    contract = logreg_predictor_contract(
        estimators,
        trainer._last_logreg_convergence,
        input_feature_count=features.shape[1],
    )

    assert all(isinstance(item["value"], LogisticRegression) for item in estimators)
    assert together.shape == (9, 2)
    assert np.isfinite(together).all()
    assert ((0.0 <= together) & (together <= 1.0)).all()
    np.testing.assert_allclose(together, separately, rtol=0.0, atol=1e-15)
    assert contract["contract_version"] == LOGREG_PREDICTOR_CONTRACT_VERSION
    assert contract["algorithm"] == "logistic_regression"
    assert contract["implementation"] == "sklearn.linear_model.LogisticRegression"
    assert contract["implementation_version"] == sklearn.__version__
    assert contract["status"] == "verified_platform_variant"
    assert contract["solver"] == "liblinear"
    assert contract["link"] == "logit"
    assert contract["penalty"] == "l2"
    assert contract["score_kind"] == "probability"
    assert contract["calibrated"] is False
    assert contract["convergence"]["status"] == "converged"


@pytest.mark.parametrize(
    ("features_transform", "labels_transform", "message"),
    [
        (lambda x: x[:0], lambda y: y[:0], "non-empty"),
        (None, lambda y: np.zeros_like(y), "both binary classes"),
        (
            lambda x: np.concatenate(
                [np.full((1, x.shape[1]), np.nan, dtype=x.dtype), x[1:]], axis=0
            ),
            None,
            "finite",
        ),
    ],
)
def test_logreg_rejects_invalid_or_degenerate_training_data(
    features_transform, labels_transform, message
):
    features, labels = _multioutput_data()
    if features_transform is not None:
        features = features_transform(features)
    if labels_transform is not None:
        labels = labels_transform(labels)
    trainer = ClassicalMLTrainer(_args())

    with pytest.raises(ValueError, match=message):
        trainer._fit_logreg_estimators(features, labels)


def test_logreg_rejects_nonpositive_c():
    features, labels = _multioutput_data()

    with pytest.raises(ValueError, match="greater than 0"):
        ClassicalMLTrainer(_args(logreg_c=0.0))._fit_logreg_estimators(features, labels)


def test_logreg_strict_policy_rejects_nonconverged_solver():
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args(logreg_c=10_000.0, logreg_max_iter=1))

    with pytest.raises(LogRegConvergenceError, match="did not converge") as exc:
        trainer._fit_logreg_estimators(features, labels)

    assert exc.value.code == "logreg_convergence_failed"
    assert exc.value.details["attempted_max_iters"] == "1"


def test_logreg_bounded_retry_changes_only_iteration_budget():
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(
        _args(logreg_max_iter=1, logreg_convergence_policy="bounded_retry")
    )

    estimators = trainer._fit_logreg_estimators(features, labels)
    trace = trainer._last_logreg_convergence["outputs"][0]
    fitted = estimators[0]["value"]

    assert trace["recovered"] is True
    assert trace["attempts"][0]["max_iter"] == 1
    assert trace["attempts"][-1]["status"] == "converged"
    assert fitted.max_iter in {5, 25}
    assert fitted.C == 1.0
    assert fitted.class_weight is None
    assert fitted.solver == "liblinear"
    assert fitted.random_state == 17


def test_logreg_joblib_reload_preserves_probabilities(tmp_path):
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args())
    estimators = trainer._fit_logreg_estimators(features, labels)
    expected = trainer._probability_ensemble_scores(estimators, features)
    path = tmp_path / "logreg.joblib"

    joblib.dump(estimators, path)
    restored = joblib.load(path)

    np.testing.assert_array_equal(
        expected, trainer._probability_ensemble_scores(restored, features)
    )


def test_logreg_training_entry_emits_current_contract(tmp_path):
    features, labels = _multioutput_data()
    train = TensorDataset(torch.from_numpy(features[:64]), torch.from_numpy(labels[:64]).float())
    validation = TensorDataset(
        torch.from_numpy(features[64:]), torch.from_numpy(labels[64:]).float()
    )
    args = _args(
        artifact_dir=str(tmp_path),
        reduction="none",
        ml_scaler="standard",
        ml_reduction="none",
        ml_auto_tune=False,
        report_aupr=False,
        _artifact_source="sequence_features",
        _artifact_info={"label_names": ["alpha", "beta"]},
    )

    result = ClassicalMLTrainer(args).run_logreg(train, validation)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    contract = metadata["predictor_contract"]
    assert np.isfinite(result)
    assert contract["algorithm"] == "logistic_regression"
    assert contract["convergence"]["status"] == "converged"
    assert metadata["operating_point"]["score_kind"] == "probability"
    ModelService._validate_logreg_predictor_contract(metadata)


def test_logreg_legacy_artifact_is_explicitly_legacy_only():
    metadata = {"args": {"classifier": "logreg"}, "label_names": ["label"]}

    with pytest.warns(RuntimeWarning, match="legacy_only"):
        ModelService._validate_logreg_predictor_contract(metadata)


def test_logreg_web_validator_rejects_version_mismatch():
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args())
    estimators = trainer._fit_logreg_estimators(features, labels)
    contract = logreg_predictor_contract(
        estimators,
        trainer._last_logreg_convergence,
        input_feature_count=features.shape[1],
    )
    contract["implementation_version"] = "0.0"

    with pytest.raises(InferenceUnsupportedError) as exc:
        ModelService._validate_logreg_predictor_contract(
            {"args": {"classifier": "logreg"}, "predictor_contract": contract}
        )

    assert exc.value.code == "logreg_runtime_version_mismatch"


def test_logreg_tuning_skips_nonconverged_candidate(monkeypatch):
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(
        _args(
            ml_auto_tune=True,
            ml_tuning_mode="holdout",
            ml_tuning_objective="f1",
            ml_tuning_max_candidates=2,
        )
    )
    candidates = [
        {"logreg_c": 1.0, "logreg_class_weight": "none"},
        {"logreg_c": 0.1, "logreg_class_weight": "none"},
    ]
    monkeypatch.setattr(trainer, "_candidate_params", lambda *args: candidates)
    calls = {"count": 0}

    def score(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise LogRegConvergenceError("candidate did not converge")
        return {
            "metrics": {
                "acc": 1.0,
                "sacc": 1.0,
                "f1": 1.0,
                "auc": 1.0,
                "balanced_accuracy": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "specificity": 1.0,
                "mcc": 1.0,
            },
            "fold_metrics": [],
        }

    monkeypatch.setattr(trainer, "_score_candidate_holdout", score)
    trainer._maybe_auto_tune("logreg", features, labels, features[:8], labels[:8])

    assert trainer._last_tuning_result["candidates_failed"] == 1
    assert trainer._last_tuning_result["best_params"]["logreg_c"] == 0.1


def test_logreg_cli_and_paper_entrypoints_default_to_strict_convergence():
    assert build_sequence_parser().parse_args([]).logreg_convergence_policy == "strict"
    assert build_pair_parser().parse_args([]).logreg_convergence_policy == "strict"
