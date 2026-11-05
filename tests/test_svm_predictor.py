from argparse import Namespace
import json

import joblib
import numpy as np
import pytest
import sklearn
import torch
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import TensorDataset

from pepplatform.ml_trainers import (
    SVM_PREDICTOR_CONTRACT_VERSION,
    ClassicalMLTrainer,
    SVMConvergenceError,
    svm_predictor_contract,
)


def _args(**overrides):
    values = {
        "seed": 17,
        "svm_c": 1.0,
        "svm_class_weight": "none",
        "svm_kernel": "linear",
        "svm_gamma": "scale",
        "svm_max_iter": 10_000,
        "svm_tol": 1e-4,
    }
    values.update(overrides)
    return Namespace(**values)


def _binary_data():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(60, 6)).astype(np.float32)
    labels = (features[:, 0] + 0.25 * features[:, 1] > 0).astype(int)[:, None]
    return features, labels


def _multioutput_data():
    features, labels = _binary_data()
    second = (features[:, 2] - features[:, 3] > 0).astype(int)
    return features, np.column_stack([labels[:, 0], second])


def test_linear_svm_identity_contract_and_margin_behavior():
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args())

    estimators = trainer._fit_svm_estimators(features, labels)
    together = trainer._svm_scores(estimators, features[:8])
    separately = np.concatenate(
        [trainer._svm_scores(estimators, row[None, :]) for row in features[:8]], axis=0
    )
    contract = svm_predictor_contract(estimators)

    assert all(isinstance(estimator, LinearSVC) for estimator in estimators)
    assert together.shape == (8, 2)
    assert np.isfinite(together).all()
    np.testing.assert_allclose(together, separately, rtol=0.0, atol=1e-12)
    assert contract["contract_version"] == SVM_PREDICTOR_CONTRACT_VERSION
    assert contract["algorithm"] == "support_vector_machine"
    assert contract["implementation"] == "sklearn.svm.LinearSVC"
    assert contract["implementation_version"] == sklearn.__version__
    assert contract["status"] == "verified_platform_variant"
    assert contract["kernel"] == "linear"
    assert contract["solver"] == "liblinear"
    assert contract["score_kind"] == "margin"
    assert contract["calibrated"] is False
    assert contract["multioutput"] == "independent_binary_one_vs_rest"
    assert contract["num_outputs"] == 2
    assert contract["estimator_params"]["loss"] == "squared_hinge"
    assert contract["estimator_params"]["penalty"] == "l2"
    assert contract["estimator_params"]["dual"] == "auto"
    assert contract["estimator_params"]["random_state"] == 17
    assert contract["convergence"]["status"] == "converged"


def test_rbf_svm_identity_contract_uses_support_vectors():
    features, labels = _binary_data()
    trainer = ClassicalMLTrainer(_args(svm_kernel="rbf", svm_gamma=0.2))

    estimators = trainer._fit_svm_estimators(features, labels)
    scores = trainer._svm_scores(estimators, features[:5])
    contract = svm_predictor_contract(estimators)

    assert all(isinstance(estimator, SVC) for estimator in estimators)
    assert estimators[0].support_vectors_.shape[0] > 0
    assert scores.shape == (5, 1)
    assert np.isfinite(scores).all()
    assert contract["implementation"] == "sklearn.svm.SVC"
    assert contract["kernel"] == "rbf"
    assert contract["solver"] == "libsvm"
    assert contract["estimator_params"]["kernel"] == "rbf"
    assert contract["estimator_params"]["gamma"] == 0.2
    assert contract["estimator_params"]["fitted_gamma"] == pytest.approx(0.2)
    assert contract["estimator_params"]["probability"] is False


def test_svm_contract_records_per_output_effective_iteration_budgets():
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args(svm_max_iter=2000))
    estimators = trainer._fit_svm_estimators(features, labels)
    estimators[1].set_params(max_iter=10_000)

    contract = svm_predictor_contract(
        estimators,
        convergence={"requested_max_iter": 2000},
    )

    assert contract["estimator_params"]["max_iter"] == 2000
    assert contract["estimator_params"]["requested_max_iter"] == 2000
    assert contract["estimator_params"]["effective_max_iter_by_output"] == [2000, 10_000]
    assert contract["convergence"]["effective_max_iter_by_output"] == [2000, 10_000]


def test_svm_contract_still_rejects_model_parameter_mismatch():
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args())
    estimators = trainer._fit_svm_estimators(features, labels)
    estimators[1].set_params(C=2.0)

    with pytest.raises(ValueError, match="shared parameters"):
        svm_predictor_contract(estimators)


def test_svm_joblib_reload_preserves_margins(tmp_path):
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args())
    estimators = trainer._fit_svm_estimators(features, labels)
    expected = trainer._svm_scores(estimators, features)
    path = tmp_path / "svm.joblib"

    joblib.dump(estimators, path)
    restored = joblib.load(path)

    np.testing.assert_array_equal(expected, trainer._svm_scores(restored, features))


@pytest.mark.parametrize(
    ("features", "labels", "message"),
    [
        (np.empty((0, 2), dtype=np.float32), np.empty((0, 1), dtype=int), "non-empty"),
        (np.ones((4, 2), dtype=np.float32), np.zeros((4, 1), dtype=int), "both binary classes"),
        (
            np.ones((4, 2), dtype=np.float32),
            np.asarray([[-1], [-1], [1], [1]]),
            "both binary classes",
        ),
        (
            np.asarray([[0.0, np.nan], [1.0, 0.0]], dtype=np.float32),
            np.asarray([[0], [1]]),
            "finite",
        ),
    ],
)
def test_new_svm_training_rejects_invalid_or_degenerate_data(features, labels, message):
    trainer = ClassicalMLTrainer(_args())

    with pytest.raises(ValueError, match=message):
        trainer._fit_svm_estimators(features, labels)


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_new_svm_training_rejects_nonconverged_solver(kernel):
    rng = np.random.default_rng(19)
    features = rng.normal(size=(80, 12)).astype(np.float32)
    labels = (rng.normal(size=80) > 0).astype(int)[:, None]
    trainer = ClassicalMLTrainer(
        _args(
            svm_kernel=kernel,
            svm_c=10_000.0,
            svm_max_iter=1,
            svm_tol=1e-12,
        )
    )

    with pytest.raises(SVMConvergenceError, match="did not converge"):
        trainer._fit_svm_estimators(features, labels)


def test_web_bounded_policy_recovers_by_changing_only_iteration_budget():
    features, labels = _binary_data()
    trainer = ClassicalMLTrainer(
        _args(
            svm_kernel="linear",
            svm_max_iter=1,
            svm_tol=1e-4,
            svm_convergence_policy="bounded_retry",
        )
    )

    estimators = trainer._fit_svm_estimators(features, labels)
    trace = trainer._last_svm_convergence["outputs"][0]

    assert estimators[0].max_iter in {5, 25, 50_000}
    assert trace["recovered"] is True
    assert trace["attempts"][0]["max_iter"] == 1
    assert trace["attempts"][-1]["status"] == "converged"
    assert estimators[0].C == 1.0
    assert estimators[0].tol == 1e-4


def test_svm_tuning_records_and_skips_nonconverged_candidate(monkeypatch):
    features, labels = _binary_data()
    trainer = ClassicalMLTrainer(
        _args(
            task_type="pep_function",
            ml_auto_tune=True,
            ml_tuning_mode="holdout",
            ml_tuning_objective="f1",
            ml_tuning_max_candidates=2,
        )
    )
    candidates = [
        {"svm_c": 1.0, "svm_class_weight": "none", "svm_kernel": "linear", "svm_gamma": "scale"},
        {"svm_c": 0.1, "svm_class_weight": "none", "svm_kernel": "linear", "svm_gamma": "scale"},
    ]
    monkeypatch.setattr(trainer, "_candidate_params", lambda *args: candidates)
    calls = {"count": 0}

    def score(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise SVMConvergenceError("candidate did not converge")
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
    trainer._maybe_auto_tune("svm", features, labels, features[:8], labels[:8])

    assert trainer._last_tuning_result["candidates_failed"] == 1
    assert trainer._last_tuning_result["candidates_succeeded"] == 1
    assert trainer._last_tuning_result["candidates"][0]["status"] == "failed"
    assert trainer._last_tuning_result["best_params"]["svm_c"] == 0.1
    assert trainer.args.svm_c == 0.1


def test_svm_tuning_fails_when_every_candidate_is_nonconverged(monkeypatch):
    features, labels = _binary_data()
    trainer = ClassicalMLTrainer(
        _args(
            task_type="pep_function",
            ml_auto_tune=True,
            ml_tuning_mode="holdout",
            ml_tuning_objective="f1",
            ml_tuning_max_candidates=1,
        )
    )
    monkeypatch.setattr(
        trainer,
        "_score_candidate_holdout",
        lambda *args, **kwargs: (_ for _ in ()).throw(SVMConvergenceError("no convergence")),
    )

    with pytest.raises(SVMConvergenceError, match="all SVM tuning candidates"):
        trainer._maybe_auto_tune("svm", features, labels, features[:8], labels[:8])


def test_sequence_svm_artifact_records_contract_v1(tmp_path):
    features, labels = _binary_data()
    train = TensorDataset(torch.from_numpy(features[:48]), torch.from_numpy(labels[:48]).float())
    val = TensorDataset(torch.from_numpy(features[48:]), torch.from_numpy(labels[48:]).float())
    args = _args(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="bow",
        model="svm",
        classifier="svm",
        reduction="none",
        ml_scaler="standard",
        ml_reduction="none",
        ml_auto_tune=False,
        report_aupr=False,
    )

    ClassicalMLTrainer(args).run_svm(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    contract = metadata["predictor_contract"]
    assert metadata["bundle_type"] == "sequence_classical"
    assert contract["contract_version"] == SVM_PREDICTOR_CONTRACT_VERSION
    assert contract["algorithm"] == "support_vector_machine"
    assert contract["score_kind"] == "margin"
    assert contract["convergence"]["status"] == "converged"
    assert metadata["operating_point"]["score_kind"] == "margin"
