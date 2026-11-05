from argparse import Namespace
import json

import pytest
import sklearn
import torch
from torch.utils.data import TensorDataset

from pepplatform.evaluators import Evaluator
from pepplatform.ml_trainers import ClassicalMLTrainer
from pepplatform.workflow import CLASSIFIER_PARAMETER_DEFAULTS


def _binary_dataset():
    train = TensorDataset(
        torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [1.0, 2.0],
            ],
            dtype=torch.float,
        ),
        torch.tensor([[0], [0], [1], [1], [1], [0]], dtype=torch.float),
    )
    val = TensorDataset(
        torch.tensor([[0.0, 0.0], [2.0, 1.0], [1.0, 2.0]], dtype=torch.float),
        torch.tensor([[0], [1], [0]], dtype=torch.float),
    )
    return train, val


def _base_args(tmp_path, model):
    return Namespace(
        task_type="pep_function",
        artifact_dir=str(tmp_path),
        tokenizer="kmer",
        encoder="bow",
        reduction="none",
        ml_scaler="none",
        ml_reduction="none",
        ml_auto_tune=True,
        ml_tuning_objective="f1",
        ml_tuning_mode="holdout",
        ml_tuning_max_candidates=2,
        ml_cv_folds=2,
        report_aupr=False,
        model=model,
        classifier=model,
    )


def test_rf_auto_tuning_records_selected_params(tmp_path):
    args = _base_args(tmp_path, "rf")
    args.rf_estimators = 5
    args.rf_max_depth = 0
    args.rf_min_samples_leaf = 1
    trainer = ClassicalMLTrainer(args)
    train, val = _binary_dataset()

    trainer.run_rf(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["tuning"]["classifier"] == "rf"
    assert metadata["tuning"]["mode"] == "holdout"
    assert metadata["tuning"]["objective"] == "f1"
    assert metadata["args"]["rf_estimators"] == metadata["tuning"]["best_params"]["rf_estimators"]
    assert metadata["tuning"]["candidates_evaluated"] == 2
    assert len(metadata["tuning"]["candidates"]) == 2
    assert any(candidate["selected"] for candidate in metadata["tuning"]["candidates"])
    contract = metadata["predictor_contract"]
    assert contract["contract_version"] == 1
    assert contract["implementation"] == "sklearn.ensemble.RandomForestClassifier"
    assert contract["implementation_version"] == sklearn.__version__
    assert contract["status"] == "verified"
    assert contract["aggregation"] == "mean_class_probability"
    assert contract["score_kind"] == "probability"
    assert contract["calibrated"] is False
    assert contract["estimator_params"]["n_estimators"] == metadata["args"]["rf_estimators"]


def test_rf_dual_tuning_records_best_score_without_key_error(tmp_path):
    args = _base_args(tmp_path, "rf")
    args.ml_tuning_objective = "dual_f1_sacc"
    args.tuning_objective_params = {"reference_f1": 0.4, "reference_sacc": 0.6}
    args.threshold_objective = "dual_f1_sacc"
    args.threshold_objective_params = args.tuning_objective_params
    args.rf_estimators = 5
    args.rf_max_depth = 0
    args.rf_min_samples_leaf = 1
    trainer = ClassicalMLTrainer(args)
    train, val = _binary_dataset()

    train = TensorDataset(
        train.tensors[0], torch.cat([train.tensors[1], 1 - train.tensors[1]], dim=1)
    )
    val = TensorDataset(val.tensors[0], torch.cat([val.tensors[1], 1 - val.tensors[1]], dim=1))

    trainer.run_rf(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    tuning = metadata["tuning"]
    assert tuning["objective"] == "dual_f1_sacc"
    assert "best_score" in tuning
    assert tuning["best_score"] == pytest.approx(
        Evaluator.dual_primary_score(
            tuning["best_metrics"]["f1"],
            tuning["best_metrics"]["sacc"],
            **args.tuning_objective_params,
        )
    )


def test_rbf_svm_cv_tuning_records_gamma(tmp_path):
    args = _base_args(tmp_path, "svm")
    args.ml_tuning_mode = "cv"
    args.ml_tuning_objective = "auc"
    args.ml_tuning_max_candidates = 4
    args.svm_c = 1.0
    args.svm_class_weight = "balanced"
    args.svm_max_iter = 1000
    args.svm_tol = 1e-3
    args.svm_kernel = "rbf"
    args.svm_gamma = "scale"
    trainer = ClassicalMLTrainer(args)
    train, val = _binary_dataset()

    trainer.run_svm(train, val)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["tuning"]["classifier"] == "svm"
    assert metadata["tuning"]["mode"] == "cv"
    assert metadata["tuning"]["objective"] == "auc"
    assert metadata["tuning"]["cv_folds"] == 2
    assert metadata["args"]["svm_kernel"] == metadata["tuning"]["best_params"]["svm_kernel"]
    assert "svm_gamma" in metadata["tuning"]["best_params"]
    assert metadata["tuning"]["candidates"][0]["params"]["svm_kernel"] == "rbf"
    assert metadata["tuning"]["candidates"][0]["params"]["svm_gamma"] == "scale"
    assert len(metadata["tuning"]["candidates"][0]["fold_metrics"]) == 2
    assert metadata["predictor_contract"]["algorithm"] == "support_vector_machine"


def test_missing_tuning_objective_uses_task_selection_policy():
    function_trainer = ClassicalMLTrainer(Namespace(task_type="pep_function"))
    pair_trainer = ClassicalMLTrainer(Namespace(task_type="pep_hlaii"))
    explicit_pair_trainer = ClassicalMLTrainer(
        Namespace(task_type="pep_hlaii", ml_tuning_objective="f1")
    )

    assert function_trainer._tuning_objective() == "f1"
    assert pair_trainer._tuning_objective() == "auc"
    assert explicit_pair_trainer._tuning_objective() == "f1"


@pytest.mark.parametrize(
    "classifier",
    ["rf", "svm", "logreg", "extratrees", "histgb", "xgb", "lgb"],
)
def test_all_classical_classifiers_have_deterministic_bounded_candidates(tmp_path, classifier):
    args = _base_args(tmp_path, classifier)
    for values in CLASSIFIER_PARAMETER_DEFAULTS.values():
        for key, value in values.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    trainer = ClassicalMLTrainer(args)

    first = trainer._candidate_params(classifier, "cv")
    second = trainer._candidate_params(classifier, "cv")

    assert first == second
    assert first
    assert len(first[:6]) <= len(first[:12]) <= len(first[:18])
    assert first[0] == {key: getattr(args, key) for key in first[0]}


def test_cv_preprocessing_is_fit_inside_each_fold(tmp_path, monkeypatch):
    args = _base_args(tmp_path, "rf")
    args.ml_scaler = "standard"
    args.rf_estimators = 2
    args.rf_max_depth = 0
    args.rf_min_samples_leaf = 1
    trainer = ClassicalMLTrainer(args)
    train, _ = _binary_dataset()
    train_x, train_y = trainer.collect_xy(train, "train")
    fit_sizes = []
    original = trainer.apply_ml_scaler

    def recording_scaler(fold_train_x, fold_val_x, test_x=None):
        fit_sizes.append(len(fold_train_x))
        return original(fold_train_x, fold_val_x, test_x)

    monkeypatch.setattr(trainer, "apply_ml_scaler", recording_scaler)
    result = trainer._score_candidate_cv(
        "rf",
        train_x,
        train_y,
        {"rf_estimators": 2, "rf_max_depth": 0, "rf_min_samples_leaf": 1},
        2,
    )

    assert fit_sizes == [3, 3]
    assert len(result["fold_metrics"]) == 2
    assert trainer._ml_scaler is None
