from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.ensemble import ExtraTreesClassifier
from torch.utils.data import TensorDataset

from pepplatform.featurizers import FeaturizerFactory
from pepplatform.ml_trainers import ClassicalMLTrainer
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from pepplatform.web.preflight import preflight_experiment
from pepplatform.workflow import CLASSIC_ENCODER_OPTIONS, CLASSIC_TOKENIZERS, CLASSICAL_CLASSIFIERS


def _trainer_args(**overrides):
    base = dict(
        artifact_dir="",
        task_type="pep_function",
        reduction="none",
        ml_scaler="standard",
        ml_reduction="none",
        ml_reduction_dim=2,
        kpca_kernel="rbf",
        kpca_gamma=0.1,
        ml_auto_tune=False,
        ml_tuning_objective="f1",
        ml_tuning_mode="holdout",
        ml_tuning_max_candidates=2,
        ml_cv_folds=2,
        report_aupr=False,
        svm_c=1.0,
        svm_class_weight="balanced",
        svm_max_iter=200,
        svm_tol=1e-3,
        svm_kernel="linear",
        svm_gamma="scale",
        et_estimators=8,
        et_max_depth=0,
        et_min_samples_leaf=1,
        histgb_max_iter=8,
        histgb_max_leaf_nodes=7,
        histgb_lr=0.1,
        histgb_l2_regularization=0.0,
        _artifact_source="sequence_features",
        _artifact_info={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _toy_splits():
    train_x = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
        ],
        dtype=torch.float32,
    )
    train_y = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=torch.long)
    val_x = torch.tensor([[0.2, 0.0], [0.0, 0.2], [0.9, 1.0], [1.0, 0.9]], dtype=torch.float32)
    val_y = torch.tensor([[0], [0], [1], [1]], dtype=torch.long)
    return TensorDataset(train_x, train_y), TensorDataset(val_x, val_y)


def test_workflow_exposes_new_options():
    removed = {"mismatch", "multi-kmer", "reduced-aa"}
    assert removed.isdisjoint(CLASSIC_TOKENIZERS)
    assert removed.isdisjoint(CLASSIC_ENCODER_OPTIONS)
    for classifier in ["svm", "extratrees", "histgb"]:
        assert classifier in CLASSICAL_CLASSIFIERS
    assert "calibrated_svm" not in CLASSICAL_CLASSIFIERS


@pytest.mark.parametrize("tokenizer", ["mismatch", "multi-kmer", "reduced-aa"])
def test_removed_tokenizers_are_rejected(tokenizer):
    factory = FeaturizerFactory(bpe_path="")

    with pytest.raises(ValueError, match="unsupported tokenizer"):
        factory.create(tokenizer, "tfidf", max_length=40, k=2)


@pytest.mark.parametrize("tokenizer", ["multi-kmer", "reduced-aa"])
def test_preflight_rejects_removed_tokenizer(tokenizer):
    dataset = {
        "id": "toy",
        "name": "toy",
        "task_type": "pep_function",
        "format": "sequence_dir",
        "metadata": {"splits": {"train": 6, "val": 4}},
    }
    report = preflight_experiment(
        dataset,
        "pep_function",
        {"tokenizer": tokenizer, "encoder": "tfidf", "model": "histgb", "classifier": "histgb"},
    )
    assert report["valid"] is False
    assert "route_resolution_error" in report["error_codes"]
    assert any(f"unsupported tokenizer: {tokenizer}" in error for error in report["errors"])


@pytest.mark.parametrize("tokenizer", ["multi-kmer", "reduced-aa"])
def test_inference_rejects_removed_tokenizer_artifact(tokenizer):
    with pytest.raises(
        InferenceUnsupportedError,
        match="unsupported tokenizer in classic artifact",
    ):
        ModelService._build_classic_featurizer(
            {"classic": {"tokenizer": tokenizer, "encoder": "tfidf"}}
        )


def test_new_classical_trainers_smoke():
    train_ds, val_ds = _toy_splits()
    for method_name in ["run_calibrated_svm", "run_extratrees", "run_histgb"]:
        trainer = ClassicalMLTrainer(_trainer_args())
        score = getattr(trainer, method_name)(train_ds, val_ds)
        assert 0.0 <= score <= 1.0


def test_tree_ensemble_prediction_helper_returns_positive_probabilities():
    train_x = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32)
    train_y = np.array([0, 0, 1, 1])
    estimator = ExtraTreesClassifier(n_estimators=4, random_state=42).fit(train_x, train_y)
    scores = ModelService._predict_classical_scores(
        {"estimator_kind": "tree_ensemble", "estimator": estimator},
        train_x,
    )
    assert scores.shape == (4, 1)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
