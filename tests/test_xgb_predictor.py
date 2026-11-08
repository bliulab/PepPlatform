from argparse import Namespace
import hashlib
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import sklearn
import torch
import xgboost
from torch.utils.data import TensorDataset

from pepplatform.ml_trainers import (
    XGB_MANIFEST_SCHEMA_VERSION,
    XGB_PREDICTOR_CONTRACT_VERSION,
    ClassicalMLTrainer,
    XGBTrainingDataError,
    xgb_predictor_contract,
)
from pepplatform.web.inference import InferenceUnsupportedError, ModelService


def _args(**overrides):
    values = {
        "task_type": "pep_function",
        "model": "xgb",
        "classifier": "xgb",
        "tokenizer": "kmer",
        "encoder": "bow",
        "seed": 17,
        "xgb_estimators": 8,
        "xgb_max_depth": 2,
        "xgb_lr": 0.15,
        "xgb_subsample": 1.0,
        "xgb_colsample": 1.0,
        "xgb_reg_alpha": 0.0,
        "xgb_reg_lambda": 1.0,
        "artifact_dir": "",
    }
    values.update(overrides)
    return Namespace(**values)


def _multioutput_data():
    rng = np.random.default_rng(29)
    features = rng.normal(size=(64, 7)).astype(np.float32)
    first = (features[:, 0] + 0.3 * features[:, 1] > 0).astype(int)
    second = (features[:, 2] - 0.2 * features[:, 3] > 0).astype(int)
    return features, np.column_stack([first, second])


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_xgb_identity_contract_probability_and_batch_invariance():
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args(xgb_max_depth=0))

    estimators = trainer._fit_boosting_estimators("xgb", features, labels)
    together = trainer._probability_ensemble_scores(estimators, features[:9])
    separately = np.concatenate(
        [trainer._probability_ensemble_scores(estimators, row[None, :]) for row in features[:9]],
        axis=0,
    )
    contract = xgb_predictor_contract(estimators, input_feature_count=features.shape[1])

    assert all(isinstance(item["value"], xgboost.XGBClassifier) for item in estimators)
    assert together.shape == (9, 2)
    assert np.isfinite(together).all()
    assert ((0.0 <= together) & (together <= 1.0)).all()
    np.testing.assert_array_equal(together, separately)
    assert contract["contract_version"] == XGB_PREDICTOR_CONTRACT_VERSION
    assert contract["algorithm"] == "xgboost"
    assert contract["implementation"] == "xgboost.XGBClassifier"
    assert contract["implementation_version"] == xgboost.__version__
    assert contract["sklearn_version"] == sklearn.__version__
    assert contract["status"] == "verified_platform_variant"
    assert (
        contract["architecture"]
        == "independent_binary_regularized_gradient_tree_booster_per_output"
    )
    assert contract["objective"] == "binary:logistic"
    assert contract["score_kind"] == "probability"
    assert contract["calibrated"] is False
    assert contract["estimator_params"]["max_depth"] == 0
    assert all(item["max_depth"] == 0 for item in contract["effective_params_by_output"])


@pytest.mark.parametrize(
    ("overrides", "labels_transform", "message"),
    [
        ({"xgb_subsample": 0.0}, None, "subsample"),
        ({"xgb_colsample": 0.0}, None, "colsample"),
        ({"xgb_max_depth": -1}, None, "max_depth"),
        ({}, lambda labels: np.zeros_like(labels), "both binary classes"),
    ],
)
def test_xgb_rejects_invalid_parameters_and_degenerate_outputs(
    overrides, labels_transform, message
):
    features, labels = _multioutput_data()
    if labels_transform is not None:
        labels = labels_transform(labels)
    trainer = ClassicalMLTrainer(_args(**overrides))

    with pytest.raises(XGBTrainingDataError, match=message):
        trainer._fit_boosting_estimators("xgb", features, labels)


def test_xgb_hybrid_artifact_roundtrip(tmp_path):
    features, labels = _multioutput_data()
    args = _args(
        artifact_dir=str(tmp_path),
        _artifact_source="sequence_features",
        _artifact_info={"label_names": ["alpha", "beta"]},
    )
    trainer = ClassicalMLTrainer(args)
    estimators = trainer._fit_boosting_estimators("xgb", features, labels)
    expected = trainer._probability_ensemble_scores(estimators, features[:11])
    trainer._maybe_save_artifact(
        {
            "classifier": "xgb",
            "estimator_kind": "probability_ensemble",
            "estimators": estimators,
            "input_feature_count": features.shape[1],
            "_artifact_verification_x": features[:11],
        },
        SimpleNamespace(label_names=["alpha", "beta"]),
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    manifest_path = tmp_path / "xgb_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = joblib.load(tmp_path / "model.joblib")

    assert payload["estimator_kind"] == "xgboost_native_ensemble"
    assert payload["manifest_path"] == "xgb_manifest.json"
    assert "estimators" not in payload
    assert manifest["schema_version"] == XGB_MANIFEST_SCHEMA_VERSION
    assert manifest["reload_verification"]["status"] == "passed"
    assert [item["label_name"] for item in manifest["outputs"]] == ["alpha", "beta"]
    for output in manifest["outputs"]:
        model_path = tmp_path / output["model_path"]
        assert model_path.suffix == ".ubj"
        assert output["size_bytes"] == model_path.stat().st_size
        assert output["sha256"] == _sha256(model_path)
    assert metadata["predictor_contract"]["serialization"]["manifest_sha256"] == _sha256(
        manifest_path
    )

    ModelService._validate_xgb_predictor_contract(metadata, tmp_path)
    restored = ModelService._load_classical_payload(tmp_path, metadata)
    actual = ModelService._predict_classical_scores(restored, features[:11])
    np.testing.assert_array_equal(expected, actual)


def test_xgb_training_entry_emits_current_native_artifact(tmp_path):
    features, labels = _multioutput_data()
    train = TensorDataset(torch.from_numpy(features[:48]), torch.from_numpy(labels[:48]).float())
    validation = TensorDataset(
        torch.from_numpy(features[48:]), torch.from_numpy(labels[48:]).float()
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

    result = ClassicalMLTrainer(args).run_xgb(train, validation)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    payload = joblib.load(tmp_path / "model.joblib")
    assert np.isfinite(result)
    assert metadata["predictor_contract"]["algorithm"] == "xgboost"
    assert metadata["operating_point"]["score_kind"] == "probability"
    assert payload["estimator_kind"] == "xgboost_native_ensemble"
    assert (tmp_path / "evaluation.json").is_file()


def test_xgb_web_validator_rejects_native_model_tampering(tmp_path):
    features, labels = _multioutput_data()
    args = _args(
        artifact_dir=str(tmp_path),
        _artifact_source="sequence_features",
        _artifact_info={"label_names": ["alpha", "beta"]},
    )
    trainer = ClassicalMLTrainer(args)
    estimators = trainer._fit_boosting_estimators("xgb", features, labels)
    trainer._maybe_save_artifact(
        {
            "classifier": "xgb",
            "estimator_kind": "probability_ensemble",
            "estimators": estimators,
            "input_feature_count": features.shape[1],
            "_artifact_verification_x": features[:8],
        },
        SimpleNamespace(label_names=["alpha", "beta"]),
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    model_path = tmp_path / "xgb_models" / "output_000.ubj"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    with pytest.raises(InferenceUnsupportedError, match="failed manifest validation") as exc:
        ModelService._validate_xgb_predictor_contract(metadata, tmp_path)

    assert exc.value.code == "xgb_model_integrity_failed"


def test_xgb_legacy_artifact_is_explicitly_legacy_only(tmp_path):
    metadata = {"args": {"classifier": "xgb"}, "label_names": ["label"]}

    with pytest.warns(RuntimeWarning, match="legacy_only"):
        ModelService._validate_xgb_predictor_contract(metadata, tmp_path)
