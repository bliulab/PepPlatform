from argparse import Namespace
import hashlib
import json
from types import SimpleNamespace

import joblib
import lightgbm
import numpy as np
import pytest
import sklearn
import torch
from torch.utils.data import TensorDataset

from pepplatform.ml_trainers import (
    LGB_MANIFEST_SCHEMA_VERSION,
    LGB_PREDICTOR_CONTRACT_VERSION,
    ClassicalMLTrainer,
    LGBTrainingDataError,
    lgb_predictor_contract,
)
from pepplatform.web.config import normalize_experiment_config_with_sources
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from pepplatform.training_failures import failure_from_exception


def _args(**overrides):
    values = {
        "task_type": "pep_function",
        "model": "lgb",
        "classifier": "lgb",
        "tokenizer": "kmer",
        "encoder": "bow",
        "seed": 17,
        "lgb_estimators": 8,
        "lgb_max_depth": 2,
        "lgb_lr": 0.15,
        "lgb_subsample": 1.0,
        "lgb_colsample": 1.0,
        "lgb_reg_alpha": 0.0,
        "lgb_reg_lambda": 0.0,
        "artifact_dir": "",
    }
    values.update(overrides)
    return Namespace(**values)


def _multioutput_data():
    rng = np.random.default_rng(37)
    features = rng.normal(size=(72, 7)).astype(np.float32)
    first = (features[:, 0] + 0.3 * features[:, 1] > 0).astype(int)
    second = (features[:, 2] - 0.2 * features[:, 3] > 0).astype(int)
    return features, np.column_stack([first, second])


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lgb_identity_contract_probability_and_batch_invariance():
    features, labels = _multioutput_data()
    trainer = ClassicalMLTrainer(_args(lgb_max_depth=0, lgb_subsample=0.8))

    estimators = trainer._fit_boosting_estimators("lgb", features, labels)
    together = trainer._probability_ensemble_scores(estimators, features[:9])
    separately = np.concatenate(
        [trainer._probability_ensemble_scores(estimators, row[None, :]) for row in features[:9]],
        axis=0,
    )
    contract = lgb_predictor_contract(estimators, input_feature_count=features.shape[1])

    assert all(isinstance(item["value"], lightgbm.LGBMClassifier) for item in estimators)
    assert together.shape == (9, 2)
    assert np.isfinite(together).all()
    assert ((0.0 <= together) & (together <= 1.0)).all()
    np.testing.assert_array_equal(together, separately)
    assert contract["contract_version"] == LGB_PREDICTOR_CONTRACT_VERSION
    assert contract["algorithm"] == "lightgbm"
    assert contract["implementation"] == "lightgbm.LGBMClassifier"
    assert contract["implementation_version"] == lightgbm.__version__
    assert contract["sklearn_version"] == sklearn.__version__
    assert contract["status"] == "verified_platform_variant"
    assert contract["architecture"] == "independent_binary_lightgbm_gbdt_per_output"
    assert contract["booster"] == "gbdt"
    assert contract["objective"] == "binary"
    assert contract["score_kind"] == "probability"
    assert contract["calibrated"] is False
    assert contract["estimator_params"]["max_depth"] == -1
    assert contract["estimator_params"]["subsample_freq"] == 1
    assert contract["execution_policy"] == {
        "n_jobs": 4,
        "deterministic": True,
        "histogram_mode": "row_wise",
        "device_type": "cpu",
        "seed": 17,
        "child_seed_policy": "experiment_seed",
    }
    assert all(
        item["value"].get_params()["n_jobs"] == 4
        and item["value"].get_params()["deterministic"] is True
        and item["value"].get_params()["force_row_wise"] is True
        and item["value"].get_params()["device_type"] == "cpu"
        and item["value"].get_params()["bagging_seed"] == 17
        and item["value"].get_params()["feature_fraction_seed"] == 17
        and item["value"].get_params()["data_random_seed"] == 17
        for item in estimators
    )
    assert all(
        item["max_depth"] == -1 and item["subsample_freq"] == 1
        for item in contract["effective_params_by_output"]
    )


def test_lgb_fixed_four_thread_policy_is_bitwise_reproducible():
    features, labels = _multioutput_data()
    first_trainer = ClassicalMLTrainer(
        _args(lgb_estimators=24, lgb_subsample=0.8, lgb_colsample=0.8)
    )
    second_trainer = ClassicalMLTrainer(
        _args(lgb_estimators=24, lgb_subsample=0.8, lgb_colsample=0.8)
    )

    first = first_trainer._fit_boosting_estimators("lgb", features, labels)
    second = second_trainer._fit_boosting_estimators("lgb", features, labels)

    first_models = [item["value"].booster_.model_to_string() for item in first]
    second_models = [item["value"].booster_.model_to_string() for item in second]
    assert first_models == second_models
    np.testing.assert_array_equal(
        first_trainer._probability_ensemble_scores(first, features),
        second_trainer._probability_ensemble_scores(second, features),
    )


@pytest.mark.parametrize(
    ("overrides", "labels_transform", "message"),
    [
        ({"lgb_lr": 0.0}, None, "lgb_lr"),
        ({"lgb_lr": float("nan")}, None, "finite"),
        ({"lgb_subsample": 0.0}, None, "subsample"),
        ({"lgb_colsample": 0.0}, None, "colsample"),
        ({"lgb_max_depth": -1}, None, "max_depth"),
        ({"lgb_reg_alpha": -0.1}, None, "regularization"),
        ({"lgb_reg_lambda": -0.1}, None, "regularization"),
        ({}, lambda labels: np.zeros_like(labels), "both binary classes"),
    ],
)
def test_lgb_rejects_invalid_parameters_and_degenerate_outputs(
    overrides, labels_transform, message
):
    features, labels = _multioutput_data()
    if labels_transform is not None:
        labels = labels_transform(labels)
    trainer = ClassicalMLTrainer(_args(**overrides))

    with pytest.raises(LGBTrainingDataError, match=message):
        trainer._fit_boosting_estimators("lgb", features, labels)


def test_lgb_rejects_nonfinite_training_features():
    features, labels = _multioutput_data()
    features[0, 0] = np.nan

    with pytest.raises(LGBTrainingDataError, match="finite"):
        ClassicalMLTrainer(_args())._fit_boosting_estimators("lgb", features, labels)


def test_lgb_training_error_maps_to_structured_data_failure():
    error = LGBTrainingDataError("invalid labels", details={"output_index": 0})

    failure = failure_from_exception(error)

    assert failure["code"] == "lgb_training_data_invalid"
    assert failure["category"] == "data"
    assert failure["stage"] == "training"
    assert failure["context"]["classifier"] == "lgb"


def test_lgb_hybrid_artifact_roundtrip(tmp_path):
    features, labels = _multioutput_data()
    args = _args(
        artifact_dir=str(tmp_path),
        _artifact_source="sequence_features",
        _artifact_info={"label_names": ["alpha", "beta"]},
    )
    trainer = ClassicalMLTrainer(args)
    estimators = trainer._fit_boosting_estimators("lgb", features, labels)
    expected = trainer._probability_ensemble_scores(estimators, features[:11])
    trainer._maybe_save_artifact(
        {
            "classifier": "lgb",
            "estimator_kind": "probability_ensemble",
            "estimators": estimators,
            "input_feature_count": features.shape[1],
            "_artifact_verification_x": features[:11],
        },
        SimpleNamespace(label_names=["alpha", "beta"]),
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    manifest_path = tmp_path / "lgb_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = joblib.load(tmp_path / "model.joblib")

    assert payload["estimator_kind"] == "lightgbm_native_ensemble"
    assert payload["manifest_path"] == "lgb_manifest.json"
    assert "estimators" not in payload
    assert manifest["schema_version"] == LGB_MANIFEST_SCHEMA_VERSION
    assert manifest["native_loader"] == "lightgbm.Booster"
    assert manifest["execution_policy"] == metadata["predictor_contract"]["execution_policy"]
    assert manifest["reload_verification"]["status"] == "passed"
    assert [item["label_name"] for item in manifest["outputs"]] == [
        "alpha",
        "beta",
    ]
    for output in manifest["outputs"]:
        model_path = tmp_path / output["model_path"]
        assert model_path.suffix == ".txt"
        assert output["size_bytes"] == model_path.stat().st_size
        assert output["sha256"] == _sha256(model_path)
    assert metadata["predictor_contract"]["serialization"]["manifest_sha256"] == _sha256(
        manifest_path
    )

    ModelService._validate_lgb_predictor_contract(metadata, tmp_path)
    restored = ModelService._load_classical_payload(tmp_path, metadata)
    actual = ModelService._predict_classical_scores(restored, features[:11])
    np.testing.assert_array_equal(expected, actual)


def test_lgb_training_entry_emits_current_native_artifact(tmp_path, capsys):
    features, labels = _multioutput_data()
    train = TensorDataset(torch.from_numpy(features[:56]), torch.from_numpy(labels[:56]).float())
    validation = TensorDataset(
        torch.from_numpy(features[56:]), torch.from_numpy(labels[56:]).float()
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

    result = ClassicalMLTrainer(args).run_lgb(train, validation)
    progress_lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("[run-progress]")
    ]

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    payload = joblib.load(tmp_path / "model.joblib")
    assert np.isfinite(result)
    assert metadata["predictor_contract"]["algorithm"] == "lightgbm"
    assert metadata["operating_point"]["score_kind"] == "probability"
    assert payload["estimator_kind"] == "lightgbm_native_ensemble"
    assert (tmp_path / "evaluation.json").is_file()
    assert any(
        "stage=classifier_fit status=running" in line and "threads=4" in line
        for line in progress_lines
    )
    assert any("stage=classifier_fit status=complete" in line for line in progress_lines)
    assert progress_lines[-1].startswith("[run-progress] stage=finalization status=complete")


def test_lgb_web_validator_rejects_native_model_tampering(tmp_path):
    features, labels = _multioutput_data()
    args = _args(
        artifact_dir=str(tmp_path),
        _artifact_source="sequence_features",
        _artifact_info={"label_names": ["alpha", "beta"]},
    )
    trainer = ClassicalMLTrainer(args)
    estimators = trainer._fit_boosting_estimators("lgb", features, labels)
    trainer._maybe_save_artifact(
        {
            "classifier": "lgb",
            "estimator_kind": "probability_ensemble",
            "estimators": estimators,
            "input_feature_count": features.shape[1],
            "_artifact_verification_x": features[:8],
        },
        SimpleNamespace(label_names=["alpha", "beta"]),
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    model_path = tmp_path / "lgb_models" / "output_000.txt"
    model_path.write_text(
        model_path.read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(InferenceUnsupportedError, match="failed manifest validation") as exc:
        ModelService._validate_lgb_predictor_contract(metadata, tmp_path)

    assert exc.value.code == "lgb_model_integrity_failed"


def test_lgb_legacy_artifact_is_explicitly_legacy_only(tmp_path):
    metadata = {"args": {"classifier": "lgb"}, "label_names": ["label"]}

    with pytest.warns(RuntimeWarning, match="legacy_only"):
        ModelService._validate_lgb_predictor_contract(metadata, tmp_path)


def test_lgb_web_default_matches_cli_and_documented_regularization_default():
    config, sources = normalize_experiment_config_with_sources(
        "pep_function", {"model": "lgb", "classifier": "lgb"}
    )

    assert config["lgb_reg_lambda"] == 0.0
    assert sources["lgb_reg_lambda"] == "default"
    assert config["seed"] == 42
    assert sources["seed"] == "default"
    assert config["lgb_n_jobs"] == 4
    assert config["lgb_deterministic"] is True
    assert config["lgb_histogram_mode"] == "row_wise"
    assert config["lgb_device_type"] == "cpu"
    assert sources["lgb_n_jobs"] == "derived"
