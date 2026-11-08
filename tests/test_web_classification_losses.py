from copy import deepcopy

import pytest

from pepplatform.web.classification_losses import CLASSIFICATION_LOSS_FIELDS
from pepplatform.web.config import normalize_experiment_config_with_sources
from pepplatform.web.paths import RuntimePaths
from pepplatform.web.preflight import preflight_experiment
from pepplatform.web.runner import ExperimentRunner
from scripts import train_pair_pipeline, train_sequence_pipeline


TASKS = ["pep_function", "pep_protein_interaction", "pep_hlai", "pep_hlaii"]


def dataset(task):
    return {"id": "data", "name": "loss data", "task_type": task,
            "format": "sequence_dir" if task == "pep_function" else "pair_split_dir",
            "root_path": "/unused/loss-test", "metadata": {"splits": {"train": 10, "val": 2}}}


def config():
    return {"tokenizer": "kmer", "encoder": "onehot", "model": "cnn", "pair_strategy": "single_stream"}


@pytest.mark.parametrize("task", TASKS)
def test_legacy_configs_do_not_gain_classification_loss_fields(task):
    raw = config()
    before = deepcopy(raw)
    normalized, sources = normalize_experiment_config_with_sources(task, raw)
    report = preflight_experiment(dataset(task), task, raw)
    assert report["valid"], report["errors"]
    assert not CLASSIFICATION_LOSS_FIELDS.intersection(normalized)
    assert not CLASSIFICATION_LOSS_FIELDS.intersection(report["normalized_config"])
    assert not CLASSIFICATION_LOSS_FIELDS.intersection(sources)
    assert raw == before


@pytest.mark.parametrize("task,overrides", [
    ("pep_function", {"multilabel_loss": "bce"}),
    ("pep_function", {"multilabel_loss": "balanced_bce", "multilabel_pos_weight_cap": 3}),
    ("pep_function", {"multilabel_loss": "asymmetric_focal", "asl_gamma_pos": 0.5, "asl_gamma_neg": 3, "asl_clip": 0.1}),
    *[(task, {"loss_type": loss, "pos_weight_max": 7}) for task in TASKS[1:] for loss in ("bce", "wbce")],
])
def test_preflight_runner_and_cli_round_trip(tmp_path, task, overrides):
    raw = {**config(), **overrides, "enable_semantic_similarity": True, "semantic_loss_type": "jaccard_supcon"}
    report = preflight_experiment(dataset(task), task, raw)
    assert report["valid"], report["errors"]
    for key, value in overrides.items():
        assert report["normalized_config"][key] == value
        assert report["parameter_sources"][key] == "explicit"
    runner = ExperimentRunner(None, RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure())
    command = runner.build_command(
        {"task_type": task, "config": report["normalized_config"], "artifact_dir": str(tmp_path / "artifact")},
        dataset(task),
    )
    module = train_sequence_pipeline if task == "pep_function" else train_pair_pipeline
    parsed = module.build_argument_parser().parse_args(command[2:])
    for key, value in overrides.items():
        assert getattr(parsed, key) == value
    assert parsed.semantic_loss_type == "jaccard_supcon"


@pytest.mark.parametrize("key,value", [
    ("multilabel_loss", "typo"), ("multilabel_loss", "wbce"),
    ("multilabel_pos_weight_cap", 0.99), ("multilabel_pos_weight_cap", True),
    ("asl_gamma_pos", -1), ("asl_gamma_neg", "nan"), ("asl_clip", 1),
    ("asl_clip", float("inf")), ("asl_clip", None), ("asl_gamma_pos", ""),
])
def test_preflight_rejects_invalid_sequence_loss_parameters(key, value):
    report = preflight_experiment(dataset("pep_function"), "pep_function", {**config(), key: value})
    assert not report["valid"]
    assert {"classification_loss_invalid", "classification_loss_parameter_invalid"}.intersection(report["error_codes"])


@pytest.mark.parametrize("task", TASKS[1:])
@pytest.mark.parametrize("override", [{"loss_type": "asymmetric_focal"}, {"pos_weight_max": 0}, {"pos_weight_max": "inf"}, {"pos_weight_max": True}])
def test_pair_loss_validation(task, override):
    report = preflight_experiment(dataset(task), task, {**config(), **override})
    assert not report["valid"]


@pytest.mark.parametrize("task,override", [("pep_function", {"loss_type": "bce"}), ("pep_hlai", {"multilabel_loss": "bce"})])
def test_wrong_task_loss_fields_are_rejected(task, override):
    report = preflight_experiment(dataset(task), task, {**config(), **override})
    assert "classification_loss_task_unsupported" in report["error_codes"]


@pytest.mark.parametrize("raw", [
    {"tokenizer": "kmer", "encoder": "bow", "model": "svm", "multilabel_loss": "bce"},
    {**config(), "ensemble_members": ["first", "second"], "multilabel_loss": "bce"},
])
def test_non_neural_or_ensemble_selection_is_rejected(raw):
    report = preflight_experiment(dataset("pep_function"), "pep_function", raw)
    assert "classification_loss_route_unsupported" in report["error_codes"]


def test_runner_filters_loss_fields_for_the_other_cli(tmp_path):
    runner = ExperimentRunner(None, RuntimePaths.from_project_root(tmp_path / "p", tmp_path / "r").ensure())
    for task in ("pep_function", "pep_hlai"):
        command = runner.build_command({"task_type": task, "config": {**config(), "multilabel_loss": "bce", "loss_type": "bce", "asl_clip": 0.1, "pos_weight_max": 20}, "artifact_dir": str(tmp_path / "a")}, dataset(task))
        if task == "pep_function":
            assert "--loss_type" not in command and "--pos_weight_max" not in command
        else:
            assert "--multilabel_loss" not in command and "--asl_clip" not in command
