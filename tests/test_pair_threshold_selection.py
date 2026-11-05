from argparse import Namespace
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from pepplatform.evaluators import Evaluator
from scripts import train_pair_pipeline as pair


TASKS = ("pep_protein_interaction", "pep_hlai", "pep_mhci", "pep_hlaii")
LABELS = np.array([0, 1, 0, 1])
SCORES = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)


def selection_args(task="pep_protein_interaction", **overrides):
    return Namespace(
        **{
            "task_type": task,
            "checkpoint_objective": "auc",
            "threshold_objective": "f1",
            "threshold_search_profile": "f1_exact_v1",
            "checkpoint_threshold_objective": "macro_f1",
            "checkpoint_threshold_search_profile": "coordinate_all_v1",
            **overrides,
        }
    )


@pytest.mark.parametrize("task", TASKS)
def test_checkpoint_threshold_configuration_is_independent_and_never_freezes_a_final_point(task):
    args = selection_args(task)
    checkpoint = pair._fit_pair_operating_point(args, LABELS, SCORES, for_checkpoint=True)

    assert checkpoint.objective == "macro_f1"
    assert checkpoint.search_config["profile"] == "coordinate_all_v1"
    assert not hasattr(args, "_operating_point")
    assert args._selection_validation_metrics["operating_point"] == checkpoint.to_dict()

    final = pair._fit_pair_operating_point(args, LABELS, SCORES)
    assert final.objective == "f1"
    assert final.search_config["profile"] == "f1_exact_v1"
    assert args._operating_point == final.to_dict()
    pair._pair_checkpoint_metrics(args, LABELS, SCORES)
    assert args._operating_point == final.to_dict()


@pytest.mark.parametrize("task", TASKS)
def test_omitted_checkpoint_threshold_configuration_inherits_the_final_settings(task):
    args = selection_args(
        task, checkpoint_threshold_objective=None, checkpoint_threshold_search_profile=None
    )
    checkpoint = pair._fit_pair_operating_point(args, LABELS, SCORES, for_checkpoint=True)
    final = pair._fit_pair_operating_point(args, LABELS, SCORES)

    assert checkpoint == final
    assert final == Evaluator.fit_operating_point(LABELS, SCORES, objective="f1")
    assert pair.best_threshold_by_f1(LABELS, SCORES) == final.threshold


def test_nn_checkpoint_validation_collects_scores_once_and_applies_checkpoint_settings(monkeypatch):
    args = selection_args()
    calls = []

    def collect(*values):
        calls.append(values)
        return LABELS, SCORES

    monkeypatch.setattr(pair, "collect_binary_scores", collect)
    metrics = pair._pair_nn_validation_metrics(args, "model", "loader", "cpu")

    assert calls == [("model", "loader", "cpu")]
    assert metrics["operating_point"]["objective"] == "macro_f1"
    assert metrics["operating_point"]["search_config"]["profile"] == "coordinate_all_v1"
    assert not hasattr(args, "_operating_point")


@pytest.mark.parametrize("provide_point", [False, True])
def test_score_analysis_preserves_legacy_calls_and_reuses_an_explicit_final_point(
    tmp_path, monkeypatch, provide_point
):
    args = selection_args(artifact_dir=str(tmp_path))
    point = Evaluator.fit_operating_point(LABELS, SCORES)
    fit_calls = []
    original = pair._fit_pair_operating_point

    def fit(*values, **options):
        fit_calls.append(options)
        return original(*values, **options)

    monkeypatch.setattr(pair, "_fit_pair_operating_point", fit)
    options = {"operating_point": point} if provide_point else {}
    pair.write_score_analysis(args, LABELS, SCORES, LABELS, SCORES, **options)
    evaluation = json.loads((tmp_path / "evaluation.json").read_text())

    assert len(fit_calls) == (0 if provide_point else 1)
    assert evaluation["operating_point"] == point.to_dict()
    assert evaluation["test"]["operating_point"] == args._operating_point == point.to_dict()


@pytest.mark.parametrize("aligned", [False, True], ids=["mlp", "aligned_gated"])
def test_test_log_evaluation_and_saved_model_use_the_final_point(
    tmp_path, monkeypatch, capsys, aligned
):
    args = pair.build_argument_parser().parse_args(
        [
            "--task_type",
            "pep_protein_interaction",
            "--tokenizer",
            "kmer",
            "--encoder",
            "onehot",
            "--model",
            "mlp",
            "--epochs",
            "2",
            "--batch_size",
            "4",
            "--head_hidden_dims",
            "[8]",
            "--checkpoint_threshold_objective",
            "macro_f1",
            "--checkpoint_threshold_search_profile",
            "coordinate_all_v1",
            "--threshold_objective",
            "f1",
            "--artifact_dir",
            str(tmp_path),
            "--no-report_aupr",
        ]
    )
    args._artifact_info = {"label_names": ["interaction"]}
    monkeypatch.setattr(pair, "_pair_vector_artifacts_supported", lambda _: True)
    # Both policies are valid binary F1 variants. Force different thresholds to
    # expose accidental reuse of an epoch's point in final reporting or export.
    original_fit = Evaluator.fit_operating_point
    objectives = []

    def fit(*values, **options):
        point = original_fit(*values, **options)
        objectives.append(point.objective)
        return replace(point, values=(0.3 if point.objective == "macro_f1" else 0.7,))

    monkeypatch.setattr(Evaluator, "fit_operating_point", fit)
    monkeypatch.setattr(pair, "collect_binary_scores", lambda *unused: (LABELS, SCORES))
    monkeypatch.setattr(pair, "collect_binary_scores_pair", lambda *unused: (LABELS, SCORES))
    torch.manual_seed(42)
    features = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
    partner = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 20
    targets = torch.from_numpy(LABELS.astype(np.float32))[:, None]

    if aligned:
        pair.train_mlp_aligned_gated(
            args,
            features,
            partner,
            targets,
            features,
            partner,
            targets,
            features,
            partner,
            targets,
        )
    else:
        pair.train_mlp(args, features, targets, features, targets, features, targets)

    output = capsys.readouterr().out
    test_line = next(line for line in output.splitlines() if "] Test |" in line)
    evaluation = json.loads((tmp_path / "evaluation.json").read_text())
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    history = json.loads((tmp_path / "selection_history.json").read_text())

    assert objectives == ["macro_f1", "macro_f1", "f1"]
    assert all(row["metrics"]["operating_point"]["threshold"] == 0.3 for row in history["epochs"])
    assert evaluation["operating_point"]["threshold"] == metadata["threshold"] == 0.7
    assert metadata["operating_point"] == evaluation["operating_point"] == args._operating_point
    assert evaluation["test"]["f1"] == pytest.approx(2 / 3)
    assert " | F1: 0.6667 |" in test_line
    assert (tmp_path / "model.pt").is_file()
