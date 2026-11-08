from argparse import Namespace
import copy

import numpy as np
import pytest

from pepplatform.evaluators import Evaluator, OperatingPoint
from pepplatform.objectives import ObjectiveError, ObjectiveSpec, resolve_selection


REF = {"reference_f1": 0.4, "reference_sacc": 0.6}


def test_checkpoint_and_threshold_and_tuning_are_independent():
    args = Namespace(
        task_type="pep_function",
        checkpoint_objective="dual_f1_sacc",
        checkpoint_objective_params=REF,
        tuning_objective="auc",
    )
    config = resolve_selection(args)
    assert config.threshold.id == "macro_f1"
    assert config.tuning.id == "auc"
    args.threshold_objective = "sacc"
    changed = resolve_selection(args)
    assert changed.checkpoint == config.checkpoint
    assert changed.tuning == config.tuning


@pytest.mark.parametrize("objective", ["auc", "aupr"])
def test_ranking_metric_rejected_as_threshold_before_training(objective):
    with pytest.raises(ObjectiveError, match="threshold objective"):
        resolve_selection({"task_type": "pep_hlai", "threshold_objective": objective})


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"reference_f1": 0},
        {**REF, "reference_sacc": float("nan")},
        {**REF, "reference_f1": -1},
        {**REF, "unknown": 1},
    ],
)
def test_dual_parameters_fail_closed(params):
    with pytest.raises((ObjectiveError, TypeError)):
        ObjectiveSpec.parse("dual_f1_sacc", params)


def test_dual_reference_changes_selection_not_metric_definition():
    first = {"f1": 0.6, "sacc": 0.4}
    second = {"f1": 0.4, "sacc": 0.6}
    a = ObjectiveSpec.parse("dual_f1_sacc", {"reference_f1": 1, "reference_sacc": 2})
    b = ObjectiveSpec.parse("dual_f1_sacc", {"reference_f1": 2, "reference_sacc": 1})
    assert a.key(first) < a.key(second)
    assert b.key(first) > b.key(second)
    y = np.array([[1, 0], [0, 1], [1, 1]])
    scores = np.array([[0.8, 0.3], [0.2, 0.7], [0.8, 0.8]])
    pa = OperatingPoint((0.5, 0.5), objective=a.id, objective_definition=a.to_dict())
    pb = OperatingPoint((0.5, 0.5), objective=b.id, objective_definition=b.to_dict())
    ma, mb = Evaluator.score_metrics(y, scores, pa), Evaluator.score_metrics(y, scores, pb)
    for key in ("f1", "sacc", "auc", "aupr", "mcc"):
        assert ma[key] == mb[key]


def test_definition_tampering_detected():
    definition = ObjectiveSpec.parse("dual_f1_sacc", REF).to_dict()
    altered = copy.deepcopy(definition)
    altered["params"]["reference_f1"] = 1
    with pytest.raises(ObjectiveError, match="digest"):
        ObjectiveSpec.parse(altered)


def test_pair_defaults_preserve_auc_and_f1():
    for task in ("pep_hlai", "pep_hlaii", "pep_protein_interaction"):
        c = resolve_selection({"task_type": task})
        assert (c.checkpoint.id, c.threshold.id, c.tuning.id) == ("auc", "f1", "auc")


def test_historical_pooled_checkpoint_and_final_operating_points_are_explicit():
    config = resolve_selection(
        {
            "task_type": "pep_function",
            "checkpoint_objective": "macro_f1",
            "threshold_objective": "dual_f1_sacc",
            "threshold_objective_params": REF,
            "checkpoint_threshold_objective": "macro_f1",
        }
    )
    assert config.checkpoint_threshold.id == "macro_f1"
    assert config.checkpoint_search_profile == "f1_exact_v1"
    assert config.threshold.id == "dual_f1_sacc"
    assert config.search_profile == "coordinate_quantile32_v1"


def test_selection_requires_all_requested_auc_labels_available():
    with pytest.raises(ObjectiveError, match="lacks both classes"):
        ObjectiveSpec.parse("auc").key(
            {"auc": 0.9, "averaging": "macro", "auc_by_label": [0.9, None]}
        )


def test_dual_string_and_definition_use_the_same_recorded_search():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=(70, 3))
    scores = rng.random((70, 3))
    from_string = Evaluator.fit_operating_point(
        y, scores, objective="dual_f1_sacc", objective_params=REF
    )
    spec = ObjectiveSpec.parse("dual_f1_sacc", REF)
    explicit = Evaluator.fit_operating_point(y, scores, objective=spec)
    assert explicit == from_string
    assert explicit.search_config["max_candidates"] == 32
    assert OperatingPoint.from_dict(explicit.to_dict()) == explicit
