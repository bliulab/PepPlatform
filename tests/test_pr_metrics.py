import numpy as np
import pytest
from sklearn.metrics import auc, precision_recall_curve

from pepplatform.pr_metrics import (
    binary_aupr, has_current_aupr, metric_definition, weighted_aupr, without_legacy_aupr,
)


def test_area_uses_trapezoids_instead_of_average_precision():
    assert binary_aupr([1, 0, 1], [0.9, 0.8, 0.7]) == pytest.approx(19 / 24)
    assert binary_aupr([1, 0, 1], [0.9, 0.8, 0.7]) != pytest.approx(5 / 6)


def test_ties_are_threshold_groups_and_input_order_does_not_matter():
    y = np.array([1, 0, 1, 0, 1])
    s = np.array([0.9, 0.9, 0.5, 0.2, 0.2])
    p, r, _ = precision_recall_curve(y, s, drop_intermediate=False)
    order = [4, 2, 0, 3, 1]
    assert binary_aupr(y, s) == auc(r, p)
    assert binary_aupr(y[order], s[order]) == binary_aupr(y, s)


@pytest.mark.parametrize("labels", [[], [0, 0], [1, 1]])
def test_unavailable_without_two_classes(labels):
    assert binary_aupr(labels, np.zeros(len(labels))) is None


def test_weighted_area_matches_independent_sklearn_curve():
    y, s = [1, 0, 1, 0], [0.9, 0.9, 0.4, 0.1]
    weights = np.array([[1, 1, 1, 1], [0.3, 2, 0.1, 3], [0, 0, 1, 0]])
    values = weighted_aupr(y, s, weights)
    for i, w in enumerate(weights[:2]):
        p, r, _ = precision_recall_curve(y, s, sample_weight=w, drop_intermediate=False)
        assert values[i] == pytest.approx(auc(r, p), abs=1e-14)
    assert np.isnan(values[2])


@pytest.mark.parametrize("labels,scores", [([1, 0], [0.1]), ([2, 0], [0.1, 0.2]), ([1, 0], [np.nan, 1])])
def test_invalid_inputs_are_rejected(labels, scores):
    with pytest.raises(ValueError):
        binary_aupr(labels, scores)


def test_historical_scores_are_hidden_without_mutating_originals():
    raw = {"summary": {"test_aupr": 0.9, "test_auc": 0.8}, "curves": {"pr": {"aupr": 0.9}}}
    view = without_legacy_aupr(raw)
    assert view["summary"]["test_aupr"] is None
    assert view["curves"]["pr"]["aupr"] is None
    assert view["summary"]["test_auc"] == raw["summary"]["test_auc"]
    assert raw["summary"]["test_aupr"] == 0.9


def test_current_summary_cannot_relabel_old_events():
    data = {**metric_definition(), "summary": {"test_aupr": 0.7}, "events": [
        {"val_aupr": 0.9}, {**metric_definition(), "val_aupr": 0.8},
    ]}
    view = without_legacy_aupr(data)
    assert has_current_aupr(view)
    assert view["summary"]["test_aupr"] == 0.7
    assert view["events"][0]["val_aupr"] is None
    assert view["events"][1]["val_aupr"] == 0.8


def test_explicit_legacy_child_does_not_inherit_current_parent_definition():
    value = {**metric_definition(), "metrics": {"metric_namespace_version": 1, "aupr": 0.9}}
    assert without_legacy_aupr(value)["metrics"]["aupr"] is None
