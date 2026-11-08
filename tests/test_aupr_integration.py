"""The reported PR area must agree across evaluation, selection and display."""

import numpy as np
import pytest
from sklearn.metrics import auc, precision_recall_curve

from pepplatform.analysis import build_curve_bundle
from pepplatform.evaluators import Evaluator
from pepplatform.pr_metrics import has_current_aupr
from pepplatform.web import heatmap_jobs
from pepplatform.web.ensemble_math import fuse, normalize_definition, select


def test_macro_and_micro_areas_share_the_full_threshold_definition():
    labels = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]])
    scores = np.array([[0.9, 0.4, 0.8], [0.8, 0.5, 0.3], [0.7, 0.2, 0.6], [0.2, 0.1, 0.9]])
    expected = []
    for column in (0, 1):
        precision, recall, _ = precision_recall_curve(
            labels[:, column], scores[:, column], drop_intermediate=False
        )
        expected.append(auc(recall, precision))
    precision, recall, _ = precision_recall_curve(
        labels.ravel(), scores.ravel(), drop_intermediate=False
    )
    metrics = Evaluator.ranking_metrics(labels, scores)
    assert has_current_aupr(metrics)
    assert metrics["aupr_by_label"][:2] == pytest.approx(expected)
    assert metrics["aupr_by_label"][2] is None
    assert metrics["aupr"] == metrics["aupr_macro"] == pytest.approx(np.mean(expected))
    assert metrics["aupr_micro"] == pytest.approx(auc(recall, precision))
    curves = build_curve_bundle(labels, scores)
    assert has_current_aupr(curves["pr"])
    assert curves["pr"]["aupr"] == metrics["aupr_micro"]


def test_display_decimation_cannot_change_reported_area():
    rng = np.random.default_rng(193)
    labels = rng.integers(0, 2, 5000)
    scores = rng.random(len(labels))
    precision, recall, _ = precision_recall_curve(labels, scores, drop_intermediate=False)
    curves = build_curve_bundle(labels, scores)
    assert len(curves["pr"]["points"]) < len(recall)
    assert curves["pr"]["aupr"] == pytest.approx(auc(recall, precision), abs=1e-14)


@pytest.mark.parametrize("objective", ["auc", "aupr"])
def test_binary_weight_selection_uses_trapezoidal_area_for_auc_ties(objective):
    # Every mixture has ROC-AUC 0.5. Equal weights tie all scores and have
    # trapezoidal area 5/6; either member has area 19/24. AP prefers a member.
    labels = np.array([1, 0, 1])
    members = [np.array([0.9, 0.8, 0.7]), np.array([0.7, 0.8, 0.9])]
    definition = normalize_definition({
        "task_type": "pep_hlai",
        "member_ids": ["a", "b"],
        "weight_source": "validation",
        "grid_step": 0.5,
        "objective": objective,
    })
    assert has_current_aupr(definition)
    weights, point, _ = select(members, labels, ["positive"], definition)
    np.testing.assert_array_equal(weights, [0.5, 0.5])
    metrics = Evaluator.score_metrics(labels, fuse(members, weights), point)
    assert metrics["aupr"] == pytest.approx(5 / 6)
    assert definition["objective_definition"]["metric_namespace_version"] == (
        2 if objective == "aupr" else 1
    )


def test_heatmap_cache_identity_tracks_the_shared_pr_definition(monkeypatch):
    original_digest = heatmap_jobs.digest
    before = heatmap_jobs.implementation_digest()

    def changed_pr_digest(path):
        return "changed-pr-definition" if path.name == "pr_metrics.py" else original_digest(path)

    monkeypatch.setattr(heatmap_jobs, "digest", changed_pr_digest)
    assert heatmap_jobs.implementation_digest() != before
