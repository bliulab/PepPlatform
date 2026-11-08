"""The public selection API works without experiment code or benchmark defaults."""

import ast
from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from pepplatform.evaluation_audit import audit_artifact
from pepplatform.evaluators import Evaluator, OperatingPoint
from pepplatform.objectives import ObjectiveSpec
from pepplatform.web.ensemble_math import EnsembleError, normalize_definition, select


def sample():
    return np.array([[1, 0], [0, 1], [1, 1], [0, 0]]), np.array(
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.7], [0.1, 0.2]]
    )


def test_plain_f1_is_self_describing_and_does_not_report_dual():
    labels, scores = sample()
    point = Evaluator.fit_operating_point(labels, scores, objective="f1")
    assert point.objective == "macro_f1"
    assert point.objective_definition == ObjectiveSpec.parse("macro_f1").to_dict()
    assert point.search_config["profile"] == "f1_exact_v1"
    assert Evaluator.score_metrics(labels, scores, point)["dual_score"] is None
    # A caller may also evaluate fixed thresholds without requesting a dual score.
    assert Evaluator.score_metrics(labels, scores, OperatingPoint((0.5, 0.5)))["dual_score"] is None


def test_every_dual_entry_requires_parameters():
    labels, scores = sample()
    with pytest.raises(ValueError, match="parameters"):
        Evaluator.fit_operating_point(labels, scores, objective="dual_f1_sacc")
    with pytest.raises(TypeError, match="reference"):
        Evaluator.dual_primary_key(0.5, 0.6)
    with pytest.raises(ValueError, match="explicit objective definition"):
        Evaluator.score_metrics(
            labels, scores, OperatingPoint((0.5, 0.5), objective="dual_f1_sacc")
        )


def test_explicit_parameters_control_dual_reporting_and_survive_serialization():
    labels, scores = sample()
    spec = ObjectiveSpec.parse("dual_f1_sacc", {"reference_f1": 0.4, "reference_sacc": 0.6})
    point = Evaluator.fit_operating_point(labels, scores, objective=spec)
    assert OperatingPoint.from_dict(json.loads(json.dumps(point.to_dict()))) == point
    metrics = Evaluator.score_metrics(labels, scores, point)
    assert metrics["dual_score"] == spec.key(metrics)[0]
    doubled = ObjectiveSpec.parse("dual_f1_sacc", {"reference_f1": 0.8, "reference_sacc": 1.2})
    other = Evaluator.score_metrics(
        labels, scores, replace(point, objective_definition=doubled.to_dict())
    )
    assert other["dual_score"] == metrics["dual_score"] / 2
    for key in ("f1", "sacc", "auc", "aupr"):
        assert other[key] == metrics[key]


@pytest.mark.parametrize("missing", ["objective_definition", "search_config"])
def test_audit_refuses_to_guess_missing_selection_definitions(tmp_path, missing):
    labels, scores = sample()
    point = Evaluator.fit_operating_point(labels, scores).to_dict()
    metrics = Evaluator.score_metrics(labels, scores, point)
    point.pop(missing)
    metadata = {
        "evaluation_protocol_version": 3,
        "checkpoint_metric": "validation_macro_f1",
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": point,
        "thresholds": point["values"],
        "threshold_source": point["threshold_source"],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "evaluation.json").write_text(json.dumps({**metadata, "validation": metrics}))
    np.savez(tmp_path / "evaluation_scores.npz", validation_labels=labels, validation_scores=scores)
    assert any(
        f"{missing} is required" in error for error in audit_artifact(tmp_path / "evaluation.json")
    )


def test_web_selection_rejects_implicit_historical_definitions():
    request = {"schema_version": 1, "task_type": "pep_function", "member_ids": ["a", "b"]}
    with pytest.raises(EnsembleError, match="version"):
        normalize_definition(request)
    labels, scores = sample()
    with pytest.raises(EnsembleError, match="version"):
        select([scores, scores], labels, ["a", "b"], request)
    with pytest.raises(EnsembleError, match="version"):
        select([scores, scores], labels, ["a", "b"], {})


def test_audit_checks_the_explicit_dual_score(tmp_path):
    labels, scores = sample()
    spec = ObjectiveSpec.parse("dual_f1_sacc", {"reference_f1": 0.4, "reference_sacc": 0.6})
    point = Evaluator.fit_operating_point(labels, scores, objective=spec).to_dict()
    metrics = Evaluator.score_metrics(labels, scores, point)
    metadata = {
        "evaluation_protocol_version": 3,
        "checkpoint_metric": "validation_dual_f1_sacc",
        "feature_fit_scope": "train_only",
        "evaluation_mode": "independent_iid",
        "operating_point": point,
        "thresholds": point["values"],
        "threshold_source": point["threshold_source"],
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    metrics["dual_score"] += 0.1
    (tmp_path / "evaluation.json").write_text(json.dumps({**metadata, "validation": metrics}))
    np.savez(tmp_path / "evaluation_scores.npz", validation_labels=labels, validation_scores=scores)
    assert any(
        "validation.dual_score mismatch" in e for e in audit_artifact(tmp_path / "evaluation.json")
    )


def test_component_sources_contain_no_paper_selection_dependencies():
    root = Path(__file__).resolve().parents[1]
    forbidden_values = re.compile(r"(?<![\d.])0\.(?:422|536)(?!\d)")
    for directory in ("pepplatform", "scripts"):
        for path in (root / directory).rglob("*.py"):
            source = path.read_text()
            assert not forbidden_values.search(source), path
            assert "TPPRED_LE_" not in source, path
            assert "legacy_selection" not in source, path
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("experiment_paper"), path
                elif isinstance(node, ast.Import):
                    assert all(not n.name.startswith("experiment_paper") for n in node.names), path
    for path in (root / "frontend/src").rglob("*"):
        if path.suffix in {".vue", ".ts"} and not path.name.endswith(".test.ts"):
            assert not forbidden_values.search(path.read_text()), path


def test_public_components_run_when_experiment_imports_are_blocked(tmp_path):
    program = r"""
import importlib.abc
import sys
class BlockExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "experiment_paper" or fullname.startswith("experiment_paper."):
            raise AssertionError("component imported experiment code: " + fullname)
sys.meta_path.insert(0, BlockExperiments())
import numpy as np
from pepplatform.evaluators import Evaluator
from pepplatform.ensemble import select_weights, build_ensemble_artifact
from pepplatform.evaluation_audit import audit_artifact
from pepplatform.web.ensemble_math import normalize_definition, select
from pepplatform.web.capabilities import build_capabilities
from scripts.train_ensemble import build_parser
assert len(build_capabilities()["selection_objectives"]) == 6
assert build_parser().get_default("weight_objective") == "macro_f1"
y = np.array([[0,1],[1,0],[1,1],[0,0]])
s = .2 + y * .6
result = select_weights([s, s], y, label_names=["a", "b"])
build_ensemble_artifact(members=[{"name":"a"},{"name":"b"}], weights=result["weights"],
    y=y, fused_scores=s, label_names=["a","b"], artifact_dir=sys.argv[1],
    operating_point=result["operating_point"], weight_objective=result["weight_objective"])
assert audit_artifact(sys.argv[1] + "/evaluation.json") == []
definition = normalize_definition({"task_type":"pep_function", "member_ids":["a","b"]})
assert select([s,s], y, ["a","b"], definition)[1].objective == "macro_f1"
assert Evaluator.score_metrics(y,s,result["operating_point"])["dual_score"] is None
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "artifact")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
