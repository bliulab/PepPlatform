import json

import numpy as np
import pytest

from pepplatform import ensemble
from pepplatform.web.inference import InferenceUnsupportedError, ModelService
from pepplatform.web.paths import RuntimePaths
from pepplatform.web.preflight import preflight_experiment
from pepplatform.web.runner import ExperimentRunner
from scripts import train_ensemble


def _sample_data():
    y = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=np.int32,
    )
    scores_a = np.array(
        [
            [0.9, 0.1, 0.8],
            [0.2, 0.8, 0.3],
            [0.7, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.8, 0.3, 0.2],
        ],
        dtype=np.float64,
    )
    scores_b = np.array(
        [
            [0.7, 0.3, 0.9],
            [0.4, 0.9, 0.2],
            [0.6, 0.5, 0.4],
            [0.2, 0.1, 0.9],
            [0.9, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    return y, scores_a, scores_b


def test_probability_mean_ensemble_fuse():
    model = ensemble.ProbabilityMeanEnsemble(
        members=({"artifact_dir": "a"}, {"artifact_dir": "b"}),
        weights=(0.25, 0.75),
        label_names=("A", "B"),
    )
    fused = model.fuse(
        [
            np.array([[0.2, 0.8]]),
            np.array([[0.4, 0.4]]),
        ]
    )
    np.testing.assert_allclose(fused, [[0.35, 0.5]])


def test_select_weights_equal():
    y, scores_a, scores_b = _sample_data()
    result = ensemble.select_weights(
        [scores_a, scores_b],
        y,
        label_names=["A", "B", "C"],
        weight_source="equal",
    )
    assert result["weights"] == (0.5, 0.5)
    assert result["operating_point"].objective == "macro_f1"
    assert result["validation_metrics"]["sacc"] > 0


def test_select_weights_validation_two_members():
    y, scores_a, scores_b = _sample_data()
    result = ensemble.select_weights(
        [scores_a, scores_b],
        y,
        label_names=["A", "B", "C"],
        weight_source="validation",
        grid_step=0.5,
    )
    assert len(result["weights"]) == 2
    assert abs(sum(result["weights"]) - 1.0) < 1e-9
    assert len(result["grid"]) == 3
    assert result["grid"][0]["alpha_member_0"] == 0.0


def test_select_weights_validation_requires_two_members():
    y, scores_a, _ = _sample_data()
    with pytest.raises(ensemble.EnsembleConfigError):
        ensemble.select_weights(
            [scores_a, scores_a, scores_a],
            y,
            label_names=["A", "B", "C"],
            weight_source="validation",
        )


def test_build_ensemble_artifact_writes_protocol_bundle(tmp_path):
    y, scores_a, scores_b = _sample_data()
    fused = 0.5 * scores_a + 0.5 * scores_b
    operating_point = ensemble.Evaluator.fit_operating_point(
        y,
        fused,
        label_names=["A", "B", "C"],
        objective="macro_f1",
    )
    artifact_dir = tmp_path / "artifact"
    result = ensemble.build_ensemble_artifact(
        members=[
            {"name": "a", "artifact_dir": str(tmp_path / "a")},
            {"name": "b", "artifact_dir": str(tmp_path / "b")},
        ],
        weights=(0.5, 0.5),
        y=y,
        fused_scores=fused,
        label_names=["A", "B", "C"],
        artifact_dir=artifact_dir,
        operating_point=operating_point,
    )
    assert (artifact_dir / "metadata.json").is_file()
    assert (artifact_dir / "ensemble.json").is_file()
    assert (artifact_dir / "model.joblib").is_file()
    assert (artifact_dir / "evaluation.json").is_file()
    assert (artifact_dir / "evaluation_scores.npz").is_file()
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["bundle_type"] == "ensemble"
    assert metadata["supports_inference"] is True
    assert metadata["threshold_source"] == "validation_macro_f1"
    assert result["validation_metrics"]["sacc"] > 0


def _write_member_artifact(root, name, y, scores):
    artifact_dir = root / name
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {
                "label": name,
                "label_names": ["A", "B", "C"],
                "args": {"task_type": "pep_function"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        artifact_dir / "evaluation_scores.npz",
        validation_labels=y,
        validation_scores=scores,
    )
    return artifact_dir


def test_train_ensemble_cli_validation_only(tmp_path):
    y, scores_a, scores_b = _sample_data()
    member_a = _write_member_artifact(tmp_path, "member_a", y, scores_a)
    member_b = _write_member_artifact(tmp_path, "member_b", y, scores_b)
    artifact_dir = tmp_path / "ensemble"
    exit_code = train_ensemble.main(
        [
            "--members",
            f"{member_a},{member_b}",
            "--artifact-dir",
            str(artifact_dir),
            "--ensemble-weight-source",
            "validation",
            "--ensemble-grid-step",
            "0.5",
            "--no-evaluate-test",
        ]
    )
    assert exit_code == 0
    assert (artifact_dir / "ensemble.json").is_file()
    payload = json.loads((artifact_dir / "ensemble.json").read_text(encoding="utf-8"))
    assert len(payload["members"]) == 2
    assert abs(sum(payload["weights"]) - 1.0) < 1e-9


def test_web_runner_builds_ensemble_command(tmp_path):
    y, scores_a, scores_b = _sample_data()
    member_a = _write_member_artifact(tmp_path, "member_a", y, scores_a)
    member_b = _write_member_artifact(tmp_path, "member_b", y, scores_b)
    paths = RuntimePaths.from_project_root(
        tmp_path / "project",
        tmp_path / "runtime",
    ).ensure()
    runner = ExperimentRunner(None, paths)
    command = runner.build_command(
        {
            "task_type": "pep_function",
            "config": {
                "ensemble_members": [str(member_a), str(member_b)],
                "ensemble_method": "probability_mean",
                "ensemble_weight_source": "validation",
                "ensemble_grid_step": 0.05,
            },
            "artifact_dir": str(tmp_path / "ensemble-out"),
        },
        {
            "format": "sequence_dir",
            "root_path": str(tmp_path / "data"),
        },
    )
    assert command[1].endswith("scripts/train_ensemble.py")
    assert command[command.index("--members") + 1] == f"{member_a},{member_b}"
    assert command[command.index("--ensemble-weight-source") + 1] == "validation"


def test_preflight_accepts_ensemble_config(tmp_path):
    dataset = {
        "id": "d1",
        "name": "d",
        "task_type": "pep_function",
        "format": "sequence_dir",
        "root_path": str(tmp_path),
        "metadata": {},
    }
    report = preflight_experiment(
        dataset,
        "pep_function",
        {
            "ensemble_members": [str(tmp_path / "a"), str(tmp_path / "b")],
            "ensemble_method": "probability_mean",
            "ensemble_weight_source": "validation",
        },
    )
    assert report["valid"] is True
    assert report["execution"]["script"] == "scripts/train_ensemble.py"
    assert report["execution"]["supports_reusable_inference"] is True


def test_preflight_rejects_bad_ensemble_config(tmp_path):
    dataset = {
        "id": "d1",
        "name": "d",
        "task_type": "pep_function",
        "format": "sequence_dir",
        "root_path": str(tmp_path),
        "metadata": {},
    }
    report = preflight_experiment(
        dataset,
        "pep_function",
        {
            "ensemble_members": [str(tmp_path / "a")],
            "ensemble_method": "probability_mean",
            "ensemble_weight_source": "validation",
            "ensemble_grid_step": 2.0,
        },
    )
    assert report["valid"] is False
    assert "ensemble_members_required" in report["error_codes"]
    assert "ensemble_validation_weights_two_members" in report["error_codes"]
    assert "ensemble_grid_step_invalid" in report["error_codes"]


class _FakeStore:
    def __init__(self, records):
        self.records = records

    def get_model(self, model_id, user=None):
        return self.records.get(model_id)


def test_model_service_ensemble_prediction(tmp_path):
    member_a = tmp_path / "member-a"
    member_b = tmp_path / "member-b"
    member_a.mkdir()
    member_b.mkdir()
    (member_a / "metadata.json").write_text(
        json.dumps({"bundle_type": "pretrained_embedding_nn"}),
        encoding="utf-8",
    )
    (member_b / "metadata.json").write_text(
        json.dumps({"bundle_type": "pretrained_embedding_nn"}),
        encoding="utf-8",
    )
    store = _FakeStore(
        {
            "m1": {"bundle_path": str(member_a), "supports_inference": True},
            "m2": {"bundle_path": str(member_b), "supports_inference": True},
        }
    )
    service = ModelService(store)

    def fake_member_predict(model_id, sequences=None, pairs=None):
        if model_id == "m1":
            rows = [[0.8, 0.2], [0.6, 0.4]]
        else:
            rows = [[0.4, 0.6], [0.2, 0.8]]
        return {
            "labels": ["A", "B"],
            "predictions": [{"scores": row} for row in rows],
        }

    ensemble_bundle = tmp_path / "ensemble"
    ensemble_bundle.mkdir()
    (ensemble_bundle / "ensemble.json").write_text(
        json.dumps(
            {
                "members": [
                    {"model_id": "m1", "artifact_dir": str(member_a), "weight": 0.5},
                    {"model_id": "m2", "artifact_dir": str(member_b), "weight": 0.5},
                ],
                "weights": [0.5, 0.5],
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "bundle_type": "ensemble",
        "label_names": ["A", "B"],
        "thresholds": [0.5, 0.5],
        "evaluation_protocol_version": 3,
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(service, "_predict", fake_member_predict)
    try:
        response = service._predict_ensemble(
            "ens",
            ensemble_bundle,
            metadata,
            sequences=["ACD", "EFG"],
        )
    finally:
        monkeypatch.undo()
    assert response["labels"] == ["A", "B"]
    assert response["predictions"][0]["scores"] == pytest.approx([0.6, 0.4])
    assert response["predictions"][0]["predicted_labels"] == ["A"]
    assert response["predictions"][1]["predicted_labels"] == ["B"]


def test_model_service_ensemble_rejects_unregistered_member(tmp_path):
    store = _FakeStore({})
    service = ModelService(store)
    ensemble_bundle = tmp_path / "ensemble"
    ensemble_bundle.mkdir()
    (ensemble_bundle / "ensemble.json").write_text(
        json.dumps(
            {
                "members": [
                    {"model_id": "", "artifact_dir": "x", "weight": 0.5},
                    {"model_id": "", "artifact_dir": "y", "weight": 0.5},
                ],
                "weights": [0.5, 0.5],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(InferenceUnsupportedError):
        service._predict_ensemble(
            "ens",
            ensemble_bundle,
            {"bundle_type": "ensemble"},
            sequences=["ACD"],
        )
