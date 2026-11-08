from pepplatform.web.model_artifacts import delete_experiment_artifact, delete_model_artifact
from pepplatform.web.paths import RuntimePaths


def test_delete_model_artifact_removes_runtime_bundle(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    artifact_dir = paths.experiments / "experiment-1" / "artifact"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "metadata.json").write_text("{}", encoding="utf-8")

    deleted = delete_model_artifact(artifact_dir, paths)

    assert deleted is True
    assert not artifact_dir.exists()


def test_delete_model_artifact_ignores_paths_outside_runtime(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    outside_artifact = tmp_path / "outside-artifact"
    outside_artifact.mkdir()
    (outside_artifact / "model.pt").write_text("model", encoding="utf-8")

    deleted = delete_model_artifact(outside_artifact, paths)

    assert deleted is False
    assert outside_artifact.exists()


def test_delete_model_artifact_does_not_remove_runtime_roots(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()

    assert delete_model_artifact(paths.experiments, paths) is False
    assert paths.experiments.exists()


def test_delete_experiment_artifact_removes_runtime_experiment_dir(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    experiment_dir = paths.experiments / "experiment-1"
    experiment_dir.mkdir()
    (experiment_dir / "train.log").write_text("done", encoding="utf-8")

    deleted = delete_experiment_artifact("experiment-1", paths)

    assert deleted is True
    assert not experiment_dir.exists()


def test_delete_experiment_artifact_ignores_paths_outside_runtime(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    outside = paths.root / "datasets"
    outside.mkdir(exist_ok=True)

    deleted = delete_experiment_artifact("../datasets", paths)

    assert deleted is False
    assert outside.exists()
