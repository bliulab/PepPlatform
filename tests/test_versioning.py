import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select

from pepplatform.versioning.core import VersionRepository, records, tree_files
from pepplatform.versioning.service import RunVersions, command_options, source_files
from pepplatform.web.training_inputs import WebTrainingInputs


@pytest.fixture
def repo(tmp_path):
    return VersionRepository(
        create_engine(f"sqlite:///{tmp_path / 'versions.db'}"), tmp_path / "versions"
    )


@pytest.fixture
def scenario(tmp_path, repo):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts/train.py").write_text('print("training")\n')
    data = tmp_path / "data"
    for split in ("train", "val", "test"):
        directory = data / split
        directory.mkdir(parents=True)
        (directory / "seqs.fasta").write_text(f">{split}\nACDE\n")
        (directory / "labels.csv").write_text("label\n1\n")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    experiment = {
        "id": "run_1",
        "task_type": "pep_function",
        "owner_user_id": "alice",
        "config": {"seed": 42, "epochs": 2},
        "artifact_dir": str(artifact),
    }
    service = RunVersions(repo, project)
    command = [
        "python",
        str(project / "scripts/train.py"),
        "--data_dir",
        str(data),
        "--seed",
        "42",
        "--epochs",
        "2",
        "--artifact_dir",
        str(artifact),
    ]
    return service, experiment, command, data, artifact


def test_config_content_identity_is_order_independent_and_binding_is_immutable(repo):
    first = repo.register("config", {"config": {"seed": 42, "epochs": 2}})
    same = repo.register("config", {"config": {"epochs": 2, "seed": 42}})
    changed = repo.register("config", {"config": {"epochs": 3, "seed": 42}})
    assert same["id"] == first["id"] != changed["id"]
    repo.start_run("one")
    repo.bind("one", "config", first["id"])
    repo.bind("one", "config", same["id"])
    with pytest.raises(ValueError, match="immutable"):
        repo.bind("one", "config", changed["id"])


def test_snapshot_survives_original_overwrite_and_detects_tampering(repo, tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("original")
    version = repo.snapshot("data_files", {"train/data.csv": source})
    source.write_text("replaced")
    target = repo.object_path(version["id"]) / "files/train/data.csv"
    assert target.read_text() == "original"
    assert target.stat().st_ino != source.stat().st_ino
    assert repo.verify(version["id"])["status"] == "verified"
    target.chmod(0o644)
    target.write_text("tampered")
    with pytest.raises(ValueError, match="mismatch"):
        repo.verify(version["id"])


def test_source_snapshot_retains_layered_runtime_requirements(repo, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    requirements = {
        "requirements.txt": "numpy>=1.24\n",
        "requirements-web.txt": "-r requirements.txt\nfastapi>=0.115\n",
        "requirements-pretrained.txt": "-r requirements.txt\ntransformers==4.48.1\n",
    }
    for name, content in requirements.items():
        (project / name).write_text(content)
    (project / "requirements-dev.txt").write_text("-r requirements-web.txt\npytest>=8\n")

    version = repo.snapshot("source", source_files(project))
    for name in requirements:
        (project / name).unlink()
    retained = repo.object_path(version["id"]) / "files"
    assert {path.name for path in retained.iterdir()} == set(requirements)
    for name, content in requirements.items():
        assert (retained / name).read_text() == content
    assert repo.verify(version["id"])["status"] == "verified"


def test_concurrent_snapshot_deduplicates_without_partial_records(repo, tmp_path):
    source = tmp_path / "weight.pt"
    source.write_bytes(b"model" * 100)
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(
            pool.map(lambda _: repo.snapshot("checkpoint", {"model.pt": source}), range(8))
        )
    assert len({item["id"] for item in result}) == 1
    assert repo.verify(result[0]["id"])["status"] == "verified"
    with repo.engine.connect() as conn:
        assert len(conn.execute(select(records)).all()) == 1
    assert not list((repo.root / "staging").iterdir())


def test_copy_failure_does_not_register_partial_version(repo, tmp_path, monkeypatch):
    from pepplatform.versioning import core

    source = tmp_path / "input"
    source.write_bytes(b"x")
    monkeypatch.setattr(core, "copy_file", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        repo.snapshot("data_files", {"data": source})
    with repo.engine.connect() as conn:
        assert not conn.execute(select(records)).all()
    assert not list((repo.root / "staging").iterdir())


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../b", "a\\b"])
def test_snapshot_rejects_escaping_members(repo, tmp_path, name):
    source = tmp_path / "input"
    source.write_text("x")
    with pytest.raises(ValueError):
        repo.snapshot("data_files", {name: source})


def test_prepare_preserves_parameters_order_and_records_unknown_split_seed(repo, scenario):
    service, experiment, command, data, _ = scenario
    service.begin(experiment)
    result = service.prepare(experiment, command)
    assert result[:2] == command[:2]
    assert result[4:] == command[4:]
    assert result[3] != command[3]
    assert (Path(result[3]) / "train/seqs.fasta").read_bytes() == (
        data / "train/seqs.fasta"
    ).read_bytes()
    run = repo.run(experiment["id"])
    split = next(v for v in run["versions"] if v["role"] == "split")
    assert split["identity"]["split_seed"] is None
    (data / "train/seqs.fasta").write_text("changed outside retained run")
    service.verify_inputs(experiment["id"])


def test_prepare_without_git_retains_verified_source_and_inputs(repo, scenario, monkeypatch):
    service, experiment, command, _, _ = scenario
    monkeypatch.setenv("PATH", str(service.project_root / "no-executables"))
    service.begin(experiment)

    service.prepare(experiment, command)

    run = repo.run(experiment["id"])
    assert run["status"] == "running"
    assert run["details"]["source_revision"] == {"git_commit": None, "git_dirty": None}
    source = next(version for version in run["versions"] if version["role"] == "source")
    assert repo.verify(source["id"])["status"] == "verified"
    service.verify_inputs(experiment["id"])


def test_input_and_source_drift_prevent_sealing(repo, scenario):
    service, experiment, command, _, artifact = scenario
    service.begin(experiment)
    result = service.prepare(experiment, command)
    (artifact / "metadata.json").write_text(json.dumps({"args": experiment["config"]}))
    (artifact / "model.pt").write_bytes(b"weights")
    materialized = Path(result[3]) / "train/seqs.fasta"
    original = materialized.read_bytes()
    materialized.write_text("different")
    with pytest.raises(ValueError, match="input changed"):
        service.seal(experiment)
    materialized.write_bytes(original)
    (service.project_root / "scripts/train.py").write_text('print("changed")')
    with pytest.raises(ValueError, match="source changed"):
        service.seal(experiment)
    assert "model" not in {v["role"] for v in repo.run(experiment["id"])["versions"]}


def test_seal_and_failed_run_never_invent_model_parent(repo, scenario):
    service, experiment, command, _, artifact = scenario
    experiment["rerun_source_experiment_id"] = "older"
    service.begin(experiment)
    service.prepare(experiment, command)
    (artifact / "metadata.json").write_text(
        json.dumps({"args": {"seed": 42, "epochs": 2}, "bundle_type": "sequence_nn"})
    )
    (artifact / "model.pt").write_bytes(b"weights")
    bundle = service.seal(experiment)
    assert bundle != artifact
    assert (bundle / "model.pt").read_bytes() == b"weights"
    result = repo.run(experiment["id"])
    assert result["status"] == "sealed"
    assert {"model", "checkpoint", "effective_config"}.issubset(
        {v["role"] for v in result["versions"]}
    )
    assert result["relations"] == [{"relation": "rerun_of", "target": "older"}]
    failed = {
        **experiment,
        "id": "failed",
        "status": "failed",
        "artifact_dir": str(artifact.parent / "missing"),
    }
    service.begin(failed)
    service.finish(failed)
    assert repo.run("failed")["status"] == "failed"
    assert not repo.run("failed")["versions"]


@pytest.mark.parametrize(
    "task", ["pep_function", "pep_protein_interaction", "pep_hlai", "pep_hlaii"]
)
def test_all_four_tasks_preserve_existing_command_semantics(tmp_path, repo, scenario, task):
    from pepplatform.web.paths import RuntimePaths
    from pepplatform.web.runner import ExperimentRunner

    service, experiment, _, data, artifact = scenario
    experiment.update(
        task_type=task,
        config={"seed": 42, "epochs": 2, "tokenizer": "kmer", "encoder": "onehot", "model": "cnn"},
    )
    if task != "pep_function":
        for split in ("train", "val", "test"):
            (data / split / "pairs.csv").write_text("peptide,protein,label\nACD,EFG,1\n")
    paths = RuntimePaths.from_project_root(service.project_root, tmp_path / "runtime")
    original = ExperimentRunner(None, paths).build_command(
        experiment,
        {
            "format": "sequence_dir" if task == "pep_function" else "pair_split_dir",
            "root_path": str(data),
        },
    )
    service.begin(experiment)
    prepared = service.prepare(experiment, original)
    before, after = command_options(original), command_options(prepared)
    assert set(before) == set(after)
    for flag in set(before) - {"data_dir", "task_root", "train_csv", "val_csv", "test_csv"}:
        assert before[flag][1] == after[flag][1]
    for flag in {"train_csv", "val_csv", "test_csv"} & set(before):
        assert Path(before[flag][1]).read_bytes() == Path(after[flag][1]).read_bytes()


def test_api_scopes_deduplicated_versions_and_redacts_local_paths(repo):
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from pepplatform.versioning.web import register_version_routes

    app = FastAPI()
    record = repo.register("config", {"config": {"private_path": "/private/data.csv"}})
    for name, owner in [("a", "alice"), ("b", "bob")]:
        repo.start_run(name, owner)
        repo.bind(name, "config", record["id"])

    def current(request):
        return request.headers.get("x-user", "alice")

    def require(run_id, user):
        run = repo.run(run_id)
        if run.get("owner_user_id") != user:
            raise HTTPException(403)
        return {"id": run_id, "name": run_id, "status": "running"}

    register_version_routes(app, repo, current, require)
    client = TestClient(app)
    assert client.get("/api/experiments/b/versions").status_code == 403
    payload = client.get("/api/experiments/a/versions").json()
    assert "/private" not in json.dumps(payload)
    assert client.get(f"/api/versions/{record['id']}/runs").json()["items"] == [
        {"id": "a", "name": "a", "status": "running"}
    ]
    assert (
        client.get(f"/api/versions/{record['id']}", headers={"x-user": "outsider"}).status_code
        == 404
    )
    assert (
        client.get("/api/experiments/a/versions?download=true")
        .headers["content-disposition"]
        .endswith('"versions.json"')
    )


def test_model_inference_copy_does_not_mutate_retained_artifact(repo, scenario, tmp_path):
    from pepplatform.versioning.web import VersionedModelStore

    service, experiment, command, _, artifact = scenario
    service.begin(experiment)
    service.prepare(experiment, command)
    (artifact / "metadata.json").write_text('{"args": {}, "bundle_type": "sequence_nn"}')
    (artifact / "model.pt").write_bytes(b"weights")
    bundle = service.seal(experiment)

    class Store:
        def get_model(self, *args, **kwargs):
            return {"id": "m", "experiment_id": experiment["id"], "bundle_path": str(bundle)}

    adapter = VersionedModelStore(Store(), repo)
    working = Path(adapter.get_model("m")["bundle_path"])
    assert working != bundle
    (working / "profile_cache").mkdir()
    (working / "profile_cache/tmp").write_text("derived")
    assert adapter.get_model("m")["id"] == "m"
    (working / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="working copy changed"):
        adapter.get_model("m")
    assert (bundle / "model.pt").read_bytes() == b"weights"


@pytest.mark.parametrize("execution_fails", [False, True])
def test_runner_only_pins_commands_during_execution(
    repo, scenario, tmp_path, monkeypatch, execution_fails
):
    from pepplatform.versioning.web import VersionedExperimentRunner
    from pepplatform.web.ensemble_runner import WebExperimentRunner
    from pepplatform.web.paths import RuntimePaths

    service, experiment, command, _, _ = scenario
    experiment["status"] = "queued"

    class Store:
        def get_experiment(self, experiment_id):
            assert experiment_id == experiment["id"]
            return experiment

    paths = RuntimePaths.from_project_root(service.project_root, tmp_path / "runtime")
    runner = VersionedExperimentRunner(Store(), paths, versions=repo)
    monkeypatch.setattr(WebExperimentRunner, "build_command", lambda *args: list(command))

    assert runner.build_command(experiment, {}) == command
    assert repo.run(experiment["id"])["status"] == "not_recorded"
    assert not (repo.root / "workspaces").exists()

    def execute(self, experiment_id):
        pinned = self.build_command(experiment, {})
        assert pinned[3] != command[3]
        assert Path(pinned[3], "train/seqs.fasta").is_file()
        assert repo.run(experiment_id)["status"] == "running"
        experiment["status"] = "failed"
        if execution_fails:
            raise RuntimeError("execution failed")

    monkeypatch.setattr(WebExperimentRunner, "_run_experiment", execute)
    if execution_fails:
        with pytest.raises(RuntimeError, match="execution failed"):
            runner._run_experiment(experiment["id"])
    else:
        runner._run_experiment(experiment["id"])
    assert repo.run(experiment["id"])["status"] == "failed"
    assert runner.build_command(experiment, {}) == command


def test_web_sequence_pins_splits_and_isolates_writable_caches(repo, scenario):
    original, experiment, command, data, artifact = scenario
    cache = data / "features"
    cache.mkdir()
    (cache / "large-cache.npy").write_bytes(b"historical cache")
    (data / "processed").symlink_to(cache, target_is_directory=True)
    service = RunVersions(repo, original.project_root, input_policy=WebTrainingInputs())
    command += ["--profile_cache_dir", str(data / "old-profile-cache")]
    service.begin(experiment)
    prepared = command_options(service.prepare(experiment, command))
    run = repo.run(experiment["id"])
    assert run["details"]["input_policy"] == WebTrainingInputs.name
    version = next(v for v in run["versions"] if v["role"] == "input:data_dir")
    assert set(version["identity"]["files"]) == {
        f"{split}/{name}"
        for split in ("train", "val", "test")
        for name in ("seqs.fasta", "labels.csv")
    }
    assert prepared["profile_cache_dir"][1] == str(artifact.parent / "profile_cache")
    assert (cache / "large-cache.npy").read_bytes() == b"historical cache"
    service.verify_inputs(experiment["id"])
    (Path(prepared["data_dir"][1]) / "train/seqs.fasta").write_text("tampered")
    with pytest.raises(ValueError, match="input changed"):
        service.verify_inputs(experiment["id"])


@pytest.mark.parametrize("task", ["pep_protein_interaction", "pep_hlai", "pep_hlaii"])
def test_web_pair_pins_required_inputs_without_scanning_task_history(
    repo, scenario, tmp_path, task
):
    from pepplatform.web.paths import RuntimePaths
    from pepplatform.web.runner import ExperimentRunner

    original, experiment, _, data, artifact = scenario
    task_root = tmp_path / "task-history"
    task_root.mkdir()
    old_cache = task_root / "features"
    old_cache.mkdir()
    (old_cache / "large.npy").write_bytes(b"do not copy or overwrite")
    (task_root / "processed").symlink_to(data, target_is_directory=True)
    pseudo = task_root / "pseudoseqs_decoded.csv"
    pseudo.write_text("allele,pseudoseq\nHLA-A,ACDE\n")
    for split in ("train", "val", "test"):
        (data / split / "pairs.csv").write_text("peptide,protein,label\nACD,EFG,1\n")
    experiment.update(
        task_type=task,
        config={"tokenizer": "kmer", "encoder": "onehot", "model": "cnn", "seed": 42},
    )
    paths = RuntimePaths.from_project_root(original.project_root, tmp_path / "runtime")
    command = ExperimentRunner(None, paths).build_command(
        experiment,
        {
            "format": "pair_split_dir",
            "root_path": str(data),
            "metadata": {"source_root": str(task_root), "feature_dir": str(old_cache)},
        },
    )
    service = RunVersions(repo, original.project_root, input_policy=WebTrainingInputs())
    service.begin(experiment)
    prepared = command_options(service.prepare(experiment, command))
    root = Path(prepared["task_root"][1])
    assert (root / ".web_input_root.json").is_file()
    assert not (root / "features").exists()
    assert (root / "pseudoseqs_decoded.csv").exists() == (task == "pep_hlai")
    assert prepared["feature_dir"][1] == str(artifact.parent / "feature_cache")
    for split in ("train", "val", "test"):
        assert (
            Path(prepared[f"{split}_csv"][1]).read_bytes()
            == (data / split / "pairs.csv").read_bytes()
        )
    # Existing trainers may create default tokenizers/profile caches here.
    (root / "generated-cache").mkdir()
    (root / "generated-cache/value").write_text("generated")
    service.verify_inputs(experiment["id"])
    assert (old_cache / "large.npy").read_bytes() == b"do not copy or overwrite"
    before = command_options(command)
    for flag in set(before) - {"task_root", "train_csv", "val_csv", "test_csv", "feature_dir"}:
        assert prepared[flag][1] == before[flag][1]


def test_web_profile_dependency_paths_are_pinned_without_editing_originals(
    repo, scenario, tmp_path
):
    original, experiment, command, _, _ = scenario
    directory = tmp_path / "profile"
    directory.mkdir()
    matrix = directory / "matrix.pt"
    matrix.write_bytes(b"profile matrix")
    external = tmp_path / "external.pt"
    external.write_bytes(b"external matrix")
    index = directory / "index.json"
    index.write_text(json.dumps({"sample": str(matrix), "other": str(external)}))
    original_index = index.read_bytes()
    command += ["--pssm_index_path", str(index)]
    service = RunVersions(repo, original.project_root, input_policy=WebTrainingInputs())
    service.begin(experiment)
    prepared = command_options(service.prepare(experiment, command))
    pinned_index = Path(prepared["pssm_index_path"][1])
    pinned = json.loads(pinned_index.read_text())
    assert (pinned_index.parent / pinned["sample"]).read_bytes() == b"profile matrix"
    assert (pinned_index.parent / pinned["other"]).read_bytes() == b"external matrix"
    assert all(
        not Path(value).is_absolute() and ".." not in Path(value).parts for value in pinned.values()
    )
    assert index.read_bytes() == original_index
    assert {v["role"] for v in repo.run(experiment["id"])["versions"]} >= {
        "original:pssm_index_path",
        "input:pssm_index_path",
    }
    matrix.write_bytes(b"original changed after pinning")
    external.write_bytes(b"original changed after pinning")
    service.verify_inputs(experiment["id"])
    (pinned_index.parent / pinned["sample"]).write_bytes(b"tampered retained dependency")
    with pytest.raises(ValueError, match="input changed"):
        service.verify_inputs(experiment["id"])


def test_default_versioning_policy_keeps_strict_directory_symlink_behavior(repo, scenario):
    service, experiment, command, data, _ = scenario
    (data / "alias").symlink_to(data / "train", target_is_directory=True)
    service.begin(experiment)
    with pytest.raises(ValueError, match="directory symlink"):
        service.prepare(experiment, command)


def test_web_relative_profile_manifest_retains_its_original_identity(repo, scenario, tmp_path):
    original, experiment, command, _, _ = scenario
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "matrix.pt").write_bytes(b"matrix")
    index = profile / "index.json"
    index.write_text('{"samples": {"sample": {"path": "matrix.pt"}}}\n')
    service = RunVersions(repo, original.project_root, input_policy=WebTrainingInputs())
    service.begin(experiment)
    prepared = command_options(
        service.prepare(experiment, command + ["--pssm_index_path", str(index)])
    )
    assert Path(prepared["pssm_index_path"][1]).read_bytes() == index.read_bytes()
    assert "original:pssm_index_path" not in {
        v["role"] for v in repo.run(experiment["id"])["versions"]
    }
    service.verify_inputs(experiment["id"])


def test_web_pair_resolves_original_default_csv_paths_before_pinning(repo, scenario, tmp_path):
    original, experiment, _, _, artifact = scenario
    history = tmp_path / "history"
    processed = tmp_path / "resolved_splits"
    history.mkdir()
    processed.mkdir()
    (history / "processed").symlink_to(processed, target_is_directory=True)
    for name in ("train.csv", "val.csv", "test_1440.csv"):
        (processed / name).write_text("peptide,protein,label\nACD,EFG,1\n")
    experiment.update(task_type="pep_protein_interaction")
    command = ["python", "train.py", "--task_root", str(history), "--artifact_dir", str(artifact)]
    service = RunVersions(repo, original.project_root, input_policy=WebTrainingInputs())
    service.begin(experiment)
    prepared = command_options(service.prepare(experiment, command))
    for split, name in (("train", "train.csv"), ("val", "val.csv"), ("test", "test_1440.csv")):
        assert Path(prepared[f"{split}_csv"][1]).read_bytes() == (processed / name).read_bytes()


def test_web_local_stream_model_source_is_an_immutable_input(repo, scenario, tmp_path):
    original, experiment, command, _, _ = scenario
    model = tmp_path / "local-model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    (model / "model.safetensors").write_bytes(b"model weights")
    service = RunVersions(repo, original.project_root, input_policy=WebTrainingInputs())
    service.begin(experiment)
    prepared = command_options(
        service.prepare(experiment, command + ["--partner_model_source", str(model)])
    )
    pinned = Path(prepared["partner_model_source"][1])
    (model / "model.safetensors").write_bytes(b"later original revision")
    assert (pinned / "model.safetensors").read_bytes() == b"model weights"
    service.verify_inputs(experiment["id"])


def test_web_version_preparation_failure_does_not_block_next_task(tmp_path, monkeypatch):
    import sys
    import time
    from pepplatform.versioning.web import VersionedExperimentRunner
    from pepplatform.web.paths import RuntimePaths
    from pepplatform.web.store import MetadataStore

    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts/train.py").write_text("print('training')\n")
    paths = RuntimePaths.from_project_root(project, tmp_path / "runtime").ensure()
    store = MetadataStore(paths.database)
    data = tmp_path / "data"
    (data / "train").mkdir(parents=True)
    (data / "train/seqs.fasta").write_text(">a\nACD\n")
    (data / "train/labels.csv").write_text("label\n1\n")
    dataset = store.create_dataset("demo", "pep_function", "upload", "sequence_dir", data)
    for name in ("broken", "next"):
        store.create_experiment(
            name,
            dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
            paths.experiments / name / "train.log",
            paths.experiments / name / "artifact",
            experiment_id=name,
        )
    runner = VersionedExperimentRunner(store, paths, input_policy=WebTrainingInputs())
    prepare = runner.version_service.prepare

    def pin(experiment, command):
        if experiment["id"] == "broken":
            raise OSError("input snapshot disk failure")
        # Keep a real version preparation lifecycle but a tiny execution command.
        command = [sys.executable, "-c", "print('trained')", "--data_dir", str(data)]
        return prepare(experiment, command)

    monkeypatch.setattr(runner.version_service, "prepare", pin)
    runner.start()
    try:
        runner.submit("broken")
        runner.submit("next")
        deadline = time.monotonic() + 10
        while store.get_experiment("next")["status"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.01)
        while store.get_experiment("next")["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.get_experiment("broken")["status"] == "failed"
        assert runner.versions.run("broken")["status"] == "failed"
        assert (
            "input snapshot disk failure"
            in Path(store.get_experiment("broken")["log_path"]).read_text()
        )
        assert store.get_experiment("next")["status"] == "succeeded"
        assert runner.thread.is_alive()
    finally:
        runner.stop()
        runner.thread.join(timeout=2)
