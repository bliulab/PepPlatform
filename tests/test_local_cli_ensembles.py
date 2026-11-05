"""Local application acceptance tests; every model/data output is temporary."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pepplatform.web.ensemble_math import fuse, select
from pepplatform.web.inference import ModelService
from pepplatform_cli import data
from pepplatform_cli.ensemble import prepare, preflight, assemble
from pepplatform_cli.inference import predict_artifact
from pepplatform_cli.storage import (CLIError, inventory, read_json, write_json,
                                     digest_json, file_digest, guard_output)

TASKS = ["pep_function", "pep_hlai", "pep_hlaii", "pep_protein_interaction"]
METHODS = ["probability_mean", "logit_mean", "rank_mean"]
SOURCES = ["equal", "custom", "validation"]


def dataset_files(root, task):
    for split, sequences in {"train": ["AAAA", "CCCC", "DDDD", "EEEE"],
                             "val": ["FFFF", "GGGG", "HHHH", "IIII"],
                             "test": ["KKKK", "LLLL", "MMMM", "NNNN"]}.items():
        directory = root / split
        directory.mkdir(parents=True)
        if task == "pep_function":
            (directory / "seqs.fasta").write_text("".join(f">{i}\n{x}\n" for i, x in enumerate(sequences)))
            (directory / "labels.csv").write_text("A,B\n0,1\n1,0\n0,1\n1,0\n")
        else:
            (directory / "pairs.csv").write_text("sample_id,peptide,protein,label\n" +
                "".join(f"{i},{sequence},ACDE,{i % 2}\n" for i, sequence in enumerate(sequences)))
            (directory / "labels.csv").write_text("label\n0\n1\n0\n1\n")


def members_config(tmp_path, task="pep_function", count=3, proof=True):
    root = tmp_path / "data"
    dataset_files(root, task)
    identity = data.validation_identity(data.dataset({"task_type": task, "data_dir": str(root)}))
    members = []
    for index in range(count):
        path = tmp_path / f"member{index}"
        path.mkdir()
        metadata = {"bundle_type": "sequence_nn" if task == "pep_function" else "pair_vector_nn",
                    "supports_inference": True, "evaluation_protocol_version": 3,
                    "feature_fit_scope": "train_only", "label_names": ["A", "B"] if task == "pep_function" else ["interaction"],
                    "args": {"task_type": task}, "test_member_index": index}
        write_json(path / "metadata.json", metadata)
        # Non-object JSON resources must be copied without being reinterpreted as metadata.
        (path / "vocabulary.json").write_text('["A", "C"]')
        if proof:
            write_json(path / "web_ensemble_provenance.json", {**identity, "dataset_id": "original-web-id"})
        members.append(str(path))
    return {"task_type": task, "data_dir": str(root), "members": members, "grid_step": 0.5}


def fake_prediction(self, model_id, sequences=None, pairs=None):
    record = self.store.get_model(model_id)
    metadata = read_json(Path(record["bundle_path"]) / "metadata.json")
    values = sequences if sequences is not None else [p["peptide"] for p in pairs]
    columns = len(metadata["label_names"])
    index = metadata["test_member_index"]
    scores = [[((sum(map(ord, x)) + col * 7 + index * 3) % 19 + 1) / 21
               for col in range(columns)] for x in values]
    return {"labels": metadata["label_names"], "score_kind": "probability",
            "predictions": [{"scores": row} for row in scores]}


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(ModelService, "_predict", fake_prediction)
    monkeypatch.setattr(ModelService, "readiness", lambda *_: {"ready": True, "warnings": []})


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("source", SOURCES)
def test_full_matrix_matches_existing_core_and_is_independent(tmp_path, fake_runtime, task, method, source):
    config = {**members_config(tmp_path, task), "method": method, "weight_source": source, "weights": [2, 3, 1]}
    original = [inventory(path) for path in config["members"]]
    request = prepare(config)
    output = tmp_path / "ensemble"
    result = assemble(request, output)
    val = data.read_split(request["dataset"], "val")
    # Compare with the same public numerical core used by the Web implementation.
    service = ModelService(type("Store", (), {"get_model": lambda self, id: next(
        r for r in request["members"] if r["id"] == id)})())
    matrices = [np.asarray([p["scores"] for p in service.predict(r["id"], **val["inputs"])["predictions"]])
                for r in request["members"]]
    if task != "pep_function":
        matrices = [m[:, 0] for m in matrices]
    weights, point, _ = select(matrices, val["y"], request["label_names"], request["definition"])
    np.testing.assert_array_equal(result["weights"], weights)
    assert read_json(output / "ensemble.json")["operating_point"] == point.to_dict()
    assert [inventory(path) for path in config["members"]] == original
    for path in config["members"]:
        shutil.rmtree(path)
    predicted = predict_artifact(output, val["inputs"], tmp_path / "runtime")
    scores = np.asarray([row["scores"] for row in predicted["predictions"]])
    expected = fuse(matrices, weights, method).reshape(scores.shape)
    np.testing.assert_allclose(scores, expected, atol=0, rtol=0)
    assert predicted["provenance_status"] == "verified"
    assert predicted["score_kind"] == ("rank" if method == "rank_mean" else "probability")
    assert (output / "audit.json").is_file()


def test_strict_legacy_mode_and_invalid_proof_are_distinct(tmp_path, fake_runtime):
    config = members_config(tmp_path, proof=False)
    assert preflight(config)["error_codes"] == ["provenance_required"]
    config["allow_unverified_provenance"] = True
    assert preflight(config)["provenance_status"] == "unverified"
    result = assemble(prepare(config), tmp_path / "ensemble")
    assert result["provenance_status"] == "unverified"
    assert set(read_json(tmp_path / "ensemble/ensemble.json")["validation_score_sources"]) == {"fresh_frozen_member_inference"}
    write_json(Path(config["members"][0]) / "web_ensemble_provenance.json", {"schema_version": 999})
    assert preflight(config)["error_codes"] == ["provenance_invalid"]


@pytest.mark.parametrize("change,code", [("duplicate", "member_duplicate"), ("labels", "label_mismatch"),
    ("nested", "nested_unsupported"), ("protocol", "protocol_required"), ("data", "provenance_data_changed"),
    ("weight", "ensemble_weights_invalid"), ("test_name", "test_splits_invalid")])
def test_preflight_failures(tmp_path, fake_runtime, change, code):
    config = members_config(tmp_path)
    path = Path(config["members"][1]) / "metadata.json"
    metadata = read_json(path)
    if change == "duplicate":
        config["members"][1] = config["members"][0]
    elif change == "labels":
        metadata["label_names"].reverse()
    elif change == "nested":
        metadata["bundle_type"] = "cli_ensemble"
    elif change == "protocol":
        metadata["feature_fit_scope"] = "train_val"
    elif change == "data":
        (Path(config["data_dir"]) / "val/seqs.fasta").write_text(">0\nVVVV\n>1\nGGGG\n>2\nHHHH\n>3\nIIII\n")
    elif change == "weight":
        config.update(weight_source="custom", weights=[0, 0, 0])
    else:
        config["test_splits"] = ["val"]
    write_json(path, metadata)
    assert preflight(config)["error_codes"] == [code]


def test_member_mutation_and_output_guards(tmp_path, fake_runtime):
    config = members_config(tmp_path)
    request = prepare(config)
    (Path(config["members"][0]) / "changed.txt").write_text("changed")
    with pytest.raises(CLIError, match="changed after preflight"):
        assemble(request, tmp_path / "ensemble")
    with pytest.raises(CLIError):
        guard_output(Path(config["members"][0]) / "new", config["members"])
    link = tmp_path / "link"
    link.symlink_to(config["data_dir"], target_is_directory=True)
    with pytest.raises(CLIError):
        guard_output(link / "new", [config["data_dir"]])


@pytest.mark.parametrize("mutation", ["definition", "threshold", "snapshot"])
def test_frozen_bundle_tampering_fails(tmp_path, fake_runtime, mutation):
    config = members_config(tmp_path)
    request = prepare(config)
    output = tmp_path / "ensemble"
    assemble(request, output)
    if mutation == "definition":
        manifest = read_json(output / "ensemble.json")
        manifest["weights"] = [1, 0, 0]
        write_json(output / "ensemble.json", manifest)
    elif mutation == "threshold":
        metadata = read_json(output / "metadata.json")
        metadata["operating_point"]["values"][0] = 0.99
        write_json(output / "metadata.json", metadata)
    else:
        (output / "members/0/vocabulary.json").write_text("[]")
    with pytest.raises(CLIError):
        predict_artifact(output, {"sequences": ["FFFF", "GGGG"]}, tmp_path / "runtime")


def test_test_access_only_after_freeze_and_named_splits(tmp_path, fake_runtime, monkeypatch):
    config = members_config(tmp_path)
    root, output = Path(config["data_dir"]), tmp_path / "ensemble"
    (root / "test").rename(root / "test_warm")
    (root / "test_cold").mkdir()
    (root / "test_cold/seqs.fasta").write_text(">0\nPPPP\n>1\nQQQQ\n>2\nRRRR\n>3\nSSSS\n")
    shutil.copyfile(root / "test_warm/labels.csv", root / "test_cold/labels.csv")
    config.update(evaluate_test=True, test_splits=["test_warm", "test_cold"])
    request = prepare(config)
    original = data.read_split

    def spy(dataset, name):
        if name.startswith("test"):
            assert (output / "ensemble.json").is_file()
            assert (output / "weight_search.json").is_file()
        return original(dataset, name)

    monkeypatch.setattr(data, "read_split", spy)
    result = assemble(request, output)
    assert set(result["test_splits"]) == {"test_warm", "test_cold"}
    assert "test" not in read_json(output / "evaluation.json")


def test_default_never_reads_test_and_verified_cache_is_bound(tmp_path, fake_runtime, monkeypatch):
    config = members_config(tmp_path)
    val = data.read_split(data.dataset(config), "val")
    for member in config["members"]:
        path = Path(member)
        np.savez_compressed(path / "evaluation_scores.npz", validation_labels=val["y"],
                            validation_scores=0.1 + val["y"] * 0.8,
                            validation_sample_ids=np.asarray(val["identity"]["sample_ids"]))
        proof = read_json(path / "web_ensemble_provenance.json")
        proof.update(evaluation_scores_sha256=file_digest(path / "evaluation_scores.npz"), score_kind="probability")
        write_json(path / "web_ensemble_provenance.json", proof)
    request = prepare(config)
    original = data.read_split
    monkeypatch.setattr(data, "read_split", lambda d, n: pytest.fail("Test read") if n.startswith("test") else original(d, n))
    monkeypatch.setattr(ModelService, "_predict", lambda *_a, **_kw: pytest.fail("Cache should be usable"))
    output = tmp_path / "ensemble"
    assemble(request, output)
    assert set(read_json(output / "ensemble.json")["validation_score_sources"]) == {"verified_cached_validation"}


def test_legacy_local_v1_and_rank_single_row(tmp_path, fake_runtime):
    config = members_config(tmp_path)
    legacy = tmp_path / "v1"
    legacy.mkdir()
    write_json(legacy / "metadata.json", {"bundle_type": "ensemble", "args": {"task_type": "pep_function"},
                "label_names": ["A", "B"], "thresholds": [0.5, 0.5]})
    write_json(legacy / "ensemble.json", {"schema_version": 1, "method": "probability_mean",
        "members": [{"artifact_dir": p} for p in config["members"]], "weights": [0.2, 0.3, 0.5], "label_names": ["A", "B"]})
    result = predict_artifact(legacy, {"sequences": ["FFFF", "GGGG"]}, tmp_path / "runtime1")
    assert result["provenance_status"] == "unverified"
    config["method"] = "rank_mean"
    output = tmp_path / "rank"
    assemble(prepare(config), output)
    with pytest.raises(CLIError, match="at least two"):
        predict_artifact(output, {"sequences": ["FFFF"]}, tmp_path / "runtime2")


def test_real_cpu_pair_bundle_reload_and_cli_history(tmp_path, monkeypatch, capsys):
    import joblib
    import torch
    from sklearn.ensemble import RandomForestClassifier
    from pepplatform_cli.__main__ import main
    config = members_config(tmp_path, "pep_protein_interaction", count=2)
    train = data.read_split(data.dataset(config), "train")
    for index, member in enumerate(config["members"]):
        path = Path(member)
        glove = path / "glove.vec"
        glove.write_text("".join(f"{letter} {i / 20} {(20 - i) / 20}\n" for i, letter in enumerate("ACDEFGHIKLMNPQRSTVWY")) + "<UNK> 0 0\n")
        metadata = {**read_json(path / "metadata.json"), "bundle_type": "pair_vector_classical", "threshold": 0.5,
            "classic": {"tokenizer": "kmer", "encoder": "glove", "max_length": 4, "k": 1, "glove_path": str(glove)},
            "pair": {"fusion_mode": "concat_all"}, "args": {"task_type": "pep_protein_interaction", "tokenizer": "kmer",
                "encoder": "glove", "model": "rf", "classifier": "rf"}}
        features = ModelService._extract_pair_vector_features(metadata, train["inputs"]["pairs"], torch.device("cpu"), bundle=path).numpy()
        estimator = RandomForestClassifier(n_estimators=7, random_state=42 + index, n_jobs=1).fit(features, train["y"])
        joblib.dump({"estimator_kind": "rf", "estimator": estimator}, path / "model.joblib")
        write_json(path / "metadata.json", metadata)
    workspace = tmp_path / "workspace"
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    assert main(["ensemble", "create", "--config", str(config_path), "--workspace", str(workspace)]) == 0
    from pepplatform_cli.storage import history
    artifact = next(r["artifact_dir"] for r in history(workspace) if r["kind"] == "ensemble")
    assert main(["predict", "--artifact-dir", artifact, "--input", str(Path(config["data_dir"]) / "val/pairs.csv"), "--workspace", str(workspace)]) == 0
    runs = history(workspace)
    prediction = next(r for r in runs if r["kind"] == "predict")
    assert prediction["status"] == "succeeded"
    assert "score_kind" in Path(prediction["csv"]).read_text()
    assert main(["history", "export", prediction["run_id"], "--output", str(tmp_path / "export.csv"), "--workspace", str(workspace)]) == 0
    assert (tmp_path / "export.csv").read_bytes() == Path(prediction["csv"]).read_bytes()
    assert main(["history", "list", "--workspace", str(workspace)]) == 0
