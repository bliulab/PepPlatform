"""Verify exact checkpoint reconstruction and atomic installation using tiny weights."""

import copy
import json
import random
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from test_docker_models import models, no_network
from test_docker_models import paper as paper_fixture

paper = paper_fixture


def manifest(root):
    return {
        p.name: {"size_bytes": p.stat().st_size, "sha256": models.sha256_file(p)}
        for p in root.iterdir()
        if p.is_file()
    }


@pytest.fixture
def official(paper, monkeypatch):
    model = next(m for m in paper.lock["models"] if m["key"] == "prott5")
    original = paper.source / "prott5"
    # Preserve non-contiguous views in the official .bin file.
    shared = torch.arange(12, dtype=torch.float32).reshape(4, 3).t()
    state = {
        "shared.weight": shared,
        "encoder.embed_tokens.weight": shared,
        "decoder.embed_tokens.weight": shared,
        "lm_head.weight": shared,
        "encoder.block.0.weight": torch.arange(20, dtype=torch.float32).reshape(4, 5).t(),
        "decoder.block.0.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
    }
    assert not shared.is_contiguous()
    filenames = model["weight_files"][1:]
    mapping = {
        "shared.weight": filenames[0],
        "encoder.block.0.weight": filenames[0],
        "decoder.block.0.weight": filenames[1],
    }
    for filename in filenames:
        save_file(
            {k: state[k].contiguous() for k, name in mapping.items() if name == filename},
            original / filename,
            metadata={"format": "pt"},
        )
    index_path = original / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": mapping}, indent=2) + "\n")
    model["files"] = manifest(original)
    model["model_fingerprint"] = models.fingerprint(model["files"])
    model["weight_sha256"] = models.weight_digest(model, model["files"])
    source = paper.root.parent / "official"
    source.mkdir()
    torch.save(state, source / "pytorch_model.bin")
    (source / "config.json").write_text('{"architectures": ["T5ForConditionalGeneration"]}\n')
    (source / "vocab.txt").write_bytes((original / "vocab.txt").read_bytes())
    headers = {}
    for filename in filenames:
        with (original / filename).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            headers[filename] = stream.read(size).decode("utf-8")
    recipe = {
        "schema_version": 1,
        "source": {
            "repo_id": model["repo_id"],
            "revision": "b" * 40,
            "weight_file": "pytorch_model.bin",
            "files": manifest(source),
        },
        "target": {k: model[k] for k in ["weight_sha256", "model_fingerprint"]},
        "copied_files": ["vocab.txt"],
        "text_files": {
            name: (original / name).read_bytes().decode("utf-8")
            for name in ["config.json", "model.safetensors.index.json"]
        },
        "shard_headers": headers,
    }
    recipe_path = paper.root.parent / "recipe.json"

    def update_recipe():
        recipe_path.write_text(json.dumps(recipe, indent=2) + "\n")
        monkeypatch.setattr(models, "PROTT5_RECIPE_SHA256", models.sha256_file(recipe_path))

    update_recipe()
    monkeypatch.setattr(models, "PROTT5_RECIPE_PATH", recipe_path)
    models.LOCK_PATH.write_text(json.dumps(paper.lock))
    return SimpleNamespace(
        model=model,
        original=original,
        source=source,
        recipe=recipe,
        state=state,
        update_recipe=update_recipe,
    )


def prepare_local(paper, official):
    def download(model):
        assert model["key"] != "prott5", "Local official ProtT5 must never download"
        return paper.source / model["key"]

    models.prepare(paper.root, paper.lock, downloader=download, prott5_source=official.source)


def test_release_recipe_pins_upstream_and_preserves_existing_package_lock():
    model = next(m for m in models.load_lock()["models"] if m["key"] == "prott5")
    recipe = models.load_prott5_recipe(model)
    assert recipe["source"]["revision"] == "973be27c52ee6474de9c945952a8008aeb2a1a73"
    assert recipe["source"]["files"]["pytorch_model.bin"]["sha256"] == (
        "c06141d42e93c12b5f6d95c701952156bd4079661f5f2d981d0d2036ba96dae9"
    )
    assert len(recipe["shard_headers"]) == 6
    assert models.sha256_file(models.LOCK_PATH) == (
        "93945413afa13c15f06872a79d4325d0c5a78894c714a3afb04282675407ebc5"
    )


def test_reconstruction_matches_all_original_bytes_without_consuming_rng(paper, official):
    before = {p.name: p.read_bytes() for p in official.source.iterdir()}
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    models.restore_prott5(official.source, paper.root, official.model)
    models.verify_tree(paper.root, official.model)
    assert {p.name: p.read_bytes() for p in paper.root.iterdir()} == {
        p.name: p.read_bytes() for p in official.original.iterdir()
    }
    assert before == {p.name: p.read_bytes() for p in official.source.iterdir()}
    assert random.getstate() == python_rng
    assert np.random.get_state()[0] == numpy_rng[0]
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert np.random.get_state()[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_default_download_and_restore_needs_no_original_shard_directory(
    paper, official, monkeypatch
):
    original_bytes = {p.name: p.read_bytes() for p in official.original.iterdir()}
    official.original.rename(official.original.with_name("paper-shards-not-available"))
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        if kwargs["repo_id"] == official.model["repo_id"]:
            return str(official.source)
        key = next(m["key"] for m in paper.lock["models"] if m["repo_id"] == kwargs["repo_id"])
        return str(paper.source / key)

    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot, hf_hub_download=lambda **_: pytest.fail()),
    )
    models.prepare(paper.root, paper.lock)
    installed = models.installed_model(paper.root, official.model)
    assert {p.name: p.read_bytes() for p in installed.iterdir()} == original_bytes
    request = next(c for c in calls if c["repo_id"] == official.model["repo_id"])
    assert request["revision"] == official.recipe["source"]["revision"]
    assert set(request["allow_patterns"]) == set(official.recipe["source"]["files"])
    assert request["endpoint"] == "https://mirror.example"


def test_local_preparation_is_idempotent_and_check_only_stays_offline(paper, official):
    prepare_local(paper, official)
    installed = models.installed_model(paper.root, official.model)
    before = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in installed.iterdir()}
    models.prepare(paper.root, paper.lock, downloader=no_network, prott5_source=Path("not-needed"))
    models.prepare(paper.root, paper.lock, check_only=True, downloader=no_network)
    assert before == {p.name: (p.stat().st_ino, p.stat().st_mtime_ns) for p in installed.iterdir()}


@pytest.mark.parametrize("filename", ["pytorch_model.bin", "config.json", "vocab.txt"])
def test_bad_official_source_is_rejected_before_torch_load(paper, official, monkeypatch, filename):
    path = official.source / filename
    path.write_bytes(b"x" * path.stat().st_size)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Unverified weights loaded"))
    with pytest.raises(models.ModelError, match="official source size/SHA256 mismatch"):
        models.restore_prott5(official.source, paper.root, official.model)
    assert not paper.root.exists()


def test_recipe_tampering_fails_before_creating_output(paper, official):
    models.PROTT5_RECIPE_PATH.write_text(models.PROTT5_RECIPE_PATH.read_text() + " ")
    with pytest.raises(models.ModelError, match="recipe SHA256 mismatch"):
        models.restore_prott5(official.source, paper.root, official.model)
    assert not paper.root.exists()


@pytest.mark.parametrize("change", ["missing", "shape", "dtype", "offset"])
def test_invalid_tensor_cannot_replace_previous_installation(paper, official, change):
    prepare_local(paper, official)
    installed = models.installed_model(paper.root, official.model)
    bad = installed / official.model["weight_files"][1]
    bad.write_bytes(b"x" * bad.stat().st_size)
    before = {p.name: p.read_bytes() for p in installed.iterdir()}
    state = copy.copy(official.state)
    key = "encoder.block.0.weight"
    if change == "missing":
        state.pop(key)
    elif change == "shape":
        state[key] = state[key].t()
    elif change == "dtype":
        state[key] = state[key].double()
    else:
        name = official.model["weight_files"][1]
        header = json.loads(official.recipe["shard_headers"][name])
        header[key]["data_offsets"][1] += 4
        official.recipe["shard_headers"][name] = json.dumps(header)
    torch.save(state, official.source / "pytorch_model.bin")
    official.recipe["source"]["files"] = manifest(official.source)
    official.update_recipe()
    with pytest.raises(models.ModelError, match="Preparation incomplete: prott5"):
        prepare_local(paper, official)
    assert before == {p.name: p.read_bytes() for p in installed.iterdir()}
    assert (paper.root / "models/prott5").resolve() == installed
    assert not list(installed.parent.glob(".model-install-*"))


def test_output_disk_space_checked_before_download(paper, monkeypatch):
    monkeypatch.setattr(models.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(models.ModelError, match="Preparation incomplete"):
        models.prepare(paper.root, paper.lock, downloader=no_network)
    assert not (paper.root / "installed.lock.json").exists()


def test_interrupted_restore_can_retry_without_replacing_previous_files(
    paper, official, monkeypatch
):
    prepare_local(paper, official)
    installed = models.installed_model(paper.root, official.model)
    (installed / "config.json").write_text("corrupt installed copy")
    before = {p.name: p.read_bytes() for p in installed.iterdir()}
    restore = models.restore_prott5

    def interrupted(_source, destination, _model):
        destination.mkdir()
        (destination / "partial").write_bytes(b"partial data")
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(models, "restore_prott5", interrupted)
    with pytest.raises(models.ModelError, match="Preparation incomplete: prott5"):
        prepare_local(paper, official)
    assert before == {p.name: p.read_bytes() for p in installed.iterdir()}
    monkeypatch.setattr(models, "restore_prott5", restore)
    prepare_local(paper, official)
    models.installed_model(paper.root, official.model)
    assert list(installed.parent.glob("*.previous-*"))


@pytest.mark.parametrize(
    "flags",
    [["--check-only"], ["--import-from", "package"], ["--export", "new", "--reference", "ref"]],
)
def test_local_source_flag_rejects_other_modes(flags):
    with pytest.raises(SystemExit) as error:
        models.main(["--prott5-source", "official", *flags])
    assert error.value.code == 2


def test_cli_help_does_not_require_optional_model_libraries():
    script = Path(models.__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"], capture_output=True, text=True, check=True
    )
    assert "--prott5-source" in result.stdout


def test_target_file_hash_still_rejects_an_altered_payload(paper, official):
    state = copy.copy(official.state)
    state["decoder.block.0.weight"] = state["decoder.block.0.weight"] + 1
    torch.save(state, official.source / "pytorch_model.bin")
    official.recipe["source"]["files"] = manifest(official.source)
    official.update_recipe()
    with pytest.raises(models.ModelError, match="restored file size/SHA256 mismatch"):
        models.restore_prott5(official.source, paper.root, official.model)
