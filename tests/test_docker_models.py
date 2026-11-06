"""Exercise portable model preparation with tiny, fully identified assets."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docker" / "download_models.py"
spec = importlib.util.spec_from_file_location("paper_docker_models", SCRIPT)
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


def test_default_transport_does_not_import_download_libraries(monkeypatch):
    import builtins

    monkeypatch.setenv("PEP_HF_TLS12", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"requests", "huggingface_hub"}:
            pytest.fail("Default transport configuration must remain lazy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert models.configure_download_http() is False
    assert models.os.environ["HF_HUB_DISABLE_XET"] == "0"


def test_tls12_transport_preserves_certificate_checks_for_direct_and_proxy(monkeypatch):
    import ssl

    pytest.importorskip("requests")
    factories = []
    monkeypatch.setenv("PEP_HF_TLS12", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            configure_http_backend=lambda **kwargs: factories.append(kwargs["backend_factory"])
        ),
    )
    assert models.configure_download_http() is True
    assert models.os.environ["HF_HUB_DISABLE_XET"] == "1"
    session = factories[0]()
    try:
        adapter = session.adapters["https://"]
        direct = adapter.poolmanager.connection_pool_kw["ssl_context"]
        proxy = adapter.proxy_manager_for("http://proxy.invalid:9981").connection_pool_kw[
            "ssl_context"
        ]
        assert session.verify is True
        assert direct is proxy
        assert direct.maximum_version == ssl.TLSVersion.TLSv1_2
        assert direct.verify_mode == ssl.CERT_REQUIRED
        assert direct.check_hostname is True
    finally:
        session.close()


def test_invalid_transport_mode_fails_before_download(monkeypatch):
    monkeypatch.setenv("PEP_HF_TLS12", "invalid")
    with pytest.raises(models.ModelError, match="must be 0 or 1"):
        models.configure_download_http()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def paper(tmp_path, monkeypatch):
    source = tmp_path / "originals"
    lock = {"schema_version": 1, "reference_environment": {}, "models": []}
    for key in sorted(models.MODEL_KEYS):
        root = source / key
        root.mkdir(parents=True)
        write_json(root / "config.json", {"model": key})
        (root / "vocab.txt").write_text("ACDEFGHIK", encoding="utf-8")
        if key == "prott5":
            shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
            for name in shards:
                (root / name).write_bytes((key + name).encode())
            write_json(
                root / "model.safetensors.index.json", {"weight_map": dict(zip("ab", shards))}
            )
            weight_files = ["model.safetensors.index.json", *shards]
        else:
            (root / "model.safetensors").write_bytes((key + "-weights").encode())
            weight_files = ["model.safetensors"]
        files = {
            p.name: {"size_bytes": p.stat().st_size, "sha256": models.sha256_file(p)}
            for p in root.iterdir()
        }
        model = {
            "key": key,
            "tokenizer": "esmc" if key.startswith("esmc-") else key,
            "repo_id": "paper/" + key,
            "revision": None if key == "prott5" else "a" * 40,
            "origin": "paper-local" if key == "prott5" else "huggingface",
            "files": files,
            "weight_files": weight_files,
            "model_fingerprint": models.fingerprint(files),
        }
        model["weight_sha256"] = models.weight_digest(model, files)
        lock["models"].append(model)
    cohort = {"candidates": []}
    index = {"passed": True, "counts": {"pep_function": 6}, "candidate_count": 6, "candidates": []}
    extras = []
    for i, model in enumerate(lock["models"]):
        candidate = f"c{i:03d}"
        artifact = tmp_path / "artifacts" / candidate
        identity = {
            "model_id": model["repo_id"],
            "weight_sha256": model["weight_sha256"],
            "model_fingerprint": model["model_fingerprint"],
            "snapshot_revision": model["revision"] or "",
            "weight_path": str(source / model["key"] / model["weight_files"][0]),
        }
        write_json(artifact / "metadata.json", {"pretrained": {"model_identity": identity}})
        sha = models.sha256_file(artifact / "metadata.json")
        evidence = {"task": "pep_function", "candidate": candidate, "metadata_sha256": sha}
        model["evidence"] = [evidence]
        cohort["candidates"].append(
            {"task": "pep_function", "candidate": candidate, "artifact": str(artifact)}
        )
        index["candidates"].append(
            {
                **evidence,
                "artifact": str(artifact),
                "passed": True,
                "errors": [],
                "policy": {"passed": True, "errors": []},
                "tests": [],
            }
        )
        extras.append({"task": "pep_function", "candidate": candidate})
    write_json(tmp_path / "cohort.json", cohort)
    protocol = {
        "manifest": str(tmp_path / "cohort.json"),
        "manifest_sha256": models.sha256_file(tmp_path / "cohort.json"),
    }
    write_json(tmp_path / "completion_protocol.json", protocol)
    index["protocol_sha256"] = models.sha256_file(tmp_path / "completion_protocol.json")
    write_json(tmp_path / "results_index.json", index)
    registration = {
        "results_id": "test-paper",
        "results_index": {
            "path": str(tmp_path / "results_index.json"),
            "sha256": models.sha256_file(tmp_path / "results_index.json"),
        },
        "plot_inputs": extras,
    }
    reference = tmp_path / "current_results.json"
    write_json(reference, registration)
    lock["reference"] = {
        "results_id": "test-paper",
        "registration_sha256": models.sha256_file(reference),
        "results_index_sha256": registration["results_index"]["sha256"],
        "protocol_sha256": index["protocol_sha256"],
        "cohort_sha256": protocol["manifest_sha256"],
        "candidate_count": 6,
        "counts": index["counts"],
    }
    lock_path = tmp_path / "models.lock.json"
    write_json(lock_path, lock)
    monkeypatch.setattr(models, "LOCK_PATH", lock_path)
    return SimpleNamespace(
        lock=lock,
        source=source,
        reference=reference,
        root=tmp_path / "installed",
        bundle=tmp_path / "portable",
    )


def no_network(_model):
    pytest.fail("Import and verification must not download anything")


def test_release_lock_pins_paper_assets_not_only_current_defaults():
    lock = models.load_lock()
    by_key = {m["key"]: m for m in lock["models"]}
    assert by_key["prott5"]["origin"] == "paper-local"
    assert by_key["prott5"]["revision"] is None
    assert len(by_key["prott5"]["weight_files"]) == 7
    assert (
        by_key["prott5"]["weight_sha256"]
        == "c1e7b7328276917e729bf11585ac6b4b7965aa33251edc882ee2fad7d47d307d"
    )
    assert by_key["esmc-300m"]["revision"] == "7f10b20ae75017b2dbc884070e03434515709a8d"
    assert by_key["esmc-600m"]["revision"] == "e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c"
    assert lock["reference"]["candidate_count"] == 100


def test_export_import_is_portable_offline_and_idempotent(paper):
    models.export_bundle(paper.bundle, paper.reference, paper.lock)
    # The exported bundle must survive removal/relocation of the original files.
    paper.source.rename(paper.source.with_name("originals-relocated"))
    models.prepare(paper.root, paper.lock, bundle=paper.bundle, downloader=no_network)
    before = {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in paper.root.rglob("*.safetensors")}
    models.prepare(paper.root, paper.lock, bundle=paper.bundle, downloader=no_network)
    models.prepare(paper.root, paper.lock, check_only=True, downloader=no_network)
    assert before == {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in before}
    for model in paper.lock["models"]:
        root = models.installed_model(paper.root, model)
        assert root.is_dir()
        if model["revision"]:
            assert root.parts[-2:] == ("snapshots", model["revision"])
    assert not any(p.is_symlink() for p in paper.bundle.rglob("*"))


def test_same_size_corruption_fails_and_reimport_repairs_only_installed_copy(paper):
    models.export_bundle(paper.bundle, paper.reference, paper.lock)
    models.prepare(paper.root, paper.lock, bundle=paper.bundle)
    model = paper.lock["models"][0]
    target = models.installed_path(paper.root, model) / model["weight_files"][0]
    target.write_bytes(b"x" * target.stat().st_size)
    with pytest.raises(models.ModelError, match="Preparation incomplete"):
        models.prepare(paper.root, paper.lock, check_only=True)
    models.prepare(paper.root, paper.lock, bundle=paper.bundle)
    models.verify_tree(paper.source / model["key"], model)
    models.installed_model(paper.root, model)
    assert list(target.parent.parent.glob("*.previous-*"))


def test_missing_prott5_shard_and_changed_package_lock_are_rejected(paper):
    models.export_bundle(paper.bundle, paper.reference, paper.lock)
    t5 = next(m for m in paper.lock["models"] if m["key"] == "prott5")
    (paper.bundle / "models/prott5" / t5["weight_files"][-1]).unlink()
    with pytest.raises(models.ModelError, match="Preparation incomplete"):
        models.prepare(paper.root, paper.lock, bundle=paper.bundle)
    assert not (paper.root / "installed.lock.json").exists()
    (paper.bundle / "models.lock.json").write_text("{}", encoding="utf-8")
    with pytest.raises(models.ModelError, match="lock differs"):
        models.prepare(paper.root, paper.lock, bundle=paper.bundle)


def test_check_only_never_creates_an_installation(paper):
    with pytest.raises(models.ModelError, match="Preparation incomplete"):
        models.prepare(paper.root, paper.lock, check_only=True, downloader=no_network)
    assert not paper.root.exists()


def test_export_checks_reference_hashes_and_preserves_existing_packages(paper):
    models.export_bundle(paper.bundle, paper.reference, paper.lock)
    with pytest.raises(models.ModelError, match="new output directory"):
        models.export_bundle(paper.bundle, paper.reference, paper.lock)
    paper.reference.write_text("{}", encoding="utf-8")
    with pytest.raises(models.ModelError, match="reference hash mismatch"):
        models.export_bundle(paper.bundle.with_name("other"), paper.reference, paper.lock)
    assert not paper.bundle.with_name("other").exists()


def test_online_mode_rejects_unregistered_prott5_target(paper, capsys):
    def download(model):
        if model["key"] == "prott5":
            return models.download_model(model)
        return paper.source / model["key"]

    with pytest.raises(models.ModelError, match="Preparation incomplete: prott5"):
        models.prepare(paper.root, paper.lock, downloader=download)
    assert "restoration recipe does not match" in capsys.readouterr().err
    assert not (paper.root / "installed.lock.json").exists()


def test_download_uses_fixed_revision_mirror_and_repairs_only_bad_file(paper, monkeypatch):
    model = copy.deepcopy(paper.lock["models"][0])
    source = paper.source / model["key"]
    name = model["weight_files"][0]
    original = (source / name).read_bytes()
    (source / name).write_bytes(b"x" * len(original))
    calls = []

    def snapshot(**kwargs):
        calls.append(("snapshot", kwargs))
        return str(source)

    def repair(**kwargs):
        calls.append(("repair", kwargs))
        (source / name).write_bytes(original)

    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot, hf_hub_download=repair),
    )
    assert models.download_model(model) == source
    assert calls[0][1]["revision"] == model["revision"]
    assert calls[0][1]["endpoint"] == "https://mirror.example"
    assert calls[0][1]["allow_patterns"] == list(model["files"])
    assert calls[1][1]["filename"] == name and calls[1][1]["force_download"]
    assert len(calls) == 2


def test_errors_do_not_print_token(paper, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_private_test_token")

    def download(_model):
        raise OSError("request failed with hf_private_test_token")

    with pytest.raises(models.ModelError):
        models.prepare(paper.root, paper.lock, downloader=download)
    assert "hf_private_test_token" not in capsys.readouterr().err


@pytest.mark.parametrize("fail_builder", [False, True])
def test_local_esm3_auxiliary_data_uses_snapshot_and_restores_sdk_roots(
    tmp_path, monkeypatch, fail_builder
):
    import importlib

    import torch
    from esm import pretrained
    from esm.models.esm3 import ESM3

    from pepplatform.encoders import Esm3Encoder

    constants = importlib.import_module("esm.utils.constants.esm3")
    original_builder_root = pretrained.data_root
    original_tokenizer_root = constants.data_root
    root = tmp_path / "esm3" / "snapshots" / ("a" * 40)
    weights = root / "data/weights/esm3_sm_open_v1.pth"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fake checkpoint for SDK path test")

    class Client(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = SimpleNamespace(embed_dim=1536)

    def builder(_cls, name, device=None):
        assert name == "esm3_sm_open_v1"
        assert pretrained.data_root("esm3") == root
        assert constants.data_root("esm3") == root
        if fail_builder:
            raise RuntimeError("SDK builder failed")
        return Client()

    monkeypatch.setattr(ESM3, "from_pretrained", classmethod(builder))
    if fail_builder:
        with pytest.raises(RuntimeError, match="SDK builder failed"):
            Esm3Encoder(model_name=str(root), device="cpu")
    else:
        assert Esm3Encoder(model_name=str(root), device="cpu").output_dim == 1536
    assert pretrained.data_root is original_builder_root
    assert constants.data_root is original_tokenizer_root
