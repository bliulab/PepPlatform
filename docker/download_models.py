"""Prepare the exact pretrained assets used by the registered paper candidates.

Normal export/import/check operations need only Python's standard library.
Hugging Face and model libraries are imported only for downloads, restoration or smoke tests.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path(__file__).with_name("models.lock.json")
PROTT5_RECIPE_PATH = Path(__file__).with_name("prott5.restore.json")
PROTT5_RECIPE_SHA256 = "37b912f08cf6b8a12dba725123946c14c849528627411ff32ecb1759d88b75bc"
IDENTITY_SUFFIXES = {
    ".bin",
    ".json",
    ".model",
    ".pt",
    ".pth",
    ".safetensors",
    ".spm",
    ".txt",
    ".vocab",
}
MODEL_KEYS = {"esm2", "protbert", "prott5", "esmc-300m", "esmc-600m", "esm3"}


class ModelError(ValueError):
    """An asset does not match the frozen deployment contract."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ModelError(f"Unsafe model-relative path: {value}")
    return path


def fingerprint(files):
    digest = hashlib.sha256()
    for name, record in sorted(files.items()):
        if PurePosixPath(name).suffix.lower() in IDENTITY_SUFFIXES:
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(record["sha256"]))
    return "tree:" + digest.hexdigest()


def weight_digest(model, files):
    if model["key"] != "prott5":
        return files[model["weight_files"][0]]["sha256"]
    digest = hashlib.sha256()
    for name in model["weight_files"]:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(files[name]["sha256"]))
        digest.update(b"\0")
    return digest.hexdigest()


def load_lock():
    lock = read_json(LOCK_PATH)
    if lock.get("schema_version") != 1:
        raise ModelError("Unsupported model lock version")
    if {model["key"] for model in lock["models"]} != MODEL_KEYS or len(lock["models"]) != 6:
        raise ModelError("Model lock must contain all six paper checkpoints")
    for model in lock["models"]:
        revision = model["revision"]
        if model["origin"] == "huggingface" and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ModelError(f"{model['key']}: download requires an immutable revision")
        for name, record in model["files"].items():
            safe_relative(name)
            if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) or record["size_bytes"] < 0:
                raise ModelError(f"{model['key']}: invalid file identity")
        if fingerprint(model["files"]) != model["model_fingerprint"]:
            raise ModelError(f"{model['key']}: inconsistent tree fingerprint in lock")
        if weight_digest(model, model["files"]) != model["weight_sha256"]:
            raise ModelError(f"{model['key']}: inconsistent weight identity in lock")
    return lock


def inventory(root):
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(root).parts
        and not path.name.endswith((".lock", ".incomplete"))
    }


def verify_tree(root, model):
    root = Path(root)
    expected = set(model["files"])
    actual = inventory(root) if root.is_dir() else set()
    if actual != expected:
        raise ModelError(
            f"{model['key']}: file set differs; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    for name, record in model["files"].items():
        path = root / name
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ModelError(f"{model['key']}: size/SHA256 mismatch: {name}")
    # An index alone cannot prove that all of its tensor shards are present.
    if model["key"] == "prott5":
        index = read_json(root / "model.safetensors.index.json")
        shards = sorted(set(index["weight_map"].values()))
        if model["weight_files"] != ["model.safetensors.index.json", *shards]:
            raise ModelError("prott5: shard index disagrees with the frozen weight list")
    return root


def identities(value):
    if isinstance(value, dict):
        if isinstance(value.get("model_identity"), dict):
            yield value["model_identity"]
        for key, child in value.items():
            if key != "args":
                yield from identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from identities(child)


def verified_json(path, expected):
    if sha256_file(path) != expected:
        raise ModelError(f"Frozen reference hash mismatch: {path}")
    return read_json(path)


def reference_sources(reference, lock):
    """Follow the accepted index, never discover experiments by mtime or names."""
    expected = lock["reference"]
    registration = verified_json(reference, expected["registration_sha256"])
    if registration["results_id"] != expected["results_id"]:
        raise ModelError("Wrong registered paper results")
    declared = registration["results_index"]
    if declared["sha256"] != expected["results_index_sha256"]:
        raise ModelError("Wrong accepted results index")
    index_path = Path(declared["path"])
    index = verified_json(index_path, declared["sha256"])
    if (
        not index.get("passed")
        or index["counts"] != expected["counts"]
        or index["candidate_count"] != expected["candidate_count"]
        or len(index["candidates"]) != expected["candidate_count"]
    ):
        raise ModelError("Incomplete accepted candidate cohort")
    if index["protocol_sha256"] != expected["protocol_sha256"]:
        raise ModelError("Wrong completion protocol")
    protocol = verified_json(
        index_path.parent / "completion_protocol.json", expected["protocol_sha256"]
    )
    if protocol["manifest_sha256"] != expected["cohort_sha256"]:
        raise ModelError("Wrong frozen cohort manifest")
    cohort = verified_json(protocol["manifest"], expected["cohort_sha256"])
    originals = {(row["task"], row["candidate"]): row for row in cohort["candidates"]}
    rows = {(row["task"], row["candidate"]): row for row in index["candidates"]}
    extras = {(row["task"], row["candidate"]): row for row in registration["plot_inputs"]}
    if (
        len(rows) != expected["candidate_count"]
        or set(rows) != set(originals)
        or set(rows) != set(extras)
    ):
        raise ModelError("Accepted, frozen and registered candidate identities differ")
    models = {(m["repo_id"], m["weight_sha256"]): m for m in lock["models"]}
    roots = {m["key"]: set() for m in lock["models"]}
    observed_evidence = {m["key"]: [] for m in lock["models"]}
    for key, row in rows.items():
        if (
            not row.get("passed")
            or row.get("errors")
            or not row["policy"].get("passed")
            or row["policy"].get("errors")
            or any(t.get("errors") for t in row["tests"])
        ):
            raise ModelError(f"Unaccepted paper candidate: {key}")
        source = row.get("replacement") or originals[key]
        if Path(source["artifact"]).resolve() != Path(row["artifact"]).resolve():
            raise ModelError(f"Accepted artifact mapping differs: {key}")
        records = [(Path(row["artifact"]) / "metadata.json", row["metadata_sha256"])]
        records.extend(
            (Path(m["path"]), m["sha256"]) for m in extras[key].get("member_metadata", [])
        )
        for path, digest in records:
            metadata = verified_json(path, digest)
            for identity in identities(metadata):
                model = models.get((identity.get("model_id"), identity.get("weight_sha256")))
                if model is None:
                    raise ModelError(f"Unregistered pretrained identity in {key}")
                name = model["key"]
                if (
                    identity.get("model_fingerprint")
                    and identity["model_fingerprint"] != model["model_fingerprint"]
                ):
                    raise ModelError(f"{name}: frozen content fingerprints disagree")
                if identity.get("snapshot_revision") not in (None, "", model["revision"]):
                    raise ModelError(f"{name}: frozen revisions disagree")
                weight = Path(identity["weight_path"])
                root = weight.parents[2] if "/data/weights/" in weight.as_posix() else weight.parent
                roots[name].add(root.resolve())
                evidence = {"task": key[0], "candidate": key[1], "metadata_sha256": digest}
                if evidence not in observed_evidence[name]:
                    observed_evidence[name].append(evidence)
    for model in lock["models"]:
        name = model["key"]
        evidence_key = lambda item: (item["task"], item["candidate"], item["metadata_sha256"])
        if not roots[name] or sorted(observed_evidence[name], key=evidence_key) != sorted(
            model["evidence"], key=evidence_key
        ):
            raise ModelError(f"{name}: model evidence does not cover the locked cohort")
    return {name: min(paths) for name, paths in roots.items()}


def require_space(parent, size):
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < size + 256 * 1024 * 1024:
        raise ModelError(
            f"Insufficient free space: need {size / 1e9:.2f} GB plus temporary headroom"
        )


def copy_model(source, destination, model):
    destination.mkdir(parents=True, exist_ok=True)
    for name in model["files"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)  # Materialize links into portable regular files.


def load_prott5_recipe(model):
    if sha256_file(PROTT5_RECIPE_PATH) != PROTT5_RECIPE_SHA256:
        raise ModelError("ProtT5 restoration recipe SHA256 mismatch")
    recipe = read_json(PROTT5_RECIPE_PATH)
    if (
        recipe.get("schema_version") != 1
        or model["key"] != "prott5"
        or recipe["source"]["repo_id"] != model["repo_id"]
        or recipe["target"]["weight_sha256"] != model["weight_sha256"]
        or recipe["target"]["model_fingerprint"] != model["model_fingerprint"]
        or not re.fullmatch(r"[0-9a-f]{40}", recipe["source"]["revision"])
    ):
        raise ModelError("ProtT5 restoration recipe does not match the frozen model lock")
    source = recipe["source"]
    for name, record in source["files"].items():
        safe_relative(name)
        if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) or record["size_bytes"] < 0:
            raise ModelError("ProtT5 restoration recipe has an invalid source identity")
    if source["weight_file"] not in source["files"]:
        raise ModelError("ProtT5 restoration recipe is missing the official weight identity")
    groups = [set(recipe[k]) for k in ("copied_files", "text_files", "shard_headers")]
    if (
        set.union(*groups) != set(model["files"])
        or any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :])
        or groups[2] != {name for name in model["weight_files"] if name.endswith(".safetensors")}
    ):
        raise ModelError("ProtT5 restoration recipe does not cover the exact target file set")
    for name in recipe["copied_files"]:
        if source["files"].get(name) != model["files"][name]:
            raise ModelError(f"ProtT5 restoration recipe changes a tokenizer file: {name}")
    return recipe


def verify_source_files(root, files):
    root = Path(root)
    for name, record in files.items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ModelError(f"ProtT5 official source size/SHA256 mismatch: {name}")


def restore_prott5(source, destination, model):
    """Reconstruct the exact paper files, including their original header bytes.

    Each source tensor is materialized in contiguous order separately. Some
    upstream tensors are stored as transposed views, so streaming raw storages
    would silently change the checkpoint. No model is instantiated here.
    """
    recipe = load_prott5_recipe(model)
    source, destination = Path(source), Path(destination)
    verify_source_files(source, recipe["source"]["files"])
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    try:
        import torch
    except ImportError as exc:
        raise ModelError(
            "ProtT5 restoration requires the pretrained dependencies (including torch)"
        ) from exc
    if sys.byteorder != "little":
        raise ModelError("ProtT5 restoration requires a little-endian host")
    state = torch.load(
        source / recipe["source"]["weight_file"],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(state, dict) or not all(isinstance(t, torch.Tensor) for t in state.values()):
        raise ModelError("ProtT5 official weights must be a tensor state dictionary")
    destination.mkdir(parents=True, exist_ok=True)
    index = json.loads(recipe["text_files"]["model.safetensors.index.json"])["weight_map"]
    seen = set()
    for name, text in recipe["shard_headers"].items():
        header = text.encode("utf-8")
        layout = json.loads(text)
        tensors = {key: value for key, value in layout.items() if key != "__metadata__"}
        if set(tensors) != {key for key, filename in index.items() if filename == name}:
            raise ModelError(f"ProtT5 restoration header disagrees with the index: {name}")
        print(f"[prott5] restoring original file {name}", flush=True)
        path = destination / name
        with path.open("wb") as stream:
            stream.write(struct.pack("<Q", len(header)))
            stream.write(header)
            position = 0
            for key, entry in sorted(tensors.items(), key=lambda row: row[1]["data_offsets"][0]):
                tensor = state.get(key)
                if (
                    not isinstance(tensor, torch.Tensor)
                    or entry["dtype"] != "F32"
                    or tensor.dtype != torch.float32
                    or list(tensor.shape) != entry["shape"]
                    or key in seen
                ):
                    raise ModelError(f"ProtT5 restoration tensor name/shape/dtype mismatch: {key}")
                start, end = entry["data_offsets"]
                if start != position or end - start != tensor.numel() * tensor.element_size():
                    raise ModelError(f"ProtT5 restoration tensor offsets are invalid: {key}")
                contiguous = tensor.contiguous()
                payload = memoryview(contiguous.reshape(-1).view(torch.uint8).numpy())
                for offset in range(0, payload.nbytes, 8 * 1024 * 1024):
                    stream.write(payload[offset : offset + 8 * 1024 * 1024])
                del payload, contiguous
                position = end
                seen.add(key)
        if (
            path.stat().st_size != model["files"][name]["size_bytes"]
            or sha256_file(path) != model["files"][name]["sha256"]
        ):
            raise ModelError(f"ProtT5 restored file size/SHA256 mismatch: {name}")
    if seen != set(index):
        raise ModelError("ProtT5 restoration did not cover the complete tensor index")
    del state
    for name, text in recipe["text_files"].items():
        (destination / name).write_bytes(text.encode("utf-8"))
    for name in recipe["copied_files"]:
        shutil.copyfile(source / name, destination / name)


def export_bundle(destination, reference, lock):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ModelError("Export requires a new output directory; existing packages are preserved")
    sources = reference_sources(reference, lock)
    total = sum(f["size_bytes"] for m in lock["models"] for f in m["files"].values())
    require_space(destination.parent, total)
    with tempfile.TemporaryDirectory(prefix=".model-export-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        for model in lock["models"]:
            print(f"[{model['key']}] verifying frozen source", flush=True)
            source = verify_tree(sources[model["key"]], model)
            target = staging / "models" / model["key"]
            print(f"[{model['key']}] exporting original files", flush=True)
            copy_model(source, target, model)
            verify_tree(target, model)
        shutil.copyfile(LOCK_PATH, staging / "models.lock.json")
        receipt = {
            "schema_version": 1,
            "lock_sha256": sha256_file(LOCK_PATH),
            "results_id": lock["reference"]["results_id"],
        }
        (staging / "bundle.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        staging.rename(destination)
    print(f"Exported six verified checkpoints: {destination}", flush=True)


def installed_path(root, model):
    if model["revision"]:
        return root / ".versions" / model["key"] / "snapshots" / model["revision"]
    return root / ".versions" / model["key"] / "paper" / model["weight_sha256"]


def installed_model(root, model):
    target = installed_path(root, model)
    alias = root / "models" / model["key"]
    if not alias.is_symlink() or alias.resolve() != target.resolve():
        raise ModelError(f"{model['key']}: missing or incorrect installed model link")
    return verify_tree(target, model)


def install_model(source, root, model, *, restore=False):
    try:
        installed_model(root, model)
        print(f"[{model['key']}] verified; reusing installed files", flush=True)
        return
    except ModelError:
        pass
    target = installed_path(root, model)
    alias = root / "models" / model["key"]
    if alias.exists() and not alias.is_symlink():
        raise ModelError(f"Refusing to replace an unmanaged model directory: {alias}")
    require_space(root, sum(f["size_bytes"] for f in model["files"].values()))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model-install-", dir=target.parent) as temporary:
        staging = Path(temporary) / "model"
        if restore:
            restore_prott5(source, staging, model)
        else:
            copy_model(source, staging, model)
        verify_tree(staging, model)
        if target.exists():
            previous = target.with_name(target.name + ".previous-" + uuid.uuid4().hex)
            target.rename(previous)
            print(f"[{model['key']}] preserved previous files at {previous}", flush=True)
        staging.rename(target)
    alias.parent.mkdir(parents=True, exist_ok=True)
    temporary_alias = alias.with_name("." + alias.name + "-" + uuid.uuid4().hex)
    temporary_alias.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=True)
    temporary_alias.replace(alias)
    print(f"[{model['key']}] installed and verified", flush=True)


def validate_bundle(source, lock):
    if sha256_file(source / "models.lock.json") != sha256_file(LOCK_PATH):
        raise ModelError("Model package lock differs from this application release")
    receipt = read_json(source / "bundle.json")
    if receipt != {
        "schema_version": 1,
        "lock_sha256": sha256_file(LOCK_PATH),
        "results_id": lock["reference"]["results_id"],
    }:
        raise ModelError("Model package receipt differs from the frozen deployment contract")


def configure_download_http():
    """Optionally use verified TLS 1.2 HTTP downloads for incompatible proxies."""
    mode = os.environ.get("PEP_HF_TLS12", "0")
    if mode not in {"0", "1"}:
        raise ModelError("PEP_HF_TLS12 must be 0 or 1")
    if mode == "0":
        return False
    # Xet has its own HTTP client and cannot use this requests SSL context.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import ssl

    import requests
    from huggingface_hub import configure_http_backend
    from requests.adapters import HTTPAdapter

    def backend_factory():
        context = ssl.create_default_context()
        context.maximum_version = ssl.TLSVersion.TLSv1_2

        class TLS12Adapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                kwargs["ssl_context"] = context
                return super().init_poolmanager(*args, **kwargs)

            def proxy_manager_for(self, *args, **kwargs):
                kwargs["ssl_context"] = context
                return super().proxy_manager_for(*args, **kwargs)

        session = requests.Session()
        session.mount("https://", TLS12Adapter())
        return session

    configure_http_backend(backend_factory=backend_factory)
    return True


def download_model(model):
    endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    if model["key"] == "prott5":
        upstream = load_prott5_recipe(model)["source"]
    elif model["origin"] == "huggingface":
        upstream = model
    else:
        raise ModelError(f"{model['key']}: unsupported download origin")
    configure_download_http()
    from huggingface_hub import hf_hub_download, snapshot_download

    print(f"[{model['key']}] downloading fixed revision {upstream['revision']}", flush=True)
    # The library retains resumable downloads and reads HF_TOKEN/HF_HOME from the environment.
    # No fallback to another endpoint, repository, weight format or revision is allowed.
    source = Path(
        snapshot_download(
            repo_id=upstream["repo_id"],
            revision=upstream["revision"],
            allow_patterns=list(upstream["files"]),
            endpoint=endpoint,
        )
    )
    for name, record in upstream["files"].items():
        path = source / name
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            hf_hub_download(
                repo_id=upstream["repo_id"],
                revision=upstream["revision"],
                filename=name,
                endpoint=endpoint,
                force_download=True,
            )
    return source


def prepare(
    root, lock, bundle=None, check_only=False, downloader=download_model, prott5_source=None
):
    if prott5_source is not None and (bundle is not None or check_only):
        raise ModelError("--prott5-source is only available in normal preparation mode")
    if bundle is not None:
        validate_bundle(bundle, lock)
    failures = []
    for model in lock["models"]:
        key = model["key"]
        try:
            if check_only:
                installed_model(root, model)
                print(f"[{key}] OK: all files match the paper lock", flush=True)
                continue
            if bundle is not None:
                print(f"[{key}] checking original model package", flush=True)
                verify_tree(bundle / "models" / key, model)
            try:
                installed_model(root, model)
            except ModelError:
                require_space(root, sum(f["size_bytes"] for f in model["files"].values()))
                restore = bundle is None and key == "prott5"
                if bundle is not None:
                    source = bundle / "models" / key
                elif restore and prott5_source is not None:
                    source = prott5_source
                else:
                    source = downloader(model)
                install_model(source, root, model, restore=restore)
            else:
                print(f"[{key}] verified; reusing installed files", flush=True)
        except Exception as exc:  # noqa: BLE001 -- report all six model outcomes before failing.
            # HF errors can include signed URLs or credentials: report their class, not a traceback.
            message = str(exc) if isinstance(exc, (ModelError, OSError)) else type(exc).__name__
            token = os.environ.get("HF_TOKEN")
            if token:
                message = message.replace(token, "[redacted]")
            failures.append(key)
            print(f"[{key}] FAILED: {message}", file=sys.stderr, flush=True)
    if failures:
        raise ModelError("Preparation incomplete: " + ", ".join(failures))
    if not check_only:
        shutil.copyfile(LOCK_PATH, root / "installed.lock.json")
    print("All six paper checkpoints verified (five model families).", flush=True)


def smoke_test(root, lock, device=None):
    # Set offline flags before any model library import; also deny accidental SDK network calls.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PEPPLATFORM_OFFLINE_MODE="1")
    sys.path.insert(0, str(PROJECT_ROOT))
    import torch

    from pepplatform import encoders, tokenizers
    from pepplatform.workflow import resolve_pretrained_model_source

    channel = os.environ.get("PEP_TORCH_CHANNEL") or ("cu121" if torch.version.cuda else "cpu")
    selected = device or ("cuda" if os.environ.get("PEP_TORCH_CHANNEL") == "cu121" else "cpu")
    if channel not in {"cu121", "cpu"}:
        raise ModelError(f"Unsupported Torch channel: {channel}")
    spec = importlib.util.spec_from_file_location(
        "paper_runtime_check", Path(__file__).with_name("check_runtime.py")
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    report = checker.inspect_runtime(channel, selected)
    print("Runtime validation: " + json.dumps(report, sort_keys=True), flush=True)
    if report["status"] != "PASS":
        raise ModelError("Runtime/device validation failed: " + json.dumps(report["mismatches"]))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    classes = {
        "esm2": (encoders.EsmEncoder, tokenizers.EsmTokenizer),
        "protbert": (encoders.ProtBertEncoder, tokenizers.ProtBertTokenizer),
        "prott5": (encoders.ProT5Encoder, tokenizers.ProT5Tokenizer),
        "esmc": (encoders.EsmcEncoder, tokenizers.EsmcTokenizer),
        "esm3": (encoders.Esm3Encoder, tokenizers.Esm3Tokenizer),
    }
    versions = {}
    for package in lock["reference_environment"]:
        versions[package] = importlib.metadata.version(package)
    print("Installed runtime: " + json.dumps(versions, sort_keys=True), flush=True)
    print(
        "Paper reference runtime: " + json.dumps(lock["reference_environment"], sort_keys=True),
        flush=True,
    )
    for model in lock["models"]:
        print(f"[{model['key']}] offline tokenizer/encoder smoke test on {selected}", flush=True)
        path = installed_path(root, model).resolve()
        tag = model["tokenizer"]
        source = resolve_pretrained_model_source(
            PROJECT_ROOT, str(path), [], tag, tag + "_model", tokenizer=tag
        )
        encoder_class, tokenizer_class = classes[tag]
        tokenizer = tokenizer_class(model_name=source, max_length=16)
        encoder = encoder_class(model_name=source, device=torch.device(selected))
        loaded_model = getattr(encoder, "model", None)
        if loaded_model is None:
            loaded_model = getattr(encoder, "client", None)
        devices = sorted({str(p.device) for p in loaded_model.parameters()})
        if not devices or any(torch.device(value).type != selected for value in devices):
            raise ModelError(f"{model['key']}: actual model devices differ: {devices}")
        if tag == "esm3":
            output = encoder.encode_sequences(["ACDEFGHIK"], max_length=16)
        else:
            output = encoder.encode(tokenizer.batch_tokenize(["ACDEFGHIK"]))
        identity = encoder.model_identity
        if identity["weight_sha256"] != model["weight_sha256"]:
            raise ModelError(f"{model['key']}: runtime loaded different pretrained weights")
        if (
            identity.get("model_fingerprint")
            and identity["model_fingerprint"] != model["model_fingerprint"]
        ):
            raise ModelError(f"{model['key']}: runtime content fingerprint differs")
        if tuple(output.shape) != (1, model["output_dim"]) or not torch.isfinite(output).all():
            raise ModelError(f"{model['key']}: invalid embedding shape or non-finite output")
        print(
            f"[{model['key']}] smoke test OK: {tuple(output.shape)}; model devices={devices}",
            flush=True,
        )
        del output, encoder, tokenizer, loaded_model
        gc.collect()
        if selected == "cuda":
            torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--export",
        type=Path,
        metavar="DIR",
        help="Export original files into a new portable model package",
    )
    modes.add_argument(
        "--import-from",
        type=Path,
        metavar="DIR",
        help="Import all six original checkpoints without network access",
    )
    modes.add_argument(
        "--check-only",
        action="store_true",
        help="Verify installed assets without downloading or changing them",
    )
    parser.add_argument(
        "--reference", type=Path, help="Frozen current_results.json; required for export"
    )
    parser.add_argument(
        "--prott5-source",
        type=Path,
        metavar="DIR",
        help="Use an existing fixed official ProtT5 snapshot without downloading ProtT5",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("PEP_RUNTIME_ROOT", "/var/lib/pepplatform")) / "pretrained",
        help="Installation root (normally inside the existing runtime volume)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="With --check-only, load each model sequentially offline on the selected device",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], help="With --smoke-test; defaults to the image profile"
    )
    args = parser.parse_args(argv)
    if bool(args.export) != bool(args.reference):
        parser.error("--export and --reference must be supplied together")
    if args.smoke_test and not args.check_only:
        parser.error("--smoke-test requires --check-only")
    if args.device is not None and not args.smoke_test:
        parser.error("--device requires --smoke-test")
    if args.prott5_source is not None and (args.export or args.import_from or args.check_only):
        parser.error("--prott5-source is only available in normal preparation mode")
    try:
        lock = load_lock()
        if args.export:
            export_bundle(args.export, args.reference, lock)
        else:
            prepare(
                args.root.absolute(),
                lock,
                bundle=args.import_from,
                check_only=args.check_only,
                prott5_source=args.prott5_source,
            )
            if args.smoke_test:
                smoke_test(args.root.absolute(), lock, device=args.device)
    except (ModelError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Model preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
