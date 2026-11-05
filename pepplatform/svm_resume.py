"""Versioned, integrity-checked resume bundles for downstream SVM training."""

from __future__ import annotations
from pepplatform.task_compat import normalize_task_type


import hashlib
import json
import os
import shutil
from pathlib import Path

import torch


SVM_RESUME_SCHEMA_VERSION = 1
SVM_RESUME_STAGE = "downstream_svm"
SVM_RESUME_RELATIVE_DIR = Path("resume") / "downstream_svm_v1"
SVM_RESUME_TENSORS = "embeddings.pt"
SVM_RESUME_MANIFEST = "manifest.json"

SVM_DOWNSTREAM_OVERRIDE_FIELDS = frozenset(
    {
        "svm_c",
        "svm_class_weight",
        "svm_max_iter",
        "svm_tol",
        "svm_kernel",
        "svm_gamma",
        "svm_convergence_policy",
        "ml_scaler",
        "ml_reduction",
        "ml_reduction_dim",
        "pca_dense_memory_limit_gib",
        "kpca_memory_limit_gib",
        "kpca_kernel",
        "kpca_gamma",
        "ml_auto_tune",
        "ml_tuning_budget",
        "ml_tuning_objective",
        "ml_tuning_mode",
        "ml_tuning_max_candidates",
        "ml_cv_folds",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_description(value):
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def save_svm_resume_bundle(
    artifact_dir,
    *,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x=None,
    test_y=None,
    task_type="",
    artifact_source="",
    artifact_info=None,
):
    task_type = normalize_task_type(task_type)
    root = Path(artifact_dir) / SVM_RESUME_RELATIVE_DIR
    root.mkdir(parents=True, exist_ok=True)
    tensors = {
        "train_x": torch.as_tensor(train_x).detach().cpu(),
        "train_y": torch.as_tensor(train_y).detach().cpu(),
        "val_x": torch.as_tensor(val_x).detach().cpu(),
        "val_y": torch.as_tensor(val_y).detach().cpu(),
        "test_x": None if test_x is None else torch.as_tensor(test_x).detach().cpu(),
        "test_y": None if test_y is None else torch.as_tensor(test_y).detach().cpu(),
    }
    tensor_path = root / SVM_RESUME_TENSORS
    torch.save(tensors, tensor_path)
    info = dict(artifact_info or {})
    safe_artifact_info = {
        key: value
        for key, value in info.items()
        if key != "dl_scaler" and isinstance(value, (str, int, float, bool, list, dict, type(None)))
    }
    manifest = {
        "schema_version": SVM_RESUME_SCHEMA_VERSION,
        "stage": SVM_RESUME_STAGE,
        "task_type": str(task_type or ""),
        "artifact_source": str(artifact_source or ""),
        "artifact_info": safe_artifact_info,
        "tensor_file": SVM_RESUME_TENSORS,
        "tensor_sha256": _sha256(tensor_path),
        "tensors": {key: _tensor_description(value) for key, value in tensors.items()},
    }
    manifest_path = root / SVM_RESUME_MANIFEST
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[resume] saved downstream SVM bundle: {root} sha256={manifest['tensor_sha256']}",
        flush=True,
    )
    return manifest


def inspect_svm_resume_bundle(artifact_dir, *, verify_integrity=True):
    root = Path(artifact_dir) / SVM_RESUME_RELATIVE_DIR
    manifest_path = root / SVM_RESUME_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SVM_RESUME_SCHEMA_VERSION
        or manifest.get("stage") != SVM_RESUME_STAGE
    ):
        return None
    tensor_path = root / str(manifest.get("tensor_file") or "")
    if not tensor_path.is_file():
        return None
    expected = str(manifest.get("tensor_sha256") or "")
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.lower()):
        return None
    if verify_integrity and _sha256(tensor_path) != expected:
        return None
    return {
        **manifest,
        "root": str(root),
        "manifest_path": str(manifest_path),
        "integrity_verified": bool(verify_integrity),
    }


def load_svm_resume_bundle(bundle_path):
    supplied = Path(bundle_path)
    artifact_dir = supplied
    if supplied.name == SVM_RESUME_RELATIVE_DIR.name:
        artifact_dir = supplied.parent.parent
    manifest = inspect_svm_resume_bundle(artifact_dir)
    if not manifest:
        raise ValueError("downstream SVM resume bundle is missing or failed integrity checks")
    tensor_path = Path(manifest["root"]) / str(manifest["tensor_file"])
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    required = {"train_x", "train_y", "val_x", "val_y", "test_x", "test_y"}
    if not isinstance(tensors, dict) or set(tensors) != required:
        raise ValueError("downstream SVM resume tensor payload is invalid")
    return manifest, tensors


def materialize_svm_resume_bundle(source_artifact_dir, destination_artifact_dir):
    source = Path(source_artifact_dir)
    manifest = inspect_svm_resume_bundle(source)
    if not manifest:
        raise ValueError("source experiment has no valid downstream SVM resume bundle")
    destination = Path(destination_artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target_root = destination / SVM_RESUME_RELATIVE_DIR
    target_root.mkdir(parents=True, exist_ok=True)
    for filename in (SVM_RESUME_TENSORS, SVM_RESUME_MANIFEST):
        source_path = Path(manifest["root"]) / filename
        target_path = target_root / filename
        try:
            os.link(source_path, target_path)
        except OSError:
            shutil.copy2(source_path, target_path)
    encoder = source / "embedding_encoder.pt"
    if encoder.is_file():
        target_encoder = destination / encoder.name
        try:
            os.link(encoder, target_encoder)
        except OSError:
            shutil.copy2(encoder, target_encoder)
    return inspect_svm_resume_bundle(destination)


def remove_svm_resume_bundle(artifact_dir):
    root = Path(artifact_dir) / SVM_RESUME_RELATIVE_DIR
    if root.is_dir():
        shutil.rmtree(root)
    resume_root = root.parent
    if resume_root.is_dir() and not any(resume_root.iterdir()):
        resume_root.rmdir()
