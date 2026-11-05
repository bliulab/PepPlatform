"""Canonical ProtBERT identity, source and representation contracts.

The public ``protbert`` route uses the Rostlab ProtBERT checkpoint through the
Hugging Face BERT implementation.  Architecture identity and checkpoint
provenance are kept separate so compatible external snapshots remain usable
without inheriting official-checkpoint claims.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from numbers import Integral
from pathlib import Path
from typing import Mapping

from pepplatform.cache_provenance import model_source_fingerprint, sha256_file


PROTBERT_OFFICIAL_MODEL_ID = "Rostlab/prot_bert"
# Reference snapshot for official-checkpoint provenance checks. Other snapshots
# remain external artifacts unless their revision and weight digest are verified.
PROTBERT_HF_REVISION = "7a894481acdc12202f0a415dd567f6cfdb698908"
PROTBERT_OFFICIAL_REVISION = PROTBERT_HF_REVISION
PROTBERT_OFFICIAL_WEIGHT_RELATIVE_PATH = Path("pytorch_model.bin")
PROTBERT_OFFICIAL_WEIGHT_SHA256 = "6ea3edd26cfefc3111176100ee2a027ed510f9c86b63e5b53a2a050b59f2af9d"

PROTBERT_MODEL_TYPE = "bert"
PROTBERT_HIDDEN_SIZE = 1024
PROTBERT_NUM_ATTENTION_HEADS = 16
PROTBERT_NUM_HIDDEN_LAYERS = 30
PROTBERT_INTERMEDIATE_SIZE = 4096
PROTBERT_VOCAB_SIZE = 30
PROTBERT_MAX_POSITION_EMBEDDINGS = 40000
PROTBERT_REPRESENTATION_CONTRACT_VERSION = 1
PROTBERT_IDENTITY_CONTRACT_VERSION = 1
PROTBERT_POOLING = "residue_mean"
PROTBERT_RESIDUE_OUTPUT_KIND = "residue_embeddings"
PROTBERT_TOKEN_OUTPUT_KIND = "residue_token_hidden"
PROTBERT_MAX_RESIDUES = PROTBERT_MAX_POSITION_EMBEDDINGS - 2
PROTBERT_VALID_SEQUENCE_AA = frozenset("ACDEFGHIKLMNPQRSTVWYX")
PROTBERT_REPLACED_RESIDUES = frozenset("UZOB")
PROTBERT_REQUIRED_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "vocab.txt",
)
PROTBERT_REQUIRED_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
PROTBERT_CONFIG_SIGNATURE = {
    "model_type": PROTBERT_MODEL_TYPE,
    "hidden_size": PROTBERT_HIDDEN_SIZE,
    "num_attention_heads": PROTBERT_NUM_ATTENTION_HEADS,
    "num_hidden_layers": PROTBERT_NUM_HIDDEN_LAYERS,
    "intermediate_size": PROTBERT_INTERMEDIATE_SIZE,
    "vocab_size": PROTBERT_VOCAB_SIZE,
    "max_position_embeddings": PROTBERT_MAX_POSITION_EMBEDDINGS,
}


def normalize_protbert_sequence(sequence: object) -> str:
    """Normalize a sequence using the ProtTrans residue-token contract."""

    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise ValueError("ProtBERT sequence must contain at least one residue")
    normalized = "".join(
        "X" if residue in PROTBERT_REPLACED_RESIDUES else residue for residue in normalized
    )
    invalid = sorted(set(normalized) - PROTBERT_VALID_SEQUENCE_AA)
    if invalid:
        raise ValueError("ProtBERT sequence contains unsupported residues: " + ", ".join(invalid))
    return normalized


def normalize_protbert_sequences(sequences) -> list[str]:
    return [normalize_protbert_sequence(value) for value in sequences]


def validate_protbert_max_residues(value: object, *, field: str = "max_length") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"ProtBERT {field} must be an integer")
    normalized = int(value)
    if normalized < 1 or normalized > PROTBERT_MAX_RESIDUES:
        raise ValueError(f"ProtBERT {field} must be between 1 and {PROTBERT_MAX_RESIDUES} residues")
    return normalized


def _expanded_path(source: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(str(source or "").strip()))


def model_snapshot_revision(source: str | os.PathLike[str]) -> str:
    parts = _expanded_path(source).parts
    if "snapshots" not in parts:
        return ""
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else ""


def _load_config(source: str | os.PathLike[str]) -> dict[str, object]:
    path = _expanded_path(source)
    config_path = path / "config.json" if path.is_dir() else Path()
    if not config_path.is_file():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def validate_protbert_config(config: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise ValueError("ProtBERT config must be a mapping")
    normalized = dict(config)
    architectures = normalized.get("architectures")
    if normalized.get("model_type") in (None, "") and isinstance(architectures, list):
        if any(name in {"BertModel", "BertForMaskedLM"} for name in architectures):
            normalized["model_type"] = PROTBERT_MODEL_TYPE
    mismatches = {}
    for key, expected in PROTBERT_CONFIG_SIGNATURE.items():
        observed = normalized.get(key)
        if observed != expected:
            mismatches[key] = {"expected": expected, "observed": observed}
    if mismatches:
        raise ValueError(
            "ProtBERT architecture signature mismatch: "
            + json.dumps(mismatches, ensure_ascii=True, sort_keys=True)
        )
    return normalized


def _has_any(directory: Path, filenames) -> bool:
    return any((directory / name).is_file() for name in filenames)


def is_complete_protbert_snapshot(source: str | os.PathLike[str]) -> bool:
    path = _expanded_path(source)
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    try:
        validate_protbert_config(_load_config(path))
    except ValueError:
        return False
    return _has_any(path, PROTBERT_REQUIRED_TOKENIZER_FILES) and _has_any(
        path, PROTBERT_REQUIRED_WEIGHT_FILES
    )


def _weight_path(source: str | os.PathLike[str]) -> Path | None:
    path = _expanded_path(source)
    if not path.is_dir():
        return None
    for name in PROTBERT_REQUIRED_WEIGHT_FILES:
        candidate = path / name
        if candidate.is_file() and not name.endswith(".index.json"):
            return candidate
    return None


def inspect_protbert_source(source: str | os.PathLike[str]) -> dict[str, object]:
    value = str(source or "").strip()
    path = _expanded_path(value)
    local = path.is_dir()
    config = _load_config(path) if local else {}
    revision = model_snapshot_revision(path)
    if not local and value == PROTBERT_OFFICIAL_MODEL_ID:
        revision = PROTBERT_HF_REVISION
    weight = _weight_path(path) if local else None
    complete = is_complete_protbert_snapshot(path) if local else False
    if local:
        fingerprint = model_source_fingerprint(value) if complete else ""
    else:
        fingerprint_source = f"{value}@{revision}" if revision else value
        fingerprint = model_source_fingerprint(fingerprint_source) if value else ""
    return {
        "source": value,
        "is_local": local,
        "complete": complete,
        "snapshot_revision": revision,
        "config": config,
        "weight_path": str(weight or ""),
        "weight_sha256": sha256_file(weight) if weight is not None else "",
        "model_fingerprint": fingerprint,
        "provenance_status": "external_artifact_unverified",
    }


def protbert_model_identity(
    source: str | os.PathLike[str],
    *,
    config: Mapping[str, object] | None = None,
    weight_path: str | os.PathLike[str] | None = None,
    weight_sha256: str = "",
    transformers_version: str = "",
) -> dict[str, object]:
    value = str(source or "").strip()
    observed = inspect_protbert_source(value)
    cfg = dict(config or observed.get("config") or {})
    validate_protbert_config(cfg)
    resolved_weight = _expanded_path(weight_path or observed.get("weight_path", ""))
    observed_sha = weight_sha256 or (
        sha256_file(resolved_weight)
        if resolved_weight.is_file()
        else str(observed.get("weight_sha256", ""))
    )
    pinned = (
        value == PROTBERT_OFFICIAL_MODEL_ID
        or observed.get("snapshot_revision") == PROTBERT_HF_REVISION
    )
    digest_matches = observed_sha == PROTBERT_OFFICIAL_WEIGHT_SHA256 if observed_sha else False
    provenance = (
        "official_artifact_verified"
        if pinned and digest_matches
        else "external_artifact_unverified"
    )
    try:
        runtime_version = transformers_version or importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        runtime_version = transformers_version
    return {
        "contract_version": PROTBERT_IDENTITY_CONTRACT_VERSION,
        "provider": "Rostlab/ProtTrans",
        "model_family": "ProtBERT",
        "model_id": PROTBERT_OFFICIAL_MODEL_ID,
        "requested_source": value,
        "resolved_source": value,
        "snapshot_revision": str(observed.get("snapshot_revision", "") or ""),
        "model_fingerprint": str(observed.get("model_fingerprint", "") or ""),
        "weight_path": str(resolved_weight) if resolved_weight.is_file() else "",
        "weight_size_bytes": resolved_weight.stat().st_size if resolved_weight.is_file() else 0,
        "weight_sha256": observed_sha,
        "provenance_status": provenance,
        "runtime_package": "transformers",
        "runtime_version": str(runtime_version or ""),
        "architecture": "bert_large_30_layer_protbert",
        **PROTBERT_CONFIG_SIGNATURE,
        "pooling": PROTBERT_POOLING,
        "output_kind": PROTBERT_RESIDUE_OUTPUT_KIND,
        "representation_contract_version": PROTBERT_REPRESENTATION_CONTRACT_VERSION,
        "max_residues": PROTBERT_MAX_RESIDUES,
        "special_tokens": "CLS/SEP/PAD excluded",
        "input_normalization": "strip_whitespace_uppercase_UZOB_to_X",
    }


def validate_protbert_model_identity(
    expected: Mapping[str, object] | None,
    observed: Mapping[str, object],
) -> None:
    if not isinstance(expected, Mapping) or not expected:
        return
    keys = (
        "provider",
        "model_id",
        "snapshot_revision",
        "weight_sha256",
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "pooling",
        "representation_contract_version",
    )
    mismatches = {
        key: {"expected": expected.get(key), "observed": observed.get(key)}
        for key in keys
        if expected.get(key) not in (None, "") and expected.get(key) != observed.get(key)
    }
    if mismatches:
        raise ValueError(
            "ProtBERT model identity differs from the training artifact: "
            + json.dumps(mismatches, ensure_ascii=True, sort_keys=True)
        )


def protbert_runtime_environment(device=None) -> dict[str, object]:
    import torch

    torch_device = torch.device(device if device is not None else "cpu")
    index = torch_device.index
    name = ""
    if torch_device.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if index is None else index
        name = torch.cuda.get_device_name(index)
    try:
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        transformers_version = ""
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers_version),
        "cuda_runtime_version": str(torch.version.cuda or ""),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": torch_device.type,
        "device_index": index,
        "device_name": name,
    }


def format_protbert_provenance(model_identity, runtime_environment) -> str:
    return "[protbert-provenance] " + json.dumps(
        {
            "model_identity": dict(model_identity),
            "runtime_environment": dict(runtime_environment),
        },
        ensure_ascii=True,
        sort_keys=True,
    )
