"""Canonical ESM-2 model identity, source and representation contracts.

The public ``esm2`` route uses the Meta ESM-2 transformer checkpoint through
the Hugging Face Transformers implementation.  This module deliberately keeps
architecture identity separate from checkpoint provenance: a compatible local
directory can be loaded, but it is not called an official artifact unless its
revision and content digest are both known.
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


ESM2_OFFICIAL_MODEL_ID = "facebook/esm2_t30_150M_UR50D"
ESM2_DEFAULT_MODEL_ID = ESM2_OFFICIAL_MODEL_ID
ESM2_HF_REVISION = "a695f6045e2e32885fa60af20c13cb35398ce30c"
ESM2_OFFICIAL_REVISION = ESM2_HF_REVISION
ESM2_OFFICIAL_WEIGHT_RELATIVE_PATH = Path("model.safetensors")
# Digest of the model.safetensors file in the pinned HF snapshot.
ESM2_OFFICIAL_WEIGHT_SHA256 = "c3f1da8aea53bddd32c246c86168c23b9fd72341fb9db9a94436f855f5053566"

ESM2_HIDDEN_SIZE = 640
ESM2_NUM_ATTENTION_HEADS = 20
ESM2_NUM_HIDDEN_LAYERS = 30
ESM2_INTERMEDIATE_SIZE = 2560
ESM2_VOCAB_SIZE = 33
ESM2_MAX_POSITION_EMBEDDINGS = 1026
ESM2_MODEL_TYPE = "esm"
ESM2_REPRESENTATION_CONTRACT_VERSION = 1
ESM2_IDENTITY_CONTRACT_VERSION = 1
ESM2_POOLING = "residue_mean"
ESM2_RESIDUE_OUTPUT_KIND = "residue_embeddings"
ESM2_TOKEN_OUTPUT_KIND = "residue_token_hidden"
ESM2_MAX_RESIDUES = ESM2_MAX_POSITION_EMBEDDINGS - 2

ESM2_VALID_SEQUENCE_AA = frozenset("ACDEFGHIKLMNPQRSTVWYBXZOU")
ESM2_REQUIRED_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "vocab.txt",
    "tokenizer.json",
)
ESM2_REQUIRED_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
ESM2_CONFIG_SIGNATURE = {
    "model_type": ESM2_MODEL_TYPE,
    "hidden_size": ESM2_HIDDEN_SIZE,
    "num_attention_heads": ESM2_NUM_ATTENTION_HEADS,
    "num_hidden_layers": ESM2_NUM_HIDDEN_LAYERS,
    "intermediate_size": ESM2_INTERMEDIATE_SIZE,
    "vocab_size": ESM2_VOCAB_SIZE,
    "max_position_embeddings": ESM2_MAX_POSITION_EMBEDDINGS,
}


def normalize_esm2_sequence(sequence: object) -> str:
    """Normalize one protein sequence and reject ambiguous empty input."""

    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise ValueError("ESM2 sequence must contain at least one residue")
    invalid = sorted(set(normalized) - ESM2_VALID_SEQUENCE_AA)
    if invalid:
        raise ValueError("ESM2 sequence contains unsupported residues: " + ", ".join(invalid))
    return normalized


def normalize_esm2_sequences(sequences) -> list[str]:
    return [normalize_esm2_sequence(value) for value in sequences]


def validate_esm2_max_residues(value: object, *, field: str = "max_length") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"ESM2 {field} must be an integer")
    normalized = int(value)
    if normalized < 1 or normalized > ESM2_MAX_RESIDUES:
        raise ValueError(f"ESM2 {field} must be between 1 and {ESM2_MAX_RESIDUES} residues")
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


def validate_esm2_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact t30/150M architecture signature."""

    if not isinstance(config, Mapping):
        raise ValueError("ESM2 config must be a mapping")
    mismatches = {}
    for key, expected in ESM2_CONFIG_SIGNATURE.items():
        observed = config.get(key)
        if observed != expected:
            mismatches[key] = {"expected": expected, "observed": observed}
    if mismatches:
        raise ValueError(
            "ESM2 architecture signature mismatch: "
            + json.dumps(mismatches, ensure_ascii=True, sort_keys=True)
        )
    return dict(config)


def _has_any(directory: Path, filenames) -> bool:
    return any((directory / name).is_file() for name in filenames)


def is_complete_esm2_snapshot(source: str | os.PathLike[str]) -> bool:
    path = _expanded_path(source)
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    try:
        validate_esm2_config(_load_config(path))
    except ValueError:
        return False
    return _has_any(path, ESM2_REQUIRED_TOKENIZER_FILES) and _has_any(
        path, ESM2_REQUIRED_WEIGHT_FILES
    )


def _weight_path(source: str | os.PathLike[str]) -> Path | None:
    path = _expanded_path(source)
    if not path.is_dir():
        return None
    for name in ESM2_REQUIRED_WEIGHT_FILES:
        candidate = path / name
        if candidate.is_file() and not name.endswith(".index.json"):
            return candidate
    return None


def inspect_esm2_source(source: str | os.PathLike[str]) -> dict[str, object]:
    """Return deterministic source facts without loading model weights."""

    value = str(source or "").strip()
    path = _expanded_path(value)
    local = path.is_dir()
    config = _load_config(path) if local else {}
    revision = model_snapshot_revision(path)
    if not local and value == ESM2_OFFICIAL_MODEL_ID:
        revision = ESM2_HF_REVISION
    weight = _weight_path(path) if local else None
    complete = is_complete_esm2_snapshot(path) if local else False
    if local:
        fingerprint = model_source_fingerprint(value) if complete else ""
    else:
        fingerprint_source = f"{value}@{revision}" if revision else value
        fingerprint = model_source_fingerprint(fingerprint_source) if value else ""
    if value == ESM2_OFFICIAL_MODEL_ID:
        provenance = "external_artifact_unverified"
    elif local:
        provenance = (
            "official_artifact_verified"
            if (
                revision == ESM2_HF_REVISION
                and weight is not None
                and sha256_file(weight) == ESM2_OFFICIAL_WEIGHT_SHA256
            )
            else "external_artifact_unverified"
        )
    else:
        provenance = "external_artifact_unverified"
    return {
        "source": value,
        "is_local": local,
        "complete": complete,
        "snapshot_revision": revision,
        "config": config,
        "weight_path": str(weight or ""),
        "weight_sha256": sha256_file(weight) if weight is not None else "",
        "model_fingerprint": fingerprint,
        "provenance_status": provenance,
    }


def resolve_esm2_source(
    project_root: str | os.PathLike[str],
    raw_model: str | os.PathLike[str] | None,
    local_candidates=(),
    *,
    offline: bool | None = None,
) -> str:
    """Resolve ESM2 to an existing complete local source or pinned HF ID."""

    value = str(raw_model or ESM2_OFFICIAL_MODEL_ID).strip() or ESM2_OFFICIAL_MODEL_ID
    project = Path(project_root)
    explicit_path = None
    if os.path.isabs(value):
        explicit_path = Path(value)
    elif value and (project / value).exists():
        explicit_path = project / value
    elif value and Path(value).exists():
        explicit_path = Path(value)
    if explicit_path is not None and (
        not explicit_path.is_dir() or not is_complete_esm2_snapshot(explicit_path)
    ):
        raise SystemExit(
            "ESM2 local model is incomplete or has the wrong architecture signature: "
            f"{explicit_path}"
        )
    candidates = []
    if os.path.isabs(value):
        candidates.append(Path(value))
    elif value and (project / value).exists():
        candidates.append(project / value)
    candidates.extend(Path(path) for path in local_candidates)
    for candidate in candidates:
        if is_complete_esm2_snapshot(candidate):
            return str(candidate)
    if value == ESM2_OFFICIAL_MODEL_ID:
        cached = os.environ.get("HF_HUB_CACHE", "")
        # Workflow's cache resolver remains the source of truth for locating
        # snapshots; this local check handles explicit candidates only.
        del cached
    offline_enabled = (
        offline
        if offline is not None
        else str(os.environ.get("PEPPLATFORM_OFFLINE_MODE", "")).lower()
        in {"1", "true", "yes", "on"}
    )
    if offline_enabled:
        raise SystemExit(
            "ESM2 local model not found. Provide a complete ESM2 snapshot or disable offline mode."
        )
    return value


def esm2_model_identity(
    source: str | os.PathLike[str],
    *,
    config: Mapping[str, object] | None = None,
    weight_path: str | os.PathLike[str] | None = None,
    weight_sha256: str = "",
    transformers_version: str = "",
    parameter_count: int | None = None,
) -> dict[str, object]:
    """Build the canonical identity for an observed ESM2 encoder."""

    value = str(source or "").strip()
    observed = inspect_esm2_source(value)
    cfg = dict(config or observed.get("config") or {})
    validate_esm2_config(cfg)
    resolved_weight = _expanded_path(weight_path or observed.get("weight_path", ""))
    observed_sha = weight_sha256 or (
        sha256_file(resolved_weight)
        if resolved_weight.is_file()
        else str(observed.get("weight_sha256", ""))
    )
    pinned = (
        value == ESM2_OFFICIAL_MODEL_ID or observed.get("snapshot_revision") == ESM2_HF_REVISION
    )
    digest_matches = observed_sha == ESM2_OFFICIAL_WEIGHT_SHA256 if observed_sha else False
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
        "contract_version": ESM2_IDENTITY_CONTRACT_VERSION,
        "provider": "Meta",
        "model_family": "ESM-2",
        "model_id": ESM2_OFFICIAL_MODEL_ID,
        "requested_source": value,
        "resolved_source": value,
        "snapshot_revision": str(observed.get("snapshot_revision", "") or ""),
        "model_fingerprint": str(observed.get("model_fingerprint", "") or ""),
        "parameter_count": int(parameter_count or 0),
        "weight_path": str(resolved_weight) if resolved_weight.is_file() else "",
        "weight_size_bytes": resolved_weight.stat().st_size if resolved_weight.is_file() else 0,
        "weight_sha256": observed_sha,
        "provenance_status": provenance,
        "runtime_package": "transformers",
        "runtime_version": str(runtime_version or ""),
        "architecture": "esm2_t30_150M_UR50D",
        **ESM2_CONFIG_SIGNATURE,
        "pooling": ESM2_POOLING,
        "output_kind": ESM2_RESIDUE_OUTPUT_KIND,
        "representation_contract_version": ESM2_REPRESENTATION_CONTRACT_VERSION,
        "max_residues": ESM2_MAX_RESIDUES,
        "special_tokens": "BOS/EOS/PAD excluded",
    }


def validate_esm2_model_identity(
    expected: Mapping[str, object] | None,
    observed: Mapping[str, object],
) -> None:
    """Reject artifact/runtime identity drift; missing legacy identity is caller-owned."""

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
            "ESM2 model identity differs from the training artifact: "
            + json.dumps(mismatches, ensure_ascii=True, sort_keys=True)
        )


def esm2_runtime_environment(device=None) -> dict[str, object]:
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


def format_esm2_provenance(model_identity, runtime_environment) -> str:
    return "[esm2-provenance] " + json.dumps(
        {"model_identity": dict(model_identity), "runtime_environment": dict(runtime_environment)},
        ensure_ascii=True,
        sort_keys=True,
    )
