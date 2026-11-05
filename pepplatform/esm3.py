"""Canonical ESM3 identity, source, runtime and representation contracts.

ESM3 is loaded through the official EvolutionaryScale ``esm`` SDK.  The
platform deliberately keeps model-family identity separate from checkpoint
provenance: a complete local SDK tree is loadable, but it is not called an
official artifact without an exact revision and content digest.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import Mapping

from pepplatform.cache_provenance import model_source_fingerprint, sha256_file

ESM3_OFFICIAL_MODEL_ID = "EvolutionaryScale/esm3-sm-open-v1"
ESM3_RUNTIME_KEY = "esm3_sm_open_v1"
ESM3_MODEL_ALIASES = frozenset(
    {ESM3_RUNTIME_KEY, "esm3-sm-open-v1", "esm3-open-2024-03", "esm3-open", ESM3_OFFICIAL_MODEL_ID}
)
ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH = Path("data/weights/esm3_sm_open_v1.pth")
ESM3_HF_DOWNLOAD_METADATA_RELATIVE_PATH = Path(
    ".cache/huggingface/download/data/weights/esm3_sm_open_v1.pth.metadata"
)
ESM3_OFFICIAL_WEIGHT_SHA256 = "5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9"
ESM3_OFFICIAL_REVISIONS = frozenset(
    {
        "2feda652c51c57f9797fc1b01f8d6181b38fb488",
        "47f0545b2b6daf26a93439a3cd610f4f7f3d5478",
    }
)
ESM3_RUNTIME_VERSION = "3.2.0"
ESM3_RUNTIME_API_CONTRACT_VERSION = 1
ESM3_IDENTITY_CONTRACT_VERSION = 1
ESM3_REPRESENTATION_CONTRACT_VERSION = 1
ESM3_HIDDEN_SIZE = 1536
ESM3_NUM_ATTENTION_HEADS = 24
ESM3_NUM_VARIATIONAL_HEADS = 256
ESM3_NUM_LAYERS = 48
ESM3_MAX_RESIDUES = 2046
ESM3_POOLING = "residue_mean"
ESM3_DEFAULT_SDPA_BACKEND = "auto"
ESM3_SDPA_BACKENDS = ("auto", "math")
ESM3_RESIDUE_OUTPUT_KIND = "residue_embeddings"
ESM3_SEQUENCE_OUTPUT_KIND = "sequence_embedding"
ESM3_VALID_SEQUENCE_AA = frozenset("ACDEFGHIKLMNPQRSTVWYBXZOU")
ESM3_CHAIN_BREAK = "|"
ESM3_SPECIAL_TOKEN_IDS = {0, 1, 2, 31, 32}


def validate_esm3_sdpa_backend(value: object) -> str:
    normalized = str(value or ESM3_DEFAULT_SDPA_BACKEND).strip().lower()
    if normalized not in ESM3_SDPA_BACKENDS:
        raise ValueError(f"ESM3 SDPA backend must be one of {ESM3_SDPA_BACKENDS}, got {value!r}")
    return normalized


@lru_cache(maxsize=2)
def inspect_esm3_runtime(check_api: bool = True) -> dict[str, object]:
    try:
        installed = importlib.metadata.version("esm")
    except importlib.metadata.PackageNotFoundError:
        return {
            "package": "esm",
            "required_version": ESM3_RUNTIME_VERSION,
            "installed_version": "",
            "compatible": False,
            "code": "esm3_runtime_missing",
            "message": f"Official ESM3 requires esm=={ESM3_RUNTIME_VERSION}, but it is not installed.",
        }
    if installed != ESM3_RUNTIME_VERSION:
        return {
            "package": "esm",
            "required_version": ESM3_RUNTIME_VERSION,
            "installed_version": installed,
            "compatible": False,
            "code": "esm3_runtime_unsupported",
            "message": f"Official ESM3 requires esm=={ESM3_RUNTIME_VERSION}, found {installed}.",
        }
    if check_api:
        try:
            from esm.models.esm3 import ESM3  # noqa: F401
            from esm.sdk.api import ESMProtein, LogitsConfig  # noqa: F401
            from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer  # noqa: F401
        except Exception as exc:
            return {
                "package": "esm",
                "required_version": ESM3_RUNTIME_VERSION,
                "installed_version": installed,
                "compatible": False,
                "code": "esm3_runtime_api_incompatible",
                "message": f"esm=={installed} does not provide the required ESM3 API: {type(exc).__name__}: {exc}",
            }
    return {
        "package": "esm",
        "required_version": ESM3_RUNTIME_VERSION,
        "installed_version": installed,
        "compatible": True,
        "code": "",
        "message": "",
    }


def require_official_esm3_runtime(*, check_api: bool = True) -> dict[str, object]:
    status = inspect_esm3_runtime(check_api)
    if status["compatible"]:
        return status
    error_type = ImportError if status["code"] == "esm3_runtime_missing" else RuntimeError
    raise error_type(str(status["message"]))


def normalize_esm3_model_name(value: object) -> str:
    """Normalize public/HF aliases to the official SDK registry key."""

    raw = str(value or "").strip()
    if raw == ESM3_OFFICIAL_MODEL_ID or raw in ESM3_MODEL_ALIASES:
        return ESM3_RUNTIME_KEY
    return raw


def normalize_esm3_sequence(sequence: object) -> str:
    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise ValueError("ESM3 sequence must contain at least one residue")
    invalid = sorted(set(normalized) - ESM3_VALID_SEQUENCE_AA - {ESM3_CHAIN_BREAK})
    if invalid:
        raise ValueError("ESM3 sequence contains unsupported symbols: " + ", ".join(invalid))
    if all(char == ESM3_CHAIN_BREAK for char in normalized):
        raise ValueError("ESM3 sequence must contain at least one residue")
    if (
        normalized.startswith(ESM3_CHAIN_BREAK)
        or normalized.endswith(ESM3_CHAIN_BREAK)
        or "||" in normalized
    ):
        raise ValueError("ESM3 chain breaks must occur between non-empty residue chains")
    return normalized


def normalize_esm3_sequences(sequences) -> list[str]:
    return [normalize_esm3_sequence(value) for value in sequences]


def esm3_residue_count(sequence: object) -> int:
    return sum(char != ESM3_CHAIN_BREAK for char in normalize_esm3_sequence(sequence))


def truncate_esm3_sequence(sequence: object, max_residues: object) -> str:
    normalized = normalize_esm3_sequence(sequence)
    limit = validate_esm3_max_residues(max_residues)
    result = []
    residue_count = 0
    for char in normalized:
        if char == ESM3_CHAIN_BREAK and residue_count >= limit:
            break
        if char != ESM3_CHAIN_BREAK:
            if residue_count >= limit:
                break
            residue_count += 1
        result.append(char)
    return "".join(result).rstrip(ESM3_CHAIN_BREAK)


def validate_esm3_max_residues(value: object, *, field: str = "max_length") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"ESM3 {field} must be an integer")
    normalized = int(value)
    if normalized < 1 or normalized > ESM3_MAX_RESIDUES:
        raise ValueError(f"ESM3 {field} must be between 1 and {ESM3_MAX_RESIDUES} residues")
    return normalized


def _expanded_path(source: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(str(source or "").strip()))


def _local_download_metadata(source: str | os.PathLike[str]) -> tuple[str, str]:
    path = _expanded_path(source)
    if not path.is_dir():
        return "", ""
    metadata_path = path / ESM3_HF_DOWNLOAD_METADATA_RELATIVE_PATH
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "", ""
    if len(lines) < 2:
        return "", ""
    revision = lines[0].strip().lower()
    digest = lines[1].strip().lower()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        return "", ""
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return "", ""
    return revision, digest


def model_snapshot_revision(source: str | os.PathLike[str]) -> str:
    path = _expanded_path(source)
    parts = path.parts
    if "snapshots" not in parts:
        revision, _digest = _local_download_metadata(path)
        return revision
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else ""


def official_esm3_weight_path(source: str | os.PathLike[str]) -> Path | None:
    path = _expanded_path(source)
    if path.is_file():
        return path if path.name == ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH.name else None
    candidate = path / ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH
    return candidate if candidate.is_file() else None


def is_complete_esm3_source(source: str | os.PathLike[str]) -> bool:
    path = _expanded_path(source)
    return path.is_dir() and official_esm3_weight_path(path) is not None


def inspect_esm3_source(source: str | os.PathLike[str]) -> dict[str, object]:
    value = str(source or "").strip()
    path = _expanded_path(value)
    local = path.is_dir()
    weight = official_esm3_weight_path(path) if local else None
    complete = bool(weight is not None)
    revision = model_snapshot_revision(path)
    metadata_revision, metadata_digest = _local_download_metadata(path)
    fingerprint = ""
    if complete:
        fingerprint = model_source_fingerprint(value)
    elif value and not local:
        fingerprint = model_source_fingerprint(f"{value}@{revision}" if revision else value)
    digest = sha256_file(weight) if weight is not None else ""
    provenance = "external_artifact_unverified"
    metadata_matches = not metadata_revision or (
        metadata_revision == revision and metadata_digest == digest
    )
    if (
        revision in ESM3_OFFICIAL_REVISIONS
        and digest == ESM3_OFFICIAL_WEIGHT_SHA256
        and metadata_matches
    ):
        provenance = "official_artifact_verified"
    return {
        "source": value,
        "is_local": local,
        "complete": complete,
        "snapshot_revision": revision,
        "weight_path": str(weight or ""),
        "weight_sha256": digest,
        "model_fingerprint": fingerprint,
        "provenance_status": provenance,
    }


def esm3_model_identity(
    source: str | os.PathLike[str],
    *,
    weight_path: str | os.PathLike[str] | None = None,
    weight_sha256: str = "",
    runtime_version: str = "",
    parameter_count: int | None = None,
) -> dict[str, object]:
    value = str(source or "").strip()
    facts = inspect_esm3_source(value)
    resolved_weight = _expanded_path(weight_path or facts.get("weight_path", ""))
    # Reuse the verified checkpoint digest to avoid hashing the large file twice.
    if weight_sha256:
        digest = weight_sha256
    elif weight_path is not None and resolved_weight.is_file():
        digest = sha256_file(resolved_weight)
    else:
        digest = str(facts.get("weight_sha256", ""))
    provenance = str(facts.get("provenance_status") or "external_artifact_unverified")
    try:
        installed = importlib.metadata.version("esm")
    except importlib.metadata.PackageNotFoundError:
        installed = ""
    return {
        "contract_version": ESM3_IDENTITY_CONTRACT_VERSION,
        "provider": "EvolutionaryScale",
        "model_family": "ESM3",
        "model_id": ESM3_OFFICIAL_MODEL_ID,
        "runtime_key": ESM3_RUNTIME_KEY,
        "requested_source": value,
        "resolved_source": value,
        "snapshot_revision": str(facts.get("snapshot_revision", "") or ""),
        "model_fingerprint": str(facts.get("model_fingerprint", "") or ""),
        "parameter_count": int(parameter_count or 0),
        "weight_path": str(resolved_weight) if resolved_weight.is_file() else "",
        "weight_size_bytes": resolved_weight.stat().st_size if resolved_weight.is_file() else 0,
        "weight_sha256": digest,
        "provenance_status": provenance,
        "runtime_package": "esm",
        "runtime_version": str(runtime_version or installed or ""),
        "runtime_required_version": ESM3_RUNTIME_VERSION,
        "runtime_api_contract_version": ESM3_RUNTIME_API_CONTRACT_VERSION,
        "hidden_size": ESM3_HIDDEN_SIZE,
        "num_attention_heads": ESM3_NUM_ATTENTION_HEADS,
        "num_variational_heads": ESM3_NUM_VARIATIONAL_HEADS,
        "num_hidden_layers": ESM3_NUM_LAYERS,
        "pooling": ESM3_POOLING,
        "output_kind": ESM3_RESIDUE_OUTPUT_KIND,
        "representation_contract_version": ESM3_REPRESENTATION_CONTRACT_VERSION,
        "max_residues": ESM3_MAX_RESIDUES,
        "special_tokens": "CLS/EOS/PAD/CHAIN_BREAK/MASK excluded from residue output",
        "input_normalization": "strip_whitespace_uppercase_validate_alphabet",
    }


def validate_esm3_model_identity(
    expected: Mapping[str, object] | None, observed: Mapping[str, object]
) -> None:
    if not isinstance(expected, Mapping) or not expected:
        return
    keys = (
        "provider",
        "model_id",
        "runtime_key",
        "runtime_version",
        "weight_sha256",
        "hidden_size",
        "num_attention_heads",
        "num_variational_heads",
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
            "ESM3 model identity differs from the training artifact: "
            + json.dumps(mismatches, ensure_ascii=True, sort_keys=True)
        )


def esm3_runtime_environment(device=None) -> dict[str, object]:
    import torch

    torch_device = torch.device(device if device is not None else "cpu")
    index = torch_device.index
    name = ""
    if torch_device.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device() if index is None else index
        name = torch.cuda.get_device_name(index)
    try:
        version = importlib.metadata.version("esm")
    except importlib.metadata.PackageNotFoundError:
        version = ""
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "esm_version": str(version),
        "cuda_runtime_version": str(torch.version.cuda or ""),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": torch_device.type,
        "device_index": index,
        "device_name": name,
    }


def format_esm3_provenance(model_identity, runtime_environment) -> str:
    return "[esm3-provenance] " + json.dumps(
        {"model_identity": dict(model_identity), "runtime_environment": dict(runtime_environment)},
        ensure_ascii=True,
        sort_keys=True,
    )
