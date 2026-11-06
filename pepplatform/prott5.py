"""Canonical ProtT5 (ProtTrans T5-XL) identity, source and representation contracts.

The public ``prott5`` route uses the Rostlab ProtT5-XL-UniRef50 checkpoint,
a T5-XL encoder-decoder model through Hugging Face Transformers. Architecture
identity and checkpoint provenance are kept separate so compatible local
directories remain usable without inheriting official-checkpoint claims.
"""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
import platform
from numbers import Integral
from pathlib import Path
from typing import Mapping

from pepplatform.cache_provenance import model_source_fingerprint, sha256_file


PROTT5_OFFICIAL_MODEL_ID = "Rostlab/prot_t5_xl_uniref50"
PROTT5_HF_REVISION = "973be27c52ee6474de9c945952a8008aeb2a1a73"
PROTT5_OFFICIAL_REVISION = PROTT5_HF_REVISION
PROTT5_OFFICIAL_WEIGHT_RELATIVE_PATH = Path("pytorch_model.bin")
PROTT5_OFFICIAL_WEIGHT_SHA256 = "c06141d42e93c12b5f6d95c701952156bd4079661f5f2d981d0d2036ba96dae9"

# T5-XL architecture constants
PROTT5_MODEL_TYPE = "t5"
PROTT5_D_MODEL = 1024  # hidden_size equivalent
PROTT5_D_KV = 128
PROTT5_D_FF = 16384
PROTT5_NUM_ENCODER_LAYERS = 24
PROTT5_NUM_DECODER_LAYERS = 24
PROTT5_NUM_ATTENTION_HEADS = 32
PROTT5_VOCAB_SIZE = 128
PROTT5_N_POSITIONS = 512
PROTT5_RELATIVE_ATTENTION_NUM_BUCKETS = 32
PROTT5_RELATIVE_ATTENTION_MAX_DISTANCE = 128
PROTT5_FEED_FORWARD_PROJ = "relu"
PROTT5_PAD_TOKEN_ID = 0
PROTT5_EOS_TOKEN_ID = 1

PROTT5_REPRESENTATION_CONTRACT_VERSION = 1
PROTT5_IDENTITY_CONTRACT_VERSION = 1
PROTT5_POOLING = "residue_mean"
PROTT5_RESIDUE_OUTPUT_KIND = "residue_embeddings"
PROTT5_TOKEN_OUTPUT_KIND = "residue_token_hidden"
PROTT5_MAX_RESIDUES = PROTT5_N_POSITIONS - 1  # room for EOS

PROTT5_VALID_SEQUENCE_AA = frozenset("ACDEFGHIKLMNPQRSTVWYX")
PROTT5_REPLACED_RESIDUES = frozenset("UZOB")

PROTT5_REQUIRED_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "spiece.model",
    "tokenizer.json",
)
PROTT5_REQUIRED_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

PROTT5_CONFIG_SIGNATURE = {
    "model_type": PROTT5_MODEL_TYPE,
    "d_model": PROTT5_D_MODEL,
    "d_kv": PROTT5_D_KV,
    "d_ff": PROTT5_D_FF,
    "num_layers": PROTT5_NUM_ENCODER_LAYERS,
    "num_decoder_layers": PROTT5_NUM_DECODER_LAYERS,
    "num_heads": PROTT5_NUM_ATTENTION_HEADS,
    "vocab_size": PROTT5_VOCAB_SIZE,
    "n_positions": PROTT5_N_POSITIONS,
    "relative_attention_num_buckets": PROTT5_RELATIVE_ATTENTION_NUM_BUCKETS,
    "relative_attention_max_distance": PROTT5_RELATIVE_ATTENTION_MAX_DISTANCE,
    "feed_forward_proj": PROTT5_FEED_FORWARD_PROJ,
    "pad_token_id": PROTT5_PAD_TOKEN_ID,
    "eos_token_id": PROTT5_EOS_TOKEN_ID,
}


def normalize_prott5_sequence(sequence: object) -> str:
    """Normalize a sequence using the ProtTrans T5 residue-token contract.

    Whitespace is removed, the sequence is uppercased, and U/Z/O/B are
    mapped to X.  Empty input is rejected.
    """

    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise ValueError("ProtT5 sequence must contain at least one residue")
    normalized = "".join(
        "X" if residue in PROTT5_REPLACED_RESIDUES else residue for residue in normalized
    )
    invalid = sorted(set(normalized) - PROTT5_VALID_SEQUENCE_AA)
    if invalid:
        raise ValueError("ProtT5 sequence contains unsupported residues: " + ", ".join(invalid))
    return normalized


def normalize_prott5_sequences(sequences) -> list[str]:
    return [normalize_prott5_sequence(value) for value in sequences]


def validate_prott5_tokenizer(tokenizer, *, probe: bool = True) -> dict[str, object]:
    """Validate the tokenizer facts required by the residue representation.

    The check is intentionally structural and works with both Hugging Face
    tokenizers and the small test doubles used by the pipelines.  Missing
    optional introspection APIs are rejected only when the tokenizer exposes
    the corresponding concept; the mandatory special-token IDs and
    one-residue-one-token probe must still be observable.
    """

    if tokenizer is None:
        raise ValueError("ProtT5 tokenizer is required")
    total_vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else 0
    base_vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    facts: dict[str, object] = {
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "vocab_size": int(total_vocab_size or base_vocab_size),
        "base_vocab_size": base_vocab_size,
        "residue_tokenization": "one_residue_one_sentencepiece_token",
        "special_tokens_to_add": int(tokenizer.num_special_tokens_to_add(pair=False))
        if hasattr(tokenizer, "num_special_tokens_to_add")
        else None,
    }
    if facts["special_tokens_to_add"] is not None and facts["special_tokens_to_add"] != 1:
        raise ValueError("ProtT5 tokenizer must add exactly one EOS special token")
    if facts["pad_token_id"] is not None and int(facts["pad_token_id"]) != PROTT5_PAD_TOKEN_ID:
        raise ValueError(f"ProtT5 tokenizer pad_token_id must be {PROTT5_PAD_TOKEN_ID}")
    if facts["eos_token_id"] is not None and int(facts["eos_token_id"]) != PROTT5_EOS_TOKEN_ID:
        raise ValueError(f"ProtT5 tokenizer eos_token_id must be {PROTT5_EOS_TOKEN_ID}")
    if facts["vocab_size"] and int(facts["vocab_size"]) != PROTT5_VOCAB_SIZE:
        raise ValueError(
            f"ProtT5 tokenizer vocab_size must be {PROTT5_VOCAB_SIZE}, got {facts['vocab_size']}"
        )
    if probe:
        try:
            encoded = tokenizer(
                ["A C"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3,
                return_special_tokens_mask=True,
            )
            input_ids = encoded.get("input_ids")
            attention = encoded.get("attention_mask")
            special = encoded.get("special_tokens_mask")
            if input_ids is None or attention is None or special is None:
                raise ValueError(
                    "ProtT5 tokenizer must return input_ids, attention_mask, "
                    "and special_tokens_mask"
                )
            active = attention[0].bool() if hasattr(attention[0], "bool") else attention[0]
            special_row = special[0].bool() if hasattr(special[0], "bool") else special[0]
            residue_count = int((active & ~special_row).sum())
            if residue_count != 2:
                raise ValueError(
                    "ProtT5 tokenizer must encode one token per residue; "
                    f"probe produced {residue_count} residue tokens"
                )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("ProtT5 tokenizer probe failed") from exc
    return facts


def validate_prott5_max_residues(value: object, *, field: str = "max_length") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"ProtT5 {field} must be an integer")
    normalized = int(value)
    if normalized < 1 or normalized > PROTT5_MAX_RESIDUES:
        raise ValueError(f"ProtT5 {field} must be between 1 and {PROTT5_MAX_RESIDUES} residues")
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


def validate_prott5_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the ProtT5 T5-XL architecture signature.

    Accepts ``T5Model`` and ``T5ForConditionalGeneration`` architecture
    variants when the core T5-XL signature matches.
    """

    if not isinstance(config, Mapping):
        raise ValueError("ProtT5 config must be a mapping")
    normalized = dict(config)
    # The pinned upstream config omits this field. T5Config defaults it to 128;
    # retain strict checks for explicit overrides and all XL-specific dimensions.
    normalized.setdefault(
        "relative_attention_max_distance", PROTT5_RELATIVE_ATTENTION_MAX_DISTANCE
    )
    architectures = normalized.get("architectures")
    if isinstance(architectures, list):
        if any(name in {"T5Model", "T5ForConditionalGeneration"} for name in architectures):
            if normalized.get("model_type") in (None, ""):
                normalized["model_type"] = PROTT5_MODEL_TYPE
    mismatches = {}
    for key, expected in PROTT5_CONFIG_SIGNATURE.items():
        observed = normalized.get(key)
        if observed != expected:
            mismatches[key] = {"expected": expected, "observed": observed}
    if mismatches:
        raise ValueError(
            "ProtT5 architecture signature mismatch: "
            + json.dumps(mismatches, ensure_ascii=True, sort_keys=True)
        )
    return normalized


def _has_any(directory: Path, filenames) -> bool:
    return any((directory / name).is_file() for name in filenames)


def _indexed_weight_files(directory: Path, index_name: str) -> list[Path]:
    index_path = directory / index_name
    if not index_path.is_file():
        return []
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return []
    files = []
    for raw_name in sorted(set(weight_map.values())):
        if not isinstance(raw_name, str) or not raw_name.strip():
            return []
        relative = Path(raw_name)
        if relative.is_absolute() or ".." in relative.parts:
            return []
        candidate = (directory / relative).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError:
            return []
        if not candidate.is_file():
            return []
        files.append(candidate)
    return files


def _declared_weight_files(directory: Path, *, safetensors: bool) -> list[Path]:
    extension = "safetensors" if safetensors else "bin"
    direct = directory / ("model.safetensors" if safetensors else "pytorch_model.bin")
    if direct.is_file():
        return [direct]
    return _indexed_weight_files(directory, f"model.{extension}.index.json")


def is_complete_prott5_snapshot(source: str | os.PathLike[str]) -> bool:
    path = _expanded_path(source)
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    try:
        validate_prott5_config(_load_config(path))
    except ValueError:
        return False
    has_tokenizer = (path / "tokenizer_config.json").is_file() and (path / "spiece.model").is_file()
    return has_tokenizer and bool(
        _declared_weight_files(path, safetensors=True)
        or _declared_weight_files(path, safetensors=False)
    )


def _weight_path(source: str | os.PathLike[str]) -> Path | None:
    path = _expanded_path(source)
    if not path.is_dir():
        return None
    if _declared_weight_files(path, safetensors=True):
        direct = path / "model.safetensors"
        return direct if direct.is_file() else path / "model.safetensors.index.json"
    if _declared_weight_files(path, safetensors=False):
        direct = path / "pytorch_model.bin"
        return direct if direct.is_file() else path / "pytorch_model.bin.index.json"
    return None


def _weight_manifest_files(source: str | os.PathLike[str]) -> list[Path]:
    """Return the index/direct weight and every referenced shard.

    A shard index is only a manifest; hashing it alone does not identify the
    actual checkpoint contents.  Keep the ordering deterministic and include
    the index itself so changing the tensor-to-shard mapping invalidates
    derived features as well.
    """

    directory = _expanded_path(source)
    if not directory.is_dir():
        return []
    for safetensors in (True, False):
        declared = _declared_weight_files(directory, safetensors=safetensors)
        if not declared:
            continue
        direct = directory / ("model.safetensors" if safetensors else "pytorch_model.bin")
        if direct.is_file():
            return [direct]
        index_name = (
            "model.safetensors.index.json" if safetensors else "pytorch_model.bin.index.json"
        )
        index = directory / index_name
        shards = _indexed_weight_files(directory, index_name)
        return [index, *shards]
    return []


def weight_manifest_sha256(source: str | os.PathLike[str]) -> str:
    """Hash the complete direct or indexed ProtT5 checkpoint."""

    files = _weight_manifest_files(source)
    if not files:
        return ""
    root = _expanded_path(source).resolve()
    digest = hashlib.sha256()
    for path in files:
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\0")
    return digest.hexdigest()


def weight_manifest_size_bytes(source: str | os.PathLike[str]) -> int:
    return sum(path.stat().st_size for path in _weight_manifest_files(source))


def inspect_prott5_source(source: str | os.PathLike[str]) -> dict[str, object]:
    """Return deterministic source facts without loading model weights."""

    value = str(source or "").strip()
    path = _expanded_path(value)
    local = path.is_dir()
    config = _load_config(path) if local else {}
    revision = model_snapshot_revision(path)
    if not local and value == PROTT5_OFFICIAL_MODEL_ID:
        revision = PROTT5_HF_REVISION
    weight = _weight_path(path) if local else None
    complete = is_complete_prott5_snapshot(path) if local else False
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
        "weight_sha256": (
            weight_manifest_sha256(path)
            if local
            else PROTT5_OFFICIAL_WEIGHT_SHA256
            if value == PROTT5_OFFICIAL_MODEL_ID
            else ""
        ),
        "weight_format": (
            "safetensors"
            if _declared_weight_files(path, safetensors=True)
            else "pytorch_bin"
            if _declared_weight_files(path, safetensors=False)
            else "pytorch_bin"
            if value == PROTT5_OFFICIAL_MODEL_ID
            else ""
        ),
        "weight_files": [
            str(item.relative_to(path.resolve()))
            for item in (
                _declared_weight_files(path, safetensors=True)
                or _declared_weight_files(path, safetensors=False)
            )
        ],
        "model_fingerprint": fingerprint,
        "provenance_status": "external_artifact_unverified",
    }


def prott5_model_identity(
    source: str | os.PathLike[str],
    *,
    config: Mapping[str, object] | None = None,
    weight_path: str | os.PathLike[str] | None = None,
    weight_sha256: str = "",
    transformers_version: str = "",
    parameter_count: int | None = None,
) -> dict[str, object]:
    """Build the canonical identity for an observed ProtT5 encoder."""

    value = str(source or "").strip()
    observed = inspect_prott5_source(value)
    cfg = dict(config or observed.get("config") or {})
    validate_prott5_config(cfg)
    resolved_weight = _expanded_path(weight_path or observed.get("weight_path", ""))
    observed_sha = weight_sha256 or str(observed.get("weight_sha256", ""))
    if not observed_sha and _expanded_path(value).is_dir():
        observed_sha = weight_manifest_sha256(value)
    provenance = "external_artifact_unverified"
    try:
        runtime_version = transformers_version or importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        runtime_version = transformers_version
    return {
        "contract_version": PROTT5_IDENTITY_CONTRACT_VERSION,
        "provider": "Rostlab/ProtTrans",
        "model_family": "ProtT5",
        "model_id": PROTT5_OFFICIAL_MODEL_ID,
        "requested_source": value,
        "resolved_source": value,
        "snapshot_revision": str(observed.get("snapshot_revision", "") or ""),
        "model_fingerprint": str(observed.get("model_fingerprint", "") or ""),
        "parameter_count": int(parameter_count or 0),
        "weight_path": str(resolved_weight) if resolved_weight.is_file() else "",
        "weight_size_bytes": (
            weight_manifest_size_bytes(value)
            if _expanded_path(value).is_dir()
            else resolved_weight.stat().st_size
            if resolved_weight.is_file()
            else 0
        ),
        "weight_sha256": observed_sha,
        "weight_format": str(observed.get("weight_format", "") or ""),
        "weight_files": list(observed.get("weight_files", []) or []),
        "provenance_status": provenance,
        "runtime_package": "transformers",
        "runtime_version": str(runtime_version or ""),
        "architecture": "t5_xl_prott5",
        **PROTT5_CONFIG_SIGNATURE,
        "pooling": PROTT5_POOLING,
        "output_kind": PROTT5_RESIDUE_OUTPUT_KIND,
        "representation_contract_version": PROTT5_REPRESENTATION_CONTRACT_VERSION,
        "max_residues": PROTT5_MAX_RESIDUES,
        "special_tokens": "EOS/PAD excluded",
        "input_normalization": "strip_whitespace_uppercase_UZOB_to_X",
        "tokenizer_contract": {
            "residue_tokenization": "one_residue_one_sentencepiece_token",
            "pad_token_id": PROTT5_PAD_TOKEN_ID,
            "eos_token_id": PROTT5_EOS_TOKEN_ID,
            "vocab_size": PROTT5_VOCAB_SIZE,
            "max_residues": PROTT5_MAX_RESIDUES,
        },
    }


def validate_prott5_model_identity(
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
        "model_fingerprint",
        "d_model",
        "num_heads",
        "num_layers",
        "pooling",
        "representation_contract_version",
        "tokenizer_contract",
    )
    mismatches = {
        key: {"expected": expected.get(key), "observed": observed.get(key)}
        for key in keys
        if expected.get(key) not in (None, "") and expected.get(key) != observed.get(key)
    }
    if mismatches:
        raise ValueError(
            "ProtT5 model identity differs from the training artifact: "
            + json.dumps(mismatches, ensure_ascii=True, sort_keys=True)
        )


def prott5_runtime_environment(device=None) -> dict[str, object]:
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


def format_prott5_provenance(model_identity, runtime_environment) -> str:
    return "[prott5-provenance] " + json.dumps(
        {
            "model_identity": dict(model_identity),
            "runtime_environment": dict(runtime_environment),
        },
        ensure_ascii=True,
        sort_keys=True,
    )
