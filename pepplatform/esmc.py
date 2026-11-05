"""Canonical ESM-C model identities and local checkpoint inspection."""

from __future__ import annotations

import json
import importlib.metadata
import os
import platform
from functools import lru_cache
from pathlib import Path

from pepplatform.cache_provenance import sha256_file


ESMC_OFFICIAL_300M_MODEL_ID = "EvolutionaryScale/esmc-300m-2024-12"
ESMC_OFFICIAL_300M_RUNTIME_KEY = "esmc-300"
ESMC_OFFICIAL_300M_WEIGHT_RELATIVE_PATH = Path("data/weights/esmc_300m_2024_12_v0.pth")
ESMC_OFFICIAL_300M_WEIGHT_SHA256 = (
    "323dff9fbf3fef297a74f4f18b6528e6f2e599b0bcf72b6927516804015becea"
)
ESMC_OFFICIAL_300M_HIDDEN_SIZE = 960
ESMC_OFFICIAL_300M_NUM_HEADS = 15
ESMC_OFFICIAL_300M_NUM_LAYERS = 30

ESMC_OFFICIAL_600M_MODEL_ID = "EvolutionaryScale/esmc-600m-2024-12"
ESMC_OFFICIAL_600M_RUNTIME_KEY = "esmc-600"
ESMC_OFFICIAL_600M_WEIGHT_RELATIVE_PATH = Path("data/weights/esmc_600m_2024_12_v0.pth")
ESMC_OFFICIAL_600M_WEIGHT_SHA256 = (
    "8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324"
)
ESMC_OFFICIAL_600M_HIDDEN_SIZE = 1152
ESMC_OFFICIAL_600M_NUM_HEADS = 18
ESMC_OFFICIAL_600M_NUM_LAYERS = 36

ESMC_OFFICIAL_VARIANTS = {
    "300m": {
        "model_id": ESMC_OFFICIAL_300M_MODEL_ID,
        "runtime_key": ESMC_OFFICIAL_300M_RUNTIME_KEY,
        "weight_relative_path": ESMC_OFFICIAL_300M_WEIGHT_RELATIVE_PATH,
        "weight_sha256": ESMC_OFFICIAL_300M_WEIGHT_SHA256,
        "parameter_count": 300_000_000,
        "hidden_size": ESMC_OFFICIAL_300M_HIDDEN_SIZE,
        "num_attention_heads": ESMC_OFFICIAL_300M_NUM_HEADS,
        "num_hidden_layers": ESMC_OFFICIAL_300M_NUM_LAYERS,
    },
    "600m": {
        "model_id": ESMC_OFFICIAL_600M_MODEL_ID,
        "runtime_key": ESMC_OFFICIAL_600M_RUNTIME_KEY,
        "weight_relative_path": ESMC_OFFICIAL_600M_WEIGHT_RELATIVE_PATH,
        "weight_sha256": ESMC_OFFICIAL_600M_WEIGHT_SHA256,
        "parameter_count": 600_000_000,
        "hidden_size": ESMC_OFFICIAL_600M_HIDDEN_SIZE,
        "num_attention_heads": ESMC_OFFICIAL_600M_NUM_HEADS,
        "num_hidden_layers": ESMC_OFFICIAL_600M_NUM_LAYERS,
    },
}
ESMC_OFFICIAL_MODEL_IDS = tuple(variant["model_id"] for variant in ESMC_OFFICIAL_VARIANTS.values())
ESMC_300M_LICENSE_URL = "https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement"
ESMC_600M_LICENSE_URL = (
    "https://www.evolutionaryscale.ai/policies/cambrian-non-commercial-license-agreement"
)
ESMC_RUNTIME_VERSION = "3.2.0"
ESMC_RUNTIME_API_CONTRACT_VERSION = 1
ESMC_RUNTIME_INSTALL_COMMAND = "python -m pip install --no-cache-dir 'esm==3.2.0'"

ESMPLUSPLUS_SMALL_MODEL_ID = "Synthyra/ESMplusplus_small"
ESMPLUSPLUS_SMALL_HIDDEN_SIZE = 960

ESMC_BACKEND_OFFICIAL = "evolutionaryscale_esm"
ESMC_BACKEND_ESMPLUSPLUS = "synthyra_transformers"
ESMC_VALID_SEQUENCE_AA = frozenset("ACDEFGHIKLMNPQRSTVWYXBUZO")


def _runtime_error_message(code: str, installed_version: str = "", detail: str = "") -> str:
    if code == "esmc_runtime_missing":
        summary = "Official ESM-C requires the esm package, but it is not installed."
    elif code == "esmc_runtime_unsupported":
        summary = (
            f"Official ESM-C requires esm=={ESMC_RUNTIME_VERSION}, "
            f"found {installed_version or '<unknown>'}."
        )
    else:
        summary = (
            f"esm=={installed_version or ESMC_RUNTIME_VERSION} does not provide "
            "the official ESM-C API required by PepPlatform."
        )
    if detail:
        summary += f" Detail: {detail}"
    return f"{summary} Run: {ESMC_RUNTIME_INSTALL_COMMAND}"


@lru_cache(maxsize=2)
def inspect_esmc_runtime(check_api: bool = True) -> dict[str, object]:
    """Return the exact ESM-C runtime status without loading model weights."""
    try:
        installed_version = importlib.metadata.version("esm")
    except importlib.metadata.PackageNotFoundError:
        code = "esmc_runtime_missing"
        return {
            "package": "esm",
            "required_version": ESMC_RUNTIME_VERSION,
            "installed_version": "",
            "api_contract_version": ESMC_RUNTIME_API_CONTRACT_VERSION,
            "compatible": False,
            "code": code,
            "message": _runtime_error_message(code),
        }

    if installed_version != ESMC_RUNTIME_VERSION:
        code = "esmc_runtime_unsupported"
        return {
            "package": "esm",
            "required_version": ESMC_RUNTIME_VERSION,
            "installed_version": installed_version,
            "api_contract_version": ESMC_RUNTIME_API_CONTRACT_VERSION,
            "compatible": False,
            "code": code,
            "message": _runtime_error_message(code, installed_version),
        }

    if check_api:
        try:
            from esm.models.esmc import ESMC  # noqa: F401
            from esm.pretrained import ESMC_300M_202412, ESMC_600M_202412  # noqa: F401
            from esm.tokenization import get_esmc_model_tokenizers  # noqa: F401
            from esm.utils.constants.esm3 import data_root  # noqa: F401
            from esm.utils.encoding import tokenize_sequence  # noqa: F401
            from esm.utils.misc import stack_variable_length_tensors  # noqa: F401
        except Exception as exc:
            code = "esmc_runtime_api_incompatible"
            detail = f"{type(exc).__name__}: {exc}"
            return {
                "package": "esm",
                "required_version": ESMC_RUNTIME_VERSION,
                "installed_version": installed_version,
                "api_contract_version": ESMC_RUNTIME_API_CONTRACT_VERSION,
                "compatible": False,
                "code": code,
                "message": _runtime_error_message(code, installed_version, detail),
            }

    return {
        "package": "esm",
        "required_version": ESMC_RUNTIME_VERSION,
        "installed_version": installed_version,
        "api_contract_version": ESMC_RUNTIME_API_CONTRACT_VERSION,
        "compatible": True,
        "code": "",
        "message": "",
    }


def require_official_esmc_runtime(*, check_api: bool = True) -> dict[str, object]:
    status = inspect_esmc_runtime(check_api)
    if status["compatible"]:
        return status
    error_type = ImportError if status["code"] == "esmc_runtime_missing" else RuntimeError
    raise error_type(str(status["message"]))


def esmc_runtime_environment(device=None) -> dict[str, object]:
    import torch

    torch_device = torch.device(device if device is not None else "cpu")
    device_name = ""
    device_index = torch_device.index
    if torch_device.type == "cuda" and torch.cuda.is_available():
        device_index = torch.cuda.current_device() if device_index is None else device_index
        device_name = torch.cuda.get_device_name(device_index)
    runtime = inspect_esmc_runtime(check_api=False)
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "esm_version": str(runtime.get("installed_version", "")),
        "cuda_runtime_version": str(torch.version.cuda or ""),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": torch_device.type,
        "device_index": device_index,
        "device_name": device_name,
    }


def format_esmc_provenance(
    model_identity: dict[str, object], runtime_environment: dict[str, object]
) -> str:
    payload = {
        "model_identity": dict(model_identity),
        "runtime_environment": dict(runtime_environment),
    }
    return "[esmc-provenance] " + json.dumps(payload, ensure_ascii=True, sort_keys=True)


def validate_esmc_model_identity(
    expected: dict[str, object] | None,
    observed: dict[str, object],
) -> None:
    if not isinstance(expected, dict) or not expected:
        return
    if expected.get("provider") != "EvolutionaryScale":
        return
    keys = (
        "provider",
        "model_id",
        "variant",
        "parameter_count",
        "weight_sha256",
        "runtime_version",
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "pooling",
    )
    mismatches = {
        key: {"expected": expected.get(key), "observed": observed.get(key)}
        for key in keys
        if expected.get(key) not in (None, "") and expected.get(key) != observed.get(key)
    }
    if mismatches:
        raise ValueError(
            "ESM-C model identity differs from the training artifact: "
            + json.dumps(mismatches, ensure_ascii=True, sort_keys=True)
        )


def _expanded_path(source: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(str(source or "").strip()))


def _official_variant_for_model_id(model_id: str) -> tuple[str, dict[str, object]] | None:
    value = str(model_id or "").strip()
    for key, variant in ESMC_OFFICIAL_VARIANTS.items():
        if value in {str(variant["model_id"]), str(variant["runtime_key"]), key}:
            return key, variant
    return None


def official_esmc_model_options() -> list[dict[str, object]]:
    """Return JSON-safe official variants for CLI and Web selection."""

    options = []
    for key, variant in ESMC_OFFICIAL_VARIANTS.items():
        options.append(
            {
                "id": str(variant["model_id"]),
                "label": f"ESM-C {key.upper()}",
                "variant": key,
                "parameter_count": int(variant["parameter_count"]),
                "hidden_size": int(variant["hidden_size"]),
                "runtime_key": str(variant["runtime_key"]),
                "license_url": (ESMC_300M_LICENSE_URL if key == "300m" else ESMC_600M_LICENSE_URL),
                "download_policy": "official_huggingface_on_first_use",
            }
        )
    return options


def esmc_variant_for_source(
    source: str | os.PathLike[str], *, model_id: str = ""
) -> tuple[str, dict[str, object]] | None:
    """Resolve one official variant from an ID or an unambiguous local layout."""

    explicit = _official_variant_for_model_id(model_id)
    if explicit is not None:
        return explicit
    value = str(source or "").strip()
    by_id = _official_variant_for_model_id(value)
    if by_id is not None:
        return by_id
    path = _expanded_path(value)
    if not path.exists():
        return None
    matches = []
    for key, variant in ESMC_OFFICIAL_VARIANTS.items():
        relative = Path(variant["weight_relative_path"])
        if path.is_file():
            if path.name != relative.name:
                continue
            candidate = path
        else:
            candidate = path / relative
        if candidate.is_file():
            matches.append((key, variant))
    if len(matches) > 1:
        raise ValueError(f"ambiguous official ESM-C source contains multiple variants: {path}")
    return matches[0] if matches else None


def official_esmc_weight_path(source: str | os.PathLike[str], *, model_id: str = "") -> Path | None:
    resolved = esmc_variant_for_source(source, model_id=model_id)
    if resolved is None:
        return None
    _key, variant = resolved
    path = _expanded_path(source)
    if not path.exists():
        return None
    if path.is_file():
        expected_name = Path(variant["weight_relative_path"]).name
        return path if path.name == expected_name else None
    candidate = path / Path(variant["weight_relative_path"])
    return candidate if candidate.is_file() else None


def _local_model_type(source: str | os.PathLike[str]) -> str:
    path = _expanded_path(source)
    config_path = path / "config.json" if path.is_dir() else Path()
    if not config_path.is_file():
        return ""
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("model_type", "") or "").strip().lower()


def esmc_backend_for_source(source: str | os.PathLike[str]) -> str:
    value = str(source or "").strip()
    if esmc_variant_for_source(value) is not None:
        return ESMC_BACKEND_OFFICIAL
    if value == ESMPLUSPLUS_SMALL_MODEL_ID or _local_model_type(value) == "esmplusplus":
        return ESMC_BACKEND_ESMPLUSPLUS
    raise ValueError(
        "unsupported ESM-C model source; use "
        f"one of {ESMC_OFFICIAL_MODEL_IDS!r}, {ESMPLUSPLUS_SMALL_MODEL_ID!r}, "
        "or a verified local mirror of one of those checkpoints"
    )


@lru_cache(maxsize=8)
def validate_official_esmc_weight(
    source: str | os.PathLike[str], *, model_id: str = ""
) -> tuple[Path, str]:
    resolved = esmc_variant_for_source(source, model_id=model_id)
    if resolved is None:
        raise FileNotFoundError(f"unknown official ESM-C variant or source: {source}")
    variant_key, variant = resolved
    weight_path = official_esmc_weight_path(source, model_id=str(variant["model_id"]))
    if weight_path is None:
        raise FileNotFoundError(
            f"ESM-C {variant_key} checkpoint is missing "
            f"{Path(variant['weight_relative_path']).as_posix()} under {source}"
        )
    observed = sha256_file(weight_path)
    expected = str(variant.get("weight_sha256") or "")
    if expected and observed != expected:
        raise ValueError(
            f"ESM-C {variant_key} weight SHA256 mismatch: expected={expected} observed={observed}"
        )
    return weight_path, observed


def model_snapshot_revision(source: str | os.PathLike[str]) -> str:
    path = _expanded_path(source)
    parts = path.parts
    if "snapshots" not in parts:
        return ""
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else ""


def esmc_model_identity(
    source: str | os.PathLike[str],
    *,
    model_id: str = "",
    weight_path: str | os.PathLike[str] | None = None,
    weight_sha256: str = "",
    runtime_version: str = "",
    parameter_count: int | None = None,
) -> dict[str, object]:
    value = str(source or "").strip()
    resolved_variant = esmc_variant_for_source(value, model_id=model_id)
    if resolved_variant is None:
        backend = esmc_backend_for_source(value)
    else:
        _variant_key, variant = resolved_variant
        backend = ESMC_BACKEND_OFFICIAL
    if backend == ESMC_BACKEND_OFFICIAL:
        variant_key, variant = resolved_variant or ("", {})
        resolved_weight = Path(weight_path) if weight_path else None
        observed_digest = weight_sha256 or str(variant.get("weight_sha256") or "")
        expected_digest = str(variant.get("weight_sha256") or "")
        provenance = (
            "official_artifact_verified"
            if resolved_weight is not None
            and resolved_weight.is_file()
            and bool(expected_digest)
            and observed_digest == expected_digest
            else "external_artifact_unverified"
        )
        return {
            "provider": "EvolutionaryScale",
            "model_id": str(variant["model_id"]),
            "variant": variant_key,
            "parameter_count": int(
                parameter_count if parameter_count is not None else variant["parameter_count"]
            ),
            "backend": backend,
            "resolved_source": value,
            "snapshot_revision": model_snapshot_revision(weight_path or value),
            "weight_path": str(resolved_weight or ""),
            "weight_size_bytes": (
                resolved_weight.stat().st_size
                if resolved_weight is not None and resolved_weight.is_file()
                else 0
            ),
            "weight_sha256": observed_digest,
            "provenance_status": provenance,
            "runtime_package": "esm",
            "runtime_version": runtime_version or ESMC_RUNTIME_VERSION,
            "runtime_required_version": ESMC_RUNTIME_VERSION,
            "runtime_api_contract_version": ESMC_RUNTIME_API_CONTRACT_VERSION,
            "hidden_size": int(variant["hidden_size"]),
            "num_attention_heads": int(variant["num_attention_heads"]),
            "num_hidden_layers": int(variant["num_hidden_layers"]),
            "pooling": "residue_mean",
            "flash_attention": False,
        }
    local_weight = _expanded_path(value) / "model.safetensors"
    return {
        "provider": "Synthyra",
        "model_id": ESMPLUSPLUS_SMALL_MODEL_ID,
        "parameter_count": int(parameter_count or 0),
        "backend": backend,
        "resolved_source": value,
        "snapshot_revision": model_snapshot_revision(value),
        "weight_path": str(local_weight) if local_weight.is_file() else "",
        "weight_sha256": sha256_file(local_weight) if local_weight.is_file() else "",
        "provenance_status": "external_artifact_unverified",
        "runtime_package": "transformers",
        "runtime_version": runtime_version,
        "hidden_size": ESMPLUSPLUS_SMALL_HIDDEN_SIZE,
        "pooling": "attention_mask_mean",
    }
