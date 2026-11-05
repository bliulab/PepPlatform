"""Strict contract and artifact helpers for the platform's protein BPE variant."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from pepplatform.cache_provenance import sha256_file


BPE_CONTRACT_VERSION = 2
BPE_NORMALIZATION_VERSION = 1
BPE_PROVENANCE_VERSION = 2
BPE_VARIANT = "protein_byte_level_bpe"
BPE_IMPLEMENTATION = "huggingface_tokenizers_bpe"
BPE_TOKENIZERS_VERSION = "0.21.4"
BPE_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWYXBZJUO")
BPE_SPECIAL_TOKEN_IDS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<CLS>": 2,
    "<SEP>": 3,
    "<MASK>": 4,
}
BPE_SPECIAL_TOKENS = tuple(BPE_SPECIAL_TOKEN_IDS)
BPE_MIN_FREQUENCY = 1
# Leave room for at least one learned merge in addition to the base alphabet and
# fixed special tokens. A smaller vocabulary would not be meaningfully "BPE".
BPE_MIN_VOCAB_SIZE = len(BPE_ALPHABET) + len(BPE_SPECIAL_TOKENS) + 1


class BPEContractError(ValueError):
    """Raised when an input or artifact violates the platform BPE contract."""

    def __init__(self, message: str, *, code: str = "bpe_artifact_invalid"):
        super().__init__(message)
        self.code = code


def bpe_provenance_path(tokenizer_path: str | os.PathLike[str]) -> Path:
    return Path(f"{tokenizer_path}.provenance.json")


def normalize_bpe_sequence(sequence: object, *, context: str = "BPE sequence") -> str:
    """Apply the sole supported protein-BPE input normalization policy."""

    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise BPEContractError(
            f"{context} is empty after removing whitespace",
            code="bpe_sequence_invalid",
        )
    invalid = sorted(set(normalized).difference(BPE_ALPHABET))
    if invalid:
        rendered = ", ".join(repr(character) for character in invalid)
        raise BPEContractError(
            f"{context} contains unsupported character(s): {rendered}; "
            f"allowed alphabet is {''.join(BPE_ALPHABET)}",
            code="bpe_sequence_invalid",
        )
    return normalized


def normalize_bpe_sequences(
    sequences: Iterable[object],
    *,
    context: str = "BPE training sequence",
) -> list[str]:
    normalized = [
        normalize_bpe_sequence(sequence, context=f"{context} #{index}")
        for index, sequence in enumerate(sequences, start=1)
    ]
    if not normalized:
        raise BPEContractError(
            "BPE training corpus is empty",
            code="bpe_sequence_invalid",
        )
    return normalized


def bpe_training_digest(normalized_sequences: Iterable[str]) -> str:
    """Hash ordered, duplicate-preserving normalized training sequences."""

    digest = hashlib.sha256()
    for sequence in normalized_sequences:
        encoded = sequence.encode("ascii")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _load_json_object(path: Path, *, description: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BPEContractError(
            f"{description} is missing: {path}",
            code="bpe_artifact_missing",
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BPEContractError(
            f"{description} cannot be read as JSON: {path}: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise BPEContractError(f"{description} must contain a JSON object: {path}")
    return payload


def _special_token_id(payload: Mapping[str, object], token: str) -> int | None:
    model = payload.get("model")
    if isinstance(model, Mapping):
        vocab = model.get("vocab")
        if isinstance(vocab, Mapping) and token in vocab:
            try:
                return int(vocab[token])
            except (TypeError, ValueError):
                return None
    added_tokens = payload.get("added_tokens")
    if isinstance(added_tokens, list):
        for entry in added_tokens:
            if isinstance(entry, Mapping) and entry.get("content") == token:
                try:
                    return int(entry.get("id"))
                except (TypeError, ValueError):
                    return None
    return None


def inspect_bpe_artifact(
    tokenizer_path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Validate a tokenizer JSON without trusting its filename or CLI label."""

    path = Path(tokenizer_path)
    payload = _load_json_object(path, description="BPE tokenizer artifact")
    model = payload.get("model")
    if not isinstance(model, Mapping) or model.get("type") != "BPE":
        actual = model.get("type") if isinstance(model, Mapping) else None
        raise BPEContractError(f"BPE tokenizer model.type must be 'BPE', got {actual!r}: {path}")
    if model.get("unk_token") != "<UNK>":
        raise BPEContractError(f"BPE tokenizer model.unk_token must be '<UNK>': {path}")
    if model.get("dropout") is not None:
        raise BPEContractError(
            f"BPE tokenizer dropout must be disabled for deterministic inference: {path}"
        )
    for option in ("continuing_subword_prefix", "end_of_word_suffix"):
        if model.get(option) is not None:
            raise BPEContractError(f"BPE tokenizer model.{option} must be null: {path}")
    if model.get("fuse_unk") not in (False, None):
        raise BPEContractError(f"BPE tokenizer model.fuse_unk must be false: {path}")
    if model.get("ignore_merges") not in (False, None):
        raise BPEContractError(f"BPE tokenizer model.ignore_merges must be false: {path}")
    if payload.get("normalizer") is not None:
        raise BPEContractError(
            f"BPE tokenizer JSON normalizer must be null; normalization is external: {path}"
        )

    pre_tokenizer = payload.get("pre_tokenizer")
    if (
        not isinstance(pre_tokenizer, Mapping)
        or pre_tokenizer.get("type") != "ByteLevel"
        or pre_tokenizer.get("add_prefix_space") is not False
    ):
        raise BPEContractError(
            f"BPE tokenizer pre_tokenizer must be ByteLevel(add_prefix_space=False): {path}"
        )

    added_tokens = payload.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise BPEContractError(f"BPE tokenizer added_tokens must be an array: {path}")
    for token, expected_id in BPE_SPECIAL_TOKEN_IDS.items():
        actual_id = _special_token_id(payload, token)
        if actual_id != expected_id:
            raise BPEContractError(
                f"BPE tokenizer requires {token} id={expected_id}, got {actual_id!r}: {path}"
            )
        added_entry = next(
            (
                entry
                for entry in added_tokens
                if isinstance(entry, Mapping) and entry.get("content") == token
            ),
            None,
        )
        if not isinstance(added_entry, Mapping) or added_entry.get("special") is not True:
            raise BPEContractError(
                f"BPE tokenizer requires {token} to be registered as a special token: {path}"
            )

    vocab = model.get("vocab")
    if not isinstance(vocab, Mapping):
        raise BPEContractError(f"BPE tokenizer model.vocab must be an object: {path}")
    missing_alphabet = [token for token in BPE_ALPHABET if token not in vocab]
    if missing_alphabet:
        raise BPEContractError(
            "BPE tokenizer is missing required protein alphabet token(s) "
            f"{missing_alphabet}: {path}"
        )

    post_processor = payload.get("post_processor")
    special_map = (
        post_processor.get("special_tokens") if isinstance(post_processor, Mapping) else None
    )
    if (
        not isinstance(post_processor, Mapping)
        or post_processor.get("type") != "TemplateProcessing"
        or not isinstance(special_map, Mapping)
    ):
        raise BPEContractError(f"BPE tokenizer must use TemplateProcessing with CLS/SEP: {path}")
    expected_single = [
        {"SpecialToken": {"id": "<CLS>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<SEP>", "type_id": 0}},
    ]
    if post_processor.get("single") != expected_single:
        raise BPEContractError(
            f"BPE tokenizer single-sequence template must be '<CLS> $A <SEP>': {path}"
        )
    for token in ("<CLS>", "<SEP>"):
        entry = special_map.get(token)
        ids = entry.get("ids") if isinstance(entry, Mapping) else None
        if ids != [BPE_SPECIAL_TOKEN_IDS[token]]:
            raise BPEContractError(
                f"BPE tokenizer post-processor has invalid {token} mapping: {path}"
            )

    artifact_sha256 = sha256_file(path)
    if expected_sha256 and artifact_sha256 != str(expected_sha256):
        raise BPEContractError(
            f"BPE tokenizer SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {artifact_sha256}",
            code="bpe_artifact_digest_mismatch",
        )
    return {
        "contract_version": BPE_CONTRACT_VERSION,
        "model_type": "BPE",
        "variant": BPE_VARIANT,
        "vocab_size": len(vocab),
        "sha256": artifact_sha256,
    }


def load_bpe_provenance(
    tokenizer_path: str | os.PathLike[str],
    *,
    provenance_path: str | os.PathLike[str] | None = None,
) -> dict:
    path = Path(provenance_path) if provenance_path else bpe_provenance_path(tokenizer_path)
    return _load_json_object(path, description="BPE provenance")


def build_bpe_provenance(
    *,
    normalized_sequences: list[str],
    requested_vocab_size: int,
    actual_vocab_size: int,
    artifact_sha256: str,
    tokenizers_version: str,
    sequence_role: str = "sequence",
) -> dict:
    return {
        "schema_version": BPE_PROVENANCE_VERSION,
        "contract_version": BPE_CONTRACT_VERSION,
        "normalization_version": BPE_NORMALIZATION_VERSION,
        "implementation": BPE_IMPLEMENTATION,
        "variant": BPE_VARIANT,
        "feature_fit_scope": "train_only",
        "sequence_role": str(sequence_role or "sequence"),
        "training_digest": bpe_training_digest(normalized_sequences),
        "training_sequences": len(normalized_sequences),
        "normalization": {
            "uppercase": True,
            "remove_whitespace": True,
            "alphabet": "".join(BPE_ALPHABET),
        },
        "trainer": {
            "vocab_size": int(requested_vocab_size),
            "actual_vocab_size": int(actual_vocab_size),
            "min_frequency": BPE_MIN_FREQUENCY,
            "initial_alphabet": list(BPE_ALPHABET),
            "special_tokens": dict(BPE_SPECIAL_TOKEN_IDS),
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": False,
            },
        },
        "runtime": {
            "library": "tokenizers",
            "tokenizers_version": str(tokenizers_version),
        },
        "artifact_sha256": str(artifact_sha256),
    }


def bpe_cache_is_current(
    tokenizer_path: str | os.PathLike[str],
    *,
    normalized_sequences: list[str],
    vocab_size: int,
    tokenizers_version: str,
    sequence_role: str = "sequence",
) -> bool:
    """Return False for every stale, legacy, corrupt, or incompatible cache."""

    try:
        provenance = load_bpe_provenance(tokenizer_path)
        inspection = inspect_bpe_artifact(
            tokenizer_path,
            expected_sha256=provenance.get("artifact_sha256"),
        )
    except (BPEContractError, OSError, TypeError, ValueError):
        return False
    trainer = provenance.get("trainer")
    runtime = provenance.get("runtime")
    normalization = provenance.get("normalization")
    return bool(
        provenance.get("schema_version") == BPE_PROVENANCE_VERSION
        and provenance.get("contract_version") == BPE_CONTRACT_VERSION
        and provenance.get("normalization_version") == BPE_NORMALIZATION_VERSION
        and provenance.get("implementation") == BPE_IMPLEMENTATION
        and provenance.get("variant") == BPE_VARIANT
        and provenance.get("feature_fit_scope") == "train_only"
        and provenance.get("sequence_role") == str(sequence_role or "sequence")
        and provenance.get("training_digest") == bpe_training_digest(normalized_sequences)
        and provenance.get("training_sequences") == len(normalized_sequences)
        and isinstance(normalization, Mapping)
        and normalization.get("uppercase") is True
        and normalization.get("remove_whitespace") is True
        and normalization.get("alphabet") == "".join(BPE_ALPHABET)
        and isinstance(trainer, Mapping)
        and trainer.get("vocab_size") == int(vocab_size)
        and trainer.get("min_frequency") == BPE_MIN_FREQUENCY
        and trainer.get("initial_alphabet") == list(BPE_ALPHABET)
        and trainer.get("special_tokens") == BPE_SPECIAL_TOKEN_IDS
        and isinstance(runtime, Mapping)
        and runtime.get("tokenizers_version") == str(tokenizers_version)
        and inspection["vocab_size"] == trainer.get("actual_vocab_size")
    )


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary_name)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def bundle_bpe_artifact(
    tokenizer_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    *,
    role: str | None = None,
) -> dict:
    """Copy a validated BPE tokenizer and provenance into a portable bundle."""

    source = Path(tokenizer_path)
    source_provenance = bpe_provenance_path(source)
    provenance = load_bpe_provenance(source, provenance_path=source_provenance)
    inspection = inspect_bpe_artifact(
        source,
        expected_sha256=provenance.get("artifact_sha256"),
    )
    if (
        provenance.get("schema_version") != BPE_PROVENANCE_VERSION
        or provenance.get("contract_version") != BPE_CONTRACT_VERSION
        or provenance.get("normalization_version") != BPE_NORMALIZATION_VERSION
        or provenance.get("implementation") != BPE_IMPLEMENTATION
        or provenance.get("variant") != BPE_VARIANT
        or provenance.get("feature_fit_scope") != "train_only"
        or provenance.get("artifact_sha256") != inspection["sha256"]
        or (provenance.get("runtime") or {}).get("tokenizers_version") != BPE_TOKENIZERS_VERSION
        or (provenance.get("trainer") or {}).get("actual_vocab_size") != inspection["vocab_size"]
    ):
        raise BPEContractError(
            f"BPE provenance is incompatible with contract v{BPE_CONTRACT_VERSION}: "
            f"{source_provenance}"
        )

    normalized_role = str(role or "").strip().lower()
    stem = {
        "peptide": "peptide_bpe",
        "partner": "partner_bpe",
        "protein": "partner_bpe",
    }.get(normalized_role, "bpe")
    bundle_root = Path(artifact_dir)
    relative_tokenizer = Path("tokenizers") / f"{stem}.json"
    relative_provenance = Path("tokenizers") / f"{stem}.provenance.json"
    bundled_tokenizer = bundle_root / relative_tokenizer
    bundled_provenance = bundle_root / relative_provenance
    _atomic_copy(source, bundled_tokenizer)
    _atomic_copy(source_provenance, bundled_provenance)

    # Verify the copy rather than assuming the filesystem operation was faithful.
    inspect_bpe_artifact(
        bundled_tokenizer,
        expected_sha256=inspection["sha256"],
    )
    descriptor = {
        "contract_version": BPE_CONTRACT_VERSION,
        "path": relative_tokenizer.as_posix(),
        "sha256": inspection["sha256"],
        "provenance_path": relative_provenance.as_posix(),
        "model_type": "BPE",
        "variant": BPE_VARIANT,
        "vocab_size": inspection["vocab_size"],
        "tokenizers_version": str((provenance.get("runtime") or {}).get("tokenizers_version", "")),
    }
    return {
        "bpe_path": relative_tokenizer.as_posix(),
        "bpe_artifact": descriptor,
    }
