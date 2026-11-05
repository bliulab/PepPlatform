"""Strict contract and artifact helpers for the platform's protein WordPiece variant."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from pepplatform.cache_provenance import sha256_file


WORDPIECE_CONTRACT_VERSION = 2
WORDPIECE_NORMALIZATION_VERSION = 1
WORDPIECE_PROVENANCE_VERSION = 2
WORDPIECE_VARIANT = "protein_wordpiece_bpe_trained"
WORDPIECE_IMPLEMENTATION = "huggingface_tokenizers_wordpiece"
WORDPIECE_TRAINING_ALGORITHM = "huggingface_bpe_based_wordpiece_trainer"
WORDPIECE_TOKENIZERS_VERSION = "0.21.4"
WORDPIECE_CONTINUING_SUBWORD_PREFIX = "##"
WORDPIECE_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWYX")
WORDPIECE_AMBIGUOUS_RESIDUES = tuple("BZJUO")
WORDPIECE_SPECIAL_TOKEN_IDS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<CLS>": 2,
    "<SEP>": 3,
    "<MASK>": 4,
}
WORDPIECE_SPECIAL_TOKENS = tuple(WORDPIECE_SPECIAL_TOKEN_IDS)
WORDPIECE_DEFAULT_MAX_RAW_LENGTH = 50_000
WORDPIECE_ABSOLUTE_MAX_RAW_LENGTH = 100_000
WORDPIECE_MAX_LENGTH_UNIT = "tokens_including_special_tokens"
# Reserve every initial/continuation residue token and at least one learned
# multi-residue token in addition to the fixed special tokens.
WORDPIECE_MIN_VOCAB_SIZE = len(WORDPIECE_SPECIAL_TOKENS) + (2 * len(WORDPIECE_ALPHABET)) + 1


class WordPieceContractError(ValueError):
    """Raised when an input or artifact violates the platform WordPiece contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "wordpiece_artifact_invalid",
    ):
        super().__init__(message)
        self.code = code


def validate_wordpiece_max_raw_length(value: object) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise WordPieceContractError(
            f"wordpiece max_raw_length must be an integer, got {value!r}",
            code="wordpiece_length_invalid",
        ) from exc
    if normalized < 1 or normalized > WORDPIECE_ABSOLUTE_MAX_RAW_LENGTH:
        raise WordPieceContractError(
            "wordpiece max_raw_length must be in "
            f"[1, {WORDPIECE_ABSOLUTE_MAX_RAW_LENGTH}], got {normalized}",
            code="wordpiece_length_invalid",
        )
    return normalized


def validate_wordpiece_vocab_size(value: object) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise WordPieceContractError(
            f"wordpiece vocab_size must be an integer, got {value!r}",
            code="wordpiece_vocab_invalid",
        ) from exc
    if normalized < WORDPIECE_MIN_VOCAB_SIZE:
        raise WordPieceContractError(
            f"wordpiece vocab_size must be >= {WORDPIECE_MIN_VOCAB_SIZE} "
            "to reserve special tokens, residue fallbacks, and capacity for "
            "a learned piece",
            code="wordpiece_vocab_invalid",
        )
    return normalized


def wordpiece_provenance_path(
    tokenizer_path: str | os.PathLike[str],
) -> Path:
    return Path(f"{tokenizer_path}.provenance.json")


def normalize_wordpiece_sequence(
    sequence: object,
    *,
    max_raw_length: int = WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
    context: str = "WordPiece sequence",
) -> str:
    """Apply the platform's sole supported protein WordPiece normalization."""

    limit = validate_wordpiece_max_raw_length(max_raw_length)
    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise WordPieceContractError(
            f"{context} is empty after removing whitespace",
            code="wordpiece_sequence_invalid",
        )
    normalized = normalized.translate(
        str.maketrans({residue: "X" for residue in WORDPIECE_AMBIGUOUS_RESIDUES})
    )
    invalid = sorted(set(normalized).difference(WORDPIECE_ALPHABET))
    if invalid:
        rendered = ", ".join(repr(character) for character in invalid)
        raise WordPieceContractError(
            f"{context} contains unsupported character(s): {rendered}; "
            f"allowed normalized alphabet is {''.join(WORDPIECE_ALPHABET)}",
            code="wordpiece_sequence_invalid",
        )
    if len(normalized) > limit:
        raise WordPieceContractError(
            f"{context} contains {len(normalized)} residues, exceeding configured "
            f"wordpiece_max_raw_length={limit}",
            code="wordpiece_sequence_too_long",
        )
    return normalized


def normalize_wordpiece_sequences(
    sequences: Iterable[object],
    *,
    max_raw_length: int = WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
    context: str = "WordPiece training sequence",
) -> list[str]:
    normalized = [
        normalize_wordpiece_sequence(
            sequence,
            max_raw_length=max_raw_length,
            context=f"{context} #{index}",
        )
        for index, sequence in enumerate(sequences, start=1)
    ]
    if not normalized:
        raise WordPieceContractError(
            "WordPiece training corpus is empty",
            code="wordpiece_sequence_invalid",
        )
    return normalized


def wordpiece_training_digest(normalized_sequences: Iterable[str]) -> str:
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
        raise WordPieceContractError(
            f"{description} is missing: {path}",
            code="wordpiece_artifact_missing",
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WordPieceContractError(
            f"{description} cannot be read as JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise WordPieceContractError(f"{description} must contain a JSON object: {path}")
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


def inspect_wordpiece_artifact(
    tokenizer_path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Validate a tokenizer JSON without trusting its path or CLI label."""

    path = Path(tokenizer_path)
    payload = _load_json_object(path, description="WordPiece tokenizer artifact")
    model = payload.get("model")
    if not isinstance(model, Mapping) or model.get("type") != "WordPiece":
        actual = model.get("type") if isinstance(model, Mapping) else None
        raise WordPieceContractError(
            f"WordPiece tokenizer model.type must be 'WordPiece', got {actual!r}: {path}"
        )
    if model.get("unk_token") != "<UNK>":
        raise WordPieceContractError(f"WordPiece tokenizer model.unk_token must be '<UNK>': {path}")
    if model.get("continuing_subword_prefix") != WORDPIECE_CONTINUING_SUBWORD_PREFIX:
        raise WordPieceContractError(
            "WordPiece tokenizer continuing_subword_prefix must be "
            f"{WORDPIECE_CONTINUING_SUBWORD_PREFIX!r}: {path}"
        )
    try:
        stored_max_raw_length = validate_wordpiece_max_raw_length(
            model.get("max_input_chars_per_word")
        )
    except WordPieceContractError as exc:
        raise WordPieceContractError(
            f"WordPiece tokenizer has invalid max_input_chars_per_word: {path}: {exc}"
        ) from exc

    pre_tokenizer = payload.get("pre_tokenizer")
    if not isinstance(pre_tokenizer, Mapping) or pre_tokenizer.get("type") != "WhitespaceSplit":
        raise WordPieceContractError(
            f"WordPiece tokenizer pre_tokenizer must be WhitespaceSplit: {path}"
        )

    for token, expected_id in WORDPIECE_SPECIAL_TOKEN_IDS.items():
        actual_id = _special_token_id(payload, token)
        if actual_id != expected_id:
            raise WordPieceContractError(
                f"WordPiece tokenizer requires {token} id={expected_id}, got {actual_id!r}: {path}"
            )

    vocab = model.get("vocab")
    if not isinstance(vocab, Mapping):
        raise WordPieceContractError(f"WordPiece tokenizer model.vocab must be an object: {path}")
    required_tokens = [
        token
        for residue in WORDPIECE_ALPHABET
        for token in (
            residue,
            f"{WORDPIECE_CONTINUING_SUBWORD_PREFIX}{residue}",
        )
    ]
    missing_tokens = [token for token in required_tokens if token not in vocab]
    if missing_tokens:
        raise WordPieceContractError(
            "WordPiece tokenizer is missing required residue fallback token(s) "
            f"{missing_tokens}: {path}"
        )
    if any(type(value) is not int for value in vocab.values()):
        raise WordPieceContractError(f"WordPiece tokenizer vocabulary IDs must be integers: {path}")
    vocab_ids = list(vocab.values())
    if sorted(vocab_ids) != list(range(len(vocab_ids))):
        raise WordPieceContractError(
            "WordPiece tokenizer vocabulary IDs must be contiguous integers "
            f"from 0 to {len(vocab_ids) - 1}: {path}"
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
        raise WordPieceContractError(
            f"WordPiece tokenizer must use TemplateProcessing with CLS/SEP: {path}"
        )
    for token in ("<CLS>", "<SEP>"):
        entry = special_map.get(token)
        ids = entry.get("ids") if isinstance(entry, Mapping) else None
        if ids != [WORDPIECE_SPECIAL_TOKEN_IDS[token]]:
            raise WordPieceContractError(
                f"WordPiece tokenizer post-processor has invalid {token} mapping: {path}"
            )
    expected_single_template = [
        {"SpecialToken": {"id": "<CLS>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<SEP>", "type_id": 0}},
    ]
    if post_processor.get("single") != expected_single_template:
        raise WordPieceContractError(
            f"WordPiece tokenizer single-sequence template must be '<CLS> $A <SEP>': {path}"
        )

    artifact_sha256 = sha256_file(path)
    if expected_sha256 and artifact_sha256 != str(expected_sha256):
        raise WordPieceContractError(
            f"WordPiece tokenizer SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {artifact_sha256}",
            code="wordpiece_artifact_digest_mismatch",
        )
    return {
        "contract_version": WORDPIECE_CONTRACT_VERSION,
        "model_type": "WordPiece",
        "variant": WORDPIECE_VARIANT,
        "vocab_size": len(vocab),
        "max_raw_length": stored_max_raw_length,
        "sha256": artifact_sha256,
    }


def load_wordpiece_provenance(
    tokenizer_path: str | os.PathLike[str],
    *,
    provenance_path: str | os.PathLike[str] | None = None,
) -> dict:
    path = Path(provenance_path) if provenance_path else wordpiece_provenance_path(tokenizer_path)
    return _load_json_object(path, description="WordPiece provenance")


def validate_wordpiece_provenance(
    provenance: Mapping[str, object],
    inspection: Mapping[str, object],
    *,
    context: str = "WordPiece provenance",
) -> None:
    """Validate the corpus-independent portion of WordPiece provenance."""

    trainer = provenance.get("trainer")
    runtime = provenance.get("runtime")
    normalization = provenance.get("normalization")
    lengths = provenance.get("lengths")
    training_digest = provenance.get("training_digest")
    artifact_sha256 = provenance.get("artifact_sha256")
    expected = (
        provenance.get("schema_version") == WORDPIECE_PROVENANCE_VERSION
        and provenance.get("contract_version") == WORDPIECE_CONTRACT_VERSION
        and provenance.get("normalization_version") == WORDPIECE_NORMALIZATION_VERSION
        and provenance.get("implementation") == WORDPIECE_IMPLEMENTATION
        and provenance.get("training_algorithm") == WORDPIECE_TRAINING_ALGORITHM
        and provenance.get("variant") == WORDPIECE_VARIANT
        and provenance.get("feature_fit_scope") == "train_only"
        and isinstance(provenance.get("sequence_role"), str)
        and bool(str(provenance.get("sequence_role") or "").strip())
        and type(provenance.get("training_sequences")) is int
        and int(provenance.get("training_sequences") or 0) > 0
        and isinstance(training_digest, str)
        and len(training_digest) == 64
        and all(character in "0123456789abcdef" for character in training_digest)
        and isinstance(normalization, Mapping)
        and normalization.get("uppercase") is True
        and normalization.get("remove_whitespace") is True
        and normalization.get("map_to_x") == list(WORDPIECE_AMBIGUOUS_RESIDUES)
        and normalization.get("alphabet") == "".join(WORDPIECE_ALPHABET)
        and isinstance(trainer, Mapping)
        and type(trainer.get("vocab_size")) is int
        and trainer.get("vocab_size") >= WORDPIECE_MIN_VOCAB_SIZE
        and type(trainer.get("trainer_vocab_size")) is int
        and 0 < trainer.get("trainer_vocab_size") <= trainer.get("vocab_size")
        and trainer.get("actual_vocab_size") == inspection.get("vocab_size")
        and trainer.get("initial_alphabet") == list(WORDPIECE_ALPHABET)
        and isinstance(trainer.get("added_fallback_tokens"), list)
        and trainer.get("continuing_subword_prefix") == WORDPIECE_CONTINUING_SUBWORD_PREFIX
        and trainer.get("tie_breaking") == "tokenizers_internal"
        and trainer.get("special_tokens") == WORDPIECE_SPECIAL_TOKEN_IDS
        and trainer.get("pre_tokenizer") == {"type": "WhitespaceSplit"}
        and isinstance(lengths, Mapping)
        and lengths.get("max_raw_length") == inspection.get("max_raw_length")
        and lengths.get("absolute_max_raw_length") == WORDPIECE_ABSOLUTE_MAX_RAW_LENGTH
        and lengths.get("max_length_unit") == WORDPIECE_MAX_LENGTH_UNIT
        and isinstance(runtime, Mapping)
        and runtime.get("library") == "tokenizers"
        and runtime.get("tokenizers_version") == WORDPIECE_TOKENIZERS_VERSION
        and isinstance(artifact_sha256, str)
        and artifact_sha256 == inspection.get("sha256")
    )
    if not expected:
        raise WordPieceContractError(
            f"{context} is incompatible with WordPiece contract v{WORDPIECE_CONTRACT_VERSION}"
        )


def build_wordpiece_provenance(
    *,
    normalized_sequences: list[str],
    requested_vocab_size: int,
    trainer_vocab_size: int,
    actual_vocab_size: int,
    added_fallback_tokens: list[str],
    max_raw_length: int,
    artifact_sha256: str,
    tokenizers_version: str,
    sequence_role: str = "sequence",
) -> dict:
    return {
        "schema_version": WORDPIECE_PROVENANCE_VERSION,
        "contract_version": WORDPIECE_CONTRACT_VERSION,
        "normalization_version": WORDPIECE_NORMALIZATION_VERSION,
        "implementation": WORDPIECE_IMPLEMENTATION,
        "training_algorithm": WORDPIECE_TRAINING_ALGORITHM,
        "variant": WORDPIECE_VARIANT,
        "feature_fit_scope": "train_only",
        "sequence_role": str(sequence_role or "sequence"),
        "training_digest": wordpiece_training_digest(normalized_sequences),
        "training_sequences": len(normalized_sequences),
        "normalization": {
            "uppercase": True,
            "remove_whitespace": True,
            "map_to_x": list(WORDPIECE_AMBIGUOUS_RESIDUES),
            "alphabet": "".join(WORDPIECE_ALPHABET),
        },
        "trainer": {
            "vocab_size": int(requested_vocab_size),
            "trainer_vocab_size": int(trainer_vocab_size),
            "actual_vocab_size": int(actual_vocab_size),
            "initial_alphabet": list(WORDPIECE_ALPHABET),
            "added_fallback_tokens": list(added_fallback_tokens),
            "continuing_subword_prefix": WORDPIECE_CONTINUING_SUBWORD_PREFIX,
            "tie_breaking": "tokenizers_internal",
            "special_tokens": dict(WORDPIECE_SPECIAL_TOKEN_IDS),
            "pre_tokenizer": {"type": "WhitespaceSplit"},
        },
        "lengths": {
            "max_raw_length": validate_wordpiece_max_raw_length(max_raw_length),
            "absolute_max_raw_length": WORDPIECE_ABSOLUTE_MAX_RAW_LENGTH,
            "max_length_unit": WORDPIECE_MAX_LENGTH_UNIT,
        },
        "runtime": {
            "library": "tokenizers",
            "tokenizers_version": str(tokenizers_version),
        },
        "artifact_sha256": str(artifact_sha256),
    }


def wordpiece_cache_is_current(
    tokenizer_path: str | os.PathLike[str],
    *,
    normalized_sequences: list[str],
    vocab_size: int,
    max_raw_length: int,
    tokenizers_version: str,
    sequence_role: str = "sequence",
) -> bool:
    """Return False for every stale, legacy, corrupt, or incompatible cache."""

    try:
        provenance = load_wordpiece_provenance(tokenizer_path)
        inspection = inspect_wordpiece_artifact(
            tokenizer_path,
            expected_sha256=provenance.get("artifact_sha256"),
        )
        validate_wordpiece_provenance(provenance, inspection)
    except (WordPieceContractError, OSError, TypeError, ValueError):
        return False
    trainer = provenance.get("trainer")
    runtime = provenance.get("runtime")
    normalization = provenance.get("normalization")
    lengths = provenance.get("lengths")
    return bool(
        provenance.get("schema_version") == WORDPIECE_PROVENANCE_VERSION
        and provenance.get("contract_version") == WORDPIECE_CONTRACT_VERSION
        and provenance.get("normalization_version") == WORDPIECE_NORMALIZATION_VERSION
        and provenance.get("implementation") == WORDPIECE_IMPLEMENTATION
        and provenance.get("training_algorithm") == WORDPIECE_TRAINING_ALGORITHM
        and provenance.get("variant") == WORDPIECE_VARIANT
        and provenance.get("feature_fit_scope") == "train_only"
        and provenance.get("sequence_role") == str(sequence_role or "sequence")
        and provenance.get("training_digest") == wordpiece_training_digest(normalized_sequences)
        and provenance.get("training_sequences") == len(normalized_sequences)
        and isinstance(normalization, Mapping)
        and normalization.get("uppercase") is True
        and normalization.get("remove_whitespace") is True
        and normalization.get("map_to_x") == list(WORDPIECE_AMBIGUOUS_RESIDUES)
        and normalization.get("alphabet") == "".join(WORDPIECE_ALPHABET)
        and isinstance(trainer, Mapping)
        and trainer.get("vocab_size") == int(vocab_size)
        and trainer.get("initial_alphabet") == list(WORDPIECE_ALPHABET)
        and trainer.get("continuing_subword_prefix") == WORDPIECE_CONTINUING_SUBWORD_PREFIX
        and trainer.get("tie_breaking") == "tokenizers_internal"
        and trainer.get("special_tokens") == WORDPIECE_SPECIAL_TOKEN_IDS
        and isinstance(lengths, Mapping)
        and lengths.get("max_raw_length") == validate_wordpiece_max_raw_length(max_raw_length)
        and lengths.get("absolute_max_raw_length") == WORDPIECE_ABSOLUTE_MAX_RAW_LENGTH
        and lengths.get("max_length_unit") == WORDPIECE_MAX_LENGTH_UNIT
        and isinstance(runtime, Mapping)
        and runtime.get("tokenizers_version") == str(tokenizers_version)
        and inspection["vocab_size"] == trainer.get("actual_vocab_size")
        and inspection["max_raw_length"] == lengths.get("max_raw_length")
    )


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Mapping[str, object],
) -> None:
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


def bundle_wordpiece_artifact(
    tokenizer_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    *,
    role: str | None = None,
) -> dict:
    """Copy a validated WordPiece tokenizer and provenance into a portable bundle."""

    source = Path(tokenizer_path)
    source_provenance = wordpiece_provenance_path(source)
    provenance = load_wordpiece_provenance(
        source,
        provenance_path=source_provenance,
    )
    inspection = inspect_wordpiece_artifact(
        source,
        expected_sha256=provenance.get("artifact_sha256"),
    )
    validate_wordpiece_provenance(
        provenance,
        inspection,
        context=f"WordPiece provenance {source_provenance}",
    )

    normalized_role = str(role or "").strip().lower()
    stem = {
        "peptide": "peptide_wordpiece",
        "partner": "partner_wordpiece",
        "protein": "partner_wordpiece",
    }.get(normalized_role, "wordpiece")
    bundle_root = Path(artifact_dir)
    relative_tokenizer = Path("tokenizers") / f"{stem}.json"
    relative_provenance = Path("tokenizers") / f"{stem}.provenance.json"
    bundled_tokenizer = bundle_root / relative_tokenizer
    bundled_provenance = bundle_root / relative_provenance
    _atomic_copy(source, bundled_tokenizer)
    _atomic_copy(source_provenance, bundled_provenance)
    inspect_wordpiece_artifact(
        bundled_tokenizer,
        expected_sha256=inspection["sha256"],
    )
    descriptor = {
        "contract_version": WORDPIECE_CONTRACT_VERSION,
        "path": relative_tokenizer.as_posix(),
        "sha256": inspection["sha256"],
        "provenance_path": relative_provenance.as_posix(),
        "model_type": "WordPiece",
        "variant": WORDPIECE_VARIANT,
        "vocab_size": inspection["vocab_size"],
        "max_raw_length": int(
            (provenance.get("lengths") or {}).get(
                "max_raw_length",
                WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
        "tokenizers_version": str((provenance.get("runtime") or {}).get("tokenizers_version", "")),
    }
    return {
        "wordpiece_path": relative_tokenizer.as_posix(),
        "wordpiece_artifact": descriptor,
    }
