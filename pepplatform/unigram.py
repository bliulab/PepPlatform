"""Strict contract, provenance, and deterministic sampling for protein Unigram."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from pepplatform.cache_provenance import sha256_file


UNIGRAM_CONTRACT_VERSION = 1
UNIGRAM_NORMALIZATION_VERSION = 1
UNIGRAM_PROVENANCE_VERSION = 1
UNIGRAM_SAMPLING_VERSION = 1
UNIGRAM_VARIANT = "protein_unigram_lm"
UNIGRAM_IMPLEMENTATION = "huggingface_tokenizers_unigram"
UNIGRAM_TRAINING_ALGORITHM = "kudo_unigram_em_pruning"
UNIGRAM_TOKENIZERS_VERSION = "0.21.4"
UNIGRAM_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWYXBZJUO")
UNIGRAM_SPECIAL_TOKEN_IDS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<CLS>": 2,
    "<SEP>": 3,
    "<MASK>": 4,
}
UNIGRAM_SPECIAL_TOKENS = tuple(UNIGRAM_SPECIAL_TOKEN_IDS)
UNIGRAM_MIN_VOCAB_SIZE = len(UNIGRAM_SPECIAL_TOKENS) + len(UNIGRAM_ALPHABET)
UNIGRAM_DEFAULT_MAX_RAW_LENGTH = 50_000
UNIGRAM_ABSOLUTE_MAX_RAW_LENGTH = 100_000
UNIGRAM_MAX_LENGTH_UNIT = "tokens_including_special_tokens"
UNIGRAM_SHRINKING_FACTOR = 0.75
UNIGRAM_N_SUB_ITERATIONS = 2
UNIGRAM_MAX_PIECE_LENGTH = 16
UNIGRAM_SEED_SIZE = 1_000_000
UNIGRAM_SCORE_DECIMALS = 12
UNIGRAM_SINGLE_SCORE_TOLERANCE = 0.005
UNIGRAM_DEFAULT_NBEST_SIZE = 64
UNIGRAM_MAX_NBEST_SIZE = 256
UNIGRAM_DEFAULT_ALPHA = 0.1


class UnigramContractError(ValueError):
    """Raised when input, configuration, or an artifact violates the contract."""

    def __init__(self, message: str, *, code: str = "unigram_artifact_invalid"):
        super().__init__(message)
        self.code = code


def validate_unigram_max_raw_length(value: object) -> int:
    if isinstance(value, bool):
        raise UnigramContractError(
            f"unigram max_raw_length must be an integer, got {value!r}",
            code="unigram_length_invalid",
        )
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise UnigramContractError(
            f"unigram max_raw_length must be an integer, got {value!r}",
            code="unigram_length_invalid",
        ) from exc
    if normalized < 1 or normalized > UNIGRAM_ABSOLUTE_MAX_RAW_LENGTH:
        raise UnigramContractError(
            "unigram max_raw_length must be in "
            f"[1, {UNIGRAM_ABSOLUTE_MAX_RAW_LENGTH}], got {normalized}",
            code="unigram_length_invalid",
        )
    return normalized


def validate_unigram_vocab_size(value: object) -> int:
    if isinstance(value, bool):
        raise UnigramContractError(
            f"unigram vocab_size must be an integer, got {value!r}",
            code="unigram_vocab_invalid",
        )
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise UnigramContractError(
            f"unigram vocab_size must be an integer, got {value!r}",
            code="unigram_vocab_invalid",
        ) from exc
    if normalized < UNIGRAM_MIN_VOCAB_SIZE:
        raise UnigramContractError(
            f"unigram vocab_size must be >= {UNIGRAM_MIN_VOCAB_SIZE} "
            "to reserve special tokens and one-character residue fallbacks",
            code="unigram_vocab_invalid",
        )
    return normalized


def validate_unigram_sampling(
    nbest_size: object = UNIGRAM_DEFAULT_NBEST_SIZE,
    alpha: object = UNIGRAM_DEFAULT_ALPHA,
) -> tuple[int, float]:
    if isinstance(nbest_size, bool):
        raise UnigramContractError(
            f"unigram nbest_size must be an integer, got {nbest_size!r}",
            code="unigram_sampling_invalid",
        )
    try:
        normalized_nbest = int(nbest_size)
    except (TypeError, ValueError) as exc:
        raise UnigramContractError(
            f"unigram nbest_size must be an integer, got {nbest_size!r}",
            code="unigram_sampling_invalid",
        ) from exc
    if isinstance(alpha, bool):
        raise UnigramContractError(
            f"unigram alpha must be a finite number, got {alpha!r}",
            code="unigram_sampling_invalid",
        )
    try:
        normalized_alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise UnigramContractError(
            f"unigram alpha must be a finite number, got {alpha!r}",
            code="unigram_sampling_invalid",
        ) from exc
    if normalized_nbest < 2 or normalized_nbest > UNIGRAM_MAX_NBEST_SIZE:
        raise UnigramContractError(
            f"unigram nbest_size must be in [2, {UNIGRAM_MAX_NBEST_SIZE}] when sampling is enabled",
            code="unigram_sampling_invalid",
        )
    if not math.isfinite(normalized_alpha) or normalized_alpha <= 0.0:
        raise UnigramContractError(
            "unigram alpha must be finite and > 0 when sampling is enabled",
            code="unigram_sampling_invalid",
        )
    return normalized_nbest, normalized_alpha


def unigram_provenance_path(
    tokenizer_path: str | os.PathLike[str],
) -> Path:
    return Path(f"{tokenizer_path}.provenance.json")


def normalize_unigram_sequence(
    sequence: object,
    *,
    max_raw_length: int = UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    context: str = "Unigram sequence",
) -> str:
    limit = validate_unigram_max_raw_length(max_raw_length)
    normalized = "".join(str(sequence or "").split()).upper()
    if not normalized:
        raise UnigramContractError(
            f"{context} is empty after removing whitespace",
            code="unigram_sequence_invalid",
        )
    invalid = sorted(set(normalized).difference(UNIGRAM_ALPHABET))
    if invalid:
        rendered = ", ".join(repr(character) for character in invalid)
        raise UnigramContractError(
            f"{context} contains unsupported character(s): {rendered}; "
            f"allowed alphabet is {''.join(UNIGRAM_ALPHABET)}",
            code="unigram_sequence_invalid",
        )
    if len(normalized) > limit:
        raise UnigramContractError(
            f"{context} contains {len(normalized)} residues, exceeding configured "
            f"unigram_max_raw_length={limit}",
            code="unigram_sequence_too_long",
        )
    return normalized


def normalize_unigram_sequences(
    sequences: Iterable[object],
    *,
    max_raw_length: int = UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    context: str = "Unigram training sequence",
) -> list[str]:
    normalized = [
        normalize_unigram_sequence(
            sequence,
            max_raw_length=max_raw_length,
            context=f"{context} #{index}",
        )
        for index, sequence in enumerate(sequences, start=1)
    ]
    if not normalized:
        raise UnigramContractError(
            "Unigram training corpus is empty",
            code="unigram_sequence_invalid",
        )
    return normalized


def unigram_training_digest(normalized_sequences: Iterable[str]) -> str:
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
        raise UnigramContractError(
            f"{description} is missing: {path}",
            code="unigram_artifact_missing",
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UnigramContractError(f"{description} cannot be read as JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise UnigramContractError(f"{description} must contain a JSON object: {path}")
    return payload


def _unigram_vocab(payload: Mapping[str, object], path: Path) -> list[tuple[str, float]]:
    model = payload.get("model")
    raw_vocab = model.get("vocab") if isinstance(model, Mapping) else None
    if not isinstance(raw_vocab, list):
        raise UnigramContractError(f"Unigram tokenizer model.vocab must be an array: {path}")
    vocab: list[tuple[str, float]] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw_vocab):
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or isinstance(entry[1], bool)
            or not isinstance(entry[1], (int, float))
        ):
            raise UnigramContractError(
                f"Unigram vocabulary entry {index} must be [token, finite_score]: {path}"
            )
        token = entry[0]
        score = float(entry[1])
        if not token or token in seen or not math.isfinite(score):
            raise UnigramContractError(
                f"Unigram vocabulary entry {index} is empty, duplicate, or non-finite: {path}"
            )
        seen.add(token)
        vocab.append((token, score))
    return vocab


def inspect_unigram_artifact(
    tokenizer_path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> dict:
    """Validate a deterministic platform Unigram JSON artifact."""

    path = Path(tokenizer_path)
    payload = _load_json_object(path, description="Unigram tokenizer artifact")
    model = payload.get("model")
    if not isinstance(model, Mapping) or model.get("type") != "Unigram":
        actual = model.get("type") if isinstance(model, Mapping) else None
        raise UnigramContractError(
            f"Unigram tokenizer model.type must be 'Unigram', got {actual!r}: {path}"
        )
    if model.get("unk_id") != UNIGRAM_SPECIAL_TOKEN_IDS["<UNK>"]:
        raise UnigramContractError(f"Unigram tokenizer model.unk_id must be 1: {path}")
    if model.get("byte_fallback") not in (False, None):
        raise UnigramContractError(f"Unigram tokenizer byte_fallback must be false: {path}")
    if model.get("alpha") not in (None,):
        raise UnigramContractError(
            f"Unigram tokenizer model.alpha must not persist runtime sampling: {path}"
        )
    if model.get("nbest_size") not in (None,):
        raise UnigramContractError(
            f"Unigram tokenizer model.nbest_size must not persist runtime sampling: {path}"
        )
    if payload.get("normalizer") is not None or payload.get("pre_tokenizer") is not None:
        raise UnigramContractError(
            "Unigram tokenizer normalizer and pre_tokenizer must be null; "
            f"protein normalization is external: {path}"
        )

    vocab = _unigram_vocab(payload, path)
    token_to_id = {token: index for index, (token, _) in enumerate(vocab)}
    for token, expected_id in UNIGRAM_SPECIAL_TOKEN_IDS.items():
        if token_to_id.get(token) != expected_id:
            raise UnigramContractError(
                f"Unigram tokenizer requires {token} id={expected_id}, "
                f"got {token_to_id.get(token)!r}: {path}"
            )
    missing_alphabet = [token for token in UNIGRAM_ALPHABET if token not in token_to_id]
    if missing_alphabet:
        raise UnigramContractError(
            "Unigram tokenizer is missing required one-character residue pieces "
            f"{missing_alphabet}: {path}"
        )
    invalid_pieces = [
        token
        for token in token_to_id
        if token not in UNIGRAM_SPECIAL_TOKEN_IDS
        and (len(token) > UNIGRAM_MAX_PIECE_LENGTH or not set(token).issubset(UNIGRAM_ALPHABET))
    ]
    if invalid_pieces:
        raise UnigramContractError(
            "Unigram tokenizer contains pieces outside the protein alphabet "
            f"or longer than {UNIGRAM_MAX_PIECE_LENGTH}: "
            f"{invalid_pieces[:5]}: {path}"
        )

    added_tokens = payload.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise UnigramContractError(f"Unigram tokenizer added_tokens must be an array: {path}")
    if len(added_tokens) != len(UNIGRAM_SPECIAL_TOKEN_IDS):
        raise UnigramContractError(
            "Unigram tokenizer added_tokens must contain exactly the five "
            f"platform special tokens: {path}"
        )
    for token, expected_id in UNIGRAM_SPECIAL_TOKEN_IDS.items():
        entry = next(
            (
                item
                for item in added_tokens
                if isinstance(item, Mapping) and item.get("content") == token
            ),
            None,
        )
        if (
            not isinstance(entry, Mapping)
            or entry.get("id") != expected_id
            or entry.get("special") is not True
        ):
            raise UnigramContractError(
                f"Unigram tokenizer must register {token} as special id={expected_id}: {path}"
            )

    post_processor = payload.get("post_processor")
    special_map = (
        post_processor.get("special_tokens") if isinstance(post_processor, Mapping) else None
    )
    expected_single = [
        {"SpecialToken": {"id": "<CLS>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<SEP>", "type_id": 0}},
    ]
    if (
        not isinstance(post_processor, Mapping)
        or post_processor.get("type") != "TemplateProcessing"
        or post_processor.get("single") != expected_single
        or not isinstance(special_map, Mapping)
    ):
        raise UnigramContractError(
            f"Unigram tokenizer must use single template '<CLS> $A <SEP>': {path}"
        )
    for token in ("<CLS>", "<SEP>"):
        entry = special_map.get(token)
        ids = entry.get("ids") if isinstance(entry, Mapping) else None
        if ids != [UNIGRAM_SPECIAL_TOKEN_IDS[token]]:
            raise UnigramContractError(
                f"Unigram post-processor has invalid {token} mapping: {path}"
            )

    artifact_sha256 = sha256_file(path)
    if expected_sha256 and artifact_sha256 != str(expected_sha256):
        raise UnigramContractError(
            f"Unigram tokenizer SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {artifact_sha256}",
            code="unigram_artifact_digest_mismatch",
        )
    return {
        "contract_version": UNIGRAM_CONTRACT_VERSION,
        "model_type": "Unigram",
        "variant": UNIGRAM_VARIANT,
        "vocab_size": len(vocab),
        "sha256": artifact_sha256,
    }


def load_unigram_provenance(
    tokenizer_path: str | os.PathLike[str],
    *,
    provenance_path: str | os.PathLike[str] | None = None,
) -> dict:
    path = Path(provenance_path) if provenance_path else unigram_provenance_path(tokenizer_path)
    return _load_json_object(path, description="Unigram provenance")


def validate_unigram_provenance(
    provenance: Mapping[str, object],
    inspection: Mapping[str, object],
    *,
    context: str = "Unigram provenance",
) -> None:
    trainer = provenance.get("trainer")
    runtime = provenance.get("runtime")
    normalization = provenance.get("normalization")
    lengths = provenance.get("lengths")
    canonicalization = provenance.get("canonicalization")
    expected = (
        provenance.get("schema_version") == UNIGRAM_PROVENANCE_VERSION
        and provenance.get("contract_version") == UNIGRAM_CONTRACT_VERSION
        and provenance.get("normalization_version") == UNIGRAM_NORMALIZATION_VERSION
        and provenance.get("implementation") == UNIGRAM_IMPLEMENTATION
        and provenance.get("training_algorithm") == UNIGRAM_TRAINING_ALGORITHM
        and provenance.get("variant") == UNIGRAM_VARIANT
        and provenance.get("feature_fit_scope") == "train_only"
        and isinstance(provenance.get("sequence_role"), str)
        and bool(str(provenance.get("sequence_role") or "").strip())
        and type(provenance.get("training_sequences")) is int
        and int(provenance.get("training_sequences") or 0) > 0
        and isinstance(provenance.get("training_digest"), str)
        and len(str(provenance.get("training_digest"))) == 64
        and isinstance(normalization, Mapping)
        and normalization.get("uppercase") is True
        and normalization.get("remove_whitespace") is True
        and normalization.get("alphabet") == "".join(UNIGRAM_ALPHABET)
        and isinstance(trainer, Mapping)
        and type(trainer.get("vocab_size")) is int
        and trainer.get("vocab_size") >= UNIGRAM_MIN_VOCAB_SIZE
        and trainer.get("actual_vocab_size") == inspection.get("vocab_size")
        and trainer.get("initial_alphabet") == list(UNIGRAM_ALPHABET)
        and trainer.get("special_tokens") == UNIGRAM_SPECIAL_TOKEN_IDS
        and trainer.get("unk_token") == "<UNK>"
        and trainer.get("shrinking_factor") == UNIGRAM_SHRINKING_FACTOR
        and trainer.get("n_sub_iterations") == UNIGRAM_N_SUB_ITERATIONS
        and trainer.get("max_piece_length") == UNIGRAM_MAX_PIECE_LENGTH
        and trainer.get("seed_size") == UNIGRAM_SEED_SIZE
        and trainer.get("pre_tokenizer") is None
        and isinstance(canonicalization, Mapping)
        and canonicalization.get("ordinary_token_order") == "unicode_lexicographic"
        and canonicalization.get("score_decimals") == UNIGRAM_SCORE_DECIMALS
        and canonicalization.get("single_residue_score_tolerance") == UNIGRAM_SINGLE_SCORE_TOLERANCE
        and canonicalization.get("missing_residue_score") == "learned_min_minus_one"
        and isinstance(lengths, Mapping)
        and lengths.get("max_raw_length") == provenance.get("max_raw_length")
        and lengths.get("absolute_max_raw_length") == UNIGRAM_ABSOLUTE_MAX_RAW_LENGTH
        and lengths.get("max_length_unit") == UNIGRAM_MAX_LENGTH_UNIT
        and isinstance(runtime, Mapping)
        and runtime.get("library") == "tokenizers"
        and runtime.get("tokenizers_version") == UNIGRAM_TOKENIZERS_VERSION
        and provenance.get("artifact_sha256") == inspection.get("sha256")
    )
    if not expected:
        raise UnigramContractError(
            f"{context} is incompatible with Unigram contract v{UNIGRAM_CONTRACT_VERSION}"
        )


def build_unigram_provenance(
    *,
    normalized_sequences: list[str],
    requested_vocab_size: int,
    actual_vocab_size: int,
    max_raw_length: int,
    artifact_sha256: str,
    tokenizers_version: str,
    sequence_role: str = "sequence",
) -> dict:
    limit = validate_unigram_max_raw_length(max_raw_length)
    return {
        "schema_version": UNIGRAM_PROVENANCE_VERSION,
        "contract_version": UNIGRAM_CONTRACT_VERSION,
        "normalization_version": UNIGRAM_NORMALIZATION_VERSION,
        "implementation": UNIGRAM_IMPLEMENTATION,
        "training_algorithm": UNIGRAM_TRAINING_ALGORITHM,
        "variant": UNIGRAM_VARIANT,
        "feature_fit_scope": "train_only",
        "sequence_role": str(sequence_role or "sequence"),
        "training_digest": unigram_training_digest(normalized_sequences),
        "training_sequences": len(normalized_sequences),
        "max_raw_length": limit,
        "normalization": {
            "uppercase": True,
            "remove_whitespace": True,
            "alphabet": "".join(UNIGRAM_ALPHABET),
        },
        "trainer": {
            "vocab_size": int(requested_vocab_size),
            "actual_vocab_size": int(actual_vocab_size),
            "initial_alphabet": list(UNIGRAM_ALPHABET),
            "special_tokens": dict(UNIGRAM_SPECIAL_TOKEN_IDS),
            "unk_token": "<UNK>",
            "shrinking_factor": UNIGRAM_SHRINKING_FACTOR,
            "n_sub_iterations": UNIGRAM_N_SUB_ITERATIONS,
            "max_piece_length": UNIGRAM_MAX_PIECE_LENGTH,
            "seed_size": UNIGRAM_SEED_SIZE,
            "pre_tokenizer": None,
        },
        "canonicalization": {
            "ordinary_token_order": "unicode_lexicographic",
            "score_decimals": UNIGRAM_SCORE_DECIMALS,
            "single_residue_score_tolerance": UNIGRAM_SINGLE_SCORE_TOLERANCE,
            "missing_residue_score": "learned_min_minus_one",
        },
        "lengths": {
            "max_raw_length": limit,
            "absolute_max_raw_length": UNIGRAM_ABSOLUTE_MAX_RAW_LENGTH,
            "max_length_unit": UNIGRAM_MAX_LENGTH_UNIT,
        },
        "runtime": {
            "library": "tokenizers",
            "tokenizers_version": str(tokenizers_version),
        },
        "sampling": {
            "schema_version": UNIGRAM_SAMPLING_VERSION,
            "default_enabled": False,
            "default_nbest_size": UNIGRAM_DEFAULT_NBEST_SIZE,
            "default_alpha": UNIGRAM_DEFAULT_ALPHA,
            "seed_derivation": "sha256_context_v1",
        },
        "artifact_sha256": str(artifact_sha256),
    }


def unigram_cache_is_current(
    tokenizer_path: str | os.PathLike[str],
    *,
    normalized_sequences: list[str],
    vocab_size: int,
    max_raw_length: int,
    tokenizers_version: str,
    sequence_role: str = "sequence",
) -> bool:
    try:
        provenance = load_unigram_provenance(tokenizer_path)
        inspection = inspect_unigram_artifact(
            tokenizer_path,
            expected_sha256=provenance.get("artifact_sha256"),
        )
        validate_unigram_provenance(provenance, inspection)
    except (UnigramContractError, OSError, TypeError, ValueError):
        return False
    trainer = provenance.get("trainer") or {}
    lengths = provenance.get("lengths") or {}
    runtime = provenance.get("runtime") or {}
    return bool(
        provenance.get("sequence_role") == str(sequence_role or "sequence")
        and provenance.get("training_digest") == unigram_training_digest(normalized_sequences)
        and provenance.get("training_sequences") == len(normalized_sequences)
        and trainer.get("vocab_size") == int(vocab_size)
        and lengths.get("max_raw_length") == validate_unigram_max_raw_length(max_raw_length)
        and runtime.get("tokenizers_version") == str(tokenizers_version)
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


def bundle_unigram_artifact(
    tokenizer_path: str | os.PathLike[str],
    artifact_dir: str | os.PathLike[str],
    *,
    role: str | None = None,
) -> dict:
    source = Path(tokenizer_path)
    source_provenance = unigram_provenance_path(source)
    provenance = load_unigram_provenance(source)
    inspection = inspect_unigram_artifact(
        source,
        expected_sha256=provenance.get("artifact_sha256"),
    )
    validate_unigram_provenance(provenance, inspection)
    normalized_role = str(role or "").strip().lower()
    stem = {
        "peptide": "peptide_unigram",
        "partner": "partner_unigram",
        "protein": "partner_unigram",
    }.get(normalized_role, "unigram")
    bundle_root = Path(artifact_dir)
    relative_tokenizer = Path("tokenizers") / f"{stem}.json"
    # Keep the sidecar adjacent under the same convention used by
    # ``unigram_provenance_path`` so bundled artifacts can be loaded directly
    # by UnigramTokenizer without an out-of-band provenance override.
    relative_provenance = Path("tokenizers") / f"{stem}.json.provenance.json"
    _atomic_copy(source, bundle_root / relative_tokenizer)
    _atomic_copy(source_provenance, bundle_root / relative_provenance)
    inspect_unigram_artifact(
        bundle_root / relative_tokenizer,
        expected_sha256=inspection["sha256"],
    )
    descriptor = {
        "contract_version": UNIGRAM_CONTRACT_VERSION,
        "path": relative_tokenizer.as_posix(),
        "sha256": inspection["sha256"],
        "provenance_path": relative_provenance.as_posix(),
        "model_type": "Unigram",
        "variant": UNIGRAM_VARIANT,
        "vocab_size": inspection["vocab_size"],
        "max_raw_length": int(
            (provenance.get("lengths") or {}).get(
                "max_raw_length",
                UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
        "tokenizers_version": str((provenance.get("runtime") or {}).get("tokenizers_version", "")),
    }
    return {
        "unigram_path": relative_tokenizer.as_posix(),
        "unigram_artifact": descriptor,
    }


class _UnigramLatticePath:
    """Compact persistent backpointer for one lattice path."""

    __slots__ = ("score", "token_id", "next")

    def __init__(self, score, token_id, next_path):
        self.score = float(score)
        self.token_id = token_id
        self.next = next_path


class SeededUnigramSampler:
    """Stable finite-n-best sampler over the model's Unigram lattice."""

    _TERMINALS = object()

    def __init__(
        self,
        tokenizer_path: str | os.PathLike[str],
        *,
        max_raw_length: int = UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    ):
        inspect_unigram_artifact(tokenizer_path)
        payload = _load_json_object(
            Path(tokenizer_path),
            description="Unigram tokenizer artifact",
        )
        self.vocab = _unigram_vocab(payload, Path(tokenizer_path))
        self.max_raw_length = validate_unigram_max_raw_length(max_raw_length)
        self.root: dict = {}
        for token_id, (token, score) in enumerate(self.vocab):
            if token in UNIGRAM_SPECIAL_TOKEN_IDS:
                continue
            node = self.root
            for character in token:
                node = node.setdefault(character, {})
            node.setdefault(self._TERMINALS, []).append((token_id, score))
        for node in self._walk_nodes(self.root):
            terminals = node.get(self._TERMINALS)
            if terminals:
                terminals.sort(key=lambda item: item[0])

    @classmethod
    def _walk_nodes(cls, node):
        yield node
        for key, child in node.items():
            if key is not cls._TERMINALS:
                yield from cls._walk_nodes(child)

    def nbest_paths(
        self,
        sequence: object,
        *,
        nbest_size: int = UNIGRAM_DEFAULT_NBEST_SIZE,
    ) -> list[tuple[float, tuple[int, ...]]]:
        normalized_nbest, _ = validate_unigram_sampling(nbest_size, 1.0)
        normalized = normalize_unigram_sequence(
            sequence,
            max_raw_length=self.max_raw_length,
        )
        paths: list[list[_UnigramLatticePath] | None] = [None for _ in range(len(normalized) + 1)]
        paths[len(normalized)] = [_UnigramLatticePath(0.0, None, None)]
        for start in range(len(normalized) - 1, -1, -1):
            node = self.root
            candidates: list[tuple[_UnigramLatticePath, int]] = []
            for end in range(start, len(normalized)):
                node = node.get(normalized[end])
                if node is None:
                    break
                terminals = node.get(self._TERMINALS, ())
                suffixes = paths[end + 1]
                if not terminals or not suffixes:
                    continue
                for token_id, token_score in terminals:
                    for suffix_rank, suffix in enumerate(suffixes):
                        candidates.append(
                            (
                                _UnigramLatticePath(
                                    token_score + suffix.score,
                                    token_id,
                                    suffix,
                                ),
                                suffix_rank,
                            )
                        )
            # For equal total scores, token_id is the first lexicographic
            # component. Equal token_ids imply the same piece end, so the
            # already ordered suffix rank completes the exact token-id
            # lexicographic tie-break without copying every suffix tuple.
            candidates.sort(
                key=lambda item: (
                    -item[0].score,
                    item[0].token_id,
                    item[1],
                )
            )
            paths[start] = [candidate for candidate, _ in candidates[:normalized_nbest]]
            no_longer_needed = start + UNIGRAM_MAX_PIECE_LENGTH
            if no_longer_needed < len(paths):
                paths[no_longer_needed] = None
        if not paths[0]:
            raise UnigramContractError(
                "Unigram vocabulary cannot segment a valid normalized sequence; "
                "retrain the tokenizer",
                code="unigram_vocabulary_incomplete",
            )
        results = []
        for head in paths[0]:
            token_ids = []
            current = head
            while current.token_id is not None:
                token_ids.append(current.token_id)
                current = current.next
            results.append((head.score, tuple(token_ids)))

        # Break the retained persistent chains iteratively. This avoids a deep
        # reference-count deallocation cascade for very long protein inputs.
        for bucket in paths:
            if not bucket:
                continue
            for head in bucket:
                current = head
                while current is not None:
                    next_path = current.next
                    current.next = None
                    current = next_path
        return results

    @staticmethod
    def _uniform_from_context(
        *,
        seed: int,
        epoch: int,
        sample_id: object,
        role: str,
        normalized_sequence: str,
    ) -> float:
        if sample_id is None or not str(sample_id):
            raise UnigramContractError(
                "sample_id is required for strictly reproducible Unigram sampling",
                code="unigram_sampling_invalid",
            )
        payload = {
            "sampling_version": UNIGRAM_SAMPLING_VERSION,
            "seed": int(seed),
            "epoch": int(epoch),
            "sample_id": str(sample_id),
            "role": str(role or "sequence"),
            "sequence_sha256": hashlib.sha256(normalized_sequence.encode("ascii")).hexdigest(),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        value = int.from_bytes(hashlib.sha256(encoded).digest(), "big")
        return value / float(1 << 256)

    def sample_ids(
        self,
        sequence: object,
        *,
        seed: int,
        epoch: int,
        sample_id: object,
        role: str = "sequence",
        nbest_size: int = UNIGRAM_DEFAULT_NBEST_SIZE,
        alpha: float = UNIGRAM_DEFAULT_ALPHA,
    ) -> list[int]:
        normalized_nbest, normalized_alpha = validate_unigram_sampling(
            nbest_size,
            alpha,
        )
        normalized = normalize_unigram_sequence(
            sequence,
            max_raw_length=self.max_raw_length,
        )
        paths = self.nbest_paths(normalized, nbest_size=normalized_nbest)
        maximum = max(score for score, _ in paths)
        weights = [math.exp(normalized_alpha * (score - maximum)) for score, _ in paths]
        total = math.fsum(weights)
        if not math.isfinite(total) or total <= 0.0:
            raise UnigramContractError(
                "Unigram sampling weights are not finite and positive",
                code="unigram_sampling_invalid",
            )
        threshold = (
            self._uniform_from_context(
                seed=int(seed),
                epoch=int(epoch),
                sample_id=sample_id,
                role=role,
                normalized_sequence=normalized,
            )
            * total
        )
        cumulative = 0.0
        for weight, (_, token_ids) in zip(weights, paths):
            cumulative += weight
            if threshold < cumulative:
                return list(token_ids)
        return list(paths[-1][1])
