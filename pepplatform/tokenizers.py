import os
import itertools
import re
import json
import math
import shutil
import tempfile
from numbers import Integral
import torch

try:
    import tokenizers as tokenizers_package
    from tokenizers import Tokenizer
except ImportError:
    tokenizers_package = None
    Tokenizer = None
try:
    from transformers import AutoTokenizer, T5Tokenizer, AutoModelForMaskedLM
except ImportError:
    AutoTokenizer = None
    T5Tokenizer = None
    AutoModelForMaskedLM = None
from pepplatform.dataset import AA_VOCAB
from pepplatform.esmc import (
    ESMC_BACKEND_OFFICIAL,
    ESMC_OFFICIAL_600M_MODEL_ID,
    ESMC_RUNTIME_INSTALL_COMMAND,
    ESMC_VALID_SEQUENCE_AA,
    esmc_backend_for_source,
    require_official_esmc_runtime,
)
from pepplatform.esm2 import (
    ESM2_HF_REVISION,
    ESM2_OFFICIAL_MODEL_ID,
    ESM2_VALID_SEQUENCE_AA,
    is_complete_esm2_snapshot,
    normalize_esm2_sequence,
    validate_esm2_max_residues,
)
from pepplatform.esm3 import (
    ESM3_OFFICIAL_MODEL_ID,
    esm3_residue_count,
    is_complete_esm3_source,
    normalize_esm3_model_name,
    normalize_esm3_sequence,
    require_official_esm3_runtime,
    truncate_esm3_sequence,
    validate_esm3_max_residues,
)
from pepplatform.protbert import (
    PROTBERT_OFFICIAL_MODEL_ID,
    is_complete_protbert_snapshot,
    normalize_protbert_sequence,
    validate_protbert_max_residues,
)
from pepplatform.prott5 import (
    PROTT5_OFFICIAL_MODEL_ID,
    PROTT5_HF_REVISION,
    is_complete_prott5_snapshot,
    normalize_prott5_sequence,
    validate_prott5_max_residues,
    validate_prott5_tokenizer,
)
from pepplatform.kmer import validate_kmer_k
from pepplatform.frequency import (
    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    FREQUENCY_MAX_RAW_LENGTH,
    normalize_frequency_sequence,
    validate_frequency_max_raw_length,
)
from pepplatform.dr import (
    DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
    DR_DEFAULT_MAX_DISTANCE,
    DR_DEFAULT_MAX_RAW_LENGTH,
    dr_feature_tokens,
    iter_dr_tokens,
    normalize_dr_sequence,
    validate_dr_max_distance,
    validate_dr_max_raw_length,
    validate_dr_unknown_residue_policy,
)
from pepplatform.bpe import (
    BPE_ALPHABET,
    BPE_MIN_FREQUENCY,
    BPE_MIN_VOCAB_SIZE,
    BPE_SPECIAL_TOKEN_IDS,
    BPE_SPECIAL_TOKENS,
    BPE_TOKENIZERS_VERSION,
    atomic_write_json,
    bpe_cache_is_current,
    bpe_provenance_path,
    build_bpe_provenance,
    inspect_bpe_artifact,
    normalize_bpe_sequence,
    normalize_bpe_sequences,
)
from pepplatform.wordpiece import (
    WORDPIECE_ALPHABET,
    WORDPIECE_CONTINUING_SUBWORD_PREFIX,
    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
    WORDPIECE_SPECIAL_TOKEN_IDS,
    WORDPIECE_SPECIAL_TOKENS,
    WORDPIECE_TOKENIZERS_VERSION,
    WordPieceContractError,
    atomic_write_json as atomic_write_wordpiece_json,
    build_wordpiece_provenance,
    inspect_wordpiece_artifact,
    normalize_wordpiece_sequence,
    normalize_wordpiece_sequences,
    validate_wordpiece_max_raw_length,
    validate_wordpiece_vocab_size,
    wordpiece_cache_is_current,
    wordpiece_provenance_path,
)
from pepplatform.unigram import (
    UNIGRAM_ALPHABET,
    UNIGRAM_DEFAULT_ALPHA,
    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    UNIGRAM_DEFAULT_NBEST_SIZE,
    UNIGRAM_MAX_PIECE_LENGTH,
    UNIGRAM_N_SUB_ITERATIONS,
    UNIGRAM_SCORE_DECIMALS,
    UNIGRAM_SEED_SIZE,
    UNIGRAM_SHRINKING_FACTOR,
    UNIGRAM_SINGLE_SCORE_TOLERANCE,
    UNIGRAM_SPECIAL_TOKEN_IDS,
    UNIGRAM_SPECIAL_TOKENS,
    UNIGRAM_TOKENIZERS_VERSION,
    SeededUnigramSampler,
    UnigramContractError,
    atomic_write_json as atomic_write_unigram_json,
    build_unigram_provenance,
    inspect_unigram_artifact,
    load_unigram_provenance,
    normalize_unigram_sequence,
    normalize_unigram_sequences,
    unigram_cache_is_current,
    unigram_provenance_path,
    validate_unigram_max_raw_length,
    validate_unigram_provenance,
    validate_unigram_vocab_size,
)
from pepplatform.top_ngram import (
    TOP_N_GRAM_ALPHABET,
    TargetFrequencyProfileStore,
    TopNGramContractError,
    stable_top_residue_indices,
    top_ngram_feature_dim,
    top_residue_ids_to_feature_ids,
    validate_top_ngram_n,
)
from pepplatform.dt import (
    DT_ALPHABET,
    DT_DEFAULT_MAX_DISTANCE,
    DTContractError,
    dt_feature_dim,
    dt_feature_index,
    validate_dt_max_distance,
)


class BaseTokenizer:
    def __init__(self, max_length=60):
        self.max_length = max_length

    def tokenize(self, sequence: str):
        raise NotImplementedError


def _batch_tokenize_residues(tokenizer, sequences, max_residues, device=None, normalize=None):
    """Tokenize one-token-per-residue inputs while retaining special-token metadata."""
    max_residues = int(max_residues)
    if max_residues <= 0:
        raise ValueError("max_residues must be positive")
    prepared = [str(sequence or "")[:max_residues] for sequence in sequences]
    if normalize is not None:
        prepared = [normalize(sequence) for sequence in prepared]
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    inputs = tokenizer(
        prepared,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_residues + special_count,
        return_special_tokens_mask=True,
    )
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs


class KmerTokenizer(BaseTokenizer):
    """Contiguous, overlapping, stride-1 K-mer tokenizer.

    ``max_length`` limits emitted K-mer tokens, not source residues.
    """

    def __init__(
        self,
        max_length=60,
        k=2,
        frequency_max_raw_length=FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    ):
        super().__init__(max_length=max_length)
        self.k = validate_kmer_k(k)
        self.frequency_max_raw_length = validate_frequency_max_raw_length(frequency_max_raw_length)
        aa_tokens = [aa for aa in AA_VOCAB.keys() if aa != "<PAD>"]
        aa_tokens = [aa for aa in aa_tokens if len(aa) == 1]
        kmers = ["".join(p) for p in itertools.product(aa_tokens, repeat=self.k)]
        self.kmer_vocab = {"<PAD>": 0, "<UNK>": 1}
        for i, kmer in enumerate(kmers, start=2):
            self.kmer_vocab[kmer] = i
        self.vocab_size = len(self.kmer_vocab)

    def tokenize(self, sequence: str):
        sequence = sequence.upper()
        kmers = []
        if len(sequence) >= self.k:
            for i in range(len(sequence) - self.k + 1):
                kmers.append(sequence[i : i + self.k])
        kmers = kmers[: self.max_length]
        return kmers

    def to_indices(self, kmers):
        indices = [self.kmer_vocab.get(kmer, self.kmer_vocab["<UNK>"]) for kmer in kmers]
        padding_len = self.max_length - len(indices)
        if padding_len > 0:
            indices.extend([self.kmer_vocab["<PAD>"]] * padding_len)
        return indices

    def iter_frequency_tokens(self, sequence: str):
        sequence = normalize_frequency_sequence(
            sequence,
            max_raw_length=self.frequency_max_raw_length,
        )
        for index in range(max(0, len(sequence) - self.k + 1)):
            yield sequence[index : index + self.k]

    def iter_frequency_feature_ids(self, sequence: str):
        unknown = self.kmer_vocab["<UNK>"]
        for token in self.iter_frequency_tokens(sequence):
            yield self.kmer_vocab.get(token, unknown)


class DRTokenizer(BaseTokenizer):
    """Distance-based Residue tokenizer using the platform DR v2 contract.

    The complete token stream contains distance-zero residue words and all
    valid directed residue pairs up to ``max_distance``. ``max_length`` only
    limits ordered encoders through :meth:`to_indices`; global DR encoders use
    the complete normalized sequence.
    """

    def __init__(
        self,
        max_length=60,
        max_distance=DR_DEFAULT_MAX_DISTANCE,
        max_raw_length=DR_DEFAULT_MAX_RAW_LENGTH,
        unknown_residue_policy=DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
    ):
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, Integral)
            or int(max_length) < 1
        ):
            raise ValueError(
                f"DR tokenizer max_length must be a positive integer, got {max_length!r}"
            )
        super().__init__(max_length=int(max_length))
        self.max_distance = validate_dr_max_distance(max_distance)
        self.max_raw_length = validate_dr_max_raw_length(max_raw_length)
        self.unknown_residue_policy = validate_dr_unknown_residue_policy(unknown_residue_policy)
        feature_tokens = dr_feature_tokens(self.max_distance)
        self.feature_tokens = feature_tokens
        self.feature_to_index = {token: index for index, token in enumerate(feature_tokens)}
        self.feature_dim = len(feature_tokens)
        self.dr_vocab = {"<PAD>": 0, "<UNK>": 1}
        self.dr_vocab.update({token: index for index, token in enumerate(feature_tokens, start=2)})
        self.vocab_size = len(self.dr_vocab)

    def normalize_sequence(self, sequence: str) -> str:
        return normalize_dr_sequence(
            sequence,
            max_raw_length=self.max_raw_length,
            unknown_residue_policy=self.unknown_residue_policy,
        )

    def iter_tokens(self, sequence: str):
        normalized = self.normalize_sequence(sequence)
        return iter_dr_tokens(
            normalized,
            self.max_distance,
            unknown_residue_policy=self.unknown_residue_policy,
        )

    def tokenize(self, sequence: str):
        return list(self.iter_tokens(sequence))

    def tokenize_for_ordered_encoder(self, sequence: str):
        return list(
            itertools.islice(
                self.iter_tokens(sequence),
                self.max_length,
            )
        )

    def to_indices(self, tokens):
        indices = [
            self.dr_vocab.get(token, self.dr_vocab["<UNK>"]) for token in tokens[: self.max_length]
        ]
        padding_len = self.max_length - len(indices)
        if padding_len > 0:
            indices.extend([self.dr_vocab["<PAD>"]] * padding_len)
        return indices

    def to_feature_indices(self, tokens):
        return [self.feature_to_index[token] for token in tokens if token in self.feature_to_index]


class TopNGramTokenizer(BaseTokenizer):
    """Paper-aligned Top-n-gram tokenizer over target-frequency profiles.

    ``max_length`` is a hard source-residue limit. It never truncates an
    otherwise valid profile. Global Top-n-gram encoders consume one token per
    real residue and use a feature space of exactly ``20 ** top_n``.
    """

    def __init__(
        self,
        index_path,
        max_length=60,
        top_n=2,
        strict=True,
        frequency_max_raw_length=None,
    ):
        if not strict:
            raise TopNGramContractError(
                "Top-n-gram v3 requires a validated profile manifest; "
                "fallback profiles must be authorized by its provenance",
                code="top_ngram_strict_mode_required",
            )
        super().__init__(max_length=max_length)
        self.index_path = os.fspath(index_path)
        self.top_n = validate_top_ngram_n(top_n)
        self.strict = True
        self.aa_tokens = TOP_N_GRAM_ALPHABET
        self.feature_dim = top_ngram_feature_dim(self.top_n)
        self.vocab_size = self.feature_dim
        if frequency_max_raw_length is None:
            profile_max_length = max_length
            profile_platform_limit = None
        else:
            profile_max_length = validate_frequency_max_raw_length(frequency_max_raw_length)
            profile_platform_limit = FREQUENCY_MAX_RAW_LENGTH
        self.profile_store = TargetFrequencyProfileStore(
            self.index_path,
            max_length=profile_max_length,
            platform_limit=profile_platform_limit,
        )

    def load_profile(self, sequence: str, sample_id=None):
        return self.profile_store.load(sample_id, sequence)

    def residue_indices(self, sequence: str, sample_id=None):
        matrix = self.load_profile(sequence, sample_id=sample_id)
        return stable_top_residue_indices(matrix.numpy(), self.top_n)

    def feature_ids(self, sequence: str, sample_id=None):
        return top_residue_ids_to_feature_ids(self.residue_indices(sequence, sample_id=sample_id))

    def tokenize(self, sequence: str, sample_id=None):
        rows = self.residue_indices(sequence, sample_id=sample_id)
        return ["".join(self.aa_tokens[index] for index in row) for row in rows.tolist()]

    def to_indices(self, tokens):
        ids = []
        for token in tokens:
            if not isinstance(token, str) or len(token) != self.top_n:
                raise TopNGramContractError(
                    f"invalid Top-n-gram token {token!r}",
                    code="top_ngram_token_value_invalid",
                )
            feature_id = 0
            for residue in token:
                try:
                    residue_id = self.aa_tokens.index(residue)
                except ValueError as exc:
                    raise TopNGramContractError(
                        f"invalid Top-n-gram token {token!r}",
                        code="top_ngram_token_value_invalid",
                    ) from exc
                feature_id = feature_id * len(self.aa_tokens) + residue_id
            ids.append(feature_id)
        return ids


class DTTokenizer(BaseTokenizer):
    """Paper-aligned Distance-based Top-1-gram tokenizer over current profiles."""

    def __init__(
        self,
        index_path,
        max_length=60,
        max_distance=DT_DEFAULT_MAX_DISTANCE,
        strict=True,
        frequency_max_raw_length=None,
    ):
        if not strict:
            raise DTContractError(
                "DT v1 requires a validated target-frequency profile manifest",
                code="dt_strict_mode_required",
            )
        super().__init__(max_length=max_length)
        self.index_path = os.fspath(index_path)
        self.max_distance = validate_dt_max_distance(max_distance)
        self.feature_dim = dt_feature_dim(self.max_distance)
        self.vocab_size = self.feature_dim
        if frequency_max_raw_length is None:
            profile_max_length = max_length
            profile_platform_limit = None
        else:
            profile_max_length = validate_frequency_max_raw_length(frequency_max_raw_length)
            profile_platform_limit = FREQUENCY_MAX_RAW_LENGTH
        self.profile_store = TargetFrequencyProfileStore(
            self.index_path,
            max_length=profile_max_length,
            platform_limit=profile_platform_limit,
        )
        self._profile_to_dt = {
            profile_index: DT_ALPHABET.index(residue)
            for profile_index, residue in enumerate(TOP_N_GRAM_ALPHABET)
        }

    def residue_ids(self, sequence: str, sample_id=None):
        matrix = self.profile_store.load(sample_id, sequence)
        profile_ids = stable_top_residue_indices(matrix.numpy(), 1)[:, 0]
        return [self._profile_to_dt[int(index)] for index in profile_ids]

    def iter_feature_ids(self, sequence: str, sample_id=None):
        residue_ids = self.residue_ids(sequence, sample_id=sample_id)
        alphabet_size = len(DT_ALPHABET)
        length = len(residue_ids)
        for position, first in enumerate(residue_ids):
            yield first
            upper = min(self.max_distance, length - position - 1)
            for distance in range(1, upper + 1):
                second = residue_ids[position + distance]
                yield (
                    alphabet_size
                    + (distance - 1) * alphabet_size * alphabet_size
                    + first * alphabet_size
                    + second
                )

    def feature_ids(self, sequence: str, sample_id=None):
        return list(self.iter_feature_ids(sequence, sample_id=sample_id))

    def tokenize(self, sequence: str, sample_id=None):
        residue_ids = self.residue_ids(sequence, sample_id=sample_id)
        residues = [DT_ALPHABET[index] for index in residue_ids]
        tokens = []
        for position, first in enumerate(residues):
            tokens.append(first)
            upper = min(self.max_distance, len(residues) - position - 1)
            for distance in range(1, upper + 1):
                tokens.append(f"{first}_{distance}_{residues[position + distance]}")
        return tokens

    def to_indices(self, tokens):
        return [dt_feature_index(token, max_distance=self.max_distance) for token in tokens]


class BPETokenizer(BaseTokenizer):
    def __init__(self, tokenizer_path, max_length=60):
        super().__init__(max_length=max_length)
        if Tokenizer is None:
            raise ImportError(
                f"BPE requires tokenizers=={BPE_TOKENIZERS_VERSION}. "
                f"Run `pip install tokenizers=={BPE_TOKENIZERS_VERSION}`."
            )
        runtime_version = str(getattr(tokenizers_package, "__version__", "") or "")
        if runtime_version != BPE_TOKENIZERS_VERSION:
            raise ImportError(
                f"BPE requires tokenizers=={BPE_TOKENIZERS_VERSION}; "
                f"found {runtime_version or 'unknown'}."
            )
        inspect_bpe_artifact(tokenizer_path)
        self.tokenizer = Tokenizer.from_file(os.fspath(tokenizer_path))
        self.tokenizer.enable_padding(
            length=max_length,
            pad_id=BPE_SPECIAL_TOKEN_IDS["<PAD>"],
            pad_token="<PAD>",
        )
        self.tokenizer.enable_truncation(max_length=max_length)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.tokenizer_path = str(tokenizer_path)

    def tokenize(self, sequence: str):
        normalized = normalize_bpe_sequence(sequence)
        encoding = self.tokenizer.encode(normalized)
        ids = encoding.ids
        return ids


class UnigramTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer_path,
        max_length=60,
        max_raw_length=UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    ):
        max_length = int(max_length)
        if max_length < 2:
            raise UnigramContractError(
                "Unigram max_length must be >= 2 because it includes <CLS>/<SEP>",
                code="unigram_length_invalid",
            )
        super().__init__(max_length=max_length)
        if Tokenizer is None:
            raise ImportError(
                f"UnigramTokenizer requires tokenizers=={UNIGRAM_TOKENIZERS_VERSION}. "
                f"Run `pip install tokenizers=={UNIGRAM_TOKENIZERS_VERSION}`."
            )
        runtime_version = str(getattr(tokenizers_package, "__version__", "") or "")
        if runtime_version != UNIGRAM_TOKENIZERS_VERSION:
            raise ImportError(
                f"UnigramTokenizer requires tokenizers=={UNIGRAM_TOKENIZERS_VERSION}; "
                f"found {runtime_version or 'unknown'}."
            )
        self.max_raw_length = validate_unigram_max_raw_length(max_raw_length)
        inspection = inspect_unigram_artifact(tokenizer_path)
        provenance = load_unigram_provenance(tokenizer_path)
        validate_unigram_provenance(provenance, inspection)
        if (provenance.get("lengths") or {}).get("max_raw_length") != self.max_raw_length:
            raise UnigramContractError(
                "Configured unigram_max_raw_length does not match tokenizer provenance",
                code="unigram_length_invalid",
            )
        self.tokenizer = Tokenizer.from_file(os.fspath(tokenizer_path))
        self.tokenizer.enable_padding(
            length=max_length,
            pad_id=UNIGRAM_SPECIAL_TOKEN_IDS["<PAD>"],
            pad_token="<PAD>",
        )
        self.tokenizer.enable_truncation(max_length=max_length)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.tokenizer_path = str(tokenizer_path)
        self.sampler = SeededUnigramSampler(
            tokenizer_path,
            max_raw_length=self.max_raw_length,
        )

    def tokenize(self, sequence: str):
        normalized = normalize_unigram_sequence(
            sequence,
            max_raw_length=self.max_raw_length,
        )
        ids = self.tokenizer.encode(normalized).ids
        if UNIGRAM_SPECIAL_TOKEN_IDS["<UNK>"] in ids:
            raise UnigramContractError(
                "Unigram produced <UNK> for a valid normalized sequence; retrain the tokenizer",
                code="unigram_vocabulary_incomplete",
            )
        return ids

    def tokenize_sampled(
        self,
        sequence: str,
        *,
        seed: int,
        epoch: int,
        sample_id,
        role="sequence",
        nbest_size=UNIGRAM_DEFAULT_NBEST_SIZE,
        alpha=UNIGRAM_DEFAULT_ALPHA,
    ):
        content = self.sampler.sample_ids(
            sequence,
            seed=seed,
            epoch=epoch,
            sample_id=sample_id,
            role=role,
            nbest_size=nbest_size,
            alpha=alpha,
        )
        content = content[: self.max_length - 2]
        ids = [
            UNIGRAM_SPECIAL_TOKEN_IDS["<CLS>"],
            *content,
            UNIGRAM_SPECIAL_TOKEN_IDS["<SEP>"],
        ]
        ids.extend([UNIGRAM_SPECIAL_TOKEN_IDS["<PAD>"]] * (self.max_length - len(ids)))
        if UNIGRAM_SPECIAL_TOKEN_IDS["<UNK>"] in ids:
            raise UnigramContractError(
                "Unigram sampler produced <UNK> for a valid normalized sequence",
                code="unigram_vocabulary_incomplete",
            )
        return ids


class _WordPieceTrie:
    """Vocabulary trie implementing greedy longest-prefix lookup."""

    _TOKEN_ID = None

    def __init__(self):
        self.root = {}

    def add(self, token, token_id):
        node = self.root
        for character in token:
            node = node.setdefault(character, {})
        node[self._TOKEN_ID] = int(token_id)

    def longest_match(self, sequence, start):
        node = self.root
        best = None
        position = start
        while position < len(sequence):
            node = node.get(sequence[position])
            if node is None:
                break
            position += 1
            token_id = node.get(self._TOKEN_ID)
            if token_id is not None:
                best = (token_id, position)
        return best


class WordPieceTokenizer(BaseTokenizer):
    """Strict protein WordPiece with bounded, trie-based longest matching."""

    def __init__(
        self,
        tokenizer_path,
        max_length=60,
        max_raw_length=WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
    ):
        max_length = int(max_length)
        if max_length < 2:
            raise WordPieceContractError(
                "WordPiece max_length must be >= 2 because it includes <CLS>/<SEP>",
                code="wordpiece_length_invalid",
            )
        super().__init__(max_length=max_length)
        if Tokenizer is None:
            raise ImportError(
                f"WordPiece requires tokenizers=={WORDPIECE_TOKENIZERS_VERSION}. "
                f"Run `pip install tokenizers=={WORDPIECE_TOKENIZERS_VERSION}`."
            )
        runtime_version = str(getattr(tokenizers_package, "__version__", "") or "")
        if runtime_version != WORDPIECE_TOKENIZERS_VERSION:
            raise ImportError(
                f"WordPiece requires tokenizers=={WORDPIECE_TOKENIZERS_VERSION}; "
                f"found {runtime_version or 'unknown'}."
            )
        self.max_raw_length = validate_wordpiece_max_raw_length(max_raw_length)
        inspect_wordpiece_artifact(tokenizer_path)
        self.tokenizer = Tokenizer.from_file(os.fspath(tokenizer_path))
        self.tokenizer.model.max_input_chars_per_word = self.max_raw_length
        self.tokenizer.enable_padding(
            length=max_length,
            pad_id=WORDPIECE_SPECIAL_TOKEN_IDS["<PAD>"],
            pad_token="<PAD>",
        )
        self.tokenizer.enable_truncation(max_length=max_length)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.tokenizer_path = str(tokenizer_path)
        self._initial_trie = _WordPieceTrie()
        self._continuation_trie = _WordPieceTrie()
        self._build_tries()

    def _build_tries(self):
        alphabet = set(WORDPIECE_ALPHABET)
        prefix = WORDPIECE_CONTINUING_SUBWORD_PREFIX
        vocab = self.tokenizer.get_vocab(with_added_tokens=False)
        for token, token_id in vocab.items():
            if token in WORDPIECE_SPECIAL_TOKEN_IDS:
                continue
            if token.startswith(prefix):
                piece = token[len(prefix) :]
                if piece and set(piece).issubset(alphabet):
                    self._continuation_trie.add(piece, token_id)
            elif token and set(token).issubset(alphabet):
                self._initial_trie.add(token, token_id)

    def tokenize(self, sequence: str):
        normalized = normalize_wordpiece_sequence(
            sequence,
            max_raw_length=self.max_raw_length,
        )
        content_budget = self.max_length - 2
        ids = [WORDPIECE_SPECIAL_TOKEN_IDS["<CLS>"]]
        start = 0
        while start < len(normalized) and len(ids) - 1 < content_budget:
            trie = self._initial_trie if start == 0 else self._continuation_trie
            match = trie.longest_match(normalized, start)
            if match is None:
                raise WordPieceContractError(
                    "WordPiece vocabulary cannot segment a valid normalized "
                    f"sequence at residue offset {start}; retrain the tokenizer",
                    code="wordpiece_vocabulary_incomplete",
                )
            token_id, start = match
            if token_id == WORDPIECE_SPECIAL_TOKEN_IDS["<UNK>"]:
                raise WordPieceContractError(
                    "WordPiece produced <UNK> for a valid normalized sequence; "
                    "retrain the tokenizer",
                    code="wordpiece_vocabulary_incomplete",
                )
            ids.append(token_id)
        ids.append(WORDPIECE_SPECIAL_TOKEN_IDS["<SEP>"])
        padding = self.max_length - len(ids)
        if padding > 0:
            ids.extend([WORDPIECE_SPECIAL_TOKEN_IDS["<PAD>"]] * padding)
        return ids


class ProtBertTokenizer(BaseTokenizer):
    def __init__(
        self,
        model_name=PROTBERT_OFFICIAL_MODEL_ID,
        max_length=60,
        *,
        legacy_total_token_length=False,
    ):
        super().__init__(max_length=max_length)
        if AutoTokenizer is None:
            raise ImportError(
                "ProtBertTokenizer requires 'transformers' library. Run `pip install transformers`."
            )
        self.model_name = str(model_name or PROTBERT_OFFICIAL_MODEL_ID).strip()
        self.legacy_total_token_length = bool(legacy_total_token_length)
        if not self.model_name:
            raise ValueError("ProtBERT model source must not be empty")
        if os.path.isdir(self.model_name) and not is_complete_protbert_snapshot(self.model_name):
            raise ValueError(
                "ProtBERT local source is incomplete or has the wrong architecture signature: "
                f"{self.model_name}"
            )
        kwargs = {"do_lower_case": False}
        if not os.path.isdir(self.model_name) and self.model_name == PROTBERT_OFFICIAL_MODEL_ID:
            from pepplatform.protbert import PROTBERT_HF_REVISION

            kwargs["revision"] = PROTBERT_HF_REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        if self.legacy_total_token_length:
            self.max_length = int(max_length)
            if self.max_length < 2:
                raise ValueError("ProtBERT legacy max_length must be at least 2 tokens")
        else:
            self.max_length = validate_protbert_max_residues(max_length)

    def normalize_sequence(self, sequence: str):
        return " ".join(list(normalize_protbert_sequence(sequence)))

    def residue_length(self, sequence: str, max_residues=None):
        normalized = normalize_protbert_sequence(sequence)
        if max_residues is not None:
            normalized = normalized[
                : validate_protbert_max_residues(max_residues, field="max_residues")
            ]
        return len(normalized)

    def _prepare(self, sequences):
        if self.legacy_total_token_length:
            prepared = [self.normalize_sequence(value) for value in sequences]
        else:
            prepared = [
                self.normalize_sequence(normalize_protbert_sequence(value)[: self.max_length])
                for value in sequences
            ]
        if any(not value for value in prepared):
            raise ValueError("ProtBERT sequences must contain at least one residue")
        return prepared

    def _tokenize(self, sequences):
        prepared = self._prepare(sequences)
        special_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        token_limit = (
            self.max_length if self.legacy_total_token_length else self.max_length + special_count
        )
        return self.tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=token_limit,
            return_special_tokens_mask=True,
        )

    def tokenize(self, sequence: str):
        return self._tokenize([sequence])

    def batch_tokenize(self, sequences, device=None):
        inputs = self._tokenize(sequences)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def batch_tokenize_residues(self, sequences, max_residues, device=None):
        max_residues = validate_protbert_max_residues(max_residues, field="max_residues")
        normalized = [
            self.normalize_sequence(normalize_protbert_sequence(sequence)[:max_residues])
            for sequence in sequences
        ]
        special_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        inputs = self.tokenizer(
            normalized,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_residues + special_count,
            return_special_tokens_mask=True,
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs


ProBertTokenizer = ProtBertTokenizer


class ProT5Tokenizer(BaseTokenizer):
    """ProtT5 tokenizer with residue-length and alignment semantics.

    ``max_length`` counts residues, with extra room for EOS. With
    ``legacy_total_token_length=True``, it counts all tokens including EOS.
    Both modes retain the special-token mask.
    """

    def __init__(
        self,
        model_name="Rostlab/prot_t5_xl_uniref50",
        max_length=60,
        *,
        legacy_total_token_length=False,
    ):
        self.legacy_total_token_length = bool(legacy_total_token_length)
        if self.legacy_total_token_length:
            if (
                isinstance(max_length, bool)
                or not isinstance(max_length, Integral)
                or int(max_length) < 2
            ):
                raise ValueError("ProtT5 legacy max_length must be at least 2 tokens")
            validated_max_length = int(max_length)
        else:
            validated_max_length = validate_prott5_max_residues(max_length)
        super().__init__(max_length=validated_max_length)
        if AutoTokenizer is None and T5Tokenizer is None:
            raise ImportError(
                "ProT5Tokenizer requires 'transformers' library. Run `pip install transformers`."
            )
        self.model_name = str(model_name or PROTT5_OFFICIAL_MODEL_ID).strip()
        if not self.model_name:
            raise ValueError("ProtT5 model source must not be empty")
        if os.path.isdir(self.model_name) and not is_complete_prott5_snapshot(self.model_name):
            raise ValueError(
                "ProtT5 local source is incomplete or has the wrong architecture signature: "
                f"{self.model_name}"
            )
        kwargs = {}
        if not os.path.isdir(self.model_name) and self.model_name == PROTT5_OFFICIAL_MODEL_ID:
            kwargs["revision"] = PROTT5_HF_REVISION
        try:
            if T5Tokenizer is not None:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    self.model_name, do_lower_case=False, **kwargs
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, do_lower_case=False, use_fast=False, **kwargs
                )
        except Exception as e:
            raise RuntimeError(
                f"ProT5Tokenizer failed to load model source: {self.model_name}"
            ) from e
        # Record the tokenizer contract once at construction.  Some lightweight
        # test doubles do not expose IDs, so validation remains capability-aware.
        self.tokenizer_contract = validate_prott5_tokenizer(self.tokenizer)

    @staticmethod
    def normalize_sequence(sequence: str) -> str:
        """Normalize a sequence for T5 SentencePiece tokenization.

        Strips whitespace, uppercases, maps U/Z/O/B to X, and space-separates
        residues so each is a SentencePiece token.
        """
        raw = normalize_prott5_sequence(sequence)
        return " ".join(list(raw))

    def _prepare(self, sequences):
        prepared = []
        for value in sequences:
            normalized = self.normalize_sequence(value)
            if self.legacy_total_token_length:
                prepared.append(normalized)
            else:
                # Split by spaces to count residues, truncate, rejoin.
                residues = normalized.split()
                prepared.append(" ".join(residues[: self.max_length]))
        return prepared

    def _tokenize(self, sequences):
        prepared = self._prepare(sequences)
        if any(not seq for seq in prepared):
            raise ValueError("ProtT5 sequences must contain at least one residue")
        special_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        token_limit = (
            self.max_length if self.legacy_total_token_length else self.max_length + special_count
        )
        return self.tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=token_limit,
            return_special_tokens_mask=True,
        )

    def tokenize(self, sequence: str):
        return self._tokenize([sequence])

    def batch_tokenize(self, sequences, device=None):
        inputs = self._tokenize(sequences)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def batch_tokenize_residues(self, sequences, max_residues, device=None):
        max_residues = validate_prott5_max_residues(
            max_residues,
            field="max_residues",
        )
        prepared = [self.normalize_sequence(value) for value in sequences]
        # Truncate to max_residues after normalization (preserves residue alignment)
        prepared = [" ".join(seq.split()[:max_residues]) for seq in prepared]
        special_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        inputs = self.tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_residues + special_count,
            return_special_tokens_mask=True,
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def residue_length(self, sequence: str, max_residues=None) -> int:
        normalized = normalize_prott5_sequence(sequence)
        if max_residues is not None:
            normalized = normalized[
                : validate_prott5_max_residues(max_residues, field="max_residues")
            ]
        return len(normalized)


class UnigramTokenizerTrainer:
    def __init__(
        self,
        vocab_size=1000,
        initial_alphabet=None,
        max_raw_length=UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
        show_progress=True,
    ):
        if Tokenizer is None:
            raise ImportError(
                "UnigramTokenizerTrainer requires "
                f"tokenizers=={UNIGRAM_TOKENIZERS_VERSION}. "
                f"Run `pip install tokenizers=={UNIGRAM_TOKENIZERS_VERSION}`."
            )
        self.tokenizers_version = str(getattr(tokenizers_package, "__version__", "") or "")
        if self.tokenizers_version != UNIGRAM_TOKENIZERS_VERSION:
            raise ImportError(
                "UnigramTokenizerTrainer requires "
                f"tokenizers=={UNIGRAM_TOKENIZERS_VERSION}; "
                f"found {self.tokenizers_version or 'unknown'}."
            )
        self.vocab_size = validate_unigram_vocab_size(vocab_size)
        requested_alphabet = tuple(initial_alphabet or UNIGRAM_ALPHABET)
        if requested_alphabet != UNIGRAM_ALPHABET:
            raise UnigramContractError(
                "Unigram initial_alphabet is fixed by the platform contract: "
                f"{''.join(UNIGRAM_ALPHABET)}",
                code="unigram_vocab_invalid",
            )
        self.initial_alphabet = list(UNIGRAM_ALPHABET)
        self.max_raw_length = validate_unigram_max_raw_length(max_raw_length)
        self.show_progress = bool(show_progress)
        self.last_cache_reused = False
        self.last_provenance = None

    @staticmethod
    def _commit_staged_cache(
        staged_tokenizer,
        staged_provenance,
        destination,
    ):
        destination = os.path.abspath(os.fspath(destination))
        provenance_destination = os.fspath(unigram_provenance_path(destination))
        destinations = (destination, provenance_destination)
        staged = (staged_tokenizer, staged_provenance)
        backups = {}
        try:
            for target in destinations:
                if not os.path.exists(target):
                    backups[target] = None
                    continue
                descriptor, backup = tempfile.mkstemp(
                    prefix=f".{os.path.basename(target)}.",
                    suffix=".backup",
                    dir=os.path.dirname(target),
                )
                os.close(descriptor)
                shutil.copy2(target, backup)
                backups[target] = backup
            for source, target in zip(staged, destinations):
                os.replace(source, target)
        except Exception:
            for target, backup in backups.items():
                if backup is None:
                    try:
                        os.unlink(target)
                    except FileNotFoundError:
                        pass
                else:
                    shutil.copy2(backup, target)
            raise
        finally:
            for backup in backups.values():
                if backup:
                    try:
                        os.unlink(backup)
                    except FileNotFoundError:
                        pass

    def _canonicalize(self, tokenizer, normalized_sequences):
        from tokenizers import models, processors

        payload = json.loads(tokenizer.to_str())
        raw_vocab = (payload.get("model") or {}).get("vocab") or []
        score_by_token = {
            str(entry[0]): float(entry[1])
            for entry in raw_vocab
            if isinstance(entry, list) and len(entry) == 2
        }
        observed = set("".join(normalized_sequences))
        absent_residues = set(UNIGRAM_ALPHABET).difference(observed)
        residue_scores = sorted(
            (
                float(score_by_token[residue]),
                residue,
            )
            for residue in UNIGRAM_ALPHABET
            if residue in observed and residue in score_by_token
        )
        clusters = []
        for score, residue in residue_scores:
            if not clusters or score - clusters[-1][-1][0] > UNIGRAM_SINGLE_SCORE_TOLERANCE:
                clusters.append([])
            clusters[-1].append((score, residue))
        for cluster in clusters:
            stable_score = math.fsum(score for score, _ in cluster) / len(cluster)
            for _, residue in cluster:
                score_by_token[residue] = stable_score
        learned_scores = [
            score
            for token, score in score_by_token.items()
            if token not in UNIGRAM_SPECIAL_TOKEN_IDS
            and token not in absent_residues
            and math.isfinite(score)
        ]
        if not learned_scores:
            raise UnigramContractError(
                "Unigram trainer did not produce any finite learned piece scores"
            )
        missing_residue_score = min(learned_scores) - 1.0
        for residue in UNIGRAM_ALPHABET:
            if residue not in score_by_token or residue in absent_residues:
                score_by_token[residue] = missing_residue_score
        ordinary_tokens = sorted(set(score_by_token).difference(UNIGRAM_SPECIAL_TOKEN_IDS))
        canonical_vocab = [(token, 0.0) for token in UNIGRAM_SPECIAL_TOKENS]
        canonical_vocab.extend(
            (
                token,
                round(float(score_by_token[token]), UNIGRAM_SCORE_DECIMALS),
            )
            for token in ordinary_tokens
        )
        canonical = Tokenizer(
            models.Unigram(
                vocab=canonical_vocab,
                unk_id=UNIGRAM_SPECIAL_TOKEN_IDS["<UNK>"],
                byte_fallback=False,
            )
        )
        canonical.add_special_tokens(list(UNIGRAM_SPECIAL_TOKENS))
        canonical.post_processor = processors.TemplateProcessing(
            single="<CLS> $A <SEP>",
            special_tokens=[
                ("<CLS>", UNIGRAM_SPECIAL_TOKEN_IDS["<CLS>"]),
                ("<SEP>", UNIGRAM_SPECIAL_TOKEN_IDS["<SEP>"]),
            ],
        )
        return canonical

    def _train_normalized(
        self,
        normalized_sequences,
        save_path,
        sequence_role,
    ):
        from tokenizers import models, trainers

        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(UNIGRAM_SPECIAL_TOKENS),
            show_progress=self.show_progress,
            initial_alphabet=self.initial_alphabet,
            shrinking_factor=UNIGRAM_SHRINKING_FACTOR,
            unk_token="<UNK>",
            max_piece_length=UNIGRAM_MAX_PIECE_LENGTH,
            n_sub_iterations=UNIGRAM_N_SUB_ITERATIONS,
        )
        tokenizer.train_from_iterator(
            normalized_sequences,
            trainer=trainer,
            length=len(normalized_sequences),
        )
        tokenizer = self._canonicalize(tokenizer, normalized_sequences)

        destination = os.path.abspath(os.fspath(save_path))
        destination_dir = os.path.dirname(destination)
        os.makedirs(destination_dir, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(destination)}.",
            suffix=".tmp",
            dir=destination_dir,
        )
        os.close(descriptor)
        temporary_provenance = f"{temporary_path}.provenance.json"
        try:
            tokenizer.save(temporary_path)
            inspection = inspect_unigram_artifact(temporary_path)
            provenance = build_unigram_provenance(
                normalized_sequences=normalized_sequences,
                requested_vocab_size=self.vocab_size,
                actual_vocab_size=inspection["vocab_size"],
                max_raw_length=self.max_raw_length,
                artifact_sha256=inspection["sha256"],
                tokenizers_version=self.tokenizers_version,
                sequence_role=sequence_role,
            )
            atomic_write_unigram_json(
                temporary_provenance,
                provenance,
            )
            self._commit_staged_cache(
                temporary_path,
                temporary_provenance,
                destination,
            )
        finally:
            for staged_path in (temporary_path, temporary_provenance):
                try:
                    os.unlink(staged_path)
                except FileNotFoundError:
                    pass
        self.last_cache_reused = False
        self.last_provenance = provenance
        return tokenizer

    def train_sequences(
        self,
        sequences,
        save_path,
        *,
        sequence_role="sequence",
    ):
        normalized = normalize_unigram_sequences(
            sequences,
            max_raw_length=self.max_raw_length,
        )
        return self._train_normalized(normalized, save_path, sequence_role)

    def ensure_trained(
        self,
        sequences,
        save_path,
        *,
        sequence_role="sequence",
    ):
        normalized = normalize_unigram_sequences(
            sequences,
            max_raw_length=self.max_raw_length,
        )
        if unigram_cache_is_current(
            save_path,
            normalized_sequences=normalized,
            vocab_size=self.vocab_size,
            max_raw_length=self.max_raw_length,
            tokenizers_version=self.tokenizers_version,
            sequence_role=sequence_role,
        ):
            self.last_cache_reused = True
            self.last_provenance = load_unigram_provenance(save_path)
            return Tokenizer.from_file(os.fspath(save_path))
        return self._train_normalized(normalized, save_path, sequence_role)

    def train(self, files, save_path):
        """Compatibility entry point for newline-delimited sequence corpora."""

        sequences = []
        for corpus_path in files:
            with open(corpus_path, "r", encoding="utf-8") as handle:
                sequences.extend(line for line in handle if line.strip())
        return self.train_sequences(sequences, save_path)


class WordPieceTokenizerTrainer:
    def __init__(
        self,
        vocab_size=1000,
        initial_alphabet=None,
        continuing_subword_prefix=WORDPIECE_CONTINUING_SUBWORD_PREFIX,
        max_raw_length=WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
        show_progress=True,
    ):
        if Tokenizer is None:
            raise ImportError(
                "WordPieceTokenizerTrainer requires "
                f"tokenizers=={WORDPIECE_TOKENIZERS_VERSION}. "
                f"Run `pip install tokenizers=={WORDPIECE_TOKENIZERS_VERSION}`."
            )
        self.tokenizers_version = str(getattr(tokenizers_package, "__version__", "") or "")
        if self.tokenizers_version != WORDPIECE_TOKENIZERS_VERSION:
            raise ImportError(
                "WordPieceTokenizerTrainer requires "
                f"tokenizers=={WORDPIECE_TOKENIZERS_VERSION}; "
                f"found {self.tokenizers_version or 'unknown'}."
            )
        self.vocab_size = validate_wordpiece_vocab_size(vocab_size)
        requested_alphabet = tuple(initial_alphabet or WORDPIECE_ALPHABET)
        if requested_alphabet != WORDPIECE_ALPHABET:
            raise ValueError(
                "WordPiece initial_alphabet is fixed by the platform contract: "
                f"{''.join(WORDPIECE_ALPHABET)}"
            )
        if continuing_subword_prefix != WORDPIECE_CONTINUING_SUBWORD_PREFIX:
            raise ValueError(
                "WordPiece continuing_subword_prefix is fixed by the platform "
                f"contract: {WORDPIECE_CONTINUING_SUBWORD_PREFIX!r}"
            )
        self.initial_alphabet = list(WORDPIECE_ALPHABET)
        self.continuing_subword_prefix = WORDPIECE_CONTINUING_SUBWORD_PREFIX
        self.max_raw_length = validate_wordpiece_max_raw_length(max_raw_length)
        self.show_progress = bool(show_progress)
        self.last_cache_reused = False
        self.last_provenance = None

    @staticmethod
    def _commit_staged_cache(
        staged_tokenizer,
        staged_provenance,
        destination,
    ):
        destination = os.path.abspath(os.fspath(destination))
        provenance_destination = os.fspath(wordpiece_provenance_path(destination))
        destinations = (destination, provenance_destination)
        staged = (staged_tokenizer, staged_provenance)
        backups = {}
        try:
            for target in destinations:
                if not os.path.exists(target):
                    backups[target] = None
                    continue
                descriptor, backup = tempfile.mkstemp(
                    prefix=f".{os.path.basename(target)}.",
                    suffix=".backup",
                    dir=os.path.dirname(target),
                )
                os.close(descriptor)
                shutil.copy2(target, backup)
                backups[target] = backup
            for source, target in zip(staged, destinations):
                os.replace(source, target)
        except Exception:
            for target, backup in backups.items():
                if backup is None:
                    try:
                        os.unlink(target)
                    except FileNotFoundError:
                        pass
                else:
                    shutil.copy2(backup, target)
            raise
        finally:
            for backup in backups.values():
                if backup:
                    try:
                        os.unlink(backup)
                    except FileNotFoundError:
                        pass

    def _train_normalized(
        self,
        normalized_sequences,
        save_path,
        sequence_role,
    ):
        from tokenizers import models, pre_tokenizers, processors, trainers

        observed_continuations = {
            residue for sequence in normalized_sequences for residue in sequence[1:]
        }
        reserved_fallbacks = [
            f"{self.continuing_subword_prefix}{residue}"
            for residue in WORDPIECE_ALPHABET
            if residue not in observed_continuations
        ]
        trainer_vocab_size = self.vocab_size - len(reserved_fallbacks)
        tokenizer = Tokenizer(
            models.WordPiece(
                unk_token="<UNK>",
                continuing_subword_prefix=self.continuing_subword_prefix,
                max_input_chars_per_word=self.max_raw_length,
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = trainers.WordPieceTrainer(
            vocab_size=trainer_vocab_size,
            special_tokens=list(WORDPIECE_SPECIAL_TOKENS),
            show_progress=self.show_progress,
            initial_alphabet=self.initial_alphabet,
            continuing_subword_prefix=self.continuing_subword_prefix,
        )
        tokenizer.train_from_iterator(
            normalized_sequences,
            trainer=trainer,
            length=len(normalized_sequences),
        )

        vocab = tokenizer.get_vocab(with_added_tokens=False)
        added_fallback_tokens = []
        for residue in WORDPIECE_ALPHABET:
            for token in (
                residue,
                f"{self.continuing_subword_prefix}{residue}",
            ):
                if token not in vocab:
                    vocab[token] = -1
                    added_fallback_tokens.append(token)
        # WordPiece inference depends on vocabulary membership rather than merge
        # ranks. Canonicalize IDs after training while retaining the fixed
        # special IDs; the pinned trainer may still choose different, equally
        # scored pieces when its internal tie order differs.
        canonical_vocab = dict(WORDPIECE_SPECIAL_TOKEN_IDS)
        for token in sorted(set(vocab).difference(WORDPIECE_SPECIAL_TOKEN_IDS)):
            canonical_vocab[token] = len(canonical_vocab)
        tokenizer.model = models.WordPiece(
            vocab=canonical_vocab,
            unk_token="<UNK>",
            continuing_subword_prefix=self.continuing_subword_prefix,
            max_input_chars_per_word=self.max_raw_length,
        )
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<CLS> $A <SEP>",
            special_tokens=[
                ("<CLS>", WORDPIECE_SPECIAL_TOKEN_IDS["<CLS>"]),
                ("<SEP>", WORDPIECE_SPECIAL_TOKEN_IDS["<SEP>"]),
            ],
        )

        destination = os.path.abspath(os.fspath(save_path))
        destination_dir = os.path.dirname(destination)
        os.makedirs(destination_dir, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(destination)}.",
            suffix=".tmp",
            dir=destination_dir,
        )
        os.close(descriptor)
        temporary_provenance = f"{temporary_path}.provenance.json"
        try:
            tokenizer.save(temporary_path)
            inspection = inspect_wordpiece_artifact(temporary_path)
            provenance = build_wordpiece_provenance(
                normalized_sequences=normalized_sequences,
                requested_vocab_size=self.vocab_size,
                trainer_vocab_size=trainer_vocab_size,
                actual_vocab_size=inspection["vocab_size"],
                added_fallback_tokens=added_fallback_tokens,
                max_raw_length=self.max_raw_length,
                artifact_sha256=inspection["sha256"],
                tokenizers_version=self.tokenizers_version,
                sequence_role=sequence_role,
            )
            atomic_write_wordpiece_json(
                temporary_provenance,
                provenance,
            )
            self._commit_staged_cache(
                temporary_path,
                temporary_provenance,
                destination,
            )
        finally:
            for staged_path in (temporary_path, temporary_provenance):
                try:
                    os.unlink(staged_path)
                except FileNotFoundError:
                    pass
        self.last_cache_reused = False
        self.last_provenance = provenance
        return tokenizer

    def train_sequences(
        self,
        sequences,
        save_path,
        *,
        sequence_role="sequence",
    ):
        normalized = normalize_wordpiece_sequences(
            sequences,
            max_raw_length=self.max_raw_length,
        )
        return self._train_normalized(normalized, save_path, sequence_role)

    def ensure_trained(
        self,
        sequences,
        save_path,
        *,
        sequence_role="sequence",
    ):
        normalized = normalize_wordpiece_sequences(
            sequences,
            max_raw_length=self.max_raw_length,
        )
        if wordpiece_cache_is_current(
            save_path,
            normalized_sequences=normalized,
            vocab_size=self.vocab_size,
            max_raw_length=self.max_raw_length,
            tokenizers_version=self.tokenizers_version,
            sequence_role=sequence_role,
        ):
            self.last_cache_reused = True
            with open(
                wordpiece_provenance_path(save_path),
                "r",
                encoding="utf-8",
            ) as handle:
                self.last_provenance = json.load(handle)
            return Tokenizer.from_file(os.fspath(save_path))
        return self._train_normalized(normalized, save_path, sequence_role)

    def train(self, files, save_path):
        """Compatibility entry point for newline-delimited sequence corpora."""

        sequences = []
        for corpus_path in files:
            with open(corpus_path, "r", encoding="utf-8") as handle:
                sequences.extend(line for line in handle if line.strip())
        return self.train_sequences(sequences, save_path)


class EsmTokenizer(BaseTokenizer):
    """ESM-2 tokenizer with residue-length and alignment semantics.

    ``max_length`` counts residues, with extra room for BOS/EOS. With
    ``legacy_total_token_length=True``, it counts all tokens including BOS/EOS.
    Both modes retain the special-token mask.
    """

    def __init__(
        self,
        model_name=ESM2_OFFICIAL_MODEL_ID,
        max_length=60,
        *,
        legacy_total_token_length=False,
    ):
        super().__init__(max_length=max_length)
        if AutoTokenizer is None:
            raise ImportError(
                "EsmTokenizer requires 'transformers' library. Run `pip install transformers`."
            )
        self.model_name = str(model_name or ESM2_OFFICIAL_MODEL_ID).strip()
        self.legacy_total_token_length = bool(legacy_total_token_length)
        if not self.model_name:
            raise ValueError("ESM2 model source must not be empty")
        if os.path.isdir(self.model_name) and not is_complete_esm2_snapshot(self.model_name):
            raise ValueError(
                "ESM2 local source is incomplete or has the wrong architecture signature: "
                f"{self.model_name}"
            )
        kwargs = {}
        if not os.path.isdir(self.model_name) and self.model_name == ESM2_OFFICIAL_MODEL_ID:
            kwargs["revision"] = ESM2_HF_REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        self.max_length = validate_esm2_max_residues(max_length)

    @staticmethod
    def normalize_sequence(sequence):
        return normalize_esm2_sequence(sequence)

    def _prepare(self, sequences):
        if self.legacy_total_token_length:
            prepared = [str(value or "") for value in sequences]
            if any(not value for value in prepared):
                raise ValueError("ESM2 sequences must contain at least one residue")
            return prepared
        prepared = [self.normalize_sequence(value)[: self.max_length] for value in sequences]
        if not prepared:
            return []
        if any(not value for value in prepared):
            raise ValueError("ESM2 sequences must contain at least one residue")
        invalid = sorted(set("".join(prepared)) - ESM2_VALID_SEQUENCE_AA)
        if invalid:
            raise ValueError("ESM2 sequence contains unsupported residues: " + ", ".join(invalid))
        return prepared

    def _tokenize(self, sequences):
        prepared = self._prepare(sequences)
        special_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        token_limit = (
            self.max_length if self.legacy_total_token_length else self.max_length + special_count
        )
        return self.tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=token_limit,
            return_special_tokens_mask=True,
        )

    def tokenize(self, sequence: str):
        return self._tokenize([sequence])

    def batch_tokenize(self, sequences, device=None):
        inputs = self._tokenize(sequences)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def batch_tokenize_residues(self, sequences, max_residues, device=None):
        max_residues = validate_esm2_max_residues(
            max_residues,
            field="max_residues",
        )
        prepared = [self.normalize_sequence(value)[:max_residues] for value in sequences]
        special_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        inputs = self.tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_residues + special_count,
            return_special_tokens_mask=True,
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs


class EsmcTokenizer(BaseTokenizer):
    def __init__(self, model_name=ESMC_OFFICIAL_600M_MODEL_ID, max_length=60):
        super().__init__(max_length=max_length)
        self.model_name = model_name
        self.backend = esmc_backend_for_source(model_name)
        if self.backend == ESMC_BACKEND_OFFICIAL:
            require_official_esmc_runtime(check_api=False)
            try:
                from esm.tokenization import get_esmc_model_tokenizers
            except Exception as exc:
                raise RuntimeError(
                    "esm==3.2.0 does not provide the official ESM-C tokenizer API. "
                    f"Run: {ESMC_RUNTIME_INSTALL_COMMAND}"
                ) from exc

            self.tokenizer = get_esmc_model_tokenizers()
            return
        if AutoTokenizer is None:
            raise ImportError("ESM++ tokenization requires the 'transformers' library.")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            if AutoModelForMaskedLM is None:
                raise ImportError(
                    "EsmcTokenizer init failed and AutoModelForMaskedLM is unavailable."
                )
            model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
            if not hasattr(model, "tokenizer") or model.tokenizer is None:
                raise ImportError(
                    "EsmcTokenizer init failed: remote model does not provide tokenizer."
                )
            self.tokenizer = model.tokenizer

    @staticmethod
    def _normalize_official_sequence(sequence):
        normalized = str(sequence or "").strip().upper()
        invalid = sorted(set(normalized) - ESMC_VALID_SEQUENCE_AA)
        if invalid:
            raise ValueError("ESM-C sequence contains unsupported residues: " + ", ".join(invalid))
        return normalized

    def normalize_sequence(self, sequence):
        """Return the residue string used by explicit-mask ESM-C routes."""
        return self._normalize_official_sequence(sequence)

    def residue_length(self, sequence, max_residues=None):
        normalized = self.normalize_sequence(sequence)
        return (
            min(len(normalized), int(max_residues)) if max_residues is not None else len(normalized)
        )

    def _official_batch(self, sequences, max_residues, device=None):
        from esm.utils.encoding import tokenize_sequence
        from esm.utils.misc import stack_variable_length_tensors

        max_residues = int(max_residues)
        if max_residues <= 0:
            raise ValueError("max_residues must be positive")
        prepared = [self._normalize_official_sequence(value)[:max_residues] for value in sequences]
        rows = [
            tokenize_sequence(value, self.tokenizer, add_special_tokens=True) for value in prepared
        ]
        if rows:
            sequence_tokens = stack_variable_length_tensors(
                rows,
                constant_value=int(self.tokenizer.pad_token_id),
            )
        else:
            sequence_tokens = torch.empty((0, max_residues + 2), dtype=torch.long)
        attention_mask = sequence_tokens.ne(int(self.tokenizer.pad_token_id))
        special_tokens_mask = torch.ones_like(sequence_tokens, dtype=torch.bool)
        for index, value in enumerate(prepared):
            if value:
                special_tokens_mask[index, 1 : 1 + len(value)] = False
        result = {
            "sequence_tokens": sequence_tokens,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask,
        }
        if device is not None:
            result = {key: value.to(device) for key, value in result.items()}
        return result

    def tokenize(self, sequence: str):
        if self.backend == ESMC_BACKEND_OFFICIAL:
            return self._official_batch([sequence], self.max_length)
        return self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

    def batch_tokenize(self, sequences, device=None):
        if self.backend == ESMC_BACKEND_OFFICIAL:
            return self._official_batch(sequences, self.max_length, device=device)
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        return inputs

    def batch_tokenize_residues(self, sequences, max_residues, device=None):
        if self.backend == ESMC_BACKEND_OFFICIAL:
            return self._official_batch(sequences, max_residues, device=device)
        return _batch_tokenize_residues(
            self.tokenizer,
            sequences,
            max_residues,
            device=device,
        )


class Esm3Tokenizer(BaseTokenizer):
    def __init__(self, model_name=ESM3_OFFICIAL_MODEL_ID, max_length=60):
        super().__init__(max_length=validate_esm3_max_residues(max_length))
        require_official_esm3_runtime()
        self.model_name = str(model_name or ESM3_OFFICIAL_MODEL_ID).strip()
        local_path = os.path.expanduser(self.model_name)
        if os.path.isdir(local_path) and not is_complete_esm3_source(local_path):
            raise FileNotFoundError(
                "ESM3 local source is incomplete; expected data/weights/esm3_sm_open_v1.pth"
            )
        if (
            not os.path.isdir(local_path)
            and normalize_esm3_model_name(self.model_name) != "esm3_sm_open_v1"
        ):
            raise ValueError(
                "unsupported ESM3 tokenizer source; use the ESM3 open-small model identity"
            )
        try:
            from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        except Exception as exc:
            raise ImportError(
                "Esm3Tokenizer requires the official 'esm' SDK. Run `pip install esm==3.2.0`."
            ) from exc
        self.tokenizer = EsmSequenceTokenizer()

    @staticmethod
    def normalize_sequence(sequence):
        return normalize_esm3_sequence(sequence)

    def residue_length(self, sequence, max_residues=None):
        value = (
            truncate_esm3_sequence(sequence, max_residues)
            if max_residues is not None
            else normalize_esm3_sequence(sequence)
        )
        return esm3_residue_count(value)

    def _batch(self, sequences, max_residues, device=None):
        limit = validate_esm3_max_residues(max_residues)
        normalized = [truncate_esm3_sequence(sequence, limit) for sequence in sequences]
        inputs = self.tokenizer(
            normalized,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
        )
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        return inputs

    def tokenize(self, sequence: str):
        return self._batch([sequence], self.max_length)

    def batch_tokenize(self, sequences, device=None):
        return self._batch(sequences, self.max_length, device=device)

    def batch_tokenize_residues(self, sequences, max_residues, device=None):
        return self._batch(sequences, max_residues, device=device)


class BPETokenizerTrainer:
    def __init__(self, vocab_size=1000, initial_alphabet=None, show_progress=True):
        if Tokenizer is None:
            raise ImportError(
                f"BPETokenizerTrainer requires tokenizers=={BPE_TOKENIZERS_VERSION}. "
                f"Run `pip install tokenizers=={BPE_TOKENIZERS_VERSION}`."
            )
        self.tokenizers_version = str(getattr(tokenizers_package, "__version__", "") or "")
        if self.tokenizers_version != BPE_TOKENIZERS_VERSION:
            raise ImportError(
                f"BPETokenizerTrainer requires tokenizers=={BPE_TOKENIZERS_VERSION}; "
                f"found {self.tokenizers_version or 'unknown'}."
            )
        self.vocab_size = int(vocab_size)
        if self.vocab_size < BPE_MIN_VOCAB_SIZE:
            raise ValueError(
                f"BPE vocab_size must be >= {BPE_MIN_VOCAB_SIZE} so the fixed "
                "protein alphabet, special tokens, and at least one merge fit"
            )
        requested_alphabet = tuple(initial_alphabet or BPE_ALPHABET)
        if requested_alphabet != BPE_ALPHABET:
            raise ValueError(
                f"BPE initial_alphabet is fixed by the platform contract: {''.join(BPE_ALPHABET)}"
            )
        self.initial_alphabet = list(BPE_ALPHABET)
        self.show_progress = bool(show_progress)
        self.last_cache_reused = False
        self.last_provenance = None

    @staticmethod
    def _commit_staged_cache(staged_tokenizer, staged_provenance, destination):
        destination = os.path.abspath(os.fspath(destination))
        provenance_destination = os.fspath(bpe_provenance_path(destination))
        destinations = (destination, provenance_destination)
        staged = (staged_tokenizer, staged_provenance)
        backups = {}
        try:
            for target in destinations:
                if not os.path.exists(target):
                    backups[target] = None
                    continue
                descriptor, backup = tempfile.mkstemp(
                    prefix=f".{os.path.basename(target)}.",
                    suffix=".backup",
                    dir=os.path.dirname(target),
                )
                os.close(descriptor)
                shutil.copy2(target, backup)
                backups[target] = backup
            for source, target in zip(staged, destinations):
                os.replace(source, target)
        except Exception:
            for target, backup in backups.items():
                if backup is None:
                    try:
                        os.unlink(target)
                    except FileNotFoundError:
                        pass
                else:
                    shutil.copy2(backup, target)
            raise
        finally:
            for backup in backups.values():
                if backup:
                    try:
                        os.unlink(backup)
                    except FileNotFoundError:
                        pass

    def _train_normalized(self, normalized_sequences, save_path, sequence_role):
        from tokenizers import models, pre_tokenizers, trainers, processors

        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=BPE_MIN_FREQUENCY,
            special_tokens=list(BPE_SPECIAL_TOKENS),
            show_progress=self.show_progress,
            initial_alphabet=self.initial_alphabet,
        )
        tokenizer.train_from_iterator(
            normalized_sequences,
            trainer=trainer,
            length=len(normalized_sequences),
        )
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<CLS> $A <SEP>",
            special_tokens=[
                ("<CLS>", BPE_SPECIAL_TOKEN_IDS["<CLS>"]),
                ("<SEP>", BPE_SPECIAL_TOKEN_IDS["<SEP>"]),
            ],
        )
        destination = os.path.abspath(os.fspath(save_path))
        destination_dir = os.path.dirname(destination)
        os.makedirs(destination_dir, exist_ok=True)
        temporary_path = os.path.join(
            destination_dir,
            f".{os.path.basename(destination)}.{os.getpid()}.tmp",
        )
        temporary_provenance = f"{temporary_path}.provenance.json"
        try:
            tokenizer.save(temporary_path)
            inspection = inspect_bpe_artifact(temporary_path)
            provenance = build_bpe_provenance(
                normalized_sequences=normalized_sequences,
                requested_vocab_size=self.vocab_size,
                actual_vocab_size=inspection["vocab_size"],
                artifact_sha256=inspection["sha256"],
                tokenizers_version=self.tokenizers_version,
                sequence_role=sequence_role,
            )
            atomic_write_json(temporary_provenance, provenance)
            # All fallible training and validation work happens before replacing
            # a previously valid cache.
            self._commit_staged_cache(
                temporary_path,
                temporary_provenance,
                destination,
            )
        finally:
            for staged_path in (temporary_path, temporary_provenance):
                try:
                    os.unlink(staged_path)
                except FileNotFoundError:
                    pass
        self.last_cache_reused = False
        self.last_provenance = provenance
        return tokenizer

    def train_sequences(self, sequences, save_path, *, sequence_role="sequence"):
        normalized = normalize_bpe_sequences(sequences)
        return self._train_normalized(normalized, save_path, sequence_role)

    def ensure_trained(self, sequences, save_path, *, sequence_role="sequence"):
        normalized = normalize_bpe_sequences(sequences)
        if bpe_cache_is_current(
            save_path,
            normalized_sequences=normalized,
            vocab_size=self.vocab_size,
            tokenizers_version=self.tokenizers_version,
            sequence_role=sequence_role,
        ):
            self.last_cache_reused = True
            with open(bpe_provenance_path(save_path), "r", encoding="utf-8") as handle:
                self.last_provenance = json.load(handle)
            return Tokenizer.from_file(os.fspath(save_path))
        return self._train_normalized(normalized, save_path, sequence_role)

    def train(self, files, save_path):
        """Compatibility entry point for newline-delimited sequence corpora."""

        sequences = []
        for corpus_path in files:
            with open(corpus_path, "r", encoding="utf-8") as handle:
                sequences.extend(line for line in handle if line.strip())
        return self.train_sequences(sequences, save_path)
