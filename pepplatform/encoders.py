import os
import contextlib
import importlib.metadata
import math
from dataclasses import dataclass
from numbers import Integral

import torch

from pepplatform.covariance import (
    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
    build_covariance_encoder_contract,
    compute_covariance_features,
    covariance_feature_names,
    validate_covariance_lag,
    validate_covariance_max_raw_length,
)

from pepplatform.frequency import (
    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    count_feature_ids,
    normalize_frequency_sequence,
    validate_frequency_max_raw_length,
    validate_idf_vector,
)
from pepplatform.pssm_covariance import (
    build_pssm_covariance_encoder_contract,
    compute_pssm_covariance_features,
    pssm_covariance_feature_names,
    validate_pssm_covariance_lag,
)
from pepplatform.pdt import (
    PDT_DEFAULT_LAG,
    PDT_DEFAULT_MAX_RAW_LENGTH,
    PDT_PROPERTY_COUNT,
    build_pdt_encoder_contract,
    compute_pdt_features,
    pdt_feature_names,
    validate_pdt_lag,
    validate_pdt_max_raw_length,
)
from pepplatform.textrank import (
    TEXTRANK_DEFAULT_DAMPING,
    TEXTRANK_DEFAULT_MAX_ITERATIONS,
    TEXTRANK_DEFAULT_TOLERANCE,
    TEXTRANK_DEFAULT_WINDOW_SIZE,
    validate_feature_ids,
    validate_textrank_parameters,
    weighted_textrank,
)
from pepplatform.sequence_profiles import (
    AA_ORDER as PROFILE_AA_ORDER,
    BLOSUM62_MATRIX_SHA256,
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
    FEATURE_AC_PSSM_BLOSUM62_HYBRID,
    FEATURE_CC_PSSM_BLOSUM62_HYBRID,
    FEATURE_PSFM,
    FEATURE_PSFM_BLOSUM62_HYBRID,
    FEATURE_PSSM,
    FEATURE_PSSM_BLOSUM62_HYBRID,
    ProfileStore,
    blosum62_token_mean,
)

from pepplatform.esmc import (
    ESMC_BACKEND_ESMPLUSPLUS,
    ESMC_BACKEND_OFFICIAL,
    ESMC_OFFICIAL_600M_MODEL_ID,
    esmc_backend_for_source,
    esmc_model_identity,
    esmc_runtime_environment,
    esmc_variant_for_source,
    require_official_esmc_runtime,
    validate_official_esmc_weight,
)
from pepplatform.esm2 import (
    ESM2_HF_REVISION,
    ESM2_OFFICIAL_MODEL_ID,
    esm2_model_identity,
    esm2_runtime_environment,
    is_complete_esm2_snapshot,
    validate_esm2_config,
)
from pepplatform.esm3 import (
    ESM3_CHAIN_BREAK,
    ESM3_DEFAULT_SDPA_BACKEND,
    ESM3_HIDDEN_SIZE,
    ESM3_MAX_RESIDUES,
    ESM3_OFFICIAL_MODEL_ID,
    ESM3_POOLING,
    ESM3_RESIDUE_OUTPUT_KIND,
    esm3_model_identity,
    esm3_residue_count,
    esm3_runtime_environment,
    normalize_esm3_model_name,
    normalize_esm3_sequence,
    normalize_esm3_sequences,
    require_official_esm3_runtime,
    truncate_esm3_sequence,
    validate_esm3_max_residues,
    validate_esm3_sdpa_backend,
)
from pepplatform.protbert import (
    PROTBERT_MAX_RESIDUES,
    PROTBERT_OFFICIAL_MODEL_ID,
    protbert_model_identity,
    protbert_runtime_environment,
    is_complete_protbert_snapshot,
    validate_protbert_config,
)
from pepplatform.prott5 import (
    PROTT5_D_MODEL,
    PROTT5_HF_REVISION,
    PROTT5_MAX_RESIDUES,
    PROTT5_OFFICIAL_MODEL_ID,
    inspect_prott5_source,
    is_complete_prott5_snapshot,
    prott5_model_identity,
    prott5_runtime_environment,
    validate_prott5_config,
    validate_prott5_max_residues,
)


def _model_parameter_count(model) -> int:
    try:
        return sum(int(parameter.numel()) for parameter in model.parameters())
    except Exception:
        return 0


RESIDUE_ALIGNMENT_VERSION = 2


class _WeightedTextRankMixin:
    def _configure_textrank(
        self,
        *,
        feature_dim,
        window_size,
        damping,
        max_iter,
        tol,
        context,
    ):
        (
            self.feature_dim,
            self.window_size,
            self.damping,
            self.tol,
            self.max_iter,
        ) = validate_textrank_parameters(
            feature_dim=feature_dim,
            window_size=window_size,
            damping=damping,
            tolerance=tol,
            max_iterations=max_iter,
        )
        self.vocab_size = self.feature_dim
        self._textrank_context = str(context)

    def _rank_feature_ids(self, raw_ids, *, padding_id=None):
        ids = validate_feature_ids(
            raw_ids,
            feature_dim=self.feature_dim,
            context=self._textrank_context,
        )
        if padding_id is not None:
            ids = [feature_id for feature_id in ids if feature_id != padding_id]
        if not ids:
            return torch.zeros(self.feature_dim, dtype=torch.float32)
        result = weighted_textrank(
            ids,
            feature_dim=self.feature_dim,
            window_size=self.window_size,
            damping=self.damping,
            tolerance=self.tol,
            max_iterations=self.max_iter,
            context=self._textrank_context,
        )
        return result.l1_normalized()


@dataclass(frozen=True)
class ResidueEncoding:
    features: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self):
        if self.features.ndim != 3:
            raise ValueError(f"residue features must be [B,L,D], got {tuple(self.features.shape)}")
        if self.mask.ndim != 2 or self.mask.shape != self.features.shape[:2]:
            raise ValueError(
                f"residue mask must match [B,L], got {tuple(self.mask.shape)} for {tuple(self.features.shape)}"
            )


def _last_hidden_state(outputs):
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    raise RuntimeError("pretrained encoder did not return token hidden states")


def _encode_hf_residue_batch(encoder, inputs, sequence_lengths, max_length):
    device_inputs = {key: value.to(encoder.device) for key, value in inputs.items()}
    model_inputs = {
        key: value for key, value in device_inputs.items() if key != "special_tokens_mask"
    }
    with torch.no_grad():
        try:
            outputs = encoder.model(**model_inputs, output_hidden_states=True)
        except TypeError:
            outputs = encoder.model(**model_inputs)
    hidden = _last_hidden_state(outputs)
    attention_mask = device_inputs.get("attention_mask")
    special_tokens_mask = device_inputs.get("special_tokens_mask")
    if attention_mask is None or special_tokens_mask is None:
        raise RuntimeError("residue extraction requires attention_mask and special_tokens_mask")
    valid_tokens = attention_mask.bool() & ~special_tokens_mask.bool()
    rows = []
    masks = []
    for index, raw_length in enumerate(sequence_lengths):
        residue_count = min(int(raw_length), int(max_length))
        residue_states = hidden[index][valid_tokens[index]]
        if residue_states.shape[0] != residue_count:
            raise RuntimeError(
                f"pretrained token/residue alignment failed for sample {index}: "
                f"states={residue_states.shape[0]}, residues={residue_count}"
            )
        padded = hidden.new_zeros((max_length, hidden.shape[-1]))
        padded_mask = torch.zeros(max_length, dtype=torch.bool, device=hidden.device)
        if residue_count:
            padded[:residue_count] = residue_states
            padded_mask[:residue_count] = True
        rows.append(padded.detach().cpu().to(dtype=torch.float16))
        masks.append(padded_mask.detach().cpu())
    if not rows:
        width = int(hidden.shape[-1])
        return ResidueEncoding(
            torch.empty((0, max_length, width), dtype=torch.float16),
            torch.empty((0, max_length), dtype=torch.bool),
        )
    return ResidueEncoding(torch.stack(rows), torch.stack(masks))


try:
    from transformers import AutoModel, T5EncoderModel, AutoModelForMaskedLM
except ImportError:
    AutoModel = None
    T5EncoderModel = None
    AutoModelForMaskedLM = None


def _normalize_token_ids(ids, *, max_length, vocab_size):
    normalized = []
    for position, value in enumerate(list(ids)[:max_length]):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"token id at position {position} must be an integer, got {value!r}")
        index = int(value)
        if index < 0 or index >= vocab_size:
            raise ValueError(
                f"token id at position {position} is outside [0, {vocab_size}): {index}"
            )
        normalized.append(index)
    normalized.extend([0] * (max_length - len(normalized)))
    return normalized


class IndexEncoder:
    def __init__(self, max_length, vocab_size, to_indices=None):
        if isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, Integral) or vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        self.max_length = int(max_length)
        self.vocab_size = int(vocab_size)
        self.to_indices = to_indices

    def encode(self, tokens):
        if len(tokens) == 0 or isinstance(tokens[0], Integral):
            ids = tokens
        else:
            if self.to_indices is None:
                raise ValueError("to_indices required for non-index tokens")
            ids = self.to_indices(tokens)
        ids = _normalize_token_ids(
            ids,
            max_length=self.max_length,
            vocab_size=self.vocab_size,
        )
        return torch.tensor(ids, dtype=torch.long)


class OneHotEncoder:
    def __init__(self, max_length, vocab_size, to_indices):
        if isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, Integral) or vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        if to_indices is None:
            raise ValueError("to_indices is required")
        self.max_length = int(max_length)
        self.vocab_size = int(vocab_size)
        self.to_indices = to_indices

    def encode(self, tokens):
        ids = _normalize_token_ids(
            self.to_indices(tokens),
            max_length=self.max_length,
            vocab_size=self.vocab_size,
        )
        one_hot = torch.zeros(self.max_length, self.vocab_size, dtype=torch.float)
        for i, idx in enumerate(ids):
            if idx != 0:
                one_hot[i, idx] = 1.0
        return one_hot


class DRCountEncoder:
    """Exact Distance-based Residue occurrence-count feature map."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.feature_dim
        self.feature_dim = tokenizer.feature_dim

    def _encode_tokens(self, tokens):
        return count_feature_ids(
            (
                self.tokenizer.feature_to_index[token]
                for token in tokens
                if token in self.tokenizer.feature_to_index
            ),
            feature_dim=self.feature_dim,
            context="DR BoW",
        )

    def encode(self, tokens):
        return self._encode_tokens(tokens)

    def encode_sequence(self, sequence):
        return self._encode_tokens(self.tokenizer.iter_tokens(sequence))


class DRTfidfEncoder:
    """BioSeq-BLM-style TF-IDF over the complete DR v2 word stream."""

    def __init__(self, tokenizer, idf):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.feature_dim
        self.feature_dim = tokenizer.feature_dim
        self.idf = validate_idf_vector(
            idf,
            feature_dim=self.feature_dim,
            context="DR TF-IDF",
        )

    def _encode_tokens(self, tokens):
        counts = count_feature_ids(
            (
                self.tokenizer.feature_to_index[token]
                for token in tokens
                if token in self.tokenizer.feature_to_index
            ),
            feature_dim=self.feature_dim,
            context="DR TF-IDF",
        )
        total = counts.sum()
        if total > 0:
            counts = counts / total
        return counts * self.idf

    def encode(self, tokens):
        return self._encode_tokens(tokens)

    def encode_sequence(self, sequence):
        return self._encode_tokens(self.tokenizer.iter_tokens(sequence))


class DRTextRankEncoder(_WeightedTextRankMixin):
    """TextRank extension over the complete position-major DR token stream."""

    def __init__(
        self,
        tokenizer,
        window_size=TEXTRANK_DEFAULT_WINDOW_SIZE,
        damping=TEXTRANK_DEFAULT_DAMPING,
        max_iter=TEXTRANK_DEFAULT_MAX_ITERATIONS,
        tol=TEXTRANK_DEFAULT_TOLERANCE,
    ):
        self.tokenizer = tokenizer
        self._configure_textrank(
            feature_dim=tokenizer.feature_dim,
            window_size=window_size,
            damping=damping,
            max_iter=max_iter,
            tol=tol,
            context="DR TextRank",
        )

    def _encode_tokens(self, tokens):
        ids = []
        for position, token in enumerate(tokens):
            if token not in self.tokenizer.feature_to_index:
                raise ValueError(
                    f"DR TextRank token at position {position} is not in the feature map: {token!r}"
                )
            ids.append(self.tokenizer.feature_to_index[token])
        return self._rank_feature_ids(ids)

    def encode(self, tokens):
        return self._encode_tokens(tokens)

    def encode_sequence(self, sequence):
        return self._encode_tokens(self.tokenizer.iter_tokens(sequence))


class TopNGramCountEncoder:
    """Exact occurrence counts over a complete Top-n-gram v3 stream."""

    def __init__(
        self,
        tokenizer,
        max_raw_length=FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    ):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.feature_dim
        self.feature_dim = tokenizer.feature_dim
        self.max_raw_length = validate_frequency_max_raw_length(max_raw_length)

    def _encode_ids(self, ids):
        return count_feature_ids(
            ids,
            feature_dim=self.feature_dim,
            context="Top-n-gram BoW",
        )

    def encode(self, tokens):
        return self._encode_ids(self.tokenizer.to_indices(tokens))

    def encode_sequence(self, sequence, sample_id=None):
        sequence = normalize_frequency_sequence(
            sequence,
            max_raw_length=self.max_raw_length,
        )
        return self._encode_ids(self.tokenizer.feature_ids(sequence, sample_id=sample_id))


class TopNGramTfidfEncoder:
    """Train-only TF-IDF extension over complete Top-n-gram v3 streams."""

    def __init__(
        self,
        tokenizer,
        idf,
        max_raw_length=FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    ):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.feature_dim
        self.feature_dim = tokenizer.feature_dim
        self.max_raw_length = validate_frequency_max_raw_length(max_raw_length)
        self.idf = validate_idf_vector(
            idf,
            feature_dim=self.feature_dim,
            context="Top-n-gram TF-IDF",
        )

    def _encode_ids(self, ids):
        counts = count_feature_ids(
            ids,
            feature_dim=self.feature_dim,
            context="Top-n-gram TF-IDF",
        )
        total = counts.sum()
        if total > 0:
            counts = counts / total
        return counts * self.idf

    def encode(self, tokens):
        return self._encode_ids(self.tokenizer.to_indices(tokens))

    def encode_sequence(self, sequence, sample_id=None):
        sequence = normalize_frequency_sequence(
            sequence,
            max_raw_length=self.max_raw_length,
        )
        return self._encode_ids(self.tokenizer.feature_ids(sequence, sample_id=sample_id))


class TopNGramTextRankEncoder(_WeightedTextRankMixin):
    """TextRank platform extension over the full Top-n-gram v3 token stream."""

    def __init__(
        self,
        tokenizer,
        window_size=TEXTRANK_DEFAULT_WINDOW_SIZE,
        damping=TEXTRANK_DEFAULT_DAMPING,
        max_iter=TEXTRANK_DEFAULT_MAX_ITERATIONS,
        tol=TEXTRANK_DEFAULT_TOLERANCE,
    ):
        self.tokenizer = tokenizer
        self._configure_textrank(
            feature_dim=tokenizer.feature_dim,
            window_size=window_size,
            damping=damping,
            max_iter=max_iter,
            tol=tol,
            context="Top-n-gram TextRank",
        )

    def _encode_ids(self, raw_ids):
        return self._rank_feature_ids(raw_ids)

    def encode(self, tokens):
        return self._encode_ids(self.tokenizer.to_indices(tokens))

    def encode_sequence(self, sequence, sample_id=None):
        return self._encode_ids(self.tokenizer.feature_ids(sequence, sample_id=sample_id))


class DTCountEncoder:
    """Exact occurrence counts for the 2014 Distance-based Top-1-gram map."""

    def __init__(
        self,
        tokenizer,
        max_raw_length=FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    ):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.feature_dim
        self.feature_dim = tokenizer.feature_dim
        self.max_raw_length = validate_frequency_max_raw_length(max_raw_length)

    def _encode_ids(self, ids):
        return count_feature_ids(
            ids,
            feature_dim=self.feature_dim,
            context="DT BoW",
        )

    def encode(self, tokens):
        return self._encode_ids(self.tokenizer.to_indices(tokens))

    def encode_sequence(self, sequence, sample_id=None):
        sequence = normalize_frequency_sequence(
            sequence,
            max_raw_length=self.max_raw_length,
        )
        return self._encode_ids(self.tokenizer.iter_feature_ids(sequence, sample_id=sample_id))


class DTTfidfEncoder:
    """Train-only TF-IDF extension over the complete DT v1 word stream."""

    def __init__(
        self,
        tokenizer,
        idf,
        max_raw_length=FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    ):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.feature_dim
        self.feature_dim = tokenizer.feature_dim
        self.max_raw_length = validate_frequency_max_raw_length(max_raw_length)
        self.idf = validate_idf_vector(
            idf,
            feature_dim=self.feature_dim,
            context="DT TF-IDF",
        )

    def _encode_ids(self, ids):
        counts = count_feature_ids(
            ids,
            feature_dim=self.feature_dim,
            context="DT TF-IDF",
        )
        total = counts.sum()
        if total > 0:
            counts = counts / total
        return counts * self.idf

    def encode(self, tokens):
        return self._encode_ids(self.tokenizer.to_indices(tokens))

    def encode_sequence(self, sequence, sample_id=None):
        sequence = normalize_frequency_sequence(
            sequence,
            max_raw_length=self.max_raw_length,
        )
        return self._encode_ids(self.tokenizer.iter_feature_ids(sequence, sample_id=sample_id))


class DTTextRankEncoder(TopNGramTextRankEncoder):
    """TextRank extension over the complete source-position-major DT stream."""

    def encode_sequence(self, sequence, sample_id=None):
        return self._encode_ids(self.tokenizer.iter_feature_ids(sequence, sample_id=sample_id))


class BoWEncoder:
    def __init__(
        self,
        max_length,
        vocab_size,
        to_indices,
        normalize=False,
        sequence_to_ids=None,
    ):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.to_indices = to_indices
        self.normalize = normalize
        self.feature_dim = vocab_size
        if sequence_to_ids is not None:

            def encode_sequence(sequence):
                return self._encode_ids(sequence_to_ids(sequence))

            self.encode_sequence = encode_sequence

    def _encode_ids(self, ids):
        counts = count_feature_ids(
            ids,
            feature_dim=self.vocab_size,
            padding_id=0,
            context="BoW",
        )
        if self.normalize and counts.sum() > 0:
            counts = counts / counts.sum()
        return counts

    def encode(self, tokens):
        return self._encode_ids(self.to_indices(tokens))


class TfidfEncoder:
    def __init__(
        self,
        max_length,
        vocab_size,
        to_indices,
        idf,
        normalize=True,
        sequence_to_ids=None,
    ):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.to_indices = to_indices
        self.normalize = normalize
        self.feature_dim = vocab_size
        self.idf = validate_idf_vector(
            idf,
            feature_dim=self.vocab_size,
            context="TF-IDF",
        )
        if sequence_to_ids is not None:

            def encode_sequence(sequence):
                return self._encode_ids(sequence_to_ids(sequence))

            self.encode_sequence = encode_sequence

    def _encode_ids(self, ids):
        counts = count_feature_ids(
            ids,
            feature_dim=self.vocab_size,
            padding_id=0,
            context="TF-IDF",
        )
        if self.normalize and counts.sum() > 0:
            counts = counts / counts.sum()
        return counts * self.idf

    def encode(self, tokens):
        return self._encode_ids(self.to_indices(tokens))


class TextRankEncoder(_WeightedTextRankMixin):
    def __init__(
        self,
        max_length,
        vocab_size,
        to_indices,
        window_size=TEXTRANK_DEFAULT_WINDOW_SIZE,
        damping=TEXTRANK_DEFAULT_DAMPING,
        max_iter=TEXTRANK_DEFAULT_MAX_ITERATIONS,
        tol=TEXTRANK_DEFAULT_TOLERANCE,
    ):
        if isinstance(max_length, bool) or not isinstance(max_length, Integral) or max_length < 1:
            raise ValueError("TextRank max_length must be a positive integer")
        if not callable(to_indices):
            raise ValueError("TextRank to_indices must be callable")
        self.max_length = int(max_length)
        self.to_indices = to_indices
        self._configure_textrank(
            feature_dim=vocab_size,
            window_size=window_size,
            damping=damping,
            max_iter=max_iter,
            tol=tol,
            context="TextRank",
        )

    def encode(self, tokens):
        return self._rank_feature_ids(self.to_indices(tokens), padding_id=0)


class Blosum62Encoder:
    def __init__(self, max_length):
        if isinstance(max_length, bool) or not isinstance(max_length, Integral):
            raise ValueError("BLOSUM62 max_length must be a positive integer")
        self.max_length = int(max_length)
        if self.max_length < 1:
            raise ValueError("BLOSUM62 max_length must be a positive integer")
        self.feature_dim = len(PROFILE_AA_ORDER)
        self.matrix_sha256 = BLOSUM62_MATRIX_SHA256
        self.score_unit = "half_bit"
        self.token_aggregation = "mean_within_kmer"

    def encode(self, tokens):
        feats: list[torch.Tensor] = []
        for token in tokens[: self.max_length]:
            if not isinstance(token, str):
                raise ValueError(f"BLOSUM62 token must be a string, got {type(token).__name__}")
            feats.append(torch.from_numpy(blosum62_token_mean(token)))
        if len(feats) < self.max_length:
            padding = torch.zeros(self.feature_dim, dtype=torch.float)
            feats.extend([padding] * (self.max_length - len(feats)))
        return torch.stack(feats, dim=0)


class _GuoCovarianceEncoder:
    encoder_name = ""

    def __init__(
        self,
        *,
        lag=10,
        max_raw_length=COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
    ):
        self.lag = validate_covariance_lag(lag)
        self.max_raw_length = validate_covariance_max_raw_length(max_raw_length)
        self.base_feature_dim = 7
        self.feature_names = covariance_feature_names(self.encoder_name, self.lag)
        self.feature_dim = len(self.feature_names)
        self.covariance_encoder_contract = build_covariance_encoder_contract(
            encoder=self.encoder_name,
            lag=self.lag,
            max_raw_length=self.max_raw_length,
        )

    def encode_sequence(self, sequence):
        features = compute_covariance_features(
            sequence,
            encoder=self.encoder_name,
            lag=self.lag,
            max_raw_length=self.max_raw_length,
        )
        return torch.from_numpy(features.astype("float32", copy=False))

    def encode(self, sequence):
        return self.encode_sequence(sequence)


class AutoCovarianceEncoder(_GuoCovarianceEncoder):
    encoder_name = "ac"


class CrossCovarianceEncoder(_GuoCovarianceEncoder):
    encoder_name = "cc"


class AutoCrossCovarianceEncoder(_GuoCovarianceEncoder):
    encoder_name = "acc"


class PDTEncoder:
    def __init__(
        self,
        *,
        lag=PDT_DEFAULT_LAG,
        max_raw_length=PDT_DEFAULT_MAX_RAW_LENGTH,
    ):
        self.lag = validate_pdt_lag(lag)
        self.max_raw_length = validate_pdt_max_raw_length(max_raw_length)
        self.base_feature_dim = PDT_PROPERTY_COUNT
        self.feature_names = pdt_feature_names(self.lag)
        self.feature_dim = len(self.feature_names)
        self.pdt_encoder_contract = build_pdt_encoder_contract(
            lag=self.lag,
            max_raw_length=self.max_raw_length,
        )

    def encode_sequence(self, sequence):
        features = compute_pdt_features(
            sequence,
            lag=self.lag,
            max_raw_length=self.max_raw_length,
        )
        return torch.from_numpy(features.astype("float32", copy=False))

    def encode(self, sequence):
        return self.encode_sequence(sequence)


def _validate_profile_max_length(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError("profile max_length must be a positive integer")
    return int(value)


def _validate_profile_lag(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError("PSSM lag must be a positive integer")
    return int(value)


class _PositionProfileEncoder:
    feature_dim = len(PROFILE_AA_ORDER)
    profile_feature_type = FEATURE_PSSM

    def __init__(self, max_length, index_path, strict=True):
        if strict is not True:
            raise ValueError(
                "profile encoders require an explicit feature-specific manifest; "
                "select the named PSSM/PSFM BLOSUM62 Hybrid encoder when its "
                "fallback is scientifically acceptable"
            )
        self.max_length = _validate_profile_max_length(max_length)
        self.index_path = index_path
        self.strict = True
        self.store = ProfileStore(
            index_path,
            feature_type=self.profile_feature_type,
        )

    def _pad_or_truncate(self, matrix):
        if matrix.shape[0] >= self.max_length:
            return matrix[: self.max_length]
        output = torch.zeros(
            (self.max_length, self.feature_dim),
            dtype=torch.float,
        )
        output[: matrix.shape[0]] = matrix
        return output

    def encode_sequence(self, sequence, sample_id=None):
        loaded = self.store.load(sample_id, sequence)
        return self._pad_or_truncate(loaded.matrix)

    def encode_sequence_with_mask(self, sequence, sample_id=None):
        loaded = self.store.load(sample_id, sequence)
        features = self._pad_or_truncate(loaded.matrix)
        residue_count = min(int(loaded.matrix.shape[0]), self.max_length)
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:residue_count] = True
        return features, mask

    def encode(self, tokens, sample_id=None):
        return self.encode_sequence("".join(tokens), sample_id=sample_id)


class PSSMEncoder(_PositionProfileEncoder):
    profile_feature_type = FEATURE_PSSM


class PSSMBlosum62HybridEncoder(_PositionProfileEncoder):
    profile_feature_type = FEATURE_PSSM_BLOSUM62_HYBRID


class PSFMEncoder(_PositionProfileEncoder):
    profile_feature_type = FEATURE_PSFM
    requires_explicit_mask = True


class PSFMBlosum62HybridEncoder(_PositionProfileEncoder):
    profile_feature_type = FEATURE_PSFM_BLOSUM62_HYBRID
    requires_explicit_mask = True


class _PSSMCovarianceEncoder:
    base_feature_dim = len(PROFILE_AA_ORDER)
    encoder_name = ""
    profile_feature_type = FEATURE_PSSM

    def __init__(self, max_length, index_path, lag=10, zscore=None, strict=True):
        if strict is not True:
            raise ValueError("PSSM covariance encoders require strict profiles")
        if zscore not in (None, False):
            raise ValueError("PSSM covariance encoders do not permit column z-score scaling")
        self.max_length = _validate_profile_max_length(max_length)
        self.index_path = index_path
        self.lag = validate_pssm_covariance_lag(lag)
        self.zscore = False
        self.strict = True
        self.store = ProfileStore(
            index_path,
            feature_type=self.profile_feature_type,
        )
        self.feature_names = pssm_covariance_feature_names(
            self.encoder_name,
            self.lag,
        )
        self.feature_dim = len(self.feature_names)
        self.pssm_covariance_encoder_contract = build_pssm_covariance_encoder_contract(
            encoder=self.encoder_name,
            lag=self.lag,
        )

    def encode_sequence(self, sequence, sample_id=None):
        matrix = self.store.load(sample_id, sequence).matrix
        features = compute_pssm_covariance_features(
            matrix,
            encoder=self.encoder_name,
            lag=self.lag,
        )
        return torch.from_numpy(features).to(dtype=torch.float32)

    def encode(self, tokens, sample_id=None):
        return self.encode_sequence("".join(tokens), sample_id=sample_id)


class ACPSSMEncoder(_PSSMCovarianceEncoder):
    encoder_name = "ac_pssm"


class CCPSSMEncoder(_PSSMCovarianceEncoder):
    encoder_name = "cc_pssm"


class ACCPSSMEncoder(_PSSMCovarianceEncoder):
    encoder_name = "acc_pssm"


class ACPSSMBlosum62HybridEncoder(_PSSMCovarianceEncoder):
    encoder_name = FEATURE_AC_PSSM_BLOSUM62_HYBRID
    profile_feature_type = FEATURE_AC_PSSM_BLOSUM62_HYBRID


class CCPSSMBlosum62HybridEncoder(_PSSMCovarianceEncoder):
    encoder_name = FEATURE_CC_PSSM_BLOSUM62_HYBRID
    profile_feature_type = FEATURE_CC_PSSM_BLOSUM62_HYBRID


class ACCPSSMBlosum62HybridEncoder(_PSSMCovarianceEncoder):
    encoder_name = FEATURE_ACC_PSSM_BLOSUM62_HYBRID
    profile_feature_type = FEATURE_ACC_PSSM_BLOSUM62_HYBRID


class StaticEmbeddingEncoder:
    def __init__(self, max_length, embedding_path, has_header=False):
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, Integral)
            or int(max_length) < 1
        ):
            raise ValueError("static embedding max_length must be a positive integer")
        self.max_length = int(max_length)
        self.embedding_path = str(embedding_path)
        self.vectors, self.feature_dim = self._load_vectors(embedding_path, has_header)
        self.unk = self.vectors.get("<UNK>")
        if self.unk is None:
            self.unk = self.vectors.get("UNK")
        if self.unk is None:
            self.unk = torch.zeros(self.feature_dim, dtype=torch.float)
        self.lookup_count = 0
        self.in_vocabulary_count = 0
        self.oov_count = 0

    def _load_vectors(self, path, has_header):
        if not path or not os.path.isfile(path):
            raise ValueError(f"embedding file not found: {path}")
        vectors = {}
        dim = None
        declared_count = None
        with open(path, "r", encoding="utf-8") as f:
            first_data_line = True
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if has_header and first_data_line:
                    parts = line.split()
                    if len(parts) != 2 or not all(part.isdigit() for part in parts):
                        raise ValueError(
                            "word2vec text artifacts must begin with a "
                            f"'<vocab_size> <dimension>' header; got line "
                            f"{line_number}: {line!r}"
                        )
                    declared_count = int(parts[0])
                    dim = int(parts[1])
                    if declared_count < 1 or dim < 1:
                        raise ValueError(
                            f"invalid embedding header at line {line_number}: {line!r}"
                        )
                    first_data_line = False
                    continue
                first_data_line = False
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(
                        f"invalid embedding row at line {line_number}: expected token and values"
                    )
                token = parts[0]
                if token in vectors:
                    raise ValueError(f"duplicate embedding token {token!r} at line {line_number}")
                vals = parts[1:]
                if dim is None:
                    dim = len(vals)
                if len(vals) != dim:
                    raise ValueError(
                        f"embedding dimension mismatch at line {line_number}: "
                        f"expected {dim}, got {len(vals)}"
                    )
                try:
                    numbers = [float(value) for value in vals]
                except ValueError as exc:
                    raise ValueError(f"non-numeric embedding value at line {line_number}") from exc
                if not all(math.isfinite(value) for value in numbers):
                    raise ValueError(f"non-finite embedding value at line {line_number}")
                vectors[token] = torch.tensor(numbers, dtype=torch.float)
        if dim is None or len(vectors) == 0:
            raise ValueError(f"invalid embedding file: {path}")
        if declared_count is not None and declared_count != len(vectors):
            raise ValueError(
                f"embedding header declares {declared_count} vectors, loaded {len(vectors)}"
            )
        return vectors, dim

    def lookup(self, token):
        key = str(token)
        self.lookup_count += 1
        if key in self.vectors:
            self.in_vocabulary_count += 1
            return self.vectors[key]
        self.oov_count += 1
        return self.unk

    def coverage_stats(self):
        covered = self.in_vocabulary_count
        total = self.lookup_count
        return {
            "lookups": total,
            "in_vocabulary": covered,
            "oov": self.oov_count,
            "coverage": float(covered / total) if total else 1.0,
            "oov_policy": "provided_unk_else_null_vector",
        }

    def encode(self, tokens):
        feats = [self.lookup(t) for t in tokens[: self.max_length]]
        if len(feats) < self.max_length:
            pad = torch.zeros(self.feature_dim, dtype=torch.float)
            feats.extend([pad] * (self.max_length - len(feats)))
        return torch.stack(feats, dim=0)


class Word2VecEncoder(StaticEmbeddingEncoder):
    def __init__(self, max_length, embedding_path):
        super().__init__(max_length=max_length, embedding_path=embedding_path, has_header=True)


class GloveEncoder(StaticEmbeddingEncoder):
    def __init__(self, max_length, embedding_path):
        super().__init__(max_length=max_length, embedding_path=embedding_path, has_header=False)


class FastTextEncoder(StaticEmbeddingEncoder):
    """Load a complete Facebook-compatible fastText binary, including n-grams."""

    def __init__(self, max_length, embedding_path):
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, Integral)
            or int(max_length) < 1
        ):
            raise ValueError("fastText max_length must be a positive integer")
        path = str(embedding_path or "")
        if not path or not os.path.isfile(path):
            raise ValueError(f"fastText embedding file not found: {embedding_path}")
        if os.path.splitext(path)[1].lower() in {".vec", ".txt"}:
            raise ValueError(
                "legacy fastText text vectors are unsupported because they omit "
                "character n-gram state; provide a complete Facebook-compatible .bin artifact"
            )
        try:
            from gensim.models.fasttext import load_facebook_vectors
        except Exception as exc:
            raise RuntimeError(
                "gensim is required to load complete fastText binary artifacts"
            ) from exc
        try:
            keyed_vectors = load_facebook_vectors(path)
        except Exception as exc:
            raise ValueError(f"invalid fastText binary artifact: {path}") from exc
        self.max_length = int(max_length)
        self.embedding_path = path
        self.keyed_vectors = keyed_vectors
        self.feature_dim = int(keyed_vectors.vector_size)
        self.min_n = int(keyed_vectors.min_n)
        self.max_n = int(keyed_vectors.max_n)
        self.bucket = int(keyed_vectors.bucket)
        self.cache = {}
        self.unk = torch.zeros(self.feature_dim, dtype=torch.float)
        self.lookup_count = 0
        self.in_vocabulary_count = 0
        self.subword_oov_count = 0
        self.failed_oov_count = 0

    def lookup(self, token):
        key = str(token)
        self.lookup_count += 1
        if key in self.keyed_vectors.key_to_index:
            self.in_vocabulary_count += 1
        else:
            self.subword_oov_count += 1
        if key in self.cache:
            return self.cache[key]
        try:
            values = self.keyed_vectors.get_vector(key)
        except KeyError:
            self.failed_oov_count += 1
            return self.unk
        if values.shape != (self.feature_dim,) or not bool(
            torch.isfinite(torch.as_tensor(values)).all()
        ):
            raise RuntimeError(f"fastText returned an invalid vector for token {key!r}")
        self.cache[key] = torch.as_tensor(values.copy(), dtype=torch.float)
        return self.cache[key]

    def coverage_stats(self):
        return {
            "lookups": self.lookup_count,
            "in_vocabulary": self.in_vocabulary_count,
            "subword_synthesized": self.subword_oov_count,
            "failed_oov": self.failed_oov_count,
            "coverage": (
                float((self.lookup_count - self.failed_oov_count) / self.lookup_count)
                if self.lookup_count
                else 1.0
            ),
            "oov_policy": "character_ngram_synthesis",
        }


class ProtBertEncoder:
    """Frozen ProtBERT final hidden-state encoder.

    Formal sequence embeddings are the mean of residue positions only;
    ``[CLS]``, ``[SEP]`` and padding are excluded by the explicit tokenizer
    special-token mask.  The historical attention-mask mean is available only
    through ``encode_legacy_attention_mask_mean`` for old artifacts.
    """

    def __init__(self, model_name=PROTBERT_OFFICIAL_MODEL_ID, device=None):
        if AutoModel is None:
            raise ImportError(
                "ProtBertEncoder requires 'transformers' library. Run `pip install transformers`."
            )
        self.model_name = str(model_name or PROTBERT_OFFICIAL_MODEL_ID).strip()
        if not self.model_name:
            raise ValueError("ProtBERT model source must not be empty")
        is_local_dir = os.path.isdir(self.model_name)
        if is_local_dir and not is_complete_protbert_snapshot(self.model_name):
            raise ValueError(
                "ProtBERT local source is incomplete or has the wrong architecture signature: "
                f"{self.model_name}"
            )
        load_kwargs = {}
        if not is_local_dir and self.model_name == PROTBERT_OFFICIAL_MODEL_ID:
            from pepplatform.protbert import PROTBERT_HF_REVISION

            load_kwargs["revision"] = PROTBERT_HF_REVISION
        has_safetensors = is_local_dir and (
            os.path.isfile(os.path.join(self.model_name, "model.safetensors"))
            or os.path.isfile(os.path.join(self.model_name, "model.safetensors.index.json"))
        )
        self.device = torch.device(device if device is not None else "cpu")
        self.model = AutoModel.from_pretrained(
            self.model_name,
            use_safetensors=bool(has_safetensors),
            **load_kwargs,
        ).to(self.device)
        self.model.eval()
        config = getattr(self.model, "config", None)
        if config is None:
            raise RuntimeError("ProtBERT encoder did not expose a model config")
        config_dict = config.to_dict() if hasattr(config, "to_dict") else vars(config)
        validate_protbert_config(config_dict)
        self.output_dim = int(getattr(config, "hidden_size", 0) or 0)
        weight_path = (
            os.path.join(
                self.model_name, "model.safetensors" if has_safetensors else "pytorch_model.bin"
            )
            if is_local_dir
            else ""
        )
        self.model_identity = protbert_model_identity(
            self.model_name,
            config=config_dict,
            weight_path=weight_path,
        )
        self.runtime_environment = protbert_runtime_environment(self.device)

    @staticmethod
    def _residue_mask(inputs, attention_mask):
        special = inputs.get("special_tokens_mask")
        if special is None:
            special = torch.zeros_like(attention_mask)
            if attention_mask.shape[1]:
                special[:, 0] = attention_mask[:, 0]
                valid_lengths = attention_mask.long().sum(dim=1)
                for row, length in enumerate(valid_lengths.tolist()):
                    if length > 1:
                        special[row, length - 1] = 1
        return attention_mask.bool() & ~special.bool()

    def encode(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model_inputs = {key: value for key, value in inputs.items() if key != "special_tokens_mask"}
        with torch.inference_mode():
            outputs = self.model(**model_inputs)
        last_hidden_state = _last_hidden_state(outputs)
        attention_mask = inputs["attention_mask"]
        residue_mask = self._residue_mask(inputs, attention_mask)
        if not residue_mask.any(dim=1).all():
            raise ValueError("ProtBERT sequence has no residue tokens after special-token masking")
        input_mask_expanded = (
            residue_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        if not bool(torch.isfinite(embeddings).all()):
            raise RuntimeError("ProtBERT encoder produced non-finite pooled embeddings")
        return embeddings

    def encode_legacy_attention_mask_mean(self, inputs):
        """Preserve historical CLS/SEP-inclusive pooling for old artifacts."""

        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key != "special_tokens_mask"
        }
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        hidden = _last_hidden_state(outputs)
        mask = model_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        if not bool(torch.isfinite(embeddings).all()):
            raise RuntimeError("legacy ProtBERT encoder produced non-finite pooled embeddings")
        return embeddings

    def encode_residue_batch(self, inputs, sequence_lengths, max_length):
        if int(max_length) < 1 or int(max_length) > PROTBERT_MAX_RESIDUES:
            raise ValueError(
                f"ProtBERT max_length must be between 1 and {PROTBERT_MAX_RESIDUES} residues"
            )
        return _encode_hf_residue_batch(self, inputs, sequence_lengths, max_length)


ProBertEncoder = ProtBertEncoder


class ProT5Encoder:
    """Frozen ProtT5-XL final hidden-state encoder.

    Sequence embeddings are a mean over residue positions only.  EOS/PAD
    are excluded using the tokenizer-provided special-token mask.
    """

    def __init__(self, model_name="Rostlab/prot_t5_xl_uniref50", device=None):
        if AutoModel is None or T5EncoderModel is None:
            raise ImportError(
                "ProT5Encoder requires 'transformers' library. Run `pip install transformers`."
            )
        self.model_name = str(model_name or PROTT5_OFFICIAL_MODEL_ID).strip()
        if not self.model_name:
            raise ValueError("ProtT5 model source must not be empty")
        is_local = os.path.isdir(self.model_name)
        if is_local and not is_complete_prott5_snapshot(self.model_name):
            raise ValueError(
                "ProtT5 local source is incomplete or has the wrong architecture signature: "
                f"{self.model_name}"
            )
        self.device = torch.device(device if device is not None else "cpu")

        # Determine weight format explicitly without silent retry.
        has_safetensors = is_local and (
            os.path.exists(os.path.join(self.model_name, "model.safetensors"))
            or os.path.exists(os.path.join(self.model_name, "model.safetensors.index.json"))
        )
        has_bin = is_local and (
            os.path.exists(os.path.join(self.model_name, "pytorch_model.bin"))
            or os.path.exists(os.path.join(self.model_name, "pytorch_model.bin.index.json"))
        )

        load_kwargs = {}
        if not is_local and self.model_name == PROTT5_OFFICIAL_MODEL_ID:
            load_kwargs["revision"] = PROTT5_HF_REVISION

        if is_local:
            # Declared format: use the file that exists, prefer safetensors if both exist.
            use_safetensors = has_safetensors
            if not use_safetensors and not has_bin:
                raise FileNotFoundError(
                    f"ProtT5 local source has no weight file: {self.model_name}"
                )
        else:
            # The official full checkpoint is published as pytorch_model.bin.
            # Remote sources therefore use an explicit, non-fallback format.
            use_safetensors = False

        self.model = T5EncoderModel.from_pretrained(
            self.model_name,
            use_safetensors=use_safetensors,
            **load_kwargs,
        ).to(self.device)

        self.model.eval()

        config = getattr(self.model, "config", None)
        if config is None:
            raise RuntimeError("ProtT5 encoder did not expose a model config")
        validate_prott5_config(config.to_dict() if hasattr(config, "to_dict") else vars(config))
        self.output_dim = int(
            getattr(config, "d_model", 0) or getattr(config, "hidden_size", 0) or 0
        )
        source_facts = inspect_prott5_source(self.model_name)
        weight_path = str(source_facts.get("weight_path", "") or "")
        self.model_identity = prott5_model_identity(
            self.model_name,
            config=config.to_dict() if hasattr(config, "to_dict") else vars(config),
            weight_path=weight_path,
            parameter_count=_model_parameter_count(self.model),
        )
        self.runtime_environment = prott5_runtime_environment(self.device)

    @staticmethod
    def _residue_mask(inputs, attention_mask):
        special = inputs.get("special_tokens_mask")
        if special is None:
            raise RuntimeError(
                "ProtT5 residue extraction requires tokenizer-provided "
                "special_tokens_mask; use encode_legacy_attention_mask_mean "
                "for legacy tensor-only artifacts"
            )
        return attention_mask.bool() & ~special.bool()

    def encode(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model_inputs = {key: value for key, value in inputs.items() if key != "special_tokens_mask"}
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        last_hidden_state = _last_hidden_state(outputs)
        attention_mask = inputs["attention_mask"]
        residue_mask = self._residue_mask(inputs, attention_mask)
        if not residue_mask.any(dim=1).all():
            raise ValueError("ProtT5 sequence has no residue tokens after special-token masking")
        input_mask_expanded = (
            residue_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        if not bool(torch.isfinite(embeddings).all()):
            raise RuntimeError("ProtT5 encoder produced non-finite pooled embeddings")
        return embeddings

    def encode_legacy_attention_mask_mean(self, inputs):
        """Preserve historical EOS-inclusive attention-mask mean pooling."""

        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key != "special_tokens_mask"
        }
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        hidden = _last_hidden_state(outputs)
        mask = model_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        if not bool(torch.isfinite(embeddings).all()):
            raise RuntimeError("legacy ProtT5 encoder produced non-finite pooled embeddings")
        return embeddings

    def encode_residue_batch(self, inputs, sequence_lengths, max_length):
        max_length = validate_prott5_max_residues(max_length)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None or attention_mask.shape[0] != len(sequence_lengths):
            raise ValueError("ProtT5 residue sequence lengths must match the input batch")
        encoding = _encode_hf_residue_batch(self, inputs, sequence_lengths, max_length)
        if encoding.features.numel() and not bool(torch.isfinite(encoding.features.float()).all()):
            raise RuntimeError("ProtT5 residue encoder produced non-finite features")
        return encoding


class EsmEncoder:
    """Frozen ESM-2 final hidden-state encoder.

    Sequence embeddings are a mean over residue positions only.  BOS/EOS/PAD
    are excluded using the tokenizer-provided special-token mask.
    """

    def __init__(self, model_name=ESM2_OFFICIAL_MODEL_ID, device=None):
        if AutoModel is None:
            raise ImportError(
                "EsmEncoder requires 'transformers' library. Run `pip install transformers`."
            )
        self.model_name = str(model_name or ESM2_OFFICIAL_MODEL_ID).strip()
        if not self.model_name:
            raise ValueError("ESM2 model source must not be empty")
        is_local = os.path.isdir(self.model_name)
        if is_local and not is_complete_esm2_snapshot(self.model_name):
            raise ValueError(
                "ESM2 local source is incomplete or has the wrong architecture signature: "
                f"{self.model_name}"
            )
        self.device = torch.device(device if device is not None else "cpu")
        load_kwargs = {}
        if not is_local and self.model_name == ESM2_OFFICIAL_MODEL_ID:
            load_kwargs["revision"] = ESM2_HF_REVISION
        try:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                use_safetensors=True,
                add_pooling_layer=False,
                **load_kwargs,
            ).to(self.device)
        except TypeError:
            # Keep compatibility with small test doubles and older Transformers.
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    **load_kwargs,
                ).to(self.device)
            except Exception:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    use_safetensors=False,
                    **load_kwargs,
                ).to(self.device)
        except Exception:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                use_safetensors=False,
                add_pooling_layer=False,
                **load_kwargs,
            ).to(self.device)
        self.model.eval()
        config = getattr(self.model, "config", None)
        if config is None:
            raise RuntimeError("ESM2 encoder did not expose a model config")
        validate_esm2_config(config.to_dict() if hasattr(config, "to_dict") else vars(config))
        self.output_dim = int(getattr(config, "hidden_size", 0) or 0)
        weight_path = os.path.join(self.model_name, "model.safetensors") if is_local else ""
        self.model_identity = esm2_model_identity(
            self.model_name,
            config=config.to_dict() if hasattr(config, "to_dict") else vars(config),
            weight_path=weight_path,
            parameter_count=_model_parameter_count(self.model),
        )
        self.runtime_environment = esm2_runtime_environment(self.device)

    @staticmethod
    def _residue_mask(inputs, attention_mask):
        special = inputs.get("special_tokens_mask")
        if special is None:
            special = torch.zeros_like(attention_mask)
            # ESM2's alphabet always places BOS first and EOS immediately
            # before padding.  This fallback preserves old tensor-only callers.
            special[:, 0] = attention_mask[:, 0]
            valid_lengths = attention_mask.long().sum(dim=1)
            for row, length in enumerate(valid_lengths.tolist()):
                if length > 1:
                    special[row, length - 1] = 1
        return attention_mask.bool() & ~special.bool()

    def encode(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        model_inputs = {key: value for key, value in inputs.items() if key != "special_tokens_mask"}
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        last_hidden_state = _last_hidden_state(outputs)
        attention_mask = inputs["attention_mask"]
        residue_mask = self._residue_mask(inputs, attention_mask)
        if not residue_mask.any(dim=1).all():
            raise ValueError("ESM2 sequence has no residue tokens after special-token masking")
        input_mask_expanded = (
            residue_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        if not bool(torch.isfinite(embeddings).all()):
            raise RuntimeError("ESM2 encoder produced non-finite pooled embeddings")
        return embeddings

    def encode_legacy_attention_mask_mean(self, inputs):
        """Preserve historical BOS/EOS-inclusive pooling for legacy artifacts."""

        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key != "special_tokens_mask"
        }
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        hidden = _last_hidden_state(outputs)
        mask = model_inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        if not bool(torch.isfinite(embeddings).all()):
            raise RuntimeError("legacy ESM2 encoder produced non-finite pooled embeddings")
        return embeddings

    def encode_residues(self, inputs, sequence_lengths, max_length):
        """Backward-compatible tensor-only residue extraction."""
        if "special_tokens_mask" not in inputs:
            special = torch.ones_like(inputs["attention_mask"])
            for index, raw_length in enumerate(sequence_lengths):
                count = min(int(raw_length), int(max_length))
                special[index, 1 : 1 + count] = 0
            inputs = {**inputs, "special_tokens_mask": special}
        return self.encode_residue_batch(inputs, sequence_lengths, max_length).features

    def encode_residue_batch(self, inputs, sequence_lengths, max_length):
        return _encode_hf_residue_batch(self, inputs, sequence_lengths, max_length)


class EsmcEncoder:
    def __init__(self, model_name=ESMC_OFFICIAL_600M_MODEL_ID, device=None):
        self.model_name = model_name
        self.device = torch.device(device if device is not None else "cpu")
        self.backend = esmc_backend_for_source(model_name)
        if self.backend == ESMC_BACKEND_OFFICIAL:
            resolved = esmc_variant_for_source(model_name)
            if resolved is None:
                raise ValueError(f"cannot resolve official ESM-C variant: {model_name}")
            self.variant_key, self.variant = resolved
            self.model, weight_path, weight_sha256, runtime_version = self._load_official_model()
            self.output_dim = int(self.variant["hidden_size"])
            self.model_identity = esmc_model_identity(
                model_name,
                model_id=str(self.variant["model_id"]),
                weight_path=weight_path,
                weight_sha256=weight_sha256,
                runtime_version=runtime_version,
                parameter_count=_model_parameter_count(self.model),
            )
            self.runtime_environment = esmc_runtime_environment(self.device)
            return
        if AutoModelForMaskedLM is None:
            raise ImportError("ESM++ encoding requires the 'transformers' library.")
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True, use_safetensors=True
            ).to(self.device)
        except Exception:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name, trust_remote_code=True, use_safetensors=False
            ).to(self.device)
        self.model.eval()
        self.output_dim = int(getattr(self.model.config, "hidden_size", 0) or 0)
        try:
            runtime_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            runtime_version = ""
        self.model_identity = esmc_model_identity(
            model_name,
            runtime_version=runtime_version,
            parameter_count=_model_parameter_count(self.model),
        )
        self.runtime_environment = esmc_runtime_environment(self.device)
        self.runtime_environment.update(
            {
                "backend": ESMC_BACKEND_ESMPLUSPLUS,
                "transformers_version": runtime_version,
            }
        )

    def _load_official_model(self):
        runtime = require_official_esmc_runtime()
        runtime_version = str(runtime["installed_version"])
        from esm.models.esmc import ESMC
        from esm.pretrained import ESMC_300M_202412, ESMC_600M_202412
        from esm.tokenization import get_esmc_model_tokenizers
        from esm.utils.constants.esm3 import data_root

        model_id = str(self.variant["model_id"])
        if os.path.exists(os.path.expanduser(str(self.model_name))):
            weight_path, weight_sha256 = validate_official_esmc_weight(
                str(self.model_name), model_id=model_id
            )
            with torch.device(self.device):
                model = ESMC(
                    d_model=int(self.variant["hidden_size"]),
                    n_heads=int(self.variant["num_attention_heads"]),
                    n_layers=int(self.variant["num_hidden_layers"]),
                    tokenizer=get_esmc_model_tokenizers(),
                    use_flash_attn=False,
                ).eval()
            state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
        else:
            model_root = data_root(str(self.variant["runtime_key"]))
            weight_path, weight_sha256 = validate_official_esmc_weight(
                str(model_root), model_id=model_id
            )
            builder = ESMC_300M_202412 if self.variant_key == "300m" else ESMC_600M_202412
            model = builder(device=self.device, use_flash_attn=False)
        model.eval()
        return model, weight_path, weight_sha256, runtime_version

    def _official_hidden(self, inputs):
        sequence_tokens = inputs["sequence_tokens"].to(self.device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            outputs = self.model(sequence_tokens=sequence_tokens)
        if outputs.embeddings is None:
            raise RuntimeError("official ESM-C did not return embeddings")
        return outputs.embeddings

    def token_hidden(self, inputs):
        if self.backend == ESMC_BACKEND_OFFICIAL:
            return self._official_hidden(inputs)
        model_inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key != "special_tokens_mask"
        }
        with torch.no_grad():
            outputs = self.model(**model_inputs, output_hidden_states=True)
        return _last_hidden_state(outputs)

    def encode(self, inputs):
        device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        last_hidden_state = self.token_hidden(device_inputs)
        attention_mask = device_inputs["attention_mask"].bool()
        if getattr(self, "backend", None) == ESMC_BACKEND_OFFICIAL:
            attention_mask = attention_mask & ~device_inputs["special_tokens_mask"].bool()
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state = last_hidden_state.float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        return embeddings

    def encode_residue_batch(self, inputs, sequence_lengths, max_length):
        if getattr(self, "backend", None) == ESMC_BACKEND_OFFICIAL:
            device_inputs = {key: value.to(self.device) for key, value in inputs.items()}
            hidden = self._official_hidden(device_inputs)
            valid_tokens = (
                device_inputs["attention_mask"].bool()
                & ~device_inputs["special_tokens_mask"].bool()
            )
            rows = []
            masks = []
            for index, raw_length in enumerate(sequence_lengths):
                residue_count = min(int(raw_length), int(max_length))
                residue_states = hidden[index][valid_tokens[index]]
                if residue_states.shape[0] != residue_count:
                    raise RuntimeError(
                        f"ESM-C token/residue alignment failed for sample {index}: "
                        f"states={residue_states.shape[0]}, residues={residue_count}"
                    )
                padded = hidden.new_zeros((max_length, hidden.shape[-1]))
                padded_mask = torch.zeros(max_length, dtype=torch.bool, device=hidden.device)
                if residue_count:
                    padded[:residue_count] = residue_states
                    padded_mask[:residue_count] = True
                rows.append(padded.detach().cpu().to(dtype=torch.float16))
                masks.append(padded_mask.detach().cpu())
            if not rows:
                return ResidueEncoding(
                    torch.empty((0, max_length, self.output_dim), dtype=torch.float16),
                    torch.empty((0, max_length), dtype=torch.bool),
                )
            return ResidueEncoding(torch.stack(rows), torch.stack(masks))
        return _encode_hf_residue_batch(self, inputs, sequence_lengths, max_length)


class Esm3Encoder:
    output_dim = ESM3_HIDDEN_SIZE

    def __init__(
        self,
        model_name=ESM3_OFFICIAL_MODEL_ID,
        device=None,
        sdpa_backend=ESM3_DEFAULT_SDPA_BACKEND,
    ):
        self.model_name = str(model_name or ESM3_OFFICIAL_MODEL_ID).strip()
        self.device = device if device is not None else torch.device("cpu")
        self.sdpa_backend = validate_esm3_sdpa_backend(sdpa_backend)
        require_official_esm3_runtime()
        try:
            from esm.models.esm3 import ESM3
            import esm.pretrained as esm_pretrained
        except Exception as e:
            raise ImportError("Esm3Encoder requires 'esm' library. Run `pip install esm`.") from e
        from pathlib import Path

        local_root = Path(os.path.expanduser(self.model_name))
        model_key = normalize_esm3_model_name(self.model_name)
        if local_root.is_dir():
            weights_file = local_root / "data" / "weights" / "esm3_sm_open_v1.pth"
            if not weights_file.is_file():
                raise FileNotFoundError(f"ESM3 local source is incomplete; expected {weights_file}")
            # The SDK builder closes over this module-level function. Scope the
            # override to model construction so another route cannot observe a
            # different local source after the constructor returns.
            previous_root = esm_pretrained.data_root
            esm_pretrained.data_root = lambda _model: local_root
            try:
                self.client = ESM3.from_pretrained("esm3_sm_open_v1", device=self.device)
            finally:
                esm_pretrained.data_root = previous_root
        else:
            if model_key != "esm3_sm_open_v1":
                raise ValueError(
                    "unsupported ESM3 model source; use "
                    f"{ESM3_OFFICIAL_MODEL_ID!r} or a complete local SDK tree"
                )
            self.client = ESM3.from_pretrained(model_key, device=self.device)
        self.client = self.client.to(str(self.device)).eval()
        self.output_dim = (
            int(getattr(self.client, "encoder", None).embed_dim)
            if getattr(getattr(self.client, "encoder", None), "embed_dim", None)
            else ESM3_HIDDEN_SIZE
        )
        if self.output_dim != ESM3_HIDDEN_SIZE:
            raise RuntimeError(
                "ESM3 architecture signature mismatch: "
                f"expected hidden_size={ESM3_HIDDEN_SIZE} observed={self.output_dim}"
            )
        self.model_identity = esm3_model_identity(
            self.model_name,
            parameter_count=_model_parameter_count(self.client),
        )
        self.model_identity["output_dim"] = self.output_dim
        self.model_identity["sdpa_backend"] = self.sdpa_backend
        self.runtime_environment = esm3_runtime_environment(self.device)
        self.runtime_environment["sdpa_backend"] = self.sdpa_backend

    def _sdpa_context(self):
        if (
            getattr(self, "sdpa_backend", ESM3_DEFAULT_SDPA_BACKEND) != "math"
            or self.device.type != "cuda"
        ):
            return contextlib.nullcontext()
        return torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])

    def _embedding_and_residue_states(self, client, sequence, constants):
        from esm.sdk.api import ESMProtein, LogitsConfig

        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        with self._sdpa_context():
            output = client.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )
        emb = output.embeddings
        if emb.ndim == 3:
            emb = emb.squeeze(0)
        elif emb.ndim == 1:
            emb = emb.unsqueeze(0)
        if emb.ndim != 2:
            raise RuntimeError(f"Unexpected ESM3 embedding shape: {tuple(emb.shape)}")
        token_ids = getattr(protein_tensor, "sequence", None)
        if token_ids is None:
            raise RuntimeError("ESM3 residue extraction did not return sequence token ids")
        token_ids = token_ids.reshape(-1).to(emb.device)
        special_ids = {
            int(getattr(constants, name))
            for name in (
                "SEQUENCE_BOS_TOKEN",
                "SEQUENCE_EOS_TOKEN",
                "SEQUENCE_PAD_TOKEN",
                "SEQUENCE_CHAINBREAK_TOKEN",
                "SEQUENCE_MASK_TOKEN",
            )
            if hasattr(constants, name)
        }
        keep = torch.ones_like(token_ids, dtype=torch.bool)
        for special_id in special_ids:
            keep &= token_ids != special_id
        residue_states = emb[keep]
        expected = esm3_residue_count(sequence)
        if residue_states.shape[0] != expected:
            raise RuntimeError(
                "ESM3 token/residue alignment failed: "
                f"states={residue_states.shape[0]}, residues={expected}"
            )
        if int(residue_states.shape[1]) != int(
            getattr(self, "output_dim", residue_states.shape[1])
        ):
            if hasattr(self, "model_identity"):
                raise RuntimeError(
                    "ESM3 hidden size mismatch: "
                    f"expected={self.output_dim} observed={residue_states.shape[1]}"
                )
            self.output_dim = int(residue_states.shape[1])
        if not torch.isfinite(residue_states).all():
            raise RuntimeError("ESM3 produced non-finite residue embeddings")
        return emb, residue_states

    def encode_residue_batch(self, sequences, max_length):
        import importlib

        # Import the leaf module directly so lightweight SDK test doubles and
        # installations that do not expose ``esm.utils`` on the package still
        # work when the constants module itself is available.
        esm3_constants = importlib.import_module("esm.utils.constants.esm3")

        max_length = validate_esm3_max_residues(max_length)
        normalized = normalize_esm3_sequences(sequences)
        all_tokens = []
        all_masks = []
        with torch.inference_mode():
            for idx, raw_seq in enumerate(normalized):
                seq = truncate_esm3_sequence(raw_seq, max_length)
                _, residue_states = self._embedding_and_residue_states(
                    self.client, seq, esm3_constants
                )
                if residue_states.shape[0] != esm3_residue_count(seq):
                    raise RuntimeError(
                        f"ESM3 token/residue alignment failed for sample {idx}: "
                        f"states={residue_states.shape[0]}, residues={esm3_residue_count(seq)}"
                    )
                padded = residue_states.new_zeros((max_length, self.output_dim))
                mask = torch.zeros(max_length, dtype=torch.bool, device=residue_states.device)
                residue_count = residue_states.shape[0]
                if residue_count:
                    padded[:residue_count] = residue_states
                    mask[:residue_count] = True
                all_tokens.append(padded.detach().cpu().to(dtype=torch.float16))
                all_masks.append(mask.detach().cpu())
        if len(all_tokens) == 0:
            return ResidueEncoding(
                torch.empty((0, max_length, self.output_dim), dtype=torch.float16),
                torch.empty((0, max_length), dtype=torch.bool),
            )
        return ResidueEncoding(torch.stack(all_tokens, dim=0), torch.stack(all_masks, dim=0))

    def encode_token_sequences(self, sequences, max_length):
        return self.encode_residue_batch(sequences, max_length).features

    def encode_sequences(self, sequences, max_length=None):
        import importlib

        normalized = normalize_esm3_sequences(sequences)
        limit = validate_esm3_max_residues(ESM3_MAX_RESIDUES if max_length is None else max_length)
        if not normalized:
            return torch.empty((0, self.output_dim), dtype=torch.float32)
        esm3_constants = importlib.import_module("esm.utils.constants.esm3")
        pooled = []
        with torch.inference_mode():
            for index, raw_sequence in enumerate(normalized):
                sequence = truncate_esm3_sequence(raw_sequence, limit)
                _, residue_states = self._embedding_and_residue_states(
                    self.client, sequence, esm3_constants
                )
                if residue_states.numel() == 0:
                    raise RuntimeError("ESM3 sequence produced no residue embeddings")
                pooled.append(residue_states.float().mean(dim=0))
        result = torch.stack(pooled, dim=0)
        if not torch.isfinite(result).all():
            raise RuntimeError("ESM3 produced non-finite sequence embeddings")
        return result
