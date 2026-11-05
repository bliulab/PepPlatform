from pepplatform.stream_compat import normalize_stream_fields, install_legacy_class_exports, upgrade_strategy_contract, is_pair_bow_route, normalize_attention_contract
from pepplatform.task_compat import normalize_task_type

import csv
import gc
import json
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from argparse import Namespace
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:
    xgboost = None
    XGBClassifier = None

try:
    import lightgbm
    from lightgbm import Booster as LightGBMBooster
except ImportError:
    lightgbm = None
    LightGBMBooster = None

from pepplatform.kmer import validate_kmer_k
from pepplatform.lsa import LSAContractError, validate_lsa_contract
from pepplatform.tsvd import TSVDContractError, validate_tsvd_contract
from pepplatform.pca import (
    PCAContractError,
    transform_pca_held_out,
    validate_pca_contract,
)
from pepplatform.kpca import (
    KPCAContractError,
    transform_kpca_held_out,
    validate_kpca_contract,
)
from pepplatform.lda import LDAContractError, validate_lda_contract
from pepplatform.plsa import (
    PLSA_CONTRACT_VERSION,
    PLSAContractError,
    validate_plsa_contract,
)
from pepplatform.models import AlignedGatedClassifier
from pepplatform.pair_fusion import (
    AUDITED_PAIR_FUSION_MODES,
    AlignedGatedFusion,
    PairFusionError,
    validate_aligned_gated_contract,
    fuse_pair_embeddings as fuse_audited_pair_embeddings,
    normalize_pair_fusion_mode,
    validate_pair_fusion_contract,
)
from pepplatform.pair_strategy import (
    PairStrategyContractError,
    join_single_stream_batches,
    validate_two_stream_strategy_contract,
    validate_single_stream_strategy_contract,
)
from pepplatform.covariance import (
    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
    COVARIANCE_ENCODERS,
    validate_covariance_encoder_contract,
    validate_covariance_lag,
    validate_covariance_max_raw_length,
)
from pepplatform.pdt import (
    PDT_DEFAULT_LAG,
    PDT_DEFAULT_MAX_RAW_LENGTH,
    PDT_ENCODERS,
    validate_pdt_encoder_contract,
    validate_pdt_lag,
    validate_pdt_max_raw_length,
)
from pepplatform.frequency import (
    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    FREQUENCY_ENCODER_CONTRACT_VERSION,
    FREQUENCY_LEGACY_CONTRACT_VERSION,
    sha256_file as frequency_sha256_file,
    validate_frequency_max_raw_length,
)
from pepplatform.pssm_covariance import (
    HYBRID_PSSM_COVARIANCE_ENCODERS,
    PSSM_COVARIANCE_ENCODERS,
    pssm_covariance_lag_key,
    pssm_covariance_profile_feature_type,
    validate_pssm_covariance_encoder_contract,
    validate_pssm_covariance_lag,
)
from pepplatform.dr import (
    DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
    DR_DEFAULT_MAX_RAW_LENGTH,
    DR_CONTRACT_VERSION,
    DRContractError,
    dr_contract_metadata,
    validate_dr_max_distance,
    validate_dr_max_raw_length,
    validate_dr_unknown_residue_policy,
)
from pepplatform.dt import (
    DT_CONTRACT_VERSION,
    DTContractError,
    dt_contract_metadata,
    validate_dt_max_distance,
)
from pepplatform.top_ngram import (
    TOP_N_GRAM_CONTRACT_VERSION,
    TOP_N_GRAM_FALLBACK_POLICY,
    TOP_N_GRAM_FALLBACK_POLICY_VERSION,
    TOP_N_GRAM_STRICT_PROFILE_POLICY,
    TOP_N_GRAM_TIE_POLICY,
    TopNGramContractError,
    normalize_top_ngram_sequence,
    top_ngram_feature_dim,
    validate_top_ngram_n,
)
from pepplatform.bpe import (
    BPE_CONTRACT_VERSION,
    BPE_TOKENIZERS_VERSION,
    BPE_VARIANT,
    BPEContractError,
    inspect_bpe_artifact,
    load_bpe_provenance,
)
from pepplatform.wordpiece import (
    WORDPIECE_CONTRACT_VERSION,
    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
    WORDPIECE_TOKENIZERS_VERSION,
    WORDPIECE_VARIANT,
    WordPieceContractError,
    inspect_wordpiece_artifact,
    load_wordpiece_provenance,
    validate_wordpiece_max_raw_length,
    validate_wordpiece_provenance,
)
from pepplatform.unigram import (
    UNIGRAM_CONTRACT_VERSION,
    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    UNIGRAM_TOKENIZERS_VERSION,
    UNIGRAM_VARIANT,
    UnigramContractError,
    inspect_unigram_artifact,
    load_unigram_provenance,
    unigram_provenance_path,
    validate_unigram_max_raw_length,
    validate_unigram_provenance,
)
from pepplatform.workflow import (
    LGB_EXECUTION_POLICY,
    PRETRAINED_TOKENIZERS,
    PROFILE_DEPENDENT_INFERENCE_REASON,
    normalize_encoder_name,
    normalize_tokenizer_name,
    pretrained_model_local_candidates,
    resolve_pretrained_model_source,
    uses_profile_features,
    validate_prediction_head_artifact_config,
)
from pepplatform.protbert import (
    PROTBERT_OFFICIAL_MODEL_ID,
    PROTBERT_POOLING,
    PROTBERT_REPRESENTATION_CONTRACT_VERSION,
    validate_protbert_model_identity,
)
from pepplatform.esm3 import ESM3_POOLING
from pepplatform.web.profile_runtime import (
    profile_environment_readiness,
    profile_resource_lock,
)
from pepplatform.sequence_profiles import (
    PROFILE_CONTRACT_VERSION,
    PSSM_HYBRID_FEATURE_TYPES,
    PSFM_HYBRID_FEATURE_TYPES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ENCODERS = {
    "pssm",
    "pssm_blosum62_hybrid",
    "psfm",
    "psfm_blosum62_hybrid",
    "ac_pssm",
    "cc_pssm",
    "acc_pssm",
    "ac_pssm_blosum62_hybrid",
    "cc_pssm_blosum62_hybrid",
    "acc_pssm_blosum62_hybrid",
}
PROFILE_TOKENIZERS = {"top-n-gram", "dt"}
_PROFILE_DIAGNOSTICS = ContextVar("profile_inference_diagnostics", default=None)
_INFERENCE_PROGRESS = ContextVar("inference_progress_callback", default=None)
_INFERENCE_TIMING = ContextVar("inference_timing", default=None)
_LEGACY_TFIDF_IDF_CACHE = {}
_RAW_INDEX_VECTOR_BUNDLE_TYPES = {
    "sequence_classical",
    "pair_vector_nn",
    "pair_vector_classical",
    "pair_graph_nn",
}
_CLASSICAL_BUNDLE_TYPES = {
    "sequence_classical",
    "pretrained_embedding_classical",
    "pretrained_residue_embedding_classical",
    "sequence_embedding_classical",
    "pair_vector_classical",
    "pair_embedding_classical",
}
_RF_PREDICTOR_CONTRACT_VERSION = 1
_EXTRATREES_PREDICTOR_CONTRACT_VERSION = 1
_SVM_PREDICTOR_CONTRACT_VERSION = 1
_CALIBRATED_SVM_PREDICTOR_CONTRACT_VERSION = 1
_XGB_PREDICTOR_CONTRACT_VERSION = 1
_XGB_MANIFEST_SCHEMA_VERSION = 1
_LGB_PREDICTOR_CONTRACT_VERSION = 1
_LGB_MANIFEST_SCHEMA_VERSION = 1
_LOGREG_PREDICTOR_CONTRACT_VERSION = 1
_HISTGB_PREDICTOR_CONTRACT_VERSION = 1
_PAIR_ATTENTION_CONTRACTS = {
    "cross_attn": {
        "implementation": "pepplatform_pair_token_cross_attention",
        "direction": "peptide_query_partner_key_value",
        "attention_parameterization": "one directional attention module",
        "readout": "unidirectional_cross_attention_pair_fusion_8h",
    },
    "bi_cross_attn": {
        "implementation": "pepplatform_pair_token_bidirectional_cross_attention",
        "direction": "peptide_query_partner_key_value_and_partner_query_peptide_key_value",
        "attention_parameterization": "independent parameters for each direction",
        "readout": "bidirectional_cross_attention_pair_fusion_8h",
    },
}


def _matches_pair_attention_contract(contract, interaction_model):
    expected = _PAIR_ATTENTION_CONTRACTS.get(str(interaction_model))
    if not isinstance(contract, dict) or expected is None:
        return False
    version = contract.get("contract_version")
    hidden_dim = contract.get("hidden_dim")
    heads = contract.get("heads")
    embedding_dim = contract.get("embedding_dim")
    return (
        isinstance(version, int)
        and not isinstance(version, bool)
        and version == 1
        and contract.get("status") == "verified_platform_variant"
        and all(contract.get(key) == value for key, value in expected.items())
        and all(
            isinstance(contract.get(key), str) and bool(contract.get(key).strip())
            for key in ("input_adapter", "attention", "masking", "output")
        )
        and isinstance(hidden_dim, int)
        and not isinstance(hidden_dim, bool)
        and hidden_dim > 0
        and isinstance(heads, int)
        and not isinstance(heads, bool)
        and heads > 0
        and hidden_dim % heads == 0
        and embedding_dim == hidden_dim * 8
    )


def _validate_loaded_pair_attention_model(model, contract, interaction_model):
    from scripts.train_pair_pipeline import pair_attention_predictor_contract

    contract = normalize_attention_contract(contract)
    actual = pair_attention_predictor_contract(model)
    if (
        actual is None
        or not _matches_pair_attention_contract(actual, interaction_model)
        or any(contract.get(key) != value for key, value in actual.items())
    ):
        raise InferenceUnsupportedError(
            f"loaded {interaction_model} model does not match its predictor contract"
        )


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InferenceUnsupportedError(RuntimeError):
    def __init__(
        self, message, code="inference_unsupported", stage="inference", hint="", technical_detail=""
    ):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.hint = hint
        self.technical_detail = technical_detail

    def to_detail(self):
        detail = {
            "code": self.code,
            "stage": self.stage,
            "message": str(self),
        }
        if self.hint:
            detail["hint"] = self.hint
        if self.technical_detail:
            detail["technical_detail"] = self.technical_detail
        return detail


class _Esm3RuntimeEntry:
    def __init__(self, key, encoder):
        self.key = key
        self.encoder = encoder
        self.lock = threading.RLock()
        self.active = 0
        self.last_used = time.monotonic()
        self.timer = None


class _Esm3RuntimeHandle:
    """A cached encoder facade that leases the GPU model per forward call."""

    def __init__(self, pool, entry):
        self._pool = pool
        self._entry = entry

    def __getattr__(self, name):
        return getattr(self._entry.encoder, name)

    def encode_sequences(self, sequences, max_length=None):
        with self._pool.lease(self._entry) as encoder:
            return encoder.encode_sequences(sequences, max_length=max_length)

    def encode_residue_batch(self, sequences, max_length):
        with self._pool.lease(self._entry) as encoder:
            return encoder.encode_residue_batch(sequences, max_length)


class _Esm3RuntimePool:
    def __init__(self):
        self._condition = threading.Condition()
        self._entry = None
        self._loading = False
        self._suspended_reason = ""
        self._requests_active = 0

    @staticmethod
    def _idle_ttl_seconds():
        try:
            value = float(os.environ.get("PEPPLATFORM_ESM3_IDLE_TTL_SECONDS", "300"))
        except (TypeError, ValueError):
            value = 300.0
        return max(1.0, value)

    @staticmethod
    def _device_is_cuda(device):
        return torch.device(device).type == "cuda"

    def status(self):
        with self._condition:
            if self._suspended_reason:
                cache_status = "suspended"
            elif self._loading:
                cache_status = "loading"
            elif self._entry is not None:
                cache_status = "warm"
            else:
                cache_status = "cold"
            return {
                "cache_status": cache_status,
                "suspended_reason": self._suspended_reason,
                "idle_ttl_seconds": self._idle_ttl_seconds(),
                "active_requests": self._requests_active,
            }

    def _raise_if_suspended(self):
        if self._suspended_reason:
            raise InferenceUnsupportedError(
                "ESM3 Web inference is paused while a training job uses the GPU",
                code="gpu_busy_training",
                stage="resource_wait",
                hint="Wait for the training job to finish, then retry inference.",
            )

    @staticmethod
    def _cancel_timer(entry):
        timer = entry.timer
        entry.timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_eviction(self, entry):
        self._cancel_timer(entry)
        timer = threading.Timer(
            self._idle_ttl_seconds(),
            self._evict_if_idle,
            args=(entry,),
        )
        timer.daemon = True
        entry.timer = timer
        timer.start()

    def _evict_if_idle(self, entry):
        should_clear = False
        with self._condition:
            if self._entry is not entry:
                return
            elapsed = time.monotonic() - entry.last_used
            if entry.active or elapsed < self._idle_ttl_seconds():
                self._schedule_eviction(entry)
                return
            self._entry = None
            entry.timer = None
            should_clear = True
            self._condition.notify_all()
        if should_clear:
            del entry.encoder
            gc.collect()
            if self._device_is_cuda(getattr(entry, "device", "cpu")) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_or_load(self, key, loader):
        retired = None
        while True:
            with self._condition:
                self._raise_if_suspended()
                if self._entry is not None and self._entry.key == key:
                    self._entry.last_used = time.monotonic()
                    self._cancel_timer(self._entry)
                    return _Esm3RuntimeHandle(self, self._entry), True
                if self._loading:
                    self._condition.wait()
                    continue
                if self._entry is not None:
                    if self._entry.active:
                        self._condition.wait()
                        continue
                    old = self._entry
                    self._entry = None
                    self._cancel_timer(old)
                    retired = old
                self._loading = True
                break
        if retired is not None:
            device = getattr(retired, "device", "cpu")
            del retired.encoder
            gc.collect()
            if self._device_is_cuda(device) and torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            encoder = loader()
        except Exception:
            with self._condition:
                self._loading = False
                self._condition.notify_all()
            raise
        entry = _Esm3RuntimeEntry(key, encoder)
        entry.device = getattr(encoder, "device", "cpu")
        with self._condition:
            self._entry = entry
            self._loading = False
            self._schedule_eviction(entry)
            self._condition.notify_all()
        return _Esm3RuntimeHandle(self, entry), False

    @contextmanager
    def request(self):
        with self._condition:
            self._raise_if_suspended()
            self._requests_active += 1
        try:
            yield
        finally:
            with self._condition:
                self._requests_active = max(0, self._requests_active - 1)
                self._condition.notify_all()

    @contextmanager
    def lease(self, entry):
        with self._condition:
            self._raise_if_suspended()
            if self._entry is not entry:
                raise InferenceUnsupportedError(
                    "ESM3 runtime was evicted before the request acquired it",
                    code="esm3_runtime_evicted",
                    stage="feature_extraction",
                )
            self._cancel_timer(entry)
            entry.active += 1
            entry.last_used = time.monotonic()
        try:
            with entry.lock:
                yield entry.encoder
        finally:
            with self._condition:
                entry.active = max(0, entry.active - 1)
                entry.last_used = time.monotonic()
                if entry.active == 0 and self._entry is entry:
                    self._schedule_eviction(entry)
                self._condition.notify_all()

    def suspend(self, reason="training"):
        with self._condition:
            self._suspended_reason = reason
            while (
                self._loading
                or self._requests_active
                or (self._entry is not None and self._entry.active)
            ):
                self._condition.wait()
            entry = self._entry
            self._entry = None
            if entry is not None:
                self._cancel_timer(entry)
            self._condition.notify_all()
        if entry is not None:
            device = getattr(entry, "device", "cpu")
            del entry.encoder
            gc.collect()
            if self._device_is_cuda(device) and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def resume(self):
        with self._condition:
            self._suspended_reason = ""
            self._condition.notify_all()

    def close(self):
        self.suspend("shutdown")
        self.resume()


_ESM3_RUNTIME_POOL = _Esm3RuntimePool()


def _report_inference_progress(stage, **payload):
    callback = _INFERENCE_PROGRESS.get()
    if callable(callback):
        callback({"stage": str(stage), **payload})


class ModelService:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def esm3_runtime_status():
        return _ESM3_RUNTIME_POOL.status()

    @staticmethod
    def suspend_esm3_runtime(reason="training"):
        _ESM3_RUNTIME_POOL.suspend(reason)

    @staticmethod
    def resume_esm3_runtime():
        _ESM3_RUNTIME_POOL.resume()

    @staticmethod
    def close():
        _ESM3_RUNTIME_POOL.close()

    @staticmethod
    def _metadata_uses_esm3(metadata):
        metadata = normalize_stream_fields(metadata)
        def visit(value):
            if isinstance(value, dict):
                return any(visit(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(visit(item) for item in value)
            return "esm3" in str(value or "").strip().lower()

        return visit(metadata)

    @staticmethod
    def _validate_pair_strategy_metadata(metadata, peptide=None, partner=None):
        metadata = normalize_stream_fields(metadata)
        pair = metadata.get("pair") if isinstance(metadata.get("pair"), dict) else {}
        strategy = str(pair.get("pair_strategy") or "").strip().lower()
        if strategy != "two_stream":
            return None
        try:
            contract = validate_two_stream_strategy_contract(
                metadata.get("pair_strategy_contract"),
                peptide=peptide,
                partner=partner,
            )
        except PairStrategyContractError as exc:
            contract = None
            if (
                exc.code == "two_stream_strategy_contract_shape_mismatch"
                and peptide is not None
                and partner is not None
            ):
                contract = ModelService._legacy_concatenated_two_stream_contract(
                    metadata,
                    peptide,
                    partner,
                )
            if contract is None:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="input" if peptide is not None else "bundle",
                ) from exc
            warnings.warn(
                "legacy pair_token_nn artifact recorded the concatenated training "
                "tensor length for both homogeneous Two-stream roles; the two "
                "independently reconstructed role tensors passed the strict "
                "compatibility contract",
                RuntimeWarning,
                stacklevel=2,
            )
        for key in ("stream_mode", "interaction_level"):
            recorded = str(pair.get(key, "") or "").strip().lower()
            if recorded and recorded != contract[key]:
                raise InferenceUnsupportedError(
                    f"Two-stream {key} does not match pair_strategy_contract",
                    code="two_stream_strategy_contract_invalid",
                    stage="bundle",
                )
        return contract

    @staticmethod
    def _legacy_concatenated_two_stream_contract(metadata, peptide, partner):
        """Recognize one historical, otherwise fully verified contract defect.

        Early homogeneous pair-token artifacts built the strategy contract from
        the already concatenated training tensor.  Consequently both role
        descriptors recorded ``2 * seq_len`` even though inference correctly
        reconstructs two independent ``seq_len`` tensors before concatenation.
        No other descriptor mismatch is relaxed here.
        """
        metadata = normalize_stream_fields(metadata)

        if metadata.get("bundle_type") != "pair_token_nn":
            return None
        if metadata.get("streams") not in (None, {}):
            return None
        pair = metadata.get("pair") if isinstance(metadata.get("pair"), dict) else {}
        classic = metadata.get("classic") if isinstance(metadata.get("classic"), dict) else {}
        try:
            seq_len = int(pair.get("seq_len"))
            token_dim = int(pair.get("token_dim"))
            partner_seq_len = int(pair.get("partner_seq_len"))
            partner_token_dim = int(pair.get("partner_token_dim"))
            classic_length = int(classic.get("max_length"))
            classic_token_dim = int(classic.get("token_dim"))
        except (TypeError, ValueError):
            return None
        exact_signature = (
            seq_len > 0
            and token_dim > 0
            and partner_seq_len == 0
            and partner_token_dim == 0
            and classic_length == seq_len
            and classic_token_dim == token_dim
            and str(classic.get("tokenizer") or "").strip().lower() == "kmer"
            and str(classic.get("encoder") or "").strip().lower() == "onehot"
            and str(pair.get("stream_mode") or "").strip().lower() == "homogeneous"
            and str(pair.get("interaction_level") or "").strip().lower() == "token"
            and str(pair.get("interaction_model") or "").strip().lower() == "cross_attn"
        )
        if not exact_signature:
            return None
        try:
            recorded = validate_two_stream_strategy_contract(metadata.get("pair_strategy_contract"))
        except PairStrategyContractError:
            return None
        if (
            recorded.get("stream_mode") != "homogeneous"
            or recorded.get("interaction_level") != "token"
            or recorded.get("predictor_type") != "classifier"
            or recorded.get("input_adapter") != "two_stream_token_interaction"
            or recorded.get("interaction_model") != "cross_attn"
        ):
            return None
        recorded_streams = recorded["streams"]
        if any(
            int(recorded_streams[role].get("token_length", -1)) != 2 * seq_len
            for role in ("peptide", "partner")
        ):
            return None
        adjusted = {
            **recorded,
            "streams": {
                role: {**recorded_streams[role], "token_length": seq_len}
                for role in ("peptide", "partner")
            },
            "compatibility_mode": "legacy_concatenated_role_length_v1",
        }
        try:
            validate_two_stream_strategy_contract(
                adjusted,
                peptide=peptide,
                partner=partner,
            )
        except PairStrategyContractError:
            return None
        return adjusted

    def predict(self, model_id, sequences=None, pairs=None, progress_callback=None):
        diagnostics = {}
        inference_timing = {
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "encoder_cache_hit": None,
            "encoder_load_ms": 0.0,
            "total_ms": 0.0,
        }
        started = time.perf_counter()

        def report(event):
            event = dict(event or {})
            event.setdefault("elapsed_ms", round((time.perf_counter() - started) * 1000.0, 2))
            if callable(progress_callback):
                try:
                    progress_callback(event)
                except Exception:
                    pass

        profile_token = _PROFILE_DIAGNOSTICS.set(diagnostics)
        progress_token = _INFERENCE_PROGRESS.set(report)
        timing_token = _INFERENCE_TIMING.set(inference_timing)
        response = None
        try:
            report({"stage": "validating_bundle"})
            report({"stage": "waiting_for_runtime"})
            with _ESM3_RUNTIME_POOL.request():
                response = self._predict(model_id, sequences=sequences, pairs=pairs)
        finally:
            inference_timing["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
            _INFERENCE_TIMING.reset(timing_token)
            _INFERENCE_PROGRESS.reset(progress_token)
            _PROFILE_DIAGNOSTICS.reset(profile_token)
        if response is None:  # pragma: no cover - exceptions leave this method early
            raise RuntimeError("prediction did not produce a response")
        diagnostics_payload = {}
        if diagnostics:
            diagnostics_payload["profile"] = diagnostics
        diagnostics_payload["inference"] = inference_timing
        response["diagnostics"] = diagnostics_payload
        return response

    def _predict(self, model_id, sequences=None, pairs=None):
        model_record = self.store.get_model(model_id)
        if not model_record:
            raise KeyError(model_id)
        bundle = Path(model_record["bundle_path"])
        metadata_path = bundle / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        )
        metadata = normalize_stream_fields(metadata)
        if str(metadata.get("bundle_type", "")).strip().lower() == "ensemble":
            return self._predict_ensemble(model_id, bundle, metadata, sequences, pairs)
        self._reject_raw_index_vector_bundle(metadata)
        profile_reason = self._profile_inference_unavailable_reason(metadata)
        if profile_reason:
            raise InferenceUnsupportedError(
                profile_reason,
                code=metadata.get("inference_unavailable_code")
                or metadata.get("reason_code")
                or "profile_features_require_sample_id",
                stage="bundle",
                hint="Re-run or re-export the experiment with Profile generation artifacts included.",
            )
        if not model_record["supports_inference"]:
            raise InferenceUnsupportedError(
                "this model route does not yet emit a reusable inference bundle"
            )
        args = Namespace(**metadata["args"])
        bundle_type = metadata.get("bundle_type")
        if bundle_type in {
            "pair_vector_nn",
            "pair_vector_classical",
            "pair_token_nn",
            "pair_embedding_nn",
            "pair_embedding_classical",
            "pair_graph_nn",
        }:
            self._validate_pair_strategy_metadata(metadata)
        _report_inference_progress("extracting_features", total=0, completed=0)
        if bundle_type in _CLASSICAL_BUNDLE_TYPES:
            self._validate_lsa_artifact(metadata, bundle)
            self._validate_lda_artifact(metadata, bundle)
            self._validate_plsa_artifact(metadata, bundle)
            self._validate_tsvd_artifact(metadata, bundle)
            self._validate_pca_artifact(metadata, bundle)
            self._validate_kpca_artifact(metadata, bundle)
            self._validate_rf_predictor_contract(metadata)
            self._validate_extratrees_predictor_contract(metadata)
            self._validate_svm_predictor_contract(metadata)
            self._validate_logreg_predictor_contract(metadata)
            self._validate_histgb_predictor_contract(metadata)
            self._validate_xgb_predictor_contract(metadata, bundle)
            self._validate_lgb_predictor_contract(metadata, bundle)
        pipeline = Namespace(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        sequence_pipeline = None
        score_kind = "probability"

        def get_sequence_pipeline():
            nonlocal sequence_pipeline, pipeline
            if sequence_pipeline is None:
                from scripts.train_sequence_pipeline import Pipeline

                sequence_pipeline = Pipeline(args)
                pipeline = sequence_pipeline
            return sequence_pipeline

        if bundle_type == "sequence_nn":
            sequences = self._require_sequences(sequences)
            pipeline = get_sequence_pipeline()
            inputs = self._extract_sequence_features(metadata, sequences, pipeline, bundle)
            if hasattr(inputs, "mask") and hasattr(inputs, "features"):
                inputs = self._transform_pretrained_residue_features(
                    bundle,
                    metadata,
                    inputs,
                )
            else:
                inputs = self._transform_pretrained_tensor_features(bundle, metadata, inputs)
            model = torch.load(
                bundle / "model.pt", map_location=pipeline.device, weights_only=False
            )
            model = model.to(pipeline.device)
            model.eval()
            self._validate_prediction_head_contract(metadata, model)
            scores = self._predict_tensor_model_in_batches(
                model,
                inputs,
                pipeline.device,
                metadata,
            )
        elif bundle_type == "sequence_classical":
            sequences = self._require_sequences(sequences)
            pipeline = get_sequence_pipeline()
            encoded = self._extract_sequence_features(metadata, sequences, pipeline, bundle)
            encoded_values = encoded.features if hasattr(encoded, "features") else encoded
            features = torch.stack(
                [item.view(-1) if item.ndim > 1 else item for item in encoded_values]
            )
            features = features.cpu().numpy().astype(np.float32)
            payload = self._load_classical_payload(bundle, metadata)
            features = self._transform_classical_features(payload, features)
            scores = self._predict_classical_scores(payload, features)
            score_kind = self._classical_score_kind(payload)
        elif bundle_type == "pretrained_embedding_nn":
            sequences = self._require_sequences(sequences)
            features = self._extract_pretrained_embeddings(metadata, sequences, pipeline.device)
            features = self._transform_pretrained_tensor_features(bundle, metadata, features)
            model = torch.load(
                bundle / "model.pt", map_location=pipeline.device, weights_only=False
            )
            model = model.to(pipeline.device)
            model.eval()
            self._validate_prediction_head_contract(metadata, model)
            scores = self._predict_tensor_model_in_batches(
                model,
                features,
                pipeline.device,
                metadata,
            )
        elif bundle_type == "pretrained_embedding_classical":
            sequences = self._require_sequences(sequences)
            features = self._extract_pretrained_embeddings(metadata, sequences, pipeline.device)
            features_np = features.cpu().numpy().astype(np.float32)
            payload = self._load_classical_payload(bundle, metadata)
            features_np = self._transform_classical_features(payload, features_np)
            scores = self._predict_classical_scores(payload, features_np)
            score_kind = self._classical_score_kind(payload)
        elif bundle_type == "pretrained_residue_nn":
            sequences = self._require_sequences(sequences)
            features = self._extract_pretrained_residue_features(
                metadata,
                sequences,
                pipeline.device,
            )
            features = self._transform_pretrained_residue_features(bundle, metadata, features)
            model = torch.load(
                bundle / "model.pt", map_location=pipeline.device, weights_only=False
            )
            model = model.to(pipeline.device)
            model.eval()
            self._validate_prediction_head_contract(metadata, model)
            scores = self._predict_residue_model_in_batches(
                model,
                features,
                pipeline.device,
                metadata,
            )
        elif bundle_type == "pretrained_residue_embedding_classical":
            sequences = self._require_sequences(sequences)
            features = self._extract_pretrained_residue_export_features(
                metadata,
                sequences,
                pipeline.device,
                bundle,
            )
            features_np = features.cpu().numpy().astype(np.float32)
            payload = self._load_classical_payload(bundle, metadata)
            features_np = self._transform_classical_features(payload, features_np)
            scores = self._predict_classical_scores(payload, features_np)
            score_kind = self._classical_score_kind(payload)
        elif bundle_type == "sequence_embedding_classical":
            sequences = self._require_sequences(sequences)
            pipeline = get_sequence_pipeline()
            features = self._extract_sequence_embedding_features(
                metadata, sequences, pipeline, bundle
            )
            features_np = features.cpu().numpy().astype(np.float32)
            payload = self._load_classical_payload(bundle, metadata)
            features_np = self._transform_classical_features(payload, features_np)
            scores = self._predict_classical_scores(payload, features_np)
            score_kind = self._classical_score_kind(payload)
        elif bundle_type == "pair_vector_nn":
            pairs = self._require_pairs(pairs)
            model = self._load_torch_model(bundle / "model.pt", pipeline.device)
            model = model.to(pipeline.device)
            model.eval()
            self._validate_prediction_head_contract(metadata, model)
            fusion_mode = (
                str((metadata.get("pair") or {}).get("fusion_mode", "") or "").strip().lower()
            )
            if fusion_mode in {"aligned_gated", "gated_trainable", "gated(trainable)"}:
                self._validate_aligned_gated_model(metadata, model)
                peptide_embeddings, protein_embeddings = self._extract_pair_embeddings(
                    metadata, pairs, pipeline.device, bundle=bundle
                )
                peptide_embeddings, protein_embeddings = self._transform_graph_pair_embeddings(
                    bundle,
                    metadata,
                    peptide_embeddings,
                    protein_embeddings,
                )
                with torch.no_grad():
                    scores = (
                        torch.sigmoid(
                            model(
                                peptide_embeddings.to(pipeline.device),
                                protein_embeddings.to(pipeline.device),
                            )
                        )
                        .cpu()
                        .numpy()
                    )
            else:
                features = self._extract_pair_vector_features(
                    metadata, pairs, pipeline.device, bundle=bundle
                )
                features = self._transform_pretrained_tensor_features(bundle, metadata, features)
                with torch.no_grad():
                    scores = torch.sigmoid(model(features.to(pipeline.device))).cpu().numpy()
        elif bundle_type == "pair_vector_classical":
            pairs = self._require_pairs(pairs)
            features = self._extract_pair_vector_features(
                metadata, pairs, pipeline.device, bundle=bundle
            )
            features_np = features.cpu().numpy().astype(np.float32)
            payload = self._load_classical_payload(bundle, metadata)
            features_np = self._transform_classical_features(payload, features_np)
            scores = self._predict_classical_scores(payload, features_np)
            score_kind = self._classical_score_kind(payload)
        elif bundle_type == "pair_token_nn":
            pairs = self._require_pairs(pairs)
            if metadata.get("streams"):
                peptide_tokens, partner_tokens = self._extract_pair_token_streams(
                    metadata,
                    pairs,
                    pipeline.device,
                    bundle=bundle,
                )
                model = self._load_heterogeneous_pair_token_model(
                    bundle,
                    metadata,
                    pipeline.device,
                )
                model.eval()
                self._validate_prediction_head_contract(metadata, model)
                scores = self._predict_pair_tensor_model_in_batches(
                    model,
                    peptide_tokens,
                    partner_tokens,
                    pipeline.device,
                    metadata,
                )
            else:
                features = self._extract_pair_token_features(
                    metadata, pairs, pipeline.device, bundle=bundle
                )
                model = self._load_pair_token_model(bundle, metadata, features, pipeline.device)
                model.eval()
                self._validate_prediction_head_contract(metadata, model)
                scores = self._predict_tensor_model_in_batches(
                    model, features, pipeline.device, metadata
                )
        elif bundle_type == "pair_sequence_nn":
            pairs = self._require_pairs(pairs)
            features = self._extract_pair_sequence_features(
                metadata, pairs, pipeline.device, bundle=bundle
            )
            model = torch.load(
                bundle / "model.pt", map_location=pipeline.device, weights_only=False
            )
            model = model.to(pipeline.device)
            model.eval()
            self._validate_prediction_head_contract(metadata, model)
            scores = self._predict_tensor_model_in_batches(
                model, features, pipeline.device, metadata
            )
        elif bundle_type == "pair_embedding_nn":
            pairs = self._require_pairs(pairs)
            features = self._extract_pair_embedding_features(
                metadata, pairs, pipeline.device, bundle=bundle
            )
            features = self._transform_pretrained_tensor_features(bundle, metadata, features)
            model = torch.load(
                bundle / "model.pt", map_location=pipeline.device, weights_only=False
            )
            model = model.to(pipeline.device)
            model.eval()
            self._validate_prediction_head_contract(metadata, model)
            scores = self._predict_tensor_model_in_batches(
                model, features, pipeline.device, metadata
            )
        elif bundle_type == "pair_embedding_classical":
            pairs = self._require_pairs(pairs)
            features = self._extract_pair_embedding_features(
                metadata, pairs, pipeline.device, bundle=bundle
            )
            features_np = features.cpu().numpy().astype(np.float32)
            payload = self._load_classical_payload(bundle, metadata)
            features_np = self._transform_classical_features(payload, features_np)
            scores = self._predict_classical_scores(payload, features_np)
            score_kind = self._classical_score_kind(payload)
        elif bundle_type == "pair_graph_nn":
            pairs = self._require_pairs(pairs)
            scores = self._predict_pair_graph(metadata, pairs, pipeline.device, bundle)
        else:
            raise InferenceUnsupportedError("unsupported bundle type")
        _report_inference_progress("running_predictor", total=1, completed=1)
        if bundle_type in {
            "pair_vector_nn",
            "pair_token_nn",
            "pair_sequence_nn",
            "pair_embedding_nn",
            "pair_graph_nn",
        }:
            score_kind = "probability"
        scores = np.asarray(scores)
        if scores.ndim == 1:
            scores = scores[:, None]
        labels = metadata.get("label_names", [])
        protocol_version = int(metadata.get("evaluation_protocol_version", 1) or 1)
        if bundle_type in {
            "pair_vector_nn",
            "pair_vector_classical",
            "pair_token_nn",
            "pair_sequence_nn",
            "pair_embedding_nn",
            "pair_embedding_classical",
            "pair_graph_nn",
        }:
            threshold = self._interaction_threshold(metadata)
            score_range = [0.0, 1.0] if score_kind == "probability" else None
            return {
                "model_id": model_id,
                "labels": labels,
                "evaluation_protocol_version": protocol_version,
                "legacy_evaluation_protocol": protocol_version < 2,
                "predictions": [
                    self._pair_prediction_record(pair, row, threshold, score_kind, score_range)
                    for pair, row in zip(pairs, scores)
                ],
            }
        thresholds = self._sequence_thresholds(metadata, len(labels) or scores.shape[1])
        return {
            "model_id": model_id,
            "labels": labels,
            "thresholds": thresholds,
            "score_kind": score_kind,
            "evaluation_protocol_version": protocol_version,
            "legacy_evaluation_protocol": protocol_version < 2,
            "predictions": [
                {
                    "sequence": sequence,
                    "scores": [float(value) for value in row],
                    "predicted_labels": [
                        labels[index] if index < len(labels) else f"label_{index + 1}"
                        for index, (value, threshold) in enumerate(zip(row, thresholds))
                        if float(value) >= float(threshold)
                    ],
                }
                for sequence, row in zip(sequences, scores)
            ],
        }

    def _predict_ensemble(self, model_id, bundle, metadata, sequences=None, pairs=None):
        metadata = normalize_stream_fields(metadata)
        if pairs is not None:
            raise InferenceUnsupportedError(
                "ensemble v1 supports sequence prediction only",
                code="ensemble_pair_unsupported",
                stage="bundle",
            )
        ensemble_path = Path(bundle) / "ensemble.json"
        if not ensemble_path.is_file():
            raise InferenceUnsupportedError(
                "ensemble bundle is missing ensemble.json",
                code="ensemble_manifest_missing",
                stage="bundle",
            )
        payload = json.loads(ensemble_path.read_text(encoding="utf-8"))
        members = payload.get("members")
        weights = payload.get("weights")
        if not isinstance(members, list) or len(members) < 2:
            raise InferenceUnsupportedError(
                "ensemble bundle must contain at least two members",
                code="ensemble_members_invalid",
                stage="bundle",
            )
        if not isinstance(weights, list) or len(weights) != len(members):
            raise InferenceUnsupportedError(
                "ensemble member/weight mismatch",
                code="ensemble_weights_invalid",
                stage="bundle",
            )
        score_rows = None
        labels = None
        for member, weight in zip(members, weights):
            member_id = str(member.get("model_id") or "").strip()
            if not member_id:
                raise InferenceUnsupportedError(
                    "ensemble members must be registered model ids for Web inference",
                    code="ensemble_member_not_registered",
                    stage="bundle",
                )
            member_record = self.store.get_model(member_id)
            if not member_record:
                raise InferenceUnsupportedError(
                    f"ensemble member model is not available: {member_id}",
                    code="ensemble_member_missing",
                    stage="bundle",
                )
            member_metadata_path = Path(member_record["bundle_path"]) / "metadata.json"
            member_metadata = (
                json.loads(member_metadata_path.read_text(encoding="utf-8"))
                if member_metadata_path.is_file()
                else {}
            )
            if str(member_metadata.get("bundle_type", "")).strip().lower() == "ensemble":
                raise InferenceUnsupportedError(
                    "nested ensembles are not supported",
                    code="ensemble_nested_unsupported",
                    stage="bundle",
                )
            response = self._predict(member_id, sequences=sequences)
            predictions = response.get("predictions") or []
            if len(predictions) != len(sequences or []):
                raise InferenceUnsupportedError(
                    "ensemble member returned a prediction count mismatch",
                    code="ensemble_member_prediction_mismatch",
                    stage="bundle",
                )
            member_rows = [list(prediction.get("scores") or []) for prediction in predictions]
            if labels is None:
                labels = list(response.get("labels") or [])
            if score_rows is None:
                score_rows = [
                    [float(weight) * float(value) for value in row] for row in member_rows
                ]
            else:
                if any(len(row) != len(existing) for row, existing in zip(member_rows, score_rows)):
                    raise InferenceUnsupportedError(
                        "ensemble member score widths differ",
                        code="ensemble_member_score_mismatch",
                        stage="bundle",
                    )
                for index, row in enumerate(member_rows):
                    for column, value in enumerate(row):
                        score_rows[index][column] += float(weight) * float(value)
        if score_rows is None:
            raise InferenceUnsupportedError(
                "ensemble produced no predictions",
                code="ensemble_empty_prediction",
                stage="bundle",
            )
        metadata_labels = metadata.get("label_names") or []
        if metadata_labels:
            labels = [str(value) for value in metadata_labels]
        protocol_version = int(metadata.get("evaluation_protocol_version", 1) or 1)
        thresholds = self._sequence_thresholds(
            metadata,
            len(labels) or (len(score_rows[0]) if score_rows else 0),
        )
        return {
            "model_id": model_id,
            "labels": labels,
            "thresholds": thresholds,
            "score_kind": "probability",
            "evaluation_protocol_version": protocol_version,
            "legacy_evaluation_protocol": protocol_version < 2,
            "predictions": [
                {
                    "sequence": sequence,
                    "scores": [float(value) for value in row],
                    "predicted_labels": [
                        labels[index] if index < len(labels) else f"label_{index + 1}"
                        for index, value in enumerate(row)
                        if float(value) >= float(thresholds[index])
                    ],
                }
                for sequence, row in zip(sequences, score_rows)
            ],
        }

    def readiness(self, model_id):
        model_record = self.store.get_model(model_id)
        if not model_record:
            raise KeyError(model_id)
        bundle = Path(model_record["bundle_path"])
        metadata_path = bundle / "metadata.json"
        if not metadata_path.exists():
            return {
                "model_id": model_id,
                "ready": False,
                "profile_required": False,
                "status": "blocked",
                "mode": "unavailable",
                "checks": [
                    {
                        "id": "metadata",
                        "ok": False,
                        "status": "error",
                        "code": "bundle_metadata_missing",
                        "message": "The model bundle is missing metadata.json",
                    }
                ],
                "errors": [
                    {
                        "code": "bundle_metadata_missing",
                        "message": "The model bundle is missing metadata.json",
                        "hint": "Re-run or re-export the experiment to create a complete model bundle.",
                    }
                ],
                "warnings": [],
            }
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "model_id": model_id,
                "ready": False,
                "profile_required": False,
                "status": "blocked",
                "mode": "unavailable",
                "checks": [],
                "errors": [
                    {
                        "code": "bundle_metadata_invalid",
                        "message": f"The model metadata could not be read: {exc}",
                        "hint": "Re-run or re-export the experiment to replace the damaged metadata.",
                    }
                ],
                "warnings": [],
            }

        runtime_status = self.esm3_runtime_status()
        uses_esm3 = self._metadata_uses_esm3(metadata)
        if uses_esm3 and runtime_status.get("cache_status") == "suspended":
            return {
                "model_id": model_id,
                "ready": False,
                "profile_required": False,
                "status": "blocked",
                "mode": "gpu_busy_training",
                "runtime": runtime_status,
                "checks": [],
                "errors": [
                    {
                        "code": "gpu_busy_training",
                        "stage": "resource_wait",
                        "message": "ESM3 Web inference is paused while a training job uses the GPU.",
                        "hint": "Wait for the training job to finish, then retry inference.",
                    }
                ],
                "warnings": [],
            }

        try:
            self._reject_raw_index_vector_bundle(metadata)
        except InferenceUnsupportedError as exc:
            detail = exc.to_detail()
            return {
                "model_id": model_id,
                "ready": False,
                "profile_required": False,
                "status": "blocked",
                "mode": "unavailable",
                "checks": [],
                "errors": [detail],
                "warnings": [],
            }

        if not model_record["supports_inference"]:
            message = (
                metadata.get("reason")
                or "This model route does not emit a reusable inference bundle."
            )
            return {
                "model_id": model_id,
                "ready": False,
                "profile_required": self._classic_uses_profile(metadata),
                "status": "blocked",
                "mode": "unavailable",
                "checks": [],
                "errors": [
                    {
                        "code": metadata.get("reason_code") or "inference_unsupported",
                        "message": message,
                        "hint": "Choose a model marked as reusable for inference.",
                    }
                ],
                "warnings": [],
            }

        if not self._classic_uses_profile(metadata):
            return {
                "model_id": model_id,
                "ready": True,
                "profile_required": False,
                "status": "not_required",
                "mode": "standard",
                "runtime": runtime_status,
                "checks": [],
                "errors": [],
                "warnings": [],
            }

        report = self._profile_model_readiness(model_id, bundle, metadata)
        report["runtime"] = runtime_status
        return report

    @staticmethod
    def _classic_index_roles(metadata):
        metadata = normalize_stream_fields(metadata)
        roles = []

        def inspect(role, source):
            if not isinstance(source, dict):
                return
            tokenizer = normalize_tokenizer_name(source.get("tokenizer", ""))
            encoder = source.get("encoder", "")
            if isinstance(encoder, dict):
                encoder = encoder.get("id", "")
            encoder = normalize_encoder_name(encoder)
            if tokenizer not in PRETRAINED_TOKENIZERS and encoder == "index":
                roles.append(role)

        streams = metadata.get("streams")
        if isinstance(streams, dict):
            for role, stream in streams.items():
                inspect(str(role), stream)
        args = metadata.get("args")
        if isinstance(args, dict):
            arg_streams = args.get("streams")
            if isinstance(arg_streams, dict):
                for role, stream in arg_streams.items():
                    inspect(str(role), stream)
        inspect("classic", metadata.get("classic"))
        inspect("sequence", args)
        return sorted(set(roles))

    @classmethod
    def _reject_raw_index_vector_bundle(cls, metadata):
        metadata = normalize_stream_fields(metadata)
        bundle_type = str(metadata.get("bundle_type", "") or "")
        if bundle_type not in _RAW_INDEX_VECTOR_BUNDLE_TYPES:
            return
        roles = cls._classic_index_roles(metadata)
        if not roles:
            return
        role_text = ", ".join(roles)
        raise InferenceUnsupportedError(
            "raw index token IDs cannot be used as flattened vector or graph "
            f"features; affected sources: {role_text}",
            code="classical_index_unsupported",
            stage="bundle",
            hint="Retrain with a learned token embedding or a categorical feature encoder such as onehot.",
        )

    @staticmethod
    def _matching_profile_training_contract(
        metadata,
        profile,
        scientific_parameters,
    ):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.sequence_profiles import (
            BLOSUM62_BACKGROUND_SHA256,
            BLOSUM62_MATRIX_SHA256,
            FEATURE_TARGET_FREQUENCY,
            PROFILE_HIT_SUMMARY_FIELDS,
            PROFILE_QUERY_COVERAGE_FORMULA,
            PROFILE_SCIENTIFIC_REFERENCES,
        )

        evidence = []
        expected_feature = ModelService._profile_feature_kind(metadata)
        fallback_policies = profile.get("fallback_policies") or {}
        expected_fallback_policy = str(
            fallback_policies.get(
                expected_feature,
                profile.get("on_missing", "error"),
            )
            or "error"
        )
        expected_manifest_fallback_policy = (
            TOP_N_GRAM_FALLBACK_POLICY_VERSION
            if expected_feature == FEATURE_TARGET_FREQUENCY
            and expected_fallback_policy == TOP_N_GRAM_FALLBACK_POLICY
            else expected_fallback_policy
        )
        if isinstance(metadata.get("profile_contract"), dict):
            evidence.append(metadata["profile_contract"])
        if profile.get("manifest_sha256"):
            evidence.append(
                {
                    "manifest_sha256": profile.get("manifest_sha256"),
                    "provenance": profile.get("training_profile_provenance"),
                }
            )
        manifests = profile.get("training_manifests") or {}
        if isinstance(manifests, dict):
            evidence.extend(value for value in manifests.values() if isinstance(value, dict))

        def valid_digest(value):
            return (
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            )

        expected_reference_dois = {
            str(reference["doi"]) for reference in PROFILE_SCIENTIFIC_REFERENCES
        }

        def valid_references(value):
            return (
                isinstance(value, list)
                and {
                    str(reference.get("doi") or "")
                    for reference in value
                    if isinstance(reference, dict)
                }
                == expected_reference_dois
            )

        for contract in evidence:
            provenance = contract.get("provenance")
            if (
                valid_digest(contract.get("manifest_sha256"))
                and isinstance(provenance, dict)
                and provenance.get("profile_contract_version") == PROFILE_CONTRACT_VERSION
                and provenance.get("algorithm") == "NCBI PSI-BLAST"
                and provenance.get("implementation") == "NCBI BLAST+"
                and provenance.get("feature_type") == expected_feature
                and provenance.get("fallback_policy") == expected_manifest_fallback_policy
                and isinstance(provenance.get("blast_version"), str)
                and bool(provenance.get("blast_version"))
                and isinstance(provenance.get("makeblastdb_version"), str)
                and bool(provenance.get("makeblastdb_version"))
                and provenance.get("scientific_parameters") == scientific_parameters
                and isinstance(provenance.get("runtime_parameters"), dict)
                and isinstance(provenance.get("database_scope"), str)
                and bool(provenance.get("database_scope"))
                and provenance.get("database_scope") == profile.get("db_scope")
                and valid_digest(provenance.get("subject_fasta_sha256"))
                and valid_digest(provenance.get("generation_fingerprint"))
                and provenance.get("blosum62_matrix_sha256") == BLOSUM62_MATRIX_SHA256
                and provenance.get("blosum62_background_sha256") == BLOSUM62_BACKGROUND_SHA256
                and valid_references(provenance.get("scientific_references"))
                and provenance.get("quality_summary")
                == {
                    "outfmt_fields": list(PROFILE_HIT_SUMMARY_FIELDS),
                    "query_coverage": PROFILE_QUERY_COVERAGE_FORMULA,
                    "iteration_scope": "final_completed_iteration",
                    "hit_count_unit": "unique_subject",
                }
                and provenance.get("self_exclusion")
                == "all subject IDs with an identical normalized query sequence"
            ):
                return contract
        return None

    @staticmethod
    def _profile_model_readiness(model_id, bundle, metadata):
        metadata = normalize_stream_fields(metadata)
        profile = metadata.get("profile_generation") or {}
        checks = []
        errors = []
        warnings = []

        def add_check(check_id, ok, message, code="", hint="", status=None, **extra):
            check = {
                "id": check_id,
                "ok": bool(ok),
                "status": status or ("ok" if ok else "error"),
                "message": message,
                **extra,
            }
            if code:
                check["code"] = code
            if hint:
                check["hint"] = hint
            checks.append(check)
            if check["status"] == "error":
                errors.append({"code": code, "message": message, "hint": hint})
            elif check["status"] == "warning":
                warnings.append({"code": code, "message": message, "hint": hint})

        if not profile:
            add_check(
                "profile_metadata",
                False,
                "The model bundle is missing profile generation metadata",
                "profile_metadata_missing",
                "Re-run or re-export the experiment with Profile artifact generation enabled.",
            )
            return {
                "model_id": model_id,
                "ready": False,
                "profile_required": True,
                "status": "blocked",
                "mode": "unavailable",
                "checks": checks,
                "errors": errors,
                "warnings": warnings,
            }

        add_check(
            "profile_metadata",
            True,
            "Profile generation metadata is present",
            feature=ModelService._profile_feature_kind(metadata),
        )
        contract_current = profile.get("profile_contract_version") == PROFILE_CONTRACT_VERSION
        add_check(
            "profile_contract",
            contract_current,
            (
                f"Profile scientific contract v{PROFILE_CONTRACT_VERSION} is present"
                if contract_current
                else "Profile bundle uses a legacy or incomplete contract"
            ),
            "profile_contract_legacy" if not contract_current else "",
            (
                f"Retrain and export with the v{PROFILE_CONTRACT_VERSION} profile contract."
                if not contract_current
                else ""
            ),
        )
        try:
            from pepplatform.profile_generation import ProfileGenerationConfig

            readiness_config = ProfileGenerationConfig(
                iterations=int(profile.get("iterations", 3)),
                evalue=float(profile.get("evalue", 0.001)),
                inclusion_evalue=float(profile.get("inclusion_evalue", 0.001)),
                threads=int(profile.get("threads", 1)),
            )
            scientific_parameters = readiness_config.scientific_parameters()
            parameters_valid = profile.get("scientific_parameters") == scientific_parameters
        except (TypeError, ValueError):
            scientific_parameters = None
            parameters_valid = False
        add_check(
            "profile_scientific_parameters",
            parameters_valid,
            (
                "Profile scientific parameters are complete and pinned"
                if parameters_valid
                else "Profile scientific parameters are missing or inconsistent"
            ),
            "profile_provenance_mismatch" if not parameters_valid else "",
            f"Retrain and export with the v{PROFILE_CONTRACT_VERSION} profile contract.",
        )
        training_contract = (
            ModelService._matching_profile_training_contract(
                metadata,
                profile,
                scientific_parameters,
            )
            if scientific_parameters is not None
            else None
        )
        add_check(
            "profile_training_manifest",
            training_contract is not None,
            (
                "Validated training profile provenance is present"
                if training_contract is not None
                else "Validated training profile provenance is missing"
            ),
            ("profile_training_manifest_missing" if training_contract is None else ""),
            f"Retrain and export with a validated v{PROFILE_CONTRACT_VERSION} profile manifest.",
        )
        feature = ModelService._profile_feature_kind(metadata)
        fallback_policies = profile.get("fallback_policies") or {}
        fallback_mode = str(
            fallback_policies.get(
                feature,
                profile.get("on_missing", "error"),
            )
            or "error"
        )
        fallback_valid = (
            fallback_mode == "blosum62_hybrid"
            if feature in (PSSM_HYBRID_FEATURE_TYPES | PSFM_HYBRID_FEATURE_TYPES)
            else fallback_mode
            in (
                {"error", TOP_N_GRAM_FALLBACK_POLICY}
                if feature == "target_frequency"
                else {"error"}
            )
        )
        add_check(
            "profile_fallback_policy",
            fallback_valid,
            (
                f"Profile fallback policy is explicit ({fallback_mode})"
                if fallback_valid
                else "Profile fallback policy is incompatible with its feature"
            ),
            "profile_fallback_policy_mismatch" if not fallback_valid else "",
            "Retrain using strict PSSM/PSFM or the named hybrid encoder.",
        )
        subject_fasta = ModelService._resolve_bundle_path(bundle, profile.get("subject_fasta", ""))
        subject_ready = bool(
            subject_fasta and subject_fasta.is_file() and subject_fasta.stat().st_size > 0
        )
        add_check(
            "subject_fasta",
            subject_ready,
            "Profile subject FASTA is available"
            if subject_ready
            else "Profile subject FASTA is missing or empty",
            "profile_subject_missing" if not subject_ready else "",
            "Re-run or re-export the experiment so profile_generation/subject.fasta is included.",
        )
        subject_digest_valid = False
        if subject_ready and training_contract is not None:
            from pepplatform.sequence_profiles import sha256_file

            subject_digest = sha256_file(subject_fasta)
            subject_digest_valid = (
                profile.get("subject_fasta_sha256") == subject_digest
                and (training_contract.get("provenance") or {}).get("subject_fasta_sha256")
                == subject_digest
            )
        add_check(
            "profile_subject_digest",
            subject_digest_valid,
            (
                "Bundled profile database matches the training manifest"
                if subject_digest_valid
                else "Bundled profile database differs from the training manifest"
            ),
            ("profile_subject_digest_mismatch" if not subject_digest_valid else ""),
            "Re-export the model from the validated training profile cache.",
        )

        profile_dir = bundle / "profile_generation"
        cache_writable = profile_dir.is_dir() and os.access(profile_dir, os.W_OK)
        add_check(
            "profile_cache",
            cache_writable,
            "Profile cache directory is writable"
            if cache_writable
            else "Profile cache directory is not writable",
            "profile_cache_unwritable" if not cache_writable else "",
            "Grant the backend process write access to the model's profile_generation directory.",
        )

        configured_db = ModelService._resolve_bundle_path(bundle, profile.get("db_prefix", ""))
        default_db = profile_dir / "blastdb" / "ppi_profile_db"
        db_prefix = (
            configured_db
            if configured_db and ModelService._blast_db_ready(str(configured_db))
            else default_db
        )
        prebuilt = ModelService._blast_db_ready(str(db_prefix))
        mode = "prebuilt_db" if prebuilt else "build_on_demand"
        add_check(
            "blast_database",
            prebuilt,
            "A prebuilt BLAST database is available"
            if prebuilt
            else "BLAST database will be built on first inference",
            "blast_database_build_required" if not prebuilt else "",
            status="ok" if prebuilt else "warning",
            mode=mode,
        )

        environment = profile_environment_readiness()
        environment_checks = {check["id"]: check for check in environment["checks"]}
        psiblast = environment_checks.get("psiblast") or {}
        add_check(
            "psiblast",
            psiblast.get("ok", False),
            psiblast.get("message", "psiblast readiness is unknown"),
            psiblast.get("code", ""),
            psiblast.get("hint", ""),
            version=psiblast.get("version", ""),
        )
        makeblastdb = environment_checks.get("makeblastdb") or {}
        makeblastdb_required = not prebuilt
        makeblastdb_ok = makeblastdb.get("ok", False)
        add_check(
            "makeblastdb",
            makeblastdb_ok or not makeblastdb_required,
            makeblastdb.get("message", "makeblastdb readiness is unknown"),
            makeblastdb.get("code", "") if makeblastdb_required else "makeblastdb_optional",
            makeblastdb.get("hint", ""),
            status=("ok" if makeblastdb_ok else ("error" if makeblastdb_required else "warning")),
            required=makeblastdb_required,
            version=makeblastdb.get("version", ""),
        )
        smoke = environment_checks.get("profile_smoke_test") or {}
        if makeblastdb_ok and psiblast.get("ok", False):
            add_check(
                "profile_smoke_test",
                smoke.get("ok", False),
                smoke.get("message", "BLAST+ profile smoke test status is unknown"),
                smoke.get("code", ""),
                smoke.get("hint", ""),
            )

        try:
            iterations = int(profile.get("iterations", 3) or 3)
        except (TypeError, ValueError):
            iterations = 0
        if iterations < 2:
            add_check(
                "profile_iterations",
                True,
                "This legacy bundle uses fewer than two PSI-BLAST iterations and may always fall back",
                "profile_iterations_legacy",
                "Create a new experiment with profile_iterations set to at least 2.",
                status="warning",
                value=iterations,
            )

        ready = not errors
        return {
            "model_id": model_id,
            "ready": ready,
            "profile_required": True,
            "status": "blocked" if not ready else ("warning" if warnings else "ready"),
            "mode": mode,
            "feature": feature,
            "fallback_mode": fallback_mode,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "environment_checked_at": environment.get("checked_at"),
        }

    @staticmethod
    def _require_sequences(sequences):
        rows = [str(sequence).strip() for sequence in (sequences or []) if str(sequence).strip()]
        if not rows:
            raise ValueError("at least one sequence is required")
        return rows

    @staticmethod
    def _require_pairs(pairs):
        normalized = []
        for pair in pairs or []:
            if not isinstance(pair, dict):
                raise ValueError("each pair must include peptide and protein")
            peptide = str(pair.get("peptide", "") or "").strip()
            protein = str(pair.get("protein", "") or "").strip()
            if not peptide or not protein:
                raise ValueError("each pair must include peptide and protein")
            normalized.append({"peptide": peptide, "protein": protein})
        if not normalized:
            raise ValueError("at least one peptide-protein pair is required")
        return normalized

    @staticmethod
    def _profile_inference_unavailable_reason(metadata):
        metadata = normalize_stream_fields(metadata)
        bundle_type = str(metadata.get("bundle_type", "") or "")
        if bundle_type not in {"sequence_nn", "sequence_classical", "sequence_embedding_classical"}:
            return ""
        if not uses_profile_features(metadata.get("args") or {}):
            return ""
        if (metadata.get("profile_generation") or {}).get("subject_fasta"):
            return ""
        return metadata.get("inference_unavailable_reason") or PROFILE_DEPENDENT_INFERENCE_REASON

    @staticmethod
    def _interaction_threshold(metadata):
        metadata = normalize_stream_fields(metadata)
        try:
            threshold = float(metadata.get("threshold", 0.5))
        except (TypeError, ValueError):
            return 0.5
        if not np.isfinite(threshold):
            return 0.5
        return threshold

    @staticmethod
    def _sequence_thresholds(metadata, width):
        metadata = normalize_stream_fields(metadata)
        raw = metadata.get("thresholds")
        if raw is None:
            raw = (metadata.get("operating_point") or {}).get("thresholds")
        if raw is None and int(width) == 1:
            scalar = metadata.get("threshold")
            if scalar is None:
                scalar = (metadata.get("operating_point") or {}).get("threshold")
            if scalar is not None:
                raw = [scalar]
        try:
            values = [float(value) for value in (raw or [])]
        except (TypeError, ValueError):
            values = []
        if len(values) != int(width):
            values = [0.5] * int(width)
        return [value if np.isfinite(value) else 0.5 for value in values]

    @staticmethod
    def _pair_prediction_record(pair, row, threshold, score_kind="probability", score_range=None):
        row_scores = [float(value) for value in row]
        score = row_scores[0] if row_scores else 0.0
        is_interaction = bool(score >= threshold)
        record = {
            "peptide": pair["peptide"],
            "protein": pair["protein"],
            "score": score,
            "score_kind": score_kind,
            "threshold": float(threshold),
            "is_interaction": is_interaction,
            "prediction_label": "Interact" if is_interaction else "No interaction",
            "scores": row_scores,
        }
        if score_range is not None:
            record["score_range"] = [float(score_range[0]), float(score_range[1])]
        return record

    @staticmethod
    def _inference_batch_size(metadata, default=64):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        values = [
            os.environ.get("PEPPLATFORM_INFERENCE_BATCH_SIZE"),
            args.get("inference_batch_size"),
            args.get("predict_batch_size"),
            args.get("batch_size"),
        ]
        for value in values:
            try:
                batch_size = int(value)
            except (TypeError, ValueError):
                continue
            if batch_size > 0:
                return batch_size
        return int(default)

    @staticmethod
    def _load_torch_model(path, device):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except AttributeError as exc:
            error_text = str(exc)
            from scripts.train_pair_pipeline import (
                LegacyAlignedGatedClassifier,
                PairBiCrossAttentionClassifier,
            )

            compatibility_classes = {
                "AlignedGatedClassifier": LegacyAlignedGatedClassifier,
                "GatedTrainableClassifier": LegacyAlignedGatedClassifier,
                "PairBiCrossAttentionClassifier": PairBiCrossAttentionClassifier,
            }
            class_name = next(
                (name for name in compatibility_classes if name in error_text),
                "",
            )
            if not class_name:
                raise

            main_module = sys.modules.get("__main__")
            if main_module is None:
                raise
            missing = object()
            previous = getattr(main_module, class_name, missing)
            setattr(main_module, class_name, compatibility_classes[class_name])
            try:
                return torch.load(path, map_location=device, weights_only=False)
            finally:
                if previous is missing:
                    delattr(main_module, class_name)
                else:
                    setattr(main_module, class_name, previous)

    @staticmethod
    def _validate_aligned_gated_model(metadata, model):
        metadata = normalize_stream_fields(metadata)
        contract = metadata.get("aligned_gated_contract")
        if isinstance(model, AlignedGatedClassifier) and hasattr(model, "fusion"):
            if contract is None:
                warnings.warn(
                    "unversioned aligned_gated vector bundle is legacy_only; "
                    "loading its historical model structure",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            try:
                validate_aligned_gated_contract(contract, model.fusion)
            except PairFusionError as exc:
                raise InferenceUnsupportedError(str(exc)) from exc
            return
        warnings.warn(
            "historical trainable-gated vector bundle is legacy_only; "
            "loading its compatibility model structure",
            RuntimeWarning,
            stacklevel=2,
        )

    @staticmethod
    def _prediction_head_module(model):
        if isinstance(model, (torch.nn.Linear, torch.nn.Sequential)):
            return model
        for name in ("head", "net", "fc", "classifier"):
            candidate = getattr(model, name, None)
            if isinstance(candidate, (torch.nn.Linear, torch.nn.Sequential)):
                return candidate
        return None

    @staticmethod
    def _prediction_head_structure(model):
        module = ModelService._prediction_head_module(model)
        if isinstance(module, torch.nn.Linear):
            return {"kind": "linear", "hidden_dims": []}
        if not isinstance(module, torch.nn.Sequential):
            return None
        layers = list(module.children())
        if len(layers) < 4 or (len(layers) - 1) % 3:
            return None
        activation_names = {
            torch.nn.ReLU: "relu",
            torch.nn.GELU: "gelu",
            torch.nn.SiLU: "silu",
        }
        hidden_dims = []
        activation = None
        dropout = None
        previous_width = None
        for offset in range(0, len(layers) - 1, 3):
            linear, hidden_activation, hidden_dropout = layers[offset : offset + 3]
            if (
                not isinstance(linear, torch.nn.Linear)
                or type(hidden_activation) not in activation_names
                or not isinstance(hidden_dropout, torch.nn.Dropout)
            ):
                return None
            if previous_width is not None and int(linear.in_features) != previous_width:
                return None
            layer_activation = activation_names[type(hidden_activation)]
            layer_dropout = float(hidden_dropout.p)
            if activation is not None and layer_activation != activation:
                return None
            if dropout is not None and layer_dropout != dropout:
                return None
            activation = layer_activation
            dropout = layer_dropout
            previous_width = int(linear.out_features)
            hidden_dims.append(previous_width)
        output = layers[-1]
        if (
            not isinstance(output, torch.nn.Linear)
            or previous_width is None
            or int(output.in_features) != previous_width
        ):
            return None
        return {
            "kind": "mlp",
            "hidden_dims": hidden_dims,
            "activation": activation,
            "dropout": dropout,
        }

    @staticmethod
    def _metadata_claims_standard_prediction_head(metadata):
        metadata = normalize_stream_fields(metadata)
        if str(metadata.get("bundle_type") or "") == "pair_graph_nn":
            return False
        args = metadata.get("args") or {}
        route_head = ((metadata.get("effective_route") or {}).get("predictor") or {}).get(
            "head"
        ) or {}
        candidates = [args.get("classifier"), args.get("model"), route_head.get("id")]
        return any(str(value or "").strip().lower() in {"mlp", "linear"} for value in candidates)

    @staticmethod
    def _validate_prediction_head_contract(metadata, model):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_standard_prediction_head(metadata):
            return
        persisted = metadata.get("prediction_head")
        if persisted is None:
            warnings.warn(
                "neural artifact is missing the current prediction-head contract and is "
                "legacy_only; historical inference is preserved, but retraining is required "
                "for a verified MLP or Linear baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        try:
            contract = validate_prediction_head_artifact_config(persisted)
        except ValueError as exc:
            raise InferenceUnsupportedError(
                f"unsupported prediction-head contract: {exc}",
                code="prediction_head_contract_unsupported",
                stage="bundle",
                hint="Retrain and re-export the model with the current prediction-head contract.",
            ) from exc
        route_head = ((metadata.get("effective_route") or {}).get("predictor") or {}).get(
            "head"
        ) or {}
        route_kind = str(route_head.get("id") or "").strip().lower()
        if route_kind in {"mlp", "linear"} and route_kind != contract["kind"]:
            raise InferenceUnsupportedError(
                "prediction-head contract disagrees with the effective route",
                code="prediction_head_contract_mismatch",
                stage="bundle",
            )
        actual = ModelService._prediction_head_structure(model)
        comparable = {
            key: contract[key]
            for key in ("kind", "hidden_dims", "activation", "dropout")
            if key in contract
        }
        if actual != comparable:
            raise InferenceUnsupportedError(
                "prediction-head contract disagrees with the loaded model structure",
                code="prediction_head_model_mismatch",
                stage="bundle",
                hint="Replace the damaged bundle or retrain and re-export the model.",
                technical_detail=f"metadata={comparable!r}; model={actual!r}",
            )

    @staticmethod
    def _graph_inference_batch_size(metadata, query_count):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        values = [
            os.environ.get("PEPPLATFORM_GRAPH_INFERENCE_BATCH_SIZE"),
            args.get("graph_inference_batch_size"),
        ]
        for value in values:
            try:
                batch_size = int(value)
            except (TypeError, ValueError):
                continue
            if batch_size > 0:
                return min(batch_size, max(1, int(query_count)))
        return max(1, int(query_count))

    @staticmethod
    def _predict_tensor_model_in_batches(model, features, device, metadata):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.models import ResidueModelInput

        if isinstance(features, ResidueModelInput):
            return ModelService._predict_residue_model_in_batches(
                model,
                features,
                device,
                metadata,
            )
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)
        batch_size = ModelService._inference_batch_size(metadata)
        host_features = features.detach().cpu()
        outputs = []
        total = int(host_features.shape[0])
        with torch.no_grad():
            for start in range(0, host_features.shape[0], batch_size):
                batch = host_features[start : start + batch_size].to(device)
                batch_count = int(batch.shape[0])
                logits = model(batch)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                outputs.append(torch.sigmoid(logits).detach().cpu())
                del batch, logits
                _report_inference_progress(
                    "running_predictor",
                    completed=min(start + batch_count, total),
                    total=total,
                )
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
        if not outputs:
            return np.empty((0, 1), dtype=np.float32)
        return torch.cat(outputs, dim=0).numpy()

    @staticmethod
    def _predict_residue_model_in_batches(model, features, device, metadata):
        metadata = normalize_stream_fields(metadata)
        batch_size = ModelService._inference_batch_size(metadata)
        host_features = features.detach().cpu()
        outputs = []
        total = int(host_features.features.shape[0])
        with torch.no_grad():
            for start in range(0, host_features.features.shape[0], batch_size):
                batch = host_features.batch_slice(start, start + batch_size).to(device)
                logits = model(batch)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                outputs.append(torch.sigmoid(logits).detach().cpu())
                del batch, logits
                _report_inference_progress(
                    "running_predictor",
                    completed=min(start + batch_size, total),
                    total=total,
                )
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
        if not outputs:
            return np.empty((0, 1), dtype=np.float32)
        return torch.cat(outputs, dim=0).numpy()

    @staticmethod
    def _predict_pair_tensor_model_in_batches(
        model,
        peptide_features,
        partner_features,
        device,
        metadata,
    ):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.models import ResidueModelInput

        peptide_explicit = isinstance(peptide_features, ResidueModelInput)
        partner_explicit = isinstance(partner_features, ResidueModelInput)
        if peptide_explicit:
            peptide_features = peptide_features.detach().cpu()
        else:
            peptide_features = torch.as_tensor(peptide_features).detach().cpu()
        if partner_explicit:
            partner_features = partner_features.detach().cpu()
        else:
            partner_features = torch.as_tensor(partner_features).detach().cpu()
        if peptide_features.shape[0] != partner_features.shape[0]:
            raise InferenceUnsupportedError("pair streams produced different batch sizes")
        batch_size = ModelService._inference_batch_size(metadata)
        outputs = []
        total = int(peptide_features.shape[0])
        with torch.no_grad():
            for start in range(0, peptide_features.shape[0], batch_size):
                peptide_batch = (
                    peptide_features.batch_slice(start, start + batch_size).to(device)
                    if peptide_explicit
                    else peptide_features[start : start + batch_size].to(device)
                )
                partner_batch = (
                    partner_features.batch_slice(start, start + batch_size).to(device)
                    if partner_explicit
                    else partner_features[start : start + batch_size].to(device)
                )
                logits = model(peptide_batch, partner_batch)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                outputs.append(torch.sigmoid(logits).detach().cpu())
                del peptide_batch, partner_batch, logits
                _report_inference_progress(
                    "running_predictor",
                    completed=min(start + batch_size, total),
                    total=total,
                )
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
        if not outputs:
            return np.empty((0, 1), dtype=np.float32)
        return torch.cat(outputs, dim=0).numpy()

    @staticmethod
    def _extract_pretrained_embeddings(metadata, sequences, device):
        metadata = normalize_stream_fields(metadata)
        pretrained = metadata.get("pretrained") or {}
        tag = normalize_tokenizer_name(pretrained.get("tokenizer", ""))
        model_name = pretrained.get("model_name")
        max_length = int(
            pretrained.get("max_length") or metadata.get("args", {}).get("max_length") or 60
        )
        batch_size = max(1, int(metadata.get("args", {}).get("batch_size") or 1))
        if not tag or not model_name:
            raise InferenceUnsupportedError("pretrained embedding bundle is missing model metadata")
        if tag == "esm3":
            _tokenizer, encoder, _legacy = ModelService._load_esm3_for_inference(
                pretrained, max_length, device
            )
            batches = []
            total = len(sequences)
            batch_total = (total + batch_size - 1) // batch_size if total else 0
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start : start + batch_size]
                batches.append(encoder.encode_sequences(batch, max_length=max_length).float())
                _report_inference_progress(
                    "extracting_features",
                    completed=min(start + len(batch), total),
                    total=total,
                    batch_completed=(start // batch_size) + 1,
                    batch_total=batch_total,
                )
            return torch.cat(batches, dim=0)
        tokenizer_classes = {}
        encoder_classes = {}
        protbert_legacy = False
        prott5_legacy = False
        if tag == "esm2":
            tokenizer, encoder, esm2_legacy = ModelService._load_esm2_for_inference(
                pretrained, max_length, device
            )
        elif tag == "protbert":
            tokenizer, encoder, protbert_legacy = ModelService._load_protbert_for_inference(
                pretrained, max_length, device
            )
        elif tag == "prott5":
            tokenizer, encoder, prott5_legacy = ModelService._load_prott5_for_inference(
                pretrained, max_length, device
            )
        elif tag == "esmc":
            tokenizer, encoder = ModelService._load_esmc_for_inference(
                pretrained, max_length, device
            )
        else:
            raise InferenceUnsupportedError(f"unsupported pretrained tokenizer: {tag}")
        if tag not in {"esm2", "protbert", "prott5", "esmc"}:
            tokenizer = tokenizer_classes[tag](model_name=model_name, max_length=max_length)
            encoder = encoder_classes[tag](model_name=model_name, device=device)
        batches = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            inputs = tokenizer.batch_tokenize(batch, device=device)
            if (
                tag == "esm2"
                and esm2_legacy
                and hasattr(encoder, "encode_legacy_attention_mask_mean")
            ):
                embeddings = encoder.encode_legacy_attention_mask_mean(inputs)
            elif tag == "protbert" and protbert_legacy:
                embeddings = encoder.encode_legacy_attention_mask_mean(inputs)
            elif tag == "prott5" and prott5_legacy:
                embeddings = encoder.encode_legacy_attention_mask_mean(inputs)
            else:
                embeddings = encoder.encode(inputs)
            batches.append(embeddings.detach().cpu().float())
        return torch.cat(batches, dim=0)

    @staticmethod
    def _extract_pretrained_residue_features(metadata, sequences, device):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.models import ResidueModelInput

        pretrained = metadata.get("pretrained") or {}
        tag = normalize_tokenizer_name(pretrained.get("tokenizer", ""))
        model_name = pretrained.get("model_name")
        max_length = int(
            pretrained.get("max_length") or metadata.get("args", {}).get("max_length") or 60
        )
        batch_size = ModelService._inference_batch_size(metadata, default=1)
        if not tag or not model_name:
            raise InferenceUnsupportedError("pretrained residue bundle is missing model metadata")
        prott5_legacy = False

        if tag == "esm3":
            tokenizer, encoder, _legacy = ModelService._load_esm3_for_inference(
                pretrained, max_length, device
            )
        elif tag == "esm2":
            tokenizer, encoder, _legacy = ModelService._load_esm2_for_inference(
                pretrained, max_length, device
            )
        elif tag == "protbert":
            tokenizer, encoder, protbert_legacy = ModelService._load_protbert_for_inference(
                pretrained, max_length, device
            )
        elif tag == "prott5":
            tokenizer, encoder, prott5_legacy = ModelService._load_prott5_for_inference(
                pretrained, max_length, device
            )
        elif tag == "esmc":
            tokenizer, encoder = ModelService._load_esmc_for_inference(
                pretrained, max_length, device
            )
        else:
            raise InferenceUnsupportedError(f"unsupported pretrained tokenizer: {tag}")

        feature_batches = []
        mask_batches = []
        total = len(sequences)
        batch_total = (total + batch_size - 1) // batch_size if total else 0
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            if tag == "esm3":
                encoded = encoder.encode_residue_batch(batch_sequences, max_length)
            else:
                if tag == "prott5" and prott5_legacy:
                    raise InferenceUnsupportedError(
                        "legacy ProtT5 residue artifacts cannot prove residue alignment; retrain required",
                        code="prott5_legacy_residue_unavailable",
                        stage="feature_extraction",
                    )
                tokenized = tokenizer.batch_tokenize_residues(
                    batch_sequences,
                    max_length,
                    device=device,
                )
                encoded = encoder.encode_residue_batch(
                    tokenized,
                    [
                        len(tokenizer.normalize_sequence(sequence)[:max_length])
                        if tag == "esm2"
                        else (
                            tokenizer.residue_length(sequence, max_length)
                            if tag in {"protbert", "prott5", "esmc"}
                            else min(len(sequence), max_length)
                        )
                        for sequence in batch_sequences
                    ],
                    max_length,
                )
            feature_batches.append(encoded.features.detach().cpu().float())
            mask_batches.append(encoded.mask.detach().cpu().bool())
            _report_inference_progress(
                "extracting_features",
                completed=min(start + len(batch_sequences), total),
                total=total,
                batch_completed=(start // batch_size) + 1,
                batch_total=batch_total,
            )
        return ResidueModelInput(
            torch.cat(feature_batches, dim=0),
            torch.cat(mask_batches, dim=0),
        )

    @staticmethod
    def _extract_pretrained_residue_export_features(
        metadata,
        sequences,
        device,
        bundle,
    ):
        metadata = normalize_stream_fields(metadata)
        export = metadata.get("embedding_export") or {}
        encoder_path = str(export.get("encoder_path", "") or "").strip()
        if not encoder_path:
            raise InferenceUnsupportedError(
                "pretrained residue export bundle is missing encoder_path"
            )
        encoder_file = Path(encoder_path).expanduser()
        if not encoder_file.is_absolute():
            encoder_file = Path(bundle) / encoder_file
        if not encoder_file.exists():
            raise InferenceUnsupportedError(
                "pretrained residue export bundle is missing stage-1 encoder"
            )
        inputs = ModelService._extract_pretrained_residue_features(
            metadata,
            sequences,
            device,
        )
        inputs = ModelService._transform_pretrained_residue_features(
            bundle,
            metadata,
            inputs,
        )
        encoder = ModelService._load_torch_model(encoder_file, device)
        encoder = encoder.to(device)
        encoder.eval()
        if not hasattr(encoder, "forward_with_embedding"):
            raise InferenceUnsupportedError(
                "pretrained residue stage-1 encoder cannot export embeddings"
            )
        embedding_batches = []
        batch_size = ModelService._inference_batch_size(metadata)
        with torch.no_grad():
            for start in range(0, inputs.features.shape[0], batch_size):
                batch = inputs.batch_slice(start, start + batch_size).to(device)
                _logits, embeddings = encoder.forward_with_embedding(batch)
                embedding_batches.append(embeddings.detach().cpu().float())
        return torch.cat(embedding_batches, dim=0)

    @staticmethod
    def _extract_sequence_embedding_features(metadata, sequences, pipeline, bundle):
        metadata = normalize_stream_fields(metadata)
        export = metadata.get("embedding_export") or {}
        encoder_path = str(export.get("encoder_path", "") or "").strip()
        if not encoder_path:
            raise InferenceUnsupportedError(
                "sequence embedding-export bundle is missing encoder_path"
            )
        encoder_file = Path(encoder_path).expanduser()
        if not encoder_file.is_absolute():
            encoder_file = Path(bundle) / encoder_file
        if not encoder_file.exists():
            raise InferenceUnsupportedError(
                "sequence embedding-export bundle is missing stage-1 encoder"
            )
        inputs = ModelService._extract_sequence_features(metadata, sequences, pipeline, bundle)
        inputs = ModelService._transform_pretrained_tensor_features(bundle, metadata, inputs)
        encoder = torch.load(encoder_file, map_location=pipeline.device, weights_only=False)
        encoder = encoder.to(pipeline.device)
        encoder.eval()
        if not hasattr(encoder, "forward_with_embedding"):
            raise InferenceUnsupportedError(
                "sequence embedding-export stage-1 encoder cannot export embeddings"
            )
        with torch.no_grad():
            _logits, embeddings = encoder.forward_with_embedding(inputs.to(pipeline.device))
        return embeddings.detach().cpu().float()

    @staticmethod
    def _prepare_sequence_profile_metadata(metadata, sequences, bundle):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._classic_uses_profile(metadata):
            return metadata, None
        if bundle is None:
            raise InferenceUnsupportedError(
                "profile-backed sequence inference is missing bundle context"
            )
        ModelService._validate_top_ngram_bundle_contract(metadata)
        ModelService._validate_dt_bundle_contract(metadata)
        sample_ids = [f"inference_{index}" for index in range(len(sequences))]
        sequence_by_key = {
            sample_id: sequence for sample_id, sequence in zip(sample_ids, sequences)
        }
        profile_paths = ModelService._build_inference_profile_cache(
            Path(bundle), metadata, sequence_by_key
        )
        prepared = dict(metadata)
        prepared["classic"] = dict(metadata.get("classic") or {})
        prepared["classic"].update(profile_paths)
        return prepared, sample_ids

    @staticmethod
    def _extract_sequence_features(metadata, sequences, pipeline, bundle=None):
        metadata = normalize_stream_fields(metadata)
        tokenizer = (
            str(ModelService._classic_value(metadata, "tokenizer", "") or "").strip().lower()
        )
        classic_encoder = normalize_encoder_name(
            ModelService._classic_value(metadata, "encoder", "index")
        )
        if tokenizer == "bpe":
            try:
                featurizer, _ = ModelService._build_classic_featurizer(
                    metadata,
                    bundle=bundle,
                )
                return ModelService._stack_sequence_model_inputs(
                    [featurizer.encode(sequence) for sequence in sequences]
                )
            except InferenceUnsupportedError:
                raise
            except BPEContractError as exc:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="input",
                ) from exc
            except Exception as exc:
                raise InferenceUnsupportedError(
                    f"BPE sequence feature reconstruction failed: {exc}",
                    code="bpe_artifact_invalid",
                ) from exc
        if ModelService._classic_uses_profile(metadata):
            from pepplatform.models import ResidueModelInput

            prepared, sample_ids = ModelService._prepare_sequence_profile_metadata(
                metadata, sequences, bundle
            )
            try:
                featurizer, _ = ModelService._build_classic_featurizer(prepared, bundle=bundle)
                rows = []
                for sequence, sample_id in zip(sequences, sample_ids):
                    rows.append(featurizer.encode(sequence, sample_id=sample_id))
                return ModelService._stack_sequence_model_inputs(rows)
            except InferenceUnsupportedError:
                raise
            except Exception as exc:
                raise InferenceUnsupportedError(
                    f"profile-backed sequence feature reconstruction failed: {exc}"
                ) from exc
        if metadata.get("classic") or classic_encoder in {"bow", "tfidf"}:
            try:
                featurizer, _ = ModelService._build_classic_featurizer(
                    metadata,
                    bundle=bundle,
                )
                return ModelService._stack_sequence_model_inputs(
                    [featurizer.encode(sequence) for sequence in sequences]
                )
            except InferenceUnsupportedError:
                raise
            except Exception as exc:
                raise InferenceUnsupportedError(
                    f"classic sequence feature reconstruction failed: {exc}"
                ) from exc
        featurizer = pipeline.build_featurizer()
        return ModelService._stack_sequence_model_inputs(
            [featurizer.encode(sequence) for sequence in sequences]
        )

    @staticmethod
    def _stack_sequence_model_inputs(rows):
        from pepplatform.models import ResidueModelInput

        if not rows:
            return torch.empty(0)
        explicit = [isinstance(row, ResidueModelInput) for row in rows]
        if any(explicit):
            if not all(explicit):
                raise InferenceUnsupportedError(
                    "sequence feature rows disagree on explicit position-mask usage"
                )
            return ResidueModelInput(
                torch.stack([row.features.float() for row in rows]),
                torch.stack([row.mask.bool() for row in rows]),
            )
        normalized = [
            row if isinstance(row, torch.Tensor) else torch.as_tensor(row) for row in rows
        ]
        return torch.stack(normalized)

    @staticmethod
    def _extract_pair_vector_features(metadata, pairs, device, bundle=None):
        metadata = normalize_stream_fields(metadata)
        peptide_embeddings, protein_embeddings = ModelService._extract_pair_embeddings(
            metadata, pairs, device, bundle=bundle
        )
        return ModelService._fuse_pair_artifact_embeddings(
            metadata,
            peptide_embeddings,
            protein_embeddings,
        ).float()

    @staticmethod
    def _stream_legacy_metadata(metadata, role):
        metadata = normalize_stream_fields(metadata)
        stream = dict((metadata.get("streams") or {}).get(role) or {})
        spec = dict(stream.get("spec") or stream)
        params = dict(spec.get("params") or {})
        args = dict(metadata.get("args") or {})
        args.update(params)
        args.update(
            {
                "tokenizer": spec.get("tokenizer", ""),
                "encoder": spec.get("encoder", "index"),
                "max_length": int(spec.get("max_length") or stream.get("max_length") or 60),
            }
        )
        prepared = {
            "args": args,
            "profile_generation": metadata.get("profile_generation"),
        }
        if isinstance(stream.get("profile_contract"), dict):
            prepared["profile_contract"] = dict(stream["profile_contract"])
        if isinstance(stream.get("static_embedding"), dict):
            prepared["static_embedding"] = dict(stream["static_embedding"])
        if spec.get("feature_route") == "pretrained":
            model_identity = spec.get("model_identity")
            pooling = (
                str(model_identity.get("pooling"))
                if isinstance(model_identity, dict)
                else (ESM3_POOLING if spec.get("tokenizer") == "esm3" else "attention_mask_mean")
            )
            prepared["pretrained"] = {
                "tokenizer": spec.get("tokenizer", ""),
                "model_name": spec.get("resolved_model_source") or spec.get("model_source", ""),
                "max_length": int(spec.get("max_length") or stream.get("max_length") or 512),
                "pooling": pooling,
            }
            if isinstance(model_identity, dict):
                prepared["pretrained"]["model_identity"] = dict(model_identity)
        else:
            classic = {
                "tokenizer": spec.get("tokenizer", ""),
                "encoder": spec.get("encoder", "index"),
                "max_length": int(spec.get("max_length") or stream.get("max_length") or 60),
                **params,
            }
            if stream.get("tfidf_idf_path"):
                classic["tfidf_idf_path"] = stream["tfidf_idf_path"]
            elif (metadata.get("classic") or {}).get("tfidf_idf_path"):
                classic["legacy_shared_tfidf_idf_path"] = metadata["classic"]["tfidf_idf_path"]
            if isinstance(stream.get("tfidf_idf"), dict):
                classic["tfidf_idf"] = dict(stream["tfidf_idf"])
            if isinstance(stream.get("frequency_encoder_contract"), dict):
                classic["frequency_encoder_contract"] = dict(stream["frequency_encoder_contract"])
            if isinstance(stream.get("textrank_contract"), dict):
                classic["textrank_contract"] = dict(stream["textrank_contract"])
            if isinstance(stream.get("covariance_encoder_contract"), dict):
                classic["covariance_encoder_contract"] = dict(stream["covariance_encoder_contract"])
            if isinstance(stream.get("pdt_encoder_contract"), dict):
                classic["pdt_encoder_contract"] = dict(stream["pdt_encoder_contract"])
            if isinstance(
                stream.get("pssm_covariance_encoder_contract"),
                dict,
            ):
                classic["pssm_covariance_encoder_contract"] = dict(
                    stream["pssm_covariance_encoder_contract"]
                )
            prepared["classic"] = classic
        return prepared

    @staticmethod
    def _extract_stream_embeddings(metadata, pairs, role, device, bundle=None):
        metadata = normalize_stream_fields(metadata)
        stream_metadata = ModelService._stream_legacy_metadata(metadata, role)
        sequence_key = "peptide" if role == "peptide" else "protein"
        sequences = [pair[sequence_key] for pair in pairs]
        if stream_metadata.get("pretrained"):
            return ModelService._extract_pretrained_embeddings(
                stream_metadata,
                sequences,
                device,
            )
        if stream_metadata.get("classic"):
            sample_ids = None
            prepared = stream_metadata
            if ModelService._classic_uses_profile(stream_metadata):
                prepared, peptide_ids, protein_ids = ModelService._prepare_classic_profile_metadata(
                    stream_metadata,
                    pairs,
                    bundle,
                )
                sample_ids = peptide_ids if role == "peptide" else protein_ids
            return ModelService._extract_classic_pair_embeddings(
                prepared,
                sequences,
                device,
                sample_ids=sample_ids,
                bundle=bundle,
                role="peptide" if role == "peptide" else "protein",
            )
        raise InferenceUnsupportedError(f"{role} stream is missing feature metadata")

    @staticmethod
    def _extract_pair_embeddings(metadata, pairs, device, bundle=None):
        metadata = normalize_stream_fields(metadata)
        artifact_metadata = metadata
        peptides = [pair["peptide"] for pair in pairs]
        proteins = [pair["protein"] for pair in pairs]
        use_stream_metadata = bool(metadata.get("streams")) and (
            str((metadata.get("pair") or {}).get("stream_mode", "") or "").lower() == "heterogeneous"
            or not metadata.get("pretrained")
            and not metadata.get("classic")
            or ModelService._homogeneous_tfidf_uses_stream_metadata(metadata)
        )
        if use_stream_metadata:
            peptide_embeddings = ModelService._extract_stream_embeddings(
                metadata,
                pairs,
                "peptide",
                device,
                bundle=bundle,
            )
            protein_embeddings = ModelService._extract_stream_embeddings(
                metadata,
                pairs,
                "partner",
                device,
                bundle=bundle,
            )
        elif metadata.get("pretrained"):
            peptide_embeddings = ModelService._extract_pretrained_embeddings(
                metadata, peptides, device
            )
            protein_embeddings = ModelService._extract_pretrained_embeddings(
                metadata, proteins, device
            )
        elif metadata.get("classic"):
            metadata, peptide_ids, protein_ids = ModelService._prepare_classic_profile_metadata(
                metadata,
                pairs,
                bundle,
            )
            peptide_embeddings = ModelService._extract_classic_pair_embeddings(
                metadata,
                peptides,
                device,
                sample_ids=peptide_ids,
                bundle=bundle,
                role="peptide",
            )
            protein_embeddings = ModelService._extract_classic_pair_embeddings(
                metadata,
                proteins,
                device,
                sample_ids=protein_ids,
                bundle=bundle,
                role="protein",
            )
        else:
            raise InferenceUnsupportedError("pair-vector bundle is missing feature metadata")
        peptide_embeddings = peptide_embeddings.detach().cpu().float()
        protein_embeddings = protein_embeddings.detach().cpu().float()
        ModelService._validate_pair_strategy_metadata(
            artifact_metadata,
            peptide_embeddings,
            protein_embeddings,
        )
        return peptide_embeddings, protein_embeddings

    @staticmethod
    def _classic_uses_profile(metadata):
        metadata = normalize_stream_fields(metadata)
        tokenizer = (
            str(ModelService._classic_value(metadata, "tokenizer", "") or "").strip().lower()
        )
        encoder = normalize_encoder_name(ModelService._classic_value(metadata, "encoder", "index"))
        if encoder == "ppi_fusion":
            encoder = normalize_encoder_name(
                ModelService._classic_value(metadata, "aux_encoder", "index")
            )
        return tokenizer in PROFILE_TOKENIZERS or encoder in PROFILE_ENCODERS

    @staticmethod
    def _profile_feature_kind(metadata):
        metadata = normalize_stream_fields(metadata)
        tokenizer = (
            str(ModelService._classic_value(metadata, "tokenizer", "") or "").strip().lower()
        )
        encoder = normalize_encoder_name(ModelService._classic_value(metadata, "encoder", "index"))
        if tokenizer in PROFILE_TOKENIZERS:
            return "target_frequency"
        if encoder in PSSM_COVARIANCE_ENCODERS:
            return pssm_covariance_profile_feature_type(encoder)
        if encoder == "pssm_blosum62_hybrid":
            return "pssm_blosum62_hybrid"
        if encoder == "psfm_blosum62_hybrid":
            return "psfm_blosum62_hybrid"
        need_pssm = encoder in {"pssm", "ac_pssm", "cc_pssm", "acc_pssm"}
        need_psfm = encoder == "psfm"
        if need_pssm and need_psfm:
            return "both"
        if need_psfm:
            return "psfm"
        return "pssm"

    @staticmethod
    def _prepare_classic_profile_metadata(metadata, pairs, bundle):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._classic_uses_profile(metadata):
            return metadata, None, None
        if bundle is None:
            raise InferenceUnsupportedError(
                "profile-backed pair-vector inference is missing bundle context"
            )
        ModelService._validate_top_ngram_bundle_contract(metadata)
        ModelService._validate_dt_bundle_contract(metadata)
        peptide_ids = [f"inference_{index}::pep" for index in range(len(pairs))]
        protein_ids = [f"inference_{index}::pro" for index in range(len(pairs))]
        sequences = {}
        for index, pair in enumerate(pairs):
            sequences[peptide_ids[index]] = pair["peptide"]
            sequences[protein_ids[index]] = pair["protein"]
        profile_paths = ModelService._build_inference_profile_cache(
            Path(bundle), metadata, sequences
        )
        prepared = dict(metadata)
        prepared["classic"] = dict(metadata.get("classic") or {})
        prepared["classic"].update(profile_paths)
        return prepared, peptide_ids, protein_ids

    @staticmethod
    def _validate_top_ngram_bundle_contract(metadata):
        metadata = normalize_stream_fields(metadata)
        tokenizer = (
            str(ModelService._classic_value(metadata, "tokenizer", "") or "").strip().lower()
        )
        if tokenizer != "top-n-gram":
            return
        contract = ModelService._classic_value(
            metadata,
            "top_ngram_contract",
            None,
        )
        try:
            top_n = validate_top_ngram_n(
                ModelService._classic_value(
                    metadata,
                    "top_ngram_n",
                    None,
                )
            )
        except TopNGramContractError as exc:
            raise InferenceUnsupportedError(
                "Top-n-gram artifact is missing valid v3 parameters; retrain and re-export it",
                code="top_ngram_contract_unsupported",
                stage="bundle",
                technical_detail=str(exc),
            ) from exc
        expected_dim = top_ngram_feature_dim(top_n)
        manifest_digest = (
            contract.get("profile_manifest_sha256") if isinstance(contract, dict) else None
        )
        fallback_policy = contract.get("fallback_policy") if isinstance(contract, dict) else None
        fallback_profiles_allowed = fallback_policy == TOP_N_GRAM_FALLBACK_POLICY_VERSION
        profile_on_missing = (
            str(
                (metadata.get("profile_generation") or {}).get(
                    "on_missing",
                    TOP_N_GRAM_STRICT_PROFILE_POLICY,
                )
                or TOP_N_GRAM_STRICT_PROFILE_POLICY
            )
            .strip()
            .lower()
        )
        expected_profile_on_missing = (
            TOP_N_GRAM_FALLBACK_POLICY
            if fallback_profiles_allowed
            else TOP_N_GRAM_STRICT_PROFILE_POLICY
        )
        if (
            not isinstance(contract, dict)
            or contract.get("contract_version") != TOP_N_GRAM_CONTRACT_VERSION
            or contract.get("profile_type") != "target_frequency"
            or contract.get("tie_policy") != TOP_N_GRAM_TIE_POLICY
            or contract.get("full_length") is not True
            or fallback_policy
            not in {
                TOP_N_GRAM_STRICT_PROFILE_POLICY,
                TOP_N_GRAM_FALLBACK_POLICY_VERSION,
            }
            or contract.get("fallback_profiles_allowed") is not fallback_profiles_allowed
            or profile_on_missing != expected_profile_on_missing
            or contract.get("feature_dim") != expected_dim
            or not isinstance(manifest_digest, str)
            or len(manifest_digest) != 64
            or any(character not in "0123456789abcdef" for character in manifest_digest)
        ):
            raise InferenceUnsupportedError(
                "Top-n-gram artifact does not implement the platform v3 "
                "target-frequency contract; older or legacy models cannot "
                "be converted and must be retrained",
                code="top_ngram_contract_unsupported",
                stage="bundle",
            )

    @staticmethod
    def _validate_dt_bundle_contract(metadata):
        metadata = normalize_stream_fields(metadata)
        tokenizer = (
            str(ModelService._classic_value(metadata, "tokenizer", "") or "").strip().lower()
        )
        if tokenizer != "dt":
            return
        contract = ModelService._classic_value(metadata, "dt_contract", None)
        try:
            max_distance = validate_dt_max_distance(
                ModelService._classic_value(
                    metadata,
                    "dt_max_distance",
                    None,
                )
            )
        except DTContractError as exc:
            raise InferenceUnsupportedError(
                "DT artifact is missing valid v1 parameters; retrain and re-export it",
                code="dt_contract_unsupported",
                stage="bundle",
                technical_detail=str(exc),
            ) from exc
        expected_contract = dt_contract_metadata(max_distance=max_distance)
        manifest_digest = (
            contract.get("profile_manifest_sha256") if isinstance(contract, dict) else None
        )
        fallback_policy = contract.get("fallback_policy") if isinstance(contract, dict) else None
        fallback_profiles_allowed = fallback_policy == TOP_N_GRAM_FALLBACK_POLICY_VERSION
        profile_on_missing = (
            str(
                (metadata.get("profile_generation") or {}).get(
                    "on_missing",
                    TOP_N_GRAM_STRICT_PROFILE_POLICY,
                )
                or TOP_N_GRAM_STRICT_PROFILE_POLICY
            )
            .strip()
            .lower()
        )
        expected_profile_on_missing = (
            TOP_N_GRAM_FALLBACK_POLICY
            if fallback_profiles_allowed
            else TOP_N_GRAM_STRICT_PROFILE_POLICY
        )
        core_matches = isinstance(contract, dict) and all(
            contract.get(key) == value for key, value in expected_contract.items()
        )
        if (
            not core_matches
            or contract.get("contract_version") != DT_CONTRACT_VERSION
            or fallback_policy
            not in {
                TOP_N_GRAM_STRICT_PROFILE_POLICY,
                TOP_N_GRAM_FALLBACK_POLICY_VERSION,
            }
            or contract.get("fallback_profiles_allowed") is not fallback_profiles_allowed
            or profile_on_missing != expected_profile_on_missing
            or ModelService._classic_value(metadata, "feature_dim", None)
            != expected_contract["feature_dim"]
            or not isinstance(manifest_digest, str)
            or len(manifest_digest) != 64
            or any(character not in "0123456789abcdef" for character in manifest_digest)
        ):
            raise InferenceUnsupportedError(
                "DT artifact does not implement the platform v1 exact "
                "Distance-based Top-n-gram contract; legacy hashed models "
                "cannot be converted and must be retrained",
                code="dt_contract_unsupported",
                stage="bundle",
            )

    @staticmethod
    def _resolve_bundle_path(bundle, raw_path):
        value = str(raw_path or "").strip()
        if not value:
            return None
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return Path(bundle) / path

    @staticmethod
    def _blast_db_ready(db_prefix):
        if not db_prefix:
            return False
        return all(Path(f"{db_prefix}{suffix}").exists() for suffix in [".pin", ".phr", ".psq"])

    @staticmethod
    def _ensure_inference_blast_db(bundle, metadata):
        metadata = normalize_stream_fields(metadata)
        profile = metadata.get("profile_generation") or {}
        db_prefix = ModelService._resolve_bundle_path(bundle, profile.get("db_prefix", ""))
        if db_prefix is not None and ModelService._blast_db_ready(str(db_prefix)):
            return str(db_prefix)
        subject_fasta = ModelService._resolve_bundle_path(bundle, profile.get("subject_fasta", ""))
        if subject_fasta is None or not subject_fasta.exists():
            raise InferenceUnsupportedError(
                "Profile-backed inference bundle is missing the PSI-BLAST subject FASTA.",
                code="profile_subject_missing",
                stage="bundle",
                hint="Re-run or re-export the experiment so profile_generation/subject.fasta is included.",
            )
        if shutil.which("makeblastdb") is None:
            raise InferenceUnsupportedError(
                "Profile-backed inference requires makeblastdb from ncbi-blast+.",
                code="blast_binary_missing",
                stage="database",
                hint="Install ncbi-blast+ in the backend runtime.",
            )
        db_dir = Path(bundle) / "profile_generation" / "blastdb"
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise InferenceUnsupportedError(
                "Profile-backed inference cannot create the BLAST database directory.",
                code="profile_cache_unwritable",
                stage="database",
                hint="Grant the backend process write access to the model bundle.",
                technical_detail=str(exc),
            ) from exc
        db_prefix = db_dir / "ppi_profile_db"
        with profile_resource_lock(db_prefix):
            if not ModelService._blast_db_ready(str(db_prefix)):
                cmd = [
                    "makeblastdb",
                    "-in",
                    str(subject_fasta),
                    "-dbtype",
                    "prot",
                    "-out",
                    str(db_prefix),
                    "-parse_seqids",
                ]
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=ModelService._profile_command_timeout(),
                    )
                except subprocess.TimeoutExpired as exc:
                    raise InferenceUnsupportedError(
                        "Building the Profile BLAST database timed out.",
                        code="blast_db_build_timeout",
                        stage="database",
                        hint="Increase PEP_PROFILE_BLAST_TIMEOUT_SECONDS or reduce the subject FASTA size.",
                    ) from exc
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise InferenceUnsupportedError(
                        "Profile-backed inference failed to build the BLAST database.",
                        code="blast_db_build_failed",
                        stage="database",
                        hint="Verify the subject FASTA and the ncbi-blast+ installation.",
                        technical_detail=ModelService._subprocess_detail(exc),
                    ) from exc
                if not ModelService._blast_db_ready(str(db_prefix)):
                    raise InferenceUnsupportedError(
                        "makeblastdb completed without producing a usable protein database.",
                        code="blast_db_incomplete",
                        stage="database",
                        hint="Check filesystem permissions and the makeblastdb output.",
                    )
        return str(db_prefix)

    @staticmethod
    def _run_inference_psiblast(
        query_fa,
        db_prefix,
        out_ascii_pssm,
        iterations,
        evalue,
        threads,
        *,
        out_checkpoint=None,
        out_hits=None,
        excluded_subjects=None,
    ):
        if shutil.which("psiblast") is None:
            raise InferenceUnsupportedError(
                "Profile-backed inference requires psiblast from ncbi-blast+.",
                code="blast_binary_missing",
                stage="profile_generation",
                hint="Install ncbi-blast+ in the backend runtime.",
            )
        cmd = [
            "psiblast",
            "-query",
            str(query_fa),
            "-db",
            str(db_prefix),
            "-num_iterations",
            str(max(1, int(iterations))),
            "-evalue",
            str(float(evalue)),
            "-num_threads",
            str(max(1, int(threads))),
            "-out_ascii_pssm",
            str(out_ascii_pssm),
        ]
        if out_checkpoint is not None:
            cmd.extend(["-out_pssm", str(out_checkpoint)])
        if out_hits is not None:
            cmd.extend(
                [
                    "-out",
                    str(out_hits),
                    "-outfmt",
                    "6 qseqid sseqid pident length qlen evalue bitscore",
                ]
            )
        else:
            cmd.extend(["-out", os.devnull])
        if excluded_subjects:
            excluded_path = Path(query_fa).with_suffix(".exclude")
            excluded_path.write_text(
                "\n".join(excluded_subjects) + "\n",
                encoding="utf-8",
            )
            cmd.extend(["-negative_seqidlist", str(excluded_path)])
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=ModelService._profile_command_timeout(),
            )
        except subprocess.TimeoutExpired as exc:
            raise InferenceUnsupportedError(
                "PSI-BLAST profile generation timed out.",
                code="psiblast_timeout",
                stage="profile_generation",
                hint="Increase PEP_PROFILE_BLAST_TIMEOUT_SECONDS or submit a smaller batch.",
            ) from exc
        except (OSError, subprocess.CalledProcessError) as exc:
            raise InferenceUnsupportedError(
                "PSI-BLAST profile generation failed.",
                code="psiblast_failed",
                stage="profile_generation",
                hint="Inspect the sequence, subject database, and BLAST+ installation.",
                technical_detail=ModelService._subprocess_detail(exc),
            ) from exc

    @staticmethod
    def _build_inference_profile_cache_v3(bundle, metadata, sequence_by_key):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.profile_generation import (
            ProfileGenerationConfig,
            build_profile_caches,
            index_subject_fasta,
        )
        from pepplatform.sequence_profiles import (
            FEATURE_PSFM,
            FEATURE_PSFM_BLOSUM62_HYBRID,
            FEATURE_PSSM,
            FEATURE_PSSM_BLOSUM62_HYBRID,
            FEATURE_TARGET_FREQUENCY,
            PSSM_HYBRID_FEATURE_TYPES,
            PSFM_HYBRID_FEATURE_TYPES,
            SequenceProfileError,
            read_json_strict,
            sha256_file,
        )

        feature = ModelService._profile_feature_kind(metadata)
        profile = metadata.get("profile_generation") or {}
        if not profile:
            raise InferenceUnsupportedError(
                "profile-backed bundle is missing profile generation metadata",
                code="profile_contract_missing",
                stage="bundle",
            )
        if profile.get("profile_contract_version") != PROFILE_CONTRACT_VERSION:
            raise InferenceUnsupportedError(
                "Legacy profile bundles are not accepted because their "
                "scientific provenance and fallback state are incomplete.",
                code="profile_contract_legacy",
                stage="bundle",
                hint=(
                    f"Retrain and export with the v{PROFILE_CONTRACT_VERSION} "
                    "sequence-profile contract."
                ),
            )
        subject_fasta = ModelService._resolve_bundle_path(
            bundle,
            profile.get("subject_fasta", ""),
        )
        if subject_fasta is None or not subject_fasta.is_file():
            raise InferenceUnsupportedError(
                "Profile-backed inference bundle is missing its subject FASTA.",
                code="profile_subject_missing",
                stage="bundle",
                hint=(
                    f"Re-export the experiment with the v{PROFILE_CONTRACT_VERSION} "
                    "profile provenance."
                ),
            )
        fallback_policies = profile.get("fallback_policies") or {}
        on_missing = str(
            fallback_policies.get(
                feature,
                profile.get("on_missing", "error"),
            )
            or "error"
        )
        if (
            feature in (PSSM_HYBRID_FEATURE_TYPES | PSFM_HYBRID_FEATURE_TYPES)
            and on_missing != "blosum62_hybrid"
        ):
            raise InferenceUnsupportedError(
                "The hybrid profile bundle is missing its explicit BLOSUM62 fallback policy.",
                code="profile_fallback_policy_mismatch",
                stage="bundle",
            )
        if feature in {FEATURE_PSSM, FEATURE_PSFM} and on_missing != "error":
            raise InferenceUnsupportedError(
                f"{feature} bundles must use strict profile generation.",
                code="profile_fallback_forbidden",
                stage="bundle",
            )
        if feature == FEATURE_TARGET_FREQUENCY and on_missing not in {
            TOP_N_GRAM_STRICT_PROFILE_POLICY,
            TOP_N_GRAM_FALLBACK_POLICY,
        }:
            raise InferenceUnsupportedError(
                "Target-frequency bundles have an unsupported fallback policy.",
                code="profile_tokenizer_contract_unsupported",
                stage="bundle",
            )

        try:
            config = ProfileGenerationConfig(
                iterations=int(profile.get("iterations", 3)),
                evalue=float(profile.get("evalue", 0.001)),
                inclusion_evalue=float(profile.get("inclusion_evalue", 0.001)),
                threads=int(profile.get("threads", 1)),
                timeout_seconds=ModelService._profile_command_timeout(),
            )
            if profile.get("scientific_parameters") != config.scientific_parameters():
                raise InferenceUnsupportedError(
                    "Profile bundle scientific parameters are incomplete or "
                    "inconsistent with its executable configuration.",
                    code="profile_provenance_mismatch",
                    stage="bundle",
                    hint=(
                        f"Retrain and export the model with the "
                        f"v{PROFILE_CONTRACT_VERSION} profile contract."
                    ),
                )

            matching_contract = ModelService._matching_profile_training_contract(
                metadata,
                profile,
                config.scientific_parameters(),
            )
            if matching_contract is None:
                raise InferenceUnsupportedError(
                    "Profile bundle does not contain a validated training "
                    "manifest digest with full PSI-BLAST provenance.",
                    code="profile_training_manifest_missing",
                    stage="bundle",
                    hint=(
                        f"Retrain and export the model with the "
                        f"v{PROFILE_CONTRACT_VERSION} profile contract."
                    ),
                )
            subject_sha256 = sha256_file(subject_fasta)
            if (
                profile.get("subject_fasta_sha256") != subject_sha256
                or (matching_contract.get("provenance") or {}).get("subject_fasta_sha256")
                != subject_sha256
            ):
                raise InferenceUnsupportedError(
                    "Bundled profile subject FASTA does not match the validated training manifest.",
                    code="profile_subject_digest_mismatch",
                    stage="bundle",
                    hint=("Re-export the model from the validated training profile cache."),
                )
            normalized_items = [
                (str(sample_id), str(sequence)) for sample_id, sequence in sequence_by_key.items()
            ]
            sequence_subjects = index_subject_fasta(subject_fasta)

            cache_identity = {
                "items": normalized_items,
                "feature": feature,
                "subject_fasta_sha256": subject_sha256,
                "scientific_parameters": config.scientific_parameters(),
                "on_missing": on_missing,
                "contract_version": 3,
            }
            batch_digest = hashlib.sha256(
                json.dumps(
                    cache_identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24]
            cache_root = Path(bundle) / "profile_generation" / "inference_cache" / batch_digest
            feature_directory = cache_root / feature
            indexes = build_profile_caches(
                samples=normalized_items,
                feature_directories={feature: feature_directory},
                subject_fasta=subject_fasta,
                sequence_subject_ids=sequence_subjects,
                config=config,
                database_scope="bundled_frozen_subject",
                database_provenance={
                    "bundle_subject_fasta": str(subject_fasta),
                    "bundle_subject_fasta_sha256": subject_sha256,
                    "training_manifest_sha256": str(matching_contract.get("manifest_sha256", "")),
                },
                target_frequency_fallback=(
                    feature == FEATURE_TARGET_FREQUENCY and on_missing == TOP_N_GRAM_FALLBACK_POLICY
                ),
                lock_path=cache_root / ".profile-cache.lock",
            )
            index_path = indexes[feature]
            manifest = read_json_strict(index_path)
        except SequenceProfileError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="profile_generation",
                hint=(
                    "Check the BLAST+ installation, bundled training database, "
                    f"and v{PROFILE_CONTRACT_VERSION} profile metadata."
                ),
            ) from exc
        except (OSError, UnicodeError, ValueError) as exc:
            raise InferenceUnsupportedError(
                "Profile inference cache could not be generated.",
                code="profile_generation_failed",
                stage="profile_generation",
                technical_detail=str(exc),
            ) from exc

        samples = manifest.get("samples") or {}
        fallback_count = sum(
            1
            for value in samples.values()
            if isinstance(value, dict) and value.get("fallback") is True
        )
        zero_hit_count = sum(
            1
            for value in samples.values()
            if isinstance(value, dict)
            and isinstance(value.get("quality"), dict)
            and value["quality"].get("zero_nonself_hits") is True
        )
        diagnostics = _PROFILE_DIAGNOSTICS.get()
        if diagnostics is not None:
            diagnostics.update(
                {
                    "status": ("degraded" if fallback_count or zero_hit_count else "ready"),
                    "feature": feature,
                    "input_count": len(normalized_items),
                    "cache_hits": 0,
                    "generated": len(normalized_items),
                    "fallbacks": fallback_count,
                    "fallback_mode": on_missing,
                    "warning_codes": (["profile_fallback_used"] if fallback_count else [])
                    + (["profile_zero_nonself_hits"] if zero_hit_count else []),
                }
            )
        if feature in HYBRID_PSSM_COVARIANCE_ENCODERS:
            return {f"{feature}_index_path": str(index_path)}
        if feature in {FEATURE_PSSM, FEATURE_PSSM_BLOSUM62_HYBRID}:
            return {"pssm_index_path": str(index_path)}
        if feature in {FEATURE_PSFM, FEATURE_PSFM_BLOSUM62_HYBRID}:
            return {"psfm_index_path": str(index_path)}
        return {"frequency_profile_index_path": str(index_path)}

    @staticmethod
    def _build_inference_profile_cache(bundle, metadata, sequence_by_key):
        metadata = normalize_stream_fields(metadata)
        return ModelService._build_inference_profile_cache_v3(
            bundle,
            metadata,
            sequence_by_key,
        )

    @staticmethod
    def _profile_command_timeout():
        try:
            return max(1, int(os.environ.get("PEP_PROFILE_BLAST_TIMEOUT_SECONDS", "300")))
        except (TypeError, ValueError):
            return 300

    @staticmethod
    def _subprocess_detail(exc):
        parts = []
        for value in [getattr(exc, "stderr", ""), getattr(exc, "stdout", "")]:
            text = str(value or "").strip()
            if text:
                parts.append(text)
        detail = "\n".join(parts) or str(exc)
        return detail[-2000:]

    @staticmethod
    def _classic_value(metadata, key, default=""):
        metadata = normalize_stream_fields(metadata)
        classic = metadata.get("classic") or {}
        args = metadata.get("args") or {}
        value = classic.get(key, args.get(key, default))
        return default if value is None else value

    @staticmethod
    def _classic_int(metadata, key, default):
        metadata = normalize_stream_fields(metadata)
        try:
            return int(ModelService._classic_value(metadata, key, default))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _resolve_metadata_path(raw_path):
        value = str(raw_path or "").strip()
        if not value:
            return ""
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        return str(PROJECT_ROOT / path)

    @staticmethod
    def _resolve_classic_path(metadata, key, default="", bundle=None):
        metadata = normalize_stream_fields(metadata)
        value = str(ModelService._classic_value(metadata, key, default) or "").strip()
        if not value:
            return ""
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        if bundle is not None:
            bundle_path = Path(bundle) / path
            if bundle_path.exists():
                return str(bundle_path)
        return str(PROJECT_ROOT / path)

    @staticmethod
    def _resolve_bpe_bundle_member(bundle, raw_path, *, description):
        if bundle is None:
            raise InferenceUnsupportedError(
                f"{description} requires bundle context",
                code="bpe_artifact_missing",
                stage="bundle",
            )
        value = str(raw_path or "").strip()
        path = Path(value)
        if not value or path.is_absolute():
            raise InferenceUnsupportedError(
                f"{description} must be a non-empty bundle-relative path",
                code="bpe_artifact_invalid",
                stage="bundle",
            )
        bundle_root = Path(bundle).resolve()
        candidate = (bundle_root / path).resolve()
        try:
            candidate.relative_to(bundle_root)
        except ValueError as exc:
            raise InferenceUnsupportedError(
                f"{description} escapes the model bundle: {value}",
                code="bpe_artifact_invalid",
                stage="bundle",
            ) from exc
        return candidate

    @staticmethod
    def _resolve_bpe_path(metadata, bundle=None):
        metadata = normalize_stream_fields(metadata)
        descriptor = ModelService._classic_value(
            metadata,
            "bpe_artifact",
            None,
        )
        if descriptor is not None:
            if not isinstance(descriptor, dict):
                raise InferenceUnsupportedError(
                    "BPE artifact descriptor must be an object",
                    code="bpe_artifact_invalid",
                    stage="bundle",
                )
            if (
                descriptor.get("contract_version") != BPE_CONTRACT_VERSION
                or descriptor.get("model_type") != "BPE"
                or descriptor.get("variant") != BPE_VARIANT
                or descriptor.get("tokenizers_version") != BPE_TOKENIZERS_VERSION
            ):
                raise InferenceUnsupportedError(
                    "BPE artifact descriptor is incompatible with the platform contract",
                    code="bpe_artifact_invalid",
                    stage="bundle",
                )
            tokenizer_path = ModelService._resolve_bpe_bundle_member(
                bundle,
                descriptor.get("path"),
                description="BPE tokenizer artifact",
            )
            provenance_path = ModelService._resolve_bpe_bundle_member(
                bundle,
                descriptor.get("provenance_path"),
                description="BPE provenance",
            )
            try:
                inspection = inspect_bpe_artifact(
                    tokenizer_path,
                    expected_sha256=descriptor.get("sha256"),
                )
                provenance = load_bpe_provenance(
                    tokenizer_path,
                    provenance_path=provenance_path,
                )
            except BPEContractError as exc:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="bundle",
                ) from exc
            if (
                provenance.get("contract_version") != BPE_CONTRACT_VERSION
                or provenance.get("variant") != BPE_VARIANT
                or provenance.get("feature_fit_scope") != "train_only"
                or provenance.get("artifact_sha256") != inspection["sha256"]
                or (provenance.get("runtime") or {}).get("tokenizers_version")
                != BPE_TOKENIZERS_VERSION
                or descriptor.get("vocab_size") != inspection["vocab_size"]
            ):
                raise InferenceUnsupportedError(
                    "BPE provenance or descriptor does not match the bundled tokenizer",
                    code="bpe_artifact_invalid",
                    stage="bundle",
                )
            return str(tokenizer_path)

        raise InferenceUnsupportedError(
            "BPE bundle is missing the v2 embedded artifact descriptor; "
            "rebuild and re-export the model",
            code="bpe_artifact_missing",
            stage="bundle",
        )

    @staticmethod
    def _resolve_wordpiece_bundle_member(
        bundle,
        raw_path,
        *,
        description,
    ):
        if bundle is None:
            raise InferenceUnsupportedError(
                f"{description} requires bundle context",
                code="wordpiece_artifact_missing",
                stage="bundle",
            )
        value = str(raw_path or "").strip()
        path = Path(value)
        if not value or path.is_absolute():
            raise InferenceUnsupportedError(
                f"{description} must be a non-empty bundle-relative path",
                code="wordpiece_artifact_invalid",
                stage="bundle",
            )
        bundle_root = Path(bundle).resolve()
        candidate = (bundle_root / path).resolve()
        try:
            candidate.relative_to(bundle_root)
        except ValueError as exc:
            raise InferenceUnsupportedError(
                f"{description} escapes the model bundle: {value}",
                code="wordpiece_artifact_invalid",
                stage="bundle",
            ) from exc
        return candidate

    @staticmethod
    def _resolve_wordpiece_path(metadata, bundle=None):
        metadata = normalize_stream_fields(metadata)
        descriptor = ModelService._classic_value(
            metadata,
            "wordpiece_artifact",
            None,
        )
        if descriptor is not None:
            if not isinstance(descriptor, dict):
                raise InferenceUnsupportedError(
                    "WordPiece artifact descriptor must be an object",
                    code="wordpiece_artifact_invalid",
                    stage="bundle",
                )
            if (
                descriptor.get("contract_version") != WORDPIECE_CONTRACT_VERSION
                or descriptor.get("model_type") != "WordPiece"
                or descriptor.get("variant") != WORDPIECE_VARIANT
                or descriptor.get("tokenizers_version") != WORDPIECE_TOKENIZERS_VERSION
            ):
                raise InferenceUnsupportedError(
                    "WordPiece artifact descriptor is incompatible with the platform contract",
                    code="wordpiece_artifact_invalid",
                    stage="bundle",
                )
            tokenizer_path = ModelService._resolve_wordpiece_bundle_member(
                bundle,
                descriptor.get("path"),
                description="WordPiece tokenizer artifact",
            )
            provenance_path = ModelService._resolve_wordpiece_bundle_member(
                bundle,
                descriptor.get("provenance_path"),
                description="WordPiece provenance",
            )
            try:
                inspection = inspect_wordpiece_artifact(
                    tokenizer_path,
                    expected_sha256=descriptor.get("sha256"),
                )
                provenance = load_wordpiece_provenance(
                    tokenizer_path,
                    provenance_path=provenance_path,
                )
                validate_wordpiece_provenance(
                    provenance,
                    inspection,
                    context=f"WordPiece provenance {provenance_path}",
                )
            except WordPieceContractError as exc:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="bundle",
                ) from exc
            if (
                descriptor.get("vocab_size") != inspection["vocab_size"]
                or descriptor.get("max_raw_length") != inspection["max_raw_length"]
            ):
                raise InferenceUnsupportedError(
                    "WordPiece provenance or descriptor does not match the bundled tokenizer",
                    code="wordpiece_artifact_invalid",
                    stage="bundle",
                )
            return str(tokenizer_path)

        raise InferenceUnsupportedError(
            "WordPiece bundle is missing the v2 embedded artifact descriptor; "
            "rebuild and re-export the model",
            code="wordpiece_artifact_missing",
            stage="bundle",
        )

    @staticmethod
    def _resolve_unigram_bundle_member(
        bundle,
        raw_path,
        *,
        description,
    ):
        if bundle is None:
            raise InferenceUnsupportedError(
                f"{description} requires bundle context",
                code="unigram_artifact_missing",
                stage="bundle",
            )
        value = str(raw_path or "").strip()
        path = Path(value)
        if not value or path.is_absolute():
            raise InferenceUnsupportedError(
                f"{description} must be a non-empty bundle-relative path",
                code="unigram_artifact_invalid",
                stage="bundle",
            )
        bundle_root = Path(bundle).resolve()
        candidate = (bundle_root / path).resolve()
        try:
            candidate.relative_to(bundle_root)
        except ValueError as exc:
            raise InferenceUnsupportedError(
                f"{description} escapes the model bundle: {value}",
                code="unigram_artifact_invalid",
                stage="bundle",
            ) from exc
        return candidate

    @staticmethod
    def _resolve_unigram_path(metadata, bundle=None):
        metadata = normalize_stream_fields(metadata)
        descriptor = ModelService._classic_value(
            metadata,
            "unigram_artifact",
            None,
        )
        if not isinstance(descriptor, dict):
            raise InferenceUnsupportedError(
                "Unigram bundle is missing the v1 embedded artifact descriptor; "
                "rebuild and re-export the model",
                code="unigram_artifact_missing",
                stage="bundle",
            )
        if (
            descriptor.get("contract_version") != UNIGRAM_CONTRACT_VERSION
            or descriptor.get("model_type") != "Unigram"
            or descriptor.get("variant") != UNIGRAM_VARIANT
            or descriptor.get("tokenizers_version") != UNIGRAM_TOKENIZERS_VERSION
        ):
            raise InferenceUnsupportedError(
                "Unigram artifact descriptor is incompatible with the platform contract",
                code="unigram_artifact_invalid",
                stage="bundle",
            )
        tokenizer_path = ModelService._resolve_unigram_bundle_member(
            bundle,
            descriptor.get("path"),
            description="Unigram tokenizer artifact",
        )
        provenance_path = ModelService._resolve_unigram_bundle_member(
            bundle,
            descriptor.get("provenance_path"),
            description="Unigram provenance",
        )
        if provenance_path != unigram_provenance_path(tokenizer_path).resolve():
            raise InferenceUnsupportedError(
                "Unigram provenance must use the tokenizer-adjacent sidecar path",
                code="unigram_artifact_invalid",
                stage="bundle",
            )
        try:
            inspection = inspect_unigram_artifact(
                tokenizer_path,
                expected_sha256=descriptor.get("sha256"),
            )
            provenance = load_unigram_provenance(
                tokenizer_path,
                provenance_path=provenance_path,
            )
            validate_unigram_provenance(
                provenance,
                inspection,
                context=f"Unigram provenance {provenance_path}",
            )
        except UnigramContractError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="bundle",
            ) from exc
        if descriptor.get("vocab_size") != inspection["vocab_size"] or descriptor.get(
            "max_raw_length"
        ) != (provenance.get("lengths") or {}).get("max_raw_length"):
            raise InferenceUnsupportedError(
                "Unigram provenance or descriptor does not match the bundled tokenizer",
                code="unigram_artifact_invalid",
                stage="bundle",
            )
        return str(tokenizer_path)

    @staticmethod
    def _load_classic_tfidf_idf(metadata, bundle=None):
        metadata = normalize_stream_fields(metadata)
        descriptor = ModelService._classic_value(
            metadata,
            "tfidf_idf",
            None,
        )
        if descriptor is not None:
            if not isinstance(descriptor, dict):
                raise InferenceUnsupportedError(
                    "classic TF-IDF state descriptor must be an object",
                    code="classic_tfidf_artifact_invalid",
                    stage="bundle",
                )
            required = {"path", "sha256", "dtype", "shape"}
            if not required.issubset(descriptor):
                raise InferenceUnsupportedError(
                    "classic TF-IDF state descriptor is incomplete",
                    code="classic_tfidf_artifact_invalid",
                    stage="bundle",
                )
            relative = Path(str(descriptor.get("path") or ""))
            if bundle is None or relative.is_absolute() or not relative.parts:
                raise InferenceUnsupportedError(
                    "classic TF-IDF state must use a bundle-relative path",
                    code="classic_tfidf_artifact_invalid",
                    stage="bundle",
                )
            bundle_root = Path(bundle).resolve()
            path = (bundle_root / relative).resolve()
            try:
                path.relative_to(bundle_root)
            except ValueError as exc:
                raise InferenceUnsupportedError(
                    "classic TF-IDF state path escapes the model bundle",
                    code="classic_tfidf_artifact_invalid",
                    stage="bundle",
                ) from exc
            if not path.is_file():
                raise InferenceUnsupportedError(
                    f"classic TF-IDF bundle is missing IDF tensor: {path}",
                    code="classic_tfidf_artifact_invalid",
                    stage="bundle",
                )
            if frequency_sha256_file(path) != str(descriptor["sha256"]):
                raise InferenceUnsupportedError(
                    "classic TF-IDF state digest mismatch",
                    code="classic_tfidf_artifact_invalid",
                    stage="bundle",
                )
            expected_shape = descriptor.get("shape")
            if (
                descriptor.get("dtype") != "float32"
                or not isinstance(expected_shape, list)
                or len(expected_shape) != 1
                or isinstance(expected_shape[0], bool)
                or not isinstance(expected_shape[0], int)
                or expected_shape[0] < 1
            ):
                raise InferenceUnsupportedError(
                    "classic TF-IDF state dtype/shape descriptor is invalid",
                    code="classic_tfidf_artifact_invalid",
                    stage="bundle",
                )
            try:
                idf = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                try:
                    idf = torch.load(path, map_location="cpu")
                except Exception as exc:
                    raise InferenceUnsupportedError(
                        "classic TF-IDF tensor could not be loaded",
                        code="classic_tfidf_artifact_invalid",
                        stage="bundle",
                    ) from exc
            except Exception as exc:
                raise InferenceUnsupportedError(
                    "classic TF-IDF tensor could not be loaded",
                    code="classic_tfidf_artifact_invalid",
                    stage="bundle",
                ) from exc
            if (
                not isinstance(idf, torch.Tensor)
                or idf.dtype != torch.float32
                or list(idf.shape) != expected_shape
            ):
                raise InferenceUnsupportedError(
                    "classic TF-IDF tensor does not match its descriptor",
                    code="classic_tfidf_artifact_invalid",
                    stage="bundle",
                )
            return idf.detach().clone()

        value = str(ModelService._classic_value(metadata, "tfidf_idf_path", "") or "").strip()
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            bundle_path = Path(bundle) / path if bundle is not None else None
            if bundle_path is not None and bundle_path.exists():
                path = bundle_path
            else:
                path = PROJECT_ROOT / path
        if not path.exists():
            raise InferenceUnsupportedError(f"classic TF-IDF bundle is missing IDF tensor: {path}")
        try:
            idf = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            try:
                idf = torch.load(path, map_location="cpu")
            except Exception as exc:
                raise InferenceUnsupportedError(
                    "classic TF-IDF tensor could not be loaded",
                    code="classic_tfidf_artifact_invalid",
                    stage="bundle",
                ) from exc
        except Exception as exc:
            raise InferenceUnsupportedError(
                "classic TF-IDF tensor could not be loaded",
                code="classic_tfidf_artifact_invalid",
                stage="bundle",
            ) from exc
        if not isinstance(idf, torch.Tensor):
            idf = torch.tensor(idf, dtype=torch.float)
        return idf.float()

    @staticmethod
    def _homogeneous_tfidf_uses_stream_metadata(metadata):
        """Prefer role-specific TF-IDF state for schema-v2 homogeneous bundles."""
        metadata = normalize_stream_fields(metadata)

        pair = metadata.get("pair") or {}
        if str(pair.get("stream_mode", "") or "").strip().lower() != "homogeneous":
            return False
        classic = metadata.get("classic") or {}
        if str(classic.get("tfidf_fit_scope", "") or "") == "combined_pair_roles":
            return False
        streams = metadata.get("streams") or {}
        if not all(isinstance(streams.get(role), dict) for role in ("peptide", "partner")):
            return False
        for role in ("peptide", "partner"):
            stream = streams[role]
            spec = stream.get("spec") or stream
            encoder = normalize_encoder_name(str(spec.get("encoder", "index") or "index"))
            if encoder != "tfidf":
                return False
        return True

    @staticmethod
    def _legacy_pair_rows(metadata, csv_key):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.hlai_input import HLAIInputError, hla_allele, require_hlai_partner_sequence

        args = dict(metadata.get("args") or {})
        raw_csv = str(args.get(csv_key, "") or "").strip()
        if not raw_csv:
            raise InferenceUnsupportedError(
                f"legacy role-specific TF-IDF reconstruction requires args.{csv_key}"
            )
        csv_path = Path(raw_csv).expanduser()
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
        if not csv_path.is_file():
            raise InferenceUnsupportedError(
                f"legacy role-specific TF-IDF CSV is missing: {csv_path}"
            )

        hla_pseudo_map = {}
        if normalize_task_type(args.get("task_type", "") or "") == "pep_hlai":
            task_root = Path(str(args.get("task_root", "") or "").strip()).expanduser()
            if task_root and not task_root.is_absolute():
                task_root = PROJECT_ROOT / task_root
            pseudo_path = task_root / "pseudoseqs_decoded.csv"
            if pseudo_path.is_file():
                with pseudo_path.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        hla = str(hla_allele(row) or "").strip()
                        pseudo = str(row.get("pseudoseq", "") or "").strip().upper()
                        status = str(row.get("status", "ok") or "ok").strip().lower()
                        if hla and pseudo and status in {"", "ok"}:
                            hla_pseudo_map[hla] = pseudo

        rows = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=1):
                sample_id = (
                    str(row.get("sample_id", "") or "").strip() or f"{csv_path.stem}_{index}"
                )
                label = row.get("label", row.get("tgt", row.get("bind", "")))
                peptide = str(row.get("peptide", row.get("pep", "")) or "").strip().upper()
                protein = str(row.get("protein", "") or "").strip()
                hla = str(hla_allele(row) or "").strip()
                pair_pseudo = str(row.get("pair_pseudo", "") or "").strip().upper()
                hla_pair = str(row.get("hla_pair", "") or "").strip()
                partner = pair_pseudo or protein or hla or hla_pair
                if protein:
                    partner = protein
                elif hla_pseudo_map and hla in hla_pseudo_map:
                    partner = hla_pseudo_map[hla]
                if label in {None, ""} or not peptide or not partner:
                    continue
                if hla or normalize_task_type(args.get("task_type", "")) == "pep_hlai":
                    try:
                        require_hlai_partner_sequence(
                            partner,
                            allele=hla,
                            resolved=bool(
                                protein or pair_pseudo or (hla_pseudo_map and hla in hla_pseudo_map)
                            ),
                            source_path=csv_path,
                            row_number=index,
                            pseudoseq_path=pseudo_path
                            if normalize_task_type(args.get("task_type", "")) == "pep_hlai"
                            else None,
                        )
                    except HLAIInputError as exc:
                        raise InferenceUnsupportedError(str(exc)) from exc
                rows.append((sample_id, peptide, partner, int(float(label))))
        if not rows:
            raise InferenceUnsupportedError(
                f"legacy role-specific TF-IDF reconstruction found no rows in {csv_path}"
            )
        return csv_path, rows

    @staticmethod
    def _legacy_tfidf_training_sequences(metadata, role):
        """Recover role-specific IDF input from a legacy bundle's locked train CSV."""
        metadata = normalize_stream_fields(metadata)

        _, rows = ModelService._legacy_pair_rows(metadata, "train_csv")
        column = 1 if role == "peptide" else 2
        return [row[column] for row in rows]

    @staticmethod
    def _legacy_feature_input_digest(items):
        digest = hashlib.sha256()
        for item in items:
            parts = [str(part) for part in item] if isinstance(item, (list, tuple)) else [str(item)]
            for part in parts:
                encoded = part.encode("utf-8", errors="replace")
                digest.update(str(len(encoded)).encode("ascii"))
                digest.update(b":")
                digest.update(encoded)
                digest.update(b"\x1f")
            digest.update(b"\x1e")
        return digest.hexdigest()

    @staticmethod
    def _legacy_tfidf_cache_path(metadata, role):
        metadata = normalize_stream_fields(metadata)
        args = dict(metadata.get("args") or {})
        raw_feature_dir = str(args.get("feature_dir", "") or "").strip()
        if not raw_feature_dir:
            return None
        feature_dir = Path(raw_feature_dir).expanduser()
        if not feature_dir.is_absolute():
            feature_dir = PROJECT_ROOT / feature_dir
        tokenizer = (
            str(ModelService._classic_value(metadata, "tokenizer", "") or "").strip().lower()
        )
        encoder = normalize_encoder_name(ModelService._classic_value(metadata, "encoder", "index"))
        if encoder != "tfidf" or tokenizer != "kmer":
            return None
        parts = [
            tokenizer.replace("-", ""),
            "tfidf",
            f"len{ModelService._classic_int(metadata, 'max_length', 60)}",
            f"k{ModelService._classic_int(metadata, 'k', 2)}",
        ]
        return feature_dir / "_".join(parts) / f"val_{role}.pt"

    @staticmethod
    def _recover_legacy_tfidf_idf(metadata, featurizer, role, bundle=None):
        metadata = normalize_stream_fields(metadata)
        cache_path = ModelService._legacy_tfidf_cache_path(metadata, role)
        if cache_path is None or not cache_path.is_file():
            return None
        meta_path = Path(f"{cache_path}.meta.json")
        if not meta_path.is_file():
            return None
        stat = cache_path.stat()
        key = (str(cache_path.resolve()), stat.st_size, stat.st_mtime_ns)
        cached = _LEGACY_TFIDF_IDF_CACHE.get(key)
        if cached is not None:
            return cached.clone()

        cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _, val_rows = ModelService._legacy_pair_rows(metadata, "val_csv")
        role_column = 1 if role == "peptide" else 2
        input_items = [(row[0], row[role_column]) for row in val_rows]
        expected = {
            "feature_fit_scope": "train_only",
            "split": "val",
            "role": role,
            "tokenizer": str(ModelService._classic_value(metadata, "tokenizer", "") or "")
            .strip()
            .lower(),
            "encoder": "tfidf",
            "max_length": ModelService._classic_int(metadata, "max_length", 60),
            "row_count": len(val_rows),
            "input_sha256": ModelService._legacy_feature_input_digest(input_items),
        }
        mismatches = {
            name: (cache_meta.get(name), value)
            for name, value in expected.items()
            if cache_meta.get(name) != value
        }
        if mismatches:
            raise InferenceUnsupportedError(
                f"legacy TF-IDF cache provenance mismatch for {role}: {mismatches}"
            )
        _, train_rows = ModelService._legacy_pair_rows(metadata, "train_csv")
        train_digest = ModelService._legacy_feature_input_digest(train_rows)
        if cache_meta.get("train_sha256") != train_digest:
            raise InferenceUnsupportedError(f"legacy TF-IDF cache train digest mismatch for {role}")

        try:
            encoded_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:
            encoded_cache = torch.load(cache_path, map_location="cpu")
        sequences = [row[role_column] for row in val_rows]
        base = torch.stack(
            [featurizer.encode(sequence).reshape(-1).float() for sequence in sequences]
        )
        if not isinstance(encoded_cache, torch.Tensor) or encoded_cache.shape != base.shape:
            raise InferenceUnsupportedError(
                f"legacy TF-IDF cache shape mismatch for {role}: "
                f"{getattr(encoded_cache, 'shape', None)} != {base.shape}"
            )

        shared_path = str(
            ModelService._classic_value(metadata, "legacy_shared_tfidf_idf_path", "") or ""
        ).strip()
        recovered = torch.ones(base.shape[1], dtype=torch.float)
        if shared_path:
            shared_metadata = {
                "classic": {"tfidf_idf_path": shared_path},
                "args": metadata.get("args") or {},
            }
            shared = ModelService._load_classic_tfidf_idf(shared_metadata, bundle=bundle)
            if shared is not None and shared.numel() == recovered.numel():
                recovered = shared.reshape(-1).clone().float()
        for column in range(base.shape[1]):
            indices = torch.nonzero(base[:, column] != 0, as_tuple=False).reshape(-1)
            if not indices.numel():
                continue
            ratios = encoded_cache[indices, column].float() / base[indices, column]
            value = ratios[0]
            if not torch.allclose(ratios, value.expand_as(ratios), rtol=1e-6, atol=1e-6):
                raise InferenceUnsupportedError(
                    f"legacy TF-IDF cache has inconsistent IDF values for {role} column {column}"
                )
            recovered[column] = value
        reconstructed = base * recovered[None, :]
        if not torch.equal(reconstructed, encoded_cache.float()):
            if not torch.allclose(reconstructed, encoded_cache.float(), rtol=1e-6, atol=1e-7):
                raise InferenceUnsupportedError(
                    f"legacy TF-IDF cache reconstruction failed for {role}"
                )
        _LEGACY_TFIDF_IDF_CACHE[key] = recovered.clone()
        return recovered

    @staticmethod
    def _build_classic_featurizer(metadata, bundle=None, role=None):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.featurizers import FeaturizerFactory

        tokenizer = (
            str(ModelService._classic_value(metadata, "tokenizer", "") or "").strip().lower()
        )
        encoder = normalize_encoder_name(ModelService._classic_value(metadata, "encoder", "index"))
        if encoder == "ppi_fusion":
            encoder = normalize_encoder_name(
                ModelService._classic_value(metadata, "aux_encoder", "index")
            )
        if encoder == "cs":
            raise InferenceUnsupportedError("CS encoder has been removed")
        if encoder == "property":
            raise InferenceUnsupportedError(
                "Property encoder has been removed; retraining with another encoder is required",
                code="property_encoder_removed",
                stage="bundle",
            )
        if not tokenizer:
            raise InferenceUnsupportedError("classic pair bundle is missing tokenizer metadata")
        if tokenizer not in {
            "kmer",
            "dr",
            "top-n-gram",
            "dt",
            "bpe",
            "wordpiece",
            "unigram",
        }:
            raise InferenceUnsupportedError(
                f"unsupported tokenizer in classic artifact: {tokenizer}"
            )
        persisted_covariance_contract = None
        dr_unknown_residue_policy = DR_DEFAULT_UNKNOWN_RESIDUE_POLICY
        covariance_max_raw_length = COVARIANCE_DEFAULT_MAX_RAW_LENGTH
        if encoder in COVARIANCE_ENCODERS:
            persisted_covariance_contract = ModelService._classic_value(
                metadata,
                "covariance_encoder_contract",
                None,
            )
            if not isinstance(persisted_covariance_contract, dict):
                raise InferenceUnsupportedError(
                    "AC/CC/ACC artifact is missing the Guo-7 v1 covariance "
                    "contract; retrain and re-export it",
                    code="covariance_contract_unsupported",
                    stage="bundle",
                )
            lag_key = f"{encoder}_lag"
            raw_lag = ModelService._classic_value(metadata, lag_key, None)
            raw_max_raw_length = ModelService._classic_value(
                metadata,
                "covariance_max_raw_length",
                None,
            )
            try:
                covariance_lag = validate_covariance_lag(raw_lag)
                covariance_max_raw_length = validate_covariance_max_raw_length(raw_max_raw_length)
                validate_covariance_encoder_contract(
                    persisted_covariance_contract,
                    encoder=encoder,
                    lag=covariance_lag,
                    max_raw_length=covariance_max_raw_length,
                )
            except (KeyError, ValueError) as exc:
                raise InferenceUnsupportedError(
                    "AC/CC/ACC artifact has an unsupported or inconsistent "
                    "covariance contract; retrain and re-export it",
                    code="covariance_contract_unsupported",
                    stage="bundle",
                    technical_detail=str(exc),
                ) from exc
        persisted_pdt_contract = None
        pdt_max_raw_length = PDT_DEFAULT_MAX_RAW_LENGTH
        if encoder in PDT_ENCODERS:
            persisted_pdt_contract = ModelService._classic_value(
                metadata,
                "pdt_encoder_contract",
                None,
            )
            if not isinstance(persisted_pdt_contract, dict):
                raise InferenceUnsupportedError(
                    "PDT artifact is missing the supported scientific "
                    "contract; retrain and re-export it",
                    code="pdt_contract_unsupported",
                    stage="bundle",
                )
            try:
                pdt_lag = validate_pdt_lag(
                    ModelService._classic_value(
                        metadata,
                        "pdt_lag",
                        PDT_DEFAULT_LAG,
                    )
                )
                pdt_max_raw_length = validate_pdt_max_raw_length(
                    ModelService._classic_value(
                        metadata,
                        "pdt_max_raw_length",
                        None,
                    )
                )
                validate_pdt_encoder_contract(
                    persisted_pdt_contract,
                    lag=pdt_lag,
                    max_raw_length=pdt_max_raw_length,
                )
            except (KeyError, ValueError) as exc:
                raise InferenceUnsupportedError(
                    "PDT artifact has an unsupported or inconsistent "
                    "scientific contract; retrain and re-export it",
                    code="pdt_contract_unsupported",
                    stage="bundle",
                    technical_detail=str(exc),
                ) from exc
        persisted_pssm_covariance_contract = None
        if encoder in PSSM_COVARIANCE_ENCODERS:
            persisted_pssm_covariance_contract = ModelService._classic_value(
                metadata,
                "pssm_covariance_encoder_contract",
                None,
            )
            if not isinstance(persisted_pssm_covariance_contract, dict):
                raise InferenceUnsupportedError(
                    "AC-PSSM/CC-PSSM/ACC-PSSM artifact is missing the "
                    "supported scientific contract; retraining and re-export "
                    "are required",
                    code="pssm_covariance_contract_unsupported",
                    stage="bundle",
                )
            lag_key = pssm_covariance_lag_key(encoder)
            try:
                pssm_covariance_lag = validate_pssm_covariance_lag(
                    ModelService._classic_value(metadata, lag_key, None)
                )
                validate_pssm_covariance_encoder_contract(
                    persisted_pssm_covariance_contract,
                    encoder=encoder,
                    lag=pssm_covariance_lag,
                )
            except (KeyError, ValueError) as exc:
                raise InferenceUnsupportedError(
                    "AC-PSSM/CC-PSSM/ACC-PSSM artifact has an unsupported "
                    "or inconsistent scientific contract; retraining and "
                    "re-export are required",
                    code="pssm_covariance_contract_unsupported",
                    stage="bundle",
                    technical_detail=str(exc),
                ) from exc
        persisted_textrank_contract = None
        if encoder == "textrank":
            persisted_textrank_contract = ModelService._classic_value(
                metadata,
                "textrank_contract",
                None,
            )
            if not isinstance(persisted_textrank_contract, dict):
                raise InferenceUnsupportedError(
                    "TextRank artifact is missing the strict weighted "
                    "TextRank contract; retrain and re-export it",
                    code="textrank_contract_unsupported",
                    stage="bundle",
                )
        if tokenizer == "kmer":
            try:
                validate_kmer_k(ModelService._classic_int(metadata, "k", 2))
            except ValueError as exc:
                raise InferenceUnsupportedError(
                    f"unsupported K-mer artifact: {exc}",
                    code="kmer_k_unsupported",
                ) from exc
        if tokenizer == "dr":
            try:
                max_distance = validate_dr_max_distance(
                    ModelService._classic_value(
                        metadata,
                        "dr_max_distance",
                        None,
                    )
                )
                max_raw_length = validate_dr_max_raw_length(
                    ModelService._classic_value(
                        metadata,
                        "dr_max_raw_length",
                        None,
                    )
                )
                dr_unknown_residue_policy = validate_dr_unknown_residue_policy(
                    ModelService._classic_value(
                        metadata,
                        "dr_unknown_residue_policy",
                        DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
                    )
                )
            except DRContractError as exc:
                raise InferenceUnsupportedError(
                    "DR artifact is missing valid v2 parameters; retrain and re-export it",
                    code="dr_contract_unsupported",
                    stage="bundle",
                    technical_detail=str(exc),
                ) from exc
            contract = ModelService._classic_value(
                metadata,
                "dr_contract",
                None,
            )
            expected_contract = dr_contract_metadata(
                max_distance=max_distance,
                max_raw_length=max_raw_length,
                unknown_residue_policy=dr_unknown_residue_policy,
            )
            if (
                not isinstance(contract, dict)
                or contract.get("contract_version") != DR_CONTRACT_VERSION
                or contract != expected_contract
            ):
                raise InferenceUnsupportedError(
                    "DR artifact does not implement the platform v2 contract; "
                    "retrain and re-export it",
                    code="dr_contract_unsupported",
                    stage="bundle",
                )
        if tokenizer == "top-n-gram":
            ModelService._validate_top_ngram_bundle_contract(metadata)
        if tokenizer == "dt":
            ModelService._validate_dt_bundle_contract(metadata)
            dt_max_distance = validate_dt_max_distance(
                ModelService._classic_value(
                    metadata,
                    "dt_max_distance",
                    None,
                )
            )
        frequency_contract = ModelService._classic_value(
            metadata,
            "frequency_encoder_contract",
            None,
        )
        if frequency_contract is None:
            frequency_contract_version = FREQUENCY_LEGACY_CONTRACT_VERSION
            frequency_max_raw_length = FREQUENCY_DEFAULT_MAX_RAW_LENGTH
        else:
            required_frequency_fields = {
                "contract_version",
                "stream_policy",
                "bow_formula",
                "tf_formula",
                "idf_formula",
                "post_normalization",
                "frequency_max_raw_length",
                "feature_dimension",
                "vocabulary_sha256",
                "training_document_count",
                "training_corpus_sha256",
                "fit_scope",
                "encoder",
            }
            if (
                not isinstance(frequency_contract, dict)
                or not required_frequency_fields.issubset(frequency_contract)
                or frequency_contract.get("contract_version") != FREQUENCY_ENCODER_CONTRACT_VERSION
            ):
                raise InferenceUnsupportedError(
                    "frequency encoder artifact has an unsupported contract",
                    code="frequency_contract_unsupported",
                    stage="bundle",
                )
            try:
                frequency_max_raw_length = validate_frequency_max_raw_length(
                    frequency_contract.get("frequency_max_raw_length")
                )
            except ValueError as exc:
                raise InferenceUnsupportedError(
                    f"frequency encoder artifact has an invalid length limit: {exc}",
                    code="frequency_contract_unsupported",
                    stage="bundle",
                ) from exc
            frequency_contract_version = FREQUENCY_ENCODER_CONTRACT_VERSION
            if encoder == "tfidf":
                document_count = frequency_contract.get("training_document_count")
                corpus_digest = frequency_contract.get("training_corpus_sha256")
                fit_scope = frequency_contract.get("fit_scope")
                if (
                    isinstance(document_count, bool)
                    or not isinstance(document_count, int)
                    or document_count < 1
                    or not isinstance(corpus_digest, str)
                    or len(corpus_digest) != 64
                    or not isinstance(fit_scope, str)
                    or not fit_scope
                    or not isinstance(
                        ModelService._classic_value(
                            metadata,
                            "tfidf_idf",
                            None,
                        ),
                        dict,
                    )
                ):
                    raise InferenceUnsupportedError(
                        "v2 TF-IDF artifact is missing fit provenance or its signed IDF descriptor",
                        code="classic_tfidf_artifact_invalid",
                        stage="bundle",
                    )
        persisted_idf = (
            ModelService._load_classic_tfidf_idf(
                metadata,
                bundle=bundle,
            )
            if encoder == "tfidf"
            else None
        )
        if encoder == "tfidf" and persisted_idf is None:
            raise InferenceUnsupportedError(
                f"{tokenizer} TF-IDF artifact is missing its train-only IDF "
                "vector; retrain and re-export it",
                code="classic_tfidf_artifact_invalid",
                stage="bundle",
            )
        bpe_path = (
            ModelService._resolve_bpe_path(metadata, bundle=bundle)
            if tokenizer == "bpe"
            else ModelService._resolve_classic_path(
                metadata,
                "bpe_path",
                bundle=bundle,
            )
        )
        wordpiece_path = (
            ModelService._resolve_wordpiece_path(
                metadata,
                bundle=bundle,
            )
            if tokenizer == "wordpiece"
            else ModelService._resolve_classic_path(
                metadata,
                "wordpiece_path",
                bundle=bundle,
            )
        )
        unigram_path = (
            ModelService._resolve_unigram_path(
                metadata,
                bundle=bundle,
            )
            if tokenizer == "unigram"
            else ModelService._resolve_classic_path(
                metadata,
                "unigram_path",
                bundle=bundle,
            )
        )
        word2vec_path = ModelService._resolve_classic_path(
            metadata,
            "word2vec_path",
            bundle=bundle,
        )
        glove_path = ModelService._resolve_classic_path(
            metadata,
            "glove_path",
            bundle=bundle,
        )
        fasttext_path = ModelService._resolve_classic_path(
            metadata,
            "fasttext_path",
            bundle=bundle,
        )
        if encoder in {
            "word2vec",
            "glove",
            "fasttext",
        }:
            descriptor = (metadata.get("classic") or {}).get("static_embedding") or metadata.get(
                "static_embedding"
            )
            resolved_embedding = {
                "word2vec": word2vec_path,
                "glove": glove_path,
                "fasttext": fasttext_path,
            }[encoder]
            # DR bundles have always required an explicit descriptor.  For
            # k-mer bundles, accept legacy metadata without one, but validate
            # every descriptor produced by the current training pipelines.
            require_descriptor = tokenizer == "dr" or isinstance(
                descriptor,
                dict,
            )
            if require_descriptor:
                error_code = (
                    "dr_static_embedding_artifact_invalid"
                    if tokenizer == "dr"
                    else "static_embedding_artifact_invalid"
                )
                if (
                    not isinstance(descriptor, dict)
                    or descriptor.get("method") != encoder
                    or descriptor.get("bundled") is not True
                    or descriptor.get("path")
                    != ModelService._classic_value(
                        metadata,
                        f"{encoder}_path",
                        "",
                    )
                    or not descriptor.get("sha256")
                    or not resolved_embedding
                    or not os.path.isfile(resolved_embedding)
                ):
                    raise InferenceUnsupportedError(
                        "Static-embedding artifact is missing a bundled model or its provenance",
                        code=error_code,
                        stage="bundle",
                    )
                digest = hashlib.sha256()
                try:
                    with open(resolved_embedding, "rb") as handle:
                        for chunk in iter(
                            lambda: handle.read(1024 * 1024),
                            b"",
                        ):
                            digest.update(chunk)
                except OSError as exc:
                    raise InferenceUnsupportedError(
                        "Static-embedding artifact could not be read",
                        code=error_code,
                        stage="bundle",
                    ) from exc
                if digest.hexdigest() != descriptor.get("sha256"):
                    raise InferenceUnsupportedError(
                        "Static-embedding artifact digest mismatch",
                        code=error_code,
                        stage="bundle",
                    )
        factory = FeaturizerFactory(
            bpe_path=bpe_path,
            unigram_path=unigram_path,
            wordpiece_path=wordpiece_path,
            word2vec_path=word2vec_path,
            glove_path=glove_path,
            fasttext_path=fasttext_path,
            pssm_index_path=ModelService._resolve_classic_path(
                metadata, "pssm_index_path", bundle=bundle
            ),
            ac_pssm_blosum62_hybrid_index_path=ModelService._resolve_classic_path(
                metadata,
                "ac_pssm_blosum62_hybrid_index_path",
                bundle=bundle,
            ),
            cc_pssm_blosum62_hybrid_index_path=ModelService._resolve_classic_path(
                metadata,
                "cc_pssm_blosum62_hybrid_index_path",
                bundle=bundle,
            ),
            acc_pssm_blosum62_hybrid_index_path=ModelService._resolve_classic_path(
                metadata,
                "acc_pssm_blosum62_hybrid_index_path",
                bundle=bundle,
            ),
            psfm_index_path=ModelService._resolve_classic_path(
                metadata, "psfm_index_path", bundle=bundle
            ),
            frequency_profile_index_path=ModelService._resolve_classic_path(
                metadata,
                "frequency_profile_index_path",
                bundle=bundle,
            ),
            ac_lag=ModelService._classic_int(metadata, "ac_lag", 10),
            cc_lag=ModelService._classic_int(metadata, "cc_lag", 10),
            acc_lag=ModelService._classic_int(metadata, "acc_lag", 10),
            pdt_lag=ModelService._classic_int(
                metadata,
                "pdt_lag",
                PDT_DEFAULT_LAG,
            ),
            ac_pssm_lag=ModelService._classic_int(metadata, "ac_pssm_lag", 10),
            cc_pssm_lag=ModelService._classic_int(metadata, "cc_pssm_lag", 10),
            acc_pssm_lag=ModelService._classic_int(metadata, "acc_pssm_lag", 10),
            ac_pssm_blosum62_hybrid_lag=ModelService._classic_int(
                metadata, "ac_pssm_blosum62_hybrid_lag", 10
            ),
            cc_pssm_blosum62_hybrid_lag=ModelService._classic_int(
                metadata, "cc_pssm_blosum62_hybrid_lag", 10
            ),
            acc_pssm_blosum62_hybrid_lag=ModelService._classic_int(
                metadata, "acc_pssm_blosum62_hybrid_lag", 10
            ),
            covariance_max_raw_length=covariance_max_raw_length,
            pdt_max_raw_length=pdt_max_raw_length,
            wordpiece_max_raw_length=validate_wordpiece_max_raw_length(
                ModelService._classic_int(
                    metadata,
                    "wordpiece_max_raw_length",
                    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            unigram_max_raw_length=validate_unigram_max_raw_length(
                ModelService._classic_int(
                    metadata,
                    "unigram_max_raw_length",
                    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            dr_max_raw_length=(max_raw_length if tokenizer == "dr" else DR_DEFAULT_MAX_RAW_LENGTH),
        )
        try:
            featurizer = factory.create(
                tokenizer,
                encoder,
                ModelService._classic_int(metadata, "max_length", 60),
                ModelService._classic_int(metadata, "k", 2),
                dr_max_distance=(
                    max_distance
                    if tokenizer == "dr"
                    else ModelService._classic_int(
                        metadata,
                        "dr_max_distance",
                        5,
                    )
                ),
                dr_unknown_residue_policy=dr_unknown_residue_policy,
                top_ngram_n=ModelService._classic_int(metadata, "top_ngram_n", 2),
                dt_max_distance=(
                    dt_max_distance
                    if tokenizer == "dt"
                    else ModelService._classic_int(metadata, "dt_max_distance", 5)
                ),
                persisted_idf=persisted_idf,
                frequency_contract_version=frequency_contract_version,
                frequency_max_raw_length=frequency_max_raw_length,
            )
        except ValueError as exc:
            if encoder in COVARIANCE_ENCODERS:
                message = f"covariance artifact cannot reconstruct the Guo-7 encoder state: {exc}"
                error_code = "covariance_contract_unsupported"
            elif encoder in PDT_ENCODERS:
                message = (
                    f"PDT artifact cannot reconstruct the supported scientific encoder state: {exc}"
                )
                error_code = "pdt_contract_unsupported"
            elif encoder in PSSM_COVARIANCE_ENCODERS:
                message = (
                    "PSSM covariance artifact cannot reconstruct the "
                    f"supported encoder state: {exc}"
                )
                error_code = "pssm_covariance_contract_unsupported"
            elif encoder == "textrank":
                message = f"TextRank artifact cannot reconstruct its tokenizer state: {exc}"
                error_code = "textrank_contract_unsupported"
            elif "vector width" in str(exc):
                message = (
                    f"classic TF-IDF artifact width does not match the tokenizer contract: {exc}"
                )
                error_code = "classic_tfidf_artifact_invalid"
            else:
                message = f"classic frequency artifact is invalid: {exc}"
                error_code = "classic_tfidf_artifact_invalid"
            raise InferenceUnsupportedError(
                message,
                code=error_code,
                stage="bundle",
            ) from exc
        if encoder in COVARIANCE_ENCODERS:
            reconstructed_covariance_contract = getattr(
                featurizer,
                "covariance_encoder_contract",
                None,
            )
            if (
                not isinstance(reconstructed_covariance_contract, dict)
                or persisted_covariance_contract != reconstructed_covariance_contract
            ):
                raise InferenceUnsupportedError(
                    "AC/CC/ACC artifact does not match the supported Guo-7 "
                    "v1 covariance contract; retrain and re-export it",
                    code="covariance_contract_unsupported",
                    stage="bundle",
                )
        if encoder in PDT_ENCODERS:
            reconstructed_pdt_contract = getattr(
                featurizer,
                "pdt_encoder_contract",
                None,
            )
            if (
                not isinstance(reconstructed_pdt_contract, dict)
                or persisted_pdt_contract != reconstructed_pdt_contract
            ):
                raise InferenceUnsupportedError(
                    "PDT artifact does not match the supported Liu et al. "
                    "v1 scientific contract; retrain and re-export it",
                    code="pdt_contract_unsupported",
                    stage="bundle",
                )
        if encoder in PSSM_COVARIANCE_ENCODERS:
            reconstructed_pssm_covariance_contract = getattr(
                featurizer,
                "pssm_covariance_encoder_contract",
                None,
            )
            if (
                not isinstance(
                    reconstructed_pssm_covariance_contract,
                    dict,
                )
                or persisted_pssm_covariance_contract != reconstructed_pssm_covariance_contract
            ):
                raise InferenceUnsupportedError(
                    "AC-PSSM/CC-PSSM/ACC-PSSM artifact does not match the "
                    "supported scientific contract; retraining and re-export "
                    "are required",
                    code="pssm_covariance_contract_unsupported",
                    stage="bundle",
                )
        if frequency_contract is not None and hasattr(
            featurizer,
            "frequency_encoder_contract",
        ):
            reconstructed = featurizer.frequency_encoder_contract
            static_keys = {
                "contract_version",
                "stream_policy",
                "bow_formula",
                "tf_formula",
                "idf_formula",
                "post_normalization",
                "frequency_max_raw_length",
                "feature_dimension",
                "vocabulary_sha256",
                "encoder",
            }
            if any(frequency_contract.get(key) != reconstructed.get(key) for key in static_keys):
                raise InferenceUnsupportedError(
                    "frequency encoder metadata does not match the tokenizer state",
                    code="frequency_contract_unsupported",
                    stage="bundle",
                )
        if encoder == "textrank":
            reconstructed_textrank_contract = getattr(
                featurizer,
                "textrank_contract",
                None,
            )
            if (
                not isinstance(persisted_textrank_contract, dict)
                or not isinstance(reconstructed_textrank_contract, dict)
                or persisted_textrank_contract != reconstructed_textrank_contract
            ):
                raise InferenceUnsupportedError(
                    "TextRank artifact is missing or does not match the strict "
                    "weighted TextRank contract; retrain and re-export it",
                    code="textrank_contract_unsupported",
                    stage="bundle",
                )
        classic_metadata = metadata.get("classic") or {}
        embedding_export = metadata.get("embedding_export") or {}
        classic_embedding_export = classic_metadata.get("embedding_export") or {}
        contracts = [
            metadata.get("predictor_contract"),
            metadata.get("architecture_contract"),
            embedding_export.get("architecture_contract"),
            classic_embedding_export.get("architecture_contract"),
        ]
        if any(
            isinstance(contract, dict)
            and contract.get("implementation")
            in {
                "pepplatform_bidirectional_elman_rnn",
                "pepplatform_bidirectional_pytorch_gru",
                "pepplatform_bidirectional_pytorch_lstm",
                "pepplatform_pytorch_transformer_encoder",
                "pepplatform_masked_chain_gcn",
            }
            and isinstance(contract.get("contract_version"), int)
            and not isinstance(contract.get("contract_version"), bool)
            and contract.get("contract_version") >= 2
            and contract.get("status") == "verified_platform_variant"
            for contract in contracts
        ):
            featurizer.require_explicit_position_mask()
        return featurizer, encoder

    @staticmethod
    def _extract_classic_pair_embeddings(
        metadata, sequences, device, sample_ids=None, bundle=None, role=None
    ):
        metadata = normalize_stream_fields(metadata)
        try:
            featurizer, _ = ModelService._build_classic_featurizer(
                metadata, bundle=bundle, role=role
            )
            rows = []
            sample_ids = sample_ids or [None] * len(sequences)
            for sequence, sample_id in zip(sequences, sample_ids):
                encoded = featurizer.encode(sequence, sample_id=sample_id)
                if hasattr(encoded, "features") and hasattr(encoded, "mask"):
                    encoded = encoded.features
                if not isinstance(encoded, torch.Tensor):
                    encoded = torch.tensor(encoded)
                rows.append(encoded.reshape(-1).float())
        except InferenceUnsupportedError:
            raise
        except BPEContractError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="input",
            ) from exc
        except Exception as exc:
            raise InferenceUnsupportedError(
                f"classic pair-vector feature reconstruction failed: {exc}"
            ) from exc
        return torch.stack(rows, dim=0).to(device)

    @staticmethod
    def _extract_classic_pair_token_sequences(
        metadata, sequences, device, sample_ids=None, bundle=None, role=None
    ):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.models import ResidueModelInput

        try:
            featurizer, _ = ModelService._build_classic_featurizer(
                metadata, bundle=bundle, role=role
            )
            rows = []
            masks = []
            sample_ids = sample_ids or [None] * len(sequences)
            for sequence, sample_id in zip(sequences, sample_ids):
                encoded = featurizer.encode(sequence, sample_id=sample_id)
                if isinstance(encoded, ResidueModelInput):
                    rows.append(encoded.features.float())
                    masks.append(encoded.mask.bool())
                    continue
                if not isinstance(encoded, torch.Tensor):
                    encoded = torch.tensor(encoded)
                if encoded.ndim == 1:
                    rows.append(encoded.long())
                elif encoded.ndim == 2:
                    rows.append(encoded.float())
                else:
                    raise ValueError(
                        f"classic token feature rank must be 1 or 2, got {encoded.ndim}"
                    )
        except InferenceUnsupportedError:
            raise
        except BPEContractError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="input",
            ) from exc
        except Exception as exc:
            raise InferenceUnsupportedError(
                f"classic pair-token feature reconstruction failed: {exc}"
            ) from exc
        if not rows:
            return torch.empty((0, 0), dtype=torch.long, device=device)
        rank = rows[0].ndim
        if any(row.ndim != rank for row in rows):
            raise InferenceUnsupportedError("classic pair-token features have mixed ranks")
        features = torch.stack(rows, dim=0).to(device)
        if masks:
            return ResidueModelInput(
                features,
                torch.stack(masks, dim=0).to(device),
            )
        return features

    @staticmethod
    def _extract_stream_tokens(metadata, pairs, role, device, bundle=None):
        metadata = normalize_stream_fields(metadata)
        stream_metadata = ModelService._stream_legacy_metadata(metadata, role)
        sequence_key = "peptide" if role == "peptide" else "protein"
        sequences = [pair[sequence_key] for pair in pairs]
        if stream_metadata.get("pretrained"):
            return ModelService._extract_pretrained_token_sequences(
                stream_metadata,
                sequences,
                device,
            )
        if stream_metadata.get("classic"):
            sample_ids = None
            prepared = stream_metadata
            if ModelService._classic_uses_profile(stream_metadata):
                prepared, peptide_ids, protein_ids = ModelService._prepare_classic_profile_metadata(
                    stream_metadata,
                    pairs,
                    bundle,
                )
                sample_ids = peptide_ids if role == "peptide" else protein_ids
            return ModelService._extract_classic_pair_token_sequences(
                prepared,
                sequences,
                device,
                sample_ids=sample_ids,
                bundle=bundle,
                role="peptide" if role == "peptide" else "protein",
            )
        raise InferenceUnsupportedError(f"{role} stream is missing feature metadata")

    @staticmethod
    def _extract_pair_token_streams(metadata, pairs, device, bundle=None):
        metadata = normalize_stream_fields(metadata)
        peptide_tokens = ModelService._extract_stream_tokens(
            metadata,
            pairs,
            "peptide",
            device,
            bundle=bundle,
        )
        partner_tokens = ModelService._extract_stream_tokens(
            metadata,
            pairs,
            "partner",
            device,
            bundle=bundle,
        )
        if peptide_tokens.shape[0] != partner_tokens.shape[0]:
            raise InferenceUnsupportedError("pair streams produced different token batch sizes")
        peptide_tokens, partner_tokens = ModelService._cast_pair_tokens_to_contract_dtype(
            metadata,
            peptide_tokens,
            partner_tokens,
        )
        ModelService._validate_pair_strategy_metadata(
            metadata,
            peptide_tokens,
            partner_tokens,
        )
        return peptide_tokens, partner_tokens

    @staticmethod
    def _cast_pair_tokens_to_contract_dtype(
        metadata,
        peptide_tokens,
        partner_tokens,
    ):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.models import ResidueModelInput

        try:
            contract = upgrade_strategy_contract(metadata.get("pair_strategy_contract") or {})
        except PairStrategyContractError as exc:
            raise InferenceUnsupportedError(str(exc), code=exc.code, stage="bundle") from exc
        streams = contract.get("streams") or {}

        def cast_role(tokens, role):
            dtype_name = (streams.get(role) or {}).get("dtype") or ""
            if not dtype_name:
                return tokens
            dtype = getattr(torch, dtype_name.split(".")[-1], None)
            if dtype is None:
                return tokens
            if isinstance(tokens, ResidueModelInput):
                return ResidueModelInput(
                    tokens.features.to(dtype=dtype),
                    tokens.mask,
                )
            if isinstance(tokens, torch.Tensor):
                return tokens.to(dtype=dtype)
            return tokens

        return (
            cast_role(peptide_tokens, "peptide"),
            cast_role(partner_tokens, "partner"),
        )

    @staticmethod
    def _extract_pair_token_features(metadata, pairs, device, bundle=None):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.models import ResidueModelInput

        artifact_metadata = metadata
        peptides = [pair["peptide"] for pair in pairs]
        proteins = [pair["protein"] for pair in pairs]
        if metadata.get("pretrained"):
            peptide_tokens = ModelService._extract_pretrained_token_sequences(
                metadata, peptides, device
            )
            protein_tokens = ModelService._extract_pretrained_token_sequences(
                metadata, proteins, device
            )
        elif metadata.get("classic"):
            metadata, peptide_ids, protein_ids = ModelService._prepare_classic_profile_metadata(
                metadata,
                pairs,
                bundle,
            )
            peptide_tokens = ModelService._extract_classic_pair_token_sequences(
                metadata,
                peptides,
                device,
                sample_ids=peptide_ids,
                bundle=bundle,
                role="peptide",
            )
            protein_tokens = ModelService._extract_classic_pair_token_sequences(
                metadata,
                proteins,
                device,
                sample_ids=protein_ids,
                bundle=bundle,
                role="protein",
            )
        else:
            raise InferenceUnsupportedError("pair-token bundle is missing feature metadata")
        peptide_tokens, protein_tokens = ModelService._cast_pair_tokens_to_contract_dtype(
            artifact_metadata,
            peptide_tokens,
            protein_tokens,
        )
        ModelService._validate_pair_strategy_metadata(
            artifact_metadata,
            peptide_tokens,
            protein_tokens,
        )
        if peptide_tokens.shape != protein_tokens.shape:
            raise InferenceUnsupportedError(
                "peptide and protein token features have incompatible dimensions"
            )
        if isinstance(peptide_tokens, ResidueModelInput) or isinstance(
            protein_tokens,
            ResidueModelInput,
        ):
            if not isinstance(peptide_tokens, ResidueModelInput) or not isinstance(
                protein_tokens,
                ResidueModelInput,
            ):
                raise InferenceUnsupportedError(
                    "pair streams disagree on explicit residue-mask usage"
                )
            return ResidueModelInput(
                torch.cat(
                    [peptide_tokens.features, protein_tokens.features],
                    dim=1,
                ).float(),
                torch.cat(
                    [peptide_tokens.mask, protein_tokens.mask],
                    dim=1,
                ).bool(),
            )
        out = torch.cat([peptide_tokens, protein_tokens], dim=1)
        return out.float() if out.ndim > 2 else out.long()

    @staticmethod
    def _extract_pair_sequence_features(metadata, pairs, device, bundle=None):
        metadata = normalize_stream_fields(metadata)
        peptides = [pair["peptide"] for pair in pairs]
        proteins = [pair["protein"] for pair in pairs]
        if metadata.get("pretrained"):

            def extract_deduplicated(values):
                unique_values = []
                unique_index = {}
                inverse = []
                for value in values:
                    identity = str(value).strip().upper()
                    position = unique_index.get(identity)
                    if position is None:
                        position = len(unique_values)
                        unique_index[identity] = position
                        unique_values.append(value)
                    inverse.append(position)
                encoded = ModelService._extract_pretrained_token_sequences(
                    metadata,
                    unique_values,
                    device,
                )
                return encoded.index_select(
                    0,
                    torch.tensor(inverse, dtype=torch.long),
                )

            peptide_tokens = extract_deduplicated(peptides)
            protein_tokens = extract_deduplicated(proteins)
        elif metadata.get("classic"):
            metadata, peptide_ids, protein_ids = ModelService._prepare_classic_profile_metadata(
                metadata,
                pairs,
                bundle,
            )
            peptide_tokens = ModelService._extract_classic_pair_token_sequences(
                metadata,
                peptides,
                device,
                sample_ids=peptide_ids,
                bundle=bundle,
                role="peptide",
            )
            protein_tokens = ModelService._extract_classic_pair_token_sequences(
                metadata,
                proteins,
                device,
                sample_ids=protein_ids,
                bundle=bundle,
                role="protein",
            )
        else:
            raise InferenceUnsupportedError(
                "Single-stream bundle is missing classic or pretrained feature metadata"
            )
        strategy_contract = metadata.get("pair_strategy_contract")
        if strategy_contract is not None:
            try:
                strategy_contract = validate_single_stream_strategy_contract(strategy_contract)
                join_contract = strategy_contract["sequence_join"]
                joined = join_single_stream_batches(
                    peptide_tokens,
                    protein_tokens,
                    separator_id=(
                        int(join_contract["index_separator_id"])
                        if join_contract["input_kind"] == "index"
                        else None
                    ),
                )
                joined_values = getattr(joined, "features", joined)
                if int(joined_values.shape[1]) != int(join_contract["joined_length"]):
                    raise PairStrategyContractError(
                        "pair_strategy_contract_shape_mismatch",
                        "Single-stream reconstructed sequence length does not match the artifact contract",
                    )
                if join_contract["input_kind"] == "feature" and int(joined_values.shape[2]) != int(
                    join_contract["joined_feature_dim"]
                ):
                    raise PairStrategyContractError(
                        "pair_strategy_contract_shape_mismatch",
                        "Single-stream reconstructed feature width does not match the artifact contract",
                    )
                return joined
            except PairStrategyContractError as exc:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="input",
                ) from exc

        warnings.warn(
            "single_stream artifact is missing pair_strategy_contract v1 and is "
            "legacy_only; historical boundary-free sequence joining is preserved, "
            "but retraining is required for a verified baseline",
            RuntimeWarning,
            stacklevel=2,
        )
        from pepplatform.models import ResidueModelInput

        if peptide_tokens.shape != protein_tokens.shape:
            raise InferenceUnsupportedError(
                "peptide and protein sequence features have incompatible dimensions"
            )
        if isinstance(peptide_tokens, ResidueModelInput) or isinstance(
            protein_tokens,
            ResidueModelInput,
        ):
            if not isinstance(peptide_tokens, ResidueModelInput) or not isinstance(
                protein_tokens,
                ResidueModelInput,
            ):
                raise InferenceUnsupportedError(
                    "pair streams disagree on explicit residue-mask usage"
                )
            return ResidueModelInput(
                torch.cat(
                    [peptide_tokens.features, protein_tokens.features],
                    dim=1,
                ).float(),
                torch.cat(
                    [peptide_tokens.mask, protein_tokens.mask],
                    dim=1,
                ).bool(),
            )
        out = torch.cat([peptide_tokens, protein_tokens], dim=1)
        return out.float() if out.ndim > 2 else out.long()

    @staticmethod
    def _extract_pair_embedding_features(metadata, pairs, device, bundle=None):
        metadata = normalize_stream_fields(metadata)
        export = metadata.get("embedding_export") or {}
        encoder_path = str(export.get("encoder_path", "") or "").strip()
        if not encoder_path:
            raise InferenceUnsupportedError("embedding-export bundle is missing encoder_path")
        encoder_file = Path(encoder_path).expanduser()
        if not encoder_file.is_absolute():
            if bundle is None:
                raise InferenceUnsupportedError("embedding-export bundle is missing bundle context")
            encoder_file = Path(bundle) / encoder_file
        if not encoder_file.exists():
            raise InferenceUnsupportedError("embedding-export bundle is missing stage-1 encoder")
        pair = metadata.get("pair") or {}
        interaction_model = str(pair.get("interaction_model", "") or "").strip().lower()
        embedding_contract = export.get("architecture_contract") or metadata.get(
            "predictor_contract"
        )
        if interaction_model == "cross_attn":
            if embedding_contract is None:
                warnings.warn(
                    "unversioned Cross-Attention embedding encoder is legacy_only; "
                    "historical bidirectional behavior will be preserved",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif not _matches_pair_attention_contract(embedding_contract, "cross_attn"):
                raise InferenceUnsupportedError(
                    "unsupported Cross-Attention embedding predictor contract"
                )
        elif interaction_model == "bi_cross_attn":
            if embedding_contract is None:
                warnings.warn(
                    "unversioned Bi-Cross-Attention embedding encoder is legacy_only; "
                    "historical bidirectional behavior will be preserved",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif not _matches_pair_attention_contract(embedding_contract, "bi_cross_attn"):
                raise InferenceUnsupportedError(
                    "unsupported Bi-Cross-Attention embedding predictor contract"
                )
        pair_strategy = (
            str(
                pair.get("pair_strategy") or (metadata.get("args") or {}).get("pair_strategy") or ""
            )
            .strip()
            .lower()
        )
        heterogeneous_dual = pair_strategy == "two_stream" and bool(metadata.get("streams"))
        if pair_strategy == "single_stream":
            inputs = ModelService._extract_pair_sequence_features(
                metadata, pairs, device, bundle=bundle
            )
        elif heterogeneous_dual:
            inputs = ModelService._extract_pair_token_streams(
                metadata,
                pairs,
                device,
                bundle=bundle,
            )
        else:
            inputs = ModelService._extract_pair_token_features(
                metadata, pairs, device, bundle=bundle
            )
        encoder = ModelService._load_torch_model(encoder_file, device)
        if embedding_contract is not None and interaction_model in _PAIR_ATTENTION_CONTRACTS:
            _validate_loaded_pair_attention_model(
                encoder,
                embedding_contract,
                interaction_model,
            )
        encoder = encoder.to(device)
        encoder.eval()
        if not hasattr(encoder, "forward_with_embedding"):
            raise InferenceUnsupportedError(
                "embedding-export stage-1 encoder cannot export embeddings"
            )
        batch_size = ModelService._inference_batch_size(metadata)
        embedding_batches = []
        if heterogeneous_dual:
            peptide_inputs, partner_inputs = inputs
            peptide_host = peptide_inputs.detach().cpu()
            partner_host = partner_inputs.detach().cpu()
            peptide_explicit = hasattr(peptide_host, "batch_slice")
            partner_explicit = hasattr(partner_host, "batch_slice")
            with torch.no_grad():
                for start in range(0, peptide_host.shape[0], batch_size):
                    peptide_batch = (
                        peptide_host.batch_slice(start, start + batch_size).to(device)
                        if peptide_explicit
                        else peptide_host[start : start + batch_size].to(device)
                    )
                    partner_batch = (
                        partner_host.batch_slice(start, start + batch_size).to(device)
                        if partner_explicit
                        else partner_host[start : start + batch_size].to(device)
                    )
                    _logits, embeddings = encoder.forward_with_embedding(
                        peptide_batch,
                        partner_batch,
                    )
                    embedding_batches.append(embeddings.detach().cpu().float())
                    del peptide_batch, partner_batch, _logits, embeddings
                    if torch.device(device).type == "cuda":
                        torch.cuda.empty_cache()
        else:
            host_inputs = inputs.detach().cpu()
            explicit_mask = hasattr(host_inputs, "batch_slice")
            with torch.no_grad():
                for start in range(0, host_inputs.shape[0], batch_size):
                    batch = (
                        host_inputs.batch_slice(start, start + batch_size).to(device)
                        if explicit_mask
                        else host_inputs[start : start + batch_size].to(device)
                    )
                    _logits, embeddings = encoder.forward_with_embedding(batch)
                    embedding_batches.append(embeddings.detach().cpu().float())
                    del batch, _logits, embeddings
                    if torch.device(device).type == "cuda":
                        torch.cuda.empty_cache()
        if not embedding_batches:
            return torch.empty((0, 0), dtype=torch.float)
        return torch.cat(embedding_batches, dim=0)

    @staticmethod
    def _extract_pretrained_token_sequences(metadata, sequences, device):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.models import ResidueModelInput

        pretrained = metadata.get("pretrained") or {}
        tag = normalize_tokenizer_name(pretrained.get("tokenizer", ""))
        model_name = pretrained.get("model_name")
        max_length = int(
            pretrained.get("max_length") or metadata.get("args", {}).get("max_length") or 512
        )
        batch_size = max(1, int(metadata.get("args", {}).get("batch_size") or 1))
        if not tag or not model_name:
            raise InferenceUnsupportedError("pretrained token bundle is missing model metadata")
        if tag == "esm3":
            tokenizer, encoder, _legacy = ModelService._load_esm3_for_inference(
                pretrained, max_length, device
            )
            batches = []
            total = len(sequences)
            batch_total = (total + batch_size - 1) // batch_size if total else 0
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start : start + batch_size]
                encoded = encoder.encode_residue_batch(batch, max_length)
                batches.append(ResidueModelInput(encoded.features.float(), encoded.mask.bool()))
                _report_inference_progress(
                    "extracting_features",
                    completed=min(start + len(batch), total),
                    total=total,
                    batch_completed=(start // batch_size) + 1,
                    batch_total=batch_total,
                )
            if batches:
                return ResidueModelInput(
                    torch.cat([item.features for item in batches], dim=0),
                    torch.cat([item.mask for item in batches], dim=0),
                )
            return ResidueModelInput(
                torch.empty((0, max_length, encoder.output_dim)),
                torch.empty((0, max_length), dtype=torch.bool),
            )
        tokenizer_classes = {}
        encoder_classes = {}
        protbert_legacy = False
        if tag == "esm2":
            tokenizer, encoder, esm2_legacy = ModelService._load_esm2_for_inference(
                pretrained, max_length, device
            )
        elif tag == "protbert":
            tokenizer, encoder, protbert_legacy = ModelService._load_protbert_for_inference(
                pretrained, max_length, device
            )
        elif tag == "prott5":
            tokenizer, encoder, prott5_legacy = ModelService._load_prott5_for_inference(
                pretrained, max_length, device
            )
        elif tag == "esmc":
            tokenizer, encoder = ModelService._load_esmc_for_inference(
                pretrained, max_length, device
            )
        else:
            raise InferenceUnsupportedError(f"unsupported pretrained tokenizer: {tag}")
        if tag not in {"esm2", "protbert", "prott5", "esmc"}:
            tokenizer = tokenizer_classes[tag](model_name=model_name, max_length=max_length)
            encoder = encoder_classes[tag](model_name=model_name, device=device)
        batches = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            if tag == "esm2" and not esm2_legacy:
                inputs = tokenizer.batch_tokenize_residues(
                    batch,
                    max_length,
                    device=device,
                )
                encoded = encoder.encode_residue_batch(
                    inputs,
                    [
                        len(tokenizer.normalize_sequence(sequence)[:max_length])
                        for sequence in batch
                    ],
                    max_length,
                )
                hidden = ResidueModelInput(encoded.features, encoded.mask)
            elif tag == "protbert" and not protbert_legacy:
                inputs = tokenizer.batch_tokenize_residues(
                    batch,
                    max_length,
                    device=device,
                )
                encoded = encoder.encode_residue_batch(
                    inputs,
                    [tokenizer.residue_length(sequence, max_length) for sequence in batch],
                    max_length,
                )
                hidden = ResidueModelInput(encoded.features, encoded.mask)
            elif tag == "prott5" and not prott5_legacy:
                inputs = tokenizer.batch_tokenize_residues(
                    batch,
                    max_length,
                    device=device,
                )
                encoded = encoder.encode_residue_batch(
                    inputs,
                    [tokenizer.residue_length(sequence, max_length) for sequence in batch],
                    max_length,
                )
                hidden = ResidueModelInput(encoded.features, encoded.mask)
            elif tag == "prott5" and prott5_legacy:
                inputs = tokenizer.batch_tokenize(batch, device=device)
                hidden = ModelService._forward_token_hidden(encoder, inputs)
                hidden = ModelService._pad_or_trim_token_hidden(hidden, max_length)
            elif tag == "esmc":
                inputs = tokenizer.batch_tokenize_residues(
                    batch,
                    max_length,
                    device=device,
                )
                encoded = encoder.encode_residue_batch(
                    inputs,
                    [tokenizer.residue_length(sequence, max_length) for sequence in batch],
                    max_length,
                )
                hidden = ResidueModelInput(encoded.features, encoded.mask)
            else:
                inputs = tokenizer.batch_tokenize(batch, device=device)
                hidden = ModelService._forward_token_hidden(encoder, inputs)
                hidden = ModelService._pad_or_trim_token_hidden(hidden, max_length)
            batches.append(hidden.detach().cpu().float())
        if batches and isinstance(batches[0], ResidueModelInput):
            return ResidueModelInput(
                torch.cat([batch.features for batch in batches], dim=0),
                torch.cat([batch.mask for batch in batches], dim=0),
            )
        return torch.cat(batches, dim=0)

    @staticmethod
    def _load_esm2_for_inference(pretrained, max_length, device):
        from pepplatform.encoders import EsmEncoder
        from pepplatform.esm2 import (
            ESM2_POOLING,
            ESM2_REPRESENTATION_CONTRACT_VERSION,
            validate_esm2_model_identity,
        )
        from pepplatform.tokenizers import EsmTokenizer

        model_name = str(pretrained.get("model_name", "") or "").strip()
        model_identity = pretrained.get("model_identity")
        current = bool(
            isinstance(model_identity, dict)
            and model_identity.get("pooling") == ESM2_POOLING
            and model_identity.get("representation_contract_version")
            == ESM2_REPRESENTATION_CONTRACT_VERSION
        )
        if not current:
            warnings.warn(
                "ESM2 artifact is missing representation contract v1 and is legacy_only; "
                "historical BOS/EOS-inclusive pooling or token semantics are preserved, "
                "but retraining is required for a current formal experiment",
                RuntimeWarning,
            )
        source = model_name
        if not Path(os.path.expanduser(model_name)).exists() and isinstance(model_identity, dict):
            source = str(model_identity.get("model_id", "") or model_name).strip()
        try:
            source = resolve_pretrained_model_source(
                PROJECT_ROOT,
                source,
                pretrained_model_local_candidates(PROJECT_ROOT, "esm2", source),
                "ESM2",
                "esm2_model",
                tokenizer="esm2",
            )
            try:
                tokenizer = EsmTokenizer(
                    model_name=source,
                    max_length=max_length,
                    legacy_total_token_length=not current,
                )
            except TypeError:
                tokenizer = EsmTokenizer(model_name=source, max_length=max_length)
            encoder = EsmEncoder(model_name=source, device=device)
            if current:
                validate_esm2_model_identity(
                    model_identity,
                    getattr(encoder, "model_identity", {}),
                )
        except InferenceUnsupportedError:
            raise
        except (ImportError, RuntimeError, ValueError, OSError, SystemExit) as exc:
            raise InferenceUnsupportedError(
                f"ESM2 runtime or checkpoint validation failed: {exc}",
                code="esm2_identity_validation_failed",
                stage="feature_extraction",
            ) from exc
        return tokenizer, encoder, not current

    @staticmethod
    def _load_protbert_for_inference(pretrained, max_length, device):
        from pepplatform.encoders import ProtBertEncoder
        from pepplatform.tokenizers import ProtBertTokenizer

        model_name = str(pretrained.get("model_name", "") or "").strip()
        model_identity = pretrained.get("model_identity")
        current = bool(
            isinstance(model_identity, dict)
            and model_identity.get("pooling") == PROTBERT_POOLING
            and model_identity.get("representation_contract_version")
            == PROTBERT_REPRESENTATION_CONTRACT_VERSION
        )
        if not current:
            warnings.warn(
                "ProtBERT artifact is missing representation contract v1 and is legacy_only; "
                "historical CLS/SEP-inclusive pooling or total-token semantics are preserved, "
                "but retraining is required for a current formal experiment",
                RuntimeWarning,
            )
        source = model_name
        if not Path(os.path.expanduser(model_name)).exists() and isinstance(model_identity, dict):
            source = str(
                model_identity.get("model_id", "") or model_name or PROTBERT_OFFICIAL_MODEL_ID
            ).strip()
        try:
            source = resolve_pretrained_model_source(
                PROJECT_ROOT,
                source,
                pretrained_model_local_candidates(PROJECT_ROOT, "protbert", source),
                "ProtBERT",
                "protbert_model",
                tokenizer="protbert",
            )
            tokenizer = ProtBertTokenizer(
                model_name=source,
                max_length=max_length,
                legacy_total_token_length=not current,
            )
            encoder = ProtBertEncoder(model_name=source, device=device)
            if current:
                validate_protbert_model_identity(
                    model_identity,
                    getattr(encoder, "model_identity", {}),
                )
        except (ImportError, RuntimeError, ValueError, OSError, SystemExit) as exc:
            raise InferenceUnsupportedError(
                f"ProtBERT runtime or checkpoint validation failed: {exc}",
                code="protbert_identity_validation_failed",
                stage="feature_extraction",
            ) from exc
        return tokenizer, encoder, not current

    @staticmethod
    def _load_prott5_for_inference(pretrained, max_length, device):
        from pepplatform.encoders import ProT5Encoder
        from pepplatform.prott5 import (
            PROTT5_OFFICIAL_MODEL_ID,
            PROTT5_POOLING,
            PROTT5_REPRESENTATION_CONTRACT_VERSION,
            validate_prott5_model_identity,
        )
        from pepplatform.tokenizers import ProT5Tokenizer

        model_name = str(pretrained.get("model_name", "") or "").strip()
        model_identity = pretrained.get("model_identity")
        current = bool(
            isinstance(model_identity, dict)
            and model_identity.get("pooling") == PROTT5_POOLING
            and model_identity.get("representation_contract_version")
            == PROTT5_REPRESENTATION_CONTRACT_VERSION
        )
        if not current:
            warnings.warn(
                "ProtT5 artifact is missing representation contract v1 and is legacy_only; "
                "historical EOS-inclusive pooling/token semantics are preserved where possible, "
                "but retraining is required for a current formal experiment",
                RuntimeWarning,
            )
        source = model_name
        if not Path(os.path.expanduser(model_name)).exists() and isinstance(model_identity, dict):
            source = str(
                model_identity.get("model_id", "") or model_name or PROTT5_OFFICIAL_MODEL_ID
            ).strip()
        try:
            source = resolve_pretrained_model_source(
                PROJECT_ROOT,
                source,
                pretrained_model_local_candidates(PROJECT_ROOT, "prott5", source),
                "ProtT5",
                "prott5_model",
                tokenizer="prott5",
            )
            tokenizer = ProT5Tokenizer(
                model_name=source,
                max_length=max_length,
                legacy_total_token_length=not current,
            )
            encoder = ProT5Encoder(model_name=source, device=device)
            if current:
                validate_prott5_model_identity(
                    model_identity,
                    getattr(encoder, "model_identity", {}),
                )
        except (ImportError, RuntimeError, ValueError, OSError, SystemExit) as exc:
            raise InferenceUnsupportedError(
                f"ProtT5 runtime or checkpoint validation failed: {exc}",
                code="prott5_identity_validation_failed",
                stage="feature_extraction",
            ) from exc
        return tokenizer, encoder, not current

    @staticmethod
    def _load_esmc_for_inference(pretrained, max_length, device):
        from pepplatform.encoders import EsmcEncoder
        from pepplatform.esmc import validate_esmc_model_identity
        from pepplatform.tokenizers import EsmcTokenizer

        model_name = str(pretrained.get("model_name", "") or "").strip()
        model_identity = pretrained.get("model_identity")
        source = model_name
        if not Path(os.path.expanduser(model_name)).exists() and isinstance(model_identity, dict):
            source = str(model_identity.get("model_id", "") or model_name).strip()
        try:
            source = resolve_pretrained_model_source(
                PROJECT_ROOT,
                source,
                pretrained_model_local_candidates(PROJECT_ROOT, "esmc", source),
                "ESM-C",
                "esmc_model",
                tokenizer="esmc",
            )
            tokenizer = EsmcTokenizer(model_name=source, max_length=max_length)
            encoder = EsmcEncoder(model_name=source, device=device)
            validate_esmc_model_identity(model_identity, encoder.model_identity)
        except (ImportError, RuntimeError, ValueError, OSError, SystemExit) as exc:
            raise InferenceUnsupportedError(
                f"ESM-C runtime or checkpoint validation failed: {exc}",
                code="esmc_identity_validation_failed",
                stage="feature_extraction",
            ) from exc
        return tokenizer, encoder

    @staticmethod
    def _load_esm3_for_inference(pretrained, max_length, device):
        from pepplatform.encoders import Esm3Encoder
        from pepplatform.esm3 import (
            ESM3_REPRESENTATION_CONTRACT_VERSION,
            ESM3_POOLING,
            official_esm3_weight_path,
            validate_esm3_model_identity,
        )
        from pepplatform.tokenizers import Esm3Tokenizer

        model_name = str(pretrained.get("model_name", "") or "").strip()
        model_identity = pretrained.get("model_identity")
        current = (
            isinstance(model_identity, dict)
            and model_identity.get("representation_contract_version")
            == ESM3_REPRESENTATION_CONTRACT_VERSION
            and model_identity.get("pooling") == ESM3_POOLING
            and model_identity.get("runtime_key") == "esm3_sm_open_v1"
        )
        if not current:
            raise InferenceUnsupportedError(
                "ESM3 artifact is missing representation contract v1 and is legacy_only; retrain required",
                code="esm3_legacy_artifact",
                stage="feature_extraction",
            )
        source = model_name
        if not Path(os.path.expanduser(model_name)).exists() and isinstance(model_identity, dict):
            recorded_source = str(model_identity.get("resolved_source", "") or "").strip()
            if recorded_source and Path(os.path.expanduser(recorded_source)).exists():
                source = recorded_source
            else:
                source = str(model_identity.get("model_id") or model_name).strip()
        try:
            source = resolve_pretrained_model_source(
                PROJECT_ROOT,
                source,
                pretrained_model_local_candidates(PROJECT_ROOT, "esm3", source),
                "ESM3",
                "esm3_model",
                tokenizer="esm3",
            )
            tokenizer = Esm3Tokenizer(model_name=source, max_length=max_length)
            source_path = Path(os.path.expanduser(source))
            weight_path = official_esm3_weight_path(source_path)
            stat_signature = None
            if weight_path is not None:
                stat = weight_path.stat()
                stat_signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            key = (
                str(source_path.resolve()) if source_path.exists() else str(source),
                str((model_identity or {}).get("weight_sha256") or ""),
                stat_signature,
                str(torch.device(device)),
            )
            load_started = time.perf_counter()

            def build_encoder():
                encoder = Esm3Encoder(model_name=source, device=device)
                validate_esm3_model_identity(model_identity, encoder.model_identity)
                return encoder

            _report_inference_progress("loading_encoder", cache_hit=False)
            encoder, cache_hit = _ESM3_RUNTIME_POOL.get_or_load(key, build_encoder)
            observed_identity = getattr(encoder, "model_identity", {})
            validate_esm3_model_identity(model_identity, observed_identity)
            timing = _INFERENCE_TIMING.get()
            if isinstance(timing, dict):
                timing["encoder_cache_hit"] = bool(cache_hit)
                timing["encoder_load_ms"] = round(
                    (time.perf_counter() - load_started) * 1000.0,
                    2,
                )
            if cache_hit:
                _report_inference_progress("encoder_ready", cache_hit=True)
            else:
                _report_inference_progress("encoder_ready", cache_hit=False)
        except InferenceUnsupportedError:
            raise
        except (ImportError, RuntimeError, ValueError, OSError, SystemExit) as exc:
            raise InferenceUnsupportedError(
                f"ESM3 runtime or checkpoint validation failed: {exc}",
                code="esm3_identity_validation_failed",
                stage="feature_extraction",
            ) from exc
        return tokenizer, encoder, False

    @staticmethod
    def _forward_token_hidden(encoder, inputs):
        model_inputs = {key: value for key, value in inputs.items() if key != "special_tokens_mask"}
        with torch.no_grad():
            try:
                outputs = encoder.model(**model_inputs, output_hidden_states=True)
            except TypeError:
                outputs = encoder.model(**model_inputs)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        else:
            raise InferenceUnsupportedError(
                "pretrained encoder does not provide token hidden states"
            )
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones(hidden.shape[:2], dtype=hidden.dtype, device=hidden.device)
        else:
            mask = mask.float()
        return hidden * mask.unsqueeze(-1)

    @staticmethod
    def _pad_or_trim_token_hidden(hidden, target_len):
        target_len = int(target_len)
        if hidden.shape[1] == target_len:
            return hidden
        if hidden.shape[1] > target_len:
            return hidden[:, :target_len, :]
        out = torch.zeros(
            (hidden.shape[0], target_len, hidden.shape[2]), dtype=hidden.dtype, device=hidden.device
        )
        out[:, : hidden.shape[1], :] = hidden
        return out

    @staticmethod
    def _load_pair_token_model(bundle, metadata, features, device):
        metadata = normalize_stream_fields(metadata)
        pair = metadata.get("pair") or {}
        interaction_model = str(pair.get("interaction_model", "") or "").strip().lower()
        if interaction_model not in {"cross_attn", "bi_cross_attn"}:
            raise InferenceUnsupportedError(
                f"unsupported token interaction model: {interaction_model}"
            )
        from scripts.train_pair_pipeline import (
            LegacyPairCrossAttentionClassifier,
            LegacyPairBiCrossAttentionClassifier,
            PairBiCrossAttentionClassifier,
            PairCrossAttentionClassifier,
        )

        predictor_contract = metadata.get("predictor_contract")
        if interaction_model == "cross_attn":
            if predictor_contract is None:
                warnings.warn(
                    "unversioned Cross-Attention bundle is legacy_only; "
                    "loading historical shared-parameter bidirectional behavior",
                    RuntimeWarning,
                    stacklevel=2,
                )
                model_cls = LegacyPairCrossAttentionClassifier
            elif _matches_pair_attention_contract(predictor_contract, "cross_attn"):
                model_cls = PairCrossAttentionClassifier
            else:
                raise InferenceUnsupportedError("unsupported Cross-Attention predictor contract")
        else:
            if predictor_contract is None:
                warnings.warn(
                    "unversioned Bi-Cross-Attention bundle is legacy_only; "
                    "loading historical bidirectional parameter layout",
                    RuntimeWarning,
                    stacklevel=2,
                )
                model_cls = LegacyPairBiCrossAttentionClassifier
            elif _matches_pair_attention_contract(predictor_contract, "bi_cross_attn"):
                model_cls = PairBiCrossAttentionClassifier
            else:
                raise InferenceUnsupportedError("unsupported Bi-Cross-Attention predictor contract")
        seq_len = int(
            pair.get("seq_len")
            or metadata.get("pretrained", {}).get("max_length")
            or features.shape[1] // 2
        )
        encoder_name = (
            str(pair.get("encoder") or metadata.get("classic", {}).get("encoder") or "ppi_fusion")
            .strip()
            .lower()
        )
        if features.ndim == 2:
            encoder_name = "index"
        token_dim = int(
            pair.get("token_dim")
            or metadata.get("pretrained", {}).get("token_dim")
            or (features.shape[2] if features.ndim > 2 else 0)
        )
        vocab_size = int(
            pair.get("vocab_size") or metadata.get("classic", {}).get("vocab_size") or 0
        )
        if encoder_name == "index" and vocab_size <= 0 and features.numel() > 0:
            vocab_size = int(features.max().item()) + 1
        hidden_dim = int(pair.get("hidden_dim") or 128)
        n_heads = int(pair.get("n_heads") or 4)
        raw_dropout = pair.get("dropout", metadata.get("args", {}).get("dropout", 0.1))
        dropout = float(0.1 if raw_dropout in [None, ""] else raw_dropout)
        kwargs = {
            "seq_len": seq_len,
            "encoder_name": encoder_name,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "n_heads": n_heads,
        }
        head = metadata.get("prediction_head") or {}
        if isinstance(head, dict) and "hidden_dims" in head:
            kwargs["head_hidden_dims"] = list(head.get("hidden_dims") or [])
            if kwargs["head_hidden_dims"]:
                kwargs["head_activation"] = str(head.get("activation") or "relu")
                kwargs["head_dropout"] = float(head.get("dropout", dropout))
        if encoder_name == "index":
            if vocab_size <= 0:
                raise InferenceUnsupportedError("classic index token bundle is missing vocab_size")
            kwargs["vocab_size"] = vocab_size
        else:
            if token_dim <= 0:
                raise InferenceUnsupportedError("token bundle is missing token_dim")
            kwargs["in_dim"] = token_dim
        model = model_cls(**kwargs)
        payload = torch.load(bundle / "model.pt", map_location=device, weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            if isinstance(model, LegacyPairBiCrossAttentionClassifier):
                incompatible = model.load_state_dict(payload["state_dict"], strict=False)
                allowed_missing = {
                    "attn.in_proj_weight",
                    "attn.in_proj_bias",
                    "attn.out_proj.weight",
                    "attn.out_proj.bias",
                }
                if (
                    set(incompatible.missing_keys).difference(allowed_missing)
                    or incompatible.unexpected_keys
                ):
                    raise InferenceUnsupportedError(
                        "legacy Bi-Cross-Attention state_dict is incomplete or incompatible"
                    )
            else:
                model.load_state_dict(payload["state_dict"])
        else:
            model = payload
        if predictor_contract is not None:
            _validate_loaded_pair_attention_model(
                model,
                predictor_contract,
                interaction_model,
            )
        return model.to(device)

    @staticmethod
    def _load_heterogeneous_pair_token_model(bundle, metadata, device):
        metadata = normalize_stream_fields(metadata)
        pair = metadata.get("pair") or {}
        interaction_model = (
            str(pair.get("interaction_model", "cross_attn") or "cross_attn").strip().lower()
        )
        if interaction_model not in {"cross_attn", "bi_cross_attn"}:
            raise InferenceUnsupportedError(
                f"unsupported heterogeneous token interaction model: {interaction_model}"
            )
        from scripts.train_pair_pipeline import (
            HeterogeneousPairBiCrossAttentionClassifier,
            HeterogeneousPairCrossAttentionClassifier,
            LegacyHeterogeneousPairCrossAttentionClassifier,
        )

        predictor_contract = metadata.get("predictor_contract")
        if interaction_model == "cross_attn":
            if predictor_contract is None:
                warnings.warn(
                    "unversioned heterogeneous Cross-Attention bundle is legacy_only; "
                    "loading historical shared-parameter bidirectional behavior",
                    RuntimeWarning,
                    stacklevel=2,
                )
                model_cls = LegacyHeterogeneousPairCrossAttentionClassifier
            elif _matches_pair_attention_contract(predictor_contract, "cross_attn"):
                model_cls = HeterogeneousPairCrossAttentionClassifier
            else:
                raise InferenceUnsupportedError(
                    "unsupported heterogeneous Cross-Attention predictor contract"
                )
        else:
            if predictor_contract is None:
                warnings.warn(
                    "unversioned heterogeneous Bi-Cross-Attention bundle is legacy_only; "
                    "loading historical bidirectional behavior",
                    RuntimeWarning,
                    stacklevel=2,
                )
                model_cls = HeterogeneousPairBiCrossAttentionClassifier
            elif _matches_pair_attention_contract(predictor_contract, "bi_cross_attn"):
                model_cls = HeterogeneousPairBiCrossAttentionClassifier
            else:
                raise InferenceUnsupportedError(
                    "unsupported heterogeneous Bi-Cross-Attention predictor contract"
                )
        kwargs = dict(pair.get("model_config") or {})
        if not kwargs:
            raise InferenceUnsupportedError(
                "heterogeneous pair-token bundle is missing model_config"
            )
        model = model_cls(**kwargs)
        payload = torch.load(bundle / "model.pt", map_location=device, weights_only=False)
        if isinstance(payload, dict) and "state_dict" in payload:
            model.load_state_dict(payload["state_dict"])
        else:
            model = payload
        if predictor_contract is not None:
            _validate_loaded_pair_attention_model(
                model,
                predictor_contract,
                interaction_model,
            )
        return model.to(device)

    @staticmethod
    def _transform_graph_pair_embeddings(bundle, metadata, peptide_embeddings, protein_embeddings):
        metadata = normalize_stream_fields(metadata)
        preprocess_path = metadata.get("preprocess_path")
        if not preprocess_path:
            return peptide_embeddings, protein_embeddings
        payload = joblib.load(bundle / preprocess_path)

        def transform(features, scaler):
            if scaler is None:
                return features
            values = scaler.transform(features.cpu().numpy())
            return torch.from_numpy(values).float()

        return (
            transform(peptide_embeddings, payload.get("peptide_scaler")),
            transform(protein_embeddings, payload.get("protein_scaler")),
        )

    @staticmethod
    def _load_pair_graph_reference(bundle):
        path = bundle / "graph_reference.pt"
        if not path.exists():
            raise InferenceUnsupportedError("pair-graph bundle is missing reference graph features")
        return torch.load(path, map_location="cpu", weights_only=False)

    @staticmethod
    def _load_pair_graph_model(bundle, metadata, input_dim, device):
        metadata = normalize_stream_fields(metadata)
        graph = metadata.get("graph") or {}
        predictor = (
            str(graph.get("predictor", "") or metadata.get("args", {}).get("model", "gcn"))
            .strip()
            .lower()
        )
        hidden_dim = int(
            graph.get("hidden_dim") or metadata.get("args", {}).get("gnn_hidden_dim") or 256
        )
        raw_dropout = graph.get("dropout", metadata.get("args", {}).get("dropout", 0.2))
        dropout = float(0.2 if raw_dropout in [None, ""] else raw_dropout)
        edge_chunk_size = int(graph.get("edge_chunk_size") or 100_000)
        from scripts.train_pair_pipeline import (
            LegacyPairHGTClassifier,
            LegacyPairGraphTransformerClassifier,
            PairGATClassifier,
            PairGCNClassifier,
            PairGraphTransformerClassifier,
            PairHGTClassifier,
            PairSAGEClassifier,
        )

        if predictor == "gcn":
            model = PairGCNClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        elif predictor == "gat":
            predictor_contract = metadata.get("predictor_contract") or {}
            model = PairGATClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                edge_chunk_size=edge_chunk_size,
                contract_version=int(predictor_contract.get("contract_version") or 1),
            )
        elif predictor == "hgt":
            predictor_contract = metadata.get("predictor_contract") or {}
            if (
                predictor_contract.get("implementation") == "pepplatform_inductive_typed_pair_hgt"
                and int(predictor_contract.get("contract_version") or 0) >= 1
            ):
                model = PairHGTClassifier(
                    peptide_input_dim=int(graph.get("peptide_input_dim") or input_dim),
                    partner_input_dim=int(graph.get("partner_input_dim") or input_dim),
                    hidden_dim=hidden_dim,
                    heads=int(graph.get("heads") or 4),
                    dropout=dropout,
                    layers=int(graph.get("layers") or 2),
                    edge_chunk_size=int(graph.get("edge_chunk_size") or 100_000),
                    contract_version=int(predictor_contract["contract_version"]),
                )
            else:
                model = LegacyPairHGTClassifier(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    heads=4,
                    dropout=dropout,
                )
        elif predictor == "sage":
            predictor_contract = metadata.get("predictor_contract") or {}
            model = PairSAGEClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                contract_version=int(predictor_contract.get("contract_version") or 0),
            )
        elif predictor == "graph_transformer":
            predictor_contract = metadata.get("predictor_contract") or {}
            current_graph_transformer = (
                predictor_contract.get("implementation")
                == "pepplatform_inductive_pair_sample_graph_graph_transformer"
                and int(predictor_contract.get("contract_version") or 0) >= 1
            )
            if current_graph_transformer:
                model = PairGraphTransformerClassifier(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    heads=int(graph.get("heads") or predictor_contract.get("heads") or 4),
                    dropout=dropout,
                    layers=int(graph.get("layers") or predictor_contract.get("layers") or 2),
                    contract_version=int(predictor_contract["contract_version"]),
                )
            else:
                warnings.warn(
                    "unversioned Graph Transformer bundle is legacy_only; "
                    "loading historical clipped-attention and multigraph behavior",
                    RuntimeWarning,
                    stacklevel=2,
                )
                model = LegacyPairGraphTransformerClassifier(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    heads=4,
                    dropout=dropout,
                )
        else:
            raise InferenceUnsupportedError(f"unsupported pair-graph predictor: {predictor}")

        payload = torch.load(bundle / "model.pt", map_location=device, weights_only=False)
        state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
        if state_dict is None:
            raise InferenceUnsupportedError("pair-graph bundle is missing model state_dict")
        model.load_state_dict(state_dict)
        return model.to(device), payload, predictor

    @staticmethod
    def _load_pair_graph_gate(payload, feature_dim, device, contract=None):
        state_dict = payload.get("gate_state_dict") if isinstance(payload, dict) else None
        if state_dict is None:
            raise InferenceUnsupportedError("pair-graph gated bundle is missing gate state_dict")
        if contract is not None:
            try:
                dimensions = contract["input_dimensions"]
                alignment_dim = int(contract["alignment_dimension"])
                hidden_dim = int(contract["gate"]["hidden_dimension"])
                dropout = float(contract["gate"]["dropout"])
                gate = AlignedGatedFusion(
                    int(dimensions["peptide"]),
                    int(dimensions["partner"]),
                    alignment_dim=alignment_dim
                    if int(dimensions["peptide"]) != int(dimensions["partner"])
                    else None,
                    gate_hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                validate_aligned_gated_contract(contract, gate)
                if any(str(key).startswith("gate.") for key in state_dict):
                    gate.load_state_dict(state_dict)
                else:
                    gate.gate.load_state_dict(state_dict)
                return gate.to(device)
            except (KeyError, TypeError, ValueError, PairFusionError) as exc:
                raise InferenceUnsupportedError(
                    f"invalid aligned_gated graph contract: {exc}"
                ) from exc
        gate = AlignedGatedFusion(
            feature_dim,
            feature_dim,
            gate_hidden_dim=feature_dim,
            dropout=0.0,
        )
        if any(str(key).startswith("gate.") for key in state_dict):
            gate.load_state_dict(state_dict)
        else:
            gate.gate.load_state_dict(state_dict)
        return gate.to(device)

    @staticmethod
    def _fuse_pair_aligned_gated(gate, peptide_embeddings, protein_embeddings):
        return gate(peptide_embeddings, protein_embeddings)

    @staticmethod
    def _pair_graph_logits(
        model,
        predictor,
        reference_features,
        query_features,
        reference_basis,
        query_basis,
        graph_config,
        device,
    ):
        from scripts.train_pair_pipeline import build_inductive_graph_structure

        if predictor == "gcn":
            graph_kind = "gcn"
        elif (
            predictor == "gat"
            and int(getattr(model, "gat_predictor_contract_version", 0) or 0) >= 2
        ):
            graph_kind = "gat"
        elif (
            predictor == "sage"
            and int(getattr(model, "sage_predictor_contract_version", 0) or 0) >= 1
        ):
            graph_kind = "sage"
        elif (
            predictor == "graph_transformer"
            and int(getattr(model, "graph_transformer_predictor_contract_version", 0) or 0) >= 1
        ):
            graph_kind = "graph_transformer"
        else:
            graph_kind = "message"
        structure = build_inductive_graph_structure(
            graph_config,
            reference_basis,
            query_basis,
            device,
            graph_kind,
        )
        features = torch.cat([reference_features, query_features], dim=0)
        return model(features, structure)

    @staticmethod
    def _predict_pair_graph(metadata, pairs, device, bundle):
        metadata = normalize_stream_fields(metadata)
        peptide_embeddings, protein_embeddings = ModelService._extract_pair_embeddings(
            metadata, pairs, device, bundle=bundle
        )
        peptide_embeddings, protein_embeddings = ModelService._transform_graph_pair_embeddings(
            bundle,
            metadata,
            peptide_embeddings,
            protein_embeddings,
        )
        reference = ModelService._load_pair_graph_reference(bundle)
        graph = metadata.get("graph") or {}
        graph_args = dict(metadata.get("args") or {})
        graph_args.update(
            {
                "gnn_k": int(graph.get("gnn_k") or graph_args.get("gnn_k") or 8),
                "gnn_exact_knn_max_nodes": int(
                    graph.get("exact_knn_max_nodes")
                    or graph_args.get("gnn_exact_knn_max_nodes")
                    or 25000
                ),
                "gnn_rp_projections": int(
                    graph.get("rp_projections") or graph_args.get("gnn_rp_projections") or 6
                ),
                "gnn_rp_window": int(
                    graph.get("rp_window") or graph_args.get("gnn_rp_window") or 0
                ),
                "seed": int(graph_args.get("seed") or 42),
            }
        )
        graph_config = Namespace(**graph_args)
        fusion_mode = str((metadata.get("pair") or {}).get("fusion_mode", "") or "").strip().lower()
        legacy_gated = fusion_mode in {"gated_trainable", "gated(trainable)"} or bool(
            graph.get("gated_trainable")
        )
        gated = bool(graph.get("aligned_gated")) or legacy_gated
        scores = []

        predictor_name = (
            str(graph.get("predictor", "") or metadata.get("args", {}).get("model", "gcn"))
            .strip()
            .lower()
        )
        predictor_contract = metadata.get("predictor_contract") or {}
        current_hgt = (
            predictor_name == "hgt"
            and predictor_contract.get("implementation") == "pepplatform_inductive_typed_pair_hgt"
            and int(predictor_contract.get("contract_version") or 0) >= 1
        )
        if current_hgt:
            if gated:
                raise InferenceUnsupportedError(
                    "typed HGT artifacts do not support aligned_gated fusion"
                )
            pair_metadata = metadata.get("pair") or {}
            input_adapter = str(pair_metadata.get("input_adapter", "") or "")
            legacy_fusion_mode = str(pair_metadata.get("fusion_mode", "") or "")
            if input_adapter == "typed_role_projection" and not legacy_fusion_mode:
                pass
            elif legacy_fusion_mode in {"concat", "concat_all"}:
                warnings.warn(
                    f"typed HGT ignores legacy fusion_mode={legacy_fusion_mode}; "
                    "the artifact keeps separate role streams",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                raise InferenceUnsupportedError(
                    "typed HGT artifact is missing its typed role input adapter"
                )
            ref_peptide = reference.get("reference_peptide_features")
            ref_partner = reference.get("reference_protein_features")
            if ref_peptide is None or ref_partner is None:
                raise InferenceUnsupportedError(
                    "typed HGT bundle is missing role-specific reference features"
                )
            input_dim = int(graph.get("input_dim") or 0)
            model, _payload, predictor = ModelService._load_pair_graph_model(
                bundle,
                metadata,
                input_dim,
                device,
            )
            if predictor != "hgt" or not hasattr(model, "hgt_predictor_contract_version"):
                raise InferenceUnsupportedError("typed HGT bundle selected an incompatible model")
            from scripts.train_pair_pipeline import build_inductive_hgt_graph

            model.eval()
            ref_peptide = ref_peptide.to(device).float()
            ref_partner = ref_partner.to(device).float()
            batch_size = ModelService._graph_inference_batch_size(metadata, len(pairs))
            for start in range(0, len(pairs), batch_size):
                query_peptide = peptide_embeddings[start : start + batch_size].to(device).float()
                query_partner = protein_embeddings[start : start + batch_size].to(device).float()
                structure = build_inductive_hgt_graph(
                    graph_config,
                    ref_peptide,
                    ref_partner,
                    query_peptide,
                    query_partner,
                    device=device,
                )
                with torch.no_grad():
                    logits = model(
                        torch.cat([ref_peptide, query_peptide], dim=0),
                        torch.cat([ref_partner, query_partner], dim=0),
                        structure,
                    )[structure.reference_pairs :]
                    scores.extend(
                        float(score)
                        for score in torch.sigmoid(logits).view(-1).detach().cpu().tolist()
                    )
            return np.asarray(scores, dtype=np.float32)[:, None]

        if gated:
            ref_pep = reference.get("reference_peptide_features")
            ref_pro = reference.get("reference_protein_features")
            if ref_pep is None or ref_pro is None:
                raise InferenceUnsupportedError(
                    "pair-graph gated bundle is missing reference peptide/protein features"
                )
            input_dim = int(graph.get("input_dim") or ref_pep.shape[1] * 3)
            model, payload, predictor = ModelService._load_pair_graph_model(
                bundle, metadata, input_dim, device
            )
            contract = metadata.get("aligned_gated_contract")
            if contract is None:
                warnings.warn(
                    "unversioned trainable-gated graph bundle is legacy_only; "
                    "loading its historical gate structure",
                    RuntimeWarning,
                    stacklevel=2,
                )
            gate = ModelService._load_pair_graph_gate(
                payload,
                int(ref_pep.shape[1]),
                device,
                contract=contract,
            )
            model.eval()
            gate.eval()
            ref_pep = ref_pep.to(device).float()
            ref_pro = ref_pro.to(device).float()
            for index in range(len(pairs)):
                query_pep = peptide_embeddings[index : index + 1].to(device).float()
                query_pro = protein_embeddings[index : index + 1].to(device).float()
                reference_basis = torch.cat(
                    [ref_pep, ref_pro, (ref_pep - ref_pro).abs(), ref_pep * ref_pro], dim=1
                )
                query_basis = torch.cat(
                    [query_pep, query_pro, (query_pep - query_pro).abs(), query_pep * query_pro],
                    dim=1,
                )
                reference_features = ModelService._fuse_pair_aligned_gated(gate, ref_pep, ref_pro)
                query_features = ModelService._fuse_pair_aligned_gated(gate, query_pep, query_pro)
                with torch.no_grad():
                    logits = ModelService._pair_graph_logits(
                        model,
                        predictor,
                        reference_features,
                        query_features,
                        reference_basis,
                        query_basis,
                        graph_config,
                        device,
                    )
                    scores.append(float(torch.sigmoid(logits[-1]).view(-1)[0].detach().cpu()))
        else:
            ref_features = reference.get("reference_features")
            if ref_features is None:
                raise InferenceUnsupportedError(
                    "pair-graph bundle is missing reference node features"
                )
            query_features = ModelService._fuse_pair_artifact_embeddings(
                metadata,
                peptide_embeddings,
                protein_embeddings,
            ).float()
            input_dim = int(graph.get("input_dim") or ref_features.shape[1])
            model, _payload, predictor = ModelService._load_pair_graph_model(
                bundle, metadata, input_dim, device
            )
            model.eval()
            ref_features = ref_features.detach().cpu().float()
            query_features = query_features.detach().cpu().float()
            batch_size = ModelService._graph_inference_batch_size(metadata, len(pairs))
            for start in range(0, len(pairs), batch_size):
                query_batch = query_features[start : start + batch_size]
                reference_model_x = ref_features.to(device).float()
                query_model_x = query_batch.to(device).float()
                with torch.no_grad():
                    logits = ModelService._pair_graph_logits(
                        model,
                        predictor,
                        reference_model_x,
                        query_model_x,
                        ref_features,
                        query_batch,
                        graph_config,
                        device,
                    )
                    batch_scores = (
                        torch.sigmoid(logits[-query_batch.shape[0] :])
                        .view(-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    scores.extend(float(score) for score in batch_scores)
                del reference_model_x, query_model_x, query_batch, logits
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
        return np.asarray(scores, dtype=np.float32)[:, None]

    @staticmethod
    def _fuse_pair_embeddings(peptide_embeddings, protein_embeddings, mode="concat_all"):
        try:
            mode = normalize_pair_fusion_mode(mode)
            return fuse_audited_pair_embeddings(
                peptide_embeddings,
                protein_embeddings,
                mode,
            )
        except PairFusionError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="input",
            ) from exc

    @staticmethod
    def _fuse_pair_artifact_embeddings(
        metadata,
        peptide_embeddings,
        protein_embeddings,
    ):
        metadata = normalize_stream_fields(metadata)
        mode = (metadata.get("pair") or {}).get("fusion_mode", "concat_all")
        fused = ModelService._fuse_pair_embeddings(
            peptide_embeddings,
            protein_embeddings,
            mode,
        )
        try:
            canonical = normalize_pair_fusion_mode(mode)
            if canonical not in AUDITED_PAIR_FUSION_MODES:
                return fused
            contract = metadata.get("pair_fusion_contract")
            if contract is None:
                warnings.warn(
                    f"{canonical} artifact is missing pair_fusion_contract v1 and is "
                    "legacy_only; historical numerical inference is preserved, but "
                    "retraining is required for a verified baseline",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return fused
            validate_pair_fusion_contract(
                contract,
                mode=canonical,
                peptide_embeddings=peptide_embeddings,
                partner_embeddings=protein_embeddings,
                fused_embeddings=fused,
            )
        except PairFusionError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="artifact",
            ) from exc
        return fused

    @staticmethod
    def _transform_pretrained_tensor_features(bundle, metadata, features):
        metadata = normalize_stream_fields(metadata)
        preprocess_path = metadata.get("preprocess_path")
        if not preprocess_path:
            return features
        payload = joblib.load(bundle / preprocess_path)
        scaler = payload.get("dl_scaler")
        if scaler is None:
            return features
        original_shape = features.shape
        features_2d = features.reshape(features.shape[0], -1).cpu().numpy()
        transformed = scaler.transform(features_2d)
        return torch.from_numpy(transformed).float().view(*original_shape)

    @staticmethod
    def _transform_pretrained_residue_features(bundle, metadata, features):
        metadata = normalize_stream_fields(metadata)
        from pepplatform.models import ResidueModelInput

        values = features.features.detach().cpu().float().clone()
        mask = features.mask.detach().cpu().bool()
        preprocess_path = metadata.get("preprocess_path")
        scaler = None
        if preprocess_path:
            scaler = joblib.load(Path(bundle) / preprocess_path).get("dl_scaler")
        if scaler is not None and bool(mask.any()):
            transformed = scaler.transform(values[mask].numpy())
            values[mask] = torch.from_numpy(transformed).float()
        values[~mask] = 0.0
        return ResidueModelInput(values, mask)

    @staticmethod
    def _transform_classical_features(payload, features):
        task_reducer = payload.get("task_reducer")
        if task_reducer:
            kind = task_reducer["kind"]
            value = task_reducer["value"]
            if kind == "labelled_lda":
                warnings.warn(
                    "Labelled LDA feature transformation is removed and legacy_only; "
                    "historical inference is preserved, but retraining with another "
                    "transform is required for current results",
                    RuntimeWarning,
                    stacklevel=2,
                )
                features = np.maximum(features, 0) @ value.T
            elif kind == "nonnegative_sklearn":
                features = value.transform(np.maximum(features, 0))
            else:
                features = value.transform(features)
        scaler = payload.get("ml_scaler")
        if scaler is not None:
            features = scaler.transform(features)
        reducer = payload.get("ml_reducer")
        if reducer is not None:
            pca_contract = payload.get("pca_contract")
            kpca_contract = payload.get("kpca_contract")
            try:
                if isinstance(pca_contract, dict):
                    features = transform_pca_held_out(
                        reducer,
                        features,
                        split="inference",
                        input_feature_count=int(pca_contract["input_feature_count"]),
                        output_dimension=int(pca_contract["output_dimension"]),
                        dense_memory_limit_gib=float(pca_contract["dense_memory_limit_gib"]),
                    )
                elif isinstance(kpca_contract, dict):
                    features = transform_kpca_held_out(
                        reducer,
                        features,
                        split="inference",
                        contract=kpca_contract,
                    )
                else:
                    features = reducer.transform(features)
            except (PCAContractError, KPCAContractError) as exc:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="preprocessing",
                    technical_detail=json.dumps(exc.details, sort_keys=True),
                ) from exc
        return features

    @staticmethod
    def _metadata_claims_lsa(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        contract = metadata.get("lsa_contract")
        return str(args.get("reduction") or "none").strip().lower() == "lsa" or isinstance(
            contract, dict
        )

    @staticmethod
    def _validate_lsa_artifact(metadata, bundle):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_lsa(metadata):
            return False
        contract = metadata.get("lsa_contract")
        if not isinstance(contract, dict):
            warnings.warn(
                "LSA artifact is missing contract v1 and is legacy_only; historical numerical inference is preserved, but retraining is required for a verified baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        task_type = normalize_task_type(metadata.get("task_type") or "")
        if normalize_task_type(task_type) in {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}:
            raise InferenceUnsupportedError(
                "current LSA contracts are not valid for Pair artifacts",
                code="lsa_pair_artifact_unsupported",
                stage="bundle",
                hint="Use the historical artifact only as legacy output, or retrain with ml_reduction=tsvd.",
            )
        try:
            validate_lsa_contract(contract, require_current_sklearn=True)
        except LSAContractError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="bundle",
                hint="Retrain and re-export LSA with the current PepPlatform contract.",
                technical_detail=json.dumps(exc.details, sort_keys=True),
            ) from exc
        descriptor = metadata.get("model_artifact")
        root = Path(bundle).resolve()
        relative = descriptor.get("path") if isinstance(descriptor, dict) else None
        path = (root / str(relative or "")).resolve()
        if (
            not isinstance(descriptor, dict)
            or relative != "model.joblib"
            or Path(str(relative or "")).is_absolute()
            or not path.is_relative_to(root)
            or not path.is_file()
            or descriptor.get("publication") != "atomic_replace"
            or descriptor.get("size_bytes") != path.stat().st_size
            or descriptor.get("sha256") != _sha256_file(path)
        ):
            raise InferenceUnsupportedError(
                "LSA model.joblib failed artifact integrity validation",
                code="lsa_model_integrity_failed",
                stage="bundle",
                hint="Restore the original artifact or retrain and re-export the model.",
            )
        return True

    @staticmethod
    def _metadata_claims_tsvd(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        contract = metadata.get("tsvd_contract")
        return str(args.get("ml_reduction") or "none").strip().lower() == "tsvd" or isinstance(
            contract, dict
        )

    @staticmethod
    def _validate_tsvd_artifact(metadata, bundle):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_tsvd(metadata):
            return False
        contract = metadata.get("tsvd_contract")
        if not isinstance(contract, dict):
            warnings.warn(
                "TSVD artifact is missing contract v1 and is legacy_only; historical numerical inference is preserved, but retraining is required for a verified baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        try:
            validate_tsvd_contract(contract, require_current_sklearn=True)
        except TSVDContractError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="bundle",
                hint="Retrain and re-export TSVD with the current PepPlatform contract.",
                technical_detail=json.dumps(exc.details, sort_keys=True),
            ) from exc
        descriptor = metadata.get("model_artifact")
        root = Path(bundle).resolve()
        relative = descriptor.get("path") if isinstance(descriptor, dict) else None
        path = (root / str(relative or "")).resolve()
        if (
            not isinstance(descriptor, dict)
            or relative != "model.joblib"
            or Path(str(relative or "")).is_absolute()
            or not path.is_relative_to(root)
            or not path.is_file()
            or descriptor.get("publication") != "atomic_replace"
            or descriptor.get("size_bytes") != path.stat().st_size
            or descriptor.get("sha256") != _sha256_file(path)
        ):
            raise InferenceUnsupportedError(
                "TSVD model.joblib failed artifact integrity validation",
                code="tsvd_model_integrity_failed",
                stage="bundle",
                hint="Restore the original artifact or retrain and re-export the model.",
            )
        return True

    @staticmethod
    def _metadata_claims_pca(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        contract = metadata.get("pca_contract")
        return str(args.get("ml_reduction") or "none").strip().lower() == "pca" or isinstance(
            contract, dict
        )

    @staticmethod
    def _validate_pca_artifact(metadata, bundle):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_pca(metadata):
            return False
        contract = metadata.get("pca_contract")
        if not isinstance(contract, dict):
            warnings.warn(
                "PCA artifact is missing contract v1 and is legacy_only; historical numerical inference is preserved, but retraining is required for a verified baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        try:
            validate_pca_contract(contract, require_current_sklearn=True)
        except PCAContractError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="bundle",
                hint="Retrain and re-export PCA with the current PepPlatform contract.",
                technical_detail=json.dumps(exc.details, sort_keys=True),
            ) from exc
        ModelService._validate_reducer_model_artifact(
            metadata,
            bundle,
            algorithm="PCA",
            error_code="pca_model_integrity_failed",
        )
        return True

    @staticmethod
    def _metadata_claims_kpca(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        contract = metadata.get("kpca_contract")
        return str(args.get("ml_reduction") or "none").strip().lower() == "kpca" or isinstance(
            contract, dict
        )

    @staticmethod
    def _validate_kpca_artifact(metadata, bundle):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_kpca(metadata):
            return False
        contract = metadata.get("kpca_contract")
        if not isinstance(contract, dict):
            warnings.warn(
                "KPCA artifact is missing contract v1 and is legacy_only; historical numerical inference is preserved, including historical sigmoid kernels, but retraining with a supported PSD kernel is required for a verified baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        try:
            validate_kpca_contract(contract, require_current_sklearn=True)
        except KPCAContractError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="bundle",
                hint="Retrain and re-export KPCA with the current PepPlatform contract.",
                technical_detail=json.dumps(exc.details, sort_keys=True),
            ) from exc
        ModelService._validate_reducer_model_artifact(
            metadata,
            bundle,
            algorithm="KPCA",
            error_code="kpca_model_integrity_failed",
        )
        return True

    @staticmethod
    def _validate_reducer_model_artifact(metadata, bundle, *, algorithm, error_code):
        metadata = normalize_stream_fields(metadata)
        descriptor = metadata.get("model_artifact")
        root = Path(bundle).resolve()
        relative = descriptor.get("path") if isinstance(descriptor, dict) else None
        path = (root / str(relative or "")).resolve()
        if (
            not isinstance(descriptor, dict)
            or relative != "model.joblib"
            or Path(str(relative or "")).is_absolute()
            or not path.is_relative_to(root)
            or not path.is_file()
            or descriptor.get("publication") != "atomic_replace"
            or descriptor.get("size_bytes") != path.stat().st_size
            or descriptor.get("sha256") != _sha256_file(path)
        ):
            raise InferenceUnsupportedError(
                f"{algorithm} model.joblib failed artifact integrity validation",
                code=error_code,
                stage="bundle",
                hint="Restore the original artifact or retrain and re-export the model.",
            )

    @staticmethod
    def _metadata_claims_lda(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        contract = metadata.get("lda_contract")
        return str(args.get("reduction") or "none").strip().lower() == "lda" or isinstance(
            contract, dict
        )

    @staticmethod
    def _validate_lda_artifact(metadata, bundle):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_lda(metadata):
            return False
        contract = metadata.get("lda_contract")
        if not isinstance(contract, dict):
            warnings.warn(
                "LDA artifact is missing contract v1 and is legacy_only; historical numerical inference is preserved, but retraining is required for a verified baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        task_type = normalize_task_type(metadata.get("task_type") or "")
        is_pair = normalize_task_type(task_type) in {
            "pep_protein_interaction",
            "pep_hlaii",
            "pep_hlai",
        }
        if is_pair and not is_pair_bow_route(contract.get("route")):
            raise InferenceUnsupportedError(
                "Pair LDA artifact is missing the strict Two-stream BoW route contract",
                code="lda_pair_artifact_unsupported",
                stage="bundle",
                hint="Use the historical artifact only as legacy output, or retrain with classic BoW streams and fusion_mode=concat.",
            )
        upstream_contract = (
            metadata.get("pair_bow_joint_contract") if is_pair else metadata.get("classic")
        )
        try:
            validate_lda_contract(
                contract,
                upstream_feature_contract=upstream_contract,
                require_current_sklearn=True,
            )
        except LDAContractError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="bundle",
                hint="Retrain and re-export LDA with the current PepPlatform contract.",
                technical_detail=json.dumps(exc.details, sort_keys=True),
            ) from exc
        descriptor = metadata.get("model_artifact")
        root = Path(bundle).resolve()
        relative = descriptor.get("path") if isinstance(descriptor, dict) else None
        path = (root / str(relative or "")).resolve()
        if (
            not isinstance(descriptor, dict)
            or relative != "model.joblib"
            or Path(str(relative or "")).is_absolute()
            or not path.is_relative_to(root)
            or not path.is_file()
            or descriptor.get("publication") != "atomic_replace"
            or descriptor.get("size_bytes") != path.stat().st_size
            or descriptor.get("sha256") != _sha256_file(path)
        ):
            raise InferenceUnsupportedError(
                "LDA model.joblib failed artifact integrity validation",
                code="lda_model_integrity_failed",
                stage="bundle",
                hint="Restore the original artifact or retrain and re-export the model.",
            )
        return True

    @staticmethod
    def _metadata_claims_plsa(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        contract = metadata.get("plsa_contract")
        return str(args.get("reduction") or "none").strip().lower() == "plsa" or isinstance(
            contract, dict
        )

    @staticmethod
    def _validate_plsa_artifact(metadata, bundle):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_plsa(metadata):
            return False
        args = metadata.get("args") or {}
        contract = metadata.get("plsa_contract")
        if not isinstance(contract, dict):
            if args.get("plsa_contract_version") in {1, PLSA_CONTRACT_VERSION}:
                raise InferenceUnsupportedError(
                    f"current PLSA artifact is missing contract v{PLSA_CONTRACT_VERSION}",
                    code="plsa_contract_missing",
                    stage="bundle",
                    hint="Retrain and re-export PLSA with the complete contract.",
                )
            warnings.warn(
                "Historical reduction=plsa used clipped KL-NMF rather than verified PLSA and is legacy_only; numerical inference is preserved, but retraining is required for a current baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        task_type = normalize_task_type(metadata.get("task_type") or "")
        is_pair = normalize_task_type(task_type) in {
            "pep_protein_interaction",
            "pep_hlaii",
            "pep_hlai",
        }
        if is_pair and not is_pair_bow_route(contract.get("route")):
            raise InferenceUnsupportedError(
                "Pair PLSA artifact is missing the strict Two-stream BoW route contract",
                code="plsa_pair_artifact_unsupported",
                stage="bundle",
                hint="Use the historical artifact only as legacy output, or retrain with classic BoW streams and fusion_mode=concat.",
            )
        try:
            validate_plsa_contract(
                contract,
                upstream_feature_contract=(
                    metadata.get("pair_bow_joint_contract") if is_pair else metadata.get("classic")
                ),
                require_current_versions=True,
            )
        except PLSAContractError as exc:
            raise InferenceUnsupportedError(
                str(exc),
                code=exc.code,
                stage="bundle",
                hint="Retrain and re-export PLSA with the current PepPlatform contract.",
                technical_detail=json.dumps(exc.details, sort_keys=True),
            ) from exc
        descriptor = metadata.get("model_artifact")
        root = Path(bundle).resolve()
        relative = descriptor.get("path") if isinstance(descriptor, dict) else None
        path = (root / str(relative or "")).resolve()
        if (
            not isinstance(descriptor, dict)
            or relative != "model.joblib"
            or Path(str(relative or "")).is_absolute()
            or not path.is_relative_to(root)
            or not path.is_file()
            or descriptor.get("publication") != "atomic_replace"
            or descriptor.get("size_bytes") != path.stat().st_size
            or descriptor.get("sha256") != _sha256_file(path)
        ):
            raise InferenceUnsupportedError(
                "PLSA model.joblib failed artifact integrity validation",
                code="plsa_model_integrity_failed",
                stage="bundle",
                hint="Restore the original artifact or retrain and re-export the model.",
            )
        return True

    @staticmethod
    def _metadata_claims_rf(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        embedding_export = metadata.get("embedding_export") or {}
        classic_export = (metadata.get("classic") or {}).get("embedding_export") or {}
        route_head = ((metadata.get("effective_route") or {}).get("predictor") or {}).get(
            "head"
        ) or {}
        candidates = [
            args.get("classifier"),
            args.get("model"),
            embedding_export.get("downstream"),
            classic_export.get("downstream"),
            route_head.get("id"),
        ]
        contract = metadata.get("predictor_contract")
        if isinstance(contract, dict):
            candidates.append(contract.get("algorithm"))
        return any(
            str(value or "").strip().lower() in {"rf", "random_forest"} for value in candidates
        )

    @staticmethod
    def _validate_rf_predictor_contract(metadata):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_rf(metadata):
            return
        contract = metadata.get("predictor_contract")
        if not isinstance(contract, dict) or contract.get("algorithm") != "random_forest":
            warnings.warn(
                "RF artifact is missing the current predictor contract and is legacy_only; "
                "historical inference is preserved, but retraining is required for a verified baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        if (
            contract.get("contract_version") != _RF_PREDICTOR_CONTRACT_VERSION
            or contract.get("implementation") != "sklearn.ensemble.RandomForestClassifier"
            or contract.get("status") != "verified"
            or contract.get("score_kind") != "probability"
            or contract.get("calibrated") is not False
        ):
            raise InferenceUnsupportedError(
                "RF artifact predictor contract is unsupported or incomplete",
                code="rf_predictor_contract_unsupported",
                stage="bundle",
                hint="Retrain and re-export the RF model with the current PepPlatform contract.",
            )
        trained_version = str(contract.get("implementation_version") or "").strip()
        if trained_version != sklearn.__version__:
            raise InferenceUnsupportedError(
                "RF artifact was trained with a different scikit-learn version",
                code="rf_runtime_version_mismatch",
                stage="bundle",
                hint=(
                    f"Use scikit-learn {trained_version} for inference or retrain with "
                    f"the current scikit-learn {sklearn.__version__}."
                ),
                technical_detail=(
                    f"artifact scikit-learn={trained_version}; "
                    f"runtime scikit-learn={sklearn.__version__}"
                ),
            )

    @staticmethod
    def _metadata_claims_extratrees(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        embedding_export = metadata.get("embedding_export") or {}
        classic_export = (metadata.get("classic") or {}).get("embedding_export") or {}
        route_head = ((metadata.get("effective_route") or {}).get("predictor") or {}).get(
            "head"
        ) or {}
        candidates = [
            args.get("classifier"),
            args.get("model"),
            embedding_export.get("downstream"),
            classic_export.get("downstream"),
            route_head.get("id"),
        ]
        contract = metadata.get("predictor_contract")
        if isinstance(contract, dict):
            candidates.append(contract.get("algorithm"))
        return any(
            str(value or "").strip().lower()
            in {"extratrees", "extra_trees", "extremely_randomized_trees"}
            for value in candidates
        )

    @staticmethod
    def _validate_extratrees_predictor_contract(metadata):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_extratrees(metadata):
            return
        contract = metadata.get("predictor_contract")
        if not isinstance(contract, dict) or contract.get("algorithm") != "extra_trees":
            warnings.warn(
                "ExtraTrees artifact is missing the current predictor contract and is legacy_only; "
                "historical inference is preserved, but retraining is required for a verified baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        if (
            contract.get("contract_version") != _EXTRATREES_PREDICTOR_CONTRACT_VERSION
            or contract.get("implementation") != "sklearn.ensemble.ExtraTreesClassifier"
            or contract.get("status") != "verified"
            or contract.get("architecture") != "whole_sample_extremely_randomized_tree_ensemble"
            or contract.get("split_rule")
            != "random_threshold_per_candidate_feature_then_best_impurity"
            or contract.get("aggregation") != "mean_class_probability"
            or contract.get("score_kind") != "probability"
            or contract.get("calibrated") is not False
            or contract.get("multioutput") != "native_sklearn_multi_output"
        ):
            raise InferenceUnsupportedError(
                "ExtraTrees artifact predictor contract is unsupported or incomplete",
                code="extratrees_predictor_contract_unsupported",
                stage="bundle",
                hint=(
                    "Retrain and re-export the ExtraTrees model with the current "
                    "PepPlatform contract."
                ),
            )
        trained_version = str(contract.get("implementation_version") or "").strip()
        if trained_version != sklearn.__version__:
            raise InferenceUnsupportedError(
                "ExtraTrees artifact was trained with a different scikit-learn version",
                code="extratrees_runtime_version_mismatch",
                stage="bundle",
                hint=(
                    f"Use scikit-learn {trained_version} for inference or retrain with "
                    f"the current scikit-learn {sklearn.__version__}."
                ),
                technical_detail=(
                    f"artifact scikit-learn={trained_version}; "
                    f"runtime scikit-learn={sklearn.__version__}"
                ),
            )

    @staticmethod
    def _metadata_claims_svm(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        embedding_export = metadata.get("embedding_export") or {}
        classic_export = (metadata.get("classic") or {}).get("embedding_export") or {}
        route_head = ((metadata.get("effective_route") or {}).get("predictor") or {}).get(
            "head"
        ) or {}
        candidates = [
            args.get("classifier"),
            args.get("model"),
            embedding_export.get("downstream"),
            classic_export.get("downstream"),
            route_head.get("id"),
        ]
        contract = metadata.get("predictor_contract")
        if isinstance(contract, dict):
            candidates.append(contract.get("algorithm"))
        return any(
            str(value or "").strip().lower() in {"svm", "calibrated_svm", "support_vector_machine"}
            for value in candidates
        )

    @staticmethod
    def _validate_svm_predictor_contract(metadata):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_svm(metadata):
            return
        contract = metadata.get("predictor_contract")
        if not isinstance(contract, dict) or contract.get("algorithm") != "support_vector_machine":
            warnings.warn(
                "SVM artifact is missing the current predictor contract and is legacy_only; "
                "historical inference is preserved, but retraining is required for a verified baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        implementation = contract.get("implementation")
        if implementation == "sklearn.calibration.CalibratedClassifierCV":
            ModelService._validate_calibrated_svm_predictor_contract(contract)
            return
        kernel = contract.get("kernel")
        implementation_matches_kernel = (
            implementation == "sklearn.svm.LinearSVC" and kernel == "linear"
        ) or (implementation == "sklearn.svm.SVC" and kernel == "rbf")
        convergence = contract.get("convergence")
        estimator_params = contract.get("estimator_params")
        effective_budgets = (
            estimator_params.get("effective_max_iter_by_output")
            if isinstance(estimator_params, dict)
            else None
        )
        num_outputs = contract.get("num_outputs")
        effective_budgets_valid = effective_budgets is None or (
            isinstance(effective_budgets, list)
            and isinstance(num_outputs, int)
            and not isinstance(num_outputs, bool)
            and len(effective_budgets) == num_outputs
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in effective_budgets
            )
        )
        convergence_budgets = (
            convergence.get("effective_max_iter_by_output")
            if isinstance(convergence, dict)
            else None
        )
        convergence_budgets_valid = convergence_budgets is None or (
            effective_budgets_valid and convergence_budgets == effective_budgets
        )
        if (
            contract.get("contract_version") != _SVM_PREDICTOR_CONTRACT_VERSION
            or not implementation_matches_kernel
            or contract.get("status") != "verified_platform_variant"
            or contract.get("input_contract") != "fixed_width_global_vector"
            or contract.get("score_kind") != "margin"
            or contract.get("output") != "decision_function_margin"
            or contract.get("calibrated") is not False
            or not isinstance(convergence, dict)
            or convergence.get("required") is not True
            or convergence.get("status") != "converged"
            or not effective_budgets_valid
            or not convergence_budgets_valid
        ):
            raise InferenceUnsupportedError(
                "SVM artifact predictor contract is unsupported or incomplete",
                code="svm_predictor_contract_unsupported",
                stage="bundle",
                hint="Retrain and re-export the SVM model with the current PepPlatform contract.",
            )
        trained_version = str(contract.get("implementation_version") or "").strip()
        if trained_version != sklearn.__version__:
            raise InferenceUnsupportedError(
                "SVM artifact was trained with a different scikit-learn version",
                code="svm_runtime_version_mismatch",
                stage="bundle",
                hint=(
                    f"Use scikit-learn {trained_version} for inference or retrain with "
                    f"the current scikit-learn {sklearn.__version__}."
                ),
                technical_detail=(
                    f"artifact scikit-learn={trained_version}; "
                    f"runtime scikit-learn={sklearn.__version__}"
                ),
            )

    @staticmethod
    def _validate_calibrated_svm_predictor_contract(contract):
        def finite_number(value):
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(float(value))
            )

        def bounded_metric(value):
            return finite_number(value) and 0.0 <= float(value) <= 1.0

        def valid_quality_output(item, index):
            if not isinstance(item, dict) or item.get("output_index") != index:
                return False
            sample_count = item.get("sample_count")
            positive_count = item.get("positive_count")
            negative_count = item.get("negative_count")
            counts_valid = all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (sample_count, positive_count, negative_count)
            )
            return (
                counts_valid
                and sample_count == positive_count + negative_count
                and sample_count > 0
                and bounded_metric(item.get("brier"))
                and finite_number(item.get("log_loss"))
                and float(item["log_loss"]) >= 0.0
                and bounded_metric(item.get("ece"))
                and finite_number(item.get("probability_min"))
                and finite_number(item.get("probability_max"))
                and 0.0 <= float(item["probability_min"]) <= float(item["probability_max"]) <= 1.0
            )

        calibration = contract.get("calibration")
        convergence = contract.get("convergence")
        quality = contract.get("probability_quality")
        estimator_params = contract.get("estimator_params")
        num_outputs = contract.get("num_outputs")
        effective_budgets = (
            estimator_params.get("effective_max_iter_by_output")
            if isinstance(estimator_params, dict)
            else None
        )
        convergence_outputs = convergence.get("outputs") if isinstance(convergence, dict) else None
        convergence_budgets_valid = (
            isinstance(effective_budgets, list)
            and isinstance(num_outputs, int)
            and not isinstance(num_outputs, bool)
            and len(effective_budgets) == num_outputs
            and isinstance(convergence_outputs, list)
            and len(convergence_outputs) == num_outputs
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in effective_budgets
            )
            and all(
                isinstance(item, dict)
                and item.get("output_index") == index
                and isinstance(item.get("effective_max_iter"), int)
                and not isinstance(item.get("effective_max_iter"), bool)
                and item["effective_max_iter"] == effective_budgets[index]
                and isinstance(item.get("attempts"), list)
                and bool(item["attempts"])
                and isinstance(item["attempts"][-1], dict)
                and item["attempts"][-1].get("status") == "converged"
                and all(
                    isinstance(attempt, dict)
                    and attempt.get("status") in {"converged", "not_converged"}
                    and isinstance(attempt.get("max_iter"), int)
                    and attempt["max_iter"] > 0
                    for attempt in item["attempts"]
                )
                for index, item in enumerate(convergence_outputs)
            )
        )
        quality_outputs = quality.get("outputs") if isinstance(quality, dict) else None
        quality_complete = (
            isinstance(quality, dict)
            and quality.get("status") == "validated"
            and quality.get("split") == "validation"
            and quality.get("acceptance") == "record_and_validate_no_dataset_threshold"
            and quality.get("ece_bins") == 10
            and isinstance(quality_outputs, list)
            and len(quality_outputs) == num_outputs
            and isinstance(quality.get("macro"), dict)
            and all(
                (
                    bounded_metric(quality["macro"].get(key))
                    if key != "log_loss"
                    else (
                        finite_number(quality["macro"].get(key))
                        and float(quality["macro"][key]) >= 0.0
                    )
                )
                for key in ("brier", "log_loss", "ece")
            )
            and all(valid_quality_output(item, index) for index, item in enumerate(quality_outputs))
        )
        if (
            contract.get("contract_version") != _CALIBRATED_SVM_PREDICTOR_CONTRACT_VERSION
            or contract.get("algorithm") != "support_vector_machine"
            or contract.get("implementation") != "sklearn.calibration.CalibratedClassifierCV"
            or contract.get("status") != "verified_platform_variant"
            or contract.get("base_implementation")
            not in {"sklearn.svm.LinearSVC", "sklearn.svm.SVC"}
            or (
                contract.get("base_implementation") == "sklearn.svm.LinearSVC"
                and contract.get("kernel") != "linear"
            )
            or (
                contract.get("base_implementation") == "sklearn.svm.SVC"
                and contract.get("kernel") != "rbf"
            )
            or (
                contract.get("base_implementation") == "sklearn.svm.LinearSVC"
                and contract.get("solver") != "liblinear"
            )
            or (
                contract.get("base_implementation") == "sklearn.svm.SVC"
                and contract.get("solver") != "libsvm"
            )
            or contract.get("input_contract") != "finite_fixed_width_global_vector"
            or not isinstance(contract.get("input_feature_count"), int)
            or isinstance(contract.get("input_feature_count"), bool)
            or contract.get("input_feature_count", 0) <= 0
            or contract.get("score_kind") != "probability"
            or contract.get("output") != "calibrated_positive_class_probability"
            or contract.get("calibrated") is not True
            or contract.get("architecture")
            != "independent_binary_cross_validated_platt_svm_per_output"
            or contract.get("multioutput") != "independent_binary_one_vs_rest"
            or not isinstance(num_outputs, int)
            or isinstance(num_outputs, bool)
            or num_outputs <= 0
            or not isinstance(calibration, dict)
            or calibration.get("method") != "sigmoid_platt"
            or calibration.get("base_score") != "decision_function"
            or calibration.get("fit_scope") != "train_only_cross_validation"
            or calibration.get("splitter") != "StratifiedKFold"
            or calibration.get("cv_folds") != 3
            or calibration.get("shuffle") is not True
            or not isinstance(calibration.get("random_state"), int)
            or isinstance(calibration.get("random_state"), bool)
            or calibration.get("ensemble") is not True
            or not isinstance(convergence, dict)
            or convergence.get("required") is not True
            or convergence.get("status") != "converged"
            or not convergence_budgets_valid
            or not isinstance(estimator_params, dict)
            or not isinstance(estimator_params.get("requested_max_iter"), int)
            or isinstance(estimator_params.get("requested_max_iter"), bool)
            or estimator_params.get("requested_max_iter", 0) <= 0
            or not quality_complete
        ):
            raise InferenceUnsupportedError(
                "calibrated SVM artifact predictor contract is unsupported or incomplete",
                code="calibrated_svm_predictor_contract_unsupported",
                stage="bundle",
                hint=(
                    "Retrain and re-export the calibrated SVM model with the current "
                    "PepPlatform contract."
                ),
            )
        trained_version = str(contract.get("implementation_version") or "").strip()
        if trained_version != sklearn.__version__:
            raise InferenceUnsupportedError(
                "calibrated SVM artifact was trained with a different scikit-learn version",
                code="calibrated_svm_runtime_version_mismatch",
                stage="bundle",
                hint=(
                    f"Use scikit-learn {trained_version} for inference or retrain with "
                    f"the current scikit-learn {sklearn.__version__}."
                ),
                technical_detail=(
                    f"artifact scikit-learn={trained_version}; "
                    f"runtime scikit-learn={sklearn.__version__}"
                ),
            )

    @staticmethod
    def _metadata_claims_xgb(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        embedding_export = metadata.get("embedding_export") or {}
        classic_export = (metadata.get("classic") or {}).get("embedding_export") or {}
        route_head = ((metadata.get("effective_route") or {}).get("predictor") or {}).get(
            "head"
        ) or {}
        candidates = [
            args.get("classifier"),
            args.get("model"),
            embedding_export.get("downstream"),
            classic_export.get("downstream"),
            route_head.get("id"),
        ]
        contract = metadata.get("predictor_contract")
        if isinstance(contract, dict):
            candidates.append(contract.get("algorithm"))
        return any(str(value or "").strip().lower() in {"xgb", "xgboost"} for value in candidates)

    @staticmethod
    def _metadata_claims_lgb(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        embedding_export = metadata.get("embedding_export") or {}
        classic_export = (metadata.get("classic") or {}).get("embedding_export") or {}
        route_head = ((metadata.get("effective_route") or {}).get("predictor") or {}).get(
            "head"
        ) or {}
        candidates = [
            args.get("classifier"),
            args.get("model"),
            embedding_export.get("downstream"),
            classic_export.get("downstream"),
            route_head.get("id"),
        ]
        contract = metadata.get("predictor_contract")
        if isinstance(contract, dict):
            candidates.append(contract.get("algorithm"))
        return any(str(value or "").strip().lower() in {"lgb", "lightgbm"} for value in candidates)

    @staticmethod
    def _metadata_claims_logreg(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        embedding_export = metadata.get("embedding_export") or {}
        classic_export = (metadata.get("classic") or {}).get("embedding_export") or {}
        route_head = ((metadata.get("effective_route") or {}).get("predictor") or {}).get(
            "head"
        ) or {}
        candidates = [
            args.get("classifier"),
            args.get("model"),
            embedding_export.get("downstream"),
            classic_export.get("downstream"),
            route_head.get("id"),
        ]
        contract = metadata.get("predictor_contract")
        if isinstance(contract, dict):
            candidates.append(contract.get("algorithm"))
        return any(
            str(value or "").strip().lower()
            in {"logreg", "logistic_regression", "logistic regression"}
            for value in candidates
        )

    @staticmethod
    def _metadata_claims_histgb(metadata):
        metadata = normalize_stream_fields(metadata)
        args = metadata.get("args") or {}
        embedding_export = metadata.get("embedding_export") or {}
        classic_export = (metadata.get("classic") or {}).get("embedding_export") or {}
        route_head = ((metadata.get("effective_route") or {}).get("predictor") or {}).get(
            "head"
        ) or {}
        candidates = [
            args.get("classifier"),
            args.get("model"),
            embedding_export.get("downstream"),
            classic_export.get("downstream"),
            route_head.get("id"),
        ]
        contract = metadata.get("predictor_contract")
        if isinstance(contract, dict):
            candidates.append(contract.get("algorithm"))
        return any(
            str(value or "").strip().lower()
            in {
                "histgb",
                "histogram_gradient_boosting",
                "histogram gradient boosting",
            }
            for value in candidates
        )

    @staticmethod
    def _validate_histgb_predictor_contract(metadata):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_histgb(metadata):
            return
        contract = metadata.get("predictor_contract")
        if (
            not isinstance(contract, dict)
            or contract.get("algorithm") != "histogram_gradient_boosting"
        ):
            warnings.warn(
                "HistGB artifact is missing the current predictor contract and is "
                "legacy_only; historical inference is preserved, but retraining is "
                "required for a verified baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        params = contract.get("estimator_params")
        iterations = contract.get("fitted_iterations_by_output")
        num_outputs = contract.get("num_outputs")
        if (
            contract.get("contract_version") != _HISTGB_PREDICTOR_CONTRACT_VERSION
            or contract.get("implementation") != "sklearn.ensemble.HistGradientBoostingClassifier"
            or contract.get("status") != "verified_platform_variant"
            or contract.get("input_contract")
            != "numeric_fixed_width_global_vector_nan_missing_supported"
            or not isinstance(contract.get("input_feature_count"), int)
            or contract.get("input_feature_count", 0) <= 0
            or contract.get("architecture")
            != "independent_binary_histogram_gradient_boosting_per_output"
            or contract.get("weak_learner") != "histogram_binned_regression_tree"
            or contract.get("loss") != "log_loss"
            or contract.get("inverse_link") != "logistic_sigmoid"
            or contract.get("binning") != "255_non_missing_bins_plus_missing_value_bin"
            or contract.get("native_missing_values") is not True
            or contract.get("early_stopping") is not False
            or contract.get("score_kind") != "probability"
            or contract.get("output") != "positive_class_probability"
            or contract.get("calibrated") is not False
            or contract.get("multioutput") != "independent_binary_one_vs_rest"
            or not isinstance(num_outputs, int)
            or isinstance(num_outputs, bool)
            or num_outputs <= 0
            or not isinstance(params, dict)
            or params.get("loss") != "log_loss"
            or params.get("early_stopping") is not False
            or params.get("warm_start") is not False
            or params.get("max_bins") != 255
            or params.get("categorical_features") != "from_dtype"
            or not isinstance(params.get("max_iter"), int)
            or isinstance(params.get("max_iter"), bool)
            or params.get("max_iter", 0) <= 0
            or not isinstance(iterations, list)
            or len(iterations) != num_outputs
            or not all(value == params.get("max_iter") for value in iterations)
        ):
            raise InferenceUnsupportedError(
                "HistGB artifact predictor contract is unsupported or incomplete",
                code="histgb_predictor_contract_unsupported",
                stage="bundle",
                hint=(
                    "Retrain and re-export the HistGB model with the current PepPlatform contract."
                ),
            )
        trained_version = str(contract.get("implementation_version") or "").strip()
        if trained_version != sklearn.__version__:
            raise InferenceUnsupportedError(
                "HistGB artifact was trained with a different scikit-learn version",
                code="histgb_runtime_version_mismatch",
                stage="bundle",
                hint=(
                    f"Use scikit-learn {trained_version} or retrain with "
                    f"scikit-learn {sklearn.__version__}."
                ),
            )

    @staticmethod
    def _validate_logreg_predictor_contract(metadata):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_logreg(metadata):
            return
        contract = metadata.get("predictor_contract")
        if not isinstance(contract, dict) or contract.get("algorithm") != "logistic_regression":
            warnings.warn(
                "LogReg artifact is missing the current predictor contract and is legacy_only; "
                "historical inference is preserved, but retraining is required for a verified baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        params = contract.get("estimator_params")
        convergence = contract.get("convergence")
        num_outputs = contract.get("num_outputs")
        effective = params.get("effective_max_iter_by_output") if isinstance(params, dict) else None
        actual = convergence.get("n_iter_by_output") if isinstance(convergence, dict) else None
        if (
            contract.get("contract_version") != _LOGREG_PREDICTOR_CONTRACT_VERSION
            or contract.get("implementation") != "sklearn.linear_model.LogisticRegression"
            or contract.get("status") != "verified_platform_variant"
            or contract.get("input_contract") != "finite_fixed_width_global_vector"
            or not isinstance(contract.get("input_feature_count"), int)
            or contract.get("input_feature_count", 0) <= 0
            or contract.get("solver") != "liblinear"
            or contract.get("link") != "logit"
            or contract.get("penalty") != "l2"
            or contract.get("score_kind") != "probability"
            or contract.get("output") != "positive_class_probability"
            or contract.get("calibrated") is not False
            or not isinstance(num_outputs, int)
            or num_outputs <= 0
            or not isinstance(params, dict)
            or params.get("solver") != "liblinear"
            or params.get("penalty") != "l2"
            or not isinstance(effective, list)
            or len(effective) != num_outputs
            or not all(isinstance(value, int) and value > 0 for value in effective)
            or not isinstance(convergence, dict)
            or convergence.get("required") is not True
            or convergence.get("status") != "converged"
            or convergence.get("effective_max_iter_by_output") != effective
            or not isinstance(actual, list)
            or len(actual) != num_outputs
            or not all(isinstance(value, int) and value > 0 for value in actual)
        ):
            raise InferenceUnsupportedError(
                "LogReg artifact predictor contract is unsupported or incomplete",
                code="logreg_predictor_contract_unsupported",
                stage="bundle",
                hint="Retrain and re-export the LogReg model with the current PepPlatform contract.",
            )
        trained_version = str(contract.get("implementation_version") or "").strip()
        if trained_version != sklearn.__version__:
            raise InferenceUnsupportedError(
                "LogReg artifact was trained with a different scikit-learn version",
                code="logreg_runtime_version_mismatch",
                stage="bundle",
                hint=f"Use scikit-learn {trained_version} or retrain with scikit-learn {sklearn.__version__}.",
            )

    @staticmethod
    def _validate_xgb_predictor_contract(metadata, bundle):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_xgb(metadata):
            return
        contract = metadata.get("predictor_contract")
        if not isinstance(contract, dict) or contract.get("algorithm") != "xgboost":
            warnings.warn(
                "XGB artifact is missing the current predictor contract and is legacy_only; "
                "historical inference is preserved, but retraining is required for a verified baseline",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        serialization = contract.get("serialization")
        params = contract.get("estimator_params")
        effective = contract.get("effective_params_by_output")
        execution_policy = contract.get("execution_policy")
        num_outputs = contract.get("num_outputs")
        execution_policy_invalid = execution_policy is not None and (
            not isinstance(execution_policy, dict)
            or execution_policy.get("n_jobs") != LGB_EXECUTION_POLICY["lgb_n_jobs"]
            or execution_policy.get("deterministic") is not True
            or execution_policy.get("histogram_mode") != LGB_EXECUTION_POLICY["lgb_histogram_mode"]
            or execution_policy.get("device_type") != LGB_EXECUTION_POLICY["lgb_device_type"]
            or not isinstance(execution_policy.get("seed"), int)
            or isinstance(execution_policy.get("seed"), bool)
            or execution_policy.get("seed", -1) < 0
            or execution_policy.get("child_seed_policy") != "experiment_seed"
        )
        if (
            contract.get("contract_version") != _XGB_PREDICTOR_CONTRACT_VERSION
            or contract.get("implementation") != "xgboost.XGBClassifier"
            or contract.get("status") != "verified_platform_variant"
            or contract.get("input_contract") != "finite_fixed_width_global_vector"
            or not isinstance(contract.get("input_feature_count"), int)
            or contract.get("input_feature_count", 0) <= 0
            or contract.get("booster") != "gbtree"
            or contract.get("tree_method") != "hist"
            or contract.get("objective") != "binary:logistic"
            or contract.get("score_kind") != "probability"
            or contract.get("output") != "positive_class_probability"
            or contract.get("calibrated") is not False
            or not isinstance(num_outputs, int)
            or num_outputs <= 0
            or not isinstance(params, dict)
            or not isinstance(effective, list)
            or len(effective) != num_outputs
            or execution_policy_invalid
            or not isinstance(serialization, dict)
            or serialization.get("format") != "joblib_preprocessors_plus_xgboost_ubjson"
            or serialization.get("xgboost_native_model") is not True
            or serialization.get("manifest_schema_version") != _XGB_MANIFEST_SCHEMA_VERSION
        ):
            raise InferenceUnsupportedError(
                "XGB artifact predictor contract is unsupported or incomplete",
                code="xgb_predictor_contract_unsupported",
                stage="bundle",
                hint="Retrain and re-export the XGB model with the current PepPlatform contract.",
            )
        if xgboost is None or XGBClassifier is None:
            raise InferenceUnsupportedError(
                "XGB artifact requires the xgboost runtime dependency",
                code="xgb_runtime_dependency_missing",
                stage="bundle",
                hint="Install the artifact's matching xgboost version before inference.",
            )
        trained_xgb = str(contract.get("implementation_version") or "").strip()
        trained_sklearn = str(contract.get("sklearn_version") or "").strip()
        if trained_xgb != xgboost.__version__:
            raise InferenceUnsupportedError(
                "XGB artifact was trained with a different xgboost version",
                code="xgb_runtime_version_mismatch",
                stage="bundle",
                hint=f"Use xgboost {trained_xgb} for inference or retrain with xgboost {xgboost.__version__}.",
                technical_detail=f"artifact xgboost={trained_xgb}; runtime xgboost={xgboost.__version__}",
            )
        if trained_sklearn != sklearn.__version__:
            raise InferenceUnsupportedError(
                "XGB artifact preprocessors were trained with a different scikit-learn version",
                code="xgb_sklearn_runtime_version_mismatch",
                stage="bundle",
                hint=f"Use scikit-learn {trained_sklearn} or retrain with scikit-learn {sklearn.__version__}.",
            )
        manifest_relative = serialization.get("manifest_path")
        if manifest_relative != "xgb_manifest.json":
            raise InferenceUnsupportedError(
                "XGB artifact manifest path is unsupported",
                code="xgb_manifest_invalid",
                stage="bundle",
            )
        root = Path(bundle).resolve()
        manifest_path = (root / manifest_relative).resolve()
        if not manifest_path.is_relative_to(root) or not manifest_path.is_file():
            raise InferenceUnsupportedError(
                "XGB artifact manifest is missing or outside the bundle",
                code="xgb_manifest_missing",
                stage="bundle",
            )
        if (
            serialization.get("manifest_sha256") != _sha256_file(manifest_path)
            or serialization.get("manifest_size_bytes") != manifest_path.stat().st_size
        ):
            raise InferenceUnsupportedError(
                "XGB artifact manifest failed its integrity check",
                code="xgb_manifest_integrity_failed",
                stage="bundle",
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InferenceUnsupportedError(
                "XGB artifact manifest is unreadable",
                code="xgb_manifest_invalid",
                stage="bundle",
                technical_detail=str(exc),
            ) from exc
        outputs = manifest.get("outputs")
        versions = manifest.get("implementation_versions") or {}
        labels = list(metadata.get("label_names") or [])
        if (
            manifest.get("schema_version") != _XGB_MANIFEST_SCHEMA_VERSION
            or manifest.get("predictor_contract_version") != _XGB_PREDICTOR_CONTRACT_VERSION
            or manifest.get("algorithm") != "xgboost"
            or manifest.get("implementation") != "xgboost.XGBClassifier"
            or versions.get("xgboost") != trained_xgb
            or versions.get("scikit_learn") != trained_sklearn
            or manifest.get("input_feature_count") != contract.get("input_feature_count")
            or manifest.get("num_outputs") != num_outputs
            or (
                execution_policy is not None
                and manifest.get("execution_policy") != execution_policy
            )
            or not isinstance(outputs, list)
            or len(outputs) != num_outputs
            or len(labels) != num_outputs
            or (manifest.get("reload_verification") or {}).get("status") != "passed"
        ):
            raise InferenceUnsupportedError(
                "XGB artifact manifest does not match its predictor contract",
                code="xgb_manifest_contract_mismatch",
                stage="bundle",
            )
        for output_index, output in enumerate(outputs):
            model_relative = output.get("model_path") if isinstance(output, dict) else None
            model_path = (root / str(model_relative or "")).resolve()
            if (
                output.get("output_index") != output_index
                or output.get("label_name") != str(labels[output_index])
                or output.get("effective_params") != effective[output_index]
                or not isinstance(model_relative, str)
                or Path(model_relative).is_absolute()
                or Path(model_relative).suffix.lower() != ".ubj"
                or not model_path.is_relative_to(root)
                or not model_path.is_file()
                or output.get("size_bytes") != model_path.stat().st_size
                or output.get("sha256") != _sha256_file(model_path)
            ):
                raise InferenceUnsupportedError(
                    f"XGB output {output_index} failed manifest validation",
                    code="xgb_model_integrity_failed",
                    stage="bundle",
                )

    @staticmethod
    def _validate_lgb_predictor_contract(metadata, bundle):
        metadata = normalize_stream_fields(metadata)
        if not ModelService._metadata_claims_lgb(metadata):
            return
        contract = metadata.get("predictor_contract")
        if not isinstance(contract, dict) or contract.get("algorithm") != "lightgbm":
            warnings.warn(
                "LGB artifact is missing the current predictor contract and is legacy_only; "
                "historical inference is preserved, but retraining is required for a verified platform variant",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        serialization = contract.get("serialization")
        params = contract.get("estimator_params")
        effective = contract.get("effective_params_by_output")
        num_outputs = contract.get("num_outputs")
        if (
            contract.get("contract_version") != _LGB_PREDICTOR_CONTRACT_VERSION
            or contract.get("implementation") != "lightgbm.LGBMClassifier"
            or contract.get("status") != "verified_platform_variant"
            or contract.get("input_contract") != "finite_fixed_width_global_vector"
            or not isinstance(contract.get("input_feature_count"), int)
            or contract.get("input_feature_count", 0) <= 0
            or contract.get("booster") != "gbdt"
            or contract.get("objective") != "binary"
            or contract.get("score_kind") != "probability"
            or contract.get("output") != "positive_class_probability"
            or contract.get("calibrated") is not False
            or not isinstance(num_outputs, int)
            or num_outputs <= 0
            or not isinstance(params, dict)
            or not isinstance(effective, list)
            or len(effective) != num_outputs
            or not isinstance(serialization, dict)
            or serialization.get("format") != "joblib_preprocessors_plus_lightgbm_native_text"
            or serialization.get("lightgbm_native_model") is not True
            or serialization.get("manifest_schema_version") != _LGB_MANIFEST_SCHEMA_VERSION
        ):
            raise InferenceUnsupportedError(
                "LGB artifact predictor contract is unsupported or incomplete",
                code="lgb_predictor_contract_unsupported",
                stage="bundle",
                hint="Retrain and re-export the LGB model with the current PepPlatform contract.",
            )
        if lightgbm is None or LightGBMBooster is None:
            raise InferenceUnsupportedError(
                "LGB artifact requires the lightgbm runtime dependency",
                code="lgb_runtime_dependency_missing",
                stage="bundle",
                hint="Install the artifact's matching lightgbm version before inference.",
            )
        trained_lgb = str(contract.get("implementation_version") or "").strip()
        trained_sklearn = str(contract.get("sklearn_version") or "").strip()
        if trained_lgb != lightgbm.__version__:
            raise InferenceUnsupportedError(
                "LGB artifact was trained with a different lightgbm version",
                code="lgb_runtime_version_mismatch",
                stage="bundle",
                hint=(
                    f"Use lightgbm {trained_lgb} for inference or retrain with "
                    f"lightgbm {lightgbm.__version__}."
                ),
                technical_detail=(
                    f"artifact lightgbm={trained_lgb}; runtime lightgbm={lightgbm.__version__}"
                ),
            )
        if trained_sklearn != sklearn.__version__:
            raise InferenceUnsupportedError(
                "LGB artifact preprocessors were trained with a different scikit-learn version",
                code="lgb_sklearn_runtime_version_mismatch",
                stage="bundle",
                hint=(
                    f"Use scikit-learn {trained_sklearn} or retrain with "
                    f"scikit-learn {sklearn.__version__}."
                ),
            )
        manifest_relative = serialization.get("manifest_path")
        if manifest_relative != "lgb_manifest.json":
            raise InferenceUnsupportedError(
                "LGB artifact manifest path is unsupported",
                code="lgb_manifest_invalid",
                stage="bundle",
            )
        root = Path(bundle).resolve()
        manifest_path = (root / manifest_relative).resolve()
        if not manifest_path.is_relative_to(root) or not manifest_path.is_file():
            raise InferenceUnsupportedError(
                "LGB artifact manifest is missing or outside the bundle",
                code="lgb_manifest_missing",
                stage="bundle",
            )
        if (
            serialization.get("manifest_sha256") != _sha256_file(manifest_path)
            or serialization.get("manifest_size_bytes") != manifest_path.stat().st_size
        ):
            raise InferenceUnsupportedError(
                "LGB artifact manifest failed its integrity check",
                code="lgb_manifest_integrity_failed",
                stage="bundle",
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InferenceUnsupportedError(
                "LGB artifact manifest is unreadable",
                code="lgb_manifest_invalid",
                stage="bundle",
                technical_detail=str(exc),
            ) from exc
        outputs = manifest.get("outputs")
        versions = manifest.get("implementation_versions") or {}
        labels = list(metadata.get("label_names") or [])
        if (
            manifest.get("schema_version") != _LGB_MANIFEST_SCHEMA_VERSION
            or manifest.get("predictor_contract_version") != _LGB_PREDICTOR_CONTRACT_VERSION
            or manifest.get("algorithm") != "lightgbm"
            or manifest.get("implementation") != "lightgbm.LGBMClassifier"
            or manifest.get("native_loader") != "lightgbm.Booster"
            or versions.get("lightgbm") != trained_lgb
            or versions.get("scikit_learn") != trained_sklearn
            or manifest.get("input_feature_count") != contract.get("input_feature_count")
            or manifest.get("num_outputs") != num_outputs
            or not isinstance(outputs, list)
            or len(outputs) != num_outputs
            or len(labels) != num_outputs
            or (manifest.get("reload_verification") or {}).get("status") != "passed"
        ):
            raise InferenceUnsupportedError(
                "LGB artifact manifest does not match its predictor contract",
                code="lgb_manifest_contract_mismatch",
                stage="bundle",
            )
        for output_index, output in enumerate(outputs):
            if not isinstance(output, dict):
                raise InferenceUnsupportedError(
                    f"LGB output {output_index} failed manifest validation",
                    code="lgb_model_integrity_failed",
                    stage="bundle",
                )
            model_relative = output.get("model_path")
            model_path = (root / str(model_relative or "")).resolve()
            if (
                output.get("output_index") != output_index
                or output.get("label_name") != str(labels[output_index])
                or output.get("effective_params") != effective[output_index]
                or not isinstance(model_relative, str)
                or Path(model_relative).is_absolute()
                or Path(model_relative).suffix.lower() != ".txt"
                or not model_path.is_relative_to(root)
                or not model_path.is_file()
                or output.get("size_bytes") != model_path.stat().st_size
                or output.get("sha256") != _sha256_file(model_path)
            ):
                raise InferenceUnsupportedError(
                    f"LGB output {output_index} failed manifest validation",
                    code="lgb_model_integrity_failed",
                    stage="bundle",
                )

    @staticmethod
    def _load_classical_payload(bundle, metadata):
        metadata = normalize_stream_fields(metadata)
        payload = joblib.load(Path(bundle) / "model.joblib")
        lsa_contract = metadata.get("lsa_contract")
        if isinstance(lsa_contract, dict):
            reducer = payload.get("task_reducer")
            payload_contract = payload.get("lsa_contract")
            if payload_contract != lsa_contract or not isinstance(reducer, dict):
                raise InferenceUnsupportedError(
                    "LSA joblib payload does not match metadata contract v1",
                    code="lsa_payload_contract_mismatch",
                    stage="bundle",
                )
            try:
                if reducer.get("kind") != "sklearn":
                    raise LSAContractError(
                        "lsa_reducer_type_invalid",
                        "LSA artifact reducer kind must be sklearn",
                    )
                validate_lsa_contract(
                    lsa_contract,
                    estimator=reducer.get("value"),
                    require_current_sklearn=True,
                )
            except LSAContractError as exc:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="bundle",
                    hint="Retrain and re-export LSA with the current PepPlatform contract.",
                    technical_detail=json.dumps(exc.details, sort_keys=True),
                ) from exc
        lda_contract = metadata.get("lda_contract")
        if isinstance(lda_contract, dict):
            reducer = payload.get("task_reducer")
            payload_contract = payload.get("lda_contract")
            if payload_contract != lda_contract or not isinstance(reducer, dict):
                raise InferenceUnsupportedError(
                    "LDA joblib payload does not match metadata contract v1",
                    code="lda_payload_contract_mismatch",
                    stage="bundle",
                )
            try:
                if reducer.get("kind") != "sklearn":
                    raise LDAContractError(
                        "lda_reducer_type_invalid",
                        "LDA artifact reducer kind must be sklearn",
                    )
                validate_lda_contract(
                    lda_contract,
                    estimator=reducer.get("value"),
                    upstream_feature_contract=(
                        metadata.get("pair_bow_joint_contract")
                        if is_pair_bow_route(lda_contract.get("route"))
                        else metadata.get("classic")
                    ),
                    require_current_sklearn=True,
                )
            except LDAContractError as exc:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="bundle",
                    hint="Retrain and re-export LDA with the current PepPlatform contract.",
                    technical_detail=json.dumps(exc.details, sort_keys=True),
                ) from exc
        plsa_contract = metadata.get("plsa_contract")
        plsa_claimed = ModelService._metadata_claims_plsa(metadata)
        reducer = payload.get("task_reducer")
        if isinstance(plsa_contract, dict):
            payload_contract = payload.get("plsa_contract")
            if payload_contract != plsa_contract or not isinstance(reducer, dict):
                raise InferenceUnsupportedError(
                    f"PLSA joblib payload does not match metadata contract v{PLSA_CONTRACT_VERSION}",
                    code="plsa_payload_contract_mismatch",
                    stage="bundle",
                )
            try:
                if reducer.get("kind") != "pepplatform_plsa":
                    raise PLSAContractError(
                        "plsa_reducer_type_invalid",
                        "PLSA artifact reducer kind must be pepplatform_plsa",
                    )
                validate_plsa_contract(
                    plsa_contract,
                    estimator=reducer.get("value"),
                    upstream_feature_contract=(
                        metadata.get("pair_bow_joint_contract")
                        if is_pair_bow_route(plsa_contract.get("route"))
                        else metadata.get("classic")
                    ),
                    require_current_versions=True,
                )
            except PLSAContractError as exc:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="bundle",
                    hint="Retrain and re-export PLSA with the current PepPlatform contract.",
                    technical_detail=json.dumps(exc.details, sort_keys=True),
                ) from exc
        elif plsa_claimed and (
            not isinstance(reducer, dict) or reducer.get("kind") != "nonnegative_sklearn"
        ):
            raise InferenceUnsupportedError(
                "legacy PLSA compatibility requires the historical nonnegative_sklearn KL-NMF payload",
                code="plsa_legacy_payload_invalid",
                stage="bundle",
            )
        tsvd_contract = metadata.get("tsvd_contract")
        if isinstance(tsvd_contract, dict):
            reducer = payload.get("ml_reducer")
            payload_contract = payload.get("tsvd_contract")
            if payload_contract != tsvd_contract or reducer is None:
                raise InferenceUnsupportedError(
                    "TSVD joblib payload does not match metadata contract v1",
                    code="tsvd_payload_contract_mismatch",
                    stage="bundle",
                )
            try:
                validate_tsvd_contract(
                    tsvd_contract,
                    estimator=reducer,
                    require_current_sklearn=True,
                )
            except TSVDContractError as exc:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="bundle",
                    hint="Retrain and re-export TSVD with the current PepPlatform contract.",
                    technical_detail=json.dumps(exc.details, sort_keys=True),
                ) from exc
        pca_contract = metadata.get("pca_contract")
        if isinstance(pca_contract, dict):
            reducer = payload.get("ml_reducer")
            payload_contract = payload.get("pca_contract")
            if payload_contract != pca_contract or reducer is None:
                raise InferenceUnsupportedError(
                    "PCA joblib payload does not match metadata contract v1",
                    code="pca_payload_contract_mismatch",
                    stage="bundle",
                )
            try:
                validate_pca_contract(
                    pca_contract,
                    estimator=reducer,
                    require_current_sklearn=True,
                )
            except PCAContractError as exc:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="bundle",
                    hint="Retrain and re-export PCA with the current PepPlatform contract.",
                    technical_detail=json.dumps(exc.details, sort_keys=True),
                ) from exc
        kpca_contract = metadata.get("kpca_contract")
        if isinstance(kpca_contract, dict):
            reducer = payload.get("ml_reducer")
            payload_contract = payload.get("kpca_contract")
            if payload_contract != kpca_contract or reducer is None:
                raise InferenceUnsupportedError(
                    "KPCA joblib payload does not match metadata contract v1",
                    code="kpca_payload_contract_mismatch",
                    stage="bundle",
                )
            try:
                validate_kpca_contract(
                    kpca_contract,
                    estimator=reducer,
                    require_current_sklearn=True,
                )
            except KPCAContractError as exc:
                raise InferenceUnsupportedError(
                    str(exc),
                    code=exc.code,
                    stage="bundle",
                    hint="Retrain and re-export KPCA with the current PepPlatform contract.",
                    technical_detail=json.dumps(exc.details, sort_keys=True),
                ) from exc
        contract = metadata.get("predictor_contract")
        if not isinstance(contract, dict):
            return payload
        algorithm = contract.get("algorithm")
        if algorithm == "xgboost":
            if (
                payload.get("classifier") != "xgb"
                or payload.get("estimator_kind") != "xgboost_native_ensemble"
                or payload.get("manifest_path") != "xgb_manifest.json"
            ):
                raise InferenceUnsupportedError(
                    "XGB joblib payload does not reference the native-model manifest",
                    code="xgb_payload_contract_mismatch",
                    stage="bundle",
                )
            manifest = json.loads(
                (Path(bundle) / payload["manifest_path"]).read_text(encoding="utf-8")
            )
            estimators = []
            for output in manifest["outputs"]:
                estimator = XGBClassifier()
                estimator.load_model(Path(bundle) / output["model_path"])
                estimators.append({"kind": "sklearn", "value": estimator})
            payload["estimators"] = estimators
            payload["estimator_kind"] = "probability_ensemble"
        elif algorithm == "lightgbm":
            if (
                payload.get("classifier") != "lgb"
                or payload.get("estimator_kind") != "lightgbm_native_ensemble"
                or payload.get("manifest_path") != "lgb_manifest.json"
            ):
                raise InferenceUnsupportedError(
                    "LGB joblib payload does not reference the native-model manifest",
                    code="lgb_payload_contract_mismatch",
                    stage="bundle",
                )
            manifest = json.loads(
                (Path(bundle) / payload["manifest_path"]).read_text(encoding="utf-8")
            )
            payload["estimators"] = [
                {
                    "kind": "lightgbm_booster",
                    "value": LightGBMBooster(model_file=str(Path(bundle) / output["model_path"])),
                }
                for output in manifest["outputs"]
            ]
            payload["estimator_kind"] = "probability_ensemble"
        return payload

    @staticmethod
    def _predict_classical_scores(payload, features):
        kind = payload["estimator_kind"]
        if kind == "svm":
            return np.stack(
                [estimator.decision_function(features) for estimator in payload["estimators"]],
                axis=1,
            )
        if kind == "calibrated_svm":
            values = np.asarray(features)
            expected_width = payload.get("input_feature_count")
            if (
                values.ndim != 2
                or not np.isfinite(values).all()
                or not isinstance(expected_width, int)
                or values.shape[1] != expected_width
            ):
                raise InferenceUnsupportedError(
                    "calibrated SVM inference features violate the artifact input contract",
                    code="calibrated_svm_input_contract_mismatch",
                    stage="inference",
                )
            columns = []
            for wrapped in payload.get("estimators") or []:
                if not isinstance(wrapped, dict) or wrapped.get("kind") != "sklearn":
                    raise InferenceUnsupportedError(
                        "calibrated SVM artifact contains an unsupported output estimator",
                        code="calibrated_svm_payload_contract_mismatch",
                        stage="bundle",
                    )
                estimator = wrapped.get("value")
                columns.append(
                    ModelService._positive_probability_column(
                        estimator.predict_proba(values), estimator.classes_
                    )
                )
            if not columns:
                raise InferenceUnsupportedError(
                    "calibrated SVM artifact does not contain output estimators",
                    code="calibrated_svm_payload_contract_mismatch",
                    stage="bundle",
                )
            scores = np.stack(columns, axis=1)
            if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
                raise InferenceUnsupportedError(
                    "calibrated SVM produced invalid probability scores",
                    code="calibrated_svm_probability_contract_failed",
                    stage="inference",
                )
            return scores
        if kind in {"rf", "tree_ensemble"}:
            proba = payload["estimator"].predict_proba(features)
            classes = payload["estimator"].classes_
            if isinstance(proba, list):
                return np.stack(
                    [
                        ModelService._positive_probability_column(item, classes[index])
                        for index, item in enumerate(proba)
                    ],
                    axis=1,
                )
            return ModelService._positive_probability_column(proba, classes)[:, None]
        if kind == "probability_ensemble":
            scores = []
            for estimator in payload["estimators"]:
                if estimator["kind"] == "constant":
                    scores.append(np.full(features.shape[0], estimator["value"], dtype=np.float32))
                elif estimator["kind"] == "lightgbm_booster":
                    scores.append(np.asarray(estimator["value"].predict(features)))
                else:
                    scores.append(estimator["value"].predict_proba(features)[:, 1])
            return np.stack(scores, axis=1)
        raise InferenceUnsupportedError("unsupported classical estimator kind")

    @staticmethod
    def _classical_score_kind(payload):
        if payload.get("estimator_kind") == "svm":
            return "margin"
        return "probability"

    @staticmethod
    def _positive_probability_column(proba, classes):
        classes_list = list(classes)
        if 1 in classes_list:
            return proba[:, classes_list.index(1)]
        return np.zeros(proba.shape[0], dtype=np.float32)


install_legacy_class_exports(ModelService)
