# Resolve local package imports before accepting historical CLI entry points.
import sys as _stream_sys
from pathlib import Path as _StreamPath
_stream_root = str(_StreamPath(__file__).resolve().parents[1])
if _stream_root not in _stream_sys.path:
    _stream_sys.path.insert(0, _stream_root)

import sys
import os
import argparse
import csv
import json
import random
import hashlib
import shutil
import subprocess
import tempfile
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm as _tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


def tqdm(*args, **kwargs):
    """Keep terminal progress bars interactive without emitting carriage-return Web logs."""
    kwargs.setdefault("disable", not sys.stderr.isatty())
    return _tqdm(*args, **kwargs)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pepplatform.task_compat import normalize_task_type

from pepplatform.dataset import PeptideDataset
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.pssm_covariance import (
    HYBRID_PSSM_COVARIANCE_ENCODERS,
    PSSM_COVARIANCE_ENCODERS,
    parse_pssm_covariance_lag,
    pssm_covariance_lag_key,
    pssm_covariance_profile_feature_type,
)
from pepplatform.models import (
    ModelFactory,
    PredictionHeadClassifier,
    PretrainedResiduePredictor,
    ResidueModelInput,
    SemanticResidualAdapter,
    sequence_predictor_contract,
)
from pepplatform.analysis import build_score_analysis, collect_nn_scores, save_analysis
from pepplatform.evaluators import EVALUATION_PROTOCOL_VERSION, Evaluator, OperatingPoint
from pepplatform.objectives import (
    REGISTRY,
    add_selection_arguments,
    resolve_selection,
    selection_fields,
    objective_capabilities,
    checkpoint_improved,
)
from pepplatform.ml_trainers import ClassicalMLTrainer
from pepplatform.cli_wizard import InteractiveConfigWizard
from pepplatform.route_registry import (
    ROUTE_SCHEMA_VERSION,
    format_effective_route,
    resolve_effective_route,
)
from pepplatform.run_progress import emit_run_progress
from pepplatform.training_failures import run_with_failure_reporting
from pepplatform.svm_resume import load_svm_resume_bundle, save_svm_resume_bundle
from pepplatform.tokenizers import (
    BPETokenizerTrainer,
    Esm3Tokenizer,
    EsmTokenizer,
    EsmcTokenizer,
    ProT5Tokenizer,
    ProtBertTokenizer,
    UnigramTokenizerTrainer,
    WordPieceTokenizerTrainer,
)
from pepplatform.encoders import (
    RESIDUE_ALIGNMENT_VERSION,
    EsmEncoder,
    ProtBertEncoder,
    ProT5Encoder,
    EsmcEncoder,
    Esm3Encoder,
    ResidueEncoding,
)
from pepplatform.workflow import (
    CLASSIC_ENCODER_OPTIONS,
    CLASSICAL_CLASSIFIERS,
    PROFILE_DEPENDENT_INFERENCE_REASON,
    CLASSIFIER_PARAMETER_DEFAULTS,
    SVM_CALIBRATION_CV_FOLDS,
    SVM_CALIBRATION_OPTIONS,
    RESIDUE_ADAPTER_DIM_DEFAULT,
    RESIDUE_ADAPTER_DIM_MAX,
    RESIDUE_ADAPTER_DIM_MIN,
    RESIDUE_ADAPTER_DIM_MULTIPLE,
    SEQUENCE_NEURAL_DEFAULTS,
    SEQUENCE_PROFILE_DEFAULTS,
    TRAINING_ENCODER_OPTIONS,
    exports_sequence_embeddings,
    default_model_selection_objective,
    ensure_encoder_available,
    normalize_encoder_name,
    normalize_prediction_head_config,
    normalize_workflow_args,
    parse_prediction_head_hidden_dims,
    prediction_head_artifact_config,
    PRETRAINED_MODEL_DEFAULTS,
    pretrained_model_local_candidates,
    resolve_pretrained_model_source as resolve_workflow_pretrained_model_source,
    uses_profile_features,
    validate_index_route_semantics,
    validate_linear_prediction_head_semantics,
    validate_training_mode_classifier,
)
from pepplatform.cache_provenance import (
    CACHE_PROVENANCE_VERSION,
    digest_text_items,
    model_source_fingerprint,
    sha256_file,
)
from pepplatform.protbert import (
    PROTBERT_OFFICIAL_MODEL_ID,
    PROTBERT_POOLING,
    PROTBERT_REPRESENTATION_CONTRACT_VERSION,
    format_protbert_provenance,
)
from pepplatform.prott5 import (
    PROTT5_MAX_RESIDUES,
    PROTT5_PAD_TOKEN_ID,
    PROTT5_EOS_TOKEN_ID,
    PROTT5_POOLING,
    PROTT5_REPRESENTATION_CONTRACT_VERSION,
    PROTT5_VOCAB_SIZE,
    format_prott5_provenance,
)
from pepplatform.frequency import (
    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    FREQUENCY_ENCODER_CONTRACT_VERSION,
    parse_frequency_max_raw_length,
    save_idf_state,
    validate_frequency_max_raw_length,
)
from pepplatform.lsa import LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
from pepplatform.tsvd import TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB
from pepplatform.pca import PCA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
from pepplatform.kpca import KPCA_DEFAULT_MEMORY_LIMIT_GIB
from pepplatform.lda import LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
from pepplatform.plsa import PLSA_DEFAULT_MEMORY_LIMIT_GIB
from pepplatform.covariance import (
    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
    parse_covariance_max_raw_length,
    validate_covariance_max_raw_length,
)
from pepplatform.pdt import (
    PDT_DEFAULT_LAG,
    PDT_DEFAULT_MAX_RAW_LENGTH,
    parse_pdt_lag,
    parse_pdt_max_raw_length,
    validate_pdt_max_raw_length,
)
from pepplatform.bpe import bundle_bpe_artifact
from pepplatform.dr import (
    DR_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    DR_DEFAULT_MAX_DISTANCE,
    DR_DEFAULT_MAX_RAW_LENGTH,
    DR_GLOBAL_MAX_LENGTH_UNIT,
    DR_ORDERED_MAX_LENGTH_UNIT,
    DRContractError,
    dr_contract_metadata,
    ensure_dr_dense_memory_budget,
    estimate_dr_dense_tensor_bytes,
    parse_dr_dense_memory_limit_gib,
    parse_dr_max_distance,
    parse_dr_max_raw_length,
    validate_dr_max_distance,
    validate_dr_max_raw_length,
)
from pepplatform.dt import (
    DT_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    DT_DEFAULT_MAX_DISTANCE,
    DT_MAX_LENGTH_UNIT,
    DTContractError,
    dt_contract_metadata,
    ensure_dt_dense_memory_budget,
    estimate_dt_dense_bytes,
    parse_dt_dense_memory_limit_gib,
    parse_dt_max_distance,
    validate_dt_max_distance,
)
from pepplatform.sequence_lengths import dataset_role_max, scan_dataset_length_stats
from pepplatform.top_ngram import (
    TOP_N_GRAM_CONTRACT_VERSION,
    TOP_N_GRAM_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    TOP_N_GRAM_FALLBACK_POLICY,
    TOP_N_GRAM_FALLBACK_POLICY_VERSION,
    TOP_N_GRAM_MAX_LENGTH_UNIT,
    TOP_N_GRAM_STRICT_PROFILE_POLICY,
    TOP_N_GRAM_TIE_POLICY,
    TargetFrequencyProfileStore,
    TopNGramContractError,
    ensure_top_ngram_dense_memory_budget,
    estimate_top_ngram_dense_bytes,
    parse_top_ngram_dense_memory_limit_gib,
    resolve_top_ngram_max_length,
    top_ngram_feature_dim,
    validate_top_ngram_n,
)
from pepplatform.profile_generation import (
    ProfileGenerationConfig,
    atomic_copy_file,
)
from pepplatform.sequence_profiles import (
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
    FEATURE_AC_PSSM_BLOSUM62_HYBRID,
    FEATURE_CC_PSSM_BLOSUM62_HYBRID,
    FEATURE_PSFM,
    FEATURE_PSFM_BLOSUM62_HYBRID,
    FEATURE_PSSM,
    FEATURE_PSSM_BLOSUM62_HYBRID,
    PSSM_HYBRID_FEATURE_TYPES,
    PSFM_HYBRID_FEATURE_TYPES,
    PROFILE_CONTRACT_VERSION,
    ProfileStore,
    SequenceProfileError,
    blosum62_contract_metadata,
)
from pepplatform.wordpiece import (
    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
    WORDPIECE_MAX_LENGTH_UNIT,
    bundle_wordpiece_artifact,
    validate_wordpiece_max_raw_length,
    validate_wordpiece_vocab_size,
)
from pepplatform.unigram import (
    UNIGRAM_DEFAULT_ALPHA,
    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    UNIGRAM_DEFAULT_NBEST_SIZE,
    UNIGRAM_MAX_LENGTH_UNIT,
    bundle_unigram_artifact,
    validate_unigram_max_raw_length,
    validate_unigram_sampling,
    validate_unigram_vocab_size,
)
from pepplatform.esmc import format_esmc_provenance
from pepplatform.esm3 import (
    ESM3_OFFICIAL_MODEL_ID,
    ESM3_POOLING,
    ESM3_REPRESENTATION_CONTRACT_VERSION,
    format_esm3_provenance,
)
from pepplatform.esm2 import (
    ESM2_OFFICIAL_MODEL_ID,
    ESM2_REPRESENTATION_CONTRACT_VERSION,
    ESM2_POOLING,
    format_esm2_provenance,
    is_complete_esm2_snapshot,
    resolve_esm2_source,
)
from pepplatform.glove import glove_cache_signature
from pepplatform.static_embeddings import (
    FASTTEXT_DEFAULT_BUCKET,
    STATIC_EMBEDDING_DEFAULT_MEMORY_LIMIT_GIB,
    fasttext_cache_signature,
    validate_positive_int,
    validate_static_embedding_memory_limit_gib,
    word2vec_cache_signature,
)
from pepplatform.semantic import (
    IntrinsicLabelRelationSource,
    RecordingRandomSampler,
    SemanticLossOutput,
    SemanticTrainingProfile,
    auxiliary_eval_forward,
    build_dense_hard_partner_table,
    build_partner_table,
    normalize_semantic_aliases,
    placebo_shuffle_table,
    prepare_semantic_projectors,
    save_semantic_training_profile,
    semantic_batch_contrastive_loss,
    semantic_contrastive_loss,
    semantic_enabled as semantic_is_enabled,
    semantic_representation_contract,
)

PAIR_TASK_TYPES = {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}


def encoder_argument(value):
    try:
        ensure_encoder_available(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return normalize_encoder_name(value)


def prediction_head_kwargs(args):
    return {
        "head_hidden_dims": getattr(args, "head_hidden_dims", None),
        "head_activation": getattr(args, "head_activation", None),
        "head_dropout": getattr(args, "head_dropout", None),
    }


def set_global_seed(seed, *, deterministic=False):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if deterministic:
            torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric focal loss for multi-label logits (Ridnik et al., 2021)."""

    def __init__(self, *, gamma_pos=1.0, gamma_neg=4.0, clip=0.05):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)

    def forward(self, logits, targets):
        targets = targets.to(dtype=logits.dtype)
        positive = torch.sigmoid(logits)
        negative = 1.0 - positive
        if self.clip > 0:
            negative = (negative + self.clip).clamp(max=1.0)
        positive_term = positive * targets
        negative_term = negative * (1.0 - targets)
        probability = positive_term + negative_term
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        weight = torch.pow(1.0 - probability, gamma)
        log_likelihood = targets * torch.log(positive.clamp_min(1e-8))
        log_likelihood += (1.0 - targets) * torch.log(negative.clamp_min(1e-8))
        return -(weight * log_likelihood).mean()


def _offline_mode_enabled():
    return str(os.environ.get("PEPPLATFORM_OFFLINE_MODE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_pretrained_model_source(project_root, raw_model, local_candidates, label, cli_arg):
    tokenizer = str(cli_arg).replace("_model", "")
    return resolve_workflow_pretrained_model_source(
        project_root,
        raw_model,
        local_candidates,
        label,
        cli_arg,
        tokenizer=tokenizer,
    )


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class ResidueTensorDataset(Dataset):
    def __init__(self, encoding, labels, *, label_names=None, dl_scaler=None):
        if not isinstance(encoding, ResidueEncoding):
            raise TypeError("ResidueTensorDataset requires ResidueEncoding")
        if encoding.features.shape[0] != labels.shape[0]:
            raise ValueError("residue features and labels have different row counts")
        self.encoding = encoding
        self.labels = labels
        self.label_names = list(label_names or [])
        self.dl_scaler = dl_scaler

    def __len__(self):
        return int(self.labels.shape[0])

    def get_num_classes(self):
        return int(self.labels.shape[1])

    def __getitem__(self, idx):
        return (
            ResidueModelInput(self.encoding.features[idx], self.encoding.mask[idx]),
            self.labels[idx],
        )


class IndexedDataset(Dataset):
    """Append the stable training-row index without changing sample order."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("semantic training requires datasets yielding (inputs, targets)")
        return item[0], item[1], torch.tensor(idx, dtype=torch.long)


class SequencePipeline:
    SEMANTIC_SIMILARITY_MODELS = {"cnn", "rnn", "gru", "lstm", "transformer", "gcn", "gat", "mlp"}
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
    PROFILE_INDEX_ATTRIBUTES = {
        FEATURE_AC_PSSM_BLOSUM62_HYBRID: ("ac_pssm_blosum62_hybrid_index_path"),
        FEATURE_CC_PSSM_BLOSUM62_HYBRID: ("cc_pssm_blosum62_hybrid_index_path"),
        FEATURE_ACC_PSSM_BLOSUM62_HYBRID: ("acc_pssm_blosum62_hybrid_index_path"),
    }

    def _resolve_top_ngram_dataset_length(self):
        tokenizer = str(getattr(self.args, "tokenizer", "")).lower()
        if tokenizer not in self.PROFILE_TOKENIZERS:
            return
        length_stats = scan_dataset_length_stats(
            getattr(self.args, "data_dir", ""),
            "sequence_dir",
        )
        required = dataset_role_max(length_stats, "sequence")
        if required < 1:
            raise SystemExit(f"{tokenizer} could not determine a positive dataset sequence length")
        requested = int(getattr(self.args, "max_length", 60))
        try:
            effective, adjusted = resolve_top_ngram_max_length(
                requested,
                required,
                context=f"{tokenizer} dataset sequences",
            )
        except TopNGramContractError as exc:
            raise SystemExit(str(exc)) from exc
        self.args.max_length = effective
        if adjusted:
            print(
                f"[{tokenizer}] max_length auto-adjusted from "
                f"{requested} to {effective} for full dataset coverage"
            )

    def _default_pretrained_feature_dir(self, feature_name):
        data_dir = str(getattr(self.args, "data_dir", "")).strip()
        if (
            normalize_task_type(self.args.task_type) == "pep_hlaii"
            and os.path.basename(os.path.normpath(data_dir)) == "data_quick_50k"
        ):
            return os.path.join(data_dir, f"{feature_name}_features")
        return os.path.join(self.project_root, self.task_root_rel, f"{feature_name}_features")

    def _jsonable_args(self):
        payload = {}
        for key, value in vars(self.args).items():
            if str(key).startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
            else:
                payload[key] = str(value)
        return payload

    def _load_label_names(self, csv_path, width=None):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return []
        header = rows[0]
        try:
            [float(x) for x in header]
            return [f"label_{idx + 1}" for idx in range(width or len(header))]
        except ValueError:
            return list(header)

    def _pretrained_embedding_artifact_info(self, tag, model_name, input_dim, label_names):
        model_identity = getattr(self.args, "_resolved_pretrained_model_identity", None)
        pooling = (
            str(model_identity.get("pooling"))
            if isinstance(model_identity, dict)
            else (
                ESM3_POOLING
                if tag == "esm3"
                else (
                    ESM2_POOLING
                    if tag == "esm2"
                    else (PROTBERT_POOLING if tag == "protbert" else "attention_mask_mean")
                )
            )
        )
        pretrained = {
            "tokenizer": tag,
            "model_name": model_name,
            "max_length": int(getattr(self.args, "max_length", 60)),
            "pooling": pooling,
            "input_dim": int(input_dim),
        }
        if isinstance(model_identity, dict):
            pretrained["model_identity"] = dict(model_identity)
        adjustment = getattr(self.args, "prott5_max_length_adjustment", None)
        if tag == "prott5" and isinstance(adjustment, dict):
            pretrained["max_length_adjustment"] = dict(adjustment)
        if tag == "esm2":
            pretrained.update(
                {
                    "representation_contract_version": ESM2_REPRESENTATION_CONTRACT_VERSION,
                    "output_kind": "sequence_embedding",
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_validate_alphabet",
                    "truncation": "max_length_residues_then_BOS_EOS",
                }
            )
        if tag == "protbert":
            pretrained.update(
                {
                    "representation_contract_version": PROTBERT_REPRESENTATION_CONTRACT_VERSION,
                    "output_kind": "sequence_embedding",
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_UZOB_to_X",
                    "truncation": "max_length_residues_then_CLS_SEP",
                }
            )
        if tag == "prott5":
            pretrained.update(
                {
                    "representation_contract_version": PROTT5_REPRESENTATION_CONTRACT_VERSION,
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_UZOB_to_X",
                    "truncation": "max_length_residues_then_EOS",
                }
            )
        if tag == "esm3":
            pretrained.update(
                {
                    "representation_contract_version": ESM3_REPRESENTATION_CONTRACT_VERSION,
                    "output_kind": "sequence_embedding",
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_validate_alphabet",
                    "truncation": "max_length_residues_then_CLS_EOS",
                }
            )
        info = {
            "pretrained": pretrained,
            "label_names": list(label_names or []),
        }
        runtime_environment = getattr(self.args, "_resolved_pretrained_runtime_environment", None)
        if tag in {"esm2", "protbert", "prott5", "esmc", "esm3"} and isinstance(
            runtime_environment, dict
        ):
            info["runtime_environment"] = dict(runtime_environment)
        return info

    def _profile_generation_needed(self):
        tokenizer = str(getattr(self.args, "tokenizer", "") or "").strip().lower()
        encoder = str(getattr(self.args, "encoder", "") or "").strip().lower()
        return tokenizer in self.PROFILE_TOKENIZERS or encoder in self.PROFILE_ENCODERS

    def _profile_feature_kind(self):
        tokenizer = str(getattr(self.args, "tokenizer", "") or "").strip().lower()
        encoder = str(getattr(self.args, "encoder", "") or "").strip().lower()
        if tokenizer in self.PROFILE_TOKENIZERS:
            return "target_frequency"
        if encoder in PSSM_COVARIANCE_ENCODERS:
            return pssm_covariance_profile_feature_type(encoder)
        if encoder == "pssm_blosum62_hybrid":
            return FEATURE_PSSM_BLOSUM62_HYBRID
        if encoder == "psfm_blosum62_hybrid":
            return FEATURE_PSFM_BLOSUM62_HYBRID
        need_pssm = encoder in {"pssm", "ac_pssm", "cc_pssm", "acc_pssm"}
        need_psfm = encoder == "psfm"
        if need_pssm and need_psfm:
            return "both"
        if need_psfm:
            return "psfm"
        return "pssm"

    def _profile_index_attribute(self, feature):
        if feature == "target_frequency":
            return "frequency_profile_index_path"
        if feature in {FEATURE_PSFM, FEATURE_PSFM_BLOSUM62_HYBRID}:
            return "psfm_index_path"
        return self.PROFILE_INDEX_ATTRIBUTES.get(feature, "pssm_index_path")

    def _artifact_relpath(self, path):
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        if not artifact_dir or not path:
            return ""
        try:
            return os.path.relpath(path, artifact_dir)
        except ValueError:
            return path

    def _blast_db_ready(self, db_prefix):
        if not db_prefix:
            return False
        return all(os.path.exists(f"{db_prefix}{suffix}") for suffix in [".pin", ".phr", ".psq"])

    def _build_profile_blast_db(self, subject_fasta, db_prefix):
        if shutil.which("makeblastdb") is None:
            print("[artifact] profile BLAST DB not prebuilt: makeblastdb not found")
            return False
        os.makedirs(os.path.dirname(db_prefix), exist_ok=True)
        cmd = [
            "makeblastdb",
            "-in",
            subject_fasta,
            "-dbtype",
            "prot",
            "-out",
            db_prefix,
            "-parse_seqids",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except Exception as exc:
            print(f"[artifact] profile BLAST DB prebuild skipped: {exc}")
            return False
        return self._blast_db_ready(db_prefix)

    def _profile_generation_config(self):
        return ProfileGenerationConfig(
            iterations=int(getattr(self.args, "profile_iterations", 3)),
            evalue=float(getattr(self.args, "profile_evalue", 0.001)),
            inclusion_evalue=float(getattr(self.args, "profile_inclusion_evalue", 0.001)),
            threads=int(getattr(self.args, "profile_threads", 1)),
        )

    def _profile_manifest_for_current_route(self):
        feature = self._profile_feature_kind()
        if feature == "target_frequency":
            index_path = str(getattr(self, "frequency_profile_index_path", "") or "")
            if not index_path:
                raise SequenceProfileError(
                    "target-frequency profile index path is missing",
                    code="profile_manifest_missing",
                )
            store = TargetFrequencyProfileStore(
                index_path,
                max_length=int(self.args.max_length),
            )
            provenance = store.manifest.get("provenance") or {}
            subject_source = store.index_path.parent / str(
                provenance.get("subject_fasta_artifact") or ""
            )
            return store.manifest, store.manifest_sha256, subject_source
        index_path = getattr(
            self,
            self._profile_index_attribute(feature),
            "",
        )
        if not index_path:
            raise SequenceProfileError(
                f"{feature} profile index path is missing",
                code="profile_manifest_missing",
            )
        store = ProfileStore(index_path, feature_type=feature)
        provenance = store.manifest.get("provenance") or {}
        subject_source = store.index_path.parent / str(
            provenance.get("subject_fasta_artifact") or ""
        )
        return store.manifest, store.manifest_sha256, subject_source

    def _sequence_profile_generation_artifact_info(self):
        if not self._profile_generation_needed():
            return {}
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        if not artifact_dir:
            return {}
        profile_dir = os.path.join(artifact_dir, "profile_generation")
        subject_fasta = os.path.join(profile_dir, "subject.fasta")
        try:
            profile_manifest, manifest_sha256, subject_source = (
                self._profile_manifest_for_current_route()
            )
        except (OSError, ValueError, TypeError, SequenceProfileError) as exc:
            raise RuntimeError(
                "cannot export a profile-backed model without a validated "
                f"v{PROFILE_CONTRACT_VERSION} profile manifest: {exc}"
            ) from exc
        if not subject_source.is_file():
            raise RuntimeError(
                "validated training profile manifest is missing its frozen subject FASTA artifact"
            )
        training_provenance = profile_manifest.get("provenance") or {}
        training_subject_sha256 = str(training_provenance.get("subject_fasta_sha256") or "")
        os.makedirs(profile_dir, exist_ok=True)
        atomic_copy_file(
            subject_source,
            subject_fasta,
            expected_sha256=training_subject_sha256,
        )
        subject_rel = self._artifact_relpath(subject_fasta)
        db_prefix_rel = ""
        db_prefix = os.path.join(profile_dir, "blastdb", "profile_db")
        if self._build_profile_blast_db(subject_fasta, db_prefix):
            db_prefix_rel = self._artifact_relpath(db_prefix)
        config = self._profile_generation_config()
        feature = self._profile_feature_kind()
        subject_sha256 = sha256_file(subject_fasta)
        database_scope = str(training_provenance.get("database_scope") or "")
        if training_subject_sha256 != subject_sha256:
            raise RuntimeError(
                "exported profile subject FASTA does not exactly match the "
                "training profile database"
            )
        on_missing = (
            "blosum62_hybrid"
            if feature in (PSSM_HYBRID_FEATURE_TYPES | PSFM_HYBRID_FEATURE_TYPES)
            else str(getattr(self.args, "profile_on_missing", "error") or "error")
        )
        return {
            "profile_generation": {
                "profile_contract_version": PROFILE_CONTRACT_VERSION,
                "strategy": "psiblast_subject_fasta",
                "feature": feature,
                "subject_fasta": subject_rel,
                "subject_fasta_sha256": subject_sha256,
                "db_prefix": db_prefix_rel,
                "db_scope": database_scope,
                "iterations": config.iterations,
                "evalue": config.evalue,
                "inclusion_evalue": config.inclusion_evalue,
                "threads": config.threads,
                "on_missing": on_missing,
                "scientific_parameters": config.scientific_parameters(),
                "manifest_sha256": manifest_sha256,
                "training_profile_provenance": dict(profile_manifest.get("provenance") or {}),
            }
        }

    def _profile_inference_supported(self, profile_info):
        return not uses_profile_features(self.args) or bool(
            (profile_info or {}).get("profile_generation", {}).get("subject_fasta")
        )

    def _sequence_feature_artifact_info(self, train_data, profile_info=None):
        info = {
            "input_schema": "peptide_sequences",
            "label_names": list(getattr(train_data, "label_names", [])),
        }
        tokenizer = str(getattr(self.args, "tokenizer", "") or "").strip().lower()
        if tokenizer == "kmer":
            info["classic"] = {
                "tokenizer": tokenizer,
                "encoder": str(getattr(self.args, "encoder", "index") or "index").strip().lower(),
                "max_length": int(getattr(self.args, "max_length", 60)),
                "k": int(getattr(self.args, "k", 2)),
                "ac_lag": int(getattr(self.args, "ac_lag", 10)),
                "cc_lag": int(getattr(self.args, "cc_lag", 10)),
                "acc_lag": int(getattr(self.args, "acc_lag", 10)),
                "pdt_lag": int(getattr(self.args, "pdt_lag", PDT_DEFAULT_LAG)),
                "pdt_max_raw_length": validate_pdt_max_raw_length(
                    getattr(
                        self.args,
                        "pdt_max_raw_length",
                        PDT_DEFAULT_MAX_RAW_LENGTH,
                    )
                ),
                "ac_pssm_lag": int(getattr(self.args, "ac_pssm_lag", 10)),
                "cc_pssm_lag": int(getattr(self.args, "cc_pssm_lag", 10)),
                "acc_pssm_lag": int(getattr(self.args, "acc_pssm_lag", 10)),
                "ac_pssm_blosum62_hybrid_lag": int(
                    getattr(self.args, "ac_pssm_blosum62_hybrid_lag", 10)
                ),
                "cc_pssm_blosum62_hybrid_lag": int(
                    getattr(self.args, "cc_pssm_blosum62_hybrid_lag", 10)
                ),
                "acc_pssm_blosum62_hybrid_lag": int(
                    getattr(self.args, "acc_pssm_blosum62_hybrid_lag", 10)
                ),
                "covariance_max_raw_length": validate_covariance_max_raw_length(
                    getattr(
                        self.args,
                        "covariance_max_raw_length",
                        COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
                    )
                ),
            }
        elif tokenizer == "dr":
            max_distance = validate_dr_max_distance(
                getattr(
                    self.args,
                    "dr_max_distance",
                    DR_DEFAULT_MAX_DISTANCE,
                )
            )
            dr_encoder = str(getattr(self.args, "encoder", "index") or "index").strip().lower()
            max_raw_length = validate_dr_max_raw_length(
                (
                    getattr(
                        self.args,
                        "frequency_max_raw_length",
                        FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
                    )
                    if dr_encoder in {"bow", "tfidf"}
                    else getattr(
                        self.args,
                        "dr_max_raw_length",
                        DR_DEFAULT_MAX_RAW_LENGTH,
                    )
                )
            )
            info["classic"] = {
                "tokenizer": "dr",
                "encoder": dr_encoder,
                "max_length": int(getattr(self.args, "max_length", 60)),
                "dr_max_distance": max_distance,
                "dr_max_raw_length": max_raw_length,
                "dr_contract": dr_contract_metadata(
                    max_distance=max_distance,
                    max_raw_length=max_raw_length,
                ),
                "max_length_unit": (
                    DR_GLOBAL_MAX_LENGTH_UNIT
                    if str(
                        getattr(
                            self.args,
                            "encoder",
                            "index",
                        )
                        or "index"
                    )
                    .strip()
                    .lower()
                    in {"bow", "tfidf", "textrank"}
                    else DR_ORDERED_MAX_LENGTH_UNIT
                ),
            }
        elif tokenizer == "top-n-gram":
            top_n = validate_top_ngram_n(getattr(self.args, "top_ngram_n", 2))
            info["classic"] = {
                "tokenizer": "top-n-gram",
                "encoder": str(getattr(self.args, "encoder", "bow") or "bow").strip().lower(),
                "max_length": int(getattr(self.args, "max_length", 60)),
                "max_length_unit": TOP_N_GRAM_MAX_LENGTH_UNIT,
                "top_ngram_n": top_n,
                "feature_dim": top_ngram_feature_dim(top_n),
                "frequency_profile_index_path": str(
                    getattr(
                        self.args,
                        "frequency_profile_index_path",
                        "",
                    )
                    or ""
                ),
                "top_ngram_contract": {
                    "contract_version": TOP_N_GRAM_CONTRACT_VERSION,
                    "profile_type": "target_frequency",
                    "tie_policy": TOP_N_GRAM_TIE_POLICY,
                    "full_length": True,
                    "fallback_profiles_allowed": (
                        str(
                            getattr(
                                self.args,
                                "profile_on_missing",
                                TOP_N_GRAM_FALLBACK_POLICY,
                            )
                        )
                        == TOP_N_GRAM_FALLBACK_POLICY
                    ),
                    "fallback_policy": (
                        TOP_N_GRAM_FALLBACK_POLICY_VERSION
                        if str(
                            getattr(
                                self.args,
                                "profile_on_missing",
                                TOP_N_GRAM_FALLBACK_POLICY,
                            )
                        )
                        == TOP_N_GRAM_FALLBACK_POLICY
                        else TOP_N_GRAM_STRICT_PROFILE_POLICY
                    ),
                    "feature_dim": top_ngram_feature_dim(top_n),
                    "profile_manifest_sha256": str(
                        getattr(
                            self.args,
                            "_frequency_profile_manifest_sha256",
                            "",
                        )
                        or ""
                    ),
                },
            }
        elif tokenizer == "dt":
            max_distance = validate_dt_max_distance(
                getattr(
                    self.args,
                    "dt_max_distance",
                    DT_DEFAULT_MAX_DISTANCE,
                )
            )
            contract = dt_contract_metadata(max_distance=max_distance)
            contract.update(
                {
                    "fallback_profiles_allowed": (
                        str(
                            getattr(
                                self.args,
                                "profile_on_missing",
                                TOP_N_GRAM_FALLBACK_POLICY,
                            )
                        )
                        == TOP_N_GRAM_FALLBACK_POLICY
                    ),
                    "fallback_policy": (
                        TOP_N_GRAM_FALLBACK_POLICY_VERSION
                        if str(
                            getattr(
                                self.args,
                                "profile_on_missing",
                                TOP_N_GRAM_FALLBACK_POLICY,
                            )
                        )
                        == TOP_N_GRAM_FALLBACK_POLICY
                        else TOP_N_GRAM_STRICT_PROFILE_POLICY
                    ),
                    "profile_manifest_sha256": str(
                        getattr(
                            self.args,
                            "_frequency_profile_manifest_sha256",
                            "",
                        )
                        or ""
                    ),
                }
            )
            info["classic"] = {
                "tokenizer": "dt",
                "encoder": str(getattr(self.args, "encoder", "bow") or "bow").strip().lower(),
                "max_length": int(getattr(self.args, "max_length", 60)),
                "max_length_unit": DT_MAX_LENGTH_UNIT,
                "dt_max_distance": max_distance,
                "feature_dim": contract["feature_dim"],
                "frequency_profile_index_path": str(
                    getattr(
                        self.args,
                        "frequency_profile_index_path",
                        "",
                    )
                    or ""
                ),
                "dt_contract": contract,
            }
        elif tokenizer == "bpe":
            artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
            if artifact_dir:
                classic = {
                    "tokenizer": "bpe",
                    "encoder": str(getattr(self.args, "encoder", "index") or "index"),
                    "max_length": int(getattr(self.args, "max_length", 60)),
                }
                classic.update(
                    bundle_bpe_artifact(
                        self.bpe_path,
                        artifact_dir,
                    )
                )
                info["classic"] = classic
        elif tokenizer == "wordpiece":
            artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
            if artifact_dir:
                classic = {
                    "tokenizer": "wordpiece",
                    "encoder": str(getattr(self.args, "encoder", "index") or "index"),
                    "max_length": int(getattr(self.args, "max_length", 60)),
                    "max_length_unit": WORDPIECE_MAX_LENGTH_UNIT,
                    "wordpiece_vocab_size": int(getattr(self.args, "wordpiece_vocab_size", 1000)),
                    "wordpiece_max_raw_length": (
                        validate_wordpiece_max_raw_length(
                            getattr(
                                self.args,
                                "wordpiece_max_raw_length",
                                WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
                            )
                        )
                    ),
                }
                classic.update(
                    bundle_wordpiece_artifact(
                        self.wordpiece_path,
                        artifact_dir,
                    )
                )
                info["classic"] = classic
        elif tokenizer == "unigram":
            artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
            if artifact_dir:
                classic = {
                    "tokenizer": "unigram",
                    "encoder": str(getattr(self.args, "encoder", "index") or "index"),
                    "max_length": int(getattr(self.args, "max_length", 60)),
                    "max_length_unit": UNIGRAM_MAX_LENGTH_UNIT,
                    "unigram_vocab_size": int(getattr(self.args, "unigram_vocab_size", 1000)),
                    "unigram_max_raw_length": (
                        validate_unigram_max_raw_length(
                            getattr(
                                self.args,
                                "unigram_max_raw_length",
                                UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
                            )
                        )
                    ),
                    "training_regularization": {
                        "enabled": bool(getattr(self.args, "unigram_sampling", False)),
                        "nbest_size": int(
                            getattr(
                                self.args,
                                "unigram_nbest_size",
                                UNIGRAM_DEFAULT_NBEST_SIZE,
                            )
                        ),
                        "alpha": float(
                            getattr(
                                self.args,
                                "unigram_alpha",
                                UNIGRAM_DEFAULT_ALPHA,
                            )
                        ),
                        "inference_mode": "viterbi",
                    },
                }
                classic.update(
                    bundle_unigram_artifact(
                        self.unigram_path,
                        artifact_dir,
                    )
                )
                info["classic"] = classic
        if profile_info:
            info.update(profile_info)
        static_info = self._static_embedding_artifact_info()
        static_encoder = str(getattr(self.args, "encoder", "") or "").strip().lower()
        if static_encoder in {"word2vec", "glove", "fasttext"} and "classic" in info:
            bundled = self._bundle_dr_static_embedding_artifact()
            if bundled:
                info["classic"].update(bundled["classic"])
                info["classic"]["static_embedding"] = bundled["static_embedding"]
                static_info = {"static_embedding": bundled["static_embedding"]}
        if "classic" in info:
            if str(info["classic"].get("encoder") or "").strip().lower() == "blosum62":
                info["classic"]["blosum62_contract"] = blosum62_contract_metadata(
                    int(info["classic"].get("k", 1))
                )
            frequency_contract = getattr(
                self.args,
                "_frequency_encoder_contract",
                None,
            )
            if frequency_contract:
                info["classic"]["frequency_encoder_contract"] = dict(frequency_contract)
                info["classic"]["frequency_max_raw_length"] = int(
                    frequency_contract["frequency_max_raw_length"]
                )
            textrank_contract = getattr(
                self.args,
                "_textrank_contract",
                None,
            )
            if textrank_contract:
                info["classic"]["textrank_contract"] = dict(textrank_contract)
            covariance_contract = getattr(
                self.args,
                "_covariance_encoder_contract",
                None,
            )
            if covariance_contract:
                info["classic"]["covariance_encoder_contract"] = dict(covariance_contract)
            pdt_contract = getattr(
                self.args,
                "_pdt_encoder_contract",
                None,
            )
            if pdt_contract:
                info["classic"]["pdt_encoder_contract"] = dict(pdt_contract)
            pssm_covariance_contract = getattr(
                self.args,
                "_pssm_covariance_encoder_contract",
                None,
            )
            if pssm_covariance_contract:
                info["classic"]["pssm_covariance_encoder_contract"] = dict(pssm_covariance_contract)
            idf = getattr(self.args, "_classic_tfidf_idf", None)
            artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
            if idf is not None and artifact_dir:
                descriptor = save_idf_state(
                    idf,
                    artifact_dir=artifact_dir,
                    filename="classic_tfidf_idf.pt",
                )
                info["classic"]["tfidf_idf"] = descriptor
                info["classic"]["tfidf_idf_path"] = descriptor["path"]
        info.update(static_info)
        return info

    def _static_embedding_artifact_info(self):
        provenance = getattr(self.args, "_static_embedding_provenance", None)
        return {"static_embedding": dict(provenance)} if provenance else {}

    def _bundle_dr_static_embedding_artifact(self):
        encoder = str(getattr(self.args, "encoder", "") or "").strip().lower()
        if encoder not in {"word2vec", "glove", "fasttext"}:
            return {}
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        source = str(getattr(self.args, f"{encoder}_path", "") or "").strip()
        if not artifact_dir or not source or not os.path.isfile(source):
            return {}
        target_dir = os.path.join(artifact_dir, "classic")
        os.makedirs(target_dir, exist_ok=True)
        extension = ".bin" if encoder == "fasttext" else ".vec"
        target = os.path.join(target_dir, f"static_{encoder}{extension}")
        if os.path.abspath(source) != os.path.abspath(target):
            shutil.copy2(source, target)
        relative = os.path.relpath(target, artifact_dir)
        provenance = dict(
            getattr(
                self.args,
                "_static_embedding_provenance",
                {},
            )
            or {}
        )
        provenance.update(
            {
                "method": encoder,
                "path": relative,
                "sha256": sha256_file(target),
                "bundled": True,
            }
        )
        return {
            "classic": {f"{encoder}_path": relative},
            "static_embedding": provenance,
        }

    def _semantic_artifact_metadata(self, model):
        provenance = dict(getattr(self, "_semantic_provenance", None) or {})
        if not provenance:
            return None
        return {
            **provenance,
            "representation": semantic_representation_contract(
                model,
                relation_type=str(provenance.get("relation_type") or ""),
                pair_mode=str(provenance.get("pair_mode") or "fused"),
            ),
            "main_batch_sampler": {
                "implementation": "torch.utils.data.RandomSampler",
                "seed": int(getattr(self.args, "seed", 42)),
                "epoch_order_hashes": list(
                    getattr(self.args, "_main_batch_order_hashes", []) or []
                ),
            },
        }

    def _save_pretrained_embedding_nn_artifact(
        self, model, artifact_info, dl_scaler=None, operating_point=None
    ):
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        if not artifact_dir:
            return
        if normalize_task_type(getattr(self.args, "task_type", "")) != "pep_function":
            print(
                "[artifact] reusable pretrained embedding bundle skipped: currently supported for pep_function only"
            )
            return
        os.makedirs(artifact_dir, exist_ok=True)
        original_device = self.device
        model_cpu = model.to("cpu")
        torch.save(model_cpu, os.path.join(artifact_dir, "model.pt"))
        model.to(original_device)
        metadata = {
            "bundle_type": "pretrained_embedding_nn",
            "supports_inference": True,
            "task_type": normalize_task_type(self.args.task_type),
            "args": self._jsonable_args(),
            "route_schema_version": ROUTE_SCHEMA_VERSION,
            "effective_route": resolve_effective_route(vars(self.args), self.args.task_type),
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "checkpoint_metric": self._checkpoint_metric(),
            **selection_fields(self.args),
            "feature_fit_scope": "train_only",
            "evaluation_mode": "independent_iid",
            **artifact_info,
        }
        head_config = prediction_head_artifact_config(self.args)
        if head_config is not None:
            metadata["prediction_head"] = head_config
        if getattr(self.args, "_resolved_multilabel_loss", None):
            metadata["multilabel_loss"] = dict(self.args._resolved_multilabel_loss)
        semantic_metadata = self._semantic_artifact_metadata(model)
        if semantic_metadata is not None:
            metadata["semantic"] = semantic_metadata
        if operating_point is not None:
            metadata["operating_point"] = operating_point.to_dict()
            metadata.update(
                {
                    key: value
                    for key, value in operating_point.to_dict().items()
                    if key in {"threshold", "thresholds", "threshold_by_label", "threshold_source"}
                }
            )
        if dl_scaler is not None:
            joblib.dump({"dl_scaler": dl_scaler}, os.path.join(artifact_dir, "preprocess.joblib"))
            metadata["preprocess_path"] = "preprocess.joblib"
        with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
        tag = artifact_info.get("pretrained", {}).get("tokenizer", "pretrained")
        print(f"[artifact] saved reusable {tag} embedding model bundle: {artifact_dir}")

    def _save_sequence_nn_artifact(self, model, train_data, dl_scaler=None, operating_point=None):
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        if not artifact_dir:
            return
        if normalize_task_type(getattr(self.args, "task_type", "")) != "pep_function":
            print(
                "[artifact] reusable inference bundle skipped: currently supported for pep_function only"
            )
            return
        if exports_sequence_embeddings(self.args):
            print(
                "[artifact] reusable inference bundle skipped: export_embeddings final predictor is not yet serialized"
            )
            return
        os.makedirs(artifact_dir, exist_ok=True)
        original_device = self.device
        model_cpu = model.to("cpu")
        torch.save(model_cpu, os.path.join(artifact_dir, "model.pt"))
        model.to(original_device)
        artifact_source = str(getattr(self.args, "_artifact_source", "sequence_features"))
        pretrained_residue = artifact_source == "pretrained_residue_embeddings"
        artifact_info = dict(getattr(self.args, "_artifact_info", {}) or {})
        profile_info = (
            {} if pretrained_residue else self._sequence_profile_generation_artifact_info()
        )
        supports_inference = (
            True if pretrained_residue else self._profile_inference_supported(profile_info)
        )
        metadata = {
            "bundle_type": "pretrained_residue_nn" if pretrained_residue else "sequence_nn",
            "supports_inference": supports_inference,
            "task_type": normalize_task_type(self.args.task_type),
            "args": self._jsonable_args(),
            "route_schema_version": ROUTE_SCHEMA_VERSION,
            "effective_route": resolve_effective_route(vars(self.args), self.args.task_type),
            "label_names": list(getattr(train_data, "label_names", [])),
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "checkpoint_metric": self._checkpoint_metric(),
            **selection_fields(self.args),
            "feature_fit_scope": "train_only",
            "evaluation_mode": "independent_iid",
            **self._static_embedding_artifact_info(),
        }
        predictor_contract = sequence_predictor_contract(model)
        if predictor_contract is not None:
            metadata["predictor_contract"] = predictor_contract
        head_config = prediction_head_artifact_config(self.args)
        if head_config is not None:
            metadata["prediction_head"] = head_config
        if getattr(self.args, "_resolved_multilabel_loss", None):
            metadata["multilabel_loss"] = dict(self.args._resolved_multilabel_loss)
        metadata.update(artifact_info)
        semantic_metadata = self._semantic_artifact_metadata(model)
        if semantic_metadata is not None:
            metadata["semantic"] = semantic_metadata
        if str(getattr(self.args, "model", "")).strip().lower() in {"gcn", "gat"}:
            metadata["graph"] = {
                "kind": str(self.args.model).lower(),
                "node_definition": "one node per residue",
                "edges": "bidirectional adjacent residues plus self-loops",
                "layers": 2,
                "hidden_dim": 128,
                "heads": 4 if str(self.args.model).lower() == "gat" else 1,
                "pooling": "mean+max",
                "padding_mask": "explicit residue mask",
                "pretrained_token_alignment": getattr(self.args, "_graph_token_alignment", ""),
            }
        if operating_point is not None:
            metadata["operating_point"] = operating_point.to_dict()
            metadata.update(
                {
                    key: value
                    for key, value in operating_point.to_dict().items()
                    if key in {"threshold", "thresholds", "threshold_by_label", "threshold_source"}
                }
            )
        metadata.update(profile_info)
        if not supports_inference:
            metadata["reason"] = PROFILE_DEPENDENT_INFERENCE_REASON
            metadata["reason_code"] = "profile_features_require_sample_id"
            metadata["inference_unavailable_reason"] = PROFILE_DEPENDENT_INFERENCE_REASON
            metadata["inference_unavailable_code"] = "profile_features_require_sample_id"
        if dl_scaler is not None:
            joblib.dump({"dl_scaler": dl_scaler}, os.path.join(artifact_dir, "preprocess.joblib"))
            metadata["preprocess_path"] = "preprocess.joblib"
        with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
        if supports_inference:
            print(
                f"[artifact] saved reusable {metadata['bundle_type']} model bundle: {artifact_dir}"
            )
        else:
            print(
                f"[artifact] saved sequence model bundle without Web raw-sequence inference: {artifact_dir}"
            )

    def _save_sequence_embedding_encoder(self, model):
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        if not artifact_dir:
            return ""
        os.makedirs(artifact_dir, exist_ok=True)
        path = os.path.join(artifact_dir, "embedding_encoder.pt")
        original_device = self.device
        model_cpu = model.to("cpu")
        torch.save(model_cpu, path)
        model.to(original_device)
        return os.path.relpath(path, artifact_dir)

    def _sequence_embedding_artifact_info(
        self,
        train_data,
        encoder_path,
        embedding_dim,
        downstream,
        dl_scaler=None,
        profile_info=None,
        architecture_contract=None,
    ):
        info = {
            "input_schema": "peptide_sequences",
            "label_names": list(getattr(train_data, "label_names", [])),
            "embedding_export": {
                "encoder_path": encoder_path,
                "embedding_dim": int(embedding_dim or 0),
                "downstream": str(downstream or "").strip().lower(),
            },
        }
        if architecture_contract is not None:
            info["embedding_export"]["architecture_contract"] = architecture_contract
        if profile_info:
            info.update(profile_info)
        info.update(self._static_embedding_artifact_info())
        if dl_scaler is not None:
            info["dl_scaler"] = dl_scaler
        return info

    def _compute_aupr_from_scores(self, y_true, scores):
        try:
            metrics = Evaluator.ranking_metrics(y_true, scores)
            value = metrics.get("aupr")
            return float(value) if value is not None else float("nan")
        except Exception:
            return float("nan")

    def _operating_point_objective(self):
        return resolve_selection(self.args).threshold

    def _checkpoint_metric(self):
        return resolve_selection(self.args).checkpoint.checkpoint_metric

    def _multilabel_loss(self, train_targets, *, auto_default="bce"):
        """Resolve an explicit loss while preserving each route's legacy default.

        Generic sequence training historically uses BCE; pretrained pooled
        MLPs historically use positive-class-weighted BCE with a cap of 20.
        """
        requested = str(getattr(self.args, "multilabel_loss", "auto") or "auto").strip().lower()
        resolved = auto_default if requested == "auto" else requested
        contract = {
            "requested": requested,
            "resolved": resolved,
            "fit_scope": "train_only",
        }
        if resolved == "bce":
            criterion = nn.BCEWithLogitsLoss()
        elif resolved == "balanced_bce":
            targets = torch.as_tensor(train_targets, dtype=torch.float32)
            positive = targets.sum(dim=0)
            negative = targets.shape[0] - positive
            cap = float(getattr(self.args, "multilabel_pos_weight_cap", 20.0))
            if not np.isfinite(cap) or cap < 1.0:
                raise ValueError("multilabel_pos_weight_cap must be finite and >= 1")
            weights = torch.clamp(negative / (positive + 1e-6), min=1.0, max=cap)
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(self.device))
            contract.update({"pos_weight_cap": cap, "pos_weight": weights.tolist()})
        elif resolved == "asymmetric_focal":
            gamma_pos = float(getattr(self.args, "asl_gamma_pos", 1.0))
            gamma_neg = float(getattr(self.args, "asl_gamma_neg", 4.0))
            clip = float(getattr(self.args, "asl_clip", 0.05))
            if (
                not np.isfinite(gamma_pos)
                or not np.isfinite(gamma_neg)
                or gamma_pos < 0
                or gamma_neg < 0
            ):
                raise ValueError("ASL gamma values must be finite and non-negative")
            if not np.isfinite(clip) or not 0.0 <= clip < 1.0:
                raise ValueError("asl_clip must be finite and in [0, 1)")
            criterion = AsymmetricFocalLoss(
                gamma_pos=gamma_pos,
                gamma_neg=gamma_neg,
                clip=clip,
            )
            contract.update({"gamma_pos": gamma_pos, "gamma_neg": gamma_neg, "clip": clip})
        else:  # protected by argparse; retained for programmatic callers
            raise ValueError(f"unsupported multilabel loss: {resolved}")
        self.args._resolved_multilabel_loss = contract
        return criterion

    def _validation_metrics_for_checkpoint(self, model, loader):
        targets, scores = collect_nn_scores(model, loader, self.device)
        config = resolve_selection(self.args)
        metrics, _point = Evaluator.fit_and_score(
            targets,
            scores,
            score_kind="probability",
            objective=config.checkpoint_threshold,
            search_profile=config.checkpoint_search_profile,
        )
        return metrics

    def _update_validation_checkpoint(
        self, model, *, val_metrics, epoch, best_key, best_epoch, best_state
    ):
        key = resolve_selection(self.args).checkpoint.key(val_metrics)
        if not np.all(np.isfinite(np.asarray(key, dtype=np.float64))):
            raise RuntimeError(
                f"validation checkpoint key is non-finite at epoch {int(epoch)}: {key!r}"
            )
        minimum_delta = float(self.args.early_stop_min_delta)
        improved = checkpoint_improved(key, None if best_state is None else best_key, minimum_delta)
        history = [] if best_state is None else getattr(self, "_selection_history", [])
        history.append(
            {
                "epoch": int(epoch),
                "selection_key": list(key),
                "metrics": val_metrics,
                "improved": bool(improved),
            }
        )
        self._selection_history = history
        self.args._selected_epoch = int(epoch) if improved else int(best_epoch)
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "")
        if artifact_dir:
            os.makedirs(artifact_dir, exist_ok=True)
            with open(os.path.join(artifact_dir, "selection_history.json"), "w") as handle:
                json.dump(
                    {
                        **selection_fields(self.args),
                        "early_stop_min_delta": minimum_delta,
                        "best_epoch": int(epoch) if improved else int(best_epoch),
                        "epochs": history,
                    },
                    handle,
                    indent=2,
                    allow_nan=False,
                )
        if not improved:
            return best_key, best_epoch, best_state, False
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        return tuple(float(value) for value in key), int(epoch), state, True

    def _fit_nn_operating_point(self, model, loader, label_names=None):
        targets, scores = collect_nn_scores(model, loader, self.device)
        operating_point = Evaluator.fit_operating_point(
            targets,
            scores,
            score_kind="probability",
            label_names=label_names,
            objective=self._operating_point_objective(),
            search_profile=resolve_selection(self.args).search_profile,
        )
        return targets, scores, operating_point

    def _save_protocol_evaluation(
        self, val_y, val_scores, operating_point, test_y=None, test_scores=None
    ):
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        if not artifact_dir:
            return
        os.makedirs(artifact_dir, exist_ok=True)
        payload = {
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "checkpoint_metric": self._checkpoint_metric(),
            **selection_fields(self.args),
            "feature_fit_scope": "train_only",
            "evaluation_mode": "independent_iid",
            "operating_point": operating_point.to_dict(),
            "validation": Evaluator.score_metrics(val_y, val_scores, operating_point),
        }
        arrays = {
            "validation_labels": np.asarray(val_y, dtype=np.int32),
            "validation_scores": np.asarray(val_scores, dtype=np.float32),
        }
        if test_y is not None and test_scores is not None:
            payload["test"] = Evaluator.score_metrics(test_y, test_scores, operating_point)
            arrays["test_labels"] = np.asarray(test_y, dtype=np.int32)
            arrays["test_scores"] = np.asarray(test_scores, dtype=np.float32)
        with open(os.path.join(artifact_dir, "evaluation.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        np.savez_compressed(os.path.join(artifact_dir, "evaluation_scores.npz"), **arrays)

    def _write_score_analysis(
        self, val_y, val_scores, test_y=None, test_scores=None, operating_point=None
    ):
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        if not artifact_dir:
            return
        try:
            save_analysis(
                artifact_dir,
                build_score_analysis(
                    val_y,
                    val_scores,
                    test_y,
                    test_scores,
                    operating_point=operating_point,
                ),
            )
        except Exception as exc:
            print(f"[analysis] skipped score analysis artifact: {exc}", flush=True)

    def _val_aupr_suffix(self, model, loader):
        if not getattr(self.args, "report_aupr", True):
            return ""
        y_true, scores = collect_nn_scores(model, loader, self.device)
        return f" | Val AUPR: {self._compute_aupr_from_scores(y_true, scores):.4f}"

    def _test_aupr_suffix(self, model, loader):
        if not getattr(self.args, "report_aupr", True):
            return ""
        y_true, scores = collect_nn_scores(model, loader, self.device)
        return f" | AUPR: {self._compute_aupr_from_scores(y_true, scores):.4f}"

    def _update_validation_f1_checkpoint(
        self,
        model,
        *,
        val_f1,
        epoch,
        best_val_f1,
        best_epoch,
        best_state,
        val_metrics=None,
    ):
        if val_metrics is not None:
            key, selected_epoch, state, improved = self._update_validation_checkpoint(
                model,
                val_metrics=val_metrics,
                epoch=epoch,
                best_key=getattr(self, "_pooled_checkpoint_key", None),
                best_epoch=best_epoch,
                best_state=best_state,
            )
            self._pooled_checkpoint_key = key
            return (float(val_f1) if improved else best_val_f1), selected_epoch, state, improved
        val_f1 = float(val_f1)
        if not np.isfinite(val_f1):
            raise RuntimeError(
                f"Validation F1 is non-finite at epoch {int(epoch)}: "
                f"{val_f1!r}; refusing to select a checkpoint or evaluate "
                "the test split"
            )
        improved = best_state is None or val_f1 > float(best_val_f1) + float(
            self.args.early_stop_min_delta
        )
        if not improved:
            return best_val_f1, best_epoch, best_state, False
        return (
            val_f1,
            int(epoch),
            {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            True,
        )

    @staticmethod
    def _restore_validation_f1_checkpoint(model, best_state, best_epoch):
        if best_state is None or int(best_epoch) < 1:
            raise RuntimeError(
                "training completed without a valid validation-F1 "
                "checkpoint; refusing to save a final model or evaluate "
                "the test split"
            )
        model.load_state_dict(best_state)

    def _write_tensor_nn_analysis(self, model, val_ds, test_ds=None):
        val_loader = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False)
        val_targets, val_scores = collect_nn_scores(model, val_loader, self.device)
        operating_point = Evaluator.fit_operating_point(
            val_targets, val_scores, score_kind="probability"
        )
        test_targets, test_scores = None, None
        if test_ds is not None:
            test_loader = DataLoader(test_ds, batch_size=self.args.batch_size, shuffle=False)
            test_targets, test_scores = collect_nn_scores(model, test_loader, self.device)
        self._write_score_analysis(
            val_targets,
            val_scores,
            test_targets,
            test_scores,
            operating_point=operating_point,
        )

    def _finalize_pretrained_embedding_nn(
        self, tag, model, val_ds, test_ds, criterion, artifact_info, dl_scaler
    ):
        val_loader = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False)
        val_targets, val_scores, operating_point = self._fit_nn_operating_point(
            model,
            val_loader,
            label_names=artifact_info.get("label_names", []),
        )
        self._save_pretrained_embedding_nn_artifact(
            model,
            artifact_info,
            dl_scaler,
            operating_point=operating_point,
        )
        test_targets, test_scores = None, None
        if test_ds is not None:
            test_loader = DataLoader(test_ds, batch_size=self.args.batch_size, shuffle=False)
            test_targets, test_scores = collect_nn_scores(model, test_loader, self.device)
            test_metrics = Evaluator.score_metrics(test_targets, test_scores, operating_point)
            test_loss = self.eval_loss(model, test_loader, criterion)
            print(
                f"[{tag}/mlp] Test | Loss: {test_loss:.4f} | Acc: {test_metrics['acc']:.4f} | "
                f"SAcc: {test_metrics['sacc']:.4f} | F1: {test_metrics['f1']:.4f} | "
                f"AUC: {float(test_metrics.get('auc') or 0.0):.4f}"
                + Evaluator.format_extra_metrics(test_metrics)
                + (
                    f" | AUPR: {float(test_metrics.get('aupr') or 0.0):.4f}"
                    if getattr(self.args, "report_aupr", True)
                    else ""
                )
            )
        self._save_protocol_evaluation(
            val_targets, val_scores, operating_point, test_targets, test_scores
        )
        self._write_score_analysis(
            val_targets,
            val_scores,
            test_targets,
            test_scores,
            operating_point=operating_point,
        )

    def __init__(self, args):
        self.args = args
        self.args.task_type = normalize_task_type(self.args.task_type)
        self.args.task_type = normalize_task_type(getattr(self.args, "task_type", "pep_function"))
        if normalize_task_type(self.args.task_type) == "pep_function_multilabel":
            self.args.task_type = "pep_function"
        if normalize_task_type(self.args.task_type) in [
            "pep_hlaii_interaction",
            "peptide_hlaii_interaction",
        ]:
            self.args.task_type = "pep_hlaii"
        if normalize_task_type(self.args.task_type) in [
            "pep_hlai_interaction",
            "peptide_hlai_interaction",
        ]:
            self.args.task_type = "pep_hlai"
        if normalize_task_type(self.args.task_type) in PAIR_TASK_TYPES:
            raise SystemExit(
                f"{self.args.task_type} is a pair-interaction task. "
                "Use scripts/train_pair_pipeline.py or the dispatcher scripts/train_pipeline.py."
            )
        normalize_workflow_args(self.args)
        normalize_semantic_aliases(self.args)
        if semantic_is_enabled(self.args):
            model_name = str(getattr(self.args, "model", "") or "").strip().lower()
            training_mode = str(getattr(self.args, "training_mode", "") or "").strip().lower()
            classifier = str(getattr(self.args, "classifier", "") or "").strip().lower()
            if (
                model_name not in self.SEMANTIC_SIMILARITY_MODELS
                or training_mode == "export_embeddings"
                or classifier in CLASSICAL_CLASSIFIERS
            ):
                raise SystemExit(
                    "The unified semantic layer requires an end-to-end differentiable neural "
                    "classifier with a sample-level embedding; classical and export-only routes "
                    "are unsupported."
                )
            loss_space = (
                str(getattr(self.args, "semantic_loss_space", "projection")).strip().lower()
            )
            residual_weight = float(getattr(self.args, "semantic_residual_weight", 0.0))
            partner_scheme = (
                str(getattr(self.args, "semantic_partner_scheme", "fixed")).strip().lower()
            )
            if residual_weight > 0.0 and loss_space == "shared":
                raise SystemExit(
                    "semantic_residual_weight and semantic_loss_space=shared are mutually exclusive"
                )
            if residual_weight > 0.0:
                if normalize_task_type(getattr(self.args, "task_type", "")) != "pep_function":
                    raise SystemExit(
                        "semantic_residual_weight currently supports pep_function only"
                    )
                if model_name != "mlp" or bool(getattr(self.args, "sequence_model", False)):
                    raise SystemExit(
                        "semantic_residual_weight currently requires a pooled-embedding MLP route"
                    )
            if partner_scheme in {"dense_hard", "placebo"}:
                if (
                    normalize_task_type(getattr(self.args, "task_type", "")) != "pep_function"
                    or model_name != "mlp"
                    or bool(getattr(self.args, "sequence_model", False))
                ):
                    raise SystemExit(
                        "semantic_partner_scheme dense_hard/placebo currently "
                        "requires a pep_function pooled-embedding MLP route"
                    )
        self._semantic_profile = None
        self._semantic_train_inputs = None
        self._semantic_train_dataset = None
        self._semantic_provenance = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        task_root_map = {
            "pep_function": "datasets/task_peptide_function",
        }
        self.task_root_rel = task_root_map.get(
            self.args.task_type, f"datasets/task_{self.args.task_type}"
        )
        raw_bpe_path = getattr(self.args, "bpe_path", "")
        if raw_bpe_path:
            self.bpe_path = (
                raw_bpe_path
                if os.path.isabs(raw_bpe_path)
                else os.path.join(self.project_root, raw_bpe_path)
            )
        else:
            bpe_candidates = [
                os.path.join(
                    self.project_root, self.task_root_rel, "bpe_tokenizers", "bpe_tokenizer.json"
                ),
                os.path.join(self.project_root, "datasets", "bpe_tokenizer.json"),
            ]
            self.bpe_path = next(
                (p for p in bpe_candidates if os.path.exists(p)), bpe_candidates[0]
            )

        raw_data_dir = getattr(self.args, "data_dir", "") or f"{self.task_root_rel}/data"
        if os.path.isabs(raw_data_dir):
            self.args.data_dir = raw_data_dir
        else:
            self.args.data_dir = os.path.join(self.project_root, raw_data_dir)
        self._ensure_hlaii_test_alias()
        raw_unigram_path = getattr(self.args, "unigram_path", "")
        if raw_unigram_path:
            unigram_path = raw_unigram_path
            if not os.path.isabs(unigram_path):
                unigram_path = os.path.join(self.project_root, unigram_path)
        else:
            unigram_path = os.path.join(
                self.project_root,
                self.task_root_rel,
                "unigram_tokenizers",
                "unigram_tokenizer.json",
            )
        self.unigram_path = unigram_path
        raw_wordpiece_path = getattr(self.args, "wordpiece_path", "")
        if raw_wordpiece_path:
            wordpiece_path = raw_wordpiece_path
            if not os.path.isabs(wordpiece_path):
                wordpiece_path = os.path.join(self.project_root, wordpiece_path)
        else:
            wordpiece_path = os.path.join(
                self.project_root,
                self.task_root_rel,
                "wordpiece_tokenizers",
                "wordpiece_tokenizer.json",
            )
        self.wordpiece_path = wordpiece_path
        esm_dir = getattr(self.args, "esm_feature_dir", "") or self._default_pretrained_feature_dir(
            "esm"
        )
        if not os.path.isabs(esm_dir):
            esm_dir = os.path.join(self.project_root, esm_dir)
        self.esm_feature_dir = esm_dir
        protbert_dir = (
            getattr(self.args, "protbert_feature_dir", "")
            or getattr(self.args, "probert_feature_dir", "")
            or self._default_pretrained_feature_dir("protbert")
        )
        if not os.path.isabs(protbert_dir):
            protbert_dir = os.path.join(self.project_root, protbert_dir)
        self.protbert_feature_dir = protbert_dir
        prott5_dir = getattr(
            self.args, "prott5_feature_dir", ""
        ) or self._default_pretrained_feature_dir("prott5")
        if not os.path.isabs(prott5_dir):
            prott5_dir = os.path.join(self.project_root, prott5_dir)
        self.prott5_feature_dir = prott5_dir
        esmc_dir = getattr(
            self.args, "esmc_feature_dir", ""
        ) or self._default_pretrained_feature_dir("esmc")
        if not os.path.isabs(esmc_dir):
            esmc_dir = os.path.join(self.project_root, esmc_dir)
        self.esmc_feature_dir = esmc_dir
        esm3_dir = getattr(
            self.args, "esm3_feature_dir", ""
        ) or self._default_pretrained_feature_dir("esm3")
        if not os.path.isabs(esm3_dir):
            esm3_dir = os.path.join(self.project_root, esm3_dir)
        self.esm3_feature_dir = esm3_dir

        def resolve_optional_path(raw_path):
            if not raw_path:
                return ""
            if os.path.isabs(raw_path):
                return raw_path
            return os.path.join(self.project_root, raw_path)

        self._resolve_optional_path = resolve_optional_path

        raw_word2vec_argument = str(getattr(self.args, "word2vec_path", "") or "").strip()
        raw_glove_argument = str(getattr(self.args, "glove_path", "") or "").strip()
        raw_fasttext_argument = str(getattr(self.args, "fasttext_path", "") or "").strip()
        self._explicit_static_embedding_paths = {
            "word2vec": resolve_optional_path(raw_word2vec_argument),
            "glove": resolve_optional_path(raw_glove_argument),
            "fasttext": resolve_optional_path(raw_fasttext_argument),
        }
        self._explicit_glove_path = self._explicit_static_embedding_paths["glove"]
        raw_word2vec_path = raw_word2vec_argument or f"{self.task_root_rel}/word2vec/word2vec.vec"
        raw_glove_path = raw_glove_argument or f"{self.task_root_rel}/glove/glove.vec"
        raw_fasttext_path = raw_fasttext_argument or f"{self.task_root_rel}/fasttext/fasttext.bin"
        profile_root = resolve_optional_path(
            getattr(self.args, "profile_cache_dir", "")
            or f"{self.task_root_rel}/profile_cache/default"
        )
        default_pssm_feature = (
            FEATURE_PSSM_BLOSUM62_HYBRID
            if str(getattr(self.args, "encoder", "") or "").strip().lower()
            == "pssm_blosum62_hybrid"
            else FEATURE_PSSM
        )
        raw_pssm_index_path = (
            getattr(self.args, "pssm_index_path", "")
            or f"{profile_root}/{default_pssm_feature}/index.json"
        )
        hybrid_covariance_index_paths = {
            feature: resolve_optional_path(
                getattr(self.args, f"{feature}_index_path", "")
                or f"{profile_root}/{feature}/index.json"
            )
            for feature in (
                FEATURE_AC_PSSM_BLOSUM62_HYBRID,
                FEATURE_CC_PSSM_BLOSUM62_HYBRID,
                FEATURE_ACC_PSSM_BLOSUM62_HYBRID,
            )
        }
        raw_psfm_index_path = (
            getattr(self.args, "psfm_index_path", "") or f"{profile_root}/psfm/index.json"
        )
        if str(
            getattr(self.args, "encoder", "") or ""
        ).strip().lower() == "psfm_blosum62_hybrid" and not getattr(
            self.args, "psfm_index_path", ""
        ):
            raw_psfm_index_path = f"{profile_root}/psfm_blosum62_hybrid/index.json"
        explicit_frequency_profile_index_path = str(
            getattr(self.args, "frequency_profile_index_path", "") or ""
        ).strip()
        self._explicit_frequency_profile_index_path = resolve_optional_path(
            explicit_frequency_profile_index_path
        )
        raw_frequency_profile_index_path = (
            explicit_frequency_profile_index_path or f"{profile_root}/target_frequency/index.json"
        )

        self.word2vec_path = resolve_optional_path(raw_word2vec_path)
        self.glove_path = resolve_optional_path(raw_glove_path)
        self.fasttext_path = resolve_optional_path(raw_fasttext_path)
        self.pssm_index_path = resolve_optional_path(raw_pssm_index_path)
        for feature, index_path in hybrid_covariance_index_paths.items():
            attribute = f"{feature}_index_path"
            setattr(self, attribute, index_path)
            setattr(self.args, attribute, index_path)
        self.psfm_index_path = resolve_optional_path(raw_psfm_index_path)
        self.frequency_profile_index_path = resolve_optional_path(raw_frequency_profile_index_path)
        self.args.frequency_profile_index_path = self.frequency_profile_index_path

        self.feat_factory = FeaturizerFactory(
            self.bpe_path,
            self.unigram_path,
            self.wordpiece_path,
            self.word2vec_path,
            self.glove_path,
            self.fasttext_path,
            self.pssm_index_path,
            self.psfm_index_path,
            getattr(self.args, "ac_lag", 10),
            getattr(self.args, "cc_lag", 10),
            getattr(self.args, "acc_lag", 10),
            getattr(self.args, "pdt_lag", PDT_DEFAULT_LAG),
            getattr(self.args, "ac_pssm_lag", 10),
            getattr(self.args, "cc_pssm_lag", 10),
            getattr(self.args, "acc_pssm_lag", 10),
            wordpiece_max_raw_length=validate_wordpiece_max_raw_length(
                getattr(
                    self.args,
                    "wordpiece_max_raw_length",
                    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            unigram_max_raw_length=validate_unigram_max_raw_length(
                getattr(
                    self.args,
                    "unigram_max_raw_length",
                    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            dr_max_raw_length=validate_dr_max_raw_length(
                getattr(
                    self.args,
                    "dr_max_raw_length",
                    DR_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            frequency_profile_index_path=self.frequency_profile_index_path,
            covariance_max_raw_length=validate_covariance_max_raw_length(
                getattr(
                    self.args,
                    "covariance_max_raw_length",
                    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            pdt_max_raw_length=validate_pdt_max_raw_length(
                getattr(
                    self.args,
                    "pdt_max_raw_length",
                    PDT_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            ac_pssm_blosum62_hybrid_index_path=(self.ac_pssm_blosum62_hybrid_index_path),
            cc_pssm_blosum62_hybrid_index_path=(self.cc_pssm_blosum62_hybrid_index_path),
            acc_pssm_blosum62_hybrid_index_path=(self.acc_pssm_blosum62_hybrid_index_path),
            ac_pssm_blosum62_hybrid_lag=getattr(self.args, "ac_pssm_blosum62_hybrid_lag", 10),
            cc_pssm_blosum62_hybrid_lag=getattr(self.args, "cc_pssm_blosum62_hybrid_lag", 10),
            acc_pssm_blosum62_hybrid_lag=getattr(self.args, "acc_pssm_blosum62_hybrid_lag", 10),
        )
        self.model_factory = ModelFactory()
        self.args.dropout = float(getattr(self.args, "dropout", 0.0))
        self.args.weight_decay = float(getattr(self.args, "weight_decay", 1e-2))
        self.args.grad_clip_norm = float(getattr(self.args, "grad_clip_norm", 0.0))
        self.args.early_stop_patience = int(getattr(self.args, "early_stop_patience", 0))
        self.args.early_stop_min_delta = float(getattr(self.args, "early_stop_min_delta", 0.0))
        self.args.enable_semantic_similarity = bool(
            getattr(self.args, "enable_semantic_similarity", False)
        )
        self.args.semantic_similarity_weight = float(
            getattr(self.args, "semantic_similarity_weight", 0.1)
        )
        self.args.ml_scaler = str(getattr(self.args, "ml_scaler", "standard")).lower()
        self.args.dl_scaler = str(getattr(self.args, "dl_scaler", "none")).lower()
        self.args.residue_adapter_dim = int(
            getattr(self.args, "residue_adapter_dim", RESIDUE_ADAPTER_DIM_DEFAULT)
        )
        if self.args.dropout < 0.0 or self.args.dropout >= 1.0:
            raise SystemExit("dropout must be in [0.0, 1.0).")
        try:
            normalize_prediction_head_config(self.args)
            validate_linear_prediction_head_semantics(self.args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if self.args.weight_decay < 0.0:
            raise SystemExit("weight_decay must be >= 0.0.")
        if self.args.grad_clip_norm < 0.0:
            raise SystemExit("grad_clip_norm must be >= 0.0.")
        if self.args.early_stop_patience < 0:
            raise SystemExit("early_stop_patience must be >= 0.")
        if self.args.early_stop_min_delta < 0.0:
            raise SystemExit("early_stop_min_delta must be >= 0.0.")
        if self.args.semantic_similarity_weight < 0.0:
            raise SystemExit("semantic_similarity_weight must be >= 0.0.")
        if self.args.ml_scaler not in ["none", "standard", "minmax"]:
            raise SystemExit("ml_scaler must be in {none,standard,minmax}.")
        if self.args.dl_scaler not in ["none", "standard"]:
            raise SystemExit("dl_scaler must be in {none,standard}.")
        if not (
            RESIDUE_ADAPTER_DIM_MIN <= self.args.residue_adapter_dim <= RESIDUE_ADAPTER_DIM_MAX
        ):
            raise SystemExit(
                f"residue_adapter_dim must be in "
                f"[{RESIDUE_ADAPTER_DIM_MIN}, {RESIDUE_ADAPTER_DIM_MAX}]."
            )
        if self.args.residue_adapter_dim % RESIDUE_ADAPTER_DIM_MULTIPLE != 0:
            raise SystemExit(
                f"residue_adapter_dim must be divisible by {RESIDUE_ADAPTER_DIM_MULTIPLE}."
            )
        self.ml_trainer = ClassicalMLTrainer(self.args)

    def _ensure_hlaii_test_alias(self):
        if normalize_task_type(getattr(self.args, "task_type", "")) != "pep_hlaii":
            return
        base_dir = self.args.data_dir
        test_dir = os.path.join(base_dir, "test")
        warm_dir = os.path.join(base_dir, "test_warm")
        warm_seq = os.path.join(warm_dir, "seqs.fasta")
        warm_label = os.path.join(warm_dir, "labels.csv")
        test_seq = os.path.join(test_dir, "seqs.fasta")
        test_label = os.path.join(test_dir, "labels.csv")
        if os.path.exists(test_seq) and os.path.exists(test_label):
            return
        if not (os.path.exists(warm_seq) and os.path.exists(warm_label)):
            return
        os.makedirs(test_dir, exist_ok=True)
        for name in ["seqs.fasta", "labels.csv", "pairs.csv"]:
            src = os.path.join(warm_dir, name)
            dst = os.path.join(test_dir, name)
            if not os.path.exists(src) or os.path.exists(dst):
                continue
            try:
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)
        print(f"[pep_hlaii] Using test_warm as test split: {test_dir}")

    def _build_hlaii_contact_csv(self, split_name, out_csv):
        src_csv = os.path.join(self.args.data_dir, split_name, "pairs.csv")
        if not os.path.exists(src_csv):
            return 0
        rows = []
        with open(src_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                label_raw = r.get("label", r.get("bind", ""))
                if label_raw in ["", None]:
                    continue
                rows.append(
                    {
                        "sample_id": str(r.get("sample_id", "")).strip(),
                        "peptide": str(r.get("peptide", "")).strip(),
                        "protein": str(r.get("pair_pseudo", "")).strip(),
                        "label": int(float(label_raw)),
                    }
                )
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "peptide", "protein", "label"])
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    def _prepare_hlaii_contact_inputs(self):
        out_dir = os.path.join(self.args.data_dir, "contact_processed")
        train_csv = os.path.join(out_dir, "train.csv")
        val_csv = os.path.join(out_dir, "val.csv")
        test_csv = os.path.join(out_dir, "test.csv")
        n_train = self._build_hlaii_contact_csv("train", train_csv)
        n_val = self._build_hlaii_contact_csv("val", val_csv)
        n_test = self._build_hlaii_contact_csv("test", test_csv)
        if n_train == 0 or n_val == 0:
            raise SystemExit(
                "pep_hlaii contact requires labeled train/val pairs.csv with pair_pseudo"
            )
        self.args.train_csv = train_csv
        self.args.val_csv = val_csv
        self.args.test_csv = test_csv if n_test > 0 else ""
        self.args.task_root = self.task_root_rel
        model_name = str(getattr(self.args, "model", "mlp")).strip().lower()
        base_encoder = str(getattr(self.args, "encoder", "index")).strip().lower()
        self.args.aux_encoder = base_encoder
        if model_name == "contact":
            self.args.model = "mlp"
        elif model_name == "cross_attn":
            self.args.model = "cross_attn+mlp"
        elif model_name == "bi_cross_attn":
            self.args.model = "bi_cross_attn+mlp"
        token_level_models = {
            "cnn",
            "rnn",
            "gru",
            "lstm",
            "transformer",
            "cross_attn+mlp",
            "bi_cross_attn+mlp",
        }
        if str(getattr(self.args, "model", "mlp")).lower() not in token_level_models:
            self.args.encoder = "ppi_fusion"
        print(
            f"[pep_hlaii-contact] train={n_train}, val={n_val}, test={n_test}, model={self.args.model}, encoder={self.args.encoder}"
        )

    def _resolve_static_embedding_target(self):
        encoder = str(getattr(self.args, "encoder", "")).lower()
        if encoder not in ["word2vec", "glove", "fasttext"]:
            return None
        tokenizer = str(getattr(self.args, "tokenizer", "")).lower()
        if tokenizer == "kmer":
            tag = f"kmer-{int(getattr(self.args, 'k', 2))}"
            tokenizer_mode = "kmer"
        elif tokenizer == "dr":
            tag = (
                "dr-v2-"
                f"{validate_dr_max_distance(getattr(self.args, 'dr_max_distance', DR_DEFAULT_MAX_DISTANCE))}"
            )
            tokenizer_mode = "dr"
        else:
            return None
        out_dir = os.path.join(self.project_root, self.task_root_rel, f"{encoder}-{tag}")
        extension = ".bin" if encoder == "fasttext" else ".vec"
        out_path = os.path.join(out_dir, f"{encoder}{extension}")
        return encoder, tokenizer_mode, out_path

    def _set_embedding_path(self, encoder, path):
        if encoder == "word2vec":
            self.word2vec_path = path
            self.args.word2vec_path = path
            self.feat_factory.word2vec_path = path
        elif encoder == "glove":
            self.glove_path = path
            self.args.glove_path = path
            self.feat_factory.glove_path = path
        elif encoder == "fasttext":
            self.fasttext_path = path
            self.args.fasttext_path = path
            self.feat_factory.fasttext_path = path

    def _needs_pssm_cache(self):
        return self.args.encoder in self.PROFILE_ENCODERS - {
            "psfm",
            "psfm_blosum62_hybrid",
        }

    def _needs_frequency_profile_cache(self):
        return self.args.tokenizer in self.PROFILE_TOKENIZERS

    def _needs_profile_cache(self):
        return (
            self._needs_pssm_cache()
            or self._needs_frequency_profile_cache()
            or self.args.encoder in {"psfm", "psfm_blosum62_hybrid"}
        )

    def ensure_bpe_tokenizer_ready(self):
        if str(getattr(self.args, "tokenizer", "")).lower() != "bpe":
            return
        seqs = []
        train_fasta = os.path.join(self.args.data_dir, "train", "seqs.fasta")
        if os.path.exists(train_fasta):
            seqs.extend(self.load_sequences(train_fasta))
        if not seqs:
            raise SystemExit(f"No sequences found for BPE training under: {self.args.data_dir}")
        trainer = BPETokenizerTrainer(vocab_size=int(getattr(self.args, "bpe_vocab_size", 1000)))
        trainer.ensure_trained(
            seqs,
            self.bpe_path,
            sequence_role="sequence",
        )
        self.args.bpe_path = self.bpe_path
        self.feat_factory.bpe_path = self.bpe_path
        state = "reused" if trainer.last_cache_reused else "rebuilt"
        print(f"[bpe] train-only cache {state}: {self.bpe_path}")

    def ensure_wordpiece_tokenizer_ready(self):
        if str(getattr(self.args, "tokenizer", "")).lower() != "wordpiece":
            return
        sequences = []
        train_fasta = os.path.join(
            self.args.data_dir,
            "train",
            "seqs.fasta",
        )
        if os.path.exists(train_fasta):
            sequences.extend(self.load_sequences(train_fasta))
        if not sequences:
            raise SystemExit(
                f"No sequences found for WordPiece training under: {self.args.data_dir}"
            )
        trainer = WordPieceTokenizerTrainer(
            vocab_size=int(getattr(self.args, "wordpiece_vocab_size", 1000)),
            max_raw_length=validate_wordpiece_max_raw_length(
                getattr(
                    self.args,
                    "wordpiece_max_raw_length",
                    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
        )
        trainer.ensure_trained(
            sequences,
            self.wordpiece_path,
            sequence_role="sequence",
        )
        self.args.wordpiece_path = self.wordpiece_path
        self.feat_factory.wordpiece_path = self.wordpiece_path
        self.feat_factory.wordpiece_max_raw_length = trainer.max_raw_length
        state = "reused" if trainer.last_cache_reused else "rebuilt"
        print(f"[wordpiece] train-only cache {state}: {self.wordpiece_path}")

    def ensure_unigram_tokenizer_ready(self):
        if str(getattr(self.args, "tokenizer", "")).lower() != "unigram":
            return
        train_fasta = os.path.join(
            self.args.data_dir,
            "train",
            "seqs.fasta",
        )
        sequences = self.load_sequences(train_fasta) if os.path.exists(train_fasta) else []
        if not sequences:
            raise SystemExit(f"No sequences found for Unigram training under: {self.args.data_dir}")
        trainer = UnigramTokenizerTrainer(
            vocab_size=validate_unigram_vocab_size(getattr(self.args, "unigram_vocab_size", 1000)),
            max_raw_length=validate_unigram_max_raw_length(
                getattr(
                    self.args,
                    "unigram_max_raw_length",
                    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
        )
        trainer.ensure_trained(
            sequences,
            self.unigram_path,
            sequence_role="sequence",
        )
        self.args.unigram_path = self.unigram_path
        self.feat_factory.unigram_path = self.unigram_path
        self.feat_factory.unigram_max_raw_length = trainer.max_raw_length
        state = "reused" if trainer.last_cache_reused else "rebuilt"
        print(f"[unigram] train-only cache {state}: {self.unigram_path}")

    def ensure_pssm_cache_ready(self):
        if not self._needs_profile_cache():
            return
        active_encoder = str(getattr(self.args, "encoder", "") or "").strip().lower()
        if active_encoder in PSSM_COVARIANCE_ENCODERS:
            lag_key = pssm_covariance_lag_key(active_encoder)
            lag = int(getattr(self.args, lag_key, 10))
            for split in ("train", "val", "test"):
                fasta_path = os.path.join(
                    self.args.data_dir,
                    split,
                    "seqs.fasta",
                )
                if not os.path.exists(fasta_path):
                    continue
                lengths = [len(sequence) for sequence in self.load_sequences(fasta_path)]
                minimum = min(lengths, default=0)
                if minimum <= lag:
                    raise SystemExit(
                        f"{active_encoder} requires every profile length > "
                        f"{lag_key}; {split} minimum sequence length is "
                        f"{minimum}, lag is {lag}"
                    )
        profile_feature = self._profile_feature_kind()
        config = self._profile_generation_config()
        top_ngram_fallback_policy = (
            TOP_N_GRAM_FALLBACK_POLICY_VERSION
            if str(
                getattr(
                    self.args,
                    "profile_on_missing",
                    TOP_N_GRAM_FALLBACK_POLICY,
                )
            )
            == TOP_N_GRAM_FALLBACK_POLICY
            else TOP_N_GRAM_STRICT_PROFILE_POLICY
        )

        train_fasta = os.path.join(
            self.args.data_dir,
            "train",
            "seqs.fasta",
        )
        with open(train_fasta, "rb") as handle:
            train_sha256 = hashlib.sha256(handle.read()).hexdigest()

        def validate_managed_index(index_path, feature_type):
            if feature_type == "target_frequency":
                store = TargetFrequencyProfileStore(
                    index_path,
                    max_length=int(self.args.max_length),
                )
                expected_fallback = top_ngram_fallback_policy
            else:
                store = ProfileStore(index_path, feature_type=feature_type)
                expected_fallback = (
                    "blosum62_hybrid"
                    if feature_type in (PSSM_HYBRID_FEATURE_TYPES | PSFM_HYBRID_FEATURE_TYPES)
                    else "error"
                )
            provenance = store.manifest.get("provenance") or {}
            database = provenance.get("database") or {}
            if (
                provenance.get("database_scope") != "train_only"
                or database.get("train_fasta_sha256") != train_sha256
                or provenance.get("self_exclusion")
                != "all subject IDs with an identical normalized query sequence"
                or provenance.get("scientific_parameters") != config.scientific_parameters()
                or provenance.get("fallback_policy") != expected_fallback
            ):
                raise SequenceProfileError(
                    "profile manifest does not match the current training "
                    "split, PSI-BLAST parameters, self-exclusion, or fallback "
                    "contract",
                    code="profile_provenance_mismatch",
                )
            return store

        if self._needs_frequency_profile_cache() and self._explicit_frequency_profile_index_path:
            try:
                store = validate_managed_index(
                    self._explicit_frequency_profile_index_path,
                    "target_frequency",
                )
            except (
                OSError,
                ValueError,
                TypeError,
                TopNGramContractError,
                SequenceProfileError,
            ) as exc:
                raise SystemExit(f"Invalid explicit --frequency_profile_index_path: {exc}") from exc
            self.frequency_profile_index_path = self._explicit_frequency_profile_index_path
            self.args.frequency_profile_index_path = self.frequency_profile_index_path
            self.args._frequency_profile_manifest_sha256 = store.manifest_sha256
            self.args._frequency_profile_provenance = dict(store.manifest.get("provenance") or {})
            self.feat_factory.frequency_profile_index_path = self.frequency_profile_index_path
            return
        index_attribute = self._profile_index_attribute(profile_feature)
        primary_index = getattr(
            self,
            index_attribute,
        )
        profile_root = os.path.dirname(os.path.dirname(primary_index))
        current_index = primary_index
        stale_cache = os.path.exists(current_index)
        if stale_cache:
            try:
                store = validate_managed_index(
                    current_index,
                    profile_feature,
                )
                if profile_feature == "target_frequency":
                    self.args._frequency_profile_manifest_sha256 = store.manifest_sha256
                    self.args._frequency_profile_provenance = dict(
                        store.manifest.get("provenance") or {}
                    )
                else:
                    self.args._profile_manifest_sha256 = store.manifest_sha256
                return
            except (
                OSError,
                ValueError,
                TypeError,
                TopNGramContractError,
                SequenceProfileError,
            ):
                pass
            print(f"[profiles] rebuilding stale cache: {profile_root}")
        print(f"[profiles] building profile cache (feature={profile_feature}) -> {profile_root}")
        cmd = [
            sys.executable,
            os.path.join(self.project_root, "tools", "build_sequence_profile_cache.py"),
            "--data-dir",
            self.args.data_dir,
            "--out-root",
            profile_root,
            "--max-length",
            str(getattr(self.args, "max_length", 128)),
            "--iterations",
            str(getattr(self.args, "profile_iterations", 3)),
            "--evalue",
            str(getattr(self.args, "profile_evalue", 0.001)),
            "--inclusion-evalue",
            str(getattr(self.args, "profile_inclusion_evalue", 0.001)),
            "--threads",
            str(getattr(self.args, "profile_threads", 4)),
            "--feature",
            profile_feature,
            "--on-missing",
            str(
                getattr(self.args, "profile_on_missing", "error")
                if profile_feature == "target_frequency"
                else "error"
            ),
        ]
        if stale_cache:
            cmd.append("--force")
        try:
            subprocess.run(cmd, check=True, cwd=self.project_root)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"{profile_feature} profile cache build failed with exit code "
                f"{exc.returncode}; see the preceding profile error for the "
                "failing sample and remediation"
            ) from None
        generated_index = os.path.join(
            profile_root,
            profile_feature,
            "index.json",
        )
        if not os.path.exists(generated_index):
            raise SystemExit(f"combined profile cache build failed: {profile_root}")
        store = validate_managed_index(generated_index, profile_feature)
        if profile_feature == "target_frequency":
            self.frequency_profile_index_path = generated_index
            self.args._frequency_profile_manifest_sha256 = store.manifest_sha256
            self.args._frequency_profile_provenance = dict(store.manifest.get("provenance") or {})
        else:
            setattr(self, index_attribute, generated_index)
            setattr(self.args, index_attribute, generated_index)
            self.args._profile_manifest_sha256 = store.manifest_sha256
        self.args.pssm_index_path = self.pssm_index_path
        self.args.psfm_index_path = self.psfm_index_path
        self.args.frequency_profile_index_path = self.frequency_profile_index_path
        self.feat_factory.pssm_index_path = self.pssm_index_path
        self.feat_factory.psfm_index_path = self.psfm_index_path
        for feature, attribute in self.PROFILE_INDEX_ATTRIBUTES.items():
            setattr(
                self.feat_factory,
                attribute,
                getattr(self, attribute),
            )
        self.feat_factory.frequency_profile_index_path = self.frequency_profile_index_path
        print(f"[profiles] ready: {profile_root}")

    def _embedding_meta_matches(self, encoder, tokenizer_mode, out_path):
        meta_path = os.path.join(os.path.dirname(out_path), "meta.json")
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return False
        expected = {
            "method": encoder,
            "tokenizer_mode": tokenizer_mode,
            "k": int(getattr(self.args, "k", 2)),
            "dr_max_distance": int(getattr(self.args, "dr_max_distance", 5)),
            "vector_size": int(getattr(self.args, "embedding_vector_size", 128)),
            "window": int(getattr(self.args, "embedding_window", 5)),
            "min_count": int(getattr(self.args, "embedding_min_count", 1)),
            "epochs": int(getattr(self.args, "embedding_epochs", 20)),
            "workers": (
                1
                if encoder == "glove"
                else validate_positive_int(
                    getattr(self.args, "embedding_workers", 1),
                    name="embedding_workers",
                )
            ),
            "seed": int(getattr(self.args, "seed", 42)),
            "feature_fit_scope": "train_only",
        }
        if tokenizer_mode == "dr":
            max_distance = validate_dr_max_distance(
                getattr(
                    self.args,
                    "dr_max_distance",
                    DR_DEFAULT_MAX_DISTANCE,
                )
            )
            max_raw_length = validate_dr_max_raw_length(
                getattr(
                    self.args,
                    "dr_max_raw_length",
                    DR_DEFAULT_MAX_RAW_LENGTH,
                )
            )
            expected["dr_max_raw_length"] = max_raw_length
            expected["dr_contract"] = dr_contract_metadata(
                max_distance=max_distance,
                max_raw_length=max_raw_length,
            )
        if encoder == "glove":
            expected.update(
                glove_cache_signature(
                    x_max=getattr(self.args, "glove_x_max", 100.0),
                    alpha=getattr(self.args, "glove_alpha", 0.75),
                    learning_rate=getattr(self.args, "glove_learning_rate", 0.05),
                    batch_size=getattr(self.args, "glove_batch_size", 4096),
                )
            )
        elif encoder == "word2vec":
            expected.update(word2vec_cache_signature(workers=expected["workers"]))
        elif encoder == "fasttext":
            expected.update(
                fasttext_cache_signature(
                    workers=expected["workers"],
                    bucket=validate_positive_int(
                        getattr(
                            self.args,
                            "fasttext_bucket",
                            FASTTEXT_DEFAULT_BUCKET,
                        ),
                        name="fasttext_bucket",
                    ),
                )
            )
        train_fasta = os.path.join(self.args.data_dir, "train", "seqs.fasta")
        if not os.path.exists(train_fasta):
            return False
        with open(train_fasta, "rb") as handle:
            expected["train_sha256"] = hashlib.sha256(handle.read()).hexdigest()
        if not all(meta.get(k) == v for k, v in expected.items()):
            return False
        if meta.get("artifact_file") != os.path.basename(out_path):
            return False
        expected_digest = str(meta.get("artifact_sha256") or "")
        return (
            len(expected_digest) == 64
            and os.path.isfile(out_path)
            and sha256_file(out_path) == expected_digest
        )

    def _record_static_embedding_provenance(self, encoder, path, source):
        provenance = {
            "method": str(encoder),
            "source": str(source),
            "path": str(path),
        }
        if path and os.path.exists(path):
            provenance["sha256"] = sha256_file(path)
        if source == "internal":
            meta_path = os.path.join(os.path.dirname(path), "meta.json")
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    provenance.update(json.load(handle))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        else:
            provenance["algorithm_verified"] = False
            provenance["verification"] = "unverified"
        self.args._static_embedding_provenance = provenance

    def ensure_static_embedding_ready(self):
        target = self._resolve_static_embedding_target()
        if target is None:
            return
        encoder, tokenizer_mode, out_path = target
        explicit_paths = getattr(
            self,
            "_explicit_static_embedding_paths",
            {"glove": getattr(self, "_explicit_glove_path", "")},
        )
        explicit_path = str(explicit_paths.get(encoder, "") or "")
        if explicit_path:
            if not os.path.isfile(explicit_path):
                raise SystemExit(f"Explicit {encoder} embedding not found: {explicit_path}")
            if encoder == "fasttext" and os.path.splitext(explicit_path)[1].lower() != ".bin":
                raise SystemExit(
                    "Explicit fastText artifacts must be complete Facebook-compatible "
                    ".bin files; legacy .vec files omit subword state and are unsupported"
                )
            self._set_embedding_path(encoder, explicit_path)
            self._record_static_embedding_provenance(
                encoder,
                explicit_path,
                "external",
            )
            return
        self._set_embedding_path(encoder, out_path)
        if os.path.exists(out_path) and self._embedding_meta_matches(
            encoder, tokenizer_mode, out_path
        ):
            self._record_static_embedding_provenance(encoder, out_path, "internal")
            return
        print(f"[{encoder}] embedding missing or stale, training cache: {out_path}")
        cmd = [
            sys.executable,
            os.path.join(self.project_root, "tools", "train_static_embeddings.py"),
            "--data_dir",
            str(getattr(self.args, "data_dir", f"{self.task_root_rel}/data")),
            "--tokenizer_mode",
            tokenizer_mode,
            "--save_root",
            self.task_root_rel,
            "--vector_size",
            str(getattr(self.args, "embedding_vector_size", 128)),
            "--window",
            str(getattr(self.args, "embedding_window", 5)),
            "--min_count",
            str(getattr(self.args, "embedding_min_count", 1)),
            "--epochs",
            str(getattr(self.args, "embedding_epochs", 20)),
            "--workers",
            str(
                validate_positive_int(
                    (1 if encoder == "glove" else getattr(self.args, "embedding_workers", 1)),
                    name="embedding_workers",
                )
            ),
            "--seed",
            str(int(getattr(self.args, "seed", 42))),
        ]
        if tokenizer_mode == "kmer":
            cmd.extend(["--k", str(getattr(self.args, "k", 2))])
        else:
            cmd.extend(
                [
                    "--dr_max_distance",
                    str(
                        validate_dr_max_distance(
                            getattr(
                                self.args,
                                "dr_max_distance",
                                DR_DEFAULT_MAX_DISTANCE,
                            )
                        )
                    ),
                    "--dr_max_raw_length",
                    str(
                        validate_dr_max_raw_length(
                            getattr(
                                self.args,
                                "dr_max_raw_length",
                                DR_DEFAULT_MAX_RAW_LENGTH,
                            )
                        )
                    ),
                ]
            )
        if encoder == "word2vec":
            cmd.append("--train_word2vec")
        elif encoder == "glove":
            cmd.extend(
                [
                    "--glove_x_max",
                    str(getattr(self.args, "glove_x_max", 100.0)),
                    "--glove_alpha",
                    str(getattr(self.args, "glove_alpha", 0.75)),
                    "--glove_learning_rate",
                    str(getattr(self.args, "glove_learning_rate", 0.05)),
                    "--glove_batch_size",
                    str(getattr(self.args, "glove_batch_size", 4096)),
                ]
            )
            cmd.append("--train_glove")
        elif encoder == "fasttext":
            cmd.extend(
                [
                    "--fasttext_bucket",
                    str(
                        validate_positive_int(
                            getattr(
                                self.args,
                                "fasttext_bucket",
                                FASTTEXT_DEFAULT_BUCKET,
                            ),
                            name="fasttext_bucket",
                        )
                    ),
                    "--static_embedding_memory_limit_gib",
                    str(
                        validate_static_embedding_memory_limit_gib(
                            getattr(
                                self.args,
                                "static_embedding_memory_limit_gib",
                                STATIC_EMBEDDING_DEFAULT_MEMORY_LIMIT_GIB,
                            )
                        )
                    ),
                ]
            )
            cmd.append("--train_fasttext")
        subprocess.run(cmd, check=True, cwd=self.project_root)
        if not os.path.exists(out_path):
            raise SystemExit(f"{encoder} embedding generation failed: {out_path}")
        if not self._embedding_meta_matches(
            encoder,
            tokenizer_mode,
            out_path,
        ):
            raise SystemExit(
                f"{encoder} embedding generation produced invalid or "
                f"incomplete cache metadata: {out_path}"
            )
        self._record_static_embedding_provenance(encoder, out_path, "internal")
        print(f"[{encoder}] ready: {out_path}")

    def build_featurizer(self):
        self.ensure_static_embedding_ready()
        self.ensure_pssm_cache_ready()
        sequences = None
        profile_samples = None
        frequency_encoder = str(getattr(self.args, "encoder", "") or "").strip().lower()
        if frequency_encoder in {"bow", "tfidf", "tf-idf", "tf_idf"}:
            train_path = os.path.join(self.args.data_dir, "train", "seqs.fasta")
            sequences = self.load_sequences(train_path)
            if (
                str(getattr(self.args, "tokenizer", "") or "").strip().lower()
                in self.PROFILE_TOKENIZERS
            ):
                profile_samples = self.load_sequence_records(train_path)
        if self.args.tokenizer == "top-n-gram":
            try:
                validate_top_ngram_n(getattr(self.args, "top_ngram_n", 2))
            except TopNGramContractError as exc:
                raise SystemExit(str(exc)) from exc
        if self.args.tokenizer == "dt":
            try:
                validate_dt_max_distance(
                    getattr(
                        self.args,
                        "dt_max_distance",
                        DT_DEFAULT_MAX_DISTANCE,
                    )
                )
            except DTContractError as exc:
                raise SystemExit(str(exc)) from exc
        if bool(getattr(self.args, "unigram_sampling", False)):
            if str(getattr(self.args, "tokenizer", "")).lower() != "unigram":
                raise SystemExit("--unigram_sampling requires --tokenizer unigram")
            if str(getattr(self.args, "encoder", "")).lower() != "index":
                raise SystemExit("Unigram sampling requires the sequence-preserving encoder=index")
            training_mode = str(getattr(self.args, "training_mode", "") or "").strip().lower()
            model_name = str(getattr(self.args, "model", "") or "").strip().lower()
            if model_name in CLASSICAL_CLASSIFIERS and training_mode != "export_embeddings":
                raise SystemExit(
                    "Unigram sampling is online neural regularization and is "
                    "not supported by direct classical classifiers"
                )
            validate_unigram_sampling(
                getattr(
                    self.args,
                    "unigram_nbest_size",
                    UNIGRAM_DEFAULT_NBEST_SIZE,
                ),
                getattr(
                    self.args,
                    "unigram_alpha",
                    UNIGRAM_DEFAULT_ALPHA,
                ),
            )
        self.ensure_bpe_tokenizer_ready()
        self.ensure_wordpiece_tokenizer_ready()
        self.ensure_unigram_tokenizer_ready()
        if self._needs_pssm_cache():
            profile_feature = self._profile_feature_kind()
            profile_index_attribute = self._profile_index_attribute(profile_feature)
            profile_index_path = getattr(
                self,
                profile_index_attribute,
                "",
            )
            if not (profile_index_path and os.path.exists(profile_index_path)):
                if profile_index_attribute == "pssm_index_path":
                    raise SystemExit(
                        f"PSSM index not found: {profile_index_path}. Pass --pssm_index_path."
                    )
                raise SystemExit(
                    f"{profile_feature} profile index not found: "
                    f"{profile_index_path}. Pass "
                    f"--{profile_index_attribute}."
                )
        if self.args.tokenizer in self.PROFILE_TOKENIZERS and not (
            self.frequency_profile_index_path and os.path.exists(self.frequency_profile_index_path)
        ):
            raise SystemExit(
                f"{self.args.tokenizer} requires target-frequency profile index: "
                f"{self.frequency_profile_index_path}. Pass "
                "--frequency_profile_index_path."
            )
        if self.args.encoder in {"psfm", "psfm_blosum62_hybrid"} and not (
            self.psfm_index_path and os.path.exists(self.psfm_index_path)
        ):
            raise SystemExit(
                f"PSFM index not found: {self.psfm_index_path}. Pass --psfm_index_path."
            )
        if self.args.encoder == "word2vec" and not (
            self.word2vec_path and os.path.exists(self.word2vec_path)
        ):
            raise SystemExit(
                f"Word2Vec embedding not found: {self.word2vec_path}. Pass --word2vec_path."
            )
        if self.args.encoder == "glove" and not (
            self.glove_path and os.path.exists(self.glove_path)
        ):
            raise SystemExit(f"GloVe embedding not found: {self.glove_path}. Pass --glove_path.")
        if self.args.encoder == "fasttext" and not (
            self.fasttext_path and os.path.exists(self.fasttext_path)
        ):
            raise SystemExit(
                f"FastText embedding not found: {self.fasttext_path}. Pass --fasttext_path."
            )
        featurizer = self.feat_factory.create(
            self.args.tokenizer,
            self.args.encoder,
            self.args.max_length,
            self.args.k,
            sequences=sequences,
            dr_max_distance=getattr(self.args, "dr_max_distance", 5),
            top_ngram_n=getattr(self.args, "top_ngram_n", 2),
            dt_max_distance=getattr(
                self.args,
                "dt_max_distance",
                DT_DEFAULT_MAX_DISTANCE,
            ),
            profile_samples=profile_samples,
            frequency_contract_version=FREQUENCY_ENCODER_CONTRACT_VERSION,
            frequency_max_raw_length=validate_frequency_max_raw_length(
                getattr(
                    self.args,
                    "frequency_max_raw_length",
                    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
        )
        architecture = (
            str(getattr(self.args, "sequence_model", "") or getattr(self.args, "model", "") or "")
            .strip()
            .lower()
        )
        if architecture in {"rnn", "gru", "lstm", "transformer", "gcn", "gat"}:
            featurizer.require_explicit_position_mask()
        if str(getattr(self.args, "encoder", "")).lower() in {"tfidf", "tf-idf"} and hasattr(
            getattr(featurizer, "encoder", None),
            "idf",
        ):
            self.args._classic_tfidf_idf = featurizer.encoder.idf.detach().cpu().float()
        if hasattr(featurizer, "frequency_encoder_contract"):
            contract = dict(featurizer.frequency_encoder_contract)
            if sequences is not None:
                corpus_identity = sequences
                if profile_samples is not None:
                    corpus_identity = [
                        featurizer.tokenizer.profile_store.manifest_sha256,
                        *profile_samples,
                    ]
                contract["training_corpus_sha256"] = digest_text_items(corpus_identity)
                contract["fit_scope"] = "sequence_train_split"
            self.args._frequency_encoder_contract = contract
        if hasattr(featurizer, "textrank_contract"):
            self.args._textrank_contract = dict(featurizer.textrank_contract)
        if hasattr(featurizer, "covariance_encoder_contract"):
            self.args._covariance_encoder_contract = dict(featurizer.covariance_encoder_contract)
        if hasattr(featurizer, "pdt_encoder_contract"):
            self.args._pdt_encoder_contract = dict(featurizer.pdt_encoder_contract)
        if hasattr(featurizer, "pssm_covariance_encoder_contract"):
            self.args._pssm_covariance_encoder_contract = dict(
                featurizer.pssm_covariance_encoder_contract
            )
        if str(getattr(self.args, "tokenizer", "")).lower() in self.PROFILE_TOKENIZERS:
            self.args._frequency_profile_manifest_sha256 = (
                featurizer.tokenizer.profile_store.manifest_sha256
            )
        return featurizer

    def build_datasets(self, featurizer):
        base_dir = self.args.data_dir
        train_data = PeptideDataset(
            os.path.join(base_dir, "train", "seqs.fasta"),
            os.path.join(base_dir, "train", "labels.csv"),
            featurizer=featurizer,
        )
        if bool(getattr(self.args, "unigram_sampling", False)):
            train_data.enable_unigram_sampling(
                seed=int(getattr(self.args, "seed", 42)),
                nbest_size=int(
                    getattr(
                        self.args,
                        "unigram_nbest_size",
                        UNIGRAM_DEFAULT_NBEST_SIZE,
                    )
                ),
                alpha=float(
                    getattr(
                        self.args,
                        "unigram_alpha",
                        UNIGRAM_DEFAULT_ALPHA,
                    )
                ),
                role="sequence",
            )
        val_data = PeptideDataset(
            os.path.join(base_dir, "val", "seqs.fasta"),
            os.path.join(base_dir, "val", "labels.csv"),
            featurizer=featurizer,
        )
        test_data = None
        test_seq_path = os.path.join(base_dir, "test", "seqs.fasta")
        test_label_path = os.path.join(base_dir, "test", "labels.csv")
        if (
            bool(getattr(self.args, "evaluate_test", True))
            and os.path.exists(test_seq_path)
            and os.path.exists(test_label_path)
        ):
            test_data = PeptideDataset(test_seq_path, test_label_path, featurizer=featurizer)
        if str(getattr(self.args, "tokenizer", "")).lower() == "dr":
            sample_count = len(train_data) + len(val_data)
            if test_data is not None:
                sample_count += len(test_data)
            try:
                tensor_bytes = estimate_dr_dense_tensor_bytes(
                    sample_count,
                    encoder=str(getattr(self.args, "encoder", "index")),
                    max_distance=getattr(
                        self.args,
                        "dr_max_distance",
                        DR_DEFAULT_MAX_DISTANCE,
                    ),
                    max_length=int(getattr(self.args, "max_length", 60)),
                    embedding_dim=getattr(
                        getattr(featurizer, "encoder", None),
                        "feature_dim",
                        None,
                    ),
                )
                ensure_dr_dense_memory_budget(
                    tensor_bytes * 2,
                    limit_gib=getattr(
                        self.args,
                        "dr_dense_memory_limit_gib",
                        DR_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                    ),
                    context=("DR sequence feature tensors and one working copy"),
                )
            except DRContractError as exc:
                raise SystemExit(str(exc)) from exc
        if str(getattr(self.args, "tokenizer", "")).lower() == "top-n-gram":
            sample_count = len(train_data) + len(val_data)
            if test_data is not None:
                sample_count += len(test_data)
            try:
                tensor_bytes = estimate_top_ngram_dense_bytes(
                    sample_count,
                    getattr(self.args, "top_ngram_n", 2),
                )
                ensure_top_ngram_dense_memory_budget(
                    tensor_bytes * 2,
                    limit_gib=getattr(
                        self.args,
                        "topngram_dense_memory_limit_gib",
                        TOP_N_GRAM_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                    ),
                    context=("Top-n-gram sequence feature tensors and one working copy"),
                )
            except TopNGramContractError as exc:
                raise SystemExit(str(exc)) from exc
        if str(getattr(self.args, "tokenizer", "")).lower() == "dt":
            sample_count = len(train_data) + len(val_data)
            if test_data is not None:
                sample_count += len(test_data)
            try:
                tensor_bytes = estimate_dt_dense_bytes(
                    sample_count,
                    max_distance=getattr(
                        self.args,
                        "dt_max_distance",
                        DT_DEFAULT_MAX_DISTANCE,
                    ),
                )
                ensure_dt_dense_memory_budget(
                    tensor_bytes * 2,
                    limit_gib=getattr(
                        self.args,
                        "dt_dense_memory_limit_gib",
                        DT_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                    ),
                    context=("DT sequence feature tensors and one working copy"),
                )
            except DTContractError as exc:
                raise SystemExit(str(exc)) from exc
        return train_data, val_data, test_data

    def load_labels(self, csv_path):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return torch.empty(0)
        header = rows[0]
        try:
            [float(x) for x in header]
            start_idx = 0
        except ValueError:
            start_idx = 1
        labels = [[float(x) for x in row] for row in rows[start_idx:]]
        return torch.tensor(labels, dtype=torch.float)

    def load_sequences(self, fasta_path):
        sequences = []
        current_seq = []
        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_seq:
                sequences.append("".join(current_seq))
        return sequences

    def load_sequence_records(self, fasta_path):
        records = []
        current_id = None
        current_sequence = []
        with open(fasta_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_sequence:
                        records.append(
                            (
                                current_id if current_id is not None else str(len(records)),
                                "".join(current_sequence),
                            )
                        )
                    current_id = line[1:].strip()
                    current_sequence = []
                else:
                    current_sequence.append(line)
            if current_sequence:
                records.append(
                    (
                        current_id if current_id is not None else str(len(records)),
                        "".join(current_sequence),
                    )
                )
        return records

    def eval_esm(self, model, tokenizer, encoder, sequences, targets, split="val"):
        model.eval()
        features = self.get_or_extract_esm_embeddings(tokenizer, encoder, sequences, split)
        if features.numel() == 0:
            return 0.0, 0.0, 0.0, 0.0
        dataset = TensorDataset(features, targets)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        return Evaluator.eval_nn(model, loader, self.device)

    def eval_protbert(self, model, tokenizer, encoder, sequences, targets, split="val"):
        model.eval()
        features = self.get_or_extract_protbert_embeddings(tokenizer, encoder, sequences, split)
        if features.numel() == 0:
            return 0.0, 0.0, 0.0, 0.0
        dataset = TensorDataset(features, targets)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        return Evaluator.eval_nn(model, loader, self.device)

    def eval_loss(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                total_loss += loss.item()
                batch_count += 1
        if batch_count == 0:
            return 0.0
        return total_loss / batch_count

    def semantic_similarity_enabled(self):
        model_name = str(getattr(self.args, "model", "")).strip().lower()
        return (
            normalize_task_type(getattr(self.args, "task_type", "")) == "pep_function"
            and semantic_is_enabled(self.args)
            and model_name in self.SEMANTIC_SIMILARITY_MODELS
        )

    def forward_with_embedding(self, model, inputs):
        if not hasattr(model, "forward_with_embedding"):
            raise SystemExit(
                "semantic similarity requires a neural model that returns a real "
                "sample-level embedding via forward_with_embedding"
            )
        logits, embeddings = model.forward_with_embedding(inputs)
        if embeddings is None or embeddings.ndim < 2 or not embeddings.requires_grad:
            raise SystemExit("semantic similarity requires a differentiable sample-level embedding")
        return logits, embeddings

    def _semantic_train_loader(self, dataset):
        generator = torch.Generator()
        generator.manual_seed(int(getattr(self.args, "seed", 42)))
        audit_log = []
        self.args._main_batch_order_hashes = audit_log
        indexed_dataset = IndexedDataset(dataset)
        sampler = RecordingRandomSampler(
            indexed_dataset,
            generator=generator,
            audit_log=audit_log,
        )
        return DataLoader(
            indexed_dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
        )

    def _prepare_semantic_model(self, model, sample_inputs=None):
        prepare_semantic_projectors(
            model,
            self.args,
            sample_inputs=sample_inputs,
        )
        weight = float(getattr(self.args, "semantic_residual_weight", 0.0))
        if weight > 0.0:
            model = SemanticResidualAdapter(model, weight)
        return model

    def _configure_intrinsic_semantics(
        self,
        train_inputs,
        train_targets,
        *,
        train_dataset=None,
        train_ids=None,
        val_ids=None,
        test_ids=None,
    ):
        if not self.semantic_similarity_enabled():
            self._semantic_profile = None
            self._semantic_train_inputs = None
            self._semantic_train_dataset = None
            return
        source = IntrinsicLabelRelationSource(train_targets.detach().cpu())
        positive_partners = int(getattr(self.args, "semantic_positive_partners", 1))
        zero_partners = int(getattr(self.args, "semantic_zero_partners", 8))
        seed = int(getattr(self.args, "seed", 42))
        partner_scheme = str(getattr(self.args, "semantic_partner_scheme", "fixed")).strip().lower()
        placebo_reference_table = None
        if partner_scheme in {"fixed", "fixed_placebo"}:
            fixed_table = build_partner_table(
                source,
                seed=seed,
                positive_partners_per_anchor=positive_partners,
                zero_partners_per_anchor=zero_partners,
            )
            if partner_scheme == "fixed_placebo":
                placebo_reference_table = fixed_table
                table = placebo_shuffle_table(
                    fixed_table,
                    seed=seed,
                    tag="intrinsic-fixed-placebo",
                )
            else:
                table = fixed_table
        else:
            if not isinstance(train_inputs, torch.Tensor):
                raise SystemExit(
                    "dense_hard/placebo semantic partner tables require "
                    "pooled-embedding tensor inputs"
                )
            table = build_dense_hard_partner_table(
                source,
                seed=seed,
                embeddings=train_inputs.detach().cpu().numpy(),
                positive_cap=int(getattr(self.args, "semantic_positive_cap", 16)),
                negative_cap=int(getattr(self.args, "semantic_negative_cap", 8)),
                placebo=(partner_scheme == "placebo"),
            )
        self._semantic_profile = SemanticTrainingProfile(
            relation_type=source.relation_type,
            seed=seed,
            fused_table=table,
            peptide_table=None,
            context_table=None,
            row_peptide_ids=(),
            row_context_ids=(),
            peptide_example_rows=(),
            context_example_rows=(),
            source_summary=source.summary(),
            config={
                "semantic_pair_mode": "fused",
                "semantic_projection_dim": int(getattr(self.args, "semantic_projection_dim", 64)),
                "semantic_contrastive_temperature": float(
                    getattr(self.args, "semantic_contrastive_temperature", 0.1)
                ),
                "semantic_positive_partners": positive_partners,
                "semantic_zero_partners": zero_partners,
                "semantic_similarity_weight": float(
                    getattr(self.args, "semantic_similarity_weight", 0.1)
                ),
                "semantic_zero_weight": float(getattr(self.args, "semantic_zero_weight", 0.25)),
                "semantic_loss_space": str(getattr(self.args, "semantic_loss_space", "projection")),
                "semantic_residual_weight": float(
                    getattr(self.args, "semantic_residual_weight", 0.0)
                ),
                "semantic_partner_scheme": partner_scheme,
                **(
                    {"semantic_loss_type": "jaccard_supcon"}
                    if getattr(self.args, "semantic_loss_type", "legacy") == "jaccard_supcon"
                    else {}
                ),
                "semantic_positive_cap": int(getattr(self.args, "semantic_positive_cap", 16)),
                "semantic_negative_cap": int(getattr(self.args, "semantic_negative_cap", 8)),
                "semantic_hard_negative_source": str(
                    getattr(
                        self.args,
                        "semantic_hard_negative_source",
                        "input_embedding",
                    )
                ),
            },
        )
        self._semantic_train_inputs = train_inputs.detach().cpu()
        self._semantic_train_dataset = train_dataset
        artifact_dir = str(getattr(self.args, "artifact_dir", "") or "").strip()
        if artifact_dir:

            def stable_ids(values, prefix, size):
                if values is None or len(values) != size:
                    return [f"{prefix}:{index}" for index in range(size)]
                return [str(value) for value in values]

            split_ids = {
                "train": stable_ids(train_ids, "train", int(train_targets.shape[0])),
                "val": [str(value) for value in (val_ids or [])],
                "test": [str(value) for value in (test_ids or [])],
            }
            self._semantic_provenance = save_semantic_training_profile(
                artifact_dir,
                profile=self._semantic_profile,
                source=source,
                split_sample_ids=split_ids,
                placebo_reference_table=placebo_reference_table,
            )

    @staticmethod
    def _semantic_sequence_ids(sequences):
        return [
            f"row:{index}:sha256:{hashlib.sha256(str(sequence).encode('utf-8')).hexdigest()}"
            for index, sequence in enumerate(sequences)
        ]

    def _semantic_auxiliary_loss(self, model, row_ids):
        if self._semantic_profile is None or self._semantic_train_inputs is None:
            return None
        anchor_indices = [int(value) for value in row_ids.detach().cpu().view(-1).tolist()]
        relations = self._semantic_profile.fused_table.relations_for(anchor_indices)
        if not relations:
            return None
        unique_rows = self._semantic_profile.fused_table.unique_row_indices(relations)
        if (
            self._semantic_train_dataset is not None
            and getattr(
                self._semantic_train_dataset,
                "_unigram_sampling",
                None,
            )
            is not None
        ):
            auxiliary_inputs = torch.stack(
                [self._semantic_train_dataset[index][0] for index in unique_rows],
                dim=0,
            ).to(self.device)
        else:
            index = torch.tensor(unique_rows, dtype=torch.long)
            auxiliary_inputs = self._semantic_train_inputs.index_select(
                0,
                index,
            ).to(self.device)
        with auxiliary_eval_forward(model):
            _, auxiliary_embeddings = self.forward_with_embedding(model, auxiliary_inputs)
        loss_space = str(getattr(self.args, "semantic_loss_space", "projection")).strip().lower()
        if loss_space == "projection":
            if not hasattr(model, "semantic_projector"):
                raise SystemExit(
                    "semantic training requires prepare_semantic_projectors to be called"
                )
            auxiliary_embeddings = model.semantic_projector(auxiliary_embeddings)
        partner_scheme = str(getattr(self.args, "semantic_partner_scheme", "fixed")).strip().lower()
        return semantic_contrastive_loss(
            auxiliary_embeddings,
            unique_rows,
            relations,
            temperature=float(getattr(self.args, "semantic_contrastive_temperature", 0.1)),
            weight_positives=(partner_scheme in {"dense_hard", "placebo"}),
            loss_type=getattr(self.args, "semantic_loss_type", "legacy"),
        )

    def training_loss(self, model, inputs, targets, criterion, row_ids=None):
        if not self.semantic_similarity_enabled():
            logits = model(inputs)
            return logits, criterion(logits, targets), None
        if row_ids is None:
            raise RuntimeError("semantic training requires stable training-row indices")
        logits = model(inputs)
        classification_loss = criterion(logits, targets)
        semantic_output = self._semantic_auxiliary_loss(model, row_ids)
        if semantic_output is None:
            return logits, classification_loss, None
        weight = float(getattr(self.args, "semantic_similarity_weight", 0.1))
        return logits, classification_loss + weight * semantic_output.loss, semantic_output

    def print_semantic_similarity_status(self, tag):
        if self.semantic_similarity_enabled():
            weight = float(getattr(self.args, "semantic_similarity_weight", 0.1))
            print(
                f"[{tag}] Semantic similarity contrastive loss enabled: "
                f"relation=intrinsic_label_jaccard, weight={weight:g}, "
                f"temperature={float(getattr(self.args, 'semantic_contrastive_temperature', 0.1)):g}, "
                f"projection_dim={int(getattr(self.args, 'semantic_projection_dim', 64))}, "
                f"loss_space={str(getattr(self.args, 'semantic_loss_space', 'projection'))}, "
                f"residual_weight={float(getattr(self.args, 'semantic_residual_weight', 0.0)):g}, "
                f"partner_scheme={str(getattr(self.args, 'semantic_partner_scheme', 'fixed'))}"
            )

    def _stack_dataset_xy(self, dataset):
        xs, ys = [], []
        for i in range(len(dataset)):
            x, y = dataset[i]
            xs.append(x.float())
            ys.append(y.float())
        if len(xs) == 0:
            return torch.empty(0), torch.empty(0)
        if isinstance(xs[0], ResidueModelInput):
            stacked_x = ResidueModelInput(
                torch.stack([item.features for item in xs]),
                torch.stack([item.mask for item in xs]),
            )
        else:
            stacked_x = torch.stack(xs)
        return stacked_x, torch.stack(ys)

    def _scale_tensor_split(self, train_x, val_x, test_x=None, return_scaler=False):
        if self.args.dl_scaler == "none":
            if return_scaler:
                return train_x, val_x, test_x, None
            return train_x, val_x, test_x
        train_shape = train_x.shape
        val_shape = val_x.shape
        train_2d = train_x.view(train_x.shape[0], -1).cpu().numpy()
        val_2d = val_x.view(val_x.shape[0], -1).cpu().numpy()
        if self.args.dl_scaler == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        train_2d = scaler.fit_transform(train_2d)
        val_2d = scaler.transform(val_2d)
        test_2d = None
        if test_x is not None and test_x.numel() > 0:
            test_2d = scaler.transform(test_x.view(test_x.shape[0], -1).cpu().numpy())
        train_x = torch.from_numpy(train_2d).float().view(*train_shape)
        val_x = torch.from_numpy(val_2d).float().view(*val_shape)
        if test_2d is not None:
            test_x = torch.from_numpy(test_2d).float().view(*test_x.shape)
        if return_scaler:
            return train_x, val_x, test_x, scaler
        return train_x, val_x, test_x

    def prepare_scaled_nn_datasets(self, train_data, val_data, test_data=None):
        self._last_sequence_dl_scaler = None
        if isinstance(train_data, ResidueTensorDataset):
            self._last_sequence_dl_scaler = train_data.dl_scaler
            return train_data, val_data, test_data
        if self.args.dl_scaler == "none" or self.args.encoder == "index":
            return train_data, val_data, test_data
        train_x, train_y = self._stack_dataset_xy(train_data)
        val_x, val_y = self._stack_dataset_xy(val_data)
        if train_x.numel() == 0 or val_x.numel() == 0:
            return train_data, val_data, test_data
        test_x, test_y = (
            self._stack_dataset_xy(test_data) if test_data is not None else (None, None)
        )
        if isinstance(train_x, ResidueModelInput):
            train_encoding = ResidueEncoding(train_x.features, train_x.mask)
            val_encoding = ResidueEncoding(val_x.features, val_x.mask)
            test_encoding = (
                ResidueEncoding(test_x.features, test_x.mask)
                if isinstance(test_x, ResidueModelInput)
                else None
            )
            (
                train_encoding,
                val_encoding,
                test_encoding,
                self._last_sequence_dl_scaler,
            ) = self._scale_residue_splits(
                train_encoding,
                val_encoding,
                test_encoding,
            )
            label_names = list(getattr(train_data, "label_names", []))
            return (
                ResidueTensorDataset(
                    train_encoding,
                    train_y,
                    label_names=label_names,
                    dl_scaler=self._last_sequence_dl_scaler,
                ),
                ResidueTensorDataset(
                    val_encoding,
                    val_y,
                    label_names=label_names,
                    dl_scaler=self._last_sequence_dl_scaler,
                ),
                (
                    ResidueTensorDataset(
                        test_encoding,
                        test_y,
                        label_names=label_names,
                        dl_scaler=self._last_sequence_dl_scaler,
                    )
                    if test_encoding is not None
                    else test_data
                ),
            )
        train_x, val_x, test_x, self._last_sequence_dl_scaler = self._scale_tensor_split(
            train_x,
            val_x,
            test_x,
            return_scaler=True,
        )
        train_ds = TensorDataset(train_x, train_y)
        val_ds = TensorDataset(val_x, val_y)
        test_ds = (
            TensorDataset(test_x, test_y)
            if test_data is not None and test_x is not None
            else test_data
        )
        label_names = list(getattr(train_data, "label_names", []))
        if label_names:
            train_ds.label_names = label_names
            val_ds.label_names = label_names
            if hasattr(test_ds, "__dict__"):
                test_ds.label_names = label_names
        return train_ds, val_ds, test_ds

    def _embedding_expected_dim(self, encoder):
        output_dim = int(getattr(encoder, "output_dim", 0) or 0)
        if output_dim:
            return output_dim
        if hasattr(encoder, "model") and hasattr(encoder.model, "config"):
            return encoder.model.config.hidden_size
        return None

    def _pretrained_cache_model_source(self, tag):
        tag = str(tag).removesuffix("_residue")
        resolvers = {
            "esm2": self.resolve_esm2_model_source,
            "protbert": self.resolve_protbert_model_source,
            "prott5": self.resolve_prott5_model_source,
            "esmc": self.resolve_esmc_model_source,
            "esm3": self.resolve_esm3_model_source,
        }
        return resolvers[tag]()

    def _embedding_cache_meta(self, tag, split, sequences, expected_dim=None):
        train_fasta = os.path.join(self.args.data_dir, "train", "seqs.fasta")
        with open(train_fasta, "rb") as handle:
            train_sha256 = hashlib.sha256(handle.read()).hexdigest()
        model_source = self._pretrained_cache_model_source(tag)
        meta = {
            "cache_version": CACHE_PROVENANCE_VERSION,
            "feature_fit_scope": "train_only",
            "split": str(split),
            "row_count": len(sequences),
            "input_sha256": digest_text_items(sequences),
            "train_sha256": train_sha256,
            "tokenizer": str(tag),
            "model_source": model_source,
            "model_fingerprint": model_source_fingerprint(model_source),
            "pooling": "residue_tokens"
            if str(tag).endswith("_residue")
            else (
                ESM3_POOLING
                if tag == "esm3"
                else (
                    ESM2_POOLING
                    if str(tag) == "esm2"
                    else (PROTBERT_POOLING if str(tag) == "protbert" else "attention_mask_mean")
                )
            ),
            "max_length": int(getattr(self.args, "max_length", 60)),
            "expected_dim": int(expected_dim) if expected_dim is not None else None,
        }
        if str(tag).removesuffix("_residue") == "prott5":
            meta.update(
                {
                    "pooling": PROTT5_POOLING,
                    "representation_contract_version": PROTT5_REPRESENTATION_CONTRACT_VERSION,
                    "output_kind": "residue_embeddings"
                    if str(tag).endswith("_residue")
                    else "sequence_embedding",
                    "special_tokens": "excluded",
                }
            )
        model_identity = getattr(self.args, "_resolved_pretrained_model_identity", None)
        if str(tag).removesuffix("_residue") == "esmc" and isinstance(model_identity, dict):
            meta["model_identity"] = dict(model_identity)
            if not str(tag).endswith("_residue"):
                meta["pooling"] = str(model_identity.get("pooling", meta["pooling"]))
        if str(tag).removesuffix("_residue") == "esm2" and isinstance(model_identity, dict):
            meta["model_identity"] = dict(model_identity)
            meta["representation_contract_version"] = ESM2_REPRESENTATION_CONTRACT_VERSION
            meta["pooling"] = "residue_tokens" if str(tag).endswith("_residue") else ESM2_POOLING
        if str(tag).removesuffix("_residue") == "protbert" and isinstance(model_identity, dict):
            meta["model_identity"] = dict(model_identity)
            meta["representation_contract_version"] = PROTBERT_REPRESENTATION_CONTRACT_VERSION
            meta["pooling"] = (
                "residue_tokens" if str(tag).endswith("_residue") else PROTBERT_POOLING
            )
        if str(tag).removesuffix("_residue") == "prott5" and isinstance(model_identity, dict):
            meta["model_identity"] = dict(model_identity)
            meta["representation_contract_version"] = PROTT5_REPRESENTATION_CONTRACT_VERSION
            meta["pooling"] = "residue_tokens" if str(tag).endswith("_residue") else PROTT5_POOLING
            meta["output_kind"] = (
                "residue_embeddings" if str(tag).endswith("_residue") else "sequence_embedding"
            )
            meta["special_tokens"] = "excluded"
            meta["input_normalization"] = "strip_whitespace_uppercase_UZOB_to_X"
        if str(tag).removesuffix("_residue") == "esm3" and isinstance(model_identity, dict):
            meta["model_identity"] = dict(model_identity)
            meta["representation_contract_version"] = ESM3_REPRESENTATION_CONTRACT_VERSION
            meta["pooling"] = "residue_tokens" if str(tag).endswith("_residue") else ESM3_POOLING
            meta["output_kind"] = (
                "residue_embeddings" if str(tag).endswith("_residue") else "sequence_embedding"
            )
            meta["special_tokens"] = "excluded"
            meta["input_normalization"] = "strip_whitespace_uppercase_validate_alphabet"
        runtime_environment = getattr(self.args, "_resolved_pretrained_runtime_environment", None)
        if str(tag).removesuffix("_residue") in {
            "esm2",
            "protbert",
            "prott5",
            "esmc",
            "esm3",
        } and isinstance(runtime_environment, dict):
            meta["runtime_environment"] = dict(runtime_environment)
        if str(tag) == "esm3":
            meta["dtype"] = "float32"
        return meta

    def _load_embedding_cache(self, save_path, tag, split, sequences, expected_dim=None):
        meta_path = f"{save_path}.meta.json"
        expected_meta = self._embedding_cache_meta(tag, split, sequences, expected_dim)
        if not os.path.exists(save_path) or not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                actual_meta = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if actual_meta != expected_meta:
            print(f"[{tag}] ignoring stale embedding cache metadata: {save_path}", flush=True)
            return None
        try:
            features = torch.load(save_path, map_location="cpu", weights_only=True)
        except TypeError:
            features = torch.load(save_path, map_location="cpu")
        valid = (
            isinstance(features, torch.Tensor)
            and features.ndim == 2
            and features.shape[0] == len(sequences)
        )
        if valid and expected_dim is not None:
            valid = features.shape[1] == expected_dim
        if valid:
            print(
                f"[{tag}] loaded {split} embeddings cache: {save_path} shape={tuple(features.shape)}",
                flush=True,
            )
            return features
        stale_shape = (
            tuple(features.shape) if isinstance(features, torch.Tensor) else type(features).__name__
        )
        print(
            f"[{tag}] stale {split} embeddings cache removed: {save_path} "
            f"shape={stale_shape}, expected_rows={len(sequences)}, expected_dim={expected_dim}",
            flush=True,
        )
        os.remove(save_path)
        return None

    def _save_embeddings(self, save_path, tag, split, sequences, features, expected_dim=None):
        torch.save(features, save_path)
        with open(f"{save_path}.meta.json", "w", encoding="utf-8") as handle:
            json.dump(
                self._embedding_cache_meta(tag, split, sequences, expected_dim),
                handle,
                indent=2,
                sort_keys=True,
            )
        print(
            f"[{tag}] saved {split} embeddings: {save_path} shape={tuple(features.shape)}",
            flush=True,
        )
        return features

    def _extract_tokenized_embeddings(
        self, tokenizer, encoder, sequences, split, save_path, tag, expected_dim=None
    ):
        cached = self._load_embedding_cache(save_path, tag, split, sequences, expected_dim)
        if cached is not None:
            return cached
        batch_size = max(1, int(getattr(self.args, "batch_size", 1)))
        total_batches = (len(sequences) + batch_size - 1) // batch_size
        print(
            f"[{tag}] extracting {split} embeddings: "
            f"{len(sequences)} samples, batch_size={batch_size}, batches={total_batches}",
            flush=True,
        )
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(
                range(0, len(sequences), batch_size),
                total=total_batches,
                desc=f"{tag} {split} embeddings",
            ):
                batch_seqs = sequences[i : i + batch_size]
                inputs = tokenizer.batch_tokenize(batch_seqs, device=self.device)
                embeddings = encoder.encode(inputs).cpu()
                all_embeddings.append(embeddings)
        if all_embeddings:
            features = torch.cat(all_embeddings, dim=0)
        else:
            features = torch.empty((0, expected_dim or 0))
        return self._save_embeddings(save_path, tag, split, sequences, features, expected_dim)

    def _extract_sequence_encoder_embeddings(self, encoder, sequences, split, save_path, tag):
        cached = self._load_embedding_cache(save_path, tag, split, sequences)
        if cached is not None:
            return cached
        batch_size = max(1, int(getattr(self.args, "batch_size", 1)))
        total_batches = (len(sequences) + batch_size - 1) // batch_size
        print(
            f"[{tag}] extracting {split} embeddings: "
            f"{len(sequences)} samples, batch_size={batch_size}, batches={total_batches}",
            flush=True,
        )
        all_embeddings = []
        for i in tqdm(
            range(0, len(sequences), batch_size),
            total=total_batches,
            desc=f"{tag} {split} embeddings",
        ):
            batch_seqs = sequences[i : i + batch_size]
            embeddings = encoder.encode_sequences(batch_seqs).cpu()
            all_embeddings.append(embeddings)
        features = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty((0, 0))
        return self._save_embeddings(save_path, tag, split, sequences, features.float())

    def get_or_extract_esm_embeddings(self, tokenizer, encoder, sequences, split):
        os.makedirs(self.esm_feature_dir, exist_ok=True)
        save_path = os.path.join(self.esm_feature_dir, f"{split}_embeddings.pt")
        return self._extract_tokenized_embeddings(
            tokenizer,
            encoder,
            sequences,
            split,
            save_path,
            "esm2",
            self._embedding_expected_dim(encoder),
        )

    def get_or_extract_esm_residue_embeddings(self, tokenizer, encoder, sequences, split):
        os.makedirs(self.esm_feature_dir, exist_ok=True)
        save_path = os.path.join(
            self.esm_feature_dir, f"{split}_residue_l{self.args.max_length}.pt"
        )
        expected_dim = self._embedding_expected_dim(encoder)
        cached = None
        meta_path = f"{save_path}.meta.json"
        if os.path.exists(save_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                expected_meta = self._embedding_cache_meta(
                    "esm2_residue", split, sequences, expected_dim
                )
                if all(metadata.get(key) == value for key, value in expected_meta.items()):
                    try:
                        cached = torch.load(save_path, map_location="cpu", weights_only=True)
                    except TypeError:
                        cached = torch.load(save_path, map_location="cpu")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                cached = None
        if (
            isinstance(cached, torch.Tensor)
            and cached.ndim == 3
            and cached.shape[1:] == (self.args.max_length, expected_dim)
        ):
            return cached
        all_embeddings = []
        batch_size = max(1, int(getattr(self.args, "batch_size", 1)))
        for start in tqdm(
            range(0, len(sequences), batch_size), desc=f"esm2 {split} residue embeddings"
        ):
            batch = sequences[start : start + batch_size]
            inputs = tokenizer.batch_tokenize(batch, device=self.device)
            states = encoder.encode_residues(
                inputs, [len(seq) for seq in batch], self.args.max_length
            )
            all_embeddings.append(states)
        features = (
            torch.cat(all_embeddings, dim=0)
            if all_embeddings
            else torch.empty((0, self.args.max_length, expected_dim))
        )
        torch.save(features, save_path)
        with open(f"{save_path}.meta.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    **self._embedding_cache_meta("esm2_residue", split, sequences, expected_dim),
                    "shape": list(features.shape),
                    "alignment": "BOS/EOS/padding stripped; one node per input residue",
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        return features

    def _residue_feature_dir(self, tag):
        directories = {
            "esm2": self.esm_feature_dir,
            "protbert": self.protbert_feature_dir,
            "prott5": self.prott5_feature_dir,
            "esmc": self.esmc_feature_dir,
            "esm3": self.esm3_feature_dir,
        }
        return directories[str(tag)]

    def _residue_cache_meta(self, tag, split, sequences, expected_dim=None):
        return {
            **self._embedding_cache_meta(f"{tag}_residue", split, sequences, expected_dim),
            "alignment_version": RESIDUE_ALIGNMENT_VERSION,
            "output_kind": "residue_embeddings",
            "padding": "right_zero",
            "mask": "explicit_boolean",
        }

    def get_or_extract_pretrained_residue_encoding(
        self,
        tag,
        tokenizer,
        encoder,
        sequences,
        split,
    ):
        feature_dir = self._residue_feature_dir(tag)
        os.makedirs(feature_dir, exist_ok=True)
        save_path = os.path.join(
            feature_dir,
            f"{split}_residue_v{RESIDUE_ALIGNMENT_VERSION}_l{self.args.max_length}.pt",
        )
        expected_dim = self._embedding_expected_dim(encoder)
        expected_meta = self._residue_cache_meta(tag, split, sequences, expected_dim)
        meta_path = f"{save_path}.meta.json"
        if os.path.exists(save_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    actual_meta = json.load(handle)
                if actual_meta == expected_meta:
                    try:
                        payload = torch.load(save_path, map_location="cpu", weights_only=True)
                    except TypeError:
                        payload = torch.load(save_path, map_location="cpu")
                    if isinstance(payload, dict):
                        cached = ResidueEncoding(payload.get("features"), payload.get("mask"))
                        valid = (
                            cached.features.shape[0] == len(sequences)
                            and cached.features.shape[1] == self.args.max_length
                            and (expected_dim is None or cached.features.shape[2] == expected_dim)
                        )
                        if valid:
                            print(
                                f"[{tag}_residue] loaded {split} cache: {save_path} "
                                f"shape={tuple(cached.features.shape)}",
                                flush=True,
                            )
                            return cached
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            print(f"[{tag}_residue] ignoring stale residue cache: {save_path}", flush=True)

        batches = []
        batch_size = max(1, int(getattr(self.args, "batch_size", 1)))
        for start in tqdm(
            range(0, len(sequences), batch_size),
            desc=f"{tag} {split} residue embeddings",
        ):
            batch_sequences = sequences[start : start + batch_size]
            if tag == "esm3":
                batch = encoder.encode_residue_batch(batch_sequences, self.args.max_length)
            else:
                inputs = tokenizer.batch_tokenize_residues(
                    batch_sequences,
                    self.args.max_length,
                    device=self.device,
                )
                sequence_lengths = [
                    len(tokenizer.normalize_sequence(sequence)[: self.args.max_length])
                    if tag == "esm2"
                    else (
                        tokenizer.residue_length(sequence, self.args.max_length)
                        if tag in {"protbert", "prott5"}
                        else min(len(sequence), self.args.max_length)
                    )
                    for sequence in batch_sequences
                ]
                batch = encoder.encode_residue_batch(
                    inputs,
                    sequence_lengths,
                    self.args.max_length,
                )
            batches.append(batch)
        if batches:
            encoding = ResidueEncoding(
                torch.cat([batch.features for batch in batches], dim=0),
                torch.cat([batch.mask for batch in batches], dim=0),
            )
        else:
            encoding = ResidueEncoding(
                torch.empty((0, self.args.max_length, expected_dim or 0), dtype=torch.float16),
                torch.empty((0, self.args.max_length), dtype=torch.bool),
            )
        if encoding.features.numel() and not torch.isfinite(encoding.features.float()).all():
            raise RuntimeError(f"{tag} residue extraction produced non-finite values")
        torch.save(
            {"features": encoding.features, "mask": encoding.mask},
            save_path,
        )
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(expected_meta, handle, indent=2, sort_keys=True)
        print(
            f"[{tag}_residue] saved {split} cache: {save_path} "
            f"shape={tuple(encoding.features.shape)}",
            flush=True,
        )
        return encoding

    @staticmethod
    def _transform_residue_encoding(encoding, scaler):
        features = encoding.features.float().clone()
        mask = encoding.mask.bool()
        if scaler is not None and bool(mask.any()):
            values = features[mask].cpu().numpy()
            features[mask] = torch.from_numpy(scaler.transform(values)).float()
        features[~mask] = 0.0
        return ResidueEncoding(features, mask)

    def _scale_residue_splits(self, train, val, test=None):
        if self.args.dl_scaler == "none":
            return (
                self._transform_residue_encoding(train, None),
                self._transform_residue_encoding(val, None),
                self._transform_residue_encoding(test, None) if test is not None else None,
                None,
            )
        train_values = train.features.float()[train.mask.bool()].cpu().numpy()
        if train_values.shape[0] == 0:
            raise SystemExit("pretrained residue training split contains no valid residues")
        scaler = StandardScaler().fit(train_values)
        return (
            self._transform_residue_encoding(train, scaler),
            self._transform_residue_encoding(val, scaler),
            self._transform_residue_encoding(test, scaler) if test is not None else None,
            scaler,
        )

    def _pretrained_residue_artifact_info(
        self,
        tag,
        model_name,
        input_dim,
        label_names,
    ):
        architecture = str(self.args.sequence_model or self.args.model).lower()
        residue = {
            "alignment_version": RESIDUE_ALIGNMENT_VERSION,
            "alignment": "one node or position per input residue; special tokens stripped",
            "adapter": "layer_norm+linear_projection",
            "adapter_dim": int(self.args.residue_adapter_dim),
            "architecture": architecture,
            "mask": "explicit_boolean",
            "padding": "right_zero",
        }
        if architecture in {"gcn", "gat"}:
            residue["graph"] = {
                "node_definition": "one node per amino-acid residue",
                "edges": "bidirectional adjacent residues plus self-loops",
            }
        pretrained = {
            "tokenizer": tag,
            "model_name": model_name,
            "max_length": int(self.args.max_length),
            "pooling": "residue_tokens",
            "output_kind": "residue_embeddings",
            "input_dim": int(input_dim),
            "frozen": True,
        }
        if tag == "esm2":
            pretrained.update(
                {
                    "representation_contract_version": ESM2_REPRESENTATION_CONTRACT_VERSION,
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_validate_alphabet",
                    "truncation": "max_length_residues_then_BOS_EOS",
                }
            )
        if tag == "protbert":
            pretrained.update(
                {
                    "representation_contract_version": PROTBERT_REPRESENTATION_CONTRACT_VERSION,
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_UZOB_to_X",
                    "truncation": "max_length_residues_then_CLS_SEP",
                }
            )
        if tag == "prott5":
            pretrained.update(
                {
                    "representation_contract_version": PROTT5_REPRESENTATION_CONTRACT_VERSION,
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_UZOB_to_X",
                    "truncation": "max_length_residues_then_EOS",
                }
            )
        if tag == "esm3":
            pretrained.update(
                {
                    "representation_contract_version": ESM3_REPRESENTATION_CONTRACT_VERSION,
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_validate_alphabet",
                    "truncation": "max_length_residues_then_CLS_EOS",
                }
            )
        model_identity = getattr(self.args, "_resolved_pretrained_model_identity", None)
        if isinstance(model_identity, dict):
            pretrained["model_identity"] = dict(model_identity)
        adjustment = getattr(self.args, "prott5_max_length_adjustment", None)
        if tag == "prott5" and isinstance(adjustment, dict):
            pretrained["max_length_adjustment"] = dict(adjustment)
        info = {
            "input_schema": "peptide_sequences",
            "pretrained": pretrained,
            "residue": residue,
            "label_names": list(label_names or []),
        }
        runtime_environment = getattr(self.args, "_resolved_pretrained_runtime_environment", None)
        if tag in {"esm2", "protbert", "prott5", "esmc", "esm3"} and isinstance(
            runtime_environment, dict
        ):
            info["runtime_environment"] = dict(runtime_environment)
        return info

    def run_pretrained_residue(
        self,
        tag,
        tokenizer,
        encoder,
        model_name,
        train_sequences,
        train_y,
        val_sequences,
        val_y,
        test_sequences,
        test_y,
        label_names,
    ):
        train_encoding = self.get_or_extract_pretrained_residue_encoding(
            tag, tokenizer, encoder, train_sequences, "train"
        )
        val_encoding = self.get_or_extract_pretrained_residue_encoding(
            tag, tokenizer, encoder, val_sequences, "val"
        )
        test_encoding = (
            self.get_or_extract_pretrained_residue_encoding(
                tag, tokenizer, encoder, test_sequences, "test"
            )
            if test_sequences
            else None
        )
        train_encoding, val_encoding, test_encoding, dl_scaler = self._scale_residue_splits(
            train_encoding,
            val_encoding,
            test_encoding,
        )
        input_dim = int(train_encoding.features.shape[-1])
        artifact_info = self._pretrained_residue_artifact_info(
            tag,
            model_name,
            input_dim,
            label_names,
        )
        train_ds = ResidueTensorDataset(
            train_encoding,
            train_y,
            label_names=label_names,
            dl_scaler=dl_scaler,
        )
        val_ds = ResidueTensorDataset(
            val_encoding,
            val_y,
            label_names=label_names,
            dl_scaler=dl_scaler,
        )
        test_ds = (
            ResidueTensorDataset(
                test_encoding,
                test_y,
                label_names=label_names,
                dl_scaler=dl_scaler,
            )
            if test_encoding is not None
            else None
        )
        model = PretrainedResiduePredictor(
            architecture=self.args.sequence_model or self.args.model,
            input_dim=input_dim,
            adapter_dim=self.args.residue_adapter_dim,
            max_length=self.args.max_length,
            num_classes=int(train_y.shape[1]),
            dropout=self.args.dropout,
            **prediction_head_kwargs(self.args),
        )
        self.args._artifact_source = "pretrained_residue_embeddings"
        self.args._artifact_info = artifact_info
        if exports_sequence_embeddings(self.args):
            return self.run_sequence_export(
                None,
                train_ds,
                val_ds,
                test_ds,
                model_override=model,
            )
        return self.run_nn(
            None,
            train_ds,
            val_ds,
            test_ds,
            model_override=model,
        )

    def get_or_extract_protbert_embeddings(self, tokenizer, encoder, sequences, split):
        os.makedirs(self.protbert_feature_dir, exist_ok=True)
        save_path = os.path.join(self.protbert_feature_dir, f"{split}_embeddings.pt")
        return self._extract_tokenized_embeddings(
            tokenizer,
            encoder,
            sequences,
            split,
            save_path,
            "protbert",
            self._embedding_expected_dim(encoder),
        )

    def get_or_extract_prott5_embeddings(self, tokenizer, encoder, sequences, split):
        os.makedirs(self.prott5_feature_dir, exist_ok=True)
        save_path = os.path.join(self.prott5_feature_dir, f"{split}_embeddings.pt")
        return self._extract_tokenized_embeddings(
            tokenizer,
            encoder,
            sequences,
            split,
            save_path,
            "prott5",
            self._embedding_expected_dim(encoder),
        )

    def get_or_extract_esmc_embeddings(self, tokenizer, encoder, sequences, split):
        os.makedirs(self.esmc_feature_dir, exist_ok=True)
        save_path = os.path.join(self.esmc_feature_dir, f"{split}_embeddings.pt")
        return self._extract_tokenized_embeddings(
            tokenizer,
            encoder,
            sequences,
            split,
            save_path,
            "esmc",
            self._embedding_expected_dim(encoder),
        )

    def get_or_extract_esm3_embeddings(self, tokenizer, encoder, sequences, split):
        os.makedirs(self.esm3_feature_dir, exist_ok=True)
        save_path = os.path.join(self.esm3_feature_dir, f"{split}_embeddings.pt")
        return self._extract_sequence_encoder_embeddings(
            encoder, sequences, split, save_path, "esm3"
        )

    def run_nn(
        self,
        featurizer,
        train_data,
        val_data,
        test_data=None,
        return_model=False,
        evaluate_test=True,
        model_override=None,
    ):
        training_mode = str(getattr(self.args, "training_mode", "") or "").strip().lower()
        if training_mode:
            validate_training_mode_classifier(
                training_mode,
                str(getattr(self.args, "classifier", "") or "mlp").strip().lower(),
            )
        if hasattr(train_data, "get_num_classes"):
            num_classes = train_data.get_num_classes()
        else:
            num_classes = int(train_data.tensors[1].shape[1])
        train_ids = list(getattr(train_data, "sequence_ids", []))
        val_ids = list(getattr(val_data, "sequence_ids", []))
        test_ids = list(getattr(test_data, "sequence_ids", [])) if test_data is not None else []
        train_data, val_data, test_data = self.prepare_scaled_nn_datasets(
            train_data, val_data, test_data
        )
        train_inputs, train_targets = self._stack_dataset_xy(train_data)
        self._configure_intrinsic_semantics(
            train_inputs,
            train_targets,
            train_dataset=train_data,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )
        train_loader = self._semantic_train_loader(train_data)
        val_loader = DataLoader(val_data, batch_size=self.args.batch_size, shuffle=False)
        model = model_override or self.model_factory.create(
            self.args.model,
            self.args.encoder,
            featurizer,
            num_classes,
            dropout=self.args.dropout,
            **prediction_head_kwargs(self.args),
        )
        model = model.to(self.device)
        if isinstance(train_inputs, ResidueModelInput):
            semantic_sample_inputs = train_inputs.batch_slice(0, min(2, train_inputs.shape[0])).to(
                self.device
            )
        else:
            semantic_sample_inputs = train_inputs[: min(2, train_inputs.shape[0])].to(self.device)
        model = self._prepare_semantic_model(
            model,
            sample_inputs=semantic_sample_inputs,
        )
        optimizer = optim.AdamW(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        criterion = self._multilabel_loss(train_targets)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=1e-5)
        self.print_semantic_similarity_status(
            f"{self.args.tokenizer}/{self.args.encoder}/{self.args.model}"
        )
        best_val_f1 = 0.0
        best_checkpoint_key = None
        best_epoch = 0
        best_state = None
        no_improve_epochs = 0
        for epoch in range(self.args.epochs):
            if hasattr(train_data, "set_epoch"):
                train_data.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            semantic_anchor_count = 0
            semantic_positive_count = 0
            semantic_zero_count = 0
            for inputs, targets, row_ids in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}"
            ):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                logits, loss, semantic_output = self.training_loss(
                    model, inputs, targets, criterion, row_ids=row_ids
                )
                loss.backward()
                if self.args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip_norm)
                optimizer.step()
                total_loss += loss.item()
                if semantic_output is not None:
                    semantic_anchor_count += semantic_output.anchors
                    semantic_positive_count += semantic_output.positive_relations
                    semantic_zero_count += semantic_output.negative_relations
            val_metrics = self._validation_metrics_for_checkpoint(model, val_loader)
            val_acc, val_sacc, val_f1, val_auc = (
                val_metrics["acc"],
                val_metrics["sacc"],
                val_metrics["f1"],
                val_metrics["auc"],
            )
            train_loss = total_loss / len(train_loader)
            val_loss = self.eval_loss(model, val_loader, criterion)
            msg = (
                f"[{self.args.tokenizer}/{self.args.encoder}/{self.args.model}] Epoch {epoch + 1}/{self.args.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Val SAcc: {val_sacc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
                f"{Evaluator.format_extra_metrics(val_metrics, prefix='Val')}"
            )
            if getattr(self.args, "report_aupr", True):
                val_targets, val_scores = collect_nn_scores(model, val_loader, self.device)
                msg += f" | Val AUPR: {self._compute_aupr_from_scores(val_targets, val_scores):.4f}"
            if self.semantic_similarity_enabled():
                msg += (
                    f" | Semantic Anchors: {semantic_anchor_count}"
                    f" | Semantic Positive Pairs: {semantic_positive_count}"
                    f" | Semantic Zero Pairs: {semantic_zero_count}"
                )
            print(msg)
            (
                best_checkpoint_key,
                best_epoch,
                best_state,
                checkpoint_improved,
            ) = self._update_validation_checkpoint(
                model,
                val_metrics=val_metrics,
                epoch=epoch + 1,
                best_key=best_checkpoint_key,
                best_epoch=best_epoch,
                best_state=best_state,
            )
            if checkpoint_improved:
                best_val_f1 = float(val_f1)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            scheduler.step()
            if (
                self.args.early_stop_patience > 0
                and no_improve_epochs >= self.args.early_stop_patience
            ):
                print(
                    f"[{self.args.tokenizer}/{self.args.encoder}/{self.args.model}] Early stop at epoch {epoch + 1}"
                )
                break
        print(
            f"[{self.args.tokenizer}/{self.args.encoder}/{self.args.model}] Finished. Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})"
        )
        self._restore_validation_f1_checkpoint(model, best_state, best_epoch)
        val_targets, val_scores, operating_point = self._fit_nn_operating_point(
            model,
            val_loader,
            label_names=getattr(train_data, "label_names", []),
        )
        self._save_sequence_nn_artifact(
            model,
            train_data,
            getattr(self, "_last_sequence_dl_scaler", None),
            operating_point=operating_point,
        )
        test_targets, test_scores = None, None
        if evaluate_test and test_data is not None:
            test_loader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False)
            test_targets, test_scores = collect_nn_scores(model, test_loader, self.device)
            test_metrics = Evaluator.score_metrics(test_targets, test_scores, operating_point)
            test_acc, test_sacc, test_f1, test_auc = (
                test_metrics["acc"],
                test_metrics["sacc"],
                test_metrics["f1"],
                test_metrics["auc"],
            )
            test_loss = self.eval_loss(model, test_loader, criterion)
            msg = (
                f"[{self.args.tokenizer}/{self.args.encoder}/{self.args.model}] Test | Loss: {test_loss:.4f} | "
                f"Acc: {test_acc:.4f} | SAcc: {test_sacc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}"
                f"{Evaluator.format_extra_metrics(test_metrics)}"
            )
            if getattr(self.args, "report_aupr", True):
                msg += f" | AUPR: {float(test_metrics.get('aupr') or 0.0):.4f}"
            print(msg)
        self._save_protocol_evaluation(
            val_targets, val_scores, operating_point, test_targets, test_scores
        )
        self._write_score_analysis(
            val_targets,
            val_scores,
            test_targets,
            test_scores,
            operating_point=operating_point,
        )
        if return_model:
            return best_val_f1, model, train_data, val_data, test_data
        return best_val_f1

    def run_ml_on_embeddings(
        self,
        train_embeddings,
        train_y,
        val_embeddings,
        val_y,
        test_embeddings=None,
        test_y=None,
        classifier=None,
        artifact_source="embeddings",
        artifact_info=None,
    ):
        train_ds = TensorDataset(train_embeddings, train_y)
        val_ds = TensorDataset(val_embeddings, val_y)
        test_ds = (
            TensorDataset(test_embeddings, test_y)
            if test_embeddings is not None and test_y is not None
            else None
        )
        classifier = classifier or self.args.model
        previous_source = getattr(self.args, "_artifact_source", "sequence_features")
        previous_info = getattr(self.args, "_artifact_info", None)
        self.args._artifact_source = artifact_source
        self.args._artifact_info = artifact_info or {}
        try:
            if classifier == "svm":
                return self.ml_trainer.run_svm(train_ds, val_ds, test_ds)
            if classifier == "calibrated_svm":
                return self.ml_trainer.run_calibrated_svm(train_ds, val_ds, test_ds)
            if classifier == "rf":
                return self.ml_trainer.run_rf(train_ds, val_ds, test_ds)
            if classifier == "extratrees":
                return self.ml_trainer.run_extratrees(train_ds, val_ds, test_ds)
            if classifier == "histgb":
                return self.ml_trainer.run_histgb(train_ds, val_ds, test_ds)
            if classifier == "xgb":
                return self.ml_trainer.run_xgb(train_ds, val_ds, test_ds)
            if classifier == "lgb":
                return self.ml_trainer.run_lgb(train_ds, val_ds, test_ds)
            if classifier == "logreg":
                return self.ml_trainer.run_logreg(train_ds, val_ds, test_ds)
            raise SystemExit(f"unsupported classifier for embedding features: {classifier}")
        finally:
            self.args._artifact_source = previous_source
            self.args._artifact_info = previous_info

    def extract_model_embeddings(self, model, dataset, split_name):
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        model.eval()
        all_embeddings, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f"Extracting {split_name} embeddings"):
                inputs = inputs.to(self.device)
                _, embeddings = model.forward_with_embedding(inputs)
                all_embeddings.append(embeddings.detach().cpu())
                all_labels.append(labels.detach().cpu())
        if not all_embeddings:
            return torch.empty((0, 0)), torch.empty((0, 0))
        return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)

    def run_sequence_export(
        self,
        featurizer,
        train_data,
        val_data,
        test_data=None,
        model_override=None,
    ):
        classifier = str(getattr(self.args, "classifier", "") or "").strip().lower()
        training_mode = str(getattr(self.args, "training_mode", "") or "").strip().lower()
        if training_mode:
            validate_training_mode_classifier(training_mode, classifier)
        if classifier not in CLASSICAL_CLASSIFIERS:
            raise SystemExit(
                "export_embeddings requires classifier in {svm,rf,extratrees,histgb,xgb,lgb,logreg}"
            )
        tag = f"{self.args.tokenizer}/{self.args.encoder}/{self.args.sequence_model}"
        print(
            f"[{tag}] Stage 1/2: Train modeling backbone with a temporary neural head "
            "for embedding learning"
        )
        emit_run_progress(
            "representation",
            "running",
            0,
            max(1, int(self.args.epochs)),
            feature="sequence",
            unit="epoch",
        )
        _, model, train_data, val_data, test_data = self.run_nn(
            featurizer,
            train_data,
            val_data,
            test_data,
            return_model=True,
            evaluate_test=False,
            model_override=model_override,
        )
        emit_run_progress(
            "representation",
            "complete",
            max(1, int(self.args.epochs)),
            max(1, int(self.args.epochs)),
            feature="sequence",
            unit="epoch",
        )
        print(f"[{tag}] Stage 1/2 complete: restored best encoder checkpoint for embedding export")
        split_total = 3 if test_data is not None else 2
        emit_run_progress(
            "embedding_export",
            "running",
            0,
            split_total,
            feature=classifier,
            unit="split",
            split="train",
        )
        train_embeddings, train_y = self.extract_model_embeddings(model, train_data, "train")
        emit_run_progress(
            "embedding_export",
            "running",
            1,
            split_total,
            feature=classifier,
            unit="split",
            split="val",
        )
        val_embeddings, val_y = self.extract_model_embeddings(model, val_data, "val")
        if test_data is not None:
            emit_run_progress(
                "embedding_export",
                "running",
                2,
                split_total,
                feature=classifier,
                unit="split",
                split="test",
            )
            test_embeddings, test_y = self.extract_model_embeddings(model, test_data, "test")
        else:
            test_embeddings, test_y = None, None
        emit_run_progress(
            "embedding_export",
            "complete",
            split_total,
            split_total,
            feature=classifier,
            unit="split",
            split="complete",
        )
        encoder_path = self._save_sequence_embedding_encoder(model)
        profile_info = self._sequence_profile_generation_artifact_info()
        artifact_info = self._sequence_embedding_artifact_info(
            train_data,
            encoder_path,
            train_embeddings.shape[1] if train_embeddings.ndim > 1 else 0,
            classifier,
            dl_scaler=getattr(self, "_last_sequence_dl_scaler", None),
            profile_info=profile_info,
            architecture_contract=sequence_predictor_contract(model),
        )
        source_info = dict(getattr(self.args, "_artifact_info", {}) or {})
        for key in ("pretrained", "residue"):
            if key in source_info:
                artifact_info[key] = source_info[key]
        print(
            f"[{tag}] Stage 2/2: Train downstream classifier on exported embeddings: {classifier}"
        )
        embedding_artifact_source = (
            "pretrained_residue_embedding_features"
            if str(getattr(self.args, "_artifact_source", "")) == "pretrained_residue_embeddings"
            else "sequence_embedding_features"
        )
        if classifier == "svm" and str(getattr(self.args, "artifact_dir", "") or ""):
            save_svm_resume_bundle(
                self.args.artifact_dir,
                train_x=train_embeddings,
                train_y=train_y,
                val_x=val_embeddings,
                val_y=val_y,
                test_x=test_embeddings,
                test_y=test_y,
                task_type=self.args.task_type,
                artifact_source=embedding_artifact_source,
                artifact_info=artifact_info,
            )
        return self.run_ml_on_embeddings(
            train_embeddings,
            train_y,
            val_embeddings,
            val_y,
            test_embeddings,
            test_y,
            classifier=classifier,
            artifact_source=embedding_artifact_source,
            artifact_info=artifact_info,
        )

    def run_esm(self):
        train_seqs = self.load_sequences(os.path.join(self.args.data_dir, "train", "seqs.fasta"))
        val_seqs = self.load_sequences(os.path.join(self.args.data_dir, "val", "seqs.fasta"))
        train_y = self.load_labels(os.path.join(self.args.data_dir, "train", "labels.csv"))
        val_y = self.load_labels(os.path.join(self.args.data_dir, "val", "labels.csv"))
        test_seq_path = os.path.join(self.args.data_dir, "test", "seqs.fasta")
        test_label_path = os.path.join(self.args.data_dir, "test", "labels.csv")
        has_test = (
            bool(getattr(self.args, "evaluate_test", True))
            and os.path.exists(test_seq_path)
            and os.path.exists(test_label_path)
        )
        test_seqs = self.load_sequences(test_seq_path) if has_test else []
        test_y = self.load_labels(test_label_path) if has_test else torch.empty(0)
        if len(train_seqs) != len(train_y):
            raise ValueError(
                f"ESM2 train split has {len(train_seqs)} sequences but {len(train_y)} label rows"
            )
        if len(val_seqs) != len(val_y):
            raise ValueError(
                f"ESM2 validation split has {len(val_seqs)} sequences but {len(val_y)} label rows"
            )
        if has_test and len(test_seqs) != len(test_y):
            raise ValueError(
                f"ESM2 test split has {len(test_seqs)} sequences but {len(test_y)} label rows"
            )
        label_names = self._load_label_names(
            os.path.join(self.args.data_dir, "train", "labels.csv"), train_y.shape[1]
        )
        train_ds = SequenceDataset(train_seqs, train_y)
        model_name = self.resolve_esm2_model_source()
        print(
            f"[esm2] {'using local model' if os.path.exists(model_name) else 'local model not found, using remote'}: {model_name}"
        )
        tokenizer = EsmTokenizer(model_name=model_name, max_length=self.args.max_length)
        encoder = EsmEncoder(model_name=model_name, device=self.device)
        self.args._resolved_pretrained_model_source = model_name
        self.args._resolved_pretrained_model_identity = dict(encoder.model_identity)
        self.args._resolved_pretrained_runtime_environment = dict(encoder.runtime_environment)
        print(
            format_esm2_provenance(encoder.model_identity, encoder.runtime_environment), flush=True
        )
        input_dim = int(
            getattr(encoder, "output_dim", 0) or getattr(encoder.model.config, "d_model", 0)
        )
        if input_dim <= 0:
            raise RuntimeError("ProtT5 encoder did not expose a positive hidden dimension")
        num_classes = train_y.shape[1]
        if self.args.sequence_model:
            return self.run_pretrained_residue(
                "esm2",
                tokenizer,
                encoder,
                model_name,
                train_seqs,
                train_y,
                val_seqs,
                val_y,
                test_seqs,
                test_y if has_test else None,
                label_names,
            )
        model = PredictionHeadClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=self.args.dropout,
            **prediction_head_kwargs(self.args),
        ).to(self.device)
        train_embeddings = self.get_or_extract_esm_embeddings(
            tokenizer, encoder, train_seqs, "train"
        )
        val_embeddings = self.get_or_extract_esm_embeddings(tokenizer, encoder, val_seqs, "val")
        test_embeddings = (
            self.get_or_extract_esm_embeddings(tokenizer, encoder, test_seqs, "test")
            if has_test
            else None
        )
        artifact_info = self._pretrained_embedding_artifact_info(
            "esm2", model_name, input_dim, label_names
        )
        if self.args.model in ["svm", "rf", "xgb", "lgb", "logreg"]:
            return self.run_ml_on_embeddings(
                train_embeddings,
                train_y,
                val_embeddings,
                val_y,
                test_embeddings,
                test_y if has_test else None,
                artifact_source="pretrained_embeddings",
                artifact_info=artifact_info,
            )
        if self.args.model != "mlp":
            raise SystemExit("esm2 supports model in {mlp,svm,rf,xgb,lgb,logreg,gcn,gat}")
        dl_scaler = None
        if self.args.dl_scaler != "none":
            train_embeddings, val_embeddings, test_embeddings, dl_scaler = self._scale_tensor_split(
                train_embeddings,
                val_embeddings,
                test_embeddings,
                return_scaler=True,
            )
        train_ds = TensorDataset(train_embeddings, train_y)
        val_ds = TensorDataset(val_embeddings, val_y)
        self._configure_intrinsic_semantics(
            train_embeddings,
            train_y,
            train_ids=self._semantic_sequence_ids(train_seqs),
            val_ids=self._semantic_sequence_ids(val_seqs),
            test_ids=self._semantic_sequence_ids(test_seqs) if has_test else [],
        )
        model = self._prepare_semantic_model(
            model,
            sample_inputs=train_embeddings[: min(2, train_embeddings.shape[0])].to(self.device),
        )
        optimizer = optim.AdamW(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        criterion = self._multilabel_loss(train_y, auto_default="balanced_bce")
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        best_val_f1 = 0.0
        best_epoch = 0
        best_state = None
        no_improve_epochs = 0
        train_loader = self._semantic_train_loader(train_ds)
        val_loader = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False)
        self.print_semantic_similarity_status("esm2/mlp")
        stage1_epochs = int(getattr(self.args, "tppredsc_stage1_epochs", 0) or 0)
        if stage1_epochs > 0:
            self._run_tppredsc_stage1(
                model,
                train_loader,
                train_y,
                scaler=dl_scaler,
                stage1_epochs=stage1_epochs,
            )
            stage2_epochs = int(getattr(self.args, "tppredsc_stage2_epochs", 10))
            stage2_lr = float(getattr(self.args, "tppredsc_stage2_lr", 1e-4))
            self.args.epochs = stage2_epochs
            self.args.tppredsc_stage1_epochs = 0
            self.args.enable_semantic_similarity = False
            optimizer.param_groups[0]["lr"] = stage2_lr
            scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs)
            print(
                f"[tppredsc-stage2] {self.args._resolved_multilabel_loss['resolved']} "
                f"fine-tune epochs={stage2_epochs} "
                f"lr={stage2_lr:g} (semantic auxiliary loss disabled)",
                flush=True,
            )
        for epoch in range(self.args.epochs):
            model.train()
            total_loss = 0.0
            for embeddings, targets, row_ids in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}"
            ):
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                logits, loss, _ = self.training_loss(
                    model, embeddings, targets, criterion, row_ids=row_ids
                )
                loss.backward()
                if self.args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip_norm)
                optimizer.step()
                total_loss += loss.item()
            val_metrics = self._validation_metrics_for_checkpoint(model, val_loader)
            val_acc, val_sacc, val_f1, val_auc = (
                val_metrics["acc"],
                val_metrics["sacc"],
                val_metrics["f1"],
                val_metrics["auc"],
            )
            train_loss = total_loss / len(train_loader)
            val_loss = self.eval_loss(model, val_loader, criterion)
            print(
                f"[esm2/mlp] Epoch {epoch + 1}/{self.args.epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val SAcc: {val_sacc:.4f} | "
                f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
                + Evaluator.format_extra_metrics(val_metrics, prefix="Val")
                + self._val_aupr_suffix(model, val_loader)
            )
            (
                best_val_f1,
                best_epoch,
                best_state,
                checkpoint_improved,
            ) = self._update_validation_f1_checkpoint(
                model,
                val_f1=val_f1,
                val_metrics=val_metrics,
                epoch=epoch + 1,
                best_val_f1=best_val_f1,
                best_epoch=best_epoch,
                best_state=best_state,
            )
            if checkpoint_improved:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            scheduler.step()
            if (
                self.args.early_stop_patience > 0
                and no_improve_epochs >= self.args.early_stop_patience
            ):
                print(f"[esm2/mlp] Early stop at epoch {epoch + 1}")
                break
        print(f"[esm2/mlp] Finished. Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})")
        self._restore_validation_f1_checkpoint(model, best_state, best_epoch)
        test_ds = TensorDataset(test_embeddings, test_y) if has_test else None
        self._finalize_pretrained_embedding_nn(
            "esm2", model, val_ds, test_ds, criterion, artifact_info, dl_scaler
        )
        return best_val_f1

    def _run_tppredsc_stage1(
        self,
        model,
        train_loader,
        train_labels,
        *,
        scaler=None,
        stage1_epochs=None,
    ):
        """Contrastive pretraining of sequence representations.

        Trains ``model`` (plus its semantic projector) with the in-batch
        multi-label supervised contrastive loss (or its SimCLR variant) for
        ``stage1_epochs`` epochs.  Augmented view embeddings, when configured
        via ``--semantic_augmented_views_dir``, replace the raw in-batch
        inputs as anchors (2N-row layout ``[view0; view1]``).
        """
        stage1_epochs = int(
            stage1_epochs
            if stage1_epochs is not None
            else getattr(self.args, "tppredsc_stage1_epochs", 0) or 0
        )
        if stage1_epochs <= 0:
            return
        mode = str(getattr(self.args, "semantic_contrastive_mode", "multilabel")).strip().lower()
        temperature = float(getattr(self.args, "semantic_contrastive_temperature", 0.07))
        theta = float(getattr(self.args, "semantic_jaccard_theta", 0.1))
        lr0 = float(getattr(self.args, "tppredsc_stage1_lr", 0.01))
        momentum = float(getattr(self.args, "tppredsc_stage1_momentum", 0.9))
        views_dir = str(getattr(self.args, "semantic_augmented_views_dir", "") or "")
        train_labels = train_labels.detach().to(self.device)
        view0 = view1 = None
        if views_dir and os.path.isdir(views_dir):
            view0 = torch.load(
                os.path.join(views_dir, "view0.pt"), map_location="cpu", weights_only=True
            )
            view1 = torch.load(
                os.path.join(views_dir, "view1.pt"), map_location="cpu", weights_only=True
            )
            if scaler is not None:

                def _scale(t):
                    return torch.from_numpy(scaler.transform(t.numpy().astype("float32"))).float()

                view0 = _scale(view0)
                view1 = _scale(view1)
            print(
                f"[tppredsc-stage1] augmented views loaded "
                f"{tuple(view0.shape)} / {tuple(view1.shape)} from {views_dir}",
                flush=True,
            )
        elif mode == "simclr":
            raise SystemExit("simclr contrastive mode requires --semantic_augmented_views_dir")
        rows_by_batch = view0 is not None
        optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=momentum)
        if not hasattr(model, "semantic_projector"):
            raise SystemExit(
                "Contrastive pretraining requires a semantic projector; "
                "enable_semantic_similarity must be true"
            )
        print(
            f"[tppredsc-stage1] mode={mode} epochs={stage1_epochs} "
            f"temperature={temperature:g} theta_w={theta:g} "
            f"optimizer=SGD(lr={lr0:g}, momentum={momentum:g}) views={rows_by_batch}",
            flush=True,
        )
        for epoch in range(1, stage1_epochs + 1):
            lr = lr0 * 0.1 ** (max(0, (epoch - 1) // (max(stage1_epochs // 3, 1))))
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            total_loss = 0.0
            total_anchors = 0
            step_count = 0
            for embeddings, targets, row_ids in train_loader:
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device)
                if rows_by_batch:
                    indices = torch.as_tensor(list(row_ids), dtype=torch.long)
                    anchors = torch.cat([view0[indices], view1[indices]], dim=0).to(self.device)
                    anchor_labels = torch.cat([targets, targets], dim=0)
                else:
                    anchors = embeddings
                    anchor_labels = targets
                optimizer.zero_grad()
                logits, embs = model.forward_with_embedding(anchors)
                if embs is None or not embs.requires_grad:
                    raise SystemExit("Contrastive pretraining requires a differentiable embedding")
                projected = model.semantic_projector(embs)
                loss, stats = semantic_batch_contrastive_loss(
                    projected,
                    anchor_labels if mode == "multilabel" else None,
                    temperature=temperature,
                    theta_w=theta,
                    mode=mode,
                )
                loss.backward()
                if getattr(self.args, "grad_clip_norm", 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(self.args.grad_clip_norm)
                    )
                optimizer.step()
                total_loss += float(loss.item())
                total_anchors += int(stats.get("anchors", 0))
                step_count += 1
            if step_count:
                print(
                    f"[tppredsc-stage1] Epoch {epoch}/{stage1_epochs} | "
                    f"Contrastive Loss: {total_loss / step_count:.4f} | "
                    f"Anchors: {total_anchors} | lr: {lr:g}",
                    flush=True,
                )
        print(f"[tppredsc-stage1] finished {stage1_epochs} contrastive epochs", flush=True)

    def resolve_esm2_model_source(self):
        raw_model = getattr(self.args, "esm2_model", ESM2_OFFICIAL_MODEL_ID)
        candidates = [
            os.path.join(self.project_root, "models", "esm2"),
            os.path.join(self.project_root, "models", "esm2_t30_150M_UR50D"),
            os.path.join(self.project_root, "models", "facebook", "esm2_t30_150M_UR50D"),
            os.path.expanduser("~/models/esm2"),
        ]
        resolved = _resolve_pretrained_model_source(
            self.project_root, raw_model, candidates, "ESM2", "esm2_model"
        )
        if os.path.isdir(resolved) and is_complete_esm2_snapshot(resolved):
            return resolved
        return resolve_esm2_source(
            self.project_root,
            raw_model,
            [resolved, *candidates],
        )

    def resolve_protbert_model_source(self):
        protbert_model = getattr(self.args, "protbert_model", "")
        legacy_model = getattr(self.args, "probert_model", "")
        raw_model = (
            legacy_model
            if legacy_model and protbert_model in ["", "Rostlab/prot_bert"]
            else protbert_model
        )
        return _resolve_pretrained_model_source(
            self.project_root,
            raw_model or PROTBERT_OFFICIAL_MODEL_ID,
            [
                os.path.join(self.project_root, "models", "protbert"),
                os.path.join(self.project_root, "models", "probert"),
                os.path.join(self.project_root, "models", "prot_bert"),
                os.path.join(self.project_root, "models", "Rostlab", "prot_bert"),
                os.path.expanduser("~/models/protbert"),
                os.path.expanduser("~/models/probert"),
            ],
            "ProtBERT",
            "protbert_model",
        )

    def run_protbert(self):
        train_seqs = self.load_sequences(os.path.join(self.args.data_dir, "train", "seqs.fasta"))
        val_seqs = self.load_sequences(os.path.join(self.args.data_dir, "val", "seqs.fasta"))
        train_y = self.load_labels(os.path.join(self.args.data_dir, "train", "labels.csv"))
        val_y = self.load_labels(os.path.join(self.args.data_dir, "val", "labels.csv"))
        test_seq_path = os.path.join(self.args.data_dir, "test", "seqs.fasta")
        test_label_path = os.path.join(self.args.data_dir, "test", "labels.csv")
        has_test = (
            bool(getattr(self.args, "evaluate_test", True))
            and os.path.exists(test_seq_path)
            and os.path.exists(test_label_path)
        )
        test_seqs = self.load_sequences(test_seq_path) if has_test else []
        test_y = self.load_labels(test_label_path) if has_test else torch.empty(0)
        if len(train_seqs) != len(train_y):
            min_len = min(len(train_seqs), len(train_y))
            train_seqs = train_seqs[:min_len]
            train_y = train_y[:min_len]
        if len(val_seqs) != len(val_y):
            min_len = min(len(val_seqs), len(val_y))
            val_seqs = val_seqs[:min_len]
            val_y = val_y[:min_len]
        if has_test and len(test_seqs) != len(test_y):
            min_len = min(len(test_seqs), len(test_y))
            test_seqs = test_seqs[:min_len]
            test_y = test_y[:min_len]
        label_names = self._load_label_names(
            os.path.join(self.args.data_dir, "train", "labels.csv"), train_y.shape[1]
        )
        model_name = self.resolve_protbert_model_source()
        print(
            f"[protbert] {'using local model' if os.path.exists(model_name) else 'local model not found, using remote'}: {model_name}"
        )
        tokenizer = ProtBertTokenizer(model_name=model_name, max_length=self.args.max_length)
        encoder = ProtBertEncoder(model_name=model_name, device=self.device)
        self.args._resolved_pretrained_model_identity = dict(encoder.model_identity)
        self.args._resolved_pretrained_runtime_environment = dict(encoder.runtime_environment)
        print(
            format_protbert_provenance(
                encoder.model_identity,
                encoder.runtime_environment,
            ),
            flush=True,
        )
        input_dim = encoder.model.config.hidden_size
        num_classes = train_y.shape[1]
        if self.args.sequence_model:
            return self.run_pretrained_residue(
                "protbert",
                tokenizer,
                encoder,
                model_name,
                train_seqs,
                train_y,
                val_seqs,
                val_y,
                test_seqs,
                test_y if has_test else None,
                label_names,
            )
        model = PredictionHeadClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=self.args.dropout,
            **prediction_head_kwargs(self.args),
        ).to(self.device)
        train_embeddings = self.get_or_extract_protbert_embeddings(
            tokenizer, encoder, train_seqs, "train"
        )
        val_embeddings = self.get_or_extract_protbert_embeddings(
            tokenizer, encoder, val_seqs, "val"
        )
        test_embeddings = (
            self.get_or_extract_protbert_embeddings(tokenizer, encoder, test_seqs, "test")
            if has_test
            else None
        )
        artifact_info = self._pretrained_embedding_artifact_info(
            "protbert", model_name, input_dim, label_names
        )
        if self.args.model in ["svm", "rf", "xgb", "lgb", "logreg"]:
            return self.run_ml_on_embeddings(
                train_embeddings,
                train_y,
                val_embeddings,
                val_y,
                test_embeddings,
                test_y if has_test else None,
                artifact_source="pretrained_embeddings",
                artifact_info=artifact_info,
            )
        if self.args.model != "mlp":
            raise SystemExit("protbert supports model in {mlp,svm,rf,xgb,lgb,logreg}")
        dl_scaler = None
        if self.args.dl_scaler != "none":
            train_embeddings, val_embeddings, test_embeddings, dl_scaler = self._scale_tensor_split(
                train_embeddings,
                val_embeddings,
                test_embeddings,
                return_scaler=True,
            )
        train_ds = TensorDataset(train_embeddings, train_y)
        val_ds = TensorDataset(val_embeddings, val_y)
        self._configure_intrinsic_semantics(
            train_embeddings,
            train_y,
            train_ids=self._semantic_sequence_ids(train_seqs),
            val_ids=self._semantic_sequence_ids(val_seqs),
            test_ids=self._semantic_sequence_ids(test_seqs) if has_test else [],
        )
        model = self._prepare_semantic_model(
            model,
            sample_inputs=train_embeddings[: min(2, train_embeddings.shape[0])].to(self.device),
        )
        optimizer = optim.AdamW(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        criterion = self._multilabel_loss(train_y, auto_default="balanced_bce")
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        best_val_f1 = 0.0
        best_epoch = 0
        best_state = None
        no_improve_epochs = 0
        train_loader = self._semantic_train_loader(train_ds)
        val_loader = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False)
        self.print_semantic_similarity_status("protbert/mlp")
        for epoch in range(self.args.epochs):
            model.train()
            total_loss = 0.0
            for embeddings, targets, row_ids in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}"
            ):
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                logits, loss, _ = self.training_loss(
                    model, embeddings, targets, criterion, row_ids=row_ids
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            val_metrics = self._validation_metrics_for_checkpoint(model, val_loader)
            val_acc, val_sacc, val_f1, val_auc = (
                val_metrics["acc"],
                val_metrics["sacc"],
                val_metrics["f1"],
                val_metrics["auc"],
            )
            train_loss = total_loss / len(train_loader)
            val_loss = self.eval_loss(model, val_loader, criterion)
            print(
                f"[protbert/mlp] Epoch {epoch + 1}/{self.args.epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val SAcc: {val_sacc:.4f} | "
                f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
                + Evaluator.format_extra_metrics(val_metrics, prefix="Val")
                + self._val_aupr_suffix(model, val_loader)
            )
            (
                best_val_f1,
                best_epoch,
                best_state,
                checkpoint_improved,
            ) = self._update_validation_f1_checkpoint(
                model,
                val_f1=val_f1,
                val_metrics=val_metrics,
                epoch=epoch + 1,
                best_val_f1=best_val_f1,
                best_epoch=best_epoch,
                best_state=best_state,
            )
            if checkpoint_improved:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            scheduler.step()
            if (
                self.args.early_stop_patience > 0
                and no_improve_epochs >= self.args.early_stop_patience
            ):
                print(f"[protbert/mlp] Early stop at epoch {epoch + 1}")
                break
        print(f"[protbert/mlp] Finished. Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})")
        self._restore_validation_f1_checkpoint(model, best_state, best_epoch)
        test_ds = TensorDataset(test_embeddings, test_y) if has_test else None
        self._finalize_pretrained_embedding_nn(
            "protbert", model, val_ds, test_ds, criterion, artifact_info, dl_scaler
        )
        return best_val_f1

    def resolve_prott5_model_source(self):
        return _resolve_pretrained_model_source(
            self.project_root,
            getattr(self.args, "prott5_model", "Rostlab/prot_t5_xl_uniref50"),
            [
                os.path.join(self.project_root, "models", "prott5"),
                os.path.join(self.project_root, "models", "prot_t5_xl_uniref50"),
                os.path.join(self.project_root, "models", "Rostlab", "prot_t5_xl_uniref50"),
                os.path.expanduser("~/models/prott5"),
            ],
            "ProT5",
            "prott5_model",
        )

    def resolve_esmc_model_source(self):
        raw_model = getattr(self.args, "esmc_model", PRETRAINED_MODEL_DEFAULTS["esmc"])
        return _resolve_pretrained_model_source(
            self.project_root,
            raw_model,
            pretrained_model_local_candidates(self.project_root, "esmc", raw_model),
            "ESM-C",
            "esmc_model",
        )

    def resolve_esm3_model_source(self):
        return _resolve_pretrained_model_source(
            self.project_root,
            getattr(self.args, "esm3_model", ESM3_OFFICIAL_MODEL_ID),
            [
                os.path.join(self.project_root, "models", "esm3"),
                os.path.join(self.project_root, "models", "esm3-sm-open-v1"),
                os.path.join(self.project_root, "models", "EvolutionaryScale", "esm3-sm-open-v1"),
                os.path.expanduser("~/models/esm3"),
            ],
            "ESM3",
            "esm3_model",
        )

    def run_prott5(self):
        train_seqs = self.load_sequences(os.path.join(self.args.data_dir, "train", "seqs.fasta"))
        val_seqs = self.load_sequences(os.path.join(self.args.data_dir, "val", "seqs.fasta"))
        train_y = self.load_labels(os.path.join(self.args.data_dir, "train", "labels.csv"))
        val_y = self.load_labels(os.path.join(self.args.data_dir, "val", "labels.csv"))
        test_seq_path = os.path.join(self.args.data_dir, "test", "seqs.fasta")
        test_label_path = os.path.join(self.args.data_dir, "test", "labels.csv")
        has_test = (
            bool(getattr(self.args, "evaluate_test", True))
            and os.path.exists(test_seq_path)
            and os.path.exists(test_label_path)
        )
        test_seqs = self.load_sequences(test_seq_path) if has_test else []
        test_y = self.load_labels(test_label_path) if has_test else torch.empty(0)
        if len(train_seqs) != len(train_y):
            min_len = min(len(train_seqs), len(train_y))
            train_seqs = train_seqs[:min_len]
            train_y = train_y[:min_len]
        if len(val_seqs) != len(val_y):
            min_len = min(len(val_seqs), len(val_y))
            val_seqs = val_seqs[:min_len]
            val_y = val_y[:min_len]
        if has_test and len(test_seqs) != len(test_y):
            min_len = min(len(test_seqs), len(test_y))
            test_seqs = test_seqs[:min_len]
            test_y = test_y[:min_len]
        label_names = self._load_label_names(
            os.path.join(self.args.data_dir, "train", "labels.csv"), train_y.shape[1]
        )
        model_name = self.resolve_prott5_model_source()
        if os.path.exists(model_name):
            print(f"[prott5] using local model: {model_name}")
        else:
            print(f"[prott5] local model not found, using remote: {model_name}")
        if int(self.args.max_length) == PROTT5_MAX_RESIDUES + 1:
            self.args.prott5_max_length_adjustment = {
                "requested": int(self.args.max_length),
                "effective": PROTT5_MAX_RESIDUES,
                "reason": "T5 n_positions includes the EOS token",
            }
            self.args.max_length = PROTT5_MAX_RESIDUES
        tokenizer = ProT5Tokenizer(model_name=model_name, max_length=self.args.max_length)
        encoder = ProT5Encoder(model_name=model_name, device=self.device)
        encoder.model_identity["tokenizer_contract"] = {
            "residue_tokenization": tokenizer.tokenizer_contract["residue_tokenization"],
            "pad_token_id": tokenizer.tokenizer_contract.get("pad_token_id", PROTT5_PAD_TOKEN_ID),
            "eos_token_id": tokenizer.tokenizer_contract.get("eos_token_id", PROTT5_EOS_TOKEN_ID),
            "vocab_size": tokenizer.tokenizer_contract.get("vocab_size", PROTT5_VOCAB_SIZE),
            "max_residues": PROTT5_MAX_RESIDUES,
        }
        self.args._resolved_pretrained_model_source = model_name
        self.args._resolved_pretrained_model_identity = dict(encoder.model_identity)
        self.args._resolved_pretrained_runtime_environment = dict(encoder.runtime_environment)
        print(
            format_prott5_provenance(encoder.model_identity, encoder.runtime_environment),
            flush=True,
        )
        input_dim = encoder.model.config.hidden_size
        num_classes = train_y.shape[1]
        if self.args.sequence_model:
            return self.run_pretrained_residue(
                "prott5",
                tokenizer,
                encoder,
                model_name,
                train_seqs,
                train_y,
                val_seqs,
                val_y,
                test_seqs,
                test_y if has_test else None,
                label_names,
            )
        model = PredictionHeadClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=self.args.dropout,
            **prediction_head_kwargs(self.args),
        ).to(self.device)
        train_embeddings = self.get_or_extract_prott5_embeddings(
            tokenizer, encoder, train_seqs, "train"
        )
        val_embeddings = self.get_or_extract_prott5_embeddings(tokenizer, encoder, val_seqs, "val")
        test_embeddings = (
            self.get_or_extract_prott5_embeddings(tokenizer, encoder, test_seqs, "test")
            if has_test
            else None
        )
        artifact_info = self._pretrained_embedding_artifact_info(
            "prott5", model_name, input_dim, label_names
        )
        if self.args.model in ["svm", "rf", "xgb", "lgb", "logreg"]:
            return self.run_ml_on_embeddings(
                train_embeddings,
                train_y,
                val_embeddings,
                val_y,
                test_embeddings,
                test_y if has_test else None,
                artifact_source="pretrained_embeddings",
                artifact_info=artifact_info,
            )
        if self.args.model != "mlp":
            raise SystemExit("prott5 supports model in {mlp,svm,rf,xgb,lgb,logreg}")
        dl_scaler = None
        if self.args.dl_scaler != "none":
            train_embeddings, val_embeddings, test_embeddings, dl_scaler = self._scale_tensor_split(
                train_embeddings,
                val_embeddings,
                test_embeddings,
                return_scaler=True,
            )
        train_ds = TensorDataset(train_embeddings, train_y)
        val_ds = TensorDataset(val_embeddings, val_y)
        self._configure_intrinsic_semantics(
            train_embeddings,
            train_y,
            train_ids=self._semantic_sequence_ids(train_seqs),
            val_ids=self._semantic_sequence_ids(val_seqs),
            test_ids=self._semantic_sequence_ids(test_seqs) if has_test else [],
        )
        model = self._prepare_semantic_model(
            model,
            sample_inputs=train_embeddings[: min(2, train_embeddings.shape[0])].to(self.device),
        )
        optimizer = optim.AdamW(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        criterion = self._multilabel_loss(train_y, auto_default="balanced_bce")
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        best_val_f1 = 0.0
        best_epoch = 0
        best_state = None
        no_improve_epochs = 0
        train_loader = self._semantic_train_loader(train_ds)
        val_loader = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False)
        self.print_semantic_similarity_status("prott5/mlp")
        for epoch in range(self.args.epochs):
            model.train()
            total_loss = 0.0
            for embeddings, targets, row_ids in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}"
            ):
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                logits, loss, _ = self.training_loss(
                    model, embeddings, targets, criterion, row_ids=row_ids
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            val_metrics = self._validation_metrics_for_checkpoint(model, val_loader)
            val_acc, val_sacc, val_f1, val_auc = (
                val_metrics["acc"],
                val_metrics["sacc"],
                val_metrics["f1"],
                val_metrics["auc"],
            )
            train_loss = total_loss / len(train_loader)
            val_loss = self.eval_loss(model, val_loader, criterion)
            print(
                f"[prott5/mlp] Epoch {epoch + 1}/{self.args.epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val SAcc: {val_sacc:.4f} | "
                f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
                + Evaluator.format_extra_metrics(val_metrics, prefix="Val")
                + self._val_aupr_suffix(model, val_loader)
            )
            (
                best_val_f1,
                best_epoch,
                best_state,
                checkpoint_improved,
            ) = self._update_validation_f1_checkpoint(
                model,
                val_f1=val_f1,
                val_metrics=val_metrics,
                epoch=epoch + 1,
                best_val_f1=best_val_f1,
                best_epoch=best_epoch,
                best_state=best_state,
            )
            if checkpoint_improved:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            scheduler.step()
            if (
                self.args.early_stop_patience > 0
                and no_improve_epochs >= self.args.early_stop_patience
            ):
                print(f"[prott5/mlp] Early stop at epoch {epoch + 1}")
                break
        print(f"[prott5/mlp] Finished. Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})")
        self._restore_validation_f1_checkpoint(model, best_state, best_epoch)
        test_ds = TensorDataset(test_embeddings, test_y) if has_test else None
        self._finalize_pretrained_embedding_nn(
            "prott5", model, val_ds, test_ds, criterion, artifact_info, dl_scaler
        )
        return best_val_f1

    def run_esmc(self):
        train_seqs = self.load_sequences(os.path.join(self.args.data_dir, "train", "seqs.fasta"))
        val_seqs = self.load_sequences(os.path.join(self.args.data_dir, "val", "seqs.fasta"))
        train_y = self.load_labels(os.path.join(self.args.data_dir, "train", "labels.csv"))
        val_y = self.load_labels(os.path.join(self.args.data_dir, "val", "labels.csv"))
        test_seq_path = os.path.join(self.args.data_dir, "test", "seqs.fasta")
        test_label_path = os.path.join(self.args.data_dir, "test", "labels.csv")
        has_test = (
            bool(getattr(self.args, "evaluate_test", True))
            and os.path.exists(test_seq_path)
            and os.path.exists(test_label_path)
        )
        test_seqs = self.load_sequences(test_seq_path) if has_test else []
        test_y = self.load_labels(test_label_path) if has_test else torch.empty(0)
        if len(train_seqs) != len(train_y):
            min_len = min(len(train_seqs), len(train_y))
            train_seqs = train_seqs[:min_len]
            train_y = train_y[:min_len]
        if len(val_seqs) != len(val_y):
            min_len = min(len(val_seqs), len(val_y))
            val_seqs = val_seqs[:min_len]
            val_y = val_y[:min_len]
        if has_test and len(test_seqs) != len(test_y):
            min_len = min(len(test_seqs), len(test_y))
            test_seqs = test_seqs[:min_len]
            test_y = test_y[:min_len]
        label_names = self._load_label_names(
            os.path.join(self.args.data_dir, "train", "labels.csv"), train_y.shape[1]
        )
        if self.args.tokenizer == "esm3":
            tag = "esm3"
            model_name = self.resolve_esm3_model_source()
            tokenizer = Esm3Tokenizer(model_name=model_name, max_length=self.args.max_length)
            encoder = Esm3Encoder(model_name=model_name, device=self.device)
            self.args._resolved_pretrained_model_source = model_name
            self.args._resolved_pretrained_model_identity = dict(encoder.model_identity)
            self.args._resolved_pretrained_runtime_environment = dict(encoder.runtime_environment)
            print(
                format_esm3_provenance(encoder.model_identity, encoder.runtime_environment),
                flush=True,
            )
            embedding_getter = self.get_or_extract_esm3_embeddings
        else:
            tag = "esmc"
            model_name = self.resolve_esmc_model_source()
            tokenizer = EsmcTokenizer(model_name=model_name, max_length=self.args.max_length)
            encoder = EsmcEncoder(model_name=model_name, device=self.device)
            self.args._resolved_pretrained_model_source = model_name
            self.args._resolved_pretrained_model_identity = dict(encoder.model_identity)
            self.args._resolved_pretrained_runtime_environment = dict(encoder.runtime_environment)
            print(
                format_esmc_provenance(encoder.model_identity, encoder.runtime_environment),
                flush=True,
            )
            embedding_getter = self.get_or_extract_esmc_embeddings
        if os.path.exists(model_name):
            print(f"[{tag}] using local model: {model_name}")
        else:
            print(f"[{tag}] local model not found, using remote: {model_name}")
        if self.args.sequence_model:
            return self.run_pretrained_residue(
                tag,
                tokenizer,
                encoder,
                model_name,
                train_seqs,
                train_y,
                val_seqs,
                val_y,
                test_seqs,
                test_y if has_test else None,
                label_names,
            )
        train_embeddings = embedding_getter(tokenizer, encoder, train_seqs, "train")
        val_embeddings = embedding_getter(tokenizer, encoder, val_seqs, "val")
        test_embeddings = (
            embedding_getter(tokenizer, encoder, test_seqs, "test") if has_test else None
        )
        train_embeddings = train_embeddings.float()
        val_embeddings = val_embeddings.float()
        if test_embeddings is not None:
            test_embeddings = test_embeddings.float()
        if train_embeddings.ndim != 2 or train_embeddings.shape[0] == 0:
            raise SystemExit(f"{tag} embedding extraction failed")
        input_dim = train_embeddings.shape[1]
        artifact_info = self._pretrained_embedding_artifact_info(
            tag, model_name, input_dim, label_names
        )
        num_classes = train_y.shape[1]
        model = PredictionHeadClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=self.args.dropout,
            **prediction_head_kwargs(self.args),
        ).to(self.device)
        model = self._prepare_semantic_model(
            model,
            sample_inputs=train_embeddings[: min(2, train_embeddings.shape[0])].to(self.device),
        )
        optimizer = optim.AdamW(
            model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )
        if self.args.model == "mlp":
            criterion = self._multilabel_loss(train_y, auto_default="balanced_bce")
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        best_val_f1 = 0.0
        best_epoch = 0
        best_state = None
        no_improve_epochs = 0
        if self.args.model in ["svm", "rf", "xgb", "lgb", "logreg"]:
            return self.run_ml_on_embeddings(
                train_embeddings,
                train_y,
                val_embeddings,
                val_y,
                test_embeddings,
                test_y if has_test else None,
                artifact_source="pretrained_embeddings",
                artifact_info=artifact_info,
            )
        if self.args.model != "mlp":
            raise SystemExit(f"{tag} supports model in {{mlp,svm,rf,xgb,lgb}}")
        dl_scaler = None
        if self.args.dl_scaler != "none":
            train_embeddings, val_embeddings, test_embeddings, dl_scaler = self._scale_tensor_split(
                train_embeddings,
                val_embeddings,
                test_embeddings,
                return_scaler=True,
            )
        train_ds = TensorDataset(train_embeddings, train_y)
        val_ds = TensorDataset(val_embeddings, val_y)
        self._configure_intrinsic_semantics(
            train_embeddings,
            train_y,
            train_ids=self._semantic_sequence_ids(train_seqs),
            val_ids=self._semantic_sequence_ids(val_seqs),
            test_ids=self._semantic_sequence_ids(test_seqs) if has_test else [],
        )
        train_loader = self._semantic_train_loader(train_ds)
        val_loader = DataLoader(val_ds, batch_size=self.args.batch_size, shuffle=False)
        self.print_semantic_similarity_status(f"{tag}/mlp")
        for epoch in range(self.args.epochs):
            model.train()
            total_loss = 0.0
            for embeddings, targets, row_ids in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}"
            ):
                embeddings = embeddings.to(self.device).float()
                targets = targets.to(self.device)
                optimizer.zero_grad()
                logits, loss, _ = self.training_loss(
                    model, embeddings, targets, criterion, row_ids=row_ids
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            val_metrics = self._validation_metrics_for_checkpoint(model, val_loader)
            val_acc, val_sacc, val_f1, val_auc = (
                val_metrics["acc"],
                val_metrics["sacc"],
                val_metrics["f1"],
                val_metrics["auc"],
            )
            train_loss = total_loss / len(train_loader)
            val_loss = self.eval_loss(model, val_loader, criterion)
            print(
                f"[{tag}/mlp] Epoch {epoch + 1}/{self.args.epochs} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val SAcc: {val_sacc:.4f} | "
                f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
                + Evaluator.format_extra_metrics(val_metrics, prefix="Val")
                + self._val_aupr_suffix(model, val_loader)
            )
            (
                best_val_f1,
                best_epoch,
                best_state,
                checkpoint_improved,
            ) = self._update_validation_f1_checkpoint(
                model,
                val_f1=val_f1,
                val_metrics=val_metrics,
                epoch=epoch + 1,
                best_val_f1=best_val_f1,
                best_epoch=best_epoch,
                best_state=best_state,
            )
            if checkpoint_improved:
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            scheduler.step()
            if (
                self.args.early_stop_patience > 0
                and no_improve_epochs >= self.args.early_stop_patience
            ):
                print(f"[{tag}/mlp] Early stop at epoch {epoch + 1}")
                break
        print(f"[{tag}/mlp] Finished. Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})")
        self._restore_validation_f1_checkpoint(model, best_state, best_epoch)
        test_ds = TensorDataset(test_embeddings, test_y) if has_test else None
        self._finalize_pretrained_embedding_nn(
            tag, model, val_ds, test_ds, criterion, artifact_info, dl_scaler
        )
        return best_val_f1

    def run_esm3(self):
        original_tokenizer = self.args.tokenizer
        self.args.tokenizer = "esm3"
        try:
            return self.run_esmc()
        finally:
            self.args.tokenizer = original_tokenizer

    def run(self):
        normalize_workflow_args(self.args)
        resume_bundle = str(getattr(self.args, "resume_embedding_bundle", "") or "").strip()
        if resume_bundle:
            manifest, tensors = load_svm_resume_bundle(resume_bundle)
            if str(getattr(self.args, "classifier", "") or "").lower() != "svm":
                raise SystemExit("downstream SVM resume requires classifier=svm")
            print("[resume] Stage 1/2 reused: loading verified exported embeddings")
            return self.run_ml_on_embeddings(
                tensors["train_x"],
                tensors["train_y"],
                tensors["val_x"],
                tensors["val_y"],
                tensors["test_x"],
                tensors["test_y"],
                classifier="svm",
                artifact_source=str(
                    manifest.get("artifact_source") or "sequence_embedding_features"
                ),
                artifact_info=dict(manifest.get("artifact_info") or {}),
            )
        try:
            validate_index_route_semantics(self.args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        self._resolve_top_ngram_dataset_length()
        for line in format_effective_route(
            resolve_effective_route(vars(self.args), self.args.task_type)
        ):
            print(line)

        if self.args.encoder == "esm2":
            return self.run_esm()

        if self.args.encoder == "protbert":
            return self.run_protbert()

        if self.args.encoder == "prott5":
            return self.run_prott5()

        if self.args.encoder == "esmc":
            return self.run_esmc()

        if self.args.encoder == "esm3":
            return self.run_esmc()

        featurizer = self.build_featurizer()
        train_data, val_data, test_data = self.build_datasets(featurizer)
        self.args._artifact_source = "sequence_features"
        self.args._artifact_info = self._sequence_feature_artifact_info(
            train_data,
            self._sequence_profile_generation_artifact_info(),
        )

        if exports_sequence_embeddings(self.args):
            return self.run_sequence_export(featurizer, train_data, val_data, test_data)

        classical_runners = {
            "svm": self.ml_trainer.run_svm,
            "calibrated_svm": self.ml_trainer.run_calibrated_svm,
            "rf": self.ml_trainer.run_rf,
            "extratrees": self.ml_trainer.run_extratrees,
            "histgb": self.ml_trainer.run_histgb,
            "xgb": self.ml_trainer.run_xgb,
            "lgb": self.ml_trainer.run_lgb,
            "logreg": self.ml_trainer.run_logreg,
        }
        if self.args.model in classical_runners:
            allowed = [encoder for encoder in CLASSIC_ENCODER_OPTIONS["kmer"] if encoder != "index"]
            if self.args.encoder not in allowed:
                raise SystemExit(f"{self.args.model} requires encoder in {{{','.join(allowed)}}}")
            return classical_runners[self.args.model](train_data, val_data, test_data)

        return self.run_nn(featurizer, train_data, val_data, test_data)


Pipeline = SequencePipeline


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Single-sequence peptide training pipeline")
    parser.add_argument("--task_type", type=str, default="pep_function")
    parser.add_argument(
        "--tokenizer",
        choices=[
            "kmer",
            "dr",
            "top-n-gram",
            "dt",
            "bpe",
            "wordpiece",
            "unigram",
            "esm2",
            "protbert",
            "probert",
            "prott5",
            "esmc",
            "esm3",
        ],
        default="kmer",
    )
    parser.add_argument(
        "--encoder",
        type=encoder_argument,
        choices=TRAINING_ENCODER_OPTIONS,
        default=None,
        help="representation encoder; pretrained routes use the same explicit id as --tokenizer",
    )
    parser.add_argument(
        "--model",
        choices=[
            "cnn",
            "rnn",
            "gru",
            "lstm",
            "transformer",
            "gcn",
            "gat",
            "mlp",
            "svm",
            "calibrated_svm",
            "rf",
            "extratrees",
            "histgb",
            "xgb",
            "lgb",
            "logreg",
        ],
        default="cnn",
    )
    parser.add_argument(
        "--sequence_model",
        choices=["cnn", "rnn", "gru", "lstm", "transformer", "gcn", "gat"],
        default="",
    )
    parser.add_argument("--training_mode", choices=["end_to_end", "export_embeddings"], default="")
    parser.add_argument(
        "--classifier",
        choices=[
            "mlp",
            "svm",
            "calibrated_svm",
            "rf",
            "extratrees",
            "histgb",
            "xgb",
            "lgb",
            "logreg",
        ],
        default="",
    )
    parser.add_argument("--head_hidden_dims", type=parse_prediction_head_hidden_dims, default=None)
    parser.add_argument("--head_activation", choices=["relu", "gelu", "silu"], default=None)
    parser.add_argument("--head_dropout", type=float, default=None)
    parser.add_argument("--residue_adapter_dim", type=int, default=RESIDUE_ADAPTER_DIM_DEFAULT)
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="window length; tokenizer=kmer accepts integers 1-3",
    )
    parser.add_argument(
        "--dr_max_distance",
        type=parse_dr_max_distance,
        default=DR_DEFAULT_MAX_DISTANCE,
    )
    parser.add_argument(
        "--dr_max_raw_length",
        type=parse_dr_max_raw_length,
        default=DR_DEFAULT_MAX_RAW_LENGTH,
    )
    parser.add_argument(
        "--frequency_max_raw_length",
        type=parse_frequency_max_raw_length,
        default=FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
        help="maximum raw sequence length for BoW/TF-IDF (hard max 100000)",
    )
    parser.add_argument(
        "--dr_dense_memory_limit_gib",
        type=parse_dr_dense_memory_limit_gib,
        default=DR_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    )
    parser.add_argument("--top_ngram_n", type=int, default=2)
    parser.add_argument(
        "--topngram_dense_memory_limit_gib",
        type=parse_top_ngram_dense_memory_limit_gib,
        default=TOP_N_GRAM_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    )
    parser.add_argument(
        "--dt_max_distance",
        type=parse_dt_max_distance,
        default=DT_DEFAULT_MAX_DISTANCE,
    )
    parser.add_argument(
        "--dt_dense_memory_limit_gib",
        type=parse_dt_dense_memory_limit_gib,
        default=DT_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=SEQUENCE_NEURAL_DEFAULTS["max_length"],
        help="maximum encoded positions; for tokenizer=kmer this is the K-mer token count",
    )
    parser.add_argument("--ac_lag", type=int, default=10)
    parser.add_argument("--cc_lag", type=int, default=10)
    parser.add_argument("--acc_lag", type=int, default=10)
    parser.add_argument(
        "--covariance_max_raw_length",
        type=parse_covariance_max_raw_length,
        default=COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
        help="maximum complete raw sequence length for AC/CC/ACC (hard max 100000)",
    )
    parser.add_argument(
        "--pdt_lag",
        type=parse_pdt_lag,
        default=PDT_DEFAULT_LAG,
    )
    parser.add_argument(
        "--pdt_max_raw_length",
        type=parse_pdt_max_raw_length,
        default=PDT_DEFAULT_MAX_RAW_LENGTH,
        help="maximum complete raw sequence length for PDT (hard max 100000)",
    )
    parser.add_argument("--ac_pssm_lag", type=parse_pssm_covariance_lag, default=10)
    parser.add_argument("--cc_pssm_lag", type=parse_pssm_covariance_lag, default=10)
    parser.add_argument("--acc_pssm_lag", type=parse_pssm_covariance_lag, default=10)
    parser.add_argument("--ac_pssm_blosum62_hybrid_lag", type=parse_pssm_covariance_lag, default=10)
    parser.add_argument("--cc_pssm_blosum62_hybrid_lag", type=parse_pssm_covariance_lag, default=10)
    parser.add_argument(
        "--acc_pssm_blosum62_hybrid_lag", type=parse_pssm_covariance_lag, default=10
    )

    parser.add_argument("--batch_size", type=int, default=SEQUENCE_NEURAL_DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=SEQUENCE_NEURAL_DEFAULTS["lr"])
    parser.add_argument("--epochs", type=int, default=SEQUENCE_NEURAL_DEFAULTS["epochs"])
    parser.add_argument("--dropout", type=float, default=SEQUENCE_NEURAL_DEFAULTS["dropout"])
    parser.add_argument(
        "--weight_decay", type=float, default=SEQUENCE_NEURAL_DEFAULTS["weight_decay"]
    )
    parser.add_argument(
        "--grad_clip_norm", type=float, default=SEQUENCE_NEURAL_DEFAULTS["grad_clip_norm"]
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=SEQUENCE_NEURAL_DEFAULTS["early_stop_patience"]
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=SEQUENCE_NEURAL_DEFAULTS["early_stop_min_delta"],
    )
    parser.add_argument(
        "--checkpoint_objective",
        choices=list(REGISTRY),
        default=None,
    )
    parser.add_argument(
        "--multilabel_loss",
        choices=["auto", "bce", "balanced_bce", "asymmetric_focal"],
        default="auto",
        help="Classification loss: auto preserves BCE for sequence models and "
        "WBCE for pretrained pooled MLPs; explicit choices also apply to pooled MLPs.",
    )
    parser.add_argument("--multilabel_pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--asl_gamma_pos", type=float, default=1.0)
    parser.add_argument("--asl_gamma_neg", type=float, default=4.0)
    parser.add_argument("--asl_clip", type=float, default=0.05)
    parser.add_argument(
        "--enable_semantic_similarity", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--enable_interaction_semantic_similarity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--semantic_similarity_weight", type=float, default=0.1)
    parser.add_argument("--semantic_zero_weight", type=float, default=0.25)
    parser.add_argument(
        "--semantic_pair_mode",
        type=str,
        default="role_fused",
        choices=["fused", "role", "role_fused"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--semantic_loss_type",
        choices=["legacy", "jaccard_supcon"],
        default=argparse.SUPPRESS,
        help="Semantic auxiliary objective (default: legacy); contrastive pretraining loss is unchanged",
    )
    parser.add_argument("--semantic_projection_dim", type=int, default=64)
    parser.add_argument("--semantic_contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--semantic_positive_partners", type=int, default=1)
    parser.add_argument("--semantic_zero_partners", type=int, default=8)
    parser.add_argument(
        "--semantic_loss_space",
        type=str,
        default="projection",
        choices=["projection", "shared"],
    )
    parser.add_argument("--semantic_residual_weight", type=float, default=0.0)
    parser.add_argument(
        "--semantic_partner_scheme",
        type=str,
        default="fixed",
        choices=["fixed", "fixed_placebo", "dense_hard", "placebo"],
    )
    parser.add_argument("--semantic_positive_cap", type=int, default=16)
    parser.add_argument("--semantic_negative_cap", type=int, default=8)
    parser.add_argument(
        "--semantic_hard_negative_source",
        type=str,
        default="input_embedding",
        choices=["input_embedding"],
    )
    parser.add_argument(
        "--tppredsc_stage1_epochs",
        type=int,
        default=0,
        help="Contrastive pretraining epochs (0 disables the two-stage mode)",
    )
    parser.add_argument("--tppredsc_stage1_lr", type=float, default=0.01)
    parser.add_argument("--tppredsc_stage1_momentum", type=float, default=0.9)
    parser.add_argument("--tppredsc_stage2_epochs", type=int, default=10)
    parser.add_argument("--tppredsc_stage2_lr", type=float, default=0.0001)
    parser.add_argument("--semantic_jaccard_theta", type=float, default=0.1)
    parser.add_argument(
        "--semantic_contrastive_mode",
        type=str,
        default="multilabel",
        choices=["multilabel", "simclr"],
    )
    parser.add_argument("--semantic_augmented_views_dir", type=str, default="")
    parser.add_argument("--embedding_vector_size", type=int, default=128)
    parser.add_argument("--embedding_window", type=int, default=5)
    parser.add_argument("--embedding_min_count", type=int, default=1)
    parser.add_argument("--embedding_epochs", type=int, default=20)
    parser.add_argument("--embedding_workers", type=int, default=1)
    parser.add_argument(
        "--fasttext_bucket",
        type=int,
        default=FASTTEXT_DEFAULT_BUCKET,
    )
    parser.add_argument(
        "--static_embedding_memory_limit_gib",
        type=float,
        default=STATIC_EMBEDDING_DEFAULT_MEMORY_LIMIT_GIB,
    )
    parser.add_argument("--glove_x_max", type=float, default=100.0)
    parser.add_argument("--glove_alpha", type=float, default=0.75)
    parser.add_argument("--glove_learning_rate", type=float, default=0.05)
    parser.add_argument("--glove_batch_size", type=int, default=4096)

    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--bpe_path", type=str, default="")
    parser.add_argument("--bpe_vocab_size", type=int, default=1000)
    parser.add_argument("--unigram_path", type=str, default="")
    parser.add_argument(
        "--unigram_vocab_size",
        type=validate_unigram_vocab_size,
        default=1000,
    )
    parser.add_argument(
        "--unigram_sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--unigram_nbest_size",
        type=int,
        default=UNIGRAM_DEFAULT_NBEST_SIZE,
    )
    parser.add_argument(
        "--unigram_alpha",
        type=float,
        default=UNIGRAM_DEFAULT_ALPHA,
    )
    parser.add_argument(
        "--unigram_max_raw_length",
        type=validate_unigram_max_raw_length,
        default=UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    )
    parser.add_argument("--wordpiece_path", type=str, default="")
    parser.add_argument(
        "--wordpiece_vocab_size",
        type=validate_wordpiece_vocab_size,
        default=1000,
    )
    parser.add_argument(
        "--wordpiece_max_raw_length",
        type=validate_wordpiece_max_raw_length,
        default=WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
    )
    parser.add_argument("--word2vec_path", type=str, default="")
    parser.add_argument("--glove_path", type=str, default="")
    parser.add_argument("--fasttext_path", type=str, default="")
    parser.add_argument("--profile_cache_dir", type=str, default="")
    parser.add_argument("--pssm_index_path", type=str, default="")
    parser.add_argument("--ac_pssm_blosum62_hybrid_index_path", type=str, default="")
    parser.add_argument("--cc_pssm_blosum62_hybrid_index_path", type=str, default="")
    parser.add_argument("--acc_pssm_blosum62_hybrid_index_path", type=str, default="")
    parser.add_argument("--psfm_index_path", type=str, default="")
    parser.add_argument("--frequency_profile_index_path", type=str, default="")
    parser.add_argument(
        "--profile_iterations", type=int, default=SEQUENCE_PROFILE_DEFAULTS["profile_iterations"]
    )
    parser.add_argument(
        "--profile_evalue", type=float, default=SEQUENCE_PROFILE_DEFAULTS["profile_evalue"]
    )
    parser.add_argument(
        "--profile_inclusion_evalue",
        type=float,
        default=SEQUENCE_PROFILE_DEFAULTS["profile_inclusion_evalue"],
    )
    parser.add_argument(
        "--profile_threads", type=int, default=SEQUENCE_PROFILE_DEFAULTS["profile_threads"]
    )
    parser.add_argument(
        "--profile_on_missing",
        choices=[
            "error",
            "blosum62_hybrid",
            TOP_N_GRAM_FALLBACK_POLICY,
        ],
        default=None,
    )
    parser.add_argument("--esm_feature_dir", type=str, default="")
    parser.add_argument("--esm2_model", type=str, default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--protbert_feature_dir", type=str, default="")
    parser.add_argument("--protbert_model", type=str, default="Rostlab/prot_bert")
    parser.add_argument("--probert_feature_dir", type=str, default="")
    parser.add_argument("--probert_model", type=str, default="")
    parser.add_argument("--prott5_feature_dir", type=str, default="")
    parser.add_argument("--prott5_model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--esmc_feature_dir", type=str, default="")
    parser.add_argument("--esmc_model", type=str, default=PRETRAINED_MODEL_DEFAULTS["esmc"])
    parser.add_argument("--esm3_feature_dir", type=str, default="")
    parser.add_argument("--esm3_model", type=str, default="EvolutionaryScale/esm3-sm-open-v1")

    parser.add_argument("--reduction", choices=["none", "lsa", "lda", "plsa"], default="none")
    parser.add_argument("--lsa_dim", type=int, default=128)
    parser.add_argument(
        "--lsa_dense_memory_limit_gib",
        type=float,
        default=LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    )
    parser.add_argument("--lda_topics", type=int, default=50)
    parser.add_argument(
        "--lda_dense_memory_limit_gib",
        type=float,
        default=LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    )
    parser.add_argument("--plsa_topics", type=int, default=50)
    parser.add_argument(
        "--plsa_memory_limit_gib",
        type=float,
        default=PLSA_DEFAULT_MEMORY_LIMIT_GIB,
    )
    parser.add_argument("--ml_scaler", choices=["none", "standard", "minmax"], default="standard")
    parser.add_argument(
        "--dl_scaler", choices=["none", "standard"], default=SEQUENCE_NEURAL_DEFAULTS["dl_scaler"]
    )
    parser.add_argument("--ml_reduction", choices=["none", "pca", "kpca", "tsvd"], default="none")
    parser.add_argument("--ml_reduction_dim", type=int, default=128)
    parser.add_argument(
        "--tsvd_dense_memory_limit_gib",
        type=float,
        default=TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    )
    parser.add_argument(
        "--pca_dense_memory_limit_gib", type=float, default=PCA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
    )
    parser.add_argument(
        "--kpca_memory_limit_gib", type=float, default=KPCA_DEFAULT_MEMORY_LIMIT_GIB
    )
    parser.add_argument("--kpca_kernel", choices=["rbf", "linear", "poly", "cosine"], default="rbf")
    parser.add_argument("--kpca_gamma", type=float, default=None)
    parser.add_argument("--ml_auto_tune", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ml_tuning_budget", choices=["quick", "standard", "thorough"], default="")
    parser.add_argument(
        "--ml_tuning_objective",
        choices=list(REGISTRY),
        default=None,
    )
    parser.add_argument("--ml_tuning_mode", choices=["holdout", "cv"], default="cv")
    parser.add_argument("--ml_tuning_max_candidates", type=int, default=18)
    parser.add_argument("--ml_cv_folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluate_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="allow loading and evaluating the optional test split",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable deterministic PyTorch algorithms in addition to seeded RNGs",
    )

    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--svm_class_weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--svm_max_iter", type=int, default=2000)
    parser.add_argument("--svm_tol", type=float, default=1e-3)
    parser.add_argument("--svm_kernel", choices=["linear", "rbf"], default="linear")
    parser.add_argument("--svm_gamma", type=str, default="scale")
    parser.add_argument("--svm_calibration", choices=SVM_CALIBRATION_OPTIONS, default="")
    parser.add_argument("--svm_calibration_cv_folds", type=int, default=SVM_CALIBRATION_CV_FOLDS)
    parser.add_argument(
        "--svm_convergence_policy",
        choices=["strict", "bounded_retry"],
        default="strict",
    )
    parser.add_argument("--logreg_c", type=float, default=1.0)
    parser.add_argument("--logreg_class_weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--logreg_max_iter", type=int, default=2000)
    parser.add_argument(
        "--logreg_convergence_policy",
        choices=["strict", "bounded_retry"],
        default="strict",
    )

    parser.add_argument(
        "--rf_estimators", type=int, default=CLASSIFIER_PARAMETER_DEFAULTS["rf"]["rf_estimators"]
    )
    parser.add_argument(
        "--rf_max_depth", type=int, default=CLASSIFIER_PARAMETER_DEFAULTS["rf"]["rf_max_depth"]
    )
    parser.add_argument(
        "--rf_min_samples_leaf",
        type=int,
        default=CLASSIFIER_PARAMETER_DEFAULTS["rf"]["rf_min_samples_leaf"],
    )

    parser.add_argument("--et_estimators", type=int, default=300)
    parser.add_argument("--et_max_depth", type=int, default=0)
    parser.add_argument("--et_min_samples_leaf", type=int, default=1)

    parser.add_argument("--histgb_max_iter", type=int, default=300)
    parser.add_argument("--histgb_max_leaf_nodes", type=int, default=31)
    parser.add_argument("--histgb_lr", type=float, default=0.1)
    parser.add_argument("--histgb_l2_regularization", type=float, default=0.0)

    parser.add_argument("--xgb_estimators", type=int, default=300)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_lr", type=float, default=0.1)
    parser.add_argument("--xgb_subsample", type=float, default=1.0)
    parser.add_argument("--xgb_colsample", type=float, default=1.0)
    parser.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0)

    parser.add_argument("--lgb_estimators", type=int, default=300)
    parser.add_argument("--lgb_max_depth", type=int, default=6)
    parser.add_argument("--lgb_lr", type=float, default=0.1)
    parser.add_argument("--lgb_subsample", type=float, default=1.0)
    parser.add_argument("--lgb_colsample", type=float, default=1.0)
    parser.add_argument("--lgb_reg_alpha", type=float, default=0.0)
    parser.add_argument("--lgb_reg_lambda", type=float, default=0.0)

    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--advanced_wizard", action="store_true")
    parser.add_argument("--report_aupr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--artifact_dir", type=str, default="")
    parser.add_argument("--resume_embedding_bundle", type=str, default="")
    add_selection_arguments(parser, checkpoint=False)
    return register_legacy_cli_aliases(parser)


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    args.task_type = normalize_task_type(args.task_type)
    if args.list_objectives:
        print(json.dumps(objective_capabilities(), indent=2))
        return
    set_global_seed(args.seed, deterministic=args.deterministic)
    normalized_task = normalize_task_type(getattr(args, "task_type", "pep_function"))
    if normalize_task_type(normalized_task) in [
        "pep_hlaii_interaction",
        "peptide_hlaii_interaction",
    ]:
        normalized_task = "pep_hlaii"
    if normalize_task_type(normalized_task) in ["pep_hlai_interaction", "peptide_hlai_interaction"]:
        normalized_task = "pep_hlai"
    if normalize_task_type(normalized_task) in PAIR_TASK_TYPES:
        raise SystemExit(
            f"{normalized_task} is a pair-interaction task. "
            "Use scripts/train_pair_pipeline.py or scripts/train_pipeline.py."
        )
    no_arguments = len(argv) == 0 if argv is not None else len(sys.argv) == 1
    if args.interactive or no_arguments:
        args = InteractiveConfigWizard().run(args)
    if args.profile_on_missing is None:
        tokenizer = str(args.tokenizer).strip().lower()
        encoder = str(args.encoder).strip().lower()
        if tokenizer in {"top-n-gram", "dt"}:
            args.profile_on_missing = TOP_N_GRAM_FALLBACK_POLICY
        elif encoder in {
            "pssm_blosum62_hybrid",
            "psfm_blosum62_hybrid",
            *HYBRID_PSSM_COVARIANCE_ENCODERS,
        }:
            args.profile_on_missing = "blosum62_hybrid"
        else:
            args.profile_on_missing = SEQUENCE_PROFILE_DEFAULTS["profile_on_missing"]
    resolve_selection(args)
    pipeline = SequencePipeline(args)
    pipeline.run()



from pepplatform.stream_compat import register_legacy_cli_aliases, install_legacy_exports as install_legacy_stream_exports

install_legacy_stream_exports(globals())


if __name__ == "__main__":
    run_with_failure_reporting(main)
