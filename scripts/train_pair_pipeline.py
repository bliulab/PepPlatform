# Resolve local package imports before accepting historical CLI entry points.
import sys as _stream_sys
from pathlib import Path as _StreamPath
_stream_root = str(_StreamPath(__file__).resolve().parents[1])
if _stream_root not in _stream_sys.path:
    _stream_sys.path.insert(0, _stream_root)

from typing import Any
import os
import sys
import csv
import json
import argparse
import gc
import hashlib
import random
import re
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm as _tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


def tqdm(*args, **kwargs):
    """Keep terminal progress bars interactive without emitting carriage-return Web logs."""
    kwargs.setdefault("disable", not sys.stderr.isatty())
    return _tqdm(*args, **kwargs)


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from pepplatform.task_compat import (
    normalize_task_type,
    legacy_keyword,
    install_legacy_exports,
    resolve_hlai_root,
    add_legacy_semantic_argument,
)

from pepplatform.tokenizers import (
    EsmTokenizer,
    ProtBertTokenizer,
    ProT5Tokenizer,
    EsmcTokenizer,
    Esm3Tokenizer,
    BPETokenizerTrainer,
    WordPieceTokenizerTrainer,
    UnigramTokenizerTrainer,
)
from pepplatform.encoders import EsmEncoder, ProtBertEncoder, ProT5Encoder, EsmcEncoder, Esm3Encoder
from pepplatform.analysis import build_score_analysis, save_analysis
from pepplatform.models import (
    AlignedGatedClassifier,
    GAT_PREDICTOR_CONTRACT_VERSION,
    ModelFactory,
    PairSemanticResidualAdapter,
    PretrainedResiduePredictor,
    PredictionHeadClassifier,
    ResidueModelInput,
    SemanticResidualAdapter,
    build_prediction_head,
    sequence_predictor_contract,
)
from pepplatform.evaluators import EVALUATION_PROTOCOL_VERSION, Evaluator, OperatingPoint
from pepplatform.pr_metrics import binary_aupr
from pepplatform.objectives import (
    REGISTRY,
    add_selection_arguments,
    resolve_selection,
    selection_fields,
    objective_capabilities,
)
from pepplatform.ml_trainers import ClassicalMLTrainer
from pepplatform.cli_wizard import InteractiveConfigWizard
from pepplatform.route_registry import (
    ROUTE_SCHEMA_VERSION,
    format_effective_route,
    resolve_effective_route,
)
from pepplatform.run_progress import emit_run_progress
from pepplatform.hlai_input import hla_allele, require_hlai_partner_sequence
from pepplatform.training_failures import run_with_failure_reporting
from pepplatform.svm_resume import load_svm_resume_bundle, save_svm_resume_bundle
from pepplatform.featurizers import FeaturizerFactory
from pepplatform.pssm_covariance import (
    HYBRID_PSSM_COVARIANCE_ENCODERS,
    PSSM_COVARIANCE_ENCODER_CONTRACT_VERSION,
    PSSM_COVARIANCE_ENCODERS,
    parse_pssm_covariance_lag,
    pssm_covariance_lag_key,
    pssm_covariance_profile_feature_type,
)
from pepplatform.workflow import (
    CLASSIC_ENCODER_OPTIONS,
    CLASSIC_TOKENIZERS,
    CLASSICAL_CLASSIFIERS,
    PAIR_CLASSIFIER_PARAMETER_DEFAULTS,
    SVM_CALIBRATION_CV_FOLDS,
    SVM_CALIBRATION_OPTIONS,
    PAIR_NEURAL_DEFAULTS,
    PAIR_PROFILE_DEFAULTS,
    PAIR_TRAINING_ENCODER_OPTIONS,
    PRETRAINED_MODEL_DEFAULTS,
    PRETRAINED_TOKENIZERS,
    RESIDUE_ADAPTER_DIM_DEFAULT,
    RESIDUE_ADAPTER_DIM_MAX,
    RESIDUE_ADAPTER_DIM_MIN,
    RESIDUE_ADAPTER_DIM_MULTIPLE,
    SEQUENCE_ENCODERS,
    SEQUENCE_MODELS,
    effective_pair_vector_encoder,
    default_model_selection_objective,
    ensure_encoder_available,
    exports_interaction_embeddings,
    exports_sequence_embeddings,
    normalize_prediction_head_config,
    normalize_pair_workflow_args,
    normalize_encoder_name,
    normalize_tokenizer_name,
    parse_prediction_head_hidden_dims,
    parse_stream_alignment_dim,
    prediction_head_artifact_config,
    pretrained_model_local_candidates,
    resolve_pretrained_model_source as resolve_workflow_pretrained_model_source,
    supports_classic_pair_token_inference,
    supports_classic_pair_vector_inference,
    validate_index_route_semantics,
    supports_pair_token_inference,
    validate_linear_prediction_head_semantics,
    validate_training_mode_classifier,
)
from pepplatform.cache_provenance import (
    CACHE_PROVENANCE_VERSION,
    digest_text_items,
    model_source_fingerprint,
    sha256_file,
)
from pepplatform.esm2 import (
    ESM2_OFFICIAL_MODEL_ID,
    ESM2_POOLING,
    ESM2_REPRESENTATION_CONTRACT_VERSION,
    format_esm2_provenance,
    is_complete_esm2_snapshot,
    resolve_esm2_source,
)
from pepplatform.protbert import (
    PROTBERT_OFFICIAL_MODEL_ID,
    PROTBERT_POOLING,
    PROTBERT_REPRESENTATION_CONTRACT_VERSION,
    format_protbert_provenance,
)
from pepplatform.prott5 import (
    PROTT5_D_MODEL,
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
from pepplatform.lda import LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
from pepplatform.tsvd import TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB
from pepplatform.pca import PCA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
from pepplatform.kpca import KPCA_DEFAULT_MEMORY_LIMIT_GIB
from pepplatform.plsa import PLSA_DEFAULT_MEMORY_LIMIT_GIB
from pepplatform.pair_bow import (
    PairBoWContractError,
    build_pair_bow_joint_contract,
    pair_bow_contract_digest,
)
from pepplatform.pair_fusion import (
    AUDITED_PAIR_FUSION_MODES,
    AlignedGatedFusion,
    PairFusionError,
    aligned_gated_contract,
    fuse_pair_embeddings as fuse_audited_pair_embeddings,
    normalize_pair_fusion_mode,
    pair_fusion_contract,
)
from pepplatform.covariance import (
    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
    COVARIANCE_ENCODERS,
    parse_covariance_max_raw_length,
    validate_covariance_max_raw_length,
)
from pepplatform.pdt import (
    PDT_DEFAULT_LAG,
    PDT_DEFAULT_MAX_RAW_LENGTH,
    PDT_ENCODERS,
    parse_pdt_lag,
    parse_pdt_max_raw_length,
    validate_pdt_max_raw_length,
)
from pepplatform.bpe import bundle_bpe_artifact
from pepplatform.dr import (
    DR_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    DR_DEFAULT_MAX_DISTANCE,
    DR_DEFAULT_MAX_RAW_LENGTH,
    DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
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
    validate_dr_unknown_residue_policy,
)
from pepplatform.dt import (
    DT_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    DT_DEFAULT_MAX_DISTANCE,
    DT_MAX_DISTANCE,
    DT_MAX_LENGTH_UNIT,
    DTContractError,
    dt_contract_metadata,
    ensure_dt_dense_memory_budget,
    estimate_dt_dense_bytes,
    parse_dt_dense_memory_limit_gib,
    parse_dt_max_distance,
    validate_dt_max_distance,
)
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
    sequence_sha256,
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
    ESM3_DEFAULT_SDPA_BACKEND,
    ESM3_OFFICIAL_MODEL_ID,
    ESM3_POOLING,
    ESM3_REPRESENTATION_CONTRACT_VERSION,
    format_esm3_provenance,
    normalize_esm3_sequence,
    truncate_esm3_sequence,
    validate_esm3_sdpa_backend,
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
from pepplatform.kmer import KMER_MAX_K, KMER_MAX_LENGTH_UNIT
from pepplatform.stream_spec import add_stream_cli_arguments, apply_pair_streams, namespace_for_stream
from pepplatform.pair_terminology import public_pair_strategy
from pepplatform.pair_strategy import (
    PairStrategyContractError,
    build_two_stream_strategy_contract,
    build_single_stream_strategy_contract,
    join_single_stream_batches,
    join_single_stream_rows,
)
from pepplatform.semantic import (
    InteractionProfileRelationSource,
    RecordingRandomSampler,
    SemanticLossOutput,
    SemanticPartnerTable,
    SemanticTrainingProfile,
    auxiliary_eval_forward,
    build_interaction_dense_hard_partner_table,
    build_partner_table,
    normalize_semantic_aliases,
    placebo_shuffle_table,
    prepare_semantic_projectors,
    save_semantic_training_profile,
    semantic_contrastive_loss,
    semantic_enabled as semantic_is_enabled,
    semantic_representation_contract,
)


DL_SEQUENCE_MODELS = SEQUENCE_MODELS
DL_SEQUENCE_ENCODERS = SEQUENCE_ENCODERS
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
PROFILE_INDEX_KEYS = {
    FEATURE_AC_PSSM_BLOSUM62_HYBRID: ("ac_pssm_blosum62_hybrid_index_path"),
    FEATURE_CC_PSSM_BLOSUM62_HYBRID: ("cc_pssm_blosum62_hybrid_index_path"),
    FEATURE_ACC_PSSM_BLOSUM62_HYBRID: ("acc_pssm_blosum62_hybrid_index_path"),
}
CROSS_ATTN_MODEL = "cross_attn+mlp"
BI_CROSS_ATTN_MODEL = "bi_cross_attn+mlp"
CROSS_ATTENTION_PREDICTOR_CONTRACT_VERSION = 1
CROSS_ATTENTION_IMPLEMENTATION = "pepplatform_pair_token_cross_attention"
BI_CROSS_ATTENTION_IMPLEMENTATION = "pepplatform_pair_token_bidirectional_cross_attention"
GCN_MODEL = "gcn"
GAT_MODEL = "gat"
HGT_MODEL = "hgt"
SAGE_MODEL = "sage"
GT_MODEL = "graph_transformer"
GRAPH_MODELS = [GCN_MODEL, GAT_MODEL, HGT_MODEL, SAGE_MODEL, GT_MODEL]
HGT_PREDICTOR_CONTRACT_VERSION = 1
HGT_GRAPH_SCHEMA = "typed_pair_hgt_v1"
HGT_NODE_PEPTIDE = 0
HGT_NODE_PARTNER = 1
HGT_REL_PEPTIDE_SEMANTIC = 0
HGT_REL_PARTNER_SEMANTIC = 1
HGT_REL_PEPTIDE_SELF = 2
HGT_REL_PARTNER_SELF = 3
GAT_EDGE_CHUNK_SIZE = 100_000
SAGE_PREDICTOR_CONTRACT_VERSION = 1
GRAPH_TRANSFORMER_PREDICTOR_CONTRACT_VERSION = 1
SUPPORTED_PAIR_SEMANTIC_MODELS = set(
    DL_SEQUENCE_MODELS
    + ["mlp", "cross_attn", CROSS_ATTN_MODEL, "bi_cross_attn", BI_CROSS_ATTN_MODEL]
)
PAIR_VECTOR_ARTIFACT_SOURCE = "pair_vector_features"
PAIR_EMBEDDING_ARTIFACT_SOURCE = "pair_embedding_features"
PAIR_INFERENCE_TASKS = {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}
HLAI_ORIGINAL_SPLIT_DIR = "data_bigmhc_el_100k_original_split"
PAIR_CLASSIFICATION_LOSS_SCHEMA_VERSION = 1
PAIR_CLASSIFICATION_LOSS_DEFAULT = "wbce"
PAIR_CLASSIFICATION_POS_WEIGHT_MAX_DEFAULT = 50.0


def encoder_argument(value):
    try:
        ensure_encoder_available(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return normalize_encoder_name(value)


def _parse_pair_pos_weight_max(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("pos_weight_max must be a finite number >= 1") from exc
    if not np.isfinite(parsed) or parsed < 1.0:
        raise argparse.ArgumentTypeError("pos_weight_max must be a finite number >= 1")
    return parsed


def _parse_feature_extraction_batch_size(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "feature extraction batch size must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("feature extraction batch size must be a positive integer")
    return parsed


def prediction_head_kwargs(args):
    return {
        "head_hidden_dims": getattr(args, "head_hidden_dims", None),
        "head_activation": getattr(args, "head_activation", None),
        "head_dropout": getattr(args, "head_dropout", None),
    }


def validate_pair_attention_hyperparameters(hidden_dim, n_heads, dropout):
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError("interaction_hidden_dim must be a positive integer")
    if isinstance(n_heads, bool) or not isinstance(n_heads, int) or n_heads <= 0:
        raise ValueError("interaction_heads must be a positive integer")
    if hidden_dim % n_heads != 0:
        raise ValueError("interaction_hidden_dim must be divisible by interaction_heads")
    if not np.isfinite(dropout) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    return int(hidden_dim), int(n_heads), float(dropout)


def pair_cross_attention_predictor_contract(model):
    if (
        getattr(model, "interaction_model_id", "") != "cross_attn"
        or getattr(model, "cross_attention_predictor_contract_version", 0)
        != CROSS_ATTENTION_PREDICTOR_CONTRACT_VERSION
    ):
        return None
    return {
        "implementation": CROSS_ATTENTION_IMPLEMENTATION,
        "contract_version": CROSS_ATTENTION_PREDICTOR_CONTRACT_VERSION,
        "status": "verified_platform_variant",
        "input_adapter": (
            "shared learned embedding or linear projection"
            if hasattr(model, "seq_len")
            else "independent peptide and partner learned embeddings or linear projections"
        ),
        "attention": "multi-head scaled dot-product cross-attention",
        "direction": "peptide_query_partner_key_value",
        "attention_parameterization": "one directional attention module",
        "masking": "partner key padding mask and peptide query masked-mean readout",
        "readout": "unidirectional_cross_attention_pair_fusion_8h",
        "embedding_dim": int(getattr(model, "hidden_dim")) * 8,
        "hidden_dim": int(getattr(model, "hidden_dim")),
        "heads": int(getattr(model, "n_heads")),
        "output": "binary logit before inference-boundary sigmoid",
    }


def pair_bi_cross_attention_predictor_contract(model):
    if (
        getattr(model, "interaction_model_id", "") != "bi_cross_attn"
        or getattr(model, "bi_cross_attention_predictor_contract_version", 0)
        != CROSS_ATTENTION_PREDICTOR_CONTRACT_VERSION
    ):
        return None
    return {
        "implementation": BI_CROSS_ATTENTION_IMPLEMENTATION,
        "contract_version": CROSS_ATTENTION_PREDICTOR_CONTRACT_VERSION,
        "status": "verified_platform_variant",
        "input_adapter": (
            "shared learned embedding or linear projection"
            if hasattr(model, "seq_len")
            else "independent peptide and partner learned embeddings or linear projections"
        ),
        "attention": "two directional multi-head scaled dot-product cross-attention modules",
        "direction": "peptide_query_partner_key_value_and_partner_query_peptide_key_value",
        "attention_parameterization": "independent parameters for each direction",
        "masking": "opposite-stream key padding masks and both query masked-mean readouts",
        "readout": "bidirectional_cross_attention_pair_fusion_8h",
        "embedding_dim": int(getattr(model, "hidden_dim")) * 8,
        "hidden_dim": int(getattr(model, "hidden_dim")),
        "heads": int(getattr(model, "n_heads")),
        "output": "binary logit before inference-boundary sigmoid",
    }


def pair_attention_predictor_contract(model):
    return pair_cross_attention_predictor_contract(
        model
    ) or pair_bi_cross_attention_predictor_contract(model)


def set_global_seed(seed, deterministic=True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
        if bool(deterministic):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def default_pair_csv_candidates(task_root, task_type, split):
    task_type = normalize_task_type(task_type)
    task = normalize_task_type(task_type or "")
    if normalize_task_type(task) == "pep_hlai":
        return [os.path.join(task_root, HLAI_ORIGINAL_SPLIT_DIR, split, "pairs.csv")]
    if split == "test":
        return [
            os.path.join(task_root, "processed", "test_1440.csv"),
            os.path.join(task_root, "processed", "test.csv"),
        ]
    return [os.path.join(task_root, "processed", f"{split}.csv")]


def default_pair_feature_dir(task_root, task_type, data_dir=""):
    task_type = normalize_task_type(task_type)
    if data_dir:
        return os.path.join(data_dir, "features")
    if normalize_task_type(task_type or "") == "pep_hlai":
        return os.path.join(task_root, HLAI_ORIGINAL_SPLIT_DIR, "features")
    return os.path.join(task_root, "features")


class PPIInteractiveWizard(InteractiveConfigWizard):
    def _prompt_partner_stream(self, args, step):
        print(f"Step {step}: Choose Two-stream representation settings")
        topology = self.prompt_choice(
            "Representation settings",
            ["homogeneous (linked)", "heterogeneous (independent)"],
            "homogeneous (linked)",
        )
        step += 1
        if topology.startswith("homogeneous"):
            return step, True

        print(f"Step {step}: Choose partner feature extraction")
        route = self.prompt_choice(
            "Partner feature extraction",
            ["pretrained model", "classic pipeline"],
            "pretrained model",
        )
        step += 1
        if route == "pretrained model":
            print(f"Step {step}: Choose partner tokenizer")
            args.partner_tokenizer = self.prompt_choice(
                "Partner tokenizer",
                PRETRAINED_TOKENIZERS,
                "prott5",
            )
            args.partner_feature_route = "pretrained"
            args.partner_encoder = args.partner_tokenizer
            print(f"Partner encoder: {args.partner_encoder} (coupled to pretrained tokenizer)")
            step += 1
            default_source = PRETRAINED_MODEL_DEFAULTS.get(args.partner_tokenizer, "")
            args.partner_model_source = self.prompt_text(
                f"Step {step}: Partner model source",
                default_source,
            )
            step += 1
            allow_token = True
        else:
            print(f"Step {step}: Choose partner tokenizer")
            args.partner_tokenizer = self.prompt_choice(
                "Partner tokenizer",
                CLASSIC_TOKENIZERS,
                "kmer",
            )
            step += 1
            encoders = CLASSIC_ENCODER_OPTIONS.get(args.partner_tokenizer, ["index"])
            print(f"Step {step}: Choose partner encoder")
            args.partner_encoder = self.prompt_choice(
                "Partner encoder",
                encoders,
                encoders[0],
            )
            args.partner_feature_route = "classic"
            step += 1
            allow_token = str(args.partner_encoder).lower() in set(DL_SEQUENCE_ENCODERS)
        args.partner_max_length = self.prompt_int(
            f"Step {step}: Partner max length",
            int(getattr(args, "max_length", 512)),
            1,
        )
        step += 1
        if args.partner_tokenizer == "kmer":
            args.partner_k = self.prompt_int(
                f"Step {step}: Partner k-mer size",
                int(getattr(args, "k", 2)),
                1,
                max_value=KMER_MAX_K,
            )
            step += 1
        return step, allow_token

    def run(self, args):
        step = 1
        print("Entering interactive mode")
        print(f"Step {step}: Choose task")
        task_options = ["pep_function", "pep_protein_interaction", "pep_hlaii", "pep_hlai"]
        default_task = normalize_task_type(getattr(args, "task_type", "pep_protein_interaction"))
        if normalize_task_type(default_task) == "pep_function_multilabel":
            default_task = "pep_function"
        if normalize_task_type(default_task) not in task_options:
            default_task = "pep_protein_interaction"
        args.task_type = self.prompt_choice("Task", task_options, default_task)
        step += 1

        if normalize_task_type(args.task_type) == "pep_function":
            print(f"{args.task_type} task is handled by scripts/train_sequence_pipeline.py")
            raise SystemExit("Please run: python scripts/train_sequence_pipeline.py")

        print(f"Step {step}: Choose feature extraction")
        default_tokenizer = normalize_tokenizer_name(getattr(args, "tokenizer", ""))
        default_mode = (
            "pretrained model" if default_tokenizer in PRETRAINED_TOKENIZERS else "classic pipeline"
        )
        feature_mode = self.prompt_choice(
            "Feature extraction", ["pretrained model", "classic pipeline"], default_mode
        )
        step += 1

        if feature_mode == "pretrained model":
            args.feature_mode = "single"
            print(f"Step {step}: Choose tokenizer")
            args.tokenizer = self.prompt_choice(
                "Tokenizer", PRETRAINED_TOKENIZERS, default_tokenizer or "esm2"
            )
            args.encoder = args.tokenizer
            print(f"Encoder: {args.encoder} (coupled to pretrained tokenizer)")
            step += 1
        elif feature_mode == "classic pipeline":
            args.feature_mode = "single"
            print(f"Step {step}: Choose tokenizer")
            args.tokenizer = self.prompt_choice(
                "Tokenizer", CLASSIC_TOKENIZERS, getattr(args, "tokenizer", "kmer")
            )
            step += 1
            print(f"Step {step}: Choose encoder")
            enc_options = CLASSIC_ENCODER_OPTIONS.get(args.tokenizer, ["index"])
            default_enc = getattr(args, "encoder", enc_options[0])
            if default_enc not in enc_options:
                default_enc = enc_options[0]
            args.encoder = self.prompt_choice("Encoder", enc_options, default_enc)
            step += 1
            step, _ = self._prompt_feature_transformation(args, step)
            if args.tokenizer in {"top-n-gram", "dt"}:
                default_frequency = getattr(
                    args, "frequency_profile_index_path", ""
                ) or os.path.join(
                    getattr(
                        args,
                        "task_root",
                        "datasets/task_peptide_protein_interaction",
                    ),
                    "frequency_profile_cache",
                    "ppi",
                    "index.json",
                )
                args.frequency_profile_index_path = self.prompt_text(
                    f"Step {step}: target-frequency profile index path",
                    default_frequency,
                )
                args.profile_on_missing = TOP_N_GRAM_FALLBACK_POLICY
                step += 1
            if args.tokenizer == "dt":
                args.dt_max_distance = self.prompt_int(
                    f"Step {step}: DT max distance",
                    int(
                        getattr(
                            args,
                            "dt_max_distance",
                            DT_DEFAULT_MAX_DISTANCE,
                        )
                    ),
                    0,
                    max_value=DT_MAX_DISTANCE,
                )
                step += 1
        enc_name = str(getattr(args, "encoder", "index")).lower()
        seq_encoder_set = set(DL_SEQUENCE_ENCODERS)
        allow_single = feature_mode == "classic pipeline" and enc_name in seq_encoder_set
        allow_token = feature_mode == "pretrained model" or enc_name in seq_encoder_set
        step, _ = self._prompt_pair_strategy(args, step, allow_single=allow_single)
        if args.pair_strategy == "single_stream":
            step, _ = self._prompt_sequence_route(args, step)
            self._set_pair_route(args, pair_strategy="single_stream")
        else:
            step, partner_allows_token = self._prompt_partner_stream(args, step)
            allow_token = allow_token and partner_allows_token
            step, _ = self._prompt_two_stream_route(args, step, allow_token=allow_token)
            if getattr(args, "partner_tokenizer", ""):
                if getattr(args, "predictor_type", "classifier") == "sample_graph":
                    if str(getattr(args, "model", "") or "").lower() == HGT_MODEL:
                        print("HGT keeps heterogeneous streams as typed role inputs.")
                        args.fusion_mode = None
                    else:
                        print("Heterogeneous sample-graph streams use concat fusion.")
                        args.fusion_mode = "concat"
                elif getattr(
                    args, "interaction_level", "vector"
                ) == "vector" and resolve_fusion_mode(
                    getattr(args, "fusion_mode", "concat_all")
                ) not in {"concat", "aligned_gated"}:
                    print("Heterogeneous streams require concat or aligned_gated; using concat.")
                    args.fusion_mode = "concat"
        step, semantic_back = self._prompt_semantic_auxiliary(args, step)
        if semantic_back == "__BACK__":
            print("Returning to previous step")
            return self.run(args)

        args.max_length = self.prompt_int(
            f"Step {step}: max length", getattr(args, "max_length", 512), 1
        )
        step += 1
        args.batch_size = self.prompt_int(
            f"Step {step}: batch size", getattr(args, "batch_size", 16), 1
        )
        step += 1

        if args.model in (DL_SEQUENCE_MODELS + ["mlp"] + GRAPH_MODELS):
            args.dl_scaler = self.prompt_choice(
                "DL scaler", ["standard", "minmax", "none"], getattr(args, "dl_scaler", "standard")
            )
            step += 1
            args.lr = self.prompt_float(
                f"Step {step}: learning rate", getattr(args, "lr", 1e-3), 0.0
            )
            step += 1
            args.epochs = self.prompt_int(f"Step {step}: epochs", getattr(args, "epochs", 20), 1)
            step += 1
            args.dropout = self.prompt_float(
                f"Step {step}: dropout", getattr(args, "dropout", 0.2), 0.0
            )
            step += 1
            args.weight_decay = self.prompt_float(
                f"Step {step}: weight decay", getattr(args, "weight_decay", 1e-2), 0.0
            )
            step += 1
            if args.model in GRAPH_MODELS:
                graph_name = {
                    GCN_MODEL: "GCN",
                    GAT_MODEL: "GAT",
                    HGT_MODEL: "HGT",
                    SAGE_MODEL: "GraphSAGE",
                    GT_MODEL: "GraphTransformer",
                }.get(args.model, str(args.model).upper())
                print(f"{graph_name} selected: sample-graph")
                args.gnn_k = self.prompt_int(
                    f"Step {step}: {graph_name} kNN k", int(getattr(args, "gnn_k", 8)), 1
                )
                step += 1
                args.gnn_hidden_dim = self.prompt_int(
                    f"Step {step}: {graph_name} hidden dim",
                    int(getattr(args, "gnn_hidden_dim", 256)),
                    8,
                )
                step += 1
            args.early_stop_patience = self.prompt_int(
                f"Step {step}: early stop patience", getattr(args, "early_stop_patience", 5), 0
            )
            step += 1
            args.early_stop_min_delta = self.prompt_float(
                f"Step {step}: early stop min delta",
                getattr(args, "early_stop_min_delta", 0.0),
                0.0,
            )
            step += 1
        if self._needs_ml_settings(args):
            args.ml_scaler = self.prompt_choice(
                "ML scaler", ["standard", "minmax", "none"], getattr(args, "ml_scaler", "standard")
            )
            step += 1

        if normalize_task_type(args.task_type) == "pep_hlai":
            default_root = str(resolve_hlai_root(self._project_root()))
        elif normalize_task_type(args.task_type) == "pep_hlaii":
            default_root = "datasets/task_peptide_HLAII_interaction"
        else:
            default_root = "datasets/task_peptide_protein_interaction"
        args.task_root = self.prompt_text(
            f"Step {step}: task root", getattr(args, "task_root", default_root)
        )
        step += 1
        args.train_csv = self.prompt_text(f"Step {step}: train csv", getattr(args, "train_csv", ""))
        step += 1
        args.val_csv = self.prompt_text(f"Step {step}: val csv", getattr(args, "val_csv", ""))
        step += 1
        args.test_csv = self.prompt_text(f"Step {step}: test csv", getattr(args, "test_csv", ""))
        step += 1
        args.feature_dir = self.prompt_text(
            f"Step {step}: feature dir", getattr(args, "feature_dir", "")
        )

        print("Configuration complete")
        print(f"task_type={args.task_type}")
        print(f"feature_mode={getattr(args, 'feature_mode', 'single')}")
        print(f"tokenizer={args.tokenizer}")
        print(f"encoder={getattr(args, 'encoder', '')}")
        if getattr(args, "partner_tokenizer", ""):
            print(f"partner_tokenizer={args.partner_tokenizer}")
            print(f"partner_encoder={getattr(args, 'partner_encoder', '')}")
            print(f"partner_model_source={getattr(args, 'partner_model_source', '')}")
            print(f"partner_max_length={getattr(args, 'partner_max_length', '')}")
        print(f"pair_strategy={getattr(args, 'pair_strategy', '')}")
        if getattr(args, "interaction_level", ""):
            print(f"interaction_level={args.interaction_level}")
        if getattr(args, "interaction_model", ""):
            print(f"interaction_model={args.interaction_model}")
        print(f"predictor_type={getattr(args, 'predictor_type', 'classifier')}")
        if getattr(args, "graph_predictor", ""):
            print(f"graph_predictor={args.graph_predictor}")
        if getattr(args, "sequence_model", ""):
            print(f"sequence_model={args.sequence_model}")
            print(f"training_mode={args.training_mode}")
            print(f"classifier={args.classifier}")
        if getattr(args, "feature_mode", "single") == "multimodal":
            print(f"primary_feature_mode={getattr(args, 'primary_feature_mode', 'pretrained')}")
            print(f"aux_feature_mode={getattr(args, 'aux_feature_mode', 'classic')}")
            print(f"aux_tokenizer={getattr(args, 'aux_tokenizer', 'kmer')}")
            print(f"aux_encoder={getattr(args, 'aux_encoder', 'index')}")
            print(f"mm_align_dim={int(getattr(args, 'mm_align_dim', 256))}")
        print(f"model={args.model}")
        if args.model == HGT_MODEL:
            print("input_adapter=typed_role_projection")
        elif args.model in ["mlp", "svm", "rf", "xgb", "lgb", "logreg"] + GRAPH_MODELS:
            print(f"fusion_mode={getattr(args, 'fusion_mode', 'concat_all')}")
        print(f"enable_semantic_similarity={self._semantic_auxiliary_enabled(args)}")
        if self._semantic_auxiliary_enabled(args):
            print("semantic_similarity_type=auxiliary_loss")
        print(f"task_root={args.task_root}")
        return args


def _jsonable_args(args):
    from pepplatform.stream_compat import normalize_stream_namespace
    normalize_stream_namespace(args)
    payload = {}
    for key, value in vars(args).items():
        if str(key).startswith("_"):
            continue
        if key == "fusion_mode" and value is None:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        else:
            payload[key] = str(value)
    return payload


def _pair_loss_value(value):
    """Return a JSON-friendly scalar for binary or per-output loss values."""

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif isinstance(value, (tuple, list)):
        value = list(value)
    else:
        return value
    if len(value) == 1:
        return value[0]
    return value


def _validate_pair_classification_loss_args(args):
    """Normalize the configurable pair classification-loss contract.

    Missing fields are intentionally interpreted as the historical fixed WBCE
    configuration so that sealed pre-option artifacts/configs remain replayable.
    """

    loss_type = (
        str(
            getattr(args, "loss_type", PAIR_CLASSIFICATION_LOSS_DEFAULT)
            or PAIR_CLASSIFICATION_LOSS_DEFAULT
        )
        .strip()
        .lower()
    )
    if loss_type not in {"bce", "wbce"}:
        raise ValueError("loss_type must be one of: bce, wbce")
    raw_pos_weight_max = getattr(
        args,
        "pos_weight_max",
        PAIR_CLASSIFICATION_POS_WEIGHT_MAX_DEFAULT,
    )
    if raw_pos_weight_max is None:
        raw_pos_weight_max = PAIR_CLASSIFICATION_POS_WEIGHT_MAX_DEFAULT
    try:
        pos_weight_max = float(raw_pos_weight_max)
    except (TypeError, ValueError) as exc:
        raise ValueError("pos_weight_max must be a finite number >= 1") from exc
    if not np.isfinite(pos_weight_max) or pos_weight_max < 1.0:
        raise ValueError("pos_weight_max must be a finite number >= 1")
    args.loss_type = loss_type
    args.pos_weight_max = pos_weight_max
    return loss_type, pos_weight_max


def build_pair_classification_criterion(args, train_y, device):
    """Build the pair neural binary criterion and record its provenance.

    ``wbce`` preserves the historical positive-class weighting formula.  ``bce``
    deliberately leaves ``pos_weight`` unset.  Counts are measured from the
    training split only and are stored on ``args`` for artifact metadata.
    """

    loss_type, pos_weight_max = _validate_pair_classification_loss_args(args)
    labels = train_y if isinstance(train_y, torch.Tensor) else torch.as_tensor(train_y)
    labels = labels.detach().float()
    if labels.ndim == 0:
        labels = labels.reshape(1)
    positive_count = labels.sum(dim=0)
    negative_count = labels.shape[0] - positive_count
    raw_pos_weight = negative_count / (positive_count + 1e-6)
    effective_pos_weight = None
    if loss_type == "wbce":
        effective_pos_weight = torch.clamp(
            raw_pos_weight,
            min=1.0,
            max=pos_weight_max,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=effective_pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    args._classification_loss_provenance = {
        "schema_version": PAIR_CLASSIFICATION_LOSS_SCHEMA_VERSION,
        "loss_type": loss_type,
        "pos_weight_max": float(pos_weight_max),
        "positive_count": _pair_loss_value(positive_count.round().to(torch.int64)),
        "negative_count": _pair_loss_value(negative_count.round().to(torch.int64)),
        "raw_pos_weight": _pair_loss_value(raw_pos_weight.float()),
        "effective_pos_weight": _pair_loss_value(
            effective_pos_weight.float() if effective_pos_weight is not None else None
        ),
    }
    return criterion


def _pair_protocol_metadata(args=None, *, model=None, evaluation_mode=None):
    if evaluation_mode is None:
        evaluation_mode = (
            "inductive"
            if str(getattr(args, "predictor_type", "") or "").strip().lower() == "sample_graph"
            else "independent_iid"
        )
    metadata = {
        "route_schema_version": ROUTE_SCHEMA_VERSION,
        "effective_route": resolve_effective_route(
            vars(args), getattr(args, "task_type", "pep_protein_interaction")
        )
        if args is not None
        else {},
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "checkpoint_metric": resolve_selection(
            args or {"task_type": normalize_task_type("pep_hlai")}
        ).checkpoint.checkpoint_metric,
        "feature_fit_scope": "train_only",
        "evaluation_mode": str(evaluation_mode),
        "metric_contract": {
            "task_type": normalize_task_type("binary_classification"),
            "positive_label": 1,
            "averaging": "binary",
            "checkpoint_metric": resolve_selection(
                args or {"task_type": normalize_task_type("pep_hlai")}
            ).checkpoint.checkpoint_metric,
            "threshold_source": "validation_f1",
        },
    }
    if args is not None:
        metadata.update(selection_fields(args))
    operating_point = getattr(args, "_operating_point", None) if args is not None else None
    if operating_point:
        metadata["operating_point"] = dict(operating_point)
    semantic_provenance = getattr(args, "_semantic_provenance", None) if args is not None else None
    if semantic_provenance:
        metadata["semantic"] = {
            **dict(semantic_provenance),
            "main_batch_sampler": {
                "implementation": "torch.utils.data.RandomSampler",
                "seed": int(getattr(args, "seed", 42)),
                "epoch_order_hashes": list(getattr(args, "_main_batch_order_hashes", []) or []),
            },
        }
        if model is not None:
            metadata["semantic"]["representation"] = semantic_representation_contract(
                model,
                relation_type=str(semantic_provenance.get("relation_type") or ""),
                pair_mode=str(semantic_provenance.get("pair_mode") or "role_fused"),
            )
    classification_loss_provenance = (
        getattr(args, "_classification_loss_provenance", None) if args is not None else None
    )
    if classification_loss_provenance:
        metadata["classification_loss"] = dict(classification_loss_provenance)
    return metadata


def _frozen_pair_threshold(args):
    try:
        operating_point = OperatingPoint.from_dict(getattr(args, "_operating_point", None))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("pair artifact requires a validation-fitted operating point") from exc
    if operating_point.source != "validation" or not operating_point.is_binary:
        raise RuntimeError("pair artifact requires one binary threshold fitted on validation data")
    return operating_point.threshold


def _pair_vector_artifacts_supported(args):
    task = normalize_task_type(getattr(args, "task_type", "") or "")
    tokenizer = str(getattr(args, "tokenizer", "") or "").strip().lower()
    pair_strategy = str(getattr(args, "pair_strategy", "") or "").strip().lower()
    interaction_level = str(getattr(args, "interaction_level", "") or "").strip().lower()
    predictor_type = str(getattr(args, "predictor_type", "") or "").strip().lower()
    model = str(getattr(args, "model", "") or "").strip().lower()
    classifier = str(getattr(args, "classifier", "") or "").strip().lower() or model
    if normalize_task_type(task) not in PAIR_INFERENCE_TASKS:
        return False
    if (
        not getattr(args, "streams", None)
        and tokenizer not in PRETRAINED_TOKENIZERS
        and not supports_classic_pair_vector_inference(args)
    ):
        return False
    if (
        pair_strategy != "two_stream"
        or interaction_level != "vector"
        or predictor_type != "classifier"
    ):
        return False
    if model not in ["mlp", *CLASSICAL_CLASSIFIERS] or classifier not in [
        "mlp",
        *CLASSICAL_CLASSIFIERS,
    ]:
        return False
    fusion_mode = resolve_fusion_mode(getattr(args, "fusion_mode", "concat_all"))
    return fusion_mode != "aligned_gated" or model == "mlp"


def _pair_graph_artifacts_supported(args):
    task = normalize_task_type(getattr(args, "task_type", "") or "")
    tokenizer = str(getattr(args, "tokenizer", "") or "").strip().lower()
    pair_strategy = str(getattr(args, "pair_strategy", "") or "").strip().lower()
    interaction_level = str(getattr(args, "interaction_level", "") or "").strip().lower()
    predictor_type = str(getattr(args, "predictor_type", "") or "").strip().lower()
    model = str(getattr(args, "model", "") or "").strip().lower()
    fusion_mode = resolve_fusion_mode(getattr(args, "fusion_mode", "concat_all"))
    if normalize_task_type(task) not in PAIR_INFERENCE_TASKS:
        return False
    if (
        not getattr(args, "streams", None)
        and tokenizer not in PRETRAINED_TOKENIZERS
        and not supports_classic_pair_vector_inference(args)
    ):
        return False
    if (
        pair_strategy != "two_stream"
        or interaction_level != "vector"
        or predictor_type != "sample_graph"
    ):
        return False
    if model not in GRAPH_MODELS:
        return False
    return fusion_mode != "aligned_gated" or model in {GCN_MODEL, GAT_MODEL}


def _pair_token_artifacts_supported(args):
    task = normalize_task_type(getattr(args, "task_type", "") or "")
    return normalize_task_type(task) in PAIR_INFERENCE_TASKS and supports_pair_token_inference(args)


def _pair_sequence_artifacts_supported(args):
    task = normalize_task_type(getattr(args, "task_type", "") or "")
    pair_strategy = str(getattr(args, "pair_strategy", "") or "").strip().lower()
    training_mode = str(getattr(args, "training_mode", "") or "").strip().lower()
    model = str(getattr(args, "model", "") or "").strip().lower()
    classifier = str(getattr(args, "classifier", "") or "").strip().lower() or "mlp"
    tokenizer = normalize_tokenizer_name(str(getattr(args, "tokenizer", "") or "").strip())
    encoder = normalize_encoder_name(effective_pair_vector_encoder(args))
    feature_supported = (
        tokenizer in PRETRAINED_TOKENIZERS and encoder in {tokenizer, "index", "ppi_fusion"}
    ) or supports_classic_pair_token_inference(args)
    return (
        normalize_task_type(task) in PAIR_INFERENCE_TASKS
        and pair_strategy == "single_stream"
        and training_mode == "end_to_end"
        and model in DL_SEQUENCE_MODELS
        and classifier == "mlp"
        and feature_supported
    )


def _pair_input_schema(args):
    task = normalize_task_type(getattr(args, "task_type", "") or "")
    if normalize_task_type(task) == "pep_hlai":
        return "peptide_hlai_pairs"
    if normalize_task_type(task) == "pep_hlaii":
        return "peptide_hla_pairs"
    return "peptide_protein_pairs"


def _resolved_pair_pretrained_model_source(args):
    explicit = str(getattr(args, "_resolved_pretrained_model_source", "") or "").strip()
    if explicit:
        return explicit
    tokenizer = str(getattr(args, "tokenizer", "") or "").strip().lower()
    if tokenizer == "probert":
        tokenizer = "protbert"
    if tokenizer == "esm2":
        return resolve_esm2_model_source(args)
    if tokenizer == "protbert":
        return resolve_protbert_model_source(args)
    if tokenizer == "prott5":
        return resolve_prott5_model_source(args)
    if tokenizer == "esmc":
        return resolve_esmc_model_source(args)
    if tokenizer == "esm3":
        return resolve_esm3_model_source(args)
    return str(getattr(args, f"{tokenizer}_model", "") or "").strip()


def _resolved_pair_pretrained_model_identity(args):
    identity = getattr(args, "_resolved_pretrained_model_identity", None)
    return dict(identity) if isinstance(identity, dict) else None


def _resolved_pair_pretrained_runtime_environment(args):
    environment = getattr(args, "_resolved_pretrained_runtime_environment", None)
    return dict(environment) if isinstance(environment, dict) else None


def _pair_profile_generation_needed(args):
    return _pair_profile_feature_kind(args) is not None


def _profile_feature_for_encoder(encoder):
    normalized = str(encoder or "").strip().lower()
    if normalized in PSSM_COVARIANCE_ENCODERS:
        return pssm_covariance_profile_feature_type(normalized)
    if normalized == "pssm_blosum62_hybrid":
        return FEATURE_PSSM_BLOSUM62_HYBRID
    if normalized == "pssm":
        return FEATURE_PSSM
    if normalized == "psfm":
        return FEATURE_PSFM
    if normalized == "psfm_blosum62_hybrid":
        return FEATURE_PSFM_BLOSUM62_HYBRID
    return None


def _profile_index_key(feature):
    if feature in {FEATURE_PSFM, FEATURE_PSFM_BLOSUM62_HYBRID}:
        return "psfm_index_path"
    return PROFILE_INDEX_KEYS.get(feature, "pssm_index_path")


def _pair_profile_feature_kind(args):
    routes = [
        (
            str(getattr(args, "tokenizer", "") or "").strip().lower(),
            effective_pair_vector_encoder(args),
        )
    ]
    for spec in (getattr(args, "streams", {}) or {}).values():
        if not isinstance(spec, dict):
            continue
        routes.append(
            (
                str(spec.get("tokenizer", "") or "").strip().lower(),
                str(spec.get("encoder", "index") or "index").strip().lower(),
            )
        )
    need_frequency = any(tokenizer in {"top-n-gram", "dt"} for tokenizer, _encoder in routes)
    requested = list(
        dict.fromkeys(
            feature
            for _tokenizer, encoder in routes
            for feature in [_profile_feature_for_encoder(encoder)]
            if feature
        )
    )
    if need_frequency:
        requested.append("target_frequency")
    return ",".join(requested) if requested else None


def _artifact_relpath(artifact_dir, path):
    if not artifact_dir or not path:
        return ""
    try:
        return os.path.relpath(path, artifact_dir)
    except ValueError:
        return path


def _blast_db_ready(db_prefix):
    if not db_prefix:
        return False
    return all(os.path.exists(f"{db_prefix}{suffix}") for suffix in [".pin", ".phr", ".psq"])


def _build_pair_profile_blast_db(subject_fasta, db_prefix):
    if shutil.which("makeblastdb") is None:
        print("[artifact] PPI profile BLAST DB not prebuilt: makeblastdb not found")
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
        print(f"[artifact] PPI profile BLAST DB prebuild skipped: {exc}")
        return False
    return _blast_db_ready(db_prefix)


def _pair_profile_generation_artifact_info(args, profile_rows=None):
    if not _pair_profile_generation_needed(args):
        return {}
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    subject_rel = ""
    db_prefix_rel = ""
    config = ProfileGenerationConfig(
        iterations=int(getattr(args, "profile_iterations", 3)),
        evalue=float(getattr(args, "profile_evalue", 0.001)),
        inclusion_evalue=float(getattr(args, "profile_inclusion_evalue", 0.001)),
        threads=int(
            getattr(
                args,
                "profile_threads",
                getattr(args, "embedding_workers", 4),
            )
        ),
    )
    feature_value = _pair_profile_feature_kind(args)
    features = [item for item in str(feature_value).split(",") if item]
    fallback_policies = {
        feature: (
            "blosum62_hybrid"
            if feature in (PSSM_HYBRID_FEATURE_TYPES | PSFM_HYBRID_FEATURE_TYPES)
            else str(getattr(args, "profile_on_missing", "error") or "error")
            if feature == "target_frequency"
            else "error"
        )
        for feature in features
    }
    manifest_contracts = {}
    subject_sources = []
    database_scopes = set()

    def register_subject_source(provenance, index_path):
        scope = str((provenance or {}).get("database_scope") or "")
        if scope:
            database_scopes.add(scope)
        artifact = str((provenance or {}).get("subject_fasta_artifact") or "")
        index_value = str(index_path or "").strip()
        if artifact and index_value:
            index_root = Path(index_value).parent.resolve()
            candidate = (index_root / artifact).resolve()
            try:
                candidate.relative_to(index_root)
            except ValueError as exc:
                raise RuntimeError(
                    "pair profile subject FASTA path escapes its cache root"
                ) from exc
            subject_sources.append(str(candidate))

    direct_digest = str(getattr(args, "_profile_manifest_sha256", "") or "")
    if direct_digest:
        manifest_contracts["classic"] = {
            "manifest_sha256": direct_digest,
            "provenance": dict(getattr(args, "_profile_manifest_provenance", {}) or {}),
        }
        direct_provenance = manifest_contracts["classic"]["provenance"]
        direct_feature = _profile_feature_for_encoder(effective_pair_vector_encoder(args))
        direct_index = getattr(
            args,
            _profile_index_key(direct_feature),
            "",
        )
        register_subject_source(direct_provenance, direct_index)
    frequency_digest = str(getattr(args, "_frequency_profile_manifest_sha256", "") or "")
    if frequency_digest:
        manifest_contracts["classic_target_frequency"] = {
            "manifest_sha256": frequency_digest,
            "provenance": dict(getattr(args, "_frequency_profile_provenance", {}) or {}),
        }
        register_subject_source(
            manifest_contracts["classic_target_frequency"]["provenance"],
            getattr(args, "frequency_profile_index_path", ""),
        )
    for role, runtime in (getattr(args, "_stream_runtime", {}) or {}).items():
        if not isinstance(runtime, dict):
            continue
        digest = str(runtime.get("profile_manifest_sha256", "") or "")
        if digest:
            manifest_contracts[str(role)] = {
                "manifest_sha256": digest,
                "provenance": dict(runtime.get("profile_manifest_provenance") or {}),
            }
            resolved = runtime.get("resolved_params") or {}
            stream_spec = (getattr(args, "streams", {}) or {}).get(role, {})
            role_feature = _profile_feature_for_encoder(
                stream_spec.get("encoder") if isinstance(stream_spec, dict) else ""
            )
            role_index = resolved.get(_profile_index_key(role_feature))
            register_subject_source(
                manifest_contracts[str(role)]["provenance"],
                role_index,
            )
        frequency_digest = str(runtime.get("frequency_profile_manifest_sha256", "") or "")
        if frequency_digest:
            manifest_contracts[f"{role}_target_frequency"] = {
                "manifest_sha256": frequency_digest,
                "provenance": dict(runtime.get("frequency_profile_provenance") or {}),
            }
            register_subject_source(
                manifest_contracts[f"{role}_target_frequency"]["provenance"],
                (runtime.get("resolved_params") or {}).get("frequency_profile_index_path"),
            )
    if artifact_dir and not manifest_contracts:
        raise RuntimeError(
            "cannot export a profile-backed pair model without validated "
            f"v{PROFILE_CONTRACT_VERSION} training profile manifests"
        )
    if artifact_dir and manifest_contracts:
        if len(database_scopes) != 1:
            raise RuntimeError("pair profile manifests disagree on database scope")
        unique_sources = list(dict.fromkeys(subject_sources))
        if not unique_sources:
            raise RuntimeError(
                "validated training profile manifests do not identify a "
                "frozen subject FASTA artifact"
            )
        missing_sources = [source for source in unique_sources if not os.path.isfile(source)]
        if missing_sources:
            raise RuntimeError(
                "training profile manifest is missing frozen subject FASTA: "
                + ", ".join(missing_sources)
            )
        source_digests = {sha256_file(source) for source in unique_sources}
        if len(source_digests) != 1:
            raise RuntimeError("pair streams were trained against different subject FASTA artifacts")
        if unique_sources:
            profile_dir = os.path.join(
                artifact_dir,
                "profile_generation",
            )
            os.makedirs(profile_dir, exist_ok=True)
            subject_fasta = os.path.join(profile_dir, "subject.fasta")
            atomic_copy_file(
                unique_sources[0],
                subject_fasta,
                expected_sha256=next(iter(source_digests)),
            )
            subject_rel = _artifact_relpath(artifact_dir, subject_fasta)
            db_prefix = os.path.join(
                profile_dir,
                "blastdb",
                "ppi_profile_db",
            )
            if _build_pair_profile_blast_db(subject_fasta, db_prefix):
                db_prefix_rel = _artifact_relpath(artifact_dir, db_prefix)
    subject_sha256 = sha256_file(os.path.join(artifact_dir, subject_rel)) if subject_rel else ""
    if subject_rel:
        if not manifest_contracts:
            raise RuntimeError(
                "cannot export a profile-backed pair model without a "
                "validated training profile manifest"
            )
        mismatched_roles = [
            role
            for role, contract in manifest_contracts.items()
            if str((contract.get("provenance") or {}).get("subject_fasta_sha256") or "")
            != subject_sha256
        ]
        if mismatched_roles:
            raise RuntimeError(
                "exported pair profile subject FASTA does not exactly match "
                "the training profile database for: " + ", ".join(sorted(mismatched_roles))
            )
    return {
        "profile_generation": {
            "profile_contract_version": PROFILE_CONTRACT_VERSION,
            "strategy": "psiblast_subject_fasta",
            "feature": feature_value,
            "features": features,
            "subject_fasta": subject_rel,
            "subject_fasta_sha256": subject_sha256,
            "db_prefix": db_prefix_rel,
            "db_scope": (next(iter(database_scopes)) if len(database_scopes) == 1 else ""),
            "iterations": config.iterations,
            "evalue": config.evalue,
            "inclusion_evalue": config.inclusion_evalue,
            "threads": config.threads,
            "scientific_parameters": config.scientific_parameters(),
            "fallback_policies": fallback_policies,
            "on_missing": (
                next(iter(fallback_policies.values()))
                if len(set(fallback_policies.values())) == 1
                else "mixed_explicit"
            ),
            "training_manifests": manifest_contracts,
        }
    }


def _classic_tfidf_artifact_info(args):
    encoder = effective_pair_vector_encoder(args)
    if encoder != "tfidf":
        return {}
    idf = getattr(args, "_classic_tfidf_idf", None)
    if idf is None:
        return {}
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    if not artifact_dir:
        return {}
    descriptor = save_idf_state(
        idf,
        artifact_dir=artifact_dir,
        filename="classic_tfidf_idf.pt",
    )
    return {
        "tfidf_idf": descriptor,
        "tfidf_idf_path": descriptor["path"],
    }


def _unigram_regularization_info(args):
    return {
        "enabled": bool(getattr(args, "unigram_sampling", False)),
        "nbest_size": int(
            getattr(
                args,
                "unigram_nbest_size",
                UNIGRAM_DEFAULT_NBEST_SIZE,
            )
        ),
        "alpha": float(getattr(args, "unigram_alpha", UNIGRAM_DEFAULT_ALPHA)),
        "inference_mode": "viterbi",
    }


def _bundle_dr_static_embedding_artifact(
    args,
    *,
    encoder,
    source,
    provenance=None,
    role=None,
):
    normalized_encoder = str(encoder or "").strip().lower()
    source_path = str(source or "").strip()
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    if (
        normalized_encoder not in {"word2vec", "glove", "fasttext"}
        or not artifact_dir
        or not source_path
        or not os.path.isfile(source_path)
    ):
        return {}
    target_dir = os.path.join(artifact_dir, "classic")
    os.makedirs(target_dir, exist_ok=True)
    role_prefix = f"{role}_" if role else ""
    extension = ".bin" if normalized_encoder == "fasttext" else ".vec"
    target = os.path.join(
        target_dir,
        f"{role_prefix}static_{normalized_encoder}{extension}",
    )
    if os.path.abspath(source_path) != os.path.abspath(target):
        shutil.copy2(source_path, target)
    relative = _artifact_relpath(artifact_dir, target)
    bundled_provenance = dict(provenance or {})
    bundled_provenance.update(
        {
            "method": normalized_encoder,
            "path": relative,
            "sha256": sha256_file(target),
            "bundled": True,
        }
    )
    return {
        f"{normalized_encoder}_path": relative,
        "static_embedding": bundled_provenance,
    }


def _classic_extra_artifact_info(args):
    info = {}
    tokenizer = str(getattr(args, "tokenizer", "") or "").strip().lower()
    if effective_pair_vector_encoder(args) == "blosum62":
        info["blosum62_contract"] = blosum62_contract_metadata(int(getattr(args, "k", 1)))
    if tokenizer == "dr":
        max_distance = validate_dr_max_distance(
            getattr(args, "dr_max_distance", DR_DEFAULT_MAX_DISTANCE)
        )
        max_raw_length = validate_dr_max_raw_length(
            (
                getattr(
                    args,
                    "frequency_max_raw_length",
                    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
                )
                if effective_pair_vector_encoder(args) in {"bow", "tfidf"}
                else getattr(
                    args,
                    "dr_max_raw_length",
                    DR_DEFAULT_MAX_RAW_LENGTH,
                )
            )
        )
        unknown_policy = validate_dr_unknown_residue_policy(
            getattr(
                args,
                "dr_unknown_residue_policy",
                DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
            )
        )
        info.update(
            {
                "dr_max_distance": max_distance,
                "dr_max_raw_length": max_raw_length,
                "dr_unknown_residue_policy": unknown_policy,
                "dr_contract": dr_contract_metadata(
                    max_distance=max_distance,
                    max_raw_length=max_raw_length,
                    unknown_residue_policy=unknown_policy,
                ),
                "max_length_unit": (
                    DR_GLOBAL_MAX_LENGTH_UNIT
                    if effective_pair_vector_encoder(args) in {"bow", "tfidf", "textrank"}
                    else DR_ORDERED_MAX_LENGTH_UNIT
                ),
            }
        )
        encoder = effective_pair_vector_encoder(args)
        bundled_static = _bundle_dr_static_embedding_artifact(
            args,
            encoder=encoder,
            source=getattr(args, f"{encoder}_path", ""),
            provenance=getattr(
                args,
                "_static_embedding_provenance",
                None,
            ),
        )
        info.update(bundled_static)
    elif tokenizer == "top-n-gram":
        top_n = validate_top_ngram_n(getattr(args, "top_ngram_n", 2))
        info.update(
            {
                "frequency_profile_index_path": str(
                    getattr(
                        args,
                        "frequency_profile_index_path",
                        "",
                    )
                    or ""
                ),
                "max_length_unit": TOP_N_GRAM_MAX_LENGTH_UNIT,
                "feature_dim": top_ngram_feature_dim(top_n),
                "top_ngram_contract": {
                    "contract_version": TOP_N_GRAM_CONTRACT_VERSION,
                    "profile_type": "target_frequency",
                    "tie_policy": TOP_N_GRAM_TIE_POLICY,
                    "full_length": True,
                    "fallback_profiles_allowed": (
                        str(
                            getattr(
                                args,
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
                                args,
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
                            args,
                            "_frequency_profile_manifest_sha256",
                            "",
                        )
                        or ""
                    ),
                },
            }
        )
    elif tokenizer == "dt":
        max_distance = validate_dt_max_distance(
            getattr(args, "dt_max_distance", DT_DEFAULT_MAX_DISTANCE)
        )
        contract = dt_contract_metadata(max_distance=max_distance)
        contract.update(
            {
                "fallback_profiles_allowed": (
                    str(
                        getattr(
                            args,
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
                            args,
                            "profile_on_missing",
                            TOP_N_GRAM_FALLBACK_POLICY,
                        )
                    )
                    == TOP_N_GRAM_FALLBACK_POLICY
                    else TOP_N_GRAM_STRICT_PROFILE_POLICY
                ),
                "profile_manifest_sha256": str(
                    getattr(
                        args,
                        "_frequency_profile_manifest_sha256",
                        "",
                    )
                    or ""
                ),
            }
        )
        info.update(
            {
                "frequency_profile_index_path": str(
                    getattr(
                        args,
                        "frequency_profile_index_path",
                        "",
                    )
                    or ""
                ),
                "max_length_unit": DT_MAX_LENGTH_UNIT,
                "feature_dim": contract["feature_dim"],
                "dt_max_distance": max_distance,
                "dt_contract": contract,
            }
        )
    elif tokenizer == "bpe":
        artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
        bpe_path = str(getattr(args, "bpe_path", "") or "").strip()
        if artifact_dir and bpe_path:
            info.update(bundle_bpe_artifact(bpe_path, artifact_dir))
    elif tokenizer == "wordpiece":
        artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
        wordpiece_path = str(getattr(args, "wordpiece_path", "") or "").strip()
        if artifact_dir and wordpiece_path:
            info.update(
                bundle_wordpiece_artifact(
                    wordpiece_path,
                    artifact_dir,
                )
            )
    elif tokenizer == "unigram":
        artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
        unigram_path = str(getattr(args, "unigram_path", "") or "").strip()
        if artifact_dir and unigram_path:
            info.update(
                bundle_unigram_artifact(
                    unigram_path,
                    artifact_dir,
                )
            )
    contract = getattr(args, "_frequency_encoder_contract", None)
    if contract:
        info["frequency_encoder_contract"] = dict(contract)
        info["frequency_max_raw_length"] = int(contract["frequency_max_raw_length"])
    textrank_contract = getattr(args, "_textrank_contract", None)
    if textrank_contract:
        info["textrank_contract"] = dict(textrank_contract)
    covariance_contract = getattr(
        args,
        "_covariance_encoder_contract",
        None,
    )
    if covariance_contract:
        info["covariance_encoder_contract"] = dict(covariance_contract)
        info["covariance_max_raw_length"] = int(covariance_contract["covariance_max_raw_length"])
    pdt_contract = getattr(args, "_pdt_encoder_contract", None)
    if pdt_contract:
        info["pdt_encoder_contract"] = dict(pdt_contract)
        info["pdt_max_raw_length"] = int(pdt_contract["pdt_max_raw_length"])
    pssm_covariance_contract = getattr(
        args,
        "_pssm_covariance_encoder_contract",
        None,
    )
    if pssm_covariance_contract:
        info["pssm_covariance_encoder_contract"] = dict(pssm_covariance_contract)
    info.update(_classic_tfidf_artifact_info(args))
    provenance = getattr(args, "_static_embedding_provenance", None)
    static_encoder = effective_pair_vector_encoder(args)
    if tokenizer != "dr" and static_encoder in {"word2vec", "glove", "fasttext"}:
        info.update(
            _bundle_dr_static_embedding_artifact(
                args,
                encoder=static_encoder,
                source=getattr(args, f"{static_encoder}_path", ""),
                provenance=provenance,
            )
        )
    if provenance and "static_embedding" not in info:
        info["static_embedding"] = dict(provenance)
    return info


def _stream_artifact_metadata(args, role, values, representation):
    spec = dict((getattr(args, "streams", {}) or {}).get(role, {}) or {})
    spec["params"] = dict(spec.get("params", {}) or {})
    runtime = dict((getattr(args, "_stream_runtime", {}) or {}).get(role, {}) or {})
    if runtime.get("resolved_model_source"):
        spec["resolved_model_source"] = runtime["resolved_model_source"]
    if runtime.get("model_identity"):
        spec["model_identity"] = dict(runtime["model_identity"])
    if runtime.get("runtime_environment"):
        metadata_runtime_environment = dict(runtime["runtime_environment"])
    else:
        metadata_runtime_environment = None
    if runtime.get("resolved_params"):
        spec["params"].update(dict(runtime["resolved_params"]))
    stream_tokenizer = str(spec.get("tokenizer", "") or "").strip().lower()
    if stream_tokenizer == "dr":
        stream_encoder = str(spec.get("encoder", "") or "").strip().lower()
        max_distance = validate_dr_max_distance(
            spec["params"].get(
                "dr_max_distance",
                DR_DEFAULT_MAX_DISTANCE,
            )
        )
        max_raw_length = validate_dr_max_raw_length(
            (
                spec["params"].get(
                    "frequency_max_raw_length",
                    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
                )
                if stream_encoder in {"bow", "tfidf"}
                else spec["params"].get(
                    "dr_max_raw_length",
                    DR_DEFAULT_MAX_RAW_LENGTH,
                )
            )
        )
        unknown_policy = validate_dr_unknown_residue_policy(
            spec["params"].get(
                "dr_unknown_residue_policy",
                DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
            )
        )
        spec["params"].update(
            {
                "dr_max_distance": max_distance,
                "dr_max_raw_length": max_raw_length,
                "dr_unknown_residue_policy": unknown_policy,
                "dr_contract": dr_contract_metadata(
                    max_distance=max_distance,
                    max_raw_length=max_raw_length,
                    unknown_residue_policy=unknown_policy,
                ),
            }
        )
        bundled_static = _bundle_dr_static_embedding_artifact(
            args,
            encoder=stream_encoder,
            source=spec["params"].get(
                f"{stream_encoder}_path",
                "",
            ),
            provenance=runtime.get("static_embedding"),
            role=role,
        )
        if bundled_static:
            runtime["static_embedding"] = bundled_static.pop("static_embedding")
            spec["params"].update(bundled_static)
    elif stream_tokenizer == "top-n-gram":
        top_n = validate_top_ngram_n(spec["params"].get("top_ngram_n", 2))
        manifest_sha256 = str(runtime.get("frequency_profile_manifest_sha256", "") or "")
        spec["params"].update(
            {
                "top_ngram_n": top_n,
                "feature_dim": top_ngram_feature_dim(top_n),
                "max_length_unit": TOP_N_GRAM_MAX_LENGTH_UNIT,
                "top_ngram_contract": {
                    "contract_version": TOP_N_GRAM_CONTRACT_VERSION,
                    "profile_type": "target_frequency",
                    "tie_policy": TOP_N_GRAM_TIE_POLICY,
                    "full_length": True,
                    "fallback_profiles_allowed": (
                        str(
                            getattr(
                                args,
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
                                args,
                                "profile_on_missing",
                                TOP_N_GRAM_FALLBACK_POLICY,
                            )
                        )
                        == TOP_N_GRAM_FALLBACK_POLICY
                        else TOP_N_GRAM_STRICT_PROFILE_POLICY
                    ),
                    "feature_dim": top_ngram_feature_dim(top_n),
                    "profile_manifest_sha256": manifest_sha256,
                },
            }
        )
    elif stream_tokenizer == "dt":
        max_distance = validate_dt_max_distance(
            spec["params"].get(
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
                            args,
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
                            args,
                            "profile_on_missing",
                            TOP_N_GRAM_FALLBACK_POLICY,
                        )
                    )
                    == TOP_N_GRAM_FALLBACK_POLICY
                    else TOP_N_GRAM_STRICT_PROFILE_POLICY
                ),
                "profile_manifest_sha256": str(
                    runtime.get(
                        "frequency_profile_manifest_sha256",
                        "",
                    )
                    or ""
                ),
            }
        )
        spec["params"].update(
            {
                "dt_max_distance": max_distance,
                "feature_dim": contract["feature_dim"],
                "max_length_unit": DT_MAX_LENGTH_UNIT,
                "dt_contract": contract,
            }
        )
    elif stream_tokenizer == "bpe":
        artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
        bpe_path = str(spec["params"].get("bpe_path", "") or "").strip()
        if artifact_dir and bpe_path:
            spec["params"].update(
                bundle_bpe_artifact(
                    bpe_path,
                    artifact_dir,
                    role=role,
                )
            )
    elif stream_tokenizer == "wordpiece":
        artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
        wordpiece_path = str(spec["params"].get("wordpiece_path", "") or "").strip()
        if artifact_dir and wordpiece_path:
            spec["params"].update(
                bundle_wordpiece_artifact(
                    wordpiece_path,
                    artifact_dir,
                    role=role,
                )
            )
    elif stream_tokenizer == "unigram":
        artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
        unigram_path = str(spec["params"].get("unigram_path", "") or "").strip()
        if artifact_dir and unigram_path:
            spec["params"].update(
                bundle_unigram_artifact(
                    unigram_path,
                    artifact_dir,
                    role=role,
                )
            )
    stream_encoder = str(spec.get("encoder", "") or "").strip().lower()
    if stream_tokenizer != "dr" and stream_encoder in {"word2vec", "glove", "fasttext"}:
        bundled_static = _bundle_dr_static_embedding_artifact(
            args,
            encoder=stream_encoder,
            source=spec["params"].get(f"{stream_encoder}_path", ""),
            provenance=runtime.get("static_embedding"),
            role=role,
        )
        if bundled_static:
            runtime["static_embedding"] = bundled_static.pop("static_embedding")
            spec["params"].update(bundled_static)
    value_tensor = values.features if isinstance(values, ResidueModelInput) else values
    metadata = {
        "spec": spec,
        "representation": representation,
        "max_length": int(spec.get("max_length") or 0),
        "input_rank": int(value_tensor.ndim) if isinstance(value_tensor, torch.Tensor) else 0,
        "explicit_residue_mask": isinstance(values, ResidueModelInput),
    }
    if isinstance(value_tensor, torch.Tensor) and value_tensor.ndim >= 2:
        metadata["output_dim"] = (
            int(value_tensor.shape[-1]) if value_tensor.ndim > 2 else int(value_tensor.shape[1])
        )
        if value_tensor.ndim == 2 and representation == "token":
            metadata["vocab_size"] = (
                int(value_tensor.max().item()) + 1 if value_tensor.numel() else 1
            )
    if runtime.get("static_embedding"):
        metadata["static_embedding"] = dict(runtime["static_embedding"])
    if metadata_runtime_environment:
        metadata["runtime_environment"] = metadata_runtime_environment
    idf = runtime.get("tfidf_idf")
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    if idf is not None and artifact_dir:
        descriptor = save_idf_state(
            idf,
            artifact_dir=artifact_dir,
            filename=f"{role}_tfidf_idf.pt",
        )
        metadata["tfidf_idf"] = descriptor
        metadata["tfidf_idf_path"] = descriptor["path"]
    frequency_contract = runtime.get("frequency_encoder_contract")
    if (
        not frequency_contract
        and str(getattr(args, "stream_mode", "") or "").strip().lower() == "homogeneous"
    ):
        frequency_contract = getattr(args, "_frequency_encoder_contract", None)
    if frequency_contract:
        metadata["frequency_encoder_contract"] = dict(frequency_contract)
    if runtime.get("textrank_contract"):
        metadata["textrank_contract"] = dict(runtime["textrank_contract"])
    if runtime.get("covariance_encoder_contract"):
        metadata["covariance_encoder_contract"] = dict(runtime["covariance_encoder_contract"])
    if runtime.get("pdt_encoder_contract"):
        metadata["pdt_encoder_contract"] = dict(runtime["pdt_encoder_contract"])
    if runtime.get("pssm_covariance_encoder_contract"):
        metadata["pssm_covariance_encoder_contract"] = dict(
            runtime["pssm_covariance_encoder_contract"]
        )
    if runtime.get("profile_manifest_sha256"):
        metadata["profile_contract"] = {
            "contract_version": PROFILE_CONTRACT_VERSION,
            "manifest_sha256": runtime["profile_manifest_sha256"],
            "provenance": dict(runtime.get("profile_manifest_provenance") or {}),
        }
    return metadata


def _pair_vector_artifact_info(
    args, peptide_embeddings, protein_embeddings, fused_features, profile_rows=None
):
    tokenizer = str(getattr(args, "tokenizer", "") or "").strip().lower()
    feature_info = {}
    if tokenizer in PRETRAINED_TOKENIZERS:
        model_identity = _resolved_pair_pretrained_model_identity(args)
        feature_info["pretrained"] = {
            "tokenizer": tokenizer,
            "model_name": _resolved_pair_pretrained_model_source(args),
            "max_length": int(getattr(args, "max_length", 512)),
            "pooling": (
                str(model_identity.get("pooling"))
                if model_identity
                else (
                    ESM3_POOLING
                    if tokenizer == "esm3"
                    else (PROTBERT_POOLING if tokenizer == "protbert" else "attention_mask_mean")
                )
            ),
            "input_dim": int(peptide_embeddings.shape[1]) if peptide_embeddings.ndim > 1 else 0,
        }
        if model_identity:
            feature_info["pretrained"]["model_identity"] = model_identity
        adjustment = getattr(args, "prott5_max_length_adjustment", None)
        if tokenizer == "prott5" and isinstance(adjustment, dict):
            feature_info["pretrained"]["max_length_adjustment"] = dict(adjustment)
        runtime_environment = _resolved_pair_pretrained_runtime_environment(args)
        if tokenizer in {"esm2", "protbert", "prott5", "esmc"} and runtime_environment:
            feature_info["runtime_environment"] = runtime_environment
        if tokenizer == "esm2":
            feature_info["pretrained"].update(
                {
                    "representation_contract_version": ESM2_REPRESENTATION_CONTRACT_VERSION,
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_validate_alphabet",
                    "truncation": "max_length_residues_then_BOS_EOS",
                }
            )
        if tokenizer == "protbert":
            feature_info["pretrained"].update(
                {
                    "representation_contract_version": PROTBERT_REPRESENTATION_CONTRACT_VERSION,
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_UZOB_to_X",
                    "truncation": "max_length_residues_then_CLS_SEP",
                }
            )
        if tokenizer == "prott5":
            feature_info["pretrained"].update(
                {
                    "representation_contract_version": PROTT5_REPRESENTATION_CONTRACT_VERSION,
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_UZOB_to_X",
                    "truncation": "max_length_residues_then_EOS",
                }
            )
        if tokenizer == "esm3":
            feature_info["pretrained"].update(
                {
                    "representation_contract_version": ESM3_REPRESENTATION_CONTRACT_VERSION,
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_validate_alphabet",
                    "truncation": "max_length_residues_then_CLS_EOS",
                }
            )
        if tokenizer in {"esm2", "protbert", "prott5", "esmc", "esm3"} and runtime_environment:
            feature_info["runtime_environment"] = runtime_environment
    else:
        encoder = effective_pair_vector_encoder(args)
        feature_info["classic"] = {
            "tokenizer": tokenizer,
            "encoder": encoder,
            "max_length": int(getattr(args, "max_length", 60)),
            "k": int(getattr(args, "k", 2)),
            "dr_max_distance": int(getattr(args, "dr_max_distance", 5)),
            "top_ngram_n": (1 if tokenizer == "dt" else int(getattr(args, "top_ngram_n", 2))),
            "dt_max_distance": int(getattr(args, "dt_max_distance", DT_DEFAULT_MAX_DISTANCE)),
            "bpe_path": str(getattr(args, "bpe_path", "") or ""),
            "unigram_path": str(getattr(args, "unigram_path", "") or ""),
            "unigram_vocab_size": int(getattr(args, "unigram_vocab_size", 1000)),
            "unigram_max_raw_length": int(
                getattr(
                    args,
                    "unigram_max_raw_length",
                    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            "wordpiece_path": str(getattr(args, "wordpiece_path", "") or ""),
            "wordpiece_vocab_size": int(getattr(args, "wordpiece_vocab_size", 1000)),
            "wordpiece_max_raw_length": int(
                getattr(
                    args,
                    "wordpiece_max_raw_length",
                    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            "word2vec_path": str(getattr(args, "word2vec_path", "") or ""),
            "glove_path": str(getattr(args, "glove_path", "") or ""),
            "fasttext_path": str(getattr(args, "fasttext_path", "") or ""),
            "pssm_index_path": str(getattr(args, "pssm_index_path", "") or ""),
            "ac_pssm_blosum62_hybrid_index_path": str(
                getattr(args, "ac_pssm_blosum62_hybrid_index_path", "") or ""
            ),
            "cc_pssm_blosum62_hybrid_index_path": str(
                getattr(args, "cc_pssm_blosum62_hybrid_index_path", "") or ""
            ),
            "acc_pssm_blosum62_hybrid_index_path": str(
                getattr(args, "acc_pssm_blosum62_hybrid_index_path", "") or ""
            ),
            "psfm_index_path": str(getattr(args, "psfm_index_path", "") or ""),
            "ac_lag": int(getattr(args, "ac_lag", 10)),
            "cc_lag": int(getattr(args, "cc_lag", 10)),
            "acc_lag": int(getattr(args, "acc_lag", 10)),
            "covariance_max_raw_length": validate_covariance_max_raw_length(
                getattr(
                    args,
                    "covariance_max_raw_length",
                    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            "pdt_lag": int(getattr(args, "pdt_lag", PDT_DEFAULT_LAG)),
            "pdt_max_raw_length": validate_pdt_max_raw_length(
                getattr(
                    args,
                    "pdt_max_raw_length",
                    PDT_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            "ac_pssm_lag": int(getattr(args, "ac_pssm_lag", 10)),
            "cc_pssm_lag": int(getattr(args, "cc_pssm_lag", 10)),
            "acc_pssm_lag": int(getattr(args, "acc_pssm_lag", 10)),
            "ac_pssm_blosum62_hybrid_lag": int(getattr(args, "ac_pssm_blosum62_hybrid_lag", 10)),
            "cc_pssm_blosum62_hybrid_lag": int(getattr(args, "cc_pssm_blosum62_hybrid_lag", 10)),
            "acc_pssm_blosum62_hybrid_lag": int(getattr(args, "acc_pssm_blosum62_hybrid_lag", 10)),
            "input_dim": int(peptide_embeddings.shape[1]) if peptide_embeddings.ndim > 1 else 0,
        }
        if tokenizer == "kmer":
            feature_info["classic"]["max_length_unit"] = KMER_MAX_LENGTH_UNIT
        elif tokenizer == "wordpiece":
            feature_info["classic"]["max_length_unit"] = WORDPIECE_MAX_LENGTH_UNIT
        elif tokenizer == "unigram":
            feature_info["classic"]["max_length_unit"] = UNIGRAM_MAX_LENGTH_UNIT
            feature_info["classic"]["training_regularization"] = _unigram_regularization_info(args)
        feature_info["classic"].update(_classic_extra_artifact_info(args))
        if encoder == "tfidf" and str(getattr(args, "stream_mode", "homogeneous")) == "homogeneous":
            feature_info["classic"]["tfidf_fit_scope"] = "combined_pair_roles"
    feature_info.update(
        _pair_profile_generation_artifact_info(
            args,
            profile_rows=profile_rows,
        )
    )
    pair_info = {
        "peptide_dim": int(peptide_embeddings.shape[1]) if peptide_embeddings.ndim > 1 else 0,
        "protein_dim": int(protein_embeddings.shape[1]) if protein_embeddings.ndim > 1 else 0,
    }
    if fused_features is None:
        pair_info["input_adapter"] = "typed_role_projection"
    else:
        pair_info.update(
            {
                "fusion_mode": resolve_fusion_mode(getattr(args, "fusion_mode", "concat_all")),
                "fused_dim": int(fused_features.shape[1]) if fused_features.ndim > 1 else 0,
            }
        )
    result = {
        "input_schema": _pair_input_schema(args),
        "label_names": ["interaction"],
        **feature_info,
        "pair": pair_info,
    }
    if result["pair"].get("fusion_mode") in AUDITED_PAIR_FUSION_MODES:
        result["pair_fusion_contract"] = pair_fusion_contract(
            result["pair"]["fusion_mode"],
            peptide_embeddings,
            protein_embeddings,
            fused_features,
        )
    if getattr(args, "streams", None):
        result["schema_version"] = 2
        result["streams"] = {
            "peptide": _stream_artifact_metadata(args, "peptide", peptide_embeddings, "vector"),
            "partner": _stream_artifact_metadata(args, "partner", protein_embeddings, "vector"),
        }
        result["pair"]["pair_strategy"] = "two_stream"
        result["pair"]["stream_mode"] = str(getattr(args, "stream_mode", "homogeneous"))
        result["pair"]["interaction_level"] = "vector"
        result["pair"]["classifier"] = str(
            getattr(args, "classifier", "") or getattr(args, "model", "")
        )
        if str(getattr(args, "stream_mode", "")) == "heterogeneous":
            result.pop("pretrained", None)
            result.pop("classic", None)
    if str(getattr(args, "pair_strategy", "") or "").strip().lower() == "two_stream":
        predictor_type = str(getattr(args, "predictor_type", "") or "").strip().lower()
        graph_predictor = (
            str(getattr(args, "graph_predictor", "") or getattr(args, "model", "") or "")
            .strip()
            .lower()
        )
        if graph_predictor == HGT_MODEL:
            predictor_type = "sample_graph"
        if not predictor_type:
            predictor_type = "classifier"
        result["pair"].update(
            {
                "pair_strategy": "two_stream",
                "stream_mode": str(getattr(args, "stream_mode", "homogeneous") or "homogeneous")
                .strip()
                .lower(),
                "interaction_level": "vector",
            }
        )
        try:
            result["pair_strategy_contract"] = build_two_stream_strategy_contract(
                args,
                peptide_embeddings,
                protein_embeddings,
                predictor_type=predictor_type,
                input_adapter=(
                    "typed_role_projection"
                    if graph_predictor == HGT_MODEL
                    else "two_stream_vector_fusion"
                ),
            )
        except PairStrategyContractError as exc:
            raise ValueError(str(exc)) from exc
    if result["pair"].get("fusion_mode") == "concat" and str(
        getattr(args, "classifier", "") or getattr(args, "model", "")
    ).strip().lower() in {"svm", "rf", "extratrees", "histgb", "xgb", "lgb", "logreg"}:
        try:
            role_contracts = result.get("streams") or {
                "peptide": result["classic"],
                "partner": result["classic"],
            }
            ordered_digest = (
                pair_bow_contract_digest(list(profile_rows)) if profile_rows is not None else None
            )
            result["pair_bow_joint_contract"] = build_pair_bow_joint_contract(
                role_contracts["peptide"],
                role_contracts["partner"],
                train_sample_count=int(peptide_embeddings.shape[0]),
                ordered_pair_train_sha256=ordered_digest,
                fused_feature_dimension=int(fused_features.shape[1]),
            )
        except (PairBoWContractError, KeyError, TypeError, ValueError):
            # Ordinary vector artifacts remain valid; topic reducers fail
            # closed when this strict contract is absent.
            pass
    return result


def save_heterogeneous_pair_token_artifact(
    args,
    model,
    peptide_tokens,
    partner_tokens,
    model_config,
    threshold=0.5,
    profile_rows=None,
):
    del profile_rows
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    if not artifact_dir:
        return
    os.makedirs(artifact_dir, exist_ok=True)
    torch.save(
        {
            "model_class": type(model).__name__,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        os.path.join(artifact_dir, "model.pt"),
    )
    metadata = {
        "schema_version": 2,
        "bundle_type": "pair_token_nn",
        "supports_inference": True,
        "task_type": normalize_task_type(getattr(args, "task_type", "pep_protein_interaction")),
        "args": _jsonable_args(args),
        "threshold": float(threshold),
        "threshold_source": "validation_f1",
        **_pair_protocol_metadata(args, model=model),
        "input_schema": _pair_input_schema(args),
        "label_names": ["interaction"],
        "streams": {
            "peptide": _stream_artifact_metadata(args, "peptide", peptide_tokens, "token"),
            "partner": _stream_artifact_metadata(args, "partner", partner_tokens, "token"),
        },
        "pair": {
            "pair_strategy": "two_stream",
            "stream_mode": str(getattr(args, "stream_mode", "heterogeneous")),
            "interaction_level": "token",
            "interaction_model": str(getattr(args, "interaction_model", "cross_attn")),
            "classifier": str(getattr(args, "classifier", "mlp") or "mlp"),
            "peptide_seq_len": int(peptide_tokens.shape[1]),
            "partner_seq_len": int(partner_tokens.shape[1]),
            "peptide_token_dim": int(peptide_tokens.shape[2]) if peptide_tokens.ndim > 2 else 0,
            "partner_token_dim": int(partner_tokens.shape[2]) if partner_tokens.ndim > 2 else 0,
            "model_config": model_config,
        },
    }
    if str(getattr(args, "pair_strategy", "") or "").strip().lower() == "two_stream":
        try:
            metadata["pair_strategy_contract"] = build_two_stream_strategy_contract(
                args,
                peptide_tokens,
                partner_tokens,
                predictor_type="classifier",
                input_adapter="two_stream_token_interaction",
            )
        except PairStrategyContractError as exc:
            raise ValueError(str(exc)) from exc
    predictor_contract = pair_attention_predictor_contract(model)
    if predictor_contract is not None:
        metadata["predictor_contract"] = predictor_contract
    head_config = prediction_head_artifact_config(args)
    if head_config is not None:
        metadata["prediction_head"] = head_config
    with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[artifact] saved reusable heterogeneous pair-token bundle: {artifact_dir}")


def configure_pair_vector_artifacts(
    args, peptide_embeddings, protein_embeddings, fused_features, profile_rows=None
):
    if not _pair_vector_artifacts_supported(args):
        return None
    artifact_info = _pair_vector_artifact_info(
        args,
        peptide_embeddings,
        protein_embeddings,
        fused_features,
        profile_rows=profile_rows,
    )
    args._artifact_source = PAIR_VECTOR_ARTIFACT_SOURCE
    args._artifact_info = artifact_info
    return artifact_info


def _pair_graph_predictor_contract(model, hidden_dim):
    if model == GCN_MODEL:
        return {
            "schema_version": 1,
            "implementation": "pepplatform_inductive_pair_sample_graph_gcn",
            "contract_version": 2,
            "status": "verified_platform_variant",
            "input_adapter": "two_stream_vector_fusion",
            "node_definition": "one node per fused peptide-partner sample",
            "edges": "training reference kNN graph plus directed reference-to-query edges and query self-loops",
            "aggregation": "symmetric degree-normalized reference adjacency and row-normalized inductive query messages",
            "layers": 2,
            "hidden_dim": int(hidden_dim),
            "readout": "node_level",
            "prediction_output": "logits_then_sigmoid_probability",
        }
    if model == GAT_MODEL:
        return {
            "schema_version": 1,
            "implementation": "pepplatform_inductive_pair_sample_graph_gat",
            "contract_version": GAT_PREDICTOR_CONTRACT_VERSION,
            "status": "verified_platform_variant",
            "input_adapter": "two_stream_vector_fusion",
            "node_definition": "one node per fused peptide-partner sample",
            "edges": "exactly k unique incoming training-reference neighbors per reference node, one self-loop per node, and directed reference-to-query edges",
            "topology_contract": "gat_unique_incoming_knn_v2",
            "attention": "single-head additive neighbor attention with destination-wise softmax",
            "aggregation": "memory-bounded exact attention-weighted neighbor sum",
            "layers": 2,
            "hidden_dim": int(hidden_dim),
            "readout": "node_level",
            "prediction_output": "logits_then_sigmoid_probability",
        }
    if model == HGT_MODEL:
        return {
            "schema_version": 1,
            "implementation": "pepplatform_inductive_typed_pair_hgt",
            "contract_version": HGT_PREDICTOR_CONTRACT_VERSION,
            "status": "verified_platform_variant",
            "input_adapter": "separate_train_only_scaled_peptide_and_partner_streams",
            "node_definition": "one peptide node and one partner node per pair row with explicit role types",
            "edges": "role-specific semantic kNN and self relations; query nodes receive reference messages only",
            "topology_contract": HGT_GRAPH_SCHEMA,
            "attention": "node-type-specific Q/K/V projections with relation-specific attention and message transforms",
            "aggregation": "destination-wise multi-head softmax over typed incoming relations",
            "target_relation": "ppi edge decoded without inserting target labels into the message graph",
            "layers": 2,
            "heads": 4,
            "hidden_dim": int(hidden_dim),
            "readout": "typed peptide-partner edge representation",
            "prediction_output": "edge_logits_then_sigmoid_probability",
        }
    if model == SAGE_MODEL:
        return {
            "schema_version": 1,
            "implementation": "pepplatform_inductive_pair_sample_graph_sage_mean",
            "contract_version": SAGE_PREDICTOR_CONTRACT_VERSION,
            "status": "verified_platform_variant",
            "input_adapter": "two_stream_vector_fusion",
            "node_definition": "one node per fused peptide-partner sample",
            "edges": "exactly k unique incoming non-self training-reference neighbors and directed reference-to-query edges",
            "topology_contract": "sage_unique_incoming_nonself_knn_v1",
            "root_update": "independent trainable root projection",
            "aggregation": "mean neighbor aggregation followed by independent trainable neighbor projection",
            "layers": 2,
            "hidden_dim": int(hidden_dim),
            "readout": "node_level",
            "prediction_output": "logits_then_sigmoid_probability",
        }
    if model == GT_MODEL:
        return {
            "schema_version": 1,
            "implementation": "pepplatform_inductive_pair_sample_graph_graph_transformer",
            "contract_version": GRAPH_TRANSFORMER_PREDICTOR_CONTRACT_VERSION,
            "status": "verified_platform_variant",
            "input_adapter": "two_stream_vector_fusion",
            "node_definition": "one node per fused peptide-partner sample",
            "edges": "exactly k unique incoming training-reference neighbors per reference node, one self-loop per node, and directed reference-to-query edges",
            "topology_contract": "graph_transformer_unique_incoming_knn_v1",
            "attention": "graph-edge-masked multi-head scaled dot-product attention with exact destination-wise softmax",
            "transformer_block": "post-norm attention residual with LayerNorm followed by position-wise GELU feed-forward residual with LayerNorm",
            "layers": 2,
            "heads": 4,
            "hidden_dim": int(hidden_dim),
            "readout": "node_level",
            "prediction_output": "logits_then_sigmoid_probability",
        }
    return None


def _pair_graph_artifact_info(
    args, peptide_embeddings, protein_embeddings, graph_features, profile_rows=None
):
    info = _pair_vector_artifact_info(
        args,
        peptide_embeddings,
        protein_embeddings,
        graph_features,
        profile_rows=profile_rows,
    )
    model = (
        str(getattr(args, "graph_predictor", "") or getattr(args, "model", "gcn")).strip().lower()
    )
    info["graph"] = {
        "predictor": model,
        "gnn_k": int(getattr(args, "gnn_k", 8)),
        "hidden_dim": int(getattr(args, "gnn_hidden_dim", 256)),
        "exact_knn_max_nodes": int(getattr(args, "gnn_exact_knn_max_nodes", 25000)),
        "rp_projections": int(getattr(args, "gnn_rp_projections", 6)),
        "rp_window": int(getattr(args, "gnn_rp_window", 0)),
        "dropout": float(getattr(args, "dropout", 0.2)),
        "reference_source": "train_only",
        "reference_path": "graph_reference.pt",
        "inference_graph": "directed_train_reference_to_query",
        "evaluation_mode": "inductive",
        "aligned_gated": resolve_fusion_mode(getattr(args, "fusion_mode", "concat_all"))
        == "aligned_gated",
    }
    if graph_features is not None:
        info["graph"]["input_dim"] = int(graph_features.shape[1]) if graph_features.ndim > 1 else 0
    predictor_contract = _pair_graph_predictor_contract(
        model,
        getattr(args, "gnn_hidden_dim", 256),
    )
    if predictor_contract is not None:
        info["predictor_contract"] = predictor_contract
    if model == GAT_MODEL:
        info["graph"].update(
            {
                "aggregation": "memory_bounded_exact_softmax",
                "topology_contract": "gat_unique_incoming_knn_v2",
                "reference_neighbors": "exactly_k_unique_incoming_plus_one_self_loop",
                "query_neighbors": "reference_to_query_only_plus_one_self_loop",
                "edge_chunk_size": int(getattr(args, "gnn_edge_chunk_size", GAT_EDGE_CHUNK_SIZE)),
            }
        )
    if model == SAGE_MODEL:
        info["graph"].update(
            {
                "topology_contract": "sage_unique_incoming_nonself_knn_v1",
                "reference_neighbors": "exactly_k_unique_incoming_nonself",
                "query_neighbors": "reference_to_query_only",
                "self_loop_policy": "no_edge_self_loops_root_update_only",
            }
        )
    if model == GT_MODEL:
        info["graph"].update(
            {
                "aggregation": "exact_destination_softmax_multi_head_attention",
                "topology_contract": "graph_transformer_unique_incoming_knn_v1",
                "reference_neighbors": "exactly_k_unique_incoming_plus_one_self_loop",
                "query_neighbors": "reference_to_query_only_plus_one_self_loop",
                "self_loop_policy": "exactly_one_per_node",
                "heads": 4,
                "layers": 2,
            }
        )
    if model == HGT_MODEL:
        info["graph"].update(
            {
                "schema": HGT_GRAPH_SCHEMA,
                "topology_contract": HGT_GRAPH_SCHEMA,
                "node_types": ["peptide", "partner"],
                "message_relations": [
                    "peptide_semantic",
                    "partner_semantic",
                    "peptide_self",
                    "partner_self",
                ],
                "target_relation": "ppi_decoder_only",
                "peptide_input_dim": int(peptide_embeddings.shape[1]),
                "partner_input_dim": int(protein_embeddings.shape[1]),
                "heads": 4,
                "layers": 2,
                "edge_chunk_size": int(getattr(args, "gnn_edge_chunk_size", GAT_EDGE_CHUNK_SIZE)),
                "reference_neighbors": "exactly_k_unique_incoming_same_role_plus_one_self_loop",
                "query_neighbors": "same_role_reference_to_query_only_plus_one_self_loop",
                "label_edges_in_message_graph": False,
                "aligned_gated": False,
            }
        )
    return info


def configure_pair_graph_artifacts(
    args, peptide_embeddings, protein_embeddings, graph_features, profile_rows=None
):
    if not _pair_graph_artifacts_supported(args):
        return None
    artifact_info = _pair_graph_artifact_info(
        args,
        peptide_embeddings,
        protein_embeddings,
        graph_features,
        profile_rows=profile_rows,
    )
    args._graph_artifact_info = artifact_info
    return artifact_info


def _pair_token_artifact_info(args, model, train_x, profile_rows=None, partner_train_x=None):
    tokenizer = str(getattr(args, "tokenizer", "") or "").strip().lower()
    tensor_input = isinstance(train_x, torch.Tensor) or _is_explicit_residue_input(train_x)
    train_tensor = train_x.features if isinstance(train_x, ResidueModelInput) else train_x
    token_dim = int(train_tensor.shape[2]) if tensor_input and train_tensor.ndim > 2 else 0
    partner_tensor = None
    if partner_train_x is not None:
        partner_tensor = (
            partner_train_x.features
            if isinstance(partner_train_x, ResidueModelInput)
            else partner_train_x
        )
        if (
            not isinstance(partner_train_x, (torch.Tensor, ResidueModelInput))
            or partner_tensor.ndim != train_tensor.ndim
        ):
            if str(getattr(args, "stream_mode", "homogeneous")).strip().lower() != "heterogeneous":
                raise ValueError("homogeneous pair-token streams must have the same tensor rank")
        if (
            str(getattr(args, "stream_mode", "homogeneous")).strip().lower() != "heterogeneous"
            and train_tensor.ndim > 2
            and int(partner_tensor.shape[2]) != token_dim
        ):
            raise ValueError("homogeneous pair-token streams must have the same token dimension")
    vocab_size = int(getattr(getattr(model, "embed", None), "num_embeddings", 0) or 0)
    hidden_dim = int(
        getattr(model, "hidden_dim", 0)
        or getattr(getattr(model, "attn", None), "embed_dim", 128)
        or 128
    )
    n_heads = int(
        getattr(model, "n_heads", 0) or getattr(getattr(model, "attn", None), "num_heads", 4) or 4
    )
    max_length = int(getattr(args, "max_length", 512))
    interaction_model = str(getattr(args, "interaction_model", "") or "").strip().lower()
    if interaction_model not in {"cross_attn", "bi_cross_attn"}:
        interaction_model = (
            "bi_cross_attn"
            if str(getattr(args, "model", "") or "").strip().lower() == BI_CROSS_ATTN_MODEL
            else "cross_attn"
        )
    encoder_name = str(getattr(args, "encoder", "ppi_fusion") or "ppi_fusion").strip().lower()
    if tokenizer not in PRETRAINED_TOKENIZERS:
        encoder_name = effective_pair_vector_encoder(args)
    info = {
        "input_schema": _pair_input_schema(args),
        "label_names": ["interaction"],
        "pair": {
            "interaction_level": "token",
            "interaction_model": interaction_model,
            "encoder": encoder_name,
            "seq_len": max_length,
            "token_dim": token_dim,
            "partner_seq_len": int(partner_tensor.shape[1])
            if partner_tensor is not None and partner_tensor.ndim > 1
            else 0,
            "partner_token_dim": int(partner_tensor.shape[2])
            if partner_tensor is not None and partner_tensor.ndim > 2
            else 0,
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "n_heads": n_heads,
            "dropout": float(getattr(args, "dropout", 0.1)),
        },
    }
    predictor_contract = pair_attention_predictor_contract(model)
    if predictor_contract is not None:
        info["predictor_contract"] = predictor_contract
    head_config = prediction_head_artifact_config(args)
    if head_config is not None:
        info["prediction_head"] = head_config
    if str(getattr(args, "pair_strategy", "") or "").strip().lower() == "two_stream":
        predictor_type = (
            str(getattr(args, "predictor_type", "classifier") or "classifier").strip().lower()
        )
        info["pair"].update(
            {
                "pair_strategy": "two_stream",
                "stream_mode": str(getattr(args, "stream_mode", "homogeneous") or "homogeneous")
                .strip()
                .lower(),
            }
        )
        try:
            info["pair_strategy_contract"] = build_two_stream_strategy_contract(
                args,
                train_x,
                partner_train_x if partner_train_x is not None else train_x,
                predictor_type=predictor_type,
                input_adapter="two_stream_token_interaction",
            )
        except PairStrategyContractError as exc:
            raise ValueError(str(exc)) from exc
    if tokenizer in PRETRAINED_TOKENIZERS:
        model_identity = _resolved_pair_pretrained_model_identity(args)
        info["pretrained"] = {
            "tokenizer": tokenizer,
            "model_name": _resolved_pair_pretrained_model_source(args),
            "max_length": max_length,
            "representation": "residue_token_hidden"
            if tokenizer in {"esm2", "protbert", "prott5", "esmc", "esm3"}
            else "token_hidden",
            "token_dim": token_dim,
        }
        if model_identity:
            info["pretrained"]["model_identity"] = model_identity
        adjustment = getattr(args, "prott5_max_length_adjustment", None)
        if tokenizer == "prott5" and isinstance(adjustment, dict):
            info["pretrained"]["max_length_adjustment"] = dict(adjustment)
        runtime_environment = _resolved_pair_pretrained_runtime_environment(args)
        if tokenizer in {"esm2", "protbert", "prott5", "esmc", "esm3"} and runtime_environment:
            info["runtime_environment"] = runtime_environment
        if tokenizer == "esm2":
            info["pretrained"].update(
                {
                    "representation_contract_version": ESM2_REPRESENTATION_CONTRACT_VERSION,
                    "pooling": "residue_tokens",
                    "explicit_residue_mask": _is_explicit_residue_input(train_x),
                    "special_tokens": "excluded",
                }
            )
        if tokenizer == "protbert":
            info["pretrained"].update(
                {
                    "representation_contract_version": PROTBERT_REPRESENTATION_CONTRACT_VERSION,
                    "pooling": "residue_tokens",
                    "explicit_residue_mask": _is_explicit_residue_input(train_x),
                    "special_tokens": "excluded",
                }
            )
        if tokenizer == "prott5":
            info["pretrained"].update(
                {
                    "representation_contract_version": PROTT5_REPRESENTATION_CONTRACT_VERSION,
                    "pooling": "residue_tokens",
                    "explicit_residue_mask": _is_explicit_residue_input(train_x),
                    "special_tokens": "excluded",
                }
            )
        if tokenizer == "esmc":
            info["pretrained"].update(
                {
                    "pooling": "residue_tokens",
                    "explicit_residue_mask": _is_explicit_residue_input(train_x),
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_validate_alphabet",
                }
            )
        if tokenizer == "esm3":
            info["pretrained"].update(
                {
                    "representation_contract_version": ESM3_REPRESENTATION_CONTRACT_VERSION,
                    "pooling": "residue_tokens",
                    "explicit_residue_mask": _is_explicit_residue_input(train_x),
                    "special_tokens": "excluded",
                    "input_normalization": "strip_whitespace_uppercase_validate_alphabet",
                }
            )
        if (
            str(getattr(args, "stream_mode", "homogeneous") or "").strip().lower() == "heterogeneous"
            and partner_train_x is not None
        ):
            info["schema_version"] = 2
            info["streams"] = {
                "peptide": _stream_artifact_metadata(args, "peptide", train_x, "token"),
                "partner": _stream_artifact_metadata(args, "partner", partner_train_x, "token"),
            }
            info.pop("pretrained", None)
        return info

    info["classic"] = {
        "tokenizer": tokenizer,
        "encoder": encoder_name,
        "max_length": max_length,
        "k": int(getattr(args, "k", 2)),
        "dr_max_distance": int(getattr(args, "dr_max_distance", 5)),
        "top_ngram_n": (1 if tokenizer == "dt" else int(getattr(args, "top_ngram_n", 2))),
        "dt_max_distance": int(getattr(args, "dt_max_distance", DT_DEFAULT_MAX_DISTANCE)),
        "bpe_path": str(getattr(args, "bpe_path", "") or ""),
        "unigram_path": str(getattr(args, "unigram_path", "") or ""),
        "unigram_vocab_size": int(getattr(args, "unigram_vocab_size", 1000)),
        "unigram_max_raw_length": int(
            getattr(
                args,
                "unigram_max_raw_length",
                UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
        "wordpiece_path": str(getattr(args, "wordpiece_path", "") or ""),
        "wordpiece_vocab_size": int(getattr(args, "wordpiece_vocab_size", 1000)),
        "wordpiece_max_raw_length": int(
            getattr(
                args,
                "wordpiece_max_raw_length",
                WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
        "word2vec_path": str(getattr(args, "word2vec_path", "") or ""),
        "glove_path": str(getattr(args, "glove_path", "") or ""),
        "fasttext_path": str(getattr(args, "fasttext_path", "") or ""),
        "pssm_index_path": str(getattr(args, "pssm_index_path", "") or ""),
        "ac_pssm_blosum62_hybrid_index_path": str(
            getattr(args, "ac_pssm_blosum62_hybrid_index_path", "") or ""
        ),
        "cc_pssm_blosum62_hybrid_index_path": str(
            getattr(args, "cc_pssm_blosum62_hybrid_index_path", "") or ""
        ),
        "acc_pssm_blosum62_hybrid_index_path": str(
            getattr(args, "acc_pssm_blosum62_hybrid_index_path", "") or ""
        ),
        "psfm_index_path": str(getattr(args, "psfm_index_path", "") or ""),
        "ac_lag": int(getattr(args, "ac_lag", 10)),
        "cc_lag": int(getattr(args, "cc_lag", 10)),
        "acc_lag": int(getattr(args, "acc_lag", 10)),
        "covariance_max_raw_length": validate_covariance_max_raw_length(
            getattr(
                args,
                "covariance_max_raw_length",
                COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
        "pdt_lag": int(getattr(args, "pdt_lag", PDT_DEFAULT_LAG)),
        "pdt_max_raw_length": validate_pdt_max_raw_length(
            getattr(
                args,
                "pdt_max_raw_length",
                PDT_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
        "ac_pssm_lag": int(getattr(args, "ac_pssm_lag", 10)),
        "cc_pssm_lag": int(getattr(args, "cc_pssm_lag", 10)),
        "acc_pssm_lag": int(getattr(args, "acc_pssm_lag", 10)),
        "ac_pssm_blosum62_hybrid_lag": int(getattr(args, "ac_pssm_blosum62_hybrid_lag", 10)),
        "cc_pssm_blosum62_hybrid_lag": int(getattr(args, "cc_pssm_blosum62_hybrid_lag", 10)),
        "acc_pssm_blosum62_hybrid_lag": int(getattr(args, "acc_pssm_blosum62_hybrid_lag", 10)),
        "vocab_size": vocab_size,
        "token_dim": token_dim,
        "input_rank": int(train_x.ndim) if tensor_input else 0,
        "explicit_residue_mask": isinstance(train_x, ResidueModelInput),
    }
    if tokenizer == "wordpiece":
        info["classic"]["max_length_unit"] = WORDPIECE_MAX_LENGTH_UNIT
    elif tokenizer == "unigram":
        info["classic"]["max_length_unit"] = UNIGRAM_MAX_LENGTH_UNIT
        info["classic"]["training_regularization"] = _unigram_regularization_info(args)
    info["classic"].update(_classic_extra_artifact_info(args))
    if (
        str(getattr(args, "stream_mode", "homogeneous") or "").strip().lower() == "heterogeneous"
        and partner_train_x is not None
    ):
        info["schema_version"] = 2
        info["streams"] = {
            "peptide": _stream_artifact_metadata(args, "peptide", train_x, "token"),
            "partner": _stream_artifact_metadata(args, "partner", partner_train_x, "token"),
        }
        info.pop("classic", None)
    info.update(_pair_profile_generation_artifact_info(args, profile_rows=profile_rows))
    return info


def save_pair_token_nn_artifact(
    args,
    model,
    train_x,
    threshold=0.5,
    profile_rows=None,
    partner_train_x=None,
):
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    if not artifact_dir or not _pair_token_artifacts_supported(args):
        return
    threshold_value = _frozen_pair_threshold(args)
    os.makedirs(artifact_dir, exist_ok=True)
    torch.save(
        {
            "model_class": type(model).__name__,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        os.path.join(artifact_dir, "model.pt"),
    )
    metadata = {
        "bundle_type": "pair_token_nn",
        "supports_inference": True,
        "task_type": normalize_task_type(getattr(args, "task_type", "pep_protein_interaction")),
        "args": _jsonable_args(args),
        "threshold": threshold_value,
        "threshold_source": "validation_f1",
        **_pair_protocol_metadata(args, model=model),
        **_pair_token_artifact_info(
            args,
            model,
            train_x,
            profile_rows=profile_rows,
            partner_train_x=partner_train_x,
        ),
    }
    with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[artifact] saved reusable PPI pair-token neural bundle: {artifact_dir}")


def _save_full_model_cpu(model, path):
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")
    model_cpu = model.to("cpu")
    torch.save(model_cpu, path)
    model.to(original_device)


def _pair_sequence_artifact_info(args, model, train_x, profile_rows=None):
    encoder_name = effective_pair_vector_encoder(args)
    tokenizer = str(getattr(args, "tokenizer", "") or "").strip().lower()
    if tokenizer in PRETRAINED_TOKENIZERS:
        strategy_contract = getattr(args, "_single_stream_strategy_contract", None)
        if not isinstance(strategy_contract, dict):
            raise ValueError("pretrained Single-stream artifact is missing its strategy contract")
        join_contract = strategy_contract.get("sequence_join") or {}
        info = _pair_token_artifact_info(
            args,
            model,
            train_x,
            profile_rows=profile_rows,
        )
        info["pair"] = {
            "pair_strategy": "single_stream",
            "interaction_level": "joint_sequence",
            "sequence_model": str(getattr(args, "sequence_model", "") or getattr(args, "model", ""))
            .strip()
            .lower(),
            "encoder": tokenizer,
            "seq_len": int(getattr(args, "max_length", 60)),
            "joined_seq_len": int(join_contract.get("joined_length") or 0),
            "residue_adapter_dim": int(
                getattr(args, "residue_adapter_dim", RESIDUE_ADAPTER_DIM_DEFAULT)
            ),
        }
        info["pair_strategy_contract"] = strategy_contract
        if isinstance(info.get("pretrained"), dict):
            info["pretrained"]["token_dim"] = int(join_contract.get("source_feature_dim") or 0)
            info["pretrained"]["cache_layout"] = (
                "unique_sequence_ragged_v1"
                if bool(getattr(args, "compact_residue_cache", False))
                else "dense_explicit_mask_v1"
            )
        return info
    model_vocab_size = int(getattr(getattr(model, "embedding", None), "num_embeddings", 0) or 0)
    strategy_contract = getattr(args, "_single_stream_strategy_contract", None)
    join_contract = (
        strategy_contract.get("sequence_join", {}) if isinstance(strategy_contract, dict) else {}
    )
    vocab_size = (
        int(join_contract.get("index_separator_id") or 0)
        if join_contract.get("input_kind") == "index"
        else model_vocab_size
    )
    tensor_input = isinstance(train_x, (torch.Tensor, ResidueModelInput))
    value_tensor = train_x.features if isinstance(train_x, ResidueModelInput) else train_x
    token_dim = int(value_tensor.shape[2]) if tensor_input and value_tensor.ndim > 2 else 0
    info = {
        "input_schema": _pair_input_schema(args),
        "label_names": ["interaction"],
        "classic": {
            "tokenizer": tokenizer,
            "encoder": encoder_name,
            "max_length": int(getattr(args, "max_length", 60)),
            "k": int(getattr(args, "k", 2)),
            "dr_max_distance": int(getattr(args, "dr_max_distance", 5)),
            "top_ngram_n": (1 if tokenizer == "dt" else int(getattr(args, "top_ngram_n", 2))),
            "dt_max_distance": int(getattr(args, "dt_max_distance", DT_DEFAULT_MAX_DISTANCE)),
            "bpe_path": str(getattr(args, "bpe_path", "") or ""),
            "unigram_path": str(getattr(args, "unigram_path", "") or ""),
            "unigram_vocab_size": int(getattr(args, "unigram_vocab_size", 1000)),
            "unigram_max_raw_length": int(
                getattr(
                    args,
                    "unigram_max_raw_length",
                    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            "wordpiece_path": str(getattr(args, "wordpiece_path", "") or ""),
            "wordpiece_vocab_size": int(getattr(args, "wordpiece_vocab_size", 1000)),
            "wordpiece_max_raw_length": int(
                getattr(
                    args,
                    "wordpiece_max_raw_length",
                    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            "word2vec_path": str(getattr(args, "word2vec_path", "") or ""),
            "glove_path": str(getattr(args, "glove_path", "") or ""),
            "fasttext_path": str(getattr(args, "fasttext_path", "") or ""),
            "pssm_index_path": str(getattr(args, "pssm_index_path", "") or ""),
            "ac_pssm_blosum62_hybrid_index_path": str(
                getattr(args, "ac_pssm_blosum62_hybrid_index_path", "") or ""
            ),
            "cc_pssm_blosum62_hybrid_index_path": str(
                getattr(args, "cc_pssm_blosum62_hybrid_index_path", "") or ""
            ),
            "acc_pssm_blosum62_hybrid_index_path": str(
                getattr(args, "acc_pssm_blosum62_hybrid_index_path", "") or ""
            ),
            "psfm_index_path": str(getattr(args, "psfm_index_path", "") or ""),
            "ac_lag": int(getattr(args, "ac_lag", 10)),
            "cc_lag": int(getattr(args, "cc_lag", 10)),
            "acc_lag": int(getattr(args, "acc_lag", 10)),
            "covariance_max_raw_length": validate_covariance_max_raw_length(
                getattr(
                    args,
                    "covariance_max_raw_length",
                    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            "pdt_lag": int(getattr(args, "pdt_lag", PDT_DEFAULT_LAG)),
            "pdt_max_raw_length": validate_pdt_max_raw_length(
                getattr(
                    args,
                    "pdt_max_raw_length",
                    PDT_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
            "ac_pssm_lag": int(getattr(args, "ac_pssm_lag", 10)),
            "cc_pssm_lag": int(getattr(args, "cc_pssm_lag", 10)),
            "acc_pssm_lag": int(getattr(args, "acc_pssm_lag", 10)),
            "ac_pssm_blosum62_hybrid_lag": int(getattr(args, "ac_pssm_blosum62_hybrid_lag", 10)),
            "cc_pssm_blosum62_hybrid_lag": int(getattr(args, "cc_pssm_blosum62_hybrid_lag", 10)),
            "acc_pssm_blosum62_hybrid_lag": int(getattr(args, "acc_pssm_blosum62_hybrid_lag", 10)),
            "vocab_size": vocab_size,
            "model_vocab_size": model_vocab_size,
            "token_dim": token_dim,
            "input_rank": int(value_tensor.ndim) if tensor_input else 0,
            "explicit_residue_mask": isinstance(train_x, ResidueModelInput),
        },
        "pair": {
            "pair_strategy": "single_stream",
            "sequence_model": str(getattr(args, "sequence_model", "") or getattr(args, "model", ""))
            .strip()
            .lower(),
            "seq_len": int(getattr(args, "max_length", 60)),
            "joined_seq_len": int(value_tensor.shape[1])
            if tensor_input and value_tensor.ndim > 1
            else 0,
        },
    }
    if tokenizer == "wordpiece":
        info["classic"]["max_length_unit"] = WORDPIECE_MAX_LENGTH_UNIT
    elif tokenizer == "unigram":
        info["classic"]["max_length_unit"] = UNIGRAM_MAX_LENGTH_UNIT
        info["classic"]["training_regularization"] = _unigram_regularization_info(args)
    info["classic"].update(_classic_extra_artifact_info(args))
    info.update(_pair_profile_generation_artifact_info(args, profile_rows=profile_rows))
    if isinstance(strategy_contract, dict):
        info["pair_strategy_contract"] = strategy_contract
    return info


def save_pair_sequence_nn_artifact(args, model, train_x, threshold=0.5, profile_rows=None):
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    if not artifact_dir or not _pair_sequence_artifacts_supported(args):
        return
    os.makedirs(artifact_dir, exist_ok=True)
    _save_full_model_cpu(model, os.path.join(artifact_dir, "model.pt"))
    metadata = {
        "bundle_type": "pair_sequence_nn",
        "supports_inference": True,
        "task_type": normalize_task_type(getattr(args, "task_type", "pep_protein_interaction")),
        "args": _jsonable_args(args),
        "threshold": _frozen_pair_threshold(args),
        "threshold_source": "validation_f1",
        **_pair_protocol_metadata(args, model=model),
        **_pair_sequence_artifact_info(args, model, train_x, profile_rows=profile_rows),
    }
    head_config = prediction_head_artifact_config(args)
    if head_config is not None:
        metadata["prediction_head"] = head_config
    predictor_contract = sequence_predictor_contract(model)
    if predictor_contract is not None:
        metadata["predictor_contract"] = predictor_contract
    with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[artifact] saved reusable PPI Single-stream neural bundle: {artifact_dir}")


def _save_pair_embedding_encoder(args, model):
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    if not artifact_dir:
        return ""
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, "embedding_encoder.pt")
    _save_full_model_cpu(model, path)
    return _artifact_relpath(artifact_dir, path)


def _configure_pair_embedding_artifacts(
    args,
    route_info,
    encoder_path,
    embedding_dim,
    downstream,
    architecture_contract=None,
):
    info = dict(route_info or {})
    info["embedding_export"] = {
        "encoder_path": encoder_path,
        "embedding_dim": int(embedding_dim or 0),
        "downstream": str(downstream or "").strip().lower(),
    }
    if architecture_contract is not None:
        info["embedding_export"]["architecture_contract"] = architecture_contract
    args._artifact_source = PAIR_EMBEDDING_ARTIFACT_SOURCE
    args._artifact_info = info
    return info


def save_pair_embedding_nn_artifact(args, model, dl_scaler=None, threshold=0.5):
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    artifact_info = getattr(args, "_artifact_info", {}) or {}
    if (
        not artifact_dir
        or str(getattr(args, "_artifact_source", "") or "") != PAIR_EMBEDDING_ARTIFACT_SOURCE
        or not artifact_info
    ):
        return
    os.makedirs(artifact_dir, exist_ok=True)
    _save_full_model_cpu(model, os.path.join(artifact_dir, "model.pt"))
    metadata = {
        "bundle_type": "pair_embedding_nn",
        "supports_inference": True,
        "task_type": normalize_task_type(getattr(args, "task_type", "pep_protein_interaction")),
        "args": _jsonable_args(args),
        "threshold": _frozen_pair_threshold(args),
        "threshold_source": "validation_f1",
        **_pair_protocol_metadata(args, model=model),
        **artifact_info,
    }
    head_config = prediction_head_artifact_config(args)
    if head_config is not None:
        metadata["prediction_head"] = head_config
    base_model = getattr(model, "base", model)
    if isinstance(base_model, AlignedGatedClassifier):
        metadata["aligned_gated_contract"] = aligned_gated_contract(base_model.fusion)
    if dl_scaler is not None:
        joblib.dump({"dl_scaler": dl_scaler}, os.path.join(artifact_dir, "preprocess.joblib"))
        metadata["preprocess_path"] = "preprocess.joblib"
    with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[artifact] saved reusable PPI embedding-export neural bundle: {artifact_dir}")


def save_pair_vector_nn_artifact(
    args, model, dl_scaler=None, threshold=0.5, peptide_scaler=None, protein_scaler=None
):
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    artifact_info = getattr(args, "_artifact_info", {}) or {}
    if not artifact_dir or not _pair_vector_artifacts_supported(args) or not artifact_info:
        return
    threshold_value = _frozen_pair_threshold(args)
    os.makedirs(artifact_dir, exist_ok=True)
    original_device = next(model.parameters()).device
    model_cpu = model.to("cpu")
    torch.save(model_cpu, os.path.join(artifact_dir, "model.pt"))
    model.to(original_device)
    metadata = {
        "bundle_type": "pair_vector_nn",
        "supports_inference": True,
        "task_type": normalize_task_type(getattr(args, "task_type", "pep_protein_interaction")),
        "args": _jsonable_args(args),
        "threshold": threshold_value,
        "threshold_source": "validation_f1",
        **_pair_protocol_metadata(args, model=model),
        **artifact_info,
    }
    head_config = prediction_head_artifact_config(args)
    if head_config is not None:
        metadata["prediction_head"] = head_config
    if isinstance(model, AlignedGatedClassifier):
        metadata["aligned_gated_contract"] = aligned_gated_contract(model.fusion)
    if dl_scaler is not None or peptide_scaler is not None or protein_scaler is not None:
        joblib.dump(
            {
                "dl_scaler": dl_scaler,
                "peptide_scaler": peptide_scaler,
                "protein_scaler": protein_scaler,
            },
            os.path.join(artifact_dir, "preprocess.joblib"),
        )
        metadata["preprocess_path"] = "preprocess.joblib"
    with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[artifact] saved reusable PPI pair-vector neural bundle: {artifact_dir}")


def _threshold_value(threshold):
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        value = 0.5
    return value if np.isfinite(value) else 0.5


def _module_state_dict_cpu(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def save_pair_graph_nn_artifact(
    args,
    model,
    reference_features=None,
    reference_peptide_features=None,
    reference_protein_features=None,
    gate=None,
    peptide_scaler=None,
    protein_scaler=None,
    threshold=0.5,
):
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    artifact_info = getattr(args, "_graph_artifact_info", {}) or {}
    if not artifact_dir or not _pair_graph_artifacts_supported(args) or not artifact_info:
        return
    os.makedirs(artifact_dir, exist_ok=True)
    payload = {
        "model_class": type(model).__name__,
        "graph_predictor": str(
            artifact_info.get("graph", {}).get("predictor", getattr(args, "model", "gcn"))
        ),
        "state_dict": _module_state_dict_cpu(model),
    }
    if gate is not None:
        payload["gate_state_dict"] = _module_state_dict_cpu(gate)
    torch.save(payload, os.path.join(artifact_dir, "model.pt"))

    reference_payload = {}
    if reference_features is not None:
        reference_payload["reference_features"] = reference_features.detach().cpu().float()
    if reference_peptide_features is not None:
        reference_payload["reference_peptide_features"] = (
            reference_peptide_features.detach().cpu().float()
        )
    if reference_protein_features is not None:
        reference_payload["reference_protein_features"] = (
            reference_protein_features.detach().cpu().float()
        )
    torch.save(reference_payload, os.path.join(artifact_dir, "graph_reference.pt"))

    metadata = {
        "bundle_type": "pair_graph_nn",
        "supports_inference": True,
        "task_type": normalize_task_type(getattr(args, "task_type", "pep_protein_interaction")),
        "args": _jsonable_args(args),
        "threshold": _frozen_pair_threshold(args),
        "threshold_source": "validation_f1",
        **_pair_protocol_metadata(args, model=model, evaluation_mode="inductive"),
        **artifact_info,
    }
    if str((metadata.get("graph") or {}).get("predictor", "") or "").strip().lower() == HGT_MODEL:

        def strip_fusion_mode(value):
            if isinstance(value, dict):
                value.pop("fusion_mode", None)
                for child in value.values():
                    strip_fusion_mode(child)
            elif isinstance(value, list):
                for child in value:
                    strip_fusion_mode(child)

        strip_fusion_mode(metadata)
    if isinstance(gate, AlignedGatedFusion):
        metadata["aligned_gated_contract"] = aligned_gated_contract(gate)
    if peptide_scaler is not None or protein_scaler is not None:
        joblib.dump(
            {"peptide_scaler": peptide_scaler, "protein_scaler": protein_scaler},
            os.path.join(artifact_dir, "preprocess.joblib"),
        )
        metadata["preprocess_path"] = "preprocess.joblib"
    with open(os.path.join(artifact_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[artifact] saved reusable PPI pair-graph neural bundle: {artifact_dir}")


def resolve_path(root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def load_hlai_pseudoseq_map(task_root):
    path = os.path.join(task_root, "pseudoseqs_decoded.csv")
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            hla = str(hla_allele(r)).strip()
            pseudo = str(r.get("pseudoseq", "")).strip().upper()
            status = str(r.get("status", "ok")).strip().lower()
            if hla and pseudo and status in ["", "ok"]:
                mapping[hla] = pseudo
    return mapping


def read_ppi_csv(
    path,
    hla_pseudo_map=None,
    return_contexts=False,
    *,
    require_hlai_pseudoseq=False,
    pseudoseq_path=None,
    **legacy,
):
    hla_pseudo_map = legacy_keyword(legacy, "hla_pseudo_map", hla_pseudo_map)
    require_hlai_pseudoseq = legacy_keyword(
        legacy, "require_hlai_pseudoseq", require_hlai_pseudoseq
    )
    if legacy:
        raise TypeError(f"unexpected keywords: {sorted(legacy)}")
    rows = []
    contexts = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader, start=1):
            sid = (
                str(r.get("sample_id", "")).strip()
                or f"{os.path.splitext(os.path.basename(path))[0]}_{i}"
            )
            label_raw = r.get("label", r.get("tgt", r.get("bind", "")))
            pep = str(r.get("peptide", r.get("pep", ""))).strip().upper()
            protein = str(r.get("protein", "") or "").strip()
            hla = str(hla_allele(r) or "").strip()
            hla_pair = str(r.get("hla_pair", "") or "").strip()
            pair_pseudo = str(r.get("pair_pseudo", "") or "").strip().upper()
            context_id = protein or hla or hla_pair or pair_pseudo
            raw_partner = protein or hla or pair_pseudo or hla_pair
            pro = pair_pseudo or raw_partner
            if protein:
                pro = protein
            elif hla_pseudo_map and hla in hla_pseudo_map:
                pro = hla_pseudo_map[hla]
            if label_raw in [None, ""] or not pep:
                continue
            if hla or require_hlai_pseudoseq:
                require_hlai_partner_sequence(
                    pro,
                    allele=hla,
                    resolved=bool(
                        protein or pair_pseudo or (hla_pseudo_map and hla in hla_pseudo_map)
                    ),
                    source_path=path,
                    row_number=i,
                    pseudoseq_path=pseudoseq_path,
                )
            if not pro:
                continue
            label = int(float(label_raw))
            rows.append((sid, pep, pro, label))
            contexts.append(context_id or pro)
    if return_contexts:
        return rows, contexts
    return rows


def resolve_pair_top_ngram_lengths(args, rows):
    streams = getattr(args, "streams", {}) or {}
    active_roles = [
        role
        for role, spec in streams.items()
        if str(spec.get("tokenizer", "") or "").strip().lower() in {"top-n-gram", "dt"}
    ]
    if not active_roles:
        return
    role_lengths = {
        "peptide": max((len(row[1]) for row in rows), default=0),
        "partner": max((len(row[2]) for row in rows), default=0),
    }
    if str(getattr(args, "stream_mode", "") or "") == "homogeneous":
        required = max(role_lengths[role] for role in active_roles)
        requested = int(getattr(args, "max_length", 512))
        try:
            effective, adjusted = resolve_top_ngram_max_length(
                requested,
                required,
                context="profile-tokenizer homogeneous pair streams",
            )
        except TopNGramContractError as exc:
            raise SystemExit(str(exc)) from exc
        args.max_length = effective
        for role in active_roles:
            streams[role]["max_length"] = effective
            setattr(args, f"{role}_max_length", effective)
        if adjusted:
            print(
                "[profile-tokenizer] homogeneous stream max_length auto-adjusted "
                f"from {requested} to {effective} for full dataset coverage"
            )
    else:
        for role in active_roles:
            requested = int(streams[role].get("max_length", 512))
            try:
                effective, adjusted = resolve_top_ngram_max_length(
                    requested,
                    role_lengths[role],
                    context=f"profile-tokenizer {role} stream",
                )
            except TopNGramContractError as exc:
                raise SystemExit(str(exc)) from exc
            streams[role]["max_length"] = effective
            setattr(args, f"{role}_max_length", effective)
            if adjusted:
                print(
                    f"[profile-tokenizer] {role} stream max_length auto-adjusted "
                    f"from {requested} to {effective} for full dataset coverage"
                )
    args.streams = streams


def stratified_split(rows, val_ratio=0.1, seed=42, contexts=None):
    if contexts is not None and len(contexts) != len(rows):
        raise ValueError("contexts must have the same length as rows")
    items = list(zip(rows, contexts)) if contexts is not None else list(rows)
    pos = [item for item in items if (item[0] if contexts is not None else item)[3] == 1]
    neg = [item for item in items if (item[0] if contexts is not None else item)[3] == 0]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_val = int(len(pos) * val_ratio)
    n_neg_val = int(len(neg) * val_ratio)
    val = pos[:n_pos_val] + neg[:n_neg_val]
    train = pos[n_pos_val:] + neg[n_neg_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    if contexts is not None:
        train_rows = [r for r, _ in train]
        train_contexts = [c for _, c in train]
        val_rows = [r for r, _ in val]
        val_contexts = [c for _, c in val]
        return train_rows, val_rows, train_contexts, val_contexts
    return train, val


def _offline_mode_enabled():
    return str(os.environ.get("PEPPLATFORM_OFFLINE_MODE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_model_source(raw_value, local_candidates, label="pretrained model", cli_arg="model"):
    tokenizer = str(cli_arg).replace("_model", "")
    return resolve_workflow_pretrained_model_source(
        project_root,
        raw_value,
        local_candidates,
        label,
        cli_arg,
        tokenizer=tokenizer,
    )


def resolve_esm2_model_source(args):
    raw_model = getattr(args, "esm2_model", ESM2_OFFICIAL_MODEL_ID)
    candidates = [
        os.path.join(project_root, "models", "esm2"),
        os.path.join(project_root, "models", "esm2_t30_150M_UR50D"),
        os.path.join(project_root, "models", "facebook", "esm2_t30_150M_UR50D"),
        os.path.expanduser("~/models/esm2"),
    ]
    resolved = _resolve_model_source(raw_model, candidates, "ESM2", "esm2_model")
    if os.path.isdir(resolved) and is_complete_esm2_snapshot(resolved):
        return resolved
    return resolve_esm2_source(project_root, raw_model, [resolved, *candidates])


def resolve_protbert_model_source(args):
    protbert_model = getattr(args, "protbert_model", "")
    legacy_model = getattr(args, "probert_model", "")
    raw_model = (
        legacy_model
        if legacy_model and protbert_model in ["", "Rostlab/prot_bert"]
        else protbert_model
    )
    return _resolve_model_source(
        raw_model or PROTBERT_OFFICIAL_MODEL_ID,
        [
            os.path.join(project_root, "models", "protbert"),
            os.path.join(project_root, "models", "probert"),
            os.path.join(project_root, "models", "prot_bert"),
            os.path.join(project_root, "models", "Rostlab", "prot_bert"),
            os.path.expanduser("~/models/protbert"),
            os.path.expanduser("~/models/probert"),
        ],
        "ProtBERT",
        "protbert_model",
    )


def resolve_prott5_model_source(args):
    return _resolve_model_source(
        getattr(args, "prott5_model", "Rostlab/prot_t5_xl_uniref50"),
        [
            os.path.join(project_root, "models", "prott5"),
            os.path.join(project_root, "models", "prot_t5_xl_uniref50"),
            os.path.join(project_root, "models", "Rostlab", "prot_t5_xl_uniref50"),
            os.path.expanduser("~/models/prott5"),
        ],
        "ProT5",
        "prott5_model",
    )


def resolve_esmc_model_source(args):
    raw_model = getattr(args, "esmc_model", PRETRAINED_MODEL_DEFAULTS["esmc"])
    return _resolve_model_source(
        raw_model,
        pretrained_model_local_candidates(project_root, "esmc", raw_model),
        "ESM-C",
        "esmc_model",
    )


def resolve_esm3_model_source(args):
    return _resolve_model_source(
        getattr(args, "esm3_model", ESM3_OFFICIAL_MODEL_ID),
        [
            os.path.join(project_root, "models", "esm3"),
            os.path.join(project_root, "models", "esm3-sm-open-v1"),
            os.path.join(project_root, "models", "EvolutionaryScale", "esm3-sm-open-v1"),
            os.path.expanduser("~/models/esm3"),
        ],
        "ESM3",
        "esm3_model",
    )


def build_pretrained_tokenizer_encoder(args, device):
    if args.tokenizer == "esm2":
        model_name = resolve_esm2_model_source(args)
        args._resolved_pretrained_model_source = model_name
        print(
            f"[esm2] {'using local model' if os.path.exists(model_name) else 'local model not found, using remote'}: {model_name}"
        )
        tok = EsmTokenizer(model_name=model_name, max_length=args.max_length)
        enc = EsmEncoder(model_name=model_name, device=device)
        args._resolved_pretrained_model_identity = dict(enc.model_identity)
        args._resolved_pretrained_runtime_environment = dict(enc.runtime_environment)
        print(format_esm2_provenance(enc.model_identity, enc.runtime_environment), flush=True)
        return tok, enc, args.tokenizer
    if args.tokenizer == "protbert":
        model_name = resolve_protbert_model_source(args)
        args._resolved_pretrained_model_source = model_name
        print(
            f"[protbert] {'using local model' if os.path.exists(model_name) else 'local model not found, using remote'}: {model_name}"
        )
        tok = ProtBertTokenizer(model_name=model_name, max_length=args.max_length)
        enc = ProtBertEncoder(model_name=model_name, device=device)
        args._resolved_pretrained_model_identity = dict(enc.model_identity)
        args._resolved_pretrained_runtime_environment = dict(enc.runtime_environment)
        print(
            format_protbert_provenance(
                enc.model_identity,
                enc.runtime_environment,
            ),
            flush=True,
        )
        return tok, enc, args.tokenizer
    if args.tokenizer == "prott5":
        model_name = resolve_prott5_model_source(args)
        if int(args.max_length) == PROTT5_MAX_RESIDUES + 1:
            args.prott5_max_length_adjustment = {
                "requested": int(args.max_length),
                "effective": PROTT5_MAX_RESIDUES,
                "reason": "T5 n_positions includes the EOS token",
            }
            args.max_length = PROTT5_MAX_RESIDUES
        args._resolved_pretrained_model_source = model_name
        print(
            f"[prott5] {'using local model' if os.path.exists(model_name) else 'local model not found, using remote'}: {model_name}"
        )
        tok = ProT5Tokenizer(model_name=model_name, max_length=args.max_length)
        enc = ProT5Encoder(model_name=model_name, device=device)
        enc.model_identity["tokenizer_contract"] = {
            "residue_tokenization": tok.tokenizer_contract["residue_tokenization"],
            "pad_token_id": tok.tokenizer_contract.get("pad_token_id", PROTT5_PAD_TOKEN_ID),
            "eos_token_id": tok.tokenizer_contract.get("eos_token_id", PROTT5_EOS_TOKEN_ID),
            "vocab_size": tok.tokenizer_contract.get("vocab_size", PROTT5_VOCAB_SIZE),
            "max_residues": PROTT5_MAX_RESIDUES,
        }
        args._resolved_pretrained_model_identity = dict(enc.model_identity)
        args._resolved_pretrained_runtime_environment = dict(enc.runtime_environment)
        print(format_prott5_provenance(enc.model_identity, enc.runtime_environment), flush=True)
        return tok, enc, args.tokenizer
    if args.tokenizer == "esmc":
        model_name = resolve_esmc_model_source(args)
        args._resolved_pretrained_model_source = model_name
        print(
            f"[esmc] {'using local model' if os.path.exists(model_name) else 'local model not found, using remote'}: {model_name}"
        )
        tok = EsmcTokenizer(model_name=model_name, max_length=args.max_length)
        enc = EsmcEncoder(model_name=model_name, device=device)
        args._resolved_pretrained_model_identity = dict(enc.model_identity)
        args._resolved_pretrained_runtime_environment = dict(enc.runtime_environment)
        print(
            format_esmc_provenance(enc.model_identity, enc.runtime_environment),
            flush=True,
        )
        return tok, enc, args.tokenizer
    if args.tokenizer == "esm3":
        model_name = resolve_esm3_model_source(args)
        args._resolved_pretrained_model_source = model_name
        print(
            f"[esm3] {'using local model' if os.path.exists(model_name) else 'local model not found, using remote'}: {model_name}"
        )
        enc = Esm3Encoder(
            model_name=model_name,
            device=device,
            sdpa_backend=validate_esm3_sdpa_backend(
                getattr(
                    args,
                    "esm3_sdpa_backend",
                    ESM3_DEFAULT_SDPA_BACKEND,
                )
            ),
        )
        tok = Esm3Tokenizer(model_name=model_name, max_length=args.max_length)
        args._resolved_pretrained_model_identity = dict(enc.model_identity)
        args._resolved_pretrained_runtime_environment = dict(enc.runtime_environment)
        print(format_esm3_provenance(enc.model_identity, enc.runtime_environment), flush=True)
        return tok, enc, args.tokenizer
    raise SystemExit(f"Unsupported tokenizer: {args.tokenizer}")


def extract_embeddings(args, tokenizer, encoder, sequences, split_name, role_name, device):
    cache_enabled = bool(getattr(args, "cache_features", True))
    save_path = ""
    input_items = list(sequences)
    expected_meta = _feature_cache_meta(
        args, split_name, role_name, input_items, encoder_name=str(args.tokenizer).lower()
    )
    unique_sequences = []
    row_to_unique = []
    if str(args.tokenizer).lower() == "esm3":
        unique_indices = {}
        for sequence in input_items:
            normalized = truncate_esm3_sequence(
                normalize_esm3_sequence(sequence),
                int(args.max_length),
            )
            index = unique_indices.get(normalized)
            if index is None:
                index = len(unique_sequences)
                unique_indices[normalized] = index
                unique_sequences.append(normalized)
            row_to_unique.append(index)
        expected_meta.update(
            {
                "cache_layout": "unique_sequence_shards_v1",
                "checkpoint_rows": 64,
                "sdpa_backend": validate_esm3_sdpa_backend(
                    getattr(
                        args,
                        "esm3_sdpa_backend",
                        ESM3_DEFAULT_SDPA_BACKEND,
                    )
                ),
                "unique_sequence_count": len(unique_sequences),
            }
        )
    if cache_enabled:
        cache_namespace = str(args.tokenizer).lower()
        if cache_namespace == "esm3":
            cache_namespace += "_" + validate_esm3_sdpa_backend(
                getattr(
                    args,
                    "esm3_sdpa_backend",
                    ESM3_DEFAULT_SDPA_BACKEND,
                )
            )
        save_dir = os.path.join(args.feature_dir, cache_namespace)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{split_name}_{role_name}.pt")
        x = _load_feature_cache(
            save_path,
            expected_meta,
            lambda item: _valid_pretrained_pooled_cache_tensor(
                item,
                row_count=len(input_items),
                output_dim=int(encoder.output_dim),
            ),
        )
        if x is not None:
            return x
    all_embeddings = []
    if args.tokenizer == "esm3":
        shard_root = Path(save_path + ".parts") if save_path else None
        if shard_root is not None:
            shard_root.mkdir(parents=True, exist_ok=True)
        checkpoint_rows = 64
        for start in tqdm(
            range(0, len(unique_sequences), checkpoint_rows),
            desc=f"Extracting {split_name}/{role_name} ({args.tokenizer})",
            leave=False,
        ):
            end = min(len(unique_sequences), start + checkpoint_rows)
            batch = unique_sequences[start:end]
            shard = None
            shard_path = (
                shard_root / f"part_{start:06d}_{end:06d}.pt" if shard_root is not None else None
            )
            shard_meta = {
                "schema_version": 1,
                "start": start,
                "end": end,
                "input_sha256": _feature_input_digest(batch),
                "full_input_sha256": expected_meta["input_sha256"],
                "model_fingerprint": expected_meta.get("model_fingerprint", ""),
                "sdpa_backend": expected_meta["sdpa_backend"],
                "max_length": int(args.max_length),
            }
            if shard_path is not None:
                shard = _load_feature_cache(
                    str(shard_path),
                    shard_meta,
                    lambda item: (
                        isinstance(item, torch.Tensor)
                        and item.ndim == 2
                        and item.shape[0] == end - start
                        and item.shape[1] == int(encoder.output_dim)
                        and bool(torch.isfinite(item).all())
                    ),
                )
            if shard is None:
                shard = encoder.encode_sequences(
                    batch,
                    max_length=args.max_length,
                ).cpu()
                if shard_path is not None:
                    _save_feature_cache(str(shard_path), shard, shard_meta)
            all_embeddings.append(shard)
        unique_embeddings = (
            torch.cat(all_embeddings, dim=0)
            if all_embeddings
            else torch.empty((0, int(encoder.output_dim)), dtype=torch.float32)
        )
        row_index = torch.tensor(row_to_unique, dtype=torch.long)
        x = unique_embeddings.index_select(0, row_index)
    else:
        starts = range(0, len(sequences), args.batch_size)
        progress = tqdm(
            starts, desc=f"Extracting {split_name}/{role_name} ({args.tokenizer})", leave=False
        )
        with torch.no_grad():
            for i in progress:
                batch = sequences[i : i + args.batch_size]
                inputs = tokenizer.batch_tokenize(batch, device=device)
                emb = encoder.encode(inputs).cpu()
                all_embeddings.append(emb)
        x = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)
    if cache_enabled:
        _save_feature_cache(save_path, x, expected_meta)
    return x


def _forward_token_hidden(encoder, inputs):
    model_inputs = {key: value for key, value in inputs.items() if key != "special_tokens_mask"}
    with torch.no_grad():
        try:
            outputs = encoder.model(**model_inputs, output_hidden_states=True)
        except TypeError:
            outputs = encoder.model(**model_inputs)
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        h = outputs.last_hidden_state
    elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        h = outputs.hidden_states[-1]
    else:
        raise SystemExit("pretrained encoder does not provide token hidden states")
    mask = inputs.get("attention_mask")
    if mask is None:
        mask = torch.ones(h.shape[:2], dtype=h.dtype, device=h.device)
    else:
        mask = mask.float()
    return h * mask.unsqueeze(-1)


def _pad_or_trim_token_hidden(h, target_len):
    if h.shape[1] == target_len:
        return h
    if h.shape[1] > target_len:
        return h[:, :target_len, :]
    out = torch.zeros((h.shape[0], target_len, h.shape[2]), dtype=h.dtype, device=h.device)
    out[:, : h.shape[1], :] = h
    return out


def _compact_sequence_identity(tokenizer, sequence, target_len):
    normalizer = getattr(tokenizer, "normalize_sequence", None)
    normalized = normalizer(sequence) if callable(normalizer) else str(sequence).strip().upper()
    if isinstance(normalized, str) and " " in normalized:
        return " ".join(normalized.split()[:target_len])
    return str(normalized)[:target_len]


def _extract_compact_pretrained_token_sequences(
    args,
    tokenizer,
    encoder,
    sequences,
    split_name,
    role_name,
    device,
):
    tok_name = str(getattr(args, "tokenizer", "")).strip().lower()
    if tok_name not in {"esm2", "protbert", "prott5", "esmc"}:
        raise SystemExit(
            "--compact_residue_cache currently requires tokenizer in "
            "{esm2,protbert,prott5,esmc} with the audited explicit-mask residue contract"
        )
    cache_enabled = bool(getattr(args, "cache_features", True))
    target_len = int(getattr(args, "max_length", 60))
    input_items = list(sequences)
    unique_sequences = []
    unique_by_identity = {}
    row_to_unique = []
    for sequence in input_items:
        identity = _compact_sequence_identity(tokenizer, sequence, target_len)
        unique_index = unique_by_identity.get(identity)
        if unique_index is None:
            unique_index = len(unique_sequences)
            unique_by_identity[identity] = unique_index
            unique_sequences.append(sequence)
        row_to_unique.append(unique_index)

    feature_dim = int(getattr(encoder, "output_dim", 0) or 0)
    if feature_dim <= 0:
        raise SystemExit("pretrained residue encoder is missing a positive output_dim")
    expected_meta = _feature_cache_meta(
        args,
        split_name,
        role_name,
        input_items,
        encoder_name=tok_name,
        target_len=target_len,
    )
    expected_meta.update(
        {
            "cache_layout": "unique_sequence_ragged_v1",
            "feature_dtype": "float32",
            "feature_dim": feature_dim,
            "unique_sequence_count": len(unique_sequences),
        }
    )
    save_path = ""
    if cache_enabled:
        save_dir = os.path.join(args.feature_dir, f"{tok_name}_residue_compact")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{split_name}_{role_name}.pt")
        cached = _load_feature_cache(
            save_path,
            expected_meta,
            lambda item: _valid_compact_residue_encoding(
                item,
                row_count=len(input_items),
                max_length=target_len,
                feature_dim=feature_dim,
            ),
        )
        if cached is not None:
            print(
                f"[feature] reuse compact residue cache: split={split_name}, "
                f"role={role_name}, unique={len(unique_sequences)}/{len(input_items)}",
                flush=True,
            )
            return cached

    print(
        f"[feature] start compact token extraction: split={split_name}, "
        f"role={role_name}, model={tok_name}, "
        f"unique={len(unique_sequences)}/{len(input_items)}",
        flush=True,
    )
    flat_batches = []
    offsets = [0]
    starts = range(0, len(unique_sequences), max(1, int(args.batch_size)))
    for start in tqdm(
        starts,
        desc=f"Encoding {split_name}/{role_name} compact tokens ({tok_name})",
        leave=False,
    ):
        batch = unique_sequences[start : start + max(1, int(args.batch_size))]
        inputs = tokenizer.batch_tokenize_residues(batch, target_len, device=device)
        if tok_name == "esm2":
            lengths = [
                len(tokenizer.normalize_sequence(sequence)[:target_len]) for sequence in batch
            ]
        else:
            lengths = [tokenizer.residue_length(sequence, target_len) for sequence in batch]
        encoded = encoder.encode_residue_batch(inputs, lengths, target_len)
        if not isinstance(encoded, ResidueModelInput):
            encoded = ResidueModelInput(encoded.features, encoded.mask)
        features = encoded.features.detach().cpu().float()
        mask = encoded.mask.detach().cpu().bool()
        if features.ndim != 3 or mask.shape != features.shape[:2]:
            raise SystemExit("pretrained residue encoder returned an invalid explicit-mask batch")
        for row in range(features.shape[0]):
            compact = features[row, mask[row]]
            if compact.shape[0] < 1:
                raise SystemExit("pretrained residue encoder returned an empty sequence")
            flat_batches.append(compact)
            offsets.append(offsets[-1] + int(compact.shape[0]))
        del encoded, features, mask

    flat = (
        torch.cat(flat_batches, dim=0)
        if flat_batches
        else torch.empty((0, feature_dim), dtype=torch.float32)
    )
    result = CompactResidueEncoding(
        flat_features=flat,
        offsets=torch.tensor(offsets, dtype=torch.int64),
        row_to_unique=torch.tensor(row_to_unique, dtype=torch.int64),
        max_length=target_len,
        feature_dim=feature_dim,
    )
    if not _valid_compact_residue_encoding(
        result,
        row_count=len(input_items),
        max_length=target_len,
        feature_dim=feature_dim,
    ):
        raise SystemExit("constructed compact residue encoding failed validation")
    if cache_enabled:
        _save_feature_cache(save_path, result, expected_meta)
    return result


def materialize_compact_single_stream(args, peptide, partner, split_name):
    if not isinstance(peptide, CompactResidueEncoding) or not isinstance(
        partner, CompactResidueEncoding
    ):
        raise TypeError("compact Single-stream join requires both Pair roles")
    if peptide.shape[0] != partner.shape[0]:
        raise ValueError("compact Pair roles have different row counts")
    if peptide.max_length != partner.max_length or peptide.feature_dim != partner.feature_dim:
        raise ValueError("compact Pair roles have incompatible residue contracts")
    result = CompactSingleStreamEncoding(peptide, partner, str(split_name))
    row_count, joined_length, joined_dim = result.shape
    source_dim = int(peptide.feature_dim)
    print(
        f"[feature] defer compact Single-stream materialization to batches: "
        f"split={split_name}, logical_shape=({row_count}, {joined_length}, {joined_dim})",
        flush=True,
    )
    if row_count < 1:
        raise ValueError("compact Single-stream input requires at least one row")
    contract_sample = result.materialize_indices([0])
    args._single_stream_strategy_contract = build_single_stream_strategy_contract(
        contract_sample,
        separator_id=None,
        source_feature_dim=source_dim,
    )
    return result


def join_pretrained_single_stream(args, peptide, partner, split_name):
    if isinstance(peptide, CompactResidueEncoding) or isinstance(partner, CompactResidueEncoding):
        return materialize_compact_single_stream(args, peptide, partner, split_name)
    if not isinstance(peptide, ResidueModelInput) or not isinstance(partner, ResidueModelInput):
        raise SystemExit("pretrained Single-stream routes require matched explicit residue masks")
    joined = join_single_stream_batches(peptide, partner)
    args._single_stream_strategy_contract = build_single_stream_strategy_contract(
        joined,
        separator_id=None,
        source_feature_dim=int(peptide.features.shape[2]),
    )
    return joined


def _token_extraction_contract(args, encoder, role_name):
    """Opt-in numerical identity for residue extraction, separate from fitting."""
    role = "partner" if role_name in {"protein", "partner"} else "peptide"
    explicit = getattr(args, f"{role}_feature_extraction_batch_size", None)
    if explicit is None:
        return None
    batch_size = _parse_feature_extraction_batch_size(explicit)
    if str(args.tokenizer).lower() not in {"esm2", "prott5", "protbert", "esmc"}:
        raise ValueError(
            "explicit residue extraction batches require a supported Hugging Face residue encoder"
        )
    if bool(getattr(args, "compact_residue_cache", False)):
        raise ValueError("explicit residue extraction batches require the dense residue cache")
    model = encoder.model
    parameter = next(model.parameters())
    return {
        "schema_version": 1,
        "role": role,
        "batch_size": batch_size,
        # CPU feature storage uses FP16 independently of model computation dtype.
        "compute_dtype": str(parameter.dtype),
        "storage_dtype": str(torch.float16),
        "device": str(parameter.device),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "attention_implementation": getattr(
            getattr(model, "config", None), "_attn_implementation", None
        ),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "sdpa_math": torch.backends.cuda.math_sdp_enabled(),
        "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(),
        "sdpa_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
    }


def _validate_token_extraction_storage(
    value, contract, *, rows, target_len, feature_dim, require_mask, source
):
    """Check explicit cache payloads without converting their numerical values."""
    if isinstance(value, dict) and value.get("kind") == "ResidueModelInput":
        features, mask = value.get("features"), value.get("mask")
    elif isinstance(value, ResidueModelInput):
        features, mask = value.features, value.mask
    else:
        features, mask = value, None
    expected_shape = (rows, target_len, feature_dim)
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 3
        or tuple(features.shape) != expected_shape
        or str(features.dtype) != contract["storage_dtype"]
    ):
        raise ValueError(
            f"{source} residue storage differs from its declared contract: "
            f"expected dtype={contract['storage_dtype']}, shape={expected_shape}; "
            f"actual dtype={getattr(features, 'dtype', None)}, shape={getattr(features, 'shape', None)}"
        )
    if require_mask and (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or tuple(mask.shape) != expected_shape[:2]
    ):
        raise ValueError(
            f"{source} residue mask differs from its declared contract: "
            f"expected dtype=torch.bool, shape={expected_shape[:2]}; "
            f"actual dtype={getattr(mask, 'dtype', None)}, shape={getattr(mask, 'shape', None)}"
        )


def extract_pretrained_token_sequences(
    args, tokenizer, encoder, sequences, split_name, role_name, device
):
    extraction_contract = _token_extraction_contract(args, encoder, role_name)
    if bool(getattr(args, "compact_residue_cache", False)):
        return _extract_compact_pretrained_token_sequences(
            args,
            tokenizer,
            encoder,
            sequences,
            split_name,
            role_name,
            device,
        )
    cache_enabled = bool(getattr(args, "cache_features", True))
    save_path = ""
    target_len = int(getattr(args, "max_length", 60))
    input_items = list(sequences)
    expected_meta = _feature_cache_meta(
        args,
        split_name,
        role_name,
        input_items,
        encoder_name=str(args.tokenizer).lower(),
        target_len=target_len,
    )
    if extraction_contract is not None:
        expected_meta["extraction_contract"] = extraction_contract
    if cache_enabled:
        save_dir = os.path.join(args.feature_dir, f"{str(args.tokenizer).lower()}_tokens")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{split_name}_{role_name}.pt")
        if extraction_contract is not None and (
            os.path.exists(save_path) or os.path.exists(_cache_meta_path(save_path))
        ):
            try:
                with open(_cache_meta_path(save_path)) as handle:
                    declared = json.load(handle)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "explicit extraction cache identity is missing or invalid"
                ) from exc
            if not os.path.isfile(save_path) or declared != expected_meta:
                raise ValueError(
                    "explicit extraction cache identity differs; use a fresh private cache"
                )
            # Validate before the legacy loader normalizes mask dtype or falls
            # back to regenerating a malformed cache with matching metadata.
            payload = torch.load(save_path, map_location="cpu", weights_only=True, mmap=True)
            _validate_token_extraction_storage(
                payload,
                extraction_contract,
                rows=len(input_items),
                target_len=target_len,
                feature_dim=int(encoder.output_dim),
                require_mask=str(args.tokenizer).lower() in {"esm2", "protbert", "prott5"},
                source="cached",
            )
            del payload
        x = _load_feature_cache(
            save_path,
            expected_meta,
            lambda item: (
                (
                    isinstance(item, torch.Tensor)
                    and item.ndim == 3
                    and item.shape[0] == len(input_items)
                    and item.shape[1] == target_len
                )
                or (
                    isinstance(item, ResidueModelInput)
                    and item.features.ndim == 3
                    and item.features.shape[0] == len(input_items)
                    and item.features.shape[1] == target_len
                    and item.mask.shape == item.features.shape[:2]
                )
            ),
        )
        if x is not None:
            return x

    print(
        f"[feature] start token extraction: split={split_name}, role={role_name}, model={args.tokenizer}",
        flush=True,
    )
    all_h = []
    tok_name = str(getattr(args, "tokenizer", "")).lower()
    if tok_name == "esm3":
        esm3_bs = max(1, int(getattr(args, "esm3_extract_batch_size", 1)))
        starts = range(0, len(sequences), esm3_bs)
        for i in tqdm(
            starts, desc=f"Encoding {split_name}/{role_name} tokens ({args.tokenizer})", leave=False
        ):
            batch = sequences[i : i + esm3_bs]
            encoded = encoder.encode_residue_batch(batch, target_len)
            h = ResidueModelInput(encoded.features.cpu(), encoded.mask.cpu())
            all_h.append(h)
    else:
        extraction_batch_size = (
            extraction_contract["batch_size"]
            if extraction_contract is not None
            else args.batch_size
        )
        starts = range(0, len(sequences), extraction_batch_size)
        for i in tqdm(
            starts, desc=f"Encoding {split_name}/{role_name} tokens ({args.tokenizer})", leave=False
        ):
            batch = sequences[i : i + extraction_batch_size]
            if tok_name in {"esm2", "protbert", "prott5", "esmc"}:
                inputs = tokenizer.batch_tokenize_residues(
                    batch,
                    target_len,
                    device=device,
                )
                lengths = [
                    len(tokenizer.normalize_sequence(sequence)[:target_len])
                    if tok_name == "esm2"
                    else (
                        tokenizer.residue_length(sequence, target_len)
                        if tok_name in {"protbert", "prott5"}
                        else min(len(sequence), target_len)
                    )
                    for sequence in batch
                ]
                encoded = encoder.encode_residue_batch(
                    inputs,
                    lengths,
                    target_len,
                )
                h = (
                    ResidueModelInput(encoded.features, encoded.mask)
                    if tok_name in {"esm2", "protbert", "prott5"}
                    else encoded.features
                )
            else:
                inputs = tokenizer.batch_tokenize(batch, device=device)
                h = _forward_token_hidden(encoder, inputs)
                h = _pad_or_trim_token_hidden(h, target_len)
            if isinstance(h, ResidueModelInput):
                h = ResidueModelInput(h.features.cpu(), h.mask.cpu())
            else:
                h = h.cpu()
            all_h.append(h)
    if all_h and isinstance(all_h[0], ResidueModelInput):
        x = ResidueModelInput(
            torch.cat([item.features for item in all_h], dim=0),
            torch.cat([item.mask for item in all_h], dim=0),
        )
    else:
        x = torch.cat(all_h, dim=0) if all_h else torch.empty((0, 0, 0), dtype=torch.float)
    if extraction_contract is not None:
        _validate_token_extraction_storage(
            x,
            extraction_contract,
            rows=len(input_items),
            target_len=target_len,
            feature_dim=int(encoder.output_dim),
            require_mask=str(args.tokenizer).lower() in {"esm2", "protbert", "prott5"},
            source="extracted",
        )
    if cache_enabled:
        _save_feature_cache(save_path, x, expected_meta)
    return x


def _resolve_optional_path(base_root, raw_path):
    if not raw_path:
        return ""
    return raw_path if os.path.isabs(raw_path) else os.path.join(base_root, raw_path)


def _static_embedding_target(args, task_root, encoder_name, role_name=None):
    tok = str(getattr(args, "tokenizer", "")).lower()
    if tok == "kmer":
        tag = f"kmer-{int(getattr(args, 'k', 2))}"
        mode = "kmer"
    elif tok == "dr":
        tag = (
            "dr-v2-"
            f"{validate_dr_max_distance(getattr(args, 'dr_max_distance', DR_DEFAULT_MAX_DISTANCE))}"
        )
        mode = "dr"
    else:
        return None
    role_suffix = f"-{role_name}" if role_name else ""
    out_dir = os.path.join(task_root, f"{encoder_name}-{tag}{role_suffix}")
    extension = ".bin" if encoder_name == "fasttext" else ".vec"
    out_path = os.path.join(out_dir, f"{encoder_name}{extension}")
    return mode, out_path


def _write_ppi_static_fasta(rows, out_path, role_name=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        idx = 0
        for sid, pep, pro, _ in rows:
            if role_name in {None, "peptide"}:
                f.write(f">{sid}_pep_{idx}\n{pep}\n")
                idx += 1
            if role_name in {None, "partner", "protein"}:
                f.write(f">{sid}_pro_{idx}\n{pro}\n")
                idx += 1


def ensure_ppi_static_embedding_ready(args, task_root, train_rows, val_rows, role_name=None):
    enc = str(getattr(args, "encoder", "")).lower()
    if enc not in ["word2vec", "glove", "fasttext"]:
        return
    raw_path = str(getattr(args, f"{enc}_path", "") or "").strip()
    auto_path = bool(getattr(args, f"_{enc}_path_auto_generated", False))
    if raw_path and not auto_path:
        explicit_path = _resolve_optional_path(project_root, raw_path)
        if not os.path.isfile(explicit_path):
            raise SystemExit(f"Explicit {enc} embedding not found: {explicit_path}")
        if enc == "fasttext" and os.path.splitext(explicit_path)[1].lower() != ".bin":
            raise SystemExit(
                "Explicit fastText artifacts must be complete Facebook-compatible "
                ".bin files; legacy .vec files omit subword state and are unsupported"
            )
        setattr(args, f"{enc}_path", explicit_path)
        args._static_embedding_provenance = {
            "method": enc,
            "source": "external",
            "path": explicit_path,
            "sha256": sha256_file(explicit_path),
            "algorithm_verified": False,
            "verification": "unverified",
        }
        return
    target = _static_embedding_target(args, task_root, enc, role_name=role_name)
    if target is None:
        raise SystemExit(f"{enc} in PPI currently supports tokenizer kmer/dr only.")
    tokenizer_mode, out_path = target
    setattr(args, f"{enc}_path", out_path)
    setattr(args, f"_{enc}_path_auto_generated", True)
    data_suffix = f"_{role_name}" if role_name else ""
    data_root = os.path.join(task_root, f"_ppi_static_embed_data{data_suffix}")
    train_fasta = os.path.join(data_root, "train", "seqs.fasta")
    _write_ppi_static_fasta(train_rows, train_fasta, role_name=role_name)
    with open(train_fasta, "rb") as handle:
        train_sha256 = hashlib.sha256(handle.read()).hexdigest()
    expected_meta = {
        "method": enc,
        "tokenizer_mode": tokenizer_mode,
        "k": int(getattr(args, "k", 2)),
        "dr_max_distance": int(getattr(args, "dr_max_distance", 5)),
        "vector_size": int(getattr(args, "embedding_vector_size", 128)),
        "window": int(getattr(args, "embedding_window", 5)),
        "min_count": int(getattr(args, "embedding_min_count", 1)),
        "epochs": int(getattr(args, "embedding_epochs", 20)),
        "workers": (
            1
            if enc == "glove"
            else validate_positive_int(
                getattr(args, "embedding_workers", 1),
                name="embedding_workers",
            )
        ),
        "seed": int(getattr(args, "seed", 42)),
        "feature_fit_scope": "train_only",
        "train_sha256": train_sha256,
    }
    if role_name:
        expected_meta["cache_suffix"] = str(role_name)
    if tokenizer_mode == "dr":
        max_distance = validate_dr_max_distance(
            getattr(
                args,
                "dr_max_distance",
                DR_DEFAULT_MAX_DISTANCE,
            )
        )
        max_raw_length = validate_dr_max_raw_length(
            getattr(
                args,
                "dr_max_raw_length",
                DR_DEFAULT_MAX_RAW_LENGTH,
            )
        )
        expected_meta["dr_max_raw_length"] = max_raw_length
        expected_meta["dr_contract"] = dr_contract_metadata(
            max_distance=max_distance,
            max_raw_length=max_raw_length,
            unknown_residue_policy=getattr(
                args,
                "dr_unknown_residue_policy",
                DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
            ),
        )
    if enc == "glove":
        expected_meta.update(
            glove_cache_signature(
                x_max=getattr(args, "glove_x_max", 100.0),
                alpha=getattr(args, "glove_alpha", 0.75),
                learning_rate=getattr(args, "glove_learning_rate", 0.05),
                batch_size=getattr(args, "glove_batch_size", 4096),
            )
        )
    elif enc == "word2vec":
        expected_meta.update(word2vec_cache_signature(workers=expected_meta["workers"]))
    elif enc == "fasttext":
        expected_meta.update(
            fasttext_cache_signature(
                workers=expected_meta["workers"],
                bucket=validate_positive_int(
                    getattr(args, "fasttext_bucket", FASTTEXT_DEFAULT_BUCKET),
                    name="fasttext_bucket",
                ),
            )
        )
    meta_path = os.path.join(os.path.dirname(out_path), "meta.json")
    if os.path.exists(out_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                cache_meta = json.load(handle)
            cache_matches = all(
                cache_meta.get(key) == value for key, value in expected_meta.items()
            )
            cache_matches = (
                cache_matches
                and cache_meta.get("artifact_file") == os.path.basename(out_path)
                and cache_meta.get("artifact_sha256") == sha256_file(out_path)
            )
            if cache_matches:
                args._static_embedding_provenance = {
                    **cache_meta,
                    "source": "internal",
                    "path": out_path,
                    "sha256": sha256_file(out_path),
                }
                return
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    if os.path.exists(out_path):
        print(f"[pair] ignoring static embedding with missing or stale metadata: {out_path}")

    cmd = [
        sys.executable,
        os.path.join(project_root, "tools", "train_static_embeddings.py"),
        "--data_dir",
        data_root,
        "--tokenizer_mode",
        tokenizer_mode,
        "--save_root",
        task_root,
        "--vector_size",
        str(getattr(args, "embedding_vector_size", 128)),
        "--window",
        str(getattr(args, "embedding_window", 5)),
        "--min_count",
        str(getattr(args, "embedding_min_count", 1)),
        "--epochs",
        str(getattr(args, "embedding_epochs", 20)),
        "--workers",
        str(expected_meta["workers"]),
        "--seed",
        str(int(getattr(args, "seed", 42))),
    ]
    if role_name:
        cmd.extend(["--cache_suffix", str(role_name)])
    if tokenizer_mode == "kmer":
        cmd.extend(["--k", str(getattr(args, "k", 2))])
    else:
        cmd.extend(
            [
                "--dr_max_distance",
                str(
                    validate_dr_max_distance(
                        getattr(
                            args,
                            "dr_max_distance",
                            DR_DEFAULT_MAX_DISTANCE,
                        )
                    )
                ),
                "--dr_max_raw_length",
                str(
                    validate_dr_max_raw_length(
                        getattr(
                            args,
                            "dr_max_raw_length",
                            DR_DEFAULT_MAX_RAW_LENGTH,
                        )
                    )
                ),
            ]
        )
    if enc == "glove":
        cmd.extend(
            [
                "--glove_x_max",
                str(getattr(args, "glove_x_max", 100.0)),
                "--glove_alpha",
                str(getattr(args, "glove_alpha", 0.75)),
                "--glove_learning_rate",
                str(getattr(args, "glove_learning_rate", 0.05)),
                "--glove_batch_size",
                str(getattr(args, "glove_batch_size", 4096)),
            ]
        )
    elif enc == "fasttext":
        cmd.extend(
            [
                "--fasttext_bucket",
                str(
                    validate_positive_int(
                        getattr(
                            args,
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
                            args,
                            "static_embedding_memory_limit_gib",
                            STATIC_EMBEDDING_DEFAULT_MEMORY_LIMIT_GIB,
                        )
                    )
                ),
            ]
        )
    cmd.append(f"--train_{enc}")

    subprocess.run(cmd, check=True, cwd=project_root)
    if not os.path.exists(out_path):
        raise SystemExit(f"{enc} embedding generation failed: {out_path}")
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            cache_meta = json.load(handle)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"{enc} embedding generation produced unreadable cache metadata: {meta_path}"
        ) from exc
    generated_cache_matches = all(
        cache_meta.get(key) == value for key, value in expected_meta.items()
    )
    generated_cache_matches = (
        generated_cache_matches
        and cache_meta.get("artifact_file") == os.path.basename(out_path)
        and cache_meta.get("artifact_sha256") == sha256_file(out_path)
    )
    if not generated_cache_matches:
        raise SystemExit(
            f"{enc} embedding generation produced invalid or incomplete cache metadata: {meta_path}"
        )
    args._static_embedding_provenance = {
        **cache_meta,
        "source": "internal",
        "path": out_path,
        "sha256": sha256_file(out_path),
    }


def ensure_pair_profile_cache_ready(args, task_root, train_csv, val_csv, test_csv):
    tok = str(getattr(args, "tokenizer", "")).strip().lower()
    enc = str(getattr(args, "encoder", "")).strip().lower()
    if enc == "ppi_fusion":
        enc = str(getattr(args, "aux_encoder", "index")).strip().lower()
    need_frequency = tok in {"top-n-gram", "dt"}
    profile_feature = _profile_feature_for_encoder(enc)
    need_psfm = profile_feature in {
        FEATURE_PSFM,
        FEATURE_PSFM_BLOSUM62_HYBRID,
    }
    need_score_profile = bool(profile_feature and not need_psfm)
    if not profile_feature and not need_frequency:
        return True

    if enc in PSSM_COVARIANCE_ENCODERS:
        lag_key = pssm_covariance_lag_key(enc)
        lag = int(getattr(args, lag_key, 10))
        stream_role = str(getattr(args, "_stream_role", "") or "").strip().lower()
        hlai_map = (
            load_hlai_pseudoseq_map(task_root)
            if normalize_task_type(getattr(args, "task_type", "")) == "pep_hlai"
            else {}
        )
        for path in (train_csv, val_csv, test_csv):
            if not path or not os.path.exists(path):
                continue
            rows = read_ppi_csv(
                path,
                hla_pseudo_map=hlai_map,
                require_hlai_pseudoseq=normalize_task_type(getattr(args, "task_type", ""))
                == "pep_hlai",
                pseudoseq_path=os.path.join(task_root, "pseudoseqs_decoded.csv"),
            )
            if stream_role == "peptide":
                lengths = [len(row[1]) for row in rows]
            elif stream_role == "partner":
                lengths = [len(row[2]) for row in rows]
            else:
                lengths = [length for row in rows for length in (len(row[1]), len(row[2]))]
            minimum = min(lengths, default=0)
            if minimum <= lag:
                role_label = stream_role or "both pair roles"
                raise SystemExit(
                    f"{enc} requires every profile length > {lag_key}; "
                    f"{os.path.basename(path)} {role_label} minimum sequence "
                    f"length is {minimum}, lag is {lag}"
                )

    if profile_feature == FEATURE_PSSM_BLOSUM62_HYBRID:
        default_pssm_root = "pssm_hybrid_cache"
    elif profile_feature == FEATURE_PSFM_BLOSUM62_HYBRID:
        default_pssm_root = "psfm_hybrid_cache"
    elif profile_feature in HYBRID_PSSM_COVARIANCE_ENCODERS:
        default_pssm_root = f"{profile_feature}_cache"
    else:
        default_pssm_root = "pssm_cache"
    profile_index_key = _profile_index_key(profile_feature)
    pssm_index = _resolve_optional_path(
        project_root,
        getattr(args, profile_index_key, "")
        or os.path.join(
            task_root,
            default_pssm_root,
            "ppi",
            "index.json",
        ),
    )
    psfm_index = _resolve_optional_path(
        project_root,
        getattr(args, "psfm_index_path", "")
        or os.path.join(
            task_root,
            "psfm_hybrid_cache"
            if profile_feature == FEATURE_PSFM_BLOSUM62_HYBRID
            else "psfm_cache",
            "ppi",
            "index.json",
        ),
    )
    explicit_frequency_index = bool(
        str(getattr(args, "frequency_profile_index_path", "") or "").strip()
    )
    query_role = str(getattr(args, "_stream_role", "") or "").strip().lower()
    if query_role not in {"peptide", "partner"} or not need_frequency:
        query_role = "all"
    frequency_suffix = os.path.join("ppi", query_role) if query_role != "all" else "ppi"
    frequency_index = _resolve_optional_path(
        project_root,
        getattr(args, "frequency_profile_index_path", "")
        or os.path.join(
            task_root,
            "frequency_profile_cache",
            frequency_suffix,
            "index.json",
        ),
    )
    if need_score_profile:
        setattr(args, profile_index_key, pssm_index)
    args.psfm_index_path = psfm_index
    args.frequency_profile_index_path = frequency_index
    top_ngram_fallback_policy = (
        TOP_N_GRAM_FALLBACK_POLICY_VERSION
        if str(
            getattr(
                args,
                "profile_on_missing",
                TOP_N_GRAM_FALLBACK_POLICY,
            )
        )
        == TOP_N_GRAM_FALLBACK_POLICY
        else TOP_N_GRAM_STRICT_PROFILE_POLICY
    )
    config = ProfileGenerationConfig(
        iterations=int(getattr(args, "profile_iterations", 3)),
        evalue=float(getattr(args, "profile_evalue", 0.001)),
        inclusion_evalue=float(getattr(args, "profile_inclusion_evalue", 0.001)),
        threads=int(
            getattr(
                args,
                "profile_threads",
                getattr(args, "embedding_workers", 4),
            )
        ),
    )

    def expected_frequency_samples():
        hlai_map = (
            load_hlai_pseudoseq_map(task_root)
            if normalize_task_type(getattr(args, "task_type", "")) == "pep_hlai"
            else {}
        )
        expected = {}
        for path in (train_csv, val_csv, test_csv):
            if not path or not os.path.exists(path):
                continue
            for sample_id, peptide, partner, _label in read_ppi_csv(
                path,
                hla_pseudo_map=hlai_map,
                require_hlai_pseudoseq=normalize_task_type(getattr(args, "task_type", ""))
                == "pep_hlai",
                pseudoseq_path=os.path.join(task_root, "pseudoseqs_decoded.csv"),
            ):
                if query_role in {"all", "peptide"}:
                    expected[f"{sample_id}::pep"] = peptide
                if query_role in {"all", "partner"}:
                    expected[f"{sample_id}::pro"] = partner
        return expected

    expected_frequency = expected_frequency_samples() if need_frequency else {}

    def frequency_cache_ready(store, provenance, train_sha256):
        database = provenance.get("database") or {}
        if not (
            provenance.get("database_scope") == "train_only"
            and database.get("train_csv_sha256") == train_sha256
            and provenance.get("self_exclusion")
            == "all subject IDs with an identical normalized query sequence"
            and str(database.get("query_role", "all")) in {"all", query_role}
            and provenance.get("fallback_policy") == top_ngram_fallback_policy
            and provenance.get("scientific_parameters") == config.scientific_parameters()
        ):
            return False
        return all(
            sample_id in store.samples
            and store.samples[sample_id].sequence_length == len(sequence)
            and store.samples[sample_id].sequence_sha256 == sequence_sha256(sequence)
            for sample_id, sequence in expected_frequency.items()
        )

    def strict_cache_ready(index_path, feature_type):
        if not index_path or not os.path.exists(index_path):
            return False
        try:
            store = ProfileStore(index_path, feature_type=feature_type)
            with open(train_csv, "rb") as handle:
                train_sha256 = hashlib.sha256(handle.read()).hexdigest()
            provenance = store.manifest.get("provenance") or {}
            database = provenance.get("database") or {}
            ready = (
                provenance.get("database_scope") == "train_only"
                and database.get("train_csv_sha256") == train_sha256
                and str(database.get("query_role", "all")) in {"all", query_role}
                and provenance.get("self_exclusion")
                == "all subject IDs with an identical normalized query sequence"
                and provenance.get("scientific_parameters") == config.scientific_parameters()
                and provenance.get("fallback_policy")
                == (
                    "blosum62_hybrid"
                    if feature_type in (PSSM_HYBRID_FEATURE_TYPES | PSFM_HYBRID_FEATURE_TYPES)
                    else "error"
                )
            )
            if ready:
                args._profile_manifest_sha256 = store.manifest_sha256
                args._profile_manifest_provenance = dict(provenance)
            return ready
        except (OSError, ValueError, TypeError, SequenceProfileError):
            return False

    ok_pssm = (not need_score_profile) or strict_cache_ready(
        pssm_index,
        profile_feature,
    )
    ok_psfm = (not need_psfm) or strict_cache_ready(
        psfm_index,
        profile_feature if need_psfm else FEATURE_PSFM,
    )
    ok_frequency = not need_frequency
    if need_frequency and os.path.exists(frequency_index):
        try:
            store = TargetFrequencyProfileStore(
                frequency_index,
                max_length=int(getattr(args, "max_length", 60)),
            )
            provenance = store.manifest.get("provenance") or {}
            with open(train_csv, "rb") as handle:
                train_sha256 = hashlib.sha256(handle.read()).hexdigest()
            ok_frequency = frequency_cache_ready(
                store,
                provenance,
                train_sha256,
            )
            if ok_frequency:
                args._frequency_profile_manifest_sha256 = store.manifest_sha256
                args._frequency_profile_provenance = dict(provenance)
        except (OSError, ValueError, TypeError, TopNGramContractError):
            ok_frequency = False
    if ok_pssm and ok_psfm and ok_frequency:
        return True
    if need_frequency and explicit_frequency_index and not ok_frequency:
        print(
            "[pair] explicit --frequency_profile_index_path is missing, stale, "
            "or violates the target-frequency provenance contract; refusing to "
            "overwrite or replace it"
        )
        return False

    requested_features = []
    if profile_feature:
        requested_features.append(profile_feature)
    if need_frequency:
        requested_features.append("target_frequency")
    feature = ",".join(requested_features)
    effective_on_missing = str(getattr(args, "profile_on_missing", "error") or "error")
    print(
        "[pair] missing or stale profile index, building cache "
        f"(feature={feature}, on_missing={effective_on_missing}) ..."
    )
    cmd = [
        sys.executable,
        os.path.join(project_root, "tools", "build_ppi_profile_cache.py"),
        "--task_root",
        task_root,
        "--train_csv",
        train_csv,
        "--val_csv",
        val_csv,
        "--test_csv",
        test_csv,
        "--db_scope",
        "train_only",
        "--feature",
        feature,
        "--query_role",
        query_role,
        "--max_length",
        str(int(getattr(args, "max_length", 60))),
        "--iterations",
        str(int(getattr(args, "profile_iterations", 3))),
        "--evalue",
        str(float(getattr(args, "profile_evalue", 0.001))),
        "--inclusion-evalue",
        str(float(getattr(args, "profile_inclusion_evalue", 0.001))),
        "--threads",
        str(int(getattr(args, "profile_threads", getattr(args, "embedding_workers", 4)))),
        "--on_missing",
        (
            (
                "blosum62_hybrid"
                if profile_feature in (PSSM_HYBRID_FEATURE_TYPES | PSFM_HYBRID_FEATURE_TYPES)
                else effective_on_missing
            )
            if need_frequency
            else TOP_N_GRAM_STRICT_PROFILE_POLICY
        ),
        "--force",
    ]
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
    except Exception as e:
        print(f"[pair] profile cache build failed: {e}")
        return False

    ok_pssm = (not need_score_profile) or strict_cache_ready(
        pssm_index,
        profile_feature,
    )
    ok_psfm = (not need_psfm) or strict_cache_ready(
        psfm_index,
        profile_feature if need_psfm else FEATURE_PSFM,
    )
    ok_frequency = not need_frequency
    if need_frequency:
        try:
            store = TargetFrequencyProfileStore(
                frequency_index,
                max_length=int(getattr(args, "max_length", 60)),
            )
            provenance = store.manifest.get("provenance") or {}
            with open(train_csv, "rb") as handle:
                train_sha256 = hashlib.sha256(handle.read()).hexdigest()
            ok_frequency = frequency_cache_ready(
                store,
                provenance,
                train_sha256,
            )
            if ok_frequency:
                args._frequency_profile_manifest_sha256 = store.manifest_sha256
                args._frequency_profile_provenance = dict(provenance)
        except (OSError, ValueError, TypeError, TopNGramContractError):
            ok_frequency = False
    return bool(ok_pssm and ok_psfm and ok_frequency)


def ensure_subword_tokenizer_ready(args, task_root, train_rows, val_rows, sequence_role=None):
    tok = str(getattr(args, "tokenizer", "")).strip().lower()
    if tok not in ["bpe", "wordpiece", "unigram"]:
        return

    if tok == "bpe":
        default_name = (
            f"bpe_tokenizer_{sequence_role}.json" if sequence_role else "bpe_tokenizer.json"
        )
        target = _resolve_optional_path(
            project_root,
            getattr(args, "bpe_path", "")
            or os.path.join(task_root, "bpe_tokenizers", default_name),
        )
        args.bpe_path = target
        trainer = BPETokenizerTrainer(vocab_size=int(getattr(args, "bpe_vocab_size", 1000)))
    elif tok == "wordpiece":
        default_name = (
            f"wordpiece_tokenizer_{sequence_role}.json"
            if sequence_role
            else "wordpiece_tokenizer.json"
        )
        target = _resolve_optional_path(
            project_root,
            getattr(args, "wordpiece_path", "")
            or os.path.join(task_root, "wordpiece_tokenizers", default_name),
        )
        args.wordpiece_path = target
        trainer = WordPieceTokenizerTrainer(
            vocab_size=int(getattr(args, "wordpiece_vocab_size", 1000)),
            max_raw_length=validate_wordpiece_max_raw_length(
                getattr(
                    args,
                    "wordpiece_max_raw_length",
                    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
        )
    else:
        default_name = (
            f"unigram_tokenizer_{sequence_role}.json" if sequence_role else "unigram_tokenizer.json"
        )
        target = _resolve_optional_path(
            project_root,
            getattr(args, "unigram_path", "")
            or os.path.join(task_root, "unigram_tokenizers", default_name),
        )
        args.unigram_path = target
        trainer = UnigramTokenizerTrainer(
            vocab_size=validate_unigram_vocab_size(getattr(args, "unigram_vocab_size", 1000)),
            max_raw_length=validate_unigram_max_raw_length(
                getattr(
                    args,
                    "unigram_max_raw_length",
                    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
                )
            ),
        )

    if tok in {"bpe", "wordpiece", "unigram"}:
        if not train_rows:
            raise SystemExit(f"No rows available to build {tok} tokenizer.")
        sequences = []
        for _, peptide, partner, _ in train_rows:
            if sequence_role == "peptide":
                sequences.append(peptide)
            elif sequence_role in {"partner", "protein"}:
                sequences.append(partner)
            else:
                sequences.extend((peptide, partner))
        normalized_role = (
            "partner" if sequence_role in {"partner", "protein"} else (sequence_role or "both")
        )
        trainer.ensure_trained(
            sequences,
            target,
            sequence_role=normalized_role,
        )
        state = "reused" if trainer.last_cache_reused else "rebuilt"
        print(f"[pair] {tok} train-only cache {state}: {target}")
        return


def build_classic_featurizer(
    args,
    task_root,
    train_rows,
    sequence_role=None,
):
    raw_bpe = str(getattr(args, "bpe_path", "")).strip()
    if raw_bpe:
        bpe_path = _resolve_optional_path(project_root, raw_bpe)
    else:
        cand = [
            os.path.join(task_root, "bpe_tokenizers", "bpe_tokenizer.json"),
            os.path.join(project_root, "datasets", "bpe_tokenizer.json"),
        ]
        bpe_path = next((p for p in cand if os.path.exists(p)), cand[0])
    default_pssm_index = os.path.join(task_root, "pssm_cache", "ppi", "index.json")
    default_ac_pssm_blosum62_hybrid_index = os.path.join(
        task_root,
        "ac_pssm_blosum62_hybrid_cache",
        "ppi",
        "index.json",
    )
    default_cc_pssm_blosum62_hybrid_index = os.path.join(
        task_root,
        "cc_pssm_blosum62_hybrid_cache",
        "ppi",
        "index.json",
    )
    default_acc_pssm_blosum62_hybrid_index = os.path.join(
        task_root,
        "acc_pssm_blosum62_hybrid_cache",
        "ppi",
        "index.json",
    )
    default_psfm_index = os.path.join(task_root, "psfm_cache", "ppi", "index.json")
    default_frequency_index = os.path.join(
        task_root,
        "frequency_profile_cache",
        "ppi",
        "index.json",
    )
    unigram_path = _resolve_optional_path(
        project_root,
        getattr(args, "unigram_path", "")
        or os.path.join(task_root, "unigram_tokenizers", "unigram_tokenizer.json"),
    )
    wordpiece_path = _resolve_optional_path(
        project_root,
        getattr(args, "wordpiece_path", "")
        or os.path.join(task_root, "wordpiece_tokenizers", "wordpiece_tokenizer.json"),
    )
    w2v_target = _static_embedding_target(args, task_root, "word2vec")
    glove_target = _static_embedding_target(args, task_root, "glove")
    ft_target = _static_embedding_target(args, task_root, "fasttext")
    default_w2v = w2v_target[1] if w2v_target is not None else ""
    default_glove = glove_target[1] if glove_target is not None else ""
    default_ft = ft_target[1] if ft_target is not None else ""
    word2vec_path = _resolve_optional_path(
        project_root, getattr(args, "word2vec_path", "") or default_w2v
    )
    glove_path = _resolve_optional_path(
        project_root, getattr(args, "glove_path", "") or default_glove
    )
    fasttext_path = _resolve_optional_path(
        project_root, getattr(args, "fasttext_path", "") or default_ft
    )
    pssm_index_path = _resolve_optional_path(
        project_root, getattr(args, "pssm_index_path", "") or default_pssm_index
    )
    ac_pssm_blosum62_hybrid_index_path = _resolve_optional_path(
        project_root,
        getattr(args, "ac_pssm_blosum62_hybrid_index_path", "")
        or default_ac_pssm_blosum62_hybrid_index,
    )
    cc_pssm_blosum62_hybrid_index_path = _resolve_optional_path(
        project_root,
        getattr(args, "cc_pssm_blosum62_hybrid_index_path", "")
        or default_cc_pssm_blosum62_hybrid_index,
    )
    acc_pssm_blosum62_hybrid_index_path = _resolve_optional_path(
        project_root,
        getattr(args, "acc_pssm_blosum62_hybrid_index_path", "")
        or default_acc_pssm_blosum62_hybrid_index,
    )
    psfm_index_path = _resolve_optional_path(
        project_root, getattr(args, "psfm_index_path", "") or default_psfm_index
    )
    frequency_profile_index_path = _resolve_optional_path(
        project_root,
        getattr(args, "frequency_profile_index_path", "") or default_frequency_index,
    )
    feat_factory = FeaturizerFactory(
        bpe_path=bpe_path,
        unigram_path=unigram_path,
        wordpiece_path=wordpiece_path,
        word2vec_path=word2vec_path,
        glove_path=glove_path,
        fasttext_path=fasttext_path,
        pssm_index_path=pssm_index_path,
        psfm_index_path=psfm_index_path,
        ac_lag=int(getattr(args, "ac_lag", 10)),
        cc_lag=int(getattr(args, "cc_lag", 10)),
        acc_lag=int(getattr(args, "acc_lag", 10)),
        pdt_lag=int(getattr(args, "pdt_lag", PDT_DEFAULT_LAG)),
        ac_pssm_lag=int(getattr(args, "ac_pssm_lag", 10)),
        cc_pssm_lag=int(getattr(args, "cc_pssm_lag", 10)),
        acc_pssm_lag=int(getattr(args, "acc_pssm_lag", 10)),
        ac_pssm_blosum62_hybrid_index_path=(ac_pssm_blosum62_hybrid_index_path),
        cc_pssm_blosum62_hybrid_index_path=(cc_pssm_blosum62_hybrid_index_path),
        acc_pssm_blosum62_hybrid_index_path=(acc_pssm_blosum62_hybrid_index_path),
        ac_pssm_blosum62_hybrid_lag=int(getattr(args, "ac_pssm_blosum62_hybrid_lag", 10)),
        cc_pssm_blosum62_hybrid_lag=int(getattr(args, "cc_pssm_blosum62_hybrid_lag", 10)),
        acc_pssm_blosum62_hybrid_lag=int(getattr(args, "acc_pssm_blosum62_hybrid_lag", 10)),
        wordpiece_max_raw_length=validate_wordpiece_max_raw_length(
            getattr(
                args,
                "wordpiece_max_raw_length",
                WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
        unigram_max_raw_length=validate_unigram_max_raw_length(
            getattr(
                args,
                "unigram_max_raw_length",
                UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
        dr_max_raw_length=validate_dr_max_raw_length(
            getattr(
                args,
                "dr_max_raw_length",
                DR_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
        frequency_profile_index_path=frequency_profile_index_path,
        covariance_max_raw_length=validate_covariance_max_raw_length(
            getattr(
                args,
                "covariance_max_raw_length",
                COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
        pdt_max_raw_length=validate_pdt_max_raw_length(
            getattr(
                args,
                "pdt_max_raw_length",
                PDT_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
    )
    if sequence_role == "peptide":
        sequences = [row[1] for row in train_rows]
    elif sequence_role in {"partner", "protein"}:
        sequences = [row[2] for row in train_rows]
    else:
        sequences = [row[1] for row in train_rows] + [row[2] for row in train_rows]
    profile_samples = None
    if str(getattr(args, "tokenizer", "")).strip().lower() in {"top-n-gram", "dt"}:
        try:
            if str(getattr(args, "tokenizer", "")).strip().lower() == "top-n-gram":
                validate_top_ngram_n(getattr(args, "top_ngram_n", 2))
            else:
                validate_dt_max_distance(
                    getattr(
                        args,
                        "dt_max_distance",
                        DT_DEFAULT_MAX_DISTANCE,
                    )
                )
        except (TopNGramContractError, DTContractError) as exc:
            raise SystemExit(str(exc)) from exc
        if sequence_role == "peptide":
            profile_samples = [
                (f"{sample_id}::pep", peptide)
                for sample_id, peptide, _partner, _label in train_rows
            ]
        elif sequence_role in {"partner", "protein"}:
            profile_samples = [
                (f"{sample_id}::pro", partner)
                for sample_id, _peptide, partner, _label in train_rows
            ]
        else:
            profile_samples = [
                (f"{sample_id}::pep", peptide)
                for sample_id, peptide, _partner, _label in train_rows
            ] + [
                (f"{sample_id}::pro", partner)
                for sample_id, _peptide, partner, _label in train_rows
            ]
    effective_encoder = str(getattr(args, "encoder", "index")).strip().lower()
    if effective_encoder == "ppi_fusion":
        effective_encoder = str(getattr(args, "aux_encoder", "index")).strip().lower()
    featurizer = feat_factory.create(
        getattr(args, "tokenizer", "kmer"),
        effective_encoder,
        int(getattr(args, "max_length", 60)),
        int(getattr(args, "k", 2)),
        sequences=sequences,
        dr_max_distance=validate_dr_max_distance(
            getattr(args, "dr_max_distance", DR_DEFAULT_MAX_DISTANCE)
        ),
        dr_unknown_residue_policy=validate_dr_unknown_residue_policy(
            getattr(
                args,
                "dr_unknown_residue_policy",
                DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
            )
        ),
        top_ngram_n=int(getattr(args, "top_ngram_n", 2)),
        dt_max_distance=validate_dt_max_distance(
            getattr(args, "dt_max_distance", DT_DEFAULT_MAX_DISTANCE)
        ),
        profile_samples=profile_samples,
        frequency_contract_version=FREQUENCY_ENCODER_CONTRACT_VERSION,
        frequency_max_raw_length=validate_frequency_max_raw_length(
            getattr(
                args,
                "frequency_max_raw_length",
                FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
            )
        ),
    )
    architecture = (
        str(getattr(args, "sequence_model", "") or getattr(args, "model", "") or "").strip().lower()
    )
    if architecture in {"rnn", "gru", "lstm", "transformer", "gcn", "gat"}:
        featurizer.require_explicit_position_mask()
    if effective_encoder == "tfidf" and hasattr(getattr(featurizer, "encoder", None), "idf"):
        args._classic_tfidf_idf = featurizer.encoder.idf.detach().cpu().float()
    if hasattr(featurizer, "frequency_encoder_contract"):
        contract = dict(featurizer.frequency_encoder_contract)
        if effective_encoder in {"bow", "tfidf"}:
            corpus_identity = sequences
            if profile_samples is not None:
                corpus_identity = [
                    featurizer.tokenizer.profile_store.manifest_sha256,
                    *profile_samples,
                ]
            contract["training_corpus_sha256"] = digest_text_items(corpus_identity)
            contract["training_document_count"] = len(sequences)
            contract["fit_scope"] = (
                "combined_pair_roles"
                if sequence_role not in {"peptide", "partner", "protein"}
                else f"{sequence_role}_role"
            )
        args._frequency_encoder_contract = contract
    if hasattr(featurizer, "textrank_contract"):
        args._textrank_contract = dict(featurizer.textrank_contract)
    if hasattr(featurizer, "covariance_encoder_contract"):
        args._covariance_encoder_contract = dict(featurizer.covariance_encoder_contract)
    if hasattr(featurizer, "pdt_encoder_contract"):
        args._pdt_encoder_contract = dict(featurizer.pdt_encoder_contract)
    if hasattr(featurizer, "pssm_covariance_encoder_contract"):
        args._pssm_covariance_encoder_contract = dict(featurizer.pssm_covariance_encoder_contract)
    if str(getattr(args, "tokenizer", "")).strip().lower() in {"top-n-gram", "dt"}:
        args.frequency_profile_index_path = frequency_profile_index_path
        args._frequency_profile_manifest_sha256 = featurizer.tokenizer.profile_store.manifest_sha256
    return featurizer


def _path_fingerprint(path):
    if not path:
        return "none"
    normalized = os.path.abspath(path)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]


def _cache_meta_path(save_path):
    return f"{save_path}.meta.json"


@dataclass(frozen=True)
class CompactResidueEncoding:
    """Deduplicated ragged residue states for a single Pair role."""

    flat_features: torch.Tensor
    offsets: torch.Tensor
    row_to_unique: torch.Tensor
    max_length: int
    feature_dim: int

    @property
    def shape(self):
        return (int(self.row_to_unique.numel()), int(self.max_length), int(self.feature_dim))

    @property
    def ndim(self):
        return 3

    def materialize_indices(self, indices):
        """Reconstruct a small explicit-mask batch from compact role storage."""

        index = torch.as_tensor(indices, dtype=torch.int64, device="cpu").reshape(-1)
        if bool((index < 0).any()) or bool((index >= self.shape[0]).any()):
            raise IndexError("compact residue row index is out of range")
        batch_size = int(index.numel())
        features = torch.zeros(
            (batch_size, int(self.max_length), int(self.feature_dim)),
            dtype=self.flat_features.dtype,
        )
        mask = torch.zeros((batch_size, int(self.max_length)), dtype=torch.bool)
        if batch_size == 0:
            return ResidueModelInput(features, mask)

        unique_rows = self.row_to_unique.index_select(0, index)
        starts = self.offsets.index_select(0, unique_rows)
        lengths = self.offsets.index_select(0, unique_rows + 1) - starts
        total = int(lengths.sum())
        batch_rows = torch.arange(batch_size, dtype=torch.int64)
        feature_rows = torch.repeat_interleave(batch_rows, lengths)
        prefixes = torch.cumsum(lengths, dim=0) - lengths
        positions = torch.arange(total, dtype=torch.int64) - torch.repeat_interleave(
            prefixes, lengths
        )
        source_rows = torch.repeat_interleave(starts, lengths) + positions
        features[feature_rows, positions] = self.flat_features[source_rows]
        mask[feature_rows, positions] = True
        return ResidueModelInput(features, mask)

    def batch_slice(self, start, end):
        return self.materialize_indices(torch.arange(int(start), int(end)))

    def index_select(self, dim, index):
        if int(dim) != 0:
            raise ValueError("compact residue encoding only supports row selection")
        return self.materialize_indices(index)


@dataclass(frozen=True)
class CompactSingleStreamEncoding:
    """Lazy joined Pair input backed by two compact residue-role caches."""

    peptide: CompactResidueEncoding
    partner: CompactResidueEncoding
    split_name: str

    def __post_init__(self):
        if self.peptide.shape[0] != self.partner.shape[0]:
            raise ValueError("compact Pair roles have different row counts")
        if (
            self.peptide.max_length != self.partner.max_length
            or self.peptide.feature_dim != self.partner.feature_dim
        ):
            raise ValueError("compact Pair roles have incompatible residue contracts")

    @property
    def shape(self):
        return (
            int(self.peptide.shape[0]),
            int(self.peptide.max_length + self.partner.max_length + 1),
            int(self.peptide.feature_dim + 3),
        )

    @property
    def ndim(self):
        return 3

    def __len__(self):
        return int(self.shape[0])

    def materialize_indices(self, indices):
        index = torch.as_tensor(indices, dtype=torch.int64, device="cpu").reshape(-1)
        return join_single_stream_batches(
            self.peptide.materialize_indices(index),
            self.partner.materialize_indices(index),
            separator_id=None,
        )

    def batch_slice(self, start, end):
        return self.materialize_indices(torch.arange(int(start), int(end)))

    def index_select(self, dim, index):
        if int(dim) != 0:
            raise ValueError("compact Single-stream encoding only supports row selection")
        return self.materialize_indices(index)


def _is_explicit_residue_input(value):
    return isinstance(value, (ResidueModelInput, CompactSingleStreamEncoding))


def _valid_compact_residue_encoding(value, *, row_count, max_length, feature_dim):
    if not isinstance(value, CompactResidueEncoding):
        return False
    flat = value.flat_features
    offsets = value.offsets
    row_map = value.row_to_unique
    if (
        not isinstance(flat, torch.Tensor)
        or flat.ndim != 2
        or not flat.is_floating_point()
        or int(flat.shape[1]) != int(feature_dim)
        or not isinstance(offsets, torch.Tensor)
        or offsets.ndim != 1
        or offsets.dtype != torch.int64
        or offsets.numel() < 2
        or not isinstance(row_map, torch.Tensor)
        or row_map.ndim != 1
        or row_map.dtype != torch.int64
        or int(row_map.numel()) != int(row_count)
        or int(value.max_length) != int(max_length)
        or int(value.feature_dim) != int(feature_dim)
    ):
        return False
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(flat.shape[0]):
        return False
    lengths = offsets[1:] - offsets[:-1]
    if bool((lengths <= 0).any()) or bool((lengths > int(max_length)).any()):
        return False
    if bool((row_map < 0).any()) or bool((row_map >= offsets.numel() - 1).any()):
        return False
    return bool(torch.isfinite(flat).all())


def _feature_input_digest(items):
    digest = hashlib.sha256()
    for item in items:
        if isinstance(item, (list, tuple)):
            parts = [str(part) for part in item]
        else:
            parts = [str(item)]
        for part in parts:
            encoded = part.encode("utf-8", errors="replace")
            digest.update(str(len(encoded)).encode("ascii"))
            digest.update(b":")
            digest.update(encoded)
            digest.update(b"\x1f")
        digest.update(b"\x1e")
    return digest.hexdigest()


def _feature_cache_meta(
    args, split_name, role_name, input_items, *, encoder_name=None, target_len=None
):
    meta = {
        "cache_version": CACHE_PROVENANCE_VERSION,
        "feature_fit_scope": "train_only",
        "split": str(split_name),
        "role": str(role_name),
        "tokenizer": str(getattr(args, "tokenizer", "")).strip().lower(),
        "encoder": str(encoder_name or getattr(args, "encoder", "")).strip().lower(),
        "max_length": int(getattr(args, "max_length", 60)),
        "row_count": len(input_items),
        "input_sha256": _feature_input_digest(input_items),
        "train_sha256": str(getattr(args, "_feature_train_sha256", "") or ""),
    }
    if meta["tokenizer"] == "prott5":
        meta.update(
            {
                "pooling": PROTT5_POOLING,
                "representation_contract_version": PROTT5_REPRESENTATION_CONTRACT_VERSION,
                "output_kind": "residue_token_hidden"
                if target_len is not None
                else "sequence_embedding",
                "special_tokens": "excluded",
            }
        )
    if not meta["train_sha256"]:
        raise ValueError("feature cache identity is missing the Train split digest")
    if meta["tokenizer"] in {"kmer", "dr", "top-n-gram", "dt"} and meta["encoder"] in {
        "bow",
        "tfidf",
    }:
        meta["frequency_contract_version"] = FREQUENCY_ENCODER_CONTRACT_VERSION
        meta["frequency_max_raw_length"] = validate_frequency_max_raw_length(
            getattr(
                args,
                "frequency_max_raw_length",
                FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    if meta["tokenizer"] == "dr":
        max_distance = validate_dr_max_distance(
            getattr(args, "dr_max_distance", DR_DEFAULT_MAX_DISTANCE)
        )
        max_raw_length = validate_dr_max_raw_length(
            (
                getattr(
                    args,
                    "frequency_max_raw_length",
                    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
                )
                if meta["encoder"] in {"bow", "tfidf"}
                else getattr(
                    args,
                    "dr_max_raw_length",
                    DR_DEFAULT_MAX_RAW_LENGTH,
                )
            )
        )
        meta["dr_contract"] = dr_contract_metadata(
            max_distance=max_distance,
            max_raw_length=max_raw_length,
            unknown_residue_policy=getattr(
                args,
                "dr_unknown_residue_policy",
                DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
            ),
        )
    elif meta["encoder"] in COVARIANCE_ENCODERS:
        contract = getattr(args, "_covariance_encoder_contract", None)
        if not isinstance(contract, dict):
            raise ValueError("covariance feature cache identity is missing its encoder contract")
        meta["covariance_encoder_contract"] = dict(contract)
    elif meta["encoder"] in PDT_ENCODERS:
        contract = getattr(args, "_pdt_encoder_contract", None)
        if not isinstance(contract, dict):
            raise ValueError("PDT feature cache identity is missing its encoder contract")
        meta["pdt_encoder_contract"] = dict(contract)
    elif meta["encoder"] in PSSM_COVARIANCE_ENCODERS:
        contract = getattr(
            args,
            "_pssm_covariance_encoder_contract",
            None,
        )
        if not isinstance(contract, dict):
            raise ValueError(
                "PSSM covariance feature cache identity is missing its encoder contract"
            )
        manifest_sha256 = str(getattr(args, "_profile_manifest_sha256", "") or "")
        if not manifest_sha256:
            raise ValueError(
                "PSSM covariance feature cache identity is missing the PSSM profile manifest digest"
            )
        meta["pssm_covariance_encoder_contract"] = dict(contract)
        meta["profile_manifest_sha256"] = manifest_sha256
    elif meta["tokenizer"] == "top-n-gram":
        top_n = validate_top_ngram_n(getattr(args, "top_ngram_n", 2))
        manifest_sha256 = str(
            getattr(
                args,
                "_frequency_profile_manifest_sha256",
                "",
            )
            or ""
        )
        if not manifest_sha256:
            raise ValueError(
                "Top-n-gram feature cache identity is missing the frequency-profile manifest digest"
            )
        meta["top_ngram_contract"] = {
            "contract_version": TOP_N_GRAM_CONTRACT_VERSION,
            "top_n": top_n,
            "feature_dim": top_ngram_feature_dim(top_n),
            "profile_type": "target_frequency",
            "tie_policy": TOP_N_GRAM_TIE_POLICY,
            "full_length": True,
            "profile_manifest_sha256": manifest_sha256,
            "fallback_profiles_allowed": (
                str(
                    getattr(
                        args,
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
                        args,
                        "profile_on_missing",
                        TOP_N_GRAM_FALLBACK_POLICY,
                    )
                )
                == TOP_N_GRAM_FALLBACK_POLICY
                else TOP_N_GRAM_STRICT_PROFILE_POLICY
            ),
        }
    elif meta["tokenizer"] == "dt":
        max_distance = validate_dt_max_distance(
            getattr(args, "dt_max_distance", DT_DEFAULT_MAX_DISTANCE)
        )
        manifest_sha256 = str(
            getattr(
                args,
                "_frequency_profile_manifest_sha256",
                "",
            )
            or ""
        )
        if not manifest_sha256:
            raise ValueError(
                "DT feature cache identity is missing the frequency-profile manifest digest"
            )
        contract = dt_contract_metadata(max_distance=max_distance)
        contract.update(
            {
                "profile_manifest_sha256": manifest_sha256,
                "fallback_profiles_allowed": (
                    str(
                        getattr(
                            args,
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
                            args,
                            "profile_on_missing",
                            TOP_N_GRAM_FALLBACK_POLICY,
                        )
                    )
                    == TOP_N_GRAM_FALLBACK_POLICY
                    else TOP_N_GRAM_STRICT_PROFILE_POLICY
                ),
            }
        )
        meta["dt_contract"] = contract
    model_source = str(getattr(args, "_resolved_pretrained_model_source", "") or "")
    if model_source:
        meta["model_source"] = model_source
        meta["model_fingerprint"] = model_source_fingerprint(model_source)
        model_identity = _resolved_pair_pretrained_model_identity(args)
        if model_identity:
            meta["model_identity"] = model_identity
        meta["pooling"] = (
            str(model_identity.get("pooling"))
            if model_identity
            else (
                ESM3_POOLING
                if meta["tokenizer"] == "esm3"
                else (
                    PROTBERT_POOLING if meta["tokenizer"] == "protbert" else "attention_mask_mean"
                )
            )
        )
        if meta["tokenizer"] == "esm2":
            meta["representation_contract_version"] = ESM2_REPRESENTATION_CONTRACT_VERSION
            meta["pooling"] = "residue_tokens" if target_len is not None else ESM2_POOLING
            meta["output_kind"] = (
                "residue_token_hidden" if target_len is not None else "sequence_embedding"
            )
        if meta["tokenizer"] == "protbert" and model_identity:
            meta["representation_contract_version"] = PROTBERT_REPRESENTATION_CONTRACT_VERSION
            meta["pooling"] = "residue_tokens" if target_len is not None else PROTBERT_POOLING
            meta["output_kind"] = (
                "residue_token_hidden" if target_len is not None else "sequence_embedding"
            )
        if meta["tokenizer"] == "prott5" and model_identity:
            meta["representation_contract_version"] = PROTT5_REPRESENTATION_CONTRACT_VERSION
            meta["pooling"] = "residue_tokens" if target_len is not None else PROTT5_POOLING
            meta["output_kind"] = (
                "residue_token_hidden" if target_len is not None else "sequence_embedding"
            )
            meta["special_tokens"] = "excluded"
            meta["input_normalization"] = "strip_whitespace_uppercase_UZOB_to_X"
        if meta["tokenizer"] == "esm3" and model_identity:
            meta["representation_contract_version"] = ESM3_REPRESENTATION_CONTRACT_VERSION
            meta["pooling"] = "residue_tokens" if target_len is not None else ESM3_POOLING
            meta["output_kind"] = (
                "residue_token_hidden" if target_len is not None else "sequence_embedding"
            )
            meta["special_tokens"] = "excluded"
            meta["input_normalization"] = "strip_whitespace_uppercase_validate_alphabet"
        runtime_environment = _resolved_pair_pretrained_runtime_environment(args)
        if (
            meta["tokenizer"] in {"esm2", "protbert", "prott5", "esmc", "esm3"}
            and runtime_environment
        ):
            meta["runtime_environment"] = runtime_environment
    if target_len is not None:
        meta["target_len"] = int(target_len)
    return meta


_PRETRAINED_POOLED_TOKENIZERS = {
    "esm2",
    "protbert",
    "prott5",
    "esmc",
    "esm3",
}
_MODEL_IDENTITY_PATH_KEYS = {
    "model_source",
    "requested_source",
    "resolved_source",
    "weight_path",
}
_MODEL_IDENTITY_RUNTIME_KEYS = {
    "runtime_package",
    "runtime_required_version",
    "runtime_version",
}
_PROTT5_V3_KEYS = {
    "cache_version",
    "encoder",
    "feature_fit_scope",
    "input_sha256",
    "max_length",
    "model_fingerprint",
    "model_source",
    "pooling",
    "role",
    "row_count",
    "split",
    "tokenizer",
    "train_sha256",
}
_PRETRAINED_SCIENTIFIC_IDENTITY_KEYS = {
    "feature_fit_scope",
    "split",
    "role",
    "row_count",
    "input_sha256",
    "train_sha256",
    "max_length",
    "tokenizer",
    "encoder",
    "pooling",
    "model_fingerprint",
}


def _is_pretrained_pooled_cache_metadata(metadata):
    if not isinstance(metadata, dict):
        return False
    tokenizer = str(metadata.get("tokenizer", "")).strip().lower()
    if tokenizer not in _PRETRAINED_POOLED_TOKENIZERS:
        return False
    if not str(metadata.get("model_fingerprint", "") or ""):
        return False
    if "target_len" in metadata:
        return False
    output_kind = str(metadata.get("output_kind", "") or "").strip().lower()
    return output_kind in {"", "sequence_embedding"}


def _scientific_model_identity(value):
    if not isinstance(value, dict):
        return value
    result = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        normalized = key.strip().lower()
        if (
            normalized in _MODEL_IDENTITY_PATH_KEYS
            or normalized.endswith("_path")
            or normalized in _MODEL_IDENTITY_RUNTIME_KEYS
            or normalized == "runtime_environment"
        ):
            continue
        if isinstance(raw_value, dict):
            result[key] = _scientific_model_identity(raw_value)
        elif isinstance(raw_value, list):
            result[key] = [
                _scientific_model_identity(item) if isinstance(item, dict) else item
                for item in raw_value
            ]
        else:
            result[key] = raw_value
    return result


def _scientific_pretrained_cache_metadata(metadata):
    normalized = dict(metadata)
    normalized.pop("model_source", None)
    normalized.pop("runtime_environment", None)
    if "model_identity" in normalized:
        normalized["model_identity"] = _scientific_model_identity(normalized["model_identity"])
    return normalized


def _prott5_v3_pooled_cache_metadata_compatible(actual_meta, expected_meta):
    if set(actual_meta) != _PROTT5_V3_KEYS:
        return False
    if actual_meta.get("cache_version") != 3:
        return False
    if expected_meta.get("cache_version") != CACHE_PROVENANCE_VERSION:
        return False
    if str(actual_meta.get("tokenizer", "")).lower() != "prott5":
        return False
    if str(actual_meta.get("encoder", "")).lower() != "prott5":
        return False
    if str(expected_meta.get("tokenizer", "")).lower() != "prott5":
        return False
    if str(expected_meta.get("encoder", "")).lower() != "prott5":
        return False
    for key in _PRETRAINED_SCIENTIFIC_IDENTITY_KEYS - {"pooling"}:
        if actual_meta.get(key) != expected_meta.get(key):
            return False
    if actual_meta.get("pooling") != "attention_mask_mean":
        return False
    if expected_meta.get("pooling") != PROTT5_POOLING:
        return False
    if (
        expected_meta.get("representation_contract_version")
        != PROTT5_REPRESENTATION_CONTRACT_VERSION
        or expected_meta.get("output_kind") != "sequence_embedding"
        or expected_meta.get("special_tokens") != "excluded"
        or expected_meta.get("input_normalization") != "strip_whitespace_uppercase_UZOB_to_X"
    ):
        return False
    identity = expected_meta.get("model_identity")
    if not isinstance(identity, dict):
        return False
    if identity.get("model_fingerprint") != expected_meta.get("model_fingerprint"):
        return False
    return int(identity.get("d_model", 0) or 0) == PROTT5_D_MODEL


def _feature_cache_metadata_compatible(actual_meta, expected_meta):
    if actual_meta == expected_meta:
        return True
    if not (
        _is_pretrained_pooled_cache_metadata(actual_meta)
        and _is_pretrained_pooled_cache_metadata(expected_meta)
    ):
        return False
    if actual_meta.get("cache_version") == 3 or expected_meta.get("cache_version") == 3:
        return _prott5_v3_pooled_cache_metadata_compatible(
            actual_meta,
            expected_meta,
        )
    if (
        actual_meta.get("cache_version") != CACHE_PROVENANCE_VERSION
        or expected_meta.get("cache_version") != CACHE_PROVENANCE_VERSION
    ):
        return False
    return _scientific_pretrained_cache_metadata(
        actual_meta
    ) == _scientific_pretrained_cache_metadata(expected_meta)


def _valid_pretrained_pooled_cache_tensor(value, *, row_count, output_dim):
    return bool(
        isinstance(value, torch.Tensor)
        and value.ndim == 2
        and value.is_floating_point()
        and int(value.shape[0]) == int(row_count)
        and int(value.shape[1]) == int(output_dim)
        and torch.isfinite(value).all()
    )


def _load_feature_cache(save_path, expected_meta, validator):
    if not os.path.exists(save_path) or not os.path.exists(_cache_meta_path(save_path)):
        return None
    try:
        with open(_cache_meta_path(save_path), "r", encoding="utf-8") as handle:
            actual_meta = json.load(handle)
    except Exception:
        return None
    if not _feature_cache_metadata_compatible(actual_meta, expected_meta):
        print(f"[feature] ignore stale cache metadata: {save_path}", flush=True)
        return None
    try:
        x = torch.load(
            save_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        try:
            x = torch.load(save_path, map_location="cpu", mmap=True)
        except (TypeError, ValueError):
            x = torch.load(save_path, map_location="cpu")
    if isinstance(x, dict) and x.get("kind") == "ResidueModelInput":
        features = x.get("features")
        mask = x.get("mask")
        if isinstance(features, torch.Tensor) and isinstance(mask, torch.Tensor):
            x = ResidueModelInput(features, mask.bool())
    elif isinstance(x, dict) and x.get("kind") == "CompactResidueEncoding":
        try:
            x = CompactResidueEncoding(
                flat_features=x["flat_features"],
                offsets=x["offsets"].long(),
                row_to_unique=x["row_to_unique"].long(),
                max_length=int(x["max_length"]),
                feature_dim=int(x["feature_dim"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            x = None
    if validator(x):
        return x
    print(f"[feature] ignore incompatible cache tensor: {save_path}", flush=True)
    return None


def _save_feature_cache(save_path, tensor, metadata):
    if isinstance(tensor, CompactResidueEncoding):
        payload = {
            "kind": "CompactResidueEncoding",
            "flat_features": tensor.flat_features,
            "offsets": tensor.offsets,
            "row_to_unique": tensor.row_to_unique,
            "max_length": int(tensor.max_length),
            "feature_dim": int(tensor.feature_dim),
        }
    else:
        payload = (
            {
                "kind": "ResidueModelInput",
                "features": tensor.features,
                "mask": tensor.mask.bool(),
            }
            if isinstance(tensor, ResidueModelInput)
            else tensor
        )
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tensor_tmp = Path(handle.name)
    meta_destination = Path(_cache_meta_path(save_path))
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=meta_destination.parent,
        prefix=meta_destination.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        meta_tmp = Path(handle.name)
    try:
        torch.save(payload, tensor_tmp)
        os.replace(tensor_tmp, destination)
        os.replace(meta_tmp, meta_destination)
    finally:
        for temporary in (tensor_tmp, meta_tmp):
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def classic_feature_cache_tag(args, encoder_name, role_name=None):
    tok = str(getattr(args, "tokenizer", "classic")).strip().lower()
    enc = str(encoder_name or getattr(args, "encoder", "index")).strip().lower()
    if enc == "ppi_fusion":
        enc = str(getattr(args, "aux_encoder", "index")).strip().lower()

    parts = [
        tok.replace("-", ""),
        enc.replace("-", ""),
        f"len{int(getattr(args, 'max_length', 60))}",
    ]
    if tok in {"kmer", "dr", "top-n-gram", "dt"} and enc in {"bow", "tfidf"}:
        parts.extend(
            [
                "freqv2",
                "raw"
                + str(
                    validate_frequency_max_raw_length(
                        getattr(
                            args,
                            "frequency_max_raw_length",
                            FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
                        )
                    )
                ),
            ]
        )
    if tok == "kmer":
        parts.append(f"k{int(getattr(args, 'k', 2))}")
    if tok == "dr":
        parts.append("v2")
        parts.append(f"d{int(getattr(args, 'dr_max_distance', 5))}")
        parts.append(
            f"raw{validate_dr_max_raw_length(getattr(args, 'dr_max_raw_length', DR_DEFAULT_MAX_RAW_LENGTH))}"
        )
    if tok == "top-n-gram":
        parts.append("v3")
        parts.append(f"top{int(getattr(args, 'top_ngram_n', 2))}")
        parts.append(
            "freq"
            + str(
                getattr(
                    args,
                    "_frequency_profile_manifest_sha256",
                    "missing",
                )
            )[:10]
        )
    if tok == "dt":
        parts.append("v1")
        parts.append(
            f"d{validate_dt_max_distance(getattr(args, 'dt_max_distance', DT_DEFAULT_MAX_DISTANCE))}"
        )
        parts.append(
            "freq"
            + str(
                getattr(
                    args,
                    "_frequency_profile_manifest_sha256",
                    "missing",
                )
            )[:10]
        )
    if tok in ["bpe", "wordpiece", "unigram"]:
        parts.append(f"tok{_path_fingerprint(getattr(args, f'{tok}_path', ''))}")

    if enc in PROFILE_ENCODERS - {"psfm", "psfm_blosum62_hybrid"}:
        profile_feature = _profile_feature_for_encoder(enc)
        index_key = _profile_index_key(profile_feature)
        parts.append(f"pssm{_path_fingerprint(getattr(args, index_key, ''))}")
    if enc == "psfm":
        parts.append(f"psfm{_path_fingerprint(getattr(args, 'psfm_index_path', ''))}")
    if enc == "psfm_blosum62_hybrid":
        parts.append(f"psfmhybrid{_path_fingerprint(getattr(args, 'psfm_index_path', ''))}")
    if enc in ["ac", "cc", "acc", "pdt"]:
        parts.append(f"lag{int(getattr(args, f'{enc}_lag', 10))}")
    if enc in PSSM_COVARIANCE_ENCODERS:
        parts.append(f"lag{int(getattr(args, pssm_covariance_lag_key(enc), 10))}")
        parts.append(f"contractv{PSSM_COVARIANCE_ENCODER_CONTRACT_VERSION}")
        parts.append("manifest" + str(getattr(args, "_profile_manifest_sha256", "missing"))[:10])
    if enc in ["word2vec", "glove", "fasttext"]:
        parts.append(f"emb{_path_fingerprint(getattr(args, f'{enc}_path', ''))}")

    return "_".join(parts)


def extract_classic_features(args, featurizer, rows, split_name, role_name, encoder_name=None):
    cache_enabled = bool(getattr(args, "cache_features", True))
    save_path = ""
    enc_name = str(encoder_name or getattr(args, "encoder", "index")).lower()
    tok_name = str(getattr(args, "tokenizer", "")).lower()
    input_items = [(sid, pep if role_name == "peptide" else pro) for sid, pep, pro, _ in rows]
    expected_meta = _feature_cache_meta(
        args, split_name, role_name, input_items, encoder_name=enc_name
    )
    if tok_name == "dr":
        try:
            matrix_bytes = estimate_dr_dense_tensor_bytes(
                len(rows),
                encoder=enc_name,
                max_distance=getattr(
                    args,
                    "dr_max_distance",
                    DR_DEFAULT_MAX_DISTANCE,
                ),
                max_length=int(getattr(args, "max_length", 60)),
                embedding_dim=getattr(
                    getattr(featurizer, "encoder", None),
                    "feature_dim",
                    None,
                ),
            )
            ensure_dr_dense_memory_budget(
                matrix_bytes * 6,
                limit_gib=getattr(
                    args,
                    "dr_dense_memory_limit_gib",
                    DR_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                ),
                context=(
                    "DR pair feature matrix plus concurrent role/split/fusion working tensors"
                ),
            )
        except DRContractError as exc:
            raise SystemExit(str(exc)) from exc
    if tok_name == "top-n-gram":
        try:
            matrix_bytes = estimate_top_ngram_dense_bytes(
                len(rows),
                getattr(args, "top_ngram_n", 2),
            )
            ensure_top_ngram_dense_memory_budget(
                matrix_bytes * 6,
                limit_gib=getattr(
                    args,
                    "topngram_dense_memory_limit_gib",
                    TOP_N_GRAM_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                ),
                context=(
                    "Top-n-gram pair feature matrix plus concurrent "
                    "role/split/fusion working tensors"
                ),
            )
        except TopNGramContractError as exc:
            raise SystemExit(str(exc)) from exc
    if tok_name == "dt":
        try:
            matrix_bytes = estimate_dt_dense_bytes(
                len(rows),
                max_distance=getattr(
                    args,
                    "dt_max_distance",
                    DT_DEFAULT_MAX_DISTANCE,
                ),
            )
            ensure_dt_dense_memory_budget(
                matrix_bytes * 6,
                limit_gib=getattr(
                    args,
                    "dt_dense_memory_limit_gib",
                    DT_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                ),
                context=(
                    "DT pair feature matrix plus concurrent role/split/fusion working tensors"
                ),
            )
        except DTContractError as exc:
            raise SystemExit(str(exc)) from exc
    if cache_enabled:
        save_dir = os.path.join(
            args.feature_dir, classic_feature_cache_tag(args, enc_name, role_name=role_name)
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{split_name}_{role_name}.pt")
        x = _load_feature_cache(
            save_path,
            expected_meta,
            lambda item: (
                isinstance(item, torch.Tensor) and item.ndim == 2 and item.shape[0] == len(rows)
            ),
        )
        if x is not None:
            return x

    feats = []
    iterator = tqdm(rows, desc=f"Encoding {split_name}/{role_name}", leave=False)
    for sid, pep, pro, _ in iterator:
        seq = pep if role_name == "peptide" else pro
        sample_key = sid
        if enc_name in PROFILE_ENCODERS or tok_name in ["top-n-gram", "dt"]:
            suffix = "pep" if role_name == "peptide" else "pro"
            sample_key = f"{sid}::{suffix}"
        ft = featurizer.encode(seq, sample_id=sample_key)
        if isinstance(ft, ResidueModelInput):
            ft = ft.features
        if not isinstance(ft, torch.Tensor):
            ft = torch.tensor(ft)
        ft = ft.reshape(-1)
        feats.append(ft.long() if enc_name == "index" else ft.float())
    x = torch.stack(feats, dim=0) if feats else torch.empty((0, 0), dtype=torch.float)
    if cache_enabled:
        _save_feature_cache(save_path, x, expected_meta)
    return x


def resolve_fusion_mode(raw_mode):
    try:
        return normalize_pair_fusion_mode(raw_mode)
    except PairFusionError as exc:
        raise SystemExit(str(exc)) from exc


def normalize_route_fusion_mode(args):
    raw_mode = getattr(args, "fusion_mode", None)
    if str(getattr(args, "model", "") or "").strip().lower() != HGT_MODEL:
        args.fusion_mode = resolve_fusion_mode(raw_mode)
        return args.fusion_mode
    if raw_mode not in {None, ""}:
        legacy_mode = resolve_fusion_mode(raw_mode)
        if legacy_mode not in {"concat", "concat_all"}:
            raise SystemExit("HGT uses typed role projection and does not accept a fusion mode")
        print(
            f"[route-warning] HGT ignores legacy fusion_mode={legacy_mode}; "
            "new configurations use typed role projection."
        )
    args.fusion_mode = None
    args.pair_input_adapter = "typed_role_projection"
    return None


def fuse_embeddings(pep, pro, mode="concat_all"):
    mode = resolve_fusion_mode(mode)
    try:
        return fuse_audited_pair_embeddings(pep, pro, mode)
    except PairFusionError as exc:
        hint = ""
        if exc.code == "pair_fusion_dimension_mismatch":
            hint = " Use fusion_mode=concat or aligned_gated for heterogeneous streams."
        raise SystemExit(f"{exc}{hint}") from exc


def rows_to_tensors(rows):
    peps = [r[1] for r in rows]
    pros = [r[2] for r in rows]
    y = torch.tensor([[float(r[3])] for r in rows], dtype=torch.float)
    return peps, pros, y


# Unified v1 semantic layer shared by every supported pair task.
def interaction_semantic_enabled(args):
    task = normalize_task_type(getattr(args, "task_type", "pep_protein_interaction"))
    model = str(getattr(args, "model", "mlp")).strip().lower()
    training_mode = str(getattr(args, "training_mode", "") or "").strip().lower()
    classifier = str(getattr(args, "classifier", "") or "").strip().lower()
    return (
        normalize_task_type(task) in PAIR_INFERENCE_TASKS
        and model in SUPPORTED_PAIR_SEMANTIC_MODELS
        and model not in GRAPH_MODELS
        and training_mode != "export_embeddings"
        and classifier not in CLASSICAL_CLASSIFIERS
        and semantic_is_enabled(args)
    )


def _pair_residual_model(model, args):
    """Wrap a neural pair model with the residual adapter when configured."""

    residual_weight = float(getattr(args, "semantic_residual_weight", 0.0))
    if residual_weight <= 0.0:
        return model
    if not interaction_semantic_enabled(args):
        raise SystemExit("semantic_residual_weight requires the unified pair semantic layer")
    learned_gate = bool(getattr(args, "semantic_learned_gate", False))
    if hasattr(model, "forward_with_role_embeddings"):
        return PairSemanticResidualAdapter(
            model,
            residual_weight,
            learned_gate=learned_gate,
        )
    if hasattr(model, "head") and hasattr(model, "semantic_embedding_dim"):
        if learned_gate:
            raise SystemExit(
                "semantic_learned_gate is only supported for pair models with "
                "forward_with_role_embeddings"
            )
        return SemanticResidualAdapter(model, residual_weight)
    raise SystemExit("semantic_residual_weight is unsupported for this pair model")


class _EpochSampledPairDataset(Dataset):
    def __init__(self, tensors, provider):
        self.tensors = tuple(tensors)
        self.provider = provider
        self.epoch = 0
        self.row_ids = torch.arange(
            int(self.tensors[0].shape[0]),
            dtype=torch.long,
        )

    def __len__(self):
        return int(self.row_ids.shape[0])

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _values(self, index):
        replacements = self.provider(index, self.epoch)
        return tuple(
            replacements.get(position, tensor[index])
            for position, tensor in enumerate(self.tensors)
        )

    def __getitem__(self, index):
        return (*self._values(index), self.row_ids[index])

    def gather_inputs(self, indices, input_count):
        columns = [[] for _ in range(input_count)]
        for index in indices:
            values = self._values(int(index))
            for position in range(input_count):
                columns[position].append(values[position])
        stacked = tuple(torch.stack(column, dim=0) for column in columns)
        return stacked[0] if input_count == 1 else stacked


class _TensorColumnsDataset(Dataset):
    """TensorDataset equivalent that preserves explicit residue masks."""

    def __init__(self, *columns):
        if not columns:
            raise ValueError("column dataset requires at least one input")
        row_count = int(columns[0].shape[0])
        if any(int(column.shape[0]) != row_count for column in columns):
            raise ValueError("column dataset inputs have different row counts")
        self.columns = tuple(columns)
        self.row_count = row_count

    def __len__(self):
        return self.row_count

    @staticmethod
    def _row(column, index):
        if isinstance(column, ResidueModelInput):
            return ResidueModelInput(
                column.features[index],
                column.mask[index],
            )
        return column[index]

    def __getitem__(self, index):
        return tuple(self._row(column, index) for column in self.columns)


def _tensor_columns_dataset(*columns):
    if all(isinstance(column, torch.Tensor) for column in columns):
        return TensorDataset(*columns)
    return _TensorColumnsDataset(*columns)


class _CompactSingleStreamDataset(Dataset):
    """Index dataset whose collate step materializes one joined batch."""

    def __init__(self, compact_input, *columns):
        if not isinstance(compact_input, CompactSingleStreamEncoding):
            raise TypeError("compact dataset requires CompactSingleStreamEncoding")
        row_count = len(compact_input)
        if any(int(column.shape[0]) != row_count for column in columns):
            raise ValueError("compact dataset columns have different row counts")
        self.compact_input = compact_input
        self.columns = tuple(columns)

    def __len__(self):
        return len(self.compact_input)

    def __getitem__(self, index):
        return int(index)

    def collate_batch(self, row_indices):
        index = torch.as_tensor(row_indices, dtype=torch.int64)
        joined = self.compact_input.materialize_indices(index)
        values = tuple(column.index_select(0, index) for column in self.columns)
        return (joined, *values)


def _sequence_data_loader(x, *columns, batch_size, sampler=None):
    if isinstance(x, CompactSingleStreamEncoding):
        dataset = _CompactSingleStreamDataset(x, *columns)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=dataset.collate_batch,
        )
    return DataLoader(
        _tensor_columns_dataset(x, *columns),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
    )


def _pair_unigram_provider(args):
    provider = getattr(args, "_unigram_sampling_provider", None)
    return provider if bool(getattr(args, "unigram_sampling", False)) else None


def _set_pair_loader_epoch(args, train_loader, epoch):
    dataset = getattr(train_loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
        args._active_unigram_dataset = dataset


def _pair_train_loader(args, *tensors):
    if not tensors:
        raise ValueError("pair training loader requires at least one tensor")
    row_ids = torch.arange(int(tensors[0].shape[0]), dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(int(getattr(args, "seed", 42)))
    audit_log = []
    args._main_batch_order_hashes = audit_log
    provider = _pair_unigram_provider(args)
    if isinstance(tensors[0], CompactSingleStreamEncoding):
        if provider is not None:
            raise ValueError("unigram sampling is incompatible with compact pretrained input")
        dataset = _CompactSingleStreamDataset(tensors[0], *tensors[1:], row_ids)
    else:
        dataset = (
            _EpochSampledPairDataset(tensors, provider)
            if provider is not None
            else _tensor_columns_dataset(*tensors, row_ids)
        )
    sampler = RecordingRandomSampler(dataset, generator=generator, audit_log=audit_log)
    if isinstance(dataset, _CompactSingleStreamDataset):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=dataset.collate_batch,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
    )


def _gather_semantic_inputs(full_inputs, row_indices, device):
    index = torch.tensor(row_indices, dtype=torch.long)
    if isinstance(
        full_inputs,
        (torch.Tensor, ResidueModelInput, CompactSingleStreamEncoding),
    ):
        return (full_inputs.index_select(0, index).to(device),)
    return tuple(value.index_select(0, index).to(device) for value in full_inputs)


def _gather_auxiliary_inputs(args, full_inputs, unique_rows, device):
    sampled_dataset = getattr(args, "_active_unigram_dataset", None)
    if isinstance(sampled_dataset, _EpochSampledPairDataset):
        input_count = (
            1 if isinstance(full_inputs, (torch.Tensor, ResidueModelInput)) else len(full_inputs)
        )
        sampled_inputs = sampled_dataset.gather_inputs(
            unique_rows,
            input_count,
        )
        if isinstance(sampled_inputs, torch.Tensor):
            return (sampled_inputs.to(device),)
        return tuple(value.to(device) for value in sampled_inputs)
    return _gather_semantic_inputs(full_inputs, unique_rows, device)


def add_partner_semantic_loss(
    args,
    model,
    classification_loss,
    semantic_profile,
    row_ids,
    full_inputs,
    device,
):
    if not interaction_semantic_enabled(args) or semantic_profile is None:
        return classification_loss, None
    if not isinstance(semantic_profile, SemanticTrainingProfile):
        raise TypeError("unified pair semantic training requires a SemanticTrainingProfile")
    pair_mode = str(getattr(args, "semantic_pair_mode", "role_fused")).strip().lower()
    temperature = float(getattr(args, "semantic_contrastive_temperature", 0.1))
    loss_space = str(getattr(args, "semantic_loss_space", "projection")).strip().lower()
    partner_scheme = str(getattr(args, "semantic_partner_scheme", "fixed")).strip().lower()
    min_negatives = 1
    anchor_indices = [int(value) for value in row_ids.detach().cpu().view(-1).tolist()]

    fused_relations = []
    peptide_relations = []
    context_relations = []
    if pair_mode in {"fused", "role_fused"} and semantic_profile.fused_table is not None:
        fused_relations = list(semantic_profile.fused_table.relations_for(anchor_indices))
    if pair_mode in {"role", "role_fused"} and semantic_profile.peptide_table is not None:
        peptide_entity_anchors = [
            semantic_profile.row_peptide_ids[index] for index in anchor_indices
        ]
        peptide_relations = list(
            semantic_profile.peptide_table.relations_for(peptide_entity_anchors)
        )
    if pair_mode in {"role", "role_fused"} and semantic_profile.context_table is not None:
        context_entity_anchors = [
            semantic_profile.row_context_ids[index] for index in anchor_indices
        ]
        context_relations = list(
            semantic_profile.context_table.relations_for(context_entity_anchors)
        )

    if not fused_relations and not peptide_relations and not context_relations:
        return classification_loss, None

    fused_unique_rows = (
        semantic_profile.fused_table.unique_row_indices(fused_relations) if fused_relations else ()
    )
    peptide_unique_entities = (
        semantic_profile.peptide_table.unique_row_indices(peptide_relations)
        if peptide_relations
        else ()
    )
    context_unique_entities = (
        semantic_profile.context_table.unique_row_indices(context_relations)
        if context_relations
        else ()
    )
    peptide_unique_rows = tuple(
        semantic_profile.peptide_example_rows[index] for index in peptide_unique_entities
    )
    context_unique_rows = tuple(
        semantic_profile.context_example_rows[index] for index in context_unique_entities
    )
    unique_rows = tuple(
        sorted(set(fused_unique_rows) | set(peptide_unique_rows) | set(context_unique_rows))
    )
    if not unique_rows:
        return classification_loss, None

    auxiliary_inputs = _gather_auxiliary_inputs(
        args,
        full_inputs,
        unique_rows,
        device,
    )
    row_to_position = {row: index for index, row in enumerate(unique_rows)}
    active_losses: list[torch.Tensor] = []
    outputs: list[SemanticLossOutput] = []

    if fused_relations:
        if not hasattr(model, "forward_with_embedding"):
            raise SystemExit(
                "semantic similarity requires a neural pair model with forward_with_embedding"
            )
        if loss_space == "shared" and hasattr(model, "forward_with_effective_embedding"):
            with auxiliary_eval_forward(model):
                _, fused_embeddings = model.forward_with_effective_embedding(*auxiliary_inputs)
        else:
            with auxiliary_eval_forward(model):
                _, fused_embeddings = model.forward_with_embedding(*auxiliary_inputs)
        if (
            fused_embeddings is None
            or fused_embeddings.ndim < 2
            or not fused_embeddings.requires_grad
        ):
            raise SystemExit(
                "semantic similarity requires a differentiable sample-level pair embedding"
            )
        if loss_space == "projection":
            if not hasattr(model, "semantic_projector"):
                raise SystemExit(
                    "semantic loss_space=projection requires prepare_semantic_projectors"
                )
            fused_embeddings = model.semantic_projector(fused_embeddings)
        fused_positions = [row_to_position[row] for row in fused_unique_rows]
        fused_output = semantic_contrastive_loss(
            fused_embeddings[fused_positions],
            fused_unique_rows,
            fused_relations,
            temperature=temperature,
            min_negatives=min_negatives,
            loss_type=getattr(args, "semantic_loss_type", "legacy"),
            weight_positives=(partner_scheme == "dense_hard"),
        )
        active_losses.append(fused_output.loss)
        outputs.append(fused_output)

    role_embeddings = None
    if peptide_relations or context_relations:
        if not hasattr(model, "forward_with_role_embeddings"):
            if pair_mode == "role":
                raise SystemExit(
                    "semantic_pair_mode=role requires a model with forward_with_role_embeddings"
                )
        else:
            with auxiliary_eval_forward(model):
                (
                    _,
                    _,
                    peptide_role_embeddings,
                    partner_role_embeddings,
                ) = model.forward_with_role_embeddings(*auxiliary_inputs)
            if (
                peptide_role_embeddings is None
                or partner_role_embeddings is None
                or not peptide_role_embeddings.requires_grad
            ):
                raise SystemExit("semantic role loss requires differentiable role embeddings")
            role_embeddings = (
                peptide_role_embeddings,
                partner_role_embeddings,
            )

    if peptide_relations and role_embeddings is not None:
        peptide_embeddings = torch.stack(
            [role_embeddings[0][row_to_position[row]] for row in peptide_unique_rows]
        )
        peptide_embeddings = model.peptide_semantic_projector(peptide_embeddings)
        peptide_output = semantic_contrastive_loss(
            peptide_embeddings,
            peptide_unique_entities,
            peptide_relations,
            temperature=temperature,
            min_negatives=min_negatives,
            loss_type=getattr(args, "semantic_loss_type", "legacy"),
        )
        active_losses.append(peptide_output.loss)
        outputs.append(peptide_output)

    if context_relations and role_embeddings is not None:
        context_embeddings = torch.stack(
            [role_embeddings[1][row_to_position[row]] for row in context_unique_rows]
        )
        context_embeddings = model.partner_semantic_projector(context_embeddings)
        context_output = semantic_contrastive_loss(
            context_embeddings,
            context_unique_entities,
            context_relations,
            temperature=temperature,
            min_negatives=min_negatives,
            loss_type=getattr(args, "semantic_loss_type", "legacy"),
        )
        active_losses.append(context_output.loss)
        outputs.append(context_output)

    if not active_losses:
        return classification_loss, None
    combined = torch.stack(active_losses).mean()
    weight = float(getattr(args, "semantic_similarity_weight", 0.1))
    return classification_loss + weight * combined, combined


def print_interaction_semantic_status(args, profile):
    if not interaction_semantic_enabled(args):
        return
    positive_edges = (
        int(profile.source_summary.get("positive_edges", 0)) if profile is not None else 0
    )
    print(
        "[semantic] Unified interaction-profile contrastive loss enabled: "
        f"mode={getattr(args, 'semantic_pair_mode', 'role_fused')}, "
        f"weight={float(getattr(args, 'semantic_similarity_weight', 0.1)):g}, "
        f"temperature={float(getattr(args, 'semantic_contrastive_temperature', 0.1)):g}, "
        f"projection_dim={int(getattr(args, 'semantic_projection_dim', 64))}, "
        f"loss_space={str(getattr(args, 'semantic_loss_space', 'projection'))}, "
        f"partner_scheme={str(getattr(args, 'semantic_partner_scheme', 'fixed'))}, "
        f"residual_weight={float(getattr(args, 'semantic_residual_weight', 0.0)):g}, "
        f"learned_gate={bool(getattr(args, 'semantic_learned_gate', False))}, "
        f"positive_edges={positive_edges}"
    )


def scale_dense_features(train_x, val_x, test_x=None, mode="none", return_scaler=False):
    if mode == "none":
        if return_scaler:
            return train_x, val_x, test_x, None
        return train_x, val_x, test_x
    scaler = MinMaxScaler() if mode == "minmax" else StandardScaler()
    tx = scaler.fit_transform(train_x.cpu().numpy())
    vx = scaler.transform(val_x.cpu().numpy())
    tx = torch.from_numpy(tx).float()
    vx = torch.from_numpy(vx).float()
    if test_x is not None and len(test_x) > 0:
        sx = scaler.transform(test_x.cpu().numpy())
        test_x = torch.from_numpy(sx).float()
    if return_scaler:
        return tx, vx, test_x, scaler
    return tx, vx, test_x


def metric_tag(args):
    tok = str(getattr(args, "tokenizer", "na"))
    model = str(getattr(args, "model", "mlp"))
    task = normalize_task_type(getattr(args, "task_type", "pep_protein_interaction"))
    prefix = {
        "pep_protein_interaction": "ppi",
        "pep_hlaii": "hlaii",
        "pep_hlai": "hlai",
    }.get(task, "pair")

    model_lower = model.lower()
    fusion_models = ["mlp", "svm", "rf", "xgb", "lgb", "logreg"] + GRAPH_MODELS
    if model_lower in [CROSS_ATTN_MODEL, BI_CROSS_ATTN_MODEL]:
        route = "two_stream"
    elif model_lower in DL_SEQUENCE_MODELS:
        route = "single_stream"
    elif model_lower in fusion_models:
        route = "two_stream"
    else:
        route = str(getattr(args, "encoder", "ppi_fusion"))

    if model_lower in fusion_models:
        fusion_mode = resolve_fusion_mode(getattr(args, "fusion_mode", "concat_all"))
        fusion = "gated(static)" if fusion_mode == "gated" else fusion_mode
        return f"{prefix}/{tok}+{route}/{model}+{fusion}"
    return f"{prefix}/{tok}+{route}/{model}"


def testset_tag(args):
    return str(getattr(args, "ppi_test_set", "none"))


def compute_aupr(y_true, y_score):
    if y_true is None or y_score is None or len(y_true) == 0:
        return float("nan")
    if y_true.max() == y_true.min():
        return float("nan")
    try:
        value = binary_aupr(y_true, y_score)
        return float(value) if value is not None else float("nan")
    except Exception:
        return float("nan")


def write_score_analysis(
    args, val_y, val_scores, test_y=None, test_scores=None, *, operating_point=None
):
    artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
    if not artifact_dir:
        return
    try:
        if operating_point is None:
            operating_point = _fit_pair_operating_point(
                args, val_y, val_scores, score_kind="probability"
            )
        else:
            operating_point = OperatingPoint.from_dict(operating_point)
        args._operating_point = operating_point.to_dict()
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
        payload = {
            **_pair_protocol_metadata(args),
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
    except Exception as exc:
        print(f"[analysis] skipped score analysis artifact: {exc}", flush=True)


def _write_masked_score_analysis(args, y_all, val_mask, test_mask, score_fn):
    def collect(mask):
        if mask is None or not bool(mask.detach().any().item()):
            return None, None
        scores = score_fn(mask)
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores).reshape(-1)
        targets = y_all[mask].view(-1).detach().cpu().numpy().astype(int)
        return targets, scores

    try:
        with torch.no_grad():
            val_y, val_scores = collect(val_mask)
            test_y, test_scores = collect(test_mask)
        if val_y is not None and val_scores is not None:
            write_score_analysis(args, val_y, val_scores, test_y, test_scores)
        return val_y, val_scores, test_y, test_scores
    except Exception as exc:
        print(f"[analysis] skipped masked score analysis artifact: {exc}", flush=True)
        return None, None, None, None


def collect_binary_scores(model, loader, device):
    model.eval()
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = torch.sigmoid(model(inputs)).view(-1)
            all_scores.append(outputs.cpu())
            all_targets.append(targets.view(-1).cpu())
    if not all_scores:
        return None, None
    scores = torch.cat(all_scores).numpy()
    targets = torch.cat(all_targets).numpy().astype(int)
    return targets, scores


def eval_binary_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = criterion(model(inputs), targets)
            batch_size = int(targets.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    return total_loss / max(1, total_count)


class LegacyAlignedGatedClassifier(nn.Module):
    """Unversioned layout retained only for historical whole-object pickles."""

    def __init__(
        self,
        input_dim=None,
        hidden_dim=256,
        dropout=0.2,
        peptide_dim=None,
        partner_dim=None,
        align_dim=None,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        self.heterogeneous = peptide_dim is not None or partner_dim is not None
        if not self.heterogeneous:
            peptide_dim = partner_dim = int(input_dim)
            align_dim = int(input_dim)
            self.peptide_projection = nn.Identity()
            self.partner_projection = nn.Identity()
        else:
            peptide_dim = int(peptide_dim)
            partner_dim = int(partner_dim)
            align_dim = int(align_dim or min(hidden_dim, max(64, min(peptide_dim, partner_dim))))
            self.peptide_projection = nn.Linear(peptide_dim, align_dim)
            self.partner_projection = nn.Linear(partner_dim, align_dim)
        self.peptide_dim = int(peptide_dim)
        self.partner_dim = int(partner_dim)
        self.align_dim = int(align_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.align_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.align_dim),
            nn.Sigmoid(),
        )
        self.head = build_prediction_head(
            self.align_dim * 3,
            1,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[128],
        )

    def forward_with_embedding(self, pep, pro):
        pep = self.peptide_projection(pep.float())
        pro = self.partner_projection(pro.float())
        absdiff = (pep - pro).abs()
        had = pep * pro
        ctx = torch.cat([pep, pro, absdiff, had], dim=1)
        g = self.gate(ctx)
        fused = g * pep + (1.0 - g) * pro
        z = torch.cat([fused, absdiff, had], dim=1)
        return self.head(z), z

    @property
    def semantic_embedding_dim(self) -> int:
        return self.align_dim * 3

    @property
    def semantic_role_embedding_dims(self) -> tuple[int, int] | None:
        return (self.align_dim, self.align_dim)

    def forward_with_role_embeddings(self, pep, pro):
        pep = self.peptide_projection(pep.float())
        pro = self.partner_projection(pro.float())
        absdiff = (pep - pro).abs()
        had = pep * pro
        ctx = torch.cat([pep, pro, absdiff, had], dim=1)
        g = self.gate(ctx)
        fused = g * pep + (1.0 - g) * pro
        z = torch.cat([fused, absdiff, had], dim=1)
        return self.head(z), z, pep, pro

    def forward(self, pep, pro):
        logits, _ = self.forward_with_embedding(pep, pro)
        return logits


# Import-path compatibility for historical artifacts. New training resolves
# the package AlignedGatedClassifier imported above; this symbol is never a
# new configuration alias.
GatedTrainableClassifier = LegacyAlignedGatedClassifier


def _sample_batch_to_device(value, device, limit=2):
    """Slice a representative batch without discarding explicit residue masks."""

    end = min(int(limit), int(value.shape[0]))
    if isinstance(value, (ResidueModelInput, CompactSingleStreamEncoding)):
        return value.batch_slice(0, end).to(device)
    return value[:end].to(device)


def collect_binary_scores_pair(model, loader, device):
    model.eval()
    all_scores, all_targets = [], []
    with torch.no_grad():
        for pep, pro, targets in loader:
            pep = pep.to(device)
            pro = pro.to(device)
            outputs = torch.sigmoid(model(pep, pro)).view(-1)
            all_scores.append(outputs.cpu())
            all_targets.append(targets.view(-1).cpu())
    if not all_scores:
        return None, None
    return torch.cat(all_targets).numpy().astype(int), torch.cat(all_scores).numpy()


def eval_binary_pair_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for pep, pro, targets in loader:
            pep = pep.to(device)
            pro = pro.to(device)
            targets = targets.to(device)
            loss = criterion(model(pep, pro), targets)
            batch_size = int(targets.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    return total_loss / max(1, total_count)


def train_mlp(
    args,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x=None,
    test_y=None,
    device="cpu",
    semantic_profile=None,
    train_pair_ids=None,
    dl_scaler=None,
):
    model = PredictionHeadClassifier(
        input_dim=train_x.shape[1],
        num_classes=1,
        dropout=args.dropout,
        **prediction_head_kwargs(args),
    ).to(device)
    prepare_semantic_projectors(
        model,
        args,
        sample_inputs=_sample_batch_to_device(train_x, device),
    )
    model = _pair_residual_model(model, args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = build_pair_classification_criterion(args, train_y, device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    val_ds = TensorDataset(val_x, val_y)
    train_loader = _pair_train_loader(args, train_x, train_y)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print_interaction_semantic_status(args, semantic_profile)

    best_score = -1.0
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(args.epochs):
        _set_pair_loader_epoch(args, train_loader, epoch)
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x, y, row_ids = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            classification_loss = criterion(logits, y)
            loss, _ = add_partner_semantic_loss(
                args,
                model,
                classification_loss,
                semantic_profile,
                row_ids,
                train_x,
                device,
            )
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()
        y_true, y_score = collect_binary_scores(model, val_loader, device)
        val_acc, val_f1, val_auc, val_threshold = binary_metrics_with_best_threshold(
            y_true, y_score, args=args, for_checkpoint=True
        )
        val_sacc = val_acc
        train_loss = total_loss / max(1, len(train_loader))
        val_loss = eval_binary_loss(model, val_loader, criterion, device)
        if getattr(args, "report_aupr", True):
            val_aupr = compute_aupr(y_true, y_score)
            print(
                f"[{metric_tag(args)}] Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val SAcc: {val_sacc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}{binary_extra_suffix(y_true, y_score, threshold=val_threshold, prefix='Val')} | Val AUPR: {val_aupr:.4f} | ValThr: {val_threshold:.2f}"
            )
        else:
            print(
                f"[{metric_tag(args)}] Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val SAcc: {val_sacc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}{binary_extra_suffix(y_true, y_score, threshold=val_threshold, prefix='Val')} | ValThr: {val_threshold:.2f}"
            )
        selection_score = _pair_checkpoint_score(args, val_auc, val_f1, epoch + 1)
        if selection_score > best_score + args.early_stop_min_delta:
            best_score = selection_score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        scheduler.step()
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[{metric_tag(args)}] Early stop at epoch {epoch + 1}")
            break

    print(
        f"[{metric_tag(args)}] Finished. Best Val {resolve_selection(args).checkpoint.id.upper()}: {best_score:.4f} (Epoch {best_epoch})"
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_true, final_val_scores = collect_binary_scores(model, val_loader, device)
    operating_point = _fit_pair_operating_point(
        args, final_val_true, final_val_scores, score_kind="probability"
    )
    final_threshold = operating_point.threshold
    final_test_true, final_test_scores = None, None
    if test_x is not None and test_y is not None and len(test_x) > 0:
        test_ds = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        final_test_true, final_test_scores = collect_binary_scores(model, test_loader, device)
        test_acc, test_f1, test_auc = binary_metrics_with_threshold(
            final_test_true, final_test_scores, threshold=final_threshold
        )
        test_sacc = test_acc
        if getattr(args, "report_aupr", True):
            test_aupr = compute_aupr(final_test_true, final_test_scores)
            print(
                f"[{metric_tag(args)}] Test | TestSet: {testset_tag(args)} | Acc: {test_acc:.4f} | SAcc: {test_sacc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}{binary_extra_suffix(final_test_true, final_test_scores, threshold=final_threshold)} | AUPR: {test_aupr:.4f} | ValThr: {final_threshold:.2f}"
            )
        else:
            print(
                f"[{metric_tag(args)}] Test | TestSet: {testset_tag(args)} | Acc: {test_acc:.4f} | SAcc: {test_sacc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}{binary_extra_suffix(final_test_true, final_test_scores, threshold=final_threshold)} | ValThr: {final_threshold:.2f}"
            )
    write_score_analysis(
        args,
        final_val_true,
        final_val_scores,
        final_test_true,
        final_test_scores,
        operating_point=operating_point,
    )
    if str(getattr(args, "_artifact_source", "") or "") == PAIR_EMBEDDING_ARTIFACT_SOURCE:
        save_pair_embedding_nn_artifact(args, model, dl_scaler=dl_scaler, threshold=final_threshold)
    else:
        save_pair_vector_nn_artifact(args, model, dl_scaler=dl_scaler, threshold=final_threshold)
    return best_score


def train_mlp_aligned_gated(
    args,
    train_pep_x,
    train_pro_x,
    train_y,
    val_pep_x,
    val_pro_x,
    val_y,
    test_pep_x=None,
    test_pro_x=None,
    test_y=None,
    device="cpu",
    semantic_profile=None,
    train_pair_ids=None,
    peptide_scaler=None,
    protein_scaler=None,
):
    heterogeneous = int(train_pep_x.shape[1]) != int(train_pro_x.shape[1])
    if heterogeneous:
        model = AlignedGatedClassifier(
            peptide_dim=train_pep_x.shape[1],
            partner_dim=train_pro_x.shape[1],
            align_dim=int(getattr(args, "stream_alignment_dim", 256)),
            hidden_dim=256,
            dropout=args.dropout,
            **prediction_head_kwargs(args),
        ).to(device)
    else:
        model = AlignedGatedClassifier(
            input_dim=train_pep_x.shape[1],
            hidden_dim=256,
            dropout=args.dropout,
            **prediction_head_kwargs(args),
        ).to(device)
    prepare_semantic_projectors(
        model,
        args,
        sample_inputs=(
            _sample_batch_to_device(train_pep_x, device),
            _sample_batch_to_device(train_pro_x, device),
        ),
    )
    model = _pair_residual_model(model, args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = build_pair_classification_criterion(args, train_y, device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    train_loader = _pair_train_loader(args, train_pep_x, train_pro_x, train_y)
    val_loader = DataLoader(
        TensorDataset(val_pep_x, val_pro_x, val_y), batch_size=args.batch_size, shuffle=False
    )
    print_interaction_semantic_status(args, semantic_profile)
    best_score, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    for epoch in range(args.epochs):
        _set_pair_loader_epoch(args, train_loader, epoch)
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            pep, pro, y, row_ids = batch
            pep, pro, y = pep.to(device), pro.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(pep, pro)
            classification_loss = criterion(logits, y)
            loss, _ = add_partner_semantic_loss(
                args,
                model,
                classification_loss,
                semantic_profile,
                row_ids,
                (train_pep_x, train_pro_x),
                device,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("aligned_gated produced a non-finite training loss")
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            if any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
            ):
                raise RuntimeError("aligned_gated produced a non-finite gradient")
            optimizer.step()
            total_loss += loss.item()
        y_true, y_score = collect_binary_scores_pair(model, val_loader, device)
        val_acc, val_f1, val_auc, val_threshold = binary_metrics_with_best_threshold(
            y_true, y_score, args=args, for_checkpoint=True
        )
        val_sacc = val_acc
        train_loss = total_loss / max(1, len(train_loader))
        val_loss = eval_binary_pair_loss(model, val_loader, criterion, device)
        val_aupr = compute_aupr(y_true, y_score) if getattr(args, "report_aupr", True) else 0.0
        msg = f"[{metric_tag(args)}] Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val SAcc: {val_sacc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}{binary_extra_suffix(y_true, y_score, threshold=val_threshold, prefix='Val')}"
        print(msg + (f" | Val AUPR: {val_aupr:.4f}" if getattr(args, "report_aupr", True) else ""))
        selection_score = _pair_checkpoint_score(args, val_auc, val_f1, epoch + 1)
        if selection_score > best_score + args.early_stop_min_delta:
            best_score, best_epoch, no_improve = (
                selection_score,
                epoch + 1,
                0,
            )
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        scheduler.step()
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[{metric_tag(args)}] Early stop at epoch {epoch + 1}")
            break
    print(
        f"[{metric_tag(args)}] Finished. Best Val {resolve_selection(args).checkpoint.id.upper()}: {best_score:.4f} (Epoch {best_epoch})"
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_true, final_val_scores = collect_binary_scores_pair(model, val_loader, device)
    operating_point = _fit_pair_operating_point(
        args, final_val_true, final_val_scores, score_kind="probability"
    )
    final_threshold = operating_point.threshold
    final_test_true, final_test_scores = None, None
    if (
        test_pep_x is not None
        and test_pro_x is not None
        and test_y is not None
        and len(test_pep_x) > 0
    ):
        test_loader = DataLoader(
            TensorDataset(test_pep_x, test_pro_x, test_y), batch_size=args.batch_size, shuffle=False
        )
        final_test_true, final_test_scores = collect_binary_scores_pair(model, test_loader, device)
        test_acc, test_f1, test_auc = binary_metrics_with_threshold(
            final_test_true, final_test_scores, threshold=final_threshold
        )
        test_sacc = test_acc
        test_aupr = (
            compute_aupr(final_test_true, final_test_scores)
            if getattr(args, "report_aupr", True)
            else 0.0
        )
        msg = f"[{metric_tag(args)}] Test | TestSet: {testset_tag(args)} | Acc: {test_acc:.4f} | SAcc: {test_sacc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}{binary_extra_suffix(final_test_true, final_test_scores, threshold=final_threshold)}"
        print(msg + (f" | AUPR: {test_aupr:.4f}" if getattr(args, "report_aupr", True) else ""))
    write_score_analysis(
        args,
        final_val_true,
        final_val_scores,
        final_test_true,
        final_test_scores,
        operating_point=operating_point,
    )
    if best_state is not None:
        save_pair_vector_nn_artifact(
            args,
            model,
            threshold=final_threshold,
            peptide_scaler=peptide_scaler,
            protein_scaler=protein_scaler,
        )
    return best_score


def build_knn_graph_edge_index(
    x,
    k=8,
    *,
    seed=42,
    exact_max_nodes=25000,
    rp_projections=6,
    rp_window=0,
):
    n = int(x.shape[0])
    if n <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    if n == 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    k = max(1, min(int(k), n - 1))
    z = x.float()
    z = z / z.norm(dim=1, keepdim=True).clamp(min=1e-6)
    exact_max_nodes = max(0, int(exact_max_nodes))
    if exact_max_nodes and n > exact_max_nodes:
        projections = max(1, int(rp_projections))
        window = int(rp_window) if int(rp_window) > 0 else k
        window = max(1, min(window, n - 1))
        rng = np.random.default_rng(int(seed))
        z_np = np.ascontiguousarray(z.numpy(), dtype=np.float32)
        directions = rng.standard_normal((z_np.shape[1], projections)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=0, keepdims=True).clip(min=1e-6)
        scores = z_np @ directions
        src_parts = []
        dst_parts = []
        for projection_idx in range(projections):
            order = np.argsort(scores[:, projection_idx], kind="mergesort").astype(
                np.int64, copy=False
            )
            for offset in range(1, window + 1):
                left = order[:-offset]
                right = order[offset:]
                src_parts.extend([left, right])
                dst_parts.extend([right, left])
        self_idx = np.arange(n, dtype=np.int64)
        src_parts.append(self_idx)
        dst_parts.append(self_idx)
        src = torch.from_numpy(np.concatenate(src_parts))
        dst = torch.from_numpy(np.concatenate(dst_parts))
        edge_index = torch.stack([src, dst], dim=0)
        print(
            "[graph] random_projection_neighbor_graph "
            f"nodes={n} k={k} projections={projections} window={window} seed={int(seed)} "
            f"edges={int(edge_index.shape[1])} exact_max_nodes={exact_max_nodes}",
            flush=True,
        )
        return edge_index
    sim = z @ z.t()
    sim.fill_diagonal_(-1e9)
    nbr = torch.topk(sim, k=k, dim=1, largest=True).indices
    dst = torch.arange(n, dtype=torch.long).unsqueeze(1).expand(n, k).reshape(-1)
    src = nbr.reshape(-1)
    self_idx = torch.arange(n, dtype=torch.long)
    src = torch.cat([src, self_idx], dim=0)
    dst = torch.cat([dst, self_idx], dim=0)
    edge_index = torch.stack([src, dst], dim=0)
    print(f"[graph] exact_dense_knn nodes={n} k={k} edges={int(edge_index.shape[1])}", flush=True)
    return edge_index


def build_candidate_graph_edge_index(args, x, k=None):
    return build_knn_graph_edge_index(
        x,
        k=int(k if k is not None else getattr(args, "gnn_k", 8)),
        seed=int(getattr(args, "seed", 42)),
        exact_max_nodes=int(getattr(args, "gnn_exact_knn_max_nodes", 25000)),
        rp_projections=int(getattr(args, "gnn_rp_projections", 6)),
        rp_window=int(getattr(args, "gnn_rp_window", 0)),
    )


def build_gat_knn_graph_edge_index(
    x,
    k=8,
    *,
    seed=42,
    exact_max_nodes=25000,
    rp_projections=6,
    rp_window=0,
    _log_name="gat",
):
    """Build the contract-v2 GAT reference graph with unique incoming edges."""
    n = int(x.shape[0])
    if n <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    if n == 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    k = max(1, min(int(k), n - 1))
    exact_max_nodes = max(0, int(exact_max_nodes))
    if not exact_max_nodes or n <= exact_max_nodes:
        return build_knn_graph_edge_index(
            x,
            k=k,
            seed=seed,
            exact_max_nodes=0,
            rp_projections=rp_projections,
            rp_window=rp_window,
        )

    z = x.detach().cpu().float()
    z = z / z.norm(dim=1, keepdim=True).clamp(min=1e-6)
    z_np = np.ascontiguousarray(z.numpy(), dtype=np.float32)
    projections = max(1, int(rp_projections))
    window = int(rp_window) if int(rp_window) > 0 else k
    window = max(k, min(window, n - 1))
    rng = np.random.default_rng(int(seed))
    directions = rng.standard_normal((z_np.shape[1], projections)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True).clip(min=1e-6)
    scores = z_np @ directions

    candidate_columns = projections * window * 2
    candidates = np.full((n, candidate_columns), -1, dtype=np.int64)
    row_ids = np.arange(n, dtype=np.int64)
    column = 0
    for projection_idx in range(projections):
        order = np.argsort(scores[:, projection_idx], kind="mergesort").astype(
            np.int64,
            copy=False,
        )
        rank = np.empty(n, dtype=np.int64)
        rank[order] = row_ids
        for offset in range(1, window + 1):
            left_rank = rank - offset
            left_valid = left_rank >= 0
            candidates[left_valid, column] = order[left_rank[left_valid]]
            column += 1
            right_rank = rank + offset
            right_valid = right_rank < n
            candidates[right_valid, column] = order[right_rank[right_valid]]
            column += 1

    neighbors = np.empty((n, k), dtype=np.int64)
    all_ids = np.arange(n, dtype=np.int64)
    for destination in range(n):
        candidate_ids = np.unique(candidates[destination])
        candidate_ids = candidate_ids[(candidate_ids >= 0) & (candidate_ids != destination)]
        if candidate_ids.size < k:
            candidate_ids = all_ids[all_ids != destination]
        similarity = z_np[candidate_ids] @ z_np[destination]
        best = np.lexsort((candidate_ids, -similarity))[:k]
        neighbors[destination] = candidate_ids[best]

    src = torch.from_numpy(neighbors.reshape(-1))
    dst = torch.arange(n, dtype=torch.long).unsqueeze(1).expand(n, k).reshape(-1)
    self_idx = torch.arange(n, dtype=torch.long)
    edge_index = torch.stack(
        [torch.cat([src, self_idx]), torch.cat([dst, self_idx])],
        dim=0,
    )
    print(
        f"[graph] {_log_name}_reranked_random_projection_knn "
        f"nodes={n} k={k} projections={projections} window={window} "
        f"seed={int(seed)} edges={int(edge_index.shape[1])} "
        f"exact_max_nodes={exact_max_nodes}",
        flush=True,
    )
    return edge_index


def build_gat_reference_edge_index(args, x, k=None):
    return build_gat_knn_graph_edge_index(
        x,
        k=int(k if k is not None else getattr(args, "gnn_k", 8)),
        seed=int(getattr(args, "seed", 42)),
        exact_max_nodes=int(getattr(args, "gnn_exact_knn_max_nodes", 25000)),
        rp_projections=int(getattr(args, "gnn_rp_projections", 6)),
        rp_window=int(getattr(args, "gnn_rp_window", 0)),
    )


def build_sage_knn_graph_edge_index(
    x,
    k=8,
    *,
    seed=42,
    exact_max_nodes=25000,
    rp_projections=6,
    rp_window=0,
):
    """Build unique non-self incoming neighbors for GraphSAGE mean aggregation."""
    edges = build_gat_knn_graph_edge_index(
        x,
        k=k,
        seed=seed,
        exact_max_nodes=exact_max_nodes,
        rp_projections=rp_projections,
        rp_window=rp_window,
        _log_name="sage",
    )
    src, dst = edges[0], edges[1]
    return edges[:, src != dst]


def build_sage_reference_edge_index(args, x, k=None):
    return build_sage_knn_graph_edge_index(
        x,
        k=int(k if k is not None else getattr(args, "gnn_k", 8)),
        seed=int(getattr(args, "seed", 42)),
        exact_max_nodes=int(getattr(args, "gnn_exact_knn_max_nodes", 25000)),
        rp_projections=int(getattr(args, "gnn_rp_projections", 6)),
        rp_window=int(getattr(args, "gnn_rp_window", 0)),
    )


def build_graph_transformer_knn_graph_edge_index(
    x,
    k=8,
    *,
    seed=42,
    exact_max_nodes=25000,
    rp_projections=6,
    rp_window=0,
):
    """Build the contract-v1 Graph Transformer reference topology."""
    return build_gat_knn_graph_edge_index(
        x,
        k=k,
        seed=seed,
        exact_max_nodes=exact_max_nodes,
        rp_projections=rp_projections,
        rp_window=rp_window,
        _log_name="graph_transformer",
    )


def build_graph_transformer_reference_edge_index(args, x, k=None):
    return build_graph_transformer_knn_graph_edge_index(
        x,
        k=int(k if k is not None else getattr(args, "gnn_k", 8)),
        seed=int(getattr(args, "seed", 42)),
        exact_max_nodes=int(getattr(args, "gnn_exact_knn_max_nodes", 25000)),
        rp_projections=int(getattr(args, "gnn_rp_projections", 6)),
        rp_window=int(getattr(args, "gnn_rp_window", 0)),
    )


def build_query_reference_edge_index(
    reference_x,
    query_x,
    k=8,
    *,
    seed=42,
    exact_max_nodes=25000,
    rp_projections=6,
    rp_window=0,
):
    """Build directed reference->query edges without query-query information flow."""
    n_reference = int(reference_x.shape[0])
    n_query = int(query_x.shape[0])
    if n_reference <= 0 or n_query <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    k = max(1, min(int(k), n_reference))
    reference = reference_x.detach().cpu().float()
    query = query_x.detach().cpu().float()
    reference = reference / reference.norm(dim=1, keepdim=True).clamp(min=1e-6)
    query = query / query.norm(dim=1, keepdim=True).clamp(min=1e-6)

    src_parts = []
    dst_parts = []
    # Keep the exact/approximate choice independent of query batch composition.
    # Reference topology is fixed for the fitted bundle; switching branches when
    # more query rows are submitted would make one query's neighbors batch-dependent.
    if n_reference <= max(1, int(exact_max_nodes)):
        chunk_size = max(1, min(1024, n_query))
        for start in range(0, n_query, chunk_size):
            stop = min(start + chunk_size, n_query)
            neighbors = torch.topk(query[start:stop] @ reference.t(), k=k, dim=1).indices
            src_parts.append(neighbors.reshape(-1))
            query_ids = torch.arange(start, stop, dtype=torch.long) + n_reference
            dst_parts.append(query_ids.unsqueeze(1).expand(-1, k).reshape(-1))
    else:
        projections = max(1, int(rp_projections))
        window = int(rp_window) if int(rp_window) > 0 else k
        window = max(k, window)
        rng = np.random.default_rng(int(seed))
        ref_np = np.ascontiguousarray(reference.numpy(), dtype=np.float32)
        query_np = np.ascontiguousarray(query.numpy(), dtype=np.float32)
        directions = rng.standard_normal((ref_np.shape[1], projections)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=0, keepdims=True).clip(min=1e-6)
        ref_projection = ref_np @ directions
        query_projection = query_np @ directions
        orders = [
            np.argsort(ref_projection[:, index], kind="mergesort").astype(np.int64, copy=False)
            for index in range(projections)
        ]
        sorted_scores = [ref_projection[orders[index], index] for index in range(projections)]
        src = np.empty((n_query, k), dtype=np.int64)
        for query_index in range(n_query):
            candidates = []
            for projection_index in range(projections):
                position = int(
                    np.searchsorted(
                        sorted_scores[projection_index],
                        query_projection[query_index, projection_index],
                    )
                )
                left = max(0, position - window)
                right = min(n_reference, position + window + 1)
                candidates.extend(orders[projection_index][left:right].tolist())
            candidate_ids = np.unique(np.asarray(candidates, dtype=np.int64))
            if candidate_ids.size < k:
                missing = np.setdiff1d(
                    np.arange(n_reference, dtype=np.int64),
                    candidate_ids,
                    assume_unique=True,
                )[: k - candidate_ids.size]
                candidate_ids = np.concatenate([candidate_ids, missing])
            similarity = ref_np[candidate_ids] @ query_np[query_index]
            best = np.argsort(-similarity, kind="mergesort")[:k]
            src[query_index] = candidate_ids[best]
        src_parts.append(torch.from_numpy(src.reshape(-1)))
        dst_parts.append(
            (torch.arange(n_query, dtype=torch.long) + n_reference)
            .unsqueeze(1)
            .expand(-1, k)
            .reshape(-1)
        )
        print(
            "[graph] inductive_random_projection_neighbors "
            f"reference={n_reference} query={n_query} k={k} "
            f"projections={projections} window={window} seed={int(seed)}",
            flush=True,
        )
    return torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)


def build_inductive_query_edge_index(args, reference_x, query_x):
    return build_query_reference_edge_index(
        reference_x,
        query_x,
        k=int(getattr(args, "gnn_k", 8)),
        seed=int(getattr(args, "seed", 42)),
        exact_max_nodes=int(getattr(args, "gnn_exact_knn_max_nodes", 25000)),
        rp_projections=int(getattr(args, "gnn_rp_projections", 6)),
        rp_window=int(getattr(args, "gnn_rp_window", 0)),
    )


class PairGCNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, adj):
        h = self.in_proj(x)
        h = torch.relu(self.gcn1(torch.sparse.mm(adj, h)))
        h = self.dropout(h)
        h = torch.relu(self.gcn2(torch.sparse.mm(adj, h)))
        h = self.dropout(h)
        return self.head(h)


def build_norm_adj(num_nodes, edge_index, device, *, bidirectional=True):
    src, dst = edge_index[0].to(device), edge_index[1].to(device)
    if bidirectional:
        src, dst = torch.cat([src, dst], dim=0), torch.cat([dst, src], dim=0)
    loop = torch.arange(num_nodes, device=device)
    src = torch.cat([src, loop], dim=0)
    dst = torch.cat([dst, loop], dim=0)
    val = torch.ones(src.shape[0], device=device)
    # Sparse matrix multiplication aggregates source messages into destination rows.
    adj = torch.sparse_coo_tensor(torch.stack([dst, src]), val, (num_nodes, num_nodes)).coalesce()
    idx = adj.indices()
    val = adj.values()
    deg = torch.zeros(num_nodes, device=device).index_add_(0, idx[0], val)
    if bidirectional:
        norm = deg[idx[0]].clamp(min=1e-6).pow(-0.5) * deg[idx[1]].clamp(min=1e-6).pow(-0.5)
    else:
        norm = val / deg[idx[0]].clamp(min=1e-6)
    return torch.sparse_coo_tensor(idx, norm, (num_nodes, num_nodes)).coalesce()


def train_gcn(
    args,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x=None,
    test_y=None,
    device="cpu",
    peptide_scaler=None,
    protein_scaler=None,
):
    model = PairGCNClassifier(
        input_dim=train_x.shape[1],
        hidden_dim=int(getattr(args, "gnn_hidden_dim", 256)),
        dropout=float(getattr(args, "dropout", 0.2)),
    ).to(device)
    return _train_inductive_graph_model(
        args,
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        device=device,
        graph_kind="gcn",
        graph_label="GCN",
        peptide_scaler=peptide_scaler,
        protein_scaler=protein_scaler,
    )


class _LegacyClippedGATAggregation(torch.autograd.Function):
    """Historical destination-wise GAT aggregation with clipped attention logits."""

    @staticmethod
    def forward(ctx, h, edge_index, a_src_weight, a_dst_weight, edge_chunk_size):
        src, dst = edge_index[0], edge_index[1]
        chunk_size = int(edge_chunk_size)
        src_score = h.matmul(a_src_weight.t()).squeeze(-1)
        dst_score = h.matmul(a_dst_weight.t()).squeeze(-1)
        denom = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)

        for start in range(0, src.numel(), chunk_size):
            stop = min(start + chunk_size, src.numel())
            chunk_src = src[start:stop]
            chunk_dst = dst[start:stop]
            logits = F.leaky_relu(
                src_score[chunk_src] + dst_score[chunk_dst],
                negative_slope=0.2,
            ).clamp(-10.0, 10.0)
            denom.index_add_(0, chunk_dst, logits.exp())

        out = torch.zeros_like(h)
        safe_denom = denom.clamp(min=1e-9)
        for start in range(0, src.numel(), chunk_size):
            stop = min(start + chunk_size, src.numel())
            chunk_src = src[start:stop]
            chunk_dst = dst[start:stop]
            logits = F.leaky_relu(
                src_score[chunk_src] + dst_score[chunk_dst],
                negative_slope=0.2,
            ).clamp(-10.0, 10.0)
            alpha = logits.exp() / safe_denom[chunk_dst]
            out.index_add_(0, chunk_dst, alpha.unsqueeze(1) * h[chunk_src])

        ctx.edge_chunk_size = chunk_size
        ctx.save_for_backward(
            h,
            edge_index,
            a_src_weight,
            a_dst_weight,
            safe_denom,
            out,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        h, edge_index, a_src_weight, a_dst_weight, denom, out = ctx.saved_tensors
        src, dst = edge_index[0], edge_index[1]
        chunk_size = ctx.edge_chunk_size
        src_score = h.matmul(a_src_weight.t()).squeeze(-1)
        dst_score = h.matmul(a_dst_weight.t()).squeeze(-1)
        grad_h = torch.zeros_like(h)
        grad_src_score = torch.zeros_like(src_score)
        grad_dst_score = torch.zeros_like(dst_score)

        for start in range(0, src.numel(), chunk_size):
            stop = min(start + chunk_size, src.numel())
            chunk_src = src[start:stop]
            chunk_dst = dst[start:stop]
            raw_logits = src_score[chunk_src] + dst_score[chunk_dst]
            activated = F.leaky_relu(raw_logits, negative_slope=0.2)
            logits = activated.clamp(-10.0, 10.0)
            alpha = logits.exp() / denom[chunk_dst]
            chunk_grad_out = grad_out[chunk_dst]
            chunk_h = h[chunk_src]

            grad_h.index_add_(
                0,
                chunk_src,
                alpha.unsqueeze(1) * chunk_grad_out,
            )
            grad_logits = alpha * (chunk_grad_out * (chunk_h - out[chunk_dst])).sum(dim=1)
            leaky_grad = torch.where(
                raw_logits >= 0,
                torch.ones_like(raw_logits),
                torch.full_like(raw_logits, 0.2),
            )
            clamp_grad = ((activated >= -10.0) & (activated <= 10.0)).to(h.dtype)
            grad_raw_logits = grad_logits * leaky_grad * clamp_grad
            grad_src_score.index_add_(0, chunk_src, grad_raw_logits)
            grad_dst_score.index_add_(0, chunk_dst, grad_raw_logits)

        grad_h.add_(grad_src_score.unsqueeze(1) * a_src_weight)
        grad_h.add_(grad_dst_score.unsqueeze(1) * a_dst_weight)
        grad_a_src = grad_src_score.unsqueeze(0).matmul(h)
        grad_a_dst = grad_dst_score.unsqueeze(0).matmul(h)
        return grad_h, None, grad_a_src, grad_a_dst, None


class _MemoryBoundedGATAggregation(torch.autograd.Function):
    """Exact, stable destination-wise GAT aggregation with bounded edge memory."""

    @staticmethod
    def forward(ctx, h, edge_index, a_src_weight, a_dst_weight, edge_chunk_size):
        src, dst = edge_index[0], edge_index[1]
        chunk_size = int(edge_chunk_size)
        src_score = h.matmul(a_src_weight.t()).squeeze(-1)
        dst_score = h.matmul(a_dst_weight.t()).squeeze(-1)
        max_score = torch.full(
            (h.shape[0],),
            -torch.inf,
            dtype=h.dtype,
            device=h.device,
        )

        for start in range(0, src.numel(), chunk_size):
            stop = min(start + chunk_size, src.numel())
            chunk_src = src[start:stop]
            chunk_dst = dst[start:stop]
            logits = F.leaky_relu(
                src_score[chunk_src] + dst_score[chunk_dst],
                negative_slope=0.2,
            )
            max_score.scatter_reduce_(
                0,
                chunk_dst,
                logits,
                reduce="amax",
                include_self=True,
            )

        denom = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
        for start in range(0, src.numel(), chunk_size):
            stop = min(start + chunk_size, src.numel())
            chunk_src = src[start:stop]
            chunk_dst = dst[start:stop]
            logits = F.leaky_relu(
                src_score[chunk_src] + dst_score[chunk_dst],
                negative_slope=0.2,
            )
            denom.index_add_(0, chunk_dst, (logits - max_score[chunk_dst]).exp())

        out = torch.zeros_like(h)
        safe_denom = denom.clamp(min=torch.finfo(h.dtype).tiny)
        for start in range(0, src.numel(), chunk_size):
            stop = min(start + chunk_size, src.numel())
            chunk_src = src[start:stop]
            chunk_dst = dst[start:stop]
            logits = F.leaky_relu(
                src_score[chunk_src] + dst_score[chunk_dst],
                negative_slope=0.2,
            )
            alpha = (logits - max_score[chunk_dst]).exp() / safe_denom[chunk_dst]
            out.index_add_(0, chunk_dst, alpha.unsqueeze(1) * h[chunk_src])

        ctx.edge_chunk_size = chunk_size
        ctx.save_for_backward(
            h,
            edge_index,
            a_src_weight,
            a_dst_weight,
            max_score,
            safe_denom,
            out,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            h,
            edge_index,
            a_src_weight,
            a_dst_weight,
            max_score,
            denom,
            out,
        ) = ctx.saved_tensors
        src, dst = edge_index[0], edge_index[1]
        chunk_size = ctx.edge_chunk_size
        src_score = h.matmul(a_src_weight.t()).squeeze(-1)
        dst_score = h.matmul(a_dst_weight.t()).squeeze(-1)
        grad_h = torch.zeros_like(h)
        grad_src_score = torch.zeros_like(src_score)
        grad_dst_score = torch.zeros_like(dst_score)

        for start in range(0, src.numel(), chunk_size):
            stop = min(start + chunk_size, src.numel())
            chunk_src = src[start:stop]
            chunk_dst = dst[start:stop]
            raw_logits = src_score[chunk_src] + dst_score[chunk_dst]
            logits = F.leaky_relu(raw_logits, negative_slope=0.2)
            alpha = (logits - max_score[chunk_dst]).exp() / denom[chunk_dst]
            chunk_grad_out = grad_out[chunk_dst]
            chunk_h = h[chunk_src]

            grad_h.index_add_(
                0,
                chunk_src,
                alpha.unsqueeze(1) * chunk_grad_out,
            )
            grad_logits = alpha * (chunk_grad_out * (chunk_h - out[chunk_dst])).sum(dim=1)
            leaky_grad = torch.where(
                raw_logits >= 0,
                torch.ones_like(raw_logits),
                torch.full_like(raw_logits, 0.2),
            )
            grad_raw_logits = grad_logits * leaky_grad
            grad_src_score.index_add_(0, chunk_src, grad_raw_logits)
            grad_dst_score.index_add_(0, chunk_dst, grad_raw_logits)

        grad_h.add_(grad_src_score.unsqueeze(1) * a_src_weight)
        grad_h.add_(grad_dst_score.unsqueeze(1) * a_dst_weight)
        grad_a_src = grad_src_score.unsqueeze(0).matmul(h)
        grad_a_dst = grad_dst_score.unsqueeze(0).matmul(h)
        return grad_h, None, grad_a_src, grad_a_dst, None


class PairGATClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        dropout=0.2,
        edge_chunk_size=GAT_EDGE_CHUNK_SIZE,
        contract_version=GAT_PREDICTOR_CONTRACT_VERSION,
    ):
        super().__init__()
        if int(edge_chunk_size) <= 0:
            raise ValueError("edge_chunk_size must be positive")
        self.edge_chunk_size = int(edge_chunk_size)
        self.gat_predictor_contract_version = int(contract_version)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.a_src1 = nn.Linear(hidden_dim, 1, bias=False)
        self.a_dst1 = nn.Linear(hidden_dim, 1, bias=False)
        self.a_src2 = nn.Linear(hidden_dim, 1, bias=False)
        self.a_dst2 = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _gat_layer(self, x, edge_index, lin, a_src, a_dst):
        h = lin(x)
        aggregation = (
            _MemoryBoundedGATAggregation
            if self.gat_predictor_contract_version >= GAT_PREDICTOR_CONTRACT_VERSION
            else _LegacyClippedGATAggregation
        )
        return aggregation.apply(
            h,
            edge_index,
            a_src.weight,
            a_dst.weight,
            self.edge_chunk_size,
        )

    def forward(self, x, edge_index):
        h = self.in_proj(x)
        h = torch.relu(self._gat_layer(h, edge_index, self.lin1, self.a_src1, self.a_dst1))
        h = self.dropout(h)
        h = torch.relu(self._gat_layer(h, edge_index, self.lin2, self.a_src2, self.a_dst2))
        h = self.dropout(h)
        return self.head(h)


def build_message_edge_index(num_nodes, edge_index, device, *, bidirectional=True):
    src, dst = edge_index[0].to(device), edge_index[1].to(device)
    if bidirectional:
        src, dst = torch.cat([src, dst], dim=0), torch.cat([dst, src], dim=0)
    loop = torch.arange(num_nodes, device=device)
    src = torch.cat([src, loop], dim=0)
    dst = torch.cat([dst, loop], dim=0)
    return torch.stack([src, dst], dim=0)


def build_inductive_gcn_adj(args, reference_x, query_x, device):
    n_reference = int(reference_x.shape[0])
    n_query = 0 if query_x is None else int(query_x.shape[0])
    reference_edges = build_candidate_graph_edge_index(args, reference_x.detach().cpu())
    reference_adj = build_norm_adj(
        n_reference, reference_edges, device, bidirectional=True
    ).coalesce()
    if n_query <= 0:
        return reference_adj

    query_edges = build_inductive_query_edge_index(
        args,
        reference_x.detach().cpu(),
        query_x.detach().cpu(),
    ).to(device)
    query_loop = torch.arange(n_query, device=device, dtype=torch.long) + n_reference
    query_src = torch.cat([query_edges[0], query_loop], dim=0)
    query_dst = torch.cat([query_edges[1], query_loop], dim=0)
    query_values = torch.ones(query_src.shape[0], device=device)
    query_degree = torch.zeros(n_reference + n_query, device=device)
    query_degree.index_add_(0, query_dst, query_values)
    query_values = query_values / query_degree[query_dst].clamp(min=1.0)

    reference_indices = reference_adj.indices()
    reference_values = reference_adj.values()
    combined_indices = torch.cat(
        [reference_indices, torch.stack([query_dst, query_src], dim=0)],
        dim=1,
    )
    combined_values = torch.cat([reference_values, query_values], dim=0)
    return torch.sparse_coo_tensor(
        combined_indices,
        combined_values,
        (n_reference + n_query, n_reference + n_query),
    ).coalesce()


def build_inductive_message_edges(args, reference_x, query_x, device):
    n_reference = int(reference_x.shape[0])
    n_query = 0 if query_x is None else int(query_x.shape[0])
    reference_edges = build_candidate_graph_edge_index(args, reference_x.detach().cpu())
    reference_messages = build_message_edge_index(
        n_reference,
        reference_edges,
        device,
        bidirectional=True,
    )
    if n_query <= 0:
        return reference_messages
    query_edges = build_inductive_query_edge_index(
        args,
        reference_x.detach().cpu(),
        query_x.detach().cpu(),
    ).to(device)
    query_loop = torch.arange(n_query, device=device, dtype=torch.long) + n_reference
    query_loops = torch.stack([query_loop, query_loop], dim=0)
    return torch.cat([reference_messages, query_edges, query_loops], dim=1)


def build_inductive_gat_edges(args, reference_x, query_x, device):
    """Build contract-v2 GAT edges without duplicate messages or query coupling."""
    n_reference = int(reference_x.shape[0])
    n_query = 0 if query_x is None else int(query_x.shape[0])
    reference_edges = build_gat_reference_edge_index(
        args,
        reference_x.detach().cpu(),
    ).to(device)
    if n_query <= 0:
        return reference_edges
    query_edges = build_inductive_query_edge_index(
        args,
        reference_x.detach().cpu(),
        query_x.detach().cpu(),
    ).to(device)
    query_nodes = torch.arange(n_query, device=device, dtype=torch.long) + n_reference
    query_loops = torch.stack([query_nodes, query_nodes], dim=0)
    return torch.cat([reference_edges, query_edges, query_loops], dim=1)


def build_inductive_sage_edges(args, reference_x, query_x, device):
    """Build GraphSAGE edges with root updates handling self information."""
    n_query = 0 if query_x is None else int(query_x.shape[0])
    reference_edges = build_sage_reference_edge_index(
        args,
        reference_x.detach().cpu(),
    ).to(device)
    if n_query <= 0:
        return reference_edges
    query_edges = build_inductive_query_edge_index(
        args,
        reference_x.detach().cpu(),
        query_x.detach().cpu(),
    ).to(device)
    return torch.cat([reference_edges, query_edges], dim=1)


def build_inductive_graph_transformer_edges(args, reference_x, query_x, device):
    """Build unique Graph Transformer edges without query information flow."""
    n_reference = int(reference_x.shape[0])
    n_query = 0 if query_x is None else int(query_x.shape[0])
    reference_edges = build_graph_transformer_reference_edge_index(
        args,
        reference_x.detach().cpu(),
    ).to(device)
    if n_query <= 0:
        return reference_edges
    query_edges = build_inductive_query_edge_index(
        args,
        reference_x.detach().cpu(),
        query_x.detach().cpu(),
    ).to(device)
    query_nodes = torch.arange(n_query, device=device, dtype=torch.long) + n_reference
    query_loops = torch.stack([query_nodes, query_nodes], dim=0)
    return torch.cat([reference_edges, query_edges, query_loops], dim=1)


def build_inductive_graph_structure(args, reference_x, query_x, device, graph_kind):
    if graph_kind == "gcn":
        return build_inductive_gcn_adj(args, reference_x, query_x, device)
    if graph_kind == "gat":
        return build_inductive_gat_edges(args, reference_x, query_x, device)
    if graph_kind == "sage":
        return build_inductive_sage_edges(args, reference_x, query_x, device)
    if graph_kind == GT_MODEL:
        return build_inductive_graph_transformer_edges(
            args,
            reference_x,
            query_x,
            device,
        )
    return build_inductive_message_edges(args, reference_x, query_x, device)


def _inductive_graph_scores(model, reference_model_x, query_model_x, structure):
    model.eval()
    with torch.no_grad():
        combined = torch.cat([reference_model_x, query_model_x], dim=0)
        scores = torch.sigmoid(model(combined, structure)[reference_model_x.shape[0] :]).view(-1)
    return scores.detach().cpu().numpy()


def _train_inductive_graph_model(
    args,
    model,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y,
    *,
    device,
    graph_kind,
    graph_label,
    peptide_scaler=None,
    protein_scaler=None,
):
    raw_test_x = test_x
    raw_test_y = test_y
    has_test = raw_test_x is not None and len(raw_test_x) > 0 and raw_test_y is not None
    train_x = train_x.float().to(device)
    val_x = val_x.float().to(device)
    train_y = train_y.float().to(device)
    val_y_np = val_y.view(-1).detach().cpu().numpy().astype(int)
    test_y_np = None
    train_structure = build_inductive_graph_structure(args, train_x, None, device, graph_kind)
    val_structure = build_inductive_graph_structure(args, train_x, val_x, device, graph_kind)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = build_pair_classification_criterion(args, train_y, device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    best_score, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_x, train_structure)
        loss = criterion(logits, train_y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{graph_label} produced a non-finite training loss")
        loss.backward()
        grad_clip_norm = float(getattr(args, "grad_clip_norm", 0.0) or 0.0)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        val_scores = _inductive_graph_scores(model, train_x, val_x, val_structure)
        threshold = best_threshold_by_f1(val_y_np, val_scores, args=args, for_checkpoint=True)
        val_acc, val_f1, val_auc = binary_metrics_with_threshold(
            val_y_np,
            val_scores,
            threshold=threshold,
        )
        print(
            f"[{metric_tag(args)}] Epoch {epoch + 1}/{args.epochs} | Train Loss: {loss.item():.4f} "
            f"| Val Acc: {val_acc:.4f} | Val SAcc: {val_acc:.4f} | Val F1: {val_f1:.4f} "
            f"| Val AUC: {val_auc:.4f}{binary_extra_suffix(val_y_np, val_scores, threshold=threshold, prefix='Val')}"
            + (
                f" | Val AUPR: {compute_aupr(val_y_np, val_scores):.4f}"
                if getattr(args, "report_aupr", True)
                else ""
            )
            + f" | ValThr: {threshold:.2f} | {graph_label}=inductive",
            flush=True,
        )
        selection_score = _pair_checkpoint_score(args, val_auc, val_f1, epoch + 1)
        if selection_score > best_score + args.early_stop_min_delta:
            best_score = selection_score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
        scheduler.step()
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[{metric_tag(args)}] Early stop at epoch {epoch + 1}")
            break

    print(
        f"[{metric_tag(args)}] Finished. Best Val {resolve_selection(args).checkpoint.id.upper()}: {best_score:.4f} "
        f"(Epoch {best_epoch}) | {graph_label}=inductive"
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_scores = _inductive_graph_scores(model, train_x, val_x, val_structure)
    operating_point = _fit_pair_operating_point(
        args,
        val_y_np,
        final_val_scores,
        score_kind="probability",
    )
    final_test_scores = None
    if has_test:
        # Test features are first admitted only after model selection and the
        # validation operating point are frozen.
        test_x = raw_test_x.float().to(device)
        test_y_np = raw_test_y.view(-1).detach().cpu().numpy().astype(int)
        test_structure = build_inductive_graph_structure(args, train_x, test_x, device, graph_kind)
        final_test_scores = _inductive_graph_scores(model, train_x, test_x, test_structure)
        metrics = Evaluator.score_metrics(test_y_np, final_test_scores, operating_point)
        print(
            f"[{metric_tag(args)}] Test | TestSet: {testset_tag(args)} | Acc: {metrics['acc']:.4f} "
            f"| SAcc: {metrics['sacc']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}"
            f"{Evaluator.format_extra_metrics(metrics)}"
            + (f" | AUPR: {metrics['aupr']:.4f}" if getattr(args, "report_aupr", True) else "")
            + f" | ValThr: {operating_point.threshold:.2f} | {graph_label}=inductive"
        )
    write_score_analysis(
        args,
        val_y_np,
        final_val_scores,
        test_y_np,
        final_test_scores,
        operating_point=operating_point,
    )
    save_pair_graph_nn_artifact(
        args,
        model,
        reference_features=train_x,
        peptide_scaler=peptide_scaler,
        protein_scaler=protein_scaler,
        threshold=operating_point.threshold,
    )
    return best_score


def _train_inductive_aligned_gated_graph_model(
    args,
    model,
    gate,
    train_pep_x,
    train_pro_x,
    train_y,
    val_pep_x,
    val_pro_x,
    val_y,
    test_pep_x,
    test_pro_x,
    test_y,
    *,
    device,
    graph_kind,
    graph_label,
    peptide_scaler=None,
    protein_scaler=None,
):
    raw_test_pep_x = test_pep_x
    raw_test_pro_x = test_pro_x
    raw_test_y = test_y
    has_test = (
        raw_test_pep_x is not None
        and raw_test_pro_x is not None
        and len(raw_test_pep_x) > 0
        and raw_test_y is not None
    )
    train_pep_x = train_pep_x.float().to(device)
    train_pro_x = train_pro_x.float().to(device)
    val_pep_x = val_pep_x.float().to(device)
    val_pro_x = val_pro_x.float().to(device)
    train_y = train_y.float().to(device)
    val_y_np = val_y.view(-1).detach().cpu().numpy().astype(int)
    test_y_np = None

    def graph_basis(pep, pro):
        return gate.topology_basis(pep, pro)

    def fuse_pair(pep, pro):
        return gate(pep, pro)

    train_basis = graph_basis(train_pep_x, train_pro_x)
    val_basis = graph_basis(val_pep_x, val_pro_x)
    train_structure = build_inductive_graph_structure(args, train_basis, None, device, graph_kind)
    val_structure = build_inductive_graph_structure(
        args, train_basis, val_basis, device, graph_kind
    )

    optimizer = optim.AdamW(
        list(model.parameters()) + list(gate.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = build_pair_classification_criterion(args, train_y, device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    best_score, best_state, best_gate, best_epoch, no_improve = -1.0, None, None, 0, 0

    def query_scores(query_pep, query_pro, structure):
        model.eval()
        gate.eval()
        with torch.no_grad():
            reference_features = fuse_pair(train_pep_x, train_pro_x)
            query_features = fuse_pair(query_pep, query_pro)
            combined = torch.cat([reference_features, query_features], dim=0)
            scores = torch.sigmoid(model(combined, structure)[reference_features.shape[0] :]).view(
                -1
            )
        return scores.detach().cpu().numpy()

    for epoch in range(args.epochs):
        model.train()
        gate.train()
        optimizer.zero_grad()
        logits = model(fuse_pair(train_pep_x, train_pro_x), train_structure)
        loss = criterion(logits, train_y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{graph_label} aligned_gated produced a non-finite training loss")
        loss.backward()
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
            for module in (model, gate)
            for parameter in module.parameters()
        ):
            raise RuntimeError(f"{graph_label} aligned_gated produced a non-finite gradient")
        optimizer.step()
        val_scores = query_scores(val_pep_x, val_pro_x, val_structure)
        threshold = best_threshold_by_f1(val_y_np, val_scores, args=args, for_checkpoint=True)
        val_acc, val_f1, val_auc = binary_metrics_with_threshold(
            val_y_np,
            val_scores,
            threshold=threshold,
        )
        print(
            f"[{metric_tag(args)}] Epoch {epoch + 1}/{args.epochs} | Train Loss: {loss.item():.4f} "
            f"| Val Acc: {val_acc:.4f} | Val SAcc: {val_acc:.4f} | Val F1: {val_f1:.4f} "
            f"| Val AUC: {val_auc:.4f}{binary_extra_suffix(val_y_np, val_scores, threshold=threshold, prefix='Val')}"
            + (
                f" | Val AUPR: {compute_aupr(val_y_np, val_scores):.4f}"
                if getattr(args, "report_aupr", True)
                else ""
            )
            + f" | ValThr: {threshold:.2f} | {graph_label}=inductive-gated",
            flush=True,
        )
        selection_score = _pair_checkpoint_score(args, val_auc, val_f1, epoch + 1)
        if selection_score > best_score + args.early_stop_min_delta:
            best_score = selection_score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_gate = {
                key: value.detach().cpu().clone() for key, value in gate.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
        scheduler.step()
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[{metric_tag(args)}] Early stop at epoch {epoch + 1}")
            break

    print(
        f"[{metric_tag(args)}] Finished. Best Val {resolve_selection(args).checkpoint.id.upper()}: {best_score:.4f} "
        f"(Epoch {best_epoch}) | {graph_label}=inductive-gated"
    )
    if best_state is not None and best_gate is not None:
        model.load_state_dict(best_state)
        gate.load_state_dict(best_gate)
    final_val_scores = query_scores(val_pep_x, val_pro_x, val_structure)
    operating_point = _fit_pair_operating_point(
        args,
        val_y_np,
        final_val_scores,
        score_kind="probability",
    )
    final_test_scores = None
    if has_test:
        # Keep test nodes completely outside training/model selection. They
        # become read-only query nodes only after the operating point is frozen.
        test_pep_x = raw_test_pep_x.float().to(device)
        test_pro_x = raw_test_pro_x.float().to(device)
        test_y_np = raw_test_y.view(-1).detach().cpu().numpy().astype(int)
        test_basis = graph_basis(test_pep_x, test_pro_x)
        test_structure = build_inductive_graph_structure(
            args,
            train_basis,
            test_basis,
            device,
            graph_kind,
        )
        final_test_scores = query_scores(test_pep_x, test_pro_x, test_structure)
        metrics = Evaluator.score_metrics(test_y_np, final_test_scores, operating_point)
        print(
            f"[{metric_tag(args)}] Test | TestSet: {testset_tag(args)} | Acc: {metrics['acc']:.4f} "
            f"| SAcc: {metrics['sacc']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}"
            f"{Evaluator.format_extra_metrics(metrics)}"
            + (f" | AUPR: {metrics['aupr']:.4f}" if getattr(args, "report_aupr", True) else "")
            + f" | ValThr: {operating_point.threshold:.2f} | {graph_label}=inductive-gated"
        )
    write_score_analysis(
        args,
        val_y_np,
        final_val_scores,
        test_y_np,
        final_test_scores,
        operating_point=operating_point,
    )
    save_pair_graph_nn_artifact(
        args,
        model,
        reference_peptide_features=train_pep_x,
        reference_protein_features=train_pro_x,
        gate=gate,
        peptide_scaler=peptide_scaler,
        protein_scaler=protein_scaler,
        threshold=operating_point.threshold,
    )
    return best_score


def train_gat(
    args,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x=None,
    test_y=None,
    device="cpu",
    peptide_scaler=None,
    protein_scaler=None,
):
    model = PairGATClassifier(
        input_dim=train_x.shape[1],
        hidden_dim=int(getattr(args, "gnn_hidden_dim", 256)),
        dropout=float(getattr(args, "dropout", 0.2)),
        edge_chunk_size=int(getattr(args, "gnn_edge_chunk_size", GAT_EDGE_CHUNK_SIZE)),
    ).to(device)
    return _train_inductive_graph_model(
        args,
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        device=device,
        graph_kind="gat",
        graph_label="GAT",
        peptide_scaler=peptide_scaler,
        protein_scaler=protein_scaler,
    )


@dataclass(frozen=True)
class TypedHGTGraph:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    node_type: torch.Tensor
    pair_index: torch.Tensor
    reference_pairs: int


def _validate_hgt_stream_features(peptide_x, partner_x, *, name):
    if not isinstance(peptide_x, torch.Tensor) or not isinstance(partner_x, torch.Tensor):
        raise TypeError(f"{name} HGT stream features must be tensors")
    if peptide_x.ndim != 2 or partner_x.ndim != 2:
        raise ValueError(f"{name} HGT stream features must be rank-2")
    if peptide_x.shape[0] != partner_x.shape[0]:
        raise ValueError(f"{name} HGT streams must contain the same number of pair rows")
    if not peptide_x.is_floating_point() or not partner_x.is_floating_point():
        raise TypeError(f"{name} HGT stream features must be floating-point")
    if peptide_x.shape[0] <= 0 or peptide_x.shape[1] <= 0 or partner_x.shape[1] <= 0:
        raise ValueError(f"{name} HGT stream features must be non-empty")
    if not torch.isfinite(peptide_x).all() or not torch.isfinite(partner_x).all():
        raise ValueError(f"{name} HGT stream features must be finite")


def _typed_hgt_role_edges(args, reference_x, query_x, *, semantic_relation, self_relation, device):
    edges = build_inductive_gat_edges(args, reference_x, query_x, device)
    relation = torch.full(
        (edges.shape[1],),
        int(semantic_relation),
        dtype=torch.long,
        device=device,
    )
    relation[edges[0] == edges[1]] = int(self_relation)
    return edges, relation


def build_inductive_hgt_graph(
    args,
    reference_peptide_x,
    reference_partner_x,
    query_peptide_x=None,
    query_partner_x=None,
    device="cpu",
):
    _validate_hgt_stream_features(reference_peptide_x, reference_partner_x, name="reference")
    if (query_peptide_x is None) != (query_partner_x is None):
        raise ValueError("HGT query streams must either both be provided or both be omitted")
    if query_peptide_x is not None:
        _validate_hgt_stream_features(query_peptide_x, query_partner_x, name="query")
        if int(reference_peptide_x.shape[1]) != int(query_peptide_x.shape[1]):
            raise ValueError("HGT peptide feature width changed between reference and query")
        if int(reference_partner_x.shape[1]) != int(query_partner_x.shape[1]):
            raise ValueError("HGT partner feature width changed between reference and query")

    n_reference = int(reference_peptide_x.shape[0])
    n_query = 0 if query_peptide_x is None else int(query_peptide_x.shape[0])
    n_peptide = n_reference + n_query
    peptide_edges, peptide_relations = _typed_hgt_role_edges(
        args,
        reference_peptide_x,
        query_peptide_x,
        semantic_relation=HGT_REL_PEPTIDE_SEMANTIC,
        self_relation=HGT_REL_PEPTIDE_SELF,
        device=device,
    )
    partner_edges, partner_relations = _typed_hgt_role_edges(
        args,
        reference_partner_x,
        query_partner_x,
        semantic_relation=HGT_REL_PARTNER_SEMANTIC,
        self_relation=HGT_REL_PARTNER_SELF,
        device=device,
    )
    partner_edges = partner_edges + n_peptide
    edge_index = torch.cat([peptide_edges, partner_edges], dim=1)
    edge_type = torch.cat([peptide_relations, partner_relations], dim=0)
    node_type = torch.cat(
        [
            torch.full((n_peptide,), HGT_NODE_PEPTIDE, dtype=torch.long, device=device),
            torch.full((n_peptide,), HGT_NODE_PARTNER, dtype=torch.long, device=device),
        ]
    )
    row_index = torch.arange(n_reference + n_query, dtype=torch.long, device=device)
    pair_index = torch.stack([row_index, row_index + n_peptide], dim=0)
    return TypedHGTGraph(
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type,
        pair_index=pair_index,
        reference_pairs=n_reference,
    )


class _TypedHGTLayer(nn.Module):
    def __init__(self, hidden_dim, heads, dropout, edge_chunk_size=GAT_EDGE_CHUNK_SIZE):
        super().__init__()
        if int(heads) <= 0 or int(hidden_dim) % int(heads) != 0:
            raise ValueError("HGT hidden_dim must be divisible by positive heads")
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        if int(edge_chunk_size) <= 0:
            raise ValueError("HGT edge_chunk_size must be positive")
        self.edge_chunk_size = int(edge_chunk_size)
        self.q_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.out_proj = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(2)])
        self.relation_att = nn.Parameter(torch.empty(4, self.heads, self.head_dim, self.head_dim))
        self.relation_msg = nn.Parameter(torch.empty(4, self.heads, self.head_dim, self.head_dim))
        self.relation_prior = nn.Parameter(torch.ones(4, self.heads))
        self.norm1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(2)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def _typed_project(self, h, node_type, projections):
        projected = torch.empty_like(h)
        for type_id, projection in enumerate(projections):
            mask = node_type == type_id
            if mask.any():
                projected[mask] = projection(h[mask])
        return projected.view(h.shape[0], self.heads, self.head_dim)

    @staticmethod
    def _typed_apply(h, node_type, modules):
        output = torch.empty_like(h)
        for type_id, module in enumerate(modules):
            mask = node_type == type_id
            if mask.any():
                output[mask] = module(h[mask])
        return output

    def _relation_transform(self, projected, src, relation, weights):
        """Apply one small relation matrix per edge without expanding it per edge."""

        output = projected.new_empty((src.numel(), self.heads, self.head_dim))
        for relation_id in range(int(weights.shape[0])):
            relation_edges = torch.nonzero(
                relation == relation_id,
                as_tuple=False,
            ).reshape(-1)
            for start in range(0, relation_edges.numel(), self.edge_chunk_size):
                edge_ids = relation_edges[start : start + self.edge_chunk_size]
                transformed = torch.einsum(
                    "ehd,hdf->ehf",
                    projected[src[edge_ids]],
                    weights[relation_id],
                )
                output.index_copy_(0, edge_ids, transformed)
        return output

    def forward(self, h, graph):
        src, dst = graph.edge_index
        relation = graph.edge_type
        if relation.numel() != src.numel() or relation.numel() == 0:
            raise ValueError("HGT graph requires one non-empty relation id per edge")
        if int(relation.min()) < 0 or int(relation.max()) >= 4:
            raise ValueError("HGT graph contains an unsupported relation id")
        q = self._typed_project(h, graph.node_type, self.q_proj)
        k = self._typed_project(h, graph.node_type, self.k_proj)
        v = self._typed_project(h, graph.node_type, self.v_proj)
        key = self._relation_transform(k, src, relation, self.relation_att)
        message = self._relation_transform(v, src, relation, self.relation_msg)
        score = (q[dst] * key).sum(dim=-1)
        score = score * self.relation_prior[relation] / float(self.head_dim) ** 0.5
        destination = dst.unsqueeze(1).expand(-1, self.heads)
        maximum = torch.full(
            (h.shape[0], self.heads),
            -torch.inf,
            dtype=h.dtype,
            device=h.device,
        )
        maximum.scatter_reduce_(0, destination, score, reduce="amax", include_self=True)
        exponent = torch.exp(score - maximum[dst])
        denominator = torch.zeros_like(maximum)
        denominator.index_add_(0, dst, exponent)
        alpha = exponent / denominator[dst].clamp(min=1e-9)
        aggregated = torch.zeros(
            (h.shape[0], self.heads, self.head_dim),
            dtype=h.dtype,
            device=h.device,
        )
        aggregated.index_add_(0, dst, alpha.unsqueeze(-1) * message)
        aggregated = aggregated.reshape(h.shape[0], self.hidden_dim)
        update = self._typed_apply(aggregated, graph.node_type, self.out_proj)
        h = self._typed_apply(h + self.dropout(update), graph.node_type, self.norm1)
        feed_forward = self._typed_apply(h, graph.node_type, self.ffn)
        return self._typed_apply(h + self.dropout(feed_forward), graph.node_type, self.norm2)


class PairHGTClassifier(nn.Module):
    def __init__(
        self,
        peptide_input_dim,
        partner_input_dim=None,
        hidden_dim=256,
        heads=4,
        dropout=0.2,
        layers=2,
        edge_chunk_size=GAT_EDGE_CHUNK_SIZE,
        contract_version=HGT_PREDICTOR_CONTRACT_VERSION,
    ):
        super().__init__()
        partner_input_dim = int(partner_input_dim or peptide_input_dim)
        if int(peptide_input_dim) <= 0 or partner_input_dim <= 0:
            raise ValueError("HGT input dimensions must be positive")
        if int(hidden_dim) < 4 or int(hidden_dim) % int(heads) != 0:
            raise ValueError("HGT hidden_dim must be at least four and divisible by heads")
        if int(layers) <= 0:
            raise ValueError("HGT layers must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("HGT dropout must be in [0, 1)")
        self.hgt_predictor_contract_version = int(contract_version)
        self.peptide_input_dim = int(peptide_input_dim)
        self.partner_input_dim = partner_input_dim
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.num_layers = int(layers)
        self.edge_chunk_size = int(edge_chunk_size)
        self.peptide_projection = nn.Linear(self.peptide_input_dim, self.hidden_dim)
        self.partner_projection = nn.Linear(self.partner_input_dim, self.hidden_dim)
        self.hgt_layers = nn.ModuleList(
            [
                _TypedHGTLayer(
                    self.hidden_dim,
                    self.heads,
                    dropout,
                    edge_chunk_size=self.edge_chunk_size,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward_with_embedding(self, peptide_x, partner_x, graph):
        _validate_hgt_stream_features(peptide_x, partner_x, name="model")
        if int(peptide_x.shape[1]) != self.peptide_input_dim:
            raise ValueError("HGT peptide input width does not match the model contract")
        if int(partner_x.shape[1]) != self.partner_input_dim:
            raise ValueError("HGT partner input width does not match the model contract")
        h = torch.cat(
            [self.peptide_projection(peptide_x), self.partner_projection(partner_x)],
            dim=0,
        )
        if graph.node_type.shape != (h.shape[0],):
            raise ValueError("HGT node_type does not match the number of nodes")
        for layer in self.hgt_layers:
            h = layer(h, graph)
        peptide_h = h[graph.pair_index[0]]
        partner_h = h[graph.pair_index[1]]
        edge_embedding = torch.cat(
            [peptide_h, partner_h, (peptide_h - partner_h).abs(), peptide_h * partner_h],
            dim=1,
        )
        return self.head(edge_embedding), edge_embedding

    def forward(self, peptide_x, partner_x, graph):
        logits, _embedding = self.forward_with_embedding(peptide_x, partner_x, graph)
        return logits


class LegacyPairHGTClassifier(nn.Module):
    """Historical homogeneous QKV graph attention retained for old bundles only."""

    def __init__(self, input_dim, hidden_dim=256, heads=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = max(1, hidden_dim // heads)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, self.heads * self.head_dim)
        self.k = nn.Linear(hidden_dim, self.heads * self.head_dim)
        self.v = nn.Linear(hidden_dim, self.heads * self.head_dim)
        self.out_proj = nn.Linear(self.heads * self.head_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _attn_layer(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        q = self.q(x).view(x.shape[0], self.heads, self.head_dim)
        k = self.k(x).view(x.shape[0], self.heads, self.head_dim)
        v = self.v(x).view(x.shape[0], self.heads, self.head_dim)
        score = (q[dst] * k[src]).sum(dim=-1) / float(self.head_dim) ** 0.5
        e = torch.exp(torch.clamp(score, -10.0, 10.0))
        denom = torch.zeros((x.shape[0], self.heads), dtype=x.dtype, device=x.device)
        denom.index_add_(0, dst, e)
        alpha = e / denom[dst].clamp(min=1e-9)
        out = torch.zeros((x.shape[0], self.heads, self.head_dim), dtype=x.dtype, device=x.device)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * v[src])
        return self.out_proj(out.reshape(x.shape[0], self.heads * self.head_dim))

    def forward(self, x, edge_index):
        h = self.in_proj(x)
        h = h + self.dropout(self._attn_layer(h, edge_index))
        h = h + self.dropout(self.ffn(h))
        return self.head(h)


def _inductive_hgt_scores(
    model,
    reference_peptide_x,
    reference_partner_x,
    query_peptide_x,
    query_partner_x,
    graph,
):
    model.eval()
    with torch.no_grad():
        peptide_x = torch.cat([reference_peptide_x, query_peptide_x], dim=0)
        partner_x = torch.cat([reference_partner_x, query_partner_x], dim=0)
        logits = model(peptide_x, partner_x, graph)[graph.reference_pairs :]
        scores = torch.sigmoid(logits).view(-1)
    return scores.detach().cpu().numpy()


def _train_inductive_hgt_model(
    args,
    model,
    train_peptide_x,
    train_partner_x,
    train_y,
    val_peptide_x,
    val_partner_x,
    val_y,
    test_peptide_x,
    test_partner_x,
    test_y,
    *,
    device,
    peptide_scaler=None,
    protein_scaler=None,
):
    raw_test_peptide_x = test_peptide_x
    raw_test_partner_x = test_partner_x
    raw_test_y = test_y
    has_test = (
        raw_test_peptide_x is not None
        and raw_test_partner_x is not None
        and raw_test_y is not None
        and len(raw_test_y) > 0
    )
    train_peptide_x = train_peptide_x.float().to(device)
    train_partner_x = train_partner_x.float().to(device)
    val_peptide_x = val_peptide_x.float().to(device)
    val_partner_x = val_partner_x.float().to(device)
    train_y = train_y.float().to(device)
    val_y_np = val_y.view(-1).detach().cpu().numpy().astype(int)
    test_y_np = None
    train_graph = build_inductive_hgt_graph(
        args,
        train_peptide_x,
        train_partner_x,
        device=device,
    )
    val_graph = build_inductive_hgt_graph(
        args,
        train_peptide_x,
        train_partner_x,
        val_peptide_x,
        val_partner_x,
        device=device,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = build_pair_classification_criterion(args, train_y, device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    best_score, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_peptide_x, train_partner_x, train_graph)
        loss = criterion(logits, train_y)
        if not torch.isfinite(loss):
            raise RuntimeError("HGT produced a non-finite training loss")
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        val_scores = _inductive_hgt_scores(
            model,
            train_peptide_x,
            train_partner_x,
            val_peptide_x,
            val_partner_x,
            val_graph,
        )
        threshold = best_threshold_by_f1(val_y_np, val_scores, args=args, for_checkpoint=True)
        val_acc, val_f1, val_auc = binary_metrics_with_threshold(
            val_y_np,
            val_scores,
            threshold=threshold,
        )
        print(
            f"[{metric_tag(args)}] Epoch {epoch + 1}/{args.epochs} | Train Loss: {loss.item():.4f} "
            f"| Val Acc: {val_acc:.4f} | Val SAcc: {val_acc:.4f} | Val F1: {val_f1:.4f} "
            f"| Val AUC: {val_auc:.4f}{binary_extra_suffix(val_y_np, val_scores, threshold=threshold, prefix='Val')}"
            + (
                f" | Val AUPR: {compute_aupr(val_y_np, val_scores):.4f}"
                if getattr(args, "report_aupr", True)
                else ""
            )
            + f" | ValThr: {threshold:.2f} | HGT=typed-inductive",
            flush=True,
        )
        selection_score = _pair_checkpoint_score(args, val_auc, val_f1, epoch + 1)
        if selection_score > best_score + args.early_stop_min_delta:
            best_score = selection_score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
        scheduler.step()
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[{metric_tag(args)}] Early stop at epoch {epoch + 1}")
            break
    print(
        f"[{metric_tag(args)}] Finished. Best Val {resolve_selection(args).checkpoint.id.upper()}: {best_score:.4f} "
        f"(Epoch {best_epoch}) | HGT=typed-inductive"
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_scores = _inductive_hgt_scores(
        model,
        train_peptide_x,
        train_partner_x,
        val_peptide_x,
        val_partner_x,
        val_graph,
    )
    operating_point = _fit_pair_operating_point(
        args,
        val_y_np,
        final_val_scores,
        score_kind="probability",
    )
    final_test_scores = None
    if has_test:
        test_peptide_x = raw_test_peptide_x.float().to(device)
        test_partner_x = raw_test_partner_x.float().to(device)
        test_y_np = raw_test_y.view(-1).detach().cpu().numpy().astype(int)
        test_graph = build_inductive_hgt_graph(
            args,
            train_peptide_x,
            train_partner_x,
            test_peptide_x,
            test_partner_x,
            device=device,
        )
        final_test_scores = _inductive_hgt_scores(
            model,
            train_peptide_x,
            train_partner_x,
            test_peptide_x,
            test_partner_x,
            test_graph,
        )
        metrics = Evaluator.score_metrics(test_y_np, final_test_scores, operating_point)
        print(
            f"[{metric_tag(args)}] Test | TestSet: {testset_tag(args)} | Acc: {metrics['acc']:.4f} "
            f"| SAcc: {metrics['sacc']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}"
            f"{Evaluator.format_extra_metrics(metrics)} | ValThr: {operating_point.threshold:.2f} "
            "| HGT=typed-inductive"
        )
    write_score_analysis(
        args,
        val_y_np,
        final_val_scores,
        test_y_np,
        final_test_scores,
        operating_point=operating_point,
    )
    save_pair_graph_nn_artifact(
        args,
        model,
        reference_peptide_features=train_peptide_x,
        reference_protein_features=train_partner_x,
        peptide_scaler=peptide_scaler,
        protein_scaler=protein_scaler,
        threshold=operating_point.threshold,
    )
    return best_score


def train_hgt(
    args,
    train_peptide_x,
    train_partner_x,
    train_y,
    val_peptide_x,
    val_partner_x,
    val_y,
    test_peptide_x=None,
    test_partner_x=None,
    test_y=None,
    device="cpu",
    peptide_scaler=None,
    protein_scaler=None,
):
    model = PairHGTClassifier(
        peptide_input_dim=train_peptide_x.shape[1],
        partner_input_dim=train_partner_x.shape[1],
        hidden_dim=int(getattr(args, "gnn_hidden_dim", 256)),
        heads=4,
        dropout=float(getattr(args, "dropout", 0.2)),
        edge_chunk_size=int(getattr(args, "gnn_edge_chunk_size", GAT_EDGE_CHUNK_SIZE)),
    ).to(device)
    return _train_inductive_hgt_model(
        args,
        model,
        train_peptide_x,
        train_partner_x,
        train_y,
        val_peptide_x,
        val_partner_x,
        val_y,
        test_peptide_x,
        test_partner_x,
        test_y,
        device=device,
        peptide_scaler=peptide_scaler,
        protein_scaler=protein_scaler,
    )


class PairSAGEClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        dropout=0.2,
        contract_version=SAGE_PREDICTOR_CONTRACT_VERSION,
    ):
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(hidden_dim) < 2:
            raise ValueError("hidden_dim must be at least 2")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.sage_predictor_contract_version = int(contract_version)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.lin_nei1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_self1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_nei2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_self2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _sage_layer(self, x, edge_index, lin_nei, lin_self):
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        deg = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones(src.shape[0], dtype=x.dtype, device=x.device))
        agg = agg / deg.unsqueeze(1).clamp(min=1.0)
        out = lin_self(x) + lin_nei(agg)
        return torch.relu(out)

    def forward(self, x, edge_index):
        h = self.in_proj(x)
        h = self._sage_layer(h, edge_index, self.lin_nei1, self.lin_self1)
        h = self.dropout(h)
        h = self._sage_layer(h, edge_index, self.lin_nei2, self.lin_self2)
        h = self.dropout(h)
        return self.head(h)


def train_sage(
    args,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x=None,
    test_y=None,
    device="cpu",
    peptide_scaler=None,
    protein_scaler=None,
):
    model = PairSAGEClassifier(
        input_dim=train_x.shape[1],
        hidden_dim=int(getattr(args, "gnn_hidden_dim", 256)),
        dropout=float(getattr(args, "dropout", 0.2)),
    ).to(device)
    return _train_inductive_graph_model(
        args,
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        device=device,
        graph_kind="sage",
        graph_label="GraphSAGE",
        peptide_scaler=peptide_scaler,
        protein_scaler=protein_scaler,
    )


class LegacyPairGraphTransformerClassifier(nn.Module):
    """Historical unversioned Graph Transformer bundle implementation."""

    def __init__(self, input_dim, hidden_dim=256, heads=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = max(1, hidden_dim // heads)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, self.heads * self.head_dim)
        self.k1 = nn.Linear(hidden_dim, self.heads * self.head_dim)
        self.v1 = nn.Linear(hidden_dim, self.heads * self.head_dim)
        self.o1 = nn.Linear(self.heads * self.head_dim, hidden_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.q2 = nn.Linear(hidden_dim, self.heads * self.head_dim)
        self.k2 = nn.Linear(hidden_dim, self.heads * self.head_dim)
        self.v2 = nn.Linear(hidden_dim, self.heads * self.head_dim)
        self.o2 = nn.Linear(self.heads * self.head_dim, hidden_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _attn(self, x, edge_index, q_layer, k_layer, v_layer, o_layer):
        src, dst = edge_index[0], edge_index[1]
        q = q_layer(x).view(x.shape[0], self.heads, self.head_dim)
        k = k_layer(x).view(x.shape[0], self.heads, self.head_dim)
        v = v_layer(x).view(x.shape[0], self.heads, self.head_dim)
        score = (q[dst] * k[src]).sum(dim=-1) / (float(self.head_dim) ** 0.5)
        score = torch.clamp(score, -10.0, 10.0)
        e = torch.exp(score)
        denom = torch.zeros((x.shape[0], self.heads), dtype=x.dtype, device=x.device)
        denom.index_add_(0, dst, e)
        alpha = e / denom[dst].clamp(min=1e-9)
        out = torch.zeros((x.shape[0], self.heads, self.head_dim), dtype=x.dtype, device=x.device)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * v[src])
        return o_layer(out.reshape(x.shape[0], self.heads * self.head_dim))

    def forward(self, x, edge_index):
        h = self.in_proj(x)
        h = h + self.dropout(self._attn(h, edge_index, self.q1, self.k1, self.v1, self.o1))
        h = h + self.dropout(self.ffn1(h))
        h = h + self.dropout(self._attn(h, edge_index, self.q2, self.k2, self.v2, self.o2))
        h = h + self.dropout(self.ffn2(h))
        return self.head(h)


class _GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _attention(self, h, edge_index):
        src, dst = edge_index
        q = self.q_proj(h).view(h.shape[0], self.heads, self.head_dim)
        k = self.k_proj(h).view(h.shape[0], self.heads, self.head_dim)
        v = self.v_proj(h).view(h.shape[0], self.heads, self.head_dim)
        score = (q[dst] * k[src]).sum(dim=-1) / float(self.head_dim) ** 0.5
        destination = dst.unsqueeze(1).expand(-1, self.heads)
        maximum = torch.full(
            (h.shape[0], self.heads),
            -torch.inf,
            dtype=h.dtype,
            device=h.device,
        )
        maximum.scatter_reduce_(
            0,
            destination,
            score,
            reduce="amax",
            include_self=True,
        )
        exponent = torch.exp(score - maximum[dst])
        denominator = torch.zeros_like(maximum)
        denominator.index_add_(0, dst, exponent)
        alpha = exponent / denominator[dst].clamp(min=torch.finfo(h.dtype).tiny)
        aggregated = torch.zeros(
            (h.shape[0], self.heads, self.head_dim),
            dtype=h.dtype,
            device=h.device,
        )
        aggregated.index_add_(0, dst, alpha.unsqueeze(-1) * v[src])
        return self.out_proj(aggregated.reshape(h.shape[0], self.hidden_dim))

    def forward(self, h, edge_index):
        h = self.norm1(h + self.dropout(self._attention(h, edge_index)))
        return self.norm2(h + self.dropout(self.ffn(h)))


class PairGraphTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        heads=4,
        dropout=0.2,
        layers=2,
        contract_version=GRAPH_TRANSFORMER_PREDICTOR_CONTRACT_VERSION,
    ):
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("Graph Transformer input_dim must be positive")
        if int(heads) <= 0:
            raise ValueError("Graph Transformer heads must be positive")
        if int(hidden_dim) < 2 or int(hidden_dim) % int(heads) != 0:
            raise ValueError(
                "Graph Transformer hidden_dim must be at least two and divisible by heads"
            )
        if int(layers) <= 0:
            raise ValueError("Graph Transformer layers must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Graph Transformer dropout must be in [0, 1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.layers_count = int(layers)
        self.graph_transformer_predictor_contract_version = int(contract_version)
        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.layers = nn.ModuleList(
            [
                _GraphTransformerLayer(self.hidden_dim, self.heads, float(dropout))
                for _ in range(self.layers_count)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def _validate_inputs(self, x, edge_index):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Graph Transformer node features must be a tensor")
        if x.ndim != 2:
            raise ValueError("Graph Transformer node features must be rank-2")
        if not x.is_floating_point():
            raise TypeError("Graph Transformer node features must be floating-point")
        if x.shape[0] <= 0 or x.shape[1] != self.input_dim:
            raise ValueError("Graph Transformer node features have an invalid shape")
        if not torch.isfinite(x).all():
            raise ValueError("Graph Transformer node features must be finite")
        if not isinstance(edge_index, torch.Tensor):
            raise TypeError("Graph Transformer edge_index must be a tensor")
        if edge_index.dtype != torch.long:
            raise TypeError("Graph Transformer edge_index must use torch.long")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] <= 0:
            raise ValueError("Graph Transformer edge_index must have shape [2, E] with E > 0")
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= int(x.shape[0]):
            raise ValueError("Graph Transformer edge_index contains an invalid node index")
        incoming = torch.bincount(edge_index[1], minlength=x.shape[0])
        if torch.any(incoming == 0):
            raise ValueError("Graph Transformer every node must have an incoming edge")

    def forward_with_embedding(self, x, edge_index):
        self._validate_inputs(x, edge_index)
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.head(h), h

    def forward(self, x, edge_index):
        logits, _embedding = self.forward_with_embedding(x, edge_index)
        return logits


def train_graph_transformer(
    args,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x=None,
    test_y=None,
    device="cpu",
    peptide_scaler=None,
    protein_scaler=None,
):
    model = PairGraphTransformerClassifier(
        input_dim=train_x.shape[1],
        hidden_dim=int(getattr(args, "gnn_hidden_dim", 256)),
        heads=4,
        dropout=float(getattr(args, "dropout", 0.2)),
    ).to(device)
    return _train_inductive_graph_model(
        args,
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        device=device,
        graph_kind=GT_MODEL,
        graph_label="GraphTransformer",
        peptide_scaler=peptide_scaler,
        protein_scaler=protein_scaler,
    )


def train_hgt_aligned_gated(
    args,
    train_pep_x,
    train_pro_x,
    train_y,
    val_pep_x,
    val_pro_x,
    val_y,
    test_pep_x=None,
    test_pro_x=None,
    test_y=None,
    device="cpu",
    peptide_scaler=None,
    protein_scaler=None,
):
    del (
        args,
        train_pep_x,
        train_pro_x,
        train_y,
        val_pep_x,
        val_pro_x,
        val_y,
        test_pep_x,
        test_pro_x,
        test_y,
        device,
        peptide_scaler,
        protein_scaler,
    )
    raise SystemExit(
        "HGT contract v1 requires separate typed stream nodes and does not support "
        "fusion_mode=aligned_gated. Historical gated HGT artifacts are legacy-only."
    )


def train_gcn_aligned_gated(
    args,
    train_pep_x,
    train_pro_x,
    train_y,
    val_pep_x,
    val_pro_x,
    val_y,
    test_pep_x=None,
    test_pro_x=None,
    test_y=None,
    device="cpu",
    peptide_scaler=None,
    protein_scaler=None,
):
    dimension = int(train_pep_x.shape[1])
    gate = AlignedGatedFusion(
        dimension,
        dimension,
        gate_hidden_dim=dimension,
        dropout=float(getattr(args, "dropout", 0.2)),
    ).to(device)
    model = PairGCNClassifier(
        input_dim=dimension * 3,
        hidden_dim=int(getattr(args, "gnn_hidden_dim", 256)),
        dropout=float(getattr(args, "dropout", 0.2)),
    ).to(device)
    return _train_inductive_aligned_gated_graph_model(
        args,
        model,
        gate,
        train_pep_x,
        train_pro_x,
        train_y,
        val_pep_x,
        val_pro_x,
        val_y,
        test_pep_x,
        test_pro_x,
        test_y,
        device=device,
        graph_kind="gcn",
        graph_label="GCN",
        peptide_scaler=peptide_scaler,
        protein_scaler=protein_scaler,
    )


def train_gat_aligned_gated(
    args,
    train_pep_x,
    train_pro_x,
    train_y,
    val_pep_x,
    val_pro_x,
    val_y,
    test_pep_x=None,
    test_pro_x=None,
    test_y=None,
    device="cpu",
    peptide_scaler=None,
    protein_scaler=None,
):
    dimension = int(train_pep_x.shape[1])
    gate = AlignedGatedFusion(
        dimension,
        dimension,
        gate_hidden_dim=dimension,
        dropout=float(getattr(args, "dropout", 0.2)),
    ).to(device)
    model = PairGATClassifier(
        input_dim=dimension * 3,
        hidden_dim=int(getattr(args, "gnn_hidden_dim", 256)),
        dropout=float(getattr(args, "dropout", 0.2)),
        edge_chunk_size=int(getattr(args, "gnn_edge_chunk_size", GAT_EDGE_CHUNK_SIZE)),
    ).to(device)
    return _train_inductive_aligned_gated_graph_model(
        args,
        model,
        gate,
        train_pep_x,
        train_pro_x,
        train_y,
        val_pep_x,
        val_pro_x,
        val_y,
        test_pep_x,
        test_pro_x,
        test_y,
        device=device,
        graph_kind="gat",
        graph_label="GAT",
        peptide_scaler=peptide_scaler,
        protein_scaler=protein_scaler,
    )


def encode_pair_sequences_for_dl(
    featurizer,
    rows,
    featurizer_peptide=None,
    featurizer_protein=None,
    split_name="data",
    args=None,
):
    pep_featurizer = featurizer_peptide if featurizer_peptide is not None else featurizer
    pro_featurizer = featurizer_protein if featurizer_protein is not None else featurizer
    if args is not None and str(getattr(args, "tokenizer", "")).strip().lower() == "dr":
        try:
            role_bytes = estimate_dr_dense_tensor_bytes(
                len(rows),
                encoder=str(getattr(args, "encoder", "index")),
                max_distance=getattr(
                    args,
                    "dr_max_distance",
                    DR_DEFAULT_MAX_DISTANCE,
                ),
                max_length=int(getattr(args, "max_length", 60)),
                embedding_dim=getattr(
                    getattr(pep_featurizer, "encoder", None),
                    "feature_dim",
                    None,
                ),
            )
            ensure_dr_dense_memory_budget(
                role_bytes * 12,
                limit_gib=getattr(
                    args,
                    "dr_dense_memory_limit_gib",
                    DR_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                ),
                context=("joined DR pair sequence tensors plus concurrent split working tensors"),
            )
        except DRContractError as exc:
            raise SystemExit(str(exc)) from exc
    items = []
    item_masks = []
    single_stream = (
        args is not None
        and str(getattr(args, "pair_strategy", "") or "").strip().lower() == "single_stream"
    )
    separator_id = (
        int(getattr(pep_featurizer, "vocab_size"))
        if single_stream and hasattr(pep_featurizer, "vocab_size")
        else None
    )
    iterator = tqdm(rows, desc=f"Encoding pair sequences ({split_name})", leave=False)
    for sid, pep, pro, _ in iterator:
        pep_id = f"{sid}::pep"
        pro_id = f"{sid}::pro"
        p = pep_featurizer.encode(pep, sample_id=pep_id)
        q = pro_featurizer.encode(pro, sample_id=pro_id)
        if single_stream:
            joined = join_single_stream_rows(
                p,
                q,
                separator_id=separator_id,
            )
            if isinstance(joined, ResidueModelInput):
                items.append(joined.features.float())
                item_masks.append(joined.mask.bool())
            else:
                items.append(joined.long())
        elif isinstance(p, ResidueModelInput) or isinstance(q, ResidueModelInput):
            if not isinstance(p, ResidueModelInput) or not isinstance(
                q,
                ResidueModelInput,
            ):
                raise SystemExit("PPI explicit-mask sequence mode requires matched stream inputs.")
            items.append(torch.cat([p.features, q.features], dim=0).float())
            item_masks.append(torch.cat([p.mask, q.mask], dim=0).bool())
        elif p.ndim == 1 and q.ndim == 1:
            items.append(torch.cat([p, q], dim=0).long())
        elif p.ndim == 2 and q.ndim == 2:
            items.append(torch.cat([p, q], dim=0).float())
        else:
            raise SystemExit(
                "PPI DL sequence mode requires tokenizer/encoder outputs with matched rank."
            )
    if not items:
        return torch.empty(0)
    features = torch.stack(items, dim=0)
    if item_masks:
        result = ResidueModelInput(
            features,
            torch.stack(item_masks, dim=0),
        )
    else:
        result = features
    if single_stream and args is not None:
        source_feature_dim = (
            int(result.features.shape[2]) - 3 if isinstance(result, ResidueModelInput) else None
        )
        args._single_stream_strategy_contract = build_single_stream_strategy_contract(
            result,
            separator_id=separator_id,
            source_feature_dim=source_feature_dim,
        )
    return result


def _sampled_unigram_ids(
    args,
    featurizer,
    sequence,
    *,
    epoch,
    sample_id,
    role,
):
    return featurizer.encode_sampled(
        sequence,
        seed=int(getattr(args, "seed", 42)),
        epoch=int(epoch),
        sample_id=sample_id,
        role=role,
        nbest_size=int(
            getattr(
                args,
                "unigram_nbest_size",
                UNIGRAM_DEFAULT_NBEST_SIZE,
            )
        ),
        alpha=float(getattr(args, "unigram_alpha", UNIGRAM_DEFAULT_ALPHA)),
    ).long()


def configure_joined_unigram_sampling(args, rows, peptide_featurizer, partner_featurizer):
    if not bool(getattr(args, "unigram_sampling", False)):
        args._unigram_sampling_provider = None
        return

    def provider(index, epoch):
        sample_id, peptide, partner, _ = rows[int(index)]
        peptide_ids = _sampled_unigram_ids(
            args,
            peptide_featurizer,
            peptide,
            epoch=epoch,
            sample_id=f"{sample_id}::pep",
            role="peptide",
        )
        partner_ids = _sampled_unigram_ids(
            args,
            partner_featurizer,
            partner,
            epoch=epoch,
            sample_id=f"{sample_id}::pro",
            role="partner",
        )
        if str(getattr(args, "pair_strategy", "") or "").strip().lower() == "single_stream":
            return {
                0: join_single_stream_rows(
                    peptide_ids,
                    partner_ids,
                    separator_id=int(peptide_featurizer.vocab_size),
                )
            }
        return {0: torch.cat((peptide_ids, partner_ids), dim=0)}

    args._unigram_sampling_provider = provider


def configure_separate_unigram_sampling(
    args,
    rows,
    *,
    peptide_featurizer=None,
    partner_featurizer=None,
):
    if not bool(getattr(args, "unigram_sampling", False)):
        args._unigram_sampling_provider = None
        return

    def provider(index, epoch):
        sample_id, peptide, partner, _ = rows[int(index)]
        replacements = {}
        if peptide_featurizer is not None:
            replacements[0] = _sampled_unigram_ids(
                args,
                peptide_featurizer,
                peptide,
                epoch=epoch,
                sample_id=f"{sample_id}::pep",
                role="peptide",
            )
        if partner_featurizer is not None:
            replacements[1] = _sampled_unigram_ids(
                args,
                partner_featurizer,
                partner,
                epoch=epoch,
                sample_id=f"{sample_id}::pro",
                role="partner",
            )
        return replacements

    args._unigram_sampling_provider = provider


class PairCrossAttentionClassifier(nn.Module):
    def __init__(
        self,
        seq_len,
        encoder_name,
        vocab_size=None,
        in_dim=None,
        hidden_dim=128,
        dropout=0.1,
        n_heads=4,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        hidden_dim, n_heads, dropout = validate_pair_attention_hyperparameters(
            int(hidden_dim), int(n_heads), float(dropout)
        )
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.encoder_name = str(encoder_name).lower()
        if self.encoder_name == "index":
            if vocab_size is None or int(vocab_size) <= 0:
                raise ValueError("index pair attention requires a positive vocab_size")
            self.embed = nn.Embedding(int(vocab_size), hidden_dim, padding_idx=0)
            self.proj = None
        else:
            if in_dim is None or int(in_dim) <= 0:
                raise ValueError("dense pair attention requires a positive in_dim")
            self.embed = None
            self.proj = nn.Linear(int(in_dim), hidden_dim)
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.interaction_model_id = "cross_attn"
        self.cross_attention_predictor_contract_version = CROSS_ATTENTION_PREDICTOR_CONTRACT_VERSION
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.head = build_prediction_head(
            hidden_dim * 8,
            1,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[hidden_dim],
        )

    def _split(self, x):
        if isinstance(x, ResidueModelInput):
            return (
                ResidueModelInput(
                    x.features[:, : self.seq_len],
                    x.mask[:, : self.seq_len],
                ),
                ResidueModelInput(
                    x.features[:, self.seq_len :],
                    x.mask[:, self.seq_len :],
                ),
            )
        return x[:, : self.seq_len], x[:, self.seq_len :]

    def _encode(self, x):
        explicit_mask = x.mask.bool() if isinstance(x, ResidueModelInput) else None
        values = x.features if isinstance(x, ResidueModelInput) else x
        if self.embed is not None:
            if values.ndim != 2 or values.dtype == torch.bool or values.is_floating_point():
                raise ValueError("index pair attention requires integer [batch, tokens] inputs")
            h = self.embed(values.long())
            mask = values != 0
        else:
            if values.ndim != 3 or not values.is_floating_point():
                raise ValueError(
                    "dense pair attention requires floating [batch, tokens, features] inputs"
                )
            h = self.proj(values.float())
            mask = values.abs().sum(dim=-1) > 0
        if explicit_mask is not None:
            mask = explicit_mask
        if h.ndim != 3 or mask.ndim != 2 or h.shape[:2] != mask.shape:
            raise ValueError("pair attention inputs must have shape [batch, tokens, features]")
        if not torch.isfinite(h).all():
            raise ValueError("pair attention inputs must contain only finite values")
        if h.shape[0] == 0 or not mask.any(dim=1).all():
            raise ValueError("pair attention inputs must contain at least one valid token per row")
        return h, mask

    def _pool(self, h, m):
        weights = m.float().unsqueeze(-1)
        return (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def _pair_role_embeddings(self, x, partner_x=None):
        pep_x, pro_x = self._split(x) if partner_x is None else (x, partner_x)
        pep_h, pep_m = self._encode(pep_x)
        pro_h, pro_m = self._encode(pro_x)
        pep2pro, _ = self.attn(
            pep_h,
            pro_h,
            pro_h,
            key_padding_mask=~pro_m,
            need_weights=False,
        )
        p0, q0 = self._pool(pep_h, pep_m), self._pool(pro_h, pro_m)
        pa = self._pool(pep2pro, pep_m)
        if int(getattr(self, "cross_attention_predictor_contract_version", 0)) < 1:
            pro2pep, _ = self.attn(
                pro_h,
                pep_h,
                pep_h,
                key_padding_mask=~pep_m,
                need_weights=False,
            )
            qa = self._pool(pro2pep, pro_m)
            z = torch.cat(
                [p0, q0, (p0 - q0).abs(), p0 * q0, pa, qa, (pa - qa).abs(), pa * qa],
                dim=1,
            )
            return z, p0, q0
        z = torch.cat(
            [p0, q0, (p0 - q0).abs(), p0 * q0, pa, q0, (pa - q0).abs(), pa * q0],
            dim=1,
        )
        return z, p0, q0

    def _pair_embedding(self, x, partner_x=None):
        return self._pair_role_embeddings(x, partner_x)[0]

    def forward_with_embedding(self, x, partner_x=None):
        z = self._pair_embedding(x, partner_x)
        return self.head(z), z

    @property
    def semantic_embedding_dim(self) -> int:
        return self.hidden_dim * 8

    @property
    def semantic_role_embedding_dims(self) -> tuple[int, int] | None:
        return (self.hidden_dim, self.hidden_dim)

    def forward_with_role_embeddings(self, x, partner_x=None):
        z, peptide_embedding, partner_embedding = self._pair_role_embeddings(x, partner_x)
        return self.head(z), z, peptide_embedding, partner_embedding

    def forward(self, x, partner_x=None):
        logits, _ = self.forward_with_embedding(x, partner_x)
        return logits


class LegacyPairCrossAttentionClassifier(PairCrossAttentionClassifier):
    """Historical shared-parameter bidirectional behavior for unversioned bundles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_attention_predictor_contract_version = 0


class PairBiCrossAttentionClassifier(PairCrossAttentionClassifier):
    def __init__(
        self,
        seq_len,
        encoder_name,
        vocab_size=None,
        in_dim=None,
        hidden_dim=128,
        dropout=0.1,
        n_heads=4,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__(
            seq_len,
            encoder_name,
            vocab_size=vocab_size,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            n_heads=n_heads,
            head_hidden_dims=head_hidden_dims,
            head_activation=head_activation,
            head_dropout=head_dropout,
        )
        self.interaction_model_id = "bi_cross_attn"
        self.bi_cross_attention_predictor_contract_version = (
            CROSS_ATTENTION_PREDICTOR_CONTRACT_VERSION
        )
        del self.attn
        self.attn_p2q = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_q2p = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=float(dropout),
            batch_first=True,
        )

    def _pair_role_embeddings(self, x, partner_x=None):
        pep_x, pro_x = self._split(x) if partner_x is None else (x, partner_x)
        pep_h, pep_m = self._encode(pep_x)
        pro_h, pro_m = self._encode(pro_x)
        pep2pro, _ = self.attn_p2q(
            pep_h,
            pro_h,
            pro_h,
            key_padding_mask=~pro_m,
            need_weights=False,
        )
        pro2pep, _ = self.attn_q2p(
            pro_h,
            pep_h,
            pep_h,
            key_padding_mask=~pep_m,
            need_weights=False,
        )
        p0, q0 = self._pool(pep_h, pep_m), self._pool(pro_h, pro_m)
        pa, qa = self._pool(pep2pro, pep_m), self._pool(pro2pep, pro_m)
        z = torch.cat(
            [p0, q0, (p0 - q0).abs(), p0 * q0, pa, qa, (pa - qa).abs(), pa * qa],
            dim=1,
        )
        return z, p0, q0


class LegacyPairBiCrossAttentionClassifier(PairBiCrossAttentionClassifier):
    """Historical Bi bundle layout with an unused inherited attention module."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bi_cross_attention_predictor_contract_version = 0
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=float(self.attn_p2q.dropout),
            batch_first=True,
        )


class HeterogeneousPairCrossAttentionClassifier(nn.Module):
    """Cross-attention head with independent peptide and partner projections."""

    def __init__(
        self,
        peptide_encoder_name,
        partner_encoder_name,
        peptide_vocab_size=None,
        partner_vocab_size=None,
        peptide_in_dim=None,
        partner_in_dim=None,
        hidden_dim=128,
        dropout=0.1,
        n_heads=4,
        separate_attention=False,
        head_hidden_dims=None,
        head_activation="relu",
        head_dropout=None,
    ):
        super().__init__()
        hidden_dim, n_heads, dropout = validate_pair_attention_hyperparameters(
            int(hidden_dim), int(n_heads), float(dropout)
        )
        self.peptide_encoder_name = str(peptide_encoder_name).lower()
        self.partner_encoder_name = str(partner_encoder_name).lower()
        self.peptide_embed, self.peptide_projection = self._build_projection(
            self.peptide_encoder_name, peptide_vocab_size, peptide_in_dim, hidden_dim
        )
        self.partner_embed, self.partner_projection = self._build_projection(
            self.partner_encoder_name, partner_vocab_size, partner_in_dim, hidden_dim
        )
        self.separate_attention = bool(separate_attention)
        self.interaction_model_id = "bi_cross_attn" if self.separate_attention else "cross_attn"
        self.cross_attention_predictor_contract_version = (
            0 if self.separate_attention else CROSS_ATTENTION_PREDICTOR_CONTRACT_VERSION
        )
        self.bi_cross_attention_predictor_contract_version = (
            CROSS_ATTENTION_PREDICTOR_CONTRACT_VERSION if self.separate_attention else 0
        )
        self.attn_p2q = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_q2p = (
            nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            if self.separate_attention
            else None
        )
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head = build_prediction_head(
            hidden_dim * 8,
            1,
            hidden_dims=head_hidden_dims,
            activation=head_activation,
            dropout=dropout if head_dropout is None else head_dropout,
            default_hidden_dims=[hidden_dim],
        )

    @staticmethod
    def _build_projection(encoder_name, vocab_size, in_dim, hidden_dim):
        if encoder_name == "index":
            if vocab_size is None or int(vocab_size) <= 0:
                raise ValueError("index stream requires vocab_size")
            return nn.Embedding(int(vocab_size), hidden_dim, padding_idx=0), None
        if in_dim is None or int(in_dim) <= 0:
            raise ValueError("dense token stream requires input dimension")
        return None, nn.Linear(int(in_dim), hidden_dim)

    @staticmethod
    def _mask(values):
        if isinstance(values, ResidueModelInput):
            return values.mask.bool()
        if values.ndim == 2:
            return values.ne(0)
        return values.abs().sum(dim=-1).gt(0)

    @staticmethod
    def _pool(hidden, mask):
        weights = mask.float().unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _encode(values, embed, projection):
        features = values.features if isinstance(values, ResidueModelInput) else values
        if embed is not None:
            if features.ndim != 2 or features.dtype == torch.bool or features.is_floating_point():
                raise ValueError("index pair attention requires integer [batch, tokens] inputs")
            return embed(features.long())
        if features.ndim != 3 or not features.is_floating_point():
            raise ValueError(
                "dense pair attention requires floating [batch, tokens, features] inputs"
            )
        return projection(features.float())

    def forward_with_embedding(self, peptide_tokens, partner_tokens):
        peptide_mask = self._mask(peptide_tokens)
        partner_mask = self._mask(partner_tokens)
        peptide_hidden = self._encode(peptide_tokens, self.peptide_embed, self.peptide_projection)
        partner_hidden = self._encode(partner_tokens, self.partner_embed, self.partner_projection)
        if (
            peptide_hidden.ndim != 3
            or partner_hidden.ndim != 3
            or peptide_mask.shape != peptide_hidden.shape[:2]
            or partner_mask.shape != partner_hidden.shape[:2]
            or peptide_hidden.shape[0] != partner_hidden.shape[0]
        ):
            raise ValueError(
                "pair attention streams must have compatible [batch, tokens, features] shapes"
            )
        if not torch.isfinite(peptide_hidden).all() or not torch.isfinite(partner_hidden).all():
            raise ValueError("pair attention inputs must contain only finite values")
        if (
            peptide_hidden.shape[0] == 0
            or not peptide_mask.any(dim=1).all()
            or not partner_mask.any(dim=1).all()
        ):
            raise ValueError("pair attention inputs must contain at least one valid token per row")
        peptide_attended, _ = self.attn_p2q(
            peptide_hidden,
            partner_hidden,
            partner_hidden,
            key_padding_mask=~partner_mask,
            need_weights=False,
        )
        p0 = self._pool(peptide_hidden, peptide_mask)
        q0 = self._pool(partner_hidden, partner_mask)
        pa = self._pool(peptide_attended, peptide_mask)
        if self.attn_q2p is None:
            pair_embedding = torch.cat(
                [p0, q0, (p0 - q0).abs(), p0 * q0, pa, q0, (pa - q0).abs(), pa * q0],
                dim=1,
            )
            return self.head(pair_embedding), pair_embedding
        partner_attended, _ = self.attn_q2p(
            partner_hidden,
            peptide_hidden,
            peptide_hidden,
            key_padding_mask=~peptide_mask,
            need_weights=False,
        )
        qa = self._pool(partner_attended, partner_mask)
        pair_embedding = torch.cat(
            [p0, q0, (p0 - q0).abs(), p0 * q0, pa, qa, (pa - qa).abs(), pa * qa],
            dim=1,
        )
        return self.head(pair_embedding), pair_embedding

    def _pair_role_embeddings(self, peptide_tokens, partner_tokens):
        peptide_mask = self._mask(peptide_tokens)
        partner_mask = self._mask(partner_tokens)
        peptide_hidden = self._encode(peptide_tokens, self.peptide_embed, self.peptide_projection)
        partner_hidden = self._encode(partner_tokens, self.partner_embed, self.partner_projection)
        peptide_attended, _ = self.attn_p2q(
            peptide_hidden,
            partner_hidden,
            partner_hidden,
            key_padding_mask=~partner_mask,
            need_weights=False,
        )
        p0 = self._pool(peptide_hidden, peptide_mask)
        q0 = self._pool(partner_hidden, partner_mask)
        pa = self._pool(peptide_attended, peptide_mask)
        if self.attn_q2p is None:
            pair_embedding = torch.cat(
                [p0, q0, (p0 - q0).abs(), p0 * q0, pa, q0, (pa - q0).abs(), pa * q0],
                dim=1,
            )
            return pair_embedding, p0, q0
        partner_attended, _ = self.attn_q2p(
            partner_hidden,
            peptide_hidden,
            peptide_hidden,
            key_padding_mask=~peptide_mask,
            need_weights=False,
        )
        qa = self._pool(partner_attended, partner_mask)
        pair_embedding = torch.cat(
            [p0, q0, (p0 - q0).abs(), p0 * q0, pa, qa, (pa - qa).abs(), pa * qa],
            dim=1,
        )
        return pair_embedding, p0, q0

    @property
    def semantic_embedding_dim(self) -> int:
        return self.hidden_dim * 8

    @property
    def semantic_role_embedding_dims(self) -> tuple[int, int] | None:
        return (self.hidden_dim, self.hidden_dim)

    def forward_with_role_embeddings(self, peptide_tokens, partner_tokens):
        pair_embedding, peptide_embedding, partner_embedding = self._pair_role_embeddings(
            peptide_tokens, partner_tokens
        )
        return (
            self.head(pair_embedding),
            pair_embedding,
            peptide_embedding,
            partner_embedding,
        )

    def forward(self, peptide_tokens, partner_tokens):
        logits, _ = self.forward_with_embedding(peptide_tokens, partner_tokens)
        return logits


class HeterogeneousPairBiCrossAttentionClassifier(HeterogeneousPairCrossAttentionClassifier):
    def __init__(self, *args, **kwargs):
        kwargs["separate_attention"] = True
        super().__init__(*args, **kwargs)


class LegacyHeterogeneousPairCrossAttentionClassifier(HeterogeneousPairCrossAttentionClassifier):
    """Historical shared-parameter bidirectional behavior for unversioned bundles."""

    def __init__(self, *args, **kwargs):
        kwargs["separate_attention"] = False
        super().__init__(*args, **kwargs)
        self.cross_attention_predictor_contract_version = 0
        self.attn_q2p = self.attn_p2q


def _collect_heterogeneous_token_scores(model, loader, device):
    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for peptide_tokens, partner_tokens, labels in loader:
            logits = model(peptide_tokens.to(device), partner_tokens.to(device))
            scores.append(torch.sigmoid(logits).view(-1).cpu())
            targets.append(labels.view(-1).cpu())
    if not scores:
        return None, None
    return torch.cat(targets).numpy().astype(int), torch.cat(scores).numpy()


def _heterogeneous_token_loss(model, loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for peptide_tokens, partner_tokens, labels in loader:
            labels = labels.to(device)
            loss = criterion(model(peptide_tokens.to(device), partner_tokens.to(device)), labels)
            total += float(loss.item()) * int(labels.shape[0])
            count += int(labels.shape[0])
    return total / max(1, count)


def train_heterogeneous_cross_attention_model(
    args,
    train_peptide,
    train_partner,
    train_y,
    val_peptide,
    val_partner,
    val_y,
    test_peptide=None,
    test_partner=None,
    test_y=None,
    device="cpu",
    semantic_profile=None,
    train_pair_ids=None,
    profile_rows=None,
    return_model=False,
    evaluate_test=True,
):
    training_mode = str(getattr(args, "training_mode", "") or "").strip().lower()
    if training_mode:
        validate_training_mode_classifier(
            training_mode,
            str(getattr(args, "classifier", "") or "mlp").strip().lower(),
        )
    peptide_encoder = (
        "index"
        if train_peptide.ndim == 2
        else str(args.streams["peptide"].get("encoder", "ppi_fusion"))
    )
    partner_encoder = (
        "index"
        if train_partner.ndim == 2
        else str(args.streams["partner"].get("encoder", "ppi_fusion"))
    )
    model_cls = (
        HeterogeneousPairBiCrossAttentionClassifier
        if str(getattr(args, "interaction_model", "cross_attn")) == "bi_cross_attn"
        else HeterogeneousPairCrossAttentionClassifier
    )
    kwargs = {
        "peptide_encoder_name": peptide_encoder,
        "partner_encoder_name": partner_encoder,
        "hidden_dim": int(getattr(args, "interaction_hidden_dim", 128)),
        "dropout": float(args.dropout),
        "n_heads": int(getattr(args, "interaction_heads", 4)),
        **prediction_head_kwargs(args),
    }
    sampled_vocab_sizes = getattr(args, "_unigram_stream_vocab_sizes", {})
    if train_peptide.ndim == 2:
        kwargs["peptide_vocab_size"] = int(
            sampled_vocab_sizes.get(
                "peptide",
                int(train_peptide.max().item()) + 1 if train_peptide.numel() else 1,
            )
        )
    else:
        kwargs["peptide_in_dim"] = int(train_peptide.shape[2])
    if train_partner.ndim == 2:
        kwargs["partner_vocab_size"] = int(
            sampled_vocab_sizes.get(
                "partner",
                int(train_partner.max().item()) + 1 if train_partner.numel() else 1,
            )
        )
    else:
        kwargs["partner_in_dim"] = int(train_partner.shape[2])
    model = model_cls(**kwargs).to(device)
    prepare_semantic_projectors(
        model,
        args,
        sample_inputs=(
            _sample_batch_to_device(train_peptide, device),
            _sample_batch_to_device(train_partner, device),
        ),
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = build_pair_classification_criterion(args, train_y, device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    train_loader = _pair_train_loader(args, train_peptide, train_partner, train_y)
    val_loader = DataLoader(
        _tensor_columns_dataset(val_peptide, val_partner, val_y),
        batch_size=args.batch_size,
        shuffle=False,
    )
    best_score, best_state, best_epoch, best_threshold, no_improve = -1.0, None, 0, 0.5, 0
    for epoch in range(args.epochs):
        _set_pair_loader_epoch(args, train_loader, epoch)
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            peptide_tokens, partner_tokens, labels, row_ids = batch
            labels = labels.to(device)
            optimizer.zero_grad()
            peptide_tokens = peptide_tokens.to(device)
            partner_tokens = partner_tokens.to(device)
            logits = model(peptide_tokens, partner_tokens)
            classification_loss = criterion(logits, labels)
            loss, _ = add_partner_semantic_loss(
                args,
                model,
                classification_loss,
                semantic_profile,
                row_ids,
                (train_peptide, train_partner),
                device,
            )
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item())
        y_true, y_score = _collect_heterogeneous_token_scores(model, val_loader, device)
        val_acc, val_f1, val_auc, threshold = binary_metrics_with_best_threshold(
            y_true, y_score, args=args, for_checkpoint=True
        )
        val_loss = _heterogeneous_token_loss(model, val_loader, criterion, device)
        val_aupr = compute_aupr(y_true, y_score)
        print(
            f"[{metric_tag(args)}] Epoch {epoch + 1}/{args.epochs} | Train Loss: {total_loss / max(1, len(train_loader)):.4f} "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} "
            f"| Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f} | ValThr: {threshold:.2f}"
        )
        selection_score = _pair_checkpoint_score(args, val_auc, val_f1, epoch + 1)
        if selection_score > best_score + args.early_stop_min_delta:
            best_score, best_epoch, best_threshold = selection_score, epoch + 1, threshold
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
        scheduler.step()
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_true, final_val_scores = _collect_heterogeneous_token_scores(
        model, val_loader, device
    )
    operating_point = _fit_pair_operating_point(
        args,
        final_val_true,
        final_val_scores,
        score_kind="probability",
    )
    args._operating_point = operating_point.to_dict()
    best_threshold = operating_point.threshold
    final_test_true, final_test_scores = None, None
    if (
        evaluate_test
        and test_peptide is not None
        and test_partner is not None
        and test_y is not None
        and len(test_y) > 0
    ):
        test_loader = DataLoader(
            _tensor_columns_dataset(test_peptide, test_partner, test_y),
            batch_size=args.batch_size,
            shuffle=False,
        )
        final_test_true, final_test_scores = _collect_heterogeneous_token_scores(
            model, test_loader, device
        )
        test_acc, test_f1, test_auc = binary_metrics_with_threshold(
            final_test_true, final_test_scores, best_threshold
        )
        print(
            f"[{metric_tag(args)}] Test | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}"
        )
    write_score_analysis(
        args,
        final_val_true,
        final_val_scores,
        final_test_true,
        final_test_scores,
        operating_point=operating_point,
    )
    save_heterogeneous_pair_token_artifact(
        args,
        model,
        train_peptide,
        train_partner,
        kwargs,
        threshold=best_threshold,
        profile_rows=profile_rows,
    )
    print(
        f"[{metric_tag(args)}] Finished. Best Val {resolve_selection(args).checkpoint.id.upper()}: {best_score:.4f} (Epoch {best_epoch})"
    )
    if return_model:
        return best_score, model
    return best_score


def train_heterogeneous_cross_attention_then_classifier(
    args,
    train_peptide,
    train_partner,
    train_y,
    val_peptide,
    val_partner,
    val_y,
    test_peptide=None,
    test_partner=None,
    test_y=None,
    device="cpu",
    semantic_profile=None,
    train_pair_ids=None,
    profile_rows=None,
):
    classifier = str(getattr(args, "classifier", "") or "").strip().lower()
    if classifier not in CLASSICAL_CLASSIFIERS:
        raise SystemExit(
            "heterogeneous interaction export_embeddings requires a classical downstream classifier"
        )
    tag = metric_tag(args)
    print(
        f"[{tag}] Stage 1/2: Train heterogeneous interaction encoder with a temporary neural head"
    )
    _, model = train_heterogeneous_cross_attention_model(
        args,
        train_peptide,
        train_partner,
        train_y,
        val_peptide,
        val_partner,
        val_y,
        test_peptide,
        test_partner,
        test_y,
        device,
        semantic_profile=semantic_profile,
        train_pair_ids=train_pair_ids,
        profile_rows=profile_rows,
        return_model=True,
        evaluate_test=False,
    )
    split_total = 3 if test_peptide is not None else 2
    emit_run_progress(
        "embedding_export",
        "running",
        0,
        split_total,
        feature=classifier,
        unit="split",
        split="train",
    )
    train_embeddings = extract_pair_model_embeddings(
        model, train_peptide, train_partner, args.batch_size, device, "train"
    )
    emit_run_progress(
        "embedding_export", "running", 1, split_total, feature=classifier, unit="split", split="val"
    )
    val_embeddings = extract_pair_model_embeddings(
        model, val_peptide, val_partner, args.batch_size, device, "val"
    )
    test_embeddings = None
    if test_peptide is not None and test_partner is not None:
        emit_run_progress(
            "embedding_export",
            "running",
            2,
            split_total,
            feature=classifier,
            unit="split",
            split="test",
        )
        test_embeddings = extract_pair_model_embeddings(
            model, test_peptide, test_partner, args.batch_size, device, "test"
        )
    emit_run_progress(
        "embedding_export",
        "complete",
        split_total,
        split_total,
        feature=classifier,
        unit="split",
        split="complete",
    )
    encoder_path = _save_pair_embedding_encoder(args, model)
    route_info = _pair_token_artifact_info(
        args,
        model,
        train_peptide,
        profile_rows=profile_rows,
        partner_train_x=train_partner,
    )
    _configure_pair_embedding_artifacts(
        args,
        route_info,
        encoder_path,
        train_embeddings.shape[1] if train_embeddings.ndim > 1 else 0,
        classifier,
        architecture_contract=pair_attention_predictor_contract(model),
    )
    if classifier == "svm" and str(getattr(args, "artifact_dir", "") or ""):
        save_svm_resume_bundle(
            args.artifact_dir,
            train_x=train_embeddings,
            train_y=train_y,
            val_x=val_embeddings,
            val_y=val_y,
            test_x=test_embeddings,
            test_y=test_y,
            task_type=args.task_type,
            artifact_source=str(
                getattr(args, "_artifact_source", "") or PAIR_EMBEDDING_ARTIFACT_SOURCE
            ),
            artifact_info=dict(getattr(args, "_artifact_info", {}) or {}),
        )
    return train_ml(
        args,
        train_embeddings,
        train_y,
        val_embeddings,
        val_y,
        test_embeddings,
        test_y,
        classifier=classifier,
    )


def train_sequence_model(
    args,
    featurizer,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x=None,
    test_y=None,
    device="cpu",
    semantic_profile=None,
    train_pair_ids=None,
    profile_rows=None,
    return_model=False,
    evaluate_test=True,
):
    training_mode = str(getattr(args, "training_mode", "") or "").strip().lower()
    if training_mode:
        validate_training_mode_classifier(
            training_mode,
            str(getattr(args, "classifier", "") or "mlp").strip().lower(),
        )
    tokenizer_name = str(getattr(args, "tokenizer", "") or "").strip().lower()
    if tokenizer_name in PRETRAINED_TOKENIZERS:
        if not _is_explicit_residue_input(train_x) or not _is_explicit_residue_input(val_x):
            raise SystemExit(
                "pretrained Single-stream sequence models require explicit "
                "residue inputs with masks"
            )
        adapter_dim = int(getattr(args, "residue_adapter_dim", RESIDUE_ADAPTER_DIM_DEFAULT))
        if (
            adapter_dim < RESIDUE_ADAPTER_DIM_MIN
            or adapter_dim > RESIDUE_ADAPTER_DIM_MAX
            or adapter_dim % RESIDUE_ADAPTER_DIM_MULTIPLE != 0
        ):
            raise SystemExit(
                "residue_adapter_dim must be between "
                f"{RESIDUE_ADAPTER_DIM_MIN} and {RESIDUE_ADAPTER_DIM_MAX} "
                f"and divisible by {RESIDUE_ADAPTER_DIM_MULTIPLE}"
            )
        model = PretrainedResiduePredictor(
            architecture=args.model,
            input_dim=int(train_x.shape[2]),
            adapter_dim=adapter_dim,
            max_length=int(train_x.shape[1]),
            num_classes=1,
            dropout=args.dropout,
            **prediction_head_kwargs(args),
        ).to(device)
    else:
        pair_meta = type("PairFeaturizerMeta", (), {})()
        pair_meta.max_length = int(train_x.shape[1])
        if getattr(args, "encoder", "") == "index":
            strategy_contract = getattr(args, "_single_stream_strategy_contract", {}) or {}
            join_contract = strategy_contract.get("sequence_join", {})
            separator_id = int(join_contract.get("index_separator_id") or 0)
            pair_meta.vocab_size = (
                separator_id + 1 if separator_id > 0 else int(featurizer.vocab_size)
            )
        elif getattr(args, "encoder", "") == "onehot":
            pair_meta.vocab_size = int(train_x.shape[2])
        else:
            pair_meta.feature_dim = int(train_x.shape[2])

        model = (
            ModelFactory()
            .create(
                args.model,
                args.encoder,
                pair_meta,
                num_classes=1,
                dropout=args.dropout,
                **prediction_head_kwargs(args),
            )
            .to(device)
        )
    prepare_semantic_projectors(
        model,
        args,
        sample_inputs=_sample_batch_to_device(train_x, device),
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = build_pair_classification_criterion(args, train_y, device)

    train_loader = _pair_train_loader(args, train_x, train_y)
    val_loader = _sequence_data_loader(
        val_x,
        val_y,
        batch_size=args.batch_size,
    )
    print_interaction_semantic_status(args, semantic_profile)

    best_score, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    for epoch in range(args.epochs):
        _set_pair_loader_epoch(args, train_loader, epoch)
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x, y, row_ids = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            classification_loss = criterion(logits, y)
            loss, _ = add_partner_semantic_loss(
                args,
                model,
                classification_loss,
                semantic_profile,
                row_ids,
                train_x,
                device,
            )
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()
        val_metrics = _pair_nn_validation_metrics(args, model, val_loader, device)
        val_acc, val_sacc, val_f1, val_auc = (
            val_metrics["acc"],
            val_metrics["sacc"],
            val_metrics["f1"],
            val_metrics["auc"],
        )
        train_loss = total_loss / max(1, len(train_loader))
        val_loss = eval_binary_loss(model, val_loader, criterion, device)
        msg = f"[{metric_tag(args)}] Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val SAcc: {val_sacc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}{Evaluator.format_extra_metrics(val_metrics, prefix='Val')}"
        if getattr(args, "report_aupr", True):
            y_true, y_score = collect_binary_scores(model, val_loader, device)
            msg += f" | Val AUPR: {compute_aupr(y_true, y_score):.4f}"
        print(msg)
        selection_score = _pair_checkpoint_score(args, val_auc, val_f1, epoch + 1)
        if selection_score > best_score + args.early_stop_min_delta:
            best_score = selection_score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[{metric_tag(args)}] Early stop at epoch {epoch + 1}")
            break

    print(
        f"[{metric_tag(args)}] Finished. Best Val {resolve_selection(args).checkpoint.id.upper()}: {best_score:.4f} (Epoch {best_epoch})"
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_true, final_val_scores = collect_binary_scores(model, val_loader, device)
    operating_point = _fit_pair_operating_point(
        args,
        final_val_true,
        final_val_scores,
        score_kind="probability",
    )
    final_test_true, final_test_scores = None, None
    if evaluate_test and test_x is not None and test_y is not None and len(test_x) > 0:
        test_loader = _sequence_data_loader(
            test_x,
            test_y,
            batch_size=args.batch_size,
        )
        final_test_true, final_test_scores = collect_binary_scores(model, test_loader, device)
        test_metrics = Evaluator.score_metrics(final_test_true, final_test_scores, operating_point)
        test_acc, test_sacc, test_f1, test_auc = (
            test_metrics["acc"],
            test_metrics["sacc"],
            test_metrics["f1"],
            test_metrics["auc"],
        )
        msg = f"[{metric_tag(args)}] Test | TestSet: {testset_tag(args)} | Acc: {test_acc:.4f} | SAcc: {test_sacc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}{Evaluator.format_extra_metrics(test_metrics)}"
        if getattr(args, "report_aupr", True):
            msg += f" | AUPR: {compute_aupr(final_test_true, final_test_scores):.4f}"
        print(msg)
    write_score_analysis(
        args,
        final_val_true,
        final_val_scores,
        final_test_true,
        final_test_scores,
        operating_point=operating_point,
    )
    best_threshold = operating_point.threshold
    save_pair_sequence_nn_artifact(
        args, model, train_x, threshold=best_threshold, profile_rows=profile_rows
    )
    if return_model:
        return best_score, model
    return best_score


def _fit_pair_operating_point(args, y_true, scores, *, for_checkpoint=False, **kwargs):
    """Keep checkpoint-only threshold fitting separate from the frozen final point."""
    if args is None:
        return Evaluator.fit_operating_point(y_true, scores, **kwargs)
    config = resolve_selection(args)
    point = Evaluator.fit_operating_point(
        y_true,
        scores,
        objective=config.checkpoint_threshold if for_checkpoint else config.threshold,
        search_profile=config.checkpoint_search_profile if for_checkpoint else config.search_profile,
        **kwargs,
    )
    args._selection_validation_metrics = Evaluator.score_metrics(y_true, scores, point)
    if not for_checkpoint:
        args._operating_point = point.to_dict()
    return point


def _pair_checkpoint_metrics(args, y, scores):
    """Score an epoch's existing validation predictions without freezing its threshold."""
    point = _fit_pair_operating_point(
        args, y, scores, score_kind="probability", for_checkpoint=True
    )
    return Evaluator.score_metrics(y, scores, point)


def _pair_nn_validation_metrics(args, model, loader, device):
    y, scores = collect_binary_scores(model, loader, device)
    return _pair_checkpoint_metrics(args, y, scores)


def _pair_checkpoint_score(args, auc, f1, epoch):
    spec = resolve_selection(args).checkpoint
    metrics = dict(getattr(args, "_selection_validation_metrics", {}) or {})
    metrics.update(auc=auc, f1=f1)
    key = spec.key(metrics)
    history = [] if epoch == 1 else getattr(args, "_selection_history", [])
    # Keep unavailable reporting metrics JSON-safe while rejecting unavailable
    # selection metrics in ObjectiveSpec.key above.
    metrics = {
        k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()
    }
    history.append({"epoch": int(epoch), "selection_key": list(key), "metrics": metrics})
    args._selection_history = history
    best = None
    selected_epoch = 0
    for item in history:
        candidate = tuple(item["selection_key"])
        if best is None or candidate[0] > best[0] + args.early_stop_min_delta:
            best, selected_epoch = candidate, item["epoch"]
    args._selected_epoch = selected_epoch
    artifact = str(getattr(args, "artifact_dir", "") or "")
    if artifact:
        os.makedirs(artifact, exist_ok=True)
        with open(os.path.join(artifact, "selection_history.json"), "w") as handle:
            json.dump(
                {
                    **selection_fields(args),
                    "epochs": history,
                    "early_stop_min_delta": float(args.early_stop_min_delta),
                },
                handle,
                indent=2,
                allow_nan=False,
            )
    return key[0]


def best_threshold_by_f1(y_true, scores, args=None, *, for_checkpoint=False):
    operating_point = _fit_pair_operating_point(
        args, y_true, scores, score_kind="probability", for_checkpoint=for_checkpoint
    )
    return operating_point.threshold


def binary_metric_details_with_threshold(y_true, scores, threshold):
    return Evaluator.binary_score_metrics(y_true, scores, threshold=threshold)


def binary_extra_suffix(y_true, scores, threshold, prefix=""):
    metrics = binary_metric_details_with_threshold(y_true, scores, threshold=threshold)
    return Evaluator.format_extra_metrics(metrics, prefix=prefix)


def binary_metrics_with_threshold(y_true, scores, threshold):
    metrics = Evaluator.binary_score_metrics(y_true, scores, threshold=threshold)
    return (
        metrics["acc"],
        metrics["f1"],
        metrics["auc"] if metrics["auc"] is not None else float("nan"),
    )


def binary_metrics_with_best_threshold(y_true, scores, args=None, *, for_checkpoint=False):
    threshold = best_threshold_by_f1(y_true, scores, args=args, for_checkpoint=for_checkpoint)
    acc, f1, auc = binary_metrics_with_threshold(y_true, scores, threshold=threshold)
    if not np.isfinite(auc) and (args is None or resolve_selection(args).checkpoint.id == "auc"):
        raise ValueError(
            "validation ROC-AUC is undefined because the validation split contains one class"
        )
    return acc, f1, auc, threshold


def train_cross_attention_model(
    args,
    featurizer,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x=None,
    test_y=None,
    device="cpu",
    semantic_profile=None,
    train_pair_ids=None,
    profile_rows=None,
    return_model=False,
    evaluate_test=True,
    train_partner_x=None,
    val_partner_x=None,
    test_partner_x=None,
):
    training_mode = str(getattr(args, "training_mode", "") or "").strip().lower()
    if training_mode:
        validate_training_mode_classifier(
            training_mode,
            str(getattr(args, "classifier", "") or "mlp").strip().lower(),
        )
    hidden_dim, n_heads, attention_dropout = validate_pair_attention_hyperparameters(
        int(getattr(args, "interaction_hidden_dim", 128)),
        int(getattr(args, "interaction_heads", 4)),
        float(args.dropout),
    )
    separate_streams = train_partner_x is not None
    if separate_streams != (val_partner_x is not None):
        raise ValueError("train and validation pair-token inputs must use the same stream layout")
    if test_x is not None and separate_streams != (test_partner_x is not None):
        raise ValueError("test pair-token inputs must use the same stream layout as training")
    if separate_streams:
        if train_x.shape[0] != train_partner_x.shape[0] or val_x.shape[0] != val_partner_x.shape[0]:
            raise ValueError("homogeneous pair-token streams must contain the same number of rows")
        if train_x.ndim != train_partner_x.ndim or val_x.ndim != val_partner_x.ndim:
            raise ValueError("homogeneous pair-token streams must have matching ranks")
        if train_x.ndim > 2 and train_x.shape[2] != train_partner_x.shape[2]:
            raise ValueError("homogeneous pair-token streams must have matching token dimensions")

    enc_name = str(getattr(args, "encoder", "index")).lower()
    if str(getattr(args, "tokenizer", "")).strip().lower() not in PRETRAINED_TOKENIZERS:
        enc_name = effective_pair_vector_encoder(args)
    model_cls = (
        PairBiCrossAttentionClassifier
        if str(getattr(args, "model", "")).lower() == BI_CROSS_ATTN_MODEL
        else PairCrossAttentionClassifier
    )
    model_encoder_name = "index" if train_x.ndim == 2 else enc_name
    seq_len = (
        int(train_x.shape[1])
        if separate_streams
        else int(getattr(args, "max_length", train_x.shape[1] // 2))
    )
    if model_encoder_name == "index":
        model = model_cls(
            seq_len=seq_len,
            encoder_name=model_encoder_name,
            vocab_size=int(featurizer.vocab_size),
            hidden_dim=hidden_dim,
            dropout=attention_dropout,
            n_heads=n_heads,
            **prediction_head_kwargs(args),
        ).to(device)
    else:
        in_dim = int(train_x.shape[2])
        model = model_cls(
            seq_len=seq_len,
            encoder_name=enc_name,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            dropout=attention_dropout,
            n_heads=n_heads,
            **prediction_head_kwargs(args),
        ).to(device)
    if separate_streams:
        prepare_semantic_projectors(
            model,
            args,
            sample_inputs=(
                _sample_batch_to_device(train_x, device),
                _sample_batch_to_device(train_partner_x, device),
            ),
        )
    else:
        prepare_semantic_projectors(
            model,
            args,
            sample_inputs=_sample_batch_to_device(train_x, device),
        )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = build_pair_classification_criterion(args, train_y, device)
    if separate_streams:
        train_loader = _pair_train_loader(args, train_x, train_partner_x, train_y)
    else:
        train_loader = _pair_train_loader(args, train_x, train_y)
    val_ds = (
        _tensor_columns_dataset(val_x, val_partner_x, val_y)
        if separate_streams
        else _tensor_columns_dataset(val_x, val_y)
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print_interaction_semantic_status(args, semantic_profile)
    best_score, best_state, best_epoch, no_improve = -1.0, None, 0, 0
    for epoch in range(args.epochs):
        _set_pair_loader_epoch(args, train_loader, epoch)
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            if separate_streams:
                x, partner_x, y, row_ids = batch
            else:
                x, y, row_ids = batch
                partner_x = None
            x, y = x.to(device), y.to(device)
            if partner_x is not None:
                partner_x = partner_x.to(device)
            optimizer.zero_grad()
            logits = model(x, partner_x)
            classification_loss = criterion(logits, y)
            full_inputs = (train_x, train_partner_x) if separate_streams else train_x
            loss, _ = add_partner_semantic_loss(
                args,
                model,
                classification_loss,
                semantic_profile,
                row_ids,
                full_inputs,
                device,
            )
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()
        if separate_streams:
            val_true, val_scores = collect_binary_scores_pair(model, val_loader, device)
            val_loss = eval_binary_pair_loss(model, val_loader, criterion, device)
        else:
            val_true, val_scores = collect_binary_scores(model, val_loader, device)
            val_loss = eval_binary_loss(model, val_loader, criterion, device)
        val_metrics = _pair_checkpoint_metrics(args, val_true, val_scores)
        val_acc, val_sacc, val_f1, val_auc = (
            val_metrics["acc"],
            val_metrics["sacc"],
            val_metrics["f1"],
            val_metrics["auc"],
        )
        train_loss = total_loss / max(1, len(train_loader))
        msg = f"[{metric_tag(args)}] Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val SAcc: {val_sacc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}{Evaluator.format_extra_metrics(val_metrics, prefix='Val')}"
        if getattr(args, "report_aupr", True):
            msg += f" | Val AUPR: {compute_aupr(val_true, val_scores):.4f}"
        print(msg)
        selection_score = _pair_checkpoint_score(args, val_auc, val_f1, epoch + 1)
        if selection_score > best_score + args.early_stop_min_delta:
            best_score = selection_score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"[{metric_tag(args)}] Early stop at epoch {epoch + 1}")
            break
    print(
        f"[{metric_tag(args)}] Finished. Best Val {resolve_selection(args).checkpoint.id.upper()}: {best_score:.4f} (Epoch {best_epoch})"
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    if separate_streams:
        final_val_true, final_val_scores = collect_binary_scores_pair(model, val_loader, device)
    else:
        final_val_true, final_val_scores = collect_binary_scores(model, val_loader, device)
    operating_point = _fit_pair_operating_point(
        args, final_val_true, final_val_scores, score_kind="probability"
    )
    final_test_true, final_test_scores = None, None
    if evaluate_test and test_x is not None and test_y is not None and len(test_x) > 0:
        test_ds = (
            _tensor_columns_dataset(test_x, test_partner_x, test_y)
            if separate_streams
            else _tensor_columns_dataset(test_x, test_y)
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        if separate_streams:
            final_test_true, final_test_scores = collect_binary_scores_pair(
                model, test_loader, device
            )
        else:
            final_test_true, final_test_scores = collect_binary_scores(model, test_loader, device)
        test_metrics = Evaluator.score_metrics(final_test_true, final_test_scores, operating_point)
        test_acc, test_sacc, test_f1, test_auc = (
            test_metrics["acc"],
            test_metrics["sacc"],
            test_metrics["f1"],
            test_metrics["auc"],
        )
        msg = f"[{metric_tag(args)}] Test | TestSet: {testset_tag(args)} | Acc: {test_acc:.4f} | SAcc: {test_sacc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}{Evaluator.format_extra_metrics(test_metrics)}"
        if getattr(args, "report_aupr", True):
            msg += f" | AUPR: {compute_aupr(final_test_true, final_test_scores):.4f}"
        print(msg)
    write_score_analysis(
        args,
        final_val_true,
        final_val_scores,
        final_test_true,
        final_test_scores,
        operating_point=operating_point,
    )
    best_threshold = operating_point.threshold
    save_pair_token_nn_artifact(
        args,
        model,
        train_x,
        threshold=best_threshold,
        profile_rows=profile_rows,
        partner_train_x=train_partner_x,
    )
    if return_model:
        return best_score, model
    return best_score


def train_ml(args, train_x, train_y, val_x, val_y, test_x=None, test_y=None, classifier=None):
    trainer = ClassicalMLTrainer(args)
    train_ds = _tensor_columns_dataset(train_x, train_y)
    val_ds = _tensor_columns_dataset(val_x, val_y)
    test_ds = (
        _tensor_columns_dataset(test_x, test_y)
        if test_x is not None and test_y is not None and len(test_x) > 0
        else None
    )
    classifier = classifier or args.model
    if classifier == "svm":
        return trainer.run_svm(train_ds, val_ds, test_ds)
    if classifier == "calibrated_svm":
        return trainer.run_calibrated_svm(train_ds, val_ds, test_ds)
    if classifier == "rf":
        return trainer.run_rf(train_ds, val_ds, test_ds)
    if classifier == "extratrees":
        return trainer.run_extratrees(train_ds, val_ds, test_ds)
    if classifier == "histgb":
        return trainer.run_histgb(train_ds, val_ds, test_ds)
    if classifier == "xgb":
        return trainer.run_xgb(train_ds, val_ds, test_ds)
    if classifier == "lgb":
        return trainer.run_lgb(train_ds, val_ds, test_ds)
    if classifier == "logreg":
        return trainer.run_logreg(train_ds, val_ds, test_ds)
    raise SystemExit(f"Unsupported classifier: {classifier}")


def extract_sequence_model_embeddings(model, x, batch_size, device, split_name):
    loader = _sequence_data_loader(x, batch_size=batch_size)
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for (batch_x,) in tqdm(loader, desc=f"Extracting {split_name} embeddings"):
            _, embeddings = model.forward_with_embedding(batch_x.to(device))
            all_embeddings.append(embeddings.detach().cpu())
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty((0, 0))


def extract_pair_model_embeddings(model, peptide_x, partner_x, batch_size, device, split_name):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for start in tqdm(
            range(0, int(peptide_x.shape[0]), int(batch_size)),
            desc=f"Extracting {split_name} pair embeddings",
        ):
            end = start + int(batch_size)
            batch_peptide = (
                peptide_x.batch_slice(start, end)
                if isinstance(peptide_x, ResidueModelInput)
                else peptide_x[start:end]
            )
            batch_partner = (
                partner_x.batch_slice(start, end)
                if isinstance(partner_x, ResidueModelInput)
                else partner_x[start:end]
            )
            _, embeddings = model.forward_with_embedding(
                batch_peptide.to(device),
                batch_partner.to(device),
            )
            all_embeddings.append(embeddings.detach().cpu())
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty((0, 0))


def train_sequence_model_then_ml(
    args,
    featurizer,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x=None,
    test_y=None,
    device="cpu",
    semantic_profile=None,
    train_pair_ids=None,
    profile_rows=None,
):
    classifier = str(getattr(args, "classifier", "") or "").strip().lower()
    training_mode = str(getattr(args, "training_mode", "") or "").strip().lower()
    if training_mode:
        validate_training_mode_classifier(training_mode, classifier)
    if classifier not in CLASSICAL_CLASSIFIERS:
        raise SystemExit(
            "export_embeddings requires classifier in {svm,rf,extratrees,histgb,xgb,lgb,logreg}"
        )
    tag = metric_tag(args)
    print(
        f"[{tag}] Stage 1/2: Train modeling backbone with a temporary neural head "
        "for embedding learning"
    )
    emit_run_progress(
        "representation", "running", 0, max(1, int(args.epochs)), feature="sequence", unit="epoch"
    )
    _, model = train_sequence_model(
        args,
        featurizer,
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        device,
        semantic_profile=semantic_profile,
        train_pair_ids=train_pair_ids,
        profile_rows=profile_rows,
        return_model=True,
        evaluate_test=False,
    )
    emit_run_progress(
        "representation",
        "complete",
        max(1, int(args.epochs)),
        max(1, int(args.epochs)),
        feature="sequence",
        unit="epoch",
    )
    print(f"[{tag}] Stage 1/2 complete: restored best encoder checkpoint for embedding export")
    split_total = 3 if test_x is not None else 2
    emit_run_progress(
        "embedding_export",
        "running",
        0,
        split_total,
        feature=classifier,
        unit="split",
        split="train",
    )
    train_embeddings = extract_sequence_model_embeddings(
        model, train_x, args.batch_size, device, "train"
    )
    emit_run_progress(
        "embedding_export", "running", 1, split_total, feature=classifier, unit="split", split="val"
    )
    val_embeddings = extract_sequence_model_embeddings(model, val_x, args.batch_size, device, "val")
    if test_x is not None:
        emit_run_progress(
            "embedding_export",
            "running",
            2,
            split_total,
            feature=classifier,
            unit="split",
            split="test",
        )
        test_embeddings = extract_sequence_model_embeddings(
            model, test_x, args.batch_size, device, "test"
        )
    else:
        test_embeddings = None
    emit_run_progress(
        "embedding_export",
        "complete",
        split_total,
        split_total,
        feature=classifier,
        unit="split",
        split="complete",
    )
    encoder_path = _save_pair_embedding_encoder(args, model)
    route_info = _pair_sequence_artifact_info(args, model, train_x, profile_rows=profile_rows)
    _configure_pair_embedding_artifacts(
        args,
        route_info,
        encoder_path,
        train_embeddings.shape[1] if train_embeddings.ndim > 1 else 0,
        classifier,
        architecture_contract=sequence_predictor_contract(model),
    )
    if classifier == "svm" and str(getattr(args, "artifact_dir", "") or ""):
        save_svm_resume_bundle(
            args.artifact_dir,
            train_x=train_embeddings,
            train_y=train_y,
            val_x=val_embeddings,
            val_y=val_y,
            test_x=test_embeddings,
            test_y=test_y,
            task_type=args.task_type,
            artifact_source=str(getattr(args, "_artifact_source", "") or "pair_embedding_features"),
            artifact_info=dict(getattr(args, "_artifact_info", {}) or {}),
        )
    print(f"[{tag}] Stage 2/2: Train downstream classifier on exported embeddings: {classifier}")
    return train_ml(
        args,
        train_embeddings,
        train_y,
        val_embeddings,
        val_y,
        test_embeddings,
        test_y,
        classifier=classifier,
    )


def train_cross_attention_then_classifier(
    args,
    featurizer,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x=None,
    test_y=None,
    device="cpu",
    semantic_profile=None,
    train_pair_ids=None,
    profile_rows=None,
    train_partner_x=None,
    val_partner_x=None,
    test_partner_x=None,
):
    classifier = str(getattr(args, "classifier", "") or "").strip().lower()
    training_mode = str(getattr(args, "training_mode", "") or "").strip().lower()
    if training_mode:
        validate_training_mode_classifier(training_mode, classifier)
    if classifier not in CLASSICAL_CLASSIFIERS:
        raise SystemExit(
            "cross-attention export_embeddings requires a classical downstream classifier"
        )
    tag = metric_tag(args)
    print(
        f"[{tag}] Stage 1/2: Train interaction encoder with a temporary neural head "
        "for interaction-embedding learning"
    )
    emit_run_progress(
        "representation",
        "running",
        0,
        max(1, int(args.epochs)),
        feature="interaction",
        unit="epoch",
    )
    _, model = train_cross_attention_model(
        args,
        featurizer,
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        device,
        semantic_profile=semantic_profile,
        train_pair_ids=train_pair_ids,
        profile_rows=profile_rows,
        return_model=True,
        evaluate_test=False,
        train_partner_x=train_partner_x,
        val_partner_x=val_partner_x,
        test_partner_x=test_partner_x,
    )
    emit_run_progress(
        "representation",
        "complete",
        max(1, int(args.epochs)),
        max(1, int(args.epochs)),
        feature="interaction",
        unit="epoch",
    )
    print(f"[{tag}] Stage 1/2 complete: restored best interaction checkpoint for embedding export")
    split_total = 3 if test_x is not None else 2
    emit_run_progress(
        "embedding_export",
        "running",
        0,
        split_total,
        feature=classifier,
        unit="split",
        split="train",
    )
    if train_partner_x is not None:
        train_embeddings = extract_pair_model_embeddings(
            model, train_x, train_partner_x, args.batch_size, device, "train"
        )
        emit_run_progress(
            "embedding_export",
            "running",
            1,
            split_total,
            feature=classifier,
            unit="split",
            split="val",
        )
        val_embeddings = extract_pair_model_embeddings(
            model, val_x, val_partner_x, args.batch_size, device, "val"
        )
        if test_x is not None:
            emit_run_progress(
                "embedding_export",
                "running",
                2,
                split_total,
                feature=classifier,
                unit="split",
                split="test",
            )
        test_embeddings = (
            extract_pair_model_embeddings(
                model, test_x, test_partner_x, args.batch_size, device, "test"
            )
            if test_x is not None
            else None
        )
    else:
        train_embeddings = extract_sequence_model_embeddings(
            model, train_x, args.batch_size, device, "train"
        )
        emit_run_progress(
            "embedding_export",
            "running",
            1,
            split_total,
            feature=classifier,
            unit="split",
            split="val",
        )
        val_embeddings = extract_sequence_model_embeddings(
            model, val_x, args.batch_size, device, "val"
        )
        if test_x is not None:
            emit_run_progress(
                "embedding_export",
                "running",
                2,
                split_total,
                feature=classifier,
                unit="split",
                split="test",
            )
        test_embeddings = (
            extract_sequence_model_embeddings(model, test_x, args.batch_size, device, "test")
            if test_x is not None
            else None
        )
    emit_run_progress(
        "embedding_export",
        "complete",
        split_total,
        split_total,
        feature=classifier,
        unit="split",
        split="complete",
    )
    encoder_path = _save_pair_embedding_encoder(args, model)
    route_info = _pair_token_artifact_info(
        args,
        model,
        train_x,
        profile_rows=profile_rows,
        partner_train_x=train_partner_x,
    )
    _configure_pair_embedding_artifacts(
        args,
        route_info,
        encoder_path,
        train_embeddings.shape[1] if train_embeddings.ndim > 1 else 0,
        classifier,
        architecture_contract=pair_attention_predictor_contract(model),
    )
    if classifier == "svm" and str(getattr(args, "artifact_dir", "") or ""):
        save_svm_resume_bundle(
            args.artifact_dir,
            train_x=train_embeddings,
            train_y=train_y,
            val_x=val_embeddings,
            val_y=val_y,
            test_x=test_embeddings,
            test_y=test_y,
            task_type=args.task_type,
            artifact_source=str(getattr(args, "_artifact_source", "") or "pair_embedding_features"),
            artifact_info=dict(getattr(args, "_artifact_info", {}) or {}),
        )
    print(
        f"[{tag}] Stage 2/2: Train downstream classifier on exported interaction embeddings: {classifier}"
    )
    if classifier == "mlp":
        train_embeddings, val_embeddings, test_embeddings = scale_dense_features(
            train_embeddings,
            val_embeddings,
            test_embeddings,
            mode=getattr(args, "dl_scaler", "standard"),
        )
        return train_mlp(
            args=args,
            train_x=train_embeddings,
            train_y=train_y,
            val_x=val_embeddings,
            val_y=val_y,
            test_x=test_embeddings,
            test_y=test_y,
            device=device,
            semantic_profile=semantic_profile,
            train_pair_ids=train_pair_ids,
        )
    return train_ml(
        args,
        train_embeddings,
        train_y,
        val_embeddings,
        val_y,
        test_embeddings,
        test_y,
        classifier=classifier,
    )


def _stream_runtime(args, role):
    runtime = getattr(args, "_stream_runtime", None)
    if runtime is None:
        runtime = {}
        args._stream_runtime = runtime
    return runtime.setdefault(role, {})


def _release_stream_encoder(tokenizer, encoder):
    model = getattr(encoder, "model", None)
    for candidate in (model, encoder):
        if candidate is not None and hasattr(candidate, "to"):
            try:
                candidate.to("cpu")
            except Exception:
                pass
    del tokenizer, encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _classic_role_token_tensor(
    args,
    featurizer,
    rows,
    role_name,
    encoder_name,
):
    if str(getattr(args, "tokenizer", "")).strip().lower() == "dr":
        try:
            matrix_bytes = estimate_dr_dense_tensor_bytes(
                len(rows),
                encoder=encoder_name,
                max_distance=getattr(
                    args,
                    "dr_max_distance",
                    DR_DEFAULT_MAX_DISTANCE,
                ),
                max_length=int(getattr(args, "max_length", 60)),
                embedding_dim=getattr(
                    getattr(featurizer, "encoder", None),
                    "feature_dim",
                    None,
                ),
            )
            ensure_dr_dense_memory_budget(
                matrix_bytes * 6,
                limit_gib=getattr(
                    args,
                    "dr_dense_memory_limit_gib",
                    DR_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
                ),
                context=("DR pair token tensor plus concurrent role/split working tensors"),
            )
        except DRContractError as exc:
            raise SystemExit(str(exc)) from exc
    values = []
    masks = []
    for sample_id, peptide, partner, _ in tqdm(
        rows,
        desc=f"Encoding tokens/{role_name}",
        leave=False,
    ):
        sequence = peptide if role_name == "peptide" else partner
        encode_id = sample_id
        if encoder_name in PROFILE_ENCODERS:
            encode_id = f"{sample_id}::{'pep' if role_name == 'peptide' else 'pro'}"
        encoded = featurizer.encode(sequence, sample_id=encode_id)
        if isinstance(encoded, ResidueModelInput):
            values.append(encoded.features.float())
            masks.append(encoded.mask.bool())
            continue
        if not isinstance(encoded, torch.Tensor):
            encoded = torch.tensor(encoded)
        values.append(encoded.long() if encoder_name == "index" else encoded.float())
    if not values:
        return torch.empty((0, 0), dtype=torch.long if encoder_name == "index" else torch.float)
    try:
        features = torch.stack(values, dim=0)
        if masks:
            return ResidueModelInput(
                features,
                torch.stack(masks, dim=0),
            )
        return features
    except RuntimeError as exc:
        raise SystemExit(
            f"Stream {role_name} produced variable token shapes; set an explicit max_length."
        ) from exc


def _prepare_classic_stream(
    args,
    role,
    task_root,
    train_rows,
    val_rows,
    train_csv,
    val_csv,
    test_csv,
):
    stream_args = namespace_for_stream(args, role)
    role_name = "peptide" if role == "peptide" else "protein"
    tokenizer_name = str(stream_args.tokenizer).strip().lower()
    encoder_name = str(stream_args.encoder).strip().lower()
    if tokenizer_name not in CLASSIC_TOKENIZERS:
        raise SystemExit(f"Unsupported classic tokenizer for {role} stream: {tokenizer_name}")
    allowed = CLASSIC_ENCODER_OPTIONS.get(tokenizer_name, [])
    if encoder_name != "ppi_fusion" and encoder_name not in allowed:
        raise SystemExit(
            f"Invalid {role} stream encoder for tokenizer={tokenizer_name}: "
            f"{encoder_name}. Allowed: {', '.join(allowed)}"
        )
    ensure_subword_tokenizer_ready(
        stream_args,
        task_root,
        train_rows,
        val_rows,
        sequence_role=role,
    )
    ensure_ppi_static_embedding_ready(
        stream_args,
        task_root,
        train_rows,
        val_rows,
        role_name=role,
    )
    if not ensure_pair_profile_cache_ready(
        stream_args,
        task_root,
        train_csv,
        val_csv,
        test_csv,
    ):
        raise SystemExit(f"Profile cache build failed for the {role} stream.")
    featurizer = build_classic_featurizer(
        stream_args,
        task_root,
        train_rows,
        sequence_role=role,
    )
    runtime = _stream_runtime(args, role)
    if getattr(stream_args, "_static_embedding_provenance", None):
        runtime["static_embedding"] = dict(stream_args._static_embedding_provenance)
    if hasattr(stream_args, "_classic_tfidf_idf"):
        runtime["tfidf_idf"] = stream_args._classic_tfidf_idf
    if hasattr(stream_args, "_frequency_encoder_contract"):
        runtime["frequency_encoder_contract"] = dict(stream_args._frequency_encoder_contract)
    if hasattr(stream_args, "_textrank_contract"):
        runtime["textrank_contract"] = dict(stream_args._textrank_contract)
    if hasattr(stream_args, "_covariance_encoder_contract"):
        runtime["covariance_encoder_contract"] = dict(stream_args._covariance_encoder_contract)
    if hasattr(stream_args, "_pdt_encoder_contract"):
        runtime["pdt_encoder_contract"] = dict(stream_args._pdt_encoder_contract)
    if hasattr(stream_args, "_pssm_covariance_encoder_contract"):
        runtime["pssm_covariance_encoder_contract"] = dict(
            stream_args._pssm_covariance_encoder_contract
        )
    if hasattr(stream_args, "_frequency_profile_manifest_sha256"):
        runtime["frequency_profile_manifest_sha256"] = str(
            stream_args._frequency_profile_manifest_sha256
        )
        runtime["frequency_profile_provenance"] = dict(
            getattr(stream_args, "_frequency_profile_provenance", {}) or {}
        )
    if hasattr(stream_args, "_profile_manifest_sha256"):
        runtime["profile_manifest_sha256"] = str(stream_args._profile_manifest_sha256)
        runtime["profile_manifest_provenance"] = dict(
            getattr(stream_args, "_profile_manifest_provenance", {}) or {}
        )
    resolved_params = {}
    for key in [
        "bpe_path",
        "unigram_path",
        "wordpiece_path",
        "word2vec_path",
        "glove_path",
        "fasttext_path",
        "pssm_index_path",
        "ac_pssm_blosum62_hybrid_index_path",
        "cc_pssm_blosum62_hybrid_index_path",
        "acc_pssm_blosum62_hybrid_index_path",
        "psfm_index_path",
        "frequency_profile_index_path",
    ]:
        value = str(getattr(stream_args, key, "") or "").strip()
        if value:
            resolved_params[key] = value
    if tokenizer_name == "wordpiece":
        resolved_params["wordpiece_vocab_size"] = int(
            getattr(stream_args, "wordpiece_vocab_size", 1000)
        )
        resolved_params["wordpiece_max_raw_length"] = validate_wordpiece_max_raw_length(
            getattr(
                stream_args,
                "wordpiece_max_raw_length",
                WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    elif tokenizer_name == "unigram":
        resolved_params["unigram_vocab_size"] = int(getattr(stream_args, "unigram_vocab_size", 1000))
        resolved_params["unigram_max_raw_length"] = validate_unigram_max_raw_length(
            getattr(
                stream_args,
                "unigram_max_raw_length",
                UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    runtime["resolved_params"] = resolved_params
    return stream_args, featurizer, encoder_name, role_name


def _load_stream_pooled_cache(
    args,
    role: str,
    sequences: list[str],
) -> torch.Tensor:
    """Load the frozen pooled embedding cache for one stream's training split.

    Dense-hard partner selection reuses these embeddings. Cache provenance is
    checked here; ``prepare_ppi_semantic_features`` validates model identity.
    """

    spec = args.streams[role]
    if str(spec.get("feature_route", "") or "").strip().lower() != "pretrained":
        raise SystemExit(
            "semantic_partner_scheme=dense_hard requires pretrained pooled "
            f"stream caches for role={role}"
        )
    role_name = "peptide" if role == "peptide" else "protein"
    tokenizer = str(spec.get("tokenizer", "") or "").strip().lower()
    stream_args = namespace_for_stream(args, role)
    save_dir = os.path.join(str(args.feature_dir), tokenizer)
    save_path = os.path.join(save_dir, f"train_{role_name}.pt")
    expected_meta = _feature_cache_meta(
        stream_args,
        "train",
        role_name,
        list(sequences),
        encoder_name=tokenizer,
    )
    meta_path = Path(_cache_meta_path(save_path))
    stable_keys = (
        "cache_version",
        "split",
        "role",
        "tokenizer",
        "encoder",
        "max_length",
        "row_count",
        "input_sha256",
        "train_sha256",
    )
    actual_meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                actual_meta = json.load(handle)
        except Exception:
            actual_meta = {}
    mismatches = [key for key in stable_keys if expected_meta.get(key) != actual_meta.get(key)]
    if mismatches:
        raise RuntimeError(
            "pooled stream embedding cache is missing or stale for "
            f"semantic_partner_scheme=dense_hard: {save_path}; mismatches=" + ",".join(mismatches)
        )
    try:
        cached = torch.load(
            save_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        cached = torch.load(save_path, map_location="cpu", mmap=True)
    if not (
        isinstance(cached, torch.Tensor) and cached.ndim == 2 and cached.shape[0] == len(sequences)
    ):
        raise RuntimeError(f"pooled stream embedding cache has an incompatible shape: {save_path}")
    return cached.float()


def _extract_stream_vector_splits(
    args,
    role,
    sequences,
    rows,
    task_root,
    train_rows,
    val_rows,
    train_csv,
    val_csv,
    test_csv,
    device,
):
    spec = args.streams[role]
    role_name = "peptide" if role == "peptide" else "protein"
    if spec["feature_route"] == "pretrained":
        stream_args = namespace_for_stream(args, role)
        tokenizer, encoder, _ = build_pretrained_tokenizer_encoder(stream_args, device)
        runtime = _stream_runtime(args, role)
        runtime["resolved_model_source"] = str(
            getattr(stream_args, "_resolved_pretrained_model_source", "")
            or spec.get("model_source", "")
        )
        if getattr(stream_args, "_resolved_pretrained_model_identity", None):
            runtime["model_identity"] = dict(stream_args._resolved_pretrained_model_identity)
        if getattr(stream_args, "_resolved_pretrained_runtime_environment", None):
            runtime["runtime_environment"] = dict(
                stream_args._resolved_pretrained_runtime_environment
            )
        result = tuple(
            extract_embeddings(
                stream_args,
                tokenizer,
                encoder,
                split_sequences,
                split_name,
                role_name,
                device,
            )
            if split_sequences
            else None
            for split_name, split_sequences in zip(("train", "val", "test"), sequences)
        )
        _release_stream_encoder(tokenizer, encoder)
        return result
    stream_args, featurizer, encoder_name, role_name = _prepare_classic_stream(
        args,
        role,
        task_root,
        train_rows,
        val_rows,
        train_csv,
        val_csv,
        test_csv,
    )
    return tuple(
        extract_classic_features(
            stream_args,
            featurizer,
            split_rows,
            split_name,
            role_name,
            encoder_name=encoder_name,
        )
        if split_rows
        else None
        for split_name, split_rows in zip(("train", "val", "test"), rows)
    )


def _extract_stream_token_splits(
    args,
    role,
    sequences,
    rows,
    task_root,
    train_rows,
    val_rows,
    train_csv,
    val_csv,
    test_csv,
    device,
):
    spec = args.streams[role]
    role_name = "peptide" if role == "peptide" else "protein"
    if spec["feature_route"] == "pretrained":
        stream_args = namespace_for_stream(args, role)
        tokenizer, encoder, _ = build_pretrained_tokenizer_encoder(stream_args, device)
        runtime = _stream_runtime(args, role)
        runtime["resolved_model_source"] = str(
            getattr(stream_args, "_resolved_pretrained_model_source", "")
            or spec.get("model_source", "")
        )
        if getattr(stream_args, "_resolved_pretrained_model_identity", None):
            runtime["model_identity"] = dict(stream_args._resolved_pretrained_model_identity)
        if getattr(stream_args, "_resolved_pretrained_runtime_environment", None):
            runtime["runtime_environment"] = dict(
                stream_args._resolved_pretrained_runtime_environment
            )
        result = tuple(
            extract_pretrained_token_sequences(
                stream_args,
                tokenizer,
                encoder,
                split_sequences,
                split_name,
                role_name,
                device,
            )
            if split_sequences
            else None
            for split_name, split_sequences in zip(("train", "val", "test"), sequences)
        )
        _release_stream_encoder(tokenizer, encoder)
        return result
    stream_args, featurizer, encoder_name, _ = _prepare_classic_stream(
        args,
        role,
        task_root,
        train_rows,
        val_rows,
        train_csv,
        val_csv,
        test_csv,
    )
    if encoder_name not in DL_SEQUENCE_ENCODERS:
        raise SystemExit(
            f"Token-level {role} stream requires a sequence-preserving encoder in "
            f"{DL_SEQUENCE_ENCODERS}, got {encoder_name}."
        )
    if (
        bool(getattr(args, "unigram_sampling", False))
        and str(getattr(stream_args, "tokenizer", "")).lower() == "unigram"
    ):
        role_featurizers = getattr(
            args,
            "_unigram_stream_featurizers",
            None,
        )
        if role_featurizers is None:
            role_featurizers = {}
            args._unigram_stream_featurizers = role_featurizers
        role_featurizers[role] = featurizer
        role_vocab_sizes = getattr(
            args,
            "_unigram_stream_vocab_sizes",
            None,
        )
        if role_vocab_sizes is None:
            role_vocab_sizes = {}
            args._unigram_stream_vocab_sizes = role_vocab_sizes
        role_vocab_sizes[role] = int(featurizer.vocab_size)
    return tuple(
        _classic_role_token_tensor(
            stream_args,
            featurizer,
            split_rows,
            role_name,
            encoder_name,
        )
        if split_rows
        else None
        for split_rows in rows
    )


def run_heterogeneous_two_stream(
    args,
    task_root,
    train_csv,
    val_csv,
    test_csv,
    train_rows,
    val_rows,
    test_rows,
    train_pep,
    train_partner,
    val_pep,
    val_partner,
    test_pep,
    test_partner,
    train_y,
    val_y,
    test_y,
    device,
    semantic_profile=None,
    train_pair_ids=None,
):
    sequence_splits = {
        "peptide": (train_pep, val_pep, test_pep),
        "partner": (train_partner, val_partner, test_partner),
    }
    row_splits = (train_rows, val_rows, test_rows)
    if str(getattr(args, "interaction_level", "vector")) == "token":
        classifier = str(getattr(args, "classifier", "") or "mlp").strip().lower()
        if classifier != "mlp" and not exports_interaction_embeddings(args):
            raise SystemExit(
                "Heterogeneous token-level interaction currently requires classifier=mlp."
            )
        peptide_splits = _extract_stream_token_splits(
            args,
            "peptide",
            sequence_splits["peptide"],
            row_splits,
            task_root,
            train_rows,
            val_rows,
            train_csv,
            val_csv,
            test_csv,
            device,
        )
        partner_splits = _extract_stream_token_splits(
            args,
            "partner",
            sequence_splits["partner"],
            row_splits,
            task_root,
            train_rows,
            val_rows,
            train_csv,
            val_csv,
            test_csv,
            device,
        )
        if bool(getattr(args, "unigram_sampling", False)):
            role_featurizers = getattr(
                args,
                "_unigram_stream_featurizers",
                {},
            )
            configure_separate_unigram_sampling(
                args,
                train_rows,
                peptide_featurizer=role_featurizers.get("peptide"),
                partner_featurizer=role_featurizers.get("partner"),
            )
        trainer = (
            train_heterogeneous_cross_attention_then_classifier
            if exports_interaction_embeddings(args)
            else train_heterogeneous_cross_attention_model
        )
        return trainer(
            args,
            peptide_splits[0],
            partner_splits[0],
            train_y,
            peptide_splits[1],
            partner_splits[1],
            val_y,
            peptide_splits[2],
            partner_splits[2],
            test_y,
            device,
            semantic_profile=semantic_profile,
            train_pair_ids=train_pair_ids,
            profile_rows=train_rows,
        )

    peptide_splits = _extract_stream_vector_splits(
        args,
        "peptide",
        sequence_splits["peptide"],
        row_splits,
        task_root,
        train_rows,
        val_rows,
        train_csv,
        val_csv,
        test_csv,
        device,
    )
    partner_splits = _extract_stream_vector_splits(
        args,
        "partner",
        sequence_splits["partner"],
        row_splits,
        task_root,
        train_rows,
        val_rows,
        train_csv,
        val_csv,
        test_csv,
        device,
    )
    if str(getattr(args, "predictor_type", "classifier")) == "sample_graph":
        graph_model = str(
            getattr(args, "graph_predictor", "") or getattr(args, "model", "gcn")
        ).lower()
        if (
            graph_model != HGT_MODEL
            and resolve_fusion_mode(getattr(args, "fusion_mode", "concat_all")) != "concat"
        ):
            raise SystemExit("Heterogeneous sample-graph streams require fusion_mode=concat.")
        peptide_train, peptide_val, peptide_test, peptide_scaler = scale_dense_features(
            peptide_splits[0].float(),
            peptide_splits[1].float(),
            peptide_splits[2].float() if peptide_splits[2] is not None else None,
            mode=getattr(args, "dl_scaler", "standard"),
            return_scaler=True,
        )
        partner_train, partner_val, partner_test, partner_scaler = scale_dense_features(
            partner_splits[0].float(),
            partner_splits[1].float(),
            partner_splits[2].float() if partner_splits[2] is not None else None,
            mode=getattr(args, "dl_scaler", "standard"),
            return_scaler=True,
        )
        if graph_model == HGT_MODEL:
            configure_pair_graph_artifacts(
                args,
                peptide_splits[0],
                partner_splits[0],
                None,
                profile_rows=train_rows,
            )
            return train_hgt(
                args=args,
                train_peptide_x=peptide_train,
                train_partner_x=partner_train,
                train_y=train_y,
                val_peptide_x=peptide_val,
                val_partner_x=partner_val,
                val_y=val_y,
                test_peptide_x=peptide_test,
                test_partner_x=partner_test,
                test_y=test_y,
                device=device,
                peptide_scaler=peptide_scaler,
                protein_scaler=partner_scaler,
            )
        train_x = fuse_embeddings(peptide_train, partner_train, "concat").float()
        val_x = fuse_embeddings(peptide_val, partner_val, "concat").float()
        test_x = (
            fuse_embeddings(peptide_test, partner_test, "concat").float()
            if peptide_test is not None and partner_test is not None
            else None
        )
        configure_pair_graph_artifacts(
            args,
            peptide_splits[0],
            partner_splits[0],
            train_x,
            profile_rows=train_rows,
        )
        graph_kwargs = {
            "args": args,
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "test_x": test_x,
            "test_y": test_y,
            "device": device,
            "peptide_scaler": peptide_scaler,
            "protein_scaler": partner_scaler,
        }
        if graph_model == GCN_MODEL:
            return train_gcn(**graph_kwargs)
        if graph_model == GAT_MODEL:
            return train_gat(**graph_kwargs)
        if graph_model == SAGE_MODEL:
            return train_sage(**graph_kwargs)
        if graph_model == GT_MODEL:
            return train_graph_transformer(**graph_kwargs)
        raise SystemExit(f"Unsupported heterogeneous sample-graph predictor: {graph_model}")
    classifier = (
        str(getattr(args, "classifier", "") or getattr(args, "model", "mlp")).strip().lower()
    )
    if classifier not in ["mlp", *CLASSICAL_CLASSIFIERS]:
        raise SystemExit(f"Unsupported heterogeneous vector classifier: {classifier}")
    if args.fusion_mode == "aligned_gated":
        if classifier != "mlp":
            raise SystemExit("fusion_mode=aligned_gated requires classifier=mlp.")
        peptide_train, peptide_val, peptide_test, peptide_scaler = scale_dense_features(
            peptide_splits[0].float(),
            peptide_splits[1].float(),
            peptide_splits[2].float() if peptide_splits[2] is not None else None,
            mode=getattr(args, "dl_scaler", "standard"),
            return_scaler=True,
        )
        partner_train, partner_val, partner_test, partner_scaler = scale_dense_features(
            partner_splits[0].float(),
            partner_splits[1].float(),
            partner_splits[2].float() if partner_splits[2] is not None else None,
            mode=getattr(args, "dl_scaler", "standard"),
            return_scaler=True,
        )
        align_dim = int(getattr(args, "stream_alignment_dim", 256))
        artifact_shape = torch.empty((len(peptide_train), align_dim * 3), dtype=torch.float)
        configure_pair_vector_artifacts(
            args,
            peptide_splits[0],
            partner_splits[0],
            artifact_shape,
            profile_rows=train_rows,
        )
        return train_mlp_aligned_gated(
            args,
            peptide_train,
            partner_train,
            train_y,
            peptide_val,
            partner_val,
            val_y,
            peptide_test,
            partner_test,
            test_y,
            device,
            semantic_profile=semantic_profile,
            train_pair_ids=train_pair_ids,
            peptide_scaler=peptide_scaler,
            protein_scaler=partner_scaler,
        )
    train_x = fuse_embeddings(peptide_splits[0], partner_splits[0], args.fusion_mode).float()
    val_x = fuse_embeddings(peptide_splits[1], partner_splits[1], args.fusion_mode).float()
    test_x = (
        fuse_embeddings(peptide_splits[2], partner_splits[2], args.fusion_mode).float()
        if peptide_splits[2] is not None and partner_splits[2] is not None
        else None
    )
    configure_pair_vector_artifacts(
        args,
        peptide_splits[0],
        partner_splits[0],
        train_x,
        profile_rows=train_rows,
    )
    if classifier == "mlp":
        train_x, val_x, test_x, scaler = scale_dense_features(
            train_x,
            val_x,
            test_x,
            mode=getattr(args, "dl_scaler", "standard"),
            return_scaler=True,
        )
        return train_mlp(
            args,
            train_x,
            train_y,
            val_x,
            val_y,
            test_x,
            test_y,
            device,
            semantic_profile=semantic_profile,
            train_pair_ids=train_pair_ids,
            dl_scaler=scaler,
        )
    return train_ml(
        args,
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        classifier=classifier,
    )


def run_pair(args):
    normalize_pair_workflow_args(args)
    try:
        _validate_pair_classification_loss_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    resume_bundle = str(getattr(args, "resume_embedding_bundle", "") or "").strip()
    if resume_bundle:
        manifest, tensors = load_svm_resume_bundle(resume_bundle)
        if str(getattr(args, "classifier", "") or "").lower() != "svm":
            raise SystemExit("downstream SVM resume requires classifier=svm")
        args._artifact_source = str(manifest.get("artifact_source") or "pair_embedding_features")
        args._artifact_info = dict(manifest.get("artifact_info") or {})
        print("[resume] Stage 1/2 reused: loading verified exported embeddings")
        return train_ml(
            args,
            tensors["train_x"],
            tensors["train_y"],
            tensors["val_x"],
            tensors["val_y"],
            tensors["test_x"],
            tensors["test_y"],
            classifier="svm",
        )
    args.dropout = float(getattr(args, "dropout", PAIR_NEURAL_DEFAULTS["dropout"]))
    try:
        normalize_prediction_head_config(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        apply_pair_streams(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if str(getattr(args, "interaction_level", "")).strip().lower() == "token":
        try:
            validate_pair_attention_hyperparameters(
                int(getattr(args, "interaction_hidden_dim", 128)),
                int(getattr(args, "interaction_heads", 4)),
                float(args.dropout),
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    try:
        validate_index_route_semantics(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if bool(getattr(args, "unigram_sampling", False)):
        validate_unigram_sampling(
            getattr(
                args,
                "unigram_nbest_size",
                UNIGRAM_DEFAULT_NBEST_SIZE,
            ),
            getattr(args, "unigram_alpha", UNIGRAM_DEFAULT_ALPHA),
        )
        stream_mode = (
            str(getattr(args, "stream_mode", "homogeneous") or "homogeneous").strip().lower()
        )
        if stream_mode == "homogeneous":
            if str(getattr(args, "tokenizer", "")).lower() != "unigram":
                raise SystemExit("--unigram_sampling requires tokenizer=unigram")
            if str(getattr(args, "encoder", "")).lower() != "index":
                raise SystemExit("Unigram sampling requires encoder=index")
            supported_models = set(DL_SEQUENCE_MODELS).union(
                {CROSS_ATTN_MODEL, BI_CROSS_ATTN_MODEL}
            )
            if str(getattr(args, "model", "")).lower() not in supported_models:
                raise SystemExit(
                    "Pair Unigram sampling requires a sequence-preserving neural "
                    f"model in {sorted(supported_models)}"
                )
        else:
            unigram_roles = [
                role
                for role, spec in (getattr(args, "streams", {}) or {}).items()
                if str(spec.get("tokenizer", "")).lower() == "unigram"
                and str(spec.get("encoder", "")).lower() == "index"
            ]
            if not unigram_roles or str(getattr(args, "interaction_level", "")).lower() != "token":
                raise SystemExit(
                    "Heterogeneous Unigram sampling requires at least one "
                    "unigram/index stream and interaction_level=token"
                )
    for line in format_effective_route(resolve_effective_route(vars(args), args.task_type)):
        print(line)
    print(
        "classification_loss="
        f"{getattr(args, 'loss_type', PAIR_CLASSIFICATION_LOSS_DEFAULT)} "
        f"pos_weight_max={float(getattr(args, 'pos_weight_max', PAIR_CLASSIFICATION_POS_WEIGHT_MAX_DEFAULT)):.6g}"
    )
    for role, spec in (getattr(args, "streams", {}) or {}).items():
        stream_model = str(spec.get("stream_model", "") or "").strip().lower()
        if stream_model not in {"", "identity", "embedding"}:
            raise SystemExit(
                f"Unsupported {role} stream_model={stream_model}. This route currently supports "
                "identity/embedding stream outputs; choose the interaction classifier separately."
            )
    if not hasattr(args, "report_aupr"):
        args.report_aupr = True
    if str(getattr(args, "feature_mode", "single")).strip().lower() == "multimodal":
        raise SystemExit(
            "feature_mode=multimodal is not implemented in train_pair_pipeline.py; use feature_mode=single."
        )
    normalize_semantic_aliases(args)
    try:
        validate_linear_prediction_head_semantics(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if semantic_is_enabled(args) and not interaction_semantic_enabled(args):
        raise SystemExit(
            "The unified semantic layer requires a differentiable neural pair model "
            "with a sample-level embedding; classical classifiers and sample-graph models are unsupported."
        )
    normalize_route_fusion_mode(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_type = normalize_task_type(getattr(args, "task_type", ""))
    if normalize_task_type(task_type) == "pep_hlai":
        default_task_root = str(getattr(args, "task_root", "") or resolve_hlai_root(project_root))
    elif normalize_task_type(task_type) == "pep_hlaii":
        default_task_root = "datasets/task_peptide_HLAII_interaction"
    else:
        default_task_root = "datasets/task_peptide_protein_interaction"
    task_root = resolve_path(project_root, getattr(args, "task_root", "") or default_task_root)

    train_csv_arg = getattr(args, "train_csv", "")
    val_csv_arg = getattr(args, "val_csv", "")
    test_csv_arg = getattr(args, "test_csv", "")
    data_dir_arg = getattr(args, "data_dir", "")
    data_dir = resolve_path(project_root, data_dir_arg) if data_dir_arg else ""
    if data_dir:
        if not train_csv_arg:
            train_csv_arg = os.path.join(data_dir, "train", "pairs.csv")
        if not val_csv_arg:
            val_csv_arg = os.path.join(data_dir, "val", "pairs.csv")
        if not test_csv_arg:
            test_candidates = [
                os.path.join(data_dir, "test", "pairs.csv"),
                os.path.join(data_dir, "test_warm", "pairs.csv"),
            ]
            test_csv_arg = next(
                (p for p in test_candidates if os.path.exists(p)), test_candidates[0]
            )
    if train_csv_arg:
        train_csv = resolve_path(project_root, train_csv_arg)
    else:
        train_candidates = default_pair_csv_candidates(task_root, task_type, "train")
        train_csv = next((p for p in train_candidates if os.path.exists(p)), train_candidates[0])
    if val_csv_arg:
        val_csv = resolve_path(project_root, val_csv_arg)
    else:
        val_candidates = default_pair_csv_candidates(task_root, task_type, "val")
        val_csv = next((p for p in val_candidates if os.path.exists(p)), "")
    if test_csv_arg:
        test_csv = resolve_path(project_root, test_csv_arg)
    else:
        test_candidates = default_pair_csv_candidates(task_root, task_type, "test")
        test_csv = next((p for p in test_candidates if os.path.exists(p)), test_candidates[0])
    if not bool(getattr(args, "evaluate_test", True)):
        test_csv = ""
    args.feature_dir = (
        resolve_path(project_root, getattr(args, "feature_dir", ""))
        if getattr(args, "feature_dir", "")
        else default_pair_feature_dir(task_root, task_type, data_dir)
    )

    from pepplatform.stream_compat import stream_cache_root
    args.feature_dir = stream_cache_root(args.feature_dir)

    is_hlai_task = normalize_task_type(getattr(args, "task_type", "")) == "pep_hlai"
    hla_pseudo_map = {}
    if is_hlai_task:
        hla_pseudo_map = load_hlai_pseudoseq_map(task_root)
        if hla_pseudo_map:
            print(f"[hlai] loaded pseudoseq map: {len(hla_pseudo_map)} alleles")

    train_rows, train_contexts = read_ppi_csv(
        train_csv,
        hla_pseudo_map=hla_pseudo_map,
        return_contexts=True,
        require_hlai_pseudoseq=is_hlai_task,
        pseudoseq_path=os.path.join(task_root, "pseudoseqs_decoded.csv"),
    )
    if val_csv and os.path.exists(val_csv):
        val_rows, val_contexts = read_ppi_csv(
            val_csv,
            hla_pseudo_map=hla_pseudo_map,
            return_contexts=True,
            require_hlai_pseudoseq=is_hlai_task,
            pseudoseq_path=os.path.join(task_root, "pseudoseqs_decoded.csv"),
        )

    else:
        val_ratio = float(getattr(args, "val_ratio", 0.1))
        split_seed = int(getattr(args, "split_seed", 42))
        train_rows, val_rows, train_contexts, val_contexts = stratified_split(
            train_rows,
            val_ratio=val_ratio,
            seed=split_seed,
            contexts=train_contexts,
        )

    args._feature_train_sha256 = _feature_input_digest(
        (sid, peptide, protein, label) for sid, peptide, protein, label in train_rows
    )

    if str(getattr(args, "stream_mode", "homogeneous")) != "heterogeneous":
        ensure_subword_tokenizer_ready(args, task_root, train_rows, val_rows)

    if os.path.exists(test_csv):
        test_rows, test_contexts = read_ppi_csv(
            test_csv,
            hla_pseudo_map=hla_pseudo_map,
            return_contexts=True,
            require_hlai_pseudoseq=is_hlai_task,
            pseudoseq_path=os.path.join(task_root, "pseudoseqs_decoded.csv"),
        )
    else:
        test_rows = []
        test_contexts = []
    args.ppi_test_set = os.path.basename(test_csv) if test_rows else "none"
    dr_unknown_audit = {}
    for role, spec in (getattr(args, "streams", {}) or {}).items():
        params = spec.get("params", {}) or {}
        if (
            str(spec.get("tokenizer", "") or "").strip().lower() != "dr"
            or validate_dr_unknown_residue_policy(
                params.get(
                    "dr_unknown_residue_policy",
                    DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
                )
            )
            != "zero"
        ):
            continue
        sequence_index = 1 if role == "peptide" else 2
        for split_name, split_rows in (("train", train_rows), ("val", val_rows)):
            sequences = [str(row[sequence_index]).upper() for row in split_rows]
            key = f"{role}_{split_name}"
            dr_unknown_audit[key] = {
                "unknown_residue": "X",
                "unknown_residue_count": sum(value.count("X") for value in sequences),
                "affected_sample_count": sum("X" in value for value in sequences),
                "sample_count": len(sequences),
            }
    if dr_unknown_audit:
        args.dr_unknown_residue_audit_json = json.dumps(
            dr_unknown_audit,
            sort_keys=True,
            separators=(",", ":"),
        )
        print(
            "[dr] unknown-residue audit: " + args.dr_unknown_residue_audit_json,
            flush=True,
        )
    resolve_pair_top_ngram_lengths(
        args,
        train_rows + val_rows + test_rows,
    )

    train_pep, train_pro, train_y = rows_to_tensors(train_rows)
    val_pep, val_pro, val_y = rows_to_tensors(val_rows)
    test_pep, test_pro, test_y = rows_to_tensors(test_rows) if test_rows else ([], [], None)
    semantic_profile = None
    train_pair_ids = None
    if interaction_semantic_enabled(args):
        relation_source = InteractionProfileRelationSource(train_rows, train_contexts)
        pair_mode = str(getattr(args, "semantic_pair_mode", "role_fused")).strip().lower()
        loss_space = str(getattr(args, "semantic_loss_space", "projection")).strip().lower()
        partner_scheme = str(getattr(args, "semantic_partner_scheme", "fixed")).strip().lower()
        if loss_space not in {"projection", "shared"}:
            raise SystemExit("pair semantic loss_space must be projection or shared")
        if partner_scheme not in {
            "fixed",
            "fixed_placebo",
            "dense_hard",
            "placebo",
        }:
            raise SystemExit(
                "pair semantic partner_scheme must be fixed, fixed_placebo, dense_hard or placebo"
            )
        positive_partners = int(getattr(args, "semantic_positive_partners", 1))
        zero_partners = int(getattr(args, "semantic_zero_partners", 8))
        semantic_seed = int(getattr(args, "seed", 42))
        if partner_scheme == "dense_hard":
            peptide_sequences = [str(row[1]).strip().upper() for row in train_rows]
            protein_sequences = [str(row[2]).strip() for row in train_rows]
            peptide_embeddings = _load_stream_pooled_cache(
                args,
                "peptide",
                peptide_sequences,
            ).numpy()
            protein_embeddings = _load_stream_pooled_cache(
                args,
                "partner",
                protein_sequences,
            ).numpy()

            def _role_normalized(values: np.ndarray) -> np.ndarray:
                norms = np.linalg.norm(values, axis=1, keepdims=True) + 1e-8
                return (values / norms).astype(np.float32)

            fused_embeddings = np.concatenate(
                [
                    _role_normalized(peptide_embeddings),
                    _role_normalized(protein_embeddings),
                ],
                axis=1,
            )
            fused_table = build_interaction_dense_hard_partner_table(
                relation_source,
                seed=semantic_seed,
                embeddings=fused_embeddings,
                positive_cap=int(getattr(args, "semantic_positive_cap", 16)),
                negative_cap=int(getattr(args, "semantic_negative_cap", 8)),
                placebo=False,
            )
        else:
            fused_table = build_partner_table(
                relation_source,
                seed=semantic_seed,
                positive_partners_per_anchor=positive_partners,
                zero_partners_per_anchor=zero_partners,
            )
        peptide_table = None
        context_table = None
        if pair_mode in {"role", "role_fused"}:
            peptide_table = build_partner_table(
                relation_source.peptide_entity_source(),
                seed=semantic_seed,
                positive_partners_per_anchor=positive_partners,
                zero_partners_per_anchor=zero_partners,
            )
            context_table = build_partner_table(
                relation_source.context_entity_source(),
                seed=semantic_seed,
                positive_partners_per_anchor=positive_partners,
                zero_partners_per_anchor=zero_partners,
            )
        placebo_reference_tables = {}
        if partner_scheme == "fixed_placebo":
            placebo_reference_tables = {
                "fused": fused_table,
                "peptide": peptide_table,
                "context": context_table,
            }
            fused_table = placebo_shuffle_table(
                fused_table,
                seed=semantic_seed,
                tag="interaction-fused-fixed-placebo",
            )
            if peptide_table is not None:
                peptide_table = placebo_shuffle_table(
                    peptide_table,
                    seed=semantic_seed,
                    tag="interaction-peptide-fixed-placebo",
                )
            if context_table is not None:
                context_table = placebo_shuffle_table(
                    context_table,
                    seed=semantic_seed,
                    tag="interaction-context-fixed-placebo",
                )
        elif partner_scheme == "placebo":
            fused_table = placebo_shuffle_table(
                fused_table,
                seed=semantic_seed,
                tag="interaction-fixed-placebo",
            )
            if peptide_table is not None:
                peptide_table = placebo_shuffle_table(
                    peptide_table,
                    seed=semantic_seed,
                    tag="interaction-peptide-placebo",
                )
            if context_table is not None:
                context_table = placebo_shuffle_table(
                    context_table,
                    seed=semantic_seed,
                    tag="interaction-context-placebo",
                )
        peptide_example_rows = tuple(
            relation_source.rows_by_peptide[index][0]
            for index in range(len(relation_source.peptide_values))
        )
        context_example_rows = tuple(
            relation_source.rows_by_context[index][0]
            for index in range(len(relation_source.context_values))
        )
        semantic_profile = SemanticTrainingProfile(
            relation_type=relation_source.relation_type,
            seed=semantic_seed,
            fused_table=fused_table,
            peptide_table=peptide_table,
            context_table=context_table,
            row_peptide_ids=relation_source.row_peptide_ids,
            row_context_ids=relation_source.row_context_ids,
            peptide_example_rows=peptide_example_rows,
            context_example_rows=context_example_rows,
            source_summary=relation_source.summary(),
            config={
                "semantic_pair_mode": pair_mode,
                "semantic_projection_dim": int(getattr(args, "semantic_projection_dim", 64)),
                "semantic_contrastive_temperature": float(
                    getattr(args, "semantic_contrastive_temperature", 0.1)
                ),
                "semantic_positive_partners": positive_partners,
                "semantic_zero_partners": zero_partners,
                "semantic_similarity_weight": float(
                    getattr(args, "semantic_similarity_weight", 0.1)
                ),
                "semantic_zero_weight": float(getattr(args, "semantic_zero_weight", 0.25)),
                "semantic_loss_space": loss_space,
                "semantic_residual_weight": float(getattr(args, "semantic_residual_weight", 0.0)),
                "semantic_learned_gate": bool(getattr(args, "semantic_learned_gate", False)),
                "semantic_partner_scheme": partner_scheme,
                **(
                    {"semantic_loss_type": "jaccard_supcon"}
                    if getattr(args, "semantic_loss_type", "legacy") == "jaccard_supcon"
                    else {}
                ),
                "semantic_positive_cap": int(getattr(args, "semantic_positive_cap", 16)),
                "semantic_negative_cap": int(getattr(args, "semantic_negative_cap", 8)),
                "semantic_hard_negative_source": str(
                    getattr(args, "semantic_hard_negative_source", "input_embedding")
                ),
            },
        )
        train_pair_ids = torch.arange(len(train_rows), dtype=torch.long)
        artifact_dir = str(getattr(args, "artifact_dir", "") or "").strip()
        if artifact_dir:
            args._semantic_provenance = save_semantic_training_profile(
                artifact_dir,
                profile=semantic_profile,
                source=relation_source,
                split_sample_ids={
                    "train": [str(row[0]) for row in train_rows],
                    "val": [str(row[0]) for row in val_rows],
                    "test": [str(row[0]) for row in test_rows],
                },
                placebo_reference_tables=placebo_reference_tables,
            )

    tok_name = str(getattr(args, "tokenizer", "esm2")).strip().lower()
    model_name = str(getattr(args, "model", "mlp")).strip().lower()
    if getattr(args, "pair_strategy", "") == "two_stream":
        peptide_spec = args.streams["peptide"]
        partner_spec = args.streams["partner"]
        print(
            "[streams] "
            f"mode={args.stream_mode} | "
            f"peptide={peptide_spec['tokenizer']}+{peptide_spec['encoder']} "
            f"(max_length={peptide_spec['max_length']}) | "
            f"partner={partner_spec['tokenizer']}+{partner_spec['encoder']} "
            f"(max_length={partner_spec['max_length']})"
        )
    if getattr(args, "pair_strategy", "") == "single_stream":
        print("[route] single_stream(role-boundary-sequence-join-v1), fusion_mode=not_used")
    elif getattr(args, "interaction_level", "") == "token":
        print(
            f"[route] {public_pair_strategy(args.pair_strategy)}(token-level interaction:{getattr(args, 'interaction_model', 'cross_attn')}), fusion_mode=not_used"
        )
    elif getattr(args, "predictor_type", "") == "sample_graph":
        if model_name == HGT_MODEL:
            print(
                "[route] two_stream(vector-level interaction), "
                "sample_graph_predictor=hgt, input_adapter=typed_role_projection"
            )
        else:
            print(
                f"[route] {public_pair_strategy(args.pair_strategy)}(vector-level interaction), sample_graph_predictor={getattr(args, 'graph_predictor', model_name)}, fusion_mode={resolve_fusion_mode(getattr(args, 'fusion_mode', 'concat_all'))}"
            )
    else:
        print(
            f"[route] {public_pair_strategy(args.pair_strategy)}(vector-level interaction), predictor=classifier, fusion_mode={resolve_fusion_mode(getattr(args, 'fusion_mode', 'concat_all'))}"
        )

    if (
        getattr(args, "pair_strategy", "") == "two_stream"
        and str(getattr(args, "stream_mode", "homogeneous")) == "heterogeneous"
    ):
        return run_heterogeneous_two_stream(
            args,
            task_root,
            train_csv,
            val_csv,
            test_csv,
            train_rows,
            val_rows,
            test_rows,
            train_pep,
            train_pro,
            val_pep,
            val_pro,
            test_pep,
            test_pro,
            train_y,
            val_y,
            test_y,
            device,
            semantic_profile=semantic_profile,
            train_pair_ids=train_pair_ids,
        )

    profile_ready = ensure_pair_profile_cache_ready(args, task_root, train_csv, val_csv, test_csv)
    if not profile_ready:
        raise SystemExit(
            "Profile cache build failed. This pipeline already uses per-sequence fallback (--on_missing blosum62) during PSSM/PSFM construction, "
            "but if cache generation infrastructure fails (e.g., BLAST+ missing), training cannot continue with profile encoders."
        )
    if tok_name in CLASSIC_TOKENIZERS:
        enc_name = str(getattr(args, "encoder", "index")).strip()
        enc_name_lower = enc_name.lower()
        allowed = CLASSIC_ENCODER_OPTIONS.get(tok_name, [])
        if enc_name != "ppi_fusion" and enc_name not in allowed:
            raise SystemExit(
                f"Invalid encoder for tokenizer={tok_name}: {enc_name}. Allowed: {', '.join(allowed)}"
            )
        if (
            str(getattr(args, "model", "mlp")).lower()
            in (DL_SEQUENCE_MODELS + [CROSS_ATTN_MODEL, BI_CROSS_ATTN_MODEL])
            and enc_name_lower not in DL_SEQUENCE_ENCODERS
        ):
            raise SystemExit(
                f"PPI DL sequence models require encoder in {DL_SEQUENCE_ENCODERS}, got {enc_name_lower}"
            )

    if tok_name in PRETRAINED_TOKENIZERS:
        tokenizer, encoder, _ = build_pretrained_tokenizer_encoder(args, device)
        if (
            model_name in DL_SEQUENCE_MODELS
            and str(getattr(args, "pair_strategy", "") or "").strip().lower() == "single_stream"
        ):
            train_pep_tok = extract_pretrained_token_sequences(
                args, tokenizer, encoder, train_pep, "train", "peptide", device
            )
            train_pro_tok = extract_pretrained_token_sequences(
                args, tokenizer, encoder, train_pro, "train", "protein", device
            )
            train_x = join_pretrained_single_stream(args, train_pep_tok, train_pro_tok, "train")
            del train_pep_tok, train_pro_tok
            gc.collect()

            val_pep_tok = extract_pretrained_token_sequences(
                args, tokenizer, encoder, val_pep, "val", "peptide", device
            )
            val_pro_tok = extract_pretrained_token_sequences(
                args, tokenizer, encoder, val_pro, "val", "protein", device
            )
            val_x = join_pretrained_single_stream(args, val_pep_tok, val_pro_tok, "val")
            del val_pep_tok, val_pro_tok
            gc.collect()

            test_x = None
            if test_rows:
                test_pep_tok = extract_pretrained_token_sequences(
                    args, tokenizer, encoder, test_pep, "test", "peptide", device
                )
                test_pro_tok = extract_pretrained_token_sequences(
                    args, tokenizer, encoder, test_pro, "test", "protein", device
                )
                test_x = join_pretrained_single_stream(args, test_pep_tok, test_pro_tok, "test")
                del test_pep_tok, test_pro_tok
            del encoder, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if exports_sequence_embeddings(args):
                return train_sequence_model_then_ml(
                    args,
                    None,
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    test_x,
                    test_y,
                    device,
                    semantic_profile=semantic_profile,
                    train_pair_ids=train_pair_ids,
                    profile_rows=train_rows,
                )
            return train_sequence_model(
                args,
                None,
                train_x,
                train_y,
                val_x,
                val_y,
                test_x,
                test_y,
                device,
                semantic_profile=semantic_profile,
                train_pair_ids=train_pair_ids,
                profile_rows=train_rows,
            )
        if str(getattr(args, "model", "mlp")).lower() in [CROSS_ATTN_MODEL, BI_CROSS_ATTN_MODEL]:
            train_pep_tok = extract_pretrained_token_sequences(
                args, tokenizer, encoder, train_pep, "train", "peptide", device
            )
            train_pro_tok = extract_pretrained_token_sequences(
                args, tokenizer, encoder, train_pro, "train", "protein", device
            )
            val_pep_tok = extract_pretrained_token_sequences(
                args, tokenizer, encoder, val_pep, "val", "peptide", device
            )
            val_pro_tok = extract_pretrained_token_sequences(
                args, tokenizer, encoder, val_pro, "val", "protein", device
            )
            test_pep_tok = (
                extract_pretrained_token_sequences(
                    args, tokenizer, encoder, test_pep, "test", "peptide", device
                )
                if test_rows
                else None
            )
            test_pro_tok = (
                extract_pretrained_token_sequences(
                    args, tokenizer, encoder, test_pro, "test", "protein", device
                )
                if test_rows
                else None
            )
            if exports_interaction_embeddings(args):
                return train_cross_attention_then_classifier(
                    args,
                    None,
                    train_pep_tok,
                    train_y,
                    val_pep_tok,
                    val_y,
                    test_pep_tok,
                    test_y,
                    device,
                    semantic_profile=semantic_profile,
                    train_pair_ids=train_pair_ids,
                    profile_rows=train_rows,
                    train_partner_x=train_pro_tok,
                    val_partner_x=val_pro_tok,
                    test_partner_x=test_pro_tok,
                )
            return train_cross_attention_model(
                args,
                None,
                train_pep_tok,
                train_y,
                val_pep_tok,
                val_y,
                test_pep_tok,
                test_y,
                device,
                semantic_profile=semantic_profile,
                train_pair_ids=train_pair_ids,
                profile_rows=train_rows,
                train_partner_x=train_pro_tok,
                val_partner_x=val_pro_tok,
                test_partner_x=test_pro_tok,
            )
        train_pep_emb = extract_embeddings(
            args, tokenizer, encoder, train_pep, "train", "peptide", device
        )
        train_pro_emb = extract_embeddings(
            args, tokenizer, encoder, train_pro, "train", "protein", device
        )
        val_pep_emb = extract_embeddings(
            args, tokenizer, encoder, val_pep, "val", "peptide", device
        )
        val_pro_emb = extract_embeddings(
            args, tokenizer, encoder, val_pro, "val", "protein", device
        )
        test_pep_emb = (
            extract_embeddings(args, tokenizer, encoder, test_pep, "test", "peptide", device)
            if test_rows
            else None
        )
        test_pro_emb = (
            extract_embeddings(args, tokenizer, encoder, test_pro, "test", "protein", device)
            if test_rows
            else None
        )
    else:
        if tok_name not in CLASSIC_TOKENIZERS:
            raise SystemExit(f"Unsupported tokenizer for PPI: {tok_name}")
        ensure_ppi_static_embedding_ready(args, task_root, train_rows, val_rows)
        feature_encoder = str(getattr(args, "encoder", "index")).lower()
        if feature_encoder == "ppi_fusion":
            feature_encoder = str(getattr(args, "aux_encoder", "index")).lower()
        featurizer = build_classic_featurizer(args, task_root, train_rows)
        featurizer_peptide = featurizer
        featurizer_protein = featurizer
        if bool(getattr(args, "unigram_sampling", False)):
            configure_joined_unigram_sampling(
                args,
                train_rows,
                featurizer_peptide,
                featurizer_protein,
            )
        if getattr(args, "model", "mlp") in [CROSS_ATTN_MODEL, BI_CROSS_ATTN_MODEL]:
            print("Building sequence features for cross-attention...", flush=True)
            train_x = encode_pair_sequences_for_dl(
                featurizer,
                train_rows,
                featurizer_peptide=featurizer_peptide,
                featurizer_protein=featurizer_protein,
                split_name="train",
                args=args,
            )
            val_x = encode_pair_sequences_for_dl(
                featurizer,
                val_rows,
                featurizer_peptide=featurizer_peptide,
                featurizer_protein=featurizer_protein,
                split_name="val",
                args=args,
            )
            test_x = (
                encode_pair_sequences_for_dl(
                    featurizer,
                    test_rows,
                    featurizer_peptide=featurizer_peptide,
                    featurizer_protein=featurizer_protein,
                    split_name="test",
                    args=args,
                )
                if test_rows
                else None
            )
            if exports_interaction_embeddings(args):
                return train_cross_attention_then_classifier(
                    args,
                    featurizer,
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    test_x,
                    test_y,
                    device,
                    semantic_profile=semantic_profile,
                    train_pair_ids=train_pair_ids,
                    profile_rows=train_rows,
                )
            return train_cross_attention_model(
                args,
                featurizer,
                train_x,
                train_y,
                val_x,
                val_y,
                test_x,
                test_y,
                device,
                semantic_profile=semantic_profile,
                train_pair_ids=train_pair_ids,
                profile_rows=train_rows,
            )
        if getattr(args, "model", "mlp") in DL_SEQUENCE_MODELS:
            print("Building sequence features for Single-stream model...", flush=True)
            train_x = encode_pair_sequences_for_dl(
                featurizer,
                train_rows,
                featurizer_peptide=featurizer_peptide,
                featurizer_protein=featurizer_protein,
                split_name="train",
                args=args,
            )
            val_x = encode_pair_sequences_for_dl(
                featurizer,
                val_rows,
                featurizer_peptide=featurizer_peptide,
                featurizer_protein=featurizer_protein,
                split_name="val",
                args=args,
            )
            test_x = (
                encode_pair_sequences_for_dl(
                    featurizer,
                    test_rows,
                    featurizer_peptide=featurizer_peptide,
                    featurizer_protein=featurizer_protein,
                    split_name="test",
                    args=args,
                )
                if test_rows
                else None
            )
            if exports_sequence_embeddings(args):
                return train_sequence_model_then_ml(
                    args,
                    featurizer,
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    test_x,
                    test_y,
                    device,
                    semantic_profile=semantic_profile,
                    train_pair_ids=train_pair_ids,
                    profile_rows=train_rows,
                )
            return train_sequence_model(
                args,
                featurizer,
                train_x,
                train_y,
                val_x,
                val_y,
                test_x,
                test_y,
                device,
                semantic_profile=semantic_profile,
                train_pair_ids=train_pair_ids,
                profile_rows=train_rows,
            )
        train_pep_emb = extract_classic_features(
            args, featurizer_peptide, train_rows, "train", "peptide", encoder_name=feature_encoder
        )
        train_pro_emb = extract_classic_features(
            args, featurizer_protein, train_rows, "train", "protein", encoder_name=feature_encoder
        )
        val_pep_emb = extract_classic_features(
            args, featurizer_peptide, val_rows, "val", "peptide", encoder_name=feature_encoder
        )
        val_pro_emb = extract_classic_features(
            args, featurizer_protein, val_rows, "val", "protein", encoder_name=feature_encoder
        )
        test_pep_emb = (
            extract_classic_features(
                args, featurizer_peptide, test_rows, "test", "peptide", encoder_name=feature_encoder
            )
            if test_rows
            else None
        )
        test_pro_emb = (
            extract_classic_features(
                args, featurizer_protein, test_rows, "test", "protein", encoder_name=feature_encoder
            )
            if test_rows
            else None
        )

    normalize_route_fusion_mode(args)

    if getattr(args, "model", "mlp") in GRAPH_MODELS:
        if args.fusion_mode == "aligned_gated":
            train_pep_emb, val_pep_emb, test_pep_emb, peptide_scaler = scale_dense_features(
                train_pep_emb.float(),
                val_pep_emb.float(),
                test_pep_emb.float() if test_pep_emb is not None else None,
                mode=getattr(args, "dl_scaler", "standard"),
                return_scaler=True,
            )
            train_pro_emb, val_pro_emb, test_pro_emb, protein_scaler = scale_dense_features(
                train_pro_emb.float(),
                val_pro_emb.float(),
                test_pro_emb.float() if test_pro_emb is not None else None,
                mode=getattr(args, "dl_scaler", "standard"),
                return_scaler=True,
            )
            graph_shape_features = torch.empty(
                (train_pep_emb.shape[0], train_pep_emb.shape[1] * 3), dtype=torch.float
            )
            configure_pair_graph_artifacts(
                args,
                train_pep_emb,
                train_pro_emb,
                graph_shape_features,
                profile_rows=train_rows,
            )
            if getattr(args, "model", "mlp") == GCN_MODEL:
                return train_gcn_aligned_gated(
                    args=args,
                    train_pep_x=train_pep_emb,
                    train_pro_x=train_pro_emb,
                    train_y=train_y,
                    val_pep_x=val_pep_emb,
                    val_pro_x=val_pro_emb,
                    val_y=val_y,
                    test_pep_x=test_pep_emb,
                    test_pro_x=test_pro_emb,
                    test_y=test_y,
                    device=device,
                    peptide_scaler=peptide_scaler,
                    protein_scaler=protein_scaler,
                )
            if getattr(args, "model", "mlp") == GAT_MODEL:
                return train_gat_aligned_gated(
                    args=args,
                    train_pep_x=train_pep_emb,
                    train_pro_x=train_pro_emb,
                    train_y=train_y,
                    val_pep_x=val_pep_emb,
                    val_pro_x=val_pro_emb,
                    val_y=val_y,
                    test_pep_x=test_pep_emb,
                    test_pro_x=test_pro_emb,
                    test_y=test_y,
                    device=device,
                    peptide_scaler=peptide_scaler,
                    protein_scaler=protein_scaler,
                )
            if getattr(args, "model", "mlp") == HGT_MODEL:
                return train_hgt_aligned_gated(
                    args=args,
                    train_pep_x=train_pep_emb,
                    train_pro_x=train_pro_emb,
                    train_y=train_y,
                    val_pep_x=val_pep_emb,
                    val_pro_x=val_pro_emb,
                    val_y=val_y,
                    test_pep_x=test_pep_emb,
                    test_pro_x=test_pro_emb,
                    test_y=test_y,
                    device=device,
                    peptide_scaler=peptide_scaler,
                    protein_scaler=protein_scaler,
                )
            raise SystemExit("fusion_mode=aligned_gated currently requires model in {mlp,gcn,gat}")
        train_pep_emb, val_pep_emb, test_pep_emb, peptide_scaler = scale_dense_features(
            train_pep_emb.float(),
            val_pep_emb.float(),
            test_pep_emb.float() if test_pep_emb is not None else None,
            mode=getattr(args, "dl_scaler", "standard"),
            return_scaler=True,
        )
        train_pro_emb, val_pro_emb, test_pro_emb, protein_scaler = scale_dense_features(
            train_pro_emb.float(),
            val_pro_emb.float(),
            test_pro_emb.float() if test_pro_emb is not None else None,
            mode=getattr(args, "dl_scaler", "standard"),
            return_scaler=True,
        )
        if getattr(args, "model", "mlp") == HGT_MODEL:
            configure_pair_graph_artifacts(
                args,
                train_pep_emb,
                train_pro_emb,
                None,
                profile_rows=train_rows,
            )
            return train_hgt(
                args=args,
                train_peptide_x=train_pep_emb,
                train_partner_x=train_pro_emb,
                train_y=train_y,
                val_peptide_x=val_pep_emb,
                val_partner_x=val_pro_emb,
                val_y=val_y,
                test_peptide_x=test_pep_emb,
                test_partner_x=test_pro_emb,
                test_y=test_y,
                device=device,
                peptide_scaler=peptide_scaler,
                protein_scaler=protein_scaler,
            )
        train_x = fuse_embeddings(train_pep_emb, train_pro_emb, mode=args.fusion_mode).float()
        val_x = fuse_embeddings(val_pep_emb, val_pro_emb, mode=args.fusion_mode).float()
        test_x = (
            fuse_embeddings(test_pep_emb, test_pro_emb, mode=args.fusion_mode).float()
            if test_rows
            else None
        )
        configure_pair_graph_artifacts(
            args,
            train_pep_emb,
            train_pro_emb,
            train_x,
            profile_rows=train_rows,
        )
        if getattr(args, "model", "mlp") == GCN_MODEL:
            return train_gcn(
                args=args,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                test_x=test_x,
                test_y=test_y,
                device=device,
                peptide_scaler=peptide_scaler,
                protein_scaler=protein_scaler,
            )
        if getattr(args, "model", "mlp") == GAT_MODEL:
            return train_gat(
                args=args,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                test_x=test_x,
                test_y=test_y,
                device=device,
                peptide_scaler=peptide_scaler,
                protein_scaler=protein_scaler,
            )
        if getattr(args, "model", "mlp") == SAGE_MODEL:
            return train_sage(
                args=args,
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                test_x=test_x,
                test_y=test_y,
                device=device,
                peptide_scaler=peptide_scaler,
                protein_scaler=protein_scaler,
            )
        return train_graph_transformer(
            args=args,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
            device=device,
            peptide_scaler=peptide_scaler,
            protein_scaler=protein_scaler,
        )

    if args.fusion_mode == "aligned_gated":
        if getattr(args, "model", "mlp") != "mlp":
            raise SystemExit("fusion_mode=aligned_gated currently requires model in {mlp,gcn,gat}")
        train_pep_emb, val_pep_emb, test_pep_emb, peptide_scaler = scale_dense_features(
            train_pep_emb.float(),
            val_pep_emb.float(),
            test_pep_emb.float() if test_pep_emb is not None else None,
            mode=getattr(args, "dl_scaler", "standard"),
            return_scaler=True,
        )
        train_pro_emb, val_pro_emb, test_pro_emb, protein_scaler = scale_dense_features(
            train_pro_emb.float(),
            val_pro_emb.float(),
            test_pro_emb.float() if test_pro_emb is not None else None,
            mode=getattr(args, "dl_scaler", "standard"),
            return_scaler=True,
        )
        gated_shape_features = torch.empty(
            (train_pep_emb.shape[0], train_pep_emb.shape[1] * 3), dtype=torch.float
        )
        configure_pair_vector_artifacts(
            args,
            train_pep_emb,
            train_pro_emb,
            gated_shape_features,
            profile_rows=train_rows,
        )
        return train_mlp_aligned_gated(
            args=args,
            train_pep_x=train_pep_emb,
            train_pro_x=train_pro_emb,
            train_y=train_y,
            val_pep_x=val_pep_emb,
            val_pro_x=val_pro_emb,
            val_y=val_y,
            test_pep_x=test_pep_emb,
            test_pro_x=test_pro_emb,
            test_y=test_y,
            device=device,
            semantic_profile=semantic_profile,
            train_pair_ids=train_pair_ids,
            peptide_scaler=peptide_scaler,
            protein_scaler=protein_scaler,
        )

    train_x = fuse_embeddings(train_pep_emb, train_pro_emb, mode=args.fusion_mode).float()
    val_x = fuse_embeddings(val_pep_emb, val_pro_emb, mode=args.fusion_mode).float()
    test_x = (
        fuse_embeddings(test_pep_emb, test_pro_emb, mode=args.fusion_mode).float()
        if test_rows
        else None
    )
    configure_pair_vector_artifacts(
        args,
        train_pep_emb,
        train_pro_emb,
        train_x,
        profile_rows=train_rows,
    )

    if getattr(args, "model", "mlp") == "mlp":
        train_x, val_x, test_x, dl_scaler = scale_dense_features(
            train_x,
            val_x,
            test_x,
            mode=getattr(args, "dl_scaler", "standard"),
            return_scaler=True,
        )
        return train_mlp(
            args=args,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            test_x=test_x,
            test_y=test_y,
            device=device,
            semantic_profile=semantic_profile,
            train_pair_ids=train_pair_ids,
            dl_scaler=dl_scaler,
        )
    return train_ml(
        args=args,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )


def run_ppi(args):
    return run_pair(args)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Pair-interaction peptide training pipeline")
    parser.add_argument("--task_type", type=str, default="pep_protein_interaction")
    parser.add_argument("--task_root", type=str, default="")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Compatibility split root containing train/val/test pairs.csv files",
    )
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--val_csv", type=str, default="")
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument(
        "--evaluate_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="evaluate the supplied test split during training; leave disabled for final independent-test evaluation",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--tokenizer",
        choices=[
            "esm2",
            "protbert",
            "probert",
            "prott5",
            "esmc",
            "esm3",
            "kmer",
            "dr",
            "bpe",
            "wordpiece",
            "unigram",
            "top-n-gram",
            "dt",
        ],
        default="esm2",
    )
    parser.add_argument("--feature_mode", choices=["single", "multimodal"], default="single")
    parser.add_argument(
        "--primary_feature_mode", choices=["pretrained", "classic"], default="pretrained"
    )
    parser.add_argument("--aux_feature_mode", choices=["pretrained", "classic"], default="classic")
    parser.add_argument(
        "--aux_tokenizer",
        choices=[
            "esm2",
            "protbert",
            "probert",
            "prott5",
            "esmc",
            "esm3",
            "kmer",
            "dr",
            "bpe",
            "wordpiece",
            "unigram",
            "top-n-gram",
            "dt",
        ],
        default="kmer",
    )
    parser.add_argument(
        "--aux_encoder",
        type=encoder_argument,
        choices=PAIR_TRAINING_ENCODER_OPTIONS,
        default="index",
    )
    parser.add_argument("--mm_align_dim", type=int, default=256)
    parser.add_argument(
        "--model",
        choices=[
            "cnn",
            "rnn",
            "gru",
            "lstm",
            "transformer",
            "cross_attn",
            "cross_attn+mlp",
            "bi_cross_attn",
            "bi_cross_attn+mlp",
            "mlp",
            "svm",
            "calibrated_svm",
            "rf",
            "extratrees",
            "histgb",
            "xgb",
            "lgb",
            "logreg",
            "gcn",
            "gat",
            "hgt",
            "sage",
            "graph_transformer",
        ],
        default="mlp",
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
    from pepplatform.pair_terminology import normalize_pair_strategy

    parser.add_argument(
        "--pair_strategy",
        type=normalize_pair_strategy,
        choices=["single_stream", "two_stream"],
        default=None,
        metavar="{single_stream,two_stream}",
        help="Downstream pair input organization: single_stream (role-boundary join) or "
        "two_stream (separate roles until fusion/interaction). "
        "Historical strategy names remain accepted by compatibility adapters.",
    )
    parser.add_argument("--interaction_level", choices=["vector", "token"], default=None)
    parser.add_argument(
        "--interaction_model", choices=["cross_attn", "bi_cross_attn"], default=None
    )
    parser.add_argument("--predictor_type", choices=["classifier", "sample_graph"], default=None)
    parser.add_argument(
        "--graph_predictor",
        choices=["gcn", "gat", "hgt", "sage", "graph_transformer"],
        default=None,
    )
    parser.add_argument("--ppi_interaction_mode", choices=["single", "dual"], default=None)
    parser.add_argument("--hlaii_interaction_mode", choices=["concat", "contact"], default=None)
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default=None,
        help=(
            "Two-stream vector fusion (effective default: concat_all). HGT uses "
            "typed role projection and ignores only legacy concat/concat_all markers."
        ),
    )
    parser.add_argument("--stream_alignment_dim", type=parse_stream_alignment_dim, default=256)
    parser.add_argument("--interaction_hidden_dim", type=int, default=128)
    parser.add_argument("--interaction_heads", type=int, default=4)
    parser.add_argument("--gnn_k", type=int, default=8)
    parser.add_argument("--gnn_hidden_dim", type=int, default=256)
    parser.add_argument("--gnn_exact_knn_max_nodes", type=int, default=25000)
    parser.add_argument("--gnn_rp_projections", type=int, default=6)
    parser.add_argument("--gnn_rp_window", type=int, default=0)
    parser.add_argument(
        "--gnn_edge_chunk_size",
        type=int,
        default=GAT_EDGE_CHUNK_SIZE,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=PAIR_NEURAL_DEFAULTS["max_length"],
        help="maximum encoded positions; for tokenizer=kmer this is the K-mer token count",
    )
    parser.add_argument(
        "--residue_adapter_dim",
        type=int,
        default=RESIDUE_ADAPTER_DIM_DEFAULT,
        help="width of the audited LayerNorm+Linear adapter for pretrained residue routes",
    )
    parser.add_argument("--batch_size", type=int, default=PAIR_NEURAL_DEFAULTS["batch_size"])
    parser.add_argument(
        "--peptide_feature_extraction_batch_size",
        type=_parse_feature_extraction_batch_size,
        default=None,
        help="training/validation peptide residue extraction batch; defaults to batch_size",
    )
    parser.add_argument(
        "--partner_feature_extraction_batch_size",
        type=_parse_feature_extraction_batch_size,
        default=None,
        help="training/validation partner residue extraction batch; defaults to batch_size",
    )
    parser.add_argument("--lr", type=float, default=PAIR_NEURAL_DEFAULTS["lr"])
    parser.add_argument("--epochs", type=int, default=PAIR_NEURAL_DEFAULTS["epochs"])
    parser.add_argument("--dropout", type=float, default=PAIR_NEURAL_DEFAULTS["dropout"])
    parser.add_argument("--weight_decay", type=float, default=PAIR_NEURAL_DEFAULTS["weight_decay"])
    parser.add_argument(
        "--grad_clip_norm", type=float, default=PAIR_NEURAL_DEFAULTS["grad_clip_norm"]
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=PAIR_NEURAL_DEFAULTS["early_stop_patience"]
    )
    parser.add_argument(
        "--early_stop_min_delta", type=float, default=PAIR_NEURAL_DEFAULTS["early_stop_min_delta"]
    )
    parser.add_argument(
        "--loss_type",
        choices=["bce", "wbce"],
        default=PAIR_CLASSIFICATION_LOSS_DEFAULT,
        help="Pair neural classification loss; wbce preserves the historical default.",
    )
    parser.add_argument(
        "--pos_weight_max",
        type=_parse_pair_pos_weight_max,
        default=PAIR_CLASSIFICATION_POS_WEIGHT_MAX_DEFAULT,
        help="Upper clamp for WBCE positive-class weighting (must be finite and >= 1).",
    )
    parser.add_argument(
        "--enable_semantic_similarity", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--semantic_similarity_weight", type=float, default=None)
    parser.add_argument("--semantic_zero_weight", type=float, default=0.25)
    parser.add_argument(
        "--semantic_pair_mode",
        type=str,
        default="role_fused",
        choices=["fused", "role", "role_fused"],
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
        choices=["projection", "shared"],
        default="projection",
    )
    parser.add_argument("--semantic_residual_weight", type=float, default=0.0)
    parser.add_argument(
        "--semantic_learned_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use a trainable gate (init 0) for the pair semantic residual",
    )
    parser.add_argument(
        "--semantic_partner_scheme",
        choices=["fixed", "fixed_placebo", "dense_hard", "placebo"],
        default="fixed",
    )
    parser.add_argument("--semantic_positive_cap", type=int, default=16)
    parser.add_argument("--semantic_negative_cap", type=int, default=8)
    parser.add_argument(
        "--semantic_hard_negative_source",
        choices=["input_embedding"],
        default="input_embedding",
    )
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
    parser.add_argument(
        "--enable_interaction_semantic_similarity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--interaction_semantic_mode", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--interaction_semantic_similarity_weight", type=float, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--hla_context_semantic_margin", type=float, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--hlai_semantic_smoothing_tau", type=float, default=None, help=argparse.SUPPRESS
    )

    parser.add_argument("--esm2_model", type=str, default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--protbert_model", type=str, default="Rostlab/prot_bert")
    parser.add_argument("--probert_model", type=str, default="")
    parser.add_argument("--prott5_model", type=str, default="Rostlab/prot_t5_xl_uniref50")
    parser.add_argument("--esmc_model", type=str, default=PRETRAINED_MODEL_DEFAULTS["esmc"])
    parser.add_argument("--esm3_model", type=str, default="EvolutionaryScale/esm3-sm-open-v1")
    parser.add_argument(
        "--esm3_sdpa_backend",
        choices=("auto", "math"),
        default=ESM3_DEFAULT_SDPA_BACKEND,
    )
    add_stream_cli_arguments(parser)

    parser.add_argument("--feature_dir", type=str, default="")
    parser.add_argument(
        "--dl_scaler",
        choices=["none", "standard", "minmax"],
        default=PAIR_NEURAL_DEFAULTS["dl_scaler"],
    )
    parser.add_argument("--ml_scaler", choices=["none", "standard", "minmax"], default="standard")
    parser.add_argument(
        "--ml_reduction",
        choices=["none", "pca", "kpca", "tsvd"],
        default="none",
        help="Generic numeric reduction for fused Pair features; tsvd uses uncentered TruncatedSVD.",
    )
    parser.add_argument("--ml_reduction_dim", type=int, default=256)
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
    parser.add_argument(
        "--reduction",
        choices=["none", "lda", "plsa"],
        default="none",
        help="Strict Pair topic reduction; requires classic BoW streams, fusion_mode=concat, and a classical classifier.",
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
    parser.add_argument(
        "--encoder",
        type=encoder_argument,
        choices=PAIR_TRAINING_ENCODER_OPTIONS,
        metavar="ENCODER",
        default=None,
        help="representation encoder; pretrained routes use the same explicit id as --tokenizer",
    )
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
        "--dr_unknown_residue_policy",
        choices=("error", "zero"),
        default=DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
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
    parser.add_argument("--pssm_index_path", type=str, default="")
    parser.add_argument("--ac_pssm_blosum62_hybrid_index_path", type=str, default="")
    parser.add_argument("--cc_pssm_blosum62_hybrid_index_path", type=str, default="")
    parser.add_argument("--acc_pssm_blosum62_hybrid_index_path", type=str, default="")
    parser.add_argument("--psfm_index_path", type=str, default="")
    parser.add_argument("--frequency_profile_index_path", type=str, default="")
    parser.add_argument("--ac_pssm_lag", type=parse_pssm_covariance_lag, default=10)
    parser.add_argument("--cc_pssm_lag", type=parse_pssm_covariance_lag, default=10)
    parser.add_argument("--acc_pssm_lag", type=parse_pssm_covariance_lag, default=10)
    parser.add_argument("--ac_pssm_blosum62_hybrid_lag", type=parse_pssm_covariance_lag, default=10)
    parser.add_argument("--cc_pssm_blosum62_hybrid_lag", type=parse_pssm_covariance_lag, default=10)
    parser.add_argument(
        "--acc_pssm_blosum62_hybrid_lag", type=parse_pssm_covariance_lag, default=10
    )
    parser.add_argument("--ac_lag", type=int, default=10)
    parser.add_argument("--cc_lag", type=int, default=10)
    parser.add_argument("--acc_lag", type=int, default=10)
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
    parser.add_argument(
        "--covariance_max_raw_length",
        type=parse_covariance_max_raw_length,
        default=COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
        help="maximum complete raw sequence length for AC/CC/ACC (hard max 100000)",
    )
    parser.add_argument(
        "--profile_iterations", type=int, default=PAIR_PROFILE_DEFAULTS["profile_iterations"]
    )
    parser.add_argument(
        "--profile_evalue", type=float, default=PAIR_PROFILE_DEFAULTS["profile_evalue"]
    )
    parser.add_argument(
        "--profile_inclusion_evalue",
        type=float,
        default=PAIR_PROFILE_DEFAULTS["profile_inclusion_evalue"],
    )
    parser.add_argument(
        "--profile_threads", type=int, default=PAIR_PROFILE_DEFAULTS["profile_threads"]
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
    parser.add_argument("--svm_c", type=float, default=1.0)
    parser.add_argument("--svm_class_weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--svm_max_iter", type=int, default=3000)
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
        "--rf_estimators",
        type=int,
        default=PAIR_CLASSIFIER_PARAMETER_DEFAULTS["rf"]["rf_estimators"],
    )
    parser.add_argument(
        "--rf_max_depth", type=int, default=PAIR_CLASSIFIER_PARAMETER_DEFAULTS["rf"]["rf_max_depth"]
    )
    parser.add_argument(
        "--rf_min_samples_leaf",
        type=int,
        default=PAIR_CLASSIFIER_PARAMETER_DEFAULTS["rf"]["rf_min_samples_leaf"],
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
    parser.add_argument("--report_aupr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compact_residue_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="deduplicate and store actual-length pretrained residue states before Pair joining",
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--advanced_wizard", action="store_true")
    parser.add_argument("--artifact_dir", type=str, default="")
    parser.add_argument("--resume_embedding_bundle", type=str, default="")
    add_selection_arguments(parser)
    add_legacy_semantic_argument(parser)
    return register_legacy_cli_aliases(parser)


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    args.task_type = normalize_task_type(args.task_type)
    if args.list_objectives:
        print(json.dumps(objective_capabilities(), indent=2))
        return
    set_global_seed(args.seed, deterministic=args.deterministic)
    normalized_task = normalize_task_type(getattr(args, "task_type", "pep_protein_interaction"))
    if normalize_task_type(normalized_task) in [
        "pep_hlaii_interaction",
        "peptide_hlaii_interaction",
    ]:
        args.task_type = "pep_hlaii"
    if normalize_task_type(normalized_task) in ["pep_hlai_interaction", "peptide_hlai_interaction"]:
        args.task_type = "pep_hlai"

    no_arguments = len(argv) == 0 if argv is not None else len(sys.argv) == 1
    if args.interactive or no_arguments:
        args = PPIInteractiveWizard().run(args)
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
            args.profile_on_missing = PAIR_PROFILE_DEFAULTS["profile_on_missing"]

    resolve_selection(args)
    run_pair(args)



install_legacy_exports(globals())



from pepplatform.stream_compat import register_legacy_cli_aliases, install_legacy_exports as install_legacy_stream_exports

install_legacy_stream_exports(globals())


if __name__ == "__main__":
    run_with_failure_reporting(main)
