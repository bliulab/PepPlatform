"""Shared workflow taxonomy for guided configuration and runtime routing."""

from pepplatform.stream_compat import normalize_stream_namespace
from pepplatform.task_compat import normalize_task_type


from pepplatform.pair_terminology import normalize_pair_strategy

import json
import math
import os
from pathlib import Path

from pepplatform.esmc import (
    ESMC_OFFICIAL_600M_MODEL_ID,
    ESMPLUSPLUS_SMALL_MODEL_ID,
    ESMC_OFFICIAL_MODEL_IDS,
    esmc_variant_for_source,
    official_esmc_weight_path,
)
from pepplatform.esm3 import (
    ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH,
    is_complete_esm3_source,
)
from pepplatform.prott5 import PROTT5_MAX_RESIDUES, validate_prott5_max_residues
from pepplatform.kmer import KMER_MAX_LENGTH_UNIT
from pepplatform.covariance import (
    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
    COVARIANCE_ENCODERS,
    validate_covariance_lag,
    validate_covariance_max_raw_length,
)
from pepplatform.pdt import (
    PDT_DEFAULT_LAG,
    PDT_DEFAULT_MAX_RAW_LENGTH,
    PDT_ENCODERS,
    validate_pdt_lag,
    validate_pdt_max_raw_length,
)
from pepplatform.frequency import (
    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    validate_frequency_max_raw_length,
)
from pepplatform.lda import (
    LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    validate_lda_route,
)
from pepplatform.plsa import (
    PLSA_CONTRACT_VERSION,
    PLSA_DEFAULT_MEMORY_LIMIT_GIB,
    validate_plsa_route,
)
from pepplatform.pssm_covariance import (
    PSSM_COVARIANCE_DEFAULT_LAG,
    PSSM_COVARIANCE_ENCODERS,
    validate_pssm_covariance_lag,
)
from pepplatform.wordpiece import WORDPIECE_MAX_LENGTH_UNIT
from pepplatform.lsa import LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB, validate_lsa_route
from pepplatform.tsvd import TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB
from pepplatform.pca import (
    PCA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    validate_pca_dense_memory_limit_gib,
    validate_pca_dimension,
)
from pepplatform.kpca import (
    KPCA_DEFAULT_MEMORY_LIMIT_GIB,
    validate_kpca_dimension,
    validate_kpca_gamma,
    validate_kpca_kernel,
    validate_kpca_memory_limit_gib,
)

TOKENIZER_ALIASES = {
    "probert": "protbert",
}


def validate_stream_alignment_dim(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("stream_alignment_dim must be a positive integer")
    return int(value)


def parse_stream_alignment_dim(value):
    """Parse a CLI value, then apply the strict positive-integer contract."""
    if isinstance(value, str):
        text = value.strip()
        try:
            value = int(text, 10)
        except (TypeError, ValueError) as exc:
            raise ValueError("stream_alignment_dim must be a positive integer") from exc
    return validate_stream_alignment_dim(value)


def normalize_tokenizer_name(tokenizer):
    value = str(tokenizer or "").strip().lower()
    return TOKENIZER_ALIASES.get(value, value)


PRETRAINED_TOKENIZERS = ["esm2", "protbert", "prott5", "esmc", "esm3"]
PRETRAINED_MODEL_FIELDS = {
    "esm2": "esm2_model",
    "protbert": "protbert_model",
    "prott5": "prott5_model",
    "esmc": "esmc_model",
    "esm3": "esm3_model",
}
PRETRAINED_MODEL_DEFAULTS = {
    "esm2": "facebook/esm2_t30_150M_UR50D",
    "protbert": "Rostlab/prot_bert",
    "prott5": "Rostlab/prot_t5_xl_uniref50",
    "esmc": ESMC_OFFICIAL_600M_MODEL_ID,
    "esm3": "EvolutionaryScale/esm3-sm-open-v1",
}
PRETRAINED_MODEL_FIELD_ALIASES = {
    "protbert": ["probert_model"],
}
PRETRAINED_FEATURE_DIR_ALIASES = {
    "protbert": ["probert_feature_dir"],
}
PRETRAINED_MODEL_CACHE_FILES = {
    "esm2": {
        "tokenizer": ["tokenizer_config.json", "vocab.txt", "tokenizer.json"],
        "weights": [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ],
    },
    "protbert": {
        "tokenizer": ["tokenizer_config.json", "vocab.txt", "tokenizer.json"],
        "weights": [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ],
    },
    "prott5": {
        "tokenizer": ["tokenizer_config.json", "spiece.model", "tokenizer.json"],
        "weights": [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ],
    },
    "esmc": {
        "tokenizer": ["tokenizer_config.json", "tokenizer.json"],
        "weights": [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ],
    },
    "esm3": {
        "tokenizer": [],
        "weights": [ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH.as_posix()],
    },
}
CLASSIC_TOKENIZERS = ["kmer", "dr", "top-n-gram", "dt", "bpe", "wordpiece", "unigram"]
PROFILE_DEPENDENT_TOKENIZERS = ["top-n-gram", "dt"]
CLASSIC_PAIR_VECTOR_INFERENCE_TOKENIZERS = list(CLASSIC_TOKENIZERS)

SEQUENCE_ARCHITECTURES = ["cnn", "rnn", "gru", "lstm", "transformer"]
CHAIN_GRAPH_ARCHITECTURES = ["gcn", "gat"]
# Backward-compatible combined field used by legacy configs and clients.
SEQUENCE_MODELS = SEQUENCE_ARCHITECTURES + CHAIN_GRAPH_ARCHITECTURES
RESIDUE_ADAPTER_DIM_DEFAULT = 128
RESIDUE_ADAPTER_DIM_MIN = 16
RESIDUE_ADAPTER_DIM_MAX = 1024
RESIDUE_ADAPTER_DIM_MULTIPLE = 4
NEURAL_CLASSIFIERS = ["mlp"]
CLASSICAL_CLASSIFIERS = ["svm", "rf", "extratrees", "histgb", "xgb", "lgb", "logreg"]
LEGACY_CLASSIFIER_ALIASES = {"calibrated_svm": "sigmoid"}
SVM_CALIBRATION_OPTIONS = ["none", "sigmoid"]
SVM_CALIBRATION_CV_FOLDS = 3
CLASSIFIERS = NEURAL_CLASSIFIERS + CLASSICAL_CLASSIFIERS
PAIR_STRATEGIES = ["single_stream", "two_stream"]
VECTOR_INTERACTION = "vector"
TOKEN_INTERACTION = "token"
INTERACTION_LEVELS = [VECTOR_INTERACTION, TOKEN_INTERACTION]
PAIR_INTERACTION_MODELS = ["cross_attn", "bi_cross_attn"]
SEMANTIC_RELATION_TYPES = ["intrinsic_label_jaccard", "interaction_profile_jaccard"]
TASK_SEMANTIC_RELATION_TYPE = {
    "pep_function": "intrinsic_label_jaccard",
    "pep_protein_interaction": "interaction_profile_jaccard",
    "pep_hlaii": "interaction_profile_jaccard",
    "pep_hlai": "interaction_profile_jaccard",
}
# Backward-compatible capability fields for older clients.  There is no
# runtime mode selection in the unified semantic layer.
INTERACTION_SEMANTIC_MODES = ["profile_jaccard"]
TASK_INTERACTION_SEMANTIC_MODES = {
    "pep_protein_interaction": ["profile_jaccard"],
    "pep_hlaii": ["profile_jaccard"],
    "pep_hlai": ["profile_jaccard"],
}
TASK_INTERACTION_SEMANTIC_DEFAULTS = {
    task: modes[0] for task, modes in TASK_INTERACTION_SEMANTIC_MODES.items()
}
PAIR_TASK_TYPES = {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}
PREDICTOR_TYPES = ["classifier", "sample_graph"]
GRAPH_PREDICTORS = ["gcn", "gat", "hgt", "sage", "graph_transformer"]
PROFILE_DEPENDENT_ENCODERS = [
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
]
RESIDUE_PROFILE_ENCODERS = frozenset(PROFILE_DEPENDENT_ENCODERS)
PROFILE_DEPENDENT_INFERENCE_REASON = "this route requires sample-specific PSSM/PSFM profiles and cannot run Web inference from raw sequences only"
REMOVED_ENCODER_MESSAGES = {
    "cs": "CS encoder has been removed",
    "property": ("Property encoder has been removed; retraining with another encoder is required"),
}
REMOVED_ENCODERS = frozenset(REMOVED_ENCODER_MESSAGES)
REMOVED_FEATURE_TRANSFORMATION_MESSAGES = {
    "labelled_lda": (
        "Labelled LDA feature transformation has been removed; retraining with "
        "none, lsa, lda, or plsa is required"
    ),
}
REMOVED_FEATURE_TRANSFORMATIONS = frozenset(REMOVED_FEATURE_TRANSFORMATION_MESSAGES)

SEQUENCE_ENCODERS = [
    "index",
    "onehot",
    "blosum62",
    "pssm",
    "pssm_blosum62_hybrid",
    "psfm",
    "psfm_blosum62_hybrid",
    "word2vec",
    "glove",
    "fasttext",
]
CHAIN_GRAPH_COMPATIBLE_ENCODERS = [
    "onehot",
    "blosum62",
    "pssm",
    "pssm_blosum62_hybrid",
    "psfm",
    "psfm_blosum62_hybrid",
    "word2vec",
    "glove",
    "fasttext",
]
GLOBAL_ENCODERS = [
    "bow",
    "tfidf",
    "textrank",
    "ac",
    "cc",
    "acc",
    "pdt",
    "ac_pssm",
    "cc_pssm",
    "acc_pssm",
    "ac_pssm_blosum62_hybrid",
    "cc_pssm_blosum62_hybrid",
    "acc_pssm_blosum62_hybrid",
]
CLASSIC_PAIR_VECTOR_INFERENCE_ENCODERS = [
    "onehot",
    "bow",
    "tfidf",
    "textrank",
    "ac",
    "cc",
    "acc",
    "pdt",
    "blosum62",
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
    "word2vec",
    "glove",
    "fasttext",
]
CLASSIC_PAIR_TOKEN_INFERENCE_ENCODERS = list(SEQUENCE_ENCODERS)
PROFILE_GENERATED_PAIR_INFERENCE_REASON = "this route will generate PSSM/PSFM profiles during Web inference and requires local ncbi-blast+ tools"

CLASSIC_ENCODER_OPTIONS = {
    "kmer": SEQUENCE_ENCODERS[:2]
    + [
        "bow",
        "tfidf",
        "textrank",
        "ac",
        "cc",
        "acc",
        "pdt",
        "blosum62",
        "pssm",
        "pssm_blosum62_hybrid",
        "ac_pssm",
        "cc_pssm",
        "acc_pssm",
        "ac_pssm_blosum62_hybrid",
        "cc_pssm_blosum62_hybrid",
        "acc_pssm_blosum62_hybrid",
        "psfm",
        "psfm_blosum62_hybrid",
        "word2vec",
        "glove",
        "fasttext",
    ],
    "dr": ["index", "onehot", "bow", "tfidf", "textrank", "word2vec", "glove", "fasttext"],
    "top-n-gram": ["bow", "tfidf", "textrank"],
    "dt": ["bow", "tfidf", "textrank"],
    "bpe": ["index"],
    "wordpiece": ["index"],
    "unigram": ["index"],
}
TRAINING_ENCODER_OPTIONS = list(
    dict.fromkeys(encoder for options in CLASSIC_ENCODER_OPTIONS.values() for encoder in options)
) + [
    tokenizer
    for tokenizer in PRETRAINED_TOKENIZERS
    if tokenizer
    not in {encoder for options in CLASSIC_ENCODER_OPTIONS.values() for encoder in options}
]
PAIR_TRAINING_ENCODER_OPTIONS = [
    "ppi_fusion",
    *TRAINING_ENCODER_OPTIONS,
]

FEATURE_TRANSFORMATIONS = {
    "bow": ["none", "lsa", "lda", "plsa"],
    "tfidf": ["none", "lsa"],
    "textrank": ["none", "lsa"],
}

END_TO_END = "end_to_end"
EXPORT_EMBEDDINGS = "export_embeddings"
TRAINING_MODES = [END_TO_END, EXPORT_EMBEDDINGS]
PREDICTION_HEAD_ACTIVATIONS = ["relu", "gelu", "silu"]
PREDICTION_HEAD_FIELDS = ("head_hidden_dims", "head_activation", "head_dropout")


class TrainingModeContractError(ValueError):
    """Raised when a training mode is paired with the wrong final classifier."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = str(code)


def validate_training_mode_classifier(training_mode, classifier):
    """Enforce the platform's two training-mode/classifier compositions."""

    mode = str(training_mode or "").strip().lower()
    final_classifier = str(classifier or "").strip().lower()
    if mode == END_TO_END and final_classifier != "mlp":
        raise TrainingModeContractError(
            "end_to_end_requires_mlp",
            "training_mode=end_to_end requires classifier=mlp",
        )
    if mode == EXPORT_EMBEDDINGS and final_classifier not in CLASSICAL_CLASSIFIERS:
        raise TrainingModeContractError(
            "export_embeddings_requires_classical_classifier",
            "training_mode=export_embeddings requires a classical downstream classifier",
        )
    return mode, final_classifier


def parse_prediction_head_hidden_dims(value):
    """Parse a JSON array of positive integer hidden-layer widths."""
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        raw = list(value)
    elif isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "head_hidden_dims must be a JSON integer array, for example [256,128]"
            ) from exc
    else:
        raise ValueError("head_hidden_dims must be a JSON integer array")
    if not isinstance(raw, list):
        raise ValueError("head_hidden_dims must be a JSON integer array")
    result = []
    for index, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"head_hidden_dims[{index}] must be a positive integer")
        if item <= 0:
            raise ValueError(f"head_hidden_dims[{index}] must be a positive integer")
        result.append(int(item))
    return result


def prediction_head_applies(source):
    """Return whether a route has a configurable terminal neural head."""
    classifier = str(_workflow_value(source, "classifier", "") or "").strip().lower()
    if not classifier:
        classifier = str(_workflow_value(source, "model", "") or "").strip().lower()
    predictor_type = str(_workflow_value(source, "predictor_type", "") or "").strip().lower()
    return classifier == "mlp" and predictor_type != "sample_graph"


def end_to_end_prediction_head_applies(source):
    """Backward-compatible alias for the broader prediction-head predicate."""
    return prediction_head_applies(source)


def default_prediction_head_hidden_dims(source):
    """Return the standard single-hidden-layer prediction-head widths."""
    return [128]


def linear_prediction_head_has_trainable_upstream(source):
    """Return whether a linear head still receives a trainable representation."""
    sequence_model = str(_workflow_value(source, "sequence_model", "") or "").strip().lower()
    interaction_level = str(_workflow_value(source, "interaction_level", "") or "").strip().lower()
    fusion_mode = str(_workflow_value(source, "fusion_mode", "") or "").strip().lower()
    return (
        sequence_model in SEQUENCE_MODELS
        or interaction_level == TOKEN_INTERACTION
        or fusion_mode == "aligned_gated"
    )


def prediction_head_artifact_config(source):
    """Return the versioned artifact contract for an explicit prediction head."""
    hidden_dims = parse_prediction_head_hidden_dims(
        _workflow_value(source, "head_hidden_dims", None)
    )
    if hidden_dims is None:
        return None
    result = {
        "schema_version": 1,
        "implementation": "standard_linear_mlp",
        "kind": "mlp" if hidden_dims else "linear",
        "hidden_dims": hidden_dims,
    }
    if hidden_dims:
        result.update(
            {
                "activation": str(_workflow_value(source, "head_activation", "relu") or "relu")
                .strip()
                .lower(),
                "dropout": float(
                    _workflow_value(
                        source,
                        "head_dropout",
                        _workflow_value(source, "dropout", 0.0),
                    )
                    or 0.0
                ),
            }
        )
    return validate_prediction_head_artifact_config(result)


def validate_prediction_head_artifact_config(config):
    """Validate and canonicalize the persisted prediction-head contract."""

    if not isinstance(config, dict):
        raise ValueError("prediction_head must be an object")
    if config.get("schema_version") != 1:
        raise ValueError("prediction_head schema_version must be 1")
    if config.get("implementation") != "standard_linear_mlp":
        raise ValueError("unsupported prediction_head implementation")
    if "hidden_dims" not in config:
        raise ValueError("prediction_head.hidden_dims is required")
    hidden_dims = parse_prediction_head_hidden_dims(config.get("hidden_dims"))
    if hidden_dims is None:
        raise ValueError("prediction_head.hidden_dims must be an integer array")
    expected_kind = "mlp" if hidden_dims else "linear"
    if config.get("kind") != expected_kind:
        raise ValueError(
            f"prediction_head.kind must be {expected_kind!r} for the persisted hidden_dims"
        )
    result = {
        "schema_version": 1,
        "implementation": "standard_linear_mlp",
        "kind": expected_kind,
        "hidden_dims": hidden_dims,
    }
    if hidden_dims:
        activation = str(config.get("activation") or "").strip().lower()
        if activation not in PREDICTION_HEAD_ACTIVATIONS:
            raise ValueError(
                "prediction_head.activation must be one of "
                f"{{{', '.join(PREDICTION_HEAD_ACTIVATIONS)}}}"
            )
        try:
            dropout = float(config.get("dropout"))
        except (TypeError, ValueError) as exc:
            raise ValueError("prediction_head.dropout must be numeric") from exc
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("prediction_head.dropout must be finite and in [0, 1)")
        result.update({"activation": activation, "dropout": dropout})
    return result


def validate_linear_prediction_head_semantics(source):
    """Reject semantic loss when a linear-only route has no trainable representation."""
    if not prediction_head_applies(source):
        return source
    hidden_dims = parse_prediction_head_hidden_dims(
        _workflow_value(source, "head_hidden_dims", None)
    )
    semantic_enabled = bool(
        _workflow_value(source, "enable_semantic_similarity", False)
        or _workflow_value(source, "enable_interaction_semantic_similarity", False)
    )
    if (
        semantic_enabled
        and hidden_dims == []
        and not linear_prediction_head_has_trainable_upstream(source)
    ):
        raise ValueError(
            "semantic similarity with a linear prediction head requires a trainable "
            "upstream representation; choose at least one hidden layer for fixed-vector routes"
        )
    return source


def normalize_prediction_head_config(source):
    """Canonicalize configurable end-to-end prediction-head fields in a Namespace or dict."""
    applies = prediction_head_applies(source)
    supplied = {key: _workflow_value(source, key, None) for key in PREDICTION_HEAD_FIELDS}
    has_supplied = any(value not in [None, ""] for value in supplied.values())
    if not applies:
        if has_supplied:
            raise ValueError(
                "head_hidden_dims, head_activation, and head_dropout require "
                "a neural MLP route with a configurable terminal prediction head"
            )
        return source

    hidden_dims = parse_prediction_head_hidden_dims(supplied["head_hidden_dims"])
    if hidden_dims is None:
        hidden_dims = default_prediction_head_hidden_dims(source)
    _set_workflow_value(source, "head_hidden_dims", hidden_dims)

    if not hidden_dims:
        if supplied["head_activation"] not in [None, ""] or supplied["head_dropout"] not in [
            None,
            "",
        ]:
            raise ValueError(
                "head_activation and head_dropout are not used by a linear prediction head"
            )
        _delete_workflow_value(source, "head_activation")
        _delete_workflow_value(source, "head_dropout")
        return source

    activation = str(supplied["head_activation"] or "relu").strip().lower()
    if activation not in PREDICTION_HEAD_ACTIVATIONS:
        choices = ", ".join(PREDICTION_HEAD_ACTIVATIONS)
        raise ValueError(f"head_activation must be one of {{{choices}}}")
    _set_workflow_value(source, "head_activation", activation)

    raw_dropout = supplied["head_dropout"]
    if raw_dropout in [None, ""]:
        raw_dropout = _workflow_value(source, "dropout", 0.0)
    try:
        head_dropout = float(raw_dropout)
    except (TypeError, ValueError) as exc:
        raise ValueError("head_dropout must be numeric") from exc
    if (head_dropout < 0.0 or head_dropout >= 1.0) and supplied["head_dropout"] not in [None, ""]:
        raise ValueError("head_dropout must be in [0, 1)")
    _set_workflow_value(source, "head_dropout", head_dropout)
    return source


def _set_workflow_value(source, key, value):
    if isinstance(source, dict):
        source[key] = value
    else:
        setattr(source, key, value)


def _delete_workflow_value(source, key):
    if isinstance(source, dict):
        source.pop(key, None)
    elif hasattr(source, key):
        delattr(source, key)


# Canonical runtime defaults shared by Web preflight and the CLI training
# entrypoints. Keeping these here makes an Agent candidate's effective config
# match the arguments that will actually be used when the run starts.
SEQUENCE_NEURAL_DEFAULTS = {
    "max_length": 60,
    "batch_size": 32,
    "lr": 1e-3,
    "epochs": 30,
    "dropout": 0.0,
    "weight_decay": 1e-2,
    "grad_clip_norm": 0.0,
    "early_stop_patience": 0,
    "early_stop_min_delta": 0.0,
    "dl_scaler": "none",
}
PAIR_NEURAL_DEFAULTS = {
    "max_length": 512,
    "batch_size": 16,
    "lr": 1e-3,
    "epochs": 20,
    "dropout": 0.2,
    "weight_decay": 1e-2,
    "grad_clip_norm": 0.0,
    "early_stop_patience": 5,
    "early_stop_min_delta": 0.0,
    "dl_scaler": "standard",
}
SEQUENCE_PROFILE_DEFAULTS = {
    "profile_iterations": 3,
    "profile_evalue": 0.001,
    "profile_inclusion_evalue": 0.001,
    "profile_threads": 1,
    "profile_on_missing": "error",
}
PAIR_PROFILE_DEFAULTS = {
    **SEQUENCE_PROFILE_DEFAULTS,
    "profile_threads": 4,
}
CLASSICAL_TRAINING_DEFAULTS = {
    "ml_scaler": "standard",
    "ml_auto_tune": False,
    "ml_tuning_objective": "f1",
    "ml_tuning_mode": "cv",
    "ml_tuning_max_candidates": 6,
    "ml_cv_folds": 2,
}

AUTO_TUNABLE_CLASSIFIERS = tuple(CLASSICAL_CLASSIFIERS)
ML_TUNING_SPACE_VERSION = "bounded_grid_v1"
ML_TUNING_BUDGETS = {
    "quick": {"max_candidates": 6, "cv_folds": 2},
    "standard": {"max_candidates": 12, "cv_folds": 3},
    "thorough": {"max_candidates": 18, "cv_folds": 5},
}
ML_TUNING_MAX_CANDIDATES = 18

LGB_EXECUTION_POLICY = {
    "lgb_n_jobs": 4,
    "lgb_deterministic": True,
    "lgb_histogram_mode": "row_wise",
    "lgb_device_type": "cpu",
}


def default_model_selection_objective(task_type):
    """Return the default validation objective used to select a model for a task."""
    task_type = normalize_task_type(task_type)
    normalized = normalize_task_type(task_type or "")
    aliases = {
        "pep_hlaii_interaction": "pep_hlaii",
        "peptide_hlaii_interaction": "pep_hlaii",
        "pep_hlai_interaction": "pep_hlai",
        "peptide_hlai_interaction": "pep_hlai",
    }
    normalized = aliases.get(normalized, normalized)
    return "auc" if normalized in PAIR_TASK_TYPES else "f1"


CLASSIFIER_PARAMETER_DEFAULTS = {
    "svm": {
        "svm_c": 1.0,
        "svm_class_weight": "balanced",
        "svm_max_iter": 2000,
        "svm_tol": 1e-3,
        "svm_kernel": "linear",
        "svm_gamma": "scale",
        "svm_convergence_policy": "bounded_retry",
        "svm_calibration": "none",
        "svm_calibration_cv_folds": SVM_CALIBRATION_CV_FOLDS,
    },
    "logreg": {
        "logreg_c": 1.0,
        "logreg_class_weight": "balanced",
        "logreg_max_iter": 2000,
        "logreg_convergence_policy": "bounded_retry",
    },
    "rf": {
        "rf_estimators": 300,
        "rf_max_depth": 0,
        "rf_min_samples_leaf": 1,
    },
    "extratrees": {
        "et_estimators": 300,
        "et_max_depth": 0,
        "et_min_samples_leaf": 1,
    },
    "histgb": {
        "histgb_max_iter": 300,
        "histgb_max_leaf_nodes": 31,
        "histgb_lr": 0.1,
        "histgb_l2_regularization": 0.0,
    },
    "xgb": {
        "xgb_estimators": 300,
        "xgb_max_depth": 6,
        "xgb_lr": 0.1,
        "xgb_subsample": 1.0,
        "xgb_colsample": 1.0,
        "xgb_reg_alpha": 0.0,
        "xgb_reg_lambda": 1.0,
    },
    "lgb": {
        "lgb_estimators": 300,
        "lgb_max_depth": 6,
        "lgb_lr": 0.1,
        "lgb_subsample": 1.0,
        "lgb_colsample": 1.0,
        "lgb_reg_alpha": 0.0,
        "lgb_reg_lambda": 0.0,
    },
}
PAIR_CLASSIFIER_PARAMETER_DEFAULTS = {
    **CLASSIFIER_PARAMETER_DEFAULTS,
    "svm": {
        **CLASSIFIER_PARAMETER_DEFAULTS["svm"],
        "svm_max_iter": 3000,
    },
}


def normalize_encoder_name(encoder):
    value = str(encoder or "").strip().lower()
    if value in {"tf-idf", "tf_idf"}:
        return "tfidf"
    if value in {"physchem", "physicochemical(property)"}:
        return "property"
    return value


def ensure_encoder_available(encoder):
    normalized = normalize_encoder_name(encoder)
    if normalized in REMOVED_ENCODER_MESSAGES:
        raise ValueError(REMOVED_ENCODER_MESSAGES[normalized])


def ensure_feature_transformation_available(reduction):
    normalized = str(reduction or "none").strip().lower()
    if normalized in REMOVED_FEATURE_TRANSFORMATIONS:
        raise ValueError(REMOVED_FEATURE_TRANSFORMATION_MESSAGES[normalized])
    return normalized


def _workflow_value(source, key, default=""):
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _workflow_first_value(source, keys, default=""):
    for key in keys:
        value = _workflow_value(source, key, "")
        if value not in [None, ""]:
            return value
    return default


def pretrained_model_fields(tokenizer):
    tokenizer = normalize_tokenizer_name(tokenizer)
    field = PRETRAINED_MODEL_FIELDS.get(tokenizer, "")
    if not field:
        return []
    return [field, *PRETRAINED_MODEL_FIELD_ALIASES.get(tokenizer, [])]


def normalize_pretrained_alias_args(args):
    tokenizer = normalize_tokenizer_name(getattr(args, "tokenizer", ""))
    if hasattr(args, "tokenizer"):
        args.tokenizer = tokenizer
    aux_tokenizer = getattr(args, "aux_tokenizer", None)
    if aux_tokenizer is not None:
        args.aux_tokenizer = normalize_tokenizer_name(aux_tokenizer)
    for canonical, aliases in PRETRAINED_MODEL_FIELD_ALIASES.items():
        canonical_field = PRETRAINED_MODEL_FIELDS.get(canonical, "")
        if not canonical_field:
            continue
        canonical_value = getattr(args, canonical_field, "")
        default_value = PRETRAINED_MODEL_DEFAULTS.get(canonical, "")
        for alias in aliases:
            alias_value = getattr(args, alias, "")
            if alias_value and (not canonical_value or canonical_value == default_value):
                setattr(args, canonical_field, alias_value)
                break
    for canonical, aliases in PRETRAINED_FEATURE_DIR_ALIASES.items():
        canonical_field = f"{canonical}_feature_dir"
        if not getattr(args, canonical_field, ""):
            for alias in aliases:
                alias_value = getattr(args, alias, "")
                if alias_value:
                    setattr(args, canonical_field, alias_value)
                    break
    return args


def uses_profile_features(source):
    tokenizer = normalize_tokenizer_name(_workflow_value(source, "tokenizer", ""))
    encoder = normalize_encoder_name(_workflow_value(source, "encoder", ""))
    if tokenizer in PROFILE_DEPENDENT_TOKENIZERS or encoder in PROFILE_DEPENDENT_ENCODERS:
        return True
    streams = _workflow_value(source, "streams", {})
    if not isinstance(streams, dict):
        return False
    return any(uses_profile_features(stream) for stream in streams.values() if isinstance(stream, dict))


def effective_pair_vector_encoder(source):
    encoder = normalize_encoder_name(_workflow_value(source, "encoder", ""))
    if encoder == "ppi_fusion":
        encoder = normalize_encoder_name(_workflow_value(source, "aux_encoder", "index"))
    return encoder


def supports_classic_pair_vector_inference(source):
    tokenizer = normalize_tokenizer_name(_workflow_value(source, "tokenizer", ""))
    encoder = effective_pair_vector_encoder(source)
    allowed_for_tokenizer = {
        normalize_encoder_name(item) for item in CLASSIC_ENCODER_OPTIONS.get(tokenizer, [])
    }
    if tokenizer not in CLASSIC_PAIR_VECTOR_INFERENCE_TOKENIZERS:
        return False
    if encoder not in CLASSIC_PAIR_VECTOR_INFERENCE_ENCODERS:
        return False
    if encoder not in allowed_for_tokenizer:
        return False
    return True


def supports_classic_pair_token_inference(source):
    tokenizer = normalize_tokenizer_name(_workflow_value(source, "tokenizer", ""))
    encoder = effective_pair_vector_encoder(source)
    allowed_for_tokenizer = {
        normalize_encoder_name(item) for item in CLASSIC_ENCODER_OPTIONS.get(tokenizer, [])
    }
    if tokenizer not in CLASSIC_TOKENIZERS:
        return False
    if encoder not in CLASSIC_PAIR_TOKEN_INFERENCE_ENCODERS:
        return False
    if encoder not in allowed_for_tokenizer:
        return False
    return True


RAW_INDEX_ROUTE_MESSAGE = (
    "raw index token IDs require a learned embedding and cannot be used as "
    "flattened vector or graph features"
)


def _classic_index_stream_roles(source):
    streams = _workflow_value(source, "streams", {})
    if isinstance(streams, dict) and streams:
        roles = []
        for role, stream in streams.items():
            if not isinstance(stream, dict):
                continue
            tokenizer = normalize_tokenizer_name(stream.get("tokenizer", ""))
            encoder = normalize_encoder_name(stream.get("encoder", ""))
            if tokenizer not in PRETRAINED_TOKENIZERS and encoder == "index":
                roles.append(str(role))
        return roles

    tokenizer = normalize_tokenizer_name(_workflow_value(source, "tokenizer", ""))
    encoder = normalize_encoder_name(_workflow_value(source, "encoder", ""))
    if tokenizer not in PRETRAINED_TOKENIZERS and encoder == "index":
        return ["sequence"]
    return []


def validate_index_route_semantics(source):
    """Require categorical token IDs to enter a learnable embedding first."""
    roles = _classic_index_stream_roles(source)
    if not roles:
        return

    task_type = normalize_task_type(_workflow_value(source, "task_type", "") or "")
    if normalize_task_type(task_type) in PAIR_TASK_TYPES:
        pair_strategy = (
            str(_workflow_value(source, "pair_strategy", "two_stream") or "two_stream")
            .strip()
            .lower()
        )
        interaction_level = (
            str(
                _workflow_value(source, "interaction_level", VECTOR_INTERACTION)
                or VECTOR_INTERACTION
            )
            .strip()
            .lower()
        )
        if pair_strategy == "two_stream" and interaction_level == VECTOR_INTERACTION:
            role_text = ", ".join(roles)
            raise ValueError(f"{RAW_INDEX_ROUTE_MESSAGE}; affected streams: {role_text}")
        if pair_strategy == "single_stream":
            sequence_model = (
                str(
                    _workflow_value(source, "sequence_model", "")
                    or _workflow_value(source, "model", "")
                )
                .strip()
                .lower()
            )
            if sequence_model not in SEQUENCE_ARCHITECTURES:
                raise ValueError(RAW_INDEX_ROUTE_MESSAGE)
        return

    sequence_model = (
        str(_workflow_value(source, "sequence_model", "") or _workflow_value(source, "model", ""))
        .strip()
        .lower()
    )
    if sequence_model not in SEQUENCE_ARCHITECTURES:
        raise ValueError(RAW_INDEX_ROUTE_MESSAGE)


def supports_pair_token_inference(source):
    tokenizer = normalize_tokenizer_name(_workflow_value(source, "tokenizer", ""))
    encoder = normalize_encoder_name(_workflow_value(source, "encoder", ""))
    pair_strategy = str(_workflow_value(source, "pair_strategy", "") or "").strip().lower()
    interaction_level = str(_workflow_value(source, "interaction_level", "") or "").strip().lower()
    interaction_model = str(_workflow_value(source, "interaction_model", "") or "").strip().lower()
    training_mode = str(_workflow_value(source, "training_mode", "") or "").strip().lower()
    model = str(_workflow_value(source, "model", "") or "").strip().lower()
    classifier = str(_workflow_value(source, "classifier", "") or "").strip().lower() or "mlp"
    feature_supported = (
        tokenizer in PRETRAINED_TOKENIZERS and encoder in {tokenizer, "index", "ppi_fusion"}
    ) or supports_classic_pair_token_inference(source)
    return (
        feature_supported
        and pair_strategy == "two_stream"
        and interaction_level == TOKEN_INTERACTION
        and interaction_model in PAIR_INTERACTION_MODELS
        and training_mode == END_TO_END
        and model in {interaction_model, f"{interaction_model}+mlp"}
        and classifier == "mlp"
    )


def pretrained_model_field(tokenizer):
    return PRETRAINED_MODEL_FIELDS.get(normalize_tokenizer_name(tokenizer), "")


def pretrained_model_source(source):
    tokenizer = normalize_tokenizer_name(_workflow_value(source, "tokenizer", ""))
    fields = pretrained_model_fields(tokenizer)
    if not fields:
        return ""
    return str(
        _workflow_first_value(source, fields, PRETRAINED_MODEL_DEFAULTS.get(tokenizer, ""))
    ).strip()


def pretrained_tokenizer_for_model_source(source):
    value = str(source or "").strip()
    if value in {*ESMC_OFFICIAL_MODEL_IDS, ESMPLUSPLUS_SMALL_MODEL_ID}:
        return "esmc"
    for tokenizer, default in PRETRAINED_MODEL_DEFAULTS.items():
        if value == default:
            return tokenizer
    return ""


def pretrained_model_local_candidates(project_root, tokenizer, model_source=""):
    root = Path(project_root)
    tokenizer = normalize_tokenizer_name(tokenizer)
    model_source = str(model_source or PRETRAINED_MODEL_DEFAULTS.get(tokenizer, "")).strip()
    candidates = {
        "esm2": [
            root / "models" / "esm2",
            root / "models" / "esm2_t30_150M_UR50D",
            root / "models" / "facebook" / "esm2_t30_150M_UR50D",
            Path(os.path.expanduser("~/models/esm2")),
        ],
        "protbert": [
            root / "models" / "protbert",
            root / "models" / "probert",
            root / "models" / "prot_bert",
            root / "models" / "Rostlab" / "prot_bert",
            Path(os.path.expanduser("~/models/protbert")),
            Path(os.path.expanduser("~/models/probert")),
        ],
        "prott5": [
            root / "models" / "prott5",
            root / "models" / "prot_t5_xl_uniref50",
            root / "models" / "Rostlab" / "prot_t5_xl_uniref50",
            Path(os.path.expanduser("~/models/prott5")),
        ],
        "esmc": [],
        "esm3": [
            root / "models" / "esm3",
            root / "models" / "esm3-sm-open-v1",
            root / "models" / "EvolutionaryScale" / "esm3-sm-open-v1",
            Path(os.path.expanduser("~/models/esm3")),
        ],
    }
    if tokenizer == "esmc":
        official_variant = esmc_variant_for_source(model_source)
        if official_variant is not None:
            variant_key, variant = official_variant
            model_slug = str(variant["model_id"]).split("/", 1)[-1]
            candidates["esmc"] = [
                root / "models" / f"esmc-{variant_key}",
                root / "models" / model_slug,
                root / "models" / "EvolutionaryScale" / model_slug,
                Path(os.path.expanduser(f"~/models/esmc-{variant_key}")),
                Path(os.path.expanduser(f"~/models/{model_slug}")),
                Path(os.path.expanduser(f"~/models/EvolutionaryScale/{model_slug}")),
            ]
            if variant_key == "600m":
                candidates["esmc"].insert(0, root / "models" / "esmc")
                candidates["esmc"].append(Path(os.path.expanduser("~/models/esmc")))
        elif model_source == ESMPLUSPLUS_SMALL_MODEL_ID:
            candidates["esmc"] = [
                root / "models" / "ESMplusplus_small",
                root / "models" / "Synthyra" / "ESMplusplus_small",
                Path(os.path.expanduser("~/models/ESMplusplus_small")),
            ]
    return [str(path) for path in candidates.get(tokenizer, [])]


def _pretrained_cache_roots():
    roots = []
    explicit_cache = str(os.environ.get("HUGGINGFACE_HUB_CACHE", "") or "").strip()
    if explicit_cache:
        roots.append(Path(os.path.expanduser(explicit_cache)))
    hf_home = str(os.environ.get("HF_HOME", "") or "").strip()
    if hf_home:
        roots.append(Path(os.path.expanduser(hf_home)) / "hub")
    roots.append(Path(os.path.expanduser("~/.cache/huggingface/hub")))

    seen = set()
    unique = []
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _snapshot_has_any(snapshot, filenames):
    return any((snapshot / filename).exists() for filename in filenames)


def _snapshot_is_complete(snapshot, tokenizer, model_source=""):
    tokenizer = normalize_tokenizer_name(tokenizer)
    if tokenizer == "esmc":
        official_variant = esmc_variant_for_source(str(model_source).strip())
        if official_variant is not None:
            _variant_key, variant = official_variant
            return (snapshot / Path(variant["weight_relative_path"])).is_file()
    if tokenizer == "esm3":
        return is_complete_esm3_source(snapshot)
    if not (snapshot / "config.json").exists():
        return False
    requirements = PRETRAINED_MODEL_CACHE_FILES.get(tokenizer, {})
    tokenizer_files = requirements.get("tokenizer") or [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "spiece.model",
    ]
    weight_files = requirements.get("weights") or ["model.safetensors", "pytorch_model.bin"]
    return _snapshot_has_any(snapshot, tokenizer_files) and _snapshot_has_any(
        snapshot, weight_files
    )


def _snapshot_candidates(repo_cache):
    snapshots = repo_cache / "snapshots"
    candidates = []
    ref = repo_cache / "refs" / "main"
    if ref.exists():
        revision = ref.read_text(encoding="utf-8").strip()
        if revision:
            candidates.append(snapshots / revision)
    if snapshots.exists():
        candidates.extend(
            sorted(
                [path for path in snapshots.iterdir() if path.is_dir()],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )

    seen = set()
    unique = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_cached_pretrained_model_source(raw_model, tokenizer=""):
    raw_model = str(raw_model or "").strip()
    tokenizer = normalize_tokenizer_name(
        tokenizer or pretrained_tokenizer_for_model_source(raw_model)
    )
    if not raw_model or not tokenizer:
        return ""

    repo_cache_name = "models--" + raw_model.replace("/", "--")
    for root in _pretrained_cache_roots():
        repo_cache = root / repo_cache_name
        if not repo_cache.exists():
            continue
        for snapshot in _snapshot_candidates(repo_cache):
            if _snapshot_is_complete(snapshot, tokenizer, raw_model):
                return str(snapshot)
    return ""


def resolve_pretrained_model_source(
    project_root, raw_model, local_candidates, label, cli_arg, tokenizer=""
):
    raw_model = str(raw_model or "").strip()
    if os.path.isabs(raw_model) and os.path.exists(raw_model):
        if normalize_tokenizer_name(tokenizer) == "esm3" and not is_complete_esm3_source(raw_model):
            raise SystemExit(
                f"{label} local model is incomplete; expected "
                f"{ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH.as_posix()} under {raw_model}"
            )
        return str(Path(raw_model).resolve())
    rel_path = os.path.join(str(project_root), raw_model)
    if raw_model and os.path.exists(rel_path):
        if normalize_tokenizer_name(tokenizer) == "esm3" and not is_complete_esm3_source(rel_path):
            raise SystemExit(
                f"{label} local model is incomplete; expected "
                f"{ESM3_OFFICIAL_WEIGHT_RELATIVE_PATH.as_posix()} under {rel_path}"
            )
        return str(Path(rel_path).resolve())
    if normalize_tokenizer_name(tokenizer) == "esmc":
        local_candidates = pretrained_model_local_candidates(project_root, tokenizer, raw_model)
    for path in local_candidates:
        if os.path.exists(path):
            if normalize_tokenizer_name(tokenizer) == "esm3" and not is_complete_esm3_source(path):
                continue
            if normalize_tokenizer_name(tokenizer) == "esmc":
                official_variant = esmc_variant_for_source(raw_model)
                if (
                    official_variant is not None
                    and official_esmc_weight_path(
                        path, model_id=str(official_variant[1]["model_id"])
                    )
                    is None
                ):
                    continue
            # Prepared model links retain snapshots/<revision> in the resolved
            # source so runtime provenance agrees with the original artifacts.
            return str(Path(path).resolve())
    cached = resolve_cached_pretrained_model_source(raw_model, tokenizer)
    if cached:
        return cached
    if str(os.environ.get("PEPPLATFORM_OFFLINE_MODE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        candidates = ", ".join([*local_candidates, "Hugging Face cache"])
        raise SystemExit(
            f"{label} local model not found. Set --{cli_arg} to a local directory or place the model under one of: {candidates}"
        )
    return raw_model


def encoder_family(encoder):
    normalized = normalize_encoder_name(encoder)
    if normalized in SEQUENCE_ENCODERS:
        return "sequence"
    if normalized in GLOBAL_ENCODERS:
        return "global"
    return "unknown"


def encoder_options_for_tokenizer(tokenizer):
    return list(CLASSIC_ENCODER_OPTIONS.get(normalize_tokenizer_name(tokenizer), ["index"]))


def transformation_options_for_encoder(encoder):
    return list(FEATURE_TRANSFORMATIONS.get(normalize_encoder_name(encoder), ["none"]))


def uses_sequence_model(args):
    return str(getattr(args, "sequence_model", "") or "").strip().lower() in SEQUENCE_MODELS


def exports_sequence_embeddings(args):
    return (
        uses_sequence_model(args)
        and str(getattr(args, "training_mode", "") or "").strip().lower() == EXPORT_EMBEDDINGS
    )


def effective_classifier(args):
    classifier = str(getattr(args, "classifier", "") or "").strip().lower()
    if classifier in LEGACY_CLASSIFIER_ALIASES:
        return "svm"
    if classifier:
        return classifier
    model = str(getattr(args, "model", "") or "").strip().lower()
    if model in LEGACY_CLASSIFIER_ALIASES:
        return "svm"
    return model if model in CLASSIFIERS else ""


def normalize_svm_calibration_args(args):
    """Normalize the historical calibrated_svm name to the canonical SVM mode."""
    model = str(getattr(args, "model", "") or "").strip().lower()
    classifier = str(getattr(args, "classifier", "") or "").strip().lower()
    raw_calibration = str(getattr(args, "svm_calibration", "") or "").strip().lower()
    legacy_alias = model == "calibrated_svm" or classifier == "calibrated_svm"
    is_svm_route = model == "svm" or classifier == "svm" or legacy_alias
    if not is_svm_route:
        if raw_calibration not in {"", "none"}:
            raise ValueError("svm_calibration is only valid when classifier=svm")
        if hasattr(args, "svm_calibration"):
            delattr(args, "svm_calibration")
        if hasattr(args, "svm_calibration_cv_folds"):
            delattr(args, "svm_calibration_cv_folds")
        return args
    if legacy_alias:
        if raw_calibration and raw_calibration != "sigmoid":
            raise ValueError(
                "calibrated_svm is a legacy alias for svm_calibration=sigmoid; "
                "conflicting svm_calibration is not allowed"
            )
        raw_calibration = "sigmoid"
        if model == "calibrated_svm":
            model = "svm"
        if classifier == "calibrated_svm":
            classifier = "svm"
    calibration = raw_calibration or "none"
    if calibration not in SVM_CALIBRATION_OPTIONS:
        raise ValueError(
            f"unsupported svm_calibration: {calibration}; expected one of {SVM_CALIBRATION_OPTIONS}"
        )
    raw_folds = getattr(args, "svm_calibration_cv_folds", SVM_CALIBRATION_CV_FOLDS)
    try:
        folds = int(raw_folds)
    except (TypeError, ValueError) as exc:
        raise ValueError("svm_calibration_cv_folds must be an integer") from exc
    if folds != SVM_CALIBRATION_CV_FOLDS:
        raise ValueError(f"svm_calibration_cv_folds is fixed at {SVM_CALIBRATION_CV_FOLDS}")
    args.model = model
    args.classifier = classifier
    args.svm_calibration = calibration
    args.svm_calibration_cv_folds = folds
    return args


def normalize_ml_reduction_args(args):
    if not hasattr(args, "ml_reduction"):
        return args
    reduction = str(getattr(args, "ml_reduction", "none") or "none").strip().lower()
    args.ml_reduction = reduction
    if reduction == "pca":
        args.ml_reduction_dim = validate_pca_dimension(getattr(args, "ml_reduction_dim", 128))
        args.pca_dense_memory_limit_gib = validate_pca_dense_memory_limit_gib(
            getattr(
                args,
                "pca_dense_memory_limit_gib",
                PCA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
            )
        )
    elif reduction == "kpca":
        args.ml_reduction_dim = validate_kpca_dimension(getattr(args, "ml_reduction_dim", 128))
        args.kpca_memory_limit_gib = validate_kpca_memory_limit_gib(
            getattr(args, "kpca_memory_limit_gib", KPCA_DEFAULT_MEMORY_LIMIT_GIB)
        )
        args.kpca_kernel = validate_kpca_kernel(getattr(args, "kpca_kernel", "rbf"))
        args.kpca_gamma = validate_kpca_gamma(
            args.kpca_kernel,
            getattr(args, "kpca_gamma", None),
        )
    return args


def normalize_workflow_args(args):
    """Fill explicit workflow roles while preserving legacy ``--model`` behavior."""
    if hasattr(args, "task_type"):
        args.task_type = normalize_task_type(args.task_type)
    normalize_pretrained_alias_args(args)
    normalize_svm_calibration_args(args)
    from pepplatform.route_registry import normalize_encoder_for_tokenizer

    tokenizer = normalize_tokenizer_name(getattr(args, "tokenizer", ""))
    raw_encoder = str(getattr(args, "encoder", "") or "").strip().lower()
    encoder, legacy_encoder = normalize_encoder_for_tokenizer(tokenizer, raw_encoder)
    args.tokenizer = tokenizer
    args.encoder = encoder
    args.reduction = ensure_feature_transformation_available(getattr(args, "reduction", "none"))
    if encoder in COVARIANCE_ENCODERS:
        lag_key = f"{encoder}_lag"
        setattr(
            args,
            lag_key,
            validate_covariance_lag(getattr(args, lag_key, 10)),
        )
        args.covariance_max_raw_length = validate_covariance_max_raw_length(
            getattr(
                args,
                "covariance_max_raw_length",
                COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    if encoder in PDT_ENCODERS:
        args.pdt_lag = validate_pdt_lag(getattr(args, "pdt_lag", PDT_DEFAULT_LAG))
        args.pdt_max_raw_length = validate_pdt_max_raw_length(
            getattr(
                args,
                "pdt_max_raw_length",
                PDT_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    if encoder in PSSM_COVARIANCE_ENCODERS:
        lag_key = f"{encoder}_lag"
        setattr(
            args,
            lag_key,
            validate_pssm_covariance_lag(getattr(args, lag_key, PSSM_COVARIANCE_DEFAULT_LAG)),
        )
    if tokenizer in {"kmer", "dr", "top-n-gram", "dt"} and encoder in {"bow", "tfidf"}:
        args.frequency_max_raw_length = validate_frequency_max_raw_length(
            getattr(
                args,
                "frequency_max_raw_length",
                FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    if tokenizer == "kmer":
        args.max_length_unit = KMER_MAX_LENGTH_UNIT
    elif tokenizer == "wordpiece":
        args.max_length_unit = WORDPIECE_MAX_LENGTH_UNIT
    elif hasattr(args, "max_length_unit"):
        delattr(args, "max_length_unit")
    if legacy_encoder:
        print(
            f"[route-warning] encoder={raw_encoder} was a legacy pretrained placeholder; "
            f"using encoder={encoder}"
        )
    model = str(getattr(args, "model", "") or "").strip().lower()
    sequence_model = str(getattr(args, "sequence_model", "") or "").strip().lower()
    training_mode = str(getattr(args, "training_mode", "") or "").strip().lower()
    classifier = str(getattr(args, "classifier", "") or "").strip().lower()

    if not sequence_model and model in SEQUENCE_MODELS:
        sequence_model = model
    if sequence_model:
        if sequence_model not in SEQUENCE_MODELS:
            raise ValueError(f"unsupported sequence_model: {sequence_model}")
        if not training_mode:
            training_mode = EXPORT_EMBEDDINGS if classifier in CLASSICAL_CLASSIFIERS else END_TO_END
        if training_mode not in TRAINING_MODES:
            raise ValueError(f"unsupported training_mode: {training_mode}")
        if not classifier and training_mode == END_TO_END:
            classifier = "mlp"
        if classifier and classifier not in CLASSIFIERS:
            raise ValueError(f"unsupported classifier: {classifier}")
        training_mode, classifier = validate_training_mode_classifier(
            training_mode,
            classifier,
        )
        args.model = sequence_model
    else:
        if not classifier and model in CLASSIFIERS:
            classifier = model
        if classifier:
            if classifier not in CLASSIFIERS:
                raise ValueError(f"unsupported classifier: {classifier}")
            args.model = classifier
        training_mode = ""

    args.sequence_model = sequence_model
    args.training_mode = training_mode
    args.classifier = classifier
    if tokenizer in PRETRAINED_TOKENIZERS and sequence_model:
        raw_adapter_dim = getattr(args, "residue_adapter_dim", RESIDUE_ADAPTER_DIM_DEFAULT)
        try:
            args.residue_adapter_dim = int(raw_adapter_dim)
        except (TypeError, ValueError) as exc:
            raise ValueError("residue_adapter_dim must be an integer") from exc
        if not RESIDUE_ADAPTER_DIM_MIN <= args.residue_adapter_dim <= RESIDUE_ADAPTER_DIM_MAX:
            raise ValueError(
                "residue_adapter_dim must be between "
                f"{RESIDUE_ADAPTER_DIM_MIN} and {RESIDUE_ADAPTER_DIM_MAX}"
            )
        if args.residue_adapter_dim % RESIDUE_ADAPTER_DIM_MULTIPLE != 0:
            raise ValueError(
                f"residue_adapter_dim must be divisible by {RESIDUE_ADAPTER_DIM_MULTIPLE}"
            )
    if args.reduction == "lsa" and not hasattr(args, "lsa_dense_memory_limit_gib"):
        args.lsa_dense_memory_limit_gib = LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
    if args.reduction == "lda" and not hasattr(args, "lda_dense_memory_limit_gib"):
        args.lda_dense_memory_limit_gib = LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
    if args.reduction == "plsa":
        if not hasattr(args, "plsa_memory_limit_gib"):
            args.plsa_memory_limit_gib = PLSA_DEFAULT_MEMORY_LIMIT_GIB
        args.plsa_contract_version = PLSA_CONTRACT_VERSION
    if not hasattr(args, "tsvd_dense_memory_limit_gib"):
        args.tsvd_dense_memory_limit_gib = TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB
    if not hasattr(args, "pca_dense_memory_limit_gib"):
        args.pca_dense_memory_limit_gib = PCA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
    if not hasattr(args, "kpca_memory_limit_gib"):
        args.kpca_memory_limit_gib = KPCA_DEFAULT_MEMORY_LIMIT_GIB
    normalize_ml_reduction_args(args)
    validate_lsa_route(args, pair=False)
    validate_lda_route(args, pair=False)
    validate_plsa_route(args, pair=False)
    return args


def _legacy_pair_strategy(args, model):
    explicit_ppi = str(getattr(args, "ppi_interaction_mode", "") or "").strip().lower()
    if explicit_ppi in {"single", "dual"}:
        return "single_stream" if explicit_ppi == "single" else "two_stream"
    explicit_hlaii = str(getattr(args, "hlaii_interaction_mode", "") or "").strip().lower()
    if explicit_hlaii in {"concat", "contact"}:
        return "single_stream" if explicit_hlaii == "concat" else "two_stream"
    if model in GRAPH_PREDICTORS:
        return "two_stream"
    if model in SEQUENCE_MODELS:
        return "single_stream"
    if model == "contact":
        return "two_stream"
    return "two_stream"


def normalize_pair_workflow_args(args):
    """Fill explicit pair-task roles while preserving legacy route flags."""
    normalize_stream_namespace(args)
    if getattr(args, "pair_strategy", None):
        args.pair_strategy = normalize_pair_strategy(args.pair_strategy)
    if hasattr(args, "task_type"):
        args.task_type = normalize_task_type(args.task_type)
    normalize_pretrained_alias_args(args)
    normalize_svm_calibration_args(args)
    from pepplatform.route_registry import legacy_encoder_messages, normalize_encoder_for_tokenizer

    for message in legacy_encoder_messages(vars(args)):
        print(f"[route-warning] {message}")
    tokenizer = normalize_tokenizer_name(getattr(args, "tokenizer", ""))
    args.tokenizer = tokenizer
    if tokenizer == "prott5":
        requested_max_length = int(getattr(args, "max_length", PAIR_NEURAL_DEFAULTS["max_length"]))
        if requested_max_length == PROTT5_MAX_RESIDUES + 1:
            args.max_length = PROTT5_MAX_RESIDUES
            args.prott5_max_length_adjustment = {
                "requested": requested_max_length,
                "effective": PROTT5_MAX_RESIDUES,
                "reason": "T5 n_positions includes the EOS token",
            }
        else:
            args.max_length = validate_prott5_max_residues(
                requested_max_length,
                field="max_length",
            )
    args.reduction = ensure_feature_transformation_available(getattr(args, "reduction", "none"))
    validate_lsa_route(args, pair=True)
    if tokenizer == "kmer":
        args.max_length_unit = KMER_MAX_LENGTH_UNIT
    elif tokenizer == "wordpiece":
        args.max_length_unit = WORDPIECE_MAX_LENGTH_UNIT
    elif hasattr(args, "max_length_unit"):
        delattr(args, "max_length_unit")
    args.encoder, _legacy_encoder = normalize_encoder_for_tokenizer(
        tokenizer,
        getattr(args, "encoder", ""),
    )
    if args.encoder in COVARIANCE_ENCODERS:
        lag_key = f"{args.encoder}_lag"
        setattr(
            args,
            lag_key,
            validate_covariance_lag(getattr(args, lag_key, 10)),
        )
        args.covariance_max_raw_length = validate_covariance_max_raw_length(
            getattr(
                args,
                "covariance_max_raw_length",
                COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    if args.encoder in PDT_ENCODERS:
        args.pdt_lag = validate_pdt_lag(getattr(args, "pdt_lag", PDT_DEFAULT_LAG))
        args.pdt_max_raw_length = validate_pdt_max_raw_length(
            getattr(
                args,
                "pdt_max_raw_length",
                PDT_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    if args.encoder in PSSM_COVARIANCE_ENCODERS:
        lag_key = f"{args.encoder}_lag"
        setattr(
            args,
            lag_key,
            validate_pssm_covariance_lag(getattr(args, lag_key, PSSM_COVARIANCE_DEFAULT_LAG)),
        )
    if tokenizer in {"kmer", "dr", "top-n-gram", "dt"} and args.encoder in {"bow", "tfidf"}:
        args.frequency_max_raw_length = validate_frequency_max_raw_length(
            getattr(
                args,
                "frequency_max_raw_length",
                FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    raw_model = str(getattr(args, "model", "") or "").strip().lower()
    if raw_model == "cross_attn+mlp":
        raw_model = "cross_attn"
    elif raw_model == "bi_cross_attn+mlp":
        raw_model = "bi_cross_attn"

    pair_strategy = str(getattr(args, "pair_strategy", "") or "").strip().lower()
    interaction_level = str(getattr(args, "interaction_level", "") or "").strip().lower()
    interaction_model = str(getattr(args, "interaction_model", "") or "").strip().lower()
    predictor_type = str(getattr(args, "predictor_type", "") or "").strip().lower()
    graph_predictor = str(getattr(args, "graph_predictor", "") or "").strip().lower()

    sequence_model = str(getattr(args, "sequence_model", "") or "").strip().lower()
    training_mode = str(getattr(args, "training_mode", "") or "").strip().lower()
    classifier = str(getattr(args, "classifier", "") or "").strip().lower()
    fusion_mode = str(getattr(args, "fusion_mode", "") or "").strip().lower()
    if fusion_mode == "aligned_gated":
        alignment_dim = getattr(args, "stream_alignment_dim", None)
        if alignment_dim is None:
            alignment_dim = 256
        args.stream_alignment_dim = validate_stream_alignment_dim(alignment_dim)

    if raw_model in PAIR_INTERACTION_MODELS and not interaction_model:
        interaction_model = raw_model
    if raw_model == "contact" and not classifier:
        classifier = "mlp"
    if raw_model in GRAPH_PREDICTORS and not graph_predictor:
        graph_predictor = raw_model
    if raw_model in CLASSIFIERS and not classifier:
        classifier = raw_model
    if raw_model in SEQUENCE_MODELS and raw_model not in GRAPH_PREDICTORS and not sequence_model:
        sequence_model = raw_model

    if not pair_strategy:
        pair_strategy = _legacy_pair_strategy(args, raw_model)
    if pair_strategy not in PAIR_STRATEGIES:
        raise ValueError(f"unsupported pair_strategy: {pair_strategy}")

    if pair_strategy == "single_stream":
        if not sequence_model:
            sequence_model = "cnn"
        if sequence_model not in SEQUENCE_MODELS:
            raise ValueError("single_stream requires a sequence_model")
        if not training_mode:
            training_mode = EXPORT_EMBEDDINGS if classifier in CLASSICAL_CLASSIFIERS else END_TO_END
        if training_mode not in TRAINING_MODES:
            raise ValueError(f"unsupported training_mode: {training_mode}")
        if not classifier:
            classifier = "mlp"
        if classifier not in CLASSIFIERS:
            raise ValueError(f"unsupported classifier: {classifier}")
        training_mode, classifier = validate_training_mode_classifier(
            training_mode,
            classifier,
        )
        interaction_level = ""
        interaction_model = ""
        predictor_type = "classifier"
        graph_predictor = ""
        args.model = sequence_model
    else:
        if not interaction_level:
            interaction_level = TOKEN_INTERACTION if interaction_model else VECTOR_INTERACTION
        if interaction_level not in INTERACTION_LEVELS:
            raise ValueError(f"unsupported interaction_level: {interaction_level}")
        sequence_model = ""
        if interaction_level == TOKEN_INTERACTION:
            if not interaction_model:
                interaction_model = "cross_attn"
            if interaction_model not in PAIR_INTERACTION_MODELS:
                raise ValueError(f"unsupported interaction_model: {interaction_model}")
            if not training_mode:
                training_mode = (
                    EXPORT_EMBEDDINGS
                    if classifier in CLASSIFIERS and classifier != "mlp"
                    else END_TO_END
                )
            if training_mode not in TRAINING_MODES:
                raise ValueError(f"unsupported training_mode: {training_mode}")
            if not classifier:
                classifier = "mlp"
            if classifier not in CLASSIFIERS:
                raise ValueError(f"unsupported classifier: {classifier}")
            training_mode, classifier = validate_training_mode_classifier(
                training_mode,
                classifier,
            )
            predictor_type = "classifier"
            graph_predictor = ""
            args.model = f"{interaction_model}+mlp"
        else:
            interaction_model = ""
            training_mode = ""
            if not predictor_type:
                predictor_type = "sample_graph" if graph_predictor else "classifier"
            if predictor_type not in PREDICTOR_TYPES:
                raise ValueError(f"unsupported predictor_type: {predictor_type}")
            if predictor_type == "sample_graph":
                if not graph_predictor:
                    graph_predictor = "gcn"
                if graph_predictor not in GRAPH_PREDICTORS:
                    raise ValueError(f"unsupported graph_predictor: {graph_predictor}")
                classifier = ""
                args.model = graph_predictor
            else:
                if not classifier:
                    classifier = "mlp"
                if classifier not in CLASSIFIERS:
                    raise ValueError(f"unsupported classifier: {classifier}")
                graph_predictor = ""
                args.model = classifier

    args.pair_strategy = pair_strategy
    args.interaction_level = interaction_level
    args.interaction_model = interaction_model
    args.predictor_type = predictor_type
    args.graph_predictor = graph_predictor
    args.sequence_model = sequence_model
    args.training_mode = training_mode
    args.classifier = classifier
    if tokenizer in PRETRAINED_TOKENIZERS and pair_strategy == "single_stream":
        raw_adapter_dim = getattr(args, "residue_adapter_dim", RESIDUE_ADAPTER_DIM_DEFAULT)
        try:
            args.residue_adapter_dim = int(raw_adapter_dim)
        except (TypeError, ValueError) as exc:
            raise ValueError("residue_adapter_dim must be an integer") from exc
        if not RESIDUE_ADAPTER_DIM_MIN <= args.residue_adapter_dim <= RESIDUE_ADAPTER_DIM_MAX:
            raise ValueError(
                "residue_adapter_dim must be between "
                f"{RESIDUE_ADAPTER_DIM_MIN} and {RESIDUE_ADAPTER_DIM_MAX}"
            )
        if args.residue_adapter_dim % RESIDUE_ADAPTER_DIM_MULTIPLE != 0:
            raise ValueError(
                f"residue_adapter_dim must be divisible by {RESIDUE_ADAPTER_DIM_MULTIPLE}"
            )
    args.ppi_interaction_mode = "single" if pair_strategy == "single_stream" else "dual"
    args.hlaii_interaction_mode = "concat" if pair_strategy == "single_stream" else "contact"
    if not hasattr(args, "tsvd_dense_memory_limit_gib"):
        args.tsvd_dense_memory_limit_gib = TSVD_DEFAULT_DENSE_MEMORY_LIMIT_GIB
    if not hasattr(args, "pca_dense_memory_limit_gib"):
        args.pca_dense_memory_limit_gib = PCA_DEFAULT_DENSE_MEMORY_LIMIT_GIB
    if not hasattr(args, "kpca_memory_limit_gib"):
        args.kpca_memory_limit_gib = KPCA_DEFAULT_MEMORY_LIMIT_GIB
    normalize_ml_reduction_args(args)
    # Validate topic reducers after legacy model aliases have been normalized,
    # so the strict Pair route can inspect the resolved classifier/strategy.
    validate_lda_route(args, pair=True)
    validate_plsa_route(args, pair=True)
    return args


def uses_pair_interaction_model(args):
    return (
        str(getattr(args, "pair_strategy", "") or "").strip().lower() == "two_stream"
        and str(getattr(args, "interaction_level", "") or "").strip().lower() == TOKEN_INTERACTION
        and str(getattr(args, "interaction_model", "") or "").strip().lower()
        in PAIR_INTERACTION_MODELS
    )


def exports_interaction_embeddings(args):
    return (
        uses_pair_interaction_model(args)
        and str(getattr(args, "training_mode", "") or "").strip().lower() == EXPORT_EMBEDDINGS
    )


from pepplatform.stream_compat import install_legacy_exports as _install_legacy_stream_exports
_install_legacy_stream_exports(globals())
