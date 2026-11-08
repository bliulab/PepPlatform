from pathlib import Path

from pepplatform.covariance import (
    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
    COVARIANCE_ENCODER_CONTRACT_VERSION,
    COVARIANCE_MAX_RAW_LENGTH,
    GUO_PROPERTY_NAMES,
)
from pepplatform.esmc import official_esmc_model_options
from pepplatform.frequency import (
    FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
    FREQUENCY_ENCODER_CONTRACT_VERSION,
    FREQUENCY_MAX_RAW_LENGTH,
)
from pepplatform.kmer import KMER_MAX_K, KMER_MAX_LENGTH_UNIT, KMER_MIN_K
from pepplatform.lda import (
    LDA_CONTRACT_VERSION,
    LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    LDA_VARIANT,
)
from pepplatform.lsa import (
    LSA_CONTRACT_VERSION,
    LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    LSA_VARIANT,
)
from pepplatform.pair_terminology import PAIR_STRATEGY_ALIASES, PAIR_STRATEGY_LABELS
from pepplatform.pdt import (
    PDT_DEFAULT_LAG,
    PDT_DEFAULT_MAX_RAW_LENGTH,
    PDT_ENCODER_CONTRACT_VERSION,
    PDT_MAX_RAW_LENGTH,
    PDT_PROPERTY_COUNT,
    PDT_SOURCE_DOI,
)
from pepplatform.plsa import (
    PLSA_CONTRACT_VERSION,
    PLSA_DEFAULT_MEMORY_LIMIT_GIB,
    PLSA_VARIANT,
)
from pepplatform.prott5 import PROTT5_MAX_RESIDUES
from pepplatform.pssm_covariance import (
    HYBRID_PSSM_COVARIANCE_ENCODERS,
    PSSM_COVARIANCE_DEFAULT_LAG,
    PSSM_COVARIANCE_ENCODER_CONTRACT_VERSION,
    PSSM_COVARIANCE_LAG_KEYS,
    PSSM_COVARIANCE_PROFILE_FEATURE_TYPES,
)
from pepplatform.route_registry import route_capabilities
from pepplatform.task_compat import resolve_hlai_root
from pepplatform.unigram import (
    UNIGRAM_ABSOLUTE_MAX_RAW_LENGTH,
    UNIGRAM_DEFAULT_ALPHA,
    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    UNIGRAM_DEFAULT_NBEST_SIZE,
    UNIGRAM_MAX_LENGTH_UNIT,
    UNIGRAM_MAX_NBEST_SIZE,
)
from pepplatform.wordpiece import (
    WORDPIECE_ABSOLUTE_MAX_RAW_LENGTH,
    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
    WORDPIECE_MAX_LENGTH_UNIT,
)
from pepplatform.workflow import (
    AUTO_TUNABLE_CLASSIFIERS,
    CHAIN_GRAPH_ARCHITECTURES,
    CHAIN_GRAPH_COMPATIBLE_ENCODERS,
    CLASSIC_ENCODER_OPTIONS,
    CLASSIC_TOKENIZERS,
    CLASSIFIERS,
    FEATURE_TRANSFORMATIONS,
    GRAPH_PREDICTORS,
    INTERACTION_LEVELS,
    ML_TUNING_BUDGETS,
    ML_TUNING_SPACE_VERSION,
    PAIR_INTERACTION_MODELS,
    PAIR_STRATEGIES,
    PREDICTOR_TYPES,
    PRETRAINED_TOKENIZERS,
    RESIDUE_ADAPTER_DIM_DEFAULT,
    RESIDUE_ADAPTER_DIM_MAX,
    RESIDUE_ADAPTER_DIM_MIN,
    RESIDUE_ADAPTER_DIM_MULTIPLE,
    RESIDUE_PROFILE_ENCODERS,
    SEMANTIC_RELATION_TYPES,
    SEQUENCE_ARCHITECTURES,
    SEQUENCE_MODELS,
    SVM_CALIBRATION_CV_FOLDS,
    SVM_CALIBRATION_OPTIONS,
    TASK_SEMANTIC_RELATION_TYPE,
    TRAINING_MODES,
)

TASKS = [
    {"id": "pep_function", "label": "Therapeutic peptide function prediction"},
    {"id": "pep_protein_interaction", "label": "Peptide–protein interaction prediction"},
    {"id": "pep_hlaii", "label": "Peptide–HLA-II ligand presentation prediction"},
    {"id": "pep_hlai", "label": "Peptide–HLA-I ligand presentation prediction"},
]


def build_capabilities():
    from pepplatform.objectives import objective_capabilities

    return {
        "selection_objectives": objective_capabilities(),
        **route_capabilities(),
        "tasks": TASKS,
        "feature_routes": ["pretrained", "classic"],
        "pretrained_tokenizers": PRETRAINED_TOKENIZERS,
        "pretrained_models": {
            "esmc": official_esmc_model_options(),
        },
        "classic_tokenizers": CLASSIC_TOKENIZERS,
        "classic_encoder_options": CLASSIC_ENCODER_OPTIONS,
        "classical_tuning": {
            "supported_classifiers": list(AUTO_TUNABLE_CLASSIFIERS),
            "objectives": ["f1", "auc", "dual_f1_sacc"],
            "mode": "cv",
            "default_budget": "quick",
            "space_version": ML_TUNING_SPACE_VERSION,
            "semantics": "best_observed_candidate",
            "budgets": {
                name: {
                    "max_candidates": values["max_candidates"],
                    "cv_folds": values["cv_folds"],
                    "max_candidate_fold_evaluations": values["max_candidates"] * values["cv_folds"],
                }
                for name, values in ML_TUNING_BUDGETS.items()
            },
        },
        "sequence_profile_contract": {
            "version": 3,
            "strict_features": sorted(
                RESIDUE_PROFILE_ENCODERS
                - {
                    "pssm_blosum62_hybrid",
                    "psfm_blosum62_hybrid",
                    *HYBRID_PSSM_COVARIANCE_ENCODERS,
                }
            ),
            "explicit_hybrid_feature": "pssm_blosum62_hybrid",
            "explicit_hybrid_encoders": [
                "pssm_blosum62_hybrid",
                "psfm_blosum62_hybrid",
                "ac_pssm_blosum62_hybrid",
                "cc_pssm_blosum62_hybrid",
                "acc_pssm_blosum62_hybrid",
            ],
            "required_k": 1,
            "fallback_policies": {
                "pssm": "error",
                "psfm": "error",
                "pssm_blosum62_hybrid": "blosum62_hybrid",
                "psfm_blosum62_hybrid": "blosum62_hybrid",
                **{encoder: "blosum62_hybrid" for encoder in HYBRID_PSSM_COVARIANCE_ENCODERS},
            },
        },
        "frequency_encoder_contract": {
            "version": FREQUENCY_ENCODER_CONTRACT_VERSION,
            "encoders": ["bow", "tfidf"],
            "tokenizers": [
                "kmer",
                "dr",
                "top-n-gram",
                "dt",
            ],
            "max_raw_length": {
                "default": FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
                "absolute_max": FREQUENCY_MAX_RAW_LENGTH,
            },
            "stream_policy": "full_token_stream",
        },
        "covariance_encoder_contract": {
            "version": COVARIANCE_ENCODER_CONTRACT_VERSION,
            "encoders": ["ac", "cc", "acc"],
            "tokenizer": "kmer",
            "required_k": 1,
            "properties": list(GUO_PROPERTY_NAMES),
            "max_raw_length": {
                "default": COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
                "absolute_max": COVARIANCE_MAX_RAW_LENGTH,
            },
            "stream_policy": "complete_raw_sequence",
        },
        "pdt_encoder_contract": {
            "version": PDT_ENCODER_CONTRACT_VERSION,
            "encoders": ["pdt"],
            "tokenizer": "kmer",
            "required_k": 1,
            "default_lag": PDT_DEFAULT_LAG,
            "dimensions_per_lag": PDT_PROPERTY_COUNT,
            "method_source_doi": PDT_SOURCE_DOI,
            "max_raw_length": {
                "default": PDT_DEFAULT_MAX_RAW_LENGTH,
                "absolute_max": PDT_MAX_RAW_LENGTH,
            },
            "stream_policy": "complete_raw_sequence",
        },
        "pssm_covariance_encoder_contract": {
            "version": PSSM_COVARIANCE_ENCODER_CONTRACT_VERSION,
            "encoders": [
                "ac_pssm",
                "cc_pssm",
                "acc_pssm",
                "ac_pssm_blosum62_hybrid",
                "cc_pssm_blosum62_hybrid",
                "acc_pssm_blosum62_hybrid",
            ],
            "strict_encoders": ["ac_pssm", "cc_pssm", "acc_pssm"],
            "hybrid_encoders": [
                "ac_pssm_blosum62_hybrid",
                "cc_pssm_blosum62_hybrid",
                "acc_pssm_blosum62_hybrid",
            ],
            "tokenizer": "kmer",
            "required_k": 1,
            "default_lag": PSSM_COVARIANCE_DEFAULT_LAG,
            "dimensions_per_lag": {
                "ac_pssm": 20,
                "cc_pssm": 380,
                "acc_pssm": 400,
                "ac_pssm_blosum62_hybrid": 20,
                "cc_pssm_blosum62_hybrid": 380,
                "acc_pssm_blosum62_hybrid": 400,
            },
            "lag_keys": dict(sorted(PSSM_COVARIANCE_LAG_KEYS.items())),
            "profile_feature_types": dict(sorted(PSSM_COVARIANCE_PROFILE_FEATURE_TYPES.items())),
            "stream_policy": "complete_pssm_profile",
        },
        "tokenizer_constraints": {
            "kmer": {
                "k": {"min": KMER_MIN_K, "max": KMER_MAX_K},
                "max_length_unit": KMER_MAX_LENGTH_UNIT,
            },
            "wordpiece": {
                "max_length_unit": WORDPIECE_MAX_LENGTH_UNIT,
                "max_raw_length": {
                    "default": WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
                    "absolute_max": WORDPIECE_ABSOLUTE_MAX_RAW_LENGTH,
                },
            },
            "unigram": {
                "max_length_unit": UNIGRAM_MAX_LENGTH_UNIT,
                "max_raw_length": {
                    "default": UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
                    "absolute_max": UNIGRAM_ABSOLUTE_MAX_RAW_LENGTH,
                },
                "sampling": {
                    "default_enabled": False,
                    "default_nbest_size": UNIGRAM_DEFAULT_NBEST_SIZE,
                    "min_nbest_size": 2,
                    "max_nbest_size": UNIGRAM_MAX_NBEST_SIZE,
                    "default_alpha": UNIGRAM_DEFAULT_ALPHA,
                    "inference_mode": "viterbi",
                },
            },
            "prott5": {
                "max_length_unit": "residues",
                "max_residues": PROTT5_MAX_RESIDUES,
                "total_token_compatibility_adjustment": {
                    "requested_total_tokens": PROTT5_MAX_RESIDUES + 1,
                    "effective_residues": PROTT5_MAX_RESIDUES,
                    "reason": "T5 n_positions includes EOS",
                },
            },
        },
        "feature_transformations": FEATURE_TRANSFORMATIONS,
        "pair_feature_transformations": {
            encoder: [
                value
                for value in values
                if value != "lsa" and (encoder == "bow" or value not in {"lda", "plsa"})
            ]
            for encoder, values in FEATURE_TRANSFORMATIONS.items()
        },
        "lsa_contract": {
            "version": LSA_CONTRACT_VERSION,
            "variant": LSA_VARIANT,
            "route": "single_sequence_classic_global",
            "pair_supported": False,
            "dense_memory_limit_gib_default": LSA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
        },
        "lda_contract": {
            "version": LDA_CONTRACT_VERSION,
            "variant": LDA_VARIANT,
            "route": "single_sequence_classic_global_bow_v2",
            "pair_route": "pair_two_stream_classic_bow_v2",
            "pair_supported": True,
            "dense_memory_limit_gib_default": LDA_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
        },
        "plsa_contract": {
            "version": PLSA_CONTRACT_VERSION,
            "variant": PLSA_VARIANT,
            "route": "single_sequence_classic_global_bow_v2",
            "pair_route": "pair_two_stream_classic_bow_v2",
            "pair_supported": True,
            "memory_limit_gib_default": PLSA_DEFAULT_MEMORY_LIMIT_GIB,
        },
        "sequence_models": SEQUENCE_MODELS,
        "modeling_architectures": {
            "sequence": SEQUENCE_ARCHITECTURES,
            "chain_graph": CHAIN_GRAPH_ARCHITECTURES,
        },
        "classifiers": CLASSIFIERS,
        "svm_calibration": {
            "options": SVM_CALIBRATION_OPTIONS,
            "default": "none",
            "cv_folds": SVM_CALIBRATION_CV_FOLDS,
        },
        "pair_strategies": PAIR_STRATEGIES,
        "pair_strategy_aliases": dict(PAIR_STRATEGY_ALIASES),
        "pair_strategy_labels": {key: dict(value) for key, value in PAIR_STRATEGY_LABELS.items()},
        "interaction_levels": INTERACTION_LEVELS,
        "interaction_models": PAIR_INTERACTION_MODELS,
        "semantic_relation_types": SEMANTIC_RELATION_TYPES,
        "task_semantic_relation_type": TASK_SEMANTIC_RELATION_TYPE,
        "semantic_defaults": {
            "semantic_similarity_weight": 0.1,
            "semantic_zero_weight": 0.25,
            "semantic_positive_partners": 1,
            "semantic_zero_partners": 8,
            # Preserve historical response keys; new controls use CLI names.
            "positive_partners_per_anchor": 1,
            "zero_partners_per_anchor": 1,
        },
        "predictor_types": PREDICTOR_TYPES,
        "graph_predictors": GRAPH_PREDICTORS,
        "training_modes": TRAINING_MODES,
        "route_constraints": {
            "single_sequence": {
                "classic": {
                    "chain_graph_encoders": CHAIN_GRAPH_COMPATIBLE_ENCODERS,
                },
                "pretrained": {
                    "pooled_predictors": CLASSIFIERS,
                    "residue_architectures": {
                        tokenizer: list(SEQUENCE_MODELS) for tokenizer in PRETRAINED_TOKENIZERS
                    },
                    "residue_training_modes": {
                        tokenizer: list(TRAINING_MODES) for tokenizer in PRETRAINED_TOKENIZERS
                    },
                    "residue_adapter_dim": {
                        "default": RESIDUE_ADAPTER_DIM_DEFAULT,
                        "min": RESIDUE_ADAPTER_DIM_MIN,
                        "max": RESIDUE_ADAPTER_DIM_MAX,
                        "multiple_of": RESIDUE_ADAPTER_DIM_MULTIPLE,
                    },
                    "residue_encoder_frozen": True,
                    "residue_graph_predictors": {
                        tokenizer: list(CHAIN_GRAPH_ARCHITECTURES)
                        for tokenizer in PRETRAINED_TOKENIZERS
                    },
                    "residue_graph_training_modes": {
                        tokenizer: list(TRAINING_MODES) for tokenizer in PRETRAINED_TOKENIZERS
                    },
                    "residue_graph_prediction_head": "configurable",
                    "unsupported_post_encoders": [],
                },
            },
            "pair": {
                "pretrained_token_interaction": {
                    "pair_strategy": "two_stream",
                    "interaction_level": "token",
                    "interaction_models": PAIR_INTERACTION_MODELS,
                    "classifier": "mlp",
                },
            },
        },
    }


def builtin_dataset_specs(project_root):
    project_root = Path(project_root)
    pep_function_root = project_root / "datasets" / "task_peptide_function"
    pep_function_quick = pep_function_root / "data_quick_384"
    hlaii_root = project_root / "datasets" / "task_peptide_HLAII_interaction"
    hlaii_data = hlaii_root / "data"
    hlaii_quick = hlaii_root / "data_quick_50k"
    ppi_root = project_root / "datasets" / "task_peptide_protein_interaction"
    ppi_data = ppi_root / "data"
    ppi_quick = ppi_root / "data_quick_2000"
    hlai_root = resolve_hlai_root(project_root)
    hlai_data = hlai_root / "data_bigmhc_el_100k_original_split"
    return [
        {
            "id": "builtin-pep-function-quick",
            "name": "Built-in peptide function quick dataset",
            "task_type": "pep_function",
            "format": "sequence_dir",
            "root_path": pep_function_quick,
        },
        {
            "id": "builtin-pep-function",
            "name": "Built-in peptide function full dataset",
            "task_type": "pep_function",
            "format": "sequence_dir",
            "root_path": pep_function_root / "data",
        },
        {
            "id": "builtin-tppred-le-benchmark",
            "name": "TPpred-LE benchmark",
            "task_type": "pep_function",
            "format": "sequence_dir",
            "root_path": pep_function_root / "data_tppred_benchmark_public",
        },
        {
            "id": "builtin-pep-hlaii",
            "name": "Built-in peptide HLA-II quick dataset",
            "task_type": "pep_hlaii",
            "format": "pair_split_dir",
            "root_path": hlaii_quick,
        },
        {
            "id": "builtin-pep-hlaii-full",
            "name": "Built-in peptide HLA-II full dataset",
            "task_type": "pep_hlaii",
            "format": "pair_split_dir",
            "root_path": hlaii_data,
        },
        {
            "id": "builtin-ppi",
            "name": "Built-in peptide-protein interaction dataset",
            "task_type": "pep_protein_interaction",
            "format": "pair_split_dir" if ppi_data.exists() else "pair_root",
            "root_path": ppi_data if ppi_data.exists() else ppi_root,
        },
        {
            "id": "builtin-ppi-quick",
            "name": "Built-in peptide-protein interaction quick dataset",
            "task_type": "pep_protein_interaction",
            "format": "pair_split_dir",
            "root_path": ppi_quick,
            "feature_dir": ppi_quick / "features",
        },
        {
            "id": "builtin-hlai",
            "name": "Built-in peptide HLA-I BigMHC-EL 100k dataset",
            "task_type": "pep_hlai",
            "format": "pair_split_dir",
            "root_path": hlai_data,
            "feature_dir": hlai_data / "features",
        },
    ]
