from pepplatform.stream_compat import normalize_stream_fields
from pepplatform.task_compat import normalize_task_type

from argparse import Namespace

from pepplatform.workflow import (
    CLASSICAL_CLASSIFIERS,
    CLASSICAL_TRAINING_DEFAULTS,
    CLASSIFIER_PARAMETER_DEFAULTS,
    GRAPH_PREDICTORS,
    LGB_EXECUTION_POLICY,
    ML_TUNING_BUDGETS,
    PAIR_CLASSIFIER_PARAMETER_DEFAULTS,
    PAIR_NEURAL_DEFAULTS,
    PAIR_PROFILE_DEFAULTS,
    PRETRAINED_TOKENIZERS,
    PROFILE_DEPENDENT_TOKENIZERS,
    RESIDUE_PROFILE_ENCODERS,
    RESIDUE_ADAPTER_DIM_DEFAULT,
    SEQUENCE_MODELS,
    SEQUENCE_NEURAL_DEFAULTS,
    SEQUENCE_PROFILE_DEFAULTS,
    default_model_selection_objective,
    normalize_prediction_head_config,
    normalize_pair_workflow_args,
    normalize_workflow_args,
    uses_profile_features,
)
from pepplatform.route_registry import canonicalize_encoder_config, legacy_encoder_messages
from pepplatform.covariance import COVARIANCE_ENCODERS
from pepplatform.pdt import PDT_ENCODERS
from pepplatform.pssm_covariance import HYBRID_PSSM_COVARIANCE_ENCODERS
from pepplatform.top_ngram import TOP_N_GRAM_FALLBACK_POLICY
from pepplatform.stream_spec import canonical_pair_config


DEFAULTS = {
    "pep_function": {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
    "pep_hlaii": {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
    "pep_protein_interaction": {"tokenizer": "esm2", "encoder": "esm2", "model": "mlp"},
    "pep_hlai": {"tokenizer": "esm2", "encoder": "esm2", "model": "mlp"},
}


def normalize_experiment_config(task_type, raw_config):
    task_type = normalize_task_type(task_type)
    config, _sources = normalize_experiment_config_with_sources(task_type, raw_config)
    return config


def _is_neural_route(task_type, config):
    task_type = normalize_task_type(task_type)
    sequence_model = str(config.get("sequence_model", "") or "").strip().lower()
    classifier = str(config.get("classifier", "") or config.get("model", "") or "").strip().lower()
    graph_predictor = str(config.get("graph_predictor", "") or "").strip().lower()
    predictor_type = str(config.get("predictor_type", "") or "").strip().lower()
    interaction_level = str(config.get("interaction_level", "") or "").strip().lower()
    if sequence_model in SEQUENCE_MODELS:
        return True
    if normalize_task_type(task_type) != "pep_function" and (
        graph_predictor in GRAPH_PREDICTORS
        or predictor_type == "sample_graph"
        or interaction_level == "token"
    ):
        return True
    return classifier == "mlp"


def _set_defaults(config, sources, defaults):
    for key, value in defaults.items():
        if config.get(key) in [None, ""]:
            config[key] = value
            sources[key] = "default"


def normalize_experiment_config_with_sources(task_type, raw_config):
    task_type = normalize_task_type(task_type)
    requested = normalize_stream_fields(dict(raw_config or {}))
    config = dict(DEFAULTS.get(task_type, {}))
    config.update(requested)
    config["task_type"] = task_type
    config = canonicalize_encoder_config(config)
    args = Namespace(**config)
    if normalize_task_type(task_type) in {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}:
        normalize_pair_workflow_args(args)
    else:
        normalize_workflow_args(args)
    normalized = dict(vars(args))
    if normalize_task_type(task_type) in {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}:
        normalized = canonical_pair_config(normalized, task_type=task_type)
    normalized = canonicalize_encoder_config(normalized)
    tokenizer = str(normalized.get("tokenizer", "") or "").strip().lower()
    encoder = str(normalized.get("encoder", "") or "").strip().lower()
    if (
        tokenizer == "kmer"
        and encoder in RESIDUE_PROFILE_ENCODERS.union(COVARIANCE_ENCODERS, PDT_ENCODERS)
        and normalized.get("k") in [None, ""]
    ):
        normalized["k"] = 1
    streams = normalized.get("streams")
    if isinstance(streams, dict):
        normalized_streams = {}
        for role, raw_stream in streams.items():
            if not isinstance(raw_stream, dict):
                normalized_streams[role] = raw_stream
                continue
            stream = dict(raw_stream)
            params = dict(stream.get("params")) if isinstance(stream.get("params"), dict) else {}
            if (
                str(stream.get("tokenizer", "") or "").strip().lower() == "kmer"
                and str(stream.get("encoder", "") or "").strip().lower()
                in RESIDUE_PROFILE_ENCODERS.union(
                    COVARIANCE_ENCODERS,
                    PDT_ENCODERS,
                )
                and params.get("k") in [None, ""]
            ):
                params["k"] = 1
            stream["params"] = params
            normalized_streams[role] = stream
        normalized["streams"] = normalized_streams
    for legacy_key in ["probert_model", "probert_feature_dir"]:
        normalized.pop(legacy_key, None)
    sources = {}
    task_defaults = DEFAULTS.get(task_type, {})
    for key in normalized:
        if key in requested and requested[key] not in [None, ""]:
            sources[key] = "explicit"
        elif key in task_defaults or key == "task_type":
            sources[key] = "default"
        else:
            sources[key] = "derived"
    if legacy_encoder_messages(requested):
        raw_encoder = str(requested.get("encoder", "") or "").strip().lower()
        if raw_encoder in {"index", "ppi_fusion"}:
            sources["encoder"] = "legacy_derived"

    if requested.get("seed") in [None, ""]:
        normalized["seed"] = 42
        sources["seed"] = "default"

    is_pair = normalize_task_type(task_type) in {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}
    if _is_neural_route(task_type, normalized):
        _set_defaults(
            normalized,
            sources,
            PAIR_NEURAL_DEFAULTS if is_pair else SEQUENCE_NEURAL_DEFAULTS,
        )
    if (
        not is_pair
        and str(normalized.get("tokenizer", "")).lower() in PRETRAINED_TOKENIZERS
        and str(normalized.get("sequence_model", "")).lower() in SEQUENCE_MODELS
    ):
        _set_defaults(
            normalized,
            sources,
            {"residue_adapter_dim": RESIDUE_ADAPTER_DIM_DEFAULT},
        )

    normalize_prediction_head_config(normalized)
    for key in ["head_hidden_dims", "head_activation", "head_dropout"]:
        if key not in normalized:
            sources.pop(key, None)
        elif key in requested and requested[key] not in [None, ""]:
            sources[key] = "explicit"
        else:
            sources[key] = "default"

    classifier = (
        str(normalized.get("classifier", "") or normalized.get("model", "") or "").strip().lower()
    )
    if classifier in CLASSICAL_CLASSIFIERS:
        _set_defaults(
            normalized,
            sources,
            {
                **CLASSICAL_TRAINING_DEFAULTS,
                "ml_tuning_objective": requested.get("tuning_objective")
                or default_model_selection_objective(task_type),
            },
        )
        classifier_defaults = (
            PAIR_CLASSIFIER_PARAMETER_DEFAULTS if is_pair else CLASSIFIER_PARAMETER_DEFAULTS
        )
        _set_defaults(normalized, sources, classifier_defaults.get(classifier, {}))
        if classifier == "lgb":
            normalized.update(LGB_EXECUTION_POLICY)
            sources.update({key: "derived" for key in LGB_EXECUTION_POLICY})
        requested_budget = str(requested.get("ml_tuning_budget", "") or "").strip().lower()
        if requested_budget in ML_TUNING_BUDGETS:
            budget = ML_TUNING_BUDGETS[requested_budget]
            normalized["ml_tuning_budget"] = requested_budget
            normalized["ml_tuning_mode"] = "cv"
            normalized["ml_tuning_max_candidates"] = budget["max_candidates"]
            normalized["ml_cv_folds"] = budget["cv_folds"]
            for key in [
                "ml_tuning_budget",
                "ml_tuning_mode",
                "ml_tuning_max_candidates",
                "ml_cv_folds",
            ]:
                sources[key] = "explicit" if key == "ml_tuning_budget" else "derived"

    if uses_profile_features(normalized):
        profile_defaults = dict(PAIR_PROFILE_DEFAULTS if is_pair else SEQUENCE_PROFILE_DEFAULTS)
        streams = normalized.get("streams") if isinstance(normalized.get("streams"), dict) else {}
        uses_profile_tokenizer = str(
            normalized.get("tokenizer", "")
        ).lower() in PROFILE_DEPENDENT_TOKENIZERS or any(
            str(stream.get("tokenizer", "")).lower() in PROFILE_DEPENDENT_TOKENIZERS
            for stream in streams.values()
            if isinstance(stream, dict)
        )
        if uses_profile_tokenizer:
            profile_defaults["profile_on_missing"] = TOP_N_GRAM_FALLBACK_POLICY
        uses_hybrid_encoder = str(normalized.get("encoder", "")).lower() in {
            "pssm_blosum62_hybrid",
            "psfm_blosum62_hybrid",
            *HYBRID_PSSM_COVARIANCE_ENCODERS,
        } or any(
            str(stream.get("encoder", "")).lower()
            in {
                "pssm_blosum62_hybrid",
                "psfm_blosum62_hybrid",
                *HYBRID_PSSM_COVARIANCE_ENCODERS,
            }
            for stream in streams.values()
            if isinstance(stream, dict)
        )
        if uses_hybrid_encoder:
            profile_defaults["profile_on_missing"] = "blosum62_hybrid"
        _set_defaults(
            normalized,
            sources,
            profile_defaults,
        )

    _set_defaults(
        normalized,
        sources,
        {
            "checkpoint_objective": "auc" if is_pair else "macro_f1",
            "threshold_objective": "f1" if is_pair else "macro_f1",
        },
    )
    return normalized, sources
