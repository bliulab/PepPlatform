from pepplatform.task_compat import (
    REMOVED_SEMANTIC_MODES,
    REMOVED_SEMANTIC_FIELDS,
    normalize_task_type,
    install_legacy_exports,
)

import math
import os
from pathlib import Path

from pepplatform.esmc import (
    ESMC_BACKEND_OFFICIAL,
    esmc_backend_for_source,
    esmc_variant_for_source,
    inspect_esmc_runtime,
    official_esmc_weight_path,
)
from pepplatform.kmer import validate_kmer_k
from pepplatform.lsa import (
    LSAContractError,
    validate_lsa_dense_memory_limit_gib,
    validate_lsa_route,
)
from pepplatform.tsvd import (
    TSVDContractError,
    validate_tsvd_dense_memory_limit_gib,
)
from pepplatform.pca import (
    PCAContractError,
    validate_pca_dense_memory_limit_gib,
)
from pepplatform.kpca import (
    KPCAContractError,
    KPCA_KERNELS,
    validate_kpca_gamma,
    validate_kpca_kernel,
    validate_kpca_memory_limit_gib,
)
from pepplatform.lda import (
    LDAContractError,
    validate_lda_dense_memory_limit_gib,
    validate_lda_route,
)
from pepplatform.plsa import (
    PLSAContractError,
    validate_plsa_memory_limit_gib,
    validate_plsa_route,
)
from pepplatform.frequency import validate_frequency_max_raw_length
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
from pepplatform.pssm_covariance import (
    HYBRID_PSSM_COVARIANCE_ENCODERS,
    PSSM_COVARIANCE_DEFAULT_LAG,
    PSSM_COVARIANCE_ENCODERS,
    STRICT_PSSM_COVARIANCE_ENCODERS,
    validate_pssm_covariance_lag,
)
from pepplatform.sequence_lengths import (
    dataset_role_max,
    dataset_role_min,
    length_stats_are_current,
    scan_dataset_length_stats,
)
from pepplatform.dr import (
    DRContractError,
    validate_dr_dense_memory_limit_gib,
    validate_dr_max_distance,
    validate_dr_max_raw_length,
    validate_dr_unknown_residue_policy,
)
from pepplatform.dt import (
    DTContractError,
    validate_dt_dense_memory_limit_gib,
    validate_dt_max_distance,
)
from pepplatform.top_ngram import (
    TOP_N_GRAM_FALLBACK_POLICY,
    TopNGramContractError,
    resolve_top_ngram_max_length,
    validate_top_ngram_dense_memory_limit_gib,
    validate_top_ngram_n,
)
from pepplatform.route_registry import legacy_encoder_messages, resolve_effective_route
from pepplatform.pair_fusion import PairFusionError, normalize_pair_fusion_mode
from pepplatform.prott5 import PROTT5_MAX_RESIDUES, validate_prott5_max_residues
from pepplatform.workflow import (
    CHAIN_GRAPH_ARCHITECTURES,
    CHAIN_GRAPH_COMPATIBLE_ENCODERS,
    CLASSIFIERS,
    CLASSICAL_CLASSIFIERS,
    END_TO_END,
    EXPORT_EMBEDDINGS,
    GRAPH_PREDICTORS,
    ML_TUNING_BUDGETS,
    ML_TUNING_MAX_CANDIDATES,
    PROFILE_DEPENDENT_INFERENCE_REASON,
    PROFILE_GENERATED_PAIR_INFERENCE_REASON,
    PRETRAINED_TOKENIZERS,
    SVM_CALIBRATION_CV_FOLDS,
    SVM_CALIBRATION_OPTIONS,
    RESIDUE_PROFILE_ENCODERS,
    RESIDUE_ADAPTER_DIM_MAX,
    RESIDUE_ADAPTER_DIM_MIN,
    RESIDUE_ADAPTER_DIM_MULTIPLE,
    SEQUENCE_MODELS,
    encoder_family,
    pretrained_model_local_candidates,
    pretrained_model_source,
    resolve_cached_pretrained_model_source,
    supports_classic_pair_token_inference,
    supports_pair_token_inference,
    supports_classic_pair_vector_inference,
    TrainingModeContractError,
    uses_profile_features,
    validate_index_route_semantics,
    validate_linear_prediction_head_semantics,
)
from pepplatform.esm3 import is_complete_esm3_source

from .config import normalize_experiment_config_with_sources
from .classification_losses import validate_classification_loss_config


ALLOWED_DATASET_FORMATS = {
    "pep_function": {"sequence_dir"},
    "pep_hlaii": {"pair_split_dir", "sequence_dir"},
    "pep_protein_interaction": {"pair_split_dir", "pair_root", "pair_csv_bundle"},
    "pep_hlai": {"pair_split_dir", "pair_root", "pair_csv_bundle"},
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAIR_REUSABLE_INFERENCE_TASKS = {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}
ROUTE_INFERENCE_UNAVAILABLE_REASON = (
    "this route will be recorded in the model library, but reusable inference is not yet available"
)
HLAI_ROUTE_INFERENCE_UNAVAILABLE_REASON = (
    "this pep_hlai route does not yet emit a reusable Web inference bundle"
)
SEQUENCE_EXPORT_INFERENCE_UNAVAILABLE_REASON = (
    "export_embeddings routes do not yet emit reusable single-sequence Web inference bundles"
)
SCALED_SEQUENCE_INFERENCE_UNAVAILABLE_REASON = "this sequence model route uses dl_scaler preprocessing that is not yet serialized for Web inference"


def _offline_mode_enabled():
    return str(os.environ.get("PEPPLATFORM_OFFLINE_MODE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _hf_endpoint():
    return (
        str(os.environ.get("HF_ENDPOINT", "")).strip()
        or str(os.environ.get("PEPPLATFORM_HF_ENDPOINT", "")).strip()
        or "https://hf-mirror.com"
    )


def _resolve_local_model_source(tokenizer, source):
    if not source:
        return ""
    expanded = Path(os.path.expanduser(source))
    if expanded.is_absolute():
        if not expanded.exists():
            return ""
        if tokenizer == "esm3" and not is_complete_esm3_source(expanded):
            return ""
        return str(expanded)
    relative = PROJECT_ROOT / expanded
    if relative.exists():
        if tokenizer == "esm3" and not is_complete_esm3_source(relative):
            return ""
        return str(relative)
    for candidate in pretrained_model_local_candidates(PROJECT_ROOT, tokenizer, source):
        if not Path(candidate).exists():
            continue
        if tokenizer == "esm3" and not is_complete_esm3_source(candidate):
            continue
        if tokenizer == "esmc":
            official_variant = esmc_variant_for_source(source)
            if (
                official_variant is not None
                and official_esmc_weight_path(
                    candidate,
                    model_id=str(official_variant[1]["model_id"]),
                )
                is None
            ):
                continue
        return candidate
    return resolve_cached_pretrained_model_source(source, tokenizer)


def _pair_streams(config):
    streams = config.get("streams") if isinstance(config, dict) else None
    if not isinstance(streams, dict):
        return {}
    if not all(isinstance(streams.get(role), dict) for role in ("peptide", "partner")):
        return {}
    return {role: dict(streams[role]) for role in ("peptide", "partner")}


def _dataset_length_stats(dataset):
    metadata = dataset.get("metadata") or {}
    length_stats = metadata.get("length_stats")
    if length_stats_are_current(length_stats):
        return length_stats
    root_path = str(dataset.get("root_path") or "").strip()
    if not root_path:
        return None
    try:
        return scan_dataset_length_stats(
            root_path,
            dataset.get("format", ""),
            source_root=metadata.get("source_root"),
        )
    except OSError:
        return None


def _resolve_top_ngram_dataset_lengths(
    normalized,
    dataset,
    parameter_sources,
    add_error,
    add_warning,
):
    length_stats = _dataset_length_stats(dataset)
    task_type = str(normalized.get("task_type") or dataset.get("task_type") or "")
    tokenizer = str(normalized.get("tokenizer", "") or "").strip().lower()
    streams = _pair_streams(normalized)
    active_roles = [
        role
        for role, stream in streams.items()
        if str(stream.get("tokenizer", "") or "").strip().lower() in {"top-n-gram", "dt"}
    ]
    uses_profile_tokenizer = tokenizer in {"top-n-gram", "dt"} or bool(active_roles)
    if not uses_profile_tokenizer:
        return
    active_tokenizers = {
        str(streams[role].get("tokenizer", "") or "").strip().lower() for role in active_roles
    }
    if tokenizer in {"top-n-gram", "dt"}:
        active_tokenizers.add(tokenizer)
    profile_label = (
        "DT"
        if active_tokenizers == {"dt"}
        else ("Top-n-gram" if active_tokenizers == {"top-n-gram"} else "Profile tokenizer")
    )
    code_prefix = "dt" if active_tokenizers == {"dt"} else "top_ngram"
    if str(normalized.get("profile_on_missing", "")).strip().lower() == TOP_N_GRAM_FALLBACK_POLICY:
        add_warning(
            f"{code_prefix}_profile_fallback_enabled",
            f"{profile_label} will use an explicit BLOSUM62 target-frequency prior "
            "when PSI-BLAST cannot produce a usable target-frequency profile",
        )
    if not length_stats_are_current(length_stats):
        add_error(
            f"{code_prefix}_length_stats_unavailable",
            f"{profile_label} requires dataset sequence-length statistics",
        )
        return

    if normalize_task_type(task_type) == "pep_function":
        required = dataset_role_max(length_stats, "sequence")
        if required < 1:
            add_error(
                f"{code_prefix}_length_stats_unavailable",
                f"{profile_label} requires a positive sequence-length maximum",
            )
            return
        requested = normalized.get("max_length", 60)
        try:
            effective, adjusted = resolve_top_ngram_max_length(
                requested,
                required,
                context=f"{profile_label} dataset sequences",
            )
        except TopNGramContractError as exc:
            add_error(exc.code, str(exc))
            return
        normalized["max_length"] = effective
        if adjusted:
            parameter_sources["max_length"] = "dataset_derived"
            add_warning(
                f"{code_prefix}_max_length_auto_adjusted",
                f"{profile_label} max_length was increased from {requested} to "
                f"{effective} to cover the full dataset",
            )
        return

    if not active_roles:
        add_error(
            f"{code_prefix}_length_stats_unavailable",
            f"{profile_label} pair configuration is missing role-aware streams",
        )
        return
    homogeneous = str(normalized.get("stream_mode", "") or "") == "homogeneous"
    if homogeneous:
        required = max(dataset_role_max(length_stats, role) for role in active_roles)
        requested = max(int(streams[role].get("max_length", 512)) for role in active_roles)
        try:
            effective, adjusted = resolve_top_ngram_max_length(
                requested,
                required,
                context=f"{profile_label} homogeneous pair streams",
            )
        except TopNGramContractError as exc:
            add_error(exc.code, str(exc))
            return
        for role in active_roles:
            streams[role]["max_length"] = effective
            if adjusted:
                parameter_sources[f"streams.{role}.max_length"] = "dataset_derived"
        normalized["max_length"] = effective
        if adjusted:
            parameter_sources["max_length"] = "dataset_derived"
            add_warning(
                f"{code_prefix}_max_length_auto_adjusted",
                f"{profile_label} homogeneous stream max_length was increased from "
                f"{requested} to {effective} to cover peptide and partner data",
            )
    else:
        for role in active_roles:
            required = dataset_role_max(length_stats, role)
            if required < 1:
                add_error(
                    f"{code_prefix}_length_stats_unavailable",
                    f"{profile_label} requires {role} sequence-length statistics",
                )
                continue
            requested = streams[role].get("max_length", normalized.get("max_length", 512))
            try:
                effective, adjusted = resolve_top_ngram_max_length(
                    requested,
                    required,
                    context=f"{profile_label} {role} stream",
                )
            except TopNGramContractError as exc:
                add_error(exc.code, str(exc))
                continue
            streams[role]["max_length"] = effective
            if adjusted:
                parameter_sources[f"streams.{role}.max_length"] = "dataset_derived"
                add_warning(
                    f"{code_prefix}_max_length_auto_adjusted",
                    f"{profile_label} {role} stream max_length was increased from "
                    f"{requested} to {effective} to cover the full dataset",
                )
    normalized["streams"] = streams


def _validate_prott5_lengths(normalized, add_error, add_warning):
    """Apply the explicit 512-token -> 511-residue compatibility adjustment."""

    adjustment = normalized.get("prott5_max_length_adjustment")
    if isinstance(adjustment, dict) and adjustment.get("requested") == PROTT5_MAX_RESIDUES + 1:
        add_warning(
            "prott5_max_length_adjusted",
            "ProtT5 max_length=512 describes total tokens; using "
            f"{PROTT5_MAX_RESIDUES} residues plus EOS",
        )

    def adjust(container, key, label):
        if not isinstance(container, dict):
            return
        raw = container.get(key)
        if raw in (None, ""):
            return
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return
        tokenizer = str(container.get("tokenizer", "") or "").strip().lower()
        if tokenizer != "prott5":
            return
        if container.get("max_length_requested") == PROTT5_MAX_RESIDUES + 1:
            add_warning(
                "prott5_max_length_adjusted",
                f"{label} max_length=512 describes total tokens; using "
                f"{PROTT5_MAX_RESIDUES} residues plus EOS",
            )
            return
        if value == PROTT5_MAX_RESIDUES + 1:
            container[key] = PROTT5_MAX_RESIDUES
            container.setdefault("max_length_requested", value)
            container["max_length_adjustment"] = {
                "requested": value,
                "effective": PROTT5_MAX_RESIDUES,
                "reason": "T5 n_positions includes the EOS token",
            }
            add_warning(
                "prott5_max_length_adjusted",
                f"{label} max_length=512 describes total tokens; using "
                f"{PROTT5_MAX_RESIDUES} residues plus EOS",
            )
            return
        try:
            validate_prott5_max_residues(value, field=f"{label} max_length")
        except ValueError as exc:
            add_error("prott5_max_length_invalid", str(exc))

    adjust(normalized, "max_length", "ProtT5")
    streams = normalized.get("streams")
    if isinstance(streams, dict):
        for role, stream in streams.items():
            adjust(stream, "max_length", f"ProtT5 {role} stream")


def _validate_covariance_dataset_lengths(normalized, dataset, add_error):
    task_type = str(normalized.get("task_type") or dataset.get("task_type") or "")
    active = []
    if normalize_task_type(task_type) == "pep_function":
        encoder = str(normalized.get("encoder", "") or "").strip().lower()
        if encoder in COVARIANCE_ENCODERS:
            active.append(("sequence", encoder, normalized))
    else:
        for role, stream in _pair_streams(normalized).items():
            encoder = str(stream.get("encoder", "") or "").strip().lower()
            if encoder in COVARIANCE_ENCODERS:
                active.append((role, encoder, _stream_flat_config(normalized, stream)))

    if not active:
        return

    length_stats = _dataset_length_stats(dataset)
    if not length_stats_are_current(length_stats):
        add_error(
            "covariance_length_stats_unavailable",
            "AC/CC/ACC requires dataset sequence-length statistics",
        )
        return

    for role, encoder, config in active:
        minimum = dataset_role_min(length_stats, role)
        maximum = dataset_role_max(length_stats, role)
        if minimum < 1 or maximum < 1:
            add_error(
                "covariance_length_stats_unavailable",
                f"{encoder.upper()} requires positive {role} sequence-length statistics",
            )
            continue

        lag_key = f"{encoder}_lag"
        try:
            lag = validate_covariance_lag(config.get(lag_key, 10))
            max_raw_length = validate_covariance_max_raw_length(
                config.get(
                    "covariance_max_raw_length",
                    COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
                )
            )
        except ValueError:
            # Numeric validation reports the user-facing parameter error.
            continue

        if lag >= minimum:
            add_error(
                "covariance_lag_too_large",
                f"{encoder.upper()} {role} stream requires {lag_key} < dataset "
                f"minimum sequence length {minimum}, got {lag}",
            )
        if max_raw_length < maximum:
            add_error(
                "covariance_raw_length_exceeded",
                f"{encoder.upper()} {role} stream requires covariance_max_raw_length "
                f">= dataset maximum sequence length {maximum}, got {max_raw_length}",
            )


def _validate_pdt_dataset_lengths(normalized, dataset, add_error):
    task_type = str(normalized.get("task_type") or dataset.get("task_type") or "")
    active = []
    if normalize_task_type(task_type) == "pep_function":
        encoder = str(normalized.get("encoder", "") or "").strip().lower()
        if encoder in PDT_ENCODERS:
            active.append(("sequence", normalized))
    else:
        for role, stream in _pair_streams(normalized).items():
            encoder = str(stream.get("encoder", "") or "").strip().lower()
            if encoder in PDT_ENCODERS:
                active.append((role, _stream_flat_config(normalized, stream)))

    if not active:
        return

    length_stats = _dataset_length_stats(dataset)
    if not length_stats_are_current(length_stats):
        add_error(
            "pdt_length_stats_unavailable",
            "PDT requires dataset sequence-length statistics",
        )
        return

    for role, config in active:
        minimum = dataset_role_min(length_stats, role)
        maximum = dataset_role_max(length_stats, role)
        if minimum < 1 or maximum < 1:
            add_error(
                "pdt_length_stats_unavailable",
                f"PDT requires positive {role} sequence-length statistics",
            )
            continue
        try:
            lag = validate_pdt_lag(config.get("pdt_lag", PDT_DEFAULT_LAG))
            max_raw_length = validate_pdt_max_raw_length(
                config.get(
                    "pdt_max_raw_length",
                    PDT_DEFAULT_MAX_RAW_LENGTH,
                )
            )
        except ValueError:
            continue
        if lag >= minimum:
            add_error(
                "pdt_lag_too_large",
                f"PDT {role} stream requires pdt_lag < dataset minimum "
                f"sequence length {minimum}, got {lag}",
            )
        if max_raw_length < maximum:
            add_error(
                "pdt_raw_length_exceeded",
                f"PDT {role} stream requires pdt_max_raw_length >= dataset "
                f"maximum sequence length {maximum}, got {max_raw_length}",
            )


def _validate_pssm_covariance_dataset_lengths(
    normalized,
    dataset,
    add_error,
):
    task_type = str(normalized.get("task_type") or dataset.get("task_type") or "")
    active = []
    if normalize_task_type(task_type) == "pep_function":
        encoder = str(normalized.get("encoder", "") or "").strip().lower()
        if encoder in PSSM_COVARIANCE_ENCODERS:
            active.append(("sequence", encoder, normalized))
    else:
        for role, stream in _pair_streams(normalized).items():
            encoder = str(stream.get("encoder", "") or "").strip().lower()
            if encoder in PSSM_COVARIANCE_ENCODERS:
                active.append((role, encoder, _stream_flat_config(normalized, stream)))

    if not active:
        return

    length_stats = _dataset_length_stats(dataset)
    if not length_stats_are_current(length_stats):
        add_error(
            "pssm_covariance_length_stats_unavailable",
            "AC-PSSM/CC-PSSM/ACC-PSSM requires current dataset sequence-length statistics",
        )
        return

    for role, encoder, config in active:
        minimum = dataset_role_min(length_stats, role)
        if minimum < 1:
            add_error(
                "pssm_covariance_length_stats_unavailable",
                f"{encoder.upper()} requires a positive {role} minimum sequence length",
            )
            continue
        lag_key = f"{encoder}_lag"
        try:
            lag = validate_pssm_covariance_lag(config.get(lag_key, PSSM_COVARIANCE_DEFAULT_LAG))
        except ValueError:
            continue
        if lag >= minimum:
            add_error(
                "pssm_covariance_lag_too_large",
                f"{encoder.upper()} {role} stream requires {lag_key} < "
                f"dataset minimum sequence length {minimum}, got {lag}",
            )


def _stream_flat_config(config, stream):
    flat = dict(config)
    params = stream.get("params") if isinstance(stream.get("params"), dict) else {}
    flat.update(params)
    flat.update(
        {
            "tokenizer": stream.get("tokenizer", ""),
            "encoder": stream.get("encoder", "index"),
            "max_length": stream.get("max_length", config.get("max_length", 60)),
        }
    )
    tokenizer = str(flat["tokenizer"] or "").lower()
    if tokenizer in PRETRAINED_TOKENIZERS:
        flat[f"{tokenizer}_model"] = stream.get("model_source", "")
    return flat


def _stream_supports_vector(config, stream):
    flat = _stream_flat_config(config, stream)
    return str(
        flat.get("tokenizer", "")
    ).lower() in PRETRAINED_TOKENIZERS or supports_classic_pair_vector_inference(flat)


def _stream_supports_token(config, stream):
    flat = _stream_flat_config(config, stream)
    return str(
        flat.get("tokenizer", "")
    ).lower() in PRETRAINED_TOKENIZERS or supports_classic_pair_token_inference(flat)


def _validate_numeric_config(config, add_error):
    def present(key):
        return key in config and config[key] not in [None, ""]

    def as_float(key):
        try:
            value = float(config[key])
        except (TypeError, ValueError):
            add_error("invalid_numeric_parameter", f"{key} must be numeric")
            return None
        if not math.isfinite(value):
            add_error("invalid_numeric_parameter", f"{key} must be finite")
            return None
        return value

    def require_float(
        key,
        min_value=None,
        max_value=None,
        inclusive_max=True,
        inclusive_min=True,
    ):
        if not present(key):
            return
        value = as_float(key)
        if value is None:
            return
        if min_value is not None:
            invalid = value < min_value if inclusive_min else value <= min_value
            if invalid:
                op = ">=" if inclusive_min else ">"
                add_error("invalid_numeric_parameter", f"{key} must be {op} {min_value}")
        if max_value is not None:
            invalid = value > max_value if inclusive_max else value >= max_value
            if invalid:
                op = "<=" if inclusive_max else "<"
                add_error("invalid_numeric_parameter", f"{key} must be {op} {max_value}")

    def require_int(key, min_value=None, max_value=None):
        if not present(key):
            return
        value = as_float(key)
        if value is None:
            return
        if not value.is_integer():
            add_error("invalid_numeric_parameter", f"{key} must be an integer")
            return
        if min_value is not None and value < min_value:
            add_error("invalid_numeric_parameter", f"{key} must be >= {min_value}")
        if max_value is not None and value > max_value:
            add_error("invalid_numeric_parameter", f"{key} must be <= {max_value}")

    tokenizer = str(config.get("tokenizer", "") or "").strip().lower()
    encoder = str(config.get("encoder", "") or "").strip().lower()
    if present("stream_alignment_dim"):
        alignment_dim = config["stream_alignment_dim"]
        if (
            isinstance(alignment_dim, bool)
            or not isinstance(alignment_dim, int)
            or alignment_dim < 1
        ):
            add_error(
                "stream_alignment_dim_invalid",
                "stream_alignment_dim must be a positive integer",
            )
    if tokenizer == "kmer":
        if present("k"):
            try:
                normalized_k = validate_kmer_k(config["k"])
            except ValueError as exc:
                add_error("invalid_numeric_parameter", str(exc))
            else:
                if encoder in RESIDUE_PROFILE_ENCODERS and normalized_k != 1:
                    add_error(
                        "profile_encoder_requires_k1",
                        f"{encoder} is defined per raw residue and requires k=1; "
                        "PSSM/PSFM profile k-mer aggregation is not supported",
                    )
                if encoder in COVARIANCE_ENCODERS and normalized_k != 1:
                    add_error(
                        "covariance_encoder_requires_k1",
                        f"{encoder} is defined per raw residue and requires k=1",
                    )
                if encoder in PDT_ENCODERS and normalized_k != 1:
                    add_error(
                        "pdt_encoder_requires_k1",
                        "pdt is defined per raw residue and requires k=1",
                    )
    else:
        require_int("k", 1)
        if encoder in RESIDUE_PROFILE_ENCODERS:
            add_error(
                "profile_encoder_requires_kmer",
                f"{encoder} requires tokenizer=kmer and k=1",
            )
        if encoder in COVARIANCE_ENCODERS:
            add_error(
                "covariance_encoder_requires_kmer",
                f"{encoder} requires tokenizer=kmer and k=1",
            )
        if encoder in PDT_ENCODERS:
            add_error(
                "pdt_encoder_requires_kmer",
                "pdt requires tokenizer=kmer and k=1",
            )

    if present("dr_max_distance"):
        try:
            validate_dr_max_distance(config["dr_max_distance"])
        except DRContractError as exc:
            add_error("invalid_numeric_parameter", str(exc))
    if present("dr_max_raw_length"):
        try:
            validate_dr_max_raw_length(config["dr_max_raw_length"])
        except DRContractError as exc:
            add_error("invalid_numeric_parameter", str(exc))
    if present("dr_unknown_residue_policy"):
        try:
            validate_dr_unknown_residue_policy(config["dr_unknown_residue_policy"])
        except DRContractError as exc:
            add_error("invalid_parameter", str(exc))
    if present("frequency_max_raw_length"):
        try:
            validate_frequency_max_raw_length(config["frequency_max_raw_length"])
        except ValueError as exc:
            add_error("invalid_numeric_parameter", str(exc))
    if present("covariance_max_raw_length"):
        try:
            validate_covariance_max_raw_length(config["covariance_max_raw_length"])
        except ValueError as exc:
            add_error("invalid_numeric_parameter", str(exc))
    if present("pdt_lag"):
        try:
            validate_pdt_lag(config["pdt_lag"])
        except ValueError as exc:
            add_error("invalid_numeric_parameter", str(exc))
    if present("pdt_max_raw_length"):
        try:
            validate_pdt_max_raw_length(config["pdt_max_raw_length"])
        except ValueError as exc:
            add_error("invalid_numeric_parameter", str(exc))
    for key in sorted(f"{name}_lag" for name in PSSM_COVARIANCE_ENCODERS):
        if present(key):
            try:
                validate_pssm_covariance_lag(config[key])
            except ValueError as exc:
                add_error("invalid_numeric_parameter", f"{key}: {exc}")
    if present("dr_dense_memory_limit_gib"):
        try:
            validate_dr_dense_memory_limit_gib(config["dr_dense_memory_limit_gib"])
        except DRContractError as exc:
            add_error("invalid_numeric_parameter", str(exc))
    if tokenizer == "top-n-gram" and present("top_ngram_n"):
        try:
            validate_top_ngram_n(config["top_ngram_n"])
        except TopNGramContractError as exc:
            add_error("invalid_numeric_parameter", str(exc))
    elif present("top_ngram_n"):
        require_int("top_ngram_n", 1)
    if present("topngram_dense_memory_limit_gib"):
        try:
            validate_top_ngram_dense_memory_limit_gib(config["topngram_dense_memory_limit_gib"])
        except TopNGramContractError as exc:
            add_error("invalid_numeric_parameter", str(exc))

    if present("dt_max_distance"):
        try:
            validate_dt_max_distance(config["dt_max_distance"])
        except DTContractError as exc:
            add_error("invalid_numeric_parameter", str(exc))
    if present("dt_dense_memory_limit_gib"):
        try:
            validate_dt_dense_memory_limit_gib(config["dt_dense_memory_limit_gib"])
        except DTContractError as exc:
            add_error("invalid_numeric_parameter", str(exc))
    if present("lsa_dense_memory_limit_gib"):
        try:
            validate_lsa_dense_memory_limit_gib(config["lsa_dense_memory_limit_gib"])
        except LSAContractError as exc:
            add_error(exc.code, str(exc))
    if present("lda_dense_memory_limit_gib"):
        try:
            validate_lda_dense_memory_limit_gib(config["lda_dense_memory_limit_gib"])
        except LDAContractError as exc:
            add_error(exc.code, str(exc))
    if present("plsa_memory_limit_gib"):
        try:
            validate_plsa_memory_limit_gib(config["plsa_memory_limit_gib"])
        except PLSAContractError as exc:
            add_error(exc.code, str(exc))
    if present("tsvd_dense_memory_limit_gib"):
        try:
            validate_tsvd_dense_memory_limit_gib(config["tsvd_dense_memory_limit_gib"])
        except TSVDContractError as exc:
            add_error(exc.code, str(exc))
    if present("pca_dense_memory_limit_gib"):
        try:
            validate_pca_dense_memory_limit_gib(config["pca_dense_memory_limit_gib"])
        except PCAContractError as exc:
            add_error(exc.code, str(exc))
    if present("kpca_memory_limit_gib"):
        try:
            validate_kpca_memory_limit_gib(config["kpca_memory_limit_gib"])
        except KPCAContractError as exc:
            add_error(exc.code, str(exc))

    for key in [
        "embedding_vector_size",
        "embedding_window",
        "embedding_min_count",
        "embedding_epochs",
        "embedding_workers",
        "fasttext_bucket",
        "glove_batch_size",
        "max_length",
        "batch_size",
        "epochs",
        "lsa_dim",
        "lda_topics",
        "plsa_topics",
        "ml_reduction_dim",
        "ml_tuning_max_candidates",
        "svm_max_iter",
        "logreg_max_iter",
        "rf_estimators",
        "rf_min_samples_leaf",
        "et_estimators",
        "et_min_samples_leaf",
        "histgb_max_iter",
        "xgb_estimators",
        "lgb_estimators",
        "profile_iterations",
        "profile_threads",
        "interaction_hidden_dim",
        "interaction_heads",
    ]:
        require_int(key, 1)
    require_int("seed", 0)
    require_int("ml_tuning_max_candidates", 1, ML_TUNING_MAX_CANDIDATES)
    require_int("ml_cv_folds", 2, 5)
    if present("svm_calibration_cv_folds"):
        require_int(
            "svm_calibration_cv_folds",
            SVM_CALIBRATION_CV_FOLDS,
            SVM_CALIBRATION_CV_FOLDS,
        )
    for key in [
        "early_stop_patience",
        "rf_max_depth",
        "et_max_depth",
        "histgb_max_leaf_nodes",
        "xgb_max_depth",
        "lgb_max_depth",
    ]:
        require_int(key, 0)

    for key in [
        "lr",
        "weight_decay",
        "grad_clip_norm",
        "early_stop_min_delta",
        "semantic_similarity_weight",
        "semantic_zero_weight",
        "interaction_semantic_similarity_weight",
        "svm_c",
        "svm_tol",
        "logreg_c",
        "histgb_l2_regularization",
        "xgb_lr",
        "xgb_reg_alpha",
        "xgb_reg_lambda",
        "lgb_lr",
        "lgb_reg_alpha",
        "lgb_reg_lambda",
        "profile_evalue",
        "profile_inclusion_evalue",
    ]:
        require_float(key, 0.0)
    require_float("histgb_lr", 0.0, inclusive_min=False)
    if str(config.get("ml_reduction") or "none").strip().lower() == "kpca":
        try:
            kernel = validate_kpca_kernel(config.get("kpca_kernel", "rbf"))
        except KPCAContractError as exc:
            add_error(exc.code, str(exc))
            kernel = None
        if kernel in KPCA_KERNELS:
            try:
                validate_kpca_gamma(kernel, config.get("kpca_gamma"))
            except KPCAContractError as exc:
                add_error(exc.code, str(exc))
    if present("histgb_max_leaf_nodes"):
        value = as_float("histgb_max_leaf_nodes")
        if value == 1:
            add_error(
                "invalid_numeric_parameter",
                "histgb_max_leaf_nodes must be 0 (unlimited) or at least 2",
            )
    for key in ["profile_evalue", "profile_inclusion_evalue"]:
        if present(key):
            value = as_float(key)
            if value is not None and value <= 0.0:
                add_error("invalid_numeric_parameter", f"{key} must be > 0")
    if present("logreg_c"):
        value = as_float("logreg_c")
        if value is not None and value <= 0.0:
            add_error("invalid_numeric_parameter", "logreg_c must be > 0.0")
    if present("lgb_lr"):
        value = as_float("lgb_lr")
        if value is not None and value <= 0.0:
            add_error("invalid_numeric_parameter", "lgb_lr must be > 0.0")

    require_float("dropout", 0.0, 1.0, inclusive_max=False)
    if str(config.get("interaction_level", "")).strip().lower() == "token":
        try:
            interaction_hidden_dim = int(config.get("interaction_hidden_dim", 128))
            interaction_heads = int(config.get("interaction_heads", 4))
        except (TypeError, ValueError):
            interaction_hidden_dim = interaction_heads = 0
        if (
            interaction_hidden_dim > 0
            and interaction_heads > 0
            and interaction_hidden_dim % interaction_heads != 0
        ):
            add_error(
                "pair_attention_dimension_invalid",
                "interaction_hidden_dim must be divisible by interaction_heads",
            )
    require_float("glove_x_max", 1e-12)
    require_float("glove_alpha", 1e-12, 1.0)
    require_float("glove_learning_rate", 1e-12)
    require_float("static_embedding_memory_limit_gib", 1e-12)
    for key in ["xgb_subsample", "xgb_colsample", "lgb_subsample", "lgb_colsample"]:
        require_float(key, 0.0, 1.0)
    for key in ["xgb_subsample", "xgb_colsample", "lgb_subsample", "lgb_colsample"]:
        if present(key):
            value = as_float(key)
            if value is not None and value <= 0.0:
                add_error("invalid_numeric_parameter", f"{key} must be > 0.0")

    for key, choices in {
        "ml_reduction": {"none", "pca", "kpca", "tsvd"},
        "kpca_kernel": set(KPCA_KERNELS),
        "ml_tuning_budget": set(ML_TUNING_BUDGETS),
        "ml_tuning_objective": {"f1", "macro_f1", "sacc", "auc", "aupr", "dual_f1_sacc"},
        "ml_tuning_mode": {"holdout", "cv"},
        "svm_kernel": {"linear", "rbf"},
        "svm_calibration": set(SVM_CALIBRATION_OPTIONS),
        "svm_convergence_policy": {"strict", "bounded_retry"},
        "logreg_convergence_policy": {"strict", "bounded_retry"},
        "profile_on_missing": {
            "error",
            "blosum62_hybrid",
            TOP_N_GRAM_FALLBACK_POLICY,
        },
    }.items():
        if present(key) and str(config[key]).strip().lower() not in choices:
            add_error("invalid_choice_parameter", f"{key} must be one of {sorted(choices)}")
    if (
        tokenizer in {"top-n-gram", "dt"}
        and present("profile_on_missing")
        and str(config["profile_on_missing"]).strip().lower()
        not in {"error", TOP_N_GRAM_FALLBACK_POLICY}
    ):
        fallback_code = (
            "top_ngram_fallback_unsupported"
            if tokenizer == "top-n-gram"
            else "dt_fallback_unsupported"
        )
        add_error(
            fallback_code,
            f"{tokenizer} requires profile_on_missing=error or {TOP_N_GRAM_FALLBACK_POLICY}",
        )
    if encoder in {"pssm", "psfm", *STRICT_PSSM_COVARIANCE_ENCODERS}:
        if (
            present("profile_on_missing")
            and str(config["profile_on_missing"]).strip().lower() != "error"
        ):
            add_error(
                "strict_profile_fallback_forbidden",
                f"{encoder} requires profile_on_missing=error; use "
                + (
                    "psfm_blosum62_hybrid for the explicit frequency fallback variant"
                    if encoder == "psfm"
                    else "a named PSSM-BLOSUM62 hybrid for an explicit score fallback"
                ),
            )
    if encoder in {
        "pssm_blosum62_hybrid",
        "psfm_blosum62_hybrid",
        *HYBRID_PSSM_COVARIANCE_ENCODERS,
    }:
        if (
            present("profile_on_missing")
            and str(config["profile_on_missing"]).strip().lower() != "blosum62_hybrid"
        ):
            add_error(
                "hybrid_profile_policy_mismatch",
                f"{encoder} requires profile_on_missing=blosum62_hybrid",
            )

    if present("svm_gamma"):
        gamma = str(config["svm_gamma"]).strip().lower()
        if gamma not in {"scale", "auto"}:
            try:
                value = float(gamma)
            except (TypeError, ValueError):
                add_error("invalid_numeric_parameter", "svm_gamma must be numeric, scale, or auto")
            else:
                if value <= 0.0:
                    add_error("invalid_numeric_parameter", "svm_gamma must be > 0")


def _validate_web_prediction_head_preset(config, add_error):
    hidden_dims = config.get("head_hidden_dims")
    if hidden_dims is None:
        return
    if list(hidden_dims) not in ([], [128], [256, 128]):
        add_error(
            "prediction_head_preset_unsupported",
            "Web experiments only support prediction-head hidden_dims [], [128], or [256,128]",
        )


def _validate_ensemble_config(normalized, add_error):
    raw = normalized.get("ensemble_members")
    if isinstance(raw, list):
        members = [str(item).strip() for item in raw if str(item).strip()]
    elif isinstance(raw, str):
        members = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        members = []
    if len(members) < 2:
        add_error(
            "ensemble_members_required",
            "ensemble_members must contain at least two trained model entries",
        )
    method = str(normalized.get("ensemble_method") or "probability_mean").strip().lower()
    if method != "probability_mean":
        add_error(
            "ensemble_method_unsupported",
            "ensemble_method must be probability_mean",
        )
    weight_source = str(normalized.get("ensemble_weight_source") or "validation").strip().lower()
    if weight_source not in {"equal", "validation"}:
        add_error(
            "ensemble_weight_source_unsupported",
            "ensemble_weight_source must be equal or validation",
        )
    if weight_source == "validation" and len(members) != 2:
        add_error(
            "ensemble_validation_weights_two_members",
            "validation weight search currently requires exactly two members",
        )
    step = normalized.get("ensemble_grid_step")
    if step not in (None, ""):
        try:
            step_value = float(step)
            if not (0.0 < step_value < 1.0):
                raise ValueError
        except (TypeError, ValueError):
            add_error(
                "ensemble_grid_step_invalid",
                "ensemble_grid_step must be a number in (0, 1)",
            )


def _supports_pair_reusable_inference(config):
    tokenizer = str(config.get("tokenizer", "")).lower()
    model = str(config.get("model", "")).lower()
    classifier = str(config.get("classifier", "")).lower() or model
    pair_strategy = str(config.get("pair_strategy", "")).lower()
    interaction_level = str(config.get("interaction_level", "")).lower()
    interaction_model = str(config.get("interaction_model", "")).lower()
    training_mode = str(config.get("training_mode", "")).lower()
    predictor_type = str(config.get("predictor_type", "")).lower()
    fusion_mode = str(config.get("fusion_mode", "concat_all") or "concat_all").lower()
    streams = _pair_streams(config)
    if (
        streams
        and pair_strategy == "two_stream"
        and str(config.get("stream_mode", "homogeneous")) == "heterogeneous"
    ):
        homogeneous = streams["peptide"] == streams["partner"]
        if interaction_level == "token":
            return (
                all(_stream_supports_token(config, stream) for stream in streams.values())
                and interaction_model in {"cross_attn", "bi_cross_attn"}
                and (
                    (training_mode == END_TO_END and classifier == "mlp")
                    or (training_mode == EXPORT_EMBEDDINGS and classifier in CLASSICAL_CLASSIFIERS)
                )
                and model in {interaction_model, f"{interaction_model}+mlp"}
            )
        if not all(_stream_supports_vector(config, stream) for stream in streams.values()):
            return False
        if predictor_type == "sample_graph":
            return model in GRAPH_PREDICTORS and (
                model == "hgt" or homogeneous or fusion_mode == "concat"
            )
        if (
            predictor_type != "classifier"
            or model not in CLASSIFIERS
            or classifier not in CLASSIFIERS
        ):
            return False
        if fusion_mode == "aligned_gated":
            return model == "mlp"
        if not homogeneous and fusion_mode != "concat":
            return False
        return True
    if pair_strategy == "single_stream":
        pretrained_residue = tokenizer in PRETRAINED_TOKENIZERS and model in SEQUENCE_MODELS
        if (
            not pretrained_residue and not supports_classic_pair_token_inference(config)
        ) or model not in SEQUENCE_MODELS:
            return False
        if training_mode == END_TO_END:
            return classifier == "mlp"
        if training_mode == EXPORT_EMBEDDINGS:
            return classifier in CLASSICAL_CLASSIFIERS
        return False
    if interaction_level == "token":
        if supports_pair_token_inference(config):
            return True
        return (
            pair_strategy == "two_stream"
            and interaction_level == "token"
            and interaction_model in {"cross_attn", "bi_cross_attn"}
            and training_mode == EXPORT_EMBEDDINGS
            and classifier in CLASSICAL_CLASSIFIERS
            and model in {interaction_model, f"{interaction_model}+mlp"}
            and (
                tokenizer in PRETRAINED_TOKENIZERS or supports_classic_pair_token_inference(config)
            )
        )
    if predictor_type == "sample_graph":
        return (
            (tokenizer in PRETRAINED_TOKENIZERS or supports_classic_pair_vector_inference(config))
            and pair_strategy == "two_stream"
            and interaction_level == "vector"
            and model in GRAPH_PREDICTORS
            and (fusion_mode != "aligned_gated" or model in {"gcn", "gat"})
        )
    return (
        (tokenizer in PRETRAINED_TOKENIZERS or supports_classic_pair_vector_inference(config))
        and pair_strategy == "two_stream"
        and interaction_level == "vector"
        and predictor_type == "classifier"
        and model in CLASSIFIERS
        and classifier in CLASSIFIERS
        and (fusion_mode != "aligned_gated" or model == "mlp")
    )


def _supports_reusable_inference(task_type, config):
    task_type = normalize_task_type(task_type)
    if normalize_task_type(task_type) in PAIR_REUSABLE_INFERENCE_TASKS:
        return _supports_pair_reusable_inference(config)
    if normalize_task_type(task_type) != "pep_function":
        return False
    tokenizer = str(config.get("tokenizer", "")).lower()
    model = str(config.get("model", "")).lower()
    classifier = str(config.get("classifier", "")).lower() or (
        model if model in CLASSIFIERS else ""
    )
    if tokenizer in PRETRAINED_TOKENIZERS:
        sequence_model = str(config.get("sequence_model", "") or "").lower()
        if not sequence_model:
            return classifier in CLASSIFIERS
        training_mode = str(config.get("training_mode", "") or "").lower()
        if sequence_model not in SEQUENCE_MODELS:
            return False
        if training_mode == EXPORT_EMBEDDINGS:
            return classifier in CLASSICAL_CLASSIFIERS
        return training_mode == END_TO_END and classifier == "mlp"
    if str(config.get("training_mode", "")).lower() == EXPORT_EMBEDDINGS:
        sequence_model = str(config.get("sequence_model", "") or "").lower()
        return sequence_model in SEQUENCE_MODELS and classifier in CLASSICAL_CLASSIFIERS
    if config.get("sequence_model") or model == "mlp":
        return True
    return classifier in CLASSICAL_CLASSIFIERS


def inference_unavailable_metadata(task_type, config):
    task_type = normalize_task_type(task_type)
    normalized = config or {}
    if normalize_task_type(task_type) == "pep_function":
        if uses_profile_features(normalized):
            return {
                "reason": PROFILE_DEPENDENT_INFERENCE_REASON,
                "reason_code": "profile_features_require_sample_id",
                "inference_unavailable_reason": PROFILE_DEPENDENT_INFERENCE_REASON,
                "inference_unavailable_code": "profile_features_require_sample_id",
            }
        pretrained_residue = (
            str(normalized.get("tokenizer", "")).lower() in PRETRAINED_TOKENIZERS
            and str(normalized.get("sequence_model", "")).lower() in SEQUENCE_MODELS
        )
        if (
            str(normalized.get("training_mode", "")).lower() == EXPORT_EMBEDDINGS
            and not pretrained_residue
        ):
            return {
                "reason": SEQUENCE_EXPORT_INFERENCE_UNAVAILABLE_REASON,
                "reason_code": "sequence_export_inference_unavailable",
            }
        model = str(normalized.get("model", "")).lower()
        if (normalized.get("sequence_model") or model == "mlp") and not pretrained_residue:
            if str(normalized.get("dl_scaler", "none") or "none").lower() != "none":
                return {
                    "reason": SCALED_SEQUENCE_INFERENCE_UNAVAILABLE_REASON,
                    "reason_code": "scaled_sequence_inference_unavailable",
                }
    if normalize_task_type(task_type) == "pep_hlai":
        return {
            "reason": HLAI_ROUTE_INFERENCE_UNAVAILABLE_REASON,
            "reason_code": "hlai_route_inference_unavailable",
        }
    return {
        "reason": ROUTE_INFERENCE_UNAVAILABLE_REASON,
        "reason_code": "route_inference_unavailable",
    }


def _preflight_inference_warning(task_type, config):
    task_type = normalize_task_type(task_type)
    metadata = inference_unavailable_metadata(task_type, config)
    if metadata.get("reason_code") == "profile_features_require_sample_id":
        return "profile_inference_unavailable", metadata["reason"]
    return metadata["reason_code"], metadata["reason"]


def missing_bundle_metadata(task_type, config):
    task_type = normalize_task_type(task_type)
    if config and _supports_reusable_inference(task_type, config):
        return {
            "reason": "legacy route did not emit a reusable model bundle",
            "reason_code": "legacy_bundle_missing",
        }
    return inference_unavailable_metadata(task_type, config)


def preflight_experiment(dataset, task_type, raw_config):
    task_type = normalize_task_type(task_type)
    errors = []
    error_codes = []
    warnings = []
    warning_codes = []
    normalized = {}
    parameter_sources = {}
    effective_route = {}
    legacy_route_messages = []
    esmc_runtime_checked = False

    def add_error(code, message):
        error_codes.append(code)
        errors.append(message)

    def add_warning(code, message):
        warning_codes.append(code)
        warnings.append(message)

    def validate_esmc_source(source, label="ESM-C"):
        nonlocal esmc_runtime_checked
        try:
            backend = esmc_backend_for_source(source)
        except ValueError as exc:
            add_error("esmc_model_source_unsupported", f"{label}: {exc}")
            return
        if backend != ESMC_BACKEND_OFFICIAL or esmc_runtime_checked:
            return
        esmc_runtime_checked = True
        runtime = inspect_esmc_runtime()
        if not runtime["compatible"]:
            add_error(str(runtime["code"]), str(runtime["message"]))

    def deprecated_dt_hash_configured(config):
        if not isinstance(config, dict):
            return False
        if config.get("dt_hash_size") not in {None, ""}:
            return True
        for role in ("peptide", "partner", "protein"):
            if config.get(f"{role}_dt_hash_size") not in {None, ""}:
                return True
        streams = config.get("streams")
        if isinstance(streams, dict):
            for stream in streams.values():
                if not isinstance(stream, dict):
                    continue
                params = stream.get("params")
                if stream.get("dt_hash_size") not in {None, ""} or (
                    isinstance(params, dict) and params.get("dt_hash_size") not in {None, ""}
                ):
                    return True
        return False

    if deprecated_dt_hash_configured(raw_config):
        add_error(
            "dt_hash_size_removed",
            "dt_hash_size belonged to the legacy hashed DT implementation and "
            "has been removed; remove it and retrain with the DT v1 contract",
        )

    if normalize_task_type(dataset["task_type"]) != normalize_task_type(task_type):
        add_error(
            "task_mismatch",
            f"dataset task_type={dataset['task_type']} does not match requested task_type={task_type}",
        )

    allowed_formats = ALLOWED_DATASET_FORMATS.get(task_type, set())
    if dataset["format"] not in allowed_formats:
        if normalize_task_type(task_type) == "pep_hlaii" and dataset["format"] == "pair_csv_bundle":
            add_error(
                "hlaii_requires_pair_split_dir",
                "pep_hlaii currently requires split directories with pairs.csv files",
            )
        else:
            add_error(
                "format_unsupported",
                f"dataset format={dataset['format']} is not supported for task_type={task_type}",
            )

    try:
        normalized, parameter_sources = normalize_experiment_config_with_sources(
            task_type, raw_config
        )
    except ValueError as exc:
        contract_errors = (
            LSAContractError,
            LDAContractError,
            PLSAContractError,
            PCAContractError,
            KPCAContractError,
            TrainingModeContractError,
        )
        code = exc.code if isinstance(exc, contract_errors) else "normalization_error"
        if (
            code == "end_to_end_requires_mlp"
            and normalize_task_type(task_type) == "pep_function"
            and str((raw_config or {}).get("tokenizer", "") or "").strip().lower()
            in PRETRAINED_TOKENIZERS
        ):
            code = "pretrained_residue_end_to_end_requires_mlp"
        add_error(code, str(exc))
    else:
        legacy_route_messages = legacy_encoder_messages(raw_config)

    requested_budget = str((raw_config or {}).get("ml_tuning_budget", "") or "").strip().lower()
    if requested_budget in ML_TUNING_BUDGETS:
        expected = ML_TUNING_BUDGETS[requested_budget]
        supplied_candidates = (raw_config or {}).get("ml_tuning_max_candidates")
        supplied_folds = (raw_config or {}).get("ml_cv_folds")
        try:
            candidate_conflict = (
                supplied_candidates not in {None, ""}
                and int(supplied_candidates) != expected["max_candidates"]
            )
            fold_conflict = (
                supplied_folds not in {None, ""} and int(supplied_folds) != expected["cv_folds"]
            )
        except (TypeError, ValueError):
            candidate_conflict = fold_conflict = False
        if candidate_conflict:
            add_error(
                "tuning_budget_conflict", "ml_tuning_max_candidates conflicts with ml_tuning_budget"
            )
        if fold_conflict:
            add_error("tuning_budget_conflict", "ml_cv_folds conflicts with ml_tuning_budget")

    if normalized:
        validate_classification_loss_config(task_type, normalized, add_error)
        from pepplatform.objectives import resolve_selection

        try:
            resolve_selection(normalized)
        except (ValueError, TypeError) as exc:
            add_error("selection_objective_invalid", str(exc))
        try:
            validate_lsa_route(
                normalized,
                pair=normalize_task_type(task_type) in PAIR_REUSABLE_INFERENCE_TASKS,
            )
        except LSAContractError as exc:
            add_error(exc.code, str(exc))
        try:
            validate_lda_route(
                normalized,
                pair=normalize_task_type(task_type) in PAIR_REUSABLE_INFERENCE_TASKS,
            )
        except LDAContractError as exc:
            add_error(exc.code, str(exc))
        try:
            validate_plsa_route(
                normalized,
                pair=normalize_task_type(task_type) in PAIR_REUSABLE_INFERENCE_TASKS,
            )
        except PLSAContractError as exc:
            add_error(exc.code, str(exc))
        if (
            normalize_task_type(task_type) in PAIR_REUSABLE_INFERENCE_TASKS
            and str(normalized.get("reduction", "") or "").strip().lower() in {"lda", "plsa"}
            and bool(
                (raw_config or {}).get(
                    "ml_auto_tune",
                    normalized.get("ml_auto_tune"),
                )
            )
            and str(
                (raw_config or {}).get(
                    "ml_tuning_mode",
                    normalized.get("ml_tuning_mode", "cv"),
                )
                or "cv"
            )
            .strip()
            .lower()
            == "cv"
        ):
            add_error(
                "pair_bow_topic_cv_unsupported",
                "Strict Pair-BoW LDA/PLSA routes cannot tune inside CV folds; "
                "use ml_tuning_mode=holdout so the locked full-training corpus "
                "defines the topic model and validation performs selection",
            )
        canonical_enable = normalized.get("enable_semantic_similarity")
        legacy_enable = normalized.get("enable_interaction_semantic_similarity")
        if (
            canonical_enable is not None
            and legacy_enable is not None
            and bool(canonical_enable) != bool(legacy_enable)
        ):
            add_error(
                "semantic_alias_conflict",
                "enable_semantic_similarity conflicts with legacy enable_interaction_semantic_similarity",
            )
        normalized["enable_semantic_similarity"] = bool(
            canonical_enable if canonical_enable is not None else legacy_enable or False
        )
        canonical_weight = normalized.get("semantic_similarity_weight")
        legacy_weight = normalized.get("interaction_semantic_similarity_weight")
        if canonical_weight is not None and legacy_weight is not None:
            try:
                weights_conflict = float(canonical_weight) != float(legacy_weight)
            except (TypeError, ValueError):
                weights_conflict = False
            if weights_conflict:
                add_error(
                    "semantic_alias_conflict",
                    "semantic_similarity_weight conflicts with legacy interaction_semantic_similarity_weight",
                )
        normalized["semantic_similarity_weight"] = (
            canonical_weight
            if canonical_weight is not None
            else legacy_weight
            if legacy_weight is not None
            else 0.1
        )
        normalized["semantic_zero_weight"] = normalized.get("semantic_zero_weight", 0.25)
        # Missing means the historical objective; keep old normalized configs
        # free of new default fields so cloning/re-running does not migrate them.
        if "semantic_loss_type" in normalized:
            normalized["semantic_loss_type"] = str(normalized["semantic_loss_type"]).strip().lower()
            if normalized["semantic_loss_type"] not in {"legacy", "jaccard_supcon"}:
                add_error(
                    "semantic_loss_type_invalid",
                    "semantic_loss_type must be one of legacy, jaccard_supcon",
                )
        normalized["semantic_pair_mode"] = (
            str(normalized.get("semantic_pair_mode", "role_fused")).strip().lower()
        )
        projection_dim = normalized.get("semantic_projection_dim")
        try:
            normalized["semantic_projection_dim"] = (
                int(projection_dim) if projection_dim is not None else 64
            )
        except (TypeError, ValueError):
            normalized["semantic_projection_dim"] = 0
        temperature = normalized.get("semantic_contrastive_temperature")
        try:
            normalized["semantic_contrastive_temperature"] = (
                float(temperature) if temperature is not None else 0.1
            )
        except (TypeError, ValueError):
            normalized["semantic_contrastive_temperature"] = 0.0
        for key, default, minimum in (
            ("semantic_positive_partners", 1, 1),
            ("semantic_zero_partners", 8, 0),
        ):
            value = normalized.get(key)
            try:
                # Counts must not silently truncate fractions or accept booleans.
                if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                    raise ValueError
                normalized[key] = int(value) if value is not None else default
            except (TypeError, ValueError, OverflowError):
                normalized[key] = minimum - 1
        if normalized["semantic_pair_mode"] not in {"fused", "role", "role_fused"}:
            add_error(
                "semantic_pair_mode_invalid",
                "semantic_pair_mode must be one of fused, role, role_fused",
            )
        if normalized["semantic_projection_dim"] <= 0:
            add_error(
                "semantic_projection_dim_invalid",
                "semantic_projection_dim must be > 0",
            )
        if normalized["semantic_contrastive_temperature"] <= 0:
            add_error(
                "semantic_temperature_invalid",
                "semantic_contrastive_temperature must be > 0",
            )
        if normalized["semantic_positive_partners"] < 1:
            add_error(
                "semantic_positive_partners_invalid",
                "semantic_positive_partners must be an integer >= 1",
            )
        if normalized["semantic_zero_partners"] < 0:
            add_error(
                "semantic_zero_partners_invalid",
                "semantic_zero_partners must be an integer >= 0",
            )
        legacy_mode = str(normalized.get("interaction_semantic_mode", "") or "").strip().lower()
        if legacy_mode in REMOVED_SEMANTIC_MODES:
            add_error(
                "semantic_mode_removed",
                f"interaction_semantic_mode={legacy_mode} was removed; pair tasks now use interaction_profile_jaccard",
            )
        elif legacy_mode not in {"", "profile_jaccard"}:
            add_error(
                "semantic_mode_invalid",
                "interaction_semantic_mode only accepts the legacy value profile_jaccard",
            )
        for removed_field in REMOVED_SEMANTIC_FIELDS:
            if removed_field in raw_config and raw_config.get(removed_field) not in {None, ""}:
                add_error(
                    "semantic_parameter_removed",
                    f"{removed_field} was removed with the task-specific semantic modes",
                )
        for legacy_field in (
            "enable_interaction_semantic_similarity",
            "interaction_semantic_similarity_weight",
            "interaction_semantic_mode",
            "hla_context_semantic_margin",
            *REMOVED_SEMANTIC_FIELDS,
        ):
            normalized.pop(legacy_field, None)

        _validate_prott5_lengths(normalized, add_error, add_warning)
        _validate_numeric_config(normalized, add_error)
        _validate_web_prediction_head_preset(normalized, add_error)
        try:
            validate_linear_prediction_head_semantics(normalized)
        except ValueError as exc:
            add_error("linear_head_semantic_unsupported", str(exc))
        try:
            validate_index_route_semantics(normalized)
        except ValueError as exc:
            add_error("classical_index_unsupported", str(exc))

        tokenizer = str(normalized.get("tokenizer", "")).lower()
        encoder = str(normalized.get("encoder", "")).lower()
        family = encoder_family(encoder)
        pair_strategy = str(normalized.get("pair_strategy", "")).lower()
        interaction_level = str(normalized.get("interaction_level", "")).lower()
        streams = _pair_streams(normalized)
        graph_model = (
            str(normalized.get("graph_predictor", "") or normalized.get("model", ""))
            .strip()
            .lower()
        )
        fusion_mode = (
            str(normalized.get("fusion_mode", "concat_all") or "concat_all").strip().lower()
        )
        if (
            normalize_task_type(task_type) in PAIR_REUSABLE_INFERENCE_TASKS
            and pair_strategy == "two_stream"
            and interaction_level == "vector"
        ):
            raw_fusion = (raw_config or {}).get("fusion_mode")
            if graph_model == "hgt":
                if raw_fusion not in {None, ""}:
                    try:
                        legacy_hgt_fusion = normalize_pair_fusion_mode(raw_fusion)
                    except PairFusionError as exc:
                        add_error(exc.code, str(exc))
                    else:
                        if legacy_hgt_fusion not in {"concat", "concat_all"}:
                            add_error(
                                "hgt_typed_graph_fusion_unsupported",
                                "HGT uses typed role projection and does not accept a fusion mode",
                            )
                        else:
                            add_warning(
                                "hgt_legacy_fusion_marker_ignored",
                                f"HGT ignores legacy fusion_mode={legacy_hgt_fusion}; "
                                "new configurations use typed role projection",
                            )
                normalized.pop("fusion_mode", None)
                fusion_mode = ""
            else:
                try:
                    fusion_mode = normalize_pair_fusion_mode(fusion_mode)
                except PairFusionError as exc:
                    add_error(exc.code, str(exc))
                else:
                    normalized["fusion_mode"] = fusion_mode
        if graph_model == "graph_transformer":
            try:
                graph_hidden_dim = int(normalized.get("gnn_hidden_dim", 256) or 256)
            except (TypeError, ValueError):
                graph_hidden_dim = None
            if graph_hidden_dim is None or graph_hidden_dim < 4 or graph_hidden_dim % 4 != 0:
                add_error(
                    "graph_transformer_hidden_dim_invalid",
                    "Graph Transformer gnn_hidden_dim must be at least 4 and divisible by 4 heads",
                )
        _resolve_top_ngram_dataset_lengths(
            normalized,
            dataset,
            parameter_sources,
            add_error,
            add_warning,
        )
        _validate_covariance_dataset_lengths(normalized, dataset, add_error)
        _validate_pdt_dataset_lengths(normalized, dataset, add_error)
        _validate_pssm_covariance_dataset_lengths(
            normalized,
            dataset,
            add_error,
        )
        streams = _pair_streams(normalized)
        if streams and str(normalized.get("stream_mode", "homogeneous")) == "heterogeneous":
            for role, stream in streams.items():
                stream_tokenizer = str(stream.get("tokenizer", "") or "").lower()
                if stream_tokenizer == "esmc":
                    validate_esmc_source(
                        str(stream.get("model_source", "") or ""),
                        f"{role} stream ESM-C",
                    )
        elif tokenizer == "esmc":
            validate_esmc_source(pretrained_model_source(normalized))
        if tokenizer == "esm3" and encoder == "esm3":
            sequence_model = str(normalized.get("sequence_model", "") or "").strip().lower()
            model = str(normalized.get("model", "") or "").strip().lower()
            dl_scaler = str(normalized.get("dl_scaler", "none") or "none").strip().lower()
            if not sequence_model and model == "mlp" and dl_scaler == "none":
                add_warning(
                    "esm3_pooled_fp16_cache_rebuild",
                    "ESM3 pooled embeddings are trained in fp32; any existing fp16 "
                    "feature cache is stale and will be rebuilt automatically",
                )
        if normalized.get("enable_semantic_similarity"):
            model = str(normalized.get("model", "") or "").strip().lower()
            predictor_type = (
                str(normalized.get("predictor_type", "classifier") or "classifier").strip().lower()
            )
            training_mode = str(normalized.get("training_mode", "") or "").strip().lower()
            classifier = str(normalized.get("classifier", "") or "").strip().lower()
            supported = model in {
                *SEQUENCE_MODELS,
                "mlp",
                "cross_attn",
                "cross_attn+mlp",
                "bi_cross_attn",
                "bi_cross_attn+mlp",
            }
            if (
                predictor_type == "sample_graph"
                or (
                    normalize_task_type(task_type) in PAIR_REUSABLE_INFERENCE_TASKS
                    and model in GRAPH_PREDICTORS
                )
                or training_mode == "export_embeddings"
                or classifier in CLASSICAL_CLASSIFIERS
                or not supported
            ):
                add_error(
                    "semantic_model_unsupported",
                    "the unified semantic layer requires an end-to-end differentiable neural classifier with a sample-level embedding",
                )
        for role, stream in streams.items():
            stream_model = str(stream.get("stream_model", "") or "").strip().lower()
            if stream_model not in {"", "identity", "embedding"}:
                add_error(
                    "stream_model_unsupported",
                    f"{role} stream_model={stream_model} is not supported; use identity or embedding",
                )

        if uses_profile_features(normalized):
            profile_encoders = {
                str(normalized.get("encoder", "") or "").strip().lower(),
                str(normalized.get("aux_encoder", "") or "").strip().lower(),
                *{
                    str(stream.get("encoder", "") or "").strip().lower()
                    for stream in streams.values()
                    if isinstance(stream, dict)
                },
            }
            strict_profile_encoders = profile_encoders.intersection(
                {"pssm", "psfm", *STRICT_PSSM_COVARIANCE_ENCODERS}
            )
            if strict_profile_encoders:
                add_warning(
                    "strict_profile_complete_coverage_required",
                    "strict PSSM/PSFM routes require PSI-BLAST to produce the "
                    "requested profile for every sample; any uncovered sample "
                    "will stop training",
                )
            if profile_encoders.intersection(
                {
                    "pssm_blosum62_hybrid",
                    *HYBRID_PSSM_COVARIANCE_ENCODERS,
                }
            ):
                add_warning(
                    "pssm_blosum62_hybrid_fallback_enabled",
                    "PSSM-BLOSUM62 Hybrid will use explicit BLOSUM62 score "
                    "rows whenever a successful PSI-BLAST search cannot yield "
                    "a valid strict PSSM for an individual sample",
                )
            if "psfm_blosum62_hybrid" in profile_encoders:
                add_warning(
                    "psfm_blosum62_hybrid_fallback_enabled",
                    "PSFM-BLOSUM62 Hybrid will use BLOSUM62-derived target "
                    "frequencies whenever a successful PSI-BLAST search cannot "
                    "yield a valid strict PSFM for an individual sample",
                )
            try:
                profile_iterations = int(normalized.get("profile_iterations", 3))
            except (TypeError, ValueError):
                profile_iterations = 0
            if profile_iterations < 2:
                add_error(
                    "profile_iterations_too_low",
                    "profile_iterations must be >= 2 so PSI-BLAST can emit an ASCII PSSM",
                )

        if normalize_task_type(task_type) in {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}:
            if (
                pair_strategy == "single_stream"
                and streams
                and str(normalized.get("stream_mode", "homogeneous")) != "homogeneous"
            ):
                add_error(
                    "single_stream_requires_homogeneous_streams",
                    "The current Single-stream implementation requires identical peptide and partner stream specifications",
                )
            single_sequence_model = (
                str(normalized.get("sequence_model", "") or normalized.get("model", ""))
                .strip()
                .lower()
            )
            pretrained_single_stream = (
                pair_strategy == "single_stream"
                and tokenizer in PRETRAINED_TOKENIZERS
                and single_sequence_model in SEQUENCE_MODELS
            )
            if pair_strategy == "single_stream" and not (
                pretrained_single_stream
                or (tokenizer not in PRETRAINED_TOKENIZERS and family == "sequence")
            ):
                add_error(
                    "single_stream_requires_sequence_encoder",
                    "single_stream requires a classic sequence-preserving encoder or "
                    "an audited pretrained residue encoder followed by a sequence model",
                )
            if pretrained_single_stream:
                training_mode = str(normalized.get("training_mode", "")).strip().lower()
                classifier = str(normalized.get("classifier", "")).strip().lower()
                if training_mode not in {END_TO_END, EXPORT_EMBEDDINGS}:
                    add_error(
                        "pretrained_residue_training_mode_unsupported",
                        "pretrained residue predictors require training_mode=end_to_end or export_embeddings",
                    )
                elif training_mode == END_TO_END and classifier != "mlp":
                    add_error(
                        "pretrained_residue_end_to_end_requires_mlp",
                        "pretrained residue end-to-end routes require classifier=mlp",
                    )
                elif training_mode == EXPORT_EMBEDDINGS and classifier not in CLASSICAL_CLASSIFIERS:
                    add_error(
                        "pretrained_residue_export_requires_classical",
                        "pretrained residue export routes require a classical downstream classifier",
                    )
            if (
                pair_strategy == "two_stream"
                and interaction_level == "token"
                and tokenizer not in PRETRAINED_TOKENIZERS
                and family != "sequence"
            ):
                add_error(
                    "token_interaction_requires_sequence_encoder",
                    "token-level interaction requires a pretrained model or a classic sequence-preserving encoder",
                )
            if (
                streams
                and pair_strategy == "two_stream"
                and str(normalized.get("stream_mode", "homogeneous")) == "heterogeneous"
            ):
                for role, stream in streams.items():
                    flat_stream = _stream_flat_config(normalized, stream)
                    _validate_numeric_config(
                        flat_stream,
                        lambda code, message, role=role: add_error(
                            code,
                            f"{role} stream: {message}",
                        ),
                    )
                    if interaction_level == "token" and not _stream_supports_token(
                        normalized, stream
                    ):
                        add_error(
                            "token_interaction_requires_sequence_encoder",
                            f"{role} stream token-level interaction requires a pretrained or sequence-preserving encoder",
                        )
                    if interaction_level == "vector" and not _stream_supports_vector(
                        normalized, stream
                    ):
                        add_error(
                            "stream_vector_inference_unsupported",
                            f"{role} stream cannot reconstruct vector features for reusable inference",
                        )
                    stream_tokenizer = str(stream.get("tokenizer", "") or "").lower()
                    if stream_tokenizer in PRETRAINED_TOKENIZERS:
                        model_source = str(stream.get("model_source", "") or "").strip()
                        local_model_source = _resolve_local_model_source(
                            stream_tokenizer, model_source
                        )
                        if not local_model_source:
                            if _offline_mode_enabled():
                                add_error(
                                    "pretrained_model_source_unavailable",
                                    f"{role} stream {stream_tokenizer} model source is not local: {model_source}",
                                )
                            else:
                                add_warning(
                                    "pretrained_model_remote_download",
                                    f"{role} stream {stream_tokenizer} will be resolved remotely via HF_ENDPOINT={_hf_endpoint()}: {model_source}",
                                )
                        else:
                            add_warning(
                                "pretrained_model_local_cache",
                                f"{role} stream {stream_tokenizer} will use local files: {local_model_source}",
                            )
                homogeneous = streams["peptide"] == streams["partner"]
                fusion_mode = str(
                    normalized.get("fusion_mode", "concat_all") or "concat_all"
                ).lower()
                predictor_type = str(
                    normalized.get("predictor_type", "classifier") or "classifier"
                ).lower()
                classifier = str(
                    normalized.get("classifier", "") or normalized.get("model", "")
                ).lower()
                if (
                    not homogeneous
                    and predictor_type == "sample_graph"
                    and graph_model != "hgt"
                    and fusion_mode != "concat"
                ):
                    add_error(
                        "heterogeneous_graph_requires_concat",
                        "heterogeneous sample-graph streams require fusion_mode=concat",
                    )
                if (
                    not homogeneous
                    and interaction_level == "vector"
                    and fusion_mode not in {"concat", "aligned_gated"}
                ):
                    add_error(
                        "heterogeneous_fusion_requires_alignment",
                        "heterogeneous vector streams require fusion_mode=concat or aligned_gated",
                    )
                if fusion_mode == "aligned_gated" and classifier != "mlp":
                    add_error(
                        "aligned_gated_requires_mlp",
                        "aligned_gated fusion requires classifier=mlp",
                    )

        if normalize_task_type(task_type) == "pep_function":
            classifier = str(normalized.get("classifier", "")).lower()
            sequence_model = str(normalized.get("sequence_model", "")).lower()
            if tokenizer in PRETRAINED_TOKENIZERS and sequence_model:
                training_mode = str(normalized.get("training_mode", "")).lower()
                if sequence_model not in SEQUENCE_MODELS:
                    add_error(
                        "pretrained_sequence_model_unsupported",
                        f"unsupported pretrained residue architecture: {sequence_model}",
                    )
                elif training_mode not in {END_TO_END, EXPORT_EMBEDDINGS}:
                    add_error(
                        "pretrained_residue_training_mode_unsupported",
                        "pretrained residue predictors require training_mode=end_to_end or export_embeddings",
                    )
                elif training_mode == END_TO_END and classifier != "mlp":
                    add_error(
                        "pretrained_residue_end_to_end_requires_mlp",
                        "pretrained residue end-to-end routes require classifier=mlp",
                    )
                elif training_mode == EXPORT_EMBEDDINGS and classifier not in CLASSICAL_CLASSIFIERS:
                    add_error(
                        "pretrained_residue_export_requires_classical",
                        "pretrained residue export routes require a classical downstream classifier",
                    )
                adapter_dim = normalized.get("residue_adapter_dim")
                try:
                    adapter_dim = int(adapter_dim)
                except (TypeError, ValueError):
                    add_error(
                        "residue_adapter_dim_invalid",
                        "residue_adapter_dim must be an integer",
                    )
                else:
                    if not RESIDUE_ADAPTER_DIM_MIN <= adapter_dim <= RESIDUE_ADAPTER_DIM_MAX:
                        add_error(
                            "residue_adapter_dim_out_of_range",
                            "residue_adapter_dim must be between "
                            f"{RESIDUE_ADAPTER_DIM_MIN} and {RESIDUE_ADAPTER_DIM_MAX}",
                        )
                    elif adapter_dim % RESIDUE_ADAPTER_DIM_MULTIPLE != 0:
                        add_error(
                            "residue_adapter_dim_not_divisible",
                            "residue_adapter_dim must be divisible by "
                            f"{RESIDUE_ADAPTER_DIM_MULTIPLE}",
                        )
            if (
                tokenizer not in PRETRAINED_TOKENIZERS
                and sequence_model in CHAIN_GRAPH_ARCHITECTURES
                and encoder not in CHAIN_GRAPH_COMPATIBLE_ENCODERS
            ):
                add_error(
                    "chain_graph_encoder_unsupported",
                    "sequence-chain graph architectures require per-position feature vectors; "
                    f"encoder={encoder or '<empty>'} is not supported",
                )
        if (
            normalize_task_type(task_type) in PAIR_REUSABLE_INFERENCE_TASKS
            and pair_strategy == "single_stream"
            and str(normalized.get("sequence_model", "")).lower() in CHAIN_GRAPH_ARCHITECTURES
            and encoder not in CHAIN_GRAPH_COMPATIBLE_ENCODERS
        ):
            add_error(
                "chain_graph_encoder_unsupported",
                "Single-stream sequence-chain graph architectures require per-position feature vectors; "
                f"encoder={encoder or '<empty>'} is not supported",
            )

        if tokenizer in PRETRAINED_TOKENIZERS and (
            not streams or str(normalized.get("stream_mode", "homogeneous")) != "heterogeneous"
        ):
            model_source = pretrained_model_source(normalized)
            local_model_source = _resolve_local_model_source(tokenizer, model_source)
            if not local_model_source:
                if _offline_mode_enabled():
                    add_error(
                        "pretrained_model_source_unavailable",
                        f"{tokenizer} model source is not local: {model_source}",
                    )
                else:
                    add_warning(
                        "pretrained_model_remote_download",
                        f"{tokenizer} model source will be resolved remotely via HF_ENDPOINT={_hf_endpoint()}: {model_source}",
                    )
            else:
                add_warning(
                    "pretrained_model_local_cache",
                    f"{tokenizer} model source will use local files: {local_model_source}",
                )

    splits = dataset.get("metadata", {}).get("splits", {})
    if splits and "test" not in splits:
        add_warning(
            "no_test_split",
            "dataset has no test split; the run will report validation metrics only",
        )
    for message in legacy_route_messages:
        add_warning("legacy_pretrained_encoder", message)

    is_ensemble = bool(normalized.get("ensemble_members"))
    if is_ensemble:
        _validate_ensemble_config(normalized, add_error)
        script = "scripts/train_ensemble.py"
        supports_inference = True
        effective_route = {
            "schema_version": 5,
            "route_id": "sequence.ensemble",
            "task_type": task_type,
            "family": "ensemble",
            "tokenizer": "ensemble",
            "encoder": "probability_mean",
            "model": "ensemble",
        }
    else:
        script = (
            "scripts/train_pair_pipeline.py"
            if normalize_task_type(task_type) in PAIR_REUSABLE_INFERENCE_TASKS
            else "scripts/train_sequence_pipeline.py"
        )
        supports_inference = (
            _supports_reusable_inference(task_type, normalized) if normalized else False
        )
        if normalized and not supports_inference:
            code, message = _preflight_inference_warning(task_type, normalized)
            add_warning(code, message)
        elif (
            normalized
            and supports_inference
            and normalize_task_type(task_type) in {*PAIR_REUSABLE_INFERENCE_TASKS, "pep_function"}
            and uses_profile_features(normalized)
        ):
            add_warning(
                "profile_inference_requires_blast",
                PROFILE_GENERATED_PAIR_INFERENCE_REASON,
            )

        if normalized:
            try:
                effective_route = resolve_effective_route(normalized, task_type)
            except ValueError as exc:
                add_error("route_resolution_error", str(exc))

    return {
        "valid": not errors,
        "errors": errors,
        "error_codes": error_codes,
        "warnings": warnings,
        "warning_codes": warning_codes,
        "normalized_config": normalized,
        "effective_route": effective_route,
        "parameter_sources": parameter_sources,
        "dataset": {
            "id": dataset["id"],
            "name": dataset["name"],
            "task_type": dataset["task_type"],
            "format": dataset["format"],
            "metadata": dataset.get("metadata", {}),
        },
        "execution": {
            "script": script,
            "supports_reusable_inference": supports_inference,
        },
    }


install_legacy_exports(globals())


from pepplatform.stream_compat import install_legacy_exports as _install_legacy_stream_exports
_install_legacy_stream_exports(globals())
