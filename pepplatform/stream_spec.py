"""Canonical stream specifications for peptide pair-interaction workflows.

The pair pipeline historically exposed one global tokenizer/encoder/model source.
This module keeps that interface readable while normalizing every pair run to an
explicit peptide/partner stream pair.
"""

from __future__ import annotations

from pepplatform.stream_compat import normalize_stream_fields, normalize_stream_namespace
from pepplatform.task_compat import normalize_task_type


from argparse import Namespace
from copy import deepcopy
from typing import Any, Mapping

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
from pepplatform.pssm_covariance import (
    PSSM_COVARIANCE_DEFAULT_LAG,
    PSSM_COVARIANCE_ENCODERS,
    validate_pssm_covariance_lag,
)
from pepplatform.dr import (
    DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
    DR_DEFAULT_MAX_DISTANCE,
    DR_DEFAULT_MAX_RAW_LENGTH,
    DR_GLOBAL_MAX_LENGTH_UNIT,
    DR_ORDERED_MAX_LENGTH_UNIT,
    validate_dr_max_distance,
    validate_dr_max_raw_length,
    validate_dr_unknown_residue_policy,
)
from pepplatform.dt import (
    DT_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    DT_DEFAULT_MAX_DISTANCE,
    DT_MAX_LENGTH_UNIT,
    validate_dt_dense_memory_limit_gib,
    validate_dt_max_distance,
)
from pepplatform.top_ngram import (
    TOP_N_GRAM_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
    TOP_N_GRAM_MAX_LENGTH_UNIT,
    validate_top_ngram_dense_memory_limit_gib,
    validate_top_ngram_n,
)
from pepplatform.wordpiece import (
    WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
    WORDPIECE_MAX_LENGTH_UNIT,
    validate_wordpiece_max_raw_length,
    validate_wordpiece_vocab_size,
)
from pepplatform.unigram import (
    UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
    validate_unigram_max_raw_length,
    validate_unigram_vocab_size,
)
from pepplatform.route_registry import normalize_encoder_for_tokenizer
from pepplatform.workflow import (
    PRETRAINED_MODEL_DEFAULTS,
    PRETRAINED_TOKENIZERS,
    normalize_tokenizer_name,
)
from pepplatform.prott5 import PROTT5_MAX_RESIDUES, validate_prott5_max_residues
from pepplatform.esm3 import ESM3_DEFAULT_SDPA_BACKEND, validate_esm3_sdpa_backend
from pepplatform.pair_strategy import require_single_stream_homogeneous


PAIR_TASKS = {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}
STREAM_ROLES = ("peptide", "partner")
STREAM_CORE_FIELDS = (
    "feature_route",
    "tokenizer",
    "encoder",
    "stream_model",
    "model_source",
    "max_length",
)
STREAM_PARAM_FIELDS = (
    "k",
    "dr_max_distance",
    "dr_max_raw_length",
    "dr_unknown_residue_policy",
    "esm3_sdpa_backend",
    "frequency_max_raw_length",
    "top_ngram_n",
    "topngram_dense_memory_limit_gib",
    "dt_max_distance",
    "dt_dense_memory_limit_gib",
    "bpe_path",
    "unigram_path",
    "unigram_vocab_size",
    "unigram_max_raw_length",
    "wordpiece_path",
    "wordpiece_vocab_size",
    "wordpiece_max_raw_length",
    "word2vec_path",
    "glove_path",
    "fasttext_path",
    "pssm_index_path",
    "ac_pssm_blosum62_hybrid_index_path",
    "cc_pssm_blosum62_hybrid_index_path",
    "acc_pssm_blosum62_hybrid_index_path",
    "psfm_index_path",
    "frequency_profile_index_path",
    "profile_inclusion_evalue",
    "ac_lag",
    "cc_lag",
    "acc_lag",
    "pdt_lag",
    "pdt_max_raw_length",
    "covariance_max_raw_length",
    "ac_pssm_lag",
    "cc_pssm_lag",
    "acc_pssm_lag",
    "ac_pssm_blosum62_hybrid_lag",
    "cc_pssm_blosum62_hybrid_lag",
    "acc_pssm_blosum62_hybrid_lag",
)
MODEL_SOURCE_FIELDS = {
    "esm2": ("esm2_model",),
    "protbert": ("protbert_model", "probert_model"),
    "prott5": ("prott5_model",),
    "esmc": ("esmc_model",),
    "esm3": ("esm3_model",),
}
_INT_STREAM_PARAMS = {
    "k",
    "dr_max_distance",
    "dr_max_raw_length",
    "frequency_max_raw_length",
    "top_ngram_n",
    "dt_max_distance",
    "unigram_vocab_size",
    "unigram_max_raw_length",
    "wordpiece_vocab_size",
    "wordpiece_max_raw_length",
    "ac_lag",
    "cc_lag",
    "acc_lag",
    "pdt_lag",
    "pdt_max_raw_length",
    "covariance_max_raw_length",
    "ac_pssm_lag",
    "cc_pssm_lag",
    "acc_pssm_lag",
    "ac_pssm_blosum62_hybrid_lag",
    "cc_pssm_blosum62_hybrid_lag",
    "acc_pssm_blosum62_hybrid_lag",
}
_FLOAT_STREAM_PARAMS = {
    "topngram_dense_memory_limit_gib",
    "dt_dense_memory_limit_gib",
    "profile_inclusion_evalue",
}


def _mapping(source: Mapping[str, Any] | Namespace | None) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return normalize_stream_fields(dict(source))
    return normalize_stream_fields(dict(vars(source)))


def _value(source: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = source.get(key, default)
    return default if value in (None, "") else value


def _legacy_model_source(source: Mapping[str, Any], tokenizer: str) -> str:
    for field in MODEL_SOURCE_FIELDS.get(tokenizer, ()):
        value = str(source.get(field, "") or "").strip()
        if value:
            return value
    return str(PRETRAINED_MODEL_DEFAULTS.get(tokenizer, "") or "")


def _legacy_stream(source: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer = normalize_tokenizer_name(_value(source, "tokenizer", "esm2"))
    encoder, _legacy = normalize_encoder_for_tokenizer(tokenizer, source.get("encoder"))
    params = {
        key: deepcopy(source[key])
        for key in STREAM_PARAM_FIELDS
        if key in source and source[key] not in (None, "")
    }
    return {
        "feature_route": "pretrained" if tokenizer in PRETRAINED_TOKENIZERS else "classic",
        "tokenizer": tokenizer,
        "encoder": encoder,
        "stream_model": str(source.get("stream_model", "") or "").strip().lower(),
        "model_source": _legacy_model_source(source, tokenizer)
        if tokenizer in PRETRAINED_TOKENIZERS
        else "",
        "max_length": int(_value(source, "max_length", 512)),
        "params": params,
    }


def _role_overrides(source: Mapping[str, Any], role: str) -> dict[str, Any]:
    aliases = (role,) if role != "partner" else ("partner", "protein")
    result: dict[str, Any] = {}
    for field in STREAM_CORE_FIELDS:
        for alias in aliases:
            key = f"{alias}_{field}"
            if key in source and source[key] not in (None, "", 0):
                result[field] = deepcopy(source[key])
                break
    params: dict[str, Any] = {}
    for field in STREAM_PARAM_FIELDS:
        for alias in aliases:
            key = f"{alias}_{field}"
            if key in source and source[key] not in (None, ""):
                params[field] = deepcopy(source[key])
                break
    if params:
        result["params"] = params
    for alias in aliases:
        entries = source.get(f"{alias}_param", []) or []
        if isinstance(entries, str):
            entries = [entries]
        for entry in entries:
            key, separator, value = str(entry).partition("=")
            key = key.strip()
            if not separator or key not in STREAM_PARAM_FIELDS:
                raise ValueError(f"invalid {alias}_param={entry!r}; expected an allowed key=value")
            value = value.strip()
            if key in _INT_STREAM_PARAMS:
                value = int(value)
            elif key in _FLOAT_STREAM_PARAMS:
                value = float(value)
            result.setdefault("params", {})[key] = value
    return result


def normalize_stream_spec(
    raw: Mapping[str, Any] | None, fallback: Mapping[str, Any]
) -> dict[str, Any]:
    spec = deepcopy(dict(fallback))
    incoming = dict(raw or {})
    fallback_tokenizer = normalize_tokenizer_name(spec.get("tokenizer", "esm2"))
    incoming_params = dict(incoming.pop("params", {}) or {})
    for field in STREAM_CORE_FIELDS:
        if field in incoming and incoming[field] not in (None, "", 0):
            spec[field] = deepcopy(incoming[field])
    params = dict(spec.get("params", {}) or {})
    params.update(incoming_params)
    for field in STREAM_PARAM_FIELDS:
        if field in incoming and incoming[field] not in (None, ""):
            params[field] = deepcopy(incoming[field])

    tokenizer = normalize_tokenizer_name(spec.get("tokenizer", "esm2"))
    expected_route = "pretrained" if tokenizer in PRETRAINED_TOKENIZERS else "classic"
    if tokenizer != fallback_tokenizer and "feature_route" not in incoming:
        spec["feature_route"] = expected_route
    feature_route = str(spec.get("feature_route", "") or "").strip().lower()
    if feature_route and feature_route != expected_route:
        raise ValueError(f"tokenizer={tokenizer} requires feature_route={expected_route}")
    spec["feature_route"] = expected_route
    spec["tokenizer"] = tokenizer
    if tokenizer == "kmer":
        spec["max_length_unit"] = KMER_MAX_LENGTH_UNIT
    elif tokenizer == "wordpiece":
        spec["max_length_unit"] = WORDPIECE_MAX_LENGTH_UNIT
    else:
        spec.pop("max_length_unit", None)
    if tokenizer != fallback_tokenizer and "encoder" not in incoming:
        spec["encoder"] = tokenizer if expected_route == "pretrained" else "index"
    spec["encoder"], _legacy = normalize_encoder_for_tokenizer(tokenizer, spec.get("encoder"))
    if spec["encoder"] in COVARIANCE_ENCODERS:
        lag_key = f"{spec['encoder']}_lag"
        params[lag_key] = validate_covariance_lag(params.get(lag_key, 10))
        params["covariance_max_raw_length"] = validate_covariance_max_raw_length(
            params.get(
                "covariance_max_raw_length",
                COVARIANCE_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    else:
        params.pop("covariance_max_raw_length", None)
    if spec["encoder"] in PDT_ENCODERS:
        params["pdt_lag"] = validate_pdt_lag(params.get("pdt_lag", PDT_DEFAULT_LAG))
        params["pdt_max_raw_length"] = validate_pdt_max_raw_length(
            params.get("pdt_max_raw_length", PDT_DEFAULT_MAX_RAW_LENGTH)
        )
    else:
        params.pop("pdt_max_raw_length", None)
    if spec["encoder"] in PSSM_COVARIANCE_ENCODERS:
        lag_key = f"{spec['encoder']}_lag"
        params[lag_key] = validate_pssm_covariance_lag(
            params.get(lag_key, PSSM_COVARIANCE_DEFAULT_LAG)
        )
    if tokenizer in {"kmer", "dr", "top-n-gram", "dt"} and spec["encoder"] in {"bow", "tfidf"}:
        params["frequency_max_raw_length"] = validate_frequency_max_raw_length(
            params.get(
                "frequency_max_raw_length",
                FREQUENCY_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    else:
        params.pop("frequency_max_raw_length", None)
    if tokenizer == "dr":
        params["dr_max_distance"] = validate_dr_max_distance(
            params.get("dr_max_distance", DR_DEFAULT_MAX_DISTANCE)
        )
        params["dr_max_raw_length"] = validate_dr_max_raw_length(
            params.get("dr_max_raw_length", DR_DEFAULT_MAX_RAW_LENGTH)
        )
        params["dr_unknown_residue_policy"] = validate_dr_unknown_residue_policy(
            params.get(
                "dr_unknown_residue_policy",
                DR_DEFAULT_UNKNOWN_RESIDUE_POLICY,
            )
        )
        spec["max_length_unit"] = (
            DR_GLOBAL_MAX_LENGTH_UNIT
            if spec["encoder"] in {"bow", "tfidf", "textrank"}
            else DR_ORDERED_MAX_LENGTH_UNIT
        )
    else:
        params.pop("dr_max_distance", None)
        params.pop("dr_max_raw_length", None)
        params.pop("dr_unknown_residue_policy", None)
    if tokenizer == "esm3":
        params["esm3_sdpa_backend"] = validate_esm3_sdpa_backend(
            params.get("esm3_sdpa_backend", ESM3_DEFAULT_SDPA_BACKEND)
        )
    else:
        params.pop("esm3_sdpa_backend", None)
    if tokenizer == "top-n-gram":
        params["top_ngram_n"] = validate_top_ngram_n(params.get("top_ngram_n", 2))
        params["topngram_dense_memory_limit_gib"] = validate_top_ngram_dense_memory_limit_gib(
            params.get(
                "topngram_dense_memory_limit_gib",
                TOP_N_GRAM_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
            )
        )
        spec["max_length_unit"] = TOP_N_GRAM_MAX_LENGTH_UNIT
    else:
        params.pop("top_ngram_n", None)
        params.pop("topngram_dense_memory_limit_gib", None)
    if tokenizer == "dt":
        params["dt_max_distance"] = validate_dt_max_distance(
            params.get("dt_max_distance", DT_DEFAULT_MAX_DISTANCE)
        )
        params["dt_dense_memory_limit_gib"] = validate_dt_dense_memory_limit_gib(
            params.get(
                "dt_dense_memory_limit_gib",
                DT_DEFAULT_DENSE_MEMORY_LIMIT_GIB,
            )
        )
        spec["max_length_unit"] = DT_MAX_LENGTH_UNIT
    else:
        params.pop("dt_max_distance", None)
        params.pop("dt_dense_memory_limit_gib", None)
    if tokenizer == "wordpiece":
        params["wordpiece_vocab_size"] = validate_wordpiece_vocab_size(
            params.get("wordpiece_vocab_size", 1000)
        )
        params["wordpiece_max_raw_length"] = validate_wordpiece_max_raw_length(
            params.get(
                "wordpiece_max_raw_length",
                WORDPIECE_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    else:
        params.pop("wordpiece_vocab_size", None)
        params.pop("wordpiece_max_raw_length", None)
    if tokenizer == "unigram":
        params["unigram_vocab_size"] = validate_unigram_vocab_size(
            params.get("unigram_vocab_size", 1000)
        )
        params["unigram_max_raw_length"] = validate_unigram_max_raw_length(
            params.get(
                "unigram_max_raw_length",
                UNIGRAM_DEFAULT_MAX_RAW_LENGTH,
            )
        )
    else:
        params.pop("unigram_vocab_size", None)
        params.pop("unigram_max_raw_length", None)
    spec["stream_model"] = str(spec.get("stream_model", "") or "").strip().lower()
    requested_max_length = int(spec.get("max_length") or 512)
    if tokenizer == "prott5":
        if requested_max_length == PROTT5_MAX_RESIDUES + 1:
            spec["max_length_requested"] = requested_max_length
            spec["max_length_adjustment"] = {
                "requested": requested_max_length,
                "effective": PROTT5_MAX_RESIDUES,
                "reason": "T5 n_positions includes the EOS token",
            }
            requested_max_length = PROTT5_MAX_RESIDUES
        else:
            requested_max_length = validate_prott5_max_residues(
                requested_max_length,
                field="stream.max_length",
            )
    spec["max_length"] = requested_max_length
    if spec["max_length"] < 1:
        raise ValueError("stream max_length must be >= 1")
    if tokenizer == "unigram" and spec["max_length"] < 2:
        raise ValueError("Unigram stream max_length must be >= 2 for <CLS>/<SEP>")
    if expected_route == "pretrained":
        if tokenizer != fallback_tokenizer and "model_source" not in incoming:
            spec["model_source"] = PRETRAINED_MODEL_DEFAULTS.get(tokenizer, "")
        spec["model_source"] = str(
            spec.get("model_source", "") or PRETRAINED_MODEL_DEFAULTS.get(tokenizer, "")
        ).strip()
    else:
        spec["model_source"] = ""
    spec["params"] = params
    return spec


def normalize_pair_streams(
    source: Mapping[str, Any] | Namespace, task_type: str | None = None
) -> dict[str, Any]:
    task_type = normalize_task_type(task_type)
    raw = _mapping(source)
    task = normalize_task_type(task_type or raw.get("task_type", "pep_protein_interaction"))
    if normalize_task_type(task) not in PAIR_TASKS:
        return {}
    base = _legacy_stream(raw)
    nested = raw.get("streams") if isinstance(raw.get("streams"), Mapping) else {}
    result: dict[str, dict[str, Any]] = {}
    for role in STREAM_ROLES:
        role_raw: dict[str, Any] = {}
        if isinstance(nested, Mapping) and isinstance(nested.get(role), Mapping):
            role_raw.update(dict(nested[role]))
        role_raw.update(_role_overrides(raw, role))
        result[role] = normalize_stream_spec(role_raw, base)
    return result


def stream_mode(streams: Mapping[str, Mapping[str, Any]]) -> str:
    return (
        "homogeneous"
        if dict(streams.get("peptide", {})) == dict(streams.get("partner", {}))
        else "heterogeneous"
    )


def apply_pair_streams(args: Namespace) -> Namespace:
    normalize_stream_namespace(args)
    streams = normalize_pair_streams(args)
    if not streams:
        return args
    args.streams = streams
    args.stream_mode = stream_mode(streams)
    if str(getattr(args, "pair_strategy", "") or "").strip().lower() == "single_stream":
        require_single_stream_homogeneous(streams)
    for role, spec in streams.items():
        for field in STREAM_CORE_FIELDS:
            setattr(args, f"{role}_{field}", spec.get(field))
        for key, value in spec.get("params", {}).items():
            setattr(args, f"{role}_{key}", value)
    if args.stream_mode == "homogeneous":
        shared = streams["peptide"]
        args.tokenizer = shared["tokenizer"]
        args.encoder = shared["encoder"]
        args.max_length = shared["max_length"]
        args.stream_model = shared.get("stream_model", "")
        for key, value in shared.get("params", {}).items():
            setattr(args, key, deepcopy(value))
        for field in MODEL_SOURCE_FIELDS.get(shared["tokenizer"], ()):
            setattr(args, field, shared.get("model_source", ""))
    return args


def namespace_for_stream(args: Namespace, role: str) -> Namespace:
    normalize_stream_namespace(args)
    if role not in STREAM_ROLES:
        raise ValueError(f"unsupported stream role: {role}")
    streams = getattr(args, "streams", None) or normalize_pair_streams(args)
    spec = deepcopy(streams[role])
    values = deepcopy(vars(args))
    values["tokenizer"] = spec["tokenizer"]
    values["encoder"] = spec["encoder"]
    values["max_length"] = spec["max_length"]
    values["stream_model"] = spec.get("stream_model", "")
    for key, value in spec.get("params", {}).items():
        values[key] = deepcopy(value)
    model_source = str(spec.get("model_source", "") or "")
    for field in MODEL_SOURCE_FIELDS.get(spec["tokenizer"], ()):
        values[field] = model_source
    values["_stream_role"] = role
    values["_stream_spec"] = spec
    return Namespace(**values)


def canonical_pair_config(
    source: Mapping[str, Any] | Namespace, task_type: str | None = None
) -> dict[str, Any]:
    task_type = normalize_task_type(task_type)
    raw = _mapping(source)
    if raw.get("pair_strategy"):
        from pepplatform.pair_terminology import normalize_pair_strategy

        raw["pair_strategy"] = normalize_pair_strategy(raw["pair_strategy"])
    streams = normalize_pair_streams(raw, task_type=task_type)
    if not streams:
        return raw
    raw["streams"] = streams
    raw["stream_mode"] = stream_mode(streams)
    return raw


def flatten_streams_for_cli(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return a CLI-friendly config with canonical streams expanded by role."""
    config = canonical_pair_config(source)
    streams = config.pop("streams", {}) or {}
    config.pop("stream_mode", None)
    for role in STREAM_ROLES:
        spec = dict(streams.get(role, {}) or {})
        params = dict(spec.pop("params", {}) or {})
        for field in STREAM_CORE_FIELDS:
            value = spec.get(field)
            if value not in (None, ""):
                config[f"{role}_{field}"] = value
        for key, value in params.items():
            if value not in (None, ""):
                config[f"{role}_{key}"] = value
    return config


def add_stream_cli_arguments(parser) -> None:
    """Register role-prefixed stream arguments on an argparse parser."""
    for role in STREAM_ROLES:
        aliases = (role,) if role != "partner" else ("partner", "protein")

        def options(field: str) -> list[str]:
            return [f"--{alias}_{field}" for alias in aliases]

        parser.add_argument(
            *options("feature_route"), choices=["classic", "pretrained"], default=""
        )
        parser.add_argument(*options("tokenizer"), type=str, default="")
        parser.add_argument(*options("encoder"), type=str, default="")
        parser.add_argument(*options("stream_model"), type=str, default="")
        parser.add_argument(*options("model_source"), type=str, default="")
        parser.add_argument(*options("max_length"), type=int, default=0)
        for field in STREAM_PARAM_FIELDS:
            parser.add_argument(
                *options(field),
                type=(
                    int
                    if field in _INT_STREAM_PARAMS
                    else (float if field in _FLOAT_STREAM_PARAMS else str)
                ),
                default=None,
            )
        parser.add_argument(*options("param"), action="append", default=[])
