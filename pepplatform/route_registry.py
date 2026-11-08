"""Canonical Tokenizer -> Encoder -> Predictor route descriptions.

This module is the semantic boundary shared by CLI, Web preflight, metadata,
and presentation code.  Runtime-specific argument names remain supported, but
they are resolved into one stable route description before execution.
"""

from __future__ import annotations

from pepplatform.stream_compat import normalize_stream_fields
from pepplatform.task_compat import normalize_task_type


from copy import deepcopy
from typing import Any, Mapping

from pepplatform.workflow import (
    CHAIN_GRAPH_ARCHITECTURES,
    CLASSIC_ENCODER_OPTIONS,
    CLASSIC_TOKENIZERS,
    GLOBAL_ENCODERS,
    GRAPH_PREDICTORS,
    PRETRAINED_TOKENIZERS,
    RESIDUE_ADAPTER_DIM_DEFAULT,
    SEQUENCE_ARCHITECTURES,
    SEQUENCE_MODELS,
    SEQUENCE_ENCODERS,
    default_prediction_head_hidden_dims,
    ensure_encoder_available,
    normalize_tokenizer_name,
    parse_prediction_head_hidden_dims,
    prediction_head_applies,
    validate_stream_alignment_dim,
)
from pepplatform.pair_strategy import SINGLE_STREAM_JOIN_MODE
from pepplatform.pair_terminology import normalize_pair_strategy


ROUTE_SCHEMA_VERSION = 6
LEGACY_PRETRAINED_ENCODERS = {"index", "ppi_fusion"}
PAIR_TASKS = {"pep_protein_interaction", "pep_hlaii", "pep_hlai"}

OUTPUT_KINDS = [
    "token_ids",
    "position_features",
    "global_features",
    "sequence_embedding",
    "residue_embeddings",
]

ROUTE_FAMILIES = [
    {
        "id": "sequence.classic_global",
        "task_scope": "sequence",
        "label": "Classic global features",
    },
    {
        "id": "sequence.classic_positional",
        "task_scope": "sequence",
        "label": "Classic positional modeling",
    },
    {
        "id": "sequence.classic_chain_graph",
        "task_scope": "sequence",
        "label": "Classic sequence-chain graph",
    },
    {
        "id": "sequence.pretrained_pooled",
        "task_scope": "sequence",
        "label": "Pretrained pooled representation",
    },
    {
        "id": "sequence.pretrained_residue_sequence",
        "task_scope": "sequence",
        "label": "Pretrained residue sequence modeling",
    },
    {
        "id": "sequence.pretrained_residue_graph",
        "task_scope": "sequence",
        "label": "Pretrained residue-chain graph",
    },
    {
        "id": "pair.shared_sequence",
        "task_scope": "pair",
        "label": "Pair shared sequence modeling",
    },
    {
        "id": "pair.dual_vector",
        "task_scope": "pair",
        "label": "Pair Two-stream vector interaction",
    },
    {
        "id": "pair.dual_token_attention",
        "task_scope": "pair",
        "label": "Pair Two-stream token interaction",
    },
    {
        "id": "pair.sample_graph",
        "task_scope": "pair",
        "label": "Pair sample-graph prediction",
    },
    {
        "id": "pair.specialized_dual_sequence",
        "task_scope": "pair",
        "label": "Specialized dual-sequence modeling",
    },
]

DISPLAY_NAMES = {
    "identity": "Identity",
    "index": "Index",
    "onehot": "One-hot",
    "word2vec": "Word2Vec",
    "glove": "GloVe",
    "fasttext": "FastText",
    "esm2": "ESM2",
    "protbert": "ProtBERT",
    "prott5": "ProtT5",
    "esmc": "ESM-C",
    "esm3": "ESM3",
    "cnn": "CNN",
    "rnn": "RNN",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "gcn": "GCN",
    "gat": "GAT",
    "hgt": "HGT",
    "sage": "GraphSAGE",
    "graph_transformer": "Graph Transformer",
    "cross_attn": "Cross-Attention",
    "bi_cross_attn": "Bi-Cross-Attention",
    "mlp": "MLP",
    "linear": "Linear",
    "svm": "SVM",
    "calibrated_svm": "Calibrated SVM",
    "rf": "Random Forest",
    "extratrees": "Extra Trees",
    "histgb": "Histogram Gradient Boosting",
    "xgb": "XGBoost",
    "lgb": "LightGBM",
    "logreg": "Logistic Regression",
    "learned_embedding": "Learned Embedding",
    "linear_projection": "Linear Projection",
    "vector_fusion": "Vector Fusion",
    "aligned_gated_fusion": "Aligned Gated Fusion",
    "flatten": "Flatten",
    "flatten_vector_fusion": "Flatten + Vector Fusion",
    "global_max": "Masked Global Max Pooling",
    "last_hidden": "Last Hidden State",
    "masked_mean": "Masked Mean Pooling",
    "mean_max": "Mean + Max Pooling",
    "node": "Node-level Readout",
    "amino_acid_residue": "Amino-acid Residues",
    "hla_pseudosequence_lookup": "HLA-I Pseudosequence Lookup",
    "blosum50": "BLOSUM50",
    "immuneapp_el_cnn_bilstm_dense": "ImmuneApp-EL CNN-BiLSTM + Dense Path",
    "dual_attention_flatten_fusion": "Dual Attention + Flatten Fusion",
    "unidirectional_cross_attention_pair_fusion": "Unidirectional Cross-Attention Pair Fusion",
    "bidirectional_cross_attention_pair_fusion": "Bidirectional Cross-Attention Pair Fusion",
    "dense_32_sigmoid": "Dense-32 + Sigmoid",
    "legacy_gated_residual": "Legacy Gated Residual",
}


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _label(component_id: str) -> str:
    if component_id in DISPLAY_NAMES:
        return DISPLAY_NAMES[component_id]
    return component_id.replace("_", " ").replace("-", " ").title()


def _component(component_id: str, **details: Any) -> dict[str, Any]:
    component_id = component_id or "identity"
    result = {"id": component_id, "label": _label(component_id)}
    result.update({key: value for key, value in details.items() if value not in (None, "")})
    return result


def normalize_encoder_for_tokenizer(tokenizer: Any, encoder: Any) -> tuple[str, bool]:
    """Return the canonical encoder and whether a legacy placeholder was used."""

    tokenizer_id = normalize_tokenizer_name(tokenizer)
    encoder_id = _text(encoder)
    ensure_encoder_available(encoder_id)
    if tokenizer_id in PRETRAINED_TOKENIZERS:
        if not encoder_id:
            return tokenizer_id, False
        if encoder_id in LEGACY_PRETRAINED_ENCODERS:
            return tokenizer_id, True
        if encoder_id != tokenizer_id:
            raise ValueError(
                f"pretrained tokenizer={tokenizer_id} requires encoder={tokenizer_id}; "
                f"got encoder={encoder_id}"
            )
        return encoder_id, False
    if encoder_id == "ppi_fusion" or encoder_id in PRETRAINED_TOKENIZERS:
        raise ValueError(
            f"classic tokenizer={tokenizer_id or '<empty>'} cannot use encoder={encoder_id}"
        )
    return encoder_id or "index", False


def canonicalize_encoder_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize top-level and pair-stream encoder identifiers."""

    result = deepcopy(dict(config))
    tokenizer = normalize_tokenizer_name(result.get("tokenizer", ""))
    if tokenizer:
        result["tokenizer"] = tokenizer
        result["encoder"], _legacy = normalize_encoder_for_tokenizer(
            tokenizer, result.get("encoder")
        )

    streams = result.get("streams")
    if isinstance(streams, Mapping):
        normalized_streams: dict[str, Any] = {}
        for role, raw_stream in streams.items():
            if not isinstance(raw_stream, Mapping):
                normalized_streams[str(role)] = deepcopy(raw_stream)
                continue
            stream = deepcopy(dict(raw_stream))
            stream_tokenizer = normalize_tokenizer_name(stream.get("tokenizer", tokenizer))
            stream["tokenizer"] = stream_tokenizer
            stream["encoder"], _legacy = normalize_encoder_for_tokenizer(
                stream_tokenizer,
                stream.get("encoder"),
            )
            normalized_streams[str(role)] = stream
        result["streams"] = normalized_streams
    return result


def legacy_encoder_messages(raw_config: Mapping[str, Any]) -> list[str]:
    """Describe legacy pretrained encoder placeholders without mutating input."""

    messages: list[str] = []

    def inspect(label: str, source: Mapping[str, Any]) -> None:
        tokenizer = normalize_tokenizer_name(source.get("tokenizer", ""))
        encoder = _text(source.get("encoder"))
        if tokenizer in PRETRAINED_TOKENIZERS and encoder in LEGACY_PRETRAINED_ENCODERS:
            messages.append(
                f"{label}encoder={encoder} is deprecated for tokenizer={tokenizer}; "
                f"use encoder={tokenizer}"
            )

    inspect("", raw_config)
    streams = raw_config.get("streams")
    if isinstance(streams, Mapping):
        for role in ("peptide", "partner"):
            stream = streams.get(role)
            if isinstance(stream, Mapping):
                inspect(f"{role} stream ", stream)
    for role in ("peptide", "partner"):
        tokenizer_key = f"{role}_tokenizer"
        encoder_key = f"{role}_encoder"
        if tokenizer_key in raw_config or encoder_key in raw_config:
            before = len(messages)
            inspect(
                f"{role} stream ",
                {
                    "tokenizer": raw_config.get(tokenizer_key, raw_config.get("tokenizer", "")),
                    "encoder": raw_config.get(encoder_key, ""),
                },
            )
            if len(messages) > before and messages[-1] in messages[:-1]:
                messages.pop()
    return messages


def _encoder_output(tokenizer: str, encoder: str, *, token_level: bool = False) -> str:
    if tokenizer in PRETRAINED_TOKENIZERS:
        return "residue_embeddings" if token_level else "sequence_embedding"
    if encoder == "index":
        return "token_ids"
    if encoder in GLOBAL_ENCODERS:
        return "global_features"
    return "position_features"


def _readout_for_architecture(architecture: str) -> str:
    if architecture == "cnn":
        return "global_max"
    if architecture in {"rnn", "gru", "lstm"}:
        return "last_hidden"
    if architecture == "transformer":
        return "masked_mean"
    if architecture in CHAIN_GRAPH_ARCHITECTURES:
        return "mean_max"
    if architecture == "cross_attn":
        return "unidirectional_cross_attention_pair_fusion"
    if architecture == "bi_cross_attn":
        return "bidirectional_cross_attention_pair_fusion"
    if architecture in GRAPH_PREDICTORS:
        return "node"
    return "identity"


def _input_adapter(output_kinds: list[str], architecture: str, fusion_mode: str = "") -> str:
    if architecture == "hgt":
        return "typed_role_projection"
    if len(output_kinds) > 1 and fusion_mode == "aligned_gated":
        return "aligned_gated_fusion"
    if len(output_kinds) > 1 and architecture not in {"cross_attn", "bi_cross_attn"}:
        if any(output in {"token_ids", "position_features"} for output in output_kinds):
            return "flatten_vector_fusion"
        return "vector_fusion"
    if "token_ids" in output_kinds:
        return "learned_embedding"
    if architecture in {"cnn", "transformer", "cross_attn", "bi_cross_attn"} and any(
        output in {"position_features", "residue_embeddings"} for output in output_kinds
    ):
        return "linear_projection"
    return "identity"


def _effective_classifier(config: Mapping[str, Any]) -> str:
    classifier = _text(config.get("classifier"))
    model = _text(config.get("model"))
    requested = classifier or model or "mlp"
    # ``calibrated_svm`` was the historical public name.  Keep it
    # recognizable in the source config, but resolve the route to the
    # canonical SVM predictor so all route consumers share one identity.
    return "svm" if requested == "calibrated_svm" else requested


def _prediction_head_component(config: Mapping[str, Any], classifier: str) -> dict[str, Any]:
    if classifier == "svm":
        raw_classifier = _text(config.get("classifier"))
        raw_model = _text(config.get("model"))
        legacy_alias = raw_classifier == "calibrated_svm" or raw_model == "calibrated_svm"
        calibration = _text(config.get("svm_calibration"))
        if legacy_alias and not calibration:
            calibration = "sigmoid"
        return _component(
            classifier,
            calibration=calibration or "none",
            legacy_alias="calibrated_svm" if legacy_alias else None,
        )
    if classifier != "mlp" or not prediction_head_applies(config):
        return _component(classifier)
    hidden_dims = parse_prediction_head_hidden_dims(config.get("head_hidden_dims"))
    if hidden_dims is None:
        hidden_dims = default_prediction_head_hidden_dims(config)
    if not hidden_dims:
        return _component("linear", hidden_dims=[])
    raw_dropout = config.get("head_dropout")
    if raw_dropout in [None, ""]:
        raw_dropout = config.get("dropout", 0.0)
    return _component(
        "mlp",
        hidden_dims=hidden_dims,
        activation=_text(config.get("head_activation")) or "relu",
        dropout=float(raw_dropout or 0.0),
    )


def _single_route(config: Mapping[str, Any], task_type: str) -> dict[str, Any]:
    task_type = normalize_task_type(task_type)
    tokenizer = normalize_tokenizer_name(config.get("tokenizer", "kmer"))
    encoder, _legacy = normalize_encoder_for_tokenizer(tokenizer, config.get("encoder"))
    architecture = _text(config.get("sequence_model"))
    classifier = _effective_classifier(config)
    pretrained = tokenizer in PRETRAINED_TOKENIZERS
    pretrained_residue = pretrained and architecture in SEQUENCE_MODELS
    residue_graph = pretrained_residue and architecture in CHAIN_GRAPH_ARCHITECTURES
    output_kind = _encoder_output(tokenizer, encoder, token_level=pretrained_residue)

    if residue_graph:
        route_id = "sequence.pretrained_residue_graph"
    elif pretrained_residue:
        route_id = "sequence.pretrained_residue_sequence"
    elif pretrained:
        route_id = "sequence.pretrained_pooled"
        architecture = "identity"
    elif architecture in CHAIN_GRAPH_ARCHITECTURES:
        route_id = "sequence.classic_chain_graph"
    elif architecture in SEQUENCE_ARCHITECTURES:
        route_id = "sequence.classic_positional"
    else:
        route_id = "sequence.classic_global"
        architecture = "identity"

    adapter = _input_adapter([output_kind], architecture)
    if pretrained_residue:
        adapter = "linear_projection"
    if route_id == "sequence.classic_global" and output_kind in {"token_ids", "position_features"}:
        adapter = "flatten"
    return {
        "schema_version": ROUTE_SCHEMA_VERSION,
        "route_id": route_id,
        "task_type": task_type,
        "streams": [
            {
                "role": "sequence",
                "feature_route": "pretrained" if pretrained else "classic",
                "tokenizer": _component(tokenizer),
                "encoder": _component(
                    encoder,
                    output_kind=output_kind,
                    transformation=_text(config.get("reduction")) or "none",
                    frozen=pretrained,
                ),
            }
        ],
        "predictor": {
            "input_adapter": _component(
                adapter,
                learnable=adapter in {"learned_embedding", "linear_projection"},
                normalization="layer_norm" if pretrained_residue else "",
                output_dimension=int(
                    config.get("residue_adapter_dim") or RESIDUE_ADAPTER_DIM_DEFAULT
                )
                if pretrained_residue
                else None,
            ),
            "architecture": _component(architecture),
            "readout": _component(_readout_for_architecture(architecture)),
            "head": _prediction_head_component(config, classifier),
        },
        "training_mode": _text(config.get("training_mode")) or "direct_fit",
    }


def _stream_specs(config: Mapping[str, Any], *, token_level: bool) -> list[dict[str, Any]]:
    streams = config.get("streams")
    if not isinstance(streams, Mapping):
        streams = {
            "peptide": {
                "tokenizer": config.get("tokenizer", "esm2"),
                "encoder": config.get("encoder"),
            },
            "partner": {
                "tokenizer": config.get("tokenizer", "esm2"),
                "encoder": config.get("encoder"),
            },
        }
    result: list[dict[str, Any]] = []
    for role in ("peptide", "partner"):
        raw = streams.get(role, {})
        if not isinstance(raw, Mapping):
            raw = {}
        tokenizer = normalize_tokenizer_name(raw.get("tokenizer", config.get("tokenizer", "esm2")))
        encoder, _legacy = normalize_encoder_for_tokenizer(
            tokenizer,
            raw.get("encoder", config.get("encoder")),
        )
        output_kind = _encoder_output(tokenizer, encoder, token_level=token_level)
        result.append(
            {
                "role": role,
                "feature_route": "pretrained" if tokenizer in PRETRAINED_TOKENIZERS else "classic",
                "tokenizer": _component(tokenizer),
                "encoder": _component(
                    encoder,
                    output_kind=output_kind,
                ),
            }
        )
    return result


def _pair_route(config: Mapping[str, Any], task_type: str) -> dict[str, Any]:
    task_type = normalize_task_type(task_type)
    pair_strategy = normalize_pair_strategy(config.get("pair_strategy")) or "two_stream"
    interaction_level = _text(config.get("interaction_level")) or "vector"
    predictor_type = _text(config.get("predictor_type")) or "classifier"
    sequence_model = _text(config.get("sequence_model"))
    graph_predictor = _text(config.get("graph_predictor"))
    interaction_model = _text(config.get("interaction_model")) or "cross_attn"
    classifier = _effective_classifier(config)

    if pair_strategy == "single_stream":
        route_id = "pair.shared_sequence"
        architecture = sequence_model or _text(config.get("model")) or "cnn"
        token_level = True
    elif interaction_level == "token":
        route_id = "pair.dual_token_attention"
        architecture = interaction_model
        token_level = True
    elif predictor_type == "sample_graph" or graph_predictor:
        route_id = "pair.sample_graph"
        architecture = graph_predictor or _text(config.get("model")) or "gcn"
        classifier = "mlp"
        token_level = False
    else:
        route_id = "pair.dual_vector"
        architecture = "identity"
        token_level = False

    streams = _stream_specs(config, token_level=token_level)
    output_kinds = [stream["encoder"]["output_kind"] for stream in streams]
    fusion_mode = _text(config.get("fusion_mode"))
    adapter = _input_adapter(output_kinds, architecture, fusion_mode)
    if pair_strategy == "single_stream" and output_kinds:
        adapter = _input_adapter(output_kinds[:1], architecture)
    if route_id == "pair.sample_graph":
        readout = "typed_edge" if architecture == "hgt" else "node"
    else:
        readout = _readout_for_architecture(architecture)
    return {
        "schema_version": ROUTE_SCHEMA_VERSION,
        "route_id": route_id,
        "task_type": task_type,
        "streams": streams,
        "predictor": {
            "input_adapter": _component(
                adapter,
                learnable=adapter
                in {"learned_embedding", "linear_projection", "aligned_gated_fusion"},
                fusion_mode=fusion_mode
                if adapter in {"vector_fusion", "flatten_vector_fusion", "aligned_gated_fusion"}
                else "",
                alignment_dimension=validate_stream_alignment_dim(
                    config.get("stream_alignment_dim", 256)
                )
                if adapter == "aligned_gated_fusion"
                else None,
                projection_policy="identity_for_equal_width_else_separate_linear"
                if adapter == "aligned_gated_fusion"
                else "",
                input_blocks=[
                    "peptide_aligned",
                    "partner_aligned",
                    "absolute_difference",
                    "hadamard_product",
                ]
                if adapter == "aligned_gated_fusion"
                else None,
                output_blocks=[
                    "convex_elementwise_mix",
                    "absolute_difference",
                    "hadamard_product",
                ]
                if adapter == "aligned_gated_fusion"
                else None,
                join_mode=SINGLE_STREAM_JOIN_MODE if pair_strategy == "single_stream" else "",
            ),
            "architecture": _component(architecture),
            "readout": _component(readout),
            "head": _prediction_head_component(config, classifier),
        },
        "training_mode": _text(config.get("training_mode")) or "direct_fit",
    }


def resolve_effective_route(
    config: Mapping[str, Any], task_type: str | None = None
) -> dict[str, Any]:
    """Resolve a normalized or legacy config into the stable three-layer contract."""

    task_type = normalize_task_type(task_type)
    task = normalize_task_type(task_type or config.get("task_type")) or "pep_function"
    canonical = canonicalize_encoder_config(normalize_stream_fields(dict(config)))
    supported_tokenizers = set(CLASSIC_TOKENIZERS) | set(PRETRAINED_TOKENIZERS)
    tokenizers = [normalize_tokenizer_name(canonical.get("tokenizer", ""))]
    streams = canonical.get("streams")
    if isinstance(streams, Mapping):
        tokenizers.extend(
            normalize_tokenizer_name(stream.get("tokenizer", ""))
            for stream in streams.values()
            if isinstance(stream, Mapping)
        )
    unsupported = sorted(
        {
            tokenizer
            for tokenizer in tokenizers
            if tokenizer and tokenizer not in supported_tokenizers
        }
    )
    if unsupported:
        raise ValueError(f"unsupported tokenizer: {', '.join(unsupported)}")
    if normalize_task_type(task) in PAIR_TASKS:
        return _pair_route(canonical, task)
    return _single_route(canonical, task)


def annotate_legacy_prediction_head(
    route: Mapping[str, Any], metadata: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Correct route presentation for bundles saved with the old gated head."""
    result = deepcopy(dict(route))
    artifact = metadata if isinstance(metadata, Mapping) else {}
    predictor = result.get("predictor")
    pair = artifact.get("pair") if isinstance(artifact.get("pair"), Mapping) else {}
    if (
        isinstance(predictor, dict)
        and str(pair.get("pair_strategy", "") or "").strip().lower() == "single_stream"
        and artifact.get("pair_strategy_contract") is None
    ):
        adapter = predictor.get("input_adapter")
        if isinstance(adapter, dict):
            adapter["join_mode"] = "legacy_encoded_sequence_join"
    if artifact.get("prediction_head") is not None:
        return result
    if str(artifact.get("bundle_type", "")) not in {
        "pretrained_embedding_nn",
        "pair_vector_nn",
        "pair_embedding_nn",
    }:
        return result
    if isinstance(predictor, dict):
        predictor["head"] = _component("legacy_gated_residual")
        result["schema_version"] = ROUTE_SCHEMA_VERSION
    return result


def resolve_registered_route(
    template_id: str,
    *,
    training_mode: str = "controlled_retrain_25_model_ensemble",
) -> dict[str, Any]:
    """Resolve a centrally registered specialized route template.

    Specialized external model routes are not selectable platform backends.
    Their saved models use the same versioned route contract as Web and CLI runs.
    """

    normalized = _text(template_id)
    if normalized != "immuneapp_el":
        raise ValueError(f"unknown registered route template: {template_id}")
    return {
        "schema_version": ROUTE_SCHEMA_VERSION,
        "route_id": "pair.specialized_dual_sequence",
        "task_type": "pep_hlai",
        "streams": [
            {
                "role": "peptide",
                "feature_route": "classic",
                "tokenizer": _component(
                    "amino_acid_residue",
                    alignment="15 N-aligned + 15 C-aligned",
                    encoded_length=30,
                ),
                "encoder": _component(
                    "blosum50",
                    output_kind="position_features",
                    output_shape=[30, 20],
                ),
            },
            {
                "role": "partner",
                "feature_route": "classic",
                "tokenizer": _component(
                    "hla_pseudosequence_lookup",
                    encoded_length=34,
                ),
                "encoder": _component(
                    "blosum50",
                    output_kind="position_features",
                    output_shape=[34, 20],
                ),
            },
        ],
        "predictor": {
            "input_adapter": _component("identity", learnable=False),
            "architecture": _component(
                "immuneapp_el_cnn_bilstm_dense",
                conv_channels=128,
                bidirectional_lstm_units=64,
                dense_path=[256, 128, 64],
            ),
            "readout": _component("dual_attention_flatten_fusion"),
            "head": _component("dense_32_sigmoid"),
        },
        "training_mode": _text(training_mode) or "controlled_retrain_25_model_ensemble",
    }


def route_capabilities() -> dict[str, Any]:
    """Return serializable route metadata for Web clients."""

    encoder_ids = sorted(
        set(PRETRAINED_TOKENIZERS)
        | set(SEQUENCE_ENCODERS)
        | set(GLOBAL_ENCODERS)
        | {encoder for values in CLASSIC_ENCODER_OPTIONS.values() for encoder in values}
    )
    return {
        "route_schema_version": ROUTE_SCHEMA_VERSION,
        "output_kinds": list(OUTPUT_KINDS),
        "route_families": deepcopy(ROUTE_FAMILIES),
        "encoder_specs": [
            {
                "id": encoder,
                "label": _label(encoder),
                "family": "pretrained"
                if encoder in PRETRAINED_TOKENIZERS
                else "global"
                if encoder in GLOBAL_ENCODERS
                else "sequence",
            }
            for encoder in encoder_ids
        ],
        "predictor_parts": ["input_adapter", "architecture", "readout", "head"],
        "defaults": {
            "pep_function": {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
            "pep_hlaii": {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
            "pep_protein_interaction": {"tokenizer": "esm2", "encoder": "esm2", "model": "mlp"},
            "pep_hlai": {"tokenizer": "esm2", "encoder": "esm2", "model": "mlp"},
        },
    }


def format_effective_route(route: Mapping[str, Any]) -> list[str]:
    """Return concise log lines for a resolved route."""

    stream_parts: list[str] = []
    for stream in route.get("streams", []):
        role = str(stream.get("role", "sequence"))
        tokenizer = stream.get("tokenizer", {})
        encoder = stream.get("encoder", {})
        stream_parts.append(
            f"{role}: Tokenizer={tokenizer.get('label', tokenizer.get('id', '-'))} -> "
            f"Encoder={encoder.get('label', encoder.get('id', '-'))} "
            f"({encoder.get('output_kind', '-')})"
        )
    predictor = route.get("predictor", {})
    predictor_parts = [
        f"Adapter={predictor.get('input_adapter', {}).get('label', '-')}",
        f"Architecture={predictor.get('architecture', {}).get('label', '-')}",
        f"Readout={predictor.get('readout', {}).get('label', '-')}",
        f"Head={predictor.get('head', {}).get('label', '-')}",
    ]
    return [
        f"[effective-route] {route.get('route_id', 'unknown')} | " + " | ".join(stream_parts),
        "[effective-route] Predictor: " + " -> ".join(predictor_parts),
    ]


from pepplatform.stream_compat import install_legacy_exports as _install_legacy_stream_exports
_install_legacy_stream_exports(globals())
