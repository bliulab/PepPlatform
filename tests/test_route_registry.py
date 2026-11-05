import pytest

from pepplatform.route_registry import (
    ROUTE_SCHEMA_VERSION,
    canonicalize_encoder_config,
    legacy_encoder_messages,
    normalize_encoder_for_tokenizer,
    resolve_effective_route,
    resolve_registered_route,
    route_capabilities,
)
from pepplatform.web.config import normalize_experiment_config_with_sources
from pepplatform.web.preflight import preflight_experiment


@pytest.mark.parametrize("model_id", ["esm2", "protbert", "prott5", "esmc", "esm3"])
def test_pretrained_encoder_is_explicit_and_matches_tokenizer(model_id):
    assert normalize_encoder_for_tokenizer(model_id, None) == (model_id, False)
    assert normalize_encoder_for_tokenizer(model_id, "index") == (model_id, True)
    assert normalize_encoder_for_tokenizer(model_id, "ppi_fusion") == (model_id, True)
    assert normalize_encoder_for_tokenizer(model_id, model_id) == (model_id, False)


def test_mismatched_pretrained_encoder_is_rejected():
    with pytest.raises(ValueError, match="requires encoder=esm2"):
        normalize_encoder_for_tokenizer("esm2", "prott5")


def test_index_embedding_is_a_predictor_input_adapter():
    route = resolve_effective_route(
        {
            "task_type": "pep_function",
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cnn",
            "sequence_model": "cnn",
            "classifier": "mlp",
            "training_mode": "end_to_end",
        }
    )

    assert route["route_id"] == "sequence.classic_positional"
    assert route["streams"][0]["encoder"]["output_kind"] == "token_ids"
    assert route["predictor"]["input_adapter"]["id"] == "learned_embedding"
    assert route["predictor"]["architecture"]["id"] == "cnn"
    assert route["predictor"]["readout"]["id"] == "global_max"


def test_static_embedding_remains_an_encoder():
    route = resolve_effective_route(
        {
            "task_type": "pep_function",
            "tokenizer": "kmer",
            "encoder": "word2vec",
            "model": "rnn",
            "sequence_model": "rnn",
            "classifier": "mlp",
        }
    )

    assert route["streams"][0]["encoder"]["id"] == "word2vec"
    assert route["streams"][0]["encoder"]["output_kind"] == "position_features"
    assert route["predictor"]["input_adapter"]["id"] == "identity"


def test_flattening_position_features_is_a_predictor_input_adapter():
    route = resolve_effective_route(
        {
            "task_type": "pep_function",
            "tokenizer": "kmer",
            "encoder": "onehot",
            "model": "svm",
            "classifier": "svm",
        }
    )

    assert route["route_id"] == "sequence.classic_global"
    assert route["streams"][0]["encoder"]["output_kind"] == "position_features"
    assert route["predictor"]["input_adapter"]["id"] == "flatten"


def test_direct_classifier_flattens_index_ids_without_claiming_learned_embedding():
    route = resolve_effective_route(
        {
            "task_type": "pep_function",
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "svm",
            "classifier": "svm",
        }
    )

    assert route["streams"][0]["encoder"]["output_kind"] == "token_ids"
    assert route["predictor"]["input_adapter"]["id"] == "flatten"


def test_pair_vector_classic_features_are_flattened_before_fusion():
    route = resolve_effective_route(
        {
            "task_type": "pep_protein_interaction",
            "tokenizer": "kmer",
            "encoder": "index",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "model": "mlp",
            "classifier": "mlp",
        }
    )

    assert [stream["encoder"]["output_kind"] for stream in route["streams"]] == [
        "token_ids",
        "token_ids",
    ]
    assert route["predictor"]["input_adapter"]["id"] == "flatten_vector_fusion"


def test_hgt_route_exposes_typed_adapter_and_edge_readout():
    route = resolve_effective_route(
        {
            "task_type": "pep_hlai",
            "tokenizer": "prott5",
            "encoder": "prott5",
            "pair_strategy": "two_stream",
            "interaction_level": "vector",
            "predictor_type": "sample_graph",
            "graph_predictor": "hgt",
            "model": "hgt",
            "fusion_mode": "concat_all",
        }
    )

    assert route["route_id"] == "pair.sample_graph"
    assert route["predictor"]["architecture"]["id"] == "hgt"
    assert route["predictor"]["input_adapter"]["id"] == "typed_role_projection"
    assert route["predictor"]["readout"]["id"] == "typed_edge"


def test_pretrained_pooled_and_residue_routes_have_distinct_outputs():
    pooled = resolve_effective_route(
        {
            "task_type": "pep_function",
            "tokenizer": "esm2",
            "encoder": "index",
            "model": "mlp",
            "classifier": "mlp",
        }
    )
    residue = resolve_effective_route(
        {
            "task_type": "pep_function",
            "tokenizer": "esm2",
            "encoder": "esm2",
            "model": "gat",
            "sequence_model": "gat",
            "classifier": "mlp",
            "training_mode": "end_to_end",
        }
    )

    assert pooled["route_id"] == "sequence.pretrained_pooled"
    assert pooled["streams"][0]["encoder"] == {
        "id": "esm2",
        "label": "ESM2",
        "output_kind": "sequence_embedding",
        "transformation": "none",
        "frozen": True,
    }
    assert residue["route_id"] == "sequence.pretrained_residue_graph"
    assert residue["streams"][0]["encoder"]["output_kind"] == "residue_embeddings"
    assert residue["predictor"]["architecture"]["id"] == "gat"
    assert residue["predictor"]["readout"]["id"] == "mean_max"
    assert residue["predictor"]["input_adapter"] == {
        "id": "linear_projection",
        "label": "Linear Projection",
        "learnable": True,
        "normalization": "layer_norm",
        "output_dimension": 128,
    }
    assert residue["schema_version"] == 6


def test_pretrained_residue_sequence_route_uses_masked_readout():
    route = resolve_effective_route(
        {
            "task_type": "pep_function",
            "tokenizer": "prott5",
            "encoder": "prott5",
            "sequence_model": "lstm",
            "model": "lstm",
            "training_mode": "end_to_end",
            "classifier": "mlp",
            "residue_adapter_dim": 256,
        }
    )

    assert route["route_id"] == "sequence.pretrained_residue_sequence"
    assert route["streams"][0]["encoder"]["output_kind"] == "residue_embeddings"
    assert route["streams"][0]["encoder"]["frozen"] is True
    assert route["predictor"]["architecture"]["id"] == "lstm"
    assert route["predictor"]["readout"]["id"] == "last_hidden"
    assert route["predictor"]["input_adapter"]["output_dimension"] == 256


def test_pair_streams_normalize_legacy_placeholders_independently():
    canonical = canonicalize_encoder_config(
        {
            "task_type": "pep_protein_interaction",
            "tokenizer": "esm2",
            "encoder": "ppi_fusion",
            "streams": {
                "peptide": {"tokenizer": "esm2", "encoder": "ppi_fusion"},
                "partner": {"tokenizer": "prott5", "encoder": "index"},
            },
        }
    )

    assert canonical["encoder"] == "esm2"
    assert canonical["streams"]["peptide"]["encoder"] == "esm2"
    assert canonical["streams"]["partner"]["encoder"] == "prott5"

    route = resolve_effective_route(
        {
            **canonical,
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "cross_attn",
            "model": "cross_attn",
            "classifier": "mlp",
            "training_mode": "end_to_end",
        }
    )
    assert route["route_id"] == "pair.dual_token_attention"
    assert [stream["encoder"]["id"] for stream in route["streams"]] == ["esm2", "prott5"]
    assert route["predictor"]["architecture"]["id"] == "cross_attn"
    assert route["predictor"]["readout"]["id"] == ("unidirectional_cross_attention_pair_fusion")
    bi_route = resolve_effective_route(
        {
            **canonical,
            "pair_strategy": "two_stream",
            "interaction_level": "token",
            "interaction_model": "bi_cross_attn",
            "model": "bi_cross_attn",
            "classifier": "mlp",
            "training_mode": "end_to_end",
        }
    )
    assert bi_route["predictor"]["readout"]["id"] == ("bidirectional_cross_attention_pair_fusion")


def test_pair_single_stream_uses_sequence_join_not_vector_fusion():
    route = resolve_effective_route(
        {
            "task_type": "pep_hlaii",
            "tokenizer": "kmer",
            "encoder": "onehot",
            "pair_strategy": "single_stream",
            "model": "cnn",
            "sequence_model": "cnn",
            "classifier": "mlp",
            "training_mode": "end_to_end",
        }
    )

    assert route["route_id"] == "pair.shared_sequence"
    assert route["predictor"]["input_adapter"]["id"] == "linear_projection"
    assert route["predictor"]["input_adapter"]["join_mode"] == ("role_boundary_sequence_join_v1")


def test_web_normalization_marks_legacy_encoder_source():
    config, sources = normalize_experiment_config_with_sources(
        "pep_function",
        {"tokenizer": "esm2", "encoder": "index", "model": "mlp"},
    )

    assert config["encoder"] == "esm2"
    assert sources["encoder"] == "legacy_derived"
    assert legacy_encoder_messages({"tokenizer": "esm2", "encoder": "index"})


def test_preflight_returns_effective_route_and_legacy_warning():
    report = preflight_experiment(
        {
            "id": "dataset",
            "name": "dataset",
            "task_type": "pep_function",
            "format": "sequence_dir",
            "metadata": {"splits": {"train": 4, "val": 2, "test": 2}},
        },
        "pep_function",
        {"tokenizer": "esm2", "encoder": "index", "model": "mlp", "classifier": "mlp"},
    )

    assert report["valid"] is True
    assert report["normalized_config"]["encoder"] == "esm2"
    assert "legacy_pretrained_encoder" in report["warning_codes"]
    assert report["effective_route"]["route_id"] == "sequence.pretrained_pooled"


def test_route_capabilities_publish_the_versioned_contract():
    capabilities = route_capabilities()

    assert capabilities["route_schema_version"] == ROUTE_SCHEMA_VERSION
    assert capabilities["predictor_parts"] == ["input_adapter", "architecture", "readout", "head"]
    assert capabilities["defaults"]["pep_protein_interaction"]["encoder"] == "esm2"
    assert "residue_embeddings" in capabilities["output_kinds"]


def test_immuneapp_el_registered_route_uses_the_three_layer_contract():
    route = resolve_registered_route("immuneapp_el")

    assert route["schema_version"] == ROUTE_SCHEMA_VERSION
    assert route["route_id"] == "pair.specialized_dual_sequence"
    assert route["task_type"] == "pep_hlai"
    assert [stream["encoder"]["output_kind"] for stream in route["streams"]] == [
        "position_features",
        "position_features",
    ]
    assert route["predictor"]["input_adapter"]["id"] == "identity"
    assert route["predictor"]["architecture"]["id"] == "immuneapp_el_cnn_bilstm_dense"
    assert route["predictor"]["readout"]["id"] == "dual_attention_flatten_fusion"
    assert route["predictor"]["head"]["id"] == "dense_32_sigmoid"

    with pytest.raises(ValueError, match="unknown registered route template"):
        resolve_registered_route("not-a-baseline")
