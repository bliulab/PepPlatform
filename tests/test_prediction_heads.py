from argparse import Namespace

import pytest
import torch
import torch.nn as nn

from pepplatform.models import (
    BoWMLP,
    ESMClassifier,
    GatedResidualClassifier,
    IndexGRU,
    PredictionHeadClassifier,
    ResidueGraphClassifier,
    build_prediction_head,
)
from pepplatform.route_registry import annotate_legacy_prediction_head, resolve_effective_route
from pepplatform.web.config import normalize_experiment_config_with_sources
from pepplatform.web.inference import ModelService
from pepplatform.web.preflight import preflight_experiment
from pepplatform.workflow import (
    normalize_prediction_head_config,
    prediction_head_artifact_config,
    validate_linear_prediction_head_semantics,
)
from scripts.train_pair_pipeline import (
    HeterogeneousPairCrossAttentionClassifier,
    PairCrossAttentionClassifier,
)


def _dataset(task_type="pep_function"):
    return {
        "id": "dataset",
        "name": "demo",
        "task_type": task_type,
        "format": "sequence_dir" if task_type == "pep_function" else "pair_split_dir",
        "metadata": {"splits": {"train": 10, "val": 2, "test": 2}},
    }


def test_prediction_head_builder_supports_linear_and_modern_activations():
    linear = build_prediction_head(8, 2, hidden_dims=[])
    assert isinstance(linear, nn.Linear)
    assert linear(torch.ones(3, 8)).shape == (3, 2)

    deep = build_prediction_head(
        8,
        2,
        hidden_dims=[16, 4],
        activation="silu",
        dropout=0.25,
    )
    assert [type(layer) for layer in deep] == [
        nn.Linear,
        nn.SiLU,
        nn.Dropout,
        nn.Linear,
        nn.SiLU,
        nn.Dropout,
        nn.Linear,
    ]
    assert deep(torch.ones(3, 8)).shape == (3, 2)


def test_prediction_head_classifier_zero_hidden_layers_is_a_single_linear_layer():
    model = PredictionHeadClassifier(8, 2, head_hidden_dims=[])
    assert isinstance(model.head, nn.Linear)
    assert [name for name, _ in model.named_parameters()] == ["head.weight", "head.bias"]
    logits, embedding = model.forward_with_embedding(torch.ones(3, 8))
    assert logits.shape == (3, 2)
    assert embedding.shape == (3, 8)


def test_bow_zero_hidden_layers_is_a_single_linear_layer():
    model = BoWMLP(in_dim=8, num_classes=2, head_hidden_dims=[])
    assert isinstance(model.net, nn.Linear)
    assert [name for name, _ in model.named_parameters()] == ["net.weight", "net.bias"]
    logits, embedding = model.forward_with_embedding(torch.ones(3, 8))
    assert logits.shape == (3, 2)
    assert embedding.shape == (3, 8)


def test_legacy_esm_classifier_symbol_resolves_to_renamed_class():
    assert ESMClassifier is GatedResidualClassifier


def test_legacy_sequence_and_graph_head_defaults_are_unchanged():
    recurrent = IndexGRU(vocab_size=24, embed_dim=8, hidden_dim=6, num_classes=3, dropout=0.1)
    assert [type(layer) for layer in recurrent.fc] == [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear]
    assert recurrent.fc[0].in_features == 12
    assert recurrent.fc[0].out_features == 64
    assert recurrent.fc[3].out_features == 3

    graph = ResidueGraphClassifier(in_dim=5, hidden_dim=12, num_classes=2, dropout=0.1)
    assert [type(layer) for layer in graph.classifier] == [
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
    ]
    assert graph.classifier[0].in_features == 24
    assert graph.classifier[0].out_features == 12


def test_prediction_head_config_normalizes_route_defaults_and_linear_contract():
    args = Namespace(
        model="gru",
        sequence_model="gru",
        training_mode="end_to_end",
        classifier="mlp",
        interaction_level="",
        dropout=0.2,
        head_hidden_dims=None,
        head_activation=None,
        head_dropout=None,
    )
    normalize_prediction_head_config(args)
    assert args.head_hidden_dims == [128]
    assert args.head_activation == "relu"
    assert args.head_dropout == pytest.approx(0.2)

    args.head_hidden_dims = []
    args.head_activation = None
    args.head_dropout = None
    normalize_prediction_head_config(args)
    assert args.head_hidden_dims == []
    assert not hasattr(args, "head_activation")
    assert not hasattr(args, "head_dropout")


def test_cli_prediction_head_config_accepts_arbitrary_positive_depth():
    args = Namespace(
        model="mlp",
        classifier="mlp",
        dropout=0.2,
        head_hidden_dims=[512, 256, 128, 64],
        head_activation="gelu",
        head_dropout=0.1,
    )
    normalize_prediction_head_config(args)
    assert args.head_hidden_dims == [512, 256, 128, 64]
    assert prediction_head_artifact_config(args) == {
        "schema_version": 1,
        "implementation": "standard_linear_mlp",
        "kind": "mlp",
        "hidden_dims": [512, 256, 128, 64],
        "activation": "gelu",
        "dropout": 0.1,
    }


def test_pair_attention_heads_accept_custom_and_linear_topologies():
    pair = PairCrossAttentionClassifier(
        seq_len=3,
        encoder_name="index",
        vocab_size=20,
        hidden_dim=8,
        n_heads=2,
        head_hidden_dims=[12, 6],
        head_activation="gelu",
        head_dropout=0.0,
    )
    assert pair(torch.ones(2, 6, dtype=torch.long)).shape == (2, 1)
    assert isinstance(pair.head[1], nn.GELU)

    heterogeneous = HeterogeneousPairCrossAttentionClassifier(
        peptide_encoder_name="index",
        partner_encoder_name="index",
        peptide_vocab_size=20,
        partner_vocab_size=25,
        hidden_dim=8,
        n_heads=2,
        head_hidden_dims=[],
    )
    assert isinstance(heterogeneous.head, nn.Linear)
    assert heterogeneous(
        torch.ones(2, 3, dtype=torch.long),
        torch.ones(2, 4, dtype=torch.long),
    ).shape == (2, 1)


def test_pair_token_artifact_loader_rebuilds_custom_prediction_head(tmp_path):
    model = PairCrossAttentionClassifier(
        seq_len=2,
        encoder_name="ppi_fusion",
        in_dim=3,
        hidden_dim=4,
        n_heads=2,
        dropout=0.0,
        head_hidden_dims=[7, 5],
        head_activation="silu",
        head_dropout=0.1,
    )
    torch.save({"state_dict": model.state_dict()}, tmp_path / "model.pt")
    metadata = {
        "pair": {
            "interaction_model": "cross_attn",
            "seq_len": 2,
            "encoder": "ppi_fusion",
            "token_dim": 3,
            "hidden_dim": 4,
            "n_heads": 2,
            "dropout": 0.0,
        },
        "prediction_head": {
            "hidden_dims": [7, 5],
            "activation": "silu",
            "dropout": 0.1,
        },
    }
    features = torch.ones(2, 4, 3)
    loaded = ModelService._load_pair_token_model(tmp_path, metadata, features, "cpu")
    assert isinstance(loaded.head[1], nn.SiLU)
    assert loaded.head[0].out_features == 7
    assert loaded.head[3].out_features == 5
    assert loaded(features).shape == (2, 1)


def test_web_normalization_materializes_head_defaults_and_linear_config():
    config, sources = normalize_experiment_config_with_sources(
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cnn",
            "sequence_model": "cnn",
            "training_mode": "end_to_end",
            "classifier": "mlp",
        },
    )
    assert config["head_hidden_dims"] == [128]
    assert config["head_activation"] == "relu"
    assert config["head_dropout"] == config["dropout"]
    assert sources["head_hidden_dims"] == "default"

    linear, _ = normalize_experiment_config_with_sources(
        "pep_function",
        {
            **config,
            "head_hidden_dims": [],
            "head_activation": None,
            "head_dropout": None,
        },
    )
    assert linear["head_hidden_dims"] == []
    assert "head_activation" not in linear
    assert "head_dropout" not in linear

    route = resolve_effective_route(linear, "pep_function")
    assert route["predictor"]["head"]["id"] == "linear"
    assert route["predictor"]["head"]["hidden_dims"] == []


@pytest.mark.parametrize("hidden_dims", [[64], [256], [128, 64], [128, 128, 128]])
def test_web_preflight_only_accepts_fixed_prediction_head_presets(hidden_dims):
    report = preflight_experiment(
        _dataset(),
        "pep_function",
        {
            "tokenizer": "kmer",
            "encoder": "index",
            "model": "cnn",
            "sequence_model": "cnn",
            "training_mode": "end_to_end",
            "classifier": "mlp",
            "head_hidden_dims": hidden_dims,
        },
    )
    assert report["valid"] is False
    assert "prediction_head_preset_unsupported" in report["error_codes"]


@pytest.mark.parametrize("hidden_dims", [[], [128], [256, 128]])
def test_web_preflight_accepts_fixed_prediction_head_presets(hidden_dims):
    report = preflight_experiment(
        _dataset(),
        "pep_function",
        {
            "tokenizer": "esm2",
            "encoder": "esm2",
            "model": "mlp",
            "classifier": "mlp",
            "head_hidden_dims": hidden_dims,
        },
    )
    assert report["valid"] is True
    assert report["normalized_config"]["head_hidden_dims"] == hidden_dims


def test_linear_semantic_loss_requires_a_trainable_upstream_representation():
    fixed = {
        "model": "mlp",
        "classifier": "mlp",
        "head_hidden_dims": [],
        "enable_semantic_similarity": True,
    }
    with pytest.raises(ValueError, match="trainable upstream representation"):
        validate_linear_prediction_head_semantics(fixed)

    trainable = {
        **fixed,
        "model": "cnn",
        "sequence_model": "cnn",
    }
    assert validate_linear_prediction_head_semantics(trainable) is trainable


def test_legacy_gated_bundle_route_is_annotated_without_rewriting_new_bundles():
    route = resolve_effective_route(
        {"tokenizer": "esm2", "encoder": "esm2", "model": "mlp", "classifier": "mlp"},
        "pep_function",
    )
    legacy = annotate_legacy_prediction_head(
        route,
        {"bundle_type": "pretrained_embedding_nn"},
    )
    assert legacy["predictor"]["head"]["id"] == "legacy_gated_residual"

    current = annotate_legacy_prediction_head(
        route,
        {
            "bundle_type": "pretrained_embedding_nn",
            "prediction_head": {"hidden_dims": []},
        },
    )
    assert current["predictor"]["head"]["id"] == "mlp"
