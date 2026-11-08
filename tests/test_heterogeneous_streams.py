import json
from argparse import Namespace

import pytest
import torch

from pepplatform.stream_spec import apply_pair_streams, normalize_pair_streams
from pepplatform.web.inference import ModelService
from pepplatform.web.paths import RuntimePaths
from pepplatform.web.preflight import preflight_experiment
from pepplatform.web.runner import ExperimentRunner
from scripts.train_pair_pipeline import (
    AlignedGatedClassifier,
    HeterogeneousPairBiCrossAttentionClassifier,
    HeterogeneousPairCrossAttentionClassifier,
    LegacyHeterogeneousPairCrossAttentionClassifier,
    configure_pair_vector_artifacts,
    save_heterogeneous_pair_token_artifact,
    save_pair_vector_nn_artifact,
)


def _dataset():
    return {
        "id": "dataset",
        "name": "pairs",
        "task_type": "pep_protein_interaction",
        "format": "pair_csv_bundle",
        "metadata": {"splits": {"train": 10, "val": 2, "test": 2}},
    }


def _heterogeneous_config(level="vector"):
    config = {
        "tokenizer": "esm2",
        "encoder": "ppi_fusion",
        "model": "mlp" if level == "vector" else "cross_attn",
        "classifier": "mlp",
        "pair_strategy": "two_stream",
        "interaction_level": level,
        "predictor_type": "classifier",
        "fusion_mode": "concat",
        "streams": {
            "peptide": {
                "feature_route": "pretrained",
                "tokenizer": "esm2",
                "encoder": "ppi_fusion",
                "model_source": "facebook/esm2_t30_150M_UR50D",
                "max_length": 64,
                "params": {},
            },
            "partner": {
                "feature_route": "pretrained",
                "tokenizer": "prott5",
                "encoder": "ppi_fusion",
                "model_source": "Rostlab/prot_t5_xl_uniref50",
                "max_length": 256,
                "params": {},
            },
        },
    }
    if level == "token":
        config["interaction_model"] = "cross_attn"
        config["training_mode"] = "end_to_end"
    return config


def test_stream_normalization_preserves_legacy_and_applies_role_overrides():
    args = Namespace(
        task_type="pep_protein_interaction",
        tokenizer="esm2",
        encoder="ppi_fusion",
        esm2_model="models/esm2",
        max_length=64,
        partner_tokenizer="prott5",
        partner_model_source="models/prott5",
        partner_max_length=256,
    )

    streams = normalize_pair_streams(args)
    assert streams["peptide"]["tokenizer"] == "esm2"
    assert streams["partner"]["tokenizer"] == "prott5"
    assert streams["partner"]["model_source"] == "models/prott5"
    apply_pair_streams(args)
    assert args.stream_mode == "heterogeneous"
    assert args.tokenizer == "esm2"


def test_kmer_stream_metadata_records_token_length_unit():
    streams = normalize_pair_streams(
        Namespace(
            task_type="pep_protein_interaction",
            tokenizer="kmer",
            encoder="index",
            max_length=60,
            k=2,
        )
    )

    assert streams["peptide"]["max_length_unit"] == "tokens"
    assert streams["partner"]["max_length_unit"] == "tokens"


def test_wordpiece_stream_defaults_and_limits_are_explicit():
    streams = normalize_pair_streams(
        Namespace(
            task_type="pep_protein_interaction",
            tokenizer="wordpiece",
            encoder="index",
            max_length=60,
        )
    )

    for stream in streams.values():
        assert stream["max_length_unit"] == "tokens_including_special_tokens"
        assert stream["params"]["wordpiece_vocab_size"] == 1000
        assert stream["params"]["wordpiece_max_raw_length"] == 50_000


def test_wordpiece_stream_rejects_raw_length_above_absolute_limit():
    with pytest.raises(
        ValueError,
        match="wordpiece max_raw_length must be in",
    ):
        normalize_pair_streams(
            Namespace(
                task_type="pep_protein_interaction",
                tokenizer="wordpiece",
                encoder="index",
                max_length=60,
                wordpiece_max_raw_length=100_001,
            )
        )


def test_classic_kmer_pair_artifact_records_token_length_unit():
    args = Namespace(
        task_type="pep_protein_interaction",
        artifact_dir="",
        tokenizer="kmer",
        encoder="bow",
        model="logreg",
        classifier="logreg",
        pair_strategy="two_stream",
        interaction_level="vector",
        predictor_type="classifier",
        fusion_mode="concat_all",
        max_length=60,
        k=2,
    )
    peptide = torch.ones(2, 4)
    partner = torch.ones(2, 4)

    info = configure_pair_vector_artifacts(
        args,
        peptide,
        partner,
        torch.cat([peptide, partner, peptide * partner, torch.abs(peptide - partner)], dim=1),
    )

    assert info["classic"]["max_length_unit"] == "tokens"


def test_web_preflight_accepts_heterogeneous_vector_and_token_routes():
    vector = preflight_experiment(
        _dataset(),
        "pep_protein_interaction",
        _heterogeneous_config("vector"),
    )
    token = preflight_experiment(
        _dataset(),
        "pep_protein_interaction",
        _heterogeneous_config("token"),
    )

    assert vector["valid"] is True
    assert vector["normalized_config"]["stream_mode"] == "heterogeneous"
    assert vector["execution"]["supports_reusable_inference"] is True
    assert token["valid"] is True
    assert token["execution"]["supports_reusable_inference"] is True


def test_web_preflight_accepts_aligned_gated_heterogeneous_vector_route():
    config = _heterogeneous_config("vector")
    config["fusion_mode"] = "aligned_gated"

    report = preflight_experiment(_dataset(), "pep_protein_interaction", config)

    assert report["valid"] is True
    assert report["normalized_config"]["fusion_mode"] == "aligned_gated"
    assert report["execution"]["supports_reusable_inference"] is True


def test_web_preflight_rejects_heterogeneous_kmer_stream_above_bound():
    config = _heterogeneous_config("vector")
    config["streams"]["peptide"] = {
        "feature_route": "classic",
        "tokenizer": "kmer",
        "encoder": "bow",
        "model_source": "",
        "max_length": 64,
        "params": {"k": 4},
    }
    report = preflight_experiment(_dataset(), "pep_protein_interaction", config)

    assert report["valid"] is False
    assert "invalid_numeric_parameter" in report["error_codes"]
    assert any(
        "peptide stream: K-mer tokenizer k must be an integer in [1, 3]" in message
        for message in report["errors"]
    )


def test_web_preflight_rejects_profile_k_for_heterogeneous_stream():
    config = _heterogeneous_config("vector")
    config["streams"]["peptide"] = {
        "feature_route": "classic",
        "tokenizer": "kmer",
        "encoder": "psfm",
        "model_source": "",
        "max_length": 64,
        "params": {"k": 2},
    }
    report = preflight_experiment(_dataset(), "pep_protein_interaction", config)

    assert report["valid"] is False
    assert "profile_encoder_requires_k1" in report["error_codes"]
    assert any(
        "peptide stream: psfm is defined per raw residue and requires k=1" in message
        for message in report["errors"]
    )


def test_web_preflight_rejects_unaligned_heterogeneous_elementwise_fusion():
    config = _heterogeneous_config("vector")
    config["fusion_mode"] = "concat_all"
    report = preflight_experiment(_dataset(), "pep_protein_interaction", config)

    assert report["valid"] is False
    assert "heterogeneous_fusion_requires_alignment" in report["error_codes"]


def test_web_preflight_accepts_classic_pretrained_mixed_vector_streams():
    config = _heterogeneous_config("vector")
    config.update(
        {
            "tokenizer": "dr",
            "encoder": "tfidf",
            "model": "xgb",
            "classifier": "xgb",
        }
    )
    config["streams"]["peptide"] = {
        "feature_route": "classic",
        "tokenizer": "dr",
        "encoder": "tfidf",
        "model_source": "",
        "max_length": 64,
        "params": {"dr_max_distance": 1},
    }
    report = preflight_experiment(_dataset(), "pep_protein_interaction", config)

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True


def test_web_preflight_accepts_heterogeneous_sample_graph_with_concat():
    config = _heterogeneous_config("vector")
    config.update(
        {
            "model": "gcn",
            "graph_predictor": "gcn",
            "predictor_type": "sample_graph",
            "fusion_mode": "concat",
        }
    )
    report = preflight_experiment(_dataset(), "pep_protein_interaction", config)

    assert report["valid"] is True
    assert report["execution"]["supports_reusable_inference"] is True


def test_web_runner_flattens_streams_to_role_prefixed_cli_flags(tmp_path):
    paths = RuntimePaths.from_project_root(tmp_path / "project", tmp_path / "runtime").ensure()
    root = tmp_path / "pairs"
    root.mkdir()
    for name in ["train.csv", "val.csv", "test.csv"]:
        (root / name).write_text("peptide,protein,label\nACD,EFG,1\n", encoding="utf-8")
    command = ExperimentRunner(None, paths).build_command(
        {
            "task_type": "pep_protein_interaction",
            "config": _heterogeneous_config("vector"),
            "artifact_dir": str(tmp_path / "artifact"),
        },
        {"format": "pair_csv_bundle", "root_path": str(root), "metadata": {}},
    )

    assert command[command.index("--peptide_tokenizer") + 1] == "esm2"
    assert command[command.index("--partner_tokenizer") + 1] == "prott5"
    assert command[command.index("--peptide_max_length") + 1] == "64"
    assert command[command.index("--partner_max_length") + 1] == "256"


def test_heterogeneous_vector_concat_inference_accepts_different_dimensions(monkeypatch):
    def fake_extract(metadata, sequences, device):
        dim = 2 if metadata["pretrained"]["tokenizer"] == "esm2" else 3
        return torch.ones((len(sequences), dim), dtype=torch.float)

    monkeypatch.setattr(ModelService, "_extract_pretrained_embeddings", fake_extract)
    features = ModelService._extract_pair_vector_features(
        {
            **_heterogeneous_config("vector"),
            "pair": {"fusion_mode": "concat"},
            "args": {"batch_size": 1},
        },
        [{"peptide": "ACD", "protein": "EFGH"}],
        torch.device("cpu"),
    )

    assert features.shape == (1, 5)


def test_learned_alignment_and_token_models_accept_unequal_stream_shapes():
    vector_model = AlignedGatedClassifier(
        peptide_dim=3,
        partner_dim=5,
        align_dim=4,
        hidden_dim=8,
        dropout=0.0,
    )
    vector_logits = vector_model(torch.ones(2, 3), torch.ones(2, 5))

    token_model = HeterogeneousPairBiCrossAttentionClassifier(
        peptide_encoder_name="ppi_fusion",
        partner_encoder_name="ppi_fusion",
        peptide_in_dim=3,
        partner_in_dim=5,
        hidden_dim=8,
        n_heads=2,
        dropout=0.0,
    )
    token_logits = token_model(torch.ones(2, 7, 3), torch.ones(2, 11, 5))

    assert vector_logits.shape == (2, 1)
    assert token_logits.shape == (2, 1)


def test_schema_v2_vector_artifact_records_both_streams(tmp_path):
    config = _heterogeneous_config("vector")
    args = Namespace(
        **config,
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        stream_mode="heterogeneous",
        max_length=64,
        _operating_point={
            "values": [0.4],
            "source": "validation",
            "objective": "f1",
            "score_kind": "probability",
            "protocol_version": 3,
        },
    )
    peptide = torch.ones(3, 2)
    partner = torch.ones(3, 3)
    fused = torch.cat([peptide, partner], dim=1)
    configure_pair_vector_artifacts(args, peptide, partner, fused)
    save_pair_vector_nn_artifact(args, torch.nn.Linear(5, 1), threshold=0.4)

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["pair"]["stream_mode"] == "heterogeneous"
    assert metadata["streams"]["peptide"]["output_dim"] == 2
    assert metadata["streams"]["partner"]["output_dim"] == 3


@pytest.mark.parametrize(
    ("interaction_model", "model_cls", "implementation"),
    [
        (
            "cross_attn",
            HeterogeneousPairCrossAttentionClassifier,
            "pepplatform_pair_token_cross_attention",
        ),
        (
            "bi_cross_attn",
            HeterogeneousPairBiCrossAttentionClassifier,
            "pepplatform_pair_token_bidirectional_cross_attention",
        ),
    ],
)
def test_schema_v2_token_artifact_can_be_reloaded(
    tmp_path, interaction_model, model_cls, implementation
):
    config = _heterogeneous_config("token")
    args = Namespace(
        **config,
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        stream_mode="heterogeneous",
        dropout=0.0,
        _operating_point={
            "values": [0.45],
            "source": "validation",
            "objective": "f1",
            "score_kind": "probability",
            "protocol_version": 3,
        },
    )
    model_config = {
        "peptide_encoder_name": "ppi_fusion",
        "partner_encoder_name": "ppi_fusion",
        "peptide_in_dim": 3,
        "partner_in_dim": 5,
        "hidden_dim": 8,
        "n_heads": 2,
        "dropout": 0.0,
    }
    model = model_cls(**model_config)
    model.eval()
    peptide = torch.ones(2, 7, 3)
    partner = torch.ones(2, 11, 5)
    expected = model(peptide, partner)
    args.interaction_model = interaction_model
    save_heterogeneous_pair_token_artifact(
        args,
        model,
        peptide,
        partner,
        model_config,
        threshold=0.45,
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    restored = ModelService._load_heterogeneous_pair_token_model(
        tmp_path,
        metadata,
        torch.device("cpu"),
    )

    assert metadata["schema_version"] == 2
    assert metadata["predictor_contract"]["implementation"] == implementation
    restored.eval()
    torch.testing.assert_close(restored(peptide, partner), expected)


@pytest.mark.parametrize(
    ("interaction_model", "model_cls"),
    [
        ("cross_attn", LegacyHeterogeneousPairCrossAttentionClassifier),
        ("bi_cross_attn", HeterogeneousPairBiCrossAttentionClassifier),
    ],
)
def test_unversioned_heterogeneous_token_artifact_preserves_legacy_outputs(
    tmp_path, interaction_model, model_cls
):
    model_config = {
        "peptide_encoder_name": "ppi_fusion",
        "partner_encoder_name": "ppi_fusion",
        "peptide_in_dim": 3,
        "partner_in_dim": 5,
        "hidden_dim": 8,
        "n_heads": 2,
        "dropout": 0.0,
    }
    model = model_cls(**model_config)
    model.eval()
    peptide = torch.randn(2, 7, 3)
    partner = torch.randn(2, 11, 5)
    expected = model(peptide, partner)
    torch.save({"state_dict": model.state_dict()}, tmp_path / "model.pt")
    metadata = {
        "pair": {
            "interaction_model": interaction_model,
            "model_config": model_config,
        }
    }

    with pytest.warns(RuntimeWarning, match="legacy_only"):
        restored = ModelService._load_heterogeneous_pair_token_model(
            tmp_path,
            metadata,
            torch.device("cpu"),
        )

    restored.eval()
    torch.testing.assert_close(restored(peptide, partner), expected)
