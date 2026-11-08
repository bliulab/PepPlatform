import json
from argparse import Namespace
from unittest.mock import patch

import pytest
import torch

from scripts.train_pair_pipeline import (
    LegacyPairCrossAttentionClassifier,
    PairCrossAttentionClassifier,
    pair_cross_attention_predictor_contract,
    save_pair_token_nn_artifact,
    validate_pair_attention_hyperparameters,
)
from pepplatform.models import ResidueModelInput
from pepplatform.web.inference import InferenceUnsupportedError, ModelService


def _model(**kwargs):
    defaults = {
        "seq_len": 2,
        "encoder_name": "ppi_fusion",
        "in_dim": 3,
        "hidden_dim": 4,
        "dropout": 0.0,
        "n_heads": 2,
    }
    defaults.update(kwargs)
    return PairCrossAttentionClassifier(**defaults)


def _pair_inputs(batch=2):
    peptide = torch.randn(batch, 2, 3)
    partner = torch.randn(batch, 3, 3)
    return peptide, partner


def test_cross_attention_is_peptide_query_partner_key_value_and_exports_8h():
    model = _model()
    model.eval()
    peptide, partner = _pair_inputs()
    with patch.object(model.attn, "forward", wraps=model.attn.forward) as forward:
        logits, embedding = model.forward_with_embedding(peptide, partner)

    assert logits.shape == (2, 1)
    assert embedding.shape == (2, 32)
    assert forward.call_count == 1
    query, key, value = forward.call_args.args[:3]
    assert query.shape == peptide.shape[:2] + (4,)
    assert key.shape == partner.shape[:2] + (4,)
    assert value.shape == partner.shape[:2] + (4,)


def test_cross_attention_masked_padding_and_all_masked_inputs_fail_closed():
    model = _model(in_dim=2)
    model.eval()
    peptide = ResidueModelInput(
        torch.tensor([[[1.0, 0.0], [0.0, 0.0], [9.0, 9.0]]]),
        torch.tensor([[True, True, False]]),
    )
    partner = ResidueModelInput(
        torch.tensor([[[0.0, 1.0], [0.0, 0.0], [8.0, 8.0]]]),
        torch.tensor([[True, False, False]]),
    )
    clean = model(peptide, partner)
    contaminated = model(
        ResidueModelInput(
            peptide.features.clone().masked_fill(~peptide.mask.unsqueeze(-1), 123.0),
            peptide.mask,
        ),
        ResidueModelInput(
            partner.features.clone().masked_fill(~partner.mask.unsqueeze(-1), -321.0),
            partner.mask,
        ),
    )
    torch.testing.assert_close(clean, contaminated)

    invalid = ResidueModelInput(
        torch.zeros((1, 3, 2)),
        torch.zeros((1, 3), dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="at least one valid token"):
        model(invalid, partner)


def test_cross_attention_padding_gradients_and_batch_composition_are_isolated():
    torch.manual_seed(9)
    model = _model(in_dim=2)
    model.eval()
    peptide_features = torch.randn(2, 3, 2, requires_grad=True)
    partner_features = torch.randn(2, 4, 2, requires_grad=True)
    peptide_mask = torch.tensor([[True, True, False], [True, False, False]])
    partner_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    peptide = ResidueModelInput(peptide_features, peptide_mask)
    partner = ResidueModelInput(partner_features, partner_mask)

    batch_logits = model(peptide, partner)
    single_logits = model(peptide.batch_slice(0, 1), partner.batch_slice(0, 1))
    torch.testing.assert_close(batch_logits[:1], single_logits)
    batch_logits.sum().backward()
    assert torch.count_nonzero(peptide_features.grad[~peptide_mask]) == 0
    assert torch.count_nonzero(partner_features.grad[~partner_mask]) == 0


def test_cross_attention_rejects_nonfinite_and_invalid_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        _model(hidden_dim=5, n_heads=2)
    with pytest.raises(ValueError, match="dropout"):
        _model(dropout=1.0)
    model = _model(in_dim=2)
    bad = torch.tensor([[[float("nan"), 0.0], [0.0, 1.0]]])
    with pytest.raises(ValueError, match="finite"):
        model(bad, torch.ones((1, 2, 2)))
    with pytest.raises(ValueError, match="floating"):
        model(torch.ones((1, 2, 2), dtype=torch.long), torch.ones((1, 2, 2)))
    index_model = _model(encoder_name="index", vocab_size=5, in_dim=None)
    with pytest.raises(ValueError, match="integer"):
        index_model(torch.ones((1, 2)), torch.ones((1, 2)))
    with pytest.raises(ValueError, match="at least one valid token"):
        model(torch.empty((0, 2, 2)), torch.empty((0, 2, 2)))


def test_cross_attention_contract_is_persisted_and_validated(tmp_path):
    args = Namespace(
        task_type="pep_protein_interaction",
        artifact_dir=str(tmp_path),
        tokenizer="esm2",
        encoder="ppi_fusion",
        model="cross_attn+mlp",
        classifier="mlp",
        pair_strategy="two_stream",
        interaction_level="token",
        interaction_model="cross_attn",
        training_mode="end_to_end",
        max_length=2,
        dropout=0.0,
        _operating_point={
            "values": [0.4],
            "source": "validation",
            "objective": "f1",
            "score_kind": "probability",
            "protocol_version": 3,
        },
        _resolved_pretrained_model_source="models/esm2",
    )
    model = _model()
    save_pair_token_nn_artifact(
        args,
        model,
        torch.ones((2, 4, 3)),
        threshold=0.4,
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    contract = metadata["predictor_contract"]
    assert contract["contract_version"] == 1
    assert contract["direction"] == "peptide_query_partner_key_value"
    assert contract["readout"] == "unidirectional_cross_attention_pair_fusion_8h"
    assert pair_cross_attention_predictor_contract(model) == contract
    model.eval()
    joined = torch.randn(2, 4, 3)
    restored = ModelService._load_pair_token_model(
        tmp_path,
        metadata,
        joined,
        torch.device("cpu"),
    )
    restored.eval()
    torch.testing.assert_close(restored(joined), model(joined))


def test_unversioned_cross_attention_loader_preserves_historical_behavior(tmp_path):
    model = _model()
    model.cross_attention_predictor_contract_version = 0
    peptide, partner = _pair_inputs()
    expected = model(peptide, partner)
    torch.save(
        {"model_class": "PairCrossAttentionClassifier", "state_dict": model.state_dict()},
        tmp_path / "model.pt",
    )
    metadata = {
        "pair": {
            "interaction_model": "cross_attn",
            "seq_len": 2,
            "encoder": "ppi_fusion",
            "token_dim": 3,
            "hidden_dim": 4,
            "n_heads": 2,
            "dropout": 0.0,
        }
    }
    restored = ModelService._load_pair_token_model(
        tmp_path,
        metadata,
        torch.cat([peptide, partner], dim=1),
        torch.device("cpu"),
    )
    torch.testing.assert_close(restored(peptide, partner), expected)
    assert isinstance(restored, LegacyPairCrossAttentionClassifier)


def test_cross_loader_rejects_tampered_present_contract(tmp_path):
    model = _model()
    torch.save(
        {"model_class": "PairCrossAttentionClassifier", "state_dict": model.state_dict()},
        tmp_path / "model.pt",
    )
    contract = pair_cross_attention_predictor_contract(model)
    contract["direction"] = "partner_query_peptide_key_value"
    metadata = {
        "predictor_contract": contract,
        "pair": {
            "interaction_model": "cross_attn",
            "seq_len": 2,
            "encoder": "ppi_fusion",
            "token_dim": 3,
            "hidden_dim": 4,
            "n_heads": 2,
            "dropout": 0.0,
        },
    }
    with pytest.raises(InferenceUnsupportedError, match="unsupported Cross-Attention"):
        ModelService._load_pair_token_model(
            tmp_path,
            metadata,
            torch.ones((1, 4, 3)),
            torch.device("cpu"),
        )


def test_attention_hyperparameter_validator_is_explicit():
    assert validate_pair_attention_hyperparameters(8, 2, 0.25) == (8, 2, 0.25)
    with pytest.raises(ValueError):
        validate_pair_attention_hyperparameters(7, 2, 0.0)


def test_unversioned_cross_embedding_pickle_preserves_legacy_forward(tmp_path, monkeypatch):
    model = _model()
    del model.cross_attention_predictor_contract_version
    model.eval()
    joined = torch.randn(2, 4, 3)
    expected = model.forward_with_embedding(joined)[1]
    torch.save(model, tmp_path / "embedding_encoder.pt")
    monkeypatch.setattr(
        ModelService,
        "_extract_pair_token_features",
        staticmethod(lambda metadata, pairs, device, bundle=None: joined),
    )
    metadata = {
        "embedding_export": {"encoder_path": "embedding_encoder.pt"},
        "pair": {
            "pair_strategy": "two_stream",
            "interaction_model": "cross_attn",
        },
        "args": {"batch_size": 2},
    }
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        actual = ModelService._extract_pair_embedding_features(
            metadata,
            [{"peptide": "AA", "protein": "BB"}] * 2,
            torch.device("cpu"),
            bundle=tmp_path,
        )
    torch.testing.assert_close(actual, expected)


def test_current_cross_embedding_contract_rejects_legacy_pickle(tmp_path, monkeypatch):
    legacy = LegacyPairCrossAttentionClassifier(
        seq_len=2,
        encoder_name="ppi_fusion",
        in_dim=3,
        hidden_dim=4,
        dropout=0.0,
        n_heads=2,
    )
    torch.save(legacy, tmp_path / "embedding_encoder.pt")
    joined = torch.randn(2, 4, 3)
    monkeypatch.setattr(
        ModelService,
        "_extract_pair_token_features",
        staticmethod(lambda metadata, pairs, device, bundle=None: joined),
    )
    metadata = {
        "embedding_export": {
            "encoder_path": "embedding_encoder.pt",
            "architecture_contract": pair_cross_attention_predictor_contract(_model()),
        },
        "pair": {
            "pair_strategy": "two_stream",
            "interaction_model": "cross_attn",
        },
    }
    with pytest.raises(InferenceUnsupportedError, match="does not match"):
        ModelService._extract_pair_embedding_features(
            metadata,
            [{"peptide": "AA", "protein": "BB"}] * 2,
            torch.device("cpu"),
            bundle=tmp_path,
        )
