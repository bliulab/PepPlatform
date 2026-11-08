from unittest.mock import patch

import pytest
import torch

from scripts.train_pair_pipeline import (
    LegacyPairBiCrossAttentionClassifier,
    PairBiCrossAttentionClassifier,
    pair_bi_cross_attention_predictor_contract,
)
from pepplatform.models import ResidueModelInput
from pepplatform.web.inference import InferenceUnsupportedError, ModelService


def _model(model_cls=PairBiCrossAttentionClassifier):
    return model_cls(
        seq_len=2,
        encoder_name="ppi_fusion",
        in_dim=3,
        hidden_dim=4,
        dropout=0.0,
        n_heads=2,
    )


def _inputs(batch=2):
    return torch.randn(batch, 2, 3), torch.randn(batch, 3, 3)


def test_bi_cross_attention_uses_two_independent_directional_modules():
    model = _model()
    model.eval()
    peptide, partner = _inputs()
    with (
        patch.object(model.attn_p2q, "forward", wraps=model.attn_p2q.forward) as p2q,
        patch.object(model.attn_q2p, "forward", wraps=model.attn_q2p.forward) as q2p,
    ):
        logits, embedding = model.forward_with_embedding(peptide, partner)

    assert logits.shape == (2, 1)
    assert embedding.shape == (2, 32)
    assert p2q.call_count == q2p.call_count == 1
    p_query, p_key, _ = p2q.call_args.args[:3]
    q_query, q_key, _ = q2p.call_args.args[:3]
    assert p_query.shape[1] == peptide.shape[1]
    assert p_key.shape[1] == partner.shape[1]
    assert q_query.shape[1] == partner.shape[1]
    assert q_key.shape[1] == peptide.shape[1]
    assert model.attn_p2q is not model.attn_q2p
    assert not hasattr(model, "attn")
    assert not any(key.startswith("attn.") for key in model.state_dict())


def test_bi_cross_attention_has_finite_gradients_for_both_streams():
    model = _model()
    peptide, partner = (value.requires_grad_(True) for value in _inputs())
    logits = model(peptide, partner)
    logits.sum().backward()

    assert torch.isfinite(logits).all()
    assert torch.isfinite(peptide.grad).all()
    assert torch.isfinite(partner.grad).all()
    assert model.attn_p2q.in_proj_weight.grad is not None
    assert model.attn_q2p.in_proj_weight.grad is not None


def test_bi_cross_attention_masks_padding_and_is_batch_invariant():
    torch.manual_seed(12)
    model = _model()
    model.eval()
    peptide_features = torch.randn(2, 3, 3, requires_grad=True)
    partner_features = torch.randn(2, 4, 3, requires_grad=True)
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


def test_bi_cross_attention_rejects_invalid_inputs():
    model = _model()
    with pytest.raises(ValueError, match="finite"):
        model(torch.tensor([[[float("inf"), 0.0, 0.0]]]), torch.ones((1, 1, 3)))
    with pytest.raises(ValueError, match="floating"):
        model(torch.ones((1, 1, 3), dtype=torch.long), torch.ones((1, 1, 3)))
    invalid = ResidueModelInput(
        torch.zeros((1, 2, 3)),
        torch.zeros((1, 2), dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="at least one valid token"):
        model(
            invalid, ResidueModelInput(torch.ones((1, 2, 3)), torch.ones((1, 2), dtype=torch.bool))
        )


def test_bi_cross_attention_contract_records_independent_directions():
    contract = pair_bi_cross_attention_predictor_contract(_model())
    assert contract["contract_version"] == 1
    assert contract["direction"] == (
        "peptide_query_partner_key_value_and_partner_query_peptide_key_value"
    )
    assert contract["attention_parameterization"] == ("independent parameters for each direction")
    assert contract["readout"] == "bidirectional_cross_attention_pair_fusion_8h"


def test_unversioned_bi_loader_preserves_historical_state_layout(tmp_path):
    legacy = _model(LegacyPairBiCrossAttentionClassifier)
    legacy.eval()
    peptide, partner = _inputs()
    expected = legacy(peptide, partner)
    torch.save(
        {"model_class": "PairBiCrossAttentionClassifier", "state_dict": legacy.state_dict()},
        tmp_path / "model.pt",
    )
    metadata = {
        "pair": {
            "interaction_model": "bi_cross_attn",
            "seq_len": 2,
            "encoder": "ppi_fusion",
            "token_dim": 3,
            "hidden_dim": 4,
            "n_heads": 2,
            "dropout": 0.0,
        }
    }
    with pytest.warns(RuntimeWarning, match="legacy_only"):
        restored = ModelService._load_pair_token_model(
            tmp_path,
            metadata,
            torch.cat([peptide, partner], dim=1),
            torch.device("cpu"),
        )

    restored.eval()
    torch.testing.assert_close(restored(peptide, partner), expected)
    assert isinstance(restored, LegacyPairBiCrossAttentionClassifier)


def test_legacy_bi_constructor_preserves_positional_dropout():
    model = LegacyPairBiCrossAttentionClassifier(
        2,
        "ppi_fusion",
        None,
        3,
        4,
        0.25,
        2,
    )
    assert model.attn.dropout == 0.25
    assert pair_bi_cross_attention_predictor_contract(model) is None


def test_bi_loader_rejects_unknown_present_contract(tmp_path):
    model = _model()
    torch.save(
        {"model_class": "PairBiCrossAttentionClassifier", "state_dict": model.state_dict()},
        tmp_path / "model.pt",
    )
    metadata = {
        "predictor_contract": {
            "implementation": "pepplatform_pair_token_bidirectional_cross_attention",
            "contract_version": 99,
        },
        "pair": {
            "interaction_model": "bi_cross_attn",
            "seq_len": 2,
            "encoder": "ppi_fusion",
            "token_dim": 3,
            "hidden_dim": 4,
            "n_heads": 2,
            "dropout": 0.0,
        },
    }
    with pytest.raises(InferenceUnsupportedError, match="unsupported Bi-Cross-Attention"):
        ModelService._load_pair_token_model(
            tmp_path,
            metadata,
            torch.ones((1, 4, 3)),
            torch.device("cpu"),
        )


def test_current_bi_contract_reload_matches_outputs(tmp_path):
    model = _model()
    model.eval()
    peptide, partner = _inputs()
    contract = pair_bi_cross_attention_predictor_contract(model)
    torch.save(
        {"model_class": "PairBiCrossAttentionClassifier", "state_dict": model.state_dict()},
        tmp_path / "model.pt",
    )
    metadata = {
        "predictor_contract": contract,
        "pair": {
            "interaction_model": "bi_cross_attn",
            "seq_len": 2,
            "encoder": "ppi_fusion",
            "token_dim": 3,
            "hidden_dim": 4,
            "n_heads": 2,
            "dropout": 0.0,
        },
    }
    restored = ModelService._load_pair_token_model(
        tmp_path,
        metadata,
        torch.cat([peptide, partner], dim=1),
        torch.device("cpu"),
    )
    restored.eval()
    torch.testing.assert_close(restored(peptide, partner), model(peptide, partner))


def test_unversioned_bi_embedding_pickle_preserves_legacy_forward(tmp_path, monkeypatch):
    model = _model()
    model.attn = torch.nn.MultiheadAttention(4, 2, dropout=0.0, batch_first=True)
    del model.bi_cross_attention_predictor_contract_version
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
            "interaction_model": "bi_cross_attn",
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
