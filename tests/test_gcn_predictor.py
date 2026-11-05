from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pepplatform.models import (
    GCN_PREDICTOR_CONTRACT_VERSION,
    PretrainedResiduePredictor,
    ResidueGraphClassifier,
    ResidueModelInput,
    gcn_predictor_contract,
    sequence_predictor_contract,
)
from scripts.train_pair_pipeline import build_classic_featurizer


def test_residue_gcn_contract_records_chain_graph_assembly():
    model = ResidueGraphClassifier(
        in_dim=5,
        hidden_dim=12,
        num_classes=2,
        dropout=0.0,
        graph_kind="gcn",
    )

    contract = gcn_predictor_contract(model)

    assert model.gcn_predictor_contract_version == GCN_PREDICTOR_CONTRACT_VERSION
    assert sequence_predictor_contract(model) == contract
    assert contract["status"] == "verified_platform_variant"
    assert contract["implementation"] == "pepplatform_masked_chain_gcn"
    assert contract["node_definition"] == "one node per residue or sequence position"
    assert contract["layers"] == 2
    assert contract["hidden_dim"] == 12
    assert contract["readout"] == "masked mean plus masked max"
    assert contract["prediction_output"] == "logits"


def test_pretrained_residue_gcn_contract_records_adapter_and_mask():
    model = PretrainedResiduePredictor(
        "gcn",
        input_dim=5,
        adapter_dim=8,
        hidden_dim=6,
        heads=2,
        num_classes=2,
        dropout=0.0,
    )

    contract = gcn_predictor_contract(model)

    assert contract["input_adapter"] == "layer_norm_linear_projection"
    assert contract["padding_mask"] == "explicit_residue_mask"
    assert contract["hidden_dim"] == 6


def test_gcn_without_contract_attribute_is_legacy_only():
    model = ResidueGraphClassifier(5, num_classes=2, dropout=0.0)
    del model.gcn_predictor_contract_version

    contract = gcn_predictor_contract(model)

    assert contract["status"] == "legacy_only"
    assert contract["contract_version"] == 0
    assert contract["padding_mask"] == "legacy_route_dependent"


def test_classic_gcn_builder_preserves_valid_zero_oov_positions(tmp_path):
    vectors = Path(tmp_path) / "vectors.txt"
    vectors.write_text("1 3\nA 1.0 2.0 3.0\n", encoding="utf-8")
    args = SimpleNamespace(
        tokenizer="kmer",
        encoder="word2vec",
        model="gcn",
        sequence_model="gcn",
        max_length=4,
        k=1,
        word2vec_path=str(vectors),
        glove_path="",
        fasttext_path="",
        bpe_path="",
        unigram_path="",
        wordpiece_path="",
        pssm_index_path="",
        psfm_index_path="",
        frequency_profile_index_path="",
        aux_encoder="",
    )

    featurizer = build_classic_featurizer(
        args,
        str(tmp_path),
        [("sample", "AX", "C", 1)],
    )
    encoded = featurizer.encode("AX")

    assert isinstance(encoded, ResidueModelInput)
    assert encoded.mask.tolist() == [True, True, False, False]
    assert torch.count_nonzero(encoded.features[1]) == 0


def test_residue_gcn_masked_values_do_not_change_logits_or_receive_gradient():
    torch.manual_seed(11)
    model = ResidueGraphClassifier(4, hidden_dim=8, num_classes=2, dropout=0.0)
    model.eval()
    features = torch.randn(1, 5, 4)
    mask = torch.tensor([[True, True, False, True, False]])
    clean = ResidueModelInput(features.clone(), mask)
    polluted_features = features.clone()
    polluted_features[:, 2] = 123.0
    polluted_features[:, 4] = -456.0
    polluted_features.requires_grad_()

    clean_logits = model(clean)
    polluted_logits = model(ResidueModelInput(polluted_features, mask))
    polluted_logits.sum().backward()

    assert torch.allclose(clean_logits, polluted_logits, atol=1e-6)
    assert torch.count_nonzero(polluted_features.grad[:, ~mask[0]]) == 0
    assert torch.isfinite(polluted_features.grad).all()


def test_residue_gcn_forward_embedding_batch_and_reload_contract(tmp_path):
    torch.manual_seed(23)
    model = ResidueGraphClassifier(4, hidden_dim=8, num_classes=3, dropout=0.0).eval()
    features = torch.randn(2, 4, 4)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    inputs = ResidueModelInput(features, mask)

    logits, embedding = model.forward_with_embedding(inputs)
    singleton = model(ResidueModelInput(features[:1], mask[:1]))

    assert logits.shape == (2, 3)
    assert embedding.shape == (2, 16)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits[:1], singleton, atol=1e-6)

    checkpoint = tmp_path / "gcn.pt"
    torch.save(model, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
    assert gcn_predictor_contract(loaded)["status"] == "verified_platform_variant"
    assert torch.equal(loaded(inputs), logits)


def test_current_residue_gcn_rejects_invalid_shape_dtype_and_empty_samples():
    model = ResidueGraphClassifier(4, hidden_dim=8, num_classes=2, dropout=0.0)
    pretrained = PretrainedResiduePredictor(
        "gcn",
        input_dim=4,
        adapter_dim=8,
        hidden_dim=8,
        heads=2,
        num_classes=2,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="shapes"):
        model(torch.ones(2, 4))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.ones(2, 3, 4, dtype=torch.long))
    with pytest.raises(TypeError, match="floating-point"):
        pretrained(
            ResidueModelInput(
                torch.ones(2, 3, 4, dtype=torch.long),
                torch.ones(2, 3, dtype=torch.bool),
            )
        )
    with pytest.raises(ValueError, match="at least one valid"):
        model(torch.zeros(2, 3, 4))
