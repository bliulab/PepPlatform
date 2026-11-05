from argparse import Namespace
from pathlib import Path

import pytest
import torch

from pepplatform.models import (
    GAT_PREDICTOR_CONTRACT_VERSION,
    PretrainedResiduePredictor,
    ResidueGATLayer,
    ResidueGraphClassifier,
    ResidueModelInput,
    gat_predictor_contract,
    sequence_predictor_contract,
)
from scripts.train_pair_pipeline import (
    build_gat_knn_graph_edge_index,
    build_inductive_gat_edges,
    build_inductive_message_edges,
)


def test_residue_gat_contract_records_chain_attention_assembly():
    model = ResidueGraphClassifier(
        in_dim=5,
        hidden_dim=12,
        heads=4,
        num_classes=2,
        dropout=0.0,
        graph_kind="gat",
    )

    contract = gat_predictor_contract(model)

    assert model.gat_predictor_contract_version == GAT_PREDICTOR_CONTRACT_VERSION
    assert sequence_predictor_contract(model) == contract
    assert contract["status"] == "verified_platform_variant"
    assert contract["implementation"] == "pepplatform_masked_chain_gat"
    assert contract["heads"] == 4
    assert contract["attention"] == (
        "multi-head additive neighbor attention with destination-wise softmax"
    )
    assert contract["readout"] == "masked mean plus masked max"


def test_pretrained_residue_gat_contract_records_adapter_and_mask():
    model = PretrainedResiduePredictor(
        "gat",
        input_dim=5,
        adapter_dim=8,
        hidden_dim=8,
        heads=2,
        num_classes=2,
        dropout=0.0,
    )

    contract = gat_predictor_contract(model)

    assert contract["input_adapter"] == "layer_norm_linear_projection"
    assert contract["padding_mask"] == "explicit_residue_mask"
    assert contract["hidden_dim"] == 8
    assert contract["heads"] == 2


def test_gat_without_contract_attribute_is_legacy_only():
    model = ResidueGraphClassifier(
        5,
        num_classes=2,
        graph_kind="gat",
        dropout=0.0,
    )
    del model.gat_predictor_contract_version

    contract = gat_predictor_contract(model)

    assert contract["status"] == "legacy_only"
    assert contract["contract_version"] == 0
    assert contract["padding_mask"] == "legacy_route_dependent"


def test_residue_gat_masked_values_do_not_change_logits_or_receive_gradient():
    torch.manual_seed(11)
    model = ResidueGraphClassifier(
        4,
        hidden_dim=8,
        heads=2,
        num_classes=2,
        graph_kind="gat",
        dropout=0.0,
    ).eval()
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

    torch.testing.assert_close(clean_logits, polluted_logits)
    assert torch.count_nonzero(polluted_features.grad[:, ~mask[0]]) == 0
    assert torch.isfinite(polluted_features.grad).all()


def test_residue_gat_forward_embedding_batch_and_reload_contract(tmp_path: Path):
    torch.manual_seed(23)
    model = ResidueGraphClassifier(
        4,
        hidden_dim=8,
        heads=2,
        num_classes=3,
        graph_kind="gat",
        dropout=0.0,
    ).eval()
    features = torch.randn(2, 4, 4)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    inputs = ResidueModelInput(features, mask)

    logits, embedding = model.forward_with_embedding(inputs)
    singleton = model(ResidueModelInput(features[:1], mask[:1]))

    assert logits.shape == (2, 3)
    assert embedding.shape == (2, 16)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits[:1], singleton)

    checkpoint = tmp_path / "gat.pt"
    torch.save(model, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
    assert gat_predictor_contract(loaded)["status"] == "verified_platform_variant"
    torch.testing.assert_close(loaded(inputs), logits)


def test_current_residue_gat_rejects_invalid_kind_heads_dtype_and_empty_samples():
    with pytest.raises(ValueError, match="unsupported residue graph kind"):
        ResidueGraphClassifier(4, graph_kind="attention")
    with pytest.raises(ValueError, match="heads must be positive"):
        ResidueGATLayer(4, 8, heads=0)
    with pytest.raises(ValueError, match="divisible by heads"):
        ResidueGATLayer(4, 7, heads=2)

    model = ResidueGraphClassifier(
        4,
        hidden_dim=8,
        heads=2,
        num_classes=2,
        graph_kind="gat",
        dropout=0.0,
    )
    pretrained = PretrainedResiduePredictor(
        "gat",
        input_dim=4,
        adapter_dim=8,
        hidden_dim=8,
        heads=2,
        num_classes=2,
        dropout=0.0,
    )

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


@pytest.mark.parametrize("exact_max_nodes", [100, 3])
def test_pair_gat_v2_reference_topology_has_exact_unique_neighbors_and_one_loop(
    exact_max_nodes,
):
    torch.manual_seed(31)
    features = torch.randn(7, 5)

    edge_index = build_gat_knn_graph_edge_index(
        features,
        k=2,
        seed=42,
        exact_max_nodes=exact_max_nodes,
        rp_projections=3,
        rp_window=2,
    )

    assert edge_index.shape == (2, 21)
    assert torch.unique(edge_index, dim=1).shape[1] == 21
    src, dst = edge_index
    for node in range(7):
        incoming = src[dst == node]
        assert incoming.shape[0] == 3
        assert int((incoming == node).sum()) == 1
        assert torch.unique(incoming[incoming != node]).shape[0] == 2


def test_pair_gat_v2_inductive_topology_is_unique_and_query_independent():
    torch.manual_seed(37)
    reference = torch.randn(7, 5)
    query = torch.randn(3, 5)
    args = Namespace(
        gnn_k=2,
        seed=42,
        gnn_exact_knn_max_nodes=100,
        gnn_rp_projections=3,
        gnn_rp_window=2,
    )

    edge_index = build_inductive_gat_edges(args, reference, query, "cpu")

    assert edge_index.shape == (2, 30)
    assert torch.unique(edge_index, dim=1).shape[1] == 30
    src, dst = edge_index
    for node in range(10):
        incoming = src[dst == node]
        assert incoming.shape[0] == 3
        assert int((incoming == node).sum()) == 1
        if node >= 7:
            assert torch.all(incoming[incoming != node] < 7)
    assert not torch.any((src >= 7) & (dst < 7))
    assert not torch.any((src >= 7) & (dst >= 7) & (src != dst))


def test_pair_gat_legacy_topology_retains_historical_duplicate_messages():
    torch.manual_seed(41)
    reference = torch.randn(7, 5)
    args = Namespace(
        gnn_k=2,
        seed=42,
        gnn_exact_knn_max_nodes=100,
        gnn_rp_projections=3,
        gnn_rp_window=2,
    )

    legacy = build_inductive_message_edges(args, reference, None, "cpu")
    current = build_inductive_gat_edges(args, reference, None, "cpu")

    assert legacy.shape[1] == 49
    assert torch.unique(legacy, dim=1).shape[1] < legacy.shape[1]
    assert current.shape[1] == 21
