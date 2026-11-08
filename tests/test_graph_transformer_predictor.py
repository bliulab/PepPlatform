from argparse import Namespace

import pytest
import torch

from scripts.train_pair_pipeline import (
    GRAPH_TRANSFORMER_PREDICTOR_CONTRACT_VERSION,
    GT_MODEL,
    LegacyPairGraphTransformerClassifier,
    PairGraphTransformerClassifier,
    _pair_graph_predictor_contract,
    build_graph_transformer_knn_graph_edge_index,
    build_inductive_graph_transformer_edges,
    build_inductive_message_edges,
)


def _graph_args(k=2, exact_max_nodes=100):
    return Namespace(
        gnn_k=k,
        seed=42,
        gnn_exact_knn_max_nodes=exact_max_nodes,
        gnn_rp_projections=3,
        gnn_rp_window=2,
    )


def _ring_edges(nodes):
    node = torch.arange(nodes, dtype=torch.long)
    return torch.stack(
        [torch.cat([node, torch.roll(node, 1)]), torch.cat([node, node])],
        dim=0,
    )


def test_graph_transformer_contract_records_identity_assembly():
    model = PairGraphTransformerClassifier(
        input_dim=5,
        hidden_dim=8,
        heads=2,
        dropout=0.0,
    )
    contract = _pair_graph_predictor_contract(GT_MODEL, hidden_dim=8)

    assert (
        model.graph_transformer_predictor_contract_version
        == GRAPH_TRANSFORMER_PREDICTOR_CONTRACT_VERSION
    )
    assert len(model.layers) == 2
    assert all(isinstance(layer.norm1, torch.nn.LayerNorm) for layer in model.layers)
    assert all(isinstance(layer.norm2, torch.nn.LayerNorm) for layer in model.layers)
    assert contract["status"] == "verified_platform_variant"
    assert contract["implementation"] == (
        "pepplatform_inductive_pair_sample_graph_graph_transformer"
    )
    assert contract["topology_contract"] == ("graph_transformer_unique_incoming_knn_v1")
    assert "scaled dot-product attention" in contract["attention"]
    assert "LayerNorm" in contract["transformer_block"]
    assert "post-norm" in contract["transformer_block"]


@pytest.mark.parametrize("exact_max_nodes", [100, 3])
def test_reference_topology_has_exact_unique_neighbors_and_one_loop(
    exact_max_nodes,
):
    torch.manual_seed(31)
    features = torch.randn(7, 5)

    edge_index = build_graph_transformer_knn_graph_edge_index(
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


def test_inductive_topology_is_unique_and_query_independent():
    torch.manual_seed(37)
    reference = torch.randn(7, 5)
    query = torch.randn(3, 5)

    edge_index = build_inductive_graph_transformer_edges(
        _graph_args(),
        reference,
        query,
        "cpu",
    )

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


def test_legacy_topology_and_model_retain_historical_behavior():
    torch.manual_seed(41)
    reference = torch.randn(7, 5)
    legacy_edges = build_inductive_message_edges(
        _graph_args(),
        reference,
        None,
        "cpu",
    )
    current_edges = build_inductive_graph_transformer_edges(
        _graph_args(),
        reference,
        None,
        "cpu",
    )
    legacy_model = LegacyPairGraphTransformerClassifier(
        input_dim=5,
        hidden_dim=8,
        heads=2,
        dropout=0.0,
    )

    assert legacy_edges.shape[1] == 49
    assert torch.unique(legacy_edges, dim=1).shape[1] < legacy_edges.shape[1]
    assert current_edges.shape[1] == 21
    assert "q1.weight" in legacy_model.state_dict()
    assert not any("norm" in key for key in legacy_model.state_dict())


def test_attention_matches_dense_destination_softmax_and_gradient():
    torch.manual_seed(43)
    model = PairGraphTransformerClassifier(4, hidden_dim=8, heads=2, dropout=0.0)
    layer = model.layers[0]
    h = torch.randn(4, 8, requires_grad=True)
    edge_index = _ring_edges(4)
    src, dst = edge_index

    actual = layer._attention(h, edge_index)
    q = layer.q_proj(h).view(4, layer.heads, layer.head_dim)
    k = layer.k_proj(h).view(4, layer.heads, layer.head_dim)
    v = layer.v_proj(h).view(4, layer.heads, layer.head_dim)
    score = (q[dst] * k[src]).sum(dim=-1) / float(layer.head_dim) ** 0.5
    aggregated = torch.zeros_like(v)
    for node in range(4):
        mask = dst == node
        weight = torch.softmax(score[mask], dim=0)
        aggregated[node] = (weight.unsqueeze(-1) * v[src[mask]]).sum(dim=0)
    expected = layer.out_proj(aggregated.reshape(4, 8))

    actual_grad = torch.autograd.grad(actual.square().sum(), h, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), h)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_forward_embedding_logits_gradients_graph_signal_and_reload():
    torch.manual_seed(47)
    model = PairGraphTransformerClassifier(4, hidden_dim=8, heads=2, dropout=0.0)
    features = torch.randn(5, 4, requires_grad=True)
    forward_edges = _ring_edges(5)
    reverse_edges = torch.stack(
        [forward_edges[1], forward_edges[0]],
        dim=0,
    )

    logits, embedding = model.forward_with_embedding(features, forward_edges)
    reverse_logits = model(features, reverse_edges)
    logits.square().sum().backward()

    assert logits.shape == (5, 1)
    assert embedding.shape == (5, 8)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(embedding).all()
    assert torch.isfinite(features.grad).all()
    assert not torch.allclose(logits, reverse_logits)

    reloaded = PairGraphTransformerClassifier(
        4,
        hidden_dim=8,
        heads=2,
        dropout=0.0,
    ).eval()
    reloaded.load_state_dict(model.state_dict())
    model.eval()
    torch.testing.assert_close(
        reloaded(features.detach(), forward_edges),
        model(features.detach(), forward_edges),
    )


def test_graph_transformer_emits_logits_not_probabilities():
    model = PairGraphTransformerClassifier(4, hidden_dim=8, heads=2, dropout=0.0)
    with torch.no_grad():
        model.head[-1].weight.zero_()
        model.head[-1].bias.fill_(2.0)

    logits = model(torch.ones(3, 4), _ring_edges(3))

    torch.testing.assert_close(logits, torch.full((3, 1), 2.0))
    assert torch.all(logits > 1.0)


def test_graph_transformer_rejects_invalid_constructor_and_inputs():
    with pytest.raises(ValueError, match="input_dim"):
        PairGraphTransformerClassifier(0)
    with pytest.raises(ValueError, match="heads"):
        PairGraphTransformerClassifier(4, hidden_dim=8, heads=0)
    with pytest.raises(ValueError, match="divisible"):
        PairGraphTransformerClassifier(4, hidden_dim=7, heads=4)
    with pytest.raises(ValueError, match="layers"):
        PairGraphTransformerClassifier(4, hidden_dim=8, heads=2, layers=0)
    with pytest.raises(ValueError, match="dropout"):
        PairGraphTransformerClassifier(4, hidden_dim=8, heads=2, dropout=1.0)

    model = PairGraphTransformerClassifier(4, hidden_dim=8, heads=2, dropout=0.0)
    edges = _ring_edges(3)
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.ones(3, 4, dtype=torch.long), edges)
    with pytest.raises(ValueError, match="rank-2"):
        model(torch.ones(1, 3, 4), edges)
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((3, 4), float("nan")), edges)
    with pytest.raises(TypeError, match="torch.long"):
        model(torch.ones(3, 4), edges.float())
    with pytest.raises(ValueError, match="E > 0"):
        model(torch.ones(3, 4), torch.empty((2, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="invalid node index"):
        model(torch.ones(3, 4), torch.tensor([[0, 3, 2], [0, 1, 2]]))
    with pytest.raises(ValueError, match="every node"):
        model(torch.ones(3, 4), torch.tensor([[0, 1], [0, 1]]))
