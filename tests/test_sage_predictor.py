from argparse import Namespace

import pytest
import torch

from scripts.train_pair_pipeline import (
    SAGE_PREDICTOR_CONTRACT_VERSION,
    PairSAGEClassifier,
    _inductive_graph_scores,
    build_inductive_graph_structure,
    build_inductive_message_edges,
    build_sage_knn_graph_edge_index,
)


def _graph_args(**overrides):
    values = {
        "gnn_k": 2,
        "gnn_exact_knn_max_nodes": 100,
        "gnn_rp_projections": 3,
        "gnn_rp_window": 2,
        "seed": 42,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("exact_max_nodes", [100, 3])
def test_sage_reference_topology_has_unique_nonself_neighbors(exact_max_nodes):
    torch.manual_seed(31)
    features = torch.randn(7, 5)

    edge_index = build_sage_knn_graph_edge_index(
        features,
        k=2,
        seed=42,
        exact_max_nodes=exact_max_nodes,
        rp_projections=3,
        rp_window=2,
    )

    assert edge_index.shape == (2, 14)
    assert torch.unique(edge_index, dim=1).shape[1] == 14
    src, dst = edge_index
    assert not torch.any(src == dst)
    for node in range(7):
        incoming = src[dst == node]
        assert incoming.shape[0] == 2
        assert torch.unique(incoming).shape[0] == 2


def test_sage_inductive_topology_is_reference_only_and_query_independent():
    torch.manual_seed(37)
    reference = torch.randn(7, 5)
    query = torch.randn(3, 5)
    args = _graph_args()

    edge_index = build_inductive_graph_structure(
        args,
        reference,
        query,
        "cpu",
        "sage",
    )

    assert edge_index.shape == (2, 20)
    assert torch.unique(edge_index, dim=1).shape[1] == 20
    src, dst = edge_index
    assert not torch.any(src == dst)
    assert torch.all(src[dst >= 7] < 7)
    assert not torch.any((src >= 7) & (dst < 7))
    assert not torch.any((src >= 7) & (dst >= 7))


def test_sage_root_and_neighbor_paths_are_independent():
    model = PairSAGEClassifier(2, hidden_dim=2, dropout=0.0).eval()
    with torch.no_grad():
        model.in_proj.weight.copy_(torch.eye(2))
        model.in_proj.bias.zero_()
        model.lin_self1.weight.copy_(torch.eye(2))
        model.lin_self1.bias.zero_()
        model.lin_nei1.weight.copy_(torch.eye(2))
        model.lin_nei1.bias.zero_()
        model.lin_self2.weight.copy_(torch.eye(2))
        model.lin_self2.bias.zero_()
        model.lin_nei2.weight.zero_()
        model.lin_nei2.bias.zero_()
        model.head[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.head[0].bias.zero_()
        model.head[3].weight.fill_(1.0)
        model.head[3].bias.zero_()

    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    base = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    neighbor_changed = base.clone()
    neighbor_changed[0, 0] = 2.0
    root_changed = base.clone()
    root_changed[1, 0] = 2.0

    base_score = model(base, edge_index)[1]
    neighbor_score = model(neighbor_changed, edge_index)[1]
    root_score = model(root_changed, edge_index)[1]

    assert base_score.item() == pytest.approx(1.0)
    assert neighbor_score.item() == pytest.approx(3.0)
    assert root_score.item() == pytest.approx(2.0)


def test_sage_contract_version_and_input_validation():
    model = PairSAGEClassifier(4, hidden_dim=8, dropout=0.0)
    assert model.sage_predictor_contract_version == SAGE_PREDICTOR_CONTRACT_VERSION
    with pytest.raises(ValueError, match="hidden_dim must be at least 2"):
        PairSAGEClassifier(4, hidden_dim=1)
    with pytest.raises(ValueError, match="dropout must be in"):
        PairSAGEClassifier(4, dropout=1.0)


def test_sage_legacy_message_topology_remains_distinct():
    reference = torch.randn(7, 5)
    args = _graph_args()
    legacy = build_inductive_message_edges(args, reference, None, "cpu")
    assert legacy.shape[1] == 49
    assert torch.unique(legacy, dim=1).shape[1] < legacy.shape[1]


def test_sage_query_score_is_invariant_to_other_queries():
    torch.manual_seed(43)
    args = _graph_args()
    reference = torch.randn(7, 4)
    first_query = torch.randn(1, 4)
    other_query = torch.randn(1, 4)
    model = PairSAGEClassifier(4, hidden_dim=8, dropout=0.0).eval()

    single_structure = build_inductive_graph_structure(args, reference, first_query, "cpu", "sage")
    batch_query = torch.cat([first_query, other_query], dim=0)
    batch_structure = build_inductive_graph_structure(args, reference, batch_query, "cpu", "sage")
    single = _inductive_graph_scores(model, reference, first_query, single_structure)[0]
    batched = _inductive_graph_scores(model, reference, batch_query, batch_structure)[0]

    assert single == pytest.approx(batched, abs=1e-6)
