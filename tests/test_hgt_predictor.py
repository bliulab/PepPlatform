from argparse import Namespace
import json

import numpy as np
import pytest
import torch

from pepplatform.web.inference import ModelService
from scripts.train_pair_pipeline import (
    HGT_GRAPH_SCHEMA,
    HGT_PREDICTOR_CONTRACT_VERSION,
    HGT_REL_PARTNER_SELF,
    HGT_REL_PARTNER_SEMANTIC,
    HGT_REL_PEPTIDE_SELF,
    HGT_REL_PEPTIDE_SEMANTIC,
    LegacyPairHGTClassifier,
    PairHGTClassifier,
    _TypedHGTLayer,
    _pair_graph_predictor_contract,
    build_query_reference_edge_index,
    build_inductive_hgt_graph,
)


def _graph_args(k=2):
    return Namespace(
        gnn_k=k,
        seed=42,
        gnn_exact_knn_max_nodes=100,
        gnn_rp_projections=2,
        gnn_rp_window=0,
    )


def _stream_features():
    peptide = torch.tensor(
        [[1.0, 0.0, 0.1], [0.0, 1.0, 0.2], [0.8, 0.2, 0.3], [0.2, 0.8, 0.4]],
        dtype=torch.float,
    )
    partner = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.7, 0.3, 0.0, 0.3],
            [0.3, 0.7, 0.0, 0.4],
        ],
        dtype=torch.float,
    )
    return peptide, partner


def test_hgt_contract_records_typed_relation_aware_edge_prediction():
    contract = _pair_graph_predictor_contract("hgt", 16)

    assert contract["contract_version"] == HGT_PREDICTOR_CONTRACT_VERSION
    assert contract["status"] == "verified_platform_variant"
    assert contract["implementation"] == "pepplatform_inductive_typed_pair_hgt"
    assert contract["topology_contract"] == HGT_GRAPH_SCHEMA
    assert "node-type-specific" in contract["attention"]
    assert contract["target_relation"].startswith("ppi edge decoded")
    assert contract["prediction_output"] == "edge_logits_then_sigmoid_probability"


def test_hgt_graph_has_role_types_relations_and_strict_query_direction():
    peptide, partner = _stream_features()
    query_peptide = torch.tensor([[0.9, 0.1, 0.5], [0.1, 0.9, 0.6]])
    query_partner = torch.tensor([[0.8, 0.2, 0.0, 0.5], [0.2, 0.8, 0.0, 0.6]], dtype=torch.float)

    graph = build_inductive_hgt_graph(_graph_args(), peptide, partner, query_peptide, query_partner)

    assert graph.reference_pairs == 4
    assert graph.node_type.tolist() == [0] * 6 + [1] * 6
    assert set(graph.edge_type.tolist()) == {
        HGT_REL_PEPTIDE_SEMANTIC,
        HGT_REL_PARTNER_SEMANTIC,
        HGT_REL_PEPTIDE_SELF,
        HGT_REL_PARTNER_SELF,
    }
    src, dst = graph.edge_index
    semantic = graph.edge_type == HGT_REL_PEPTIDE_SEMANTIC
    peptide_query = semantic & (dst >= 4)
    assert torch.all(src[peptide_query] < 4)
    semantic = graph.edge_type == HGT_REL_PARTNER_SEMANTIC
    partner_query = semantic & (dst >= 10)
    assert torch.all((src[partner_query] >= 6) & (src[partner_query] < 10))
    assert not torch.any((src >= 4) & (src < 6) & (dst < 4))
    assert not torch.any((src >= 10) & (dst < 10))


def test_hgt_forward_uses_relation_parameters_and_has_finite_gradients():
    torch.manual_seed(13)
    peptide, partner = _stream_features()
    graph = build_inductive_hgt_graph(_graph_args(), peptide, partner)
    model = PairHGTClassifier(3, 4, hidden_dim=8, heads=2, dropout=0.0).eval()
    peptide.requires_grad_()
    partner.requires_grad_()

    logits, embedding = model.forward_with_embedding(peptide, partner, graph)
    changed_relation = graph.edge_type.clone()
    first_semantic = torch.nonzero(changed_relation == HGT_REL_PEPTIDE_SEMANTIC, as_tuple=False)[
        0, 0
    ]
    changed_relation[first_semantic] = HGT_REL_PARTNER_SEMANTIC
    changed_graph = type(graph)(
        graph.edge_index,
        changed_relation,
        graph.node_type,
        graph.pair_index,
        graph.reference_pairs,
    )
    changed_logits = model(peptide, partner, changed_graph)
    logits.sum().backward()

    assert logits.shape == (4, 1)
    assert embedding.shape == (4, 32)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(peptide.grad).all()
    assert torch.isfinite(partner.grad).all()
    assert not torch.allclose(logits, changed_logits)
    relation_gradient = model.hgt_layers[0].relation_att.grad
    assert relation_gradient is not None and torch.isfinite(relation_gradient).all()
    assert torch.count_nonzero(relation_gradient) > 0


def test_hgt_grouped_relation_transform_matches_expanded_reference_and_gradients():
    torch.manual_seed(31)
    layer = _TypedHGTLayer(
        hidden_dim=8,
        heads=2,
        dropout=0.0,
        edge_chunk_size=2,
    )
    projected = torch.randn(6, 2, 4, requires_grad=True)
    src = torch.tensor([0, 1, 2, 3, 4, 5, 0, 2])
    relation = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    grouped = layer._relation_transform(
        projected,
        src,
        relation,
        layer.relation_att,
    )
    reference = torch.einsum(
        "ehd,ehdf->ehf",
        projected[src],
        layer.relation_att[relation],
    )
    torch.testing.assert_close(grouped, reference)

    grouped.sum().backward(retain_graph=True)
    grouped_projected_grad = projected.grad.detach().clone()
    grouped_weight_grad = layer.relation_att.grad.detach().clone()
    projected.grad.zero_()
    layer.relation_att.grad.zero_()
    reference.sum().backward()
    torch.testing.assert_close(projected.grad, grouped_projected_grad)
    torch.testing.assert_close(layer.relation_att.grad, grouped_weight_grad)


def test_hgt_query_prediction_is_independent_of_other_queries():
    torch.manual_seed(21)
    peptide, partner = _stream_features()
    first_peptide = torch.tensor([[0.9, 0.1, 0.5]])
    first_partner = torch.tensor([[0.8, 0.2, 0.0, 0.5]])
    other_peptide = torch.tensor([[0.1, 0.9, 0.6]])
    other_partner = torch.tensor([[0.2, 0.8, 0.0, 0.6]])
    model = PairHGTClassifier(3, 4, hidden_dim=8, heads=2, dropout=0.0).eval()

    single_graph = build_inductive_hgt_graph(
        _graph_args(), peptide, partner, first_peptide, first_partner
    )
    batch_peptide = torch.cat([first_peptide, other_peptide])
    batch_partner = torch.cat([first_partner, other_partner])
    batch_graph = build_inductive_hgt_graph(
        _graph_args(), peptide, partner, batch_peptide, batch_partner
    )
    single = model(
        torch.cat([peptide, first_peptide]),
        torch.cat([partner, first_partner]),
        single_graph,
    )[len(peptide)]
    batched = model(
        torch.cat([peptide, batch_peptide]),
        torch.cat([partner, batch_partner]),
        batch_graph,
    )[len(peptide)]

    torch.testing.assert_close(single, batched, rtol=0.0, atol=1e-6)


def test_query_neighbors_do_not_change_when_batch_crosses_knn_mode_boundary():
    torch.manual_seed(2)
    reference = torch.randn(8, 4)
    first_query = torch.randn(1, 4)
    other_queries = torch.randn(2, 4)
    kwargs = {
        "k": 3,
        "seed": 2,
        "exact_max_nodes": 9,
        "rp_projections": 2,
        "rp_window": 0,
    }

    single = build_query_reference_edge_index(reference, first_query, **kwargs)
    batched = build_query_reference_edge_index(
        reference,
        torch.cat([first_query, other_queries], dim=0),
        **kwargs,
    )

    torch.testing.assert_close(single, batched[:, : single.shape[1]])


def test_hgt_rejects_invalid_shapes_dtypes_and_hyperparameters():
    with pytest.raises(ValueError, match="divisible"):
        PairHGTClassifier(3, 4, hidden_dim=7, heads=2)
    model = PairHGTClassifier(3, 4, hidden_dim=8, heads=2, dropout=0.0)
    peptide, partner = _stream_features()
    graph = build_inductive_hgt_graph(_graph_args(), peptide, partner)
    with pytest.raises(TypeError, match="floating-point"):
        model(peptide.long(), partner, graph)
    with pytest.raises(ValueError, match="same number"):
        build_inductive_hgt_graph(_graph_args(), peptide[:2], partner)


def test_hgt_web_loader_selects_current_and_legacy_contracts(tmp_path):
    current_model = PairHGTClassifier(3, 4, hidden_dim=8, heads=2, dropout=0.0)
    torch.save({"state_dict": current_model.state_dict()}, tmp_path / "model.pt")
    current, _payload, predictor = ModelService._load_pair_graph_model(
        tmp_path,
        {
            "predictor_contract": {
                "implementation": "pepplatform_inductive_typed_pair_hgt",
                "contract_version": 1,
            },
            "graph": {
                "predictor": "hgt",
                "peptide_input_dim": 3,
                "partner_input_dim": 4,
                "hidden_dim": 8,
                "heads": 2,
                "layers": 2,
                "dropout": 0.0,
            },
        },
        input_dim=0,
        device=torch.device("cpu"),
    )
    assert predictor == "hgt"
    assert current.hgt_predictor_contract_version == 1

    legacy_model = LegacyPairHGTClassifier(12, hidden_dim=8, heads=2, dropout=0.0)
    torch.save({"state_dict": legacy_model.state_dict()}, tmp_path / "model.pt")
    legacy, _payload, legacy_predictor = ModelService._load_pair_graph_model(
        tmp_path,
        {"graph": {"predictor": "hgt", "hidden_dim": 8, "dropout": 0.0}},
        input_dim=12,
        device=torch.device("cpu"),
    )
    assert legacy_predictor == "hgt"
    assert isinstance(legacy, LegacyPairHGTClassifier)
    assert not hasattr(legacy, "hgt_predictor_contract_version")


def test_typed_hgt_web_prediction_uses_role_specific_references(tmp_path, monkeypatch):
    def fake_pair_embeddings(metadata, pairs, device, bundle=None):
        del metadata, device, bundle
        peptide = torch.tensor([[1.0, 0.0, 0.1] for _ in pairs])
        partner = torch.tensor([[0.0, 1.0, 0.0, 0.2] for _ in pairs])
        return peptide, partner

    monkeypatch.setattr(
        ModelService,
        "_extract_pair_embeddings",
        staticmethod(fake_pair_embeddings),
    )
    model = PairHGTClassifier(3, 4, hidden_dim=8, heads=2, dropout=0.0)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save({"state_dict": model.state_dict()}, tmp_path / "model.pt")
    torch.save(
        {
            "reference_peptide_features": torch.tensor(
                [[1.0, 0.0, 0.1], [0.0, 1.0, 0.2], [0.8, 0.2, 0.3]]
            ),
            "reference_protein_features": torch.tensor(
                [
                    [0.0, 1.0, 0.0, 0.1],
                    [1.0, 0.0, 0.0, 0.2],
                    [0.2, 0.8, 0.0, 0.3],
                ]
            ),
        },
        tmp_path / "graph_reference.pt",
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "bundle_type": "pair_graph_nn",
                "supports_inference": True,
                "input_schema": "peptide_protein_pairs",
                "label_names": ["interaction"],
                "threshold": 0.4,
                "pretrained": {
                    "tokenizer": "esm2",
                    "model_name": "models/esm2",
                    "max_length": 2,
                    "input_dim": 3,
                },
                "pair": {"input_adapter": "typed_role_projection"},
                "predictor_contract": {
                    "implementation": "pepplatform_inductive_typed_pair_hgt",
                    "contract_version": 1,
                },
                "graph": {
                    "predictor": "hgt",
                    "schema": HGT_GRAPH_SCHEMA,
                    "gnn_k": 1,
                    "hidden_dim": 8,
                    "heads": 2,
                    "layers": 2,
                    "dropout": 0.0,
                    "peptide_input_dim": 3,
                    "partner_input_dim": 4,
                    "reference_path": "graph_reference.pt",
                    "aligned_gated": False,
                },
                "args": {
                    "task_type": "pep_protein_interaction",
                    "tokenizer": "esm2",
                    "encoder": "ppi_fusion",
                    "model": "hgt",
                    "pair_strategy": "two_stream",
                    "interaction_level": "vector",
                    "predictor_type": "sample_graph",
                },
            }
        ),
        encoding="utf-8",
    )

    class _Store:
        def get_model(self, model_id):
            return {
                "id": model_id,
                "bundle_path": str(tmp_path),
                "supports_inference": True,
            }

    response = ModelService(_Store()).predict(
        "model",
        pairs=[{"peptide": "AC", "protein": "GG"}],
    )

    assert response["predictions"][0]["score"] == 0.5
    assert response["predictions"][0]["is_interaction"] is True


def test_hgt_eval_batch_outputs_are_finite():
    peptide, partner = _stream_features()
    graph = build_inductive_hgt_graph(_graph_args(), peptide, partner)
    model = PairHGTClassifier(3, 4, hidden_dim=8, heads=2, dropout=0.0).eval()
    scores = torch.sigmoid(model(peptide, partner, graph)).detach().numpy()
    assert scores.shape == (4, 1)
    assert np.isfinite(scores).all()
