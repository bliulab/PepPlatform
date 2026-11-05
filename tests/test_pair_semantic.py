from argparse import Namespace

import torch

from pepplatform.models import AlignedGatedClassifier, GatedResidualClassifier
from pepplatform.semantic import (
    INTERACTION_PROFILE_JACCARD,
    SemanticPartnerTable,
    SemanticRelation,
    SemanticTrainingProfile,
    normalize_semantic_aliases,
    prepare_semantic_projectors,
)
from scripts import train_pair_pipeline as pair_pipeline


def test_pair_semantic_auxiliary_forward_is_differentiable_and_partner_only():
    args = Namespace(
        task_type="pep_protein_interaction",
        model="mlp",
        enable_semantic_similarity=True,
        semantic_similarity_weight=0.1,
        semantic_zero_weight=0.25,
        semantic_pair_mode="fused",
        semantic_projection_dim=16,
        semantic_contrastive_temperature=0.1,
        semantic_positive_partners=1,
        semantic_zero_partners=8,
    )
    normalize_semantic_aliases(args)
    model = GatedResidualClassifier(input_dim=4, hidden_dim=8, num_classes=1, dropout=0.2)
    prepare_semantic_projectors(model, args)
    model.train()
    full_inputs = torch.randn(4, 4)
    main_logits = model(full_inputs[:2])
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        main_logits,
        torch.tensor([[1.0], [0.0]]),
    )
    table = SemanticPartnerTable(
        relation_type=INTERACTION_PROFILE_JACCARD,
        num_rows=4,
        seed=42,
        relations=(
            SemanticRelation(0, 2, 0.5, "positive"),
            SemanticRelation(0, 3, 0.0, "zero"),
        ),
        source_summary={"positive_edges": 2},
    )
    profile = SemanticTrainingProfile(
        relation_type=INTERACTION_PROFILE_JACCARD,
        seed=42,
        fused_table=table,
        peptide_table=None,
        context_table=None,
        row_peptide_ids=(0, 1, 2, 3),
        row_context_ids=(0, 1, 2, 3),
        peptide_example_rows=(),
        context_example_rows=(),
        source_summary={"positive_edges": 2},
        config={"semantic_pair_mode": "fused"},
    )

    total_loss, semantic = pair_pipeline.add_partner_semantic_loss(
        args,
        model,
        classification_loss,
        profile,
        torch.tensor([0, 1]),
        full_inputs,
        torch.device("cpu"),
    )

    assert semantic is not None
    assert semantic.requires_grad
    assert total_loss.requires_grad
    assert model.training is True
    total_loss.backward()
    assert model.input.weight.grad is not None
    assert torch.isfinite(model.input.weight.grad).all()


def test_aligned_gated_role_embeddings_are_differentiable_and_unmixed():
    model = AlignedGatedClassifier(
        peptide_dim=4,
        partner_dim=5,
        align_dim=8,
        hidden_dim=16,
        dropout=0.0,
    ).train()
    peptide = torch.randn(3, 4, requires_grad=True)
    partner = torch.randn(3, 5, requires_grad=True)

    logits, fused, peptide_embedding, partner_embedding = model.forward_with_role_embeddings(
        peptide, partner
    )

    assert tuple(logits.shape) == (3, 1)
    assert fused.shape[-1] == model.semantic_embedding_dim
    assert tuple(peptide_embedding.shape) == (3, model.semantic_role_embedding_dims[0])
    assert tuple(partner_embedding.shape) == (3, model.semantic_role_embedding_dims[1])
    total = logits.sum() + fused.sum() + peptide_embedding.sum() + partner_embedding.sum()
    total.backward()
    assert peptide.grad is not None
    assert partner.grad is not None
    assert torch.isfinite(peptide.grad).all()


def test_read_pair_csv_separates_stable_hla_context_from_model_pseudosequence(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "sample_id,peptide,hla_pair,pair_pseudo,label\n"
        "s1,ACDE,HLA-DPA1*01:03__HLA-DPB1*04:01,PSEUDOSEQ,1\n",
        encoding="utf-8",
    )

    rows, contexts = pair_pipeline.read_ppi_csv(csv_path, return_contexts=True)

    assert rows == [("s1", "ACDE", "PSEUDOSEQ", 1)]
    assert contexts == ["HLA-DPA1*01:03__HLA-DPB1*04:01"]


def test_read_pair_csv_uses_mhc_allele_for_graph_and_mapped_pseudosequence_for_model(tmp_path):
    csv_path = tmp_path / "pairs.csv"
    csv_path.write_text(
        "sample_id,peptide,mhc,label\ns1,ACDE,HLA-A*01:01,1\n",
        encoding="utf-8",
    )

    rows, contexts = pair_pipeline.read_ppi_csv(
        csv_path,
        mhc_pseudo_map={"HLA-A*01:01": "MAPPEDPSEUDO"},
        return_contexts=True,
    )

    assert rows == [("s1", "ACDE", "MAPPEDPSEUDO", 1)]
    assert contexts == ["HLA-A*01:01"]


def test_pair_artifact_metadata_records_semantic_provenance_and_epoch_order_hashes():
    args = Namespace(
        predictor_type="classifier",
        seed=42,
        _semantic_provenance={
            "relation_type": INTERACTION_PROFILE_JACCARD,
            "source_split": "train",
        },
        _main_batch_order_hashes=[
            {"epoch": 1, "num_rows": 4, "sha256": "a" * 64},
        ],
    )

    metadata = pair_pipeline._pair_protocol_metadata(args)

    assert metadata["semantic"]["relation_type"] == INTERACTION_PROFILE_JACCARD
    sampler = metadata["semantic"]["main_batch_sampler"]
    assert sampler["implementation"] == "torch.utils.data.RandomSampler"
    assert sampler["seed"] == 42
    assert sampler["epoch_order_hashes"][0]["sha256"] == "a" * 64
