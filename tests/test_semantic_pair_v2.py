import io
from types import SimpleNamespace

import numpy as np
import torch

from pepplatform.models import AlignedGatedClassifier, PairSemanticResidualAdapter
from pepplatform.semantic import (
    EntityProfileRelationSource,
    IntrinsicLabelRelationSource,
    InteractionProfileRelationSource,
    SemanticPartnerTable,
    SemanticRelation,
    SemanticTrainingProfile,
    auxiliary_eval_forward,
    build_interaction_dense_hard_partner_table,
    build_partner_table,
    placebo_shuffle_table,
    prepare_semantic_projectors,
    save_semantic_training_profile,
    semantic_representation_contract,
)


def _reference_build_partner_table(
    source,
    *,
    seed,
    positive_partners_per_anchor=1,
    zero_partners_per_anchor=8,
):
    """Slow reference that exactly matches the pre-fast-path generic builder."""

    relations = []
    for anchor_index in range(source.num_rows):
        positive_partners = []
        for positive_pos in range(positive_partners_per_anchor):
            candidate = source.positive_partner(
                anchor_index,
                seed + 7919 * (positive_pos + 1),
            )
            if candidate is None or candidate in positive_partners:
                continue
            similarity, valid = source.similarity(anchor_index, candidate)
            if not valid or similarity <= 0.0:
                raise RuntimeError("invalid positive relation in reference")
            positive_partners.append(candidate)
            relations.append(SemanticRelation(anchor_index, candidate, similarity, "positive"))
        if not positive_partners:
            continue
        zero_partners = []
        for zero_pos in range(zero_partners_per_anchor):
            candidate = source.zero_partner(
                anchor_index,
                seed + 104729 * (zero_pos + 1),
            )
            if candidate is None or candidate in zero_partners or candidate in positive_partners:
                continue
            similarity, valid = source.similarity(anchor_index, candidate)
            if valid and similarity == 0.0:
                zero_partners.append(candidate)
                relations.append(SemanticRelation(anchor_index, candidate, 0.0, "zero"))
    return SemanticPartnerTable(
        relation_type=source.relation_type,
        num_rows=source.num_rows,
        seed=seed,
        relations=tuple(relations),
        source_summary=source.summary(),
    )


def _small_interaction_source():
    rows = [
        ("r0", "P1", "C1", 1),
        ("r1", "P1", "C2", 1),
        ("r2", "P2", "C1", 1),
        ("r3", "P3", "C3", 1),
        ("r4", "P2", "C2", 0),
        ("r5", "P4", "C4", 0),
        ("r6", "P5", "C5", 0),
        ("r7", "P6", "C6", 0),
        ("r8", "P7", "C7", 0),
    ]
    contexts = ["C1", "C2", "C1", "C3", "C2", "C4", "C5", "C6", "C7"]
    return InteractionProfileRelationSource(rows, contexts)


def test_fast_interaction_table_matches_reference():
    source = _small_interaction_source()
    for seed in (1, 2, 3, 42):
        fast = build_partner_table(source, seed=seed)
        reference = _reference_build_partner_table(source, seed=seed)
        assert fast.relations == reference.relations


def test_fast_entity_table_matches_reference():
    source = _small_interaction_source()
    for entity_source in (
        source.peptide_entity_source(),
        source.context_entity_source(),
    ):
        for seed in (1, 2, 42):
            fast = build_partner_table(entity_source, seed=seed)
            reference = _reference_build_partner_table(entity_source, seed=seed)
            assert fast.relations == reference.relations


def test_placebo_fixed_table_preserves_shape_but_shuffles_partners():
    source = _small_interaction_source()
    table = build_partner_table(source, seed=7)
    placebo = placebo_shuffle_table(table, seed=7, tag="interaction-test-placebo")
    assert placebo.relation_type == table.relation_type
    assert placebo.num_rows == table.num_rows
    true_by_anchor = table.relations_by_anchor
    placebo_by_anchor = placebo.relations_by_anchor
    assert sorted(placebo_by_anchor) == sorted(true_by_anchor)
    changed = False
    for anchor in true_by_anchor:
        true_items = sorted((r.kind, r.similarity, r.partner_index) for r in true_by_anchor[anchor])
        placebo_items = sorted(
            (r.kind, r.similarity, r.partner_index) for r in placebo_by_anchor[anchor]
        )
        assert [item[:2] for item in placebo_items] == [item[:2] for item in true_items]
        if [item[2] for item in placebo_items] != [item[2] for item in true_items]:
            changed = True
    assert changed


def test_placebo_fixed_table_is_deterministic_and_has_no_self_pairs():
    source = _small_interaction_source()
    table = build_partner_table(source, seed=11)
    first = placebo_shuffle_table(table, seed=11, tag="fixed-placebo")
    second = placebo_shuffle_table(table, seed=11, tag="fixed-placebo")
    assert first.relations == second.relations
    assert all(relation.anchor_index != relation.partner_index for relation in first.relations)
    assert [
        (relation.anchor_index, relation.kind, relation.similarity) for relation in first.relations
    ] == [
        (relation.anchor_index, relation.kind, relation.similarity) for relation in table.relations
    ]


def test_fixed_placebo_reference_table_is_persisted(tmp_path):
    source = IntrinsicLabelRelationSource(
        torch.tensor(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 1],
            ],
            dtype=torch.float32,
        )
    )
    reference = build_partner_table(source, seed=17, zero_partners_per_anchor=1)
    placebo = placebo_shuffle_table(reference, seed=17, tag="fixed-placebo")
    profile = SemanticTrainingProfile(
        relation_type=source.relation_type,
        seed=17,
        fused_table=placebo,
        peptide_table=None,
        context_table=None,
        row_peptide_ids=(),
        row_context_ids=(),
        peptide_example_rows=(),
        context_example_rows=(),
        source_summary=source.summary(),
        config={
            "semantic_pair_mode": "fused",
            "semantic_loss_space": "projection",
            "semantic_partner_scheme": "fixed_placebo",
        },
    )
    provenance = save_semantic_training_profile(
        tmp_path,
        profile=profile,
        source=source,
        split_sample_ids={"train": [f"train:{i}" for i in range(5)]},
        placebo_reference_table=reference,
    )
    assert provenance["placebo"] is True
    assert provenance["placebo_source_scheme"] == "fixed"
    assert provenance["tables"]["fused"] == provenance["tables"]["placebo_reference"]
    assert provenance["placebo_structure_audit"]["fused"]["passed"] is True
    assert provenance["placebo_structure_audit"]["fused"]["changed_partner_count"] > 0
    assert (tmp_path / "semantic_placebo_reference_table.npz").is_file()


def test_pair_residual_adapter_forward_and_roundtrip():
    base = AlignedGatedClassifier(input_dim=8, hidden_dim=16)
    prepare_semantic_projectors(
        base,
        SimpleNamespace(
            enable_semantic_similarity=True,
            semantic_pair_mode="role_fused",
            semantic_projection_dim=8,
        ),
        sample_inputs=(
            torch.randn(3, 8),
            torch.randn(3, 8),
        ),
    )
    adapter = PairSemanticResidualAdapter(base, 0.5)
    adapter.eval()
    peptide = torch.randn(4, 8)
    partner = torch.randn(4, 8)
    logits, fused = adapter.forward_with_embedding(peptide, partner)
    role_logits, role_fused, pep_role, pro_role = adapter.forward_with_role_embeddings(
        peptide, partner
    )
    assert tuple(logits.shape) == (4, 1)
    assert torch.allclose(logits, role_logits)
    assert torch.allclose(fused, role_fused)
    assert tuple(pep_role.shape) == (4, 8)
    assert tuple(pro_role.shape) == (4, 8)

    buffer = io.BytesIO()
    torch.save(adapter, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=False)
    loaded.eval()
    logits_loaded, _ = loaded.forward_with_embedding(peptide, partner)
    assert torch.allclose(logits_loaded, logits)


def test_pair_learned_residual_contract_records_role_usage_and_gate():
    base = AlignedGatedClassifier(input_dim=8, hidden_dim=16)
    prepare_semantic_projectors(
        base,
        SimpleNamespace(
            enable_semantic_similarity=True,
            semantic_pair_mode="role_fused",
            semantic_projection_dim=8,
        ),
        sample_inputs=(torch.randn(3, 8), torch.randn(3, 8)),
    )
    adapter = PairSemanticResidualAdapter(base, 1.0, learned_gate=True)
    with torch.no_grad():
        adapter.gate.fill_(0.125)
    contract = semantic_representation_contract(
        adapter,
        relation_type="interaction_profile_jaccard",
        pair_mode="role_fused",
    )
    assert contract["contract_version"] == 2
    assert contract["classification_usage"]["fused"] == ("consumed_via_semantic_residual")
    assert contract["classification_usage"]["peptide"] == (
        "not_directly_consumed_by_prediction_head"
    )
    assert contract["classification_usage"]["partner"] == (
        "not_directly_consumed_by_prediction_head"
    )
    assert contract["residual"]["gate_kind"] == "learned"
    assert contract["residual"]["configured_weight"] == 1.0
    assert contract["residual"]["effective_gate"] == 0.125


def test_pair_semantic_loss_aligns_fused_subset_rows():
    from scripts.train_pair_pipeline import add_partner_semantic_loss

    rows = [
        ("r0", "P1", "C1", 1),
        ("r1", "P1", "C2", 1),
        ("r2", "P2", "C1", 1),
        ("r3", "P3", "C3", 1),
        ("r4", "P2", "C2", 0),
        ("r5", "P4", "C4", 0),
        ("r6", "P5", "C5", 0),
        ("r7", "P6", "C6", 0),
        ("r8", "P7", "C7", 0),
    ]
    contexts = ["C1", "C2", "C1", "C3", "C2", "C4", "C5", "C6", "C7"]
    source = InteractionProfileRelationSource(rows, contexts)
    fused_table = build_partner_table(source, seed=7)
    peptide_table = build_partner_table(source.peptide_entity_source(), seed=7)
    context_table = build_partner_table(source.context_entity_source(), seed=7)
    profile = SemanticTrainingProfile(
        relation_type=source.relation_type,
        seed=7,
        fused_table=fused_table,
        peptide_table=peptide_table,
        context_table=context_table,
        row_peptide_ids=source.row_peptide_ids,
        row_context_ids=source.row_context_ids,
        peptide_example_rows=tuple(
            source.rows_by_peptide[index][0] for index in range(len(source.peptide_values))
        ),
        context_example_rows=tuple(
            source.rows_by_context[index][0] for index in range(len(source.context_values))
        ),
        source_summary=source.summary(),
        config={
            "semantic_pair_mode": "role_fused",
            "semantic_projection_dim": 8,
            "semantic_contrastive_temperature": 0.1,
            "semantic_positive_partners": 1,
            "semantic_zero_partners": 8,
            "semantic_similarity_weight": 0.1,
            "semantic_zero_weight": 0.25,
        },
    )
    model = AlignedGatedClassifier(
        peptide_dim=8,
        partner_dim=16,
        align_dim=8,
        hidden_dim=16,
    )
    prepare_semantic_projectors(
        model,
        SimpleNamespace(
            enable_semantic_similarity=True,
            semantic_pair_mode="role_fused",
            semantic_projection_dim=8,
        ),
        sample_inputs=(torch.randn(2, 8), torch.randn(2, 16)),
    )
    args = SimpleNamespace(
        enable_semantic_similarity=True,
        semantic_pair_mode="role_fused",
        semantic_contrastive_temperature=0.1,
        semantic_similarity_weight=0.1,
    )
    peptide = torch.randn(len(rows), 8)
    protein = torch.randn(len(rows), 16)
    targets = torch.tensor([[row[3]] for row in rows], dtype=torch.float32)
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(peptide, protein),
        targets,
    )
    loss, semantic = add_partner_semantic_loss(
        args,
        model,
        classification_loss,
        profile,
        torch.arange(len(rows)),
        (peptide, protein),
        "cpu",
    )
    assert torch.isfinite(loss)
    assert semantic is not None


def _reference_interaction_dense_hard_table(
    source,
    embeddings,
    *,
    positive_cap,
    negative_cap,
):
    array = np.asarray(embeddings, dtype=np.float32)
    normalized = array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-8)
    relations = []
    for anchor_index in range(source.num_rows):
        positives = []
        for candidate in range(source.num_rows):
            if candidate == anchor_index:
                continue
            similarity, valid = source.similarity(anchor_index, candidate)
            if valid and similarity > 0.0:
                positives.append((candidate, float(similarity)))
        positives.sort(key=lambda item: (-item[1], item[0]))
        positives = positives[:positive_cap]
        if not positives:
            continue
        for partner_index, similarity in positives:
            relations.append(SemanticRelation(anchor_index, partner_index, similarity, "positive"))
        if negative_cap <= 0:
            continue
        negatives = []
        for candidate in range(source.num_rows):
            if candidate == anchor_index:
                continue
            similarity, valid = source.similarity(anchor_index, candidate)
            if valid and similarity == 0.0:
                negatives.append(candidate)
        negatives.sort(
            key=lambda index: (
                -float(normalized[anchor_index] @ normalized[index]),
                index,
            )
        )
        for partner_index in negatives[:negative_cap]:
            relations.append(SemanticRelation(anchor_index, partner_index, 0.0, "zero"))
    return SemanticPartnerTable(
        relation_type=source.relation_type,
        num_rows=source.num_rows,
        seed=42,
        relations=tuple(relations),
        source_summary=source.summary(),
    )


def test_interaction_dense_hard_table_matches_reference():
    source = _small_interaction_source()
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(source.num_rows, 6)).astype(np.float32)
    for positive_cap, negative_cap in ((2, 1), (16, 8), (1, 0)):
        fast = build_interaction_dense_hard_partner_table(
            source,
            seed=42,
            embeddings=embeddings,
            positive_cap=positive_cap,
            negative_cap=negative_cap,
        )
        reference = _reference_interaction_dense_hard_table(
            source,
            embeddings,
            positive_cap=positive_cap,
            negative_cap=negative_cap,
        )
        assert fast.relations == reference.relations


def test_interaction_dense_hard_placebo_preserves_shape_but_shuffles_partners():
    source = _small_interaction_source()
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(source.num_rows, 6)).astype(np.float32)
    table = build_interaction_dense_hard_partner_table(
        source,
        seed=42,
        embeddings=embeddings,
        positive_cap=4,
        negative_cap=2,
    )
    placebo = build_interaction_dense_hard_partner_table(
        source,
        seed=42,
        embeddings=embeddings,
        positive_cap=4,
        negative_cap=2,
        placebo=True,
    )
    assert placebo.relation_type == table.relation_type
    assert placebo.num_rows == table.num_rows
    true_by_anchor = table.relations_by_anchor
    placebo_by_anchor = placebo.relations_by_anchor
    assert sorted(placebo_by_anchor) == sorted(true_by_anchor)
    changed = False
    for anchor in true_by_anchor:
        true_items = sorted((r.kind, r.similarity, r.partner_index) for r in true_by_anchor[anchor])
        placebo_items = sorted(
            (r.kind, r.similarity, r.partner_index) for r in placebo_by_anchor[anchor]
        )
        assert [item[:2] for item in placebo_items] == [item[:2] for item in true_items]
        if [item[2] for item in placebo_items] != [item[2] for item in true_items]:
            changed = True
    assert changed


def test_pair_learned_gate_starts_at_base_and_trains():
    from scripts.train_pair_pipeline import add_partner_semantic_loss

    base = AlignedGatedClassifier(input_dim=8, hidden_dim=16)
    prepare_semantic_projectors(
        base,
        SimpleNamespace(
            enable_semantic_similarity=True,
            semantic_pair_mode="fused",
            semantic_projection_dim=8,
        ),
        sample_inputs=(torch.randn(3, 8), torch.randn(3, 8)),
    )
    adapter = PairSemanticResidualAdapter(base, 1.0, learned_gate=True)
    adapter.eval()
    peptide = torch.randn(4, 8)
    partner = torch.randn(4, 8)
    assert float(adapter.gate) == 0.0
    logits, fused = adapter.forward_with_embedding(peptide, partner)
    eff_logits, effective = adapter.forward_with_effective_embedding(peptide, partner)
    assert torch.allclose(effective, fused)
    assert torch.allclose(eff_logits, base.head(fused))
    assert torch.allclose(logits, eff_logits)

    adapter.train()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-2)
    targets = torch.rand(4, 1)
    for _ in range(5):
        optimizer.zero_grad()
        logits, effective = adapter.forward_with_effective_embedding(peptide, partner)
        loss = torch.nn.functional.mse_loss(logits, targets) + effective.float().pow(2).mean()
        loss.backward()
        optimizer.step()
    assert abs(float(adapter.gate)) > 1e-6


def test_pair_shared_loss_uses_classification_embedding():
    from scripts.train_pair_pipeline import add_partner_semantic_loss
    from pepplatform.semantic import semantic_contrastive_loss

    rows = [
        ("r0", "P1", "C1", 1),
        ("r1", "P1", "C2", 1),
        ("r2", "P2", "C1", 1),
        ("r3", "P3", "C3", 1),
        ("r4", "P2", "C2", 0),
        ("r5", "P4", "C4", 0),
    ]
    contexts = ["C1", "C2", "C1", "C3", "C2", "C4"]
    source = InteractionProfileRelationSource(rows, contexts)
    fused_table = build_partner_table(source, seed=7)
    profile = SemanticTrainingProfile(
        relation_type=source.relation_type,
        seed=7,
        fused_table=fused_table,
        peptide_table=None,
        context_table=None,
        row_peptide_ids=source.row_peptide_ids,
        row_context_ids=source.row_context_ids,
        peptide_example_rows=(),
        context_example_rows=(),
        source_summary=source.summary(),
        config={"semantic_pair_mode": "fused"},
    )
    model = AlignedGatedClassifier(input_dim=8, hidden_dim=16)
    prepare_semantic_projectors(
        model,
        SimpleNamespace(
            enable_semantic_similarity=True,
            semantic_pair_mode="fused",
            semantic_projection_dim=8,
        ),
        sample_inputs=(torch.randn(2, 8), torch.randn(2, 8)),
    )
    args = SimpleNamespace(
        enable_semantic_similarity=True,
        semantic_pair_mode="fused",
        semantic_loss_space="shared",
        semantic_partner_scheme="fixed",
        semantic_contrastive_temperature=0.1,
        semantic_similarity_weight=0.1,
    )
    peptide = torch.randn(len(rows), 8)
    protein = torch.randn(len(rows), 8)
    targets = torch.tensor([[row[3]] for row in rows], dtype=torch.float32)
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(peptide, protein),
        targets,
    )
    loss, semantic = add_partner_semantic_loss(
        args,
        model,
        classification_loss,
        profile,
        torch.arange(len(rows)),
        (peptide, protein),
        "cpu",
    )
    with auxiliary_eval_forward(model):
        with torch.no_grad():
            _, fused_embeddings = model.forward_with_embedding(peptide, protein)
    unique_rows = tuple(sorted(set(fused_table.unique_row_indices(fused_table.relations))))
    positions = list(range(len(unique_rows)))
    expected = semantic_contrastive_loss(
        fused_embeddings[positions],
        unique_rows,
        fused_table.relations,
        temperature=0.1,
        min_negatives=1,
    )
    assert semantic is not None
    assert torch.allclose(semantic, expected.loss, atol=1e-6)
    assert torch.isfinite(loss)


def test_pair_dense_hard_loss_weights_positives():
    from scripts.train_pair_pipeline import add_partner_semantic_loss
    from pepplatform.semantic import semantic_contrastive_loss

    rows = [
        ("r0", "P1", "C1", 1),
        ("r1", "P1", "C2", 1),
        ("r2", "P2", "C1", 1),
        ("r3", "P3", "C3", 1),
        ("r4", "P2", "C2", 0),
        ("r5", "P4", "C4", 0),
    ]
    contexts = ["C1", "C2", "C1", "C3", "C2", "C4"]
    source = InteractionProfileRelationSource(rows, contexts)
    rng = np.random.default_rng(3)
    embeddings = rng.normal(size=(len(rows), 6)).astype(np.float32)
    fused_table = build_interaction_dense_hard_partner_table(
        source,
        seed=7,
        embeddings=embeddings,
        positive_cap=4,
        negative_cap=2,
    )
    profile = SemanticTrainingProfile(
        relation_type=source.relation_type,
        seed=7,
        fused_table=fused_table,
        peptide_table=None,
        context_table=None,
        row_peptide_ids=source.row_peptide_ids,
        row_context_ids=source.row_context_ids,
        peptide_example_rows=(),
        context_example_rows=(),
        source_summary=source.summary(),
        config={"semantic_pair_mode": "fused"},
    )
    model = AlignedGatedClassifier(input_dim=8, hidden_dim=16)
    prepare_semantic_projectors(
        model,
        SimpleNamespace(
            enable_semantic_similarity=True,
            semantic_pair_mode="fused",
            semantic_projection_dim=8,
        ),
        sample_inputs=(torch.randn(2, 8), torch.randn(2, 8)),
    )
    args = SimpleNamespace(
        enable_semantic_similarity=True,
        semantic_pair_mode="fused",
        semantic_loss_space="shared",
        semantic_partner_scheme="dense_hard",
        semantic_contrastive_temperature=0.1,
        semantic_similarity_weight=0.1,
    )
    peptide = torch.randn(len(rows), 8)
    protein = torch.randn(len(rows), 8)
    targets = torch.tensor([[row[3]] for row in rows], dtype=torch.float32)
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(peptide, protein),
        targets,
    )
    loss, semantic = add_partner_semantic_loss(
        args,
        model,
        classification_loss,
        profile,
        torch.arange(len(rows)),
        (peptide, protein),
        "cpu",
    )
    with auxiliary_eval_forward(model):
        with torch.no_grad():
            _, fused_embeddings = model.forward_with_embedding(peptide, protein)
    unique_rows = tuple(sorted(set(fused_table.unique_row_indices(fused_table.relations))))
    positions = list(range(len(unique_rows)))
    expected = semantic_contrastive_loss(
        fused_embeddings[positions],
        unique_rows,
        fused_table.relations,
        temperature=0.1,
        min_negatives=1,
        weight_positives=True,
    )
    assert semantic is not None
    assert torch.allclose(semantic, expected.loss, atol=1e-6)
    assert torch.isfinite(loss)
