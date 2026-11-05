from __future__ import annotations

import json
from argparse import Namespace
from typing import NamedTuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from pepplatform.semantic import (
    EntityProfileRelationSource,
    INTERACTION_PROFILE_JACCARD,
    INTRINSIC_LABEL_JACCARD,
    InteractionProfileRelationSource,
    IntrinsicLabelRelationSource,
    RecordingRandomSampler,
    SemanticProjection,
    SemanticRelation,
    SemanticTrainingProfile,
    auxiliary_eval_forward,
    build_partner_table,
    infer_semantic_embedding_dims,
    normalize_semantic_aliases,
    prepare_semantic_projectors,
    save_semantic_training_profile,
    semantic_contrastive_loss,
)


def test_intrinsic_jaccard_and_fixed_partner_table_are_deterministic():
    labels = np.asarray(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    source = IntrinsicLabelRelationSource(labels)
    similarity, valid = source.similarity(0, 1)
    assert valid is True
    assert similarity == pytest.approx(0.5)
    assert source.similarity(0, 3) == (0.0, True)
    assert source.similarity(0, 4) == (0.0, False)

    first = build_partner_table(source, seed=42)
    second = build_partner_table(source, seed=42)
    assert first.relation_type == INTRINSIC_LABEL_JACCARD
    assert first.relations == second.relations
    grouped = first.relations_by_anchor
    assert 4 not in grouped
    for anchor, relations in grouped.items():
        positive = [item for item in relations if item.kind == "positive"]
        assert len(positive) == 1
        assert positive[0].anchor_index == anchor
        assert positive[0].partner_index != anchor


def test_interaction_profile_uses_only_supplied_positive_training_edges():
    train_rows = [
        ("tr1", "P1", "C1SEQ", 1),
        ("tr2", "P1", "C2SEQ", 1),
        ("tr3", "P2", "C1SEQ", 1),
        ("tr4", "P3", "C3SEQ", 1),
        ("tr5", "P2", "C2SEQ", 0),
    ]
    contexts = ["C1", "C2", "C1", "C3", "C2"]
    source = InteractionProfileRelationSource(train_rows, contexts)
    assert source.relation_type == INTERACTION_PROFILE_JACCARD
    assert source.summary()["positive_edges"] == 4

    # Peptide profiles: P1={C1,C2}, P2={C1}; context profiles are both {P1,P2}.
    similarity, valid = source.similarity(0, 2)
    assert valid is True
    assert similarity == pytest.approx(0.75)

    # A validation-only edge is not present because the source never receives it.
    assert "VALIDATION_ONLY" not in source.context_values
    table = build_partner_table(source, seed=7)
    assert all(0 <= item.partner_index < len(train_rows) for item in table.relations)
    assert all(item.anchor_index != item.partner_index for item in table.relations)


def test_empty_interaction_profiles_are_unknown_not_zero_relations():
    rows = [
        ("p1", "P1", "C1", 1),
        ("p2", "P2", "C2", 0),
        ("p3", "P3", "C3", 0),
    ]
    source = InteractionProfileRelationSource(rows)
    assert source.similarity(1, 2) == (0.0, False)
    table = build_partner_table(source, seed=42)
    assert not table.relations


def test_conflicting_pair_labels_fail_closed():
    rows = [
        ("one", "P", "C", 1),
        ("two", "P", "C", 0),
    ]
    with pytest.raises(ValueError, match="conflicting labels"):
        InteractionProfileRelationSource(rows)


def test_semantic_contrastive_loss_pulls_positives_and_repels_negatives():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.6, 0.8],
            [-1.0, 0.0],
        ],
        requires_grad=True,
    )
    relations = (
        SemanticRelation(0, 1, 1.0, "positive"),
        SemanticRelation(0, 2, 0.0, "zero"),
        SemanticRelation(3, 1, 1.0, "positive"),
        SemanticRelation(3, 2, 0.0, "zero"),
    )
    output = semantic_contrastive_loss(
        embeddings,
        [0, 1, 2, 3],
        relations,
        temperature=1.0,
    )
    assert output.anchors == 2
    assert output.positive_relations == 2
    assert output.negative_relations == 2
    assert hasattr(output, "negative_relations")
    assert not hasattr(output, "zero_relations")
    assert output.loss.item() > 0.0
    output.loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_semantic_contrastive_loss_uses_in_batch_negatives_when_table_is_sparse():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        requires_grad=True,
    )
    relations = (SemanticRelation(0, 1, 1.0, "positive"),)
    output = semantic_contrastive_loss(
        embeddings,
        [0, 1, 2, 3],
        relations,
        temperature=1.0,
        min_negatives=1,
    )
    assert output.anchors == 1
    assert output.negative_relations >= 1
    output.loss.backward()
    assert torch.isfinite(embeddings.grad).all()


class _BNDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)
        self.bn = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        return self.dropout(self.bn(self.linear(inputs)))


class _RNNBNDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(3, 4, num_layers=2, dropout=0.6, batch_first=True)
        self.bn = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        encoded, _ = self.gru(inputs)
        return self.dropout(self.bn(encoded[:, -1, :]))


def test_auxiliary_eval_forward_keeps_gradients_and_restores_state_and_rng():
    torch.manual_seed(123)
    model = _BNDropoutModel()
    model.train()
    inputs = torch.randn(8, 3)
    _ = model(inputs)
    running_mean = model.bn.running_mean.detach().clone()
    rng_state = torch.get_rng_state().clone()

    with auxiliary_eval_forward(model):
        assert model.training is False
        assert model.bn.training is False
        assert model.dropout.training is False
        first = model(inputs)
        second = model(inputs)
        assert torch.equal(first, second)
        first.sum().backward()

    assert model.training is True
    assert model.bn.training is True
    assert model.dropout.training is True
    assert torch.equal(model.bn.running_mean, running_mean)
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert model.linear.weight.grad is not None


def test_auxiliary_eval_forward_keeps_rnn_trainable_without_stochastic_dropout():
    torch.manual_seed(321)
    model = _RNNBNDropoutModel().train()
    inputs = torch.randn(8, 5, 3)
    original_dropout = model.gru.dropout
    rng_state = torch.get_rng_state().clone()

    with auxiliary_eval_forward(model):
        assert model.training is False
        assert model.gru.training is True
        assert model.gru.dropout == 0.0
        assert model.bn.training is False
        assert model.dropout.training is False
        first = model(inputs)
        second = model(inputs)
        assert torch.equal(first, second)
        first.sum().backward()

    assert model.training is True
    assert model.gru.training is True
    assert model.gru.dropout == original_dropout
    assert model.bn.training is True
    assert model.dropout.training is True
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert model.gru.weight_ih_l0.grad is not None


def test_semantic_artifacts_save_actual_relation_files_and_hashes(tmp_path):
    labels = [[1, 0], [1, 1], [0, 1], [0, 0]]
    source = IntrinsicLabelRelationSource(labels)
    table = build_partner_table(source, seed=5)
    profile = SemanticTrainingProfile(
        relation_type=source.relation_type,
        seed=5,
        fused_table=table,
        peptide_table=None,
        context_table=None,
        row_peptide_ids=(),
        row_context_ids=(),
        peptide_example_rows=(),
        context_example_rows=(),
        source_summary=source.summary(),
        config={
            "semantic_pair_mode": "fused",
            "semantic_projection_dim": 64,
            "semantic_contrastive_temperature": 0.1,
            "semantic_positive_partners": 1,
            "semantic_zero_partners": 8,
        },
    )
    payload = save_semantic_training_profile(
        tmp_path,
        profile=profile,
        source=source,
        split_sample_ids={"train": ["a", "b", "c", "d"], "val": ["v"], "test": ["t"]},
    )
    assert payload["source_split"] == "train"
    for filename in [
        "data_split_indices.npz",
        "semantic_relation_source.npz",
        "semantic_partner_table.npz",
        "semantic_role_mapping.npz",
        "semantic_provenance.json",
    ]:
        assert (tmp_path / filename).is_file()
        assert filename in payload["files"]
    assert "semantic_peptide_partner_table.npz" not in payload["files"]
    assert "semantic_context_partner_table.npz" not in payload["files"]
    provenance = json.loads((tmp_path / "semantic_provenance.json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == 2
    assert provenance["loss_mode"] == "contrastive"
    assert provenance["tables"]["fused"]["positive"] > 0


def test_interaction_profile_builds_entity_role_sources_and_tables():
    train_rows = [
        ("tr1", "P1", "C1", 1),
        ("tr2", "P1", "C2", 1),
        ("tr3", "P2", "C1", 1),
        ("tr4", "P3", "C3", 1),
        ("tr5", "P2", "C2", 0),
    ]
    source = InteractionProfileRelationSource(train_rows)
    peptide_source = source.peptide_entity_source()
    context_source = source.context_entity_source()
    assert isinstance(peptide_source, EntityProfileRelationSource)
    assert peptide_source.num_rows == len(source.peptide_values)
    assert context_source.num_rows == len(source.context_values)

    peptide_table = build_partner_table(peptide_source, seed=7)
    context_table = build_partner_table(context_source, seed=7)
    assert all(
        0 <= item.partner_index < peptide_source.num_rows for item in peptide_table.relations
    )
    assert all(
        0 <= item.partner_index < context_source.num_rows for item in context_table.relations
    )


def test_context_entity_source_accepts_peptide_neighbor_ids_outside_entity_range():
    rows = [(f"r{index}", f"P{index}", f"C{index % 3}", 1) for index in range(10)]
    source = InteractionProfileRelationSource(rows)
    context_source = source.context_entity_source()
    assert context_source.num_rows == 3
    table = build_partner_table(context_source, seed=7)
    assert all(
        0 <= item.anchor_index < context_source.num_rows
        and 0 <= item.partner_index < context_source.num_rows
        for item in table.relations
    )


def test_semantic_projection_is_attached_before_optimizer():
    class _SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(4, 8)
            self.head = nn.Linear(8, 1)

        @property
        def semantic_embedding_dim(self):
            return 8

        def forward_with_embedding(self, inputs):
            hidden = torch.relu(self.input(inputs))
            return self.head(hidden), hidden

    args = Namespace(
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
    model = _SimpleModel()
    prepare_semantic_projectors(model, args)
    assert hasattr(model, "semantic_projector")
    assert isinstance(model.semantic_projector, SemanticProjection)
    assert model.semantic_projector.output_dim == 16
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    assert any(parameter.requires_grad for parameter in model.semantic_projector.parameters())


def test_recording_sampler_preserves_random_sampler_order_and_hashes_epochs():
    data = list(range(9))
    expected_generator = torch.Generator().manual_seed(19)
    actual_generator = torch.Generator().manual_seed(19)
    expected = torch.utils.data.RandomSampler(data, generator=expected_generator)
    audit_log = []
    actual = RecordingRandomSampler(data, generator=actual_generator, audit_log=audit_log)

    assert list(actual) == list(expected)
    assert len(audit_log) == 1
    assert audit_log[0]["epoch"] == 1
    assert audit_log[0]["num_rows"] == len(data)
    assert len(audit_log[0]["sha256"]) == 64

    assert list(actual) == list(expected)
    assert len(audit_log) == 2
    assert audit_log[1]["epoch"] == 2


def test_removed_task_specific_parameter_fails_in_cli_normalization():
    args = Namespace(
        enable_semantic_similarity=True,
        semantic_similarity_weight=0.1,
        semantic_zero_weight=0.25,
        interaction_semantic_mode="",
        hla_context_semantic_margin=0.2,
        hlai_semantic_smoothing_tau=None,
    )

    with pytest.raises(SystemExit, match="hla_context_semantic_margin was removed"):
        normalize_semantic_aliases(args)


def test_legacy_generic_pair_aliases_normalize_to_canonical_fields_only():
    args = Namespace(
        enable_semantic_similarity=None,
        semantic_similarity_weight=None,
        semantic_zero_weight=0.25,
        enable_interaction_semantic_similarity=True,
        interaction_semantic_similarity_weight=0.2,
        interaction_semantic_mode="profile_jaccard",
        hla_context_semantic_margin=None,
        hlai_semantic_smoothing_tau=None,
    )

    normalize_semantic_aliases(args)

    assert args.enable_semantic_similarity is True
    assert args.semantic_similarity_weight == pytest.approx(0.2)
    assert args.semantic_zero_weight == pytest.approx(0.25)
    assert args.semantic_pair_mode == "role_fused"
    assert args.semantic_projection_dim == 64
    assert args.semantic_contrastive_temperature == pytest.approx(0.1)
    assert args.semantic_positive_partners == 1
    assert args.semantic_zero_partners == 8
    assert not hasattr(args, "enable_interaction_semantic_similarity")
    assert not hasattr(args, "interaction_semantic_mode")


def test_sequence_pipeline_canonical_only_does_not_conflict_with_unset_legacy():
    # Regression: scripts/train_sequence_pipeline.py used to default the legacy
    # alias to False, so a canonical-only "--enable_semantic_similarity" command
    # was rejected as conflicting with the unset legacy flag (multiseed crash).
    # With the legacy default now None, an unset legacy flag must be ignored.
    args = Namespace(
        enable_semantic_similarity=True,
        semantic_similarity_weight=0.1,
        semantic_zero_weight=0.25,
        enable_interaction_semantic_similarity=None,
        interaction_semantic_similarity_weight=None,
        interaction_semantic_mode="",
        hla_context_semantic_margin=None,
        hlai_semantic_smoothing_tau=None,
    )

    normalize_semantic_aliases(args)

    assert args.enable_semantic_similarity is True
    assert not hasattr(args, "enable_interaction_semantic_similarity")


def test_sequence_pipeline_parser_defaults_legacy_alias_to_none():
    from scripts.train_sequence_pipeline import build_argument_parser

    args = build_argument_parser().parse_args(["--task_type", "pep_function"])
    assert args.enable_semantic_similarity is False
    assert args.enable_interaction_semantic_similarity is None


class _ResidueCarrier(NamedTuple):
    features: torch.Tensor
    mask: torch.Tensor


class _SingleInputResidueModel(nn.Module):
    """Regression model whose embedding forward accepts one tuple-like input."""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(4, 8)
        self.head = nn.Linear(8, 1)

    def forward_with_embedding(self, inputs):
        features, mask = inputs
        hidden = torch.relu(self.input(features))
        valid = mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(pooled), pooled


def test_infer_semantic_embedding_dims_does_not_unpack_single_input_namedtuple():
    # Regression: ResidueModelInput is a NamedTuple (tuple subclass), but
    # sequence models expose forward_with_embedding(self, inputs) with one
    # positional argument. It must be passed whole, not unpacked.
    model = _SingleInputResidueModel()
    sample = _ResidueCarrier(
        torch.randn(2, 4, 4),
        torch.ones(2, 4, dtype=torch.bool),
    )

    fused_dim, role_dims = infer_semantic_embedding_dims(model, sample)

    assert fused_dim == 8
    assert role_dims is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_semantic_projector_matches_model_device_on_cuda():
    # Regression: projection heads were created on the default CPU device and
    # attached after model.to(device), causing mixed-device auxiliary forwards.
    class _SimpleDeviceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(4, 8)
            self.head = nn.Linear(8, 1)

        @property
        def semantic_embedding_dim(self):
            return 8

        def forward_with_embedding(self, inputs):
            hidden = torch.relu(self.input(inputs))
            return self.head(hidden), hidden

    args = Namespace(
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
    model = _SimpleDeviceModel().to("cuda:0")

    prepare_semantic_projectors(model, args)

    model_device = next(model.parameters()).device
    projector_device = next(model.semantic_projector.parameters()).device
    assert projector_device.type == "cuda"
    assert projector_device == model_device
