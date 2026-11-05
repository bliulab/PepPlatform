"""Formula, compatibility and training integration for opt-in semantic losses."""

from argparse import Namespace
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from pepplatform.semantic import (
    INTERACTION_PROFILE_JACCARD,
    IntrinsicLabelRelationSource,
    SemanticPartnerTable,
    SemanticRelation,
    SemanticTrainingProfile,
    normalize_semantic_aliases,
    save_semantic_training_profile,
    semantic_contrastive_loss,
)


def _relations():
    return (
        SemanticRelation(0, 1, 1.0, "positive"),
        SemanticRelation(0, 2, 0.25, "positive"),
        SemanticRelation(0, 3, 0.0, "zero"),
    )


def _embeddings():
    return torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]], requires_grad=True)


def test_jaccard_supcon_matches_weighted_log_formula_and_gradient():
    embeddings = _embeddings()
    output = semantic_contrastive_loss(
        embeddings,
        range(4),
        _relations(),
        temperature=0.7,
        loss_type="jaccard_supcon",
    )
    normalized = F.normalize(embeddings, dim=1)
    probabilities = torch.softmax(normalized[0] @ normalized[1:].T / 0.7, dim=0)
    expected = -(probabilities[0].log() + 0.25 * probabilities[1].log()) / 2
    torch.testing.assert_close(output.loss, expected)
    actual_grad = torch.autograd.grad(output.loss, embeddings, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, embeddings)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert (output.anchors, output.positive_relations, output.negative_relations) == (1, 2, 1)
    # The old scheme's exponent weighting must not be applied a second time.
    with_old_flag = semantic_contrastive_loss(
        embeddings,
        range(4),
        _relations(),
        temperature=0.7,
        loss_type="jaccard_supcon",
        weight_positives=True,
    )
    assert torch.equal(output.loss, with_old_flag.loss)
    old_weighted = semantic_contrastive_loss(
        embeddings,
        range(4),
        _relations(),
        temperature=0.7,
        weight_positives=True,
    )
    assert not torch.isclose(output.loss, old_weighted.loss)


def test_single_positive_retains_absolute_jaccard_weight():
    embeddings = _embeddings()
    relations = (SemanticRelation(0, 1, 0.25, "positive"), _relations()[-1])
    old = semantic_contrastive_loss(embeddings, range(4), relations)
    new = semantic_contrastive_loss(embeddings, range(4), relations, loss_type="jaccard_supcon")
    torch.testing.assert_close(new.loss, 0.25 * old.loss)


def test_weight_one_multiple_positives_reduces_to_supcon_not_pooled_loss():
    embeddings = _embeddings()
    relations = tuple(
        replace(r, similarity=1.0) if r.kind == "positive" else r for r in _relations()
    )
    output = semantic_contrastive_loss(embeddings, range(4), relations, loss_type="jaccard_supcon")
    z = F.normalize(embeddings, dim=1)
    expected = -F.log_softmax(z[0] @ z[1:].T / 0.1, dim=0)[:2].mean()
    torch.testing.assert_close(output.loss, expected)
    assert output.loss > semantic_contrastive_loss(embeddings, range(4), relations).loss


def test_sparse_negative_fallback_and_duplicate_relations():
    embeddings = _embeddings()
    positive = SemanticRelation(0, 1, 0.5, "positive")
    fallback = semantic_contrastive_loss(
        embeddings,
        range(4),
        [positive, positive],
        min_negatives=2,
        loss_type="jaccard_supcon",
    )
    explicit = semantic_contrastive_loss(
        embeddings,
        range(4),
        [positive, SemanticRelation(0, 2, 0.0, "zero"), SemanticRelation(0, 3, 0.0, "zero")],
        loss_type="jaccard_supcon",
    )
    assert torch.equal(fallback.loss, explicit.loss)
    assert fallback.negative_relations == 2


@pytest.mark.parametrize("relations", [(), (SemanticRelation(0, 1, 1.0, "positive"),)])
def test_unusable_anchors_return_differentiable_zero(relations):
    embeddings = _embeddings()[:2].detach().requires_grad_()
    output = semantic_contrastive_loss(embeddings, [0, 1], relations, loss_type="jaccard_supcon")
    assert output.anchors == 0
    assert output.loss.item() == 0
    output.loss.backward()
    assert torch.equal(embeddings.grad, torch.zeros_like(embeddings))


def test_low_temperature_remains_finite():
    embeddings = _embeddings()
    loss = semantic_contrastive_loss(
        embeddings,
        range(4),
        _relations(),
        temperature=1e-5,
        loss_type="jaccard_supcon",
    ).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(embeddings.grad).all()


def test_anchor_mean_excludes_anchors_without_positive_relations():
    embeddings = _embeddings()
    relations = _relations() + (
        SemanticRelation(1, 0, 0.5, "positive"),
        SemanticRelation(1, 3, 0.0, "zero"),
        SemanticRelation(3, 0, 0.0, "zero"),
    )
    output = semantic_contrastive_loss(
        embeddings,
        range(4),
        relations,
        temperature=0.7,
        loss_type="jaccard_supcon",
    )
    z = F.normalize(embeddings, dim=1)
    q0 = F.log_softmax(z[0] @ z[1:].T / 0.7, dim=0)
    q1 = F.log_softmax(z[1] @ z[[0, 3]].T / 0.7, dim=0)
    expected = ((-q0[0] - 0.25 * q0[1]) / 2 - 0.5 * q1[0]) / 2
    torch.testing.assert_close(output.loss, expected)
    assert output.anchors == 2


@pytest.mark.parametrize("weighted", [False, True])
def test_default_and_explicit_legacy_match_loss_gradient_and_rng(weighted):
    embeddings = _embeddings()
    state = torch.get_rng_state().clone()
    default = semantic_contrastive_loss(
        embeddings, range(4), _relations(), weight_positives=weighted
    )
    explicit = semantic_contrastive_loss(
        embeddings,
        range(4),
        _relations(),
        weight_positives=weighted,
        loss_type="legacy",
    )
    assert torch.equal(default.loss, explicit.loss)
    default_grad = torch.autograd.grad(default.loss, embeddings, retain_graph=True)[0]
    explicit_grad = torch.autograd.grad(explicit.loss, embeddings)[0]
    assert torch.equal(default_grad, explicit_grad)
    assert torch.equal(state, torch.get_rng_state())


def test_unknown_loss_rejected_without_materializing_legacy_default():
    args = normalize_semantic_aliases(Namespace())
    assert not hasattr(args, "semantic_loss_type")
    with pytest.raises(SystemExit, match="semantic_loss_type"):
        normalize_semantic_aliases(Namespace(semantic_loss_type="typo"))
    with pytest.raises(ValueError, match="unknown semantic loss"):
        semantic_contrastive_loss(_embeddings(), range(4), _relations(), loss_type="typo")


@pytest.mark.parametrize("pipeline", ["train_sequence_pipeline", "train_pair_pipeline"])
def test_cli_loss_is_opt_in(pipeline):
    from importlib import import_module

    parser = import_module(f"scripts.{pipeline}").build_argument_parser()
    assert not hasattr(parser.parse_args([]), "semantic_loss_type")
    for option in ("legacy", "jaccard_supcon"):
        assert parser.parse_args(["--semantic_loss_type", option]).semantic_loss_type == option
    with pytest.raises(SystemExit):
        parser.parse_args(["--semantic_loss_type", "typo"])


@pytest.mark.parametrize("scheme", ["fixed", "fixed_placebo", "dense_hard", "placebo"])
def test_sequence_profiles_keep_relations_and_version_legacy_artifacts(tmp_path, scheme):
    from scripts.train_sequence_pipeline import Pipeline

    inputs = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
    labels = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0]])
    profiles = []
    for loss_type in (None, "legacy", "jaccard_supcon"):
        pipe = object.__new__(Pipeline)
        pipe.args = normalize_semantic_aliases(
            Namespace(
                enable_semantic_similarity=True,
                task_type="pep_function",
                model="mlp",
                seed=42,
                semantic_partner_scheme=scheme,
                artifact_dir=str(tmp_path / str(loss_type)),
                **({"semantic_loss_type": loss_type} if loss_type is not None else {}),
            )
        )
        pipe.device = torch.device("cpu")
        pipe._configure_intrinsic_semantics(inputs, labels)
        profiles.append(pipe._semantic_profile)
        provenance = pipe._semantic_provenance
        if loss_type == "jaccard_supcon":
            assert provenance["schema_version"] == 3
            assert provenance["loss_contract"]["positive_reduction"] == "count_mean"
        else:
            assert provenance["schema_version"] == 2
            assert "loss_contract" not in provenance
            assert "semantic_loss_type" not in provenance["config"]
    assert profiles[0] == profiles[1]
    assert profiles[0].fused_table == profiles[2].fused_table
    # Contents, not zip timestamps: changing the objective cannot change splits or Jaccard.
    for filename in (
        "data_split_indices.npz",
        "semantic_relation_source.npz",
        "semantic_partner_table.npz",
    ):
        with (
            np.load(tmp_path / "None" / filename) as old,
            np.load(tmp_path / "jaccard_supcon" / filename) as new,
        ):
            assert old.files == new.files
            for key in old.files:
                np.testing.assert_array_equal(old[key], new[key])
    source = IntrinsicLabelRelationSource(labels)
    before = {p.name: p.read_bytes() for p in (tmp_path / "None").iterdir()}
    with pytest.raises(ValueError, match="new artifact directory"):
        save_semantic_training_profile(
            tmp_path / "None",
            profile=profiles[2],
            source=source,
            split_sample_ids={"train": ["changed"]},
        )
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "None").iterdir()}


@torch.random.fork_rng(devices=[])
def test_sequence_auxiliary_loss_uses_new_objective():
    # Fixed CPU initialization avoids dead-ReLU dependence on prior test order.
    torch.random.default_generator.manual_seed(0)
    from scripts.train_sequence_pipeline import Pipeline
    from pepplatform.models import PredictionHeadClassifier

    pipe = object.__new__(Pipeline)
    pipe.args = normalize_semantic_aliases(
        Namespace(
            enable_semantic_similarity=True,
            semantic_loss_type="jaccard_supcon",
            semantic_loss_space="shared",
            task_type="pep_function",
            model="mlp",
            seed=42,
        )
    )
    pipe.device = torch.device("cpu")
    inputs = _embeddings().detach()
    pipe._configure_intrinsic_semantics(inputs, torch.tensor([[1, 1], [1, 0], [0, 1], [0, 1]]))
    pipe._semantic_train_dataset = None
    # A smooth activation avoids a randomly all-inactive ReLU head, which would
    # turn this dispatch/gradient check into an unrelated initialization lottery.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = PredictionHeadClassifier(
            input_dim=2,
            num_classes=2,
            head_hidden_dims=[4],
            head_activation="gelu",
            head_dropout=0.0,
        )
    # Keep this integration focused on auxiliary dispatch, not unrelated encoders.
    pipe.forward_with_embedding = lambda m, x: m.forward_with_embedding(x)
    output = pipe._semantic_auxiliary_loss(model, torch.arange(4))
    output.loss.backward()
    assert output.anchors > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("task", ["pep_protein_interaction", "pep_hlaii", "pep_hlai"])
@pytest.mark.parametrize("pair_mode", ["fused", "role", "role_fused"])
def test_pair_all_roles_use_jaccard_objective(task, pair_mode):
    from scripts.train_pair_pipeline import add_partner_semantic_loss

    class ToyPairModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2, bias=False)
            self.semantic_projector = torch.nn.Identity()
            self.peptide_semantic_projector = torch.nn.Identity()
            self.partner_semantic_projector = torch.nn.Identity()

        def forward_with_embedding(self, peptide, context):
            fused = self.encoder(peptide + context)
            return fused.sum(dim=1), fused

        def forward_with_role_embeddings(self, peptide, context):
            logits, fused = self.forward_with_embedding(peptide, context)
            return logits, fused, self.encoder(peptide), self.encoder(context)

    table = SemanticPartnerTable(INTERACTION_PROFILE_JACCARD, 4, 42, _relations(), {})
    profile = SemanticTrainingProfile(
        relation_type=INTERACTION_PROFILE_JACCARD,
        seed=42,
        fused_table=table,
        peptide_table=table,
        context_table=table,
        row_peptide_ids=tuple(range(4)),
        row_context_ids=tuple(range(4)),
        peptide_example_rows=tuple(range(4)),
        context_example_rows=tuple(range(4)),
        source_summary={},
        config={},
    )
    args = normalize_semantic_aliases(
        Namespace(
            task_type=task,
            enable_semantic_similarity=True,
            semantic_pair_mode=pair_mode,
            semantic_loss_type="jaccard_supcon",
            semantic_contrastive_temperature=0.7,
        )
    )
    model = ToyPairModel()
    peptide, context = _embeddings(), _embeddings().flip(0)
    classification = model.forward_with_embedding(peptide, context)[0].square().mean()
    total, actual = add_partner_semantic_loss(
        args,
        model,
        classification,
        profile,
        torch.tensor([0]),
        (peptide, context),
        "cpu",
    )
    _, fused, pep, ctx = model.forward_with_role_embeddings(peptide, context)
    selected = (
        [fused]
        if pair_mode == "fused"
        else [pep, ctx]
        if pair_mode == "role"
        else [fused, pep, ctx]
    )
    manual = []
    for embeddings in selected:
        z = F.normalize(embeddings, dim=1)
        log_prob = F.log_softmax(z[0] @ z[1:].T / 0.7, dim=0)
        manual.append(-(log_prob[0] + 0.25 * log_prob[1]) / 2)
    torch.testing.assert_close(actual, torch.stack(manual).mean())
    torch.testing.assert_close(total, classification + 0.1 * actual)
    total.backward()
    assert torch.isfinite(model.encoder.weight.grad).all()
    args.enable_semantic_similarity = False
    unchanged, auxiliary = add_partner_semantic_loss(
        args,
        model,
        classification,
        profile,
        torch.tensor([0]),
        (peptide, context),
        "cpu",
    )
    assert unchanged is classification and auxiliary is None
