"""Loss dispatch and numerical compatibility without downloading encoders."""

from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
import torch
from torch import nn

from scripts import train_sequence_pipeline as sequence


def pipeline(**overrides):
    pipe = object.__new__(sequence.Pipeline)
    pipe.device = torch.device("cpu")
    pipe.args = Namespace(**overrides)
    return pipe


@pytest.mark.parametrize("requested", [None, "auto", "balanced_bce"])
@pytest.mark.parametrize("semantic_enabled", [False, True])
@torch.random.fork_rng(devices=[])
def test_pooled_default_matches_historical_loss_gradients_and_adamw_step(
    tmp_path, requested, semantic_enabled
):
    torch.manual_seed(42)
    # Includes an absent class, an all-positive class, and a capped rare class.
    labels = torch.zeros(33, 3)
    labels[:, 1] = 1
    labels[0, 2] = 1
    inputs = torch.randn(33, 4)
    pipe = pipeline(
        **({"multilabel_loss": requested} if requested is not None else {}),
        enable_semantic_similarity=semantic_enabled,
        task_type="pep_function",
        model="mlp",
        semantic_loss_space="shared",
        seed=42,
        artifact_dir=str(tmp_path),
    )
    # Use the real semantic objective and partner table for both old/new steps.
    pipe._configure_intrinsic_semantics(inputs, labels)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(4, 4)
            self.head = nn.Linear(4, 3)

        def forward_with_embedding(self, x):
            embedding = self.backbone(x).tanh()
            return self.head(embedding), embedding

        def forward(self, x):
            return self.forward_with_embedding(x)[0]

    old_model = Head()
    new_model = deepcopy(old_model)
    positive = labels.sum(dim=0)
    negative = labels.shape[0] - positive
    historical_weights = torch.clamp(negative / (positive + 1e-6), min=1.0, max=20.0)
    old_criterion = nn.BCEWithLogitsLoss(pos_weight=historical_weights)
    rng_before = torch.random.get_rng_state().clone()
    new_criterion = pipe._multilabel_loss(labels, auto_default="balanced_bce")
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert torch.equal(new_criterion.pos_weight, historical_weights)
    losses, gradients = [], []
    for model, criterion in ((old_model, old_criterion), (new_model, new_criterion)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        _, loss, _ = pipe.training_loss(
            model, inputs, labels, criterion, row_ids=torch.arange(len(labels))
        )
        loss.backward()
        losses.append(loss.detach())
        gradients.append([p.grad.clone() for p in model.parameters()])
        optimizer.step()
    assert torch.equal(*losses)
    assert all(torch.equal(a, b) for a, b in zip(*gradients))
    assert all(torch.equal(a, b) for a, b in zip(old_model.parameters(), new_model.parameters()))
    assert pipe.args._resolved_multilabel_loss["resolved"] == "balanced_bce"
    assert pipe.args._resolved_multilabel_loss["fit_scope"] == "train_only"


@pytest.mark.parametrize("requested", [None, "auto"])
def test_generic_sequence_default_remains_unweighted_bce(requested):
    pipe = pipeline(**({"multilabel_loss": requested} if requested else {}))
    criterion = pipe._multilabel_loss(torch.tensor([[1.0], [0.0], [0.0]]))
    assert criterion.pos_weight is None
    assert pipe.args._resolved_multilabel_loss["resolved"] == "bce"


class CapturedTrainingLoss(Exception):
    pass


@pytest.mark.parametrize("encoder", ["esm2", "protbert", "prott5", "esmc", "esm3"])
@pytest.mark.parametrize("loss", ["auto", "bce", "balanced_bce", "asymmetric_focal"])
def test_each_pooled_entrypoint_uses_the_selected_criterion(tmp_path, monkeypatch, encoder, loss):
    pipe = object.__new__(sequence.Pipeline)
    pipe.device = torch.device("cpu")
    pipe.args = sequence.build_argument_parser().parse_args([
        "--tokenizer", encoder, "--encoder", encoder, "--model", "mlp",
        "--data_dir", str(tmp_path), "--artifact_dir", str(tmp_path / "artifact"),
        "--epochs", "1", "--dl_scaler", "none", "--no-evaluate_test",
        "--multilabel_loss", loss, "--multilabel_pos_weight_cap", "2",
        "--asl_gamma_pos", "0.5", "--asl_gamma_neg", "3", "--asl_clip", "0.1",
    ])
    features = torch.tensor([[0.2, 0.4, -0.3, 0.1], [-0.2, 0.1, 0.8, 0.3], [0.3, -0.5, 0.1, 0.2]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    fake_encoder = SimpleNamespace(
        output_dim=4, model=SimpleNamespace(config=SimpleNamespace(d_model=4, hidden_size=4)),
        model_identity={}, runtime_environment={},
    )
    for name in ("EsmTokenizer", "ProtBertTokenizer", "ProT5Tokenizer", "EsmcTokenizer", "Esm3Tokenizer"):
        monkeypatch.setattr(sequence, name, lambda **kwargs: SimpleNamespace(tokenizer_contract={"residue_tokenization": "spaced_residues"}))
    for name in ("EsmEncoder", "ProtBertEncoder", "ProT5Encoder", "EsmcEncoder", "Esm3Encoder"):
        monkeypatch.setattr(sequence, name, lambda **kwargs: fake_encoder)
    for name in ("format_esm2_provenance", "format_protbert_provenance", "format_prott5_provenance", "format_esmc_provenance", "format_esm3_provenance"):
        monkeypatch.setattr(sequence, name, lambda *args: "mock encoder")
    monkeypatch.setattr(pipe, "load_sequences", lambda path: ["AA", "AC", "CC"])
    monkeypatch.setattr(pipe, "load_labels", lambda path: labels.clone())
    monkeypatch.setattr(pipe, "_load_label_names", lambda *args: ["A", "B"])
    monkeypatch.setattr(pipe, "_pretrained_embedding_artifact_info", lambda *args: {})
    monkeypatch.setattr(pipe, "_configure_intrinsic_semantics", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipe, "_prepare_semantic_model", lambda model, **kwargs: model)
    monkeypatch.setattr(pipe, "_semantic_train_loader", lambda data: [(features, labels, torch.arange(3))])
    monkeypatch.setattr(pipe, "print_semantic_similarity_status", lambda *args: None)
    for tag in ("esm2", "protbert", "prott5", "esmc", "esm3"):
        monkeypatch.setattr(pipe, f"resolve_{tag}_model_source", lambda: "mock-pretrained")
    for name in ("get_or_extract_esm_embeddings", "get_or_extract_protbert_embeddings", "get_or_extract_prott5_embeddings", "get_or_extract_esmc_embeddings", "get_or_extract_esm3_embeddings"):
        monkeypatch.setattr(pipe, name, lambda *args: features.clone())

    def capture(model, inputs, targets, criterion, **kwargs):
        resolved = "balanced_bce" if loss == "auto" else loss
        assert pipe.args._resolved_multilabel_loss["requested"] == loss
        assert pipe.args._resolved_multilabel_loss["resolved"] == resolved
        if resolved == "bce":
            assert criterion.pos_weight is None
        elif resolved == "balanced_bce":
            assert criterion.pos_weight.tolist() == pytest.approx([1.999998, 1.999998])
        else:
            assert isinstance(criterion, sequence.AsymmetricFocalLoss)
            assert (criterion.gamma_pos, criterion.gamma_neg, criterion.clip) == (0.5, 3.0, 0.1)
        value = criterion(model(inputs), targets)
        value.backward()
        assert torch.isfinite(value)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        raise CapturedTrainingLoss

    monkeypatch.setattr(pipe, "training_loss", capture)
    method = {"esm2": "run_esm", "protbert": "run_protbert", "prott5": "run_prott5", "esmc": "run_esmc", "esm3": "run_esm3"}[encoder]
    with pytest.raises(CapturedTrainingLoss):
        getattr(pipe, method)()


@pytest.mark.parametrize("loss", ["auto", "bce", "balanced_bce", "asymmetric_focal"])
def test_pooled_artifact_saves_resolved_loss_separately_from_semantics(tmp_path, monkeypatch, loss):
    pipe = object.__new__(sequence.Pipeline)
    pipe.device = torch.device("cpu")
    pipe.args = sequence.build_argument_parser().parse_args([
        "--tokenizer", "esm2", "--encoder", "esm2", "--model", "mlp",
        "--artifact_dir", str(tmp_path), "--multilabel_loss", loss,
    ])
    semantic = {"schema_version": 2, "relation_type": "intrinsic_label_jaccard", "loss_type": "legacy"}
    monkeypatch.setattr(pipe, "_semantic_artifact_metadata", lambda model: dict(semantic))
    pipe._multilabel_loss(torch.tensor([[1.0], [0.0], [0.0]]), auto_default="balanced_bce")
    pipe._save_pretrained_embedding_nn_artifact(nn.Linear(4, 1), {"pretrained": {"tokenizer": "esm2"}})
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["multilabel_loss"]["requested"] == loss
    assert metadata["multilabel_loss"]["resolved"] == ("balanced_bce" if loss == "auto" else loss)
    assert metadata["semantic"] == semantic
    assert (tmp_path / "model.pt").is_file()
