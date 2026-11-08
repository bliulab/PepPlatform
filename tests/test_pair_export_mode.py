from argparse import Namespace

import pytest
import torch

from pepplatform.cli_wizard import InteractiveConfigWizard
from pepplatform.models import IndexLSTM, IndexRNN, IndexTransformer
from pepplatform.workflow import TrainingModeContractError
from scripts import train_pair_pipeline as pair_pipeline


def test_interaction_export_wizard_explains_two_stage_flow(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(
        training_mode="",
        classifier="",
        model="cross_attn+mlp",
        pair_strategy="",
        interaction_level="",
        interaction_model="",
        predictor_type="",
        graph_predictor="",
    )
    choices = iter(["export embeddings for downstream classification", "rf"])
    wizard.prompt_choice = lambda *args, **kwargs: next(choices)

    next_step, back = wizard._prompt_training_mode_for_interaction_model(args, 6, "cross_attn")

    assert next_step == 8
    assert back is None
    assert args.training_mode == "export_embeddings"
    assert args.classifier == "rf"
    output = capsys.readouterr().out
    assert "temporary neural head" in output
    assert "final predictor" in output


@pytest.mark.parametrize(
    ("model_class", "implementation"),
    [
        (IndexRNN, "pepplatform_bidirectional_elman_rnn"),
        (IndexLSTM, "pepplatform_bidirectional_pytorch_lstm"),
        (IndexTransformer, "pepplatform_pytorch_transformer_encoder"),
    ],
)
def test_pair_sequence_export_uses_two_stages_and_skips_auxiliary_test_eval(
    monkeypatch,
    capsys,
    model_class,
    implementation,
):
    args = Namespace(classifier="rf", batch_size=4, epochs=1)
    seen = {}
    if model_class is IndexTransformer:
        recurrent = model_class(
            vocab_size=24,
            max_length=8,
            embed_dim=8,
            num_heads=2,
            ff_dim=16,
            num_classes=1,
        )
    else:
        recurrent = model_class(
            vocab_size=24,
            embed_dim=8,
            hidden_dim=6,
            num_classes=1,
        )

    def fake_train_sequence_model(*call_args, **kwargs):
        seen["return_model"] = kwargs["return_model"]
        seen["evaluate_test"] = kwargs["evaluate_test"]
        return 0.5, recurrent

    def fake_extract_sequence_model_embeddings(model, x, batch_size, device, split_name):
        return torch.ones((1, 2))

    def fake_train_ml(*call_args, classifier=None, **kwargs):
        seen["classifier"] = classifier
        return 0.25

    def fake_configure(*call_args, architecture_contract=None, **kwargs):
        seen["architecture_contract"] = architecture_contract

    monkeypatch.setattr(pair_pipeline, "metric_tag", lambda args: "pair/sequence")
    monkeypatch.setattr(pair_pipeline, "train_sequence_model", fake_train_sequence_model)
    monkeypatch.setattr(
        pair_pipeline, "extract_sequence_model_embeddings", fake_extract_sequence_model_embeddings
    )
    monkeypatch.setattr(pair_pipeline, "train_ml", fake_train_ml)
    monkeypatch.setattr(pair_pipeline, "_configure_pair_embedding_artifacts", fake_configure)

    result = pair_pipeline.train_sequence_model_then_ml(
        args,
        "feat",
        "train_x",
        "train_y",
        "val_x",
        "val_y",
        "test_x",
        "test_y",
    )
    output = capsys.readouterr().out

    assert result == 0.25
    assert seen["return_model"] is True
    assert seen["evaluate_test"] is False
    assert seen["classifier"] == "rf"
    assert seen["architecture_contract"]["implementation"] == implementation
    assert seen["architecture_contract"]["contract_version"] == 2
    assert "Stage 1/2: Train modeling backbone" in output
    assert "Stage 1/2 complete" in output
    assert "Stage 2/2: Train downstream classifier on exported embeddings: rf" in output


def test_pair_interaction_export_uses_two_stages_and_skips_auxiliary_test_eval(monkeypatch, capsys):
    args = Namespace(classifier="rf", batch_size=4, epochs=1)
    seen = {}

    def fake_train_cross_attention_model(*call_args, **kwargs):
        seen["return_model"] = kwargs["return_model"]
        seen["evaluate_test"] = kwargs["evaluate_test"]
        return 0.5, object()

    def fake_extract_sequence_model_embeddings(model, x, batch_size, device, split_name):
        return torch.ones((1, 2))

    def fake_train_ml(*call_args, classifier=None, **kwargs):
        seen["classifier"] = classifier
        return 0.25

    monkeypatch.setattr(pair_pipeline, "metric_tag", lambda args: "pair/interaction")
    monkeypatch.setattr(
        pair_pipeline, "train_cross_attention_model", fake_train_cross_attention_model
    )
    monkeypatch.setattr(
        pair_pipeline, "extract_sequence_model_embeddings", fake_extract_sequence_model_embeddings
    )
    monkeypatch.setattr(pair_pipeline, "train_ml", fake_train_ml)

    result = pair_pipeline.train_cross_attention_then_classifier(
        args,
        "feat",
        "train_x",
        "train_y",
        "val_x",
        "val_y",
        "test_x",
        "test_y",
    )
    output = capsys.readouterr().out

    assert result == 0.25
    assert seen == {
        "return_model": True,
        "evaluate_test": False,
        "classifier": "rf",
    }
    assert "Stage 1/2: Train interaction encoder" in output
    assert "Stage 1/2 complete" in output
    assert "Stage 2/2: Train downstream classifier on exported interaction embeddings: rf" in output


@pytest.mark.parametrize(
    "entrypoint",
    [
        pair_pipeline.train_sequence_model_then_ml,
        pair_pipeline.train_cross_attention_then_classifier,
    ],
)
def test_pair_export_entrypoints_reject_end_to_end_classical_mode(entrypoint):
    args = Namespace(training_mode="end_to_end", classifier="rf")
    with pytest.raises(TrainingModeContractError) as exc_info:
        entrypoint(args, None, None, None, None, None, None)
    assert exc_info.value.code == "end_to_end_requires_mlp"


def test_ppi_mlp_wizard_prompts_semantic_auxiliary_once(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(
        task_type="pep_function",
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        sequence_model="",
        training_mode="",
        classifier="",
        pair_strategy="",
        interaction_level="",
        interaction_model="",
        predictor_type="",
        graph_predictor="",
        reduction="none",
        max_length=60,
        batch_size=32,
        lr=1e-3,
        epochs=30,
        dropout=0.0,
        weight_decay=1e-2,
        early_stop_patience=0,
        early_stop_min_delta=0.0,
        enable_semantic_similarity=False,
        semantic_similarity_weight=0.1,
        enable_interaction_semantic_similarity=False,
        interaction_semantic_similarity_weight=0.1,
        dl_scaler="standard",
        ml_scaler="standard",
        report_aupr=True,
        task_root="",
        train_csv="",
        val_csv="",
        test_csv="",
        feature_dir="",
        cache_features=True,
    )
    semantic_prompts = []

    def choose(title, options, default_value, allow_back=False):
        if title == "Task":
            return "pep_protein_interaction"
        if title == "Feature extraction":
            return "classic pipeline"
        if title == "Tokenizer":
            return "kmer"
        if title == "Encoder":
            return "index"
        if title == "Pair strategy":
            return "two_stream"
        if title == "Interaction level":
            return "vector-level interaction"
        if title == "Predictor route":
            return "classifier"
        if title == "Classifier":
            return "mlp"
        if "Choose fusion" in title:
            return "concat_all"
        if "Enable semantic similarity" in title:
            semantic_prompts.append(title)
            return "yes"
        if title == "DL feature scaling":
            return "standard"
        if title == "ML scaler":
            return "standard"
        if "Report AUPR" in title:
            return "yes"
        if title == "Cache features":
            return "yes"
        return default_value

    wizard.prompt_choice = choose
    wizard.prompt_int = lambda title, default_value, min_value=None, allow_back=False: default_value
    wizard.prompt_float = lambda title, default_value, min_value=None, allow_back=False: (
        default_value
    )
    wizard.prompt_text = lambda title, default_value, allow_back=False: default_value

    result = wizard.run(args)

    assert result.enable_semantic_similarity is True
    assert result.enable_interaction_semantic_similarity is None
    assert len(semantic_prompts) == 1


def test_hlaii_interaction_export_wizard_disables_unsupported_semantic_auxiliary(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(
        task_type="pep_hlaii",
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        sequence_model="",
        training_mode="",
        classifier="",
        pair_strategy="",
        interaction_level="",
        interaction_model="",
        predictor_type="",
        graph_predictor="",
        k=2,
        max_length=60,
        batch_size=32,
        lr=1e-3,
        epochs=30,
        dropout=0.0,
        weight_decay=1e-2,
        grad_clip_norm=0.0,
        early_stop_patience=0,
        early_stop_min_delta=0.0,
        enable_semantic_similarity=False,
        semantic_similarity_weight=0.1,
        enable_interaction_semantic_similarity=False,
        interaction_semantic_similarity_weight=0.1,
        hla_context_semantic_margin=0.2,
        reduction="none",
        ml_scaler="standard",
        dl_scaler="none",
        ml_reduction="none",
        rf_estimators=300,
        rf_max_depth=0,
        rf_min_samples_leaf=1,
        enable_regularization=True,
        advanced_wizard=False,
        data_dir="",
    )
    semantic_prompts = []

    def choose(title, options, default_value, allow_back=False):
        if title == "Feature extraction":
            return "classic pipeline"
        if title == "Tokenizer":
            return "kmer"
        if title == "Encoder":
            return "index"
        if title == "Pair strategy":
            return "two_stream"
        if title == "Interaction level":
            return "token-level interaction"
        if title == "Interaction architecture":
            return "cross_attn"
        if title == "Training mode":
            return "export embeddings for downstream classification"
        if title == "Downstream classifier":
            return "rf"
        if "Enable semantic similarity" in title:
            semantic_prompts.append(title)
            return "yes"
        if title == "ML scaler":
            return "none"
        if title == "ML reduction":
            return "none"
        if "Enable regularization" in title:
            return "no"
        return default_value

    wizard.prompt_choice = choose
    wizard.prompt_int = lambda title, default_value, min_value=None, allow_back=False: default_value
    wizard.prompt_float = lambda title, default_value, min_value=None, allow_back=False: (
        default_value
    )
    wizard.prompt_text = lambda title, default_value, allow_back=False: default_value

    result = wizard.run(args)

    assert result.enable_semantic_similarity is False
    assert result.enable_interaction_semantic_similarity is None
    assert semantic_prompts == []
