from argparse import Namespace

import pytest
import torch
from torch.utils.data import DataLoader

from pepplatform.cli_wizard import InteractiveConfigWizard
from pepplatform.encoders import ResidueEncoding
from pepplatform.models import (
    IndexGRU,
    IndexLSTM,
    IndexRNN,
    IndexTransformer,
    PretrainedResiduePredictor,
    ResidueModelInput,
)
from pepplatform.workflow import TrainingModeContractError
from scripts.train_sequence_pipeline import ResidueTensorDataset
from scripts.train_pipeline import Pipeline


def test_residue_tensor_dataset_default_collation_preserves_masked_input():
    dataset = ResidueTensorDataset(
        ResidueEncoding(
            torch.randn(3, 4, 5),
            torch.tensor(
                [
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, False, False, False],
                ]
            ),
        ),
        torch.zeros(3, 2),
    )

    inputs, labels = next(iter(DataLoader(dataset, batch_size=2)))

    assert isinstance(inputs, ResidueModelInput)
    assert inputs.features.shape == (2, 4, 5)
    assert inputs.mask.dtype == torch.bool
    assert labels.shape == (2, 2)
    assert dataset.get_num_classes() == 2


def test_run_nn_accepts_pretrained_residue_dataset_with_zero_f1_checkpoint(
    monkeypatch,
    capsys,
):
    torch.manual_seed(7)
    features = torch.randn(4, 4, 5)
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, False, False, False],
        ]
    )
    labels = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]
    )
    train = ResidueTensorDataset(
        ResidueEncoding(features, mask),
        labels,
        label_names=["class_a", "class_b"],
    )
    val = ResidueTensorDataset(
        ResidueEncoding(features.clone(), mask.clone()),
        labels.clone(),
        label_names=["class_a", "class_b"],
    )
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(
        task_type="pep_function",
        tokenizer="esm2",
        encoder="esm2",
        model="gru",
        sequence_model="gru",
        training_mode="end_to_end",
        classifier="mlp",
        dl_scaler="none",
        batch_size=2,
        seed=7,
        lr=0.001,
        weight_decay=0.0,
        epochs=1,
        grad_clip_norm=0.0,
        enable_semantic_similarity=False,
        semantic_similarity_weight=0.1,
        semantic_zero_weight=0.25,
        early_stop_min_delta=0.0,
        early_stop_patience=0,
        report_aupr=False,
        artifact_dir="",
        _artifact_source="pretrained_residue_embeddings",
    )
    pipeline.device = torch.device("cpu")
    pipeline._semantic_partner_table = None
    pipeline._semantic_train_inputs = None
    pipeline._semantic_provenance = None
    model = PretrainedResiduePredictor(
        "gru",
        input_dim=5,
        adapter_dim=8,
        max_length=4,
        num_classes=2,
        dropout=0.0,
        hidden_dim=4,
        head_hidden_dims=[4],
    )
    monkeypatch.setattr(
        "scripts.train_sequence_pipeline.SequencePipeline._validation_metrics_for_checkpoint",
        lambda *_args, **_kwargs: {
            "acc": 0.0,
            "sacc": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "mcc": 0.0,
        },
    )

    best_val_f1 = pipeline.run_nn(
        None,
        train,
        val,
        model_override=model,
        evaluate_test=False,
    )

    assert best_val_f1 == 0.0
    assert "Best Val F1: 0.0000 (Epoch 1)" in capsys.readouterr().out
    inputs, _ = next(iter(DataLoader(val, batch_size=2)))
    assert model(inputs).shape == (2, 2)


def test_sequence_export_entrypoint_rejects_end_to_end_classical_mode():
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(training_mode="end_to_end", classifier="rf")
    with pytest.raises(TrainingModeContractError) as exc_info:
        pipeline.run_sequence_export(None, None, None)
    assert exc_info.value.code == "end_to_end_requires_mlp"


def test_validation_f1_checkpoint_keeps_first_zero_score_state():
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(early_stop_min_delta=0.25)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    best_f1, best_epoch, best_state, improved = pipeline._update_validation_f1_checkpoint(
        model,
        val_f1=0.0,
        epoch=1,
        best_val_f1=0.0,
        best_epoch=0,
        best_state=None,
    )

    assert improved is True
    assert best_f1 == 0.0
    assert best_epoch == 1
    with torch.no_grad():
        model.weight.fill_(2.0)
    best_f1, best_epoch, retained_state, improved = pipeline._update_validation_f1_checkpoint(
        model,
        val_f1=0.0,
        epoch=2,
        best_val_f1=best_f1,
        best_epoch=best_epoch,
        best_state=best_state,
    )
    assert improved is False
    assert retained_state is best_state

    pipeline._restore_validation_f1_checkpoint(
        model,
        retained_state,
        best_epoch,
    )
    assert model.weight.detach().item() == pytest.approx(1.0)


def test_validation_f1_checkpoint_respects_min_delta_after_baseline():
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(early_stop_min_delta=0.1)
    model = torch.nn.Linear(1, 1, bias=False)

    best_f1, best_epoch, best_state, _ = pipeline._update_validation_f1_checkpoint(
        model,
        val_f1=0.4,
        epoch=1,
        best_val_f1=0.0,
        best_epoch=0,
        best_state=None,
    )
    best_f1, best_epoch, best_state, improved = pipeline._update_validation_f1_checkpoint(
        model,
        val_f1=0.5,
        epoch=2,
        best_val_f1=best_f1,
        best_epoch=best_epoch,
        best_state=best_state,
    )
    assert improved is False
    assert (best_f1, best_epoch) == (0.4, 1)

    best_f1, best_epoch, _best_state, improved = pipeline._update_validation_f1_checkpoint(
        model,
        val_f1=0.5001,
        epoch=3,
        best_val_f1=best_f1,
        best_epoch=best_epoch,
        best_state=best_state,
    )
    assert improved is True
    assert best_f1 == pytest.approx(0.5001)
    assert best_epoch == 3


@pytest.mark.parametrize("val_f1", [float("nan"), float("inf"), float("-inf")])
def test_validation_f1_checkpoint_rejects_non_finite_metric(val_f1):
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(early_stop_min_delta=0.0)

    with pytest.raises(
        RuntimeError,
        match=r"Validation F1 is non-finite at epoch 3",
    ):
        pipeline._update_validation_f1_checkpoint(
            torch.nn.Linear(1, 1),
            val_f1=val_f1,
            epoch=3,
            best_val_f1=0.0,
            best_epoch=0,
            best_state=None,
        )


def test_validation_f1_checkpoint_restore_requires_valid_state():
    with pytest.raises(
        RuntimeError,
        match="without a valid validation-F1 checkpoint",
    ):
        Pipeline._restore_validation_f1_checkpoint(
            torch.nn.Linear(1, 1),
            None,
            0,
        )


def test_sequence_export_wizard_explains_two_stage_flow(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(training_mode="", classifier="", model="gru")
    choices = iter(["export embeddings for downstream classification", "rf"])
    wizard.prompt_choice = lambda *args, **kwargs: next(choices)

    next_step, back = wizard._prompt_training_mode_for_sequence_model(args, 5, "gru")

    assert next_step == 7
    assert back is None
    assert args.training_mode == "export_embeddings"
    assert args.classifier == "rf"
    output = capsys.readouterr().out
    assert "temporary neural head" in output
    assert "final predictor" in output


def test_cli_architecture_options_filter_chain_graphs_for_raw_indices():
    wizard = InteractiveConfigWizard()

    assert wizard._architecture_options_for_encoder("index") == [
        "cnn",
        "rnn",
        "gru",
        "lstm",
        "transformer",
    ]
    assert wizard._architecture_options_for_encoder("onehot") == [
        "cnn",
        "rnn",
        "gru",
        "lstm",
        "transformer",
        "gcn",
        "gat",
    ]


def test_esm2_cli_residue_route_offers_all_architectures_and_adapter():
    wizard = InteractiveConfigWizard()
    args = Namespace(
        tokenizer="esm2",
        encoder="index",
        model="mlp",
        sequence_model="",
        training_mode="",
        classifier="mlp",
    )
    seen = {}

    def choose(title, options, default_value, allow_back=False):
        seen[title] = list(options)
        if title == "Predictor route":
            return "residue modeling"
        if title == "Modeling architecture":
            return "gat"
        return default_value

    wizard.prompt_choice = choose
    wizard.prompt_int = lambda *args, **kwargs: 256
    next_step, back = wizard._prompt_pretrained_single_route(args, 3, allow_back=True)

    assert back is None
    assert next_step == 7
    assert seen["Predictor route"] == [
        "pooled representation + classifier",
        "residue modeling",
    ]
    assert seen["Modeling architecture"] == [
        "cnn [residue sequence architecture]",
        "rnn [residue sequence architecture]",
        "gru [residue sequence architecture]",
        "lstm [residue sequence architecture]",
        "transformer [residue sequence architecture]",
        "gcn [residue chain-graph architecture]",
        "gat [residue chain-graph architecture]",
    ]
    assert args.model == "gat"
    assert args.sequence_model == "gat"
    assert args.training_mode == "end_to_end"
    assert args.classifier == "mlp"
    assert args.residue_adapter_dim == 256


def test_non_esm2_pretrained_cli_route_offers_residue_modeling():
    wizard = InteractiveConfigWizard()
    args = Namespace(
        tokenizer="prott5",
        encoder="index",
        model="mlp",
        sequence_model="",
        training_mode="",
        classifier="mlp",
    )

    def choose(title, options, default_value, allow_back=False):
        if title == "Predictor route":
            return "residue modeling"
        if title == "Modeling architecture":
            return "lstm"
        return default_value

    wizard.prompt_choice = choose
    wizard.prompt_int = lambda *args, **kwargs: 128
    next_step, back = wizard._prompt_pretrained_single_route(args, 3, allow_back=True)

    assert back is None
    assert next_step == 7
    assert args.model == "lstm"
    assert args.sequence_model == "lstm"
    assert args.training_mode == "end_to_end"


@pytest.mark.parametrize(
    ("architecture", "model_class", "implementation"),
    [
        ("rnn", IndexRNN, "pepplatform_bidirectional_elman_rnn"),
        ("gru", IndexGRU, "pepplatform_bidirectional_pytorch_gru"),
        ("lstm", IndexLSTM, "pepplatform_bidirectional_pytorch_lstm"),
        ("transformer", IndexTransformer, "pepplatform_pytorch_transformer_encoder"),
    ],
)
def test_sequence_export_uses_two_stages_and_skips_auxiliary_test_eval(
    capsys,
    architecture,
    model_class,
    implementation,
):
    pipeline = object.__new__(Pipeline)
    pipeline.args = Namespace(
        classifier="rf",
        tokenizer="kmer",
        encoder="index",
        sequence_model=architecture,
        epochs=1,
    )
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

    def fake_run_nn(
        featurizer,
        train_data,
        val_data,
        test_data=None,
        return_model=False,
        evaluate_test=True,
        model_override=None,
    ):
        seen["return_model"] = return_model
        seen["evaluate_test"] = evaluate_test
        return 0.5, recurrent, train_data, val_data, test_data

    def fake_extract_model_embeddings(model, dataset, split_name):
        return torch.ones((1, 2)), torch.ones((1, 1))

    def fake_run_ml_on_embeddings(*args, classifier=None, **kwargs):
        seen["classifier"] = classifier
        seen["artifact_source"] = kwargs.get("artifact_source")
        seen["artifact_info"] = kwargs.get("artifact_info")
        return 0.25

    pipeline.run_nn = fake_run_nn
    pipeline.extract_model_embeddings = fake_extract_model_embeddings
    pipeline.run_ml_on_embeddings = fake_run_ml_on_embeddings

    result = pipeline.run_sequence_export("feat", "train", "val", "test")
    output = capsys.readouterr().out

    assert result == 0.25
    assert seen["return_model"] is True
    assert seen["evaluate_test"] is False
    assert seen["classifier"] == "rf"
    assert seen["artifact_source"] == "sequence_embedding_features"
    artifact_info = seen["artifact_info"]
    assert artifact_info["input_schema"] == "peptide_sequences"
    assert artifact_info["embedding_export"]["encoder_path"] == ""
    assert artifact_info["embedding_export"]["embedding_dim"] == 2
    assert artifact_info["embedding_export"]["downstream"] == "rf"
    contract = artifact_info["embedding_export"]["architecture_contract"]
    assert contract["implementation"] == implementation
    assert contract["contract_version"] == 2
    assert "Stage 1/2: Train modeling backbone" in output
    assert "Stage 1/2 complete" in output
    assert "Stage 2/2: Train downstream classifier on exported embeddings: rf" in output


def test_classic_sequence_export_wizard_disables_unsupported_semantic_auxiliary(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(
        task_type="pep_function",
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        sequence_model="",
        training_mode="",
        classifier="",
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
        if title == "Modeling architecture":
            return "cnn"
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
    wizard.prompt_int = lambda title, default_value, min_value=None, allow_back=False, **kwargs: (
        default_value
    )
    wizard.prompt_float = lambda title, default_value, min_value=None, allow_back=False: (
        default_value
    )
    wizard.prompt_text = lambda title, default_value, allow_back=False: default_value

    result = wizard.run(args)

    assert result.enable_semantic_similarity is False
    assert semantic_prompts == []


def test_classic_pssm_flattened_route_available_without_advanced_wizard(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(
        task_type="pep_function",
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        sequence_model="",
        training_mode="",
        classifier="",
        k=2,
        max_length=60,
        enable_semantic_similarity=False,
        semantic_similarity_weight=0.1,
        enable_interaction_semantic_similarity=False,
        interaction_semantic_similarity_weight=0.1,
        reduction="none",
        ml_scaler="standard",
        ml_reduction="none",
        pssm_index_path="",
        svm_c=1.0,
        svm_class_weight="balanced",
        svm_max_iter=2000,
        svm_tol=1e-3,
        enable_regularization=True,
        advanced_wizard=False,
        data_dir="",
    )
    seen_titles = []
    seen_route_options = []

    def choose(title, options, default_value, allow_back=False):
        seen_titles.append(title)
        if title == "Predictor route":
            seen_route_options.extend(options)
            return "flattened features + classifier"
        if title == "Feature extraction":
            return "classic pipeline"
        if title == "Tokenizer":
            return "kmer"
        if title == "Encoder":
            return "pssm"
        if title == "Classifier":
            return "svm"
        if title == "ML scaler":
            return "standard"
        if title == "ML reduction":
            return "none"
        if "Enable regularization" in title:
            return "no"
        return default_value

    wizard.prompt_choice = choose
    wizard.prompt_int = lambda title, default_value, min_value=None, allow_back=False, **kwargs: (
        default_value
    )
    wizard.prompt_float = lambda title, default_value, min_value=None, allow_back=False: (
        default_value
    )
    wizard.prompt_text = lambda title, default_value, allow_back=False: default_value

    result = wizard.run(args)

    assert "Predictor route" in seen_titles
    assert seen_route_options == [
        "modeling architecture + prediction head",
        "flattened features + classifier",
    ]
    assert "Modeling architecture" not in seen_titles
    assert result.model == "svm"
    assert result.classifier == "svm"
    assert result.sequence_model == ""
    assert result.training_mode == ""


def test_classic_pssm_sequence_route_remains_default(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(
        task_type="pep_function",
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        sequence_model="",
        training_mode="",
        classifier="",
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
        reduction="none",
        ml_scaler="standard",
        dl_scaler="none",
        ml_reduction="none",
        pssm_index_path="",
        enable_regularization=True,
        advanced_wizard=False,
        data_dir="",
    )
    seen_titles = []

    def choose(title, options, default_value, allow_back=False):
        seen_titles.append(title)
        if title == "Feature extraction":
            return "classic pipeline"
        if title == "Tokenizer":
            return "kmer"
        if title == "Encoder":
            return "pssm"
        if title == "Predictor route":
            return "modeling architecture + prediction head"
        if title == "Modeling architecture":
            return "gru"
        if title == "Training mode":
            return "end-to-end prediction"
        if "Enable semantic similarity" in title:
            return "no"
        if "Enable regularization" in title:
            return "no"
        if title == "DL feature scaling":
            return "none"
        return default_value

    wizard.prompt_choice = choose
    wizard.prompt_int = lambda title, default_value, min_value=None, allow_back=False, **kwargs: (
        default_value
    )
    wizard.prompt_float = lambda title, default_value, min_value=None, allow_back=False: (
        default_value
    )
    wizard.prompt_text = lambda title, default_value, allow_back=False: default_value

    result = wizard.run(args)

    assert "Predictor route" in seen_titles
    assert "Modeling architecture" in seen_titles
    assert result.model == "gru"
    assert result.sequence_model == "gru"
    assert result.training_mode == "end_to_end"
    assert result.classifier == "mlp"


def test_pretrained_mlp_wizard_prompts_semantic_auxiliary_once(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(
        task_type="pep_function",
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        sequence_model="",
        training_mode="",
        classifier="",
        reduction="none",
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
        dl_scaler="none",
        enable_regularization=True,
        data_dir="",
    )
    semantic_prompts = []

    def choose(title, options, default_value, allow_back=False):
        if title == "Feature extraction":
            return "pretrained model"
        if title == "Pretrained model":
            return "esm2"
        if title == "Classifier":
            return "mlp"
        if "Enable semantic similarity" in title:
            semantic_prompts.append(title)
            return "yes"
        if title == "DL feature scaling":
            return "none"
        if title == "Enable regularization":
            return "no"
        return default_value

    wizard.prompt_choice = choose
    wizard.prompt_int = lambda title, default_value, min_value=None, allow_back=False, **kwargs: (
        default_value
    )
    wizard.prompt_float = lambda title, default_value, min_value=None, allow_back=False: (
        default_value
    )
    wizard.prompt_text = lambda title, default_value, allow_back=False: default_value

    result = wizard.run(args)

    assert result.enable_semantic_similarity is True
    assert len(semantic_prompts) == 1


def test_pretrained_svm_without_regularization_skips_class_weight_prompt(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(
        task_type="pep_function",
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        sequence_model="",
        training_mode="",
        classifier="",
        reduction="none",
        k=2,
        max_length=60,
        enable_semantic_similarity=False,
        ml_scaler="standard",
        dl_scaler="none",
        ml_reduction="none",
        svm_class_weight="balanced",
        enable_regularization=True,
        data_dir="",
    )
    seen_titles = []

    def choose(title, options, default_value, allow_back=False):
        seen_titles.append(title)
        if title == "Feature extraction":
            return "pretrained model"
        if title == "Pretrained model":
            return "esm2"
        if title == "Classifier":
            return "svm"
        if title == "Enable regularization":
            return "no"
        if title == "ML scaler":
            return "standard"
        if title == "ML reduction":
            return "none"
        return default_value

    wizard.prompt_choice = choose
    wizard.prompt_int = lambda title, default_value, min_value=None, allow_back=False, **kwargs: (
        default_value
    )
    wizard.prompt_float = lambda title, default_value, min_value=None, allow_back=False: (
        default_value
    )
    wizard.prompt_text = lambda title, default_value, allow_back=False: default_value

    result = wizard.run(args)

    assert result.svm_class_weight == "none"
    assert "SVM class weight" not in seen_titles


def test_pretrained_rf_skips_irrelevant_regularization_prompt(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(
        task_type="pep_function",
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        sequence_model="",
        training_mode="",
        classifier="",
        reduction="none",
        k=2,
        max_length=60,
        enable_semantic_similarity=False,
        ml_scaler="standard",
        dl_scaler="none",
        ml_reduction="none",
        rf_estimators=300,
        rf_max_depth=0,
        rf_min_samples_leaf=1,
        enable_regularization=True,
        data_dir="",
    )
    seen_titles = []

    def choose(title, options, default_value, allow_back=False):
        seen_titles.append(title)
        if title == "Feature extraction":
            return "pretrained model"
        if title == "Pretrained model":
            return "esm2"
        if title == "Classifier":
            return "rf"
        if title == "ML scaler":
            return "standard"
        if title == "ML reduction":
            return "none"
        return default_value

    wizard.prompt_choice = choose
    wizard.prompt_int = lambda title, default_value, min_value=None, allow_back=False, **kwargs: (
        default_value
    )
    wizard.prompt_float = lambda title, default_value, min_value=None, allow_back=False: (
        default_value
    )
    wizard.prompt_text = lambda title, default_value, allow_back=False: default_value

    result = wizard.run(args)

    assert result.enable_regularization is False
    assert "Enable regularization" not in seen_titles


def test_pretrained_logreg_regularization_controls_class_weight_prompt(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(
        task_type="pep_function",
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        sequence_model="",
        training_mode="",
        classifier="",
        reduction="none",
        k=2,
        max_length=60,
        enable_semantic_similarity=False,
        ml_scaler="standard",
        dl_scaler="none",
        ml_reduction="none",
        logreg_class_weight="balanced",
        logreg_c=1.0,
        logreg_max_iter=2000,
        enable_regularization=True,
        data_dir="",
    )
    seen_titles = []

    def choose(title, options, default_value, allow_back=False):
        seen_titles.append(title)
        if title == "Feature extraction":
            return "pretrained model"
        if title == "Pretrained model":
            return "esm2"
        if title == "Classifier":
            return "logreg"
        if title == "Enable regularization":
            return "yes"
        if title == "ML scaler":
            return "standard"
        if title == "ML reduction":
            return "none"
        if title == "LogReg class weight":
            return "balanced"
        return default_value

    wizard.prompt_choice = choose
    wizard.prompt_int = lambda title, default_value, min_value=None, allow_back=False, **kwargs: (
        default_value
    )
    wizard.prompt_float = lambda title, default_value, min_value=None, allow_back=False: (
        default_value
    )
    wizard.prompt_text = lambda title, default_value, allow_back=False: default_value

    result = wizard.run(args)

    assert result.logreg_class_weight == "balanced"
    assert "Enable regularization" in seen_titles
    assert "LogReg class weight" in seen_titles


def test_classic_rf_skips_irrelevant_regularization_prompt(capsys):
    wizard = InteractiveConfigWizard()
    args = Namespace(
        task_type="pep_function",
        tokenizer="kmer",
        encoder="index",
        model="cnn",
        sequence_model="",
        training_mode="",
        classifier="",
        reduction="none",
        k=2,
        max_length=60,
        enable_semantic_similarity=False,
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
    seen_titles = []

    def choose(title, options, default_value, allow_back=False):
        seen_titles.append(title)
        if title == "Feature extraction":
            return "classic pipeline"
        if title == "Tokenizer":
            return "kmer"
        if title == "Encoder":
            return "bow"
        if title == "Transformation":
            return "none"
        if title == "Classifier":
            return "rf"
        if title == "ML scaler":
            return "none"
        if title == "ML reduction":
            return "none"
        return default_value

    wizard.prompt_choice = choose
    wizard.prompt_int = lambda title, default_value, min_value=None, allow_back=False, **kwargs: (
        default_value
    )
    wizard.prompt_float = lambda title, default_value, min_value=None, allow_back=False: (
        default_value
    )
    wizard.prompt_text = lambda title, default_value, allow_back=False: default_value

    result = wizard.run(args)

    assert result.enable_regularization is False
    assert not any("Enable regularization" in title for title in seen_titles)
