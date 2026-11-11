"""Input transcripts, rollback, entrypoint compatibility and scientific isolation."""

import json
import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pepplatform.cli_interaction import TASKS, Interaction, WizardCancelled, field_title
from pepplatform.cli_training import configure_training, namespace_argv, trainer_for
from pepplatform_cli.__main__ import build_parser, ensemble_config, main
from pepplatform_cli.storage import inventory
from pepplatform_cli.training import training_preflight


def dataset(root, task):
    for split, sequences in {
        "train": ["AAAA", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG"],
        "val": ["HHHH", "IIII", "KKKK", "LLLL"],
    }.items():
        directory = root / split
        directory.mkdir(parents=True)
        if task == "pep_function":
            (directory / "seqs.fasta").write_text(
                "".join(f">{i}\n{x}\n" for i, x in enumerate(sequences))
            )
            (directory / "labels.csv").write_text(
                "A,B\n" + "".join(f"{i % 2},{1 - i % 2}\n" for i in range(len(sequences)))
            )
        else:
            (directory / "pairs.csv").write_text(
                "sample_id,peptide,protein,label\n"
                + "".join(f"{i},{sequence},ACDE,{i % 2}\n" for i, sequence in enumerate(sequences))
            )
    return root


def drive(monkeypatch, respond):
    """Observe field titles while exercising the real input parser and journal."""
    original = Interaction._ask
    state = {"title": None, "fields": []}

    def observe(self, title, *args, **kwargs):
        state["title"] = field_title(title)
        state["fields"].append(state["title"])
        return original(self, title, *args, **kwargs)

    monkeypatch.setattr(Interaction, "_ask", observe)
    monkeypatch.setattr("builtins.input", lambda prompt: respond(state["title"], prompt))
    return state


def configure(root, task, language, **kwargs):
    return configure_training(
        [
            "--data_dir",
            str(root),
            "--tokenizer",
            "kmer",
            "--encoder",
            "bow",
            "--model",
            "mlp",
            *kwargs.pop("extra", []),
        ],
        task=task,
        ui=Interaction(language),
        wrapped=True,
        workspace=root.parent / "workspace",
        validate=training_preflight,
        **kwargs,
    )


def test_back_rebuilds_from_initial_state_and_discards_old_branch(monkeypatch):
    entries = iter(["alpha", "7", "b", "b", "beta", "3", ""])
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or next(entries))
    ui = Interaction("en")
    initial = SimpleNamespace(seed=17)

    def collect():
        args = deepcopy(initial)
        args.route = ui.choice("Route", ["alpha", "beta"])
        if args.route == "alpha":
            args.alpha = ui.integer("Alpha", 1)
        else:
            args.beta = ui.integer("Beta", 2)
        args.tail = ui.number("Tail", 0.5)
        return args

    result = ui.collect(collect)
    assert vars(result) == {"seed": 17, "route": "beta", "beta": 3, "tail": 0.5}
    assert vars(initial) == {"seed": 17}
    assert any("[7]" in prompt for prompt in prompts)


def test_back_retains_text_for_transformed_json_and_member_inputs(monkeypatch):
    entries = iter(['{"seed":17}', "b", "", "done"])
    monkeypatch.setattr("builtins.input", lambda *_: next(entries))
    ui = Interaction("en")
    result = ui.collect(lambda: (ui.text("Configuration", validate=json.loads), ui.text("Next")))
    assert result == ({"seed": 17}, "done")


def test_changing_an_earlier_field_keeps_other_inputs_as_defaults(monkeypatch):
    entries = iter(["7", "my data", "b", "b", "9", "", "done"])
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or next(entries))
    ui = Interaction("en")
    result = ui.collect(
        lambda: (ui.integer("epochs", 3), ui.text("Data directory", "default"), ui.text("Next"))
    )
    assert result == (9, "my data", "done")
    assert any("[my data]" in prompt for prompt in prompts)


def test_optional_text_can_be_cleared_after_revisiting(monkeypatch):
    entries = iter(["custom.csv", "b", "-", "done"])
    monkeypatch.setattr("builtins.input", lambda *_: next(entries))
    ui = Interaction("en")
    assert ui.collect(lambda: (ui.text("Val CSV", "original.csv"), ui.text("Next"))) == ("", "done")


@pytest.mark.parametrize("kind", ["choice", "integer", "number", "text"])
@pytest.mark.parametrize("control", ["b", "q"])
def test_first_field_back_and_cancel_leave_workflow(monkeypatch, kind, control):
    monkeypatch.setattr("builtins.input", lambda *_: control)
    ui = Interaction("en")
    call = lambda: getattr(ui, kind)("Field", ["x"] if kind == "choice" else 1)
    with pytest.raises(WizardCancelled):
        ui.collect(call)


def test_numeric_validation_stays_on_field_and_localizes_errors(monkeypatch, capsys):
    entries = iter(["nan", "inf", "-1", "0.5", "1.5", "4", "2"])
    monkeypatch.setattr("builtins.input", lambda *_: next(entries))
    ui = Interaction("zh")
    assert ui.collect(
        lambda: (ui.number("dropout", 0.1, 0), ui.integer("k-mer size", 1, 1, 3))
    ) == (0.5, 2)
    output = capsys.readouterr().out
    assert "请输入有限数值" in output and "请输入整数" in output
    assert "Please enter" not in output
    assert len(ui.answers) == 2


@pytest.mark.parametrize("task", TASKS)
def test_locales_produce_identical_training_parameters(monkeypatch, tmp_path, capsys, task):
    root = dataset(tmp_path / "data", task)
    state = drive(monkeypatch, lambda title, _: "Run" if title == "Review configuration" else "")
    outputs, parameters = [], []
    for language in ("zh", "en"):
        args, parser, _ = configure(root, task, language)
        parameters.append((vars(args), namespace_argv(parser, args)))
        outputs.append(capsys.readouterr().out)
    assert parameters[0] == parameters[1]
    assert "lang" not in parameters[0][0]
    assert "semantic_loss_type" not in parameters[0][0]
    assert parameters[0][0]["semantic_similarity_weight"] == 0.1
    assert "Task" not in state["fields"]
    assert "特征提取方式" in outputs[0] and "Feature extraction" not in outputs[0]
    assert "Feature extraction" in outputs[1] and "配置摘要" not in outputs[1]
    assert "--interactive" not in outputs[0].split("非交互复现命令:")[-1]
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("scheme", ["fixed", "fixed_placebo", "dense_hard", "placebo"])
@pytest.mark.parametrize("loss", ["legacy", "jaccard_supcon"])
def test_semantic_flags_survive_interactive_configuration(
    monkeypatch, tmp_path, task, scheme, loss
):
    root = dataset(tmp_path / "data", task)
    drive(monkeypatch, lambda title, _: "Run" if title == "Review configuration" else "")
    flags = [
        "--enable_semantic_similarity",
        "--semantic_similarity_weight",
        "0.23",
        "--semantic_partner_scheme",
        scheme,
        "--semantic_loss_type",
        loss,
        "--seed",
        "17",
    ]
    args, parser, _ = configure(root, task, "en", extra=flags)
    assert args.enable_semantic_similarity is True
    assert args.semantic_similarity_weight == 0.23 and args.seed == 17
    assert args.semantic_partner_scheme == scheme and args.semantic_loss_type == loss
    restored = parser.parse_args(namespace_argv(parser, args))
    assert restored.semantic_loss_type == loss and restored.semantic_partner_scheme == scheme


@pytest.mark.parametrize("task", TASKS)
def test_ui_does_not_advance_python_numpy_or_torch_rng(monkeypatch, tmp_path, task):
    root = dataset(tmp_path / "data", task)
    parser, _, _ = trainer_for(task)
    # Finish lazy scientific imports before observing the UI's RNG behavior.
    training_preflight(parser.parse_args(["--task_type", task, "--data_dir", str(root)]))
    drive(monkeypatch, lambda title, _: "Run" if title == "Review configuration" else "")
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    configure(root, task, "zh")
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and before[1][2:] == after[1][2:]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert torch.equal(before[2], after[2])


def test_navigation_asks_language_and_task_once_and_preserves_language(
    monkeypatch, tmp_path, capsys
):
    entries = iter(["2", "1", "1", "q", "5"])
    monkeypatch.setattr("builtins.input", lambda *_: next(entries))
    assert main(["--workspace", str(tmp_path / "workspace")]) == 0
    output = capsys.readouterr().out
    assert output.count("Language / 语言") == 1
    assert output.count("Step 1: Task") == 1
    assert "Step 2: Feature extraction" in output
    assert "Returned to navigation" in output
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize("outcome,code", [("q", 0), (EOFError, 130), (KeyboardInterrupt, 130)])
def test_cancel_before_confirmation_never_creates_a_run(monkeypatch, tmp_path, outcome, code):
    root = dataset(tmp_path / "data", "pep_function")
    before = inventory(root)

    def respond(title, _):
        if title == "Review configuration":
            assert not (tmp_path / "workspace").exists()
            if isinstance(outcome, type):
                raise outcome()
            return outcome
        return ""

    drive(monkeypatch, respond)
    assert (
        main(
            [
                "train",
                "--lang",
                "en",
                "--interactive",
                "--task-type",
                "pep_function",
                "--workspace",
                str(tmp_path / "workspace"),
                "--",
                "--data_dir",
                str(root),
                "--encoder",
                "bow",
                "--model",
                "rf",
            ]
        )
        == code
    )
    assert not (tmp_path / "workspace").exists()
    assert inventory(root) == before


@pytest.mark.parametrize(
    "source,target,task", [("pair", "sequence", "pep_function"), ("sequence", "pair", "pep_hlaii")]
)
def test_historical_interactive_entrypoint_dispatches_selected_task(
    monkeypatch, tmp_path, source, target, task
):
    from importlib import import_module

    root = dataset(tmp_path / "data", task)
    source_module = import_module(f"scripts.train_{source}_pipeline")
    target_module = import_module(f"scripts.train_{target}_pipeline")
    called = []
    monkeypatch.setattr(target_module, "main", lambda argv: called.append(argv))
    state = drive(
        monkeypatch,
        lambda title, _: (
            task if title == "Task" else ("Run" if title == "Review configuration" else "")
        ),
    )
    source_module.main(
        [
            "--lang",
            "en",
            "--interactive",
            "--data_dir",
            str(root),
            "--tokenizer",
            "kmer",
            "--encoder",
            "bow",
            "--model",
            "rf",
        ]
    )
    assert state["fields"].count("Task") == 1
    assert len(called) == 1 and "--interactive" not in called[0] and "--lang" not in called[0]
    parsed = target_module.build_argument_parser().parse_args(called[0])
    assert parsed.task_type == task and parsed.model == "rf"


@pytest.mark.parametrize(
    "arguments", [["--lang", "zh", "history", "list"], ["history", "list", "--lang", "en"]]
)
def test_noninteractive_language_option_preserves_json_and_never_prompts(
    monkeypatch, tmp_path, capsys, arguments
):
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("Noninteractive command prompted"))
    assert main([*arguments, "--workspace", str(tmp_path / "workspace")]) == 0
    assert json.loads(capsys.readouterr().out) == []
    args = build_parser().parse_args(
        ["--lang", "zh", "ensemble", "preflight", "--task-type", "pep_function"]
    )
    assert "lang" not in ensemble_config(args)


@pytest.mark.parametrize("task", TASKS[1:])
def test_legacy_semantic_weight_is_the_interactive_default(monkeypatch, tmp_path, task):
    root = dataset(tmp_path / "data", task)
    drive(monkeypatch, lambda title, _: "Run" if title == "Review configuration" else "")
    args, _, _ = configure(
        root,
        task,
        "zh",
        extra=[
            "--enable_interaction_semantic_similarity",
            "--interaction_semantic_similarity_weight",
            "0.23",
        ],
    )
    assert args.enable_semantic_similarity and args.semantic_similarity_weight == 0.23
    assert "semantic_loss_type" not in vars(args)


def test_homogeneous_choice_clears_inactive_explicit_partner_settings(monkeypatch):
    from scripts.train_pair_pipeline import PPIInteractiveWizard, build_argument_parser

    args = build_argument_parser().parse_args(
        ["--partner_tokenizer", "prott5", "--partner_max_length", "100"]
    )
    ui = Interaction("en")
    monkeypatch.setattr("builtins.input", lambda *_: "homogeneous (linked)")
    wizard = PPIInteractiveWizard(ui=ui)
    ui.collect(lambda: wizard._prompt_partner_stream(args, 1))
    assert args.partner_tokenizer == "" and args.partner_max_length == 0


@pytest.mark.parametrize("task", TASKS[1:])
@pytest.mark.parametrize("partner", ["esm2", "prott5", "kmer"])
def test_explicit_partner_configuration_is_used_as_the_default(
    monkeypatch, tmp_path, task, partner
):
    root = dataset(tmp_path / "data", task)
    classic = partner == "kmer"
    flags = [
        "--partner_tokenizer",
        partner,
        "--partner_encoder",
        "bow" if classic else partner,
        "--partner_feature_route",
        "classic" if classic else "pretrained",
        "--partner_max_length",
        "123",
    ]
    flags += ["--partner_k", "3"] if classic else ["--partner_model_source", "custom checkpoint"]
    drive(monkeypatch, lambda title, _: "Run" if title == "Review configuration" else "")
    args, _, _ = configure(root, task, "zh", extra=flags)
    assert args.partner_tokenizer == partner and args.partner_max_length == 123
    assert args.partner_encoder == ("bow" if classic else partner)
    assert (args.partner_k == 3) if classic else (args.partner_model_source == "custom checkpoint")


def test_changing_partner_model_does_not_reuse_the_old_model_source(monkeypatch):
    from pepplatform.workflow import PRETRAINED_MODEL_DEFAULTS
    from scripts.train_pair_pipeline import PPIInteractiveWizard, build_argument_parser

    initial = build_argument_parser().parse_args(
        ["--partner_tokenizer", "esm2", "--partner_model_source", "old checkpoint"]
    )
    entries = iter(["", "", "", "", "b", "b", "prott5", "", ""])
    monkeypatch.setattr("builtins.input", lambda *_: next(entries))
    ui = Interaction("en")

    def collect():
        args = deepcopy(initial)
        PPIInteractiveWizard(ui=ui)._prompt_partner_stream(args, 1)
        return args

    result = ui.collect(collect)
    assert result.partner_tokenizer == "prott5"
    assert result.partner_model_source == PRETRAINED_MODEL_DEFAULTS["prott5"]


@pytest.mark.parametrize("confirm", ["Run", "q"])
def test_ensemble_paths_json_validation_and_confirmation(monkeypatch, tmp_path, confirm):
    from pepplatform_cli import __main__, ensemble
    from pepplatform_cli.wizard import ensemble_wizard

    root = dataset(tmp_path / "data", "pep_function")
    for name in ("member one", "member two"):
        (tmp_path / name).mkdir()
    workspace = tmp_path / "workspace"
    called = []
    monkeypatch.setattr(ensemble, "preflight", lambda config: {"valid": True})
    monkeypatch.setattr(
        __main__,
        "create_ensemble",
        lambda config, workspace: called.append(config) or {"status": "succeeded"},
    )
    json_values = iter(
        ["[]", '{"reference_f1": NaN}', '{"reference_f1": 0.4, "reference_sacc": 0.5}']
    )

    def respond(title, _):
        if title == "Dataset directory":
            return str(root)
        if title == "Member artifact paths (quote paths containing spaces)":
            return f'"{tmp_path / "member one"}" "{tmp_path / "member two"}"'
        if title == "Weight selection objective":
            return "dual_f1_sacc"
        if title == "Objective parameters (JSON)":
            return next(json_values)
        if title == "Review configuration":
            assert not workspace.exists() and not called
            return confirm
        return ""

    drive(monkeypatch, respond)
    if confirm == "q":
        with pytest.raises(WizardCancelled):
            ensemble_wizard(workspace, Interaction("zh"))
        assert not workspace.exists() and not called
    else:
        ensemble_wizard(workspace, Interaction("zh"))
        config = json.loads((workspace / "configs/ensemble.json").read_text())
        assert config == called[0] and "lang" not in config
        assert config["objective_params"] == {"reference_f1": 0.4, "reference_sacc": 0.5}


@pytest.mark.parametrize("confirm", ["Run", "q"])
def test_predict_validates_input_before_confirmation(monkeypatch, tmp_path, confirm):
    from pepplatform_cli import __main__
    from pepplatform_cli.wizard import predict_wizard

    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "metadata.json").write_text(json.dumps({"task_type": "pep_hlai"}))
    source = tmp_path / "source"
    source.mkdir()
    inputs = tmp_path / "pairs.json"
    inputs.write_text(json.dumps({"pairs": [{"peptide": "ACDE", "protein": "FGHI"}]}))
    workspace = tmp_path / "workspace"
    called = []
    monkeypatch.setattr(
        __main__, "predict", lambda args: called.append(vars(args)) or {"status": "succeeded"}
    )

    def respond(title, _):
        if title == "Model artifact directory":
            return str(artifact)
        if title == "Input FASTA/CSV/JSON":
            return str(inputs)
        if title == "Source directory (HLA-I mapping; blank=automatic)":
            return str(source)
        if title == "Review configuration":
            assert not called and not workspace.exists()
            return confirm
        return ""

    drive(monkeypatch, respond)
    if confirm == "q":
        with pytest.raises(WizardCancelled):
            predict_wizard(workspace, Interaction("zh"))
        assert not called
    else:
        predict_wizard(workspace, Interaction("zh"))
        assert called[0]["source_root"] == str(source)
        assert "lang" not in called[0]


@pytest.mark.parametrize("confirm", ["Run", "q"])
def test_history_export_only_writes_after_confirmation(monkeypatch, tmp_path, confirm):
    from pepplatform_cli.wizard import history_wizard

    workspace = tmp_path / "workspace"
    run = workspace / "runs/done"
    run.mkdir(parents=True)
    response = run / "response.json"
    response.write_text(json.dumps({"labels": ["interaction"], "predictions": [{"scores": [0.5]}]}))
    (run / "run.json").write_text(
        json.dumps(
            {
                "run_id": "done",
                "kind": "predict",
                "status": "succeeded",
                "directory": str(run),
                "response": str(response),
                "source_artifact": str(tmp_path / "model"),
            }
        )
    )
    output = tmp_path / "export with spaces.csv"

    def respond(title, _):
        if title == "History actions":
            return "Export inference CSV"
        if title == "New CSV path":
            return f'"{output}"'
        if title == "Review configuration":
            assert not output.exists()
            return confirm
        return ""

    drive(monkeypatch, respond)
    if confirm == "q":
        with pytest.raises(WizardCancelled):
            history_wizard(workspace, Interaction("en"))
        assert not output.exists()
    else:
        history_wizard(workspace, Interaction("en"))
        assert "score:interaction" in output.read_text()
