"""Read-only configuration shared by the unified and historical CLI entrypoints.

Only callers execute training. UI state is kept out of argparse namespaces and
all parameter serialization continues to use the trainer's declared actions.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import shlex
import sys
from contextlib import redirect_stderr
from copy import deepcopy
from pathlib import Path

from .cli_interaction import TASKS, Interaction, path_value
from .task_compat import normalize_task_type

ROOT = Path(__file__).resolve().parents[1]


def add_language_argument(parser):
    parser.add_argument(
        "--lang",
        choices=("zh", "en"),
        default=argparse.SUPPRESS,
        help="Interactive display language; omitted from training metadata",
    )


def without_language(arguments):
    result = []
    skip = False
    for value in arguments:
        if skip:
            skip = False
        elif value == "--lang":
            skip = True
        elif not value.startswith("--lang="):
            result.append(value)
    return result


def explicit_task(arguments):
    for index, value in enumerate(arguments):
        flag, equal, selected = value.partition("=")
        if flag in {"--task_type", "--task-type"}:
            if not equal and index + 1 == len(arguments):
                return None  # Let argparse report the missing value normally.
            return normalize_task_type(selected if equal else arguments[index + 1])
    return None


def trainer_for(task):
    name = "sequence" if normalize_task_type(task) == "pep_function" else "pair"
    module = importlib.import_module(f"scripts.train_{name}_pipeline")
    wizard = module.InteractiveConfigWizard if name == "sequence" else module.PPIInteractiveWizard
    return module.build_argument_parser(), wizard, ROOT / f"scripts/train_{name}_pipeline.py"


def namespace_argv(parser, namespace):
    """Serialize declared scientific actions, preserving historical omissions."""
    result, seen = [], set()
    for action in parser._actions:
        dest = action.dest
        if (
            dest in seen
            or dest in {"help", "interactive", "advanced_wizard", "list_objectives", "lang"}
            or not action.option_strings
            or not hasattr(namespace, dest)
        ):
            continue
        seen.add(dest)
        value = getattr(namespace, dest)
        if value is None:
            continue
        # Workflow normalizers use an empty string for an unused route. Some
        # historical parsers represent that same state with a None default and
        # deliberately exclude "" from choices. Emitting --interaction_model ''
        # would make an otherwise valid interactive configuration unparseable.
        if value == "" and action.choices is not None and "" not in action.choices:
            continue
        if value == action.default:
            continue
        flag = next(
            (x for x in action.option_strings if x.startswith("--")), action.option_strings[0]
        )
        if isinstance(action, argparse.BooleanOptionalAction):
            result.append(
                flag if value else next(x for x in action.option_strings if x.startswith("--no-"))
            )
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                result.append(flag)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                result.append(flag)
        elif isinstance(value, (list, tuple)) and action.nargs in {"*", "+"}:
            result.extend([flag, *map(str, value)])
        else:
            text = json.dumps(value) if isinstance(value, (dict, list, tuple)) else str(value)
            result.extend([flag, text])
    return result


def replay_arguments(parser, args):
    result = namespace_argv(parser, args)
    # Even a configuration consisting entirely of defaults must never re-enter
    # a historical trainer's no-arguments interactive path.
    if "--task_type" not in result:
        result = ["--task_type", args.task_type, *result]
    return result


def print_group(ui, title, values):
    if ui.cursor < ui.replay_until:
        return
    ui.say(title)
    print(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False))


def training_summary(ui, args, parser, script, *, wrapped=False, workspace=None):
    from .objectives import resolve_selection
    from .semantic import normalize_semantic_aliases

    effective = normalize_semantic_aliases(deepcopy(args))
    selection = resolve_selection(effective)
    values = vars(effective)

    def fields(names):
        return {key: values[key] for key in names if key in values}

    ui.say("Configuration")
    print_group(
        ui,
        "Model route",
        fields(
            (
                "task_type",
                "tokenizer",
                "encoder",
                "reduction",
                "model",
                "sequence_model",
                "training_mode",
                "classifier",
                "head_hidden_dims",
                "head_activation",
                "head_dropout",
                "pair_strategy",
                "interaction_level",
                "interaction_model",
                "predictor_type",
                "graph_predictor",
                "fusion_mode",
                "partner_tokenizer",
                "partner_encoder",
                "partner_model_source",
                "partner_max_length",
            )
        ),
    )
    print_group(
        ui,
        "Training settings",
        fields(
            (
                "seed",
                "split_seed",
                "deterministic",
                "k",
                "max_length",
                "batch_size",
                "epochs",
                "lr",
                "enable_regularization",
                "dropout",
                "weight_decay",
                "grad_clip_norm",
                "early_stop_patience",
                "early_stop_min_delta",
                "ml_scaler",
                "dl_scaler",
                "ml_reduction",
                "ml_auto_tune",
                "evaluate_test",
            )
        ),
    )
    print_group(
        ui, "Classification loss", fields(("multilabel_loss", "loss_type", "pos_weight_max"))
    )
    semantic = {
        key: value
        for key, value in values.items()
        if key == "enable_semantic_similarity" or key.startswith("semantic_")
    }
    # Display the effective implicit loss without materializing this opt-in
    # field in the real namespace or historical artifact schemas.
    semantic.setdefault("semantic_loss_type", "legacy")
    print_group(ui, "Semantic settings", semantic)
    print_group(
        ui,
        "Selection protocol",
        {
            "checkpoint": {
                "id": selection.checkpoint.id,
                "params": dict(selection.checkpoint.params),
            },
            "checkpoint_threshold": {
                "id": selection.checkpoint_threshold.id,
                "params": dict(selection.checkpoint_threshold.params),
            },
            "threshold": {"id": selection.threshold.id, "params": dict(selection.threshold.params)},
            "tuning": {"id": selection.tuning.id, "params": dict(selection.tuning.params)},
            "threshold_search_profile": selection.search_profile,
            "checkpoint_threshold_search_profile": selection.checkpoint_search_profile,
        },
    )
    paths = fields(
        (
            "data_dir",
            "task_root",
            "train_csv",
            "val_csv",
            "test_csv",
            "feature_dir",
            "esm_model",
            "protbert_model",
            "prott5_model",
            "esmc_model",
            "esm3_model",
        )
    )
    for key, value in list(paths.items()):
        if value and key in {
            "data_dir",
            "task_root",
            "train_csv",
            "val_csv",
            "test_csv",
            "feature_dir",
        }:
            paths[key] = str(path_value(value, base=ROOT))
    paths["artifact_dir"] = args.artifact_dir or ui.t(
        "Automatic run directory" if wrapped else "Original trainer output layout"
    )
    if wrapped:
        paths["workspace"] = str(Path(workspace).expanduser().resolve())
    print_group(ui, "Inputs and outputs", paths)
    if ui.cursor >= ui.replay_until:
        print(ui.t("Working directory") + ": " + str(ROOT))
        if wrapped:
            command = [
                sys.executable,
                "-m",
                "pepplatform_cli",
                "train",
                "--task-type",
                args.task_type,
                "--workspace",
                str(Path(workspace).expanduser().resolve()),
                "--",
                *replay_arguments(parser, args),
            ]
        else:
            command = [sys.executable, "-B", str(script), *replay_arguments(parser, args)]
        print(ui.t("Replay") + ": " + shlex.join(command))


def validate_direct_inputs(args):
    """Check resolved files without changing the trainer's fallback policies."""
    if args.task_type == "pep_function":
        root = path_value(args.data_dir, base=ROOT, exists="directory")
        for split in ("train", "val"):
            for name in ("seqs.fasta", "labels.csv"):
                if not (root / split / name).is_file():
                    raise ValueError(f"Missing dataset file: {root / split / name}")
        return
    from scripts.train_pair_pipeline import default_pair_csv_candidates

    from .cli_wizard import InteractiveConfigWizard

    task_root = path_value(
        args.task_root or InteractiveConfigWizard()._task_root(args.task_type), base=ROOT
    )
    data_root = path_value(args.data_dir, base=ROOT, exists="directory") if args.data_dir else None
    for split in ("train", "val", "test"):
        explicit = getattr(args, f"{split}_csv", "")
        if explicit:
            path_value(explicit, base=ROOT, exists="file")
        elif data_root is not None and (
            split == "train" or split == "val" and (data_root / split / "pairs.csv").exists()
        ):
            path_value(str(data_root / split / "pairs.csv"), exists="file")
        elif split == "train":
            candidates = [
                task_root / p for p in default_pair_csv_candidates("", args.task_type, split)
            ]
            if not any(p.is_file() for p in candidates):
                raise ValueError(f"Missing dataset file: {candidates[0]}")


def configure_training(
    arguments,
    *,
    task=None,
    default_task="pep_function",
    language=None,
    ui=None,
    wrapped=False,
    advanced=False,
    artifact_dir=None,
    workspace=None,
    validate=None,
    current=None,
):
    """Return a reviewed (namespace, parser, script) without executing anything."""
    ui = ui or Interaction(language)
    ui.choose_language()
    arguments = without_language(list(arguments))
    fixed_task = normalize_task_type(task) if task else explicit_task(arguments)

    def collect():
        selected = fixed_task or ui.choice("Task", TASKS, normalize_task_type(default_task))
        if current is not None and selected == normalize_task_type(default_task):
            parser, wizard_type, script = current
        else:
            parser, wizard_type, script = trainer_for(selected)
        args = parser.parse_args(["--task_type", selected, *arguments])
        vars(args).pop("lang", None)
        if normalize_task_type(args.task_type) != selected:
            raise ValueError("Use the outer --task-type to select the training task")
        args.task_type = selected
        args.advanced_wizard = advanced or args.advanced_wizard
        if artifact_dir is not None:
            args.artifact_dir = artifact_dir
        wizard = wizard_type(ui=ui, task_selected=True, wrapped=wrapped)
        args = wizard.run(args)

        def output(value):
            if not value:
                return ""
            path = path_value(value, base=ROOT)
            if path.exists():
                raise ValueError(f"Output already exists: {path}")
            return str(path)

        args.artifact_dir = ui.text(
            "Model output directory (blank=automatic)", args.artifact_dir, validate=output
        )
        try:
            (validate or validate_direct_inputs)(args)
        except (OSError, ValueError) as exc:
            ui.say(f"Invalid configuration: {exc}")
            if getattr(exc, "code", "").startswith("output_"):
                ui.retry_field("Model output directory (blank=automatic)")
            ui.retry_field(
                "Data directory"
                if wrapped or selected == "pep_function"
                else "Data directory (blank=legacy CSV layout)"
            )
        errors = io.StringIO()
        try:
            with redirect_stderr(errors):
                parser.parse_args(replay_arguments(parser, args))
        except SystemExit:
            raise ValueError(
                "Invalid configuration: " + errors.getvalue().splitlines()[-1]
            ) from None
        training_summary(ui, args, parser, script, wrapped=wrapped, workspace=workspace)
        ui.confirm()
        return args, parser, script

    return ui.collect(collect)
