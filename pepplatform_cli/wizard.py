"""Shared-language navigation. Collection is read-only until explicit execution."""
from __future__ import annotations

import json
import math
import shlex
import sys
from pathlib import Path

from pepplatform.cli_interaction import TASKS, Interaction, WizardCancelled, path_value
from pepplatform.cli_training import print_group

from .storage import guard_output, history, read_json, write_json


def json_object(value):
    def invalid_constant(_):
        raise ValueError("Please enter a JSON object.")

    try:
        result = json.loads(value, parse_constant=invalid_constant)
    except ValueError:
        raise ValueError("Please enter a JSON object.") from None
    if not isinstance(result, dict):
        raise ValueError("Please enter a JSON object.")  # noqa: TRY004 -- JSON text has an invalid value/shape.
    return result


def directory(ui, title, default=""):
    return ui.text(title, default, validate=lambda value: str(path_value(value, exists="directory")))


def new_output(value, sources=()):
    destination = guard_output(path_value(value), sources)
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}")
    return str(destination)


def replay(ui, command):
    if ui.cursor >= ui.replay_until:
        print(ui.t("Replay") + ": " + shlex.join(command))


def ensemble_wizard(workspace, ui=None):
    from .__main__ import create_ensemble
    from .ensemble import preflight

    ui = ui or Interaction()
    workspace = Path(workspace).expanduser().resolve()
    ui.choose_language()

    def collect():
        task = ui.choice("Task", TASKS, TASKS[0])
        config = {"task_type": task, "data_dir": directory(ui, "Dataset directory")}
        if task == "pep_hlai":
            source_root = ui.text("Source directory (HLA-I mapping; blank=automatic)", validate=lambda value:
                                  str(path_value(value, exists="directory")) if value else "")
            if source_root:
                config["source_root"] = source_root

        def members(value):
            try:
                paths = shlex.split(value)
            except ValueError:
                raise ValueError("Please enter at least two member artifact paths.") from None
            if len(paths) < 2:
                raise ValueError("Please enter at least two member artifact paths.")
            return [str(path_value(path, exists="directory")) for path in paths]

        config["members"] = ui.text("Member artifact paths (quote paths containing spaces)", validate=members)
        config["method"] = ui.choice("Fusion method", ["probability_mean", "logit_mean", "rank_mean"])
        config["weight_source"] = ui.choice("Member weights", ["equal", "custom", "validation"])
        if config["weight_source"] == "custom":
            def weights(value):
                try:
                    result = [float(x) for x in value.replace(",", " ").split()]
                except ValueError:
                    result = []
                if len(result) != len(config["members"]) or any(not math.isfinite(x) or x < 0 for x in result) or not any(result):
                    raise ValueError("Please enter finite nonnegative weights with a positive sum, one per member.")
                return result
            config["weights"] = ui.text("Weights in member order", validate=weights)
        elif config["weight_source"] == "validation":
            config["grid_step"] = ui.number("Weight grid step", 0.05, 0, exclusive_minimum=True)
        config["objective"] = ui.choice("Weight selection objective",
            ["macro_f1", "sacc", "auc", "aupr", "dual_f1_sacc"] if task == "pep_function" else ["auc", "aupr", "f1"])
        if config["objective"] == "dual_f1_sacc":
            config["objective_params"] = ui.text("Objective parameters (JSON)", validate=json_object)
        config["threshold_objective"] = ui.choice("Threshold objective",
            ["macro_f1", "sacc", "dual_f1_sacc"] if task == "pep_function" else ["f1"])
        if config["threshold_objective"] == "dual_f1_sacc":
            config["threshold_objective_params"] = ui.text("Threshold parameters (JSON)", validate=json_object)
        config["seed"] = ui.integer("Seed", 42)
        config["evaluate_test"] = ui.choice("Evaluate independent tests after freezing?", ["No", "Yes"]) == "Yes"
        if config["evaluate_test"]:
            names = ui.text("Test split names (empty: test/test_* discovery)")
            if names:
                config["test_splits"] = names.replace(",", " ").split()
        config["allow_unverified_provenance"] = ui.choice("Training provenance",
            ["Strict verification", "Allow unverified legacy provenance"]) == "Allow unverified legacy provenance"
        while True:
            report = preflight(config)
            if report["valid"]:
                break
            print_group(ui, "Preflight failed", report)
            action = ui.choice("Preflight failed", ["Modify", "Edit configuration as JSON", "Return to navigation"])
            if action == "Return to navigation":
                raise WizardCancelled()
            if action == "Modify":
                ui.rewind()
            config.update(ui.text("Configuration fields to replace (JSON object)", validate=json_object))
        print_group(ui, "Configuration", config)
        sources = [config["data_dir"], *[m["artifact_dir"] if isinstance(m, dict) else m for m in config["members"]]]
        destination = ui.text("Save replayable configuration", str(Path(workspace) / "configs" / "ensemble.json"),
                              validate=lambda value: new_output(value, sources))
        replay(ui, [sys.executable, "-m", "pepplatform_cli", "ensemble", "create", "--config", destination,
                    "--workspace", str(Path(workspace).expanduser().resolve())])
        ui.confirm()
        return config, destination

    config, destination = ui.collect(collect)
    write_json(destination, config)
    ui.result(create_ensemble(config, workspace))


def predict_wizard(workspace, ui):
    from pepplatform.task_compat import normalize_task_type

    from . import __main__
    from .data import prediction_inputs
    workspace = Path(workspace).expanduser().resolve()

    def collect():
        def artifact(value):
            path = path_value(value, exists="directory")
            path_value(str(path / "metadata.json"), exists="file")
            read_json(path / "metadata.json")
            return str(path)

        bundle = ui.text("Model artifact directory", validate=artifact)
        metadata = read_json(Path(bundle) / "metadata.json")
        task = normalize_task_type(metadata.get("args", {}).get("task_type") or metadata.get("task_type"))
        input_path = ui.text("Input FASTA/CSV/JSON", validate=lambda value: str(path_value(value, exists="file")))
        source_root = ""
        if task == "pep_hlai":
            source_root = ui.text("Source directory (HLA-I mapping; blank=automatic)", validate=lambda value:
                                  str(path_value(value, exists="directory")) if value else "")
        try:
            prediction_inputs(input_path, task, source_root or None)
        except (ValueError, OSError) as exc:
            ui.say(f"Invalid configuration: {exc}")
            ui.retry_field("Input FASTA/CSV/JSON")
        command = ["predict", "--artifact-dir", bundle, "--input", input_path, "--workspace", str(workspace)]
        if source_root:
            command.extend(["--source-root", source_root])
        print_group(ui, "Configuration", {"task_type": task, "artifact_dir": bundle, "input": input_path,
                                           "source_root": source_root or None, "workspace": str(workspace)})
        replay(ui, [sys.executable, "-m", "pepplatform_cli", *command])
        ui.confirm()
        return __main__.build_parser().parse_args(command)

    ui.result(__main__.predict(ui.collect(collect)))


def history_wizard(workspace, ui):
    from . import __main__
    workspace = Path(workspace).expanduser().resolve()
    records = history(workspace)
    if not records:
        ui.say("No runs found.")
        return
    print_group(ui, "History", [{key: record.get(key) for key in ("run_id", "kind", "status")} for record in records])

    def collect():
        action = ui.choice("History actions", ["Show run", "Export inference CSV", "Return to navigation"],
                           "Return to navigation")
        if action == "Return to navigation":
            raise WizardCancelled()
        choices = records if action == "Show run" else [r for r in records if r.get("kind") == "predict" and r.get("status") == "succeeded"]
        if not choices:
            ui.say("Select a successful inference run.")
            ui.rewind()
        run_id = ui.choice("Run ID", [r["run_id"] for r in choices])
        command = ["history", "show" if action == "Show run" else "export", run_id, "--workspace", str(workspace)]
        if action != "Show run":
            record = next(r for r in choices if r["run_id"] == run_id)
            destination = ui.text("New CSV path", validate=lambda value: new_output(
                value, [record["directory"], record.get("source_artifact", "")]))
            command.extend(["--output", destination])
            print_group(ui, "Configuration", {"run_id": run_id, "output": destination})
            replay(ui, [sys.executable, "-m", "pepplatform_cli", *command])
            ui.confirm()
        return __main__.build_parser().parse_args(command)

    ui.result(__main__.dispatch(ui.collect(collect)))


def navigate(workspace, ui=None):
    from .training import train
    ui = ui or Interaction(steps=False)
    workspace = Path(workspace).expanduser().resolve()
    ui.choose_language()
    while True:
        try:
            selected = ui.collect(lambda: ui.choice("PepPlatform / Local workflows",
                                  ["Train", "Ensemble", "Predict", "History", "Exit"], "Train"))
        except WizardCancelled:
            return
        if selected == "Exit":
            return
        child = Interaction(ui.language)
        try:
            if selected == "Train":
                child.result(train([], task=None, interactive=True, workspace=workspace, ui=child))
            elif selected == "Ensemble":
                ensemble_wizard(workspace, child)
            elif selected == "Predict":
                predict_wizard(workspace, child)
            else:
                history_wizard(workspace, child)
        except WizardCancelled:
            ui.say("Returned to navigation")
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            ui.say(f"Operation failed ({getattr(exc, 'code', 'wizard_failed')}): {exc}")
