"""Unified navigation without importing the CLI from any historical trainer."""
from __future__ import annotations

import json
import shlex
from pathlib import Path

from .storage import CLIError, guard_output, history, write_json


def choice(title, options, default=0):
    print(f"\n{title}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    while True:
        raw = input(f"Choice [{default + 1}] (b: back, q: cancel): ").strip()
        if raw.lower() in {"b", "q"}:
            return None
        try:
            selected = int(raw or default + 1) - 1
            if 0 <= selected < len(options):
                return options[selected]
        except ValueError:
            pass
        print("Choose a listed number.")


def text(title, default=""):
    raw = input(f"{title}" + (f" [{default}]" if default else "") + " (b: back, q: cancel): ").strip()
    if raw.lower() in {"b", "q"}:
        raise CLIError("wizard_cancelled", "Returned to navigation")
    return raw or default


def ensemble_wizard(workspace):
    from .ensemble import preflight
    from .__main__ import create_ensemble
    from .__main__ import TASKS
    task = choice("Task", list(TASKS))
    if task is None:
        return
    config = {"task_type": task, "data_dir": str(Path(text("Dataset directory")).expanduser().resolve())}
    config["members"] = [str(Path(p).expanduser().resolve()) for p in shlex.split(text("Member artifact paths (quote paths containing spaces)"))]
    config["method"] = choice("Fusion method", ["probability_mean", "logit_mean", "rank_mean"])
    if config["method"] is None:
        return
    config["weight_source"] = choice("Member weights", ["equal", "custom", "validation"])
    if config["weight_source"] is None:
        return
    if config["weight_source"] == "custom":
        config["weights"] = [float(x) for x in text("Weights in member order").replace(",", " ").split()]
    if config["weight_source"] == "validation":
        config["grid_step"] = float(text("Weight grid step", "0.05"))
    config["objective"] = choice("Weight selection objective", ["macro_f1", "sacc", "auc", "aupr", "dual_f1_sacc"]
                                 if task == "pep_function" else ["auc", "aupr", "f1"])
    if config["objective"] is None:
        return
    if config["objective"] == "dual_f1_sacc":
        config["objective_params"] = json.loads(text("Objective parameters (JSON)"))
    config["threshold_objective"] = choice("Threshold objective", ["macro_f1", "sacc", "dual_f1_sacc"]
                                           if task == "pep_function" else ["f1"])
    if config["threshold_objective"] is None:
        return
    if config["threshold_objective"] == "dual_f1_sacc":
        config["threshold_objective_params"] = json.loads(text("Threshold parameters (JSON)"))
    config["seed"] = int(text("Seed", "42"))
    evaluate = choice("Evaluate independent tests after freezing?", ["No", "Yes"])
    if evaluate is None:
        return
    config["evaluate_test"] = evaluate == "Yes"
    if config["evaluate_test"]:
        names = text("Test split names (empty: test/test_* discovery)")
        if names:
            config["test_splits"] = names.replace(",", " ").split()
    policy = choice("Training provenance", ["Strict verification", "Allow unverified legacy provenance"])
    if policy is None:
        return
    config["allow_unverified_provenance"] = policy == "Allow unverified legacy provenance"
    while True:
        report = preflight(config)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["valid"]:
            break
        action = choice("Preflight failed", ["Edit configuration as JSON", "Return to navigation"])
        if action != "Edit configuration as JSON":
            return
        config.update(json.loads(text("Configuration fields to replace (JSON object)")))
    print("\nConfiguration:\n" + json.dumps(config, ensure_ascii=False, indent=2))
    destination = guard_output(text("Save replayable configuration", str(Path(workspace) / "configs" / "ensemble.json")),
                               [config["data_dir"], *[m["artifact_dir"] if isinstance(m, dict) else m for m in config["members"]]])
    if destination.exists():
        raise CLIError("output_exists", f"Choose a new configuration file: {destination}")
    write_json(destination, config)
    command = ["python", "-m", "pepplatform_cli", "ensemble", "create", "--config", str(destination),
               "--workspace", str(Path(workspace).resolve())]
    print("Replay: " + shlex.join(command))
    if choice("Create this ensemble?", ["Create", "Return to navigation"], default=1) == "Create":
        print(json.dumps(create_ensemble(config, workspace), ensure_ascii=False, indent=2))


def navigate(workspace):
    from .__main__ import TASKS, main
    while True:
        selected = choice("PepPlatform / 本地工作流", ["Train / 训练", "Ensemble / 创建集成", "Predict / 推理",
                                                      "History / 运行历史", "Exit / 退出"])
        if selected is None or selected.startswith("Exit"):
            return
        try:
            if selected.startswith("Train"):
                task = choice("Task", list(TASKS))
                if task is not None:
                    main(["train", "--task-type", task, "--interactive", "--workspace", str(workspace)])
            elif selected.startswith("Ensemble"):
                ensemble_wizard(workspace)
            elif selected.startswith("Predict"):
                main(["predict", "--artifact-dir", text("Model artifact directory"), "--input", text("Input FASTA/CSV/JSON"),
                      "--workspace", str(workspace)])
            else:
                records = history(workspace)
                print(json.dumps(records, ensure_ascii=False, indent=2))
                if records:
                    action = choice("History actions", ["Show run", "Export inference CSV", "Return to navigation"], 2)
                    if action == "Show run":
                        main(["history", "show", text("Run ID"), "--workspace", str(workspace)])
                    elif action == "Export inference CSV":
                        main(["history", "export", text("Run ID"), "--output", text("New CSV path"),
                              "--workspace", str(workspace)])
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            print(f"{getattr(exc, 'code', 'wizard_failed')}: {exc}")
