"""An opt-in provenance envelope around the unchanged training entrypoints."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from pepplatform.task_compat import normalize_task_type
from . import data
from .storage import (CLIError, ROOT, file_digest, finish_run, guard_output, inventory,
                      new_run, read_json, write_json)


def parser_for(task):
    if normalize_task_type(task) == "pep_function":
        from scripts.train_sequence_pipeline import build_argument_parser, InteractiveConfigWizard
        return build_argument_parser(), InteractiveConfigWizard, ROOT / "scripts/train_sequence_pipeline.py"
    from scripts.train_pair_pipeline import build_argument_parser, PPIInteractiveWizard
    return build_argument_parser(), PPIInteractiveWizard, ROOT / "scripts/train_pair_pipeline.py"


def namespace_argv(parser, namespace):
    """Serialize through argparse's declared actions; never invent trainer flags."""
    result, seen = [], set()
    for action in parser._actions:
        dest = action.dest
        if (dest in seen or dest in {"help", "interactive", "advanced_wizard", "list_objectives"}
                or not action.option_strings or not hasattr(namespace, dest)):
            continue
        seen.add(dest)
        value = getattr(namespace, dest)
        if value is None:
            continue
        if value == action.default:
            continue
        flag = next((x for x in action.option_strings if x.startswith("--")), action.option_strings[0])
        if isinstance(action, argparse.BooleanOptionalAction):
            result.append(flag if value else next(x for x in action.option_strings if x.startswith("--no-")))
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


def train(arguments, *, task="pep_function", interactive=False, advanced=False,
          workspace=None, artifact_dir=None, run_process=subprocess.run):
    from .storage import DEFAULT_WORKSPACE
    parser, wizard, script = parser_for(task)
    args = parser.parse_args(["--task_type", task, *arguments])
    if normalize_task_type(args.task_type) != normalize_task_type(task):
        raise CLIError("task_mismatch", "Use the outer --task-type to select the training task")
    args.advanced_wizard = advanced or args.advanced_wizard
    if interactive or args.interactive or not arguments:
        args = wizard().run(args)
        # The shared wizard may switch task; reparse with that task's actual parser.
        chosen = normalize_task_type(args.task_type)
        if chosen != normalize_task_type(task):
            old_parser = parser
            parser, _, script = parser_for(chosen)
            args = parser.parse_args(namespace_argv(old_parser, args))
    task = normalize_task_type(args.task_type)
    source = Path(args.data_dir).expanduser()
    source = source.resolve() if source.is_absolute() else (ROOT / source).resolve()
    if not args.data_dir or not (source / "train").is_dir() or not (source / "val").is_dir():
        raise CLIError("training_data_required", "Provenance training requires --data_dir with train/ and val/ splits")
    explicit_root = getattr(args, "task_root", "")
    original_data = data.dataset({"task_type": task, "data_dir": str(source),
                                 "source_root": str((ROOT / explicit_root).resolve()) if explicit_root else str(source.parent)})
    before = data.validation_identity(original_data)
    selected_output = artifact_dir or getattr(args, "artifact_dir", "") or None
    run = new_run(workspace or DEFAULT_WORKSPACE, "train", inputs=[source], artifact_dir=selected_output)
    directory = Path(run["directory"])
    try:
        # Independent input bytes prevent trainers from generating caches in original datasets.
        copied_data = directory / "data"
        shutil.copytree(source, copied_data, symlinks=False)
        source_root = Path(original_data["source_root"])
        mapping = source_root / "pseudoseqs_decoded.csv"
        if mapping.is_file():
            shutil.copy2(mapping, directory / mapping.name)
        copied = data.dataset({"task_type": task, "data_dir": str(copied_data), "source_root": str(directory)})
        if not data.same_data(before, data.validation_identity(copied)):
            raise CLIError("training_data_changed", "Data changed while preparing independent inputs")
        args.data_dir, args.artifact_dir = str(copied_data), run["artifact_dir"]
        if hasattr(args, "task_root"):
            args.task_root = str(directory)
        # Every trainer-managed feature/representation output gets an independent location.
        # Existing resources are copied before reuse; no hard links or writes to their sources.
        representation_paths = {"bpe_path", "wordpiece_path", "unigram_path", "word2vec_path", "glove_path", "fasttext_path"}
        for name, value in list(vars(args).items()):
            is_cache = name == "feature_dir" or name.endswith("_feature_dir") or name == "profile_cache_dir"
            if not is_cache and name not in representation_paths:
                continue
            if not value and name in representation_paths:
                continue
            destination = directory / ("features" if is_cache else "representations") / name
            if value:
                original = Path(value).expanduser()
                original = original.resolve() if original.is_absolute() else (ROOT / original).resolve()
                if original.is_dir():
                    shutil.copytree(original, destination, symlinks=False)
                elif original.is_file():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(original, destination)
            setattr(args, name, str(destination))
        for name in ("train_csv", "val_csv", "test_csv"):
            if hasattr(args, name):
                value = getattr(args, name)
                if value:
                    path = Path(value).expanduser()
                    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
                    if not path.is_relative_to(source):
                        raise CLIError("training_split_external", "Explicit split files must lie within --data_dir")
                    if name != "test_csv" and path != source / name.removesuffix("_csv") / "pairs.csv":
                        raise CLIError("training_split_mismatch", "Training provenance requires standard train/val pairs.csv inputs")
                    setattr(args, name, str(copied_data / path.relative_to(source)))
        from .ensemble import runtime_identity
        source_identity = runtime_identity()
        command = [sys.executable, "-B", str(script), *namespace_argv(parser, args)]
        # Check all serialized values with the actual parser before executing.
        parser.parse_args(command[3:])
        write_json(directory / "command.json", {"command": command, "cwd": str(ROOT)})
        completed = run_process(command, cwd=ROOT, check=False)
        if completed.returncode:
            raise CLIError("training_failed", f"Trainer exited with status {completed.returncode}")
        if runtime_identity() != source_identity:
            raise CLIError("training_runtime_changed", "Runtime source changed during training")
        after = data.validation_identity(copied)
        if not data.same_data(before, after) or before != data.validation_identity(original_data):
            raise CLIError("training_data_changed", "Train/Validation changed during training")
        artifact = Path(run["artifact_dir"])
        metadata = read_json(artifact / "metadata.json")
        if normalize_task_type(metadata.get("args", {}).get("task_type")) != task:
            raise CLIError("training_artifact_invalid", "Trainer output task differs from the requested task")
        recorded_data = metadata.get("args", {}).get("data_dir")
        if not recorded_data or Path(recorded_data).resolve() != copied_data:
            raise CLIError("training_artifact_invalid", "Trainer output does not identify the pinned input directory")
        proof = {"schema_version": 1, "kind": "cli_training", "status": "verified",
                 "dataset_identity": after, "original_dataset_identity": before,
                 "artifact_files": inventory(artifact), "command": command,
                 "trainer_sha256": file_digest(script), "runtime_sources": source_identity,
                 "python_version": platform.python_version()}
        # Receipt lives outside the bundle, avoiding circular artifact hashes.
        write_json(directory / "training_provenance.json", proof)
        if artifact.parent != directory:
            # Explicit output paths use the run record to locate their proof; never overwrite siblings.
            run["provenance"] = str(directory / "training_provenance.json")
        return finish_run(run, status="succeeded", provenance=str(directory / "training_provenance.json"))
    except Exception as exc:
        finish_run(run, status="failed", error=str(exc))
        raise
