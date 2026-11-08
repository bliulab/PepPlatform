"""Run lifecycle adapter shared by Web and the explicit versioned CLI."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import subprocess

from .core import SCHEMA, atomic_json, now, sha256, tree_files


DATA_FLAGS = {"data_dir", "task_root", "train_csv", "val_csv", "test_csv"}
OUTPUT_FLAGS = {"artifact_dir", "artifact-dir", "profile_cache_dir"}


def command_options(command):
    """Commands come from the platform builder, never from a shell parser."""
    return {
        arg[2:]: (i + 1, command[i + 1])
        for i, arg in enumerate(command[:-1])
        if arg.startswith("--") and not command[i + 1].startswith("--")
    }


def replace_paths(value, replacements):
    if isinstance(value, dict):
        return {key: replace_paths(item, replacements) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [replace_paths(item, replacements) for item in value]
    if isinstance(value, str):
        for path, replacement in sorted(
            replacements.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if value == path or value.startswith(path.rstrip("/") + "/"):
                return replacement + value[len(path) :]
    return value


def config_identity(config, replacements):
    return replace_paths(
        {
            k: v
            for k, v in config.items()
            if k not in OUTPUT_FLAGS
            and not k.startswith("_")
            and k not in {"interactive", "advanced_wizard"}
        },
        replacements,
    )


def source_files(root):
    files = {}
    for name in ("pepplatform", "scripts"):
        directory = root / name
        if directory.is_dir():
            files.update({f"{name}/{key}": path for key, path in tree_files(directory).items()})
    for name in ("requirements.txt", "requirements-web.txt", "requirements-pretrained.txt"):
        if (root / name).is_file():
            files[name] = root / name
    if not files:
        raise ValueError("runtime source tree is unavailable")
    return files


def git_revision(root):
    def git(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            # Source snapshots and hashes remain authoritative in source archives
            # and minimal images that do not ship the optional Git executable.
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=all")
    return {"git_commit": commit, "git_dirty": bool(dirty) if dirty is not None else None}


def runtime_environment():
    import platform
    import sys

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": sorted(
            {
                f"{d.metadata['Name']}=={d.version}"
                for d in importlib.metadata.distributions()
                if d.metadata.get("Name")
            }
        ),
    }


class RunVersions:
    def __init__(self, repository, project_root, input_policy=None):
        self.repo = repository
        self.project_root = Path(project_root).resolve()
        self.input_policy = input_policy

    def begin(self, experiment):
        identifier = experiment["id"]
        self.repo.start_run(identifier, experiment.get("owner_user_id"))
        self.repo.update_run(identifier, task_type=experiment["task_type"])
        if experiment.get("rerun_source_experiment_id"):
            self.repo.relate(identifier, "rerun_of", experiment["rerun_source_experiment_id"])
        resume = experiment.get("resume") or {}
        if resume:
            source = resume.get("source_experiment_id")
            if source:
                self.repo.relate(identifier, "reuses_representation_from", source)

    def prepare(self, experiment, command):
        run_id = experiment["id"]
        workspace = self.repo.root / "workspaces" / run_id
        workspace.mkdir(parents=True, exist_ok=False)
        requested_command = list(command)
        if self.input_policy is not None:
            command = self.input_policy.prepare_command(
                experiment, command, workspace, self.project_root
            )
        replacements = {}
        views = []
        data_objects = []
        split_roles = {}
        rewritten = list(command)
        resources = {}
        for flag, (index, value) in command_options(command).items():
            if flag in OUTPUT_FLAGS or (
                self.input_policy is not None and flag in self.input_policy.output_flags
            ):
                continue
            is_input = flag in DATA_FLAGS
            pathish = (
                is_input
                or flag.endswith(("_path", "_dir", "_model"))
                or flag == "resume_embedding_bundle"
            )
            if self.input_policy is not None:
                pathish = pathish or self.input_policy.is_resource_flag(flag)
            if not pathish:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            path = path.resolve()
            if not path.exists():
                if is_input:
                    raise ValueError(f"required input is missing: {flag}")
                continue  # Remote model identifiers are recorded by the trained artifact.
            if path.is_dir() and (
                self.repo.root.is_relative_to(path)
                or Path(experiment["artifact_dir"]).resolve().is_relative_to(path)
            ):
                if flag != "resume_embedding_bundle":
                    raise ValueError(
                        f"input would recursively include version storage or output: {flag}"
                    )
            if self.input_policy is not None:
                files = self.input_policy.files_for_input(flag, path, experiment, workspace)
            elif flag == "resume_embedding_bundle":
                # Existing trainer expects only these resume members; never snapshot its output recursively.
                files = {
                    name: p
                    for name, p in tree_files(path).items()
                    if name.startswith("resume/") or name == "embedding_encoder.pt"
                }
            else:
                files = tree_files(path) if path.is_dir() else {path.name: path}
            kind = "data_files" if is_input else "resource"
            record = self.repo.snapshot(kind, files)
            view = self.repo.materialize(record["id"], workspace / flag)
            if self.input_policy is not None:
                record = self.input_policy.rebase_input(
                    self.repo, run_id, flag, path, files, record, view
                )
            self.repo.bind(run_id, f"input:{flag}", record["id"])
            actual = view if path.is_dir() else view / path.name
            rewritten[index] = str(actual)
            for original in (value, str(path), str(actual)):
                replacements[original] = record["id"]
            views.append({"flag": flag, "path": str(view), "version_id": record["id"]})
            if is_input:
                data_objects.append(record["id"])
                split_roles[flag] = record["id"]
            else:
                resources[flag] = record["id"]
        if not data_objects and not experiment["config"].get("web_ensemble"):
            raise ValueError("no dataset input could be pinned")
        dataset = self.repo.register(
            "dataset", {"task_type": experiment["task_type"], "objects": sorted(set(data_objects))}
        )
        self.repo.bind(run_id, "dataset", dataset["id"])
        split = self.repo.register(
            "split",
            {
                "dataset_version": dataset["id"],
                "inputs": split_roles,
                "method": "provided_files",
                "split_seed": None,
            },
        )
        self.repo.bind(run_id, "split", split["id"])
        config = self.repo.register(
            "config", {"config": config_identity(experiment["config"], replacements)}
        )
        self.repo.bind(run_id, "config", config["id"])
        source = self.repo.snapshot("source", source_files(self.project_root))
        self.repo.bind(run_id, "source", source["id"])
        env = self.repo.register("environment", runtime_environment())
        self.repo.bind(run_id, "environment", env["id"])
        policy_details = (
            {"input_policy": self.input_policy.name} if self.input_policy is not None else {}
        )
        self.repo.update_run(
            run_id,
            status="running",
            command=rewritten,
            requested_command=requested_command,
            source_revision=git_revision(self.project_root),
            input_views=views,
            path_identities=replacements,
            checked_at=now(),
            **policy_details,
        )
        self.repo.export(run_id, workspace / "versions.inputs.json")
        return rewritten

    def prepare_ensemble(self, experiment, command):
        """Rebase already-verified provenance onto byte-identical input copies."""
        from pepplatform.web.ensemble_data import (
            PROVENANCE_FILE,
            tree_identity,
            validation_identity,
            verify_provenance,
        )

        request_index, request_path = command_options(command)["request"]
        request = json.loads(Path(request_path).read_text())
        original_dataset = request["dataset"]
        original_identity = validation_identity(original_dataset)
        if original_identity != request["dataset_identity"]:
            raise ValueError("ensemble dataset changed before snapshotting")
        virtual = [command[0], "--data_dir", original_dataset["root_path"]]
        source_root = (original_dataset.get("metadata") or {}).get("source_root")
        if source_root:
            virtual.extend(["--task_root", source_root])
        prepared = command_options(self.prepare(experiment, virtual))
        pinned_dataset = {
            **original_dataset,
            "root_path": prepared["data_dir"][1],
            "metadata": dict(original_dataset.get("metadata") or {}),
        }
        if source_root:
            pinned_dataset["metadata"]["source_root"] = prepared["task_root"][1]
        pinned_identity = validation_identity(pinned_dataset)
        # Path keys differ by design. All sample identities, order and bytes must agree.
        for split in ("train", "validation"):
            before, after = original_identity[split], pinned_identity[split]
            if {k: v for k, v in before.items() if k != "files"} != {
                k: v for k, v in after.items() if k != "files"
            } or sorted(before["files"].values()) != sorted(after["files"].values()):
                raise ValueError(
                    "ensemble input snapshot does not match historical member evidence"
                )
        run_id = experiment["id"]
        workspace = self.repo.root / "workspaces" / run_id
        views = self.repo.run(run_id)["details"]["input_views"]
        members = []
        for index, record in enumerate(request["members"]):
            provenance = verify_provenance(
                record["bundle_path"], original_dataset, original_identity
            )
            if tree_identity(record["bundle_path"]) != record["bundle_files"]:
                raise ValueError("ensemble member changed before snapshotting")
            original = self.repo.snapshot("model", tree_files(record["bundle_path"]))
            self.repo.bind(run_id, f"member:{index}", original["id"])
            self.repo.relate(run_id, "ensemble_member", original["id"])
            view = self.repo.materialize(original["id"], workspace / f"member_{index}")
            # This is new evidence for a derived input view, never a rewrite of the source bundle.
            atomic_json(view / PROVENANCE_FILE, {**provenance, **pinned_identity})
            derived = self.repo.snapshot(
                "resource",
                tree_files(view),
                {"derived_from": original["id"], "transform": "rebase_verified_dataset_paths"},
            )
            self.repo.bind(run_id, f"input:member_{index}", derived["id"])
            views.append(
                {"flag": f"member_{index}", "path": str(view), "version_id": derived["id"]}
            )
            members.append(
                {**record, "bundle_path": str(view), "bundle_files": tree_identity(view)}
            )
        request.update(dataset=pinned_dataset, dataset_identity=pinned_identity, members=members)
        target = workspace / "ensemble_request.json"
        atomic_json(target, request)
        command = list(command)
        command[request_index] = str(target)
        request_version = self.repo.snapshot("resource", {"ensemble_request.json": target})
        self.repo.bind(run_id, "ensemble_request", request_version["id"])
        self.repo.update_run(run_id, command=command, input_views=views)
        return command

    def verify_inputs(self, run_id):
        run = self.repo.run(run_id)
        for view in run.get("details", {}).get("input_views", []):
            record = self.repo.get(view["version_id"])
            self.repo.verify(record["id"])
            for name, identity in record["identity"]["files"].items():
                path = Path(view["path"]) / name
                if not path.is_file() or sha256(path) != identity["sha256"]:
                    raise ValueError(f"materialized input changed: {view['flag']}/{name}")
        source = next((v for v in run["versions"] if v["role"] == "source"), None)
        if source:
            current = source_files(self.project_root)
            expected = source["identity"]["files"]
            if set(current) != set(expected) or any(
                sha256(current[name]) != expected[name]["sha256"] for name in expected
            ):
                raise ValueError("runtime source changed during execution")

    def seal(self, experiment):
        run_id = experiment["id"]
        self.verify_inputs(run_id)
        artifact = Path(experiment["artifact_dir"])
        metadata_path = artifact / "metadata.json"
        if not metadata_path.is_file():
            self.repo.update_run(run_id, status="no_model", checked_at=now())
            return None
        metadata = json.loads(metadata_path.read_text())
        files = tree_files(artifact)
        weights = {
            name: path
            for name, path in files.items()
            if path.suffix
            in {".pt", ".pth", ".joblib", ".pkl", ".safetensors", ".ubj", ".bin", ".txt"}
            and not name.startswith(("resume/", "profile_cache/"))
        }
        # Native predictors and ensemble contracts can use JSON as their executable model format.
        if not weights and metadata.get("bundle_type") != "web_ensemble":
            weights = {
                name: path
                for name, path in files.items()
                if name != "metadata.json" and ("model" in name or "classifier" in name)
            }
        if not weights and metadata.get("bundle_type") != "web_ensemble":
            self.repo.update_run(run_id, status="no_model", checked_at=now())
            return None
        bundle = self.repo.snapshot("model", files)
        self.repo.bind(run_id, "model", bundle["id"])
        if weights:
            checkpoint = self.repo.snapshot(
                "checkpoint", weights, {"selection": "saved_final_artifact"}
            )
            self.repo.bind(run_id, "checkpoint", checkpoint["id"])
        args = metadata.get("args")
        if isinstance(args, dict):
            replacements = self.repo.run(run_id)["details"].get("path_identities", {})
            effective = self.repo.register(
                "config", {"config": config_identity(args, replacements)}
            )
            self.repo.bind(run_id, "effective_config", effective["id"])
        self.repo.update_run(run_id, status="sealed", checked_at=now())
        self.repo.export(run_id, self.repo.root / "workspaces" / run_id / "versions.json")
        return self.repo.object_path(bundle["id"]) / "files"

    def finish(self, experiment):
        run_id = experiment["id"]
        run = self.repo.run(run_id)
        if run["status"] == "not_recorded":
            return
        if run["status"] in {"preparing", "running"}:
            status = experiment["status"]
            # Failed runs can have reusable representations without a final model.
            root = Path(experiment["artifact_dir"])
            if (root / "resume").is_dir():
                try:
                    resume = self.repo.snapshot(
                        "checkpoint",
                        tree_files(root / "resume"),
                        {"selection": "resume_representation"},
                    )
                    self.repo.bind(run_id, "resume_checkpoint", resume["id"])
                except (OSError, ValueError) as exc:
                    self.repo.update_run(run_id, resume_capture_error=str(exc))
            self.repo.update_run(
                run_id, status=status if status in {"failed", "canceled"} else "incomplete"
            )
        self.repo.update_run(run_id, training_status=experiment["status"])
        workspace = self.repo.root / "workspaces" / run_id
        self.repo.export(run_id, workspace / "versions.json")
