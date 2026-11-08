"""Input selection for new Web runs; historical/CLI snapshot policy is unchanged.

Task roots and feature directories mix raw inputs with very large, writable
caches. Pin consumed data and explicit resources, then let the existing trainers
build generated resources in the run's own working directories.
"""

from __future__ import annotations
from pepplatform.task_compat import normalize_task_type


import hashlib
import json
from pathlib import Path

from pepplatform.versioning.core import atomic_json, tree_files
from pepplatform.versioning.service import command_options


class WebTrainingInputs:
    name = "web_required_inputs_v2"
    output_flags = {"feature_dir"}

    @staticmethod
    def is_resource_flag(flag):
        return flag.endswith("_model_source")

    @staticmethod
    def _path(value, project_root):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else project_root / path).resolve()

    def prepare_command(self, experiment, command, workspace, project_root):
        result = list(command)

        def set_option(flag, value):
            current = command_options(result)
            if flag in current:
                result[current[flag][0]] = str(value)
            else:
                result.extend(["--" + flag, str(value)])

        options = command_options(result)
        is_pair = normalize_task_type(experiment["task_type"]) in {
            "pep_protein_interaction",
            "pep_hlai",
            "pep_hlaii",
        }
        if is_pair and not experiment["config"].get("web_ensemble"):
            # Resolve the same defaults before replacing task_root with a small
            # working root. Explicit split files remain byte-identical inputs.
            from scripts.train_pair_pipeline import default_pair_csv_candidates

            task_root = self._path(options["task_root"][1], project_root)
            for split in ("train", "val", "test"):
                flag = split + "_csv"
                if flag in options:
                    continue
                if "data_dir" in options:
                    data = self._path(options["data_dir"][1], project_root)
                    candidates = [data / split / "pairs.csv"]
                    if split == "test":
                        candidates.append(data / "test_warm" / "pairs.csv")
                else:
                    candidates = [
                        Path(p)
                        for p in default_pair_csv_candidates(
                            str(task_root), experiment["task_type"], split
                        )
                    ]
                selected = next((p for p in candidates if p.is_file()), None)
                if selected is not None or split == "train":
                    set_option(flag, selected or candidates[0])
            set_option("feature_dir", Path(experiment["artifact_dir"]).parent / "feature_cache")
        if "profile_cache_dir" in options:
            set_option(
                "profile_cache_dir", Path(experiment["artifact_dir"]).parent / "profile_cache"
            )
        return result

    def files_for_input(self, flag, path, experiment, workspace):
        if flag == "task_root":
            # An empty task root is intentional for routes whose inputs are all
            # explicit. A pinned declaration also ensures the working root exists.
            declaration = workspace / "input_declarations" / "task_root.json"
            atomic_json(declaration, {"policy": self.name, "task_type": experiment["task_type"]})
            files = {".web_input_root.json": declaration}
            pseudoseqs = path / "pseudoseqs_decoded.csv"
            if normalize_task_type(experiment["task_type"]) == "pep_hlai" and pseudoseqs.is_file():
                files[pseudoseqs.name] = pseudoseqs
            return files
        if flag == "data_dir":
            files = {}
            for directory in sorted(path.iterdir()):
                if (
                    directory.name not in {"train", "val", "test", "test_warm", "test_cold"}
                    or not directory.is_dir()
                ):
                    continue
                for name in ("seqs.fasta", "labels.csv", "pairs.csv"):
                    member = directory / name
                    if member.is_file():
                        files[f"{directory.name}/{name}"] = member
            for name in ("label_names.json", "metadata.json", "sampling_manifest.json"):
                member = path / name
                if member.is_file():
                    files[name] = member
            if not files:
                raise ValueError(f"required dataset splits are missing: {path}")
            return files
        if flag == "resume_embedding_bundle":
            return {
                name: p
                for name, p in tree_files(path).items()
                if name.startswith("resume/") or name == "embedding_encoder.pt"
            }
        if flag.endswith("_index_path") and path.is_file():
            files = tree_files(path.parent)
            retained = {p.resolve() for p in files.values()}
            # Profile indices may refer to matrix files outside their own
            # directory. Retain those exact bytes as dependencies too.
            for value in self._strings(json.loads(path.read_text(encoding="utf-8"))):
                try:
                    referenced = Path(value)
                    if not referenced.is_absolute():
                        referenced = path.parent / referenced
                    if not referenced.is_file():
                        continue
                    referenced = referenced.resolve()
                except (OSError, ValueError):
                    continue  # Scientific metadata strings need not be paths.
                if referenced not in retained:
                    tag = hashlib.sha256(str(referenced).encode()).hexdigest()[:16]
                    files[f"dependencies/{tag}/{referenced.name}"] = referenced
                    retained.add(referenced)
            return files
        if path.is_dir():
            return tree_files(path)
        files = {path.name: path}
        # Tokenizer and embedding cache metadata is part of an explicit resource.
        for sibling in (
            path.with_suffix(path.suffix + ".meta.json"),
            path.with_suffix(".meta.json"),
        ):
            if sibling.is_file():
                files[sibling.name] = sibling
        return files

    @classmethod
    def _strings(cls, value):
        if isinstance(value, dict):
            for item in value.values():
                yield from cls._strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from cls._strings(item)
        elif isinstance(value, str):
            yield value

    def rebase_input(self, repository, run_id, flag, path, files, record, view):
        if not flag.endswith("_index_path") or not path.is_file():
            return record
        index = view / path.name
        original = json.loads(index.read_text(encoding="utf-8"))
        # Keep relative manifest paths relative: ProfileStore rejects absolute
        # paths, and copied sibling resources already retain their layout.
        destinations = {str(source.resolve()): name for name, source in files.items()}

        def rebase(value):
            if isinstance(value, dict):
                return {key: rebase(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rebase(item) for item in value]
            if isinstance(value, str):
                try:
                    candidate = Path(value)
                    if not candidate.is_absolute():
                        candidate = path.parent / candidate
                    return destinations.get(str(candidate.resolve()), value)
                except (OSError, ValueError):
                    return value
            return value

        updated = rebase(original)
        if updated == original:
            return record
        repository.bind(run_id, f"original:{flag}", record["id"])
        atomic_json(index, updated)
        return repository.snapshot(
            record["kind"],
            tree_files(view),
            {
                "derived_from": record["id"],
                "transform": "rebase_profile_input_paths_v1",
            },
        )
