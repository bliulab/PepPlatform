"""Small, standard-library-only primitives for public dataset artifacts."""

from __future__ import annotations
from pepplatform.task_compat import normalize_task_type

import hashlib
import json
from pathlib import Path

VERSION = 1
TASKS = {
    "pep_function": "dbaasp",
    "pep_protein_interaction": "intact",
    "pep_hlai": "iedb",
    "pep_hlaii": "iedb",
}
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


class DataError(ValueError):
    """An invalid recipe, source response, or incomplete data artifact."""


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    Path(path).write_bytes(canonical(value))


def new_directory(path):
    path = Path(path).expanduser().resolve()
    # Artifacts must live outside source checkouts and all historical data trees.
    if any(
        p.name in {"experiment_paper", "datasets", ".pepplatform_web", ".git"}
        for p in (path, *path.parents)
    ):
        raise DataError("output is inside a protected data/experiment directory")
    if any((p / ".git").exists() for p in (path, *path.parents)):
        raise DataError("output must be outside a Git checkout (use a separate data directory)")
    if any(
        (p / "snapshot.json").is_file() or (p / "manifest.json").is_file() for p in path.parents
    ):
        raise DataError("output cannot be nested inside an existing snapshot or artifact")
    if path.exists():
        raise DataError(f"output already exists; choose a new directory: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def file_inventory(root, exclude=()):
    root = Path(root)
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise DataError("artifact symlinks are not allowed")
        if p.is_file() and p.relative_to(root).as_posix() not in exclude:
            result[p.relative_to(root).as_posix()] = sha(p)
    return result


def check_inventory(root, expected, exclude=()):
    if file_inventory(root, exclude) != expected:
        raise DataError("artifact inventory/hash mismatch (changed, missing, or extra file)")


def sequence(value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise DataError("missing_or_ambiguous_sequence")
    # Lower-case residues can encode D-amino acids; never erase that distinction.
    if not set(value) <= AA:
        raise DataError("nonstandard_or_modified_sequence")
    return value


def require_keys(value, allowed, required=(), context="recipe"):
    if not isinstance(value, dict):
        raise DataError(f"{context} must be an object")
    if set(value) - set(allowed) or set(required) - set(value):
        raise DataError(
            f"invalid {context} keys; unexpected={sorted(set(value) - set(allowed))}, "
            f"missing={sorted(set(required) - set(value))}"
        )


def load_recipe(path):
    r = read_json(path)
    require_keys(
        r,
        {
            "version",
            "task",
            "source",
            "query",
            "page_size",
            "max_records",
            "peptide_length",
            "labels",
            "resources",
            "split",
            "binding_methods",
        },
        {"version", "task", "source", "query", "peptide_length"},
    )
    if (
        type(r["version"]) is not int
        or r["version"] != VERSION
        or TASKS.get(normalize_task_type(r["task"])) != r["source"]
    ):
        raise DataError("unsupported recipe version/task/source")
    if not isinstance(r["query"], dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in r["query"].items()
    ):
        raise DataError("query must map strings to strings")
    forbidden = {"limit", "offset", "page", "pageSize", "select", "order", "format", "sort"}
    if forbidden & r["query"].keys():
        raise DataError("pagination, ordering and projection are managed by the fetcher")
    if r["source"] == "intact" and set(r["query"]) != {"query"}:
        raise DataError("IntAct requires exactly one query string")
    for key, default, upper in [("page_size", 100, 1000), ("max_records", 10000, 1000000)]:
        value = r.setdefault(key, default)
        if type(value) is not int or not 1 <= value <= upper:
            raise DataError(f"invalid {key}")
    bounds = r["peptide_length"]
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(type(x) is not int for x in bounds)
        or not 1 <= bounds[0] <= bounds[1]
    ):
        raise DataError("peptide_length must be [minimum, maximum]")
    split = r.setdefault("split", {"seed": 42, "ratios": [80, 10, 10]})
    require_keys(split, {"seed", "ratios"}, {"seed", "ratios"}, "split")
    if (
        type(split["seed"]) is not int
        or not isinstance(split["ratios"], list)
        or len(split["ratios"]) != 3
        or any(type(x) is not int or x <= 0 for x in split["ratios"])
        or sum(split["ratios"]) != 100
    ):
        raise DataError("split requires integer seed and three positive percentages totaling 100")
    resources = r.setdefault("resources", {})
    require_keys(resources, {"pseudosequences", "supplements", "benchmarks"}, context="resources")
    if any(not isinstance(v, str) or not v for v in resources.values()):
        raise DataError("resource values must be local JSON/CSV file paths")
    if normalize_task_type(r["task"]) in {"pep_hlai", "pep_hlaii"}:
        if "pseudosequences" not in resources:
            raise DataError("HLA recipes require a pseudosequences CSV resource")
        methods = r.get("binding_methods")
        if (
            not isinstance(methods, list)
            or not methods
            or not all(isinstance(x, str) and x for x in methods)
        ):
            raise DataError(
                "HLA recipes require explicit binding_methods (IEDB assay method values)"
            )
    if normalize_task_type(r["task"]) == "pep_function":
        labels = r.get("labels")
        if not isinstance(labels, list) or not labels:
            raise DataError("function recipe requires one or more label definitions")
        names = []
        import math

        for label in labels:
            require_keys(
                label,
                {"name", "target_species", "measure", "unit", "positive_max", "negative_min"},
                {"name", "target_species", "measure", "unit", "positive_max", "negative_min"},
                "label",
            )
            if not all(
                isinstance(label[k], str) and label[k]
                for k in ["name", "target_species", "measure", "unit"]
            ):
                raise DataError("label name, species, measure and unit must be explicit")
            if label["measure"] != "MIC":
                raise DataError("v1 function labels support MIC only (lower means more active)")
            if any(
                type(label[k]) not in {int, float} or not math.isfinite(label[k])
                for k in ["positive_max", "negative_min"]
            ):
                raise DataError("label thresholds must be finite numbers")
            if not 0 < label["positive_max"] < label["negative_min"]:
                raise DataError("require 0 < positive_max < negative_min")
            names.append(label["name"])
        if len(set(names)) != len(names):
            raise DataError("duplicate label names")
    elif "labels" in r:
        raise DataError("labels configuration is only supported for pep_function")
    if normalize_task_type(r["task"]) not in {"pep_hlai", "pep_hlaii"} and (
        "binding_methods" in r or "pseudosequences" in resources
    ):
        raise DataError("binding_methods/pseudosequences are only supported for HLA tasks")
    return r


def split_for(peptide, recipe):
    value = int(digest(f"{recipe['split']['seed']}:{peptide}".encode()), 16)
    # Integer comparison avoids floating point boundary differences.
    a, b, _ = recipe["split"]["ratios"]
    return (
        "train" if value * 100 < a * 2**256 else "val" if value * 100 < (a + b) * 2**256 else "test"
    )


def source_identity():
    """Include shared naming/column helpers in new fetch provenance."""
    directory = Path(__file__).parent
    result = {p.name: sha(p) for p in sorted(directory.glob("*.py"))}
    for helper in ("task_compat.py", "hlai_input.py"):
        if helper not in result:
            result[helper] = sha(directory.parent / helper)
    return result
