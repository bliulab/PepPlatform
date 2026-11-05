"""Read-only adapters for Web-owned, frozen evaluation evidence.

No trainer, inference service, fitting or artifact upgrade is invoked here.
"""

from __future__ import annotations
from pepplatform.hlai_input import hla_allele
from pepplatform.task_compat import normalize_task_type, LEGACY_HLAI_COLUMN


import csv
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from pepplatform.evaluators import EVALUATION_PROTOCOL_VERSION, OperatingPoint
from .ensemble_data import read_split

VERSION = 1
KINDS = ("stratified", "labels", "patterns")
METRICS = ("acc", "f1", "auc", "aupr", "count", "prevalence")


class HeatmapError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (OSError, ValueError) as exc:
        raise HeatmapError("evidence_missing") from exc


def contained(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or "experiment_paper" in path.parts:
        raise HeatmapError("source_forbidden")
    return path


def bundle(context):
    exp = context["experiment"]
    if exp.get("status") != "succeeded":
        raise HeatmapError("experiment_not_complete")
    return contained(exp["artifact_dir"], context["runtime_root"])


def sources(context):
    """Discover array names from ZIP headers only; never load score arrays."""
    root = bundle(context)
    result = {}
    ordinary = root / "evaluation_scores.npz"
    if ordinary.is_file():
        contained(ordinary, root)
        try:
            with zipfile.ZipFile(ordinary) as archive:
                names = set(archive.namelist())
        except (OSError, zipfile.BadZipFile) as exc:
            raise HeatmapError("evidence_invalid") from exc
        for split in ("validation", "test"):
            if {f"{split}_labels.npy", f"{split}_scores.npy"} <= names:
                result[split] = (ordinary, root / "evaluation.json", split + "_")
    evaluations = root / "evaluations"
    if evaluations.exists():
        contained(evaluations, root)
        for directory in sorted(evaluations.iterdir()):
            if directory.is_dir() and directory.name.startswith("test"):
                contained(directory, root)
                if (directory / "scores.npz").is_file() and (
                    directory / "evaluation.json"
                ).is_file():
                    result[directory.name] = (
                        directory / "scores.npz",
                        directory / "evaluation.json",
                        "",
                    )
    return result


def point_from(metadata, evaluation, width):
    left, right = metadata.get("operating_point"), evaluation.get("operating_point")
    if not left and not right:
        return None
    if not left or not right:
        raise HeatmapError("threshold_mismatch")
    if any(p.get("source") != "validation" or not p.get("objective") for p in (left, right)):
        raise HeatmapError("threshold_mismatch")
    try:
        a, b = OperatingPoint.from_dict(left), OperatingPoint.from_dict(right)
    except (TypeError, ValueError) as exc:
        raise HeatmapError("threshold_mismatch") from exc
    if (
        a != b
        or a.source != "validation"
        or len(a.values) != width
        or not np.isfinite(a.values).all()
    ):
        raise HeatmapError("threshold_mismatch")
    labels = metadata.get("label_names", [])
    if a.label_names and list(a.label_names) != labels:
        raise HeatmapError("label_mismatch")
    for record in (left, right):
        for key in ("values", "thresholds"):
            if key in record and list(record[key]) != list(a.values):
                raise HeatmapError("threshold_mismatch")
        if "threshold" in record and (len(a.values) != 1 or record["threshold"] != a.values[0]):
            raise HeatmapError("threshold_mismatch")
        if "threshold_by_label" in record and record["threshold_by_label"] != dict(
            zip(labels, a.values)
        ):
            raise HeatmapError("threshold_mismatch")
    return a


def evidence_identity(root, split, evaluation):
    if evaluation.get("data_identity"):
        return evaluation["data_identity"], None
    if split == "validation":
        path = root / "web_ensemble_provenance.json"
        if path.is_file():
            proof = read_json(contained(path, root))
            return proof.get("validation"), proof
        path = root / "ensemble.json"
        if path.is_file():
            return read_json(contained(path, root)).get("dataset_identity", {}).get(
                "validation"
            ), None
    return None, None


def capabilities(context):
    try:
        root = bundle(context)
        metadata = read_json(contained(root / "metadata.json", root))
        labels = metadata.get("label_names") or []
        if (
            metadata.get("evaluation_protocol_version") != EVALUATION_PROTOCOL_VERSION
            or metadata.get("feature_fit_scope") != "train_only"
        ):
            raise HeatmapError("protocol_required")
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(x, str) or not x for x in labels)
            or len(set(labels)) != len(labels)
        ):
            raise HeatmapError("label_mismatch")
        options = []
        for split, (_scores, ep, _prefix) in sources(context).items():
            ev = read_json(contained(ep, root))
            if ev.get("evaluation_protocol_version") != EVALUATION_PROTOCOL_VERSION:
                raise HeatmapError("protocol_required")
            identity, _ = evidence_identity(root, split, ev)
            has_point = point_from(metadata, ev, len(labels)) is not None
            multi = (
                normalize_task_type(context["experiment"]["task_type"]) == "pep_function"
                and len(labels) > 1
            )
            group_reason = (
                "task_unsupported" if multi else ("" if identity else "alignment_missing")
            )
            if not group_reason:
                try:
                    dataset_root = Path(context["dataset"]["root_path"])
                    path = contained(
                        dataset_root / ("val" if split == "validation" else split) / "pairs.csv",
                        dataset_root,
                    )
                    with path.open(encoding="utf-8", newline="") as handle:
                        fields = set(csv.DictReader(handle).fieldnames or [])
                    task = context["experiment"]["task_type"]
                    valid = bool(fields & {"peptide", "pep"}) and (
                        (
                            normalize_task_type(task) == "pep_hlai"
                            and bool(fields & {"hla", "allele", LEGACY_HLAI_COLUMN})
                        )
                        or (
                            normalize_task_type(task) == "pep_hlaii"
                            and ("hla_pair" in fields or {"hla_alpha", "hla_beta"} <= fields)
                        )
                        or (
                            normalize_task_type(task) == "pep_protein_interaction"
                            and "protein" in fields
                        )
                    )
                    if not valid:
                        group_reason = "group_missing"
                except (OSError, HeatmapError):
                    group_reason = "group_missing"
            options.append(
                {
                    "key": split,
                    "threshold_available": has_point,
                    "types": [
                        {
                            "kind": "stratified",
                            "available": not group_reason,
                            "reason": group_reason,
                        },
                        {
                            "kind": "labels",
                            "available": multi,
                            "reason": "" if multi else "task_unsupported",
                        },
                        {
                            "kind": "patterns",
                            "available": multi,
                            "reason": "" if multi else "task_unsupported",
                        },
                    ],
                }
            )
        return {
            "version": VERSION,
            "splits": options,
            "label_names": labels,
            "task_type": context["experiment"]["task_type"],
            "reason": "" if options else "evidence_missing",
        }
    except HeatmapError as exc:
        return {"version": VERSION, "splits": [], "label_names": [], "reason": exc.code}


def normalize_params(value):
    if not isinstance(value, dict) or set(value) - {"split", "kind", "metric", "pattern", "scale"}:
        raise HeatmapError("parameters_invalid")
    split, kind = value.get("split"), value.get("kind")
    if not isinstance(split, str) or not split or Path(split).name != split or kind not in KINDS:
        raise HeatmapError("parameters_invalid")
    result = {"split": split, "kind": kind}
    if kind == "stratified":
        metric = value.get("metric", "aupr")
        if metric not in METRICS:
            raise HeatmapError("parameters_invalid")
        result["metric"] = metric
    if kind == "patterns":
        pattern, scale = value.get("pattern", "truth"), value.get("scale", "count")
        if pattern not in (
            "truth",
            "prediction",
            "false_negative",
            "false_positive",
        ) or scale not in ("count", "jaccard"):
            raise HeatmapError("parameters_invalid")
        result.update(pattern=pattern, scale=scale)
    return result


def input_files(context, params):
    root = bundle(context)
    selected = sources(context).get(params["split"])
    if not selected:
        raise HeatmapError("split_unavailable")
    paths = [root / "metadata.json", *selected[:2]]
    for name in ("web_ensemble_provenance.json", "ensemble.json"):
        if (root / name).is_file():
            paths.append(root / name)
    result = {str(contained(path, root)): digest(contained(path, root)) for path in paths}
    if params["kind"] == "stratified":
        ev = read_json(selected[1])
        identity, _ = evidence_identity(root, params["split"], ev)
        if not identity or not identity.get("files"):
            raise HeatmapError("alignment_missing")
        dataset = context["dataset"]
        data_root = Path(dataset["root_path"]).resolve()
        # read_split may include the registered HLA-I mapping in its source root.
        source_root = Path(
            (dataset.get("metadata") or {}).get("source_root") or data_root.parent
        ).resolve()
        for name, expected in identity["files"].items():
            path = Path(name).resolve()
            if not path.is_relative_to(data_root):
                if path != source_root / "pseudoseqs_decoded.csv":
                    raise HeatmapError("source_forbidden")
            contained(path, data_root if path.is_relative_to(data_root) else source_root)
            if not path.is_file() or digest(path) != expected:
                raise HeatmapError("dataset_changed")
            result[str(path)] = expected
    return result


def load_evidence(context, params):
    root = bundle(context)
    sp, ep, prefix = sources(context)[params["split"]]
    metadata, evaluation = read_json(root / "metadata.json"), read_json(ep)
    if (
        metadata.get("evaluation_protocol_version") != EVALUATION_PROTOCOL_VERSION
        or evaluation.get("evaluation_protocol_version") != EVALUATION_PROTOCOL_VERSION
    ):
        raise HeatmapError("protocol_required")
    if metadata.get("feature_fit_scope") != "train_only":
        raise HeatmapError("protocol_required")
    with np.load(contained(sp, root), allow_pickle=False) as arrays:
        y = np.asarray(arrays[prefix + "labels"], dtype=np.float64)
        scores = np.asarray(arrays[prefix + "scores"], dtype=np.float64)
        ids = (
            arrays[prefix + "sample_ids"].astype(str).tolist()
            if prefix + "sample_ids" in arrays
            else None
        )
    if y.ndim == 1:
        y = y[:, None]
    if scores.ndim == 1:
        scores = scores[:, None]
    if (
        y.ndim != 2
        or scores.shape != y.shape
        or not y.size
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(scores).all()
    ):
        raise HeatmapError("scores_invalid")
    labels = metadata.get("label_names")
    if (
        not isinstance(labels, list)
        or len(labels) != y.shape[1]
        or any(not isinstance(x, str) or not x for x in labels)
        or len(set(labels)) != len(labels)
    ):
        raise HeatmapError("label_mismatch")
    point = point_from(metadata, evaluation, y.shape[1])
    if params["kind"] != "stratified" and (
        normalize_task_type(context["experiment"]["task_type"]) != "pep_function" or len(labels) < 2
    ):
        raise HeatmapError("task_unsupported")
    if params["kind"] == "patterns" and params["pattern"] != "truth" and point is None:
        raise HeatmapError("threshold_missing")
    if params["kind"] == "stratified" and params["metric"] in ("acc", "f1") and point is None:
        raise HeatmapError("threshold_missing")
    return y.astype(np.int64), scores, labels, point, ids, evaluation


def strata(context, params, y, ids, evaluation):
    if y.shape[1] != 1:
        raise HeatmapError("task_unsupported")
    root, dataset = bundle(context), context["dataset"]
    identity, proof = evidence_identity(root, params["split"], evaluation)
    if not identity:
        raise HeatmapError("alignment_missing")
    split = "val" if params["split"] == "validation" else params["split"]
    current = read_split(dataset, split)
    if current["identity"] != identity:
        raise HeatmapError("dataset_changed")
    if not np.array_equal(current["y"].reshape(y.shape), y):
        raise HeatmapError("alignment_invalid")
    if ids is not None:
        if len(set(ids)) != len(ids) or ids != identity["sample_ids"]:
            raise HeatmapError("alignment_invalid")
    else:
        # Historical arrays cannot be joined by matching row count or labels.
        # Only the Web runner's pre/post identity bound to these exact score bytes
        # is accepted, for exporters whose fixed-order output has been verified.
        metadata = read_json(root / "metadata.json")
        supported = {
            "pair_vector_classical",
            "pair_embedding_classical",
            "pair_vector_nn",
            "pair_sequence_nn",
        }
        if (
            not proof
            or proof.get("dataset_id") != dataset["id"]
            or normalize_task_type(proof.get("task_type"))
            != normalize_task_type(dataset["task_type"])
        ):
            raise HeatmapError("alignment_missing")
        if metadata.get("bundle_type") not in supported or proof.get(
            "evaluation_scores_sha256"
        ) != digest(root / "evaluation_scores.npz"):
            raise HeatmapError("alignment_missing")
    data_root = Path(dataset["root_path"]).resolve()
    path = contained(data_root / split / "pairs.csv", data_root)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(y):
        raise HeatmapError("alignment_invalid")
    result = []
    task = context["experiment"]["task_type"]
    for row, pair in zip(rows, current["inputs"]["pairs"]):
        if normalize_task_type(task) == "pep_hlai":
            group = str(row.get("allele") or hla_allele(row) or "").strip()
            if "*" not in group:
                raise HeatmapError("group_missing")
        elif normalize_task_type(task) == "pep_hlaii":
            alpha, beta = row.get("hla_alpha", "").strip(), row.get("hla_beta", "").strip()
            group = str(
                row.get("hla_pair") or (f"{alpha}|{beta}" if alpha and beta else "")
            ).strip()
            if not group:
                raise HeatmapError("group_missing")
        elif normalize_task_type(task) == "pep_protein_interaction":
            group = "protein:" + hashlib.sha256(pair["protein"].encode()).hexdigest()
        else:
            raise HeatmapError("task_unsupported")
        result.append((group, len(pair["peptide"])))
    return result
