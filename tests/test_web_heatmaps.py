"""Read-only, frozen-score statistics and Web task lifecycle contracts."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from pepplatform.evaluators import OperatingPoint
from pepplatform.web.ensemble_data import read_split
from pepplatform.web.heatmap_compute import compute
from pepplatform.web.heatmap_data import (
    HeatmapError,
    capabilities,
    digest,
    input_files,
    normalize_params,
)
from pepplatform.web.heatmap_jobs import HeatmapJobs, atomic_json, csv_export, page_result
from pepplatform.web.heatmap_routes import register_heatmap_routes


def evidence(tmp_path, task="pep_function", split="validation", ids=True):
    root = tmp_path / "runtime"
    artifact = root / "experiments" / "one" / "artifact"
    artifact.mkdir(parents=True)
    dataset = {
        "id": "data",
        "root_path": str(root / "datasets" / "data"),
        "task_type": task,
        "format": "sequence_dir" if task == "pep_function" else "pair_split_dir",
        "metadata": {},
    }
    y = (
        np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
        if task == "pep_function"
        else np.array([[0], [1], [0], [1]])
    )
    scores = (
        np.array([[0.6, 0.8, 0.1], [0.9, 0.6, 0.1], [0.2, 0.4, 0.1], [0.1, 0.2, 0.1]])
        if task == "pep_function"
        else np.array([[0.1], [0.9], [0.2], [0.8]])
    )
    labels = ["AMP", "CPP", "EMPTY"] if task == "pep_function" else ["interaction"]
    point = OperatingPoint(
        values=(0.7, 0.5, 0.5) if task == "pep_function" else (0.5,), label_names=tuple(labels)
    ).to_dict()
    metadata = {
        "bundle_type": "pair_vector_classical",
        "task_type": task,
        "label_names": labels,
        "evaluation_protocol_version": 3,
        "feature_fit_scope": "train_only",
        "operating_point": point,
    }
    atomic_json(artifact / "metadata.json", metadata)
    (artifact / "model.bin").write_bytes(b"frozen model; analysis must not read or write this")
    ev = {
        "evaluation_protocol_version": 3,
        "feature_fit_scope": "train_only",
        "operating_point": point,
    }
    data_dir = Path(dataset["root_path"]) / ("val" if split == "validation" else split)
    data_dir.mkdir(parents=True)
    if task == "pep_function":
        (data_dir / "seqs.fasta").write_text(">a\nAAAA\n>b\nCCCC\n>c\nDDDD\n>d\nEEEE\n")
        (data_dir / "labels.csv").write_text("AMP,CPP,EMPTY\n1,1,0\n1,0,0\n0,1,0\n0,0,0\n")
    else:
        (data_dir / "pairs.csv").write_text(
            "sample_id,peptide,protein,label,mhc,hla_alpha,hla_beta,hla_pair\n"
            + "".join(
                f"{i},{'ACDE' if i < 2 else 'ACDEF'},ACDE,{i % 2},HLA-A*02:01,DPA1*01:03,DPB1*02:01,DPA1*01:03|DPB1*02:01\n"
                for i in range(4)
            )
        )
        if task == "pep_hlai":
            (Path(dataset["root_path"]).parent / "pseudoseqs_decoded.csv").write_text(
                "mhc,pseudoseq\nHLA-A*02:01,ACDE\n"
            )
    identity = read_split(dataset, "val" if split == "validation" else split)["identity"]
    if split == "validation":
        arrays = {"validation_labels": y, "validation_scores": scores}
        if ids:
            arrays["validation_sample_ids"] = np.asarray(identity["sample_ids"])
        np.savez_compressed(artifact / "evaluation_scores.npz", **arrays)
        atomic_json(artifact / "evaluation.json", ev)
        atomic_json(
            artifact / "web_ensemble_provenance.json",
            {
                "dataset_id": "data",
                "task_type": task,
                "validation": identity,
                "evaluation_scores_sha256": digest(artifact / "evaluation_scores.npz"),
            },
        )
    else:
        destination = artifact / "evaluations" / split
        destination.mkdir(parents=True)
        np.savez_compressed(
            destination / "scores.npz",
            labels=y,
            scores=scores,
            sample_ids=np.asarray(identity["sample_ids"]),
        )
        atomic_json(destination / "evaluation.json", {**ev, "data_identity": identity})
    context = {
        "runtime_root": str(root),
        "experiment": {
            "id": "one",
            "status": "succeeded",
            "artifact_dir": str(artifact),
            "task_type": task,
            "dataset_id": "data",
            "owner_user_id": "alice",
        },
        "dataset": dataset,
    }
    return context, artifact, y, scores


def params(kind="labels", **extra):
    return normalize_params({"split": "validation", "kind": kind, **extra})


def test_label_metrics_use_original_threshold_and_missing_class(tmp_path, monkeypatch):
    context, artifact, y, scores = evidence(tmp_path)
    before = {p.name: digest(p) for p in artifact.iterdir()}
    from pepplatform.evaluators import Evaluator

    monkeypatch.setattr(
        Evaluator, "fit_operating_point", lambda *a, **k: pytest.fail("must never fit")
    )
    result = compute(context, params())
    cell = {(c["row"], c["column"]): c for c in result["cells"]}
    assert cell["AMP", "f1"]["value"] == pytest.approx(2 / 3)  # .7, not .5
    assert cell["AMP", "acc"]["value"] == 0.75
    assert cell["EMPTY", "auc"]["value"] is None
    assert cell["EMPTY", "aupr"]["reason"] == "single_class"
    assert cell["EMPTY", "f1"]["value"] == 0
    assert before == {p.name: digest(p) for p in artifact.iterdir()}


@pytest.mark.parametrize(
    "pattern,intersection",
    [("truth", 1), ("prediction", 1), ("false_negative", 0), ("false_positive", 0)],
)
def test_four_event_matrices_and_empty_jaccard(tmp_path, pattern, intersection):
    context, _, _, _ = evidence(tmp_path)
    result = compute(context, params("patterns", pattern=pattern))
    cell = next(c for c in result["cells"] if c["row"] == "AMP" and c["column"] == "CPP")
    assert cell["value"] == intersection
    normalized = compute(context, params("patterns", pattern=pattern, scale="jaccard"))
    empty = next(c for c in normalized["cells"] if c["row"] == c["column"] == "EMPTY")
    assert empty["value"] is None and empty["reason"] == "empty_union"


def test_joint_errors_and_jaccard_denominator(tmp_path):
    context, artifact, y, scores = evidence(tmp_path)
    scores[0, :2] = 0.1  # Both labels missed on the same sample.
    scores[3, :2] = 0.9  # Both labels falsely predicted on the same sample.
    np.savez_compressed(
        artifact / "evaluation_scores.npz", validation_labels=y, validation_scores=scores
    )
    for pattern in ("false_negative", "false_positive"):
        result = compute(context, params("patterns", pattern=pattern))
        assert (
            next(c for c in result["cells"] if c["row"] == "AMP" and c["column"] == "CPP")["value"]
            == 1
        )
    result = compute(context, params("patterns", scale="jaccard"))
    cell = next(c for c in result["cells"] if c["row"] == "AMP" and c["column"] == "CPP")
    assert cell["value"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("task", ["pep_hlai", "pep_hlaii", "pep_protein_interaction"])
@pytest.mark.parametrize("split", ["validation", "test_warm", "test_cold"])
def test_stratified_templates_and_named_splits(tmp_path, task, split):
    context, _, _, _ = evidence(tmp_path, task, split)
    request = params("stratified", split=split)
    input_files(context, request)
    result = compute(context, request)
    assert result["columns"] == ["4", "5"]
    assert all(c["value"] == 1 and c["count"] == 2 and c["low_support"] for c in result["cells"])
    if task == "pep_hlaii":
        assert result["rows"] == ["DPA1*01:03|DPB1*02:01"]
    if task == "pep_protein_interaction":
        assert result["rows"][0].startswith("protein:") and len(result["rows"][0]) == 72


def test_capabilities_does_not_read_any_score_array(tmp_path, monkeypatch):
    context, _, _, _ = evidence(tmp_path)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("capabilities loaded scores"))
    result = capabilities(context)
    assert result["splits"][0]["key"] == "validation"


def test_missing_ids_require_preexisting_bound_provenance(tmp_path):
    context, artifact, _, _ = evidence(tmp_path, "pep_protein_interaction", ids=False)
    assert compute(context, params("stratified"))["cells"]
    (artifact / "web_ensemble_provenance.json").unlink()
    with pytest.raises(HeatmapError, match="alignment_missing"):
        compute(context, params("stratified"))


@pytest.mark.parametrize("mode", ["reorder", "duplicate", "labels", "nan"])
def test_bad_prediction_evidence_fails_closed(tmp_path, mode):
    context, artifact, y, scores = evidence(tmp_path, "pep_protein_interaction")
    path = artifact / "evaluation_scores.npz"
    with np.load(path) as arrays:
        values = dict(arrays)
    if mode == "reorder":
        values["validation_sample_ids"] = values["validation_sample_ids"][::-1]
    if mode == "duplicate":
        values["validation_sample_ids"][1] = values["validation_sample_ids"][0]
    if mode == "labels":
        values["validation_labels"] = 1 - y
    if mode == "nan":
        values["validation_scores"][0, 0] = np.nan
    np.savez_compressed(path, **values)
    with pytest.raises(HeatmapError):
        compute(context, params("stratified"))


def test_changed_data_and_label_order_are_rejected(tmp_path):
    context, artifact, _, _ = evidence(tmp_path, "pep_hlaii")
    path = Path(context["dataset"]["root_path"]) / "val" / "pairs.csv"
    path.write_text(path.read_text().replace("DPA1*01:03", "DPA1*02:01"))
    with pytest.raises(HeatmapError, match="dataset_changed"):
        input_files(context, params("stratified"))
    meta = json.loads((artifact / "metadata.json").read_text())
    meta["label_names"] = ["wrong"]
    atomic_json(artifact / "metadata.json", meta)
    with pytest.raises(HeatmapError, match="label_mismatch"):
        compute(context, params("stratified"))


def test_missing_threshold_only_disables_dependent_cells(tmp_path):
    context, artifact, _, _ = evidence(tmp_path)
    for name in ("metadata.json", "evaluation.json"):
        value = json.loads((artifact / name).read_text())
        value.pop("operating_point")
        atomic_json(artifact / name, value)
    result = compute(context, params())
    assert all(c["value"] is None for c in result["cells"] if c["column"] == "f1")
    assert any(c["value"] is not None for c in result["cells"] if c["column"] == "auc")
    assert compute(context, params("patterns"))["cells"]
    with pytest.raises(HeatmapError, match="threshold_missing"):
        compute(context, params("patterns", pattern="prediction"))


def test_threshold_alias_conflict_is_not_silently_ignored(tmp_path):
    context, artifact, _, _ = evidence(tmp_path)
    value = json.loads((artifact / "metadata.json").read_text())
    value["operating_point"]["thresholds"][0] = 0.5
    atomic_json(artifact / "metadata.json", value)
    with pytest.raises(HeatmapError, match="threshold_mismatch"):
        compute(context, params())


def test_capabilities_disables_missing_group_fields(tmp_path):
    context, _, _, _ = evidence(tmp_path, "pep_hlaii")
    path = Path(context["dataset"]["root_path"]) / "val" / "pairs.csv"
    path.write_text("sample_id,peptide,protein,label\na,AAA,ACDE,1\n")
    cap = capabilities(context)
    option = cap["splits"][0]["types"][0]
    assert not option["available"] and option["reason"] == "group_missing"


def test_paths_params_and_paper_sources_rejected(tmp_path):
    context, artifact, _, _ = evidence(tmp_path)
    for bad in (
        {"kind": "labels", "split": "../test"},
        {"kind": "labels", "split": "test", "path": "/tmp/data"},
    ):
        with pytest.raises(HeatmapError):
            normalize_params(bad)
    context["experiment"]["artifact_dir"] = str(tmp_path / "experiment_paper" / "artifact")
    assert capabilities(context)["reason"] == "source_forbidden"


def test_symlink_escape_and_nonfinite_scores(tmp_path):
    context, artifact, _, _ = evidence(tmp_path)
    original = artifact / "evaluation_scores.npz"
    outside = tmp_path / "outside.npz"
    original.rename(outside)
    original.symlink_to(outside)
    with pytest.raises(HeatmapError, match="source_forbidden"):
        input_files(context, params())


def wait_for(jobs, job):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        current = jobs.get("one", job["id"])
        if current["status"] not in ("queued", "running"):
            return current
        time.sleep(0.05)
    pytest.fail("worker timed out")


def test_real_worker_cache_page_export_and_immutable_inputs(tmp_path):
    context, artifact, _, _ = evidence(tmp_path)
    before = {p.name: digest(p) for p in artifact.iterdir()}
    jobs = HeatmapJobs(context["runtime_root"])
    try:
        job = jobs.submit(context, params())
        assert jobs.submit(context, params())["id"] == job["id"]
        assert wait_for(jobs, job)["status"] == "succeeded"
        result = jobs.result("one", job["id"])
        assert page_result(result, search="AMP")["rows"] == ["AMP"]
        assert "per_label_binary_accuracy" == result["definitions"]["acc"]
        assert "AMP" in csv_export(result)
        assert jobs.submit(context, params())["status"] == "succeeded"
        (jobs.root / job["id"] / "result.json").unlink()
        regenerated = jobs.submit(context, params())
        assert regenerated["id"] == job["id"] and regenerated["status"] == "queued"
        assert wait_for(jobs, regenerated)["status"] == "succeeded"
        with pytest.raises(HeatmapError, match="job_missing"):
            jobs.get("other", job["id"])
        assert before == {p.name: digest(p) for p in artifact.iterdir()}
        meta = json.loads((artifact / "metadata.json").read_text())
        meta["changed"] = True
        atomic_json(artifact / "metadata.json", meta)
        with pytest.raises(HeatmapError, match="input_changed"):
            jobs.result("one", job["id"])
        assert jobs.submit(context, params())["id"] != job["id"]
    finally:
        jobs.stop()


def test_bounded_queue_cancel_and_restart(tmp_path, monkeypatch):
    context, _, _, _ = evidence(tmp_path)
    jobs = HeatmapJobs(context["runtime_root"])
    jobs.thread = object()  # Hold the queue before any process is launched.
    queued = []
    for i in range(10):
        context["experiment"]["id"] = f"exp{i}"
        queued.append(jobs.submit(context, params()))
    with pytest.raises(HeatmapError, match="queue_full"):
        context["experiment"]["id"] = "overflow"
        jobs.submit(context, params())
    assert jobs.cancel("exp0", queued[0]["id"])["status"] == "canceled"
    corrupt = jobs.root / ("a" * 64)
    corrupt.mkdir(exist_ok=True)
    (corrupt / "job.json").write_text("broken")
    restarted = HeatmapJobs(context["runtime_root"])
    assert restarted.get("exp1", queued[1]["id"])["status"] == "interrupted"
    jobs.thread = None
    jobs.stop()
    restarted.stop()


def test_running_cancel_terminates_only_owned_worker(tmp_path, monkeypatch):
    import threading

    context, _, _, _ = evidence(tmp_path)
    launched, terminated = threading.Event(), threading.Event()

    class Process:
        def __init__(self, *args, **kwargs):
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
            launched.set()

        def wait(self):
            assert terminated.wait(5)
            return -15

        def terminate(self):
            terminated.set()

    monkeypatch.setattr("pepplatform.web.heatmap_jobs.subprocess.Popen", Process)
    jobs = HeatmapJobs(context["runtime_root"])
    try:
        job = jobs.submit(context, params())
        assert launched.wait(5)
        assert jobs.cancel("one", job["id"])["status"] == "canceled"
        assert terminated.is_set()
    finally:
        jobs.stop()


def test_worker_rejects_inputs_modified_during_analysis(tmp_path, monkeypatch):
    from pepplatform.web import heatmap_worker

    context, artifact, _, _ = evidence(tmp_path)
    request = tmp_path / "request.json"
    atomic_json(
        request,
        {
            "context": context,
            "params": params(),
            "inputs": input_files(context, params()),
            "provenance": {},
        },
    )

    def mutate(*_):
        metadata = json.loads((artifact / "metadata.json").read_text())
        metadata["changed"] = True
        atomic_json(artifact / "metadata.json", metadata)
        return {}

    monkeypatch.setattr(heatmap_worker, "compute", mutate)
    assert heatmap_worker.run(request) == 1
    assert json.loads((tmp_path / "error.json").read_text())["code"] == "input_changed"
    assert not (tmp_path / "result.json").exists()


def test_paging_preserves_full_export_and_escapes_spreadsheet_formulas():
    result = {
        "rows": [f"={i}" for i in range(70)],
        "columns": [str(i) for i in range(60)],
        "cells": [
            {"row": f"={i}", "column": str(j), "value": 0.5, "count": 1}
            for i in range(70)
            for j in range(60)
        ],
        "params": params(),
        "experiment_id": "one",
    }
    page = page_result(result)
    assert len(page["rows"]) == len(page["columns"]) == 50
    assert len(page["cells"]) == 2500
    assert len(page_result(result, 50, 50)["cells"]) == 200
    csv = csv_export(result)
    assert len(csv.splitlines()) == 4201 and "'=69" in csv


def test_api_authentication_and_job_ownership(tmp_path):
    from types import SimpleNamespace

    context, _, _, _ = evidence(tmp_path)
    app = FastAPI()
    app.state.paths = SimpleNamespace(root=Path(context["runtime_root"]))
    app.state.store = SimpleNamespace(get_dataset=lambda _: context["dataset"])
    jobs = HeatmapJobs(context["runtime_root"])

    def current(request):
        name = request.headers.get("x-user")
        if not name:
            raise HTTPException(401)
        return name

    def require(experiment_id, user):
        if user != "alice":
            raise HTTPException(403)
        return {**context["experiment"], "id": experiment_id}

    register_heatmap_routes(app, jobs, current, require)
    try:
        with TestClient(app) as client:
            base = "/api/experiments/one/visualizations"
            assert client.get(base + "/capabilities").status_code == 401
            assert client.get(base + "/capabilities", headers={"x-user": "bob"}).status_code == 403
            assert (
                client.get(base + "/capabilities", headers={"x-user": "alice"}).status_code == 200
            )
            job = client.post(base + "/jobs", json=params(), headers={"x-user": "alice"}).json()
            assert (
                client.get(
                    f"/api/experiments/other/visualizations/jobs/{job['id']}",
                    headers={"x-user": "alice"},
                ).status_code
                == 404
            )
            assert (
                client.post(
                    base + "/jobs",
                    json={**params(), "path": "/etc/passwd"},
                    headers={"x-user": "alice"},
                ).status_code
                == 422
            )
    finally:
        jobs.stop()
