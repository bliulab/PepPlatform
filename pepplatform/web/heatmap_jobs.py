"""Bounded, durable Web-only post-hoc jobs and immutable result caches."""

from __future__ import annotations

import csv
import hashlib
import io
from importlib.metadata import version
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from .heatmap_data import VERSION, HeatmapError, digest, input_files, normalize_params, read_json


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def implementation_digest():
    paths = [
        Path(__file__).with_name(name)
        for name in (
            "heatmap_data.py",
            "heatmap_compute.py",
            "heatmap_worker.py",
            "heatmap_jobs.py",
        )
    ]
    paths.append(Path(__file__).parents[1] / "evaluators.py")
    paths.append(Path(__file__).parents[1] / "pr_metrics.py")
    paths.append(Path(__file__).with_name("ensemble_data.py"))
    paths.append(Path(__file__).with_name("datasets.py"))
    return hashlib.sha256("".join(digest(p) for p in paths).encode()).hexdigest()


class HeatmapJobs:
    def __init__(self, runtime_root):
        self.root = Path(runtime_root) / "visualizations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.queue = deque()
        self.active = None
        self.process = None
        self.closed = False
        self.thread = None
        for path in self.root.glob("*/job.json"):
            try:
                job = read_json(path)
                if job.get("status") in ("queued", "running"):
                    job.update(status="interrupted", error="interrupted")
                    atomic_json(path, job)
            except (HeatmapError, OSError):
                # A damaged derived cache must never prevent the Web app starting.
                continue

    def _path(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{64}", job_id):
            raise HeatmapError("job_missing")
        return self.root / job_id

    def submit(self, context, raw):
        params = normalize_params(raw)
        inputs = input_files(context, params)
        provenance = {
            "analysis_version": VERSION,
            "implementation_sha256": implementation_digest(),
            "libraries": {name: version(name) for name in ("numpy", "scikit-learn")},
            "experiment_id": context["experiment"]["id"],
            "dataset_id": context["dataset"]["id"],
            "split": params["split"],
            "parameters": params,
            "sources": [{"file": Path(p).name, "sha256": sha} for p, sha in sorted(inputs.items())],
        }
        identity = {
            "provenance": provenance,
            "inputs": inputs,
            "owner": context["experiment"].get("owner_user_id"),
        }
        job_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if self.closed:
                raise HeatmapError("service_stopped")
            directory = self._path(job_id)
            if (directory / "job.json").exists():
                try:
                    previous = read_json(directory / "job.json")
                    if previous.get("status") in ("queued", "running"):
                        return previous
                    if previous.get("status") == "succeeded":
                        read_json(directory / "result.json")
                        return previous
                except HeatmapError:
                    pass
            if len(self.queue) >= 10:
                raise HeatmapError("queue_full")
            directory.mkdir(exist_ok=True)
            for name in ("error.json", "result.json"):
                (directory / name).unlink(missing_ok=True)
            job = {
                "id": job_id,
                "experiment_id": context["experiment"]["id"],
                "status": "queued",
                "params": params,
                "created_at": time.time(),
                "error": None,
            }
            atomic_json(
                directory / "request.json",
                {"context": context, "params": params, "inputs": inputs, "provenance": provenance},
            )
            atomic_json(directory / "job.json", job)
            self.queue.append(job_id)
            if self.thread is None:
                self.thread = threading.Thread(target=self._work, name="heatmap-jobs", daemon=True)
                self.thread.start()
            self.wake.set()
            return job

    def get(self, experiment_id, job_id):
        path = self._path(job_id) / "job.json"
        if not path.is_file():
            raise HeatmapError("job_missing")
        job = read_json(path)
        if job["experiment_id"] != experiment_id:
            raise HeatmapError("job_missing")
        return job

    def cancel(self, experiment_id, job_id):
        with self.lock:
            job = self.get(experiment_id, job_id)
            if job["status"] in ("queued", "running"):
                job.update(status="canceled", error="canceled")
                atomic_json(self._path(job_id) / "job.json", job)
                self.queue = deque(x for x in self.queue if x != job_id)
                if self.active == job_id and self.process:
                    self.process.terminate()  # Only the CPU worker created here.
            return job

    def _work(self):
        while True:
            self.wake.wait()
            with self.lock:
                if self.closed:
                    return
                if not self.queue:
                    self.wake.clear()
                    continue
                job_id = self.queue.popleft()
                directory = self._path(job_id)
                job = read_json(directory / "job.json")
                job["status"] = "running"
                atomic_json(directory / "job.json", job)
                self.active = job_id
                env = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
                try:
                    self.process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "pepplatform.web.heatmap_worker",
                            str(directory / "request.json"),
                        ],
                        env=env,
                        cwd=Path(__file__).resolve().parents[2],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError:
                    self.process = None
            code = self.process.wait() if self.process else 1
            with self.lock:
                job = read_json(directory / "job.json")
                if job["status"] == "running":
                    error_path = directory / "error.json"
                    error = (
                        read_json(error_path).get("code")
                        if error_path.exists()
                        else "analysis_failed"
                    )
                    success = code == 0 and (directory / "result.json").exists()
                    job.update(
                        status="succeeded" if success else "failed",
                        error=None if success else error,
                    )
                    atomic_json(directory / "job.json", job)
                self.active, self.process = None, None

    def result(self, experiment_id, job_id):
        job = self.get(experiment_id, job_id)
        if job["status"] != "succeeded":
            raise HeatmapError("result_not_ready")
        directory = self._path(job_id)
        request = read_json(directory / "request.json")
        if (
            input_files(request["context"], request["params"]) != request["inputs"]
            or implementation_digest() != request["provenance"]["implementation_sha256"]
        ):
            raise HeatmapError("input_changed")
        if request["provenance"].get("libraries") != {
            name: version(name) for name in ("numpy", "scikit-learn")
        }:
            raise HeatmapError("input_changed")
        return read_json(directory / "result.json")

    def stop(self):
        with self.lock:
            self.closed = True
            for job_id in list(self.queue) + ([self.active] if self.active else []):
                path = self._path(job_id) / "job.json"
                job = read_json(path)
                if job["status"] in ("queued", "running"):
                    job.update(status="interrupted", error="interrupted")
                    atomic_json(path, job)
            self.queue.clear()
            if self.process:
                self.process.terminate()
            self.wake.set()
        if self.thread:
            self.thread.join(timeout=5)


def page_result(result, row_offset=0, column_offset=0, search=""):
    rows = [r for r in result["rows"] if search.casefold() in r.casefold()]
    visible_rows, columns = (
        rows[row_offset : row_offset + 50],
        result["columns"][column_offset : column_offset + 50],
    )
    return {
        **result,
        "rows": visible_rows,
        "columns": columns,
        "cells": [
            c for c in result["cells"] if c["row"] in visible_rows and c["column"] in columns
        ],
        "row_total": len(rows),
        "column_total": len(result["columns"]),
        "row_offset": row_offset,
        "column_offset": column_offset,
        "search": search,
    }


def csv_export(result):
    handle = io.StringIO(newline="")
    fields = [
        "experiment_id",
        "split",
        "kind",
        "metric",
        "pattern",
        "scale",
        "row",
        "column",
        "value",
        "count",
        "positive",
        "negative",
        "prevalence",
        "intersection",
        "union",
        "row_support",
        "column_support",
        "reason",
        "low_support",
        "imbalanced_support",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for cell in result["cells"]:
        value = {"experiment_id": result["experiment_id"], **result["params"], **cell}
        # Prevent spreadsheet formula execution in user-controlled label names.
        value = {
            k: (
                "'" + v
                if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r"))
                else v
            )
            for k, v in value.items()
        }
        writer.writerow(value)
    return "\ufeff" + handle.getvalue()
