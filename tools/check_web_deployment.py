#!/usr/bin/env python3
"""Check a live Web deployment using a small synthetic CPU training run.

Set PEP_SMOKE_EMAIL and PEP_SMOKE_PASSWORD to an existing account. The first
invocation creates a dataset, experiment and model; --verify-existing checks
those same resources after a restart, upgrade or backup restore.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import math
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request
import uuid

from build_web_route_smoke_dataset import build_archive


class DeploymentCheck:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
        self.client = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )

    def request(self, path, payload=None, *, data=None, content_type=None):
        if payload is not None:
            data = json.dumps(payload).encode()
            content_type = "application/json"
        request = urllib.request.Request(self.base_url + path, data=data)
        if content_type:
            request.add_header("Content-Type", content_type)
        with self.client.open(request, timeout=60) as response:
            body = response.read()
            if "application/json" in response.headers.get("Content-Type", ""):
                return json.loads(body)
            return body.decode()

    def login(self):
        email = os.environ.get("PEP_SMOKE_EMAIL")
        password = os.environ.get("PEP_SMOKE_PASSWORD")
        if not email or not password:
            raise ValueError("set PEP_SMOKE_EMAIL and PEP_SMOKE_PASSWORD")
        if "<html" not in self.request("/").lower():
            raise ValueError("frontend did not return an HTML page")
        if self.request("/api/health").get("ok") is not True:
            raise ValueError("backend health check failed")
        try:
            self.request("/api/datasets")
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
        else:
            raise ValueError("authentication is disabled; use an authenticated deployment")
        self.request("/api/auth/login", {"email": email, "password": password})
        if not self.request("/api/auth/me").get("user"):
            raise ValueError("login did not establish a session")

    def upload(self, archive):
        boundary = uuid.uuid4().hex
        parts = []
        for name, value in [("task_type", "pep_function"), ("name", "Deployment smoke test")]:
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n".encode()
            )
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="archive"; '
            f'filename="smoke.zip"\r\nContent-Type: application/zip\r\n\r\n'.encode()
            + archive.read_bytes()
            + f"\r\n--{boundary}--\r\n".encode()
        )
        return self.request(
            "/api/datasets/upload",
            data=b"".join(parts),
            content_type=f"multipart/form-data; boundary={boundary}",
        )

    def train(self, timeout):
        with tempfile.TemporaryDirectory(prefix="pepplatform-smoke-") as directory:
            archive = build_archive(Path(directory) / "smoke.zip")
            dataset = self.upload(archive)
        experiment = self.request(
            "/api/experiments",
            {
                "name": "Deployment smoke test",
                "dataset_id": dataset["id"],
                "task_type": "pep_function",
                "config": {
                    "tokenizer": "kmer",
                    "encoder": "onehot",
                    "model": "cnn",
                    "epochs": 1,
                    "batch_size": 2,
                    "seed": 7,
                },
            },
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            experiment = self.request(f"/api/experiments/{experiment['id']}")
            if experiment["status"] == "succeeded":
                return {
                    "dataset_id": dataset["id"],
                    "experiment_id": experiment["id"],
                    "model_id": experiment["model_id"],
                }
            if experiment["status"] in {"failed", "canceled", "cancelled"}:
                raise ValueError(
                    f"experiment {experiment['id']} ended with status {experiment['status']}; "
                    "inspect its training log"
                )
            time.sleep(2)
        raise TimeoutError(f"experiment {experiment['id']} did not finish within {timeout}s")

    def verify(self, state):
        datasets = self.request("/api/datasets")
        if not any(item["id"] == state["dataset_id"] for item in datasets):
            raise ValueError("uploaded dataset was not retained")
        experiment = self.request(f"/api/experiments/{state['experiment_id']}")
        if experiment["status"] != "succeeded" or experiment["model_id"] != state["model_id"]:
            raise ValueError("completed experiment/model was not retained")
        prediction = self.request(
            f"/api/models/{state['model_id']}/predict",
            {"sequences": ["AAAAAAAAAAAA", "CCCCCCCCCCCC"]},
        )
        rows = prediction.get("predictions", [])
        if len(rows) != 2 or any(
            len(row.get("scores", [])) != 2
            or any(not math.isfinite(score) or not 0 <= score <= 1 for score in row["scores"])
            for row in rows
        ):
            raise ValueError("inference did not return two finite two-label probability rows")
        versions = self.request(f"/api/experiments/{state['experiment_id']}/versions")
        if versions.get("status") != "sealed":
            raise ValueError("trained model version was not sealed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not args.verify_existing and args.state.exists():
        parser.error("state file already exists; use --verify-existing or a new path")
    check = DeploymentCheck(args.base_url)
    check.login()
    if args.verify_existing:
        state = json.loads(args.state.read_text())
    else:
        state = check.train(args.timeout)
        # Retain resource IDs even if inference fails, so the run can be inspected.
        with args.state.open("x") as output:
            json.dump(state, output, indent=2)
    check.verify(state)
    print(json.dumps({"ok": True, **state}))


if __name__ == "__main__":
    main()
