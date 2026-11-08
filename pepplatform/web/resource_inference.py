"""A supervised inference process; timeouts never leave a GPU thread running."""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from pepplatform.versioning.web import VersionedModelService

from .resource_policy import ResourceLimitError
from .resource_runner import uses_gpu


class RemoteInferenceError(RuntimeError):
    def __init__(self, status_code, detail):
        super().__init__(detail.get("message") or detail.get("code") or "inference failed")
        self.status_code, self.detail = status_code, detail


def model_uses_gpu(model, _depth=0):
    metadata = model.get("metadata") or {}
    if not uses_gpu({}):
        return False
    if _depth > 4:
        return True
    bundle = Path(model.get("bundle_path") or "")
    try:
        metadata_file = bundle / "metadata.json"
        if metadata_file.is_file():
            metadata = json.loads(metadata_file.read_text())
        if metadata.get("bundle_type") == "web_ensemble":
            manifest = json.loads((bundle / "ensemble.json").read_text())
            members = []
            for member in manifest["members"]:
                member_path = (bundle / member["snapshot_path"]).resolve()
                if not member_path.is_relative_to(bundle.resolve() / "members"):
                    return True
                members.append({"bundle_path": str(member_path)})
            return not members or any(model_uses_gpu(member, _depth + 1) for member in members)
    except (OSError, ValueError, KeyError, TypeError):
        return True
    # Unknown and neural bundles take the conservative shared slot. Classical
    # ensembles made entirely of CPU members can continue during GPU training.
    if metadata.get("bundle_type") not in {
        "sequence_classical",
        "pair_vector_classical",
        "pair_classical",
    }:
        return True
    text = json.dumps(metadata).lower()
    return any(
        token in text
        for token in ("esm2", "protbert", "prott5", "esmc", "esm3", "embedding_encoder.pt")
    )


class InferenceProcess:
    def __init__(self, store, paths, governor):
        self.store, self.paths, self.governor = store, paths, governor
        self._lock = threading.Lock()
        self._process = None
        self._events = None

    def close(self):
        process = self._process
        self._process = None
        if process:
            if process.poll() is None:
                if os.name != "nt":
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()

    def _start(self):
        if self._process and self._process.poll() is None:
            return
        self.close()
        env = dict(
            os.environ,
            PEP_RESOURCE_DATABASE_URL=self.store.database_url,
            PEP_RESOURCE_RUNTIME_ROOT=str(self.paths.root),
            PEP_RESOURCE_PARENT_PID=str(os.getpid()),
            PYTHONPATH=os.pathsep.join(
                filter(
                    None,
                    [
                        str(Path(__file__).resolve().parents[2]),
                        str(self.paths.project_root),
                        os.environ.get("PYTHONPATH"),
                    ],
                )
            ),
        )
        process = subprocess.Popen(
            [sys.executable, "-m", "pepplatform.web.resource_inference_worker"],
            cwd=self.paths.project_root
            if self.paths.project_root.is_dir()
            else Path(__file__).resolve().parents[2],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            start_new_session=os.name != "nt",
        )
        events = queue.Queue()
        self._process, self._events = process, events

        def read():
            try:
                for line in process.stdout:
                    try:
                        events.put(json.loads(line))
                    except ValueError:
                        continue
            except (OSError, ValueError):
                pass
            finally:
                events.put({"event": "exit"})

        threading.Thread(
            target=read, name="pepplatform-inference-process-output", daemon=True
        ).start()

    def predict(
        self, model_id, *, sequences=None, pairs=None, progress_callback=None, resource_limits=None
    ):
        if not self._lock.acquire(blocking=False):
            raise ResourceLimitError("inference_busy", retry_after=5)
        try:
            self._start()
            timeout = self.governor.policies.policy()["limits"]["inference_timeout_seconds"]
            deadline = time.monotonic() + timeout
            self._process.stdin.write(
                json.dumps(
                    {
                        "model_id": model_id,
                        "sequences": sequences,
                        "pairs": pairs,
                        "resource_limits": resource_limits,
                    }
                )
                + "\n"
            )
            self._process.stdin.flush()
            while True:
                if time.monotonic() >= deadline:
                    self.close()
                    raise ResourceLimitError("inference_timeout", status_code=504, limit=timeout)
                if self.governor.disk()["state"] == "critical":
                    self.close()
                    raise ResourceLimitError("disk_space_insufficient", status_code=507)
                try:
                    event = self._events.get(timeout=min(1, max(0.01, deadline - time.monotonic())))
                except queue.Empty:
                    continue
                if event["event"] == "stage" and progress_callback:
                    progress_callback(event["data"])
                elif event["event"] == "done":
                    return event["data"]
                elif event["event"] == "error":
                    raise RemoteInferenceError(event["status_code"], event["data"])
                elif event["event"] == "exit":
                    self.close()
                    raise ResourceLimitError("inference_worker_stopped", status_code=503)
        finally:
            self._lock.release()


class ManagedModelService(VersionedModelService):
    def __init__(self, store, versions, governor):
        super().__init__(store, versions)
        self.governor = governor
        self._gpu_worker = InferenceProcess(store, governor.paths, governor)
        self._cpu_worker = InferenceProcess(store, governor.paths, governor)

    def predict(
        self, model_id, *, sequences=None, pairs=None, progress_callback=None, resource_limits=None
    ):
        model = self.governor.store.get_model(model_id)
        if not model:
            raise KeyError(model_id)
        worker = self._gpu_worker if model_uses_gpu(model) else self._cpu_worker
        return worker.predict(
            model_id,
            sequences=sequences,
            pairs=pairs,
            progress_callback=progress_callback,
            resource_limits=resource_limits or self.governor.policies.quota(None),
        )

    def suspend_esm3_runtime(self, reason="training"):
        self._gpu_worker.close()
        super().suspend_esm3_runtime(reason)

    def resume_esm3_runtime(self):
        super().resume_esm3_runtime()

    def close(self):
        self._gpu_worker.close()
        self._cpu_worker.close()
        super().close()

    def cancel_prediction(self, model_id):
        model = self.governor.store.get_model(model_id)
        worker = self._gpu_worker if model and model_uses_gpu(model) else self._cpu_worker
        worker.close()
