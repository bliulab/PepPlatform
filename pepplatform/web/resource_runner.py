"""Durable fair scheduling around the unchanged versioned training runner."""

import fcntl
import logging
import os
import queue
import signal
import sys
import threading
import time
from contextlib import nullcontext
from datetime import datetime

from sqlalchemy import delete, insert, select, update

from pepplatform.versioning.web import VersionedExperimentRunner

from .resource_governor import tree_size
from .resource_policy import (
    ResourceLimitError,
    dump,
    now_utc,
    quota_exempt,
    resource_executions,
    resource_pending_jobs,
)

logger = logging.getLogger(__name__)


def process_start(pid):
    try:
        # comm can contain spaces and parentheses; field 22 follows its final ')'.
        return open(f"/proc/{int(pid)}/stat", encoding="utf-8").read().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def uses_gpu(config):
    import torch

    # The existing training commands select their own device. A raw config key
    # that isn't forwarded to those commands must never bypass the GPU lease.
    return torch.cuda.is_available()


class GovernedExperimentRunner(VersionedExperimentRunner):
    def __init__(self, *args, governor, **kwargs):
        super().__init__(*args, **kwargs)
        self.governor = governor
        self._leader = None
        self._last_owner = None
        self._execution_quota = None
        self._gpu_execution = False
        before, after = self.before_experiment, self.after_experiment
        self.before_experiment = lambda: (
            before() if self._gpu_execution and callable(before) else None
        )
        self.after_experiment = lambda: after() if self._gpu_execution and callable(after) else None

    def start(self):
        if self.thread and self.thread.is_alive():
            return self
        self._leader = (self.paths.root / ".resource-scheduler.lock").open("a")
        try:
            fcntl.flock(self._leader, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._leader.close()
            self._leader = None
            return self  # The other dispatcher consumes the same durable queue.
        self._recover()
        self.governor.start()
        return super().start()

    def stop(self):
        super().stop()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=10)
        self.governor.stop()
        if self._leader and (not self.thread or not self.thread.is_alive()):
            fcntl.flock(self._leader, fcntl.LOCK_UN)
            self._leader.close()
            self._leader = None

    def _recover(self):
        from .store import experiments, models

        with self.store.atomic(), self.store.connect() as conn:
            recovering = set(
                conn.execute(
                    select(experiments.c.id).where(experiments.c.status == "running")
                ).scalars()
            )
            interrupted = (
                conn.execute(
                    select(resource_executions).where(resource_executions.c.status == "running")
                )
                .mappings()
                .all()
            )
            for row in interrupted:
                recovering.add(row["experiment_id"])
                # Only terminate a process whose recorded birth identity matches.
                if (
                    row["pid"]
                    and row["process_start"]
                    and process_start(row["pid"]) == row["process_start"]
                ):
                    try:
                        os.killpg(row["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                self.governor.account_execution(
                    row["experiment_id"],
                    finished=True,
                    at=datetime.fromisoformat(row["heartbeat_at"]),
                )
                self.store.update_experiment(
                    row["experiment_id"],
                    status="canceled",
                    error="Training was interrupted because the Web service restarted.",
                    finished_at=now_utc().isoformat(),
                )
            # Upgrade/restart recovery is performed once, by the leader only.
            conn.execute(
                update(experiments)
                .where(experiments.c.status == "running")
                .values(
                    status="canceled",
                    error="Training was interrupted because the Web service restarted.",
                    finished_at=now_utc().isoformat(),
                )
            )
            existing = set(conn.execute(select(resource_pending_jobs.c.experiment_id)).scalars())
            for identifier in conn.execute(
                select(experiments.c.id).where(
                    experiments.c.status == "queued",
                    ~select(models.c.id).where(models.c.experiment_id == experiments.c.id).exists(),
                )
            ).scalars():
                if identifier not in existing:
                    conn.execute(
                        insert(resource_pending_jobs).values(
                            experiment_id=identifier,
                            enqueued_at=now_utc().isoformat(),
                            policy_json=dump(
                                {
                                    **self.governor.policies.quota(
                                        self.store.get_experiment(identifier).get("owner_user_id")
                                    ),
                                    "snapshot_source": "upgrade",
                                }
                            ),
                        )
                    )
        # VersionRepository uses its own transactions. Finalize only after the
        # admission transaction commits, so SQLite never deadlocks on recovery.
        for identifier in recovering:
            experiment = self.store.get_experiment(identifier)
            if experiment:
                try:
                    self.version_service.finish(experiment)
                except Exception:
                    logger.exception("Could not finalize interrupted version run: %s", identifier)

    def cancel(self, experiment_id, reason=None):
        result = super().cancel(experiment_id, reason)
        if result:
            with self.store.connect() as conn:
                conn.execute(
                    delete(resource_pending_jobs).where(
                        resource_pending_jobs.c.experiment_id == experiment_id
                    )
                )
        return result

    def submit(self, experiment_id):
        with self.store.atomic(), self.store.connect() as conn:
            if not conn.execute(
                select(resource_pending_jobs.c.experiment_id).where(
                    resource_pending_jobs.c.experiment_id == experiment_id
                )
            ).first():
                conn.execute(
                    insert(resource_pending_jobs).values(
                        experiment_id=experiment_id,
                        enqueued_at=now_utc().isoformat(),
                        policy_json=dump(
                            {
                                **self.governor.policies.quota(
                                    self.store.get_experiment(experiment_id).get("owner_user_id")
                                ),
                                "snapshot_source": "admission",
                            }
                        ),
                    )
                )
            self.store.after_commit(lambda: self.pending.put(experiment_id))

    def _next(self):
        from .store import experiments

        with self.store.connect() as conn:
            rows = (
                conn.execute(
                    select(experiments.c.id, experiments.c.owner_user_id)
                    .join(
                        resource_pending_jobs,
                        resource_pending_jobs.c.experiment_id == experiments.c.id,
                    )
                    .where(experiments.c.status == "queued")
                    .order_by(resource_pending_jobs.c.enqueued_at, experiments.c.id)
                )
                .mappings()
                .all()
            )
        first_by_owner = {}
        for row in rows:
            first_by_owner.setdefault(row["owner_user_id"] or "", row["id"])
        if not first_by_owner:
            return None
        owners = sorted(first_by_owner)
        owner = next(
            (owner for owner in owners if self._last_owner is None or owner > self._last_owner),
            owners[0],
        )
        self._last_owner = owner
        return first_by_owner[owner]

    def _loop(self):
        try:
            while not self._stopped.is_set():
                identifier = self._next()
                if identifier:
                    try:
                        self._run_experiment(identifier)
                    except Exception as exc:
                        logger.exception("Governed experiment failed: %s", identifier)
                        self._record_preparation_failure(identifier, exc)
                    current = self.store.get_experiment(identifier)
                    if not current or current["status"] not in {"queued", "running"}:
                        with self.store.connect() as conn:
                            conn.execute(
                                delete(resource_pending_jobs).where(
                                    resource_pending_jobs.c.experiment_id == identifier
                                )
                            )
                        continue
                try:
                    self.pending.get(timeout=1)
                    self.pending.task_done()
                except queue.Empty:
                    pass
        finally:
            if self._leader:
                fcntl.flock(self._leader, fcntl.LOCK_UN)
                self._leader.close()
                self._leader = None

    def prepare_process_command(self, command, experiment):
        quota = self._execution_quota
        if not quota:
            return command
        return [
            sys.executable,
            "-m",
            "pepplatform.web.worker_limits",
            *(
                []
                if quota_exempt(quota)
                else ["--max-file-bytes", str(quota["max_model_file_bytes"])]
            ),
            "--parent-pid",
            str(os.getpid()),
            "--",
            *command,
        ]

    def _run_experiment(self, experiment_id):
        experiment = self.store.get_experiment(experiment_id)
        if not experiment or experiment["status"] != "queued":
            return
        self.governor.validate_model_sources(experiment["config"])
        try:
            self.governor.require_disk()
            lease = (
                self.governor.gpu(training=True)
                if uses_gpu(experiment["config"])
                else nullcontext()
            )
            with lease:
                self._gpu_execution = uses_gpu(experiment["config"])
                try:
                    self._execute(experiment)
                finally:
                    self._gpu_execution = False
        except ResourceLimitError as exc:
            # Capacity shortages leave the original configuration queued. Invalid
            # user policy is reported as a preparation failure, never downgraded.
            if exc.detail["code"] not in {"gpu_busy", "disk_space_insufficient"}:
                raise

    def _execute(self, experiment):
        quota = self.governor.start_execution(experiment)
        self._execution_quota = quota
        done = threading.Event()

        def watch():
            last_scan = time.monotonic()
            while not done.wait(1):
                try:
                    self.governor.account_execution(experiment["id"])
                    with self.store.connect() as conn:
                        row = (
                            conn.execute(
                                select(resource_executions).where(
                                    resource_executions.c.experiment_id == experiment["id"]
                                )
                            )
                            .mappings()
                            .one()
                        )
                    elapsed = (
                        now_utc() - datetime.fromisoformat(row["started_at"])
                    ).total_seconds()
                    reason = None
                    if not quota_exempt(quota) and elapsed >= quota["max_experiment_seconds"]:
                        reason = "max_experiment_seconds"
                    elif (
                        not quota_exempt(quota)
                        and experiment.get("owner_user_id")
                        and self.governor.policies.daily(experiment["owner_user_id"])[
                            "training_milliseconds"
                        ]
                        / 1000
                        >= quota["daily_training_seconds_limit"]
                    ):
                        reason = "daily_training_seconds_limit"
                    elif (
                        not quota_exempt(quota)
                        and tree_size(experiment["artifact_dir"]) > quota["max_model_bundle_bytes"]
                    ):
                        reason = "max_model_bundle_bytes"
                    elif (
                        not quota_exempt(quota)
                        and tree_size(experiment["log_path"]) > quota["max_model_file_bytes"]
                    ):
                        reason = "max_model_file_bytes"
                    elif self.governor.disk()["state"] == "critical":
                        reason = "disk_space_critical"
                    if (
                        not reason
                        and not quota_exempt(quota)
                        and experiment.get("owner_user_id")
                        and time.monotonic() - last_scan >= 10
                    ):
                        self.governor.scan()
                        last_scan = time.monotonic()
                        if (
                            self.governor.storage_usage(experiment["owner_user_id"])["used_bytes"]
                            > quota["storage_limit_bytes"]
                        ):
                            reason = "storage_limit_bytes"
                    process = self._active_processes.get(experiment["id"])
                    if process and not row["pid"]:
                        with self.store.connect() as conn:
                            conn.execute(
                                update(resource_executions)
                                .where(resource_executions.c.experiment_id == experiment["id"])
                                .values(pid=process.pid, process_start=process_start(process.pid))
                            )
                    if reason:
                        self.cancel(
                            experiment["id"],
                            reason=f"Web resource policy stopped this incomplete experiment: {reason}",
                        )
                        return
                except Exception:
                    logger.exception("Resource watchdog failed: %s", experiment["id"])
                    self.cancel(
                        experiment["id"],
                        reason="Web resource monitoring failed; the experiment is incomplete.",
                    )
                    return

        monitor = threading.Thread(
            target=watch, name="pepplatform-training-resource-watch", daemon=True
        )
        monitor.start()
        try:
            user = (
                self.store.get_user(experiment["owner_user_id"])
                if experiment.get("owner_user_id")
                else None
            )
            budget = (
                self.versions.write_budget(user, quota)
                if hasattr(self.versions, "write_budget")
                else nullcontext()
            )
            with budget:
                super()._run_experiment(experiment["id"])
        finally:
            done.set()
            monitor.join(timeout=3)
            self.governor.account_execution(experiment["id"], finished=True)
            self._execution_quota = None
            self.governor.scan()

    def _register_model_if_present(self, experiment_id, experiment):
        quota = self._execution_quota
        if quota and not quota_exempt(quota):
            size = tree_size(experiment["artifact_dir"])
            if size > quota["max_model_bundle_bytes"]:
                raise ResourceLimitError(
                    "model_bundle_too_large", limit=quota["max_model_bundle_bytes"], used=size
                )
        record = super()._register_model_if_present(experiment_id, experiment)
        if self._is_canceled(experiment_id):
            if record:
                self.store.delete_model(record["id"])
            raise ResourceLimitError("experiment_canceled_during_publication")
        return record
