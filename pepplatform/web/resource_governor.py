"""Admission, storage inventory and conservative retention for the Web service.

Retained inputs and pretrained assets are GC roots. Model availability is kept
in an independent administrative ledger, without rewriting content identities.
"""

import csv
import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import delete, insert, select, update

from pepplatform.versioning.core import VersionRepository, sha256

from .resource_policy import (
    ResourceLimitError,
    announcement_requested,
    check_quota,
    dump,
    model_protection,
    now_utc,
    quota_exempt,
    resource_cleanup_logs,
    resource_executions,
    resource_inventory,
    resource_leases,
    resource_pending_jobs,
    resource_version_owners,
    retired_model_versions,
)

logger = logging.getLogger(__name__)


class BudgetedVersionRepository(VersionRepository):
    """Keep snapshot identities/copy semantics intact; budget their Web writes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.governor = None
        self._owner = ContextVar(f"version_write_owner_{id(self)}", default=False)

    @contextmanager
    def write_budget(self, user, policy=None):
        token = self._owner.set(user)
        try:
            with self.governor.use_policy(policy) if self.governor else nullcontext():
                yield
        finally:
            self._owner.reset(token)

    @contextmanager
    def _budget(self, requested):
        owner = self._owner.get()
        if self.governor is None or owner is False:
            yield
            return
        self.governor.scan()
        with self.governor.reserve_write(owner, requested):
            yield

    def snapshot(self, kind, files, extra=None):
        requested = sum(Path(path).stat().st_size for path in files.values())
        with self._budget(requested):
            record = super().snapshot(kind, files, extra)
            if self.governor:
                with self.governor.store.connect() as conn:
                    conn.execute(
                        delete(retired_model_versions).where(
                            retired_model_versions.c.version_id == record["id"]
                        )
                    )
            owner = self._owner.get()
            if self.governor and owner:
                with self.governor.store.atomic(), self.governor.store.connect() as conn:
                    if not conn.execute(
                        select(resource_version_owners).where(
                            resource_version_owners.c.version_id == record["id"],
                            resource_version_owners.c.user_id == owner["id"],
                        )
                    ).first():
                        conn.execute(
                            insert(resource_version_owners).values(
                                version_id=record["id"], user_id=owner["id"]
                            )
                        )
            return record

    def verify(self, identifier):
        if self.governor and not self.object_path(identifier).exists():
            with self.governor.store.connect() as conn:
                retired = conn.execute(
                    select(retired_model_versions.c.version_id).where(
                        retired_model_versions.c.version_id == identifier
                    )
                ).first()
            if retired:
                raise ResourceLimitError("model_artifact_removed", status_code=410)
        return super().verify(identifier)

    def materialize(self, identifier, destination):
        requested = sum(
            member["bytes"] for member in self.get(identifier)["identity"]["files"].values()
        )
        with self._budget(requested):
            return super().materialize(identifier, destination)


def tree_size(path):
    """Never traverse directory symlinks or charge files outside a managed tree."""
    path = Path(path)
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    if path.is_dir():
        for root, directories, files in os.walk(path, followlinks=False):
            directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
            for name in files:
                member = Path(root) / name
                try:
                    if not member.is_symlink():
                        total += member.stat().st_size
                except FileNotFoundError:
                    pass
    return total


class ResourceGovernor:
    def __init__(self, store, paths, versions):
        self.store, self.paths, self.versions = store, paths, versions
        self.policies = store.resources
        self._admission_policy = ContextVar(f"resource_admission_policy_{id(self)}", default=None)
        self._scan_lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._last_scan = None
        self._gpu_metrics = (0.0, [])
        from .model_retention import ModelRetention

        self.models = ModelRetention(self)

    def gpu_metrics(self):
        """Device totals include allocations outside PyTorch's own allocator."""
        if time.monotonic() - self._gpu_metrics[0] < 15:
            return self._gpu_metrics[1]
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            selected = visible.split(",") if visible is not None else None
            devices = []
            for row in csv.reader(result.stdout.splitlines()):
                index, identifier, name, total, used, utilization = (value.strip() for value in row)
                if selected is not None and index not in selected and identifier not in selected:
                    continue
                devices.append(
                    {
                        "index": index,
                        "name": name,
                        "total_bytes": int(float(total) * 1024**2),
                        "used_bytes": int(float(used) * 1024**2),
                        "utilization_percent": float(utilization)
                        if utilization.replace(".", "", 1).isdigit()
                        else None,
                    }
                )
        except (OSError, subprocess.SubprocessError, ValueError):
            devices = []
        self._gpu_metrics = (time.monotonic(), devices)
        return devices

    def disk(self):
        usage = shutil.disk_usage(self.paths.root)
        limits = self.policies.policy()["limits"]
        reserve = max(
            limits["disk_reserve_bytes"],
            math.ceil(usage.total * limits["disk_reserve_percent"] / 100),
        )
        critical = max(
            limits["disk_critical_bytes"],
            math.ceil(usage.total * limits["disk_critical_percent"] / 100),
        )
        return {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "reserve_bytes": reserve,
            "critical_bytes": critical,
            "state": "critical"
            if usage.free < critical
            else "paused"
            if usage.free < reserve
            else "ready",
        }

    def require_disk(self, requested=0, *, critical=False):
        disk = self.disk()
        reserved = sum(
            lease["reserved_bytes"]
            for lease in self.policies.leases()
            if lease["kind"] != "allocated"
        )
        threshold = disk["critical_bytes"] if critical else disk["reserve_bytes"]
        if disk["free_bytes"] - reserved - requested < threshold:
            raise ResourceLimitError(
                "disk_space_insufficient",
                status_code=507,
                free_bytes=disk["free_bytes"],
                reserved_bytes=reserved,
                requested=requested,
                reserve_bytes=threshold,
            )

    def storage_usage(self, user_id=None):
        with self.store.connect() as conn:
            rows = (
                conn.execute(
                    select(resource_inventory).where(resource_inventory.c.state != "deleted")
                )
                .mappings()
                .all()
            )
        total, protected, reclaimable = 0, 0, 0
        for row in rows:
            if user_id and user_id not in json.loads(row["owners_json"]):
                continue
            total += row["size_bytes"]
            if row["protected"]:
                protected += row["size_bytes"]
            else:
                reclaimable += row["size_bytes"]
        reserved = sum(lease["reserved_bytes"] for lease in self.policies.leases(user_id=user_id))
        return {
            "used_bytes": total,
            "protected_bytes": protected,
            "reclaimable_bytes": reclaimable,
            "reserved_bytes": reserved,
            "scanned_at": self._last_scan.isoformat() if self._last_scan else None,
        }

    @contextmanager
    def use_policy(self, policy):
        token = self._admission_policy.set(policy)
        try:
            yield
        finally:
            self._admission_policy.reset(token)

    def execution_policy(self, user):
        policy = self._admission_policy.get()
        if policy is not None and policy.get("user_id") == (user or {}).get("id"):
            return policy
        return self.policies.quota((user or {}).get("id"))

    def require_storage(self, user, requested=0):
        if not user:
            return
        quota = self.execution_policy(user)
        if quota_exempt(quota):
            return
        # Only the initial inventory blocks a request; subsequent scans are batched
        # by the maintenance thread, never a full walk on each HTTP request.
        if self._last_scan is None:
            self.scan()
        usage = self.storage_usage(user["id"])
        check_quota(
            "storage_limit_bytes",
            quota["storage_limit_bytes"],
            usage["used_bytes"] + usage["reserved_bytes"],
            requested,
        )

    @contextmanager
    def reserve_write(self, user, requested, *, kind="write"):
        with self.store.atomic():
            self.require_disk(requested)
            self.require_storage(user, requested)
            lease = self.policies.create_lease(
                kind, user and user["id"], reserved_bytes=requested, seconds=3600
            )
        try:
            yield
        finally:
            # Keep a logical-space charge until the next inventory includes the
            # newly written files. Already allocated bytes aren't reserved twice
            # against the volume's current free-space probe.
            with self.store.atomic(), self.store.connect() as conn:
                conn.execute(
                    update(resource_leases)
                    .where(resource_leases.c.id == lease)
                    .values(
                        kind="allocated",
                        created_at=now_utc().isoformat(),
                    )
                )

    def admit_training(self, user, count):
        from .store import experiments

        with self.store.connect() as conn:
            queued = conn.execute(
                select(experiments.c.id).where(experiments.c.status == "queued")
            ).all()
        check_quota(
            "queue_capacity", self.policies.policy()["limits"]["queue_capacity"], len(queued), count
        )
        self.require_disk()
        if user:
            quota = self.policies.quota(user["id"])
            if quota_exempt(quota):
                return
            self.models.admit(user["id"], count)
            used = self.policies.daily(user["id"])["training_milliseconds"] / 1000
            check_quota(
                "daily_training_seconds_limit", quota["daily_training_seconds_limit"], used, 0.001
            )
            check_quota("max_experiment_seconds", quota["max_experiment_seconds"], 0, 1)
            check_quota("max_model_file_bytes", quota["max_model_file_bytes"], 0, 1)
            check_quota("max_model_bundle_bytes", quota["max_model_bundle_bytes"], 0, 1)
            self.require_storage(user)

    def validate_model_sources(self, config):
        """Reject a disallowed source; never replace a model or rewrite its revision."""
        allowed = self.policies.policy()["limits"]["allowed_pretrained_models"]
        from pepplatform.workflow import PRETRAINED_MODEL_FIELDS

        fields = set(PRETRAINED_MODEL_FIELDS.values()) | {"esm_model", "model_source"}

        def visit(value):
            if isinstance(value, dict):
                for key, member in value.items():
                    if (
                        (key in fields or key.endswith("_model_source"))
                        and isinstance(member, str)
                        and member.strip()
                    ):
                        source = member.strip()
                        path = Path(source).expanduser()
                        candidate = path if path.is_absolute() else self.paths.project_root / path
                        if source not in allowed and not candidate.exists():
                            raise ResourceLimitError(
                                "pretrained_model_not_allowed", status_code=403, model_source=source
                            )
                    visit(member)
            elif isinstance(value, list):
                for member in value:
                    visit(member)

        visit(config)

    def reserve_inference(self, user, input_count, *, model_id=None):
        with self.store.atomic():
            quota = self.policies.quota((user or {}).get("id"))
            if model_id:
                self.models.require_available(model_id)
            if user and not quota_exempt(quota):
                usage = self.store.get_quota_usage(user["id"])
                leases = [
                    lease
                    for lease in self.policies.leases(user_id=user["id"], kind="inference")
                    if lease["usage_date"] == usage["usage_date"]
                ]
                check_quota(
                    "daily_inference_request_limit",
                    quota["daily_inference_request_limit"],
                    usage["inference_requests"] + sum(row["request_count"] for row in leases),
                )
                check_quota(
                    "daily_inference_input_limit",
                    quota["daily_inference_input_limit"],
                    usage["inference_inputs"] + sum(row["input_count"] for row in leases),
                    input_count,
                )
            requested = 0
            if model_id and not (self.versions.root / "inference" / model_id).exists():
                model = self.store.get_model(model_id)
                if model and self.versions.run(model["experiment_id"])["status"] == "sealed":
                    requested = tree_size(model["bundle_path"])
                    self.require_disk(requested)
                    self.require_storage(user, requested)
            return self.policies.create_lease(
                "inference",
                user and user["id"],
                resource_id=model_id,
                reserved_bytes=requested,
                request_count=1,
                input_count=input_count,
                seconds=self.policies.policy()["limits"]["inference_timeout_seconds"] + 120,
                policy=quota,
            )

    def inference_policy(self, lease_id):
        with self.store.connect() as conn:
            value = conn.execute(
                select(resource_leases.c.policy_json).where(resource_leases.c.id == lease_id)
            ).scalar()
        if value is None:
            raise ResourceLimitError("inference_reservation_missing", status_code=409)
        return json.loads(value)

    def settle_inference(self, lease_id, *, success=False):
        with self.store.atomic(), self.store.connect() as conn:
            row = (
                conn.execute(select(resource_leases).where(resource_leases.c.id == lease_id))
                .mappings()
                .first()
            )
            if row and success and row["user_id"]:
                self.store.increment_quota_usage(
                    row["user_id"],
                    inference_requests=row["request_count"],
                    inference_inputs=row["input_count"],
                    usage_date=row["usage_date"],
                )
            if row and row["reserved_bytes"]:
                conn.execute(
                    update(resource_leases)
                    .where(resource_leases.c.id == lease_id)
                    .values(
                        kind="allocated",
                        request_count=0,
                        input_count=0,
                        created_at=now_utc().isoformat(),
                    )
                )
            else:
                conn.execute(delete(resource_leases).where(resource_leases.c.id == lease_id))

    def start_execution(self, experiment):
        user_id = experiment.get("owner_user_id")
        current = now_utc().isoformat()
        with self.store.atomic(), self.store.connect() as conn:
            snapshot = conn.execute(
                select(resource_pending_jobs.c.policy_json).where(
                    resource_pending_jobs.c.experiment_id == experiment["id"]
                )
            ).scalar()
            quota = json.loads(snapshot) if snapshot else self.policies.quota(user_id)
            # Admission already reserved capacity. Only platform disk/GPU protections
            # can revoke it after a quota change; the watchdog uses this snapshot.
            self.require_disk()
            conn.execute(
                delete(resource_executions).where(
                    resource_executions.c.experiment_id == experiment["id"]
                )
            )
            conn.execute(
                insert(resource_executions).values(
                    experiment_id=experiment["id"],
                    user_id=user_id,
                    status="running",
                    started_at=current,
                    accounted_at=current,
                    heartbeat_at=current,
                    policy_json=dump(quota),
                )
            )
        return quota

    def account_execution(self, experiment_id, *, finished=False, at=None):
        current = at or now_utc()
        with self.store.atomic(), self.store.connect() as conn:
            row = (
                conn.execute(
                    select(resource_executions).where(
                        resource_executions.c.experiment_id == experiment_id
                    )
                )
                .mappings()
                .first()
            )
            if not row or row["status"] != "running":
                return
            cursor = datetime.fromisoformat(row["accounted_at"])
            if current < cursor:
                return
            if row["user_id"]:
                while cursor < current:
                    midnight = (cursor + timedelta(days=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    end = min(midnight, current)
                    milliseconds = max(0, int((end - cursor).total_seconds() * 1000))
                    self.policies.add_daily(
                        row["user_id"], cursor.date().isoformat(), milliseconds=milliseconds
                    )
                    cursor = end
            conn.execute(
                update(resource_executions)
                .where(resource_executions.c.experiment_id == experiment_id)
                .values(
                    status="finished" if finished else "running",
                    accounted_at=current.isoformat(),
                    heartbeat_at=current.isoformat(),
                )
            )

    @contextmanager
    def gpu(self, *, training=False):
        """An interprocess device lease; taking it never changes a task's device."""
        if not training and self.disk()["state"] == "ready":
            from .resource_policy import resource_pending_jobs
            from .store import experiments

            with self.store.connect() as conn:
                waiting = conn.execute(
                    select(resource_pending_jobs.c.experiment_id)
                    .join(
                        experiments,
                        experiments.c.id == resource_pending_jobs.c.experiment_id,
                    )
                    .where(experiments.c.status == "queued")
                    .limit(1)
                ).first()
            if waiting:
                raise ResourceLimitError("gpu_reserved_for_training", retry_after=10)
        path = self.paths.root / ".gpu-execution.lock"
        with path.open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ResourceLimitError(
                    "gpu_busy_training" if not training else "gpu_busy", retry_after=10
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _entries(self):
        from pepplatform.versioning.core import bindings, records, runs

        from .store import datasets, experiments, models

        with self.store.connect() as conn:
            dataset_rows = [dict(row) for row in conn.execute(select(datasets)).mappings()]
            experiment_rows = [dict(row) for row in conn.execute(select(experiments)).mappings()]
            model_rows = [dict(row) for row in conn.execute(select(models)).mappings()]
            version_rows = [dict(row) for row in conn.execute(select(records)).mappings()]
            binding_rows = [dict(row) for row in conn.execute(select(bindings)).mappings()]
            run_rows = [dict(row) for row in conn.execute(select(runs)).mappings()]
            pins = {
                row["model_id"]: dict(row)
                for row in conn.execute(select(model_protection)).mappings()
            }
            owner_rows = conn.execute(select(resource_version_owners)).mappings().all()
        runs_by_id = {row["run_id"]: row for row in run_rows}
        owners = {}
        identities = {row["id"]: json.loads(row["identity_json"]) for row in version_rows}

        def references(value):
            if isinstance(value, dict):
                for member in value.values():
                    yield from references(member)
            elif isinstance(value, list):
                for member in value:
                    yield from references(member)
            elif isinstance(value, str) and value in identities:
                yield value

        def retain(identifier, owner, visited):
            if identifier in visited:
                return
            visited.add(identifier)
            owners.setdefault(identifier, set())
            if owner:
                owners[identifier].add(owner)
            for dependency in references(identities.get(identifier, {})):
                retain(dependency, owner, visited)

        for binding in binding_rows:
            retain(
                binding["version_id"],
                runs_by_id.get(binding["run_id"], {}).get("owner_user_id"),
                set(),
            )
        for row in owner_rows:
            retain(row["version_id"], row["user_id"], set())
        active = any(row["status"] in {"queued", "running"} for row in experiment_rows) or any(
            row["status"] in {"preparing", "running"} for row in run_rows
        )
        covered = set()

        def entry(
            path,
            kind,
            *,
            owner=None,
            owner_set=None,
            model=None,
            protected=True,
            reason="reproducibility_dependency",
        ):
            path = Path(path)
            if (
                not path.exists()
                or path.is_symlink()
                or not path.resolve().is_relative_to(self.paths.root)
            ):
                return None
            covered.add(str(path.resolve()))
            owner_set = sorted(owner_set if owner_set is not None else ([owner] if owner else []))
            return {
                "id": hashlib.sha256(str(path.resolve()).encode()).hexdigest(),
                "path": str(path.resolve()),
                "kind": kind,
                "owner_user_id": owner_set[0] if len(owner_set) == 1 else None,
                "owners_json": dump(owner_set),
                "model_id": model,
                "size_bytes": tree_size(path),
                "protected": int(protected),
                "reason": reason,
            }

        for row in dataset_rows:
            value = entry(
                row["root_path"],
                "dataset",
                owner=row.get("owner_user_id"),
                reason="original_dataset",
            )
            if value:
                yield value
        for row in experiment_rows:
            value = entry(
                self.paths.experiments / row["id"], "experiment", owner=row.get("owner_user_id")
            )
            if value:
                yield value
        for row in model_rows:
            path = Path(row["bundle_path"])
            # Version objects are inventoried separately, with every owner.
            if not any(
                path.resolve().is_relative_to(Path(parent)) for parent in covered
            ) and not path.resolve().is_relative_to(self.versions.root):
                value = entry(path, "model", owner=row.get("owner_user_id"), model=row["id"])
                if value:
                    yield value
        for row in version_rows:
            if "files" not in identities[row["id"]]:
                continue
            path = self.versions.object_path(row["id"])
            # Unbound published objects may still be inputs to a CLI or another
            # publication. Keep them; only unregistered abandoned trees are GC'd.
            value = entry(
                path, "version", owner_set=owners.get(row["id"], set()), reason="immutable_version"
            )
            if value:
                yield value
        workspaces = self.versions.root / "workspaces"
        if workspaces.is_dir():
            for path in workspaces.iterdir():
                run = runs_by_id.get(path.name)
                value = entry(
                    path,
                    "workspace",
                    owner=(run or {}).get("owner_user_id"),
                    protected=bool(run) or active,
                    reason="retained_input_view" if run else "abandoned_workspace",
                )
                if value:
                    yield value
        inference = self.versions.root / "inference"
        if inference.is_dir():
            model_map = {row["id"]: row for row in model_rows}
            for path in inference.iterdir():
                model = model_map.get(path.name)
                run = runs_by_id.get((model or {}).get("experiment_id"))
                reconstructible = bool(
                    model
                    and run
                    and run["status"] == "sealed"
                    and any(
                        b["run_id"] == model["experiment_id"] and b["role"] == "model"
                        for b in binding_rows
                    )
                )
                pinned = bool(pins.get(path.name, {}).get("pinned"))
                value = entry(
                    path,
                    "inference_cache",
                    owner=(model or {}).get("owner_user_id"),
                    model=path.name,
                    protected=not reconstructible or pinned,
                    reason="pinned_model"
                    if pinned
                    else "rebuildable_copy"
                    if reconstructible
                    else "unverified_dependency",
                )
                if value:
                    yield value
        staging = self.versions.root / "staging"
        if staging.is_dir():
            for path in staging.iterdir():
                value = entry(
                    path,
                    "staging",
                    protected=active,
                    reason="active_publication" if active else "abandoned_temporary",
                )
                if value:
                    yield value

        # Unknown files are never assumed disposable. Include their storage in
        # the platform total without traversing already inventoried directories.
        def remainder(path):
            if str(path.resolve()) in covered or path.is_symlink():
                return
            if path.is_dir() and any(
                Path(member).is_relative_to(path.resolve()) for member in covered
            ):
                for child in path.iterdir():
                    yield from remainder(child)
            else:
                value = entry(path, "system", reason="platform_or_unclassified")
                if value:
                    yield value

        for path in self.paths.root.iterdir():
            if path.name.startswith(".") or path.name.startswith("metadata.sqlite3"):
                continue
            yield from remainder(path)

    def scan(self):
        with self._scan_lock:
            began = now_utc().isoformat()
            entries = list(self._entries())
            current = now_utc().isoformat()
            with self.store.atomic(), self.store.connect() as conn:
                old = {
                    row["id"]: dict(row)
                    for row in conn.execute(select(resource_inventory)).mappings()
                }
                for value in entries:
                    previous = old.pop(value["id"], None)
                    if previous:
                        state = (
                            "protected"
                            if value["protected"]
                            else previous["state"]
                            if previous["state"] in {"scheduled", "notified", "failed"}
                            else "available"
                        )
                        conn.execute(
                            update(resource_inventory)
                            .where(resource_inventory.c.id == value["id"])
                            .values(
                                **value,
                                state=state,
                                last_seen_at=current,
                                **(
                                    {
                                        "scheduled_at": None,
                                        "notified_at": None,
                                        "delete_after": None,
                                    }
                                    if value["protected"]
                                    else {}
                                ),
                            )
                        )
                    else:
                        conn.execute(
                            insert(resource_inventory).values(
                                **value,
                                state="protected" if value["protected"] else "available",
                                last_seen_at=current,
                            )
                        )
                for value in old.values():
                    conn.execute(
                        update(resource_inventory)
                        .where(resource_inventory.c.id == value["id"])
                        .values(state="deleted", size_bytes=0, last_seen_at=current)
                    )
                conn.execute(
                    delete(resource_leases).where(
                        resource_leases.c.kind == "allocated", resource_leases.c.created_at <= began
                    )
                )
            self._last_scan = now_utc()
            return self.inventory()

    def inventory(self, user_id=None):
        with self.store.connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    select(resource_inventory).where(resource_inventory.c.state != "deleted")
                ).mappings()
            ]
        if user_id:
            rows = [row for row in rows if user_id in json.loads(row["owners_json"])]
        return rows

    def _idle_since(self, row):
        path = Path(row["path"])
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if row["model_id"]:
            with self.store.connect() as conn:
                pin = (
                    conn.execute(
                        select(model_protection).where(
                            model_protection.c.model_id == row["model_id"]
                        )
                    )
                    .mappings()
                    .first()
                )
            if pin:
                modified = max(modified, datetime.fromisoformat(pin["updated_at"]))
        return modified

    def _cleanup_expiry(self, row, policy):
        idle = self._idle_since(row)
        duration = (
            timedelta(
                days=self.policies.quota(row["owner_user_id"])["model_retention_days"]
                if row["owner_user_id"]
                else 1
            )
            if row["kind"] == "inference_cache"
            else timedelta(hours=policy["cache_idle_hours"])
        )
        return idle, idle + duration

    def _schedule_cleanup(self, identifier, *, publish_announcements=False, actor_user_id=None):
        """Persist a cleanup plan; silent plans are immediately executable."""
        with self.store.atomic(), self.store.connect() as conn:
            row = (
                conn.execute(
                    select(resource_inventory).where(resource_inventory.c.id == identifier)
                )
                .mappings()
                .first()
            )
            if not row:
                raise KeyError(identifier)
            row = dict(row)
            if row["state"] == "deleted":
                return row
            policy = self.policies.policy()["retention"]
            if not policy["enabled"]:
                raise ResourceLimitError("resource_cleanup_disabled", status_code=409)
            fresh = next((entry for entry in self._entries() if entry["id"] == identifier), None)
            if (
                not fresh
                or fresh["protected"]
                or fresh["kind"] not in {"inference_cache", "staging", "workspace"}
                or any(lease["resource_id"] == row["model_id"] for lease in self.policies.leases())
            ):
                raise ResourceLimitError("resource_is_retained_or_in_use", status_code=409)
            current = now_utc()
            if self._cleanup_expiry(row, policy)[1] > current:
                raise ResourceLimitError("retention_not_due", status_code=409)
            if row["scheduled_at"] and row["delete_after"]:
                return row
            deadline = (
                current + timedelta(hours=policy["notice_hours"])
                if publish_announcements
                else current
            )
            if publish_announcements:
                self.policies.publish_announcement(
                    {
                        "title": "存储清理预告 / Scheduled storage cleanup",
                        "body": f"缓存副本 {Path(row['path']).name}（{row['size_bytes']} bytes）将在 {deadline.isoformat()} 后清理。该操作只移除缓存。 / A cached copy will be removed after the deadline. This operation removes cached files only.",
                        "audience": [row["owner_user_id"]] if row["owner_user_id"] else "all",
                    },
                    identifier="cleanup-" + identifier,
                    kind="cleanup",
                    resource_id=identifier,
                )
            values = {
                "state": "notified" if publish_announcements else "scheduled",
                "scheduled_at": current.isoformat(),
                "notified_at": current.isoformat() if publish_announcements else None,
                "delete_after": deadline.isoformat(),
            }
            conn.execute(
                update(resource_inventory)
                .where(resource_inventory.c.id == identifier)
                .values(**values)
            )
            self.store.create_audit_log(
                "resource.cleanup_scheduled",
                actor_user_id=actor_user_id,
                target_user_id=row["owner_user_id"],
                metadata={
                    "resource_id": identifier,
                    "kind": row["kind"],
                    "publish_announcements": publish_announcements,
                    "scheduled_at": current.isoformat(),
                    "delete_after": deadline.isoformat(),
                },
            )
            return {**row, **values}

    def retention_tick(self, *, publish_announcements=None, actor_user_id=None):
        policy = self.policies.policy()["retention"]
        publish = announcement_requested(
            policy
            if publish_announcements is None
            else {"publish_announcements": publish_announcements}
        )
        if not policy["enabled"]:
            return
        self.scan()
        current = now_utc()
        for row in self.inventory():
            if row["protected"] or row["kind"] not in {"inference_cache", "staging", "workspace"}:
                continue
            try:
                idle, expires = self._cleanup_expiry(row, policy)
                if expires > current:
                    # Recent use restarts the idle retention period.
                    if row["scheduled_at"] and idle > datetime.fromisoformat(row["scheduled_at"]):
                        with self.store.atomic(), self.store.connect() as conn:
                            conn.execute(
                                update(resource_inventory)
                                .where(resource_inventory.c.id == row["id"])
                                .values(
                                    state="available",
                                    scheduled_at=None,
                                    notified_at=None,
                                    delete_after=None,
                                )
                            )
                    continue
                plan = self._schedule_cleanup(
                    row["id"], publish_announcements=publish, actor_user_id=actor_user_id
                )
                if (
                    not plan["notified_at"]
                    or datetime.fromisoformat(plan["delete_after"]) <= current
                ):
                    self.cleanup(row["id"], actor_user_id=actor_user_id)
            except (OSError, ValueError, KeyError, ResourceLimitError) as exc:
                logger.warning("Retention inspection failed for %s: %s", row["id"], exc)
        self.models.tick(publish_announcements=publish, actor_user_id=actor_user_id)

    def cleanup(self, identifier, *, actor_user_id=None):
        with self._scan_lock, self.versions.locked():
            self._schedule_cleanup(identifier, actor_user_id=actor_user_id)
            return self._cleanup_scheduled(identifier, actor_user_id=actor_user_id)

    def _cleanup_scheduled(self, identifier, *, actor_user_id=None):
        # Recompute protection while publication is locked; a stale preview can
        # never authorize removal of a newly retained or pinned dependency.
        with self.store.atomic():
            with self.store.connect() as conn:
                row = (
                    conn.execute(
                        select(resource_inventory).where(resource_inventory.c.id == identifier)
                    )
                    .mappings()
                    .first()
                )
            if not row:
                raise KeyError(identifier)
            row = dict(row)
            if row["state"] == "deleted":
                return {"deleted": True, "released_bytes": 0}
            current = now_utc()
            if (
                not row["scheduled_at"]
                or not row["delete_after"]
                or (row["notified_at"] and datetime.fromisoformat(row["delete_after"]) > current)
            ):
                raise ResourceLimitError("cleanup_notice_pending", status_code=409)
            fresh = next((entry for entry in self._entries() if entry["id"] == identifier), None)
            leases = self.policies.leases()
            policy = self.policies.policy()["retention"]
            if not policy["enabled"]:
                raise ResourceLimitError("resource_cleanup_disabled", status_code=409)
            if (
                not fresh
                or fresh["protected"]
                or fresh["kind"] not in {"inference_cache", "staging", "workspace"}
                or any(lease["resource_id"] == row["model_id"] for lease in leases)
            ):
                raise ResourceLimitError("resource_is_retained_or_in_use", status_code=409)
            idle, expires = self._cleanup_expiry(row, policy)
            if expires > current:
                raise ResourceLimitError("retention_not_due", status_code=409)
            if row["notified_at"] and idle > datetime.fromisoformat(row["scheduled_at"]):
                raise ResourceLimitError("resource_used_after_notice", status_code=409)
            if not row["notified_at"] and (
                datetime.fromisoformat(row["delete_after"]) > current
                or idle > datetime.fromisoformat(row["scheduled_at"])
            ):
                previous_deadline = row["delete_after"]
                row = {
                    **row,
                    "scheduled_at": max(row["scheduled_at"], idle.isoformat()),
                    "delete_after": current.isoformat(),
                }
                with self.store.connect() as conn:
                    conn.execute(
                        update(resource_inventory)
                        .where(resource_inventory.c.id == identifier)
                        .values(
                            scheduled_at=row["scheduled_at"],
                            delete_after=row["delete_after"],
                        )
                    )
                self.store.create_audit_log(
                    "resource.cleanup_scheduled",
                    actor_user_id=actor_user_id,
                    target_user_id=row["owner_user_id"],
                    metadata={
                        "resource_id": identifier,
                        "kind": row["kind"],
                        "publish_announcements": False,
                        "scheduled_at": row["scheduled_at"],
                        "delete_after": row["delete_after"],
                        "previous_delete_after": previous_deadline,
                    },
                )
            if row["kind"] == "inference_cache":
                model = self.store.get_model(row["model_id"])
                run = self.versions.run(model["experiment_id"])
                version = next(item for item in run["versions"] if item["role"] == "model")
                self.versions.verify(version["id"])
                for view in run.get("details", {}).get("input_views", []):
                    dependency = self.versions.get(view["version_id"])
                    self.versions.verify(dependency["id"])
                    for name, expected in dependency["identity"]["files"].items():
                        retained = Path(view["path"]) / name
                        if (
                            not retained.is_file()
                            or retained.is_symlink()
                            or sha256(retained) != expected["sha256"]
                        ):
                            raise ValueError("retained model dependency changed; cleanup skipped")
            path = Path(row["path"])
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(self.paths.root)
                or path.resolve() == self.paths.root
            ):
                raise ResourceLimitError("unsafe_cleanup_path", status_code=409)
            free_before = shutil.disk_usage(self.paths.root).free
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
                released = max(0, shutil.disk_usage(self.paths.root).free - free_before)
                state, error = "deleted", None
            except OSError as exc:
                state, released, error = "failed", 0, str(exc)
            import uuid

            with self.store.connect() as conn:
                conn.execute(
                    update(resource_inventory)
                    .where(resource_inventory.c.id == identifier)
                    .values(
                        state=state,
                        **({"size_bytes": 0} if state == "deleted" else {}),
                    )
                )
                conn.execute(
                    insert(resource_cleanup_logs).values(
                        id=uuid.uuid4().hex,
                        resource_id=identifier,
                        status=state,
                        released_bytes=released,
                        error=error,
                        created_at=current.isoformat(),
                    )
                )
            return {"deleted": state == "deleted", "released_bytes": released, "error": error}

    def protect_model(self, model, pinned):
        with self.store.atomic(), self.store.connect() as conn:
            self.models.require_available(model["id"])
            current = conn.execute(
                select(model_protection.c.pinned).where(model_protection.c.model_id == model["id"])
            ).scalar()
            if pinned and not current:
                from .store import models

                owned = select(models.c.id).where(
                    models.c.owner_user_id == model.get("owner_user_id")
                )
                count = len(
                    conn.execute(
                        select(model_protection.c.model_id).where(
                            model_protection.c.model_id.in_(owned),
                            model_protection.c.pinned == 1,
                            model_protection.c.model_id != model["id"],
                        )
                    ).all()
                )
                quota = self.policies.quota(model.get("owner_user_id"))
                if model.get("owner_user_id") and not quota_exempt(quota):
                    check_quota("protected_model_limit", quota["protected_model_limit"], count)
            row = conn.execute(
                select(model_protection.c.model_id).where(
                    model_protection.c.model_id == model["id"]
                )
            ).first()
            values = {"pinned": int(pinned), "updated_at": now_utc().isoformat()}
            if row:
                conn.execute(
                    update(model_protection)
                    .where(model_protection.c.model_id == model["id"])
                    .values(**values)
                )
            else:
                conn.execute(insert(model_protection).values(model_id=model["id"], **values))
            self.models.cancel_notice(model["id"])
        return {
            "model_id": model["id"],
            "pinned": pinned,
            "reproducibility_retained": self.versions.run(model["experiment_id"])["status"]
            == "sealed",
        }

    def touch_model(self, model_id):
        with self.store.atomic(), self.store.connect() as conn:
            self.models.require_available(model_id)
            self.models.cancel_notice(model_id)
            row = conn.execute(
                select(model_protection).where(model_protection.c.model_id == model_id)
            ).first()
            if row:
                conn.execute(
                    update(model_protection)
                    .where(model_protection.c.model_id == model_id)
                    .values(updated_at=now_utc().isoformat())
                )
            else:
                conn.execute(
                    insert(model_protection).values(
                        model_id=model_id, pinned=0, updated_at=now_utc().isoformat()
                    )
                )

    def start(self):
        if self._thread:
            return

        def maintain():
            while not self._stop.wait(60):
                try:
                    self.scan()
                    self.retention_tick()
                except Exception:
                    logger.exception("Resource maintenance failed")

        self._thread = threading.Thread(
            target=maintain, name="pepplatform-resource-maintenance", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
