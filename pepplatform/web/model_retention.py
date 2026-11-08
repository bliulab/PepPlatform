"""Administrative model lifetimes, separate from scientific content identities."""

import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import func, insert, select, update

from pepplatform.versioning.core import bindings, records, runs, sha256, tree_files, version_id

from .resource_policy import (
    ResourceLimitError,
    announcement_requested,
    check_quota,
    dump,
    model_protection,
    model_retention,
    now_utc,
    platform_announcements,
    quota_exempt,
    resource_cleanup_logs,
    retired_model_versions,
)

logger = logging.getLogger(__name__)


def active_model_condition():
    """Historical rows stay available to history, but do not use library slots."""
    from .store import models

    return (
        ~select(model_retention.c.model_id)
        .where(
            model_retention.c.model_id == models.c.id,
            model_retention.c.state == "removed",
        )
        .exists()
    )


def decorate_models(store, values, *, include_removed=False):
    if not values:
        return []
    ids = [row["id"] for row in values]
    states, pins = {}, set()
    with store.connect() as conn:
        for offset in range(0, len(ids), 500):
            chunk = ids[offset : offset + 500]
            states.update(
                {
                    row["model_id"]: dict(row)
                    for row in conn.execute(
                        select(model_retention).where(model_retention.c.model_id.in_(chunk))
                    ).mappings()
                }
            )
            pins.update(
                conn.execute(
                    select(model_protection.c.model_id).where(
                        model_protection.c.model_id.in_(chunk), model_protection.c.pinned == 1
                    )
                ).scalars()
            )
    result = []
    for row in values:
        state = states.get(row["id"], {})
        status = state.get("state", "active")
        if status == "removed" and not include_removed:
            continue
        result.append(
            {
                **row,
                "supports_inference": row["supports_inference"]
                and status not in {"removed", "cleaning", "failed"},
                "retention": {
                    "state": status,
                    "pinned": row["id"] in pins,
                    "scheduled_at": state.get("scheduled_at"),
                    "notified_at": state.get("notified_at"),
                    "delete_after": state.get("delete_after"),
                    "removed_at": state.get("removed_at"),
                },
            }
        )
    return result


class ModelRetention:
    def __init__(self, governor):
        self.governor = governor
        self.store, self.versions, self.paths = governor.store, governor.versions, governor.paths
        self.policies = self.store.resources
        # Operators can exclude managed subtrees too. External paths are never
        # cleanup candidates, regardless of whether they have an exclusion.
        self.excluded = [
            Path(p).expanduser().resolve()
            for p in os.environ.get("PEP_RETENTION_EXCLUDED_ROOTS", "").split(os.pathsep)
            if p
        ]

    def usage(self, user_id):
        from .store import experiments, models

        with self.store.connect() as conn:
            retained = conn.scalar(
                select(func.count())
                .select_from(models)
                .where(models.c.owner_user_id == user_id, active_model_condition())
            )
            protected = conn.scalar(
                select(func.count())
                .select_from(models)
                .join(model_protection, model_protection.c.model_id == models.c.id)
                .where(
                    models.c.owner_user_id == user_id,
                    active_model_condition(),
                    model_protection.c.pinned == 1,
                )
            )
            # Durable queued/running experiments reserve one model slot. Once
            # their model is registered, the same slot is counted by that row.
            reserved = conn.scalar(
                select(func.count())
                .select_from(experiments)
                .where(
                    experiments.c.owner_user_id == user_id,
                    experiments.c.status.in_(["queued", "running"]),
                    ~select(models.c.id).where(models.c.experiment_id == experiments.c.id).exists(),
                )
            )
        quota = self.policies.quota(user_id)
        exempt = quota_exempt(quota)
        return {
            "retained": retained,
            "protected": protected,
            "reserved": reserved,
            "remaining": None if exempt else max(0, quota["max_model_count"] - retained - reserved),
            "protect_remaining": None
            if exempt
            else max(0, quota["protected_model_limit"] - protected),
            "over_limit": 0 if exempt else max(0, retained + reserved - quota["max_model_count"]),
            "protection_over_limit": 0
            if exempt
            else max(0, protected - quota["protected_model_limit"]),
        }

    def admit(self, user_id, count):
        if not user_id or not count:
            return
        if quota_exempt(self.policies.quota(user_id)):
            return
        usage = self.usage(user_id)
        check_quota(
            "max_model_count",
            self.policies.quota(user_id)["max_model_count"],
            usage["retained"] + usage["reserved"],
            count,
        )

    def state(self, model_id):
        with self.store.connect() as conn:
            row = (
                conn.execute(select(model_retention).where(model_retention.c.model_id == model_id))
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def require_available(self, model_id):
        state = self.state(model_id)
        if state and state["state"] == "removed":
            raise ResourceLimitError("model_removed_by_retention", status_code=410)
        if state and state["state"] in {"cleaning", "failed"}:
            raise ResourceLimitError("model_cleanup_in_progress", status_code=409)

    def cancel_notice(self, model_id):
        with self.store.connect() as conn:
            conn.execute(
                update(model_retention)
                .where(
                    model_retention.c.model_id == model_id,
                    model_retention.c.state.in_(["scheduled", "notified"]),
                )
                .values(
                    state="active",
                    scheduled_at=None,
                    notified_at=None,
                    delete_after=None,
                    updated_at=now_utc().isoformat(),
                )
            )
            conn.execute(
                update(platform_announcements)
                .where(
                    platform_announcements.c.id == "model-cleanup-" + model_id,
                    platform_announcements.c.kind == "cleanup",
                )
                .values(expires_at=now_utc().isoformat())
            )

    def _context(self):
        from .store import experiments, models

        with self.store.connect() as conn:
            return {
                "models": [dict(r) for r in conn.execute(select(models)).mappings()],
                "experiments": {
                    r["id"]: dict(r) for r in conn.execute(select(experiments)).mappings()
                },
                "runs": {r["run_id"]: dict(r) for r in conn.execute(select(runs)).mappings()},
                "bindings": [dict(r) for r in conn.execute(select(bindings)).mappings()],
                "states": {
                    r["model_id"]: dict(r) for r in conn.execute(select(model_retention)).mappings()
                },
                "pins": {
                    r["model_id"]: dict(r)
                    for r in conn.execute(select(model_protection)).mappings()
                },
                "identities": {
                    r["id"]: json.loads(r["identity_json"])
                    for r in conn.execute(select(records)).mappings()
                },
                "leases": self.policies.leases(),
            }

    def _safe(self, path):
        path = Path(path)
        resolved = path.resolve()
        return (
            not path.is_symlink()
            and resolved != self.paths.root
            and resolved.is_relative_to(self.paths.root)
            and not any(resolved == p or resolved.is_relative_to(p) for p in self.excluded)
        )

    def _status(self, model, ctx):
        state = ctx["states"].get(model["id"], {})
        if state.get("state") == "removed":
            return "removed"
        if not model["owner_user_id"]:
            return "system_model"
        if ctx["pins"].get(model["id"], {}).get("pinned"):
            return "pinned_model"
        run = ctx["runs"].get(model["experiment_id"], {})
        if (
            run.get("status") != "sealed"
            or run.get("owner_user_id") != model["owner_user_id"]
            or not any(
                b["run_id"] == model["experiment_id"] and b["role"] == "model"
                for b in ctx["bindings"]
            )
            or not self._safe(model["bundle_path"])
        ):
            return "unverified_dependency"
        experiment = ctx["experiments"].get(model["experiment_id"], {})
        if experiment.get("status") in {"queued", "running"} or any(
            lease["resource_id"] == model["id"] for lease in ctx["leases"]
        ):
            return "model_in_use"
        return None

    def preview(self, *, include_sizes=True):
        ctx = self._context()
        quota = self.policies.quotas(
            list({m["owner_user_id"] for m in ctx["models"] if m["owner_user_id"]})
        )
        result = []
        for model in ctx["models"]:
            state = ctx["states"].get(model["id"], {})
            reason = self._status(model, ctx)
            if reason == "removed":
                continue
            used_at = ctx["pins"].get(model["id"], {}).get("updated_at", model["created_at"])
            expires = datetime.fromisoformat(used_at) + timedelta(
                days=quota.get(model["owner_user_id"], {}).get("model_retention_days", 14)
            )
            if not reason and expires > now_utc():
                reason = "retention_not_due"
            estimated = 0
            if include_sizes and not self._status(model, ctx):
                from .resource_governor import tree_size

                try:
                    paths, _ = self._paths(model, ctx)
                    estimated = sum(tree_size(item["path"]) for item in paths)
                except OSError:
                    reason = "unverified_dependency"
            result.append(
                {
                    "model_id": model["id"],
                    "name": model["name"],
                    "owner_user_id": model["owner_user_id"],
                    "pinned": bool(ctx["pins"].get(model["id"], {}).get("pinned")),
                    "state": state.get("state", "active"),
                    "skip_reason": reason,
                    "eligible": reason is None,
                    "expires_at": expires.isoformat(),
                    "scheduled_at": state.get("scheduled_at"),
                    "notified_at": state.get("notified_at"),
                    "delete_after": state.get("delete_after"),
                    "estimated_bytes": estimated,
                }
            )
        policy = self.policies.policy()["retention"]
        return {
            "enabled": policy["enabled"] and policy["delete_unprotected_models"],
            "notice_hours": policy["notice_hours"],
            "items": result,
            "eligible_count": sum(row["eligible"] for row in result),
        }

    def tick(self, *, publish_announcements=None, actor_user_id=None):
        policy = self.policies.policy()["retention"]
        publish = announcement_requested(
            policy
            if publish_announcements is None
            else {"publish_announcements": publish_announcements}
        )
        preview = self.preview(include_sizes=False)
        if not preview["enabled"]:
            return
        for row in preview["items"]:
            if not row["eligible"]:
                continue
            try:
                state = self._schedule_cleanup(
                    row["model_id"], publish_announcements=publish, actor_user_id=actor_user_id
                )
                if (
                    not state["notified_at"]
                    or datetime.fromisoformat(state["delete_after"]) <= now_utc()
                ):
                    self.cleanup(row["model_id"], actor_user_id=actor_user_id)
            except (OSError, ValueError, KeyError, ResourceLimitError) as exc:
                logger.warning("Model cleanup deferred for %s: %s", row["model_id"], exc)

    def _require_expired(self, model, ctx):
        used = ctx["pins"].get(model["id"], {}).get("updated_at", model["created_at"])
        days = self.policies.quota(model["owner_user_id"])["model_retention_days"]
        if datetime.fromisoformat(used) + timedelta(days=days) > now_utc():
            raise ResourceLimitError("retention_not_due", status_code=409)
        return used

    def _schedule_cleanup(self, model_id, *, publish_announcements=False, actor_user_id=None):
        """Persist intent before removal; only an actually published notice adds lead time."""
        with self.store.atomic():
            policy = self.policies.policy()["retention"]
            if not (policy["enabled"] and policy["delete_unprotected_models"]):
                raise ResourceLimitError("model_cleanup_disabled", status_code=409)
            existing = self.state(model_id)
            if existing and existing["state"] == "removed":
                return existing
            model = self.store.get_model(model_id, include_removed=True)
            if not model:
                raise KeyError(model_id)
            ctx = self._context()
            if self._status(model, ctx):
                raise ResourceLimitError("resource_is_retained_or_in_use", status_code=409)
            self._require_expired(model, ctx)
            if existing and existing["scheduled_at"] and existing["delete_after"]:
                return existing  # A later setting never changes whether this plan was announced.
            current = now_utc()
            deadline = (
                current + timedelta(hours=policy["notice_hours"])
                if publish_announcements
                else current
            )
            values = {
                "model_id": model["id"],
                "experiment_id": model["experiment_id"],
                "owner_user_id": model["owner_user_id"],
                "state": "notified" if publish_announcements else "scheduled",
                "scheduled_at": current.isoformat(),
                "notified_at": current.isoformat() if publish_announcements else None,
                "delete_after": deadline.isoformat(),
                "removed_at": None,
                "reason": "retention",
                "manifest_json": "{}",
                "error": None,
                "updated_at": current.isoformat(),
            }
            with self.store.connect() as conn:
                if existing:
                    conn.execute(
                        update(model_retention)
                        .where(model_retention.c.model_id == model_id)
                        .values(**values)
                    )
                else:
                    conn.execute(insert(model_retention).values(**values))
            if publish_announcements:
                self.policies.publish_announcement(
                    {
                        "title": "模型清理预告 / Scheduled model removal",
                        "body": f"模型 {model['name']} 将在 {deadline.isoformat()} 后清理，并停止在线推理和权重下载。请在此前下载备份或保护模型。 / This model will be removed after the deadline. Download a backup or protect it before then.",
                        "audience": [model["owner_user_id"]],
                    },
                    identifier="model-cleanup-" + model_id,
                    kind="cleanup",
                    resource_id=model_id,
                )
            self.store.create_audit_log(
                "resource.cleanup_scheduled",
                actor_user_id=actor_user_id,
                target_user_id=model["owner_user_id"],
                metadata={
                    "resource_id": model_id,
                    "kind": "model",
                    "publish_announcements": publish_announcements,
                    "scheduled_at": current.isoformat(),
                    "delete_after": deadline.isoformat(),
                },
            )
            return values

    def _path_references(self, value, path):
        if isinstance(value, dict):
            return any(self._path_references(v, path) for v in value.values())
        if isinstance(value, list):
            return any(self._path_references(v, path) for v in value)
        if isinstance(value, str) and value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = self.paths.project_root / candidate
            try:
                candidate = candidate.resolve()
            except (OSError, ValueError):
                return False
            return candidate == path or candidate.is_relative_to(path)
        return False

    def _referenced(self, model, path, version_id, ctx):
        retired_runs = {
            r["experiment_id"] for r in ctx["states"].values() if r["state"] == "removed"
        }
        live_models = [
            m
            for m in ctx["models"]
            if m["id"] != model["id"] and ctx["states"].get(m["id"], {}).get("state") != "removed"
        ]
        if any(m["experiment_id"] == model["experiment_id"] for m in live_models):
            return True
        if version_id and any(
            b["version_id"] == version_id
            and b["run_id"] != model["experiment_id"]
            and b["run_id"] not in retired_runs
            for b in ctx["bindings"]
        ):
            return True
        for other in live_models:
            if self._path_references(other["bundle_path"], path) or self._path_references(
                json.loads(other["metadata_json"]), path
            ):
                return True
        for identifier, run in ctx["runs"].items():
            if (
                identifier != model["experiment_id"]
                and identifier not in retired_runs
                and self._path_references(json.loads(run["details_json"]), path)
            ):
                return True
        for experiment in ctx["experiments"].values():
            if experiment["status"] in {"queued", "running"} and self._path_references(
                json.loads(experiment["config_json"]), path
            ):
                return True
        return False

    def _paths(self, model, ctx):
        selected = [
            b
            for b in ctx["bindings"]
            if b["run_id"] == model["experiment_id"]
            and (
                b["role"] in {"model", "checkpoint"}
                or b["role"].startswith(("member:", "input:member_"))
            )
        ]
        output = next(b["version_id"] for b in selected if b["role"] == "model")
        candidates = [
            {
                "path": str(self.versions.object_path(b["version_id"])),
                "version_id": b["version_id"],
                "object": True,
            }
            for b in selected
        ]
        experiment = ctx["experiments"].get(model["experiment_id"], {})
        artifact = Path(experiment.get("artifact_dir", ""))
        if artifact.resolve().is_relative_to(self.paths.experiments / model["experiment_id"]):
            candidates.append({"path": str(artifact), "version_id": output, "object": False})
        candidates.append(
            {
                "path": str(self.versions.root / "inference" / model["id"]),
                "version_id": None,
                "object": False,
            }
        )
        # Only member views contain model copies; original retained datasets,
        # source inputs and pretrained assets are never reclaimed here.
        details = json.loads(ctx["runs"][model["experiment_id"]]["details_json"])
        for view in details.get("input_views", []):
            if str(view.get("flag", "")).startswith("member_"):
                path = Path(view["path"])
                if path.resolve().is_relative_to(
                    self.versions.root / "workspaces" / model["experiment_id"]
                ):
                    candidates.append(
                        {"path": str(path), "version_id": view["version_id"], "object": False}
                    )
        result, skipped = {}, []
        for candidate in candidates:
            path = Path(candidate["path"])
            if not path.exists():
                continue
            if not self._safe(path) or self._referenced(
                model, path.resolve(), candidate["version_id"], ctx
            ):
                skipped.append({"name": path.name, "reason": "shared_or_system_retained"})
                continue
            result[str(path)] = candidate
        return list(result.values()), skipped

    def _verify_path(self, item, ctx, *, resume=False):
        path = Path(item["path"])
        if not self._safe(path):
            raise ValueError("unsafe model cleanup path")
        if any(p.is_symlink() for p in path.rglob("*")):
            raise ValueError("model cleanup cannot traverse symlinks")
        identifier = item["version_id"]
        if not identifier:
            return
        identity = ctx["identities"][identifier]
        if version_id(identity["kind"], identity) != identifier:
            raise ValueError("registered model identity changed")
        if item["object"] and not resume:
            self.versions.verify(identifier)
            return
        files = path / "files" if item["object"] else path
        actual = (
            tree_files(files)
            if files.is_dir() and any(p.is_file() for p in files.rglob("*"))
            else {}
        )
        if (set(actual) - set(identity["files"])) or (
            not resume and set(actual) != set(identity["files"])
        ):
            raise ValueError("model file set changed; cleanup skipped")
        for name in actual:
            expected = identity["files"][name]
            if (
                actual[name].stat().st_size != expected["bytes"]
                or sha256(actual[name]) != expected["sha256"]
            ):
                raise ValueError("model content changed; cleanup skipped")

    def _verify_retained_inputs(self, model, ctx, removed_paths):
        """Validate retained inputs without changing their identities or files."""
        details = json.loads(ctx["runs"][model["experiment_id"]]["details_json"])
        verified = set()
        for binding in ctx["bindings"]:
            if binding["run_id"] != model["experiment_id"] or not binding["role"].startswith(
                "input:"
            ):
                continue
            identifier = binding["version_id"]
            if str(self.versions.object_path(identifier)) in removed_paths:
                continue
            self.versions.verify(identifier)
            verified.add(identifier)
        for view in details.get("input_views", []):
            path, identifier = Path(view["path"]), view["version_id"]
            if str(path) in removed_paths:
                continue
            if identifier not in verified:
                self.versions.verify(identifier)
                verified.add(identifier)
            identity = ctx["identities"][identifier]
            if path.is_symlink() or any(p.is_symlink() for p in path.rglob("*")):
                raise ValueError("retained model dependency changed; cleanup skipped")
            actual = tree_files(path)
            if set(actual) != set(identity["files"]):
                raise ValueError("retained model dependency changed; cleanup skipped")
            for name, retained in actual.items():
                expected = identity["files"][name]
                if (
                    retained.stat().st_size != expected["bytes"]
                    or sha256(retained) != expected["sha256"]
                ):
                    raise ValueError("retained model dependency changed; cleanup skipped")

    def cleanup(self, model_id, *, actor_user_id=None):
        with self.governor._scan_lock, self.versions.locked():
            try:
                self._schedule_cleanup(model_id, actor_user_id=actor_user_id)
                return self._cleanup_locked(model_id, actor_user_id=actor_user_id)
            except (ValueError, OSError) as exc:
                with self.store.atomic(), self.store.connect() as conn:
                    conn.execute(
                        update(model_retention)
                        .where(model_retention.c.model_id == model_id)
                        .values(error=str(exc), updated_at=now_utc().isoformat())
                    )
                    conn.execute(
                        insert(resource_cleanup_logs).values(
                            id=uuid.uuid4().hex,
                            resource_id="model:" + model_id,
                            status="failed",
                            released_bytes=0,
                            error=str(exc),
                            created_at=now_utc().isoformat(),
                        )
                    )
                raise ResourceLimitError(
                    "model_cleanup_verification_failed", status_code=409
                ) from exc

    def _cleanup_locked(self, model_id, *, actor_user_id=None):
        with self.store.atomic():
            policy = self.policies.policy()["retention"]
            if not (policy["enabled"] and policy["delete_unprotected_models"]):
                raise ResourceLimitError("model_cleanup_disabled", status_code=409)
            state = self.state(model_id)
            if state and state["state"] == "removed":
                return {"deleted": True, "released_bytes": 0, "already_removed": True}
            if (
                not state
                or not state["scheduled_at"]
                or not state["delete_after"]
                or (
                    state["notified_at"]
                    and datetime.fromisoformat(state["delete_after"]) > now_utc()
                )
            ):
                raise ResourceLimitError("cleanup_notice_pending", status_code=409)
            model = self.store.get_model(model_id, include_removed=True)
            if not model:
                raise KeyError(model_id)
            ctx = self._context()
            if self._status(model, ctx):
                raise ResourceLimitError("resource_is_retained_or_in_use", status_code=409)
            used = self._require_expired(model, ctx)
            if state["notified_at"] and used > state["scheduled_at"]:
                raise ResourceLimitError("resource_used_after_notice", status_code=409)
            if not state["notified_at"] and (
                datetime.fromisoformat(state["delete_after"]) > now_utc()
                or used > state["scheduled_at"]
            ):
                previous_deadline = state["delete_after"]
                state = {
                    **state,
                    "scheduled_at": max(state["scheduled_at"], used),
                    "delete_after": now_utc().isoformat(),
                }
                with self.store.connect() as conn:
                    conn.execute(
                        update(model_retention)
                        .where(model_retention.c.model_id == model_id)
                        .values(
                            scheduled_at=state["scheduled_at"],
                            delete_after=state["delete_after"],
                            updated_at=now_utc().isoformat(),
                        )
                    )
                self.store.create_audit_log(
                    "resource.cleanup_scheduled",
                    actor_user_id=actor_user_id,
                    target_user_id=model["owner_user_id"],
                    metadata={
                        "resource_id": model_id,
                        "kind": "model",
                        "publish_announcements": False,
                        "scheduled_at": state["scheduled_at"],
                        "delete_after": state["delete_after"],
                        "previous_delete_after": previous_deadline,
                    },
                )
            paths, skipped = self._paths(model, ctx)
            previous = json.loads(state["manifest_json"])
            previous_paths = {p["path"] for p in previous} if isinstance(previous, list) else set()
            attempted = {p["path"]: p for p in previous} if isinstance(previous, list) else {}
            attempted.update({p["path"]: p for p in paths})
            self._verify_retained_inputs(
                model, ctx, previous_paths if state["state"] in {"failed", "cleaning"} else set()
            )
            for item in paths:
                self._verify_path(
                    item,
                    ctx,
                    resume=state["state"] in {"failed", "cleaning"}
                    and item["path"] in previous_paths,
                )
            # Preserve small reports independently of removable weight objects.
            report_root = self.paths.root / "model-history" / model_id
            source = Path(model["bundle_path"])
            for name in (
                "metadata.json",
                "evaluation.json",
                "analysis.json",
                "selection_history.json",
            ):
                member = source / name
                if (
                    member.is_file()
                    and not member.is_symlink()
                    and member.stat().st_size <= 4 * 1024**2
                ):
                    report_root.mkdir(parents=True, exist_ok=True)
                    destination = report_root / name
                    if destination.exists():
                        if destination.is_symlink() or sha256(destination) != sha256(member):
                            raise ValueError("retained model report changed; cleanup skipped")
                    else:
                        shutil.copy2(member, destination)
            current = now_utc().isoformat()
            with self.store.connect() as conn:
                conn.execute(
                    update(model_retention)
                    .where(model_retention.c.model_id == model_id)
                    .values(
                        state="cleaning",
                        manifest_json=dump(list(attempted.values())),
                        updated_at=current,
                    )
                )
        # The intent above is durable before unlinking anything. A crash during
        # removal can be resumed without labelling a partial model available.
        free_before = shutil.disk_usage(self.paths.root).free
        error = None
        try:
            for item in paths:
                with self.store.atomic(), self.store.connect() as conn:
                    path = Path(item["path"])
                    if self._referenced(model, path.resolve(), item["version_id"], self._context()):
                        skipped.append({"name": path.name, "reason": "shared_or_system_retained"})
                        continue
                    if path.exists():
                        shutil.rmtree(path)
                    if item["object"]:
                        identifier = item["version_id"]
                        if not conn.execute(
                            select(retired_model_versions.c.version_id).where(
                                retired_model_versions.c.version_id == identifier
                            )
                        ).first():
                            conn.execute(
                                insert(retired_model_versions).values(
                                    version_id=identifier, model_id=model_id, removed_at=current
                                )
                            )
        except OSError as exc:
            error = str(exc)
        with self.store.atomic(), self.store.connect() as conn:
            # A crash after unlink but before committing its marker must still
            # yield an explicit unavailable version on a successful retry.
            for item in attempted.values():
                identifier = item["version_id"]
                if item["object"] and identifier:
                    path = self.versions.object_path(identifier)
                    if (
                        str(path) == item["path"]
                        and self._safe(path)
                        and not path.exists()
                        and not conn.execute(
                            select(retired_model_versions.c.version_id).where(
                                retired_model_versions.c.version_id == identifier
                            )
                        ).first()
                    ):
                        conn.execute(
                            insert(retired_model_versions).values(
                                version_id=identifier, model_id=model_id, removed_at=current
                            )
                        )
            conn.execute(
                update(model_retention)
                .where(model_retention.c.model_id == model_id)
                .values(
                    state="failed" if error else "removed",
                    removed_at=None if error else current,
                    error=error,
                    updated_at=current,
                )
            )
            released = 0 if error else max(0, shutil.disk_usage(self.paths.root).free - free_before)
            conn.execute(
                insert(resource_cleanup_logs).values(
                    id=uuid.uuid4().hex,
                    resource_id="model:" + model_id,
                    status="failed" if error else "deleted",
                    released_bytes=released,
                    error=error,
                    created_at=current,
                )
            )
        return {
            "deleted": error is None,
            "released_bytes": released,
            "error": error,
            "skipped": skipped,
        }
