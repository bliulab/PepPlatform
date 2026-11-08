"""HTTP adapters for resource administration and user notices."""

import io
import json
import queue
import tarfile
import threading
import time
from functools import wraps
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import delete, select, update

from .resource_policy import (
    PolicyConflict,
    ResourceLimitError,
    announcement_reads,
    now_utc,
    platform_announcements,
    resource_cleanup_logs,
    resource_leases,
)


def http_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ResourceLimitError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.detail,
                headers={"Retry-After": str(exc.detail["retry_after"])}
                if "retry_after" in exc.detail
                else None,
            ) from exc
        except PolicyConflict as exc:
            raise HTTPException(
                status_code=409, detail={"code": "resource_policy_conflict", "message": str(exc)}
            ) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="resource not found") from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail={"code": "invalid_resource_policy", "message": str(exc)}
            ) from exc

    return wrapped


def serialize_resource_mutations(app, store):
    """Keep preflight/creation/accounting atomic for every current training entry."""
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if not endpoint or not hasattr(endpoint, "__code__"):
            continue
        if "_check_experiment_quota" not in endpoint.__code__.co_freevars:
            continue

        def wrap(function):
            @wraps(function)
            def atomic(*args, **kwargs):
                with store.atomic():
                    return function(*args, **kwargs)

            return http_errors(atomic)

        wrapped = wrap(endpoint)
        route.endpoint = wrapped
        route.dependant.call = wrapped


def public_inventory(row):
    return {
        **{key: value for key, value in row.items() if key not in {"path", "owners_json"}},
        "name": Path(row["path"]).name,
        "protected": bool(row["protected"]),
    }


def register_resource_routes(app, governor, current_user, require_admin, require_model, audit):
    store, policies = governor.store, governor.policies

    async def resource_error(_request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers={"Retry-After": str(exc.detail["retry_after"])}
            if "retry_after" in exc.detail
            else None,
        )

    app.add_exception_handler(ResourceLimitError, resource_error)

    @app.get("/api/admin/email-delivery-usage")
    @http_errors
    def email_delivery_usage(request: Request = None):
        require_admin(request)
        return policies.email_delivery_usage()

    @app.get("/api/admin/resource-policies")
    @http_errors
    def resource_policy(request: Request = None):
        require_admin(request)
        return policies.policy()

    @app.post("/api/admin/resource-policies/preview")
    @http_errors
    def preview_policy(payload: dict, request: Request = None):
        require_admin(request)
        return policies.preview(payload)

    @app.patch("/api/admin/resource-policies")
    @http_errors
    def update_policy(payload: dict, request: Request = None):
        actor = require_admin(request)
        with store.atomic():
            before = policies.policy()
            result = policies.save_policy(payload, actor_user_id=actor["id"])
            audit(
                "admin.resource_policy_updated",
                request=request,
                actor=actor,
                metadata={
                    "before": before,
                    "after": result["policy"],
                    "application": result["application"],
                    "impacts": result["impacts"],
                    "daily_reduction_mode": payload.get("daily_reduction_mode", "next_reset"),
                    "publish_announcements": payload.get("publish_announcements", False),
                },
            )
        return result

    def overview():
        from .store import experiments, users

        inventory = governor.inventory()
        with store.connect() as conn:
            owners = dict(
                conn.execute(
                    select(users.c.id, users.c.email).where(
                        users.c.id.in_(
                            {row["owner_user_id"] for row in inventory if row["owner_user_id"]}
                        )
                    )
                ).all()
            )
            counts = conn.execute(
                select(experiments.c.status, experiments.c.id).where(
                    experiments.c.status.in_(["queued", "running"]),
                )
            ).all()
            logs = [
                dict(row)
                for row in conn.execute(
                    select(resource_cleanup_logs)
                    .order_by(resource_cleanup_logs.c.created_at.desc())
                    .limit(30)
                ).mappings()
            ]
        return {
            "disk": governor.disk(),
            "gpus": governor.gpu_metrics(),
            "storage": governor.storage_usage(),
            "workload": {
                "queued": sum(row.status == "queued" for row in counts),
                "running": sum(row.status == "running" for row in counts),
            },
            "registration": {
                "used": policies.daily("__public_registration__")["registrations"],
                "limit": policies.policy()["limits"]["daily_registration_limit"],
            },
            "items": [
                {**public_inventory(row), "owner_email": owners.get(row["owner_user_id"])}
                for row in inventory
            ],
            "cleanup_logs": logs,
            "retention_policy": "managed_user_models_with_reference_checks",
        }

    @app.get("/api/admin/resources")
    @http_errors
    def admin_resources(request: Request = None):
        require_admin(request)
        return overview()

    @app.post("/api/admin/resources/scan")
    @http_errors
    def scan_resources(request: Request = None):
        require_admin(request)
        governor.scan()
        return overview()

    @app.post("/api/admin/resources/retention")
    @http_errors
    def retention_resources(request: Request = None, payload: dict | None = None):
        actor = require_admin(request)
        from .resource_policy import announcement_requested

        publish = announcement_requested(payload or {})
        governor.retention_tick(publish_announcements=publish, actor_user_id=actor["id"])
        audit(
            "admin.retention_checked",
            request=request,
            actor=actor,
            metadata={"publish_announcements": publish},
        )
        return overview()

    @app.post("/api/admin/resources/{resource_id}/cleanup")
    @http_errors
    def cleanup_resource(resource_id: str, request: Request = None):
        actor = require_admin(request)
        result = governor.cleanup(resource_id, actor_user_id=actor["id"])
        audit(
            "admin.resource_cleaned",
            request=request,
            actor=actor,
            metadata={"resource_id": resource_id, **result},
        )
        return result

    @app.post("/api/admin/model-cleanup/preview")
    @http_errors
    def model_cleanup_preview(request: Request = None):
        require_admin(request)
        result = governor.models.preview()
        from .store import users

        with store.connect() as conn:
            owners = dict(conn.execute(select(users.c.id, users.c.email)).all())
        result["items"] = [
            {**row, "owner_email": owners.get(row["owner_user_id"])} for row in result["items"]
        ]
        return result

    @app.post("/api/admin/model-cleanup/{model_id}")
    @http_errors
    def cleanup_model(model_id: str, request: Request = None):
        actor = require_admin(request)
        result = governor.models.cleanup(model_id, actor_user_id=actor["id"])
        audit(
            "admin.model_cleaned",
            request=request,
            actor=actor,
            metadata={"model_id": model_id, **result},
        )
        governor.scan()
        return result

    @app.get("/api/resources/me")
    @http_errors
    def own_resources(request: Request = None):
        user = current_user(request)
        if not user:
            raise HTTPException(status_code=401, detail="authentication required")
        return {
            "storage": governor.storage_usage(user["id"]),
            "quota": policies.quota(user["id"]),
            "usage": store.get_quota_usage(user["id"]),
            "models": governor.models.usage(user["id"]),
            "items": [public_inventory(row) for row in governor.inventory(user["id"])],
            "disk_state": governor.disk()["state"],
            "reset_timezone": "UTC",
        }

    @app.get("/api/admin/announcements")
    @http_errors
    def admin_announcements(request: Request = None):
        require_admin(request)
        return {"items": policies.announcements(admin=True)}

    @app.post("/api/admin/announcements")
    @http_errors
    def create_announcement(payload: dict, request: Request = None):
        actor = require_admin(request)
        with store.atomic():
            result = policies.publish_announcement(payload)
            audit(
                "admin.announcement_published",
                request=request,
                actor=actor,
                metadata={"announcement_id": result["id"], "audience": result["audience"]},
            )
        return result

    @app.patch("/api/admin/announcements/{announcement_id}")
    @http_errors
    def update_announcement(announcement_id: str, payload: dict, request: Request = None):
        actor = require_admin(request)
        with store.atomic():
            before = next(
                (row for row in policies.announcements(admin=True) if row["id"] == announcement_id),
                None,
            )
            if not before:
                raise KeyError(announcement_id)
            if before["kind"] != "manual":
                raise ResourceLimitError("cleanup_notice_is_managed", status_code=409)
            result = policies.publish_announcement(
                {**before, **payload}, identifier=announcement_id
            )
            audit(
                "admin.announcement_updated",
                request=request,
                actor=actor,
                metadata={"announcement_id": announcement_id},
            )
        return result

    @app.delete("/api/admin/announcements/{announcement_id}")
    @http_errors
    def remove_announcement(announcement_id: str, request: Request = None):
        actor = require_admin(request)
        with store.atomic(), store.connect() as conn:
            row = (
                conn.execute(
                    select(platform_announcements).where(
                        platform_announcements.c.id == announcement_id
                    )
                )
                .mappings()
                .first()
            )
            if not row:
                raise KeyError(announcement_id)
            if row["kind"] != "manual":
                raise ResourceLimitError("cleanup_notice_is_managed", status_code=409)
            conn.execute(
                delete(announcement_reads).where(
                    announcement_reads.c.announcement_id == announcement_id
                )
            )
            conn.execute(
                delete(platform_announcements).where(platform_announcements.c.id == announcement_id)
            )
            audit(
                "admin.announcement_deleted",
                request=request,
                actor=actor,
                metadata={"announcement_id": announcement_id},
            )
        return {"deleted": True}

    @app.get("/api/announcements")
    @http_errors
    def announcements(request: Request = None):
        return {"items": policies.announcements(current_user(request))}

    @app.post("/api/announcements/{announcement_id}/read")
    @http_errors
    def read_announcement(announcement_id: str, request: Request = None):
        user = current_user(request)
        if not user:
            raise HTTPException(status_code=401, detail="authentication required")
        policies.mark_read(announcement_id, user)
        return {"read": True}

    @app.get("/api/models/{model_id}/protection")
    @http_errors
    def model_protection(model_id: str, request: Request = None):
        from .resource_policy import model_protection as protection_table

        model = require_model(model_id, current_user(request))
        with store.connect() as conn:
            row = (
                conn.execute(
                    select(protection_table).where(protection_table.c.model_id == model["id"])
                )
                .mappings()
                .first()
            )
        run = governor.versions.run(model["experiment_id"])
        return {
            "model_id": model_id,
            "pinned": bool(row and row["pinned"]),
            "reproducibility_retained": run["status"] == "sealed",
        }

    @app.patch("/api/models/{model_id}/protection")
    @http_errors
    def pin_model(model_id: str, payload: dict, request: Request = None):
        user = current_user(request)
        model = require_model(model_id, user)
        if not user or (user.get("role") != "admin" and model.get("owner_user_id") != user["id"]):
            raise HTTPException(status_code=403, detail="permission denied")
        if not isinstance(payload.get("pinned"), bool):
            raise ValueError("pinned must be a boolean")  # noqa: TRY004 -- HTTP validation uses ValueError.
        result = governor.protect_model(model, payload["pinned"])
        audit(
            "resource.model_protection_updated",
            request=request,
            actor=user,
            target_user_id=model.get("owner_user_id"),
            metadata=result,
        )
        return result

    @app.get("/api/models/{model_id}/download")
    @http_errors
    def download_model(model_id: str, request: Request = None):
        user = current_user(request)
        model = require_model(model_id, user)
        run = governor.versions.run(model["experiment_id"])
        version = next((item for item in run["versions"] if item["role"] == "model"), None)
        with store.atomic():
            governor.models.require_available(model_id)
            lease_id = policies.create_lease("read", user and user["id"], model_id, seconds=300)
            governor.touch_model(model_id)
        try:
            if version:
                governor.versions.verify(version["id"])
                root = governor.versions.object_path(version["id"]) / "files"
            else:
                root = Path(model["bundle_path"])
            if not root.is_dir():
                raise KeyError(model_id)
        except Exception:
            policies.release_lease(lease_id)
            raise
        chunks = queue.Queue(maxsize=4)
        stopped = threading.Event()

        class Writer:
            refreshed = 0

            def write(self, content):
                for offset in range(0, len(content), 65536):
                    while not stopped.is_set():
                        if time.monotonic() - self.refreshed > 30:
                            from datetime import timedelta

                            with store.connect() as conn:
                                conn.execute(
                                    update(resource_leases)
                                    .where(resource_leases.c.id == lease_id)
                                    .values(
                                        expires_at=(now_utc() + timedelta(seconds=300)).isoformat()
                                    )
                                )
                            self.refreshed = time.monotonic()
                        try:
                            chunks.put(content[offset : offset + 65536], timeout=1)
                            break
                        except queue.Full:
                            continue
                    else:
                        raise BrokenPipeError("download disconnected")
                return len(content)

            def flush(self):
                pass

        def produce():
            try:
                with tarfile.open(mode="w|", fileobj=Writer()) as archive:
                    for member in sorted(root.rglob("*")):
                        if member.is_symlink():
                            raise ValueError("model bundle contains a symlink")
                        if member.is_file():
                            archive.add(
                                member, arcname=str(member.relative_to(root)), recursive=False
                            )
                    from pepplatform.versioning.web import public_payload

                    manifest = json.dumps(public_payload(run), ensure_ascii=False).encode()
                    info = tarfile.TarInfo("versions.download.json")
                    info.size = len(manifest)
                    archive.addfile(info, io.BytesIO(manifest))
            except Exception as exc:  # noqa: BLE001 -- Propagate producer failures to the download stream.
                while not stopped.is_set():
                    try:
                        chunks.put(exc, timeout=1)
                        break
                    except queue.Full:
                        pass
            finally:
                while not stopped.is_set():
                    try:
                        chunks.put(None, timeout=1)
                        break
                    except queue.Full:
                        pass
                policies.release_lease(lease_id)

        def stream():
            producer = threading.Thread(
                target=produce, name="pepplatform-model-download", daemon=True
            )
            producer.start()
            try:
                while True:
                    part = chunks.get()
                    if part is None:
                        return
                    if isinstance(part, Exception):
                        raise part
                    yield part
            finally:
                stopped.set()
                policies.release_lease(lease_id)

        return StreamingResponse(
            stream(),
            media_type="application/x-tar",
            headers={
                "Content-Disposition": f'attachment; filename="model-{model_id}.tar"',
                "Cache-Control": "no-store",
            },
        )
