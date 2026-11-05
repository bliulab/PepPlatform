"""Application-only adapters; historical training/import closures stay unchanged."""

from contextvars import ContextVar
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import update

from pepplatform.web.ensemble_runner import WebExperimentRunner
from pepplatform.web.ensemble_service import WebModelService
from pepplatform.web.store import models, utcnow
from .core import VersionRepository, canonical, sha256
from .service import RunVersions


class VersionedExperimentRunner(WebExperimentRunner):
    def __init__(
        self,
        store,
        paths,
        before_experiment=None,
        after_experiment=None,
        versions=None,
        input_policy=None,
    ):
        super().__init__(store, paths, before_experiment, after_experiment)
        self.versions = versions or VersionRepository(store.engine, paths.root / "versions")
        self.version_service = RunVersions(
            self.versions, paths.project_root, input_policy=input_policy
        )
        self._versioning_run = ContextVar("versioning_run", default=None)

    def _run_experiment(self, experiment_id):
        experiment = self.store.get_experiment(experiment_id)
        if not experiment or experiment["status"] != "queued":
            return
        try:
            self.version_service.begin(experiment)
        except Exception as exc:
            self._record_preparation_failure(experiment_id, exc)
            return
        token = self._versioning_run.set(experiment_id)
        try:
            super()._run_experiment(experiment_id)
        finally:
            self._versioning_run.reset(token)
            self.version_service.finish(self.store.get_experiment(experiment_id))

    def build_command(self, experiment, dataset):
        command = super().build_command(experiment, dataset)
        # Command previews must not snapshot inputs or create run bindings.
        if self._versioning_run.get() != experiment["id"]:
            return command
        if experiment["config"].get("web_ensemble"):
            return self.version_service.prepare_ensemble(experiment, command)
        return self.version_service.prepare(experiment, command)

    def _register_model_if_present(self, experiment_id, experiment):
        record = super()._register_model_if_present(experiment_id, experiment)
        try:
            bundle = self.version_service.seal(experiment)
            if record and bundle:
                with self.store.connect() as conn:
                    conn.execute(
                        update(models)
                        .where(models.c.id == record["id"])
                        .values(bundle_path=str(bundle))
                    )
                return self.store.get_model(record["id"])
            return record
        except Exception as exc:
            # Training succeeded, but an unsealed new model must not be served.
            self.versions.update_run(experiment_id, status="seal_failed", error=str(exc))
            if record:
                with self.store.connect() as conn:
                    conn.execute(
                        update(models)
                        .where(models.c.id == record["id"])
                        .values(supports_inference=0)
                    )
                record = self.store.get_model(record["id"])
            return record


class VersionedModelStore:
    """Expose a verified working copy to inference, keeping cache writes out of CAS."""

    def __init__(self, store, versions):
        self.store = store
        self.versions = versions

    def __getattr__(self, name):
        return getattr(self.store, name)

    def get_model(self, model_id, **kwargs):
        record = self.store.get_model(model_id, **kwargs)
        if not record:
            return record
        run = self.versions.run(record["experiment_id"])
        if run["status"] == "not_recorded":
            return record
        model = next((v for v in run["versions"] if v["role"] == "model"), None)
        if run["status"] != "sealed" or not model:
            raise ValueError("model version has not been sealed")
        self.versions.verify(model["id"])
        for view in run.get("details", {}).get("input_views", []):
            identity = self.versions.get(view["version_id"])["identity"]
            for name, expected in identity["files"].items():
                path = Path(view["path"]) / name
                if not path.is_file() or sha256(path) != expected["sha256"]:
                    raise ValueError("retained model dependency changed")
        view = self.versions.root / "inference" / record["id"]
        with self.versions.locked():
            if not view.exists():
                self.versions.materialize(model["id"], view)
            for name, expected in model["identity"]["files"].items():
                path = view / name
                if not path.is_file() or path.is_symlink() or sha256(path) != expected["sha256"]:
                    raise ValueError("inference working copy changed")
        return {**record, "bundle_path": str(view)}


class VersionedModelService(WebModelService):
    def __init__(self, store, versions):
        super().__init__(VersionedModelStore(store, versions))


def public_payload(value):
    """Do not expose storage paths, commands, host environments, or local errors."""
    hidden = {
        "owner_user_id",
        "command",
        "requested_command",
        "input_views",
        "path_identities",
        "error",
        "resume_capture_error",
    }
    if isinstance(value, dict):
        return {k: public_payload(v) for k, v in value.items() if k not in hidden}
    if isinstance(value, list):
        return [public_payload(x) for x in value]
    if isinstance(value, str) and (value.startswith("/") or ":\\" in value):
        return "<local path>"
    return value


def register_version_routes(app, repository, current_user, require_experiment):
    def visible_run(run_id, user):
        try:
            return require_experiment(run_id, user)
        except HTTPException as exc:
            if exc.status_code in {403, 404}:
                return None
            raise

    def accessible_runs(identifier, user):
        return [
            record
            for run_id in repository.referencing_runs(identifier)
            if (record := visible_run(run_id, user)) is not None
        ]

    @app.get("/api/experiments/{experiment_id}/versions")
    def experiment_versions(experiment_id: str, request: Request, download: bool = False):
        user = current_user(request)
        require_experiment(experiment_id, user)
        payload = public_payload(repository.run(experiment_id))
        payload["relations"] = [
            edge
            for edge in payload["relations"]
            if edge["relation"] not in {"rerun_of", "reuses_representation_from"}
            or visible_run(edge["target"], user)
        ]
        if download:
            return Response(
                canonical(payload),
                media_type="application/json",
                headers={"Content-Disposition": 'attachment; filename="versions.json"'},
            )
        return payload

    @app.get("/api/versions/{version_id}")
    def content_version(version_id: str, request: Request):
        if not accessible_runs(version_id, current_user(request)):
            raise HTTPException(404, detail="version not found")
        return public_payload(repository.get(version_id))

    @app.get("/api/versions/{version_id}/runs")
    def version_runs(version_id: str, request: Request):
        records = accessible_runs(version_id, current_user(request))
        if not records:
            raise HTTPException(404, detail="version not found")
        return {
            "items": [{"id": x["id"], "name": x["name"], "status": x["status"]} for x in records]
        }
