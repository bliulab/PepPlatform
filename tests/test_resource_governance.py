"""Concurrency, retention and scientific-boundary contracts for Web governance."""

import asyncio
import hashlib
import io
import json
import os
import queue
import subprocess
import sys
import tarfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, insert, select
from sqlalchemy.engine import make_url

from pepplatform.versioning.core import VersionRepository
from pepplatform.versioning.web import VersionedModelStore
from pepplatform.web.app import create_app
from pepplatform.web.auth import SESSION_COOKIE_NAME, secret_hash, session_expires_at
from pepplatform.web.resource_governor import BudgetedVersionRepository
from pepplatform.web.resource_inference import InferenceProcess
from pepplatform.web.resource_policy import (
    DAILY_QUOTA_FIELDS,
    DEFAULT_QUOTAS,
    PolicyConflict,
    ResourceLimitError,
    resource_cleanup_logs,
    resource_pending_jobs,
    resource_quota_changes,
)
from pepplatform.web.runner import ExperimentRunner
from pepplatform.web.store import user_quotas


def endpoint(app, path, method):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == path and method in route.methods
    )


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PEP_AUTH_ENABLED", "1")
    monkeypatch.setenv("PEP_ENV", "development")
    monkeypatch.setenv("PEP_AUTH_DEBUG_CODES", "1")
    database_url = os.environ.get("PEP_RESOURCE_TEST_DATABASE_URL")
    database_engine, schema = None, None
    if database_url:
        database_engine = create_engine(database_url)
        schema = "resource_test_" + uuid.uuid4().hex
        with database_engine.begin() as conn:
            conn.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        isolated_url = make_url(database_url).update_query_dict(
            {"options": "-csearch_path=" + schema}
        )
        monkeypatch.setenv("PEP_DATABASE_URL", isolated_url.render_as_string(hide_password=False))
    else:
        monkeypatch.delenv("PEP_DATABASE_URL", raising=False)
    application = create_app(runtime_root=tmp_path / "runtime")
    application.state.runner.stop()
    yield application
    application.state.runner.stop()
    application.state.models.close()
    application.state.store.engine.dispose()
    if database_engine:
        with database_engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        database_engine.dispose()


def user(app, name="user", *, admin=False):
    store = app.state.store
    value = store.create_user(
        f"{name}@example.test", "unused", email_verified=True, role="admin" if admin else "user"
    )
    token = f"test-session-{value['id']}"
    settings = app.state.auth_settings
    store.create_session(
        value["id"], secret_hash(token, settings.session_secret), session_expires_at(settings)
    )
    return value, SimpleNamespace(cookies={SESSION_COOKIE_NAME: token}, headers={}, client=None)


def dataset(app, owner):
    root = app.state.paths.datasets / owner["id"]
    for split, label in (("train", 1), ("val", 0)):
        (root / split).mkdir(parents=True)
        (root / split / "seqs.fasta").write_text(f">{split}\nACDEFG\n")
        (root / split / "labels.csv").write_text(f"active\n{label}\n")
    return app.state.store.create_dataset(
        "demo",
        "pep_function",
        "upload",
        "sequence_dir",
        root,
        metadata={"splits": {"train": 1, "val": 1}},
        owner_user_id=owner["id"],
    )


def model(app, owner, name="one", *, data=None):
    store, versions = app.state.store, app.state.versions
    data = data or dataset(app, owner)
    directory = app.state.paths.experiments / name
    artifact = directory / "artifact"
    artifact.mkdir(parents=True)
    metadata = {
        "bundle_type": "sequence_classical",
        "args": {"tokenizer": "kmer", "encoder": "onehot", "classifier": "rf"},
    }
    (artifact / "metadata.json").write_text(json.dumps(metadata))
    (artifact / "model.joblib").write_bytes(b"retained scientific weights")
    store.create_experiment(
        name,
        data["id"],
        "pep_function",
        metadata["args"],
        directory / "train.log",
        artifact,
        experiment_id=name,
        owner_user_id=owner["id"],
    )
    store.update_experiment(name, status="succeeded")
    versions.start_run(name, owner["id"])
    version = versions.snapshot("model", {path.name: path for path in artifact.iterdir()})
    versions.bind(name, "model", version["id"])
    versions.update_run(name, status="sealed", input_views=[])
    record = store.create_model(
        name,
        name,
        "pep_function",
        versions.object_path(version["id"]) / "files",
        True,
        metadata,
        owner_user_id=owner["id"],
    )
    VersionedModelStore(store, versions).get_model(record["id"])
    return record, version


def change_policy(app, **groups):
    policies = app.state.store.resources
    return policies.save_policy({"revision": policies.policy()["revision"], **groups})


def test_personal_inheritance_and_opt_in_global_overwrite(app):
    store = app.state.store
    protected, _ = user(app, "protected")
    writable, _ = user(app, "writable")
    inherited, _ = user(app, "inherited")
    store.set_user_quota(protected["id"], {"daily_experiment_limit": 30})
    store.set_user_quota(
        writable["id"],
        {"daily_experiment_limit": 20, "storage_limit_bytes": 12345, "allow_global_override": True},
    )
    payload = {"revision": 1, "defaults": {"daily_experiment_limit": 4}, "apply_to_users": True}
    preview = store.resources.preview(payload)
    assert preview["affected_count"] == preview["skipped_count"] == 1
    result = store.resources.save_policy({**payload, "preview_token": preview["preview_token"]})
    assert result["policy"]["defaults"]["daily_experiment_limit"] == 4
    assert store.get_user_quota(protected["id"])["daily_experiment_limit"] == 30
    assert store.get_user_quota(writable["id"])["sources"]["daily_experiment_limit"] == "global"
    assert store.get_user_quota(writable["id"])["storage_limit_bytes"] == 12345
    assert store.get_user_quota(inherited["id"])["daily_experiment_limit"] == 10
    assert store.get_user_quota(inherited["id"])["pending"]["daily_experiment_limit"]["value"] == 4
    future, _ = user(app, "future")
    assert store.get_user_quota(future["id"])["daily_experiment_limit"] == 4
    store.set_user_quota(protected["id"], {"daily_experiment_limit": None})
    assert store.get_user_quota(protected["id"])["daily_experiment_limit"] == 30
    assert store.get_user_quota(protected["id"])["configured"]["daily_experiment_limit"] == 4


def test_existing_quotas_are_preserved_and_protected(app):
    owner, _ = user(app)
    with app.state.store.connect() as conn:
        conn.execute(
            insert(user_quotas).values(
                user_id=owner["id"],
                daily_experiment_limit=77,
                daily_concurrent_experiment_limit=3,
                daily_inference_request_limit=200,
                daily_inference_input_limit=10000,
                updated_at="2026-01-01",
            )
        )
    change_policy(app, defaults={"daily_experiment_limit": 2})
    quota = app.state.store.get_user_quota(owner["id"])
    assert quota["daily_experiment_limit"] == 77 and not quota["allow_global_override"]
    assert quota["daily_training_seconds_limit"] == DEFAULT_QUOTAS["daily_training_seconds_limit"]


def test_bulk_preview_detects_concurrent_personal_edits(app):
    owner, _ = user(app)
    app.state.store.set_user_quota(
        owner["id"], {"daily_experiment_limit": 20, "allow_global_override": True}
    )
    payload = {"revision": 1, "defaults": {"daily_experiment_limit": 5}, "apply_to_users": True}
    preview = app.state.store.resources.preview(payload)
    app.state.store.set_user_quota(owner["id"], {"allow_global_override": False})
    with pytest.raises(PolicyConflict):
        app.state.store.resources.save_policy(
            {**payload, "preview_token": preview["preview_token"]}
        )
    assert app.state.store.resources.policy()["revision"] == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"daily_experiment_limit": True},
        {"max_experiment_seconds": 1.2},
        {"storage_limit_bytes": -1},
        {"model_retention_days": 0},
        {"allow_global_override": "true"},
    ],
)
def test_invalid_personal_quotas_do_not_change_settings(app, updates):
    owner, _ = user(app)
    with pytest.raises(ValueError):
        app.state.store.set_user_quota(owner["id"], updates)
    assert app.state.store.get_user_quota(owner["id"])["overrides"] == {}


def test_training_admission_is_atomic_across_requests(app, quota_clock):
    owner, request = user(app)
    data = dataset(app, owner)
    app.state.store.set_user_quota(owner["id"], {"daily_experiment_limit": 1})
    quota_clock.next_day()
    create = endpoint(app, "/api/experiments", "POST")
    payload = {
        "dataset_id": data["id"],
        "config": {"tokenizer": "kmer", "encoder": "onehot", "model": "cnn"},
    }

    def submit(_):
        try:
            return create(payload, request)["id"]
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(submit, range(4)))
    assert sum(isinstance(value, str) for value in results) == 1
    assert results.count(429) == 3
    assert app.state.store.get_quota_usage(owner["id"])["experiments_created"] == 1
    assert len(app.state.store.list_experiments(user=owner)) == 1
    with app.state.store.connect() as conn:
        assert len(conn.execute(select(resource_pending_jobs)).all()) == 1


def test_public_registration_limit_is_atomic_and_failed_validation_refunds(app):
    change_policy(app, limits={"daily_registration_limit": 1})
    register = endpoint(app, "/api/auth/register", "POST")
    with pytest.raises(HTTPException):
        register({"email": "invalid", "password": "short"})

    def submit(index):
        try:
            return register({"email": f"new-{index}@example.test", "password": "password123"})[
                "user"
            ]["id"]
        except HTTPException as exc:
            return exc.status_code

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(submit, range(3)))
    assert sum(isinstance(value, str) for value in results) == 1
    assert results.count(429) == 2
    assert app.state.store.resources.daily("__public_registration__")["registrations"] == 1
    user(app, "bootstrap", admin=True)
    assert app.state.store.resources.daily("__public_registration__")["registrations"] == 1


def test_inference_reservations_prevent_parallel_overcommit_and_settle_once(app, quota_clock):
    owner, _ = user(app)
    store, governor = app.state.store, app.state.resource_governor
    store.set_user_quota(owner["id"], {"daily_inference_request_limit": 1})
    quota_clock.next_day()
    lease = governor.reserve_inference(owner, 2)
    with pytest.raises(ResourceLimitError):
        governor.reserve_inference(owner, 1)
    governor.settle_inference(lease, success=True)
    governor.settle_inference(lease, success=True)
    usage = store.get_quota_usage(owner["id"])
    assert usage["inference_requests"] == 1 and usage["inference_inputs"] == 2


def test_runtime_accounting_splits_midnight_and_is_idempotent(app, monkeypatch):
    from pepplatform.web import resource_governor, resource_policy

    owner, _ = user(app)
    data = dataset(app, owner)
    experiment = app.state.store.create_experiment(
        "runtime",
        data["id"],
        "pep_function",
        {},
        "/unused/log",
        "/unused/artifact",
        owner_user_id=owner["id"],
    )
    start = datetime(2026, 11, 7, 23, 59, 59, tzinfo=timezone.utc)
    monkeypatch.setattr(resource_governor, "now_utc", lambda: start)
    monkeypatch.setattr(resource_policy, "now_utc", lambda: start)
    governor = app.state.resource_governor
    governor.start_execution(experiment)
    end = start + timedelta(seconds=3)
    governor.account_execution(experiment["id"], finished=True, at=end)
    governor.account_execution(experiment["id"], finished=True, at=end)
    assert governor.policies.daily(owner["id"], "2026-11-07")["training_milliseconds"] == 1000
    assert governor.policies.daily(owner["id"], "2026-11-08")["training_milliseconds"] == 2000


def test_gpu_lease_is_exclusive_and_exception_releases_it(app):
    governor = app.state.resource_governor
    with pytest.raises(RuntimeError), governor.gpu(training=True):
        with pytest.raises(ResourceLimitError) as error, governor.gpu():
            pass
        assert error.value.detail["code"] == "gpu_busy_training"
        raise RuntimeError("task failed")
    with governor.gpu():
        pass


def expired_cache(app, monkeypatch, *, publish_announcements=True, advance=True):
    owner, _ = user(app)
    record, version = model(app, owner)
    cache = app.state.versions.root / "inference" / record["id"]
    old = datetime.now(timezone.utc) - timedelta(days=30)
    os.utime(cache, (old.timestamp(), old.timestamp()))
    change_policy(app, retention={"enabled": True, "publish_announcements": publish_announcements})
    governor = app.state.resource_governor
    governor.scan()
    entry = next(
        row
        for row in governor.inventory()
        if row["model_id"] == record["id"] and row["kind"] == "inference_cache"
    )
    entry = governor._schedule_cleanup(entry["id"], publish_announcements=publish_announcements)
    assert entry["state"] == ("notified" if publish_announcements else "scheduled")
    assert entry["notified_at"] == (entry["scheduled_at"] if publish_announcements else None)
    assert datetime.fromisoformat(entry["delete_after"]) - datetime.fromisoformat(
        entry["scheduled_at"]
    ) == timedelta(hours=72 if publish_announcements else 0)
    future = datetime.fromisoformat(entry["delete_after"]) + timedelta(seconds=1)
    from pepplatform.web import resource_governor, resource_policy

    if advance and publish_announcements:
        monkeypatch.setattr(resource_governor, "now_utc", lambda: future)
        monkeypatch.setattr(resource_policy, "now_utc", lambda: future)
    return owner, record, version, cache, entry


@pytest.mark.parametrize("publish", [False, True])
def test_cleanup_retains_all_scientific_versions_and_reconstructs_copy(app, monkeypatch, publish):
    owner, record, version, cache, entry = expired_cache(
        app, monkeypatch, publish_announcements=publish
    )
    weights = app.state.versions.object_path(version["id"]) / "files" / "model.joblib"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert app.state.resource_governor.cleanup(entry["id"])["deleted"]
    assert not cache.exists() and hashlib.sha256(weights.read_bytes()).hexdigest() == digest
    assert app.state.versions.verify(version["id"])["status"] == "verified"
    rebuilt = VersionedModelStore(app.state.store, app.state.versions).get_model(record["id"])
    assert Path(rebuilt["bundle_path"], "model.joblib").read_bytes() == weights.read_bytes()
    notices = app.state.store.resources.announcements(owner)
    assert len(notices) == int(publish)
    if publish:
        assert notices[0]["kind"] == "cleanup"


@pytest.mark.parametrize("publish", [False, True])
def test_cleanup_skips_pinned_and_in_use_copies(app, monkeypatch, publish):
    owner, record, _, cache, entry = expired_cache(app, monkeypatch, publish_announcements=publish)
    governor = app.state.resource_governor
    governor.protect_model(record, True)
    with pytest.raises(ResourceLimitError):
        governor.cleanup(entry["id"])
    governor.protect_model(record, False)
    lease = governor.policies.create_lease("read", owner["id"], record["id"])
    with pytest.raises(ResourceLimitError):
        governor.cleanup(entry["id"])
    assert cache.exists()
    governor.policies.release_lease(lease)


def test_cleanup_failure_reports_zero_release_and_can_retry(app, monkeypatch):
    _, _, version, cache, entry = expired_cache(app, monkeypatch)
    from pepplatform.web import resource_governor

    original = resource_governor.shutil.rmtree

    def fail(_path):
        raise OSError("simulated read-only volume")

    monkeypatch.setattr(resource_governor.shutil, "rmtree", fail)
    result = app.state.resource_governor.cleanup(entry["id"])
    assert not result["deleted"] and result["released_bytes"] == 0 and cache.exists()
    with app.state.store.connect() as conn:
        assert conn.execute(select(resource_cleanup_logs.c.status)).scalar_one() == "failed"
    monkeypatch.setattr(resource_governor.shutil, "rmtree", original)
    assert app.state.resource_governor.cleanup(entry["id"])["deleted"]
    assert app.state.resource_governor.cleanup(entry["id"])["released_bytes"] == 0
    assert app.state.versions.verify(version["id"])["status"] == "verified"


def test_direct_cache_cleanup_requires_enabled_policy_and_idle_expiry(app):
    owner, _ = user(app)
    model(app, owner)
    governor = app.state.resource_governor
    governor.scan()
    entry = next(row for row in governor.inventory() if row["kind"] == "inference_cache")
    with pytest.raises(ResourceLimitError, match="resource_cleanup_disabled"):
        governor.cleanup(entry["id"])
    change_policy(app, retention={"enabled": True})
    with pytest.raises(ResourceLimitError, match="retention_not_due"):
        governor.cleanup(entry["id"])


@pytest.mark.parametrize("publish", [False, True])
def test_cleanup_refuses_changed_retained_inputs(app, monkeypatch, tmp_path, publish):
    _, record, _, cache, entry = expired_cache(app, monkeypatch, publish_announcements=publish)
    source = tmp_path / "retained.bin"
    source.write_bytes(b"original input")
    versions = app.state.versions
    dependency = versions.snapshot("resource", {"input.bin": source})
    view = versions.root / "workspaces" / record["experiment_id"] / "input"
    versions.materialize(dependency["id"], view)
    versions.update_run(
        record["experiment_id"], input_views=[{"path": str(view), "version_id": dependency["id"]}]
    )
    (view / "input.bin").write_bytes(b"changed input")
    with pytest.raises(ValueError, match="retained model dependency"):
        app.state.resource_governor.cleanup(entry["id"])
    assert cache.exists()


@pytest.mark.parametrize(
    "task", ["pep_function", "pep_hlai", "pep_hlaii", "pep_protein_interaction"]
)
@pytest.mark.parametrize(
    "semantic,loss", [(False, None), (True, "legacy"), (True, "jaccard_supcon")]
)
@pytest.mark.parametrize("mode", ["next_reset", "immediate"])
@pytest.mark.parametrize("publish", [False, True])
@pytest.mark.parametrize("email_state", ["enabled", "exhausted", "paused"])
def test_governance_never_rewrites_scientific_training_commands(
    app, task, semantic, loss, mode, publish, email_state, quota_clock
):
    config = {
        "seed": 42,
        "epochs": 123,
        "batch_size": 37,
        "tokenizer": "kmer",
        "encoder": "onehot",
        "model": "cnn",
        "enable_semantic_similarity": semantic,
        "semantic_similarity_weight": 0.17,
        "semantic_positive_partners": 7,
        "semantic_zero_partners": 11,
    }
    if loss is not None:
        config["semantic_loss_type"] = loss
    original_config = json.dumps(config, sort_keys=True)
    experiment = {
        "id": "scientific-boundary",
        "task_type": task,
        "config": config,
        "artifact_dir": str(app.state.paths.experiments / "scientific" / "artifact"),
    }
    data = {
        "root_path": str(app.state.paths.datasets / "scientific"),
        "format": "sequence_dir" if task == "pep_function" else "pair_split_dir",
    }
    original = ExperimentRunner(app.state.store, app.state.paths).build_command(experiment, data)
    owner, _ = user(app)
    if email_state == "exhausted":
        for _ in range(20):
            app.state.store.resources.reserve_verification_email()
    change_policy(
        app,
        defaults={key: 0 for key in DAILY_QUOTA_FIELDS},
        limits={"daily_verification_email_limit": 0 if email_state == "paused" else 20},
        daily_reduction_mode=mode,
        publish_announcements=publish,
    )
    actual = app.state.runner.build_command(experiment, data)
    assert actual == original
    assert json.dumps(experiment["config"], sort_keys=True) == original_config
    assert len(app.state.store.resources.announcements(owner)) == int(publish)
    quota_clock.next_day()
    assert app.state.runner.build_command(experiment, data) == original
    assert json.dumps(experiment["config"], sort_keys=True) == original_config


@pytest.mark.parametrize("mode", ["next_reset", "immediate"])
@pytest.mark.parametrize("publish", [False, True])
def test_budgeted_snapshots_keep_the_same_content_identity(app, tmp_path, mode, publish):
    source = tmp_path / "input"
    source.write_bytes(b"unchanged scientific input")
    plain = VersionRepository(app.state.store.engine, tmp_path / "plain-versions")
    budgeted = app.state.versions
    assert isinstance(budgeted, BudgetedVersionRepository)
    owner, _ = user(app)
    accepted = app.state.store.get_user_quota(owner["id"])
    app.state.store.set_user_quota(
        owner["id"],
        {
            "daily_training_seconds_limit": 0,
            "max_model_file_bytes": 0,
            "daily_reduction_mode": mode,
            "publish_announcements": publish,
        },
    )
    with budgeted.write_budget(owner, accepted):
        actual = budgeted.snapshot("resource", {"input.bin": source})
    expected = plain.snapshot("resource", {"input.bin": source})
    assert actual["id"] == expected["id"] and actual["identity"] == expected["identity"]


def test_announcements_are_scoped_and_read_receipts_are_private(app):
    first, first_request = user(app, "first")
    _, second_request = user(app, "second")
    _, admin_request = user(app, "admin", admin=True)
    create = endpoint(app, "/api/admin/announcements", "POST")
    with pytest.raises(HTTPException) as exc:
        create({"title": "test", "body": "message"}, first_request)
    assert exc.value.status_code == 403
    notice = create(
        {"title": "Storage policy", "body": "Keep original inputs.", "audience": [first["id"]]},
        admin_request,
    )
    listing = endpoint(app, "/api/announcements", "GET")
    assert listing(second_request)["items"] == []
    assert listing(first_request)["items"][0]["read"] is False
    assert "audience_labels" not in listing(first_request)["items"][0]
    admin_notice = endpoint(app, "/api/admin/announcements", "GET")(admin_request)["items"][0]
    assert admin_notice["audience_labels"][first["id"]] == first["email"]
    endpoint(app, "/api/announcements/{announcement_id}/read", "POST")(notice["id"], first_request)
    assert listing(first_request)["items"][0]["read"] is True
    with pytest.raises(HTTPException) as exc:
        endpoint(app, "/api/announcements/{announcement_id}/read", "POST")(
            notice["id"], second_request
        )
    assert exc.value.status_code == 404


def test_inference_timeout_terminates_owned_process(app, monkeypatch):
    change_policy(app, limits={"inference_timeout_seconds": 1})
    executor = InferenceProcess(app.state.store, app.state.paths, app.state.resource_governor)
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys,time;sys.stdin.readline();time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    executor._process, executor._events = process, queue.Queue()
    monkeypatch.setattr(executor, "_start", lambda: None)
    with pytest.raises(ResourceLimitError, match="inference_timeout"):
        executor.predict("dummy", sequences=["ACD"])
    assert process.poll() is not None


def test_disk_pressure_applies_even_to_a_personal_exception(app, monkeypatch):
    from pepplatform.web import resource_governor

    owner, _ = user(app)
    app.state.store.set_user_quota(owner["id"], {"storage_limit_bytes": 1000 * 1024**3})
    monkeypatch.setattr(
        resource_governor.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, used=95 * 1024**3, free=5 * 1024**3),
    )
    with pytest.raises(ResourceLimitError) as error:
        app.state.resource_governor.admit_training(owner, 1)
    assert error.value.status_code == 507


def test_completed_write_remains_charged_until_inventory(app):
    owner, _ = user(app)
    governor = app.state.resource_governor
    app.state.store.set_user_quota(owner["id"], {"storage_limit_bytes": 100})
    with governor.reserve_write(owner, 80):
        pass
    with pytest.raises(ResourceLimitError), governor.reserve_write(owner, 30):
        pass
    governor.scan()
    assert governor.storage_usage(owner["id"])["reserved_bytes"] == 0


def test_weight_download_is_streamed_and_scoped(app):
    owner, request = user(app)
    _, other_request = user(app, "other")
    record, version = model(app, owner)
    download = endpoint(app, "/api/models/{model_id}/download", "GET")
    with pytest.raises(HTTPException) as error:
        download(record["id"], other_request)
    assert error.value.status_code == 403
    response = download(record["id"], request)

    async def consume():
        return b"".join([chunk async for chunk in response.body_iterator])

    with tarfile.open(fileobj=io.BytesIO(asyncio.run(consume()))) as archive:
        assert archive.extractfile("model.joblib").read() == b"retained scientific weights"
        assert "versions.download.json" in archive.getnames()
    assert app.state.versions.verify(version["id"])["status"] == "verified"
    assert app.state.store.resources.leases(kind="read") == []


def test_legacy_model_records_are_never_automatically_retrained(app):
    owner, _ = user(app)
    data = dataset(app, owner)
    store = app.state.store
    root = app.state.paths.experiments / "legacy"
    experiment = store.create_experiment(
        "legacy",
        data["id"],
        "pep_function",
        {},
        root / "log",
        root / "artifact",
        owner_user_id=owner["id"],
    )
    store.create_model(
        experiment["id"],
        "legacy",
        "pep_function",
        root / "artifact",
        True,
        owner_user_id=owner["id"],
    )
    app.state.runner._recover()
    with store.connect() as conn:
        assert (
            conn.execute(
                select(resource_pending_jobs).where(
                    resource_pending_jobs.c.experiment_id == experiment["id"]
                )
            ).first()
            is None
        )
    assert app.state.versions.run(experiment["id"])["status"] == "not_recorded"


def test_durable_queue_rotates_users_without_changing_their_configs(app):
    owners = [user(app, "first")[0], user(app, "second")[0]]
    chosen = {}
    for index, owner in enumerate(owners):
        data = dataset(app, owner)
        for number in range(2):
            root = app.state.paths.experiments / f"queued-{index}-{number}"
            record = app.state.store.create_experiment(
                "queued",
                data["id"],
                "pep_function",
                {"seed": 42, "epochs": 123},
                root / "log",
                root / "artifact",
                owner_user_id=owner["id"],
            )
            app.state.runner.submit(record["id"])
            chosen[record["id"]] = owner["id"]
    first = app.state.runner._next()
    app.state.store.update_experiment(first, status="succeeded")
    second = app.state.runner._next()
    assert chosen[first] != chosen[second]
    assert app.state.store.get_experiment(second)["config"] == {"seed": 42, "epochs": 123}


def test_training_timeout_stops_its_process_and_keeps_result_incomplete(app, monkeypatch):
    from pepplatform.versioning.web import VersionedExperimentRunner

    owner, _ = user(app)
    data = dataset(app, owner)
    root = app.state.paths.experiments / "timeout"
    record = app.state.store.create_experiment(
        "timeout",
        data["id"],
        "pep_function",
        {},
        root / "log",
        root / "artifact",
        owner_user_id=owner["id"],
    )
    app.state.store.set_user_quota(owner["id"], {"max_experiment_seconds": 1})
    spawned = []

    def work(runner, identifier):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
        )
        spawned.append(process)
        runner._active_processes[identifier] = process
        runner.store.update_experiment(identifier, status="running")
        try:
            process.wait(timeout=10)
        finally:
            runner._active_processes.pop(identifier, None)

    monkeypatch.setattr(VersionedExperimentRunner, "_run_experiment", work)
    app.state.runner._run_experiment(record["id"])
    assert spawned[0].poll() is not None
    result = app.state.store.get_experiment(record["id"])
    assert result["status"] == "canceled" and "max_experiment_seconds" in result["error"]


def test_worker_file_limit_preserves_original_command_and_bounds_writes(tmp_path):
    output = tmp_path / "oversized.bin"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pepplatform.web.worker_limits",
            "--max-file-bytes",
            "1024",
            "--",
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'x'*2048)",
            str(output),
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert output.stat().st_size <= 1024


# Model allowances count catalogue entries, not the number of physical copies.
def _age_model(app, record):
    from pepplatform.web.resource_policy import model_protection
    from pepplatform.web.store import models

    at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    with app.state.store.connect() as conn:
        conn.execute(models.update().where(models.c.id == record["id"]).values(created_at=at))
        conn.execute(
            model_protection.update()
            .where(model_protection.c.model_id == record["id"])
            .values(updated_at=at)
        )


def _model_notice_due(app, record, *, publish_announcements=True):
    from pepplatform.web.resource_policy import model_retention

    _age_model(app, record)
    change_policy(
        app,
        retention={
            "enabled": True,
            "delete_unprotected_models": True,
            "publish_announcements": publish_announcements,
        },
    )
    app.state.resource_governor.models._schedule_cleanup(
        record["id"], publish_announcements=publish_announcements
    )
    with app.state.store.connect() as conn:
        conn.execute(
            model_retention.update()
            .where(model_retention.c.model_id == record["id"])
            .values(
                scheduled_at=(datetime.now(timezone.utc) - timedelta(hours=73)).isoformat()
                if publish_announcements
                else datetime.now(timezone.utc).isoformat(),
                notified_at=(datetime.now(timezone.utc) - timedelta(hours=73)).isoformat()
                if publish_announcements
                else None,
                delete_after=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
                if publish_announcements
                else datetime.now(timezone.utc).isoformat(),
            )
        )


def test_model_total_and_protection_allowances_are_separate(app):
    owner, request = user(app)
    data = dataset(app, owner)
    entries = [model(app, owner, f"saved-{i}", data=data)[0] for i in range(5)]
    governor = app.state.resource_governor
    for record in entries[:3]:
        governor.protect_model(record, True)
    usage = endpoint(app, "/api/resources/me", "GET")(request)["models"]
    assert usage == {
        "retained": 5,
        "protected": 3,
        "reserved": 0,
        "remaining": 0,
        "protect_remaining": 0,
        "over_limit": 0,
        "protection_over_limit": 0,
    }
    with pytest.raises(ResourceLimitError) as count:
        governor.models.admit(owner["id"], 1)
    assert count.value.detail["quota_type"] == "max_model_count"
    with pytest.raises(ResourceLimitError) as protected:
        governor.protect_model(entries[3], True)
    assert protected.value.detail["quota_type"] == "protected_model_limit"
    governor.protect_model(entries[0], False)
    governor.protect_model(entries[3], True)
    assert governor.models.usage(owner["id"])["protected"] == 3
    assert all("retention" in r for r in app.state.store.list_models(user=owner))


def test_model_slots_are_reserved_atomically_and_released_on_cancel(app):
    owner, request = user(app)
    data = dataset(app, owner)
    app.state.store.set_user_quota(
        owner["id"],
        {"max_model_count": 1, "protected_model_limit": 1, "daily_concurrent_experiment_limit": 5},
    )
    create = endpoint(app, "/api/experiments", "POST")
    payload = {
        "dataset_id": data["id"],
        "config": {"tokenizer": "kmer", "encoder": "onehot", "model": "cnn"},
    }

    def submit(_):
        try:
            return create(payload, request)
        except HTTPException as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, range(2)))
    accepted = [r for r in results if not isinstance(r, HTTPException)]
    rejected = [r for r in results if isinstance(r, HTTPException)]
    assert len(accepted) == len(rejected) == 1
    assert rejected[0].detail["quota_type"] == "max_model_count"
    assert app.state.resource_governor.models.usage(owner["id"])["reserved"] == 1
    app.state.runner.cancel(accepted[0]["id"])
    assert app.state.resource_governor.models.usage(owner["id"])["remaining"] == 1


def test_effective_protection_limit_cannot_exceed_total(app):
    owner, _ = user(app)
    with pytest.raises(ValueError, match="protected_model_limit"):
        app.state.store.set_user_quota(owner["id"], {"max_model_count": 2})
    with pytest.raises(ValueError, match="protected_model_limit"):
        change_policy(app, defaults={"max_model_count": 2})
    app.state.store.set_user_quota(owner["id"], {"protected_model_limit": 4})
    with pytest.raises(ValueError, match="Invalid model limits"):
        change_policy(app, defaults={"max_model_count": 3})
    assert app.state.store.get_user_quota(owner["id"])["max_model_count"] == 5


def test_old_protection_settings_are_preserved_when_adding_model_count(app):
    from pepplatform.web.resource_policy import dump, platform_policies, user_resource_settings

    owner, _ = user(app)
    policy = app.state.store.resources.policy()
    policy["defaults"].pop("max_model_count")
    policy["defaults"]["protected_model_limit"] = 8
    policy["retention"].pop("delete_unprotected_models")
    policy["retention"]["enabled"] = True
    with app.state.store.connect() as conn:
        conn.execute(platform_policies.update().values(config_json=dump(policy)))
        conn.execute(
            insert(user_resource_settings).values(
                user_id=owner["id"],
                overrides_json=dump({"protected_model_limit": 9}),
                allow_global_override=0,
                updated_at="2026-01-01",
            )
        )
    app.state.store.resources.init_schema()
    quota = app.state.store.get_user_quota(owner["id"])
    assert quota["protected_model_limit"] == quota["max_model_count"] == 9
    assert not app.state.store.resources.policy()["retention"]["delete_unprotected_models"]
    assert app.state.store.resources.policy()["retention"]["enabled"]


def test_lowering_model_quota_preserves_existing_models_and_protection(app):
    owner, _ = user(app)
    data = dataset(app, owner)
    entries = [model(app, owner, f"existing-{i}", data=data)[0] for i in range(3)]
    for record in entries:
        app.state.resource_governor.protect_model(record, True)
    app.state.store.set_user_quota(owner["id"], {"max_model_count": 1, "protected_model_limit": 1})
    usage = app.state.resource_governor.models.usage(owner["id"])
    assert usage["over_limit"] == usage["protection_over_limit"] == 2
    assert len(app.state.store.list_models(user=owner)) == 3
    app.state.resource_governor.protect_model(entries[0], True)  # Idempotent retry remains valid.


@pytest.mark.parametrize("publish", [False, True])
def test_model_cleanup_is_opt_in_and_only_announced_plans_wait(app, publish):
    owner, _ = user(app)
    record, version = model(app, owner)
    _age_model(app, record)
    change_policy(app, retention={"enabled": True, "publish_announcements": publish})
    control = app.state.resource_governor.models
    control.tick()
    assert control.state(record["id"]) is None
    change_policy(app, retention={"delete_unprotected_models": True})
    control.tick()
    state = control.state(record["id"])
    assert datetime.fromisoformat(state["delete_after"]) - datetime.fromisoformat(
        state["scheduled_at"]
    ) == timedelta(hours=72 if publish else 0)
    if not publish:
        assert state["state"] == "removed"
        assert app.state.store.get_model(record["id"]) is None
        return
    with pytest.raises(ResourceLimitError, match="cleanup_notice_pending"):
        control.cleanup(record["id"])
    change_policy(app, retention={"delete_unprotected_models": False})
    assert control.state(record["id"])["delete_after"] is None
    assert app.state.versions.verify(version["id"])["status"] == "verified"


@pytest.mark.parametrize("publish", [False, True])
def test_due_model_cleanup_removes_exclusive_weights_but_preserves_history_and_inputs(app, publish):
    owner, request = user(app)
    record, version = model(app, owner)
    store, versions = app.state.store, app.state.versions
    experiment = store.get_experiment(record["experiment_id"])
    data = store.get_dataset(experiment["dataset_id"])
    data_file = Path(data["root_path"]) / "train/seqs.fasta"
    input_version = versions.snapshot("dataset", {"seqs.fasta": data_file})
    versions.bind(record["experiment_id"], "input:data_dir", input_version["id"])
    checkpoint = versions.snapshot(
        "checkpoint", {"model.joblib": Path(experiment["artifact_dir"]) / "model.joblib"}
    )
    versions.bind(record["experiment_id"], "checkpoint", checkpoint["id"])
    history = store.create_inference_run(
        record["id"],
        {"sequences": ["ACD"]},
        {"labels": ["x"], "predictions": [{"scores": {"x": 0.5}}]},
        owner_user_id=owner["id"],
    )
    _model_notice_due(app, record, publish_announcements=publish)
    result = app.state.resource_governor.models.cleanup(record["id"])
    assert result["deleted"]
    assert not versions.object_path(version["id"]).exists()
    assert not versions.object_path(checkpoint["id"]).exists()
    assert not Path(experiment["artifact_dir"]).exists()
    assert data_file.read_text() == ">train\nACDEFG\n"
    assert versions.verify(input_version["id"])["status"] == "verified"
    assert store.get_experiment(experiment["id"])["status"] == "succeeded"
    assert store.get_model(record["id"]) is None
    assert store.get_model(record["id"], include_removed=True)["retention"]["state"] == "removed"
    assert store.get_inference_run(history["id"])
    with pytest.raises(ResourceLimitError) as gone:
        endpoint(app, "/api/models/{model_id}/predict", "POST")(
            record["id"], {"sequences": ["ACD"]}, request
        )
    assert gone.value.status_code == 410
    with pytest.raises(HTTPException) as download_gone:
        endpoint(app, "/api/models/{model_id}/download", "GET")(record["id"], request)
    assert download_gone.value.status_code == 410
    assert endpoint(app, "/api/models/{model_id}/inference-runs", "GET")(record["id"], request)
    assert app.state.resource_governor.models.usage(owner["id"])["remaining"] == 5
    assert app.state.resource_governor.models.cleanup(record["id"])["released_bytes"] == 0


def test_model_cleanup_preserves_shared_content_and_protected_models(app):
    owner, _ = user(app)
    data = dataset(app, owner)
    first, one = model(app, owner, "first-shared", data=data)
    second, two = model(app, owner, "second-shared", data=data)
    assert one["id"] == two["id"]
    app.state.resource_governor.protect_model(second, True)
    _model_notice_due(app, first)
    assert app.state.resource_governor.models.cleanup(first["id"])["deleted"]
    assert app.state.versions.verify(two["id"])["status"] == "verified"
    assert VersionedModelStore(app.state.store, app.state.versions).get_model(second["id"])
    assert app.state.resource_governor.models.usage(owner["id"])["protected"] == 1


def test_download_or_prediction_lease_defers_model_cleanup_and_pin_cancels_notice(app):
    owner, _ = user(app)
    record, _ = model(app, owner)
    _model_notice_due(app, record)
    governor = app.state.resource_governor
    lease = app.state.store.resources.create_lease("read", owner["id"], record["id"])
    with pytest.raises(ResourceLimitError, match="resource_is_retained_or_in_use"):
        governor.models.cleanup(record["id"])
    app.state.store.resources.release_lease(lease)
    governor.protect_model(record, True)
    assert governor.models.state(record["id"])["state"] == "active"
    assert governor.models.state(record["id"])["delete_after"] is None


@pytest.mark.parametrize("publish", [False, True])
def test_model_cleanup_partial_failure_is_retryable(app, monkeypatch, publish):
    import pepplatform.web.model_retention as module

    owner, _ = user(app)
    record, version = model(app, owner)
    _model_notice_due(app, record, publish_announcements=publish)
    real_remove = module.shutil.rmtree

    def fail_partial(path):
        (Path(path) / "files/model.joblib").unlink()
        raise OSError("test interrupted removal")

    monkeypatch.setattr(
        module,
        "shutil",
        SimpleNamespace(
            rmtree=fail_partial, copy2=module.shutil.copy2, disk_usage=module.shutil.disk_usage
        ),
    )
    result = app.state.resource_governor.models.cleanup(record["id"])
    assert not result["deleted"] and result["released_bytes"] == 0
    assert app.state.store.get_model(record["id"])["retention"]["state"] == "failed"
    monkeypatch.setattr(module.shutil, "rmtree", real_remove)
    assert app.state.resource_governor.models.cleanup(record["id"])["deleted"]
    assert not app.state.versions.object_path(version["id"]).exists()


@pytest.mark.parametrize("publish", [False, True])
def test_model_cleanup_crash_intent_is_durable(app, monkeypatch, publish):
    import pepplatform.web.model_retention as module

    owner, _ = user(app)
    record, _ = model(app, owner)
    _model_notice_due(app, record, publish_announcements=publish)
    real_remove = module.shutil.rmtree

    def interrupted(_path):
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        module,
        "shutil",
        SimpleNamespace(
            rmtree=interrupted, copy2=module.shutil.copy2, disk_usage=module.shutil.disk_usage
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        app.state.resource_governor.models.cleanup(record["id"])
    assert app.state.resource_governor.models.state(record["id"])["state"] == "cleaning"
    monkeypatch.setattr(module.shutil, "rmtree", real_remove)
    assert app.state.resource_governor.models.cleanup(record["id"])["deleted"]


def test_system_exclusion_prevents_model_cleanup(app):
    owner, _ = user(app)
    record, version = model(app, owner)
    _model_notice_due(app, record)
    control = app.state.resource_governor.models
    control.excluded.append(app.state.versions.object_path(version["id"]))
    assert not control.preview()["items"][0]["eligible"]
    with pytest.raises(ResourceLimitError, match="resource_is_retained_or_in_use"):
        control.cleanup(record["id"])
    assert app.state.versions.verify(version["id"])["status"] == "verified"


def test_model_cleanup_verifies_retained_inputs_before_removing_any_weight(app):
    owner, _ = user(app)
    record, version = model(app, owner)
    source = app.state.paths.root / "retained-input.txt"
    source.write_text("original data")
    dependency = app.state.versions.snapshot("data_files", {"input.txt": source})
    retained = app.state.versions.materialize(
        dependency["id"],
        app.state.versions.root / "workspaces" / record["experiment_id"] / "data_dir",
    )
    app.state.versions.bind(record["experiment_id"], "input:data_dir", dependency["id"])
    app.state.versions.update_run(
        record["experiment_id"],
        input_views=[{"flag": "data_dir", "path": str(retained), "version_id": dependency["id"]}],
    )
    target = retained / "input.txt"
    target.chmod(0o644)
    target.write_text("tampered data")
    _model_notice_due(app, record)
    with pytest.raises(ResourceLimitError, match="model_cleanup_verification_failed"):
        app.state.resource_governor.models.cleanup(record["id"])
    assert app.state.versions.object_path(version["id"]).exists()
    with app.state.store.connect() as conn:
        log = conn.execute(select(resource_cleanup_logs)).mappings().one()
    assert log["status"] == "failed" and log["released_bytes"] == 0


def test_model_cleanup_recovers_after_unlink_before_database_commit(app, monkeypatch):
    import pepplatform.web.model_retention as module

    owner, _ = user(app)
    record, version = model(app, owner)
    _model_notice_due(app, record)
    real_remove = module.shutil.rmtree

    def interrupted_after_unlink(path):
        real_remove(path)
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        module,
        "shutil",
        SimpleNamespace(
            rmtree=interrupted_after_unlink,
            copy2=module.shutil.copy2,
            disk_usage=module.shutil.disk_usage,
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        app.state.resource_governor.models.cleanup(record["id"])
    monkeypatch.setattr(module.shutil, "rmtree", real_remove)
    assert app.state.resource_governor.models.cleanup(record["id"])["deleted"]
    with pytest.raises(ResourceLimitError) as exc:
        app.state.versions.verify(version["id"])
    assert exc.value.status_code == 410


def test_republication_of_identical_content_does_not_restore_retired_model(app):
    owner, _ = user(app)
    data = dataset(app, owner)
    record, version = model(app, owner, "retired-content", data=data)
    _model_notice_due(app, record)
    app.state.resource_governor.models.cleanup(record["id"])
    fresh, repeated = model(app, owner, "new-content", data=data)
    assert repeated["id"] == version["id"]
    assert app.state.versions.verify(repeated["id"])["status"] == "verified"
    assert app.state.store.get_model(fresh["id"])
    assert app.state.store.get_model(record["id"]) is None


@pytest.mark.parametrize(
    "route", ["/api/models/{model_id}/predict", "/api/models/{model_id}/predict/stream"]
)
@pytest.mark.parametrize("quota", ["daily_inference_request_limit", "daily_inference_input_limit"])
def test_prediction_quota_errors_keep_type_usage_and_reset_time(app, route, quota, quota_clock):
    owner, request = user(app)
    record, _ = model(app, owner)
    app.state.store.set_user_quota(owner["id"], {quota: 1})
    quota_clock.next_day()
    if quota == "daily_inference_request_limit":
        app.state.store.increment_quota_usage(owner["id"], inference_requests=1)
    payload = {"sequences": ["ACD", "EFG"]}
    with pytest.raises(HTTPException) as exc:
        endpoint(app, route, "POST")(record["id"], payload, request)
    assert exc.value.status_code == 429
    assert exc.value.detail["code"] == "quota_exceeded"
    assert exc.value.detail["quota_type"] == quota
    assert exc.value.detail["limit"] == 1
    assert exc.value.detail["requested"] == (1 if quota == "daily_inference_request_limit" else 2)
    assert datetime.fromisoformat(exc.value.detail["reset_at"]) > datetime.now(timezone.utc)
    assert not app.state.store.resources.leases(kind="inference")


@pytest.mark.parametrize(
    "field",
    [
        "daily_experiment_limit",
        "daily_inference_request_limit",
        "daily_inference_input_limit",
        "daily_training_seconds_limit",
    ],
)
def test_daily_reduction_keeps_today_and_latest_target_survives_restart(app, quota_clock, field):
    from pepplatform.web.store import MetadataStore

    change_policy(app, defaults={field: 10})
    owner, _ = user(app)
    store = app.state.store
    store.increment_quota_usage(
        owner["id"], experiments_created=8, inference_requests=8, inference_inputs=8
    )
    store.resources.add_daily(owner["id"], quota_clock.value.date().isoformat(), milliseconds=8000)
    for target in (5, 7):
        quota = store.set_user_quota(owner["id"], {field: target, "publish_announcements": True})
        assert quota[field] == 10
        assert quota["configured"][field] == target
        assert quota["pending"][field]["value"] == target
    notices = store.resources.announcements(owner)
    store.set_user_quota(owner["id"], {field: 7})
    assert len(store.resources.announcements(owner)) == len(notices) == 2
    assert all(notice["kind"] == "quota_change" for notice in notices)
    assert {notice["quota_change"][field]["to"] for notice in notices} == {5, 7}
    restarted = MetadataStore(store.database_url)
    try:
        assert restarted.get_user_quota(owner["id"])[field] == 10
        quota = restarted.set_user_quota(owner["id"], {field: 12})
        assert quota[field] == 12 and field not in quota["pending"]
        restarted.set_user_quota(owner["id"], {field: 3})
        quota_clock.next_day()
        assert restarted.get_user_quota(owner["id"])[field] == 3
        assert restarted.get_user_quota(owner["id"])["pending"] == {}
        assert restarted.get_quota_usage(owner["id"])["experiments_created"] == 0
        assert restarted.get_quota_usage(owner["id"])["training_seconds"] == 0
    finally:
        restarted.engine.dispose()


def test_global_reductions_preview_all_inheritors_and_exclude_admins(app, quota_clock):
    ordinary, _ = user(app)
    protected, _ = user(app, "personal")
    administrator, _ = user(app, "admin", admin=True)
    store = app.state.store
    store.set_user_quota(protected["id"], {"daily_experiment_limit": 30})
    payload = {
        "revision": store.resources.policy()["revision"],
        "defaults": {"daily_experiment_limit": 5},
    }
    preview = store.resources.preview(payload)
    assert {row["user_id"] for row in preview["impacts"]} == {ordinary["id"]}
    assert preview["deferred_count"] == preview["exempt_count"] == 1
    store.resources.save_policy({**payload, "preview_token": preview["preview_token"]})
    assert store.get_user_quota(ordinary["id"])["daily_experiment_limit"] == 10
    assert store.get_user_quota(protected["id"])["daily_experiment_limit"] == 30
    assert store.get_user_quota(administrator["id"])["pending"] == {}
    new_user, _ = user(app, "new")
    assert store.get_user_quota(new_user["id"])["daily_experiment_limit"] == 5
    store.set_user_quota(protected["id"], {"daily_experiment_limit": None})
    assert store.get_user_quota(protected["id"])["daily_experiment_limit"] == 30
    quota_clock.next_day()
    assert store.get_user_quota(protected["id"])["daily_experiment_limit"] == 5


def test_administrator_exemption_covers_admission_storage_and_model_protection(app):
    limits = {key: 0 for key in DEFAULT_QUOTAS if key != "model_retention_days"}
    change_policy(app, defaults=limits)
    administrator, request = user(app, "admin", admin=True)
    ordinary, _ = user(app)
    data = dataset(app, administrator)
    governor, store = app.state.resource_governor, app.state.store
    records = [model(app, administrator, f"admin-{i}", data=data)[0] for i in range(6)]
    for record in records:
        governor.protect_model(record, True)
    governor.require_storage(administrator, 100 * 1024**3)
    usage = governor.models.usage(administrator["id"])
    assert usage["retained"] == usage["protected"] == 6
    assert usage["remaining"] is None and usage["over_limit"] == 0
    created = endpoint(app, "/api/experiments", "POST")(
        {
            "dataset_id": data["id"],
            "config": {"tokenizer": "kmer", "encoder": "onehot", "model": "cnn"},
        },
        request,
    )
    assert created["status"] == "queued"
    lease = governor.reserve_inference(administrator, 100000)
    assert governor.inference_policy(lease)["enforcement"] == "exempt_admin"
    governor.settle_inference(lease, success=True)
    assert store.get_quota_usage(administrator["id"])["inference_inputs"] == 100000
    assert store.get_quota_usage(administrator["id"])["experiments_created"] == 1
    with pytest.raises(ResourceLimitError):
        governor.require_storage(ordinary, 1)
    with pytest.raises(ResourceLimitError):
        governor.reserve_inference(ordinary, 1)
    ordinary_data = dataset(app, ordinary)
    ordinary_model, _ = model(app, ordinary, "ordinary-model", data=ordinary_data)
    with pytest.raises(HTTPException) as denied:
        endpoint(app, "/api/models/{model_id}/protection", "PATCH")(
            ordinary_model["id"], {"pinned": True}, request
        )
    assert denied.value.status_code == 429
    store.update_user_admin_fields(administrator["id"], role="user")
    assert store.get_user_quota(administrator["id"])["enforcement"] == "limited"
    with pytest.raises(ResourceLimitError):
        governor.models.admit(administrator["id"], 1)
    store.update_user_admin_fields(administrator["id"], role="admin")
    governor.models.admit(administrator["id"], 1)
    with pytest.raises(ValueError):
        store.set_user_quota(ordinary["id"], {"enforcement": "exempt_admin"})
    change_policy(app, limits={"disk_reserve_bytes": 2**62})
    with pytest.raises(ResourceLimitError) as disk:
        governor.admit_training(administrator, 1)
    assert disk.value.detail["code"] == "disk_space_insufficient"


@pytest.mark.parametrize("mode", ["next_reset", "immediate"])
def test_accepted_training_and_inference_keep_admission_policy(app, quota_clock, mode):
    from pepplatform.web.store import MetadataStore

    owner, request = user(app)
    data = dataset(app, owner)
    store, governor = app.state.store, app.state.resource_governor
    created = endpoint(app, "/api/experiments", "POST")(
        {
            "dataset_id": data["id"],
            "config": {"tokenizer": "kmer", "encoder": "onehot", "model": "cnn", "seed": 42},
        },
        request,
    )
    lease = governor.reserve_inference(owner, 2)
    accepted = governor.inference_policy(lease)
    with store.connect() as conn:
        pending = conn.execute(
            select(resource_pending_jobs.c.policy_json).where(
                resource_pending_jobs.c.experiment_id == created["id"]
            )
        ).scalar_one()
    admitted_day = quota_clock.value.date().isoformat()
    store.set_user_quota(
        owner["id"],
        {
            "max_model_count": 0,
            "protected_model_limit": 0,
            "storage_limit_bytes": 0,
            "max_experiment_seconds": 0,
            "daily_training_seconds_limit": 0,
            "daily_inference_request_limit": 0,
            "daily_reduction_mode": mode,
        },
    )
    if mode == "next_reset":
        quota_clock.next_day()
    restarted = MetadataStore(store.database_url)
    try:
        with restarted.connect() as conn:
            assert (
                conn.execute(
                    select(resource_pending_jobs.c.policy_json).where(
                        resource_pending_jobs.c.experiment_id == created["id"]
                    )
                ).scalar_one()
                == pending
            )
    finally:
        restarted.engine.dispose()
    snapshot = governor.start_execution(created)
    assert snapshot["max_experiment_seconds"] == accepted["max_experiment_seconds"] == 3600
    assert snapshot["max_model_count"] == 5
    assert governor.inference_policy(lease) == accepted
    with governor.use_policy(snapshot):
        governor.require_storage(owner, 1)
    with pytest.raises(ResourceLimitError):
        governor.require_storage(owner, 1)
    with pytest.raises(ResourceLimitError):
        governor.admit_training(owner, 1)
    with pytest.raises(ResourceLimitError):
        governor.reserve_inference(owner, 1)
    assert store.get_experiment(created["id"])["config"] == created["config"]
    governor.settle_inference(lease, success=True)
    governor.settle_inference(lease, success=True)
    assert store.get_quota_usage(owner["id"], admitted_day)["inference_requests"] == 1


@pytest.mark.parametrize("field", DAILY_QUOTA_FIELDS)
@pytest.mark.parametrize("mode", ["next_reset", "immediate"])
@pytest.mark.parametrize("target", [0, 5])
def test_daily_modes_preserve_usage_other_pending_fields_and_restart(
    app, quota_clock, field, mode, target
):
    from pepplatform.web.store import MetadataStore

    change_policy(app, defaults=dict.fromkeys(DAILY_QUOTA_FIELDS, 10))
    owner, _ = user(app)
    store = app.state.store
    other = next(key for key in DAILY_QUOTA_FIELDS if key != field)
    store.set_user_quota(owner["id"], {other: 3})
    store.increment_quota_usage(
        owner["id"], experiments_created=8, inference_requests=8, inference_inputs=8
    )
    store.resources.add_daily(owner["id"], quota_clock.value.date().isoformat(), milliseconds=8000)
    usage = store.get_quota_usage(owner["id"])
    updates = {field: target, "daily_reduction_mode": mode}
    quota = store.set_user_quota(owner["id"], updates)
    assert quota[field] == (target if mode == "immediate" else 10)
    assert quota["configured"][field] == target
    assert (field in quota["pending"]) == (mode == "next_reset")
    assert quota[other] == 10 and quota["pending"][other]["value"] == 3
    assert "daily_reduction_mode" not in quota["overrides"]
    assert store.get_quota_usage(owner["id"]) == usage
    with store.connect() as conn:
        events = [
            json.loads(row)
            for row in conn.execute(
                select(resource_quota_changes.c.changes_json).where(
                    resource_quota_changes.c.user_id == owner["id"]
                )
            ).scalars()
        ]
    change = next(event[field] for event in events if field in event)
    assert change["effective_from"] == 10 and change["effective_to"] == target
    assert change["daily_reduction_mode"] == mode
    assert datetime.fromisoformat(change["effective_at"]) == (
        quota_clock.value
        if mode == "immediate"
        else quota_clock.value.replace(hour=0, minute=0, second=0, microsecond=0)
        + timedelta(days=1)
    )
    notices = len(store.resources.announcements(owner))
    store.set_user_quota(owner["id"], updates)
    assert len(store.resources.announcements(owner)) == notices
    restarted = MetadataStore(store.database_url)
    try:
        assert restarted.get_user_quota(owner["id"])[field] == quota[field]
        quota_clock.next_day()
        assert restarted.get_user_quota(owner["id"])[field] == target
        assert restarted.get_user_quota(owner["id"])[other] == 3
        assert restarted.get_user_quota(owner["id"])["pending"] == {}
    finally:
        restarted.engine.dispose()


@pytest.mark.parametrize("field", DAILY_QUOTA_FIELDS)
@pytest.mark.parametrize("inherited", [False, True])
def test_pending_reduction_can_be_applied_now_without_changing_its_source(
    app, quota_clock, field, inherited
):
    owner, _ = user(app)
    administrator, request = user(app, "admin", admin=True)
    store = app.state.store
    if inherited:
        change_policy(app, defaults={field: 5})
    else:
        store.set_user_quota(owner["id"], {field: 5})
    before = store.get_user_quota(owner["id"])
    update = endpoint(app, "/api/admin/users/{user_id}/quotas", "PATCH")
    payload = {
        field: None if inherited else 5,
        "daily_reduction_mode": "immediate",
        "publish_announcements": True,
    }
    quota = update(owner["id"], payload, request)
    assert quota[field] == 5 and field not in quota["pending"]
    assert quota["configured"] == before["configured"]
    assert quota["overrides"] == before["overrides"]
    assert quota["sources"][field] == ("global" if inherited else "personal")
    logs = store.list_audit_logs(event_type="admin.user_quota_updated")["items"]
    assert len(logs) == 1 and logs[0]["actor_user_id"] == administrator["id"]
    assert logs[0]["metadata"]["effective_changes"][field] == {"from": before[field], "to": 5}
    assert logs[0]["metadata"]["daily_reduction_mode"] == "immediate"
    events = store.resources.announcements(owner)
    promoted = next(
        event["quota_change"][field]
        for event in events
        if event["quota_change"][field].get("daily_reduction_mode") == "immediate"
    )
    assert promoted["from"] == promoted["to"] == 5
    assert promoted["effective_from"] == before[field] and promoted["effective_to"] == 5
    update(owner["id"], payload, request)
    assert len(store.resources.announcements(owner)) == len(events)
    assert len(store.list_audit_logs(event_type="admin.user_quota_updated")["items"]) == 1


@pytest.mark.parametrize("field", DAILY_QUOTA_FIELDS)
def test_default_reductions_and_increases_follow_the_latest_immediate_entitlement(app, field):
    change_policy(app, defaults={field: 10})
    owner, _ = user(app)
    store = app.state.store
    for target, mode, expected in (
        (5, "immediate", 5),
        (3, "next_reset", 5),
        (7, "next_reset", 7),
        (2, "immediate", 2),
        (12, "next_reset", 12),
    ):
        quota = store.set_user_quota(owner["id"], {field: target, "daily_reduction_mode": mode})
        assert quota[field] == expected
        assert quota["configured"][field] == target


@pytest.mark.parametrize("mode", ["next_reset", "immediate"])
def test_global_modes_respect_personal_protection_and_audit_affected_users(app, mode):
    ordinary, _ = user(app)
    protected, _ = user(app, "protected")
    writable, _ = user(app, "writable")
    _, request = user(app, "admin", admin=True)
    store = app.state.store
    field = "daily_experiment_limit"
    store.set_user_quota(protected["id"], {field: 7})
    store.set_user_quota(writable["id"], {field: 20, "allow_global_override": True})
    store.increment_quota_usage(ordinary["id"], experiments_created=8)
    payload = {
        "revision": 1,
        "defaults": {field: 5},
        "apply_to_users": True,
        "daily_reduction_mode": mode,
    }
    preview = store.resources.preview(payload)
    assert {row["user_id"] for row in preview["impacts"]} == {ordinary["id"], writable["id"]}
    assert preview["immediate_count"] == (2 if mode == "immediate" else 0)
    assert preview["deferred_count"] == (0 if mode == "immediate" else 2)
    assert preview["blocked_count"] == (1 if mode == "immediate" else 0)
    endpoint(app, "/api/admin/resource-policies", "PATCH")(
        {**payload, "preview_token": preview["preview_token"]}, request
    )
    assert store.get_user_quota(ordinary["id"])[field] == (5 if mode == "immediate" else 10)
    assert store.get_user_quota(writable["id"])[field] == (5 if mode == "immediate" else 20)
    assert store.get_user_quota(writable["id"])["sources"][field] == "global"
    assert store.get_user_quota(protected["id"])[field] == 10
    assert store.get_user_quota(protected["id"])["pending"][field]["value"] == 7
    assert store.get_quota_usage(ordinary["id"])["experiments_created"] == 8
    audit = store.list_audit_logs(event_type="admin.resource_policy_updated")["items"][0]
    assert audit["metadata"]["daily_reduction_mode"] == mode
    assert len(audit["metadata"]["impacts"]) == 2
    assert "daily_reduction_mode" not in store.resources.policy()


def test_global_pending_promotion_only_applies_selected_inherited_fields(app):
    owner, _ = user(app)
    protected, _ = user(app, "protected")
    store = app.state.store
    fields = ("daily_experiment_limit", "daily_inference_request_limit")
    store.set_user_quota(protected["id"], {fields[0]: 5})
    change_policy(app, defaults=dict.fromkeys(fields, 5))
    payload = {
        "revision": store.resources.policy()["revision"],
        "defaults": {fields[0]: 5},
        "daily_reduction_mode": "immediate",
    }
    preview = store.resources.preview(payload)
    assert {row["user_id"] for row in preview["impacts"]} == {owner["id"]}
    assert preview["immediate_count"] == 1 and preview["deferred_count"] == 0
    store.resources.save_policy({**payload, "preview_token": preview["preview_token"]})
    assert store.get_user_quota(owner["id"])[fields[0]] == 5
    assert fields[1] in store.get_user_quota(owner["id"])["pending"]
    assert fields[0] in store.get_user_quota(protected["id"])["pending"]
    payload["revision"] = store.resources.policy()["revision"]
    assert store.resources.preview(payload)["impacts"] == []


def test_preview_token_binds_mode_synchronization_and_reset_day(app, quota_clock):
    user(app)
    store = app.state.store
    payload = {"revision": 1, "defaults": {"daily_experiment_limit": 5}}
    token = store.resources.preview(payload)["preview_token"]
    for change in ({"daily_reduction_mode": "immediate"}, {"apply_to_users": True}):
        with pytest.raises(PolicyConflict):
            store.resources.save_policy({**payload, **change, "preview_token": token})
    quota_clock.next_day()
    with pytest.raises(PolicyConflict):
        store.resources.save_policy({**payload, "preview_token": token})
    assert store.resources.policy()["revision"] == 1


@pytest.mark.parametrize("mode", [None, True, 0, [], {}, "later"])
def test_invalid_modes_return_400_without_changing_policy_or_quotas(app, mode):
    owner, _ = user(app)
    _, request = user(app, "admin", admin=True)
    for path, method, args in (
        (
            "/api/admin/users/{user_id}/quotas",
            "PATCH",
            [owner["id"], {"daily_experiment_limit": 0, "daily_reduction_mode": mode}, request],
        ),
        (
            "/api/admin/resource-policies/preview",
            "POST",
            [{"revision": 1, "daily_reduction_mode": mode}, request],
        ),
        (
            "/api/admin/resource-policies",
            "PATCH",
            [{"revision": 1, "daily_reduction_mode": mode}, request],
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            endpoint(app, path, method)(*args)
        assert exc.value.status_code == 400
    assert app.state.store.resources.policy()["revision"] == 1
    assert app.state.store.get_user_quota(owner["id"])["overrides"] == {}
    assert app.state.store.resources.announcements(owner) == []


def test_immediate_change_rolls_back_if_notification_cannot_be_saved(app, monkeypatch):
    owner, _ = user(app)
    _, request = user(app, "admin", admin=True)
    store = app.state.store
    before = store.get_user_quota(owner["id"])

    def fail(*args, **kwargs):
        raise RuntimeError("notice write failed")

    monkeypatch.setattr(store.resources, "publish_announcement", fail)
    with pytest.raises(RuntimeError, match="notice write failed"):
        endpoint(app, "/api/admin/users/{user_id}/quotas", "PATCH")(
            owner["id"],
            {
                "daily_experiment_limit": 0,
                "daily_reduction_mode": "immediate",
                "publish_announcements": True,
            },
            request,
        )
    assert store.get_user_quota(owner["id"]) == before
    with store.connect() as conn:
        assert conn.execute(select(resource_quota_changes.c.id)).first() is None
    assert store.list_audit_logs(event_type="admin.user_quota_updated")["items"] == []


@pytest.mark.parametrize("field", DAILY_QUOTA_FIELDS)
@pytest.mark.parametrize("target", [0, 5])
def test_immediate_reductions_reject_new_work_with_original_usage(app, quota_clock, field, target):
    owner, request = user(app)
    store, governor = app.state.store, app.state.resource_governor
    store.increment_quota_usage(
        owner["id"], experiments_created=8, inference_requests=8, inference_inputs=8
    )
    store.resources.add_daily(owner["id"], quota_clock.value.date().isoformat(), milliseconds=8000)
    store.set_user_quota(owner["id"], {field: target, "daily_reduction_mode": "immediate"})
    if field in ("daily_inference_request_limit", "daily_inference_input_limit"):
        with pytest.raises(ResourceLimitError) as exc:
            governor.reserve_inference(owner, 1)
    else:
        data = dataset(app, owner)
        with pytest.raises(HTTPException) as exc:
            endpoint(app, "/api/experiments", "POST")(
                {
                    "dataset_id": data["id"],
                    "config": {
                        "tokenizer": "kmer",
                        "encoder": "onehot",
                        "model": "cnn",
                    },
                },
                request,
            )
    assert exc.value.detail["quota_type"] == field
    assert exc.value.detail["used"] == 8
    assert exc.value.detail["limit"] == target
    assert ("reset_at" in exc.value.detail) == (target > 0)
    assert store.count_active_experiments_for_user(owner["id"]) == 0
    assert governor.policies.leases(user_id=owner["id"], kind="inference") == []


@pytest.mark.parametrize("field", ["daily_inference_request_limit", "daily_inference_input_limit"])
def test_immediate_preview_and_admission_include_existing_inference_reservations(app, field):
    owner, _ = user(app)
    store, governor = app.state.store, app.state.resource_governor
    lease = governor.reserve_inference(owner, 5)
    accepted = governor.inference_policy(lease)
    target = 1 if field == "daily_inference_request_limit" else 5
    payload = {"revision": 1, "defaults": {field: target}, "daily_reduction_mode": "immediate"}
    preview = store.resources.preview(payload)
    assert preview["blocked_count"] == 1
    store.resources.save_policy({**payload, "preview_token": preview["preview_token"]})
    with pytest.raises(ResourceLimitError) as exc:
        governor.reserve_inference(owner, 1)
    assert exc.value.detail["quota_type"] == field
    assert governor.inference_policy(lease) == accepted
    governor.settle_inference(lease, success=True)
    governor.settle_inference(lease, success=True)
    usage = store.get_quota_usage(owner["id"])
    assert usage["inference_requests"] == 1 and usage["inference_inputs"] == 5


def test_immediate_change_and_training_admission_share_one_atomic_boundary(app):
    import threading

    owner, request = user(app)
    _, admin_request = user(app, "admin", admin=True)
    data = dataset(app, owner)
    store = app.state.store
    store.increment_quota_usage(owner["id"], experiments_created=9)
    barrier = threading.Barrier(2)

    def submit():
        barrier.wait(timeout=5)
        try:
            return endpoint(app, "/api/experiments", "POST")(
                {
                    "dataset_id": data["id"],
                    "config": {
                        "tokenizer": "kmer",
                        "encoder": "onehot",
                        "model": "cnn",
                        "seed": 42,
                    },
                },
                request,
            )
        except HTTPException as exc:
            assert exc.status_code == 429 and exc.detail["limit"] == 1
            return None

    def reduce():
        barrier.wait(timeout=5)
        return endpoint(app, "/api/admin/users/{user_id}/quotas", "PATCH")(
            owner["id"],
            {"daily_experiment_limit": 1, "daily_reduction_mode": "immediate"},
            admin_request,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        submitted = executor.submit(submit)
        reduced = executor.submit(reduce)
        created = submitted.result(timeout=15)
        assert reduced.result(timeout=15)["daily_experiment_limit"] == 1
    assert store.get_quota_usage(owner["id"])["experiments_created"] == (10 if created else 9)
    with store.connect() as conn:
        rows = conn.execute(select(resource_pending_jobs)).mappings().all()
    assert len(rows) == (1 if created else 0)
    if created:
        assert json.loads(rows[0]["policy_json"])["daily_experiment_limit"] == 10
        assert store.get_experiment(created["id"])["config"] == created["config"]


def test_running_training_keeps_its_policy_after_immediate_reduction(app, monkeypatch):
    from pepplatform.versioning.web import VersionedExperimentRunner

    owner, request = user(app)
    data = dataset(app, owner)
    store = app.state.store
    record = endpoint(app, "/api/experiments", "POST")(
        {
            "dataset_id": data["id"],
            "config": {"tokenizer": "kmer", "encoder": "onehot", "model": "cnn", "seed": 42},
        },
        request,
    )

    def work(runner, identifier):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(1.5); print('completed')"],
            start_new_session=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        runner._active_processes[identifier] = process
        runner.store.update_experiment(identifier, status="running")
        try:
            store.set_user_quota(
                owner["id"],
                {
                    "daily_training_seconds_limit": 0,
                    "max_experiment_seconds": 0,
                    "daily_reduction_mode": "immediate",
                },
            )
            output, _ = process.communicate(timeout=10)
            assert process.returncode == 0 and output.strip() == "completed"
            assert runner.store.get_experiment(identifier)["status"] == "running"
            runner.store.update_experiment(identifier, status="succeeded")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            runner._active_processes.pop(identifier, None)

    monkeypatch.setattr(VersionedExperimentRunner, "_run_experiment", work)
    app.state.runner._run_experiment(record["id"])
    assert store.get_experiment(record["id"])["status"] == "succeeded"
    assert store.get_experiment(record["id"])["config"] == record["config"]
    assert store.get_quota_usage(owner["id"])["training_seconds"] >= 1


def test_model_reduction_preserves_five_and_reopens_only_after_freeing_capacity(app):
    owner, _ = user(app)
    data = dataset(app, owner)
    records = [model(app, owner, f"old-{i}", data=data)[0] for i in range(5)]
    governor, store = app.state.resource_governor, app.state.store
    store.set_user_quota(owner["id"], {"max_model_count": 2, "protected_model_limit": 2})
    assert governor.models.usage(owner["id"])["retained"] == 5
    assert governor.models.usage(owner["id"])["remaining"] == 0
    with pytest.raises(ResourceLimitError):
        governor.models.admit(owner["id"], 1)
    for record in records[:3]:
        store.delete_model(record["id"])
    with pytest.raises(ResourceLimitError):
        governor.models.admit(owner["id"], 1)
    store.delete_model(records[3]["id"])
    governor.models.admit(owner["id"], 1)
    assert governor.models.usage(owner["id"])["remaining"] == 1


def test_inference_worker_resets_personal_file_limit_between_roles(tmp_path):
    script = """
import json, sys
from pathlib import Path
from pepplatform.web.resource_inference_worker import apply_file_limit
path = Path(sys.argv[1])
results = []
for policy in ({"max_model_file_bytes": 32}, {"enforcement": "exempt_admin", "max_model_file_bytes": 0}, {"max_model_file_bytes": 32}):
    apply_file_limit(policy)
    try:
        with path.open("wb") as handle:
            handle.write(b"x" * 64)
        results.append(True)
    except OSError:
        results.append(False)
print(json.dumps(results))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "worker-output")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == [False, True, False]


@pytest.mark.parametrize("publish", [None, False, True])
@pytest.mark.parametrize("field,target", [
    ("daily_experiment_limit", 5),
    ("daily_concurrent_experiment_limit", 2),
    ("max_experiment_seconds", 1800),
])
def test_quota_announcements_are_optional_without_losing_history(app, publish, field, target):
    owner, _ = user(app)
    other, _ = user(app, "other")
    administrator, request = user(app, "admin", admin=True)
    store = app.state.store
    payload = {field: target}
    if publish is not None:
        payload["publish_announcements"] = publish
    update = endpoint(app, "/api/admin/users/{user_id}/quotas", "PATCH")
    quota = update(owner["id"], payload, request)
    if field == "daily_experiment_limit":
        assert quota[field] == 10
        assert quota["pending"][field]["value"] == target
    else:
        assert quota[field] == target
    assert "publish_announcements" not in quota["configured"]
    with store.connect() as conn:
        change = conn.execute(select(resource_quota_changes)).mappings().one()
    assert change["actor_user_id"] == administrator["id"]
    assert json.loads(change["changes_json"])[field]["to"] == target
    notices = store.resources.announcements(owner)
    assert len(notices) == int(bool(publish))
    assert store.resources.announcements(other) == []
    audit = store.list_audit_logs(event_type="admin.user_quota_updated")["items"]
    assert audit[0]["metadata"]["publish_announcements"] is bool(publish)
    # Opting in later must not republish an unchanged value.
    update(owner["id"], {**payload, "publish_announcements": True}, request)
    assert store.resources.announcements(owner) == notices


@pytest.mark.parametrize("publish", [False, True])
@pytest.mark.parametrize("synchronize", [False, True])
@pytest.mark.parametrize("field,personal,target", [
    ("daily_experiment_limit", 30, 5),
    ("daily_concurrent_experiment_limit", 3, 2),
    ("max_experiment_seconds", 7200, 1800),
])
def test_global_announcement_choice_binds_preview_and_targets_actual_changes(
    app, publish, synchronize, field, personal, target
):
    owner, _ = user(app)
    protected, _ = user(app, "protected")
    eligible, _ = user(app, "eligible")
    administrator, request = user(app, "admin", admin=True)
    store = app.state.store
    store.set_user_quota(protected["id"], {field: personal})
    store.set_user_quota(
        eligible["id"],
        {
            field: personal,
            "allow_global_override": True,
        },
    )
    payload = {
        "revision": store.resources.policy()["revision"],
        "defaults": {field: target},
        "apply_to_users": synchronize,
        "publish_announcements": publish,
    }
    preview = store.resources.preview(payload)
    assert preview["announcement_count"] == ((2 if synchronize else 1) if publish else 0)
    alternative = store.resources.preview({**payload, "publish_announcements": not publish})
    assert alternative["preview_token"] != preview["preview_token"]
    save = endpoint(app, "/api/admin/resource-policies", "PATCH")
    with pytest.raises(HTTPException) as exc:
        save({**payload, "preview_token": alternative["preview_token"]}, request)
    assert exc.value.status_code == 409
    save({**payload, "preview_token": preview["preview_token"]}, request)
    assert len(store.resources.announcements(owner)) == int(publish)
    assert len(store.resources.announcements(eligible)) == int(publish and synchronize)
    assert store.resources.announcements(protected) == []
    assert store.resources.announcements(administrator) == []
    with store.connect() as conn:
        assert len(conn.execute(select(resource_quota_changes)).all()) == 2 + (
            2 if synchronize else 1
        )
    assert "publish_announcements" not in store.resources.policy()["defaults"]
    audit = store.list_audit_logs(event_type="admin.resource_policy_updated")["items"]
    assert audit[0]["metadata"]["publish_announcements"] is publish


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_invalid_announcement_options_are_rejected_before_mutation(app, value):
    owner, _ = user(app)
    _, request = user(app, "admin", admin=True)
    paths = [
        (
            "/api/admin/users/{user_id}/quotas",
            "PATCH",
            [
                owner["id"],
                {
                    "daily_experiment_limit": 5,
                    "publish_announcements": value,
                },
                request,
            ],
        ),
        (
            "/api/admin/resource-policies/preview",
            "POST",
            [
                {
                    "revision": 1,
                    "publish_announcements": value,
                },
                request,
            ],
        ),
        (
            "/api/admin/resource-policies",
            "PATCH",
            [
                {
                    "revision": 1,
                    "publish_announcements": value,
                },
                request,
            ],
        ),
        (
            "/api/admin/resource-policies",
            "PATCH",
            [
                {
                    "revision": 1,
                    "retention": {"publish_announcements": value},
                },
                request,
            ],
        ),
        ("/api/admin/resources/retention", "POST", [request, {"publish_announcements": value}]),
    ]
    for path, method, args in paths:
        with pytest.raises(HTTPException) as exc:
            endpoint(app, path, method)(*args)
        assert exc.value.status_code == 400
    assert app.state.store.get_user_quota(owner["id"])["configured"]["daily_experiment_limit"] == 10
    assert app.state.store.resources.policy()["revision"] == 1
    assert app.state.store.resources.announcements(owner) == []
    assert app.state.store.list_audit_logs()["items"] == []


def test_silent_quota_changes_never_call_the_publisher(app, monkeypatch):
    owner, _ = user(app)
    _, request = user(app, "admin", admin=True)

    def fail(*args, **kwargs):
        raise AssertionError("silent changes must not call the announcement publisher")

    monkeypatch.setattr(app.state.store.resources, "publish_announcement", fail)
    update = endpoint(app, "/api/admin/users/{user_id}/quotas", "PATCH")
    update(owner["id"], {"daily_experiment_limit": 5}, request)
    quota = update(
        owner["id"], {"daily_experiment_limit": 5, "daily_reduction_mode": "immediate"}, request
    )
    assert quota["daily_experiment_limit"] == 5 and not quota["pending"]
    change_policy(app, defaults={"daily_inference_request_limit": 300})
    with app.state.store.connect() as conn:
        assert len(conn.execute(select(resource_quota_changes)).all()) == 3


@pytest.mark.parametrize("publish", [False, True])
def test_manual_retention_choice_is_independent_of_the_background_policy(app, publish):
    owner, _ = user(app)
    record, _ = model(app, owner)
    _age_model(app, record)
    administrator, request = user(app, "admin", admin=True)
    change_policy(
        app,
        retention={
            "enabled": True,
            "delete_unprotected_models": True,
            "publish_announcements": not publish,
        },
    )
    check = endpoint(app, "/api/admin/resources/retention", "POST")
    payload = {"publish_announcements": True} if publish else None
    check(request, payload)
    state = app.state.resource_governor.models.state(record["id"])
    assert state["state"] == ("notified" if publish else "removed")
    assert state["notified_at"] == (state["scheduled_at"] if publish else None)
    assert app.state.store.resources.policy()["retention"]["publish_announcements"] is not publish
    logs = app.state.store.list_audit_logs(event_type="resource.cleanup_scheduled")["items"]
    event = next(row for row in logs if row["metadata"]["resource_id"] == record["id"])
    assert event["actor_user_id"] == administrator["id"]
    assert event["metadata"]["publish_announcements"] is publish
    # Later background checks do not republish or reset a pending plan.
    app.state.resource_governor.retention_tick()
    assert app.state.resource_governor.models.state(record["id"]) == state


@pytest.mark.parametrize("kind", ["cache", "model", "staging", "workspace"])
def test_new_silent_cleanup_runs_in_the_same_tick_without_advancing_time(app, kind):
    owner, _ = user(app)
    governor = app.state.resource_governor
    old = datetime.now(timezone.utc) - timedelta(days=30)
    if kind in {"cache", "model"}:
        record, version = model(app, owner)
        _age_model(app, record)
        path = app.state.versions.root / "inference" / record["id"]
        os.utime(path, (old.timestamp(), old.timestamp()))
        if kind == "model":
            path = app.state.versions.object_path(version["id"])
    else:
        folder = "workspaces" if kind == "workspace" else "staging"
        path = app.state.versions.root / folder / "expired"
        path.mkdir(parents=True)
        (path / "unused.txt").write_text("abandoned temporary data")
        os.utime(path, (old.timestamp(), old.timestamp()))
    change_policy(app, retention={"enabled": True, "delete_unprotected_models": kind == "model"})
    governor.retention_tick()
    assert not path.exists()
    assert app.state.store.resources.announcements(owner) == []
    events = app.state.store.list_audit_logs(event_type="resource.cleanup_scheduled")["items"]
    assert events
    assert all(
        event["metadata"]["delete_after"] == event["metadata"]["scheduled_at"] for event in events
    )
    if kind == "cache":
        assert app.state.versions.verify(version["id"])["status"] == "verified"
    if kind == "model":
        assert governor.models.state(record["id"])["state"] == "removed"
        assert governor.models.usage(owner["id"])["remaining"] == 5


@pytest.mark.parametrize("kind", ["cache", "model"])
@pytest.mark.parametrize("blocker", ["disabled", "young", "pinned", "lease"])
def test_direct_cleanup_cannot_bypass_policy_idle_expiry_or_protection(app, kind, blocker):
    owner, _ = user(app)
    record, version = model(app, owner)
    governor = app.state.resource_governor
    cache = app.state.versions.root / "inference" / record["id"]
    if blocker != "young":
        _age_model(app, record)
        old = datetime.now(timezone.utc) - timedelta(days=30)
        os.utime(cache, (old.timestamp(), old.timestamp()))
    change_policy(
        app, retention={"enabled": blocker != "disabled", "delete_unprotected_models": True}
    )
    if blocker == "pinned":
        governor.protect_model(record, True)
    if blocker == "lease":
        app.state.store.resources.create_lease("read", owner["id"], record["id"])
    if kind == "cache":
        governor.scan()
        identifier = next(
            row["id"] for row in governor.inventory() if row["kind"] == "inference_cache"
        )
        cleanup = governor.cleanup
    else:
        identifier, cleanup = record["id"], governor.models.cleanup
    with pytest.raises(ResourceLimitError) as exc:
        cleanup(identifier)
    expected = (
        ("model_cleanup_disabled" if kind == "model" else "resource_cleanup_disabled")
        if blocker == "disabled"
        else "retention_not_due"
        if blocker == "young"
        else "resource_is_retained_or_in_use"
    )
    assert exc.value.detail["code"] == expected
    assert app.state.versions.verify(version["id"])["status"] == "verified"
    assert cache.exists()
    assert not app.state.store.list_audit_logs(event_type="resource.cleanup_scheduled")["items"]


@pytest.mark.parametrize("kind", ["cache", "model"])
def test_announced_cleanup_keeps_its_deadline_when_silent_mode_is_selected(app, monkeypatch, kind):
    if kind == "cache":
        owner, record, _, path, plan = expired_cache(app, monkeypatch, advance=False)
        control, tick = app.state.resource_governor, app.state.resource_governor.retention_tick
        identifier = plan["id"]
    else:
        owner, _ = user(app)
        record, _ = model(app, owner)
        _age_model(app, record)
        change_policy(app, retention={"enabled": True, "delete_unprotected_models": True})
        control, tick = app.state.resource_governor.models, app.state.resource_governor.models.tick
        plan = control._schedule_cleanup(record["id"], publish_announcements=True)
        identifier, path = record["id"], Path(record["bundle_path"])
    change_policy(app, retention={"publish_announcements": False})
    tick(publish_announcements=False)
    assert path.exists()
    with pytest.raises(ResourceLimitError, match="cleanup_notice_pending"):
        control.cleanup(identifier)
    assert len(app.state.store.resources.announcements(owner)) == 1
    from pepplatform.web import model_retention, resource_governor, resource_policy

    future = datetime.fromisoformat(plan["delete_after"]) + timedelta(seconds=1)
    for module in (model_retention, resource_governor, resource_policy):
        monkeypatch.setattr(module, "now_utc", lambda: future)
    tick(publish_announcements=False)
    assert not path.exists()


@pytest.mark.parametrize("kind", ["cache", "model"])
def test_existing_silent_plans_ignore_old_deadlines_after_restart_and_notification_toggles(
    app, monkeypatch, kind
):
    from pepplatform.web.store import MetadataStore

    if kind == "cache":
        owner, record, _, _, entry = expired_cache(
            app, monkeypatch, publish_announcements=False, advance=False
        )
        control = app.state.resource_governor
        tick = control.retention_tick
        current = lambda: next(row for row in control.inventory() if row["id"] == entry["id"])
    else:
        owner, _ = user(app)
        record, _ = model(app, owner)
        _age_model(app, record)
        change_policy(app, retention={"enabled": True, "delete_unprotected_models": True})
        control = app.state.resource_governor.models
        tick = control.tick
        current = lambda: control.state(record["id"])
        control._schedule_cleanup(record["id"])
    before = current()
    assert before["state"] == "scheduled" and before["notified_at"] is None
    from pepplatform.web.resource_policy import model_retention, resource_inventory

    table = resource_inventory if kind == "cache" else model_retention
    key = table.c.id if kind == "cache" else table.c.model_id
    identifier = before["id"] if kind == "cache" else record["id"]
    old_deadline = (datetime.now(timezone.utc) + timedelta(hours=71)).isoformat()
    with app.state.store.connect() as conn:
        conn.execute(table.update().where(key == identifier).values(delete_after=old_deadline))
    restarted = MetadataStore(app.state.store.database_url)
    restarted.engine.dispose()
    change_policy(app, retention={"publish_announcements": True})
    tick(publish_announcements=True)
    with app.state.store.connect() as conn:
        after = conn.execute(select(table).where(key == identifier)).mappings().one()
    assert after["state"] == ("deleted" if kind == "cache" else "removed")
    assert after["notified_at"] is None
    assert datetime.fromisoformat(after["delete_after"]) < datetime.fromisoformat(old_deadline)
    assert app.state.store.resources.announcements(owner) == []
    events = app.state.store.list_audit_logs(event_type="resource.cleanup_scheduled")["items"]
    assert any(event["metadata"].get("previous_delete_after") == old_deadline for event in events)


@pytest.mark.parametrize("kind", ["cache", "model"])
def test_cleanup_announcement_failure_rolls_back_new_schedule_and_silent_mode_still_works(
    app, monkeypatch, kind
):
    owner, _ = user(app)
    record, _ = model(app, owner)
    _age_model(app, record)
    cache = app.state.versions.root / "inference" / record["id"]
    old = datetime.now(timezone.utc) - timedelta(days=30)
    os.utime(cache, (old.timestamp(), old.timestamp()))
    change_policy(app, retention={"enabled": True, "delete_unprotected_models": kind == "model"})
    governor = app.state.resource_governor
    tick = governor.models.tick if kind == "model" else governor.retention_tick

    def fail(*args, **kwargs):
        raise RuntimeError("announcement storage failed")

    monkeypatch.setattr(app.state.store.resources, "publish_announcement", fail)
    with pytest.raises(RuntimeError, match="announcement storage failed"):
        tick(publish_announcements=True)
    if kind == "model":
        assert governor.models.state(record["id"]) is None
    else:
        row = next(row for row in governor.inventory() if row["kind"] == "inference_cache")
        assert row["scheduled_at"] is None and row["delete_after"] is None
    assert app.state.store.list_audit_logs(event_type="resource.cleanup_scheduled")["items"] == []
    tick(publish_announcements=False)
    assert app.state.store.resources.announcements(owner) == []
    assert app.state.store.list_audit_logs(event_type="resource.cleanup_scheduled")["items"]


def test_admission_crossing_midnight_checks_and_charges_the_same_day(app, quota_clock):
    owner, _ = user(app)
    store = app.state.store
    quota_clock.value = quota_clock.value.replace(hour=23, minute=59, second=59)
    admitted_day = quota_clock.value.date().isoformat()
    with store.atomic():
        store.get_user_quota(owner["id"])
        quota_clock.next_day()
        store.increment_quota_usage(owner["id"], experiments_created=1)
        lease = store.resources.create_lease(
            "inference", owner["id"], request_count=1, policy=store.get_user_quota(owner["id"])
        )
        assert store.get_quota_usage(owner["id"])["usage_date"] == admitted_day
    assert store.get_quota_usage(owner["id"], admitted_day)["experiments_created"] == 1
    assert store.get_quota_usage(owner["id"])["experiments_created"] == 0
    with store.connect() as conn:
        from pepplatform.web.resource_policy import resource_leases

        assert (
            conn.execute(
                select(resource_leases.c.usage_date).where(resource_leases.c.id == lease)
            ).scalar()
            == admitted_day
        )
