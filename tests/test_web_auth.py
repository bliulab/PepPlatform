import pytest

fastapi = pytest.importorskip("fastapi")

from pepplatform.web.app import create_app
from pepplatform.web.auth import AuthSettings, cookie_kwargs, hash_password


def _auth_app(tmp_path, monkeypatch):
    monkeypatch.setenv("PEP_AUTH_ENABLED", "1")
    monkeypatch.setenv("PEP_SESSION_SECRET", "test-session-secret")
    monkeypatch.delenv("PEP_SMTP_HOST", raising=False)
    project_root = tmp_path / "project"
    project_root.mkdir()
    return create_app(project_root=project_root, runtime_root=tmp_path / "runtime")


class _Response:
    def __init__(self):
        self.cookies = {}

    def set_cookie(self, key, value, **_kwargs):
        self.cookies[key] = value

    def delete_cookie(self, key, **_kwargs):
        self.cookies.pop(key, None)


class _Request:
    def __init__(self, cookies=None):
        self.cookies = cookies or {}


def _endpoint(app, path, method):
    for route in app.routes:
        if getattr(route, "path", "") == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"{method} {path} endpoint was not registered")


def test_auth_enabled_defaults_on(monkeypatch):
    monkeypatch.delenv("PEP_AUTH_ENABLED", raising=False)
    assert AuthSettings.from_env().enabled is True

    monkeypatch.setenv("PEP_AUTH_ENABLED", "0")
    assert AuthSettings.from_env().enabled is False


def test_cookie_secure_defaults_follow_public_base_url(monkeypatch):
    monkeypatch.setenv("PEP_AUTH_ENABLED", "1")
    monkeypatch.delenv("PEP_COOKIE_SECURE", raising=False)

    monkeypatch.setenv("PEP_PUBLIC_BASE_URL", "http://127.0.0.1:5173")
    http_settings = AuthSettings.from_env()
    assert http_settings.cookie_secure is False
    assert cookie_kwargs(http_settings)["secure"] is False

    monkeypatch.setenv("PEP_PUBLIC_BASE_URL", "https://pepplatform.example")
    https_settings = AuthSettings.from_env()
    assert https_settings.cookie_secure is True
    assert cookie_kwargs(https_settings)["secure"] is True


def test_cookie_secure_env_overrides_public_base_url(monkeypatch):
    monkeypatch.setenv("PEP_AUTH_ENABLED", "1")

    monkeypatch.setenv("PEP_PUBLIC_BASE_URL", "https://pepplatform.example")
    monkeypatch.setenv("PEP_COOKIE_SECURE", "false")
    assert AuthSettings.from_env().cookie_secure is False

    monkeypatch.setenv("PEP_PUBLIC_BASE_URL", "http://127.0.0.1:5173")
    monkeypatch.setenv("PEP_COOKIE_SECURE", "true")
    assert AuthSettings.from_env().cookie_secure is True


def _register_verified(app, email):
    register = _endpoint(app, "/api/auth/register", "POST")
    verify = _endpoint(app, "/api/auth/verify-email", "POST")
    response = register({"email": email, "password": "password123"})
    code = response["debug_code"]
    verify_response = _Response()
    verified = verify({"email": email, "code": code}, verify_response)
    return verified["user"], verify_response.cookies


def _register_admin(app, email="admin@example.com"):
    user, cookies = _register_verified(app, email)
    app.state.store.update_user_admin_fields(user["id"], role="admin")
    return app.state.store.get_user(user["id"]), cookies


def test_register_verify_login_me_and_logout(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        register = _endpoint(app, "/api/auth/register", "POST")
        verify = _endpoint(app, "/api/auth/verify-email", "POST")
        login = _endpoint(app, "/api/auth/login", "POST")
        me = _endpoint(app, "/api/auth/me", "GET")
        preferences = _endpoint(app, "/api/auth/preferences", "PATCH")
        logout = _endpoint(app, "/api/auth/logout", "POST")

        response = register(
            {
                "email": "User@Example.com",
                "password": "password123",
                "preferred_locale": "zh",
            }
        )
        assert response["debug_code"]

        with pytest.raises(fastapi.HTTPException) as exc_info:
            login({"email": "user@example.com", "password": "password123"}, _Response())
        assert exc_info.value.status_code == 403

        verify_response = _Response()
        verified = verify(
            {"email": "user@example.com", "code": response["debug_code"]}, verify_response
        )
        assert verified["user"]["email"] == "user@example.com"
        assert verified["user"]["display_name"] == "user"
        assert "avatar_url" not in verified["user"]
        assert verified["user"]["preferred_locale"] == "zh"

        updated_preferences = preferences(
            {"preferred_locale": "en"}, _Request(verify_response.cookies)
        )
        assert updated_preferences["user"]["preferred_locale"] == "en"

        current = me(_Request(verify_response.cookies))
        assert current["user"]["email"] == "user@example.com"
        assert current["user"]["preferred_locale"] == "en"
        assert current["quota"]["daily_concurrent_experiment_limit"] == 1
        assert current["usage"]["experiments_created"] == 0
        assert current["resource_counts"]["active_experiments"] == 0

        logout_response = _Response()
        assert logout(logout_response, _Request(verify_response.cookies))["ok"] is True
        with pytest.raises(fastapi.HTTPException) as exc_info:
            me(_Request(verify_response.cookies))
        assert exc_info.value.status_code == 401

        login_response = _Response()
        logged_in = login({"email": "user@example.com", "password": "password123"}, login_response)
        assert logged_in["user"]["email"] == "user@example.com"
        assert me(_Request(login_response.cookies))["user"]["email"] == "user@example.com"
    finally:
        app.state.runner.stop()


def test_admin_user_lifecycle_quota_and_audit(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        admin_user, admin_cookies = _register_admin(app)
        target_user, target_cookies = _register_verified(app, "managed@example.com")
        list_users = _endpoint(app, "/api/admin/users", "GET")
        update_user = _endpoint(app, "/api/admin/users/{user_id}", "PATCH")
        update_quotas = _endpoint(app, "/api/admin/users/{user_id}/quotas", "PATCH")
        audit_logs = _endpoint(app, "/api/admin/audit-logs", "GET")
        login = _endpoint(app, "/api/auth/login", "POST")
        me = _endpoint(app, "/api/auth/me", "GET")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            list_users(_Request(target_cookies))
        assert exc_info.value.status_code == 403

        users_response = list_users(_Request(admin_cookies), search="managed")
        assert users_response["total"] == 1
        assert users_response["items"][0]["email"] == "managed@example.com"
        assert users_response["items"][0]["quota"]["daily_experiment_limit"] == 10

        with pytest.raises(fastapi.HTTPException) as exc_info:
            update_user(admin_user["id"], {"role": "user"}, _Request(admin_cookies))
        assert exc_info.value.status_code == 400

        disabled = update_user(target_user["id"], {"status": "disabled"}, _Request(admin_cookies))
        assert disabled["status"] == "disabled"
        with pytest.raises(fastapi.HTTPException) as exc_info:
            me(_Request(target_cookies))
        assert exc_info.value.status_code == 401
        with pytest.raises(fastapi.HTTPException) as exc_info:
            login({"email": "managed@example.com", "password": "password123"}, _Response())
        assert exc_info.value.status_code == 403

        quota = update_quotas(
            target_user["id"],
            {"daily_inference_request_limit": 7, "daily_inference_input_limit": 70},
            _Request(admin_cookies),
        )
        assert quota["daily_inference_request_limit"] == 7
        assert quota["daily_inference_input_limit"] == 70

        events = {
            item["event_type"] for item in audit_logs(_Request(admin_cookies), limit=20)["items"]
        }
        assert "admin.user_updated" in events
        assert "admin.user_quota_updated" in events
        assert "auth.login_failed" in events
    finally:
        app.state.runner.stop()


def test_admin_overview_pagination_sort_and_audit_discovery(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        admin_user, admin_cookies = _register_admin(app)
        managed_user, managed_cookies = _register_verified(app, "managed@example.com")
        register = _endpoint(app, "/api/auth/register", "POST")
        register({"email": "pending@example.com", "password": "password123"})

        store = app.state.store
        store.update_user_profile(managed_user["id"], "Zeta Researcher")
        store.increment_quota_usage(
            managed_user["id"],
            experiments_created=2,
            inference_requests=3,
            inference_inputs=12,
        )
        dataset = store.create_dataset(
            "admin overview data",
            "pep_function",
            "upload",
            "sequence_dir",
            app.state.paths.datasets / "admin-overview",
            owner_user_id=managed_user["id"],
        )
        store.create_experiment(
            "active admin overview run",
            dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
            app.state.paths.experiments / "admin-overview" / "train.log",
            app.state.paths.experiments / "admin-overview" / "artifact",
            owner_user_id=managed_user["id"],
        )
        store.create_audit_log(
            "auth.login_failed",
            actor_user_id=admin_user["id"],
            target_user_id=managed_user["id"],
            metadata={"email": managed_user["email"], "reason": "test"},
            ip_address="203.0.113.8",
        )

        overview = _endpoint(app, "/api/admin/overview", "GET")
        list_users = _endpoint(app, "/api/admin/users", "GET")
        audit_logs = _endpoint(app, "/api/admin/audit-logs", "GET")
        audit_event_types = _endpoint(app, "/api/admin/audit-event-types", "GET")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            overview(_Request(managed_cookies))
        assert exc_info.value.status_code == 403

        summary = overview(_Request(admin_cookies))
        assert summary["users"] == {
            "total": 3,
            "active": 3,
            "disabled": 0,
            "unverified": 1,
            "new_last_7_days": 3,
        }
        assert summary["workload"]["experiments_created_today"] == 2
        assert summary["workload"]["inference_requests_today"] == 3
        assert summary["workload"]["inference_inputs_today"] == 12
        assert summary["workload"]["active_experiments"] == 1
        assert summary["security_24h"]["failed_logins"] == 1

        first_page = list_users(
            _Request(admin_cookies),
            sort="display_name",
            order="asc",
            limit=2,
            offset=0,
        )
        assert first_page["total"] == 3
        assert [item["display_name"] for item in first_page["items"]] == ["admin", "pending"]
        assert all("resource_counts" in item for item in first_page["items"])

        with pytest.raises(fastapi.HTTPException) as exc_info:
            list_users(_Request(admin_cookies), sort="quota")
        assert exc_info.value.status_code == 400

        by_search = audit_logs(_Request(admin_cookies), search="managed@example.com")
        assert by_search["total"] == 1
        assert by_search["items"][0]["target"]["email"] == "managed@example.com"
        by_user_scope = audit_logs(_Request(admin_cookies), user_id=admin_user["id"])
        assert by_user_scope["total"] == 1
        assert by_user_scope["items"][0]["event_type"] == "auth.login_failed"
        assert audit_event_types(_Request(admin_cookies))["items"][0] == {
            "event_type": "auth.login_failed",
            "count": 1,
        }
    finally:
        app.state.runner.stop()


def test_admin_resends_verification_for_unverified_user(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        _admin_user, admin_cookies = _register_admin(app)
        register = _endpoint(app, "/api/auth/register", "POST")
        resend = _endpoint(app, "/api/admin/users/{user_id}/resend-verification", "POST")
        audit_logs = _endpoint(app, "/api/admin/audit-logs", "GET")

        registered = register({"email": "needs-verify@example.com", "password": "password123"})
        target = app.state.store.get_user_by_email("needs-verify@example.com")
        response = resend(target["id"], _Request(admin_cookies))

        assert registered["debug_code"]
        assert response["verification_sent"] is False
        assert response["debug_code"]
        assert response["user"]["email_verified"] is False
        events = {item["event_type"] for item in audit_logs(_Request(admin_cookies))["items"]}
        assert "admin.verification_resend" in events
    finally:
        app.state.runner.stop()


def test_admin_user_detail_revoke_sessions_and_delete_empty_user(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        admin_user, admin_cookies = _register_admin(app)
        target_user, target_cookies = _register_verified(app, "empty-user@example.com")
        get_user = _endpoint(app, "/api/admin/users/{user_id}", "GET")
        revoke_sessions = _endpoint(app, "/api/admin/users/{user_id}/revoke-sessions", "POST")
        delete_user = _endpoint(app, "/api/admin/users/{user_id}", "DELETE")
        audit_logs = _endpoint(app, "/api/admin/audit-logs", "GET")
        me = _endpoint(app, "/api/auth/me", "GET")

        detail = get_user(target_user["id"], _Request(admin_cookies))
        assert detail["can_delete"] is True
        assert detail["sessions"][0]["active"] is True
        assert detail["verification_codes"][0]["active"] is False

        with pytest.raises(fastapi.HTTPException) as exc_info:
            revoke_sessions(admin_user["id"], _Request(admin_cookies))
        assert exc_info.value.status_code == 400

        revoked = revoke_sessions(target_user["id"], _Request(admin_cookies))
        assert revoked["revoked_sessions"] == 1
        assert revoked["user"]["sessions"][0]["active"] is False
        with pytest.raises(fastapi.HTTPException) as exc_info:
            me(_Request(target_cookies))
        assert exc_info.value.status_code == 401

        deleted = delete_user(target_user["id"], _Request(admin_cookies))
        assert deleted["deleted"] is True
        assert app.state.store.get_user(target_user["id"]) is None

        with pytest.raises(fastapi.HTTPException) as exc_info:
            delete_user(admin_user["id"], _Request(admin_cookies))
        assert exc_info.value.status_code == 400

        events = {
            item["event_type"] for item in audit_logs(_Request(admin_cookies), limit=20)["items"]
        }
        assert "admin.sessions_revoked" in events
        assert "admin.user_deleted" in events
    finally:
        app.state.runner.stop()


def test_admin_delete_user_rejects_accounts_with_resources(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        _admin_user, admin_cookies = _register_admin(app)
        target_user, _target_cookies = _register_verified(app, "resource-user@example.com")
        store = app.state.store
        paths = app.state.paths
        dataset = store.create_dataset(
            "resource data",
            "pep_function",
            "upload",
            "sequence_dir",
            paths.datasets / "resource",
            owner_user_id=target_user["id"],
        )
        store.create_experiment(
            "resource run",
            dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
            paths.experiments / "resource" / "train.log",
            paths.experiments / "resource" / "artifact",
            owner_user_id=target_user["id"],
        )
        get_user = _endpoint(app, "/api/admin/users/{user_id}", "GET")
        delete_user = _endpoint(app, "/api/admin/users/{user_id}", "DELETE")

        detail = get_user(target_user["id"], _Request(admin_cookies))
        assert detail["can_delete"] is False
        assert detail["delete_block_reason"] == "user_has_resources"

        with pytest.raises(fastapi.HTTPException) as exc_info:
            delete_user(target_user["id"], _Request(admin_cookies))
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == "user_has_resources"
        assert store.get_user(target_user["id"]) is not None
    finally:
        app.state.runner.stop()


def test_inference_quota_blocks_and_records_usage(tmp_path, monkeypatch):
    class _FakeModelService:
        def predict(self, model_id, sequences=None, pairs=None):
            inputs = pairs if pairs is not None else sequences or []
            return {
                "model_id": model_id,
                "labels": ["active"],
                "predictions": [{"sequence": value, "scores": [0.5]} for value in inputs],
            }

    app = _auth_app(tmp_path, monkeypatch)
    try:
        user, cookies = _register_verified(app, "quota@example.com")
        store = app.state.store
        paths = app.state.paths
        dataset = store.create_dataset(
            "quota data",
            "pep_function",
            "upload",
            "sequence_dir",
            paths.datasets / "quota",
            owner_user_id=user["id"],
        )
        experiment = store.create_experiment(
            "quota run",
            dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
            paths.experiments / "quota" / "train.log",
            paths.experiments / "quota" / "artifact",
            owner_user_id=user["id"],
        )
        model = store.create_model(
            experiment["id"],
            "quota model",
            "pep_function",
            paths.models / "quota",
            True,
            owner_user_id=user["id"],
        )
        app.state.models = _FakeModelService()
        predict = _endpoint(app, "/api/models/{model_id}/predict", "POST")

        store.set_user_quota(user["id"], {"daily_inference_request_limit": 0})
        with pytest.raises(fastapi.HTTPException) as exc_info:
            predict(model["id"], {"sequences": ["ACD"]}, _Request(cookies))
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["quota_type"] == "daily_inference_request_limit"

        store.set_user_quota(
            user["id"],
            {"daily_inference_request_limit": 5, "daily_inference_input_limit": 1},
        )
        with pytest.raises(fastapi.HTTPException) as exc_info:
            predict(model["id"], {"sequences": ["ACD", "EFG"]}, _Request(cookies))
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["quota_type"] == "daily_inference_input_limit"

        store.set_user_quota(user["id"], {"daily_inference_input_limit": 5})
        response = predict(model["id"], {"sequences": ["ACD", "EFG"]}, _Request(cookies))
        usage = store.get_quota_usage(user["id"])

        assert response["run_id"]
        assert usage["inference_requests"] == 1
        assert usage["inference_inputs"] == 2
    finally:
        app.state.runner.stop()


def test_update_profile_changes_display_name(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        _user, cookies = _register_verified(app, "profile@example.com")
        update_profile = _endpoint(app, "/api/auth/profile", "PATCH")
        me = _endpoint(app, "/api/auth/me", "GET")

        updated = update_profile({"display_name": "Lab Operator"}, _Request(cookies))

        assert updated["user"]["display_name"] == "Lab Operator"
        assert me(_Request(cookies))["user"]["display_name"] == "Lab Operator"
        with pytest.raises(fastapi.HTTPException) as exc_info:
            update_profile({"display_name": ""}, _Request(cookies))
        assert exc_info.value.status_code == 400
    finally:
        app.state.runner.stop()


def test_user_data_isolation_returns_403_for_other_owner(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        alice_user, alice_cookies = _register_verified(app, "alice@example.com")
        _bob_user, bob_cookies = _register_verified(app, "bob@example.com")

        store = app.state.store
        paths = app.state.paths
        dataset = store.create_dataset(
            "alice data",
            "pep_function",
            "upload",
            "sequence_dir",
            paths.datasets / "alice",
            owner_user_id=alice_user["id"],
        )
        experiment = store.create_experiment(
            "alice run",
            dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
            paths.experiments / "alice" / "train.log",
            paths.experiments / "alice" / "artifact",
            owner_user_id=alice_user["id"],
        )

        list_experiments = _endpoint(app, "/api/experiments", "GET")
        get_experiment = _endpoint(app, "/api/experiments/{experiment_id}", "GET")
        assert list_experiments(_Request(alice_cookies))[0]["id"] == experiment["id"]
        assert list_experiments(_Request(bob_cookies)) == []
        with pytest.raises(fastapi.HTTPException) as exc_info:
            get_experiment(experiment["id"], _Request(bob_cookies))
        assert exc_info.value.status_code == 403
    finally:
        app.state.runner.stop()


def test_inference_run_delete_respects_owner_and_admin_permissions(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        alice_user, alice_cookies = _register_verified(app, "alice-inference@example.com")
        _bob_user, bob_cookies = _register_verified(app, "bob-inference@example.com")
        _admin_user, admin_cookies = _register_admin(app, "admin-inference@example.com")
        store = app.state.store
        paths = app.state.paths
        dataset = store.create_dataset(
            "alice data",
            "pep_function",
            "upload",
            "sequence_dir",
            paths.datasets / "alice-inference",
            owner_user_id=alice_user["id"],
        )
        experiment = store.create_experiment(
            "alice run",
            dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
            paths.experiments / "alice-inference" / "train.log",
            paths.experiments / "alice-inference" / "artifact",
            owner_user_id=alice_user["id"],
        )
        model = store.create_model(
            experiment["id"],
            "alice model",
            "pep_function",
            paths.experiments / "alice-inference" / "artifact",
            True,
            owner_user_id=alice_user["id"],
        )
        run = store.create_inference_run(
            model["id"],
            {"sequences": ["ACD"]},
            {
                "model_id": model["id"],
                "labels": ["active"],
                "predictions": [{"sequence": "ACD", "scores": [0.8]}],
            },
            owner_user_id=alice_user["id"],
        )
        delete_run = _endpoint(app, "/api/inference-runs/{run_id}", "DELETE")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            delete_run(run["id"], _Request(bob_cookies))
        assert exc_info.value.status_code == 403
        assert store.get_inference_run(run["id"]) is not None

        assert delete_run(run["id"], _Request(admin_cookies)) == {
            "run_id": run["id"],
            "deleted": True,
        }
        assert store.get_inference_run(run["id"]) is None
        with pytest.raises(fastapi.HTTPException) as exc_info:
            delete_run(run["id"], _Request(alice_cookies))
        assert exc_info.value.status_code == 404
    finally:
        app.state.runner.stop()


def test_dataset_report_respects_dataset_permissions(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        alice_user, alice_cookies = _register_verified(app, "alice-report@example.com")
        _bob_user, bob_cookies = _register_verified(app, "bob-report@example.com")

        root = app.state.paths.datasets / "alice-report"
        for split in ["train", "val"]:
            split_dir = root / split
            split_dir.mkdir(parents=True)
            (split_dir / "seqs.fasta").write_text(">s1\nACD\n", encoding="utf-8")
            (split_dir / "labels.csv").write_text("label\n1\n", encoding="utf-8")
        dataset = app.state.store.create_dataset(
            "alice report data",
            "pep_function",
            "upload",
            "sequence_dir",
            root,
            owner_user_id=alice_user["id"],
        )
        report = _endpoint(app, "/api/datasets/{dataset_id}/report", "GET")

        assert report(dataset["id"], 5, _Request(alice_cookies))["summary"]["total_samples"] == 2
        with pytest.raises(fastapi.HTTPException) as exc_info:
            report(dataset["id"], 5, _Request(bob_cookies))
        assert exc_info.value.status_code == 403
    finally:
        app.state.runner.stop()


def test_admin_leaderboard_global_view_switches_owner_scope(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch)
    try:
        store = app.state.store
        paths = app.state.paths
        admin_user = store.ensure_initial_admin("admin@example.com", hash_password("password123"))
        bob_user, _bob_cookies = _register_verified(app, "bob-board@example.com")

        admin_dataset = store.create_dataset(
            "admin board",
            "pep_function",
            "upload",
            "sequence_dir",
            paths.datasets / "admin-board",
            owner_user_id=admin_user["id"],
        )
        bob_dataset = store.create_dataset(
            "bob board",
            "pep_function",
            "upload",
            "sequence_dir",
            paths.datasets / "bob-board",
            owner_user_id=bob_user["id"],
        )
        admin_experiment = store.create_experiment(
            "admin run",
            admin_dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "index", "model": "cnn"},
            paths.experiments / "admin-board.log",
            paths.experiments / "admin-board",
            owner_user_id=admin_user["id"],
        )
        bob_experiment = store.create_experiment(
            "bob run",
            bob_dataset["id"],
            "pep_function",
            {"tokenizer": "kmer", "encoder": "index", "model": "rf"},
            paths.experiments / "bob-board.log",
            paths.experiments / "bob-board",
            owner_user_id=bob_user["id"],
        )
        admin_model = store.create_model(
            admin_experiment["id"],
            "admin model",
            "pep_function",
            paths.models / "admin-board",
            True,
            owner_user_id=admin_user["id"],
        )
        bob_model = store.create_model(
            bob_experiment["id"],
            "bob model",
            "pep_function",
            paths.models / "bob-board",
            True,
            owner_user_id=bob_user["id"],
        )
        store.append_metric_event(admin_experiment["id"], {"kind": "test", "test_f1": 0.7})
        store.append_metric_event(bob_experiment["id"], {"kind": "test", "test_f1": 0.8})
        store.update_experiment(
            admin_experiment["id"], status="succeeded", model_id=admin_model["id"]
        )
        store.update_experiment(bob_experiment["id"], status="succeeded", model_id=bob_model["id"])

        login_response = _Response()
        _endpoint(app, "/api/auth/login", "POST")(
            {"email": "admin@example.com", "password": "password123"},
            login_response,
        )
        leaderboard = _endpoint(app, "/api/leaderboard", "GET")
        own_response = leaderboard(
            request=_Request(login_response.cookies),
            task_type="",
            dataset_id="",
            metric="f1",
            split="auto",
            supports_inference=None,
            status="succeeded",
            limit=20,
            global_view=False,
        )
        global_response = leaderboard(
            request=_Request(login_response.cookies),
            task_type="",
            dataset_id="",
            metric="f1",
            split="auto",
            supports_inference=None,
            status="succeeded",
            limit=20,
            global_view=True,
        )

        assert own_response["filters"]["global_view"] is False
        assert [row["experiment"]["name"] for row in own_response["rows"]] == ["admin run"]
        assert global_response["filters"]["global_view"] is True
        assert [row["experiment"]["name"] for row in global_response["rows"]] == [
            "bob run",
            "admin run",
        ]
    finally:
        app.state.runner.stop()


def test_removed_avatar_routes_return_not_found(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app = _auth_app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        for method in ("GET", "POST", "DELETE"):
            assert client.request(method, "/api/auth/avatar").status_code == 404


def test_profile_response_ignores_legacy_avatar_columns(tmp_path, monkeypatch):
    from sqlalchemy import text

    app = _auth_app(tmp_path, monkeypatch)
    try:
        user, cookies = _register_verified(app, "legacy-profile@example.com")
        with app.state.store.engine.begin() as connection:
            connection.execute(
                text("UPDATE users SET avatar_object_key = :key, avatar_content_type = :kind, "
                     "avatar_updated_at = :stamp WHERE id = :id"),
                {"key": "avatars/legacy.png", "kind": "image/png", "stamp": "preserved", "id": user["id"]},
            )
        response = _endpoint(app, "/api/auth/me", "GET")(_Request(cookies))
        assert "avatar_url" not in response["user"]
        assert not any(key.startswith("avatar_") for key in response["user"])
        app.state.store.update_user_profile(user["id"], "Updated Name")
        with app.state.store.engine.connect() as connection:
            row = connection.execute(
                text("SELECT avatar_object_key, avatar_content_type, avatar_updated_at FROM users WHERE id = :id"),
                {"id": user["id"]},
            ).one()
        assert tuple(row) == ("avatars/legacy.png", "image/png", "preserved")
    finally:
        app.state.store.engine.dispose()


def test_app_has_no_object_storage_dependency(tmp_path, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_object_storage(name, *args, **kwargs):
        if name.split(".")[0] in {"boto3", "botocore"}:
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_object_storage)
    app = _auth_app(tmp_path, monkeypatch)
    try:
        assert not hasattr(app.state, "avatar_storage")
        assert app.state.store is not None
    finally:
        app.state.store.engine.dispose()
