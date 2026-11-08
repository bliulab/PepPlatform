"""The SMTP account has one durable budget, independent of scientific quotas."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from test_resource_governance import app, change_policy, endpoint, user  # noqa: F401
from test_web_auth_smtp import install_fake_smtp

from pepplatform.web import auth as auth_module
from pepplatform.web.auth import CODE_PURPOSE_EMAIL_VERIFY, CODE_PURPOSE_PASSWORD_RESET, secret_hash
from pepplatform.web.resource_policy import ResourceLimitError, verification_email_daily_usage
from pepplatform.web.store import MetadataStore, email_verification_codes


@pytest.fixture
def mail_app(request, monkeypatch):
    for name, value in {
        "PEP_SMTP_HOST": "smtp.example.invalid",
        "PEP_SMTP_USER": "sender@example.invalid",
        "PEP_SMTP_PASSWORD": "test-only",
        "PEP_SMTP_FROM": "sender@example.invalid",
    }.items():
        monkeypatch.setenv(name, value)
    created = install_fake_smtp(monkeypatch)
    application = request.getfixturevalue("app")
    original = auth_module.open_smtp_connection

    def outside_transaction(settings):
        assert application.state.store._transaction.get() is None
        return original(settings)

    monkeypatch.setattr(auth_module, "open_smtp_connection", outside_transaction)
    return application, created


def pending_user(application, name="pending"):
    return application.state.store.create_user(f"{name}@example.test", "unused")


def test_twenty_attempts_are_shared_by_all_four_sending_entries(mail_app):
    application, smtp = mail_app
    store = application.state.store
    owner = pending_user(application)
    _, administrator = user(application, "administrator", admin=True)
    client = TestClient(application)
    for index in range(5):
        responses = [
            client.post(
                "/api/auth/register",
                json={
                    "email": f"new-{index}@example.test",
                    "password": "password123",
                },
            ),
            client.post("/api/auth/resend-verification", json={"email": owner["email"]}),
            client.post("/api/auth/password-reset/request", json={"email": owner["email"]}),
            client.post(
                f"/api/admin/users/{owner['id']}/resend-verification",
                json={},
                headers={"Cookie": "; ".join(f"{k}={v}" for k, v in administrator.cookies.items())},
            ),
        ]
        assert [response.status_code for response in responses] == [200] * 4
        assert all(
            response.json().get("verification_sent", response.json().get("reset_sent"))
            for response in responses
        )
    assert len(smtp) == 20
    assert sum(len(connection.sent) for connection in smtp) == 20
    assert store.resources.email_delivery_usage()["used"] == 20
    response = client.post("/api/auth/resend-verification", json={"email": owner["email"]})
    assert response.status_code == 429
    assert len(smtp) == 20


@pytest.mark.parametrize("kind", ["register", "resend", "reset", "admin"])
@pytest.mark.parametrize("limit", [0, 1])
def test_blocked_attempts_preserve_accounts_codes_and_registration_count(mail_app, kind, limit):
    application, smtp = mail_app
    store = application.state.store
    owner = pending_user(application)
    _, administrator = user(application, "administrator", admin=True)
    purpose = CODE_PURPOSE_PASSWORD_RESET if kind == "reset" else CODE_PURPOSE_EMAIL_VERIFY
    old = store.create_email_code(
        owner["id"],
        purpose,
        secret_hash(
            f"{owner['id']}:{purpose}:314159", application.state.auth_settings.session_secret
        ),
        "2099-01-01T00:00:00+00:00",
    )
    change_policy(application, limits={"daily_verification_email_limit": limit})
    if limit:
        store.resources.reserve_verification_email()
    paths = {
        "register": "/api/auth/register",
        "resend": "/api/auth/resend-verification",
        "reset": "/api/auth/password-reset/request",
        "admin": f"/api/admin/users/{owner['id']}/resend-verification",
    }
    payload = (
        {"email": "blocked-new@example.test", "password": "password123"}
        if kind == "register"
        else {"email": owner["email"]}
    )
    response = TestClient(application).post(
        paths[kind],
        json=payload,
        headers={
            "Cookie": "; ".join(f"{k}={v}" for k, v in administrator.cookies.items()),
        },
    )
    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail["code"] == "quota_exceeded"
    assert detail["quota_type"] == "daily_verification_email_limit"
    assert detail["limit"] == limit and detail["used"] == limit
    assert ("reset_at" in detail) is bool(limit)
    assert ("Retry-After" in response.headers) is bool(limit)
    assert "debug_code" not in response.json()
    assert smtp == []
    assert store.get_active_email_code(owner["id"], purpose) == old
    assert store.get_user_by_email("blocked-new@example.test") is None
    assert store.resources.daily("__public_registration__")["registrations"] == 0
    assert store.resources.email_delivery_usage()["used"] == limit


def test_failed_smtp_attempt_still_uses_budget_and_creates_only_one_account(mail_app, monkeypatch):
    import smtplib

    application, _ = mail_app
    change_policy(application, limits={"daily_verification_email_limit": 1})
    monkeypatch.setattr(auth_module.smtplib, "SMTP", lambda *a, **kw: failing_smtp)
    from test_web_auth_smtp import FakeSMTP

    failing_smtp = FakeSMTP()
    failing_smtp.raise_on_send = smtplib.SMTPDataError(550, b"delivery rejected")
    register = endpoint(application, "/api/auth/register", "POST")
    result = register({"email": "failed@example.test", "password": "password123"})
    assert result["verification_sent"] is False
    assert application.state.store.resources.email_delivery_usage()["used"] == 1
    with pytest.raises(HTTPException) as error:
        register({"email": "next@example.test", "password": "password123"})
    assert error.value.status_code == 429
    assert application.state.store.get_user_by_email("next@example.test") is None


def test_invalid_or_unnecessary_requests_do_not_consume_email_budget(mail_app):
    application, smtp = mail_app
    verified, administrator = user(application, "verified", admin=True)
    client = TestClient(application)
    for payload in [
        {"email": "bad", "password": "password123"},
        {"email": "short@example.test", "password": "bad"},
        {"email": verified["email"], "password": "password123"},
    ]:
        assert client.post("/api/auth/register", json=payload).status_code in {400, 409}
    assert (
        client.post("/api/auth/resend-verification", json={"email": verified["email"]}).status_code
        == 200
    )
    assert (
        client.post(
            "/api/auth/password-reset/request", json={"email": "absent@example.test"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/api/admin/users/{verified['id']}/resend-verification",
            json={},
            headers={
                "Cookie": "; ".join(f"{k}={v}" for k, v in administrator.cookies.items()),
            },
        ).status_code
        == 200
    )
    assert application.state.store.resources.email_delivery_usage()["used"] == 0
    assert application.state.store.resources.daily("__public_registration__")["registrations"] == 0
    assert smtp == []


def test_last_email_slot_is_atomic_across_independent_stores(mail_app):
    application, smtp = mail_app
    store = application.state.store
    owner = pending_user(application)
    change_policy(application, limits={"daily_verification_email_limit": 1})
    other = MetadataStore(store.database)
    resend = endpoint(application, "/api/auth/resend-verification", "POST")

    def send(index):
        try:
            if index % 2:
                other.resources.reserve_verification_email()
                return "reserved"
            resend({"email": owner["email"]})
            return "sent"
        except ResourceLimitError:
            return "blocked"

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(send, range(8)))
        assert results.count("blocked") == 7
        assert len(smtp) == results.count("sent")
        assert store.resources.email_delivery_usage()["used"] == 1
        assert other.resources.email_delivery_usage()["used"] == 1
    finally:
        other.engine.dispose()


def test_counter_survives_code_deletion_restart_and_does_not_follow_user_usage(mail_app):
    application, smtp = mail_app
    store = application.state.store
    owner = pending_user(application)
    endpoint(application, "/api/auth/resend-verification", "POST")({"email": owner["email"]})
    with store.connect() as conn:
        conn.execute(
            delete(email_verification_codes).where(
                email_verification_codes.c.user_id == owner["id"]
            )
        )
    removed, _ = store.delete_user_without_resources(owner["id"])
    assert removed is not None
    restarted = MetadataStore(store.database)
    try:
        assert restarted.resources.email_delivery_usage()["used"] == 1
        assert restarted.resources.daily(owner["id"])["registrations"] == 0
        assert len(smtp) == 1
    finally:
        restarted.engine.dispose()


def test_email_attempt_crossing_utc_midnight_charges_its_original_day(
    mail_app, quota_clock, monkeypatch
):
    application, _ = mail_app
    policies = application.state.store.resources
    quota_clock.value = datetime(2026, 11, 8, 23, 59, 59, tzinfo=timezone.utc)
    original = policies.policy

    def crossing():
        quota_clock.value = datetime(2026, 11, 9, tzinfo=timezone.utc)
        return original()

    monkeypatch.setattr(policies, "policy", crossing)
    policies.reserve_verification_email()
    monkeypatch.setattr(policies, "policy", original)
    assert policies.email_delivery_usage()["used"] == 0
    assert policies.email_delivery_usage()["reset_at"] == "2026-11-10T00:00:00+00:00"
    policies.reserve_verification_email()
    with application.state.store.connect() as conn:
        assert dict(
            conn.execute(
                select(
                    verification_email_daily_usage.c.usage_date,
                    verification_email_daily_usage.c.attempts,
                )
            ).all()
        ) == {
            "2026-11-08": 1,
            "2026-11-09": 1,
        }


def test_limit_changes_keep_attempts_and_never_publish_user_announcements(mail_app, quota_clock):
    application, _ = mail_app
    policies = application.state.store.resources
    owner, _ = user(application)
    for _ in range(2):
        policies.reserve_verification_email()
    for value, remaining in [(1, 0), (0, 0), (3, 1)]:
        change_policy(
            application,
            limits={"daily_verification_email_limit": value},
            publish_announcements=True,
        )
        assert policies.email_delivery_usage()["used"] == 2
        assert policies.email_delivery_usage()["remaining"] == remaining
        assert policies.announcements(owner) == []
        if not remaining:
            with pytest.raises(ResourceLimitError):
                policies.reserve_verification_email()
    policies.reserve_verification_email()
    with pytest.raises(ResourceLimitError):
        policies.reserve_verification_email()
    quota_clock.next_day()
    assert policies.email_delivery_usage()["remaining"] == 3


@pytest.mark.parametrize("value", [-1, 0.5, True, "20", None])
def test_invalid_email_limits_leave_policy_unchanged(mail_app, value):
    application, _ = mail_app
    policies = application.state.store.resources
    before = policies.policy()
    with pytest.raises(ValueError):
        change_policy(application, limits={"daily_verification_email_limit": value})
    assert policies.policy() == before


def test_email_usage_requires_an_administrator(mail_app):
    application, _ = mail_app
    _, ordinary = user(application)
    _, administrator = user(application, "administrator", admin=True)
    usage = endpoint(application, "/api/admin/email-delivery-usage", "GET")
    for request in [None, ordinary]:
        with pytest.raises(HTTPException):
            usage(request)
    assert usage(administrator)["limit"] == 20


@pytest.mark.parametrize("limit", [0, 20])
def test_development_codes_without_smtp_do_not_consume_a_real_attempt(request, limit):
    application = request.getfixturevalue("app")
    change_policy(application, limits={"daily_verification_email_limit": limit})
    register = endpoint(application, "/api/auth/register", "POST")
    if limit == 0:
        with pytest.raises(HTTPException) as error:
            register({"email": "development@example.test", "password": "password123"})
        assert error.value.detail["quota_type"] == "daily_verification_email_limit"
        assert application.state.store.get_user_by_email("development@example.test") is None
        assert application.state.store.resources.email_delivery_usage()["used"] == 0
        return
    result = endpoint(application, "/api/auth/register", "POST")(
        {
            "email": "development@example.test",
            "password": "password123",
        }
    )
    assert result["debug_code"]
    assert application.state.store.resources.email_delivery_usage()["used"] == 0
