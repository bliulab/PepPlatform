import smtplib
import ssl

import pytest

from pepplatform.web import auth as auth_module
from pepplatform.web.auth import (
    AuthSettings,
    open_smtp_connection,
    send_verification_email,
    smtp_ssl_context,
)

SMTP_ENV = [
    "PEP_SMTP_HOST",
    "PEP_SMTP_PORT",
    "PEP_SMTP_USER",
    "PEP_SMTP_PASSWORD",
    "PEP_SMTP_FROM",
    "PEP_SMTP_SSL",
    "PEP_SMTP_STARTTLS",
    "PEP_SMTP_TIMEOUT",
    "PEP_ENV",
    "PEP_AUTH_DEBUG_CODES",
]

QQ_ENV = {
    "PEP_SMTP_HOST": "smtp.qq.com",
    "PEP_SMTP_PORT": "587",
    "PEP_SMTP_USER": "sender@qq.com",
    "PEP_SMTP_PASSWORD": "mailbox-authorization-code",
    "PEP_SMTP_FROM": "sender@qq.com",
}


@pytest.fixture(autouse=True)
def clean_smtp_env(monkeypatch):
    for name in SMTP_ENV:
        monkeypatch.delenv(name, raising=False)


def settings_with(monkeypatch, **overrides):
    values = dict(QQ_ENV)
    values.update(overrides)
    for name, value in values.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(value))
    return AuthSettings.from_env()


class FakeSMTP:
    def __init__(self, host=None, port=None, timeout=None, context=None, **_kwargs):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.context = context
        self.starttls_kwargs = None
        self.login_calls = []
        self.sent = []
        self.closed = False
        self.raise_on_login = None
        self.raise_on_send = None
        self.raise_on_starttls = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.closed = True
        return False

    def starttls(self, **kwargs):
        if self.raise_on_starttls is not None:
            raise self.raise_on_starttls
        self.starttls_kwargs = kwargs

    def login(self, user, password):
        self.login_calls.append((user, password))
        if self.raise_on_login is not None:
            raise self.raise_on_login

    def send_message(self, message):
        if self.raise_on_send is not None:
            raise self.raise_on_send
        self.sent.append(message)
        return {}

    def close(self):
        self.closed = True


def install_fake_smtp(monkeypatch, **behavior):
    created = []

    def factory(*args, **kwargs):
        instance = FakeSMTP(*args, **kwargs)
        for name, value in behavior.items():
            setattr(instance, name, value)
        created.append(instance)
        return instance

    monkeypatch.setattr(auth_module.smtplib, "SMTP", factory)
    monkeypatch.setattr(auth_module.smtplib, "SMTP_SSL", factory)
    return created


def test_port_587_uses_starttls(monkeypatch):
    settings = settings_with(monkeypatch)
    assert settings.smtp_security == "starttls"
    assert settings.smtp_ssl is False
    assert settings.smtp_endpoint == "smtp.qq.com:587/starttls"


def test_port_465_switches_to_implicit_tls(monkeypatch):
    settings = settings_with(monkeypatch, PEP_SMTP_PORT="465")
    assert settings.smtp_security == "ssl"
    assert settings.smtp_starttls is False
    assert settings.smtp_endpoint == "smtp.qq.com:465/ssl"


@pytest.mark.parametrize(
    ("port", "override", "expected"),
    [
        ("587", "1", "ssl"),
        ("465", "0", "starttls"),
        ("2525", "yes", "ssl"),
        ("2525", "off", "starttls"),
        ("587", "auto", "starttls"),
    ],
)
def test_explicit_transport_override_wins(monkeypatch, port, override, expected):
    settings = settings_with(monkeypatch, PEP_SMTP_PORT=port, PEP_SMTP_SSL=override)
    assert settings.smtp_security == expected


def test_starttls_verifies_the_server_certificate(monkeypatch):
    created = install_fake_smtp(monkeypatch)
    settings = settings_with(monkeypatch)

    assert send_verification_email("user@example.com", "123456", "email_verify", settings) is True

    context = created[0].starttls_kwargs["context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert created[0].login_calls == [("sender@qq.com", "mailbox-authorization-code")]
    assert created[0].sent and created[0].closed is True


def test_implicit_tls_connection_uses_verifying_context(monkeypatch):
    created = install_fake_smtp(monkeypatch)
    settings = settings_with(monkeypatch, PEP_SMTP_PORT="465")

    connection = open_smtp_connection(settings)
    connection.close()

    assert created[0].port == 465
    assert created[0].starttls_kwargs is None
    assert created[0].context.verify_mode == ssl.CERT_REQUIRED
    assert created[0].context.check_hostname is True


def test_disabling_starttls_skips_the_upgrade(monkeypatch):
    created = install_fake_smtp(monkeypatch)
    settings = settings_with(monkeypatch, PEP_SMTP_PORT="1025", PEP_SMTP_STARTTLS="0")

    connection = open_smtp_connection(settings)
    connection.close()

    assert settings.smtp_security == "plain"
    assert created[0].starttls_kwargs is None


def test_smtp_ssl_context_factory_is_verifying():
    context = smtp_ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_authentication_failure_is_contained(monkeypatch, caplog):
    install_fake_smtp(
        monkeypatch,
        raise_on_login=smtplib.SMTPAuthenticationError(
            535, b"Login Fail. Please enter your authorization code"
        ),
    )

    settings = settings_with(monkeypatch)
    with caplog.at_level("ERROR"):
        sent = send_verification_email("user@example.com", "123456", "email_verify", settings)

    assert sent is False
    assert "authorization code" in caplog.text
    assert "mailbox-authorization-code" not in caplog.text


def test_delivery_failure_is_contained(monkeypatch, caplog):
    created = install_fake_smtp(monkeypatch, raise_on_send=OSError("network unreachable"))

    settings = settings_with(monkeypatch)
    with caplog.at_level("ERROR"):
        sent = send_verification_email("user@example.com", "123456", "password_reset", settings)

    assert sent is False
    assert created[0].sent == []
    assert "network unreachable" in caplog.text


def test_starttls_failure_closes_the_connection(monkeypatch):
    created = install_fake_smtp(monkeypatch, raise_on_starttls=OSError("handshake failed"))

    settings = settings_with(monkeypatch)
    with pytest.raises(OSError):
        open_smtp_connection(settings)

    assert created[0].closed is True


def test_unconfigured_smtp_skips_delivery(monkeypatch):
    created = install_fake_smtp(monkeypatch)
    monkeypatch.delenv("PEP_SMTP_HOST", raising=False)

    settings = settings_with(monkeypatch, PEP_SMTP_HOST=None)
    assert settings.smtp_configured is False
    assert send_verification_email("user@example.com", "123456", "email_verify", settings) is False
    assert created == []


def test_smtp_from_falls_back_to_user(monkeypatch):
    settings = settings_with(monkeypatch, PEP_SMTP_FROM=None)
    assert settings.smtp_from == "sender@qq.com"
    assert settings.smtp_config_issues() == []


def test_configuration_issues_report_cleartext_and_sender_mismatch(monkeypatch):
    settings = settings_with(
        monkeypatch,
        PEP_SMTP_STARTTLS="0",
        PEP_SMTP_FROM="alias@qq.com",
    )
    issues = " ".join(settings.smtp_config_issues())
    assert "cleartext" in issues
    assert "PEP_SMTP_FROM differs from PEP_SMTP_USER" in issues


def test_configuration_issues_report_missing_credentials(monkeypatch):
    settings = settings_with(
        monkeypatch, PEP_SMTP_USER=None, PEP_SMTP_FROM=None, PEP_SMTP_PASSWORD=None
    )
    issues = " ".join(settings.smtp_config_issues())
    assert "PEP_SMTP_USER is not set" in issues
    assert "PEP_SMTP_FROM is not set" in issues


def test_debug_codes_default_on_in_development(monkeypatch):
    settings = settings_with(monkeypatch)
    assert settings.env == "development"
    assert settings.debug_codes is True


def test_debug_codes_default_off_in_production(monkeypatch):
    settings = settings_with(monkeypatch, PEP_ENV="production")
    assert settings.is_production is True
    assert settings.debug_codes is False


def test_debug_codes_explicit_override(monkeypatch):
    assert (
        settings_with(monkeypatch, PEP_ENV="production", PEP_AUTH_DEBUG_CODES="1").debug_codes
        is True
    )
    assert (
        settings_with(monkeypatch, PEP_ENV="development", PEP_AUTH_DEBUG_CODES="0").debug_codes
        is False
    )


def test_smtp_timeout_env_is_respected(monkeypatch):
    assert settings_with(monkeypatch, PEP_SMTP_TIMEOUT="3").smtp_timeout == 3
    assert settings_with(monkeypatch, PEP_SMTP_TIMEOUT="0").smtp_timeout == 1


def _register_endpoint(app):
    for route in app.routes:
        if getattr(route, "path", "") == "/api/auth/register" and "POST" in getattr(
            route, "methods", set()
        ):
            return route.endpoint
    raise AssertionError("POST /api/auth/register endpoint was not registered")


def _auth_app(tmp_path, monkeypatch, **env):
    pytest.importorskip("fastapi")
    from pepplatform.web.app import create_app

    monkeypatch.setenv("PEP_AUTH_ENABLED", "1")
    monkeypatch.setenv("PEP_SESSION_SECRET", "test-session-secret")
    monkeypatch.delenv("PEP_SMTP_HOST", raising=False)
    for name in env:
        monkeypatch.setenv(name, env[name])
    project_root = tmp_path / "project"
    project_root.mkdir()
    return create_app(project_root=project_root, runtime_root=tmp_path / "runtime")


def test_register_hides_debug_code_in_production(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, PEP_ENV="production")

    response = _register_endpoint(app)({"email": "new-user@example.com", "password": "password123"})

    assert response["verification_sent"] is False
    assert "debug_code" not in response


def test_register_returns_debug_code_in_development(tmp_path, monkeypatch):
    app = _auth_app(tmp_path, monkeypatch, PEP_ENV="development")

    response = _register_endpoint(app)({"email": "new-user@example.com", "password": "password123"})

    assert response["verification_sent"] is False
    assert response["debug_code"]
