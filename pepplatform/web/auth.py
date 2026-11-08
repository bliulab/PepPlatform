import hashlib
import hmac
import logging
import os
import secrets
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from urllib.parse import urlparse

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError


SESSION_COOKIE_NAME = "pep_session"
CODE_PURPOSE_EMAIL_VERIFY = "email_verify"
CODE_PURPOSE_PASSWORD_RESET = "password_reset"

# Ports that serve SMTP over implicit TLS and never speak STARTTLS.
SMTP_IMPLICIT_TLS_PORTS = frozenset({465, 994})
PRODUCTION_ENV_NAMES = frozenset({"production", "prod"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})

logger = logging.getLogger(__name__)

_password_hasher = PasswordHasher(time_cost=2, memory_cost=19_456, parallelism=1)


def env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in _TRUTHY


def env_tristate(name):
    """Read an on/off/auto switch: True, False, or None when unset (auto)."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    return None


def redact_email(address):
    """Mask the local part so addresses can appear in logs."""
    text = str(address or "").strip()
    local, _, domain = text.partition("@")
    if not domain:
        return (local[:2] + "***") if local else ""
    return f"{local[:2]}***@{domain}"


def smtp_error_text(exc):
    """Render an SMTP exception without leaking the configured password."""
    code = getattr(exc, "smtp_code", None)
    detail = getattr(exc, "smtp_error", None)
    if isinstance(detail, bytes):
        detail = detail.decode("utf-8", "replace")
    if code and detail:
        return f"{code} {detail}".strip()
    return str(exc).strip() or exc.__class__.__name__


def env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def public_base_url_uses_https(public_base_url):
    return urlparse(public_base_url).scheme.lower() == "https"


@dataclass(frozen=True)
class AuthSettings:
    enabled: bool
    session_secret: str
    public_base_url: str
    cookie_secure: bool
    session_days: int
    code_ttl_minutes: int
    code_max_attempts: int
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    smtp_from: str
    initial_admin_email: str
    initial_admin_password: str
    smtp_ssl: bool = False
    smtp_starttls: bool = True
    smtp_timeout: int = 15
    env: str = "development"
    debug_codes: bool = True

    @classmethod
    def from_env(cls):
        enabled = env_bool("PEP_AUTH_ENABLED", True)
        public_base_url = os.environ.get("PEP_PUBLIC_BASE_URL", "http://127.0.0.1:5173").rstrip("/")
        smtp_port = env_int("PEP_SMTP_PORT", 587)
        explicit_ssl = env_tristate("PEP_SMTP_SSL")
        # Implicit TLS when the port is a known SMTPS port unless explicitly overridden.
        smtp_ssl = (
            explicit_ssl if explicit_ssl is not None else smtp_port in SMTP_IMPLICIT_TLS_PORTS
        )
        env_name = str(os.environ.get("PEP_ENV") or "development").strip().lower() or "development"
        return cls(
            enabled=enabled,
            session_secret=os.environ.get("PEP_SESSION_SECRET", "dev-insecure-session-secret"),
            public_base_url=public_base_url,
            cookie_secure=env_bool(
                "PEP_COOKIE_SECURE", public_base_url_uses_https(public_base_url)
            ),
            session_days=env_int("PEP_SESSION_DAYS", 14),
            code_ttl_minutes=env_int("PEP_EMAIL_CODE_TTL_MINUTES", 10),
            code_max_attempts=env_int("PEP_EMAIL_CODE_MAX_ATTEMPTS", 5),
            smtp_host=os.environ.get("PEP_SMTP_HOST", ""),
            smtp_port=smtp_port,
            smtp_user=os.environ.get("PEP_SMTP_USER", ""),
            smtp_password=os.environ.get("PEP_SMTP_PASSWORD", ""),
            smtp_from=os.environ.get("PEP_SMTP_FROM", os.environ.get("PEP_SMTP_USER", "")),
            initial_admin_email=os.environ.get("PEP_INITIAL_ADMIN_EMAIL", ""),
            initial_admin_password=os.environ.get("PEP_INITIAL_ADMIN_PASSWORD", ""),
            smtp_ssl=smtp_ssl,
            smtp_starttls=(not smtp_ssl) and env_bool("PEP_SMTP_STARTTLS", True),
            smtp_timeout=max(1, env_int("PEP_SMTP_TIMEOUT", 15)),
            env=env_name,
            debug_codes=env_bool("PEP_AUTH_DEBUG_CODES", env_name not in PRODUCTION_ENV_NAMES),
        )

    @property
    def smtp_configured(self):
        return bool(self.smtp_host and self.smtp_from)

    @property
    def smtp_security(self):
        """Transport in use: ``ssl`` (implicit TLS), ``starttls``, or ``plain``."""
        if self.smtp_ssl:
            return "ssl"
        if self.smtp_starttls:
            return "starttls"
        return "plain"

    @property
    def smtp_endpoint(self):
        return f"{self.smtp_host or '-'}:{self.smtp_port}/{self.smtp_security}"

    @property
    def is_production(self):
        return self.env in PRODUCTION_ENV_NAMES

    def smtp_config_issues(self):
        """Return human-readable configuration problems for email delivery."""
        issues = []
        if not self.smtp_host:
            issues.append("PEP_SMTP_HOST is not set")
        if not self.smtp_from:
            issues.append("PEP_SMTP_FROM is not set (it falls back to PEP_SMTP_USER)")
        if self.smtp_host and not self.smtp_user:
            issues.append("PEP_SMTP_USER is not set; most providers require authenticated SMTP")
        if self.smtp_user and not self.smtp_password:
            issues.append("PEP_SMTP_PASSWORD is not set; login will fail")
        if self.smtp_host and self.smtp_security == "plain":
            issues.append(
                "both PEP_SMTP_SSL and PEP_SMTP_STARTTLS are disabled; credentials and messages "
                "would travel in cleartext"
            )
        if self.smtp_user and self.smtp_from and self.smtp_user.lower() != self.smtp_from.lower():
            issues.append(
                "PEP_SMTP_FROM differs from PEP_SMTP_USER; providers such as QQ/163 reject a "
                "sender that is not the authenticated account"
            )
        return issues


def utcnow():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


def parse_iso(value):
    return datetime.fromisoformat(str(value))


def hash_password(password):
    return _password_hasher.hash(password)


def verify_password(password, password_hash):
    try:
        return _password_hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def generate_code():
    return f"{secrets.randbelow(1_000_000):06d}"


def generate_session_token():
    return secrets.token_urlsafe(48)


def secret_hash(value, secret):
    return hmac.new(secret.encode("utf-8"), str(value).encode("utf-8"), hashlib.sha256).hexdigest()


def session_expires_at(settings):
    return iso(utcnow() + timedelta(days=max(1, settings.session_days)))


def code_expires_at(settings):
    return iso(utcnow() + timedelta(minutes=max(1, settings.code_ttl_minutes)))


def cookie_kwargs(settings):
    return {
        "httponly": True,
        "secure": settings.cookie_secure,
        "samesite": "lax",
        "path": "/",
        "max_age": max(1, settings.session_days) * 24 * 60 * 60,
    }


def clear_cookie_kwargs(settings):
    return {
        "httponly": True,
        "secure": settings.cookie_secure,
        "samesite": "lax",
        "path": "/",
    }


def verification_body(email, code, purpose, settings):
    if purpose == CODE_PURPOSE_PASSWORD_RESET:
        title = "PepPlatform password reset code"
        action = "reset your password"
    else:
        title = "PepPlatform email verification code"
        action = "verify your email"
    body = (
        f"Use this code to {action}: {code}\n\n"
        f"The code expires in {settings.code_ttl_minutes} minutes. "
        "If you did not request this email, you can ignore it."
    )
    return title, body


def smtp_ssl_context():
    """Client TLS context that verifies the mail server certificate.

    ``smtplib``'s default STARTTLS context is unverified (``CERT_NONE``), which
    lets an on-path attacker harvest the mailbox password or authorization code.
    """
    return ssl.create_default_context()


def open_smtp_connection(settings):
    """Open an SMTP connection matching the configured transport.

    Returns a live ``smtplib`` connection that is already upgraded to TLS when
    the transport requires it; the caller owns it and must close it.
    """
    context = smtp_ssl_context()
    if settings.smtp_ssl:
        return smtplib.SMTP_SSL(
            settings.smtp_host,
            settings.smtp_port,
            timeout=settings.smtp_timeout,
            context=context,
        )
    smtp = smtplib.SMTP(
        settings.smtp_host,
        settings.smtp_port,
        timeout=settings.smtp_timeout,
    )
    if not settings.smtp_starttls:
        return smtp
    try:
        smtp.starttls(context=context)
    except (smtplib.SMTPException, ssl.SSLError, OSError):
        smtp.close()
        raise
    return smtp


def log_auth_mail_configuration(settings, logger=None):
    """Report the effective email delivery configuration at startup."""
    log = logger or logging.getLogger(__name__)
    if not settings.smtp_configured:
        log.info("auth: SMTP is not configured; verification and reset emails are disabled")
    else:
        log.info(
            "auth: SMTP %s sender=%s code_ttl=%smin",
            settings.smtp_endpoint,
            redact_email(settings.smtp_from),
            settings.code_ttl_minutes,
        )
    for issue in settings.smtp_config_issues():
        log.warning("auth: SMTP configuration warning: %s", issue)
    if settings.debug_codes:
        log.warning(
            "auth: plaintext verification codes are returned in API responses "
            "(PEP_ENV=%s); set PEP_ENV=production or PEP_AUTH_DEBUG_CODES=0 for public "
            "deployments",
            settings.env,
        )


def send_verification_email(email, code, purpose, settings):
    """Deliver a verification code. Returns ``True`` only when handed to the server."""
    if not settings.smtp_configured:
        logger.warning(
            "email delivery skipped for %s: SMTP is not configured (set PEP_SMTP_HOST/PEP_SMTP_FROM)",
            redact_email(email),
        )
        return False
    subject, body = verification_body(email, code, purpose, settings)
    try:
        message = EmailMessage()
        message["From"] = settings.smtp_from
        message["To"] = email
        message["Subject"] = subject
        message.set_content(body)
        with open_smtp_connection(settings) as smtp:
            if settings.smtp_user:
                smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        logger.error(
            "SMTP authentication rejected by %s for %s: %s -- use the mailbox authorization code, "
            "not the account login password",
            settings.smtp_endpoint,
            redact_email(email),
            smtp_error_text(exc),
        )
        return False
    except (smtplib.SMTPException, ssl.SSLError, OSError, ValueError) as exc:
        logger.error(
            "SMTP delivery to %s via %s failed: %s",
            redact_email(email),
            settings.smtp_endpoint,
            smtp_error_text(exc),
        )
        return False
    return True
