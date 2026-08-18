"""Dashboard cookie and GitHub Action bearer auth."""

from __future__ import annotations

import hashlib
import hmac
import os

COOKIE_NAME = "regression_ci"


def trigger_secret() -> str:
    return os.environ.get("CI_SHARED_SECRET", "")


def dashboard_password() -> str:
    return os.environ.get("DASHBOARD_PASSWORD", "")


def session_token(password: str | None = None) -> str:
    """Return the cookie value for a successful dashboard login."""
    secret = password if password is not None else dashboard_password()
    return hmac.new(secret.encode("utf-8"), b"session", hashlib.sha256).hexdigest()


def bearer_matches(header: str | None, secret: str | None = None) -> bool:
    """Return True when Authorization is Bearer <shared secret>."""
    expected = secret if secret is not None else trigger_secret()
    if not expected or header is None:
        return False
    prefix = "Bearer "
    if not header.startswith(prefix):
        return False
    return hmac.compare_digest(header[len(prefix) :], expected)


def cookie_matches(value: str | None, password: str | None = None) -> bool:
    """Return True when the dashboard cookie matches the password token."""
    if not value:
        return False
    return hmac.compare_digest(value, session_token(password))
