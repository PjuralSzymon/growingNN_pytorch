"""Unit tests for trigger bearer auth and dashboard cookies."""

from worker.auth import bearer_matches, cookie_matches, session_token


def test_bearer_matches_accepts_exact_shared_secret():
    """
    bearer_matches should accept Authorization Bearer with the shared secret.
    """

    # Arrange / Act / Assert
    assert bearer_matches("Bearer secret", secret="secret") is True


def test_bearer_matches_rejects_missing_or_wrong_secret():
    """
    bearer_matches should reject a missing header, a wrong token, or an empty secret.
    """

    # Arrange / Act / Assert
    assert bearer_matches(None, secret="secret") is False
    assert bearer_matches("Bearer other", secret="secret") is False
    assert bearer_matches("Bearer secret", secret="") is False
    assert bearer_matches("secret", secret="secret") is False


def test_cookie_matches_accepts_token_derived_from_password():
    """
    cookie_matches should accept the hmac session token for the dashboard password.
    """

    # Arrange
    token = session_token("pass")

    # Act / Assert
    assert cookie_matches(token, password="pass") is True
    assert cookie_matches("nope", password="pass") is False
    assert cookie_matches(None, password="pass") is False
