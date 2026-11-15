"""
auth_tools.py - Human Authorization Backend for Geotechnical Sign-Off.

Replaces the former "role is a string" check with a verifiable credential:

1. An engineer proves possession of the shared pit-authorization passphrase
   (stored only as a SHA-256 digest in ROCKWATCH_GEOTECH_PASSPHRASE_SHA256).
2. On success the system mints a short-lived HMAC-signed credential binding
   operator_id + role + issued_at + expires_at.
3. Desensitization approval requires a valid, unexpired credential whose
   embedded role is authorized. Missing configuration fails CLOSED: no
   credential can be minted or verified, so no attenuation can be approved.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any, Dict, Optional

AUTHORIZED_ROLES = frozenset({"GEOTECH_ENGINEER", "PRINCIPAL_GEOTECH_ENGINEER"})
DEFAULT_CREDENTIAL_TTL_SECONDS = 300


def _secret() -> Optional[bytes]:
    secret = os.getenv("ROCKWATCH_AUTH_SECRET", "").strip()
    return secret.encode("utf-8") if secret else None


def _passphrase_digest() -> Optional[str]:
    digest = os.getenv("ROCKWATCH_GEOTECH_PASSPHRASE_SHA256", "").strip().lower()
    return digest or None


def authorization_configured() -> bool:
    """True when both the signing secret and the passphrase digest are present."""
    return _secret() is not None and _passphrase_digest() is not None


def hash_passphrase(passphrase: str) -> str:
    """Helper for provisioning ROCKWATCH_GEOTECH_PASSPHRASE_SHA256."""
    return hashlib.sha256(passphrase.encode("utf-8")).hexdigest()


def verify_passphrase(passphrase: str) -> bool:
    """Constant-time comparison of a supplied passphrase against the configured digest."""
    expected = _passphrase_digest()
    if not expected or not passphrase:
        return False
    return hmac.compare_digest(hashlib.sha256(passphrase.encode("utf-8")).hexdigest(), expected)


def mint_credential(
    operator_id: str,
    role: str,
    ttl_seconds: int = DEFAULT_CREDENTIAL_TTL_SECONDS,
    issued_at: Optional[float] = None,
) -> str:
    """
    Mints a signed credential. Raises PermissionError when the auth backend is
    not configured or the role is not authorized to sign safety decisions.
    """
    secret = _secret()
    if secret is None:
        raise PermissionError(
            "Authorization backend is not configured (ROCKWATCH_AUTH_SECRET missing); "
            "safety decisions cannot be signed."
        )
    if role not in AUTHORIZED_ROLES:
        raise PermissionError(f"Role {role!r} is not authorized to sign safety decisions.")
    if not operator_id:
        raise ValueError("operator_id is required to mint a credential.")

    now = int(issued_at if issued_at is not None else time.time())
    expires = now + max(1, int(ttl_seconds))
    payload = f"{operator_id}|{role}|{now}|{expires}"
    signature = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}|{signature}"


def verify_credential(
    token: str,
    now: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Returns the decoded credential when the signature, expiry, and role are all
    valid; otherwise None. Never raises on malformed input.
    """
    secret = _secret()
    if secret is None or not token:
        return None

    parts = token.split("|")
    if len(parts) != 5:
        return None
    operator_id, role, issued_raw, expires_raw, signature = parts

    payload = f"{operator_id}|{role}|{issued_raw}|{expires_raw}"
    expected = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        return None

    try:
        issued_at = int(issued_raw)
        expires_at = int(expires_raw)
    except ValueError:
        return None

    current = now if now is not None else time.time()
    if current >= expires_at:
        return None
    if role not in AUTHORIZED_ROLES:
        return None

    return {
        "operator_id": operator_id,
        "role": role,
        "issued_at": issued_at,
        "expires_at": expires_at,
    }


def require_geotech_credential(token: str, now: Optional[float] = None) -> Dict[str, Any]:
    """Fail-closed gate used by the governance layer."""
    credential = verify_credential(token, now=now)
    if credential is None:
        raise PermissionError(
            "Sensitivity attenuation requires a valid, unexpired geotechnical credential. "
            "The supplied credential is missing, malformed, expired, or not authorized."
        )
    return credential
