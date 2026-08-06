"""
utils/signed_link.py
---------------------
Short-lived HMAC-signed tokens for the document preview file link.

Why this exists: every other endpoint in this app is authorized via
X-Tenant-ID/X-Org-Unit-ID headers. A plain clickable link or an <iframe>
src can't carry custom headers — browsers don't attach them to a link
click or an inline embed. So the one endpoint that needs to work as a
plain URL (GET /documents/{id}/file) is authorized differently: the
tenant_id, org_unit_id, and doc_id it's allowed to serve are baked into
a signed token in the URL itself, with a short expiry. Same idea as an
S3 presigned URL.

Deliberately stdlib-only (hmac/hashlib/base64/json/time) rather than
pulling in a new dependency (e.g. itsdangerous) for something this small
— see the requirements.txt cleanup earlier in this project for why that
restraint matters here specifically.
"""

import base64
import hashlib
import hmac
import json
import time
from typing import Optional

from config.settings import get_settings

settings = get_settings()


def _secret_key() -> bytes:
    return settings.SECRET_KEY.encode("utf-8")


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def generate_file_token(
    doc_id: str,
    tenant_id: str,
    org_unit_id: str,
    ttl_seconds: Optional[int] = None,
) -> str:
    """
    Generate a signed token encoding which document, which tenant, which
    org unit this link is allowed to serve, and when it expires.
    Format: <base64-payload>.<base64-signature>
    """
    ttl = ttl_seconds if ttl_seconds is not None else settings.FILE_LINK_TTL_SECONDS
    payload = {
        "doc_id": doc_id,
        "tenant_id": tenant_id,
        "org_unit_id": org_unit_id,
        "exp": int(time.time()) + ttl,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_b64 = _b64encode(payload_bytes)

    sig = hmac.new(_secret_key(), payload_b64.encode("ascii"), hashlib.sha256).digest()
    sig_b64 = _b64encode(sig)

    return f"{payload_b64}.{sig_b64}"


def verify_file_token(token: str) -> Optional[dict]:
    """
    Verify a token's signature and expiry. Returns the decoded payload
    dict (doc_id/tenant_id/org_unit_id/exp) if valid, None if the token
    is malformed, tampered with, or expired. Never raises — callers
    should treat None as "reject with 403/404", not as an error to debug.
    """
    try:
        payload_b64, sig_b64 = token.split(".", 1)
    except ValueError:
        return None

    expected_sig = hmac.new(_secret_key(), payload_b64.encode("ascii"), hashlib.sha256).digest()
    expected_sig_b64 = _b64encode(expected_sig)

    # Constant-time comparison — avoids leaking signature-match info via timing
    if not hmac.compare_digest(sig_b64, expected_sig_b64):
        return None

    try:
        payload = json.loads(_b64decode(payload_b64))
    except Exception:
        return None

    if not isinstance(payload, dict) or payload.get("exp", 0) < time.time():
        return None

    return payload