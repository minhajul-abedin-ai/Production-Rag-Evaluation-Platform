"""Tenant credential verification. File-backed keys support atomic rotation without code changes.

This single-process demo is not a replacement for production IAM or an API gateway.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import secrets
import stat

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)
_TENANT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _tenant_keys() -> dict[str, str]:
    filename = os.getenv("RAG_API_KEYS_FILE", "")
    if filename:
        try:
            path = Path(filename)
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Invalid secret file")
            if os.name == "posix" and info.st_mode & 0o077:
                raise ValueError("Secret file is not owner-only")
            if info.st_size > 32_768:
                raise ValueError("Secret file too large")
            raw = path.read_text(encoding="utf-8")
        except (OSError, ValueError, UnicodeError) as exc:
            raise HTTPException(status_code=503, detail="API authentication configuration is invalid") from exc
    else:
        raw = os.getenv("RAG_API_KEYS_JSON", "")
    if not raw:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    try:
        mapping = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=503, detail="API authentication configuration is invalid") from exc
    if (not isinstance(mapping, dict) or not 1 <= len(mapping) <= 64
            or any(not isinstance(tenant, str) or not _TENANT_ID.fullmatch(tenant)
                   or not isinstance(key, str) or len(key) < 32 or len(key) > 512
                   for tenant, key in mapping.items())
            or len(set(mapping.values())) != len(mapping)):
        raise HTTPException(status_code=503, detail="API authentication configuration is invalid")
    return mapping


def require_tenant(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    mapping = _tenant_keys()
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401, detail="Valid bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    matches = [tenant for tenant, key in mapping.items()
               if secrets.compare_digest(credentials.credentials, key)]
    if len(matches) != 1:
        raise HTTPException(
            status_code=401, detail="Valid bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.app.state.limiter.check(matches[0])
    return matches[0]
