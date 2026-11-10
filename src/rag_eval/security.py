"""Small explicit tenant API-key verifier for the single-process RAG demo.

Keys are supplied out of band; never store their values in Git.
This is not a replacement for a production identity provider.
"""
from __future__ import annotations

import json
import os
import re
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)
_TENANT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _tenant_keys() -> dict[str, str]:
    raw = os.getenv("RAG_API_KEYS_JSON", "")
    if not raw:
        raise HTTPException(status_code=503, detail="API authentication is not configured")
    try:
        mapping = json.loads(raw)
    except (ValueError, TypeError):
        raise HTTPException(status_code=503, detail="API authentication configuration is invalid")
    if (not isinstance(mapping, dict) or not mapping
            or any(not isinstance(tenant, str) or not _TENANT_ID.fullmatch(tenant)
                   or not isinstance(key, str) or len(key) < 32
                   for tenant, key in mapping.items())
            or len(set(mapping.values())) != len(mapping)):
        raise HTTPException(status_code=503, detail="API authentication configuration is invalid")
    return mapping


def require_tenant(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    mapping = _tenant_keys()  # Fail closed even when the caller has a bearer token.
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401, detail="Valid bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Iterate through all entries to avoid disclosing which key is used in timing.
    matches = [tenant for tenant, key in mapping.items()
               if secrets.compare_digest(credentials.credentials, key)]
    if len(matches) != 1:
        raise HTTPException(
            status_code=401, detail="Valid bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return matches[0]
