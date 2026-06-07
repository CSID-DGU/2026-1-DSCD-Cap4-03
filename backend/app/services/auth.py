from __future__ import annotations

import hashlib

from fastapi import HTTPException, status

from app.db.memory import store


def hash_password(raw_password: str) -> str:
    return hashlib.sha256(raw_password.encode("utf-8")).hexdigest()


def build_token(user_id: int) -> str:
    return f"mock-token-{user_id}"


def parse_token(token: str) -> int:
    prefix = "mock-token-"
    if not token.startswith(prefix):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return int(token.replace(prefix, "", 1))
