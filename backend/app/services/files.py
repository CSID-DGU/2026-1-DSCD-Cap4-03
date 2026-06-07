from __future__ import annotations

import boto3
from botocore.config import Config
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

from fastapi import HTTPException, status

from app.core.config import settings


def _build_s3_client():
    session_kwargs: dict[str, str] = {"region_name": settings.s3_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        session_kwargs["aws_access_key_id"] = settings.aws_access_key_id
        session_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    if settings.aws_session_token:
        session_kwargs["aws_session_token"] = settings.aws_session_token
    session = boto3.session.Session(**session_kwargs)
    return session.client(
        "s3",
        endpoint_url=f"https://s3.{settings.s3_region}.amazonaws.com",
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


def build_presigned_payload(user_id: int, file_name: str, mime_type: str) -> dict:
    ext = file_name.split(".")[-1].lower() if "." in file_name else "jpg"
    object_key = f"{settings.s3_prefix}/{user_id}/images/{uuid4().hex}.{ext}"
    client = _build_s3_client()

    try:
        upload_url = client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": settings.s3_bucket,
                "Key": object_key,
                "ContentType": mime_type,
            },
            ExpiresIn=settings.presign_expire_seconds,
            HttpMethod="PUT",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create S3 presigned URL: {exc}",
        ) from exc

    return {
        "upload_url": upload_url,
        "public_url": f"{settings.resolved_s3_public_base_url}/{object_key}",
        "s3_key": object_key,
        "expires_in": settings.presign_expire_seconds,
        "issued_at": datetime.now(UTC).isoformat(),
    }


def _s3_key_from_storage_url(storage_url: str | None) -> str | None:
    if not storage_url:
        return None

    parsed = urlparse(storage_url)
    if parsed.scheme not in {"http", "https"}:
        return None

    host = parsed.netloc.split(":", 1)[0]
    bucket_hosts = {
        f"{settings.s3_bucket}.s3.amazonaws.com",
        f"{settings.s3_bucket}.s3.{settings.s3_region}.amazonaws.com",
    }
    if host in bucket_hosts:
        return unquote(parsed.path.lstrip("/"))

    path_parts = parsed.path.lstrip("/").split("/", 1)
    region_hosts = {"s3.amazonaws.com", f"s3.{settings.s3_region}.amazonaws.com"}
    if host in region_hosts and len(path_parts) == 2:
        bucket, key = path_parts
        if bucket == settings.s3_bucket:
            return unquote(key)

    return None


def build_presigned_get_url(s3_key: str) -> str:
    try:
        return _build_s3_client().generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": settings.s3_bucket,
                "Key": s3_key,
            },
            ExpiresIn=settings.presign_expire_seconds,
            HttpMethod="GET",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create S3 download URL: {exc}",
        ) from exc


def resolve_image_display_url(storage_url: str | None, s3_key: str | None = None) -> str | None:
    key = s3_key or _s3_key_from_storage_url(storage_url)
    if key and not key.startswith("local-upload/"):
        return build_presigned_get_url(key)

    if storage_url and Path(storage_url).exists():
        return storage_url

    return storage_url
