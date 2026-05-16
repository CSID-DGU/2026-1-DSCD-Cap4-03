from __future__ import annotations

import boto3
from datetime import UTC, datetime
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
    return session.client("s3")


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
