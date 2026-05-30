from __future__ import annotations

import sys
import tempfile
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from fastapi import HTTPException, status

from app.core.config import settings
from app.services.files import _build_s3_client


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT_PATH = PROJECT_ROOT / "model" / "skin_analysis" / "best_260507_21.pt"
MODEL_VERSION = "skin-model-260507-21"


def _ensure_project_root() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


@lru_cache(maxsize=1)
def _get_analyzer():
    _ensure_project_root()
    ckpt_path = Path(settings.skin_model_checkpoint or DEFAULT_CKPT_PATH)
    if not ckpt_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Skin model checkpoint not found: {ckpt_path}",
        )
    try:
        from model.skin_analysis.inference import SkinAnalyzer
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Skin model dependency is missing: {exc.name}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Skin model import failed: {exc}",
        ) from exc

    try:
        return SkinAnalyzer(
            ckpt_path=str(ckpt_path),
            device=settings.skin_model_device,
            img_size=settings.skin_model_img_size,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Skin model load failed: {exc}",
        ) from exc


def _display_score(value: Any) -> float:
    score = 0.0 if value is None else float(value)
    score = score / 100.0 if score > 1.0 else score
    return round(max(0.0, min(score, 1.0)), 4)


def _s3_key_from_url(image_source: str) -> str | None:
    parsed = urlparse(image_source)
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
    if host in {f"s3.amazonaws.com", f"s3.{settings.s3_region}.amazonaws.com"} and len(path_parts) == 2:
        bucket, key = path_parts
        if bucket == settings.s3_bucket:
            return unquote(key)

    return None


def _download_s3_image_to_temp(image_source: str) -> str | None:
    key = _s3_key_from_url(image_source)
    if not key:
        return None

    suffix = Path(key).suffix or ".jpg"
    try:
        response = _build_s3_client().get_object(Bucket=settings.s3_bucket, Key=key)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(response["Body"].read())
            return tmp.name
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load S3 image for skin analysis: {exc}",
        ) from exc


def analyze_skin_image(image_source: str) -> dict[str, Any]:
    local_source = _download_s3_image_to_temp(image_source)
    predict_source = local_source or image_source
    try:
        result = _get_analyzer().predict(predict_source)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Skin model inference failed: {exc}",
        ) from exc
    finally:
        if local_source:
            try:
                Path(local_source).unlink(missing_ok=True)
            except OSError:
                pass

    metrics = ["acne", "dryness", "sagging", "pore", "pigmentation", "wrinkle"]
    display_scores = {
        metric: _display_score(result.get(f"{metric}_score"))
        for metric in metrics
    }
    raw_metrics = {
        metric: int(result.get(f"{metric}_grade", 0) or 0)
        for metric in metrics
    }
    return {
        "display_scores": display_scores,
        "raw_metrics": raw_metrics,
        "model_raw": result,
        "analyzed_at": datetime.now(UTC).isoformat(),
        "model_version": MODEL_VERSION,
        "analysis_status": "SUCCESS",
    }
