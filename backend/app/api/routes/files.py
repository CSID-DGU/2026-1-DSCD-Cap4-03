from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.files import PresignRequest, PresignResponse
from app.schemas.images import ImageCreateResponse
from app.services.deps import get_current_user
from app.services.db_user import serialize_user_image
from app.services.files import build_presigned_payload
from app.models import UserImage


router = APIRouter()

BACKEND_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BACKEND_ROOT / "uploads"


@router.post("/presign", response_model=PresignResponse)
def create_presigned_url(payload: PresignRequest, current_user: dict = Depends(get_current_user)) -> PresignResponse:
    data = build_presigned_payload(current_user["user_id"], payload.file_name, payload.mime_type)
    return PresignResponse(**data)


@router.post("/local-upload", response_model=ImageCreateResponse, status_code=status.HTTP_201_CREATED)
async def upload_local_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ImageCreateResponse:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    original_name = file.filename or "image.jpg"
    suffix = Path(original_name).suffix or ".jpg"
    safe_name = f"user_{current_user['user_id']}_{uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / safe_name

    content = await file.read()
    saved_path.write_bytes(content)

    image = UserImage(
        user_id=current_user["user_id"],
        storage_url=str(saved_path),
        s3_key=f"local-upload/{safe_name}",
        original_file_name=original_name,
        mime_type=file.content_type,
        file_size=len(content),
        crop_data=None,
        upload_status="UPLOADED",
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return ImageCreateResponse(**serialize_user_image(image))
