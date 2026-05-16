from fastapi import APIRouter, Depends

from app.schemas.files import PresignRequest, PresignResponse
from app.services.deps import get_current_user
from app.services.files import build_presigned_payload


router = APIRouter()


@router.post("/presign", response_model=PresignResponse)
def create_presigned_url(payload: PresignRequest, current_user: dict = Depends(get_current_user)) -> PresignResponse:
    data = build_presigned_payload(current_user["user_id"], payload.file_name, payload.mime_type)
    return PresignResponse(**data)
