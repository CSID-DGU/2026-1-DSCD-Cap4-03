from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.images import ImageCreateRequest, ImageCreateResponse
from app.services.deps import get_current_user
from app.services.db_user import create_user_image, serialize_user_image


router = APIRouter()


@router.post("/images", response_model=ImageCreateResponse, status_code=201)
def create_image(
    payload: ImageCreateRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ImageCreateResponse:
    image = create_user_image(db, current_user["user_id"], payload)
    return ImageCreateResponse(**serialize_user_image(image))
