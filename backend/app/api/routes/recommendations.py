from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.memory import store
from app.db.session import get_db
from app.schemas.recommendations import (
    RecommendationExplanationRequest,
    RecommendationExplanationResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from app.services.deps import get_current_user
from app.services.db_user import get_user_image
from app.services.mock_recommendation import build_recommendation_explanation
from app.services.recommendation_model import (
    create_recommendation_with_model,
    get_recommendation_session_or_404,
    serialize_recommendation_session,
)


router = APIRouter()


@router.post("/recommendations", response_model=RecommendationResponse, status_code=201)
def create_recommendation(
    payload: RecommendationRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RecommendationResponse:
    image = get_user_image(db, payload.image_id)
    if not image or image.user_id != current_user["user_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    session = create_recommendation_with_model(db, payload, current_user["user_id"])
    return RecommendationResponse(**serialize_recommendation_session(db, session))


@router.get("/recommendations/{session_id}", response_model=RecommendationResponse)
def get_recommendation(
    session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RecommendationResponse:
    session = get_recommendation_session_or_404(db, session_id, current_user["user_id"])
    return RecommendationResponse(**serialize_recommendation_session(db, session))


@router.post("/recommendation-explanations", response_model=RecommendationExplanationResponse, status_code=201)
def create_recommendation_explanation(
    payload: RecommendationExplanationRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RecommendationExplanationResponse:
    session = get_recommendation_session_or_404(db, payload.session_id, current_user["user_id"])

    explanation = {"session_id": payload.session_id, **build_recommendation_explanation({"session_id": session.session_id})}
    store.recommendation_explanations[payload.session_id] = explanation
    return RecommendationExplanationResponse(**explanation)
