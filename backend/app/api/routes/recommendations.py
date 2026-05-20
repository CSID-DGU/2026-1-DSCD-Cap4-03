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
from app.services.llm_service import generate_routine_explanation_llm
from app.services.recommendation_model import (
    create_recommendation_with_model,
    get_recommendation_session_or_404,
    serialize_recommendation_session,
)


router = APIRouter()


def _client_routine_type(value: object) -> str:
    routine_type = str(value or "").strip().lower()
    if routine_type in {"value", "budget"}:
        return "value"
    return "best"


def _session_routine_type(routine: object) -> str:
    label = str(getattr(routine, "routine_label", "") or "").lower()
    rank = int(getattr(routine, "routine_rank", 0) or 0)
    if "value" in label or "budget" in label or rank == 2:
        return "value"
    return "best"


def _routine_metadata(session: object) -> dict[str, dict]:
    metadata = {}
    for routine in sorted(getattr(session, "routines", []), key=lambda row: row.routine_rank):
        routine_type = _session_routine_type(routine)
        metadata[routine_type] = {
            "routine_id": routine.routine_id,
            "routine_rank": routine.routine_rank,
            "ampm_mode": routine.ampm_mode or "",
            "categories": {
                item.slot_order: item.category or (item.product.category if item.product else "")
                for item in sorted(routine.items, key=lambda row: row.slot_order)
            },
        }
    return metadata


def _step_guides_with_categories(step_guides: object, categories: dict[int, str]) -> list[dict]:
    if not isinstance(step_guides, list):
        return []

    normalized = []
    for step in step_guides:
        if not isinstance(step, dict):
            continue
        slot_order = int(step.get("slot_order") or 0)
        normalized.append(
            {
                "slot_order": slot_order,
                "category": step.get("category") or categories.get(slot_order, ""),
                "usage_guide": step.get("usage_guide", ""),
            }
        )
    return normalized


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


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

    llm_result = generate_routine_explanation_llm(session)
    metadata = _routine_metadata(session)
    routines = [
        routine
        for routine in llm_result.get("routines", [])
        if isinstance(routine, dict)
    ]
    routines = [
        {
            **routine,
            "routine_type": _client_routine_type(routine.get("routine_type")),
        }
        for routine in routines
    ]

    response_routines = [
        {
            "routine_id": metadata.get(routine.get("routine_type"), {}).get("routine_id", routine.get("routine_id")),
            "routine_type": routine.get("routine_type", ""),
            "routine_rank": metadata.get(routine.get("routine_type"), {}).get("routine_rank", routine.get("routine_rank", 0)),
            "ampm_mode": metadata.get(routine.get("routine_type"), {}).get("ampm_mode", routine.get("ampm_mode", "")),
            "recommend_summary": routine.get("recommend_summary", ""),
            "ampm_comment": routine.get("ampm_comment", ""),
            "step_guides": _step_guides_with_categories(
                routine.get("step_guides", []),
                metadata.get(routine.get("routine_type"), {}).get("categories", {}),
            ),
            "strengths": _string_list(routine.get("strengths", [])),
            "cautions": _string_list(routine.get("cautions", [])),
        }
        for routine in routines
    ]

    explanation = {
        "session_id": payload.session_id,
        "llm_model": llm_result["model_name"],
        "prompt_version": llm_result["prompt_version"],
        "routines": response_routines,
    }
    store.recommendation_explanations[payload.session_id] = explanation
    return RecommendationExplanationResponse(**explanation)
