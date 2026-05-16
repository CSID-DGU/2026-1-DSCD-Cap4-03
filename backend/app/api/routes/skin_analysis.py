from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.memory import store
from app.db.session import get_db
from app.schemas.skin_analysis import (
    SkinAnalysisCreateRequest,
    SkinAnalysisCreateResponse,
    SkinAnalysisDetailResponse,
    SkinSummaryRequest,
    SkinSummaryResponse,
)
from app.services.deps import get_current_user
from app.services.db_user import ensure_profile, get_user_image
from app.services.mock_analysis import build_skin_scores, build_skin_summary


router = APIRouter()


@router.post("/skin-analysis", response_model=SkinAnalysisCreateResponse, status_code=201)
def create_skin_analysis(
    payload: SkinAnalysisCreateRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SkinAnalysisCreateResponse:
    image = get_user_image(db, payload.image_id)
    if not image or image.user_id != current_user["user_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    result_id = next(store.result_seq)
    scores = build_skin_scores(payload.image_id, current_user["user_id"])
    row = {
        "result_id": result_id,
        "image_id": payload.image_id,
        "user_id": current_user["user_id"],
        **scores,
    }
    store.skin_results[result_id] = row
    return SkinAnalysisCreateResponse(
        result_id=result_id,
        image_id=payload.image_id,
        user_id=current_user["user_id"],
        analyzed_at=row["analyzed_at"],
        model_version=row["model_version"],
        analysis_status=row["analysis_status"],
    )


@router.get("/skin-analysis/{result_id}", response_model=SkinAnalysisDetailResponse)
def get_skin_analysis(
    result_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SkinAnalysisDetailResponse:
    result = store.skin_results.get(result_id)
    if not result or result["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis result not found")
    image = get_user_image(db, result["image_id"])
    summary = store.skin_summaries.get(result_id)
    profile = ensure_profile(db, current_user["user_id"])
    if not summary:
        summary = {"prompt_version": "skin_v1", **build_skin_summary(result)}
        store.skin_summaries[result_id] = summary
    return SkinAnalysisDetailResponse(
        result_id=result_id,
        user_id=result["user_id"],
        image_id=result["image_id"],
        model_name=result["model_version"],
        prompt_version=summary["prompt_version"],
        analyzed_at=result["analyzed_at"],
        generated_at=summary["generated_at"],
        summary_comment=summary["summary_comment"],
        indicator_comments=summary["indicator_comments"],
        image_url=image.storage_url if image else "",
        skin_type=profile.skin_type if profile else None,
        raw_metrics=result["raw_metrics"],
        display_scores=result["display_scores"],
    )


@router.post("/skin-analysis/summaries", response_model=SkinSummaryResponse, status_code=201)
def create_skin_summary(payload: SkinSummaryRequest, current_user: dict = Depends(get_current_user)) -> SkinSummaryResponse:
    result = store.skin_results.get(payload.result_id)
    if not result or result["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis result not found")

    summary = {"result_id": payload.result_id, **build_skin_summary(result)}
    store.skin_summaries[payload.result_id] = summary
    return SkinSummaryResponse(**summary)
