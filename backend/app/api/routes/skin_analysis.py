from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.memory import save_skin_summaries, store
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
from app.services.llm_service import generate_skin_summary_llm
from app.services.skin_model_service import analyze_skin_image


router = APIRouter()
SKIN_INDICATORS = ("acne", "dryness", "sagging", "pore", "pigmentation", "wrinkle")


def _empty_skin_summary(result_id: int) -> dict:
    return {
        "result_id": result_id,
        "llm_model": "",
        "prompt_version": "",
        "summary_comment": "아직 분석 요약이 생성되지 않았습니다.",
        "indicator_comments": {
            key: "아직 분석 요약이 생성되지 않았습니다."
            for key in SKIN_INDICATORS
        },
        "generated_at": "",
    }


def _next_result_id(db: Session) -> int:
    max_db_id = db.scalar(text("SELECT COALESCE(MAX(result_id), 0) FROM skin_analysis_result")) or 0
    max_memory_id = max(store.skin_results.keys(), default=0)
    max_used_id = max(max_db_id, max_memory_id)
    result_id = next(store.result_seq)
    while result_id <= max_used_id:
        result_id = next(store.result_seq)
    return result_id


def _save_skin_analysis_result(db: Session, row: dict) -> None:
    display_scores = row["display_scores"]
    db.execute(
        text(
            """
            INSERT INTO skin_analysis_result (
                result_id, image_id, user_id,
                acne_score, dryness_score, sagging_score,
                pore_score, pigmentation_score, wrinkle_score
            ) VALUES (
                :result_id, :image_id, :user_id,
                :acne_score, :dryness_score, :sagging_score,
                :pore_score, :pigmentation_score, :wrinkle_score
            )
            """
        ),
        {
            "result_id": row["result_id"],
            "image_id": row["image_id"],
            "user_id": row["user_id"],
            "acne_score": display_scores.get("acne"),
            "dryness_score": display_scores.get("dryness"),
            "sagging_score": display_scores.get("sagging"),
            "pore_score": display_scores.get("pore"),
            "pigmentation_score": display_scores.get("pigmentation"),
            "wrinkle_score": display_scores.get("wrinkle"),
        },
    )
    db.commit()


def _load_skin_analysis_result(db: Session, result_id: int, user_id: int) -> dict | None:
    row = db.execute(
        text(
            """
            SELECT
                result_id, image_id, user_id,
                acne_score, dryness_score, sagging_score,
                pore_score, pigmentation_score, wrinkle_score,
                analyzed_at
            FROM skin_analysis_result
            WHERE result_id = :result_id
              AND user_id = :user_id
            LIMIT 1
            """
        ),
        {"result_id": result_id, "user_id": user_id},
    ).mappings().first()
    if row is None:
        return None

    display_scores = {
        "acne": float(row["acne_score"] or 0),
        "dryness": float(row["dryness_score"] or 0),
        "sagging": float(row["sagging_score"] or 0),
        "pore": float(row["pore_score"] or 0),
        "pigmentation": float(row["pigmentation_score"] or 0),
        "wrinkle": float(row["wrinkle_score"] or 0),
    }
    analyzed_at = row["analyzed_at"].isoformat() if hasattr(row["analyzed_at"], "isoformat") else str(row["analyzed_at"])
    return {
        "result_id": int(row["result_id"]),
        "image_id": int(row["image_id"]),
        "user_id": int(row["user_id"]),
        "display_scores": display_scores,
        "raw_metrics": {key: 0 for key in display_scores},
        "model_version": "skin-model-260507-21",
        "analysis_status": "SUCCESS",
        "analyzed_at": analyzed_at,
    }


@router.post("/skin-analysis", response_model=SkinAnalysisCreateResponse, status_code=201)
def create_skin_analysis(
    payload: SkinAnalysisCreateRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SkinAnalysisCreateResponse:
    image = get_user_image(db, payload.image_id)
    if not image or image.user_id != current_user["user_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    result_id = _next_result_id(db)
    scores = analyze_skin_image(image.storage_url)
    row = {
        "result_id": result_id,
        "image_id": payload.image_id,
        "user_id": current_user["user_id"],
        **scores,
    }
    try:
        _save_skin_analysis_result(db, row)
    except Exception:
        db.rollback()
        raise
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
    if not result:
        result = _load_skin_analysis_result(db, result_id, current_user["user_id"])
        if result:
            store.skin_results[result_id] = result
    if not result or result["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis result not found")
    image = get_user_image(db, result["image_id"])
    summary = store.skin_summaries.get(result_id)
    profile = ensure_profile(db, current_user["user_id"])
    if summary:
        save_skin_summaries(store.skin_summaries)
    else:
        summary = _empty_skin_summary(result_id)
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
def create_skin_summary(
    payload: SkinSummaryRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SkinSummaryResponse:
    result = store.skin_results.get(payload.result_id)
    if not result:
        result = _load_skin_analysis_result(db, payload.result_id, current_user["user_id"])
        if result:
            store.skin_results[payload.result_id] = result
    if not result or result["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis result not found")

    summary = store.skin_summaries.get(payload.result_id)
    if summary:
        save_skin_summaries(store.skin_summaries)
        return SkinSummaryResponse(**summary)

    llm_result = generate_skin_summary_llm(result)
    summary = {
        "result_id": payload.result_id,
        "llm_model": llm_result["model_name"],
        "prompt_version": llm_result["prompt_version"],
        "summary_comment": llm_result["summary_comment"],
        "indicator_comments": llm_result["indicator_comments"],
        "generated_at": llm_result["generated_at"],
    }
    store.skin_summaries[payload.result_id] = summary
    save_skin_summaries(store.skin_summaries)
    return SkinSummaryResponse(**summary)
