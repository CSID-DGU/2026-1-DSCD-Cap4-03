from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import Select, desc, select, text
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.memory import store
from app.models import RecommendationItem, RecommendationRoutine, RecommendationSession
from app.schemas.recommendations import RecommendationRequest
from app.services.db_catalog import _usage_for_category, serialize_product_list_item


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _prepare_model_imports() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    os.environ["ROUPLE_MYSQL_HOST"] = settings.mysql_host
    os.environ["ROUPLE_MYSQL_PORT"] = str(settings.mysql_port)
    os.environ["ROUPLE_MYSQL_USER"] = settings.mysql_user
    os.environ["ROUPLE_MYSQL_PASSWORD"] = settings.mysql_password
    os.environ["ROUPLE_MYSQL_DB"] = settings.mysql_db
    os.environ.setdefault("EMBED_LOCAL_FILES_ONLY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _candidate_count(db: Session, image_id: int) -> int:
    return int(
        db.scalar(
            text("SELECT COUNT(*) FROM recommendation_candidate WHERE image_id = :image_id"),
            {"image_id": image_id},
        )
        or 0
    )


def _load_skin_query_for_request(db: Session, payload: RecommendationRequest, user_id: int):
    import pandas as pd

    rows = db.execute(
        text(
            """
            SELECT
                ui.image_id,
                ui.user_id,
                COALESCE(LOWER(up.gender), 'female') AS gender,
                ui.storage_url,
                sar.dryness_score,
                sar.pore_score,
                sar.wrinkle_score,
                sar.pigmentation_score,
                sar.sagging_score,
                sar.acne_score
            FROM user_image ui
            LEFT JOIN user_profile up ON up.user_id = ui.user_id
            JOIN skin_analysis_result sar ON sar.image_id = ui.image_id
            WHERE ui.user_id = :user_id
              AND ui.image_id = :image_id
              AND sar.result_id = :result_id
            LIMIT 1
            """
        ),
        {"user_id": user_id, "image_id": payload.image_id, "result_id": payload.result_id},
    ).mappings().all()

    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skin analysis result not found")
    return pd.DataFrame(rows)


def ensure_embedding_candidates(db: Session, payload: RecommendationRequest, user_id: int) -> None:
    if _candidate_count(db, payload.image_id) > 0:
        return

    _prepare_model_imports()
    try:
        from model.recommendation.embedding_pipeline import config as embedding_config
        from model.recommendation.embedding_pipeline.data_loader import load_corpus_from_db
        from model.recommendation.embedding_pipeline.db_uploader import upload_recommendation_candidates
        from model.recommendation.embedding_pipeline.retriever import run_retrieval
        from model.recommendation.embedding_pipeline.run_embedding import build_load_output
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"임베딩 모델 의존성 누락: {exc.name}. requirements.txt 설치 필요",
        ) from exc

    embedding_config.MYSQL_HOST = settings.mysql_host
    embedding_config.MYSQL_PORT = settings.mysql_port
    embedding_config.MYSQL_USER = settings.mysql_user
    embedding_config.MYSQL_PASSWORD = settings.mysql_password
    embedding_config.MYSQL_DB = settings.mysql_db

    emb_path = embedding_config.DEFAULT_OUTPUT_DIR / f"cosmetic_emb_{embedding_config.MODEL_NAME.replace('/', '_')}.npy"
    embedding_config.DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        corpus_df = load_corpus_from_db()
        skin_df = _load_skin_query_for_request(db, payload, user_id)
        result_df = run_retrieval(
            corpus_df=corpus_df,
            skin_df=skin_df,
            topk_per_category=embedding_config.TOPK_PER_CATEGORY,
            model_name=embedding_config.MODEL_NAME,
            emb_path=emb_path,
        )
        if result_df.empty:
            raise ValueError("No embedding candidates generated.")
        upload_recommendation_candidates(build_load_output(result_df))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"임베딩 후보 생성 실패: {exc}",
        ) from exc


def _run_recommendation_pipeline(payload: RecommendationRequest, user_id: int) -> None:
    _prepare_model_imports()
    try:
        from model.recommendation.kg_pipeline.neo4j_skincare import config as kg_config
        from model.recommendation.kg_pipeline.neo4j_skincare.pipeline import run_pipeline
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"추천 모델 의존성 누락: {exc.name}. requirements.txt 설치 필요",
        ) from exc

    # config.py가 이미 import된 경우를 대비해 런타임 설정도 동기화함.
    kg_config.MYSQL_HOST = settings.mysql_host
    kg_config.MYSQL_PORT = settings.mysql_port
    kg_config.MYSQL_USER = settings.mysql_user
    kg_config.MYSQL_PASSWORD = settings.mysql_password
    kg_config.MYSQL_DB = settings.mysql_db
    kg_config.NEO4J_URI = settings.neo4j_uri
    kg_config.NEO4J_USER = settings.neo4j_user
    kg_config.NEO4J_PASS = settings.neo4j_password

    slot_budget_min_map = {
        key: value
        for key, value in {
            "Toner": payload.toner_budget_min,
            "Emulsions": payload.emulsion_budget_min,
            "Essences/Ampoules/Serums": payload.ampoule_budget_min,
            "Cream/Gel": payload.cream_budget_min,
        }.items()
        if value is not None
    }
    slot_budget_max_map = {
        key: value
        for key, value in {
            "Toner": payload.toner_budget_max,
            "Emulsions": payload.emulsion_budget_max,
            "Essences/Ampoules/Serums": payload.ampoule_budget_max,
            "Cream/Gel": payload.cream_budget_max,
        }.items()
        if value is not None
    }

    try:
        run_pipeline(
            user_id=user_id,
            image_id=payload.image_id,
            top_n=3,
            total_budget_min=payload.total_budget_min,
            total_budget_max=payload.total_budget_max,
            slot_budget_min_map=slot_budget_min_map or None,
            slot_budget_max_map=slot_budget_max_map or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"추천 모델 실행 실패: {exc}",
        ) from exc


def ensure_skin_result_for_recommendation(db: Session, payload: RecommendationRequest, user_id: int) -> None:
    existing = db.execute(
        text(
            """
            SELECT user_id, image_id
            FROM skin_analysis_result
            WHERE result_id = :result_id
            LIMIT 1
            """
        ),
        {"result_id": payload.result_id},
    ).mappings().first()
    if existing:
        if existing["user_id"] == user_id and existing["image_id"] == payload.image_id:
            return
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Skin analysis result_id already exists for another image or user. Run /skin-analysis again.",
        )

    memory_result = store.skin_results.get(payload.result_id)
    if not memory_result or memory_result["user_id"] != user_id or memory_result["image_id"] != payload.image_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skin analysis result not found")

    display_scores = memory_result.get("display_scores", {})
    db.execute(
        text(
            """
            INSERT INTO skin_analysis_result (
                result_id, image_id, user_id,
                acne_score, dryness_score, sagging_score, pore_score, pigmentation_score, wrinkle_score
            ) VALUES (
                :result_id, :image_id, :user_id,
                :acne_score, :dryness_score, :sagging_score, :pore_score, :pigmentation_score, :wrinkle_score
            )
            """
        ),
        {
            "result_id": payload.result_id,
            "image_id": payload.image_id,
            "user_id": user_id,
            "acne_score": display_scores.get("acne"),
            "dryness_score": display_scores.get("dryness"),
            "sagging_score": display_scores.get("sagging"),
            "pore_score": display_scores.get("pore"),
            "pigmentation_score": display_scores.get("pigmentation"),
            "wrinkle_score": display_scores.get("wrinkle"),
        },
    )
    db.commit()


def create_recommendation_with_model(db: Session, payload: RecommendationRequest, user_id: int) -> RecommendationSession:
    ensure_skin_result_for_recommendation(db, payload, user_id)
    ensure_embedding_candidates(db, payload, user_id)
    _run_recommendation_pipeline(payload, user_id)
    db.commit()
    db.expire_all()
    session = get_latest_recommendation_session(db, user_id, payload.result_id, payload.image_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Recommendation session was not created.")
    if not session.routines:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation routines were not created. Check recommendation pipeline logs.",
        )
    return session


def _session_query() -> Select[tuple[RecommendationSession]]:
    return select(RecommendationSession).options(
        selectinload(RecommendationSession.routines)
        .selectinload(RecommendationRoutine.items)
        .selectinload(RecommendationItem.product)
    )


def get_latest_recommendation_session(
    db: Session,
    user_id: int,
    result_id: int,
    image_id: int,
) -> RecommendationSession | None:
    return db.scalar(
        _session_query()
        .where(
            RecommendationSession.user_id == user_id,
            RecommendationSession.result_id == result_id,
            RecommendationSession.image_id == image_id,
        )
        .order_by(desc(RecommendationSession.session_id))
        .limit(1)
    )


def get_recommendation_session_or_404(db: Session, session_id: int, user_id: int) -> RecommendationSession:
    session = db.scalar(
        _session_query().where(
            RecommendationSession.session_id == session_id,
            RecommendationSession.user_id == user_id,
        )
    )
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recommendation session not found")
    return session


def _routine_type(routine: RecommendationRoutine) -> str:
    label = (routine.routine_label or "").lower()
    if "value" in label:
        return "value"
    if routine.routine_rank == 2:
        return "value"
    return "best"


def _routine_label(routine: RecommendationRoutine) -> str:
    if _routine_type(routine) == "value":
        return "가성비 루틴"
    return "AI BEST 루틴"


def _routine_time(routine: RecommendationRoutine) -> str:
    mode = (routine.ampm_mode or "").lower()
    if mode in {"am", "pm", "both"}:
        return mode
    if "pm" in mode and "am" not in mode:
        return "pm"
    if "am" in mode and "pm" not in mode:
        return "am"
    return "both"


CORE_ROUTINE_CATEGORIES = {
    "toner",
    "toner pads",
    "emulsions",
    "essences/ampoules/serums",
    "cream/gel",
    "face moisturizers",
}


def _is_core_routine_category(category: object) -> bool:
    normalized = " ".join(str(category or "").strip().lower().split())
    return normalized in CORE_ROUTINE_CATEGORIES


def _routine_description(routine: RecommendationRoutine, total_cost: int, product_count: int) -> str:
    label = _routine_label(routine)
    routine_time = _routine_time(routine).upper()
    return f"{label}은 {routine_time} 기준 {product_count}단계 루틴으로 구성되었고, 총 예상 금액은 {total_cost:,}원입니다."


def serialize_recommendation_session(db: Session, session: RecommendationSession) -> dict[str, Any]:
    routines = []
    sorted_routines = sorted(session.routines, key=lambda item: item.routine_rank)
    for routine in sorted_routines:
        products = []
        total_cost = 0
        sorted_items = sorted(routine.items, key=lambda item: item.slot_order)
        for item in sorted_items:
            if not item.product:
                continue
            product_data = serialize_product_list_item(db, item.product)
            if _is_core_routine_category(item.category or item.product.category):
                total_cost += product_data["price"]
            products.append(
                {
                    "product_id": item.product.product_id,
                    "step": item.slot_order,
                    "time_tag": item.time_tag,
                }
            )

        routines.append(
            {
                "routine_id": str(routine.routine_id),
                "type": _routine_type(routine),
                "label": _routine_label(routine),
                "routine_time": _routine_time(routine),
                "total_cost": total_cost,
                "duration": len(products),
                "products": products,
            }
        )

    budget_fallback_applied = session.session_status == "SUCCESS" and not bool(session.budget_check_passed)
    budget_message = (
        "예산 조건에 맞는 추천이 없어, 예산 제한 없이 가장 유사한 루틴을 제공합니다."
        if budget_fallback_applied
        else None
    )

    slot_budget_min = {}
    slot_budget_max = {}
    if session.slot_budget_min_json:
        try:
            slot_budget_min = json.loads(session.slot_budget_min_json)
        except json.JSONDecodeError:
            slot_budget_min = {}
    if session.slot_budget_max_json:
        try:
            slot_budget_max = json.loads(session.slot_budget_max_json)
        except json.JSONDecodeError:
            slot_budget_max = {}

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "result_id": session.result_id or 0,
        "session_status": session.session_status,
        "budget_check_passed": bool(session.budget_check_passed),
        "budget_fallback_applied": budget_fallback_applied,
        "budget_message": budget_message,
        "total_budget_min": session.total_budget_min,
        "total_budget_max": session.total_budget_max,
        "toner_budget_min": slot_budget_min.get("Toner"),
        "toner_budget_max": slot_budget_max.get("Toner"),
        "emulsion_budget_min": slot_budget_min.get("Emulsions"),
        "emulsion_budget_max": slot_budget_max.get("Emulsions"),
        "ampoule_budget_min": slot_budget_min.get("Essences/Ampoules/Serums"),
        "ampoule_budget_max": slot_budget_max.get("Essences/Ampoules/Serums"),
        "cream_budget_min": slot_budget_min.get("Cream/Gel"),
        "cream_budget_max": slot_budget_max.get("Cream/Gel"),
        "routines": routines,
    }
