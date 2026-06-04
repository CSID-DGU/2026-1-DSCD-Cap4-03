from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import bindparam, text
from sqlalchemy.orm import object_session

from app.core.config import settings
from app.db.memory import store
from app.models import RecommendationRoutine, RecommendationSession


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.llm.routine_recommendation_llm import (
    MODEL_NAME as ROUTINE_LLM_MODEL_NAME,
    PROMPT_VERSION as ROUTINE_PROMPT_VERSION,
    generate_routine_llm_result,
)
from model.llm.skin_analysis_llm import (
    MODEL_NAME as SKIN_LLM_MODEL_NAME,
    PROMPT_VERSION as SKIN_PROMPT_VERSION,
    generate_skin_llm_result,
)


AM_AVOID_INGREDIENTS = {
    "Retinol",
    "Retinyl Palmitate",
    "Glycolic Acid",
    "Lactic Acid",
    "Salicylic Acid",
}
PM_AVOID_INGREDIENTS = {"Oxybenzone", "Avobenzone"}
SKIN_SCORE_KEYS = {
    "acne_score",
    "dryness_score",
    "sagging_score",
    "pore_score",
    "pigmentation_score",
    "wrinkle_score",
}
ROUTINE_CORE_CATEGORIES = {
    "toner",
    "toner pads",
    "emulsions",
    "essences/ampoules/serums",
    "cream/gel",
    "face moisturizers",
}


def _ensure_llm_ready() -> None:
    if not (settings.dgu_llm_api_key or settings.openai_api_key):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is not set. backend/.env에 OPENAI_API_KEY를 설정하세요.",
        )


def _normalize_score(value: Any) -> float:
    score = 0.0 if value is None else float(value)
    score = score / 100.0 if score > 1.0 else score
    return max(0.0, min(score, 1.0))


def _skin_scores_from_result(result: dict[str, Any]) -> dict[str, float]:
    display_scores = result.get("display_scores", {})
    return _complete_skin_scores({
        "acne_score": _normalize_score(display_scores.get("acne")),
        "dryness_score": _normalize_score(display_scores.get("dryness")),
        "sagging_score": _normalize_score(display_scores.get("sagging")),
        "pore_score": _normalize_score(display_scores.get("pore")),
        "pigmentation_score": _normalize_score(display_scores.get("pigmentation")),
        "wrinkle_score": _normalize_score(display_scores.get("wrinkle")),
    })


def _complete_skin_scores(scores: dict[str, float]) -> dict[str, float]:
    return {key: float(scores.get(key) or 0.0) for key in SKIN_SCORE_KEYS}


def _skin_scores_from_db(session: RecommendationSession) -> dict[str, float]:
    db = object_session(session)
    if db is None or session.result_id is None:
        return {}

    row = db.execute(
        text(
            """
            SELECT
                acne_score,
                dryness_score,
                sagging_score,
                pore_score,
                pigmentation_score,
                wrinkle_score
            FROM skin_analysis_result
            WHERE result_id = :result_id
              AND user_id = :user_id
            LIMIT 1
            """
        ),
        {"result_id": session.result_id, "user_id": session.user_id},
    ).mappings().first()

    if not row:
        return {}

    return _complete_skin_scores({
        "acne_score": _normalize_score(row.get("acne_score")),
        "dryness_score": _normalize_score(row.get("dryness_score")),
        "sagging_score": _normalize_score(row.get("sagging_score")),
        "pore_score": _normalize_score(row.get("pore_score")),
        "pigmentation_score": _normalize_score(row.get("pigmentation_score")),
        "wrinkle_score": _normalize_score(row.get("wrinkle_score")),
    })


def _matched_ingredients(db: Any, product_id: int | None, target_names: set[str]) -> list[str]:
    if db is None or product_id is None or not target_names:
        return []

    rows = db.execute(
        text(
            """
            SELECT DISTINCT i.ingredient_name
            FROM product_ingredient pi
            JOIN ingredient i ON i.ingredient_id = pi.ingredient_id
            WHERE pi.product_id = :product_id
              AND i.ingredient_name IN :target_names
            ORDER BY i.ingredient_name
            """
        ).bindparams(bindparam("target_names", expanding=True)),
        {"product_id": product_id, "target_names": list(target_names)},
    ).scalars().all()

    return [str(row) for row in rows]


def _parse_warning(text_value: str) -> dict[str, str]:
    warning_type = "warning"
    body = text_value
    if text_value.startswith("[") and "]" in text_value:
        warning_type, body = text_value.split("]", 1)
        warning_type = warning_type.strip("[] ")
        body = body.strip()

    ingredient_pair = body
    product_a = ""
    product_b = ""
    if "|" in body:
        ingredient_pair, products = body.split("|", 1)
        ingredient_pair = ingredient_pair.strip()
        if "<->" in products:
            product_a, product_b = [part.strip() for part in products.split("<->", 1)]

    return {
        "warning_type": warning_type,
        "ingredient_pair": ingredient_pair,
        "product_a": product_a,
        "product_b": product_b,
    }


def _routine_warnings(conflict_pairs: str | None, limit: int = 3) -> list[dict[str, str]]:
    if not conflict_pairs:
        return []

    warnings = [part.strip() for part in conflict_pairs.split(";") if part.strip()]
    parsed = [_parse_warning(item) for item in warnings[:limit]]
    if len(warnings) > limit:
        parsed.append(
            {
                "warning_type": "summary",
                "ingredient_pair": f"추가 주의 성분 조합 {len(warnings) - limit}개",
                "product_a": "",
                "product_b": "",
            }
        )
    return parsed


def _top_skin_concerns(skin_scores: dict[str, float], top_n: int = 3) -> list[dict[str, Any]]:
    label_map = {
        "acne_score": ("acne", "트러블"),
        "dryness_score": ("dryness", "건조"),
        "sagging_score": ("sagging", "처짐"),
        "pore_score": ("pore", "모공"),
        "pigmentation_score": ("pigmentation", "색소침착"),
        "wrinkle_score": ("wrinkle", "주름"),
    }
    sorted_items = sorted(skin_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [
        {"key": label_map[key][0], "label": label_map[key][1], "score": score}
        for key, score in sorted_items
        if key in label_map
    ]


def generate_skin_summary_llm(result: dict[str, Any]) -> dict[str, Any]:
    _ensure_llm_ready()
    llm_input = {
        "result_id": result["result_id"],
        "image_id": result["image_id"],
        "user_id": result["user_id"],
        **_skin_scores_from_result(result),
        "analyzed_at": result.get("analyzed_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    llm_result = generate_skin_llm_result(
        llm_input=llm_input,
        output_dir=str(BACKEND_ROOT / "llm_outputs" / "skin_analysis"),
    )
    return {
        "model_name": SKIN_LLM_MODEL_NAME,
        "prompt_version": SKIN_PROMPT_VERSION,
        **llm_result,
    }


def _routine_type(routine: RecommendationRoutine) -> str:
    label = (routine.routine_label or "").lower()
    if "value" in label or routine.routine_rank == 2:
        return "value"
    return "best"


def _normalize_category(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_core_routine_category(value: Any) -> bool:
    return _normalize_category(value) in ROUTINE_CORE_CATEGORIES


def build_routine_llm_input(session: RecommendationSession) -> dict[str, Any]:
    db = object_session(session)
    skin_result = store.skin_results.get(int(session.result_id or 0), {})
    skin_scores = _skin_scores_from_result(skin_result) if skin_result else {}
    if not any(skin_scores.values()):
        skin_scores = _skin_scores_from_db(session)
    skin_scores = _complete_skin_scores(skin_scores)
    routines = []

    for routine in sorted(session.routines, key=lambda row: row.routine_rank):
        items = []
        pm_only_products = []
        total_price = 0
        for item in sorted(routine.items, key=lambda row: row.slot_order):
            product = item.product
            if not product:
                continue
            price = int(product.price or 0)
            brand = product.brand_name_kor or product.brand_name or ""
            product_name = product.product_name_kor or product.product_name or ""
            category = item.category or product.category or ""
            if _is_core_routine_category(category):
                total_price += price
            items.append(
                {
                    "slot_order": item.slot_order,
                    "category": category,
                    "brand": brand,
                    "product_name": product_name,
                    "product_score": float(item.product_score or 0),
                    "time_tag": item.time_tag,
                    "price": price,
                }
            )
            if item.time_tag in {"pm", "check"}:
                ingredients = _matched_ingredients(db, item.product_id, AM_AVOID_INGREDIENTS)
                pm_only_products.append(
                    {
                        "product_name": product_name,
                        "brand": brand,
                        "ingredient": ", ".join(ingredients) if ingredients else None,
                    }
                )

        routines.append(
            {
                "routine_id": routine.routine_id,
                "routine_type": _routine_type(routine),
                "routine_rank": routine.routine_rank,
                "ampm_mode": routine.ampm_mode or "am+pm",
                "routine_score": float(routine.routine_score or 0),
                "total_price": total_price,
                "budget_limit": session.total_budget_max,
                "items": items,
                "warnings": _routine_warnings(routine.conflict_pairs),
                "pm_only_products": pm_only_products,
            }
        )

    return {
        "rec_session_id": session.session_id,
        "user_id": session.user_id,
        "image_id": session.image_id,
        "result_id": session.result_id,
        "recommended_at": session.created_at.strftime("%Y-%m-%d %H:%M:%S") if session.created_at else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "skin_scores": skin_scores,
        "top_skin_concerns": _top_skin_concerns(skin_scores),
        "routines": routines,
    }


def generate_routine_explanation_llm(session: RecommendationSession) -> dict[str, Any]:
    _ensure_llm_ready()
    llm_input = build_routine_llm_input(session)
    if not llm_input["routines"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Recommendation session has no routines. Run POST /recommendations again with a successful session_id.",
        )
    llm_result = generate_routine_llm_result(
        llm_input=llm_input,
        output_dir=str(BACKEND_ROOT / "llm_outputs" / "routine_recommendation"),
    )
    return {
        "model_name": ROUTINE_LLM_MODEL_NAME,
        "prompt_version": ROUTINE_PROMPT_VERSION,
        **llm_result,
    }
