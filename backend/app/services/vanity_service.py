from __future__ import annotations

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.memory import (
    save_vanity_routine_explanations,
    save_vanity_skin_match_explanations,
    store,
)
from app.services.db_catalog import get_product_or_404, serialize_product_list_item


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]

DISPLAY_LABELS = {
    "excellent_match": "아주 잘 맞아요",
    "good_match": "괜찮은 편이에요",
    "so_so": "보통이에요",
    "weak_match": "아쉬워요",
    "poor_match": "주의가 필요해요",
}


def _now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _http_error(status_code: int, error_code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"error_code": error_code, "message": message},
    )


def _prepare_model_imports() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    os.environ["ROUPLE_MYSQL_HOST"] = settings.mysql_host
    os.environ["ROUPLE_MYSQL_PORT"] = str(settings.mysql_port)
    os.environ["ROUPLE_MYSQL_USER"] = settings.mysql_user
    os.environ["ROUPLE_MYSQL_PASSWORD"] = settings.mysql_password
    os.environ["ROUPLE_MYSQL_DB"] = settings.mysql_db


def _sync_model_db_config() -> None:
    _prepare_model_imports()
    try:
        from model.recommendation.kg_pipeline.neo4j_skincare import config as kg_config

        kg_config.MYSQL_HOST = settings.mysql_host
        kg_config.MYSQL_PORT = settings.mysql_port
        kg_config.MYSQL_USER = settings.mysql_user
        kg_config.MYSQL_PASSWORD = settings.mysql_password
        kg_config.MYSQL_DB = settings.mysql_db
    except ModuleNotFoundError:
        pass

    try:
        from model.vanity import data_loader

        data_loader.MYSQL_HOST = settings.mysql_host
        data_loader.MYSQL_PORT = settings.mysql_port
        data_loader.MYSQL_USER = settings.mysql_user
        data_loader.MYSQL_PASSWORD = settings.mysql_password
        data_loader.MYSQL_DB = settings.mysql_db
    except ModuleNotFoundError:
        pass


def _skin_concern_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).replace("|", ",").split(",") if item.strip()]


def _score_concerns(row: dict[str, Any], top_n: int = 3) -> list[str]:
    score_names = {
        "acne_score": "acne",
        "dryness_score": "dryness",
        "sagging_score": "sagging",
        "pore_score": "pore",
        "pigmentation_score": "pigmentation",
        "wrinkle_score": "wrinkle",
    }
    scores = []
    for column, name in score_names.items():
        value = row.get(column)
        if value is None:
            continue
        scores.append((name, float(value)))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [name for name, score in scores[:top_n] if score > 0]


def get_latest_skin_result(db: Session, user_id: int) -> dict[str, Any]:
    row = db.execute(
        text(
            """
            SELECT *
            FROM skin_analysis_result
            WHERE user_id = :user_id
            ORDER BY analyzed_at DESC, result_id DESC
            LIMIT 1
            """
        ),
        {"user_id": user_id},
    ).mappings().first()
    if not row:
        raise _http_error(
            status.HTTP_404_NOT_FOUND,
            "SKIN_ANALYSIS_RESULT_NOT_FOUND",
            "피부 분석 결과가 없어 내 화장대 분석을 진행할 수 없습니다.",
        )
    return dict(row)


def basis_skin_result(db: Session, user_id: int, result_id: int | None = None) -> dict[str, Any]:
    if result_id is None:
        row = get_latest_skin_result(db, user_id)
    else:
        found = db.execute(
            text(
                """
                SELECT *
                FROM skin_analysis_result
                WHERE user_id = :user_id AND result_id = :result_id
                LIMIT 1
                """
            ),
            {"user_id": user_id, "result_id": result_id},
        ).mappings().first()
        if not found:
            raise _http_error(
                status.HTTP_404_NOT_FOUND,
                "SKIN_ANALYSIS_RESULT_NOT_FOUND",
                "피부 분석 결과를 찾을 수 없습니다.",
            )
        row = dict(found)

    return {
        "result_id": int(row["result_id"]),
        "image_id": row.get("image_id"),
        "analyzed_at": str(row.get("analyzed_at")) if row.get("analyzed_at") is not None else None,
        "main_concerns": _score_concerns(row),
    }


def list_vanity_products(db: Session, user_id: int) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT uv.vanity_id, uv.product_id, uv.created_at
            FROM user_vanity uv
            JOIN product p ON p.product_id = uv.product_id
            WHERE uv.user_id = :user_id
            ORDER BY uv.created_at DESC, uv.vanity_id DESC
            """
        ),
        {"user_id": user_id},
    ).mappings().all()

    products = []
    for row in rows:
        product = get_product_or_404(db, int(row["product_id"]))
        product_data = serialize_product_list_item(db, product)
        products.append(
            {
                "vanity_id": int(row["vanity_id"]),
                "created_at": str(row["created_at"]) if row.get("created_at") is not None else None,
                **product_data,
            }
        )
    return products


def add_vanity_product(db: Session, user_id: int, product_id: int) -> dict[str, Any]:
    get_product_or_404(db, product_id)
    existing_vanity_id = db.scalar(
        text(
            """
            SELECT vanity_id
            FROM user_vanity
            WHERE user_id = :user_id AND product_id = :product_id
            LIMIT 1
            """
        ),
        {"user_id": user_id, "product_id": product_id},
    )
    if existing_vanity_id is not None:
        return {
            "vanity_id": int(existing_vanity_id),
            "product_id": product_id,
            "saved": True,
            "match_session_id": _latest_match_session_id(db, user_id),
            "message": "내 화장대에 이미 등록된 제품입니다.",
        }

    db.execute(
        text(
            """
            INSERT INTO user_vanity (user_id, product_id)
            VALUES (:user_id, :product_id)
            ON DUPLICATE KEY UPDATE created_at = created_at
            """
        ),
        {"user_id": user_id, "product_id": product_id},
    )
    db.commit()
    vanity_id = db.scalar(
        text(
            """
            SELECT vanity_id
            FROM user_vanity
            WHERE user_id = :user_id AND product_id = :product_id
            LIMIT 1
            """
        ),
        {"user_id": user_id, "product_id": product_id},
    )
    skin_match_session_id = None
    try:
        skin_match = run_skin_match(db, user_id, product_ids=None)
        skin_match_session_id = skin_match.get("match_session_id")
    except HTTPException:
        pass

    return {
        "vanity_id": int(vanity_id) if vanity_id is not None else None,
        "product_id": product_id,
        "saved": True,
        "match_session_id": skin_match_session_id,
        "message": "내 화장대에 제품이 등록되었습니다.",
    }


def delete_vanity_product(db: Session, user_id: int, product_id: int) -> dict[str, Any]:
    db.execute(
        text(
            """
            DELETE FROM user_vanity
            WHERE user_id = :user_id AND product_id = :product_id
            """
        ),
        {"user_id": user_id, "product_id": product_id},
    )
    db.commit()
    return {
        "product_id": product_id,
        "saved": False,
        "match_session_id": _latest_match_session_id(db, user_id),
        "message": "내 화장대에서 제품이 삭제되었습니다.",
    }


def _owned_product_ids(db: Session, user_id: int) -> set[int]:
    rows = db.execute(
        text("SELECT product_id FROM user_vanity WHERE user_id = :user_id"),
        {"user_id": user_id},
    ).mappings().all()
    return {int(row["product_id"]) for row in rows}


def _ensure_owned_products(db: Session, user_id: int, product_ids: list[int]) -> None:
    owned = _owned_product_ids(db, user_id)
    if not owned:
        raise _http_error(status.HTTP_400_BAD_REQUEST, "VANITY_PRODUCT_EMPTY", "내 화장대에 등록된 제품이 없습니다.")
    missing = [product_id for product_id in product_ids if product_id not in owned]
    if missing:
        raise _http_error(
            status.HTTP_400_BAD_REQUEST,
            "FIXED_PRODUCT_NOT_IN_VANITY",
            f"내 화장대에 없는 제품이 포함되어 있습니다: {missing}",
        )


def _normalize_product_match(item: dict[str, Any]) -> dict[str, Any]:
    fit_label = str(item.get("fit_label") or "")
    return {
        **item,
        "display_label": DISPLAY_LABELS.get(fit_label, fit_label),
    }


def _routine_results_from_pipeline(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {
            "fixed_products": [],
            "recommended_products": [],
            "final_routine": [],
            "warnings": [],
            "total_price": None,
        }
    return value


def _skin_match_explanation(product_match_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(product_match_results)
    excellent = sum(1 for item in product_match_results if item.get("fit_label") == "excellent_match")
    good = sum(1 for item in product_match_results if item.get("fit_label") == "good_match")
    overall = f"총 {total}개 제품을 분석했어요. 아주 적합 {excellent}개, 적합 {good}개로 확인되었습니다."

    comments = []
    for item in product_match_results:
        display_label = item.get("display_label") or DISPLAY_LABELS.get(str(item.get("fit_label")), "")
        reason_tags = item.get("reason_tags") or []
        caution_tags = item.get("caution_tags") or []
        comments.append(
            {
                "product_id": int(item["product_id"]),
                "summary": f"{item.get('brand_name', '')} {item.get('product_name', '')}은(는) {display_label} 제품이에요.",
                "fit_reason": "피부 고민, 피부 타입, 리뷰 정보를 함께 반영해 계산했습니다." if reason_tags else "강한 긍정 태그는 적지만 기본 적합도 기준으로 판단했습니다.",
                "caution_comment": "처음 사용할 때 피부 반응을 가볍게 확인해 주세요." if caution_tags else "",
                "action_comment": "현재 루틴에서 유지해도 괜찮아요." if item.get("recommend_action") in {"strong_keep", "keep"} else "부족한 부분은 루틴 추천에서 보완해 주세요.",
            }
        )
    return {"overall_summary": overall, "product_comments": comments}


def _routine_explanation(routine_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not routine_result:
        return None
    final_routine = routine_result.get("final_routine") or []
    fixed_count = sum(1 for item in final_routine if item.get("source") == "vanity")
    recommended_count = sum(1 for item in final_routine if item.get("source") == "recommendation")
    comments = [
        {
            "slot_order": int(item["slot_order"]),
            "product_id": int(item["product_id"]),
            "comment": "내 화장대 고정 제품입니다." if item.get("source") == "vanity" else "빈 루틴 단계를 채우기 위해 새로 추천된 제품입니다.",
        }
        for item in final_routine
    ]
    warnings = routine_result.get("warnings") or []
    return {
        "overall_summary": f"고정 제품 {fixed_count}개를 유지하고 추천 제품 {recommended_count}개로 빈 단계를 보완했어요.",
        "step_comments": comments,
        "warning_comment": " / ".join(str(warning) for warning in warnings),
    }


def build_vanity_llm_explanation(
    product_match_results: list[dict[str, Any]],
    routine_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "prompt_version": "vanity_v1",
        "generated_at": _now_string(),
        "skin_match": _skin_match_explanation(product_match_results),
        "vanity_routine": _routine_explanation(routine_result),
    }


def _load_user_profile_for_llm(db: Session, user_id: int) -> dict[str, Any]:
    row = db.execute(
        text(
            """
            SELECT gender, skin_type, skin_concern
            FROM user_profile
            WHERE user_id = :user_id
            LIMIT 1
            """
        ),
        {"user_id": user_id},
    ).mappings().first()
    if not row:
        return {"gender": None, "skin_type": None, "skin_concern": []}
    return {
        "gender": row.get("gender"),
        "skin_type": row.get("skin_type"),
        "skin_concern": _skin_concern_list(row.get("skin_concern")),
    }


def _load_skin_analysis_for_llm(db: Session, user_id: int, result_id: int) -> dict[str, float]:
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
            WHERE user_id = :user_id AND result_id = :result_id
            LIMIT 1
            """
        ),
        {"user_id": user_id, "result_id": result_id},
    ).mappings().first()
    if not row:
        return {
            "acne_score": 0.0,
            "dryness_score": 0.0,
            "sagging_score": 0.0,
            "pore_score": 0.0,
            "pigmentation_score": 0.0,
            "wrinkle_score": 0.0,
        }
    return {
        key: float(row.get(key) or 0.0)
        for key in [
            "acne_score",
            "dryness_score",
            "sagging_score",
            "pore_score",
            "pigmentation_score",
            "wrinkle_score",
        ]
    }


def _prepare_llm_env() -> None:
    if settings.dgu_llm_api_key:
        os.environ["DGU_LLM_API_KEY"] = settings.dgu_llm_api_key
    if settings.dgu_llm_base_url:
        os.environ["DGU_LLM_BASE_URL"] = settings.dgu_llm_base_url
    if settings.dgu_llm_model:
        os.environ["DGU_LLM_MODEL"] = settings.dgu_llm_model


def _build_vanity_llm_input(
    db: Session,
    user_id: int,
    result_id: int,
    product_match_results: list[dict[str, Any]],
    routine_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "result_id": result_id,
        "user_profile": _load_user_profile_for_llm(db, user_id),
        "skin_analysis_result": _load_skin_analysis_for_llm(db, user_id, result_id),
        "product_match_results": product_match_results,
        "routine_recommendation_results": {
            "final_routine": routine_result.get("final_routine") or [],
            "warnings": routine_result.get("warnings") or [],
            "total_price": routine_result.get("total_price"),
        },
    }


def _skin_match_only_routine(product_match_results: list[dict[str, Any]]) -> dict[str, Any]:
    final_routine = []
    for idx, product in enumerate(product_match_results, start=1):
        final_routine.append(
            {
                "slot_order": idx,
                "product_id": int(product["product_id"]),
                "category": product.get("category"),
                "brand_name": product.get("brand_name"),
                "product_name": product.get("product_name"),
                "source": "vanity",
                "product_score": float(product.get("vanity_fit_score") or 0.0),
                "price": int(product.get("price") or 0),
            }
        )
    return {
        "final_routine": final_routine,
        "warnings": [],
        "total_price": None,
    }


def generate_vanity_llm_explanation(
    db: Session,
    user_id: int,
    result_id: int,
    product_match_results: list[dict[str, Any]],
    routine_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _prepare_llm_env()
    try:
        from model.llm.vanity_llm import generate_vanity_llm_result

        if not routine_result or not routine_result.get("final_routine"):
            routine_result = _skin_match_only_routine(product_match_results)

        llm_input = _build_vanity_llm_input(
            db=db,
            user_id=user_id,
            result_id=result_id,
            product_match_results=product_match_results,
            routine_result=routine_result,
        )
        return generate_vanity_llm_result(
            llm_input=llm_input,
            output_dir=str(BACKEND_ROOT / "llm_outputs" / "vanity"),
        )
    except Exception as exc:
        print(f"[vanity-llm] failed: {type(exc).__name__}: {exc}")
        return build_vanity_llm_explanation(product_match_results, routine_result)


def run_skin_match(db: Session, user_id: int, product_ids: list[int] | None = None) -> dict[str, Any]:
    basis = basis_skin_result(db, user_id)
    if product_ids:
        _ensure_owned_products(db, user_id, product_ids)
    else:
        product_ids = None

    _sync_model_db_config()
    try:
        from model.vanity.pipeline import run_vanity_pipeline
        from model.vanity.schemas import VanityPipelineInput

        result = run_vanity_pipeline(
            VanityPipelineInput(
                user_id=user_id,
                result_id=basis["result_id"],
                vanity_product_ids=product_ids,
                fixed_product_ids=None,
                budget=None,
            ),
            save_skin_match=True,
            save_routine=False,
        )
    except ValueError as exc:
        message = str(exc)
        if "USER_VANITY products not found" in message:
            raise _http_error(status.HTTP_400_BAD_REQUEST, "VANITY_PRODUCT_EMPTY", "분석할 내 화장대 제품이 없습니다.") from exc
        if "SKIN_ANALYSIS_RESULT not found" in message:
            raise _http_error(status.HTTP_404_NOT_FOUND, "SKIN_ANALYSIS_RESULT_NOT_FOUND", "피부 분석 결과를 찾을 수 없습니다.") from exc
        raise _http_error(status.HTTP_400_BAD_REQUEST, "VANITY_SKIN_MATCH_FAILED", message) from exc
    except Exception as exc:
        raise _http_error(status.HTTP_500_INTERNAL_SERVER_ERROR, "VANITY_SKIN_MATCH_FAILED", str(exc)) from exc

    matches = [_normalize_product_match(item) for item in result["product_match_results"]]
    llm_explanation = generate_vanity_llm_explanation(
        db=db,
        user_id=user_id,
        result_id=basis["result_id"],
        product_match_results=matches,
    )
    match_session_id = result.get("match_session_id")
    if match_session_id is not None:
        store.vanity_skin_match_explanations[int(match_session_id)] = llm_explanation
        save_vanity_skin_match_explanations(store.vanity_skin_match_explanations)

    return {
        "match_session_id": match_session_id,
        "user_id": user_id,
        "basis_skin_result": basis,
        "product_match_results": matches,
        "llm_explanation": llm_explanation,
    }


def _slot_budget_maps(budget_payload: Any) -> tuple[dict[str, int], dict[str, int]]:
    if budget_payload is None:
        return {}, {}
    slot_budget_min = {
        key: value
        for key, value in {
            "Toner": getattr(budget_payload, "toner_budget_min", None),
            "Emulsions": getattr(budget_payload, "emulsion_budget_min", None),
            "Essences/Ampoules/Serums": getattr(budget_payload, "ampoule_budget_min", None),
            "Cream/Gel": getattr(budget_payload, "cream_budget_min", None),
        }.items()
        if value is not None
    }
    slot_budget_max = {
        key: value
        for key, value in {
            "Toner": getattr(budget_payload, "toner_budget_max", None),
            "Emulsions": getattr(budget_payload, "emulsion_budget_max", None),
            "Essences/Ampoules/Serums": getattr(budget_payload, "ampoule_budget_max", None),
            "Cream/Gel": getattr(budget_payload, "cream_budget_max", None),
        }.items()
        if value is not None
    }
    return slot_budget_min, slot_budget_max


def _update_vanity_budget_session(db: Session, session_id: int | None, budget_payload: Any) -> None:
    if session_id is None or budget_payload is None:
        return
    slot_budget_min, slot_budget_max = _slot_budget_maps(budget_payload)
    db.execute(
        text(
            """
            UPDATE recommendation_session
            SET
                total_budget_min = :total_budget_min,
                total_budget_max = :total_budget_max,
                slot_budget_min_json = :slot_budget_min_json,
                slot_budget_max_json = :slot_budget_max_json
            WHERE session_id = :session_id
            """
        ),
        {
            "session_id": session_id,
            "total_budget_min": getattr(budget_payload, "total_budget_min", None),
            "total_budget_max": getattr(budget_payload, "total_budget_max", None),
            "slot_budget_min_json": json.dumps(slot_budget_min, ensure_ascii=False) if slot_budget_min else None,
            "slot_budget_max_json": json.dumps(slot_budget_max, ensure_ascii=False) if slot_budget_max else None,
        },
    )
    db.commit()


def run_vanity_routine(db: Session, user_id: int, fixed_product_ids: list[int], budget_payload: Any = None) -> dict[str, Any]:
    basis = basis_skin_result(db, user_id)
    _ensure_owned_products(db, user_id, fixed_product_ids)
    total_budget_max = getattr(budget_payload, "total_budget_max", None) if budget_payload is not None else None

    _sync_model_db_config()
    try:
        from model.vanity.pipeline import run_vanity_pipeline
        from model.vanity.schemas import VanityPipelineInput

        result = run_vanity_pipeline(
            VanityPipelineInput(
                user_id=user_id,
                result_id=basis["result_id"],
                vanity_product_ids=None,
                fixed_product_ids=fixed_product_ids,
                budget=total_budget_max,
            ),
            save_skin_match=True,
            save_routine=True,
        )
    except ValueError as exc:
        message = str(exc)
        if "only one product per category" in message:
            raise _http_error(status.HTTP_400_BAD_REQUEST, "DUPLICATE_FIXED_CATEGORY", message) from exc
        raise _http_error(status.HTTP_400_BAD_REQUEST, "VANITY_ROUTINE_FAILED", message) from exc
    except Exception as exc:
        detail = str(exc)
        if "Neo4j" in detail or "Failed to establish connection" in detail:
            raise _http_error(status.HTTP_503_SERVICE_UNAVAILABLE, "NEO4J_CONNECTION_FAILED", detail) from exc
        if "candidate" in detail.lower() or "embedding" in detail.lower():
            raise _http_error(status.HTTP_500_INTERNAL_SERVER_ERROR, "RECOMMENDATION_CANDIDATE_GENERATION_FAILED", detail) from exc
        raise _http_error(status.HTTP_500_INTERNAL_SERVER_ERROR, "VANITY_ROUTINE_FAILED", detail) from exc

    _update_vanity_budget_session(db, result.get("recommendation_session_id"), budget_payload)
    matches = [_normalize_product_match(item) for item in result.get("product_match_results", [])]
    routine_result = _routine_results_from_pipeline(result.get("routine_recommendation_results"))
    llm_explanation = generate_vanity_llm_explanation(
        db=db,
        user_id=user_id,
        result_id=basis["result_id"],
        product_match_results=matches,
        routine_result=routine_result,
    )
    recommendation_session_id = result.get("recommendation_session_id")
    if recommendation_session_id is not None:
        store.vanity_routine_explanations[int(recommendation_session_id)] = llm_explanation
        save_vanity_routine_explanations(store.vanity_routine_explanations)

    return {
        "recommendation_session_id": recommendation_session_id,
        "user_id": user_id,
        "basis_skin_result": basis,
        "product_match_results": matches,
        "routine_recommendation_results": routine_result,
        "llm_explanation": llm_explanation,
    }


def _latest_match_session_id(db: Session, user_id: int) -> int | None:
    value = db.scalar(
        text(
            """
            SELECT match_session_id
            FROM vanity_match_session
            WHERE user_id = :user_id
            ORDER BY created_at DESC, match_session_id DESC
            LIMIT 1
            """
        ),
        {"user_id": user_id},
    )
    return int(value) if value is not None else None


def _latest_skin_match_summary(db: Session, user_id: int) -> dict[str, Any] | None:
    match_session_id = _latest_match_session_id(db, user_id)
    if match_session_id is None:
        return None

    session_row = db.execute(
        text(
            """
            SELECT match_session_id, created_at
            FROM vanity_match_session
            WHERE match_session_id = :match_session_id AND user_id = :user_id
            LIMIT 1
            """
        ),
        {"match_session_id": match_session_id, "user_id": user_id},
    ).mappings().first()
    if not session_row:
        return None

    rows = db.execute(
        text(
            """
            SELECT fit_label, COUNT(*) AS count
            FROM vanity_match_item vmi
            JOIN user_vanity uv ON uv.user_id = :user_id AND uv.product_id = vmi.product_id
            WHERE vmi.match_session_id = :match_session_id
            GROUP BY fit_label
            """
        ),
        {"match_session_id": match_session_id, "user_id": user_id},
    ).mappings().all()

    summary = {key: 0 for key in DISPLAY_LABELS}
    for row in rows:
        fit_label = str(row.get("fit_label") or "")
        summary[fit_label] = int(row.get("count") or 0)

    return {
        "match_session_id": int(session_row["match_session_id"]),
        "created_at": str(session_row["created_at"]) if session_row.get("created_at") is not None else None,
        "summary": summary,
    }


def get_latest_skin_match(db: Session, user_id: int) -> dict[str, Any]:
    match_session_id = _latest_match_session_id(db, user_id)
    if match_session_id is None:
        created = run_skin_match(db, user_id, product_ids=None)
        match_session_id = created.get("match_session_id")
        if match_session_id is None:
            raise _http_error(status.HTTP_404_NOT_FOUND, "VANITY_MATCH_NOT_FOUND", "Skin Match 결과가 없습니다.")

    session_row = db.execute(
        text(
            """
            SELECT match_session_id, result_id, created_at
            FROM vanity_match_session
            WHERE match_session_id = :match_session_id AND user_id = :user_id
            LIMIT 1
            """
        ),
        {"match_session_id": match_session_id, "user_id": user_id},
    ).mappings().first()
    item_rows = db.execute(
        text(
            """
            SELECT
                vmi.product_id,
                p.category,
                COALESCE(p.brand_name_kor, p.brand_name, '') AS brand_name,
                COALESCE(p.product_name_kor, p.product_name, '') AS product_name,
                vmi.vanity_fit_score,
                vmi.concern_match_score,
                vmi.skin_type_bonus,
                vmi.review_score,
                vmi.irritation_penalty,
                vmi.fit_label,
                vmi.recommend_action,
                vmi.reason_tags,
                vmi.caution_tags
            FROM vanity_match_item vmi
            JOIN product p ON p.product_id = vmi.product_id
            JOIN user_vanity uv ON uv.user_id = :user_id AND uv.product_id = vmi.product_id
            WHERE vmi.match_session_id = :match_session_id
            ORDER BY vmi.vanity_fit_score DESC, vmi.match_item_id
            """
        ),
        {"match_session_id": match_session_id, "user_id": user_id},
    ).mappings().all()

    import json

    results = []
    summary = {key: 0 for key in DISPLAY_LABELS}
    for row in item_rows:
        fit_label = str(row["fit_label"])
        summary[fit_label] = summary.get(fit_label, 0) + 1
        results.append(
            _normalize_product_match(
                {
                    "product_id": int(row["product_id"]),
                    "category": row.get("category"),
                    "brand_name": row.get("brand_name"),
                    "product_name": row.get("product_name"),
                    "vanity_fit_score": float(row["vanity_fit_score"] or 0.0),
                    "scores": {
                        "concern_match_score": float(row["concern_match_score"] or 0.0),
                        "skin_type_bonus": float(row["skin_type_bonus"] or 0.0),
                        "review_score": float(row["review_score"] or 0.0),
                        "irritation_penalty": float(row["irritation_penalty"] or 0.0),
                        "vanity_fit_score": float(row["vanity_fit_score"] or 0.0),
                    },
                    "fit_label": fit_label,
                    "recommend_action": row.get("recommend_action") or "",
                    "reason_tags": json.loads(row.get("reason_tags") or "[]"),
                    "caution_tags": json.loads(row.get("caution_tags") or "[]"),
                }
            )
        )

    basis = basis_skin_result(db, user_id, int(session_row["result_id"]))
    llm_explanation = store.vanity_skin_match_explanations.get(int(match_session_id))
    if llm_explanation is None:
        llm_explanation = build_vanity_llm_explanation(results, None)
        store.vanity_skin_match_explanations[int(match_session_id)] = llm_explanation
        save_vanity_skin_match_explanations(store.vanity_skin_match_explanations)

    return {
        "match_session_id": match_session_id,
        "created_at": str(session_row["created_at"]) if session_row and session_row.get("created_at") is not None else None,
        "basis_skin_result": basis,
        "summary": summary,
        "product_match_results": results,
        "llm_explanation": llm_explanation,
    }


def _routine_item_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "slot_order": int(row["slot_order"]),
        "category": row.get("category"),
        "product_id": int(row["product_id"]),
        "source": row.get("source") or "recommendation",
        "product_score": float(row["product_score"]) if row.get("product_score") is not None else None,
        "brand_name": row.get("brand_name"),
        "product_name": row.get("product_name"),
        "price": int(row["price"] or 0),
    }


def get_vanity_routine_detail(db: Session, user_id: int, session_id: int) -> dict[str, Any]:
    session_row = db.execute(
        text(
            """
            SELECT session_id, result_id, created_at
            FROM recommendation_session
            WHERE session_id = :session_id
              AND user_id = :user_id
              AND COALESCE(recommendation_type, 'basic') = 'vanity'
            LIMIT 1
            """
        ),
        {"session_id": session_id, "user_id": user_id},
    ).mappings().first()
    if not session_row:
        raise _http_error(status.HTTP_404_NOT_FOUND, "VANITY_ROUTINE_NOT_FOUND", "Vanity Routine 결과가 없습니다.")

    rows = db.execute(
        text(
            """
            SELECT
                ri.slot_order,
                ri.category,
                ri.product_id,
                ri.product_score,
                COALESCE(ri.source, 'recommendation') AS source,
                COALESCE(p.brand_name_kor, p.brand_name, '') AS brand_name,
                COALESCE(p.product_name_kor, p.product_name, '') AS product_name,
                COALESCE(p.price, 0) AS price,
                rr.conflict_pairs
            FROM recommendation_routine rr
            JOIN recommendation_item ri ON ri.routine_id = rr.routine_id
            JOIN product p ON p.product_id = ri.product_id
            WHERE rr.session_id = :session_id
              AND rr.routine_label = 'Vanity'
            ORDER BY ri.slot_order
            """
        ),
        {"session_id": session_id},
    ).mappings().all()

    final_routine = [_routine_item_from_row(dict(row)) for row in rows]
    fixed_products = [item for item in final_routine if item["source"] == "vanity"]
    recommended_products = [item for item in final_routine if item["source"] == "recommendation"]
    warnings = []
    if rows and rows[0].get("conflict_pairs"):
        import json

        try:
            warnings = json.loads(rows[0]["conflict_pairs"])
        except json.JSONDecodeError:
            warnings = [str(rows[0]["conflict_pairs"])]
    routine_result = {
        "fixed_products": fixed_products,
        "recommended_products": recommended_products,
        "final_routine": final_routine,
        "warnings": warnings,
        "total_price": sum(int(item.get("price") or 0) for item in final_routine),
    }
    basis = basis_skin_result(db, user_id, int(session_row["result_id"]))
    llm_explanation = store.vanity_routine_explanations.get(int(session_id))
    if llm_explanation is None:
        match = get_latest_skin_match(db, user_id)
        llm_explanation = generate_vanity_llm_explanation(
            db=db,
            user_id=user_id,
            result_id=basis["result_id"],
            product_match_results=match.get("product_match_results") or [],
            routine_result=routine_result,
        )
        store.vanity_routine_explanations[int(session_id)] = llm_explanation
        save_vanity_routine_explanations(store.vanity_routine_explanations)

    return {
        "recommendation_session_id": session_id,
        "created_at": str(session_row["created_at"]) if session_row.get("created_at") is not None else None,
        "user_id": user_id,
        "basis_skin_result": basis,
        "routine_recommendation_results": routine_result,
        "llm_explanation": llm_explanation,
    }


def get_latest_vanity_routine(db: Session, user_id: int) -> dict[str, Any]:
    session_id = db.scalar(
        text(
            """
            SELECT session_id
            FROM recommendation_session
            WHERE user_id = :user_id
              AND COALESCE(recommendation_type, 'basic') = 'vanity'
            ORDER BY created_at DESC, session_id DESC
            LIMIT 1
            """
        ),
        {"user_id": user_id},
    )
    if session_id is None:
        raise _http_error(status.HTTP_404_NOT_FOUND, "VANITY_ROUTINE_NOT_FOUND", "Vanity Routine 결과가 없습니다.")
    return get_vanity_routine_detail(db, user_id, int(session_id))


def _latest_vanity_routine_summary(db: Session, user_id: int) -> dict[str, Any] | None:
    row = db.execute(
        text(
            """
            SELECT
                rs.session_id AS recommendation_session_id,
                rs.created_at,
                SUM(CASE WHEN COALESCE(ri.source, 'recommendation') = 'vanity' THEN 1 ELSE 0 END) AS fixed_product_count,
                SUM(COALESCE(p.price, 0)) AS total_price
            FROM recommendation_session rs
            JOIN recommendation_routine rr ON rr.session_id = rs.session_id
            JOIN recommendation_item ri ON ri.routine_id = rr.routine_id
            JOIN product p ON p.product_id = ri.product_id
            WHERE rs.user_id = :user_id
              AND COALESCE(rs.recommendation_type, 'basic') = 'vanity'
              AND rr.routine_label = 'Vanity'
            GROUP BY rs.session_id, rs.created_at
            ORDER BY rs.created_at DESC, rs.session_id DESC
            LIMIT 1
            """
        ),
        {"user_id": user_id},
    ).mappings().first()
    if not row:
        return None

    return {
        "recommendation_session_id": int(row["recommendation_session_id"]),
        "created_at": str(row["created_at"]) if row.get("created_at") is not None else None,
        "fixed_product_count": int(row["fixed_product_count"] or 0),
        "total_price": int(row["total_price"] or 0),
    }


def list_vanity_routines(db: Session, user_id: int) -> list[dict[str, Any]]:
    rows = db.execute(
        text(
            """
            SELECT
                rs.session_id AS recommendation_session_id,
                rs.result_id AS basis_result_id,
                rs.created_at,
                SUM(CASE WHEN COALESCE(ri.source, 'recommendation') = 'vanity' THEN 1 ELSE 0 END) AS fixed_product_count,
                SUM(COALESCE(p.price, 0)) AS total_price
            FROM recommendation_session rs
            JOIN recommendation_routine rr ON rr.session_id = rs.session_id
            JOIN recommendation_item ri ON ri.routine_id = rr.routine_id
            JOIN product p ON p.product_id = ri.product_id
            WHERE rs.user_id = :user_id
              AND COALESCE(rs.recommendation_type, 'basic') = 'vanity'
              AND rr.routine_label = 'Vanity'
            GROUP BY rs.session_id, rs.result_id, rs.created_at
            ORDER BY rs.created_at DESC, rs.session_id DESC
            """
        ),
        {"user_id": user_id},
    ).mappings().all()
    return [
        {
            "recommendation_session_id": int(row["recommendation_session_id"]),
            "created_at": str(row["created_at"]) if row.get("created_at") is not None else None,
            "basis_result_id": row.get("basis_result_id"),
            "fixed_product_count": int(row["fixed_product_count"] or 0),
            "total_price": int(row["total_price"] or 0),
        }
        for row in rows
    ]


def get_vanity_summary(db: Session, user_id: int) -> dict[str, Any]:
    products = list_vanity_products(db, user_id)
    try:
        basis = basis_skin_result(db, user_id)
    except HTTPException:
        basis = None

    latest_match = None
    latest_match = _latest_skin_match_summary(db, user_id)

    latest_routine = _latest_vanity_routine_summary(db, user_id)

    return {
        "product_summary": {
            "total_count": len(products),
            "products": products,
        },
        "latest_skin_match": latest_match,
        "latest_vanity_routine": latest_routine,
        "basis_skin_result": basis,
    }
