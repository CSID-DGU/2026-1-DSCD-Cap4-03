from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.memory import store
from app.db.session import get_db
from app.schemas.users import UpdateAllergiesRequest, UpdateAllergiesResponse, UpdateProfileRequest, UserProfileResponse
from app.services.db_catalog import list_wishlist_products, serialize_product_list_item
from app.services.db_user import (
    ensure_profile,
    get_user_allergies,
    get_user_by_id,
    get_user_image,
    replace_user_allergies,
    serialize_user_profile,
    update_user_profile,
)
from app.services.deps import get_current_user
from app.services.files import resolve_image_display_url
from app.services.recommendation_model import get_recommendation_session_or_404, serialize_recommendation_session


router = APIRouter()


@router.get("/me", response_model=UserProfileResponse)
def get_me(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)) -> UserProfileResponse:
    user = get_user_by_id(db, current_user["user_id"])
    profile = ensure_profile(db, current_user["user_id"])
    return UserProfileResponse(**serialize_user_profile(user, profile))


@router.patch("/me/profile", response_model=UserProfileResponse)
def update_profile(
    payload: UpdateProfileRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserProfileResponse:
    user = get_user_by_id(db, current_user["user_id"])
    return UserProfileResponse(**update_user_profile(db, user, payload))


@router.put("/me/allergies", response_model=UpdateAllergiesResponse)
def update_allergies(
    payload: UpdateAllergiesRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UpdateAllergiesResponse:
    return UpdateAllergiesResponse(**replace_user_allergies(db, current_user["user_id"], payload))


@router.get("/me/allergies", response_model=UpdateAllergiesResponse)
def get_allergies(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UpdateAllergiesResponse:
    return UpdateAllergiesResponse(**get_user_allergies(db, current_user["user_id"]))


@router.get("/me/wishlist")
def get_my_wishlist(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    return {"items": list_wishlist_products(db, current_user["user_id"])}


@router.get("/me/routines")
def get_my_routines(
    response: Response,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    sessions = db.execute(
        text(
            """
            SELECT session_id, result_id, image_id, created_at
            FROM recommendation_session
            WHERE user_id = :user_id
              AND session_status = 'SUCCESS'
            ORDER BY created_at DESC, session_id DESC
            """
        ),
        {"user_id": current_user["user_id"]},
    ).mappings().all()

    results = []
    for session in sessions:
        session_id = int(session["session_id"])
        session_obj = get_recommendation_session_or_404(db, session_id, current_user["user_id"])
        session_data = serialize_recommendation_session(db, session_obj)
        saved_at = session["created_at"].isoformat() if hasattr(session["created_at"], "isoformat") else str(session["created_at"])

        for routine in session_data["routines"]:
            routine_obj = next(
                (row for row in session_obj.routines if str(row.routine_id) == routine["routine_id"]),
                None,
            )
            product_map = {}
            if routine_obj:
                for item in routine_obj.items:
                    if item.product:
                        product_map[item.product.product_id] = item.product

            detailed_products = []
            for item in routine["products"]:
                product = product_map.get(item["product_id"])
                if not product:
                    continue
                product_data = serialize_product_list_item(db, product)
                detailed_products.append(
                    {
                        "product_id": item["product_id"],
                        "step": item["step"],
                        "product_name": product_data["product_name"],
                        "brand_name": product_data["brand_name"],
                        "category": product_data["category"],
                        "price": int(product_data["price"] or 0),
                        "image_url": product_data["image_url"],
                    }
                )

            results.append(
                {
                    "routine_id": int(routine["routine_id"]),
                    "session_id": session_id,
                    "result_id": int(session["result_id"] or 0),
                    "routine_type": routine["type"],
                    "label": routine["label"],
                    "routine_time": routine["routine_time"],
                    "total_cost": routine["total_cost"],
                    "duration": routine["duration"],
                    "saved_at": saved_at,
                    "products": detailed_products,
                }
            )

    return {"items": results}


@router.get("/me/skin-analysis")
def get_my_skin_analysis(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    profile = ensure_profile(db, current_user["user_id"])
    rows = db.execute(
        text(
            """
            SELECT
                sar.result_id,
                sar.image_id,
                sar.analyzed_at,
                sar.acne_score,
                sar.dryness_score,
                sar.sagging_score,
                sar.pore_score,
                sar.pigmentation_score,
                sar.wrinkle_score,
                ui.storage_url AS image_url,
                ui.s3_key AS s3_key
            FROM skin_analysis_result sar
            LEFT JOIN user_image ui
              ON ui.image_id = sar.image_id
            WHERE sar.user_id = :user_id
            ORDER BY sar.analyzed_at DESC, sar.result_id DESC
            """
        ),
        {"user_id": current_user["user_id"]},
    ).mappings().all()

    results = []
    for row in rows:
        summary = store.skin_summaries.get(int(row["result_id"]), {})
        analyzed_at = row["analyzed_at"].isoformat() if hasattr(row["analyzed_at"], "isoformat") else str(row["analyzed_at"])
        results.append(
            {
                "result_id": int(row["result_id"]),
                "image_id": int(row["image_id"]),
                "analyzed_at": analyzed_at,
                "skin_type": profile.skin_type if profile else None,
                "image_url": resolve_image_display_url(row["image_url"], row["s3_key"]),
                "display_scores": {
                    "acne": float(row["acne_score"] or 0),
                    "dryness": float(row["dryness_score"] or 0),
                    "sagging": float(row["sagging_score"] or 0),
                    "pore": float(row["pore_score"] or 0),
                    "pigmentation": float(row["pigmentation_score"] or 0),
                    "wrinkle": float(row["wrinkle_score"] or 0),
                },
                "ai_comment": summary.get("summary_comment", "아직 분석 요약이 준비되지 않았습니다."),
            }
        )

    for row in store.skin_results.values():
        if row["user_id"] != current_user["user_id"]:
            continue
        if any(item["result_id"] == row["result_id"] for item in results):
            continue
        image = get_user_image(db, row["image_id"])
        summary = store.skin_summaries.get(row["result_id"], {})
        results.append(
            {
                "result_id": row["result_id"],
                "image_id": row["image_id"],
                "analyzed_at": row["analyzed_at"],
                "skin_type": profile.skin_type if profile else None,
                "image_url": resolve_image_display_url(image.storage_url, image.s3_key) if image else None,
                "display_scores": row.get("display_scores", {}),
                "ai_comment": summary.get("summary_comment", "아직 분석 요약이 준비되지 않았습니다."),
                "display_scores": row.get("display_scores", {}),
            }
        )

    results.sort(key=lambda item: item["analyzed_at"], reverse=True)
    return {"items": results}
