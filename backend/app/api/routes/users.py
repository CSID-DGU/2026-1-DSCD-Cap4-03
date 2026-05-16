from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.memory import store
from app.db.session import get_db
from app.schemas.users import UpdateAllergiesRequest, UpdateAllergiesResponse, UpdateProfileRequest, UserProfileResponse
from app.services.db_catalog import list_wishlist_products
from app.services.db_user import (
    ensure_profile,
    get_user_by_id,
    get_user_image,
    replace_user_allergies,
    serialize_user_profile,
    update_user_profile,
)
from app.services.deps import get_current_user


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


@router.get("/me/wishlist")
def get_my_wishlist(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    return {"items": list_wishlist_products(db, current_user["user_id"])}


@router.get("/me/routines")
def get_my_routines(current_user: dict = Depends(get_current_user)) -> dict:
    routines = [row for row in store.saved_routines.values() if row["user_id"] == current_user["user_id"]]
    return {"items": routines}


@router.get("/me/skin-analysis")
def get_my_skin_analysis(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    profile = ensure_profile(db, current_user["user_id"])
    results = []
    for row in store.skin_results.values():
        if row["user_id"] != current_user["user_id"]:
            continue
        image = get_user_image(db, row["image_id"])
        summary = store.skin_summaries.get(row["result_id"], {})
        results.append(
            {
                "result_id": row["result_id"],
                "image_id": row["image_id"],
                "analyzed_at": row["analyzed_at"],
                "skin_type": profile.skin_type if profile else None,
                "image_url": image.storage_url if image else None,
                "ai_comment": summary.get("summary_comment", "아직 분석 요약이 준비되지 않았습니다."),
            }
        )
    results.sort(key=lambda item: item["analyzed_at"], reverse=True)
    return {"items": results}
