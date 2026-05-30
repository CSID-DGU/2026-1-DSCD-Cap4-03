from __future__ import annotations

from datetime import date, datetime

from fastapi import HTTPException, status
from sqlalchemy import delete, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.models import Ingredient, User, UserAllergy, UserImage, UserProfile
from app.schemas.users import UpdateAllergiesRequest, UpdateProfileRequest
from app.services.auth import hash_password


def invalidate_recommendation_rerank_cache(db: Session, user_id: int) -> None:
    try:
        db.execute(
            text("DELETE FROM recommendation_reranked WHERE user_id = :user_id"),
            {"user_id": user_id},
        )
        db.commit()
    except DBAPIError as exc:
        db.rollback()
        err_code = exc.orig.args[0] if getattr(exc, "orig", None) and getattr(exc.orig, "args", None) else None
        if err_code == 1146:
            return
        raise


def create_user(
    db: Session,
    *,
    email: str,
    password: str,
    nickname: str,
    user_name: str | None,
    login_type: str,
) -> User:
    existing = db.scalar(select(User).where(User.email == email))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")

    user = User(
        email=email,
        password=hash_password(password),
        nickname=nickname,
        user_name=user_name,
        login_type=login_type,
        is_active=True,
    )
    db.add(user)
    db.flush()

    profile = UserProfile(user_id=user.user_id)
    db.add(profile)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, *, email: str, password: str) -> User:
    user = db.scalar(select(User).where(User.email == email))
    if not user or user.password != hash_password(password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    return user


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.scalar(select(User).where(User.user_id == user_id))


def ensure_profile(db: Session, user_id: int) -> UserProfile:
    profile = db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
    if profile is None:
        profile = UserProfile(user_id=user_id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile


def serialize_user_profile(user: User, profile: UserProfile | None) -> dict:
    concerns: list[str] = []
    if profile and profile.skin_concern:
        concerns = [item for item in profile.skin_concern.split(",") if item]
    birth = None
    if profile and profile.birth:
        birth = profile.birth.isoformat()
    return {
        "user_id": user.user_id,
        "email": user.email,
        "user_name": user.user_name,
        "nickname": user.nickname,
        "login_type": user.login_type,
        "gender": profile.gender if profile else None,
        "birth": birth,
        "skin_type": profile.skin_type if profile else None,
        "skin_concerns": concerns,
    }


def update_user_profile(db: Session, user: User, payload: UpdateProfileRequest) -> dict:
    profile = ensure_profile(db, user.user_id)

    if payload.nickname is not None:
        user.nickname = payload.nickname
    if payload.user_name is not None:
        user.user_name = payload.user_name
    if payload.gender is not None:
        profile.gender = payload.gender
    if payload.birth is not None:
        profile.birth = date.fromisoformat(payload.birth) if payload.birth else None
    if payload.skin_type is not None:
        profile.skin_type = payload.skin_type
    if payload.skin_concerns is not None:
        profile.skin_concern = ",".join(payload.skin_concerns)

    db.commit()
    db.refresh(user)
    db.refresh(profile)
    invalidate_recommendation_rerank_cache(db, user.user_id)
    return serialize_user_profile(user, profile)


def replace_user_allergies(db: Session, user_id: int, payload: UpdateAllergiesRequest) -> dict:
    db.execute(delete(UserAllergy).where(UserAllergy.user_id == user_id))

    if payload.allergy_items is not None:
        raw_items = [
            {
                "category": item.category,
                "ingredient_id": item.ingredient_id,
            }
            for item in payload.allergy_items
        ]
    else:
        fallback_category = payload.allergy_categories[0] if len(payload.allergy_categories) == 1 else None
        raw_items = [
            {
                "category": fallback_category,
                "ingredient_id": ingredient_id,
            }
            for ingredient_id in payload.allergy_ingredient_ids
        ]

    deduped_items: list[dict] = []
    seen_ingredient_ids: set[int] = set()
    for item in raw_items:
        ingredient_id = int(item["ingredient_id"])
        if ingredient_id in seen_ingredient_ids:
            continue
        seen_ingredient_ids.add(ingredient_id)
        deduped_items.append({**item, "ingredient_id": ingredient_id})

    ingredient_ids = [item["ingredient_id"] for item in deduped_items]
    ingredients = []
    if ingredient_ids:
        ingredients = list(
            db.scalars(
                select(Ingredient).where(Ingredient.ingredient_id.in_(ingredient_ids))
            )
        )

    ingredient_by_id = {ingredient.ingredient_id: ingredient for ingredient in ingredients}

    for item in deduped_items:
        ingredient_id = item["ingredient_id"]
        ingredient = ingredient_by_id.get(ingredient_id)
        category = item.get("category") or (ingredient.allergy_category if ingredient else None)
        db.add(
            UserAllergy(
                user_id=user_id,
                allergy_category=category,
                allergy_ingredient=str(ingredient_id),
            )
        )

    db.commit()
    invalidate_recommendation_rerank_cache(db, user_id)
    saved_categories = [
        item.get("category") or (ingredient_by_id.get(item["ingredient_id"]).allergy_category if ingredient_by_id.get(item["ingredient_id"]) else None)
        for item in deduped_items
    ]
    return {
        "user_id": user_id,
        "allergy_categories": [category for category in dict.fromkeys(saved_categories) if category],
        "allergy_ingredient_ids": ingredient_ids,
        "saved_count": len(ingredient_ids),
    }


def get_user_allergies(db: Session, user_id: int) -> dict:
    rows = list(
        db.scalars(
            select(UserAllergy)
            .where(UserAllergy.user_id == user_id)
            .order_by(UserAllergy.allergy_id.asc())
        )
    )

    categories = []
    ingredient_ids = []
    for row in rows:
        if row.allergy_category:
            categories.append(row.allergy_category)
        if row.allergy_ingredient:
            try:
                ingredient_ids.append(int(row.allergy_ingredient))
            except ValueError:
                continue

    return {
        "user_id": user_id,
        "allergy_categories": list(dict.fromkeys(categories)),
        "allergy_ingredient_ids": list(dict.fromkeys(ingredient_ids)),
        "saved_count": len(set(ingredient_ids)),
    }


def create_user_image(db: Session, user_id: int, payload) -> UserImage:
    image = UserImage(
        user_id=user_id,
        storage_url=payload.storage_url,
        s3_key=payload.s3_key,
        original_file_name=payload.original_file_name,
        mime_type=payload.mime_type,
        file_size=payload.file_size,
        crop_data=str(payload.crop_data) if payload.crop_data is not None else None,
        upload_status=payload.upload_status,
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return image


def get_user_image(db: Session, image_id: int) -> UserImage | None:
    return db.scalar(select(UserImage).where(UserImage.image_id == image_id))


def serialize_user_image(image: UserImage) -> dict:
    uploaded_at = image.uploaded_at.isoformat() if isinstance(image.uploaded_at, datetime) else str(image.uploaded_at)
    return {
        "image_id": image.image_id,
        "user_id": image.user_id,
        "storage_url": image.storage_url,
        "s3_key": image.s3_key,
        "uploaded_at": uploaded_at,
    }
