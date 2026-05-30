from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, joinedload

from app.models import Ingredient, Product, ProductIngredient, UserWishlist
from app.services.db_user import invalidate_recommendation_rerank_cache


def _split_pipe_text(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace("\r", "").split("|") if item.strip()]


def _usage_for_category(category: str | None) -> tuple[str, str]:
    mapping = {
        "Toner": ("세안 후 첫 단계에서 피부결을 따라 가볍게 흡수시켜 주세요.", "both"),
        "Emulsions": ("토너 다음 단계에서 얼굴 전체에 부드럽게 펴 발라 주세요.", "both"),
        "Essences/Ampoules/Serums": ("토너 다음 단계에서 적당량을 덜어 고민 부위 중심으로 흡수시켜 주세요.", "both"),
        "Cream/Gel": ("스킨케어 마지막 단계에서 보습막을 만들듯 마무리해 주세요.", "both"),
        "Face Mists": ("건조함이 느껴질 때 얼굴에서 거리를 두고 가볍게 분사해 주세요.", "both"),
    }
    return mapping.get(category or "", ("제품 특성에 맞춰 적당량을 얼굴에 펴 발라 주세요.", "both"))


def _image_url_for_product(db: Session, product_id: int) -> str:
    image_url = db.scalar(
        select(ProductIngredient.image_url)
        .where(ProductIngredient.product_id == product_id, ProductIngredient.image_url.is_not(None))
        .order_by(ProductIngredient.product_ingredient_id)
        .limit(1)
    )
    return image_url or ""


def serialize_product_list_item(db: Session, product: Product) -> dict:
    return {
        "product_id": product.product_id,
        "brand_name": product.brand_name_kor or product.brand_name or "",
        "product_name": product.product_name_kor or product.product_name or "",
        "category": product.category or "",
        "price": product.price or 0,
        "image_url": _image_url_for_product(db, product.product_id),
        "tags": [tag for tag in [product.category, product.function] if tag],
    }


def list_products(db: Session, category: str | None = None) -> list[dict]:
    stmt = select(Product).order_by(Product.category, Product.ranking, Product.product_id)
    if category and category not in {"전체", "all"}:
        stmt = stmt.where(Product.category == category)
    products = db.scalars(stmt).all()
    return [serialize_product_list_item(db, product) for product in products]


def get_product_or_404(db: Session, product_id: int) -> Product:
    product = db.scalar(
        select(Product)
        .options(joinedload(Product.review))
        .where(Product.product_id == product_id)
    )
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    return product


def serialize_product_detail(db: Session, product: Product) -> dict:
    usage, apply_time = _usage_for_category(product.category)
    ingredient_names = db.scalars(
        select(Ingredient.ingredient_name)
        .join(ProductIngredient, ProductIngredient.ingredient_id == Ingredient.ingredient_id)
        .where(ProductIngredient.product_id == product.product_id)
        .order_by(ProductIngredient.product_ingredient_id)
    ).all()

    base = serialize_product_list_item(db, product)
    base.update(
        {
            "ingredients": ingredient_names,
            "pros": _split_pipe_text(product.review.pros_text if product.review else None),
            "cons": _split_pipe_text(product.review.cons_text if product.review else None),
            "how_to_use": usage,
            "apply_time": apply_time,
        }
    )
    return base


def add_wishlist_product(db: Session, user_id: int, product_id: int) -> dict:
    get_product_or_404(db, product_id)
    exists = db.scalar(
        select(UserWishlist).where(UserWishlist.user_id == user_id, UserWishlist.product_id == product_id)
    )
    if exists is None:
        db.add(UserWishlist(user_id=user_id, product_id=product_id))
        db.commit()
        invalidate_recommendation_rerank_cache(db, user_id)
    return {"product_id": product_id, "saved": True}


def delete_wishlist_product(db: Session, user_id: int, product_id: int) -> dict:
    db.execute(delete(UserWishlist).where(UserWishlist.user_id == user_id, UserWishlist.product_id == product_id))
    db.commit()
    invalidate_recommendation_rerank_cache(db, user_id)
    return {"product_id": product_id, "saved": False}


def list_wishlist_products(db: Session, user_id: int) -> list[dict]:
    products = db.scalars(
        select(Product)
        .join(UserWishlist, UserWishlist.product_id == Product.product_id)
        .where(UserWishlist.user_id == user_id)
        .order_by(UserWishlist.created_at.desc())
    ).all()
    return [serialize_product_list_item(db, product) for product in products]


def count_products(db: Session) -> int:
    return db.scalar(select(func.count()).select_from(Product)) or 0
