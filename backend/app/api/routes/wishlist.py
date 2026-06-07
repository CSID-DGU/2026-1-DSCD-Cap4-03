from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.db_catalog import add_wishlist_product, delete_wishlist_product
from app.services.deps import get_current_user


router = APIRouter()


@router.post("/wishlist/{product_id}", status_code=201)
def add_wishlist(
    product_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return add_wishlist_product(db, current_user["user_id"], product_id)


@router.delete("/wishlist/{product_id}")
def delete_wishlist(
    product_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return delete_wishlist_product(db, current_user["user_id"], product_id)
