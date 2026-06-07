from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.products import ProductDetailResponse, ProductListResponse
from app.services.db_catalog import get_product_or_404, list_products, serialize_product_detail


router = APIRouter()


@router.get("/products", response_model=list[ProductListResponse])
def get_products(
    category: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[ProductListResponse]:
    return [ProductListResponse(**product) for product in list_products(db, category)]


@router.get("/products/{product_id}", response_model=ProductDetailResponse)
def get_product(product_id: int, db: Session = Depends(get_db)) -> ProductDetailResponse:
    product = get_product_or_404(db, product_id)
    return ProductDetailResponse(**serialize_product_detail(db, product))
