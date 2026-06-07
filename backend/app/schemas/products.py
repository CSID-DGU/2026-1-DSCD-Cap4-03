from pydantic import BaseModel, Field


class ProductListResponse(BaseModel):
    product_id: int
    brand_name: str
    product_name: str
    category: str
    price: int
    image_url: str
    tags: list[str] = Field(default_factory=list)


class ProductDetailResponse(ProductListResponse):
    ingredients: list[str] = Field(default_factory=list)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    how_to_use: str
    apply_time: str
