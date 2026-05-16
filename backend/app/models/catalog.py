from __future__ import annotations

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Product(Base):
    __tablename__ = "product"

    product_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brand_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    brand_name_kor: Mapped[str | None] = mapped_column(String(50), nullable=True)
    product_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    product_name_kor: Mapped[str | None] = mapped_column(String(100), nullable=True)
    category: Mapped[str | None] = mapped_column(String(30), nullable=True)
    function: Mapped[str | None] = mapped_column(String(20), nullable=True)
    ranking: Mapped[int | None] = mapped_column(Integer, nullable=True)
    price: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sim_1: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sim_2: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sim_3: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sim_4: Mapped[int | None] = mapped_column(Integer, nullable=True)

    ingredients: Mapped[list[ProductIngredient]] = relationship(back_populates="product")
    review: Mapped[ProductReview | None] = relationship(back_populates="product", uselist=False)


class Ingredient(Base):
    __tablename__ = "ingredient"

    ingredient_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingredient_name: Mapped[str] = mapped_column(String(200), nullable=False)
    function: Mapped[str | None] = mapped_column(Text, nullable=True)
    rating: Mapped[str | None] = mapped_column(String(10), nullable=True)
    irritation: Mapped[str | None] = mapped_column(String(20), nullable=True)
    comedogenicity: Mapped[str | None] = mapped_column(String(20), nullable=True)
    cas_no: Mapped[str | None] = mapped_column(String(100), nullable=True)
    ec_no: Mapped[str | None] = mapped_column(String(100), nullable=True)
    allergy_category: Mapped[str | None] = mapped_column(String(30), nullable=True)


class ProductIngredient(Base):
    __tablename__ = "product_ingredient"

    product_ingredient_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("product.product_id"), nullable=False)
    ingredient_id: Mapped[int] = mapped_column(ForeignKey("ingredient.ingredient_id"), nullable=False)
    inci_brand: Mapped[str | None] = mapped_column(String(100), nullable=True)
    inci_product_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    function: Mapped[str | None] = mapped_column(String(200), nullable=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    product: Mapped[Product] = relationship(back_populates="ingredients")
    ingredient: Mapped[Ingredient] = relationship()


class ProductReview(Base):
    __tablename__ = "product_review"

    review_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("product.product_id"), nullable=False, unique=True)
    pros_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    cons_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    product: Mapped[Product] = relationship(back_populates="review")


class UserWishlist(Base):
    __tablename__ = "user_wishlist"

    wishlist_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.user_id"), nullable=False)
    product_id: Mapped[int] = mapped_column(ForeignKey("product.product_id"), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())

    product: Mapped[Product] = relationship()
