from __future__ import annotations

from decimal import Decimal

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.catalog import Product


class RecommendationSession(Base):
    __tablename__ = "recommendation_session"

    session_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.user_id"), nullable=False)
    image_id: Mapped[int | None] = mapped_column(ForeignKey("user_image.image_id"), nullable=True)
    result_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strict_budget: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0")
    total_budget_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_budget_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    slot_budget_min_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    slot_budget_max_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    budget_check_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="1")
    session_status: Mapped[str] = mapped_column(String(30), nullable=False, server_default="SUCCESS")
    failure_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())

    routines: Mapped[list[RecommendationRoutine]] = relationship(back_populates="session")


class RecommendationRoutine(Base):
    __tablename__ = "recommendation_routine"

    routine_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("recommendation_session.session_id"), nullable=False)
    routine_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    routine_label: Mapped[str | None] = mapped_column(String(20), nullable=True)
    ampm_mode: Mapped[str | None] = mapped_column(String(10), nullable=True)
    routine_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    has_conflict: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="0")
    conflict_pairs: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())

    session: Mapped[RecommendationSession] = relationship(back_populates="routines")
    items: Mapped[list[RecommendationItem]] = relationship(back_populates="routine")


class RecommendationItem(Base):
    __tablename__ = "recommendation_item"

    routine_item_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    routine_id: Mapped[int] = mapped_column(ForeignKey("recommendation_routine.routine_id"), nullable=False)
    slot_order: Mapped[int] = mapped_column(Integer, nullable=False)
    category: Mapped[str | None] = mapped_column(String(30), nullable=True)
    product_id: Mapped[int | None] = mapped_column(ForeignKey("product.product_id"), nullable=True)
    product_score: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)

    routine: Mapped[RecommendationRoutine] = relationship(back_populates="items")
    product: Mapped[Product | None] = relationship()
