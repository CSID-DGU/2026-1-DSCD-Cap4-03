from __future__ import annotations

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class User(Base):
    __tablename__ = "user"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_name: Mapped[str | None] = mapped_column(String(10), nullable=True)
    login_type: Mapped[str] = mapped_column(String(10), nullable=False)
    email: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    nickname: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="1")

    profile: Mapped[UserProfile | None] = relationship(back_populates="user", uselist=False)
    images: Mapped[list[UserImage]] = relationship(back_populates="user")
    allergies: Mapped[list[UserAllergy]] = relationship(back_populates="user")


class UserProfile(Base):
    __tablename__ = "user_profile"

    user_id: Mapped[int] = mapped_column(ForeignKey("user.user_id"), primary_key=True)
    gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    birth: Mapped[Date | None] = mapped_column(Date, nullable=True)
    skin_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    skin_concern: Mapped[str | None] = mapped_column(String(100), nullable=True)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    user: Mapped[User] = relationship(back_populates="profile")


class UserAllergy(Base):
    __tablename__ = "user_allergy"

    allergy_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.user_id"), nullable=False)
    allergy_category: Mapped[str | None] = mapped_column(String(30), nullable=True)
    allergy_ingredient: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())

    user: Mapped[User] = relationship(back_populates="allergies")


class UserImage(Base):
    __tablename__ = "user_image"

    image_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.user_id"), nullable=False)
    storage_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    uploaded_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False, server_default=func.current_timestamp())
    s3_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    original_file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    crop_data: Mapped[str | None] = mapped_column(Text, nullable=True)
    upload_status: Mapped[str | None] = mapped_column(String(30), nullable=True)

    user: Mapped[User] = relationship(back_populates="images")
