from pydantic import BaseModel, Field


class UserProfileResponse(BaseModel):
    user_id: int
    email: str
    user_name: str | None = None
    nickname: str
    login_type: str
    gender: str | None = None
    birth: str | None = None
    skin_type: str | None = None
    skin_concerns: list[str] = Field(default_factory=list)


class UpdateProfileRequest(BaseModel):
    nickname: str | None = Field(default=None, min_length=2, max_length=20)
    user_name: str | None = Field(default=None, max_length=10)
    gender: str | None = Field(default=None, max_length=20)
    birth: str | None = None
    skin_type: str | None = Field(default=None, max_length=30)
    skin_concerns: list[str] | None = None


class AllergyItem(BaseModel):
    category: str | None = Field(default=None, max_length=30)
    ingredient_id: int


class UpdateAllergiesRequest(BaseModel):
    allergy_items: list[AllergyItem] | None = None
    allergy_categories: list[str] = Field(default_factory=list)
    allergy_ingredient_ids: list[int] = Field(default_factory=list)


class UpdateAllergiesResponse(BaseModel):
    user_id: int
    allergy_categories: list[str]
    allergy_ingredient_ids: list[int]
    saved_count: int
