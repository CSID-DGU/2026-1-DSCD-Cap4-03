from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=255)
    nickname: str = Field(min_length=2, max_length=20)
    user_name: str | None = Field(default=None, max_length=10)
    login_type: str = Field(default="local", max_length=10)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    nickname: str
