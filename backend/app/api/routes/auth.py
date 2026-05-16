from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.auth import AuthResponse, LoginRequest, SignupRequest
from app.services.auth import build_token
from app.services.db_user import authenticate_user, create_user


router = APIRouter()


@router.post("/signup", response_model=AuthResponse, status_code=201)
def signup(payload: SignupRequest, db: Session = Depends(get_db)) -> AuthResponse:
    user = create_user(
        db,
        email=str(payload.email),
        password=payload.password,
        nickname=payload.nickname,
        user_name=payload.user_name,
        login_type=payload.login_type,
    )
    return AuthResponse(access_token=build_token(user.user_id), user_id=user.user_id, nickname=user.nickname)


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    user = authenticate_user(db, email=str(payload.email), password=payload.password)
    return AuthResponse(
        access_token=build_token(user.user_id),
        user_id=user.user_id,
        nickname=user.nickname,
    )
