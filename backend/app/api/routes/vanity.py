from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.vanity import (
    VanityProductCreateRequest,
    VanityProductMutationResponse,
    VanityProductsResponse,
    VanityRoutineListResponse,
    VanityRoutineRequest,
    VanityRoutineResponse,
    VanitySkinMatchRequest,
    VanitySkinMatchResponse,
    VanitySummaryResponse,
)
from app.services.deps import get_current_user
from app.services.vanity_service import (
    add_vanity_product,
    delete_vanity_product,
    get_latest_skin_match,
    get_latest_vanity_routine,
    get_vanity_summary,
    get_vanity_routine_detail,
    list_vanity_products,
    list_vanity_routines,
    run_skin_match,
    run_vanity_routine,
)


router = APIRouter()


@router.get("/vanity/summary", response_model=VanitySummaryResponse)
def get_my_vanity_summary(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> VanitySummaryResponse:
    return VanitySummaryResponse(**get_vanity_summary(db, current_user["user_id"]))


@router.get("/vanity/products", response_model=VanityProductsResponse)
def get_my_vanity_products(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> VanityProductsResponse:
    return VanityProductsResponse(products=list_vanity_products(db, current_user["user_id"]))


@router.post("/vanity/products", response_model=VanityProductMutationResponse, status_code=status.HTTP_201_CREATED)
def create_my_vanity_product(
    payload: VanityProductCreateRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> VanityProductMutationResponse:
    return VanityProductMutationResponse(**add_vanity_product(db, current_user["user_id"], payload.product_id))


@router.delete("/vanity/products/{product_id}", response_model=VanityProductMutationResponse)
def remove_my_vanity_product(
    product_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> VanityProductMutationResponse:
    return VanityProductMutationResponse(**delete_vanity_product(db, current_user["user_id"], product_id))


@router.post("/vanity/skin-match", response_model=VanitySkinMatchResponse, status_code=status.HTTP_201_CREATED)
def create_vanity_skin_match(
    payload: VanitySkinMatchRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> VanitySkinMatchResponse:
    return VanitySkinMatchResponse(**run_skin_match(db, current_user["user_id"], payload.product_ids))


@router.get("/vanity/skin-match/latest")
def get_latest_vanity_skin_match(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return get_latest_skin_match(db, current_user["user_id"])


@router.post("/vanity/routines", response_model=VanityRoutineResponse, status_code=status.HTTP_201_CREATED)
def create_vanity_routine(
    payload: VanityRoutineRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> VanityRoutineResponse:
    return VanityRoutineResponse(
        **run_vanity_routine(
            db,
            current_user["user_id"],
            fixed_product_ids=payload.fixed_product_ids,
            budget_payload=payload,
        )
    )


@router.get("/vanity/routines/latest")
def get_latest_my_vanity_routine(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return get_latest_vanity_routine(db, current_user["user_id"])
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        if detail.get("error_code") != "VANITY_ROUTINE_NOT_FOUND":
            raise
        return {
            "recommendation_session_id": None,
            "created_at": None,
            "basis_skin_result": None,
            "routine_recommendation_results": {
                "fixed_products": [],
                "recommended_products": [],
                "final_routine": [],
                "warnings": [],
                "total_price": None,
            },
            "llm_explanation": None,
            "message": "아직 생성된 내 화장대 루틴이 없습니다.",
        }


@router.get("/vanity/routines", response_model=VanityRoutineListResponse)
def get_my_vanity_routines(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> VanityRoutineListResponse:
    return VanityRoutineListResponse(routines=list_vanity_routines(db, current_user["user_id"]))


@router.get("/vanity/routines/{recommendation_session_id}")
def get_my_vanity_routine_detail(
    recommendation_session_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    return get_vanity_routine_detail(db, current_user["user_id"], recommendation_session_id)
