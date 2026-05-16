from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException

from app.db.memory import store
from app.schemas.routines import SaveRoutineRequest, SaveRoutineResponse
from app.services.deps import get_current_user


router = APIRouter()


@router.post("/routines/{session_id}/save", response_model=SaveRoutineResponse, status_code=201)
def save_routine(
    session_id: int,
    payload: SaveRoutineRequest,
    current_user: dict = Depends(get_current_user),
) -> SaveRoutineResponse:
    session = store.recommendation_sessions.get(session_id)
    if not session or session["user_id"] != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Recommendation session not found")

    routine = next((item for item in session["routines"] if item["type"] == payload.routine_type), None)
    if routine is None:
        raise HTTPException(status_code=404, detail="Routine not found")

    saved_routine_id = next(store.saved_routine_seq)
    saved_at = datetime.now(UTC).isoformat()
    detailed_products = []
    for item in routine["products"]:
        product = store.products[item["product_id"]]
        detailed_products.append(
            {
                "product_id": item["product_id"],
                "step": item["step"],
                "product_name": product["product_name"],
                "brand_name": product["brand_name"],
                "category": product["category"],
                "price": product["price"],
                "image_url": product["image_url"],
            }
        )
    row = {
        "saved_routine_id": saved_routine_id,
        "session_id": session_id,
        "user_id": current_user["user_id"],
        "routine_type": routine["type"],
        "label": routine["label"],
        "routine_time": routine["routine_time"],
        "total_cost": routine["total_cost"],
        "duration": routine["duration"],
        "saved_at": saved_at,
        "products": detailed_products,
    }
    store.saved_routines[saved_routine_id] = row
    return SaveRoutineResponse(
        saved_routine_id=saved_routine_id,
        session_id=session_id,
        routine_type=routine["type"],
        saved_at=saved_at,
    )
