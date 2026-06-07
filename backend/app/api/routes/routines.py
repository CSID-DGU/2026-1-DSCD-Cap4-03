from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.memory import store
from app.db.session import get_db
from app.schemas.routines import SaveRoutineRequest, SaveRoutineResponse
from app.services.db_catalog import serialize_product_list_item
from app.services.deps import get_current_user
from app.services.recommendation_model import get_recommendation_session_or_404, serialize_recommendation_session


router = APIRouter()


@router.post("/routines/{session_id}/save", response_model=SaveRoutineResponse, status_code=201)
def save_routine(
    session_id: int,
    payload: SaveRoutineRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SaveRoutineResponse:
    session = get_recommendation_session_or_404(db, session_id, current_user["user_id"])
    session_data = serialize_recommendation_session(db, session)

    routine = next((item for item in session_data["routines"] if item["type"] == payload.routine_type), None)
    if routine is None:
        raise HTTPException(status_code=404, detail="Routine not found")

    routine_product_map = {}
    for db_routine in session.routines:
        if str(db_routine.routine_id) != routine["routine_id"]:
            continue
        for item in db_routine.items:
            if item.product:
                routine_product_map[item.product.product_id] = serialize_product_list_item(db, item.product)

    saved_routine_id = next(store.saved_routine_seq)
    saved_at = datetime.now(UTC).isoformat()
    detailed_products = []
    for item in routine["products"]:
        product = routine_product_map.get(item["product_id"])
        if product is None:
            continue
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
