from pydantic import BaseModel


class SaveRoutineRequest(BaseModel):
    routine_type: str


class SaveRoutineResponse(BaseModel):
    saved_routine_id: int
    session_id: int
    routine_type: str
    saved_at: str


class SavedRoutineProduct(BaseModel):
    product_id: int
    step: int
    product_name: str
    brand_name: str
    category: str
    price: int
    image_url: str


class SavedRoutineItem(BaseModel):
    saved_routine_id: int
    session_id: int
    routine_type: str
    label: str
    routine_time: str
    total_cost: int
    duration: int
    saved_at: str
    products: list[SavedRoutineProduct]
