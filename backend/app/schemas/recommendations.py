from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    result_id: int
    image_id: int
    total_budget: int | None = Field(default=None, ge=0)
    toner_budget: int | None = Field(default=None, ge=0)
    emulsion_budget: int | None = Field(default=None, ge=0)
    ampoule_budget: int | None = Field(default=None, ge=0)
    cream_budget: int | None = Field(default=None, ge=0)


class RecommendationProduct(BaseModel):
    product_id: int
    step: int
    application_guide: str
    time_tag: str | None = None


class RecommendationRoutine(BaseModel):
    routine_id: str
    type: str
    label: str
    routine_time: str
    total_cost: int
    duration: int
    ai_description: str
    products: list[RecommendationProduct]


class RecommendationResponse(BaseModel):
    session_id: int
    user_id: int
    result_id: int
    session_status: str
    routines: list[RecommendationRoutine]


class RecommendationExplanationRequest(BaseModel):
    session_id: int


class RecommendationExplanationResponse(BaseModel):
    session_id: int
    llm_model: str
    prompt_version: str
    summary_text: str
    usage_guide_text: str
    warning_text: str
