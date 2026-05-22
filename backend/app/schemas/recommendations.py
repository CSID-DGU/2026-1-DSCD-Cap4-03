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
    time_tag: str | None = None


class RecommendationRoutine(BaseModel):
    routine_id: str
    type: str
    label: str
    routine_time: str
    total_cost: int
    duration: int
    products: list[RecommendationProduct]


class RecommendationResponse(BaseModel):
    session_id: int
    user_id: int
    result_id: int
    session_status: str
    budget_check_passed: bool = True
    budget_fallback_applied: bool = False
    budget_message: str | None = None
    total_budget: int | None = None
    toner_budget: int | None = None
    emulsion_budget: int | None = None
    ampoule_budget: int | None = None
    cream_budget: int | None = None
    routines: list[RecommendationRoutine]


class RecommendationExplanationRequest(BaseModel):
    session_id: int


class RecommendationStepGuide(BaseModel):
    slot_order: int
    category: str
    usage_guide: str


class RecommendationExplanationRoutine(BaseModel):
    routine_id: int | str
    routine_type: str
    routine_rank: int
    ampm_mode: str
    recommend_summary: str
    ampm_comment: str
    step_guides: list[RecommendationStepGuide]
    strengths: list[str]
    cautions: list[str]


class RecommendationExplanationResponse(BaseModel):
    session_id: int
    llm_model: str
    prompt_version: str
    routines: list[RecommendationExplanationRoutine]
