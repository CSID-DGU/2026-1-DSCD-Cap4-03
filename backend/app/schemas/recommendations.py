from pydantic import BaseModel, Field, model_validator


class RecommendationRequest(BaseModel):
    result_id: int
    image_id: int
    total_budget_min: int | None = Field(default=None, ge=0)
    total_budget_max: int | None = Field(default=None, ge=0)
    toner_budget_min: int | None = Field(default=None, ge=0)
    toner_budget_max: int | None = Field(default=None, ge=0)
    emulsion_budget_min: int | None = Field(default=None, ge=0)
    emulsion_budget_max: int | None = Field(default=None, ge=0)
    ampoule_budget_min: int | None = Field(default=None, ge=0)
    ampoule_budget_max: int | None = Field(default=None, ge=0)
    cream_budget_min: int | None = Field(default=None, ge=0)
    cream_budget_max: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_budget_ranges(self) -> "RecommendationRequest":
        pairs = (
            ("total_budget_min", "total_budget_max"),
            ("toner_budget_min", "toner_budget_max"),
            ("emulsion_budget_min", "emulsion_budget_max"),
            ("ampoule_budget_min", "ampoule_budget_max"),
            ("cream_budget_min", "cream_budget_max"),
        )
        for min_field, max_field in pairs:
            min_value = getattr(self, min_field)
            max_value = getattr(self, max_field)
            if min_value is not None and max_value is not None and min_value > max_value:
                raise ValueError(f"{min_field} must be less than or equal to {max_field}")
        return self


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
    total_budget_min: int | None = None
    total_budget_max: int | None = None
    toner_budget_min: int | None = None
    toner_budget_max: int | None = None
    emulsion_budget_min: int | None = None
    emulsion_budget_max: int | None = None
    ampoule_budget_min: int | None = None
    ampoule_budget_max: int | None = None
    cream_budget_min: int | None = None
    cream_budget_max: int | None = None
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
