from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class VanityProductCreateRequest(BaseModel):
    product_id: int


class VanityProductItem(BaseModel):
    vanity_id: int | None = None
    product_id: int
    category: str | None = None
    brand_name: str
    product_name: str
    price: int = 0
    image_url: str = ""
    created_at: str | None = None


class VanityProductsResponse(BaseModel):
    products: list[VanityProductItem]


class VanityProductMutationResponse(BaseModel):
    vanity_id: int | None = None
    product_id: int
    saved: bool = True
    message: str


class VanitySkinMatchRequest(BaseModel):
    product_ids: list[int] | None = None


class BasisSkinResult(BaseModel):
    result_id: int
    image_id: int | None = None
    analyzed_at: str | None = None
    main_concerns: list[str] = Field(default_factory=list)


class ProductMatchScores(BaseModel):
    concern_match_score: float = 0.0
    skin_type_bonus: float = 0.0
    review_score: float = 0.0
    irritation_penalty: float = 0.0
    vanity_fit_score: float = 0.0


class ProductMatchResult(BaseModel):
    product_id: int
    category: str | None = None
    brand_name: str | None = None
    product_name: str | None = None
    vanity_fit_score: float
    scores: ProductMatchScores
    fit_label: str
    display_label: str
    recommend_action: str
    reason_tags: list[str] = Field(default_factory=list)
    caution_tags: list[str] = Field(default_factory=list)


class SkinMatchProductComment(BaseModel):
    product_id: int
    summary: str
    fit_reason: str
    caution_comment: str
    action_comment: str


class SkinMatchExplanation(BaseModel):
    overall_summary: str
    product_comments: list[SkinMatchProductComment] = Field(default_factory=list)


class VanityRoutineStepComment(BaseModel):
    slot_order: int
    product_id: int
    comment: str


class VanityRoutineExplanation(BaseModel):
    overall_summary: str
    step_comments: list[VanityRoutineStepComment] = Field(default_factory=list)
    warning_comment: str = ""


class VanityLLMExplanation(BaseModel):
    prompt_version: str = "vanity_v1"
    generated_at: str
    skin_match: SkinMatchExplanation | None = None
    vanity_routine: VanityRoutineExplanation | None = None


class VanitySkinMatchResponse(BaseModel):
    match_session_id: int | None = None
    user_id: int
    basis_skin_result: BasisSkinResult
    product_match_results: list[ProductMatchResult]
    llm_explanation: VanityLLMExplanation | None = None


class VanityRoutineRequest(BaseModel):
    fixed_product_ids: list[int] = Field(default_factory=list)
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
    def validate_fixed_products(self) -> "VanityRoutineRequest":
        if not self.fixed_product_ids:
            raise ValueError("fixed_product_ids must not be empty")
        if len(set(self.fixed_product_ids)) != len(self.fixed_product_ids):
            raise ValueError("fixed_product_ids must not contain duplicate product_id")
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


class VanityRoutineItem(BaseModel):
    slot_order: int
    category: str | None = None
    product_id: int
    source: str
    product_score: float | None = None
    brand_name: str | None = None
    product_name: str | None = None
    price: int | None = None


class VanityRoutineResults(BaseModel):
    fixed_products: list[VanityRoutineItem] = Field(default_factory=list)
    recommended_products: list[VanityRoutineItem] = Field(default_factory=list)
    final_routine: list[VanityRoutineItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    total_price: int | None = None


class VanityRoutineResponse(BaseModel):
    recommendation_session_id: int | None = None
    user_id: int
    basis_skin_result: BasisSkinResult
    product_match_results: list[ProductMatchResult] = Field(default_factory=list)
    routine_recommendation_results: VanityRoutineResults
    llm_explanation: VanityLLMExplanation | None = None


class VanityRoutineListItem(BaseModel):
    recommendation_session_id: int
    created_at: str | None = None
    basis_result_id: int | None = None
    fixed_product_count: int = 0
    total_price: int | None = None


class VanityRoutineListResponse(BaseModel):
    routines: list[VanityRoutineListItem]


class VanitySummaryResponse(BaseModel):
    product_summary: dict[str, Any]
    latest_skin_match: dict[str, Any] | None = None
    latest_vanity_routine: dict[str, Any] | None = None
    basis_skin_result: BasisSkinResult | None = None
