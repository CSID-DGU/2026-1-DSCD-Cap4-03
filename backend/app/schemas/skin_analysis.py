from pydantic import BaseModel


class SkinAnalysisCreateRequest(BaseModel):
    image_id: int


class SkinAnalysisCreateResponse(BaseModel):
    result_id: int
    image_id: int
    user_id: int
    analyzed_at: str
    model_version: str
    analysis_status: str


class SkinSummaryRequest(BaseModel):
    result_id: int


class SkinIndicatorComments(BaseModel):
    acne: str
    dryness: str
    sagging: str
    pore: str
    pigmentation: str
    wrinkle: str


class SkinSummaryResponse(BaseModel):
    result_id: int
    llm_model: str
    prompt_version: str
    summary_comment: str
    indicator_comments: SkinIndicatorComments
    generated_at: str


class SkinAnalysisDetailResponse(BaseModel):
    result_id: int
    user_id: int
    image_id: int
    model_name: str
    prompt_version: str
    analyzed_at: str
    generated_at: str
    summary_comment: str
    indicator_comments: SkinIndicatorComments
    image_url: str
    skin_type: str | None = None
    raw_metrics: dict[str, int]
    display_scores: dict[str, float]


class SkinAnalysisHistoryItem(BaseModel):
    result_id: int
    image_id: int
    analyzed_at: str
    skin_type: str | None = None
    image_url: str
    ai_comment: str
