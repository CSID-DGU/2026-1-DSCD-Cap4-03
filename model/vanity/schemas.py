from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VanityPipelineInput:
    user_id: int
    result_id: int | None = None
    vanity_product_ids: list[int] | None = None
    fixed_product_ids: list[int] | None = None
    budget: int | None = None
    total_budget_min: int | None = None
    total_budget_max: int | None = None
    slot_budget_min_map: dict[str, int] | None = None
    slot_budget_max_map: dict[str, int] | None = None


@dataclass(frozen=True)
class VanityProduct:
    product_id: int
    brand_name: str | None
    brand_name_kor: str | None
    product_name: str | None
    product_name_kor: str | None
    category: str | None
    function: str | None
    price: int | None
    ingredients: list[dict[str, Any]] = field(default_factory=list)
    review: dict[str, Any] = field(default_factory=dict)

    @property
    def product_key(self) -> str:
        return f"{self.brand_name}::{self.product_name}"


@dataclass(frozen=True)
class VanityContext:
    user_id: int
    profile: dict[str, Any]
    skin_result: dict[str, Any]
    products: list[VanityProduct]


@dataclass(frozen=True)
class ProductMatchResult:
    product_id: int
    category: str | None
    brand_name: str | None
    product_name: str | None
    scores: dict[str, float]
    vanity_fit_score: float
    fit_label: str
    recommend_action: str
    reason_tags: list[str]
    caution_tags: list[str]


@dataclass(frozen=True)
class RoutineItem:
    slot_order: int
    category: str | None
    product_id: int
    source: str
    product_score: float | None = None
    brand_name: str | None = None
    product_name: str | None = None
    price: int | None = None


@dataclass(frozen=True)
class VanityRoutineResult:
    fixed_products: list[RoutineItem]
    recommended_products: list[RoutineItem]
    final_routine: list[RoutineItem]
    warnings: list[str]
    total_price: int | None

