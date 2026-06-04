from __future__ import annotations

from typing import Any

from model.recommendation.kg_pipeline.neo4j_skincare.config import AM_AVOID_INGREDIENTS, PM_AVOID_INGREDIENTS
from model.recommendation.kg_pipeline.neo4j_skincare.routine.conflict_checker import check_am_pm, check_conflicts
from model.vanity.config import get_slot_order
from model.vanity.data_loader import load_products, load_user_profile
from model.vanity.schemas import RoutineItem, VanityProduct, VanityRoutineResult


CATEGORY_ALIASES = {
    "toner": "toner",
    "toner pads": "toner",
    "emulsions": "emulsions",
    "essences/ampoules/serums": "essences/ampoules/serums",
    "cream/gel": "cream/gel",
    "face moisturizers": "cream/gel",
    "facial oils": "facial oils",
    "eye treatments": "eye treatments",
    "balms/multi-balms": "balms/multi-balms",
    "all-in-one": "all-in-one",
    "shaving products": "shaving products",
}


def normalize_category(category: str | None) -> str:
    raw = " ".join(str(category or "").strip().lower().split())
    return CATEGORY_ALIASES.get(raw, raw)


def product_to_routine_item(
    product: VanityProduct,
    slot_order: int,
    source: str,
    product_score: float | None = None,
) -> RoutineItem:
    return RoutineItem(
        slot_order=slot_order,
        category=product.category,
        product_id=product.product_id,
        source=source,
        product_score=product_score,
        brand_name=product.brand_name_kor or product.brand_name,
        product_name=product.product_name_kor or product.product_name,
        price=product.price,
    )


def validate_single_fixed_product_per_category(fixed_products: list[VanityProduct]) -> None:
    seen: dict[str, VanityProduct] = {}
    duplicates = []
    for product in fixed_products:
        category_norm = normalize_category(product.category)
        if category_norm in seen:
            prev = seen[category_norm]
            prev_name = prev.product_name_kor or prev.product_name or str(prev.product_id)
            product_name = product.product_name_kor or product.product_name or str(product.product_id)
            duplicates.append(f"{product.category}: {prev_name}, {product_name}")
            continue
        seen[category_norm] = product
    if duplicates:
        raise ValueError(
            "fixed_product_ids must contain only one product per category. "
            f"Duplicate categories: {'; '.join(duplicates)}"
        )


def get_target_slots(gender: str, fixed_products: list[VanityProduct]) -> list[tuple[str, str]]:
    fixed_categories = {normalize_category(product.category) for product in fixed_products}
    targets = []
    for slot_type, categories in get_slot_order(gender):
        for category in categories:
            category_norm = normalize_category(category)
            if category_norm not in fixed_categories:
                targets.append((slot_type, category))
                break
    return targets


def pick_recommended_products(
    target_slots: list[tuple[str, str]],
    candidate_products: list[dict[str, Any]],
    used_product_ids: set[int],
    remaining_budget_max: int | None = None,
    slot_budget_min_map: dict[str, int] | None = None,
    slot_budget_max_map: dict[str, int] | None = None,
    fallback_when_budget_empty: bool = False,
) -> list[dict[str, Any]]:
    picked = []
    used_categories = set()
    used_budget = 0
    for _, category in target_slots:
        category_norm = normalize_category(category)
        if category_norm in used_categories:
            continue
        matches = [
            candidate
            for candidate in candidate_products
            if normalize_category(candidate.get("category") or candidate.get("query_category")) == category_norm
            and int(candidate["product_id"]) not in used_product_ids
        ]
        slot_min = _resolve_slot_budget(slot_budget_min_map, category)
        slot_max = _resolve_slot_budget(slot_budget_max_map, category)
        if slot_min is not None:
            matches = [candidate for candidate in matches if _price(candidate) is None or _price(candidate) >= slot_min]
        if slot_max is not None:
            matches = [candidate for candidate in matches if _price(candidate) is None or _price(candidate) <= slot_max]
        if remaining_budget_max is not None:
            budget_matches = [
                candidate
                for candidate in matches
                if _price(candidate) is None or used_budget + _price(candidate) <= remaining_budget_max
            ]
            matches = budget_matches if budget_matches or not fallback_when_budget_empty else matches
        if not matches:
            continue
        matches.sort(key=lambda row: float(row.get("S_rerank", row.get("product_score", 0.0)) or 0.0), reverse=True)
        picked.append(matches[0])
        used_product_ids.add(int(matches[0]["product_id"]))
        used_categories.add(category_norm)
        if _price(matches[0]) is not None:
            used_budget += int(_price(matches[0]) or 0)
    return picked


def build_vanity_routine(
    user_id: int,
    fixed_product_ids: list[int],
    candidate_products: list[dict[str, Any]],
    total_budget_min: int | None = None,
    total_budget_max: int | None = None,
    slot_budget_min_map: dict[str, int] | None = None,
    slot_budget_max_map: dict[str, int] | None = None,
) -> VanityRoutineResult:
    profile = load_user_profile(user_id)
    gender = str(profile.get("gender") or "female").lower()
    fixed_products = load_products(fixed_product_ids)
    validate_single_fixed_product_per_category(fixed_products)
    fixed_product_id_set = {product.product_id for product in fixed_products}
    target_slots = get_target_slots(gender, fixed_products)
    core_categories = _core_category_set(gender)
    fixed_core_total_price = sum(
        int(product.price or 0)
        for product in fixed_products
        if normalize_category(product.category) in core_categories
    )
    remaining_core_budget_max = None
    if total_budget_max is not None:
        remaining_core_budget_max = max(0, int(total_budget_max) - fixed_core_total_price)

    used_product_ids = {product.product_id for product in fixed_products}
    core_target_slots = [slot for slot in target_slots if slot[0] != "optional"]
    optional_target_slots = [slot for slot in target_slots if slot[0] == "optional"]
    core_rows = pick_recommended_products(
        target_slots=core_target_slots,
        candidate_products=candidate_products,
        used_product_ids=used_product_ids,
        remaining_budget_max=remaining_core_budget_max,
        slot_budget_min_map=slot_budget_min_map,
        slot_budget_max_map=slot_budget_max_map,
        fallback_when_budget_empty=True,
    )
    core_recommended_total = sum(int(_price(row) or 0) for row in core_rows)
    optional_budget_max = None
    if total_budget_max is not None:
        optional_budget_max = max(0, int(total_budget_max) - fixed_core_total_price - core_recommended_total)
    optional_rows = pick_recommended_products(
        target_slots=optional_target_slots,
        candidate_products=candidate_products,
        used_product_ids=used_product_ids,
        remaining_budget_max=optional_budget_max,
        slot_budget_min_map=None,
        slot_budget_max_map=None,
        fallback_when_budget_empty=False,
    )
    recommended_rows = core_rows + optional_rows
    recommended_products = load_products([int(row["product_id"]) for row in recommended_rows])
    score_by_product_id = {
        int(row["product_id"]): float(row.get("S_rerank", row.get("product_score", 0.0)) or 0.0)
        for row in recommended_rows
    }

    final_products = fixed_products + recommended_products
    category_order = _category_order(gender)
    final_products.sort(key=lambda product: category_order.get(normalize_category(product.category), 999))

    final_routine = [
        product_to_routine_item(
            product=product,
            slot_order=index,
            source="vanity" if product.product_id in fixed_product_id_set else "recommendation",
            product_score=score_by_product_id.get(product.product_id),
        )
        for index, product in enumerate(final_products, start=1)
    ]

    product_keys = [product.product_key for product in final_products]
    warnings = []
    if product_keys:
        conflict = check_conflicts(product_keys)
        warnings.extend(conflict.get("conflict_log", []))
        am_pm = check_am_pm(product_keys, AM_AVOID_INGREDIENTS, PM_AVOID_INGREDIENTS)
        for hit in am_pm.get("am_hit_details", []):
            warnings.append(f"pm_only: {hit.get('product_name')} ({hit.get('ingredient')})")
        for hit in am_pm.get("pm_hit_details", []):
            warnings.append(f"am_only: {hit.get('product_name')} ({hit.get('ingredient')})")

    fixed_items = [item for item in final_routine if item.source == "vanity"]
    recommended_items = [item for item in final_routine if item.source == "recommendation"]
    core_prices = [
        item.price
        for item in final_routine
        if item.price is not None and normalize_category(item.category) in core_categories
    ]

    return VanityRoutineResult(
        fixed_products=fixed_items,
        recommended_products=recommended_items,
        final_routine=final_routine,
        warnings=warnings,
        total_price=sum(int(price) for price in core_prices) if core_prices else None,
    )


def _category_order(gender: str) -> dict[str, int]:
    order = {}
    idx = 1
    for _, categories in get_slot_order(gender):
        for category in categories:
            order.setdefault(normalize_category(category), idx)
        idx += 1
    return order


def _core_category_set(gender: str) -> set[str]:
    core_categories: set[str] = set()
    for slot_type, categories in get_slot_order(gender):
        if slot_type == "optional":
            continue
        for category in categories:
            core_categories.add(normalize_category(category))
    return core_categories


def _price(candidate: dict[str, Any]) -> int | None:
    value = candidate.get("price")
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _resolve_slot_budget(slot_budget_map: dict[str, int] | None, category: str) -> int | None:
    if not slot_budget_map:
        return None
    category_norm = normalize_category(category)
    for key, value in slot_budget_map.items():
        if normalize_category(key) != category_norm:
            continue
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None

