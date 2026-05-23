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


def get_target_categories(gender: str, fixed_products: list[VanityProduct]) -> list[str]:
    fixed_categories = {normalize_category(product.category) for product in fixed_products}
    targets = []
    for slot_type, categories in get_slot_order(gender):
        for category in categories:
            category_norm = normalize_category(category)
            if category_norm not in fixed_categories:
                targets.append(category)
                break
    return targets


def pick_recommended_products(
    target_categories: list[str],
    candidate_products: list[dict[str, Any]],
    used_product_ids: set[int],
) -> list[dict[str, Any]]:
    picked = []
    used_categories = set()
    for category in target_categories:
        category_norm = normalize_category(category)
        if category_norm in used_categories:
            continue
        matches = [
            candidate
            for candidate in candidate_products
            if normalize_category(candidate.get("category") or candidate.get("query_category")) == category_norm
            and int(candidate["product_id"]) not in used_product_ids
        ]
        if not matches:
            continue
        matches.sort(key=lambda row: float(row.get("S_rerank", row.get("product_score", 0.0)) or 0.0), reverse=True)
        picked.append(matches[0])
        used_product_ids.add(int(matches[0]["product_id"]))
        used_categories.add(category_norm)
    return picked


def build_vanity_routine(
    user_id: int,
    fixed_product_ids: list[int],
    candidate_products: list[dict[str, Any]],
) -> VanityRoutineResult:
    profile = load_user_profile(user_id)
    gender = str(profile.get("gender") or "female").lower()
    fixed_products = load_products(fixed_product_ids)
    validate_single_fixed_product_per_category(fixed_products)
    fixed_product_id_set = {product.product_id for product in fixed_products}
    target_categories = get_target_categories(gender, fixed_products)

    used_product_ids = {product.product_id for product in fixed_products}
    recommended_rows = pick_recommended_products(
        target_categories=target_categories,
        candidate_products=candidate_products,
        used_product_ids=used_product_ids,
    )
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
    prices = [item.price for item in final_routine if item.price is not None]

    return VanityRoutineResult(
        fixed_products=fixed_items,
        recommended_products=recommended_items,
        final_routine=final_routine,
        warnings=warnings,
        total_price=sum(int(price) for price in prices) if prices else None,
    )


def _category_order(gender: str) -> dict[str, int]:
    order = {}
    idx = 1
    for _, categories in get_slot_order(gender):
        for category in categories:
            order.setdefault(normalize_category(category), idx)
        idx += 1
    return order

