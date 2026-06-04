import heapq
import hashlib
from typing import Any

import pandas as pd

from model.recommendation.kg_pipeline.neo4j_skincare.config import AM_AVOID_INGREDIENTS, PM_AVOID_INGREDIENTS, SLOT_ORDER, driver
from model.recommendation.kg_pipeline.neo4j_skincare.routine.conflict_checker import check_am_pm, check_conflicts

OPTIONAL_SLOT_BONUS = 0.01
VALUE_PRICE_PENALTY = 0.05
OPTIONAL_DIVERSITY_TIE_BREAK = 0.006
BEST_MIN_PRICE = 10000
BEST_PRICE_FILTER_WINDOW = 20


def _routine_average_score(score_sum: float, products: list[dict]) -> float:
    if not products:
        return 0.0
    return float(score_sum) / len(products)


def _optional_slot_bonus(products: list[dict]) -> float:
    optional_count = sum(1 for p in products if p.get("_slot_type") == "optional")
    return optional_count * OPTIONAL_SLOT_BONUS


def _routine_search_score(score_sum: float, products: list[dict]) -> float:
    return _routine_average_score(score_sum, products) + _optional_slot_bonus(products)


def _optional_diversity_bonus(item: dict, session_id: str) -> float:
    if item.get("_slot_type") != "optional":
        return 0.0
    key = f"{session_id}::{item.get('category')}::{item.get('product_key')}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    return fraction * OPTIONAL_DIVERSITY_TIE_BREAK


def _selection_score(item: dict) -> float:
    return float(item.get("_selection_score", item.get("S_rerank", 0.0)))


def _best_price_preferred_rows(rows: list[dict], limit: int) -> list[dict]:
    limited_rows = rows[:limit]
    window = rows[:BEST_PRICE_FILTER_WINDOW]
    preferred = [row for row in window if _safe_price(row.get("price")) >= BEST_MIN_PRICE]
    return preferred[:limit] if preferred else limited_rows



def _routine_average_score(score_sum: float, products: list[dict]) -> float:
    if not products:
        return 0.0
    return float(score_sum) / len(products)


# 
SLOT_TOPK_QUERY = """
MATCH (p:Product)-[:IN_CATEGORY]->(cat:Category)
WHERE cat.name IN $categories
  AND toLower(trim(p.product_key)) IN $candidate_keys_lower
RETURN p.product_key AS product_key,
       p.name        AS name,
       p.brand       AS brand,
       p.category    AS category,
       p.price       AS price
"""

# Beam Search: 각 단계마다 상위 b개만 유지하는 함수
def _top_b(items: list[tuple[float, list[dict]]], b: int) -> list[tuple[float, list[dict]]]:
    if len(items) <= b:
        return sorted(items, key=lambda x: _routine_search_score(x[0], x[1]), reverse=True)
    return heapq.nlargest(b, items, key=lambda x: _routine_search_score(x[0], x[1]))


def _norm_category(v: Any) -> str:
    base = " ".join(str(v or "").strip().lower().split())
    aliases = {
        "toner": "toner+toner pads",
        "toners": "toner+toner pads",
        "toner pads": "toner+toner pads",
        "toner pad": "toner+toner pads",
        "toner + toner pads": "toner+toner pads",
        "essence": "essences/ampoules/serums",
        "essences": "essences/ampoules/serums",
        "ampoule": "essences/ampoules/serums",
        "ampoules": "essences/ampoules/serums",
        "serum": "essences/ampoules/serums",
        "serums": "essences/ampoules/serums",
        "essence/ampoule/serum": "essences/ampoules/serums",
        "cream/gel": "cream/gel",
        "face moisturizers": "cream/gel",
        "all in one": "all-in-one",
    }
    return aliases.get(base, base)


def _resolve_slot_budget(slot_budget_map: dict[str, float] | None, category: str) -> float | None:
    if not slot_budget_map:
        return None
    cat_norm = _norm_category(category)
    for k, v in slot_budget_map.items():
        if _norm_category(k) == cat_norm:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _has_budget_constraint(
    total_budget_min: float | None,
    total_budget_max: float | None,
    slot_budget_min_map: dict[str, float] | None,
    slot_budget_max_map: dict[str, float] | None,
) -> bool:
    return any(
        [
            total_budget_min is not None,
            total_budget_max is not None,
            bool(slot_budget_min_map),
            bool(slot_budget_max_map),
        ]
    )


def _item_price_allowed(
    item: dict,
    total_budget_min: float | None,
    total_budget_max: float | None,
    slot_budget_min_map: dict[str, float] | None,
    slot_budget_max_map: dict[str, float] | None,
) -> bool:
    price = _safe_price(item.get("price"))
    if price >= 1e12 and _has_budget_constraint(
        total_budget_min, total_budget_max, slot_budget_min_map, slot_budget_max_map
    ):
        return False

    category = str(item.get("category") or "")
    slot_min = _resolve_slot_budget(slot_budget_min_map, category)
    slot_max = _resolve_slot_budget(slot_budget_max_map, category)

    if slot_min is not None and price < slot_min:
        return False
    if slot_max is not None and price > slot_max:
        return False
    if total_budget_max is not None and price > float(total_budget_max):
        return False
    return True


def _routine_total_price(products: list[dict]) -> float | None:
    total = 0.0
    for p in _core_products(products):
        price = _safe_price(p.get("price"))
        if price >= 1e12:
            return None
        total += price
    return total


def _routine_budget_allowed(
    products: list[dict],
    total_budget_min: float | None,
    total_budget_max: float | None,
) -> bool:
    total = _routine_total_price(products)
    if total is None:
        return total_budget_min is None and total_budget_max is None
    if total_budget_min is not None and total < float(total_budget_min):
        return False
    if total_budget_max is not None and total > float(total_budget_max):
        return False
    return True


def _apply_item_time_tags(products: list[dict], am_pm: dict) -> None:
    am_avoid_keys = {
        str(hit.get("pk") or "").strip().lower()
        for hit in am_pm.get("am_hit_details", [])
    }
    pm_avoid_keys = {
        str(hit.get("pk") or "").strip().lower()
        for hit in am_pm.get("pm_hit_details", [])
    }
    for product in products:
        key = str(product.get("product_key") or "").strip().lower()
        if key in am_avoid_keys and key in pm_avoid_keys:
            product["time_tag"] = "check"
        elif key in am_avoid_keys:
            product["time_tag"] = "pm"
        elif key in pm_avoid_keys:
            product["time_tag"] = "am"
        else:
            product["time_tag"] = None


def _core_products(products: list[dict]) -> list[dict]:
    core = [p for p in products if p.get("_slot_type") != "optional"]
    return core or products


def _apply_time_tags_and_get_routine_am_pm(products: list[dict]) -> tuple[dict, dict]:
    all_keys = [p["product_key"] for p in products]
    item_am_pm = check_am_pm(all_keys, AM_AVOID_INGREDIENTS, PM_AVOID_INGREDIENTS)
    _apply_item_time_tags(products, item_am_pm)

    core_keys = [p["product_key"] for p in _core_products(products)]
    if core_keys == all_keys:
        routine_am_pm = item_am_pm
    else:
        routine_am_pm = check_am_pm(core_keys, AM_AVOID_INGREDIENTS, PM_AVOID_INGREDIENTS)
    return item_am_pm, routine_am_pm



# Routine Builder
def build_routines(
    reranked: pd.DataFrame,
    gender: str,
    session_id: str,
    top_n: int = 3,
    beam_width: int = 200,
    total_budget_min: float | None = None,
    total_budget_max: float | None = None,
    slot_budget_min_map: dict[str, float] | None = None,
    slot_budget_max_map: dict[str, float] | None = None,
) -> list[dict]:
    # 성별에 따른 SLOT
    slots = SLOT_ORDER[gender]
    all_keys = reranked["product_key"].tolist()
    all_keys_lower = [str(k).strip().lower() for k in all_keys]

    slot_pools = []
    score_map = {
        str(k).strip().lower(): float(v)
        for k, v in reranked.set_index("product_key")["S_rerank"].to_dict().items()
    }

    with driver.session() as s:
        for slot_type, categories in slots:
            rows = s.run(
                SLOT_TOPK_QUERY,
                categories=categories,
                candidate_keys_lower=all_keys_lower,
            ).data()

            for r in rows:
                r["S_rerank"] = score_map.get(str(r["product_key"]).strip().lower(), 0.0)
                r["_slot_type"] = slot_type
                r["_selection_score"] = r["S_rerank"] + _optional_diversity_bonus(r, session_id)

            rows = [
                r
                for r in rows
                if _item_price_allowed(
                    r,
                    total_budget_min=total_budget_min,
                    total_budget_max=total_budget_max,
                    slot_budget_min_map=slot_budget_min_map,
                    slot_budget_max_map=slot_budget_max_map,
                )
            ]
            rows.sort(key=lambda x: _selection_score(x), reverse=True)
            if _has_budget_constraint(total_budget_min, total_budget_max, slot_budget_min_map, slot_budget_max_map):
                rows = _best_price_preferred_rows(rows, 5 if slot_type == "optional" else 12)
            else:
                rows = _best_price_preferred_rows(rows, 3 if slot_type == "optional" else 5)
            if slot_type == "optional":
                rows = [None] + rows
            elif not rows:
                rows = [None]
            slot_pools.append(rows)

    slot_sizes = [sum(1 for x in pool if x is not None) for pool in slot_pools]
    print(f"[debug][best] slot_pool_sizes={slot_sizes}")

    # Beam search over slots
    beam: list[tuple[float, list[dict]]] = [(0.0, [])]
    for pool in slot_pools:
        cand_next: list[tuple[float, list[dict]]] = []
        for score, partial in beam:
            for item in pool:
                if item is None:
                    cand_next.append((score, partial))
                    continue
                new_partial = partial + [item]
                new_total = _routine_total_price(new_partial)
                if total_budget_max is not None and (new_total is None or new_total > float(total_budget_max)):
                    continue
                new_score = score + _selection_score(item)
                cand_next.append((new_score, new_partial))
        beam = _top_b(cand_next, beam_width)

    routines = []
    skipped_conflict = 0
    for approx_score, products in beam:
        if not products:
            continue
        if not _routine_budget_allowed(products, total_budget_min, total_budget_max):
            continue

        product_keys = [p["product_key"] for p in products]

        conflict = check_conflicts(product_keys)
        if conflict["has_conflict"]:
            skipped_conflict += 1
            continue

        item_am_pm, routine_am_pm = _apply_time_tags_and_get_routine_am_pm(products)
        total_score = _routine_search_score(float(approx_score), products)

        routines.append(
            {
                "products": products,
                "total_score": round(total_score, 4),
                "am_pm_label": routine_am_pm["am_pm_label"],
                "conflict_log": conflict["conflict_log"],
                "rule_conflict_log": conflict.get("rule_conflict_log", []),
                "smiles_conflict_log": conflict.get("smiles_conflict_log", []),
                "am_avoid_hits": item_am_pm["am_avoid_hits"],
                "pm_avoid_hits": item_am_pm.get("pm_avoid_hits", []),
                "am_hit_details": item_am_pm.get("am_hit_details", []),
                "pm_hit_details": item_am_pm.get("pm_hit_details", []),
                "slot_count": len(products),
            }
        )

    print(f"[debug][best] beam_candidates={len(beam)}, passed={len(routines)}, dropped_conflict={skipped_conflict}")

    routines.sort(key=lambda x: x["total_score"], reverse=True)

    return routines[:top_n]


def _safe_price(v) -> float:
    try:
        if v is None or str(v) == "nan":
            return 1e12
        return float(v)
    except (TypeError, ValueError):
        return 1e12


def build_value_routines(
    reranked: pd.DataFrame,
    gender: str,
    session_id: str,
    top_n: int = 1,
    beam_width: int = 500,
    total_budget_min: float | None = None,
    total_budget_max: float | None = None,
    slot_budget_min_map: dict[str, float] | None = None,
    slot_budget_max_map: dict[str, float] | None = None,
) -> list[dict]:
    slots = SLOT_ORDER[gender]
    all_keys = reranked["product_key"].tolist()
    all_keys_lower = [str(k).strip().lower() for k in all_keys]

    slot_pools = []
    score_map = {
        str(k).strip().lower(): float(v)
        for k, v in reranked.set_index("product_key")["S_rerank"].to_dict().items()
    }

    with driver.session() as s:
        for slot_type, categories in slots:
            rows = s.run(
                SLOT_TOPK_QUERY,
                categories=categories,
                candidate_keys_lower=all_keys_lower,
            ).data()

            for r in rows:
                r["S_rerank"] = score_map.get(str(r["product_key"]).strip().lower(), 0.0)
                r["_price"] = _safe_price(r.get("price"))
                r["_slot_type"] = slot_type
                r["_selection_score"] = r["S_rerank"] + _optional_diversity_bonus(r, session_id)

            rows = [
                r
                for r in rows
                if _item_price_allowed(
                    r,
                    total_budget_min=total_budget_min,
                    total_budget_max=total_budget_max,
                    slot_budget_min_map=slot_budget_min_map,
                    slot_budget_max_map=slot_budget_max_map,
                )
            ]
            # price-first candidate pool
            rows.sort(key=lambda x: (x["_price"], -_selection_score(x)))
            rows = rows[:3] if slot_type == "optional" else rows[:12]
            if slot_type == "optional":
                rows = [None] + rows
            elif not rows:
                rows = [None]
            slot_pools.append(rows)

    slot_sizes = [sum(1 for x in pool if x is not None) for pool in slot_pools]
    print(f"[debug][value] slot_pool_sizes={slot_sizes}")

    # Phase 1: keep cheapest 100 combinations via price-oriented beam search
    beam: list[tuple[float, list[dict]]] = [(0.0, [])]
    for pool in slot_pools:
        cand_next: list[tuple[float, list[dict]]] = []
        for total_price, partial in beam:
            for item in pool:
                if item is None:
                    cand_next.append((total_price, partial))
                    continue
                new_partial = partial + [item]
                new_total = total_price + float(item.get("_price", 1e12))
                if total_budget_max is not None and new_total > float(total_budget_max):
                    continue
                cand_next.append((new_total, new_partial))
        beam = sorted(cand_next, key=lambda x: x[0])[:beam_width]

    # Phase 2: among cheap combos, pick highest score (after constraint checks)
    candidate_routines = []
    skipped_conflict = 0
    for total_price, products in beam:
        if not products:
            continue
        if not _routine_budget_allowed(products, total_budget_min, total_budget_max):
            continue

        product_keys = [p["product_key"] for p in products]
        conflict = check_conflicts(product_keys)
        if conflict["has_conflict"]:
            skipped_conflict += 1
            continue

        item_am_pm, routine_am_pm = _apply_time_tags_and_get_routine_am_pm(products)
        score_sum = sum(_selection_score(p) for p in products)
        total_score = _routine_search_score(score_sum, products)

        candidate_routines.append(
            {
                "products": products,
                "total_score": round(float(total_score), 4),
                "am_pm_label": routine_am_pm["am_pm_label"],
                "conflict_log": conflict["conflict_log"],
                "rule_conflict_log": conflict.get("rule_conflict_log", []),
                "smiles_conflict_log": conflict.get("smiles_conflict_log", []),
                "am_avoid_hits": item_am_pm["am_avoid_hits"],
                "pm_avoid_hits": item_am_pm.get("pm_avoid_hits", []),
                "am_hit_details": item_am_pm.get("am_hit_details", []),
                "pm_hit_details": item_am_pm.get("pm_hit_details", []),
                "slot_count": len(products),
                "_total_price": float(total_price),
            }
        )

    print(f"[debug][value] cheap_beam={len(beam)}, passed={len(candidate_routines)}, dropped_conflict={skipped_conflict}")

    if not candidate_routines:
        return []

    # Value routines should still be relevant, but expensive combinations need to lose ground.
    candidate_routines.sort(
        key=lambda r: (
            -(r["total_score"] - VALUE_PRICE_PENALTY * (r["_total_price"] / 100000)),
            r["_total_price"],
        )
    )

    out = []
    for r in candidate_routines[:top_n]:
        r2 = dict(r)
        r2.pop("_total_price", None)
        out.append(r2)
    return out
