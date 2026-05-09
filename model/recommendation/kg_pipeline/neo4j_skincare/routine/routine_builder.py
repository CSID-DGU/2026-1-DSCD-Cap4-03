import heapq

import pandas as pd

from model.recommendation.kg_pipeline.neo4j_skincare.config import AM_AVOID_INGREDIENTS, PM_AVOID_INGREDIENTS, SLOT_ORDER, driver
from model.recommendation.kg_pipeline.neo4j_skincare.routine.conflict_checker import check_am_pm, check_conflicts

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
        return sorted(items, key=lambda x: x[0], reverse=True)
    return heapq.nlargest(b, items, key=lambda x: x[0])

# Routine Builder
def build_routines(
    reranked: pd.DataFrame,
    gender: str,
    session_id: str,
    top_n: int = 3,
    beam_width: int = 150,
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

            rows.sort(key=lambda x: x["S_rerank"], reverse=True)
            rows = rows[:3] if slot_type == "optional" else rows[:5]
            if not rows:
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
                new_score = score + float(item.get("S_rerank", 0.0))
                cand_next.append((new_score, new_partial))
        beam = _top_b(cand_next, beam_width)

    routines = []
    skipped_conflict = 0
    for approx_score, products in beam:
        if not products:
            continue

        product_keys = [p["product_key"] for p in products]

        conflict = check_conflicts(product_keys)
        if conflict["has_conflict"]:
            skipped_conflict += 1
            continue

        am_pm = check_am_pm(product_keys, AM_AVOID_INGREDIENTS, PM_AVOID_INGREDIENTS)
        total_score = float(approx_score)

        routines.append(
            {
                "products": products,
                "total_score": round(total_score, 4),
                "am_pm_label": am_pm["am_pm_label"],
                "conflict_log": conflict["conflict_log"],
                "am_avoid_hits": am_pm["am_avoid_hits"],
                "slot_count": len(products),
            }
        )

    print(f"[debug][best] beam_candidates={len(beam)}, passed={len(routines)}, dropped_conflict={skipped_conflict}")

    routines.sort(
        key=lambda x: (
            x["am_pm_label"] == "am+pm",
            x["total_score"],
        ),
        reverse=True,
    )

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

            # price-first candidate pool
            rows.sort(key=lambda x: (x["_price"], -x["S_rerank"]))
            rows = rows[:3] if slot_type == "optional" else rows[:12]
            if not rows:
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
                cand_next.append((new_total, new_partial))
        beam = sorted(cand_next, key=lambda x: x[0])[:beam_width]

    # Phase 2: among cheap combos, pick highest score (after constraint checks)
    candidate_routines = []
    skipped_conflict = 0
    for total_price, products in beam:
        if not products:
            continue

        product_keys = [p["product_key"] for p in products]
        conflict = check_conflicts(product_keys)
        if conflict["has_conflict"]:
            skipped_conflict += 1
            continue

        am_pm = check_am_pm(product_keys, AM_AVOID_INGREDIENTS, PM_AVOID_INGREDIENTS)
        total_score = sum(float(p.get("S_rerank", 0.0)) for p in products)

        candidate_routines.append(
            {
                "products": products,
                "total_score": round(float(total_score), 4),
                "am_pm_label": am_pm["am_pm_label"],
                "conflict_log": conflict["conflict_log"],
                "am_avoid_hits": am_pm["am_avoid_hits"],
                "slot_count": len(products),
                "_total_price": float(total_price),
            }
        )

    print(f"[debug][value] cheap_beam={len(beam)}, passed={len(candidate_routines)}, dropped_conflict={skipped_conflict}")

    if not candidate_routines:
        return []

    # highest score first, then lower price tie-break
    candidate_routines.sort(key=lambda r: (-r["total_score"], r["_total_price"]))

    out = []
    for r in candidate_routines[:top_n]:
        r2 = dict(r)
        r2.pop("_total_price", None)
        out.append(r2)
    return out

