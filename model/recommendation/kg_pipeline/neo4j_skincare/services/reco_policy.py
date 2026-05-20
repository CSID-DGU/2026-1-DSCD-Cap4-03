from typing import Any

import pandas as pd

from model.recommendation.kg_pipeline.neo4j_skincare.services.user_data import _norm_brand_name_key


def _build_price_map(reranked: pd.DataFrame) -> dict[str, Any]:
    price_map: dict[str, Any] = {}
    if not reranked.empty and "price" in reranked.columns:
        for _, rr in reranked.iterrows():
            k = _norm_brand_name_key(rr.get("Brand"), rr.get("product_name"))
            price_map[k] = rr.get("price")
    return price_map

def _routine_total_price(routine: dict[str, Any], price_map: dict[str, Any]) -> float | None:
    total = 0.0
    for p in routine.get("products", []):
        k = _norm_brand_name_key(p.get("brand"), p.get("name"))
        v = price_map.get(k)
        if v is None or str(v) == "nan":
            return None
        total += float(v)
    return total

def _select_best_and_cheapest(routines: list[dict[str, Any]], price_map: dict[str, Any]) -> list[dict[str, Any]]:
    if not routines:
        return []

    best = dict(routines[0])
    best["routine_label"] = "Best Routine"

    priced = []
    for r in routines:
        tp = _routine_total_price(r, price_map)
        if tp is not None:
            priced.append((tp, r))

    selected = [best]
    cheapest = None
    if priced:
        priced.sort(key=lambda x: x[0])
        cheapest = priced[0][1]
    elif len(routines) > 1:
        cheapest = routines[1]

    if cheapest is not None and cheapest is not routines[0]:
        c = dict(cheapest)
        c["routine_label"] = "Value Routine"
        selected.append(c)

    return selected

def _attach_all_in_one_to_routines(routines: list[dict[str, Any]], all_in_one_pick: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not routines or not all_in_one_pick:
        return routines

    ai = {
        "product_key": all_in_one_pick.get("product_key"),
        "name": all_in_one_pick.get("product_name"),
        "brand": all_in_one_pick.get("Brand"),
        "category": "All-In-One",
        "price": all_in_one_pick.get("price"),
        "S_rerank": float(all_in_one_pick.get("S_rerank") or 0.0),
    }

    out = []
    ai_key = str(ai.get("product_key") or "").strip().lower()
    for r in routines:
        products = list(r.get("products", []))
        has_ai = False
        for p in products:
            pk = str(p.get("product_key") or "").strip().lower()
            cat = str(p.get("category") or "").strip().lower().replace(" ", "")
            if pk == ai_key or cat in ("all-in-one", "allinone"):
                has_ai = True
                break
        if not has_ai:
            products.append(ai)

        nr = dict(r)
        nr["products"] = products
        nr["slot_count"] = len(products)
        score_sum = sum(float(p.get("S_rerank", 0.0) or 0.0) for p in products)
        nr["total_score"] = round(score_sum / len(products), 4) if products else 0.0
        out.append(nr)

    return out

