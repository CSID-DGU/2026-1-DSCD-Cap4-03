from config import driver

# 성분 충돌
CONFLICT_QUERY = """
UNWIND $product_keys AS pk
MATCH (p:Product {product_key: pk})-[:CONTAINS]->(a:Ingredient)
      -[c:CONFLICTS]->(b:Ingredient)<-[:CONTAINS]-(p2:Product)
WHERE p2.product_key IN $product_keys
  AND p.product_key <> p2.product_key
RETURN a.name AS ing1, b.name AS ing2, c.source AS source
"""

# AM/PM 회피 성분
AM_PM_QUERY = """
UNWIND $product_keys AS pk
MATCH (p:Product {product_key: pk})-[:CONTAINS]->(i:Ingredient)
WHERE i.name IN $avoid_list
RETURN p.product_key AS pk, i.name AS ingredient
"""

def check_conflicts(product_keys: list[str]) -> dict:
    log = []

    with driver.session() as s:
        results = s.run(CONFLICT_QUERY, product_keys=product_keys)
        seen = set()
        for r in results:
            pair = tuple(sorted([r["ing1"], r["ing2"]]))
            if pair in seen:
                continue
            seen.add(pair)
            source = r["source"]
            log.append(f"[{str(source).upper()}] {r['ing1']} x {r['ing2']}")

    return {"has_conflict": len(log) > 0, "conflict_log": log}


def check_am_pm(product_keys: list[str], am_avoid: list[str], pm_avoid: list[str]) -> dict:
    with driver.session() as s:
        am_hits = s.run(AM_PM_QUERY, product_keys=product_keys, avoid_list=am_avoid).data()
        pm_hits = s.run(AM_PM_QUERY, product_keys=product_keys, avoid_list=pm_avoid).data()

    am_safe = len(am_hits) == 0
    pm_safe = len(pm_hits) == 0

    if am_safe and pm_safe:
        label = "am+pm"
    elif not am_safe and pm_safe:
        label = "pm_only"
    elif am_safe and not pm_safe:
        label = "am_only"
    else:
        label = "check_required"

    return {
        "am_pm_label": label,
        "am_avoid_hits": [h["ingredient"] for h in am_hits],
        "pm_avoid_hits": [h["ingredient"] for h in pm_hits],
    }
