from model.recommendation.kg_pipeline.neo4j_skincare.config import driver

# Ingredient conflicts across products in the same routine.
CONFLICT_QUERY = """
UNWIND $product_keys AS pk
MATCH (p:Product {product_key: pk})-[r1:CONTAINS]->(a:Ingredient)
WITH p, a, r1
ORDER BY p.product_key, coalesce(r1.order, 999999999), a.name
WITH p, collect(a)[..5] AS top_ings
UNWIND top_ings AS a
MATCH (a)-[c:CONFLICTS]-(b:Ingredient)<-[r2:CONTAINS]-(p2:Product)
WHERE p2.product_key IN $product_keys
  AND p.product_key <> p2.product_key
WITH p, a, c, b, p2, r2
ORDER BY p2.product_key, coalesce(r2.order, 999999999), b.name
WITH p, a, c, p2, collect(b)[..5] AS top_conflict_ings
UNWIND top_conflict_ings AS b
RETURN p.product_key AS product_key_1,
       p.name AS product_name_1,
       p.brand AS product_brand_1,
       a.name AS ing1,
       p2.product_key AS product_key_2,
       p2.name AS product_name_2,
       p2.brand AS product_brand_2,
       b.name AS ing2,
       c.source AS source
"""

# Ingredients that should be avoided in AM/PM usage.
AM_PM_QUERY = """
UNWIND $product_keys AS pk
MATCH (p:Product {product_key: pk})-[:CONTAINS]->(i:Ingredient)
WHERE i.name IN $avoid_list
RETURN p.product_key AS pk,
       p.brand AS brand,
       p.name AS product_name,
       i.name AS ingredient
"""


def check_conflicts(product_keys: list[str]) -> dict:
    rule_log: list[str] = []
    smiles_log: list[str] = []
    rule_pairs: set[tuple[str, str]] = set()
    smiles_pairs: set[tuple[str, str]] = set()

    with driver.session() as s:
        results = s.run(CONFLICT_QUERY, product_keys=product_keys)
        seen = set()
        for r in results:
            pair = (
                tuple(sorted([r["product_key_1"], r["product_key_2"]])),
                tuple(sorted([r["ing1"], r["ing2"]])),
                str(r["source"]).lower(),
            )
            if pair in seen:
                continue
            seen.add(pair)

            line = (
                f"{r['ing1']} x {r['ing2']} | "
                f"{r['product_brand_1']} - {r['product_name_1']} <-> "
                f"{r['product_brand_2']} - {r['product_name_2']}"
            )
            if str(r["source"]).lower() == "rule":
                rule_log.append(line)
                rule_pairs.add(tuple(sorted([r["ing1"], r["ing2"]])))
            else:
                smiles_log.append(line)
                smiles_pairs.add(tuple(sorted([r["ing1"], r["ing2"]])))

    combined_log = [f"[RULE] {x}" for x in rule_log] + [f"[SMILES] {x}" for x in smiles_log]
    return {
        "has_conflict": len(rule_log) > 0,
        "conflict_log": combined_log,
        "rule_conflict_log": rule_log,
        "smiles_conflict_log": smiles_log,
        "rule_conflict_count": len(rule_pairs),
        "smiles_conflict_count": len(smiles_pairs),
    }


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
        "am_hit_details": am_hits,
        "pm_hit_details": pm_hits,
    }
