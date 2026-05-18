from functools import lru_cache
import math
from typing import Optional

import pymysql

from model.recommendation.kg_pipeline.neo4j_skincare.config import (
    MYSQL_DB,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
    RERANK_CONCERN_WEIGHT,
    RERANK_IRRITATION_PENALTY_SCALE,
    RERANK_REVIEW_WEIGHT,
    RERANK_SKIN_WEIGHT,
    RERANK_VECTOR_WEIGHT,
    RERANK_WISHLIST_WEIGHT,
    driver,
)

SCORE_BASE_QUERY = """
MATCH (u:UserSession {session_id: $sid})
MATCH (p:Product {product_key: $product_key})
OPTIONAL MATCH (u)-[:HAS_SKIN_TYPE]->(st:SkinType)
OPTIONAL MATCH (p)-[:CONTAINS]->(i2:Ingredient)
OPTIONAL MATCH (i2)-[r:IRRITATES]->(st)
WITH count(i2) AS total,
     count(CASE WHEN (i2)-[:SUITS]->(st) THEN 1 END) AS suit_cnt,
     count(CASE WHEN r IS NOT NULL THEN 1 END) AS irr_cnt,
     coalesce(sum(coalesce(r.score, 0.0)), 0.0) AS irr_sum
RETURN
  CASE WHEN total = 0 THEN 0.0 ELSE toFloat(suit_cnt - irr_cnt) / total END AS skin_bonus,
  irr_sum
"""

SCORE_WITH_CONCERN_QUERY = """
MATCH (u:UserSession {session_id: $sid})
MATCH (p:Product {product_key: $product_key})
OPTIONAL MATCH (u)-[hc:HAS_CONCERN]->(c:Concern)<-[h:HELPS]-(i1:Ingredient)<-[:CONTAINS]-(p)
WITH u, p, coalesce(sum(coalesce(hc.importance, 1.0) * coalesce(h.weight, 1.0)), 0.0) AS concern_score
OPTIONAL MATCH (u)-[:HAS_SKIN_TYPE]->(st:SkinType)
OPTIONAL MATCH (p)-[:CONTAINS]->(i2:Ingredient)
OPTIONAL MATCH (i2)-[r:IRRITATES]->(st)
WITH concern_score,
     count(i2) AS total,
     count(CASE WHEN (i2)-[:SUITS]->(st) THEN 1 END) AS suit_cnt,
     count(CASE WHEN r IS NOT NULL THEN 1 END) AS irr_cnt,
     coalesce(sum(coalesce(r.score, 0.0)), 0.0) AS irr_sum
RETURN
  concern_score,
  CASE WHEN total = 0 THEN 0.0 ELSE toFloat(suit_cnt - irr_cnt) / total END AS skin_bonus,
  irr_sum
"""

_HAS_WISHLIST_GRAPH_CACHE: Optional[bool] = None
_GRAPH_SCHEMA_CACHE: Optional[dict] = None

WISHLIST_BONUS_QUERY = """
MATCH (u:UserSession {session_id: $sid})-[:HAS_WISHLIST]->(w:Product)
MATCH (p:Product {product_key: $product_key})
RETURN CASE WHEN count(CASE WHEN w.product_key = p.product_key THEN 1 END) > 0 THEN 1.0 ELSE 0.0 END AS wishlist_bonus
"""

# Profile-aware review mapping (simple deterministic keywords).
CONCERN_KEYWORDS = {
    "dryness": ["\uBCF4\uC2B5", "\uC218\uBD84", "\uCD09\uCD09", "\uAC74\uC870", "\uC18D\uAC74\uC870"],
    "acne": ["\uD2B8\uB7EC\uBE14", "\uC5EC\uB4DC\uB984", "\uC9C4\uC815", "\uD53C\uC9C0", "\uC881\uC300"],
    "pore": ["\uBAA8\uACF5", "\uD53C\uC9C0", "\uBE14\uB799\uD5E4\uB4DC", "\uAC01\uC9C8"],
    "pigmentation": ["\uBBF8\uBC31", "\uC7A1\uD2F0", "\uD1A4\uC5C5", "\uAE30\uBBF8", "\uC0C9\uC18C\uCE68\uCC29"],
    "wrinkle": ["\uC8FC\uB984", "\uD0C4\uB825", "\uB9C1\uD074", "\uC548\uD2F0\uC5D0\uC774\uC9D5"],
    "sagging": ["\uD0C4\uB825", "\uB9AC\uD504\uD305", "\uCC98\uC9D0", "\uD37C\uBC0D"],
}

SKIN_TYPE_KEYWORDS = {
    "dry": ["\uAC74\uC131", "\uBCF4\uC2B5", "\uCD09\uCD09", "\uC218\uBD84"],
    "oily": ["\uC9C0\uC131", "\uC0B0\uB73B", "\uD53C\uC9C0", "\uC720\uBD84"],
    "combination": ["\uBCF5\uD569\uC131", "\uC218\uBD80\uC9C0", "\uC720\uC218\uBD84", "\uBC38\uB7F0\uC2A4"],
    "sensitive": ["\uBBFC\uAC10\uC131", "\uC21C\uD568", "\uC800\uC790\uADF9", "\uC9C4\uC815"],
    "normal": ["\uC911\uC131", "\uBB34\uB09C", "\uB370\uC77C\uB9AC"],
}

NEGATIVE_HINTS = [
    "\uB530\uAC00\uC6C0",
    "\uC790\uADF9",
    "\uAC74\uC870\uD568",
    "\uB2F5\uB2F5",
    "\uB048\uC801",
    "\uD2B8\uB7EC\uBE14",
    "\uB4A4\uC9D1\uC5B4",
]

PROFILE_SKIN_TYPE_ALIASES = {
    "\uAC74\uC131": "dry",
    "\uC9C0\uC131": "oily",
    "\uBCF5\uD569\uC131": "combination",
    "\uC218\uBD80\uC9C0": "combination",
    "\uBBFC\uAC10\uC131": "sensitive",
    "\uC911\uC131": "normal",
    "\uBAA8\uB984": "",
    "dry": "dry",
    "oily": "oily",
    "combination": "combination",
    "sensitive": "sensitive",
    "normal": "normal",
}

def _mysql_connect():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _split_concerns(v: str | None) -> list[str]:
    if not v:
        return []
    raw = _norm_text(v)
    parts = [x.strip() for x in raw.replace("/", ",").replace("|", ",").split(",")]
    return [p for p in parts if p]


def _map_profile_skin_type(v: str | None) -> str:
    key = _norm_text(v or "")
    return PROFILE_SKIN_TYPE_ALIASES.get(key, "")


@lru_cache(maxsize=2048)
def _load_user_profile(user_id: int) -> tuple[str, list[str]]:
    conn = _mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(skin_type, '') AS skin_type,
                       COALESCE(skin_concern, '') AS skin_concern
                FROM USER_PROFILE
                WHERE user_id = %s
                LIMIT 1
                """,
                (int(user_id),),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return "", []

    skin_type = _map_profile_skin_type(str(row.get("skin_type") or ""))
    concerns = _split_concerns(str(row.get("skin_concern") or ""))
    return skin_type, concerns


@lru_cache(maxsize=4096)
def _load_review_text(product_key: str) -> tuple[str, str]:
    try:
        brand, name = product_key.split("::", 1)
    except ValueError:
        return "", ""

    query = """
    SELECT
      CONCAT_WS(' ', pr.pro1, pr.pro2, pr.pro3, pr.pro4, pr.pro5, pr.pro6, pr.pro7, pr.pros_text) AS pros_blob,
      CONCAT_WS(' ', pr.con1, pr.con2, pr.con3, pr.con4, pr.con5, pr.con6, pr.con7, pr.cons_text) AS cons_blob
    FROM PRODUCT p
    LEFT JOIN PRODUCT_REVIEW pr ON pr.product_id = p.product_id
    WHERE p.brand_name = %s AND p.product_name = %s
    LIMIT 1
    """

    conn = _mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (brand, name))
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return "", ""

    pros = _norm_text(str(row.get("pros_blob") or ""))
    cons = _norm_text(str(row.get("cons_blob") or ""))
    return pros, cons


def _profile_review_match_score(product_key: str, user_id: int | None) -> float:
    if user_id is None:
        return 0.0

    skin_type, concerns = _load_user_profile(int(user_id))
    pros, cons = _load_review_text(product_key)
    if not pros and not cons:
        return 0.0

    pos_hits = 0
    neg_hits = 0

    # concern-driven hits
    for c in concerns:
        kws = CONCERN_KEYWORDS.get(c, [])
        for kw in kws:
            if kw in pros:
                pos_hits += 1
            if kw in cons:
                neg_hits += 1

    # skin-type-driven hits
    for kw in SKIN_TYPE_KEYWORDS.get(skin_type, []):
        if kw in pros:
            pos_hits += 1
        if kw in cons:
            neg_hits += 1

    # generic negative hints from cons
    for kw in NEGATIVE_HINTS:
        if kw in cons:
            neg_hits += 1

    score = (pos_hits - neg_hits) / float(pos_hits + neg_hits + 1)
    if score > 1.0:
        return 1.0
    if score < -1.0:
        return -1.0
    return float(score)


def _has_wishlist_graph() -> bool:
    global _HAS_WISHLIST_GRAPH_CACHE
    if _HAS_WISHLIST_GRAPH_CACHE is not None:
        return _HAS_WISHLIST_GRAPH_CACHE

    rel_query = """
    CALL db.relationshipTypes() YIELD relationshipType
    RETURN collect(relationshipType) AS rels
    """
    with driver.session() as s:
        rels_row = s.run(rel_query).single()
        rels = set(rels_row["rels"] or []) if rels_row else set()
        _HAS_WISHLIST_GRAPH_CACHE = "HAS_WISHLIST" in rels
    return _HAS_WISHLIST_GRAPH_CACHE


def _graph_schema_flags() -> dict:
    global _GRAPH_SCHEMA_CACHE
    if _GRAPH_SCHEMA_CACHE is not None:
        return _GRAPH_SCHEMA_CACHE

    with driver.session() as s:
        rels_row = s.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS rels"
        ).single()
        labels_row = s.run("CALL db.labels() YIELD label RETURN collect(label) AS labels").single()

    rels = set((rels_row or {}).get("rels") or [])
    labels = set((labels_row or {}).get("labels") or [])

    _GRAPH_SCHEMA_CACHE = {
        "has_concern_graph": all(x in rels for x in ("HAS_CONCERN", "HELPS")) and "Concern" in labels,
    }
    return _GRAPH_SCHEMA_CACHE


def soft_score(product_key: str, session_id: str, vector_score: float, user_id: int | None = None) -> dict:
    flags = _graph_schema_flags()
    with driver.session() as s:
        if flags["has_concern_graph"]:
            rbase = s.run(SCORE_WITH_CONCERN_QUERY, sid=session_id, product_key=product_key).single()
            raw_concern_score = float((rbase or {}).get("concern_score") or 0.0)
        else:
            rbase = s.run(SCORE_BASE_QUERY, sid=session_id, product_key=product_key).single()
            raw_concern_score = 0.0

        raw_skin_bonus = float((rbase or {}).get("skin_bonus") or 0.0)
        irr_sum = float((rbase or {}).get("irr_sum") or 0.0)
        irr_penalty = min(irr_sum * RERANK_IRRITATION_PENALTY_SCALE, 1.0)

        if _has_wishlist_graph():
            r4 = s.run(WISHLIST_BONUS_QUERY, sid=session_id, product_key=product_key).single()
            wishlist_bonus = float((r4 or {}).get("wishlist_bonus") or 0.0)
        else:
            wishlist_bonus = 0.0

    vector_score = max(0.0, min(1.0, float(vector_score)))
    concern_score = 1.0 - math.exp(-raw_concern_score)
    skin_bonus = max(0.0, min(1.0, (raw_skin_bonus + 1.0) / 2.0))

    raw_review_score = _profile_review_match_score(product_key, user_id)
    review_score = max(0.0, min(1.0, (raw_review_score + 1.0) / 2.0))

    raw_s_rerank = (
        vector_score * RERANK_VECTOR_WEIGHT
        + concern_score * RERANK_CONCERN_WEIGHT
        + skin_bonus * RERANK_SKIN_WEIGHT
        + wishlist_bonus * RERANK_WISHLIST_WEIGHT
        + review_score * RERANK_REVIEW_WEIGHT
        - irr_penalty
    )
    s_rerank = max(0.0, min(1.0, (raw_s_rerank + 1.0) / 2.0))

    return {
        "product_key": product_key,
        "vector_score": round(vector_score, 4),
        "raw_concern_match_score": round(raw_concern_score, 4),
        "concern_match_score": round(concern_score, 4),
        "raw_skin_type_bonus": round(raw_skin_bonus, 4),
        "skin_type_bonus": round(skin_bonus, 4),
        "wishlist_bonus": round(wishlist_bonus, 4),
        "raw_review_score": round(raw_review_score, 4),
        "review_score": round(review_score, 4),
        "irritation_penalty": round(irr_penalty, 4),
        "raw_S_rerank": round(raw_s_rerank, 4),
        "S_rerank": round(s_rerank, 4),
    }
