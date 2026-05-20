import re
import argparse
from pathlib import Path
import sys

import pandas as pd
import pymysql

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.recommendation.kg_pipeline.neo4j_skincare.config import (
    driver,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DB,
)
from model.recommendation.kg_pipeline.neo4j_skincare.data.rule_ingredients import CONFLICT_RULES


LOAD_BATCH_SIZE = 500


def _safe_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return default
        value = v
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_float_or_none(value):
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return None
        value = v
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_severity(query_text: str) -> dict:
    return {
        m.group(1): float(m.group(2))
        for m in re.finditer(r"(\w+)\(([\d.]+)\)", query_text)
    }


def _batched_rows(rows: list[dict], batch_size: int = LOAD_BATCH_SIZE):
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def ensure_constraints(tx):
    statements = [
        "CREATE CONSTRAINT product_key_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.product_key IS UNIQUE",
        "CREATE CONSTRAINT ingredient_name_unique IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",
        "CREATE CONSTRAINT concern_name_unique IF NOT EXISTS FOR (c:Concern) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT skintype_name_unique IF NOT EXISTS FOR (s:SkinType) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT rule_id_unique IF NOT EXISTS FOR (r:Rule) REQUIRE r.rule_id IS UNIQUE",
        "CREATE CONSTRAINT usersession_id_unique IF NOT EXISTS FOR (u:UserSession) REQUIRE u.session_id IS UNIQUE",
    ]
    for stmt in statements:
        tx.run(stmt)


def load_products(tx, hwahae: pd.DataFrame):
    q = """
    UNWIND $rows AS row
    MERGE (p:Product {product_key: row.product_key})
    SET   p.brand    = row.brand,
          p.name     = row.name,
          p.category = row.category,
          p.function = row.function,
          p.price    = row.price
    WITH p, row
    MERGE (cat:Category {name: row.category})
    MERGE (p)-[:IN_CATEGORY]->(cat)
    """
    rows = []
    for _, row in hwahae.iterrows():
        rows.append(
            {
                "product_key": f"{row['brand_name']}::{row['product_name']}",
                "brand": row["brand_name"],
                "name": row["product_name"],
                "category": row["category"],
                "function": row.get("function", ""),
                "price": _safe_float_or_none(row.get("price")),
            }
        )
    for batch in _batched_rows(rows):
        tx.run(q, rows=batch)


def load_ingredients(tx, inci_product: pd.DataFrame, inci_ingredient: pd.DataFrame):
    ing_meta = {
        row["Ingredient"]: {
            "irritation": _safe_float(row.get("Irritation"), default=0.0),
            "comedogenicity": _safe_float(row.get("Comedogenicity"), default=0.0),
            "function": str(row.get("Function", "")),
            "rating": str(row.get("Rating", "")),
        }
        for _, row in inci_ingredient.iterrows()
    }

    q_ing = """
    UNWIND $rows AS row
    MERGE (i:Ingredient {name: row.name})
    SET   i.irritation     = row.irritation,
          i.comedogenicity = row.comedogenicity,
          i.function_raw   = row.function,
          i.rating         = row.rating
    """
    q_contains = """
    UNWIND $rows AS row
    MATCH (p:Product    {product_key: row.product_key})
    MATCH (i:Ingredient {name: row.ing_name})
    MERGE (p)-[r:CONTAINS]->(i)
    SET r.order = row.ingredient_order
    """

    ingredient_rows = []
    for ing_name, meta in ing_meta.items():
        ingredient_rows.append({"name": ing_name, **meta})
    for batch in _batched_rows(ingredient_rows):
        tx.run(q_ing, rows=batch)

    contains_rows = []
    for _, row in inci_product.iterrows():
        contains_rows.append(
            {
                "product_key": f"{row['brand_name']}::{row['product_name']}",
                "ing_name": row["Ingredient"],
                "ingredient_order": int(row["ingredient_order"]),
            }
        )
    for batch in _batched_rows(contains_rows):
        tx.run(q_contains, rows=batch)


FUNCTION_KEYWORD_TO_CONCERN = {
    # dryness / hydration
    "moisturizer": ("dryness", 1.0),
    "moisturizing": ("dryness", 1.0),
    "humectant": ("dryness", 1.0),
    "emollient": ("dryness", 0.9),
    "skin conditioning": ("dryness", 0.8),
    "soothing": ("dryness", 0.7),
    "hydration": ("dryness", 1.0),
    # pore / texture / sebum
    "astringent": ("pore", 0.8),
    "sebum": ("pore", 0.9),
    "oil control": ("pore", 0.9),
    "exfoliant": ("pore", 0.8),
    "exfoliation": ("pore", 0.8),
    "keratolytic": ("pore", 0.8),
    "surfactant/cleansing": ("pore", 0.5),
    "cleansing": ("pore", 0.5),
    # wrinkle / aging
    "anti-aging": ("wrinkle", 1.0),
    "anti aging": ("wrinkle", 1.0),
    "antioxidant": ("wrinkle", 0.8),
    "wrinkle": ("wrinkle", 1.0),
    # sagging / elasticity
    "firming": ("sagging", 0.9),
    "lifting": ("sagging", 0.9),
    "tightening": ("sagging", 0.8),
    "elasticity": ("sagging", 0.8),
    # pigmentation / tone
    "brightening": ("pigmentation", 1.0),
    "whitening": ("pigmentation", 1.0),
    "bleaching": ("pigmentation", 0.9),
    "tone up": ("pigmentation", 0.8),
    # acne / blemish
    "anti-acne": ("acne", 1.0),
    "anti acne": ("acne", 1.0),
    "anti-blemish": ("acne", 0.9),
    "blemish": ("acne", 0.9),
    "anti-inflammatory": ("acne", 0.8),
}


def _function_matches_to_concerns(function_text: str) -> list[tuple[str, float]]:
    """
    Convert raw ingredient function text into concern links using keyword rules.

    The source DB stores INCI-style role labels such as
    "moisturizer/humectant" or "surfactant/cleansing", so we use substring
    matching instead of exact function-name matching.
    """
    text = str(function_text or "").strip().lower()
    if not text:
        return []

    matches: list[tuple[str, float]] = []
    seen: set[str] = set()
    for keyword, (concern, weight) in FUNCTION_KEYWORD_TO_CONCERN.items():
        if keyword in text and concern not in seen:
            matches.append((concern, weight))
            seen.add(concern)
    return matches


def load_concerns(tx, inci_ingredient: pd.DataFrame):
    q_concern = "UNWIND $rows AS row MERGE (:Concern {name: row.name})"
    q_helps = """
    UNWIND $rows AS row
    MATCH (i:Ingredient {name: row.ing_name})
    MATCH (c:Concern    {name: row.concern})
    MERGE (i)-[r:HELPS]->(c)
    SET r.weight = row.weight
    """
    concern_names: set[str] = set()
    help_rows: list[dict] = []
    for _, row in inci_ingredient.iterrows():
        for concern, weight in _function_matches_to_concerns(row.get("Function", "")):
            concern_names.add(concern)
            help_rows.append(
                {
                    "ing_name": row["Ingredient"],
                    "concern": concern,
                    "weight": weight,
                }
            )
    for batch in _batched_rows([{"name": name} for name in sorted(concern_names)]):
        tx.run(q_concern, rows=batch)
    for batch in _batched_rows(help_rows):
        tx.run(q_helps, rows=batch)


IRRITATION_THRESHOLD = 2
COMEDOGENIC_THRESHOLD = 2


def load_skintypes(tx, inci_ingredient: pd.DataFrame):
    skin_types = ["dry", "oily", "combination", "sensitive", "normal"]
    tx.run("UNWIND $rows AS row MERGE (:SkinType {name: row.name})", rows=[{"name": st} for st in skin_types])

    q_irritates = """
    UNWIND $rows AS row
    MATCH (i:Ingredient {name: row.ing_name})
    MATCH (s:SkinType   {name: row.skin_type})
    MERGE (i)-[r:IRRITATES]->(s)
    SET r.score = row.score
    """
    q_suits = """
    UNWIND $rows AS row
    MATCH (i:Ingredient {name: row.ing_name})
    MATCH (s:SkinType   {name: row.skin_type})
    MERGE (i)-[:SUITS]->(s)
    """

    irritates_rows: list[dict] = []
    suits_rows: list[dict] = []
    for _, row in inci_ingredient.iterrows():
        ing = row["Ingredient"]
        irr = _safe_float(row.get("Irritation"), default=0.0)
        comed = _safe_float(row.get("Comedogenicity"), default=0.0)

        if irr >= IRRITATION_THRESHOLD:
            irritates_rows.append({"ing_name": ing, "skin_type": "sensitive", "score": irr})
        if comed >= COMEDOGENIC_THRESHOLD:
            for st in ("oily", "combination"):
                irritates_rows.append({"ing_name": ing, "skin_type": st, "score": comed})

        if irr < IRRITATION_THRESHOLD and comed < COMEDOGENIC_THRESHOLD:
            for st in skin_types:
                suits_rows.append({"ing_name": ing, "skin_type": st})
    for batch in _batched_rows(irritates_rows):
        tx.run(q_irritates, rows=batch)
    for batch in _batched_rows(suits_rows):
        tx.run(q_suits, rows=batch)


def load_conflicts(tx, conflict_pairs: pd.DataFrame):
    q = """
    UNWIND $rows AS row
    MATCH (a:Ingredient {name: row.a})
    MATCH (b:Ingredient {name: row.b})
    MERGE (a)-[r:CONFLICTS]->(b)
    SET r.source = row.source
    """

    rows: list[dict] = []
    for _, row in conflict_pairs.iterrows():
        rows.append({"a": row["Ingredient1"], "b": row["Ingredient2"], "source": "smiles"})

    for ing, rules in CONFLICT_RULES.items():
        for bad_ing in rules["bad"]:
            rows.append({"a": ing, "b": bad_ing, "source": "rule"})
    for batch in _batched_rows(rows):
        tx.run(q, rows=batch)


def load_rules(tx):
    q = """
    UNWIND $rows AS row
    MERGE (r:Rule {rule_id: row.rule_id})
    SET r.ingredient = row.ing,
        r.bad_with = row.bad,
        r.good_with = row.good
    WITH r, row
    MATCH (i:Ingredient {name: row.ing})
    MERGE (i)-[:COVERED_BY]->(r)
    """
    rows = [
        {
            "rule_id": f"rule::{ing}",
            "ing": ing,
            "bad": rules["bad"],
            "good": rules.get("good", []),
        }
        for ing, rules in CONFLICT_RULES.items()
    ]
    for batch in _batched_rows(rows):
        tx.run(q, rows=batch)


def create_user_session(
    tx,
    session_id: str,
    skin_data: dict,
    gender: str,
    allergies: list[str],
    wishlist_product_keys: list[str] | None = None,
    profile_skin_type: str | None = None,
):
    tx.run(
        """
        MERGE (u:UserSession {session_id: $sid})
        SET u.gender = $gender
        """,
        sid=session_id,
        gender=gender,
    )

    # Refresh dynamic user-session edges on every run so the graph reflects
    # the latest skin scores, allergies, wishlist, and inferred skin type.
    tx.run(
        """
        MATCH (u:UserSession {session_id: $sid})-[r]->()
        WHERE type(r) IN ['HAS_CONCERN', 'HAS_ALLERGY', 'HAS_WISHLIST', 'HAS_SKIN_TYPE']
        DELETE r
        """,
        sid=session_id,
    )

    for concern, importance in skin_data.items():
        tx.run(
            """
            MATCH (u:UserSession {session_id: $sid})
            MATCH (c:Concern     {name: $concern})
            MERGE (u)-[r:HAS_CONCERN]->(c)
            SET r.importance = $importance
            """,
            sid=session_id,
            concern=concern,
            importance=importance,
        )

    for allergy in allergies:
        tx.run(
            """
            MATCH (u:UserSession {session_id: $sid})
            MATCH (i:Ingredient  {name: $ing})
            MERGE (u)-[:HAS_ALLERGY]->(i)
            """,
            sid=session_id,
            ing=allergy,
        )

    for product_key in wishlist_product_keys or []:
        tx.run(
            """
            MATCH (u:UserSession {session_id: $sid})
            MATCH (p:Product     {product_key: $product_key})
            MERGE (u)-[:HAS_WISHLIST]->(p)
            """,
            sid=session_id,
            product_key=product_key,
        )

    skin_type = _normalize_profile_skin_type(profile_skin_type) or _infer_skin_type(skin_data)
    tx.run(
        """
        MATCH (u:UserSession {session_id: $sid})
        MATCH (s:SkinType    {name: $skin_type})
        MERGE (u)-[:HAS_SKIN_TYPE]->(s)
        """,
        sid=session_id,
        skin_type=skin_type,
    )


def _normalize_profile_skin_type(profile_skin_type: str | None) -> str:
    key = str(profile_skin_type or "").strip()
    aliases = {
        "\uAC74\uC131": "dry",
        "\uC9C0\uC131": "oily",
        "\uBCF5\uD569\uC131": "combination",
        "\uC218\uBD80\uC9C0": "combination",
        "\uBBFC\uAC10\uC131": "sensitive",
        "\uC911\uC131": "normal",
        "dry": "dry",
        "oily": "oily",
        "combination": "combination",
        "sensitive": "sensitive",
        "normal": "normal",
    }
    return aliases.get(key, "")


def _infer_skin_type(skin_data: dict) -> str:
    if skin_data.get("dryness", 0) > 0.4:
        return "dry"
    if skin_data.get("acne", 0) > 0.3 or skin_data.get("pore", 0) > 0.4:
        return "oily"
    return "combination"


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


def _fetch_df(conn, query: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def load_from_mysql() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = _mysql_connect()
    try:
        product_df = _fetch_df(
            conn,
            """
            SELECT
                brand_name,
                product_name,
                category,
                `function` AS `function`,
                price
            FROM PRODUCT
            WHERE brand_name IS NOT NULL
              AND product_name IS NOT NULL
            """,
        )

        inci_product_df = _fetch_df(
            conn,
            """
            SELECT
                p.brand_name,
                p.product_name,
                i.ingredient_name AS Ingredient,
                pi.product_ingredient_id AS ingredient_order
            FROM PRODUCT_INGREDIENT pi
            JOIN PRODUCT p ON p.product_id = pi.product_id
            JOIN INGREDIENT i ON i.ingredient_id = pi.ingredient_id
            WHERE p.brand_name IS NOT NULL
              AND p.product_name IS NOT NULL
              AND i.ingredient_name IS NOT NULL
            """,
        )

        ingredient_df = _fetch_df(
            conn,
            """
            SELECT
                ingredient_name AS Ingredient,
                `function` AS `Function`,
                rating AS Rating,
                irritation AS Irritation,
                comedogenicity AS Comedogenicity
            FROM INGREDIENT
            WHERE ingredient_name IS NOT NULL
            """,
        )

        conflict_df = _fetch_df(
            conn,
            """
            SELECT
                ingredient1_name AS Ingredient1,
                ingredient2_name AS Ingredient2
            FROM INGREDIENT_CONFLICT
            WHERE ingredient1_name IS NOT NULL
              AND ingredient2_name IS NOT NULL
              
            """,
        )
    finally:
        conn.close()

    return product_df, inci_product_df, ingredient_df, conflict_df


def load_all():
    """Backward-compatible alias for loading static cosmetics graph data."""
    load_static_graph()


def load_static_graph():
    """
    Load only static cosmetics graph data from MySQL.

    This should be run once during initialization or whenever product,
    ingredient, or conflict master data changes. User-specific dynamic data
    is loaded separately at recommendation time through create_user_session().
    """
    hwahae, inci_p, inci_i, edge_r = load_from_mysql()

    with driver.session() as s:
        s.execute_write(ensure_constraints)
        s.execute_write(load_products, hwahae)
        s.execute_write(load_ingredients, inci_p, inci_i)
        s.execute_write(load_concerns, inci_i)
        s.execute_write(load_skintypes, inci_i)
        s.execute_write(load_conflicts, edge_r)
        s.execute_write(load_rules)

    print("Static cosmetics graph loading completed from MySQL.")


def parse_args():
    parser = argparse.ArgumentParser(description="Load Neo4j graph data for recommendation pipeline.")
    parser.add_argument(
        "--mode",
        choices=["static", "all"],
        default="static",
        help="Load mode. 'static' loads only cosmetics master graph data. 'all' is kept as a backward-compatible alias.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode in {"static", "all"}:
        load_static_graph()

