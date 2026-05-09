import re
import pandas as pd
import pymysql

from config import (
    driver,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DB,
)
from data.rule_ingredients import CONFLICT_RULES


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


def load_products(tx, hwahae: pd.DataFrame):
    q = """
    MERGE (p:Product {product_key: $product_key})
    SET   p.brand    = $brand,
          p.name     = $name,
          p.category = $category,
          p.function = $function,
          p.price    = $price
    WITH p
    MERGE (cat:Category {name: $category})
    MERGE (p)-[:IN_CATEGORY]->(cat)
    """
    for _, row in hwahae.iterrows():
        tx.run(
            q,
            product_key=f"{row['brand_name']}::{row['product_name']}",
            brand=row["brand_name"],
            name=row["product_name"],
            category=row["category"],
            function=row.get("function", ""),
            price=_safe_float_or_none(row.get("price")),
        )


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
    MERGE (i:Ingredient {name: $name})
    SET   i.irritation     = $irritation,
          i.comedogenicity = $comedogenicity,
          i.function_raw   = $function,
          i.rating         = $rating
    """
    q_contains = """
    MATCH (p:Product    {product_key: $product_key})
    MATCH (i:Ingredient {name: $ing_name})
    MERGE (p)-[:CONTAINS]->(i)
    """

    seen_ings = set()
    for _, row in inci_product.iterrows():
        product_key = f"{row['brand_name']}::{row['product_name']}"
        ing_name = row["Ingredient"]

        meta = ing_meta.get(
            ing_name,
            {"irritation": 0.0, "comedogenicity": 0.0, "function": "", "rating": ""},
        )
        if ing_name not in seen_ings:
            tx.run(q_ing, name=ing_name, **meta)
            seen_ings.add(ing_name)

        tx.run(q_contains, product_key=product_key, ing_name=ing_name)


FUNCTION_TO_CONCERN = {
    "Hydration": "dryness",
    "Moisturizing": "dryness",
    "Soothing": "dryness",
    "Pores": "pore",
    "Exfoliation": "pore",
    "Anti-Aging": "wrinkle",
    "Firming": "sagging",
    "Brightening": "pigmentation",
    "Blemishes": "acne",
}

FUNCTION_WEIGHT = {
    "Hydration": 1.0,
    "Moisturizing": 0.9,
    "Soothing": 0.7,
    "Pores": 1.0,
    "Exfoliation": 0.8,
    "Anti-Aging": 1.0,
    "Firming": 0.9,
    "Brightening": 1.0,
    "Blemishes": 1.0,
}


def load_concerns(tx, inci_ingredient: pd.DataFrame):
    q_concern = "MERGE (:Concern {name: $name})"
    q_helps = """
    MATCH (i:Ingredient {name: $ing_name})
    MATCH (c:Concern    {name: $concern})
    MERGE (i)-[r:HELPS]->(c)
    SET r.weight = $weight
    """
    for _, row in inci_ingredient.iterrows():
        for fn in str(row.get("Function", "")).split(","):
            fn = fn.strip()
            concern = FUNCTION_TO_CONCERN.get(fn)
            if concern:
                tx.run(q_concern, name=concern)
                tx.run(
                    q_helps,
                    ing_name=row["Ingredient"],
                    concern=concern,
                    weight=FUNCTION_WEIGHT.get(fn, 0.5),
                )


IRRITATION_THRESHOLD = 2
COMEDOGENIC_THRESHOLD = 2


def load_skintypes(tx, inci_ingredient: pd.DataFrame):
    skin_types = ["dry", "oily", "combination", "sensitive", "normal"]
    for st in skin_types:
        tx.run("MERGE (:SkinType {name: $name})", name=st)

    q_irritates = """
    MATCH (i:Ingredient {name: $ing_name})
    MATCH (s:SkinType   {name: $skin_type})
    MERGE (i)-[r:IRRITATES]->(s)
    SET r.score = $score
    """
    q_suits = """
    MATCH (i:Ingredient {name: $ing_name})
    MATCH (s:SkinType   {name: $skin_type})
    MERGE (i)-[:SUITS]->(s)
    """

    for _, row in inci_ingredient.iterrows():
        ing = row["Ingredient"]
        irr = _safe_float(row.get("Irritation"), default=0.0)
        comed = _safe_float(row.get("Comedogenicity"), default=0.0)

        if irr >= IRRITATION_THRESHOLD:
            tx.run(q_irritates, ing_name=ing, skin_type="sensitive", score=irr)
        if comed >= COMEDOGENIC_THRESHOLD:
            for st in ("oily", "combination"):
                tx.run(q_irritates, ing_name=ing, skin_type=st, score=comed)

        if irr < IRRITATION_THRESHOLD and comed < COMEDOGENIC_THRESHOLD:
            for st in skin_types:
                tx.run(q_suits, ing_name=ing, skin_type=st)


def load_conflicts(tx, conflict_pairs: pd.DataFrame):
    q = """
    MATCH (a:Ingredient {name: $a})
    MATCH (b:Ingredient {name: $b})
    MERGE (a)-[r:CONFLICTS]->(b)
    SET r.source = $source
    """

    for _, row in conflict_pairs.iterrows():
        tx.run(
            q,
            a=row["Ingredient1"],
            b=row["Ingredient2"],
            source="smiles",
        )

    for ing, rules in CONFLICT_RULES.items():
        for bad_ing in rules["bad"]:
            tx.run(q, a=ing, b=bad_ing, source="rule")


def load_rules(tx):
    for ing, rules in CONFLICT_RULES.items():
        rule_id = f"rule::{ing}"
        tx.run(
            """
            MERGE (r:Rule {rule_id: $rule_id})
            SET r.ingredient = $ing,
                r.bad_with = $bad,
                r.good_with = $good
            WITH r
            MATCH (i:Ingredient {name: $ing})
            MERGE (i)-[:COVERED_BY]->(r)
            """,
            rule_id=rule_id,
            ing=ing,
            bad=rules["bad"],
            good=rules.get("good", []),
        )


def create_user_session(
    tx,
    session_id: str,
    skin_data: dict,
    gender: str,
    allergies: list[str],
    wishlist_product_keys: list[str] | None = None,
):
    tx.run(
        """
        MERGE (u:UserSession {session_id: $sid})
        SET u.gender = $gender
        """,
        sid=session_id,
        gender=gender,
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

    skin_type = _infer_skin_type(skin_data)
    tx.run(
        """
        MATCH (u:UserSession {session_id: $sid})
        MATCH (s:SkinType    {name: $skin_type})
        MERGE (u)-[:HAS_SKIN_TYPE]->(s)
        """,
        sid=session_id,
        skin_type=skin_type,
    )


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
                i.ingredient_name AS Ingredient
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
    hwahae, inci_p, inci_i, edge_r = load_from_mysql()

    with driver.session() as s:
        s.execute_write(load_products, hwahae)
        s.execute_write(load_ingredients, inci_p, inci_i)
        s.execute_write(load_concerns, inci_i)
        s.execute_write(load_skintypes, inci_i)
        s.execute_write(load_conflicts, edge_r)
        s.execute_write(load_rules)

    print("Graph loading completed from MySQL.")


if __name__ == "__main__":
    load_all()

