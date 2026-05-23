from __future__ import annotations

from typing import Any

import pymysql

from model.recommendation.kg_pipeline.neo4j_skincare.config import (
    MYSQL_DB,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)
from model.vanity.schemas import VanityContext, VanityProduct


SCORE_KEYS = [
    "acne_score",
    "dryness_score",
    "sagging_score",
    "pore_score",
    "pigmentation_score",
    "wrinkle_score",
]


def mysql_connect():
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


def normalize_score(value: Any) -> float:
    if value is None:
        return 0.0
    score = float(value)
    if score > 1.0:
        score = score / 100.0
    return max(0.0, min(score, 1.0))


def load_user_profile(user_id: int) -> dict[str, Any]:
    conn = mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, gender, birth, skin_type, skin_concern, updated_at
                FROM USER_PROFILE
                WHERE user_id = %s
                LIMIT 1
                """,
                (user_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise ValueError(f"USER_PROFILE not found for user_id={user_id}")
    return row


def load_skin_result(user_id: int, result_id: int | None = None) -> dict[str, Any]:
    conn = mysql_connect()
    try:
        with conn.cursor() as cur:
            if result_id is None:
                cur.execute(
                    """
                    SELECT *
                    FROM SKIN_ANALYSIS_RESULT
                    WHERE user_id = %s
                    ORDER BY analyzed_at DESC, result_id DESC
                    LIMIT 1
                    """,
                    (user_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT *
                    FROM SKIN_ANALYSIS_RESULT
                    WHERE user_id = %s AND result_id = %s
                    LIMIT 1
                    """,
                    (user_id, result_id),
                )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise ValueError(f"SKIN_ANALYSIS_RESULT not found for user_id={user_id}")

    for key in SCORE_KEYS:
        row[key] = normalize_score(row.get(key))
    return row


def load_user_allergies(user_id: int) -> list[str]:
    conn = mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT allergy_ingredient
                FROM USER_ALLERGY
                WHERE user_id = %s
                  AND allergy_ingredient IS NOT NULL
                  AND TRIM(allergy_ingredient) <> ''
                ORDER BY allergy_id
                """,
                (user_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [str(row["allergy_ingredient"]) for row in rows if row.get("allergy_ingredient")]


def load_wishlist_product_keys(user_id: int) -> list[str]:
    conn = mysql_connect()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT DISTINCT CONCAT(p.brand_name, '::', p.product_name) AS product_key
                    FROM USER_WISHLIST uw
                    JOIN PRODUCT p ON p.product_id = uw.product_id
                    WHERE uw.user_id = %s
                      AND p.brand_name IS NOT NULL
                      AND p.product_name IS NOT NULL
                    ORDER BY product_key
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()
            except pymysql.MySQLError:
                rows = []
    finally:
        conn.close()
    return [str(row["product_key"]) for row in rows if row.get("product_key")]


def load_vanity_product_ids(user_id: int) -> list[int]:
    conn = mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT product_id
                FROM USER_VANITY
                WHERE user_id = %s
                ORDER BY created_at DESC, vanity_id DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [int(row["product_id"]) for row in rows]


def load_products(product_ids: list[int]) -> list[VanityProduct]:
    if not product_ids:
        return []

    placeholders = ",".join(["%s"] * len(product_ids))
    conn = mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    product_id,
                    brand_name,
                    brand_name_kor,
                    product_name,
                    product_name_kor,
                    category,
                    `function`,
                    price
                FROM PRODUCT
                WHERE product_id IN ({placeholders})
                """,
                tuple(product_ids),
            )
            product_rows = cur.fetchall()

            cur.execute(
                f"""
                SELECT
                    pi.product_id,
                    i.ingredient_id,
                    i.ingredient_name,
                    i.`function`,
                    i.rating,
                    i.irritation,
                    i.comedogenicity,
                    i.allergy_category
                FROM PRODUCT_INGREDIENT pi
                JOIN INGREDIENT i ON i.ingredient_id = pi.ingredient_id
                WHERE pi.product_id IN ({placeholders})
                """,
                tuple(product_ids),
            )
            ingredient_rows = cur.fetchall()

            cur.execute(
                f"""
                SELECT *
                FROM PRODUCT_REVIEW
                WHERE product_id IN ({placeholders})
                """,
                tuple(product_ids),
            )
            review_rows = cur.fetchall()
    finally:
        conn.close()

    ingredients_by_product: dict[int, list[dict[str, Any]]] = {}
    for row in ingredient_rows:
        ingredients_by_product.setdefault(int(row["product_id"]), []).append(row)

    reviews_by_product = {
        int(row["product_id"]): row
        for row in review_rows
    }

    products = [
        VanityProduct(
            product_id=int(row["product_id"]),
            brand_name=row.get("brand_name"),
            brand_name_kor=row.get("brand_name_kor"),
            product_name=row.get("product_name"),
            product_name_kor=row.get("product_name_kor"),
            category=row.get("category"),
            function=row.get("function"),
            price=row.get("price"),
            ingredients=ingredients_by_product.get(int(row["product_id"]), []),
            review=reviews_by_product.get(int(row["product_id"]), {}),
        )
        for row in product_rows
    ]
    product_by_id = {product.product_id: product for product in products}
    return [product_by_id[product_id] for product_id in product_ids if product_id in product_by_id]


def load_vanity_context(
    user_id: int,
    result_id: int | None = None,
    vanity_product_ids: list[int] | None = None,
) -> VanityContext:
    profile = load_user_profile(user_id)
    skin_result = load_skin_result(user_id=user_id, result_id=result_id)
    product_ids = vanity_product_ids or load_vanity_product_ids(user_id)
    products = load_products(product_ids)

    if not products:
        raise ValueError(f"USER_VANITY products not found for user_id={user_id}")

    return VanityContext(
        user_id=user_id,
        profile=profile,
        skin_result=skin_result,
        products=products,
    )

