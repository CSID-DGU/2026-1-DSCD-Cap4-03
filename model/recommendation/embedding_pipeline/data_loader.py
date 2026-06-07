# data_loader.py
from __future__ import annotations

import pandas as pd
import pymysql

from .config import MYSQL_DB, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER
from .corpus_builder import normalize_text


def connect_mysql():
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

# Data loading from MySQL DB : Cosmetic
def load_corpus_from_db() -> pd.DataFrame:
    query = """
    SELECT
        p.product_id,
        p.brand_name AS Brand,
        p.product_name AS product_name,
        p.category AS Category,
        p.`function` AS `Function`,
        p.price AS price,
        GROUP_CONCAT(DISTINCT i.ingredient_name ORDER BY i.ingredient_name SEPARATOR ', ') AS ingredients
    FROM PRODUCT p
    LEFT JOIN PRODUCT_INGREDIENT pi ON pi.product_id = p.product_id
    LEFT JOIN INGREDIENT i ON i.ingredient_id = pi.ingredient_id
    WHERE p.brand_name IS NOT NULL
      AND p.product_name IS NOT NULL
      AND p.category IS NOT NULL
    GROUP BY
        p.product_id, p.brand_name, p.product_name, p.category, p.`function`, p.price
    """
    conn = connect_mysql()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("PRODUCT corpus is empty in DB.")

    for c in ["Brand", "product_name", "Category", "Function", "ingredients"]:
        if c in df.columns:
            df[c] = df[c].map(normalize_text)
    return df

# Data loading from MySQL DB : Skin Query
def load_skin_queries_from_db() -> pd.DataFrame:
    query = """
    SELECT
        ui.image_id,
        ui.user_id,
        COALESCE(LOWER(up.gender), 'female') AS gender,
        ui.storage_url,
        sar.dryness_score,
        sar.pore_score,
        sar.wrinkle_score,
        sar.pigmentation_score,
        sar.sagging_score,
        sar.acne_score
    FROM USER_IMAGE ui
    LEFT JOIN USER_PROFILE up ON up.user_id = ui.user_id
    JOIN (
        SELECT image_id, MAX(result_id) AS result_id
        FROM SKIN_ANALYSIS_RESULT
        GROUP BY image_id
    ) latest ON latest.image_id = ui.image_id
    JOIN SKIN_ANALYSIS_RESULT sar ON sar.result_id = latest.result_id
    ORDER BY ui.image_id
    """
    conn = connect_mysql()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("SKIN_ANALYSIS_RESULT join USER_IMAGE is empty in DB.")
    return df
