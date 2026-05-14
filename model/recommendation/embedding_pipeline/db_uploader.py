# db_uploader.py - Upload recommendation candidates to MySQL database
from __future__ import annotations

from typing import Iterable

import pandas as pd
import pymysql

from .config import MYSQL_DB, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS RECOMMENDATION_CANDIDATE (
    candidate_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    image_id INT NOT NULL,
    rank_in_category INT NOT NULL,
    product_id BIGINT NOT NULL,
    query_category VARCHAR(100) NOT NULL,
    score DECIMAL(12,8) NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_recommendation_candidate_image (image_id),
    KEY idx_recommendation_candidate_product (product_id),
    KEY idx_recommendation_candidate_category (query_category),
    KEY idx_recommendation_candidate_rank (rank_in_category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


INSERT_SQL = """
INSERT INTO RECOMMENDATION_CANDIDATE (
    image_id, rank_in_category, product_id, query_category, score
) VALUES (%s, %s, %s, %s, %s)
"""


REQUIRED_COLUMNS = [
    "image_id",
    "rank_in_category",
    "product_id",
    "query_category",
    "score",
]


def _mysql_connect():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )


def _normalize_candidate_rows(df: pd.DataFrame) -> list[tuple]:
    rows = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in rows.columns:
            raise ValueError(f"Missing required column: {col}")

    rows["image_id"] = pd.to_numeric(rows["image_id"], errors="coerce")
    rows["rank_in_category"] = pd.to_numeric(rows["rank_in_category"], errors="coerce")
    rows["product_id"] = pd.to_numeric(rows["product_id"], errors="coerce")
    rows["score"] = pd.to_numeric(rows["score"], errors="coerce")

    for col in ["query_category"]:
        rows[col] = rows[col].map(lambda v: None if pd.isna(v) or str(v).strip() == "" else str(v).strip())

    rows = rows.dropna(subset=["image_id", "rank_in_category", "product_id", "query_category", "score"]).copy()
    if rows.empty:
        return []

    rows["image_id"] = rows["image_id"].astype(int)
    rows["rank_in_category"] = rows["rank_in_category"].astype(int)
    rows["product_id"] = rows["product_id"].astype(int)

    return [
        (
            int(r.image_id),
            int(r.rank_in_category),
            int(r.product_id),
            r.query_category,
            float(r.score),
        )
        for r in rows.itertuples(index=False)
    ]


def _distinct_image_ids(prepared_rows: Iterable[tuple]) -> list[int]:
    return sorted({int(row[0]) for row in prepared_rows})


def upload_recommendation_candidates(df: pd.DataFrame) -> int:
    prepared_rows = _normalize_candidate_rows(df)
    if not prepared_rows:
        print("[db] no valid candidate rows to upload")
        return 0

    image_ids = _distinct_image_ids(prepared_rows)
    placeholders = ",".join(["%s"] * len(image_ids))

    conn = _mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(f"DELETE FROM RECOMMENDATION_CANDIDATE WHERE image_id IN ({placeholders})", image_ids)
            cur.executemany(INSERT_SQL, prepared_rows)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"[db] uploaded {len(prepared_rows)} rows to RECOMMENDATION_CANDIDATE for image_ids={image_ids}")
    return len(prepared_rows)
