from pathlib import Path
from typing import Any
import re

import pandas as pd
import pymysql

from model.recommendation.kg_pipeline.neo4j_skincare.config import MYSQL_DB, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER, RETRIEVAL_TOPK_PER_CATEGORY, SLOT_ORDER


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

def _norm_score(v: float | None) -> float:
    if v is None:
        return 0.0
    x = float(v)
    return x / 100.0 if x > 1.0 else x

def _norm_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    return re.sub(r"[^a-z0-9?-?]+", "", s)

def _norm_brand_name_key(brand: Any, name: Any) -> str:
    return f"{_norm_text(brand)}::{_norm_text(name)}"

def _normalize_gender(v: Any) -> str:
    g = str(v).strip().lower() if v is not None else ""
    gk = re.sub(r"\s+", "", g)
    female_tokens = {"female", "f", "woman", "women", "\uC5EC\uC131", "\uC5EC\uC790", "\uC5EC"}
    male_tokens = {"male", "m", "man", "men", "\uB0A8\uC131", "\uB0A8\uC790", "\uB0A8"}
    if gk in male_tokens:
        return "male"
    if gk in female_tokens:
        return "female"
    return "female"

def _load_product_catalog() -> dict[str, tuple[int, str, str]]:
    conn = _mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT product_id, brand_name, product_name
                FROM PRODUCT
                WHERE brand_name IS NOT NULL
                  AND product_name IS NOT NULL
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    catalog: dict[str, tuple[int, str, str]] = {}
    for r in rows:
        pid = r.get("product_id")
        brand = r.get("brand_name")
        name = r.get("product_name")
        if pid is None or brand is None or name is None:
            continue
        k = _norm_brand_name_key(brand, name)
        # keep first match deterministically
        if k not in catalog:
            catalog[k] = (int(pid), str(brand), str(name))
    return catalog

def _load_user_context(user_id: int, image_id: int | None = None) -> dict[str, Any]:
    conn = _mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM USER_PROFILE
                WHERE user_id = %s
                LIMIT 1
                """,
                (user_id,),
            )
            profile = cur.fetchone()
            if not profile:
                raise ValueError(f"USER_PROFILE not found for user_id={user_id}")
            gender = _normalize_gender(profile.get("gender"))
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
            allergies = [r["allergy_ingredient"] for r in cur.fetchall() if r.get("allergy_ingredient")]
            # Optional: wishlist may not exist in every local schema.
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
                wishlist_product_keys = [r["product_key"] for r in cur.fetchall() if r.get("product_key")]
            except pymysql.MySQLError:
                wishlist_product_keys = []
            if image_id is None:
                cur.execute(
                    """
                    SELECT image_id, storage_url
                    FROM USER_IMAGE
                    WHERE user_id = %s
                    ORDER BY uploaded_at DESC, image_id DESC
                    LIMIT 1
                    """,
                    (user_id,),
                )
                image_row = cur.fetchone()
                if not image_row:
                    raise ValueError(f"USER_IMAGE not found for user_id={user_id}")
                image_id = int(image_row["image_id"])
                storage_url = image_row.get("storage_url")
                requested_image_id = None
            else:
                cur.execute(
                    """
                    SELECT image_id, storage_url
                    FROM USER_IMAGE
                    WHERE user_id = %s AND image_id = %s
                    LIMIT 1
                    """,
                    (user_id, image_id),
                )
                image_row = cur.fetchone()
                if not image_row:
                    raise ValueError(f"USER_IMAGE not found for user_id={user_id}, image_id={image_id}")
                storage_url = image_row.get("storage_url")
                requested_image_id = int(image_id)
            cur.execute(
                """
                SELECT
                    result_id,
                    dryness_score, pore_score, wrinkle_score, pigmentation_score, sagging_score, acne_score
                FROM SKIN_ANALYSIS_RESULT
                WHERE user_id = %s AND image_id = %s
                ORDER BY analyzed_at DESC, result_id DESC
                LIMIT 1
                """,
                (user_id, image_id),
            )
            result = cur.fetchone()
            if not result:
                raise ValueError(
                    f"SKIN_ANALYSIS_RESULT not found for user_id={user_id}, image_id={image_id}"
                )
    finally:
        conn.close()
    # Normalize skin scores to 0-1 range
    skin_data = {
        "dryness": _norm_score(result.get("dryness_score")),
        "pore": _norm_score(result.get("pore_score")),
        "wrinkle": _norm_score(result.get("wrinkle_score")),
        "pigmentation": _norm_score(result.get("pigmentation_score")),
        "sagging": _norm_score(result.get("sagging_score")),
        "acne": _norm_score(result.get("acne_score")),
    }
    return {
        "gender": gender,
        "profile": profile or {},
        "allergies": allergies,
        "wishlist_product_keys": wishlist_product_keys,
        "skin_data": skin_data,
        "image_id": image_id,
        "requested_image_id": requested_image_id if 'requested_image_id' in locals() else image_id,
        "image_name": Path(storage_url).name if storage_url else None,
        "image_row": image_row or {},
        "result_id": int(result["result_id"]),
        "skin_result_raw": result or {},
    }

def _load_candidates_from_embedding(
    image_id: int | None, image_name: str | None, gender: str, k_per_category: int = RETRIEVAL_TOPK_PER_CATEGORY
) -> pd.DataFrame:
    if image_id is None:
        raise ValueError("image_id is required to load DB embedding candidates.")
    query = """
    SELECT
        e.query_category,
        COALESCE(e.category, p.category) AS category,
        e.brand AS Brand,
        COALESCE(p.brand_name_kor, p.brand_name) AS brand_name_kor,
        e.product_name,
        COALESCE(p.product_name_kor, p.product_name) AS product_name_kor,
        e.score,
        COALESCE(e.product_id, p.product_id) AS product_id,
        e.rank_in_category
    FROM RECOMMENDATION_CANDIDATE e
    LEFT JOIN PRODUCT p
      ON LOWER(TRIM(p.brand_name)) = LOWER(TRIM(e.brand))
     AND LOWER(TRIM(p.product_name)) = LOWER(TRIM(e.product_name))
    WHERE e.image_id = %s
    ORDER BY query_category, rank_in_category
    """
    conn = _mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (int(image_id),))
            rows = cur.fetchall()
        candidates = pd.DataFrame(rows)
    finally:
        conn.close()
    if candidates.empty:
        raise ValueError(f"No embedding candidates found in DB for image_id={image_id}.")
    # Canonicalize candidate brand/name to PRODUCT master for stronger DB/Neo4j matching.
    product_catalog = _load_product_catalog()
    if "product_id" not in candidates.columns:
        candidates["product_id"] = None
    for idx, r in candidates.iterrows():
        k = _norm_brand_name_key(r.get("Brand"), r.get("product_name"))
        hit = product_catalog.get(k)
        if not hit:
            continue
        pid, canon_brand, canon_name = hit
        candidates.at[idx, "product_id"] = pid
        candidates.at[idx, "Brand"] = canon_brand
        candidates.at[idx, "product_name"] = canon_name
    # Guard against malformed score values.
    score_raw = candidates["score"].astype(str).str.strip()
    score_raw = score_raw.str.extract(r"([-+]?\d*\.?\d+)")[0]
    candidates["score"] = pd.to_numeric(score_raw, errors="coerce")
    if candidates["score"].notna().sum() == 0:
        print(f"[warn] image_id={image_id}: score is invalid for all rows. Using rank_in_category fallback score.")
        rank_raw = candidates["rank_in_category"].astype(str).str.strip()
        rank_raw = rank_raw.str.extract(r"(\d+)")[0]
        candidates["rank_in_category"] = pd.to_numeric(rank_raw, errors="coerce")
        # If rank is also malformed, generate a fallback rank from current order.
        if candidates["rank_in_category"].notna().sum() == 0:
            candidates["rank_in_category"] = candidates.groupby("query_category").cumcount() + 1
        candidates = candidates[candidates["rank_in_category"].notna()].copy()
        if candidates.empty:
            raise ValueError(f"Embedding candidates for image_id={image_id} have no valid score or rank.")
        max_rank = float(candidates["rank_in_category"].max())
        candidates["score"] = (max_rank - candidates["rank_in_category"] + 1.0) / max_rank
    else:
        candidates = candidates[candidates["score"].notna()].copy()
        rank_raw = candidates["rank_in_category"].astype(str).str.strip()
        rank_raw = rank_raw.str.extract(r"(\d+)")[0]
        candidates["rank_in_category"] = pd.to_numeric(rank_raw, errors="coerce")
        if candidates["rank_in_category"].notna().sum() == 0:
            candidates["rank_in_category"] = candidates.groupby("query_category").cumcount() + 1
    def _norm_cat(v: Any) -> str:
        return " ".join(str(v).strip().lower().split())
    categories = []
    for _, cats in SLOT_ORDER.get(gender, SLOT_ORDER.get("female", [])):
        categories.extend(cats)
    canonical_by_norm = {_norm_cat(c): c for c in categories}
    # Alias mapping for common category notation differences.
    category_alias = {
        "toner+toner pads": "Toner+Toner Pads",
        "toner + toner pads": "Toner+Toner Pads",
        "toner pads": "Toner+Toner Pads",
        "toner pad": "Toner+Toner Pads",
        "essence/ampoule/serum": "Essences/Ampoules/Serums",
        "essences/ampoules/serums": "Essences/Ampoules/Serums",
        "essence": "Essences/Ampoules/Serums",
        "serum": "Essences/Ampoules/Serums",
        "ampoule": "Essences/Ampoules/Serums",
        "cream/gel": "Cream/Gel",
        "all in one": "All-In-One",
        "all-in-one": "All-In-One",
    }
    for k, v in category_alias.items():
        canonical_by_norm[_norm_cat(k)] = v
    candidates["query_category_norm"] = candidates["query_category"].map(_norm_cat)
    candidates["query_category_canonical"] = candidates["query_category_norm"].map(canonical_by_norm)
    candidates["category_norm"] = candidates["category"].map(_norm_cat)
    candidates["category_canonical"] = candidates["category_norm"].map(canonical_by_norm)
    matched = candidates[candidates["query_category_canonical"].notna()].copy()
    if matched.empty:
        # Fallback: keep original embedding categories instead of failing hard.
        print(
            f"[warn] No slot-category match for image_id={image_id}. "
            "Using original embedding categories."
        )
        candidates = candidates.drop(columns=["query_category_norm", "query_category_canonical"])
    else:
        matched["query_category"] = matched["query_category_canonical"]
        # Keep only rows where query slot category matches product category after canonicalization.
        before = len(matched)
        matched = matched[
            matched["category_canonical"].isna()
            | (matched["query_category_canonical"] == matched["category_canonical"])
        ].copy()
        dropped = before - len(matched)
        if dropped > 0:
            print(f"[warn] image_id={image_id}: dropped {dropped} rows due to query/category mismatch.")
        candidates = matched.drop(
            columns=[
                "query_category_norm",
                "query_category_canonical",
                "category_norm",
                "category_canonical",
            ]
        )
    # gender-slot prefilter: keep only categories defined in SLOT_ORDER for this gender
    allowed_norm = set(_norm_cat(c) for _, cats in SLOT_ORDER.get(gender, []) for c in cats)
    if allowed_norm:
        c_before = len(candidates)
        candidates = candidates[candidates["query_category"].map(_norm_cat).isin(allowed_norm)].copy()
        dropped = c_before - len(candidates)
        if dropped > 0:
            print(f"[info] gender slot prefilter dropped {dropped} candidates")

    candidates = (
        candidates.sort_values(["query_category", "score", "rank_in_category"], ascending=[True, False, True])
        .groupby("query_category", as_index=False)
        .head(k_per_category)
        .reset_index(drop=True)
    )
    return candidates

def _product_detail_map(product_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not product_ids:
        return {}
    placeholders = ",".join(["%s"] * len(product_ids))
    query = f"""
    SELECT
        p.product_id,
        p.`function` AS product_function,
        pr.pros_text,
        pr.cons_text,
        GROUP_CONCAT(DISTINCT i.ingredient_name ORDER BY i.ingredient_name SEPARATOR ', ') AS ingredients,
        GROUP_CONCAT(DISTINCT i.`function` ORDER BY i.ingredient_name SEPARATOR ' | ') AS key_ingredient_functions
    FROM PRODUCT p
    LEFT JOIN PRODUCT_REVIEW pr ON pr.product_id = p.product_id
    LEFT JOIN PRODUCT_INGREDIENT pi ON pi.product_id = p.product_id
    LEFT JOIN INGREDIENT i ON i.ingredient_id = pi.ingredient_id
    WHERE p.product_id IN ({placeholders})
    GROUP BY p.product_id, p.`function`, pr.pros_text, pr.cons_text
    """
    conn = _mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(query, product_ids)
            rows = cur.fetchall()
    finally:
        conn.close()
    return {int(r["product_id"]): r for r in rows}

def _print_run_context(ctx: dict[str, Any], user_id: int, candidates: pd.DataFrame | None = None) -> None:
    profile = ctx.get("profile", {}) or {}
    print("\n=== User Context ===")
    print(f"user_id={user_id}")
    print(
        "profile:",
        {
            "name": profile.get("name") or profile.get("user_name") or profile.get("nickname"),
            "gender_raw": profile.get("gender"),
            "gender_norm": ctx.get("gender"),
            "skin_type": profile.get("skin_type"),
            "skin_concern": profile.get("skin_concern"),
        },
    )
    print(
        "image:",
        {
            "requested_image_id": ctx.get("requested_image_id"),
            "resolved_image_id": ctx.get("image_id"),
            "image_name": ctx.get("image_name"),
            "image_row": ctx.get("image_row", {}),
        },
    )
    print(
        "skin_result:",
        {
            "result_id": ctx.get("result_id"),
            "normalized": ctx.get("skin_data", {}),
        },
    )
    print(
        "preference:",
        {
            "allergies": ctx.get("allergies", []),
            "wishlist_count": len(ctx.get("wishlist_product_keys", [])),
        },
    )
    if candidates is not None:
        qcats = sorted(set(str(x) for x in candidates.get("query_category", pd.Series(dtype=str)).dropna().tolist()))
        print(
            "embedding_candidates:",
            {
                "count": int(len(candidates)),
                "query_categories": qcats,
            },
        )

