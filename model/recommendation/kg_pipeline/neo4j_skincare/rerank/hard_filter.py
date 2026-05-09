from __future__ import annotations

from typing import Any

import pandas as pd
import pymysql

from config import MYSQL_DB, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER, SLOT_ORDER, driver

HF1_BATCH_QUERY = """
UNWIND $product_keys AS pk
MATCH (u:UserSession {session_id: $sid})
OPTIONAL MATCH (u)-[:HAS_ALLERGY]->(i:Ingredient)<-[:CONTAINS]-(p:Product)
WHERE toLower(trim(p.product_key)) = toLower(trim(pk))
RETURN pk AS product_key, count(i) AS hit
"""


CATEGORY_ALIAS = {
    "toner+toner pads": "toner+toner pads",
    "toner + toner pads": "toner+toner pads",
    "toner pads": "toner+toner pads",
    "toner pad": "toner+toner pads",
    "toner": "toner+toner pads",
    "toners": "toner+toner pads",
    "essense/ampoule/serum": "essences/ampoules/serums",
    "essence/ampoule/serum": "essences/ampoules/serums",
    "essenses/ampoules/serums": "essences/ampoules/serums",
    "essence": "essences/ampoules/serums",
    "essences": "essences/ampoules/serums",
    "ampoule": "essences/ampoules/serums",
    "ampoules": "essences/ampoules/serums",
    "serum": "essences/ampoules/serums",
    "serums": "essences/ampoules/serums",
    "all in one": "all-in-one",
    "all-in-one": "all-in-one",
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


def _norm_text(v: Any) -> str:
    if v is None:
        return ""
    base = " ".join(str(v).strip().lower().split())
    return CATEGORY_ALIAS.get(base, base)


def _build_product_key(row: pd.Series) -> str:
    if "product_key" in row and pd.notna(row["product_key"]):
        return str(row["product_key"])

    brand = row.get("Brand", row.get("brand_name"))
    name = row.get("product_name")
    if pd.isna(brand) or pd.isna(name):
        raise ValueError("Cannot infer product key. Required Brand/product_name columns are missing.")

    return f"{str(brand).strip()}::{str(name).strip()}"


def _allowed_categories(gender: str) -> set[str]:
    allowed: set[str] = set()
    for _, cats in SLOT_ORDER.get(gender, []):
        for c in cats:
            allowed.add(_norm_text(c))
    return allowed


def _core_slot_groups(gender: str) -> list[set[str]]:
    groups: list[set[str]] = []
    for slot_type, cats in SLOT_ORDER.get(gender, []):
        if slot_type != "core":
            continue
        groups.append(set(_norm_text(c) for c in cats))
    return groups


def _resolve_slot_budget(slot_budget_map: dict[str, float] | None, category: str) -> float | None:
    if not slot_budget_map:
        return None
    cat_norm = _norm_text(category)
    for k, v in slot_budget_map.items():
        if _norm_text(k) == cat_norm:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _is_out_of_budget(
    price: float | None,
    total_budget_min: float | None,
    total_budget_max: float | None,
    slot_budget_min: float | None,
    slot_budget_max: float | None,
) -> bool:
    if price is None:
        return False

    if slot_budget_min is not None and price < float(slot_budget_min):
        return True
    if slot_budget_max is not None and price > float(slot_budget_max):
        return True

    if total_budget_min is not None and price < float(total_budget_min):
        return True
    if total_budget_max is not None and price > float(total_budget_max):
        return True

    return False


def _batch_product_meta(candidates: pd.DataFrame) -> tuple[dict[int, float | None], dict[str, tuple[int, float | None]]]:
    pids = set()
    brand_names = set()
    for _, row in candidates.iterrows():
        pid_raw = row.get("product_id")
        if pd.notna(pid_raw):
            try:
                pids.add(int(float(pid_raw)))
            except (TypeError, ValueError):
                pass
        b = row.get("Brand", row.get("brand_name"))
        n = row.get("product_name")
        if pd.notna(b) and pd.notna(n):
            brand_names.add((str(b).strip().lower(), str(n).strip().lower()))

    by_pid: dict[int, float | None] = {}
    by_brand_name: dict[str, tuple[int, float | None]] = {}

    conn = _mysql_connect()
    try:
        with conn.cursor() as cur:
            if pids:
                placeholders = ",".join(["%s"] * len(pids))
                cur.execute(
                    f"SELECT product_id, price, brand_name, product_name FROM PRODUCT WHERE product_id IN ({placeholders})",
                    tuple(pids),
                )
                for r in cur.fetchall():
                    pid = int(r["product_id"])
                    price = float(r["price"]) if r.get("price") is not None else None
                    by_pid[pid] = price
                    k = f"{str(r.get('brand_name') or '').strip().lower()}::{str(r.get('product_name') or '').strip().lower()}"
                    by_brand_name[k] = (pid, price)

            if brand_names:
                cur.execute("SELECT product_id, price, brand_name, product_name FROM PRODUCT")
                for r in cur.fetchall():
                    k1 = str(r.get("brand_name") or "").strip().lower()
                    k2 = str(r.get("product_name") or "").strip().lower()
                    key = f"{k1}::{k2}"
                    if (k1, k2) in brand_names:
                        by_brand_name[key] = (
                            int(r["product_id"]),
                            float(r["price"]) if r.get("price") is not None else None,
                        )
    finally:
        conn.close()

    return by_pid, by_brand_name


def hard_filter(
    candidates: pd.DataFrame,
    session_id: str,
    gender: str = "female",
    total_budget: float | None = None,
    slot_budget_map: dict[str, float] | None = None,
    total_budget_min: float | None = None,
    total_budget_max: float | None = None,
    slot_budget_min_map: dict[str, float] | None = None,
    slot_budget_max_map: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep, drop_log = [], []
    allowed = _allowed_categories(gender)

    by_pid, by_brand_name = _batch_product_meta(candidates)

    # batch allergy hit check on Neo4j
    product_keys = []
    for _, row in candidates.iterrows():
        try:
            product_keys.append(_build_product_key(row))
        except Exception:
            pass

    allergy_hits: dict[str, int] = {}
    with driver.session() as s:
        rows = s.run(HF1_BATCH_QUERY, sid=session_id, product_keys=product_keys).data()
        for r in rows:
            allergy_hits[str(r.get("product_key"))] = int(r.get("hit") or 0)

    for _, row in candidates.iterrows():
        pid_key = _build_product_key(row)
        category = str(row.get("query_category", row.get("Category", "")))
        cat_norm = _norm_text(category)

        # HF3: SLOT_CATEGORY_MISMATCH (hard remove)
        if allowed and cat_norm not in allowed:
            drop_log.append({"product_key": pid_key, "category": category, "drop_reason": "HF3_SLOT_CATEGORY_MISMATCH"})
            continue

        # HF4: PRODUCT_UNRESOLVED (batch-resolved)
        resolved_pid = None
        price = None
        pid_raw = row.get("product_id")
        if pd.notna(pid_raw):
            try:
                pid = int(float(pid_raw))
                if pid in by_pid:
                    resolved_pid = pid
                    price = by_pid[pid]
            except (TypeError, ValueError):
                pass

        if resolved_pid is None:
            b = str(row.get("Brand", row.get("brand_name", ""))).strip().lower()
            n = str(row.get("product_name", "")).strip().lower()
            hit = by_brand_name.get(f"{b}::{n}")
            if hit:
                resolved_pid, price = hit

        if resolved_pid is None:
            drop_log.append({"product_key": pid_key, "category": category, "drop_reason": "HF4_PRODUCT_UNRESOLVED"})
            continue

        # HF1: ALLERGY_CONFLICT
        if allergy_hits.get(pid_key, 0) > 0:
            drop_log.append({"product_key": pid_key, "category": category, "drop_reason": "HF1_ALLERGY_CONFLICT"})
            continue

        # HF2: PRICE_EXCEED (min/max)
        # Backward compatibility: total_budget -> total_budget_max, slot_budget_map -> slot_budget_max_map
        eff_total_min = total_budget_min
        eff_total_max = total_budget_max if total_budget_max is not None else total_budget
        eff_slot_min = _resolve_slot_budget(slot_budget_min_map, category)
        eff_slot_max = _resolve_slot_budget(slot_budget_max_map, category)
        if eff_slot_max is None:
            eff_slot_max = _resolve_slot_budget(slot_budget_map, category)

        if _is_out_of_budget(
            price=price,
            total_budget_min=eff_total_min,
            total_budget_max=eff_total_max,
            slot_budget_min=eff_slot_min,
            slot_budget_max=eff_slot_max,
        ):
            drop_log.append({"product_key": pid_key, "category": category, "drop_reason": "HF2_PRICE_EXCEED"})
            continue

        keep.append({
            **row.to_dict(),
            "product_key": pid_key,
            "product_id": resolved_pid,
            "price": price,
        })

    keep_df = pd.DataFrame(keep)
    drop_df = pd.DataFrame(drop_log)

    # HF5: CORE_SLOT_EMPTY
    core_groups = _core_slot_groups(gender)
    if core_groups:
        present = set(_norm_text(v) for v in keep_df.get("query_category", pd.Series(dtype=str)).tolist())
        missing_groups = []
        for g in core_groups:
            if not (g & present):
                missing_groups.append(sorted(g))

        if missing_groups:
            miss_label = " || ".join(["|".join(m) for m in missing_groups])
            core_log = pd.DataFrame(
                [{"product_key": "", "category": miss_label, "drop_reason": "HF5_CORE_SLOT_EMPTY"}]
            )
            drop_df = core_log if drop_df.empty else pd.concat([drop_df, core_log], ignore_index=True)
            return pd.DataFrame(columns=keep_df.columns), drop_df

    return keep_df, drop_df
