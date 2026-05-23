from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from model.vanity.data_loader import load_skin_result, mysql_connect
from model.vanity.schemas import VanityRoutineResult


def _table_columns(table_name: str) -> set[str]:
    conn = mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SHOW COLUMNS FROM {table_name}")
            rows = cur.fetchall()
    finally:
        conn.close()
    return {str(row["Field"]) for row in rows}


def _routine_score(routine: VanityRoutineResult) -> float | None:
    scores = [
        item.product_score
        for item in routine.final_routine
        if item.product_score is not None
    ]
    if not scores:
        return None
    return round(sum(float(score) for score in scores) / len(scores), 4)


def _has_conflict(warnings: list[str]) -> bool:
    return any("conflict" in str(warning).lower() for warning in warnings)


def save_vanity_routine_result(
    user_id: int,
    result_id: int,
    routine: VanityRoutineResult,
    budget: int | None = None,
) -> int:
    skin_result = load_skin_result(user_id=user_id, result_id=result_id)
    image_id = skin_result.get("image_id")

    session_columns = _table_columns("RECOMMENDATION_SESSION")
    item_columns = _table_columns("RECOMMENDATION_ITEM")

    session_payload: dict[str, Any] = {
        "user_id": user_id,
        "image_id": image_id,
        "result_id": result_id,
        "strict_budget": 0,
        "total_budget_min": None,
        "total_budget_max": int(budget) if budget is not None else None,
        "slot_budget_min_json": None,
        "slot_budget_max_json": None,
        "budget_check_passed": 1,
        "session_status": "SUCCESS",
        "failure_reason": None,
    }
    if "recommendation_type" in session_columns:
        session_payload["recommendation_type"] = "vanity"

    conn = mysql_connect()
    try:
        with conn.cursor() as cur:
            session_cols = list(session_payload.keys())
            cur.execute(
                f"""
                INSERT INTO RECOMMENDATION_SESSION (
                    {", ".join(session_cols)}
                ) VALUES (
                    {", ".join(["%s"] * len(session_cols))}
                )
                """,
                tuple(session_payload[col] for col in session_cols),
            )
            session_id = int(cur.lastrowid)

            warnings = routine.warnings or []
            cur.execute(
                """
                INSERT INTO RECOMMENDATION_ROUTINE (
                    session_id, routine_rank, routine_label, ampm_mode, routine_score,
                    has_conflict, conflict_pairs
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    1,
                    "Vanity",
                    None,
                    _routine_score(routine),
                    1 if _has_conflict(warnings) else 0,
                    json.dumps(warnings, ensure_ascii=False) if warnings else None,
                ),
            )
            routine_id = int(cur.lastrowid)

            for item in routine.final_routine:
                item_dict = asdict(item)
                item_payload: dict[str, Any] = {
                    "routine_id": routine_id,
                    "slot_order": item.slot_order,
                    "category": item.category,
                    "product_id": item.product_id,
                    "product_score": item.product_score,
                    "time_tag": None,
                }
                if "source" in item_columns:
                    item_payload["source"] = item.source
                if "item_snapshot_json" in item_columns:
                    item_payload["item_snapshot_json"] = json.dumps(item_dict, ensure_ascii=False)

                item_cols = list(item_payload.keys())
                cur.execute(
                    f"""
                    INSERT INTO RECOMMENDATION_ITEM (
                        {", ".join(item_cols)}
                    ) VALUES (
                        {", ".join(["%s"] * len(item_cols))}
                    )
                    """,
                    tuple(item_payload[col] for col in item_cols),
                )
    finally:
        conn.close()

    return session_id
