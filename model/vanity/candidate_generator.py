from __future__ import annotations

from typing import Any

import pandas as pd

from model.recommendation.kg_pipeline.neo4j_skincare.config import driver
from model.recommendation.kg_pipeline.neo4j_skincare.graph.load_graph import create_user_session
from model.recommendation.kg_pipeline.neo4j_skincare.rerank.hard_filter import hard_filter
from model.recommendation.kg_pipeline.neo4j_skincare.rerank.soft_score import soft_score
from model.recommendation.kg_pipeline.neo4j_skincare.services.user_data import (
    _load_candidates_from_embedding,
    _normalize_gender,
)
from model.vanity.data_loader import (
    load_skin_result,
    load_user_allergies,
    load_user_profile,
    load_wishlist_product_keys,
)


SKIN_DATA_KEY_MAP = {
    "dryness": "dryness_score",
    "pore": "pore_score",
    "wrinkle": "wrinkle_score",
    "pigmentation": "pigmentation_score",
    "sagging": "sagging_score",
    "acne": "acne_score",
}


def build_vanity_session_id(user_id: int, result_id: int) -> str:
    return f"vanity::user::{user_id}::result::{result_id}"


def skin_result_to_skin_data(skin_result: dict[str, Any]) -> dict[str, float]:
    return {
        concern: float(skin_result.get(score_key) or 0.0)
        for concern, score_key in SKIN_DATA_KEY_MAP.items()
    }


def prepare_vanity_candidates(
    user_id: int,
    result_id: int | None = None,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    profile = load_user_profile(user_id)
    skin_result = load_skin_result(user_id=user_id, result_id=result_id)
    resolved_result_id = int(skin_result["result_id"])
    image_id = skin_result.get("image_id")
    if image_id is None:
        raise ValueError(f"image_id not found for result_id={resolved_result_id}")

    gender = _normalize_gender(profile.get("gender"))
    candidates = _load_candidates_from_embedding(
        image_id=int(image_id),
        image_name=None,
        gender=gender,
    )

    session_id = build_vanity_session_id(user_id=user_id, result_id=resolved_result_id)
    with driver.session() as session:
        session.execute_write(
            create_user_session,
            session_id,
            skin_result_to_skin_data(skin_result),
            gender,
            load_user_allergies(user_id),
            load_wishlist_product_keys(user_id),
            profile.get("skin_type"),
        )

    filtered, _drop_log = hard_filter(
        candidates,
        session_id=session_id,
        gender=gender,
        total_budget_max=budget,
    )
    if filtered.empty:
        return []

    scored_rows = []
    for _, row in filtered.iterrows():
        scores = soft_score(
            product_key=row["product_key"],
            session_id=session_id,
            vector_score=float(row["score"]),
            user_id=user_id,
        )
        scored_rows.append({**row.to_dict(), **scores})

    reranked = pd.DataFrame(scored_rows).sort_values("S_rerank", ascending=False)
    return reranked.to_dict("records")
