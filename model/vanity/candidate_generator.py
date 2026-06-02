from __future__ import annotations

from typing import Any

import pandas as pd

from model.recommendation.embedding_pipeline.config import DEFAULT_OUTPUT_DIR, MODEL_NAME, TOPK_PER_CATEGORY
from model.recommendation.embedding_pipeline.data_loader import load_corpus_from_db
from model.recommendation.embedding_pipeline.db_uploader import upload_recommendation_candidates
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
    mysql_connect,
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


def recommendation_candidates_exist(image_id: int) -> bool:
    conn = mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS candidate_count
                FROM RECOMMENDATION_CANDIDATE
                WHERE image_id = %s
                """,
                (image_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return int((row or {}).get("candidate_count") or 0) > 0


def build_single_skin_query_row(
    user_id: int,
    image_id: int,
    gender: str,
    skin_result: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": image_id,
                "user_id": user_id,
                "gender": gender,
                "storage_url": None,
                "dryness_score": skin_result.get("dryness_score"),
                "pore_score": skin_result.get("pore_score"),
                "wrinkle_score": skin_result.get("wrinkle_score"),
                "pigmentation_score": skin_result.get("pigmentation_score"),
                "sagging_score": skin_result.get("sagging_score"),
                "acne_score": skin_result.get("acne_score"),
            }
        ]
    )


def build_candidate_upload_df(retrieval_df: pd.DataFrame) -> pd.DataFrame:
    return retrieval_df.rename(
        columns={
            "Brand": "brand",
            "Category": "category",
            "Function": "function",
        }
    )[
        [
            "image_id",
            "user_id",
            "rank_in_category",
            "product_id",
            "query_category",
            "brand",
            "product_name",
            "category",
            "function",
            "score",
        ]
    ]


def generate_and_save_recommendation_candidates(
    user_id: int,
    image_id: int,
    gender: str,
    skin_result: dict[str, Any],
) -> int:
    # Heavy dependency is imported only when fallback generation is needed.
    try:
        from model.recommendation.embedding_pipeline.retriever import run_retrieval
    except ModuleNotFoundError as exc:
        if exc.name == "sentence_transformers":
            raise RuntimeError(
                "RECOMMENDATION_CANDIDATE is missing and embedding fallback requires "
                "sentence-transformers. Install the embedding dependencies or generate "
                "recommendation candidates before running Vanity-Based Routine."
            ) from exc
        raise

    emb_path = DEFAULT_OUTPUT_DIR / f"cosmetic_emb_{MODEL_NAME.replace('/', '_')}.npy"
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus_df = load_corpus_from_db()
    skin_df = build_single_skin_query_row(
        user_id=user_id,
        image_id=image_id,
        gender=gender,
        skin_result=skin_result,
    )
    result_df = run_retrieval(
        corpus_df=corpus_df,
        skin_df=skin_df,
        topk_per_category=TOPK_PER_CATEGORY,
        model_name=MODEL_NAME,
        emb_path=emb_path,
    )
    if result_df.empty:
        raise ValueError(f"No embedding candidates generated for image_id={image_id}")

    return upload_recommendation_candidates(build_candidate_upload_df(result_df))


def load_or_generate_recommendation_candidates(
    user_id: int,
    image_id: int,
    gender: str,
    skin_result: dict[str, Any],
) -> pd.DataFrame:
    if not recommendation_candidates_exist(image_id):
        print(f"[vanity] no candidates for image_id={image_id}; generating embedding candidates")
        generate_and_save_recommendation_candidates(
            user_id=user_id,
            image_id=image_id,
            gender=gender,
            skin_result=skin_result,
        )
    else:
        print(f"[vanity] reuse existing candidates for image_id={image_id}")

    return _load_candidates_from_embedding(
        image_id=image_id,
        image_name=None,
        gender=gender,
    )


def prepare_vanity_candidates(
    user_id: int,
    result_id: int | None = None,
    budget: int | None = None,
    total_budget_min: int | None = None,
    total_budget_max: int | None = None,
    slot_budget_min_map: dict[str, int] | None = None,
    slot_budget_max_map: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    profile = load_user_profile(user_id)
    skin_result = load_skin_result(user_id=user_id, result_id=result_id)
    resolved_result_id = int(skin_result["result_id"])
    image_id = skin_result.get("image_id")
    if image_id is None:
        raise ValueError(f"image_id not found for result_id={resolved_result_id}")

    gender = _normalize_gender(profile.get("gender"))
    candidates = load_or_generate_recommendation_candidates(
        user_id=user_id,
        image_id=int(image_id),
        gender=gender,
        skin_result=skin_result,
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
        total_budget_min=total_budget_min,
        total_budget_max=total_budget_max if total_budget_max is not None else budget,
        slot_budget_min_map=slot_budget_min_map,
        slot_budget_max_map=slot_budget_max_map,
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
