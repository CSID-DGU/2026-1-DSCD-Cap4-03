from __future__ import annotations

import json
from typing import Any

from model.vanity.config import FIT_LABELS, FIT_LABEL_THRESHOLDS, RECOMMEND_ACTIONS
from model.vanity.data_loader import load_vanity_context, mysql_connect
from model.vanity.fit_score import calculate_product_fit_scores
from model.vanity.schemas import ProductMatchResult, VanityContext, VanityPipelineInput, VanityProduct


def classify_fit_label(fit_score: float) -> tuple[str, str]:
    if fit_score >= FIT_LABEL_THRESHOLDS["excellent"]:
        return FIT_LABELS["excellent"], RECOMMEND_ACTIONS["excellent"]
    if fit_score >= FIT_LABEL_THRESHOLDS["good"]:
        return FIT_LABELS["good"], RECOMMEND_ACTIONS["good"]
    if fit_score >= FIT_LABEL_THRESHOLDS["so_so"]:
        return FIT_LABELS["so_so"], RECOMMEND_ACTIONS["so_so"]
    if fit_score >= FIT_LABEL_THRESHOLDS["weak"]:
        return FIT_LABELS["weak"], RECOMMEND_ACTIONS["weak"]
    return FIT_LABELS["poor"], RECOMMEND_ACTIONS["poor"]


def build_reason_tags(scores: dict[str, float]) -> list[str]:
    tags = []
    if scores.get("concern_match_score", 0.0) >= 0.6:
        tags.append("concern_match")
    if scores.get("skin_type_bonus", 0.0) >= 0.6:
        tags.append("skin_type_match")
    if scores.get("review_score", 0.0) >= 0.6:
        tags.append("review_match")
    return tags


def build_caution_tags(scores: dict[str, float]) -> list[str]:
    tags = []
    if scores.get("irritation_penalty", 0.0) >= 0.4:
        tags.append("irritation_check")
    if scores.get("concern_match_score", 0.0) < 0.3:
        tags.append("weak_concern_match")
    return tags


def analyze_product_match(product: VanityProduct, context: VanityContext) -> ProductMatchResult:
    scores = calculate_product_fit_scores(product, context)
    fit_score = scores["vanity_fit_score"]
    fit_label, recommend_action = classify_fit_label(fit_score)
    return ProductMatchResult(
        product_id=product.product_id,
        category=product.category,
        brand_name=product.brand_name_kor or product.brand_name,
        product_name=product.product_name_kor or product.product_name,
        scores=scores,
        vanity_fit_score=fit_score,
        fit_label=fit_label,
        recommend_action=recommend_action,
        reason_tags=build_reason_tags(scores),
        caution_tags=build_caution_tags(scores),
    )


def run_skin_match(
    pipeline_input: VanityPipelineInput,
    save_result: bool = False,
) -> dict[str, Any]:
    context = load_vanity_context(
        user_id=pipeline_input.user_id,
        result_id=pipeline_input.result_id,
        vanity_product_ids=pipeline_input.vanity_product_ids,
    )
    matches = [analyze_product_match(product, context) for product in context.products]

    match_session_id = None
    if save_result:
        match_session_id = save_skin_match_result(
            user_id=pipeline_input.user_id,
            result_id=int(context.skin_result["result_id"]),
            matches=matches,
        )

    return {
        "match_session_id": match_session_id,
        "user_id": pipeline_input.user_id,
        "result_id": int(context.skin_result["result_id"]),
        "product_match_results": [
            {
                "product_id": item.product_id,
                "category": item.category,
                "brand_name": item.brand_name,
                "product_name": item.product_name,
                "vanity_fit_score": item.vanity_fit_score,
                "scores": item.scores,
                "fit_label": item.fit_label,
                "recommend_action": item.recommend_action,
                "reason_tags": item.reason_tags,
                "caution_tags": item.caution_tags,
            }
            for item in matches
        ],
    }


def save_skin_match_result(
    user_id: int,
    result_id: int,
    matches: list[ProductMatchResult],
) -> int:
    conn = mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO VANITY_MATCH_SESSION (user_id, result_id)
                VALUES (%s, %s)
                """,
                (user_id, result_id),
            )
            match_session_id = int(cur.lastrowid)

            for match in matches:
                cur.execute(
                    """
                    INSERT INTO VANITY_MATCH_ITEM (
                        match_session_id,
                        product_id,
                        vanity_fit_score,
                        concern_match_score,
                        skin_type_bonus,
                        review_score,
                        irritation_penalty,
                        fit_label,
                        recommend_action,
                        reason_tags,
                        caution_tags
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        match_session_id,
                        match.product_id,
                        match.vanity_fit_score,
                        match.scores.get("concern_match_score"),
                        match.scores.get("skin_type_bonus"),
                        match.scores.get("review_score"),
                        match.scores.get("irritation_penalty"),
                        match.fit_label,
                        match.recommend_action,
                        json.dumps(match.reason_tags, ensure_ascii=False),
                        json.dumps(match.caution_tags, ensure_ascii=False),
                    ),
                )
    finally:
        conn.close()
    return match_session_id
