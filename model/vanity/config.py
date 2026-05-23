from __future__ import annotations

import os

from model.recommendation.kg_pipeline.neo4j_skincare.config import SLOT_ORDER


DEFAULT_TOP_N = int(os.getenv("VANITY_TOP_N", "20"))

FIT_SCORE_WEIGHTS = {
    "concern_match_score": float(os.getenv("VANITY_WEIGHT_CONCERN", "0.50")),
    "skin_type_bonus": float(os.getenv("VANITY_WEIGHT_SKIN_TYPE", "0.20")),
    "review_score": float(os.getenv("VANITY_WEIGHT_REVIEW", "0.30")),
    "irritation_penalty": float(os.getenv("VANITY_WEIGHT_IRRITATION", "0.05")),
}

FIT_LABEL_THRESHOLDS = {
    "excellent": float(os.getenv("VANITY_THRESHOLD_EXCELLENT", "0.75")),
    "good": float(os.getenv("VANITY_THRESHOLD_GOOD", "0.65")),
    "so_so": float(os.getenv("VANITY_THRESHOLD_SO_SO", "0.50")),
    "weak": float(os.getenv("VANITY_THRESHOLD_WEAK", "0.30")),
}

FIT_LABELS = {
    "excellent": "excellent_match",
    "good": "good_match",
    "so_so": "so_so",
    "weak": "weak_match",
    "poor": "poor_match",
}

RECOMMEND_ACTIONS = {
    "excellent": "strong_keep",
    "good": "keep",
    "so_so": "neutral",
    "weak": "caution",
    "poor": "replace",
}

DEFAULT_GENDER = "female"


def get_slot_order(gender: str | None) -> list[tuple[str, list[str]]]:
    return SLOT_ORDER.get(gender or DEFAULT_GENDER, SLOT_ORDER[DEFAULT_GENDER])
