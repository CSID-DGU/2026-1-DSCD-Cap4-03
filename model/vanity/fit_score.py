from __future__ import annotations

import math
from typing import Any

from model.vanity.config import FIT_SCORE_WEIGHTS
from model.vanity.schemas import VanityContext, VanityProduct


CONCERN_KEYWORDS = {
    "dryness": [
        "보습", "수분", "촉촉", "건조", "장벽", "세라마이드",
        "moisturizing", "moisturizer", "humectant", "emollient",
        "skin-identical", "squalane", "glycerin", "panthenol", "betaine",
    ],
    "acne": [
        "트러블", "여드름", "진정", "피지", "살리실릭", "병풀",
        "anti-acne", "antimicrobial", "antibacterial", "soothing",
        "salicylic", "allantoin", "panthenol", "phytosphingosine",
    ],
    "pore": [
        "모공", "피지", "각질", "블랙헤드",
        "pore", "sebum", "exfoliation", "exfoliating", "mattifier",
        "absorbent", "silica", "lha", "aha", "bha",
    ],
    "pigmentation": [
        "미백", "잡티", "톤업", "기미", "색소", "비타민",
        "brightening", "tone", "toning", "vitamin", "niacinamide",
    ],
    "wrinkle": [
        "주름", "탄력", "링클", "안티에이징", "레티놀",
        "wrinkle", "firming", "anti-aging", "retinol", "collagen",
    ],
    "sagging": [
        "탄력", "리프팅", "처짐", "퍼밍",
        "firming", "lifting", "elasticity", "collagen",
    ],
}

SKIN_TYPE_KEYWORDS = {
    "dry": [
        "보습", "수분", "촉촉", "장벽", "세라마이드",
        "moisturizing", "moisturizer", "humectant", "emollient", "skin-identical",
    ],
    "oily": [
        "산뜻", "피지", "유분", "젤", "가벼운",
        "sebum", "pore", "mattifier", "absorbent", "exfoliation",
        "gel", "lightweight", "silica", "anti-acne",
    ],
    "combination": [
        "밸런스", "수분", "산뜻", "유수분",
        "balance", "moisturizer", "humectant", "lightweight", "soothing",
    ],
    "sensitive": [
        "민감", "저자극", "진정", "순한", "병풀",
        "sensitive", "soothing", "allantoin", "panthenol", "cica",
    ],
    "normal": ["데일리", "무난", "수분", "daily", "moisturizer", "humectant"],
}

PROFILE_SKIN_TYPE_ALIASES = {
    "건성": "dry",
    "지성": "oily",
    "복합성": "combination",
    "수부지": "combination",
    "민감성": "sensitive",
    "민감": "sensitive",
    "중성": "normal",
    "dry": "dry",
    "oily": "oily",
    "combination": "combination",
    "sensitive": "sensitive",
    "normal": "normal",
}

def clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def product_text(product: VanityProduct) -> str:
    chunks = [
        product.brand_name,
        product.brand_name_kor,
        product.product_name,
        product.product_name_kor,
        product.category,
        product.function,
    ]
    for ingredient in product.ingredients:
        chunks.extend(
            [
                ingredient.get("ingredient_name"),
                ingredient.get("function"),
                ingredient.get("rating"),
                ingredient.get("irritation"),
                ingredient.get("comedogenicity"),
            ]
        )
    review = product.review or {}
    chunks.extend(str(review.get(k) or "") for k in review.keys())
    return normalize_text(" ".join(str(x) for x in chunks if x))


def top_skin_concerns(skin_result: dict[str, Any], top_n: int = 3) -> list[str]:
    score_pairs = []
    for key, value in skin_result.items():
        if key.endswith("_score"):
            score_pairs.append((key.replace("_score", ""), float(value or 0.0)))
    score_pairs.sort(key=lambda item: item[1], reverse=True)
    return [key for key, _ in score_pairs[:top_n]]


def calculate_concern_match_score(product: VanityProduct, context: VanityContext) -> float:
    text = product_text(product)
    concerns = top_skin_concerns(context.skin_result)
    if not concerns:
        return 0.0

    weighted_hits = 0.0
    total_weight = 0.0
    for concern in concerns:
        concern_weight = float(context.skin_result.get(f"{concern}_score") or 0.0)
        keywords = CONCERN_KEYWORDS.get(concern, [])
        if not keywords:
            continue
        hit_count = sum(1 for keyword in keywords if keyword.lower() in text)
        hit_ratio = min(hit_count / 3.0, 1.0)
        weighted_hits += concern_weight * hit_ratio
        total_weight += concern_weight

    if total_weight == 0.0:
        return 0.0
    return clamp01(weighted_hits / total_weight)


def calculate_skin_type_bonus(product: VanityProduct, context: VanityContext) -> float:
    skin_type_raw = normalize_text(context.profile.get("skin_type"))
    skin_type = PROFILE_SKIN_TYPE_ALIASES.get(skin_type_raw, skin_type_raw)
    keywords = SKIN_TYPE_KEYWORDS.get(skin_type, [])
    if not keywords:
        return 0.5
    text = product_text(product)
    hit_count = sum(1 for keyword in keywords if keyword.lower() in text)
    return clamp01(min(hit_count / 3.0, 1.0))


def calculate_review_score(product: VanityProduct, context: VanityContext) -> float:
    review = product.review or {}
    if not review:
        return 0.5

    pros_blob = " ".join(str(review.get(k) or "") for k in review if k.startswith("pro") or k == "pros_text")
    cons_blob = " ".join(str(review.get(k) or "") for k in review if k.startswith("con") or k == "cons_text")
    pros = normalize_text(pros_blob)
    cons = normalize_text(cons_blob)

    concerns = top_skin_concerns(context.skin_result)
    pos_hits = 0
    neg_hits = 0
    for concern in concerns:
        for keyword in CONCERN_KEYWORDS.get(concern, []):
            keyword_l = keyword.lower()
            if keyword_l in pros:
                pos_hits += 1
            if keyword_l in cons:
                neg_hits += 1

    raw = (pos_hits - neg_hits) / float(pos_hits + neg_hits + 1)
    return clamp01((raw + 1.0) / 2.0)


def _to_risk(value: Any) -> float:
    if value is None:
        return 0.0
    text = normalize_text(value)
    if "-" in text:
        parts = [part.strip() for part in text.split("-", 1)]
        try:
            numeric = sum(float(part) for part in parts if part) / len([part for part in parts if part])
            return clamp01(numeric / 5.0)
        except (ValueError, ZeroDivisionError):
            pass
    try:
        numeric = float(text)
        return clamp01(numeric / 5.0)
    except ValueError:
        pass
    if any(token in text for token in ["high", "높", "위험", "주의"]):
        return 1.0
    if any(token in text for token in ["medium", "보통"]):
        return 0.5
    if any(token in text for token in ["low", "낮", "없", "안전"]):
        return 0.0
    return 0.0


def calculate_irritation_penalty(product: VanityProduct) -> float:
    if not product.ingredients:
        return 0.0
    risks = []
    for ingredient in product.ingredients:
        risks.append(_to_risk(ingredient.get("irritation")))
        risks.append(_to_risk(ingredient.get("comedogenicity")))
    if not risks:
        return 0.0
    return clamp01(sum(risks) / len(risks))


def calculate_vanity_fit_score(component_scores: dict[str, float]) -> float:
    positive = (
        FIT_SCORE_WEIGHTS["concern_match_score"] * component_scores["concern_match_score"]
        + FIT_SCORE_WEIGHTS["skin_type_bonus"] * component_scores["skin_type_bonus"]
        + FIT_SCORE_WEIGHTS["review_score"] * component_scores["review_score"]
    )
    negative = FIT_SCORE_WEIGHTS["irritation_penalty"] * component_scores["irritation_penalty"]
    max_positive = (
        FIT_SCORE_WEIGHTS["concern_match_score"]
        + FIT_SCORE_WEIGHTS["skin_type_bonus"]
        + FIT_SCORE_WEIGHTS["review_score"]
    )
    if max_positive <= 0:
        return 0.0
    return clamp01((positive - negative) / max_positive)


def calculate_product_fit_scores(product: VanityProduct, context: VanityContext) -> dict[str, float]:
    scores = {
        "concern_match_score": calculate_concern_match_score(product, context),
        "skin_type_bonus": calculate_skin_type_bonus(product, context),
        "review_score": calculate_review_score(product, context),
        "irritation_penalty": calculate_irritation_penalty(product),
    }
    scores["vanity_fit_score"] = calculate_vanity_fit_score(scores)
    return {key: round(float(value), 4) for key, value in scores.items()}
