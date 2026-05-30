from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path


SUMMARIES_PATH = Path(__file__).parent / "skin_summaries.json"
EXPLANATIONS_PATH = Path(__file__).parent / "recommendation_explanations.json"
VANITY_SKIN_MATCH_EXPLANATIONS_PATH = Path(__file__).parent / "vanity_skin_match_explanations.json"
VANITY_ROUTINE_EXPLANATIONS_PATH = Path(__file__).parent / "vanity_routine_explanations.json"


def load_skin_summaries() -> dict[int, dict]:
    if not SUMMARIES_PATH.exists():
        return {}
    try:
        with SUMMARIES_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return {int(key): value for key, value in raw.items()}


def save_skin_summaries(summaries: dict[int, dict]) -> None:
    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARIES_PATH.open("w", encoding="utf-8") as f:
        json.dump({str(key): value for key, value in summaries.items()}, f, ensure_ascii=False, indent=2)


def load_recommendation_explanations() -> dict[int, dict]:
    if not EXPLANATIONS_PATH.exists():
        return {}
    try:
        with EXPLANATIONS_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return {int(key): value for key, value in raw.items()}


def save_recommendation_explanations(explanations: dict[int, dict]) -> None:
    EXPLANATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXPLANATIONS_PATH.open("w", encoding="utf-8") as f:
        json.dump({str(key): value for key, value in explanations.items()}, f, ensure_ascii=False, indent=2)


def load_vanity_skin_match_explanations() -> dict[int, dict]:
    if not VANITY_SKIN_MATCH_EXPLANATIONS_PATH.exists():
        return {}
    try:
        with VANITY_SKIN_MATCH_EXPLANATIONS_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return {int(key): value for key, value in raw.items()}


def save_vanity_skin_match_explanations(explanations: dict[int, dict]) -> None:
    VANITY_SKIN_MATCH_EXPLANATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VANITY_SKIN_MATCH_EXPLANATIONS_PATH.open("w", encoding="utf-8") as f:
        json.dump({str(key): value for key, value in explanations.items()}, f, ensure_ascii=False, indent=2)


def load_vanity_routine_explanations() -> dict[int, dict]:
    if not VANITY_ROUTINE_EXPLANATIONS_PATH.exists():
        return {}
    try:
        with VANITY_ROUTINE_EXPLANATIONS_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return {int(key): value for key, value in raw.items()}


def save_vanity_routine_explanations(explanations: dict[int, dict]) -> None:
    VANITY_ROUTINE_EXPLANATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VANITY_ROUTINE_EXPLANATIONS_PATH.open("w", encoding="utf-8") as f:
        json.dump({str(key): value for key, value in explanations.items()}, f, ensure_ascii=False, indent=2)


@dataclass
class MemoryStore:
    users: dict[int, dict] = field(default_factory=dict)
    user_profiles: dict[int, dict] = field(default_factory=dict)
    user_allergies: dict[int, list[dict]] = field(default_factory=dict)
    user_images: dict[int, dict] = field(default_factory=dict)
    skin_results: dict[int, dict] = field(default_factory=dict)
    skin_summaries: dict[int, dict] = field(default_factory=load_skin_summaries)
    recommendation_sessions: dict[int, dict] = field(default_factory=dict)
    recommendation_explanations: dict[int, dict] = field(default_factory=load_recommendation_explanations)
    vanity_skin_match_explanations: dict[int, dict] = field(default_factory=load_vanity_skin_match_explanations)
    vanity_routine_explanations: dict[int, dict] = field(default_factory=load_vanity_routine_explanations)
    saved_routines: dict[int, dict] = field(default_factory=dict)
    wishlists: dict[int, set[int]] = field(default_factory=dict)
    products: dict[int, dict] = field(default_factory=dict)
    user_seq: count = field(default_factory=lambda: count(1))
    image_seq: count = field(default_factory=lambda: count(1))
    result_seq: count = field(default_factory=lambda: count(1))
    session_seq: count = field(default_factory=lambda: count(1))
    saved_routine_seq: count = field(default_factory=lambda: count(1))


store = MemoryStore()


def bootstrap_products() -> None:
    if store.products:
        return

    sample_products = [
        {
            "product_id": 1001,
            "brand_name": "유리아쥬",
            "product_name": "제모스 토너",
            "category": "토너",
            "price": 29000,
            "image_url": "https://example.com/products/1001.jpg",
            "tags": ["수분", "토닝", "히알루론산"],
            "ingredients": ["블루 히알루론산", "히알루론산 5종", "알로에베라"],
            "pros": ["풍부한 수분감", "레이어링 가능", "산뜻한 마무리"],
            "cons": ["건성 피부엔 추가 보습 필요"],
            "how_to_use": "세안 후 손바닥에 적당량을 덜어 얼굴 전체에 흡수시켜 주세요.",
            "apply_time": "30초",
        },
        {
            "product_id": 1002,
            "brand_name": "토리든",
            "product_name": "다이브인 세럼",
            "category": "앰플",
            "price": 28000,
            "image_url": "https://example.com/products/1002.jpg",
            "tags": ["수분", "진정", "세럼"],
            "ingredients": ["저분자 히알루론산", "판테놀", "알란토인"],
            "pros": ["흡수 빠름", "산뜻함", "여름철 사용 편함"],
            "cons": ["고보습은 아님"],
            "how_to_use": "토너 다음 단계에서 2~3방울을 얼굴에 펴 발라주세요.",
            "apply_time": "20초",
        },
        {
            "product_id": 1003,
            "brand_name": "닥터지",
            "product_name": "레드 블레미쉬 수딩 크림",
            "category": "크림",
            "price": 24000,
            "image_url": "https://example.com/products/1003.jpg",
            "tags": ["진정", "보습", "장벽"],
            "ingredients": ["병풀추출물", "판테놀", "세라마이드"],
            "pros": ["진정감 좋음", "무난한 제형", "사계절 사용 가능"],
            "cons": ["극건성엔 부족할 수 있음"],
            "how_to_use": "마지막 단계에서 적당량을 덜어 피부결 따라 마무리해 주세요.",
            "apply_time": "40초",
        },
        {
            "product_id": 1004,
            "brand_name": "에스트라",
            "product_name": "아토베리어 365 로션",
            "category": "에멀젼",
            "price": 31000,
            "image_url": "https://example.com/products/1004.jpg",
            "tags": ["장벽", "보습", "로션"],
            "ingredients": ["세라마이드", "콜레스테롤", "지방산"],
            "pros": ["장벽 케어 강점", "민감피부 사용 쉬움", "보습 지속력 좋음"],
            "cons": ["여름엔 다소 무겁게 느껴질 수 있음"],
            "how_to_use": "토너 후 적당량을 얼굴 전체에 펴 발라 흡수시켜 주세요.",
            "apply_time": "35초",
        },
        {
            "product_id": 1005,
            "brand_name": "아누아",
            "product_name": "어성초 77 토너패드",
            "category": "토너",
            "price": 23000,
            "image_url": "https://example.com/products/1005.jpg",
            "tags": ["진정", "패드", "각질"],
            "ingredients": ["어성초추출물", "판테놀", "알란토인"],
            "pros": ["사용 편리", "패드 타입", "진정감 무난"],
            "cons": ["패드 마찰 주의 필요"],
            "how_to_use": "세안 후 피부결 방향으로 가볍게 닦아낸 뒤 남은 에센스를 흡수시켜 주세요.",
            "apply_time": "25초",
        },
    ]

    for product in sample_products:
        store.products[product["product_id"]] = product


bootstrap_products()
