from __future__ import annotations

from app.db.memory import store


def _application_guide(category: str) -> str:
    guides = {
        "토너": "세안 후 화장솜이나 손으로 얼굴 전체에 부드럽게 펴 발라주세요. 가볍게 두드려 흡수시켜 주세요.",
        "에멀젼": "토너 흡수 후 적당량을 손에 덜어 얼굴에 펴 발라주세요.",
        "앰플": "토너 다음 단계에서 2~3방울을 얼굴에 고르게 펴 발라 흡수시켜 주세요.",
        "크림": "마지막 단계에서 적당량을 덜어 피부결 따라 마무리해 주세요.",
    }
    return guides.get(category, "적당량을 얼굴 전체에 부드럽게 사용해 주세요.")


def build_mock_routines() -> list[dict]:
    products = store.products
    best_pick = [products[1001], products[1004], products[1002], products[1003]]
    budget_pick = [products[1005], products[1002], products[1003]]

    def to_products(rows: list[dict], pm_only: bool = False) -> list[dict]:
        items = []
        for index, product in enumerate(rows, start=1):
            items.append(
                {
                    "product_id": product["product_id"],
                    "step": index,
                    "application_guide": _application_guide(product["category"]),
                    "time_tag": "pm" if pm_only and index == 1 else None,
                }
            )
        return items

    return [
        {
            "routine_id": "r001",
            "type": "best",
            "label": "AI BEST 루틴",
            "routine_time": "both",
            "total_cost": sum(item["price"] for item in best_pick),
            "duration": 5,
            "ai_description": "AM+PM 공용 루틴으로 토너, 에멀젼, 앰플, 크림 4단계로 구성되어 보습 강화와 모공 관리에 맞춰 추천되었습니다.",
            "products": to_products(best_pick, pm_only=False),
        },
        {
            "routine_id": "r002",
            "type": "budget",
            "label": "가성비 루틴",
            "routine_time": "pm",
            "total_cost": sum(item["price"] for item in budget_pick),
            "duration": 4,
            "ai_description": "PM 전용 루틴으로 저자극 성분 위주로 구성해 취침 전 집중 보습과 피부 장벽 강화에 초점을 맞췄습니다.",
            "products": to_products(budget_pick, pm_only=True),
        },
    ]


def build_recommendation_explanation(session: dict) -> dict:
    return {
        "llm_model": "mock-llm-recommendation-v1",
        "prompt_version": "recommendation-explanation-v1",
        "summary_text": "현재 피부 상태를 기준으로 진정, 보습, 장벽 케어 중심 루틴을 구성했습니다.",
        "usage_guide_text": "토너부터 에센스, 로션, 크림 순서로 사용하고 자극감이 있으면 빈도를 줄이세요.",
        "warning_text": "피부 자극이 느껴질 경우 새로운 제품은 한 번에 여러 개 추가하지 마세요.",
    }
