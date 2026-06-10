# routine_recommendation_llm.py

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_client import call_llm_json


PROMPT_VERSION = "routine_v8"
OUTPUT_DIR = "outputs/llm_results/routine_recommendation"

ROUTINE_RECOMMENDATION_SYSTEM_PROMPT = """
추천 모델의 스킨케어 루틴 결과를 사용자용 JSON으로 변환한다.

score는 0~1, 높을수록 고민이 두드러진다.
지표: acne=트러블, dryness=건조, sagging=처짐, pore=모공, pigmentation=색소침착, wrinkle=주름.
routine_type: best=완성도 추천, value=가격 효율 추천.
ampm_mode: am+pm=아침과 저녁 모두, am_only=아침 중심, pm_only=저녁 중심.
카테고리: Face Mists=미스트, Toner=토너, Emulsions=로션, Essences/Ampoules/Serums=세럼, Cream/Gel=크림, Eye Treatments=아이케어 제품, Facial Oils=페이셜 오일, Multi-balm=멀티밤, Shaving=쉐이빙 제품, All-In-One=올인원 제품.

공통 규칙:
- 입력 정보만 사용. 없는 제품 특징/리뷰/성분 효능/성분명/피부타입 생성 금지.
- 입력 routines 개수와 순서 유지. routine_type은 입력값 그대로 출력.
- 진단/치료/완치/질환/증상, 제품 효능 단정, 개발자용 필드명 노출 금지.
- Best와 Value 비교 금지.
- JSON만 출력. 마크다운 금지.

금지:
- 같은 뜻 괄호 반복 금지. 예: 아침·저녁(아침·저녁 겸용), 저녁 중심(저녁 중심)
- 화살표 단계 나열 금지. 예: 미스트→토너→로션→세럼→크림 6단계
- 단계 수 강조 금지. 예: 6단계로 구성합니다
- usage_guide를 제품명/브랜드명으로 시작 금지.
- 시간 표현 금지. 예: 10초, 30초, 1분, 1~2분, 잠시, 일정 시간
- 금지 key: summary, comparison_comment, routine_summary, recommend_reason, ampm_comment, product_list
- 금지 표현: 흐름, 중심에 두고, 살피는, 돕는, 자리 잡으면, 겉면, 피부 표면, 손에 끈적임

recommend_summary:
- 2문장 이내.
- 총 가격, 주요 피부 고민, 루틴 방향, 사용 시간대 포함.
- 단계 나열 대신 역할 중심으로 설명.
- am+pm: "아침과 저녁 모두 활용할 수 있어요"
- pm_only: "저녁에 집중해서 사용하기 좋은 구성이에요"
- pm_only_products가 있으면 제품명과 저녁 권장 이유를 짧게 포함.
- 예: 총 137,200원으로 건조 관리에 초점을 맞추고, 트러블과 모공 고민까지 함께 고려한 보습 중심 루틴이에요. 아침과 저녁 모두 활용할 수 있으며, 수분을 먼저 채운 뒤 보습과 아이케어로 마무리하는 구성입니다.

step_guides:
- items의 slot_order마다 1개. slot_order, usage_guide만 출력.
- usage_guide는 1~2문장.
- 제품명/브랜드명 대신 카테고리명으로 시작. 예: 미스트를, 토너를, 로션을, 세럼을, 크림을.
- 단계 역할 + 바르는 방법 + 다음 단계 기준 포함.
- 시간 표현 금지. "피부에 남는 느낌이 줄어들면", "물기가 줄어들면", "답답함이 남지 않게"처럼 기준만 설명.
- 구체적 동작 포함. 예: 손에 덜어, 얼굴 안쪽에서 바깥쪽으로, 얇게 펴 발라, 톡톡 두드리듯, 필요한 부위에 소량만.
- 예: 토너를 손에 덜어 얼굴 안쪽에서 바깥쪽으로 부드럽게 펴 발라 주세요. 피부결을 정돈하듯 흡수시킨 뒤 다음 단계로 이어가면 좋아요.

strengths:
- 2~3개.
- recommend_summary와 반복 금지.
- 괄호 반복 금지.
- best: 단계 균형, 주요 고민 반영, 아침과 저녁 활용성, 보습 마무리 중심.
- value: 가격 부담, 실용성, 저녁 집중 관리, 핵심 단계 구성 중심.
- 예: 아침과 저녁 모두 활용하기 쉬워 같은 관리 흐름을 꾸준히 유지하기 좋아요.

cautions:
- 1~3개.
- warnings가 있으면 제품명 중심으로 설명.
- 성분명은 필요 시 괄호 안에 짧게만 작성.
- pm_only_products에 ingredient가 있으면 "{제품명}에는 {성분명}이 포함되어 있어, 처음에는 저녁 루틴에서 먼저 사용해 보고 피부 반응을 확인해 주세요." 형태로 작성.
- 위험하다/사용하면 안 된다/포함된 제품으로 보이므로 표현 금지.
- 양 조절, 단계별 적응, 피부 반응 확인 중심.

출력:
{"routines":[{"routine_type":"best","recommend_summary":"...","step_guides":[{"slot_order":1,"usage_guide":"..."}],"strengths":["..."],"cautions":["..."]},{"routine_type":"value","recommend_summary":"...","step_guides":[{"slot_order":1,"usage_guide":"..."}],"strengths":["..."],"cautions":["..."]}]}
"""


REQUIRED_INPUT_KEYS = [
    "rec_session_id",
    "user_id",
    "image_id",
    "result_id",
    "recommended_at",
    "skin_scores",
    "top_skin_concerns",
    "routines"
]

REQUIRED_SKIN_SCORE_KEYS = [
    "acne_score",
    "dryness_score",
    "sagging_score",
    "pore_score",
    "pigmentation_score",
    "wrinkle_score"
]

REQUIRED_ROUTINE_KEYS = [
    "routine_type",
    "ampm_mode",
    "total_price",
    "items",
    "warnings",
    "pm_only_products"
]

REQUIRED_ITEM_KEYS = [
    "slot_order",
    "category",
    "brand",
    "product_name",
    "price"
]

REQUIRED_OUTPUT_ROUTINE_KEYS = [
    "routine_type",
    "recommend_summary",
    "step_guides",
    "strengths",
    "cautions"
]

INDICATOR_LABELS = {
    "acne": "트러블",
    "dryness": "건조",
    "sagging": "처짐",
    "pore": "모공",
    "pigmentation": "색소침착",
    "wrinkle": "주름"
}

USER_FRIENDLY_REPLACEMENTS = {
    "routine_score": "루틴 추천 점수",
    "product_score": "제품별 추천 점수",
    "total_price": "총 가격",
    "ampm_mode": "사용 시간대",
    "pm_only_products": "저녁 사용 권장 제품",
    "pm_only": "저녁 중심",
    "am+pm": "아침과 저녁 모두",
    "am_only": "아침 중심",
    "warnings": "주의 성분 조합",
    "warning": "주의 성분 조합",
    "skin_scores": "피부 분석 결과",
    "top_skin_concerns": "주요 피부 고민",
    "Best Routine": "Best 루틴",
    "Value Routine": "Value 루틴",
    "Face Mists": "미스트",
    "Toner": "토너",
    "Emulsions": "로션",
    "Essences/Ampoules/Serums": "세럼",
    "Cream/Gel": "크림",
    "Eye Treatments": "아이케어",
    "Facial Oils": "페이셜 오일",
    "Multi-balm": "멀티밤",
    "Multi-Balm": "멀티밤",
    "Shaving": "쉐이빙",
    "All-In-One": "올인원"
}

STYLE_REPLACEMENTS = {
    # 사용 시간대 중복 괄호 제거
    "아침·저녁( 아침·저녁 겸용 )": "아침과 저녁 모두",
    "아침·저녁(아침·저녁 겸용)": "아침과 저녁 모두",
    "아침·저녁 겸용(아침·저녁 겸용)": "아침과 저녁 모두 활용하기 쉬운 구성",
    "저녁 중심(저녁 중심)": "저녁 중심",
    "아침 중심(아침 중심)": "아침 중심",

    # 사용 시간대 표현 자연화
    "아침·저녁 모두 사용 가능하며": "아침과 저녁 모두 활용할 수 있으며",
    "아침·저녁 모두 사용할 수 있으며": "아침과 저녁 모두 활용할 수 있으며",
    "아침·저녁 겸용이라": "아침과 저녁 모두 활용하기 쉬워",
    "저녁 중심으로 핵심 단계 구성이라": "저녁에 집중해 핵심 단계만 챙길 수 있어",

    # 화살표 단계 나열 완화
    "미스트→토너→로션→세럼→크림→아이 세럼 6단계로 마무리합니다.": "수분을 먼저 채운 뒤 보습과 아이케어 단계로 마무리하는 구성입니다.",
    "미스트→토너→로션→세럼→크림→아이 세럼 6단계로 마무리해요.": "수분을 먼저 채운 뒤 보습과 아이케어 단계로 마무리하는 구성입니다.",
    "미스트→토너→에멀전→포어 세럼→수분 크림→페이셜 오일 6단계로 구성합니다.": "수분을 먼저 채우고 크림과 오일로 마무리하는 저녁 루틴입니다.",
    "미스트→토너→에멀전→포어 세럼→수분 크림→페이셜 오일 6단계로 구성했어요.": "수분을 먼저 채우고 크림과 오일로 마무리하는 저녁 루틴입니다.",
    "6단계로 마무리합니다.": "단계별로 차분히 마무리하는 구성입니다.",
    "6단계로 마무리해요.": "단계별로 차분히 마무리하는 구성입니다.",
    "6단계로 구성합니다.": "단계별로 차분히 구성한 루틴입니다.",
    "6단계로 구성했어요.": "단계별로 차분히 구성한 루틴입니다.",

    # 기존 표현 보정
    "건조를 중심에 두고": "건조 관리에 초점을 맞추고",
    "트러블과 모공도 함께 살피는": "트러블과 모공 고민까지 함께 고려한",
    "함께 살피는": "함께 고려한",
    "피부 표면이 차분해지면": "피부에 남는 느낌이 줄어들면",
    "표면이 지나치게 젖어 있지 않게 느껴지면": "피부에 남는 물기가 줄어들면",
    "표면이 너무 미끄럽지 않게 자리 잡으면": "피부에 남는 미끄러운 느낌이 줄어들면",
    "겉면이 과하게 남지 않을 정도로": "피부에 남는 양이 부담스럽지 않을 정도로",
    "손에 끈적임이 덜 느껴질 때까지": "피부에 끈적임이 덜 느껴질 때까지",
    "손에 남는 느낌이 줄어들면": "피부에 남는 느낌이 줄어들면",
    "손이 다시 가볍게 느껴질 정도로": "피부에 남는 느낌이 줄어들 정도로",
    "손이 쉽게 미끄러지지 않을 정도로": "피부에 남는 미끄러운 느낌이 줄어들 정도로",
    "피부에 스며드는 느낌이 줄어들면": "피부에 남는 느낌이 줄어들면",
    "겉도는 느낌이 줄어들면": "피부에 남는 느낌이 줄어들면",
    "피부에 거의 남지 않는 느낌이 들면": "피부에 남는 느낌이 줄어들면",

    # 시간 표현 제거
    "10~20초 정도": "",
    "20~30초 정도": "",
    "30~60초 정도": "",
    "1~2분 정도": "",
    "1분 정도": "",
    "약 10~20초": "",
    "약 20~30초": "",
    "약 30~60초": "",
    "약 1~2분": "",
    "약 1분": "",
    "잠시": "",

    # 사용 가이드 자연화
    "고민 부위에 먼저 바르고 남은 양을 얼굴 전체에 가볍게 펴 주세요": "적당량을 덜어 필요한 부위와 얼굴 전체에 가볍게 펴 발라 주세요",
    "세럼은 고민 부위에 먼저 바르고": "세럼은 적당량을 덜어 필요한 부위에 먼저 펴 바르고",
    "자리 잡으면": "피부에 남는 느낌이 줄어들면",
    "표면이 차분하게 자리 잡으면": "피부에 남는 느낌이 줄어들면",
    "다음 단계가 편합니다": "다음 단계로 넘어가면 좋아요",
    "다음 단계로 넘어가세요": "다음 단계로 넘어가면 좋아요",
    "진행해 주세요": "이어가면 좋아요",
    "정도 두면 루틴 정리가 편합니다.": "마무리하면 좋아요.",

    # 문체 통일
    "마무리합니다.": "마무리해요.",
    "구성합니다.": "구성했어요.",
    "짰습니다.": "구성했어요.",
    "맞춰졌습니다.": "맞춰졌어요.",
    "좋습니다.": "좋아요.",
    "됩니다.": "돼요."
}

CATEGORY_GUIDE_NAMES = {
    "Face Mists": "미스트",
    "Toner": "토너",
    "Emulsions": "로션",
    "Essences/Ampoules/Serums": "세럼",
    "Cream/Gel": "크림",
    "Eye Treatments": "아이케어 제품",
    "Facial Oils": "페이셜 오일",
    "Multi-balm": "멀티밤",
    "Multi-Balm": "멀티밤",
    "Shaving": "쉐이빙 제품",
    "All-In-One": "올인원 제품"
}

TIME_REMOVE_PATTERN = re.compile(
    r"(약\s*)?\d+\s*(~|-|–)?\s*\d*\s*(초|분)\s*(정도|뒤|후|간)?"
)


def save_json(data: dict[str, Any], output_dir: str, file_name: str) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(file_path)


def normalize_top_skin_concerns(top_skin_concerns: list[Any]) -> list[Any]:
    normalized = []

    for concern in top_skin_concerns:
        if isinstance(concern, dict):
            item = dict(concern)
            key = item.get("key")

            if key in INDICATOR_LABELS:
                item["label"] = INDICATOR_LABELS[key]

            normalized.append(item)
        else:
            normalized.append(concern)

    return normalized


def validate_routine_llm_input(llm_input: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_INPUT_KEYS if key not in llm_input]

    if missing:
        raise ValueError(f"Routine LLM input에 필요한 key가 없습니다: {missing}")

    skin_scores = llm_input["skin_scores"]

    if not isinstance(skin_scores, dict):
        raise ValueError("skin_scores는 dict 형식이어야 합니다.")

    for key in REQUIRED_SKIN_SCORE_KEYS:
        if key not in skin_scores:
            raise ValueError(f"skin_scores에 {key}가 없습니다.")

        score = float(skin_scores[key])

        if not 0 <= score <= 1:
            raise ValueError(
                f"skin_scores['{key}'] 값이 올바르지 않습니다: {score}. "
                "score는 0~1 범위여야 합니다."
            )

    if not isinstance(llm_input["top_skin_concerns"], list) or not llm_input["top_skin_concerns"]:
        raise ValueError("top_skin_concerns는 1개 이상의 list여야 합니다.")

    if not isinstance(llm_input["routines"], list) or not llm_input["routines"]:
        raise ValueError("routines는 1개 이상의 routine dict를 포함해야 합니다.")

    for routine in llm_input["routines"]:
        for key in REQUIRED_ROUTINE_KEYS:
            if key not in routine:
                raise ValueError(f"routine에 {key}가 없습니다.")

        if not isinstance(routine["items"], list) or not routine["items"]:
            raise ValueError("routine['items']는 1개 이상의 item dict를 포함해야 합니다.")

        for item in routine["items"]:
            for key in REQUIRED_ITEM_KEYS:
                if key not in item:
                    raise ValueError(f"routine item에 {key}가 없습니다.")

        if not isinstance(routine["warnings"], list):
            raise ValueError("routine['warnings']는 list 형식이어야 합니다.")

        if not isinstance(routine["pm_only_products"], list):
            raise ValueError("routine['pm_only_products']는 list 형식이어야 합니다.")


def build_compact_routine_input(llm_input: dict[str, Any]) -> dict[str, Any]:
    compact_routines = []

    for routine in llm_input["routines"]:
        compact_routines.append(
            {
                "routine_type": routine["routine_type"],
                "ampm_mode": routine["ampm_mode"],
                "total_price": routine["total_price"],
                "items": [
                    {
                        "slot_order": item["slot_order"],
                        "category": item["category"],
                        "brand": item["brand"],
                        "product_name": item["product_name"],
                        "price": item["price"]
                    }
                    for item in sorted(
                        routine["items"],
                        key=lambda x: x["slot_order"]
                    )
                ],
                "warnings": routine["warnings"],
                "pm_only_products": routine["pm_only_products"]
            }
        )

    return {
        "rec_session_id": llm_input["rec_session_id"],
        "user_id": llm_input["user_id"],
        "image_id": llm_input["image_id"],
        "result_id": llm_input["result_id"],
        "recommended_at": llm_input["recommended_at"],
        "skin_scores": llm_input["skin_scores"],
        "top_skin_concerns": normalize_top_skin_concerns(
            llm_input["top_skin_concerns"]
        ),
        "routines": compact_routines
    }


def build_routine_user_prompt(llm_input: dict[str, Any]) -> str:
    validate_routine_llm_input(llm_input)

    return "JSON input:\n" + json.dumps(
        build_compact_routine_input(llm_input),
        ensure_ascii=False,
        separators=(",", ":")
    )


def validate_routine_llm_response(
    llm_response: dict[str, Any],
    llm_input: dict[str, Any]
) -> None:
    routines = llm_response.get("routines")

    if not isinstance(routines, list) or not routines:
        raise ValueError("LLM 응답 routines는 1개 이상의 list여야 합니다.")

    if len(routines) != len(llm_input["routines"]):
        raise ValueError(
            "LLM 응답 routines 개수가 입력 routines 개수와 다릅니다. "
            f"입력 개수={len(llm_input['routines'])}, 출력 개수={len(routines)}"
        )

    for routine in routines:
        for key in REQUIRED_OUTPUT_ROUTINE_KEYS:
            if key not in routine:
                raise ValueError(f"LLM 응답 routine에 {key}가 없습니다.")

        if not isinstance(routine["step_guides"], list):
            raise ValueError("step_guides는 list 형식이어야 합니다.")

        if not isinstance(routine["strengths"], list):
            raise ValueError("strengths는 list 형식이어야 합니다.")

        if not isinstance(routine["cautions"], list):
            raise ValueError("cautions는 list 형식이어야 합니다.")

        for step in routine["step_guides"]:
            if "slot_order" not in step:
                raise ValueError("step_guides item에 slot_order가 없습니다.")

            if "usage_guide" not in step:
                raise ValueError("step_guides item에 usage_guide가 없습니다.")


def remove_duplicate_parentheses(text: str) -> str:
    patterns = [
        (r"아침·저녁\s*\(\s*아침·저녁 겸용\s*\)", "아침과 저녁 모두"),
        (r"아침·저녁 겸용\s*\(\s*아침·저녁 겸용\s*\)", "아침과 저녁 모두 활용하기 쉬운 구성"),
        (r"저녁 중심\s*\(\s*저녁 중심\s*\)", "저녁 중심"),
        (r"아침 중심\s*\(\s*아침 중심\s*\)", "아침 중심"),
        (r"\(\s*\)", "")
    ]

    cleaned = text

    for pattern, replacement in patterns:
        cleaned = re.sub(pattern, replacement, cleaned)

    return cleaned


def remove_time_expressions(text: str) -> str:
    cleaned = TIME_REMOVE_PATTERN.sub("", text)
    cleaned = cleaned.replace("정도 기다린 뒤", "피부에 남는 느낌이 줄어들면")
    cleaned = cleaned.replace("정도 후", "")
    cleaned = cleaned.replace("정도만", "")
    cleaned = cleaned.replace("정도 두고", "")
    cleaned = cleaned.replace("기다린 뒤", "피부에 남는 느낌이 줄어들면")
    cleaned = cleaned.replace("기다리면", "피부에 남는 느낌이 줄어들면")
    cleaned = cleaned.replace("기다렸다가", "피부에 남는 느낌이 줄어들면")
    return cleaned


def make_user_friendly_text(text: Any) -> Any:
    if not isinstance(text, str):
        return text

    cleaned = text.strip()
    cleaned = remove_duplicate_parentheses(cleaned)

    for before, after in USER_FRIENDLY_REPLACEMENTS.items():
        cleaned = cleaned.replace(before, after)

    for before, after in STYLE_REPLACEMENTS.items():
        cleaned = cleaned.replace(before, after)

    cleaned = remove_duplicate_parentheses(cleaned)
    cleaned = remove_time_expressions(cleaned)

    cleaned = cleaned.replace(" ,", ",")
    cleaned = cleaned.replace(",,", ",")
    cleaned = cleaned.replace(" .", ".")
    cleaned = cleaned.replace("  ", " ")
    cleaned = cleaned.replace("뒤 다음", "다음")
    cleaned = cleaned.replace("후 다음", "다음")
    cleaned = cleaned.replace("면 면", "면")
    cleaned = cleaned.replace("면 다음 단계", "면 다음 단계")
    cleaned = cleaned.replace("피부에 남는 느낌이 줄어들면 다음 단계로 넘어가면 좋아요.", "피부에 남는 느낌이 줄어들면 다음 단계로 넘어가면 좋아요.")
    cleaned = cleaned.strip()

    return " ".join(cleaned.split())


def clean_routine_output_text(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            key: clean_routine_output_text(value)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            clean_routine_output_text(item)
            for item in data
        ]

    return make_user_friendly_text(data)


def normalize_step_guides(
    routines: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    guide_aliases = [
        "usage_guide",
        "application_guide",
        "apply_guide",
        "how_to_use",
        "how_to_apply",
        "description",
        "guide"
    ]

    for routine in routines:
        normalized_steps = []

        for step in routine.get("step_guides", []):
            usage_guide = ""

            for key in guide_aliases:
                if key in step and step[key]:
                    usage_guide = step[key]
                    break

            normalized_steps.append(
                {
                    "slot_order": step.get("slot_order"),
                    "usage_guide": make_user_friendly_text(usage_guide)
                }
            )

        routine["step_guides"] = normalized_steps

    return routines


def remove_product_names_from_usage_guide(
    usage_guide: str,
    item: dict[str, Any]
) -> str:
    text = str(usage_guide).strip()

    product_name = str(item.get("product_name", "")).strip()
    brand = str(item.get("brand", "")).strip()
    category = item.get("category", "")
    category_name = CATEGORY_GUIDE_NAMES.get(category, "제품")

    candidates = []

    if brand and product_name:
        candidates.append(f"{brand} {product_name}")

    if product_name:
        candidates.append(product_name)

    if brand:
        candidates.append(brand)

    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate:
            text = text.replace(candidate, category_name)

    text = text.replace(f"{category_name}은 ", f"{category_name}을 ")
    text = text.replace(f"{category_name}는 ", f"{category_name}을 ")
    text = text.replace(f"{category_name} {category_name}", category_name)

    return make_user_friendly_text(text)


def ensure_step_guides_match_input(
    routines: list[dict[str, Any]],
    input_routines: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    for output_routine, input_routine in zip(routines, input_routines):
        sorted_items = sorted(
            input_routine["items"],
            key=lambda x: x["slot_order"]
        )

        guide_by_slot = {
            step.get("slot_order"): step
            for step in output_routine.get("step_guides", [])
        }

        fixed_guides = []

        for item in sorted_items:
            slot_order = item["slot_order"]
            step = guide_by_slot.get(slot_order)

            if not step:
                raise ValueError(
                    f"step_guides에 slot_order={slot_order}가 없습니다."
                )

            usage_guide = str(step.get("usage_guide", "")).strip()

            if not usage_guide:
                raise ValueError(
                    f"slot_order={slot_order}의 usage_guide가 비어 있습니다."
                )

            usage_guide = remove_product_names_from_usage_guide(
                usage_guide=usage_guide,
                item=item
            )

            fixed_guides.append(
                {
                    "slot_order": slot_order,
                    "usage_guide": make_user_friendly_text(usage_guide)
                }
            )

        output_routine["step_guides"] = fixed_guides

    return routines


def build_final_routine_result(
    llm_input: dict[str, Any],
    llm_response: dict[str, Any]
) -> dict[str, Any]:
    validate_routine_llm_response(llm_response, llm_input)

    cleaned_response = clean_routine_output_text(llm_response)
    routines = cleaned_response["routines"]

    for output_routine, input_routine in zip(routines, llm_input["routines"]):
        output_routine["routine_type"] = input_routine["routine_type"]

    routines = normalize_step_guides(routines)

    routines = ensure_step_guides_match_input(
        routines=routines,
        input_routines=llm_input["routines"]
    )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "routines": routines
    }


def generate_routine_llm_result(
    llm_input: dict[str, Any],
    output_dir: str = OUTPUT_DIR
) -> dict[str, Any]:
    llm_response = call_llm_json(
        system_prompt=ROUTINE_RECOMMENDATION_SYSTEM_PROMPT,
        user_prompt=build_routine_user_prompt(llm_input)
    )

    final_result = build_final_routine_result(
        llm_input=llm_input,
        llm_response=llm_response
    )

    saved_path = save_json(
        data=final_result,
        output_dir=output_dir,
        file_name=(
            f"routine_session_{llm_input['rec_session_id']}"
            f"_user_{llm_input['user_id']}.json"
        )
    )

    final_result["saved_path"] = saved_path

    return final_result
