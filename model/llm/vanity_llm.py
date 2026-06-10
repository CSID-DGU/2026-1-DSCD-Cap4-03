import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from llm_client import call_llm_json


PROMPT_VERSION = "vanity_v1"  
OUTPUT_DIR = "outputs/llm_results/vanity"

SKIN_SCORE_KEYS = [
    "acne_score",
    "dryness_score",
    "sagging_score",
    "pore_score",
    "pigmentation_score",
    "wrinkle_score",
]

VANITY_REQUIRED_KEYS = [
    "user_id",
    "result_id",
    "user_profile",
    "skin_analysis_result",
    "product_match_results",
    "routine_recommendation_results",
]

PRODUCT_MATCH_REQUIRED_KEYS = [
    "product_id",
    "category",
    "brand_name",
    "product_name",
    "vanity_fit_score",
    "fit_label",
    "recommend_action",
    "reason_tags",
    "caution_tags",
]

ROUTINE_RESULT_REQUIRED_KEYS = [
    "final_routine",
    "warnings",
    "total_price",
]

FINAL_ROUTINE_REQUIRED_KEYS = [
    "slot_order",
    "product_id",
    "category",
    "brand_name",
    "product_name",
    "source",
    "product_score",
    "price",
]

ALLOWED_SOURCES = {"vanity", "recommendation"}

ALLOWED_FIT_LABELS = {
    "excellent_match",
    "good_match",
    "so_so",
    "weak_match",
    "poor_match",
}

FIT_LABEL_DISPLAY_TEXT = {
    "excellent_match": "아주 적합",
    "good_match": "적합",
    "so_so": "쏘쏘",
    "weak_match": "아쉬움",
    "poor_match": "잘 맞지 않음",
}

ALLOWED_RECOMMEND_ACTIONS = {
    "strong_keep",
    "keep",
    "neutral",
    "caution",
    "replace",
}

ALLOWED_REASON_TAGS = {
    "concern_match",
    "skin_type_match",
    "review_match",
}

ALLOWED_CAUTION_TAGS = {
    "irritation_check",
    "weak_concern_match",
}

SKIN_SCORE_DISPLAY_TEXT = {
    "acne_score": "트러블",
    "dryness_score": "건조",
    "sagging_score": "처짐",
    "pore_score": "모공",
    "pigmentation_score": "색소침착",
    "wrinkle_score": "주름",
}

SKIN_CONCERN_DISPLAY_TEXT = {
    "acne": "트러블",
    "dryness": "건조",
    "sagging": "처짐",
    "pore": "모공",
    "pigmentation": "색소침착",
    "wrinkle": "주름",
}

SKIN_TYPE_DISPLAY_TEXT = {
    "dry": "건성",
    "oily": "지성",
    "combination": "복합성",
    "sensitive": "민감성",
    "normal": "중성",
}

REASON_TAG_TEXT = {
    "concern_match": "현재 피부 고민과 맞음",
    "skin_type_match": "피부 타입과 잘 맞음",
    "review_match": "사용감 반응이 비교적 양호함",
}

CAUTION_TAG_TEXT = {
    "irritation_check": "초반 자극 반응 확인 필요",
    "weak_concern_match": "주요 고민 보완력은 약한 편",
}

RECOMMEND_ACTION_TEXT = {
    "strong_keep": "우선 유지",
    "keep": "유지",
    "neutral": "가볍게 사용하며 비교",
    "caution": "양과 빈도 조절",
    "replace": "비교 후 교체 고려",
}

CATEGORY_ROLE_TEXT = {
    "Face Mists": "첫 수분감과 피부 정돈",
    "Toner": "피부결 정돈과 가벼운 보습 준비",
    "Emulsions": "수분·유분 균형을 잇는 중간 보습",
    "Essences/Ampoules/Serums": "피부 고민에 맞는 성분을 더하는 집중 케어",
    "Cream/Gel": "전체 보습 유지와 마무리",
    "Balms/Multi-balms": "건조 부위 국소 보습막",
    "Eye Treatments": "눈가 건조 부위 보완",
}

CATEGORY_USAGE_GUIDE = {
    "Face Mists": "얼굴에 분사 후 가볍게 눌러 흡수",
    "Toner": "손이나 화장솜으로 피부결 따라 흡수",
    "Emulsions": "얼굴 전체에 얇게 펴 바르고 흡수",
    "Essences/Ampoules/Serums": "고민 부위 중심 소량 흡수",
    "Cream/Gel": "전체 도포 후 건조 부위 덧바름",
    "Balms/Multi-balms": "건조 부위에만 얇게 덧바름",
    "Eye Treatments": "눈가에 소량 두드려 흡수",
}

CATEGORY_FLOW_TEXT = {
    "Face Mists": "수분 준비",
    "Toner": "피부결 정돈",
    "Emulsions": "중간 보습",
    "Essences/Ampoules/Serums": "집중 케어",
    "Cream/Gel": "보습 마무리",
    "Balms/Multi-balms": "건조 부위 보강",
    "Eye Treatments": "눈가 보완",
}

CATEGORY_DISPLAY_TEXT = {
    "Face Mists": "미스트",
    "Toner": "토너",
    "Emulsions": "로션",
    "Essences/Ampoules/Serums": "에센스·앰플·세럼",
    "Cream/Gel": "크림",
    "Balms/Multi-balms": "멀티밤",
    "Eye Treatments": "아이케어",
}

VANITY_SYSTEM_PROMPT = (
    "My Vanity 결과를 사용자용 한국어 JSON으로 변환. "
    "입력 compact schema: skin{type,concerns}, "
    "match[{id,cat,fit,why,caution}], "
    "routine{steps[{order,id,cat,src,role,use}]}. "
    "LLM은 실제로 필요한 설명만 생성한다. "
    "코드가 생성하는 skin_match overall, product summary, routine overall, warning은 작성하지 않아도 된다. "
    "입력에 없는 성분·효능·리뷰·경험·피부상태 생성 금지. "
    "score숫자, 내부 label/action/tag명, 계산과정 출력 금지. "
    "의학·치료·완치·질환·증상·위험단정 금지. JSON만 출력. "
    "톤: 좋아요/도움이 돼요/확인해 주세요/조절해 주세요/비교해 보세요. "
    "금지: 적합합니다, 안정적입니다, 판단됩니다, 버리세요, 쓰지 마세요, 위험합니다, 강력 추천. "
    "compact key 이름(type,concerns,fit,why,caution,src,role,use)을 문장에 직접 쓰지 말 것. "

    "skin_match.product_comments는 match와 같은 id 순서로 모두 작성. "
    "각 항목 key는 product_id, fit_reason, caution_comment. summary/action_comment는 생략. "
    "fit_reason은 1문장에 피부 타입/피부 고민 기준으로 맞는 점, 2문장에 why 또는 fit 기준 추가 근거/아쉬운 점을 작성. "
    "fit_reason에는 카테고리 역할/처음/소량/자극/반응확인/사용량/빈도/유지/비교/교체/바르는방법 금지. "
    "caution_comment는 caution 기반으로 사용 전/초반에 확인할 점만 1~2문장, 빈 문자열 금지. "
    "caution_comment는 자극감/붉어짐/따가움/트러블/당김/건조감/답답함/보완감 확인 중심. "
    "caution_comment 금지: 적합이유/유지/비교/교체/사용량조절/빈도조절/보습보강/루틴순서/바르는방법. "
    "의미분리: summary=판정, fit_reason=이유, caution_comment=확인점, action_comment=행동방향. "
    "고정/추천/새로 추가/루틴/총액/가격 표현 금지. "

    "vanity_routine.step_comments는 steps와 같은 order/id 순서로 모두 작성. "
    "각 항목 key는 slot_order, product_id, comment. "
    "comment는 2문장 정도로 고정/추천 여부, 제품 역할, 바르는 방법과 양 조절을 설명. "
    "cat/role/use 값을 그대로 복붙하지 말고 자연스럽게 풀어 쓸 것. "
    "아침/저녁/밤 표현은 쓰지 말 것. 다음 단계/다음 제품 표현 반복 금지. "

    "출력:"
    '{"skin_match":{"product_comments":[{"product_id":int,"fit_reason":str,"caution_comment":str}]},'
    '"vanity_routine":{"step_comments":[{"slot_order":int,"product_id":int,"comment":str}]}}'
)

STYLE_REPLACEMENTS = {
    "적합합니다": "잘 맞는 편이에요",
    "안정적입니다": "무난해 보여요",
    "판단됩니다": "볼 수 있어요",
    "위험합니다": "주의해서 확인해 주세요",
    "위험해요": "주의해서 확인해 주세요",
    "쓰면 안 됩니다": "사용 전 피부 반응을 확인해 주세요",
    "쓰면 안 돼요": "사용 전 피부 반응을 확인해 주세요",
    "버리세요": "다른 제품과 비교해 보세요",
    "버리는": "다른 제품과 비교하는",
    "강력 추천": "우선 고려",
    "강력추천": "우선 고려",
    "치료": "관리",
    "완치": "개선",
    "질환": "피부 고민",
    "증상": "피부 변화",
    "여드름": "트러블",
    "해당 제품": "이 제품",
    "해당 고민": "이 고민",
    "계산되었습니다": "확인됐어요",
    "점수가 높아": "잘 맞는 편이라",
    "점수가 낮아": "아쉬운 부분이 있어",
    "문제 없음": "큰 주의 요소는 적음",
    "나쁩니다": "아쉬운 편이에요",
    "나빠요": "아쉬운 편이에요",
    "좋습니다": "좋아요",
    "입니다.": "이에요.",
    "포커스": "고민",
    "focus": "고민",
    "다음 제품로": "다음 제품으로",
    "이 제품는": "이 제품은",
    "전 제품가": "앞 제품이",
    "제품가": "제품이",
    "제품로": "제품으로",
    "제품를": "제품을",
    "보습 제품로": "보습 제품으로",
    "보습 제품를": "보습 제품을",
    "사용감를": "사용감을",
    "사용감 수": "사용 횟수",
    "고민가": "고민이",
    "고민를": "고민을",
    "제품 사용 방향": "사용 방향",
    "사용 사용 방향": "사용 방향",
    "사용 방향 사용 방향": "사용 방향",
    "고정으로 두기": "우선 유지하기",
    "고정으로": "우선 유지하는 방향으로",
    "fit 분포": "적합성 차이",
    "전반 fit": "전반적인 적합성",
    "사용감를": "사용감을",
    "사용감링": "정돈",
    "고민가": "고민이",
    "고민를": "고민을",
    "제품가": "제품이",
    "제품를": "제품을",
    "제품로": "제품으로",
    "보습 제품를": "보습 제품을",
    "보습 제품로": "보습 제품으로",
}


def save_json(data: dict[str, Any], output_dir: str, file_name: str) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(file_path)


def clean_text(text: Any) -> str:
    cleaned = "" if text is None else str(text).strip()

    for _ in range(2):
        for before, after in STYLE_REPLACEMENTS.items():
            cleaned = cleaned.replace(before, after)

    cleaned = cleaned.replace(" ,", ",")
    cleaned = cleaned.replace(",,", ",")
    cleaned = cleaned.replace(" .", ".")
    cleaned = cleaned.replace("..", ".")
    cleaned = cleaned.replace("요..", "요.")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned and not cleaned.endswith((".", "요", "요.", "다")):
        cleaned += "."

    return cleaned


def remove_price_sentences(text: str) -> str:
    patterns = [
        r"(?:이번\s*)?(?:루틴의\s*)?총\s*가격은\s*[\d,]+원(?:입니다|이에요|입니다\.|이에요\.)?",
        r"총액은\s*[\d,]+원(?:입니다|이에요|입니다\.|이에요\.)?",
        r"총\s*가격은\s*[\d,]+원(?:입니다|이에요|입니다\.|이에요\.)?",
        r"가격은\s*[\d,]+원(?:입니다|이에요|입니다\.|이에요\.)?",
        r"비용은\s*[\d,]+원(?:입니다|이에요|입니다\.|이에요\.)?",
        r"[\d,]+원",
    ]

    cleaned = text

    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    cleaned = cleaned.replace("..", ".")

    return cleaned


def format_price_text(price: Optional[int]) -> Optional[str]:
    if price is None:
        return None

    try:
        return f"{int(price):,}원"
    except (TypeError, ValueError):
        return None


def collect_brand_names(normalized_input: dict[str, Any]) -> set[str]:
    brand_names = set()

    for product in normalized_input.get("product_match_results", []):
        brand_name = str(product.get("brand_name", "")).strip()
        if brand_name:
            brand_names.add(brand_name)

    final_routine = (
        normalized_input
        .get("routine_recommendation_results", {})
        .get("final_routine", [])
    )

    for product in final_routine:
        brand_name = str(product.get("brand_name", "")).strip()
        if brand_name:
            brand_names.add(brand_name)

    return brand_names


def remove_brand_parentheses(text: str, brand_names: set[str]) -> str:
    cleaned = text

    for brand_name in brand_names:
        if not brand_name:
            continue

        cleaned = cleaned.replace(f"({brand_name})", "")
        cleaned = cleaned.replace(f"（{brand_name}）", "")

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def replace_product_name_with_generic(text: str, product_name: str) -> str:
    if not product_name:
        return text

    cleaned = text
    escaped_name = re.escape(product_name)

    particle_map = {
        "은": "은",
        "는": "은",
        "이": "이",
        "가": "이",
    }

    def replace_with_particle(match: re.Match) -> str:
        particle = match.group(1)
        return f"이 제품{particle_map.get(particle, '')}"

    cleaned = re.sub(
        rf"{escaped_name}\s*(은|는|이|가)",
        replace_with_particle,
        cleaned,
    )

    cleaned = cleaned.replace(product_name, "이 제품")
    cleaned = cleaned.replace("이 제품는", "이 제품은")
    cleaned = cleaned.replace("이 제품가", "이 제품이")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def clean_output_text(
    text: Any,
    brand_names: set[str],
    product_name: Optional[str] = None,
    hide_product_name: bool = False,
) -> str:
    cleaned = clean_text(text)
    cleaned = remove_brand_parentheses(cleaned, brand_names)

    if hide_product_name and product_name:
        cleaned = replace_product_name_with_generic(cleaned, product_name)

    return cleaned


def remove_generic_product_leading(text: str) -> str:
    cleaned = str(text or "").strip()

    leading_patterns = [
        r"^이\s*제품은\s*",
        r"^이\s*제품이\s*",
        r"^이\s*제품는\s*",
        r"^해당\s*제품은\s*",
        r"^해당\s*제품이\s*",
    ]

    for pattern in leading_patterns:
        cleaned = re.sub(pattern, "", cleaned).strip()

    if cleaned.startswith("은 ") or cleaned.startswith("는 "):
        cleaned = cleaned[2:].strip()

    return cleaned


def normalize_skin_match_terms(text: str) -> str:
    replacements = {
        "고정 아이템": "제품",
        "고정 제품": "제품",
        "고정 항목": "제품",
        "고정템": "제품",
        "고정한 ": "",
        "추천 아이템": "제품",
        "추천 제품": "제품",
        "추천 항목": "제품",
        "새로 추천된 제품": "일부 제품",
        "새로 추천된 ": "",
        "추천된 ": "",
        "새로 추가된 제품": "일부 제품",
        "새로 추가된 일부 제품": "일부 제품",
        "다음 보습 제품으로": "피부 반응을 보며",
        "다음 보습 제품을": "다른 보습 제품을",
        "중간 보습 제품을 마친 뒤": "사용 후",
        "보습 제품 사이에서": "보습 제품과 비교할 때",
        "사용 방향의 마무리 밸런스": "마무리 사용감",
        "사용 방향 중심축": "보습 중심 제품",
        "사용 방향 내": "제품 선택에서",
        "루틴 흐름": "사용 방향",
        "루틴": "사용 방향",
        "단계": "제품",
        "레이어": "사용감",
        "흐름": "사용 방향",
        "fit 분포": "적합성 차이",
        "분포": "차이",
        "유지 쪽 비중": "유지하기 좋은 제품",
        "유지 비중": "유지하기 좋은 제품",
    }

    cleaned = text

    for before, after in replacements.items():
        cleaned = cleaned.replace(before, after)

    cleaned = cleaned.replace("제품 제품", "제품")
    cleaned = cleaned.replace("사용 방향 사용 방향", "사용 방향")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned



def quote_join(items: list[str]) -> str:
    cleaned_items = [str(item).strip() for item in items if str(item).strip()]

    if not cleaned_items:
        return ""

    if len(cleaned_items) == 1:
        return cleaned_items[0]

    if len(cleaned_items) == 2:
        return f"{cleaned_items[0]}와 {cleaned_items[1]}"

    return ", ".join(cleaned_items[:-1]) + f", {cleaned_items[-1]}"


def get_category_display_text(category: Any) -> str:
    category_value = str(category or "").strip()
    return CATEGORY_DISPLAY_TEXT.get(category_value, category_value or "제품")


def dedupe_preserve_order(items: list[str]) -> list[str]:
    result = []

    for item in items:
        item_value = str(item or "").strip()
        if item_value and item_value not in result:
            result.append(item_value)

    return result


def has_korean_jongseong(text: str) -> bool:
    if not text:
        return False

    last_char = text[-1]
    code = ord(last_char)

    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0

    return False


def topic_particle(text: str) -> str:
    return "은" if has_korean_jongseong(text) else "는"


def get_category_overall_display_text(category: Any, count: int = 1) -> str:
    category_value = str(category or "").strip()

    singular_map = {
        "Cream/Gel": "크림",
        "Essences/Ampoules/Serums": "에센스·앰플·세럼",
        "Balms/Multi-balms": "멀티밤",
        "Face Mists": "미스트",
        "Toner": "토너",
        "Emulsions": "로션",
        "Eye Treatments": "아이케어",
    }

    plural_map = {
        "Cream/Gel": "크림류",
        "Essences/Ampoules/Serums": "에센스·앰플·세럼류",
        "Balms/Multi-balms": "멀티밤류",
        "Face Mists": "미스트류",
        "Toner": "토너류",
        "Emulsions": "로션류",
        "Eye Treatments": "아이케어류",
    }

    if count >= 2:
        return plural_map.get(category_value, f"{get_category_display_text(category_value)}류")

    return singular_map.get(category_value, get_category_display_text(category_value))


def get_category_overall_role_text(category_text: str) -> str:
    role_map = {
        "미스트": "첫 수분감을 가볍게 더해 피부를 정돈하는 준비 제품",
        "미스트류": "첫 수분감을 가볍게 더해 피부를 정돈하는 준비 제품",
        "토너": "피부결과 보습감을 가볍게 정돈하는 보조 제품",
        "토너류": "피부결과 보습감을 가볍게 정돈하는 보조 제품",
        "로션": "수분과 유분감을 이어 주는 중간 보습 제품",
        "로션류": "수분과 유분감을 이어 주는 중간 보습 제품",
        "에센스·앰플·세럼": "피부 고민에 맞는 성분을 더하는 집중 케어 제품",
        "에센스·앰플·세럼류": "피부 고민에 맞는 성분을 더하는 집중 케어 제품",
        "크림": "전체 보습을 잡아 주는 마무리 제품",
        "크림류": "전체 보습을 잡아 주는 마무리 제품",
        "멀티밤": "건조한 부위를 국소적으로 보강하는 마무리 제품",
        "멀티밤류": "건조한 부위를 국소적으로 보강하는 마무리 제품",
        "아이케어": "눈가처럼 얇고 건조해지기 쉬운 부위를 보완하는 제품",
        "아이케어류": "눈가처럼 얇고 건조해지기 쉬운 부위를 보완하는 제품",
    }

    return role_map.get(category_text, "현재 피부 상태에 맞춰 활용할 수 있는 제품")


def build_category_overall_phrase(
    category_text: str,
    action_group: str,
    connective: bool = False,
) -> str:
    particle = topic_particle(category_text)
    role_text = get_category_overall_role_text(category_text)

    if action_group == "mixed":
        if connective:
            return (
                f"{category_text}{particle} 제품별 차이가 있어 잘 맞는 제품은 유지하고 "
                "체감이 약한 제품은 비교해 보는 편이고"
            )
        return (
            f"{category_text}{particle} 제품별 차이가 있어 잘 맞는 제품은 유지하고 "
            "체감이 약한 제품은 비교해 보는 편이 좋아요"
        )

    if action_group == "keep":
        if connective:
            return f"{category_text}{particle} {role_text}으로 현재 피부 상태에 잘 맞아 유지하기 좋고"
        return f"{category_text}{particle} {role_text}으로 현재 피부 상태에 잘 맞아 유지하기 좋아요"

    if action_group == "neutral":
        if connective:
            return f"{category_text}{particle} {role_text}으로 무난하게 활용하기 좋고"
        return f"{category_text}{particle} {role_text}으로 무난하게 활용하기 좋아요"

    if action_group == "caution":
        if connective:
            return f"{category_text}{particle} {role_text}이지만 양과 빈도를 조절하며 피부 반응을 확인하는 것이 좋고"
        return f"{category_text}{particle} {role_text}이지만 양과 빈도를 조절하며 피부 반응을 확인하는 것이 좋아요"

    if action_group == "replace":
        if connective:
            return f"{category_text}{particle} 현재 고민 보완력이 상대적으로 약해 다른 제품과 비교하거나 교체를 고려하는 편이고"
        return f"{category_text}{particle} 현재 고민 보완력이 상대적으로 약해 다른 제품과 비교하거나 교체를 고려해 보세요"

    if connective:
        return f"{category_text}{particle} 피부 반응을 확인하며 사용하기 좋고"
    return f"{category_text}{particle} 피부 반응을 확인하며 사용하기 좋아요"

def build_skin_match_overall_summary(normalized_input: dict[str, Any]) -> str:
    user_profile = normalized_input["user_profile"]
    skin_type_text = get_skin_type_text(user_profile.get("skin_type"))
    concern_text = build_skin_concern_text(user_profile.get("skin_concern"))
    concern_text = concern_text.replace(", ", "·")

    if skin_type_text and concern_text:
        intro = f"{skin_type_text} 피부와 {concern_text} 고민을 기준으로 보면"
    elif skin_type_text:
        intro = f"{skin_type_text} 피부 기준으로 보면"
    elif concern_text:
        intro = f"{concern_text} 고민을 기준으로 보면"
    else:
        intro = "현재 피부 상태를 기준으로 보면"

    raw_category_actions: dict[str, set[str]] = {}
    raw_category_counts: dict[str, int] = {}

    for product in normalized_input.get("product_match_results", []):
        raw_category = str(product.get("category") or "").strip()
        action = str(product.get("recommend_action") or "").strip()

        if action in {"strong_keep", "keep"}:
            action_group = "keep"
        elif action == "neutral":
            action_group = "neutral"
        elif action == "caution":
            action_group = "caution"
        elif action == "replace":
            action_group = "replace"
        else:
            action_group = "neutral"

        raw_category_counts[raw_category] = raw_category_counts.get(raw_category, 0) + 1
        raw_category_actions.setdefault(raw_category, set()).add(action_group)

    category_actions: dict[str, set[str]] = {}

    for raw_category, actions in raw_category_actions.items():
        category_text = get_category_overall_display_text(
            raw_category,
            count=raw_category_counts.get(raw_category, 1),
        )
        category_actions[category_text] = actions

    category_priority = [
        "크림",
        "크림류",
        "토너",
        "토너류",
        "에센스·앰플·세럼",
        "에센스·앰플·세럼류",
        "로션",
        "로션류",
        "미스트",
        "미스트류",
        "멀티밤",
        "멀티밤류",
        "아이케어",
        "아이케어류",
    ]

    def sort_key(item: tuple[str, set[str]]) -> int:
        category_text = item[0]
        if category_text in category_priority:
            return category_priority.index(category_text)
        return len(category_priority)

    phrases = []

    for category_text, actions in sorted(category_actions.items(), key=sort_key):
        if len(actions) >= 2:
            action_group = "mixed"
        elif "keep" in actions:
            action_group = "keep"
        elif "neutral" in actions:
            action_group = "neutral"
        elif "caution" in actions:
            action_group = "caution"
        elif "replace" in actions:
            action_group = "replace"
        else:
            action_group = "neutral"

        phrases.append((category_text, action_group))

    if not phrases:
        return f"{intro}, 카테고리별 피부 적합성을 비교해 유지할 제품과 조절이 필요한 제품을 나눠 보는 것이 좋아요."

    # 카테고리가 많아질수록 overall_summary가 길어지므로,
    # 모든 카테고리를 나열하지 않고 대표 카테고리 중심으로 압축한다.
    max_overall_categories = 3
    hidden_category_count = max(0, len(phrases) - max_overall_categories)
    visible_phrases = phrases[:max_overall_categories]

    first_phrase = build_category_overall_phrase(
        category_text=visible_phrases[0][0],
        action_group=visible_phrases[0][1],
        connective=False,
    )

    if len(visible_phrases) == 1:
        if hidden_category_count:
            return f"{intro}, {first_phrase}. 그 외 제품은 제품별 코멘트에서 적합성과 조절 방향을 함께 확인해 주세요."
        return f"{intro}, {first_phrase}."

    rest_phrases = []

    for idx, (category_text, action_group) in enumerate(visible_phrases[1:], start=1):
        connective = idx < len(visible_phrases) - 1
        rest_phrases.append(
            build_category_overall_phrase(
                category_text=category_text,
                action_group=action_group,
                connective=connective,
            )
        )

    if len(rest_phrases) == 1:
        summary = f"{intro}, {first_phrase}. {rest_phrases[0]}."
    else:
        summary = f"{intro}, {first_phrase}. {', '.join(rest_phrases)}."

    if hidden_category_count:
        summary += " 그 외 제품은 제품별 코멘트에서 적합성과 조절 방향을 함께 확인해 주세요."

    return summary


def build_skin_match_summary_text(
    product: dict[str, Any],
    normalized_input: dict[str, Any],
) -> str:
    """제품별 Skin Match summary를 LLM이 아닌 입력값 기반 대표 문구로 생성한다."""
    category_text = get_category_overall_display_text(product.get("category"), count=1)
    particle = topic_particle(category_text)

    user_profile = normalized_input.get("user_profile", {})
    skin_type_text = get_skin_type_text(user_profile.get("skin_type"))
    concern_text = build_skin_concern_text(user_profile.get("skin_concern")).replace(", ", "·")

    fit_label = str(product.get("fit_label") or "").strip()
    recommend_action = str(product.get("recommend_action") or "").strip()

    if fit_label in {"excellent_match", "good_match"} or recommend_action in {"strong_keep", "keep"}:
        if product.get("category") == "Cream/Gel" and skin_type_text:
            return f"{category_text}{particle} {skin_type_text} 피부의 보습 유지에 잘 맞아 우선 유지하기 좋은 제품이에요."
        if concern_text:
            return f"{category_text}{particle} {concern_text} 고민을 보완하는 방향과 잘 맞아 꾸준히 사용하기 좋은 제품이에요."
        return f"{category_text}{particle} 현재 피부 상태와 잘 맞아 우선 유지하기 좋은 제품이에요."

    if fit_label == "so_so" or recommend_action == "neutral":
        if product.get("category") == "Toner":
            if concern_text:
                return f"{category_text}{particle} {concern_text} 고민을 가볍게 정돈하는 보조 제품으로 무난하게 맞는 편이에요."
            return f"{category_text}{particle} 피부결과 보습감을 가볍게 정돈하는 보조 제품으로 무난하게 맞는 편이에요."
        if concern_text:
            return f"{category_text}{particle} {concern_text} 고민을 가볍게 보완하는 정도로 활용하기 좋은 제품이에요."
        return f"{category_text}{particle} 현재 피부 상태에 무난하게 맞는 보조 제품이에요."

    if fit_label == "weak_match" or recommend_action == "caution":
        if product.get("category") == "Essences/Ampoules/Serums":
            return f"{category_text}{particle} 현재 고민을 크게 보완하기엔 아쉬워 양과 빈도를 조절해 보는 편이 좋아요."
        if concern_text:
            return f"{category_text}{particle} {concern_text} 고민과의 맞물림이 조금 약해 사용량과 빈도를 조절해 보는 편이 좋아요."
        return f"{category_text}{particle} 현재 피부 상태와의 맞물림이 조금 약해 조절하며 사용하는 편이 좋아요."

    if recommend_action == "replace" or fit_label == "poor_match":
        if concern_text:
            return f"{category_text}{particle} 현재 {concern_text} 고민과의 연결감이 약해 다른 제품과 비교하거나 교체를 고려해 볼 만해요."
        return f"{category_text}{particle} 현재 피부 상태와의 연결감이 약해 다른 제품과 비교하거나 교체를 고려해 볼 만해요."

    if concern_text:
        return f"{category_text}{particle} {concern_text} 고민을 기준으로 피부 반응을 확인하며 사용 여부를 정해 보는 편이 좋아요."
    return f"{category_text}{particle} 현재 피부 상태를 기준으로 피부 반응을 확인하며 사용 여부를 정해 보는 편이 좋아요."


def build_skin_match_action_comment_text(
    product: dict[str, Any],
    normalized_input: dict[str, Any],
) -> str:
    """Skin Match action_comment를 recommend_action 기반 운영 방향으로 코드 생성한다."""
    recommend_action = str(product.get("recommend_action") or "").strip()
    category_text = get_category_overall_display_text(product.get("category"), count=1)
    particle = topic_particle(category_text)

    if recommend_action == "strong_keep":
        return (
            f"{category_text}{particle} 현재 피부에 잘 맞는 편이라 우선 유지해도 좋아요. "
            "사용 중 무겁거나 예민하게 느껴지면 양만 가볍게 조절해 주세요."
        )

    if recommend_action == "keep":
        return (
            f"{category_text}{particle} 현재 제품 선택에서 유지해도 좋아요. "
            "피부 반응을 보면서 사용량만 가볍게 조절해 주세요."
        )

    if recommend_action == "neutral":
        return (
            f"{category_text}{particle} 가볍게 사용하며 비교해 보는 방향이 좋아요. "
            "더 잘 맞는 제품이 있다면 함께 비교해 보세요."
        )

    if recommend_action == "caution":
        return (
            f"{category_text}{particle} 처음에는 사용량과 빈도를 낮춰 시작해 주세요. "
            "피부가 편안하게 반응하는 범위에서만 유지하는 편이 좋아요."
        )

    if recommend_action == "replace":
        return (
            f"{category_text}{particle} 다른 제품과 비교 후 교체를 고려해 주세요. "
            "기대한 보완감이 낮다면 사용 비중을 줄이는 방향이 좋아요."
        )

    return "피부 반응을 확인하면서 유지 여부를 결정해 주세요."


def clean_skin_match_overall_text(text: Any, brand_names: set[str]) -> str:
    cleaned = clean_output_text(text, brand_names=brand_names)
    cleaned = remove_price_sentences(cleaned)
    cleaned = normalize_skin_match_terms(cleaned)

    cleaned = re.sub(r"\([^)]*고민[^)]*\)", "", cleaned)
    cleaned = cleaned.replace("포커스", "고민")
    cleaned = remove_price_sentences(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned and not cleaned.endswith((".", "요", "요.", "다")):
        cleaned += "."

    return cleaned


def expand_short_caution_text(text: str) -> str:
    stripped = text.strip()

    if stripped in {"현재 고민 보완은 약함", "현재 고민 보완은 약함.", "현재 고민 보완이 약함", "현재 고민 보완이 약함."}:
        return (
            "현재 피부 고민을 직접적으로 보완하는 힘은 크지 않아 기대한 사용감과 맞는지 비교해 볼 필요가 있어요."
        )

    if stripped in {"초반 자극 반응 확인", "초반 자극 반응 확인."}:
        return (
            "초반에는 자극감이나 트러블 반응이 생기지 않는지 소량으로 먼저 확인해 주세요."
        )

    if stripped in {"피부 고민과 맞음", "피부 고민과 맞음."}:
        return (
            "현재 피부 고민과 맞는 방향이지만, 실제 사용감은 피부 반응을 보면서 확인해 주세요."
        )

    if stripped in {"현재 고민 보완은 약함이에요.", "현재 고민 보완은 약함이에요"}:
        return (
            "현재 피부 고민을 직접적으로 보완하는 힘은 크지 않아 기대한 사용감과 맞는지 비교해 볼 필요가 있어요."
        )

    return text


def split_comment_sentences(text: str) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []

    parts = re.split(r"(?<=[.!?。])\s+", cleaned)
    if len(parts) == 1:
        parts = re.split(r"(?<=요\.)\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def build_skin_match_fit_reason_fallback(
    product: dict[str, Any],
    normalized_input: Optional[dict[str, Any]] = None,
) -> str:
    """피부 타입/피부 고민 + reason_tags/fit_label 기반 fit_reason fallback 생성."""
    reason_tags = set(product.get("reason_tags") or [])
    fit_label = str(product.get("fit_label") or "").strip()

    skin_type_text = ""
    concern_text = ""

    if isinstance(normalized_input, dict):
        user_profile = normalized_input.get("user_profile", {})
        skin_type_text = get_skin_type_text(user_profile.get("skin_type"))
        concern_text = build_skin_concern_text(user_profile.get("skin_concern")).replace(", ", "·")

    if skin_type_text and concern_text:
        base_text = f"{skin_type_text} 피부와 {concern_text} 고민을 기준으로 보면"
    elif skin_type_text:
        base_text = f"{skin_type_text} 피부 기준으로 보면"
    elif concern_text:
        base_text = f"{concern_text} 고민을 기준으로 보면"
    else:
        base_text = "현재 피부 상태를 기준으로 보면"

    has_concern = "concern_match" in reason_tags
    has_skin_type = "skin_type_match" in reason_tags
    has_review = "review_match" in reason_tags

    if has_concern and has_skin_type:
        first_sentence = f"{base_text}, 피부 고민과 피부 타입 양쪽에서 맞는 근거가 있는 편이에요."
    elif has_concern:
        first_sentence = f"{base_text}, 주요 고민을 보완하는 방향과 맞는 편이에요."
    elif has_skin_type:
        first_sentence = f"{base_text}, 피부 타입에는 크게 어긋나지 않는 편이에요."
    elif has_review:
        first_sentence = f"{base_text}, 직접적인 고민 근거는 강하지 않지만 사용감 반응은 비교적 무난하게 볼 수 있어요."
    elif fit_label in {"excellent_match", "good_match"}:
        first_sentence = f"{base_text}, 입력된 적합도 기준으로는 현재 피부 상태와의 연결성이 비교적 좋은 편이에요."
    elif fit_label == "so_so":
        first_sentence = f"{base_text}, 강한 맞춤 근거보다는 보조적으로 볼 만한 연결성이 있어요."
    elif fit_label == "weak_match":
        first_sentence = f"{base_text}, 일부 기준은 맞아도 주요 고민을 채우는 힘은 다소 약할 수 있어요."
    else:
        first_sentence = f"{base_text}, 직접적인 보완 근거가 부족해 현재 조건과의 연결성이 약한 편이에요."

    if has_concern and has_skin_type and has_review:
        second_sentence = "사용감 반응도 함께 양호한 편이라 현재 피부 조건에서 기대해 보기 좋아요."
    elif has_concern and has_skin_type:
        second_sentence = "다만 사용감 반응 근거가 함께 강하게 잡힌 것은 아니어서 실제 체감은 개인차가 있을 수 있어요."
    elif has_concern and has_review:
        second_sentence = f"{skin_type_text + ' 피부 기준의 ' if skin_type_text else ''}타입 적합 근거는 상대적으로 적지만, 고민 방향과 사용감 근거가 함께 있는 점은 긍정적이에요."
    elif has_skin_type and has_review:
        if concern_text:
            second_sentence = f"다만 {concern_text} 고민을 직접 보완하는 근거는 강하지 않아 핵심 제품으로 보기엔 아쉬울 수 있어요."
        else:
            second_sentence = "다만 주요 고민을 직접 보완하는 근거는 강하지 않아 핵심 제품으로 보기엔 아쉬울 수 있어요."
    elif has_concern:
        second_sentence = f"{skin_type_text + ' 피부 타입이나 ' if skin_type_text else ''}사용감 반응 근거가 함께 충분히 잡힌 것은 아니어서 체감은 제한적일 수 있어요."
    elif has_skin_type:
        if concern_text:
            second_sentence = f"다만 {concern_text} 고민을 직접 보완하는 근거는 약해 보조적인 제품으로 보는 편이 좋아요."
        else:
            second_sentence = "다만 주요 고민을 직접 보완하는 근거는 약해 보조적인 제품으로 보는 편이 좋아요."
    elif has_review:
        if concern_text:
            second_sentence = f"다만 {concern_text} 고민을 직접 보완하는 근거는 강하지 않아 보조적으로 보는 편이 좋아요."
        else:
            second_sentence = "다만 주요 고민을 직접 보완하는 근거는 강하지 않아 보조적으로 보는 편이 좋아요."
    elif fit_label in {"excellent_match", "good_match"}:
        second_sentence = "추가 reason 근거가 많지는 않아도 전체 적합도 기준에서는 긍정적으로 볼 수 있어요."
    elif fit_label == "so_so":
        second_sentence = "핵심 제품처럼 강하게 기대하기보다는 보조적인 연결성으로 보는 편이 좋아요."
    elif fit_label == "weak_match":
        second_sentence = "기본 사용감은 가능해도 현재 고민을 직접 채우는 근거는 다소 부족해요."
    else:
        second_sentence = "현재 피부 상태에 꼭 필요한 제품으로 보기에는 아쉬운 편이에요."

    return f"{first_sentence} {second_sentence}"


def build_skin_match_caution_fallback(product: dict[str, Any]) -> str:
    caution_tags = set(product.get("caution_tags") or [])

    if "irritation_check" in caution_tags and "weak_concern_match" in caution_tags:
        return "초반 자극 반응과 기대한 보완감이 충분한지 함께 확인해 주세요."
    if "irritation_check" in caution_tags:
        return "초반에는 자극감이나 붉어짐처럼 예민한 반응이 나타나는지 확인해 주세요."
    if "weak_concern_match" in caution_tags:
        return "주요 고민 보완감이 기대보다 약하게 느껴질 수 있는지 확인해 주세요."
    return "큰 주의 요소는 적지만, 사용 중 불편감이 생기는지만 확인해 주세요."


def enforce_skin_match_field_meaning(
    text: str,
    field_name: str,
    product: dict[str, Any],
    normalized_input: Optional[dict[str, Any]] = None,
) -> str:
    """Skin Match 하위 필드 간 의미 중복을 줄이기 위한 범용 후처리."""
    if field_name == "fit_reason":
        fallback = build_skin_match_fit_reason_fallback(product, normalized_input)
        forbidden = [
            "처음", "초반", "소량", "자극", "붉어짐", "따가움", "트러블 반응",
            "피부 반응", "반응을 확인", "사용량", "사용 양", "빈도", "횟수",
            "조절", "유지", "교체", "비교", "바르", "도포", "흡수", "문지르",
            "세안", "다음 제품", "다음 단계", "루틴", "중단",
        ]
        max_sentences = 2
    elif field_name == "caution_comment":
        fallback = build_skin_match_caution_fallback(product)
        forbidden = [
            # 적합 이유/판단 영역 금지
            "잘 맞", "적합 이유", "적합", "활용하기 좋아", "기대해 볼 만",
            "현재 피부 고민과 맞", "피부 타입과", "사용감 반응",
            # 행동 방향 영역 금지
            "유지", "교체", "비교", "제품 선택", "사용 비중", "방향을 잡",
            "중심으로 두", "중단", "제외", "갈아타",
            # 사용량/빈도 조절 지시 금지
            "사용량", "사용 양", "사용 빈도", "빈도", "횟수", "양을 줄", "양을 늘",
            "조절", "보습 보강", "보습을 늘", "텀을 두",
            # 루틴/사용법 영역 금지
            "루틴", "다음 제품", "다음 단계", "바르", "도포", "흡수", "문지르",
            "세안", "펴 바", "덧바", "레이어", "사용해 주세요",
        ]
        max_sentences = 2
    else:
        fallback = str(text or "").strip()
        forbidden = []
        max_sentences = 2

    sentences = split_comment_sentences(text)
    kept_sentences = [
        sentence for sentence in sentences
        if not contains_any_keyword(sentence, forbidden)
    ]

    if not kept_sentences:
        return fallback

    cleaned = " ".join(kept_sentences[:max_sentences]).strip()
    if len(cleaned) < 16:
        return fallback
    if cleaned and not cleaned.endswith((".", "요", "요.", "다")):
        cleaned += "."
    return cleaned


def clean_skin_match_comment_text(
    text: Any,
    brand_names: set[str],
    product_name: Optional[str] = None,
    hide_product_name: bool = False,
    fallback: Optional[str] = None,
) -> str:
    cleaned = clean_output_text(
        text=text,
        brand_names=brand_names,
        product_name=product_name,
        hide_product_name=hide_product_name,
    )

    cleaned = re.sub(r"\([^)]*고민[^)]*\)", "", cleaned)
    cleaned = cleaned.replace("포커스", "고민")
    cleaned = normalize_skin_match_terms(cleaned)
    cleaned = remove_generic_product_leading(cleaned)
    cleaned = cleaned.replace("제품 사용 방향", "사용 방향")
    cleaned = cleaned.replace("사용감를", "사용감을")
    cleaned = cleaned.replace("고민가", "고민이")
    cleaned = cleaned.replace("고민를", "고민을")
    cleaned = cleaned.replace("제품가", "제품이")
    cleaned = cleaned.replace("제품를", "제품을")
    cleaned = cleaned.replace("제품로", "제품으로")
    cleaned = cleaned.replace("보습 제품를", "보습 제품을")
    cleaned = cleaned.replace("보습 제품로", "보습 제품으로")
    cleaned = cleaned.replace("사용감링", "정돈")
    cleaned = cleaned.replace("사용감 수", "사용량")
    cleaned = cleaned.replace("사용 방향의 시작 제품", "기초 정돈용 제품")
    cleaned = cleaned.replace("전체 사용 방향의 시작 제품", "기초 정돈용 제품")
    cleaned = cleaned.replace("사용 방향의 보습 기반", "보습 중심 역할")
    cleaned = cleaned.replace("사용 방향의 중심축", "보습 중심 제품")
    cleaned = cleaned.replace("사용 방향 중심축", "보습 중심 제품")
    cleaned = cleaned.replace("사용 방향 완성도", "사용 만족도")
    cleaned = cleaned.replace("사용 방향 내", "제품 선택에서")
    cleaned = cleaned.replace("사용 방향에서", "제품 선택에서")
    cleaned = cleaned.replace("제품 사용 방향", "제품 역할")

    # fit_reason에서 생길 수 있는 중복/어색 표현 정리
    cleaned = cleaned.replace("사용 사용 방향이 잘 맞아요", "방향이 잘 맞아요")
    cleaned = cleaned.replace("사용감 사용 방향은 기대할 수 있어요", "사용감은 기대할 수 있어요")
    cleaned = cleaned.replace("사용감 사용 방향", "사용감")
    cleaned = cleaned.replace("사용 사용 방향", "사용 방향")
    cleaned = cleaned.replace("사용 방향이 잘 맞아요", "방향이 잘 맞아요")
    cleaned = cleaned.replace("아쉬운 사용 방향이에요", "아쉬운 편이에요")

    cleaned = cleaned.replace("다음 보습 제품으로", "피부 반응을 보며")
    cleaned = cleaned.replace("중간 보습 제품을 마친 뒤", "사용 후")
    cleaned = cleaned.replace("보습 제품으로 넘어가기", "보습감이 부족한지 확인하기")
    cleaned = cleaned.replace("다음 제품으로 넘기기", "피부 반응을 확인하기")

    skin_match_action_replacements = {
        "세안 후": "사용할 때는",
        "다음 제품으로": "피부 반응을 보며",
        "다음 제품을": "다른 제품을",
        "다음 단계": "다른 제품",
        "흡수 후 피부 반응을 보며": "사용 후 피부 반응을 보며",
        "흡수 후": "사용 후",
        "문지르기보다": "자극이 느껴지지 않도록",
    }

    for before, after in skin_match_action_replacements.items():
        cleaned = cleaned.replace(before, after)

    cleaned = cleaned.replace("사용할 때는 손이나 화장솜에 덜어", "사용할 때는")
    cleaned = cleaned.replace("피부결을 따라 얇게 펴 바른 뒤", "피부 반응을 확인하며")
    cleaned = cleaned.replace("고민 부위 중심으로 아주 소량만 얇게 펴 바르고", "고민 부위 중심으로 소량만 사용하며")
    cleaned = cleaned.replace("얼굴 전체에 고르게 펴 바르고", "얼굴 전체에 사용할 때도")
    cleaned = cleaned.replace("흡수가 정리될 때까지 가볍게 눌러 주세요", "피부가 편안한지 확인해 주세요")
    cleaned = cleaned.replace("손이나 화장솜으로 피부결 방향에 맞춰 흡수시키는 식으로 사용하세요", "피부 반응을 보며 가볍게 사용해 주세요")
    cleaned = cleaned.replace("중간 보습 제품을 마친 뒤", "사용 후")
    cleaned = cleaned.replace("보습 제품 사이에서", "보습 제품과 비교할 때")
    cleaned = cleaned.replace("다음 집중 케어", "집중 케어 제품")
    cleaned = cleaned.replace("다음 제품", "다른 제품")

    cleaned = expand_short_caution_text(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned and fallback:
        cleaned = fallback

    if cleaned and not cleaned.endswith((".", "요", "요.", "다")):
        cleaned += "."

    return cleaned

def clean_step_comment_text(
    text: Any,
    brand_names: set[str],
    product_name: Optional[str] = None,
) -> str:
    cleaned = clean_output_text(
        text=text,
        brand_names=brand_names,
        product_name=product_name,
        hide_product_name=False,
    )

    repeat_patterns = [
        r"바로\s*다음\s*제품으로\s*이어가면\s*좋아요",
        r"다음\s*제품으로\s*이어가면\s*좋아요",
        r"다음\s*제품으로\s*이어가\s*주세요",
        r"다음\s*단계로\s*넘어가면\s*돼요",
        r"다음\s*단계로\s*이어가\s*주세요",
        r"마무리로\s*가면\s*좋아요",
    ]

    for pattern in repeat_patterns:
        cleaned = re.sub(pattern, "충분히 흡수시켜 주세요", cleaned)

    step_text_replacements = {
        "추천 단계의": "새로 추천된",
        "추천 단계": "새로 추천된 제품",
        "부족한 루틴을 보완하기 위해 새로 추천된 제품으로": "새로 추천된 제품으로",
        "추천 제품으로": "새로 추천된 제품으로",
        "이미 가지고 있는 제품을 유지한 항목으로": "기존에 가지고 있는 제품으로",
        "고정 제품으로": "기존에 가지고 있는 제품으로",
        "고정 토너": "이미 가지고 있는 토너",
        "고정 크림": "이미 가지고 있는 크림",
        "루틴의 중심": "보습의 중심",
        "루틴 중심": "보습 중심",
        "루틴 흐름": "사용 방향",
    }

    for before, after in step_text_replacements.items():
        cleaned = cleaned.replace(before, after)

    duplicate_replacements = {
        "새로 새로 추천된": "새로 추천된",
        "새로 추천된 새로 추천된": "새로 추천된",
        "새로 추천한": "새로 추천된",
        "추천된 제품으로, 새로 추천된": "새로 추천된",
        "새로 추천된 제품으로, 새로 추천된": "새로 추천된 제품으로",
        "새로 추천된 제품으로, 새로 추천된 제품으로": "새로 추천된 제품으로",
        "새로 추천된 제품으로 새로 추천된": "새로 추천된",
        "새로 추천된 제품으로, 새로 추천된 제품": "새로 추천된 제품",
    }

    for before, after in duplicate_replacements.items():
        cleaned = cleaned.replace(before, after)

    cleaned = cleaned.replace("제품가", "제품이")
    cleaned = cleaned.replace("제품를", "제품을")
    cleaned = cleaned.replace("제품로", "제품으로")
    cleaned = cleaned.replace("보습 제품를", "보습 제품을")
    cleaned = cleaned.replace("보습 제품로", "보습 제품으로")
    cleaned = cleaned.replace("사용감링", "정돈")
    cleaned = cleaned.replace("사용감 수", "사용량")
    cleaned = cleaned.replace("사용 방향의 시작 제품", "기초 정돈용 제품")
    cleaned = cleaned.replace("전체 사용 방향의 시작 제품", "기초 정돈용 제품")
    cleaned = cleaned.replace("사용 방향의 보습 기반", "보습 중심 역할")
    cleaned = cleaned.replace("사용 방향의 중심축", "보습 중심 제품")
    cleaned = cleaned.replace("사용 방향 중심축", "보습 중심 제품")
    cleaned = cleaned.replace("사용 방향 완성도", "사용 만족도")
    cleaned = cleaned.replace("사용 방향 내", "제품 선택에서")
    cleaned = cleaned.replace("사용 방향에서", "제품 선택에서")
    cleaned = cleaned.replace("제품 사용 방향", "제품 역할")
    cleaned = cleaned.replace("사용감를", "사용감을")
    cleaned = cleaned.replace("고민가", "고민이")
    cleaned = cleaned.replace("고민를", "고민을")

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned and not cleaned.endswith((".", "요", "요.", "다")):
        cleaned += "."

    return cleaned


def ensure_step_source_prefix(comment: str, product: dict[str, Any]) -> str:
    source = str(product.get("source") or "").strip()

    if source == "vanity":
        prefix = "기존에 가지고 있는 제품으로"
        existing_prefixes = [
            "기존에 가지고 있는 제품으로",
            "이미 가지고 있는 제품으로",
            "고정 제품으로",
            "고정한 제품으로",
        ]
    elif source == "recommendation":
        prefix = "새로 추천된 제품으로"
        existing_prefixes = [
            "새로 추천된 제품으로",
            "새로 추천한 제품으로",
            "추천 제품으로",
            "추천된 제품으로",
        ]
    else:
        return str(comment or "").strip()

    cleaned = str(comment or "").strip()

    if not cleaned:
        return f"{prefix}, 현재 루틴에서 필요한 역할을 보완해 주세요."

    if any(cleaned.startswith(item) for item in existing_prefixes):
        cleaned = cleaned.replace("새로 새로 추천된", "새로 추천된")
        cleaned = cleaned.replace("새로 추천한", "새로 추천된")
        return cleaned

    return f"{prefix}, {cleaned}"


def build_routine_flow_text(final_routine: list[dict[str, Any]]) -> str:
    flow_items = []

    for product in final_routine:
        category = str(product.get("category") or "").strip()
        flow_text = CATEGORY_FLOW_TEXT.get(category)

        if flow_text and flow_text not in flow_items:
            flow_items.append(flow_text)

    if not flow_items:
        return "피부결 정돈, 보습 연결, 고민 부위 보완, 보습 마무리"

    return " → ".join(flow_items)


def build_routine_overall_prefix(
    normalized_input: dict[str, Any],
    final_routine: list[dict[str, Any]],
) -> str:
    routine_results = normalized_input["routine_recommendation_results"]
    fixed_n, rec_n = count_routine_sources(final_routine)

    user_profile = normalized_input["user_profile"]
    concerns_text = build_skin_concern_text(user_profile.get("skin_concern"))
    total_price_text = format_price_text(routine_results.get("total_price"))
    flow_text = build_routine_flow_text(final_routine)

    if concerns_text:
        concern_part = f"{concerns_text.replace(', ', '·')} 고민을 기준으로"
    else:
        concern_part = "현재 피부 상태를 기준으로"

    if total_price_text:
        return (
            f"이번 루틴은 고정한 제품 {fixed_n}개와 새로 추천된 제품 {rec_n}개로 구성됐어요. "
            f"총액은 {total_price_text}이며, {concern_part} {flow_text} 순서로 이어져요."
        )

    return (
        f"이번 루틴은 고정한 제품 {fixed_n}개와 새로 추천된 제품 {rec_n}개로 구성됐어요. "
        f"{concern_part} {flow_text} 순서로 이어져요."
    )


def clean_routine_overall_text(
    text: Any,
    brand_names: set[str],
    normalized_input: dict[str, Any],
    final_routine: list[dict[str, Any]],
) -> str:
    cleaned = clean_output_text(text, brand_names=brand_names)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    prefix = build_routine_overall_prefix(
        normalized_input=normalized_input,
        final_routine=final_routine,
    )

    has_fixed = "고정" in cleaned
    has_recommendation = "추천" in cleaned
    total_price_text = format_price_text(
        normalized_input["routine_recommendation_results"].get("total_price")
    )
    has_price = bool(total_price_text and total_price_text in cleaned)

    # 루틴 전체 요약은 고정 개수, 추천 개수, 총액, 흐름이 핵심이므로
    # LLM 문장을 덧붙이지 않고 코드 기반 문장으로 통일해 중복과 어색한 표현을 방지한다.
    return prefix


def is_mostly_ascii(text: str) -> bool:
    if not text:
        return False

    ascii_count = sum(1 for char in text if ord(char) < 128)
    return ascii_count / max(len(text), 1) > 0.8


def normalize_int_from_text(text: str) -> Optional[int]:
    patterns = [
        r"product_id\s*[:=]\s*(\d+)",
        r"id\s*[:=]\s*(\d+)",
        r"\bpid\s*[:=]\s*(\d+)",
        r"\bproduct\s*[:=]\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def parse_warning_string(warning_text: str) -> dict[str, Optional[Union[str, int]]]:
    text = warning_text.strip()

    warning_type = ""
    message = text

    if ":" in text:
        warning_type, message = text.split(":", 1)
        warning_type = warning_type.strip()
        message = message.strip()

    product_id = normalize_int_from_text(message)

    ingredient = None
    product_name = message

    match = re.search(r"\(([^()]+)\)", message)
    if match:
        ingredient = match.group(1).strip()
        product_name = re.sub(r"\([^()]+\)", "", message).strip()

    product_name = re.sub(r"product_id\s*[:=]\s*\d+", "", product_name, flags=re.IGNORECASE)
    product_name = re.sub(r"id\s*[:=]\s*\d+", "", product_name, flags=re.IGNORECASE)
    product_name = re.sub(r"\bpid\s*[:=]\s*\d+", "", product_name, flags=re.IGNORECASE)
    product_name = re.sub(r"\bproduct\s*[:=]\s*\d+", "", product_name, flags=re.IGNORECASE)
    product_name = product_name.replace("|", " ").strip()
    product_name = re.sub(r"\s+", " ", product_name).strip()

    return {
        "warning_type": warning_type or None,
        "product_id": product_id,
        "product_name": product_name or None,
        "ingredient": ingredient,
        "message": message or None,
    }


def build_final_routine_map(final_routine: Optional[list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    if not final_routine:
        return {}

    routine_map = {}

    for product in final_routine:
        try:
            product_id = int(product.get("product_id"))
        except (TypeError, ValueError):
            continue

        routine_map[product_id] = product

    return routine_map


def resolve_warning_product_name(
    product_id: Optional[Any],
    product_name: str,
    final_routine_map: dict[int, dict[str, Any]],
) -> str:
    resolved_name = product_name.strip() if product_name else ""

    if product_id is not None:
        try:
            product_id_int = int(product_id)
            matched_product = final_routine_map.get(product_id_int)

            if matched_product:
                matched_name = str(matched_product.get("product_name") or "").strip()
                if matched_name:
                    return matched_name
        except (TypeError, ValueError):
            pass

    if resolved_name and is_mostly_ascii(resolved_name):
        return "이 제품"

    return resolved_name


def build_warning_comment_from_input(
    warnings: list[Any],
    final_routine: Optional[list[dict[str, Any]]] = None,
) -> Optional[str]:
    if not warnings:
        return None

    comments = []
    final_routine_map = build_final_routine_map(final_routine)

    for warning in warnings:
        if isinstance(warning, str):
            parsed = parse_warning_string(warning)
            warning_type = str(parsed.get("warning_type") or "").strip()
            product_id = parsed.get("product_id")
            product_name = str(parsed.get("product_name") or "").strip()
            ingredient = str(parsed.get("ingredient") or "").strip()
            message = str(parsed.get("message") or "").strip()

        elif isinstance(warning, dict):
            warning_type = str(warning.get("warning_type") or "").strip()
            product_id = warning.get("product_id")
            product_name = str(warning.get("product_name") or "").strip()
            ingredient = str(warning.get("ingredient") or "").strip()
            message = str(warning.get("message") or "").strip()

        else:
            continue

        product_name = resolve_warning_product_name(
            product_id=product_id,
            product_name=product_name,
            final_routine_map=final_routine_map,
        )

        if warning_type == "pm_only":
            if product_name and ingredient:
                comments.append(
                    f"{product_name}에는 {ingredient} 성분이 포함되어 있어 저녁 사용을 권장해요. "
                    "처음 사용할 때는 소량으로 시작해 피부 반응을 확인해 주세요."
                )
            elif ingredient:
                comments.append(
                    f"{ingredient} 성분이 포함되어 있어 저녁 사용을 권장해요. "
                    "처음 사용할 때는 소량으로 시작해 피부 반응을 확인해 주세요."
                )
            elif product_name:
                comments.append(
                    f"{product_name}은 저녁 사용이 권장된 제품이에요. "
                    "처음 사용할 때는 소량으로 시작해 피부 반응을 확인해 주세요."
                )
            else:
                comments.append(
                    "저녁 사용이 권장된 제품이 있어요. "
                    "처음 사용할 때는 소량으로 시작해 피부 반응을 확인해 주세요."
                )

        elif warning_type == "am_only":
            message_has_sunscreen = any(
                keyword in message
                for keyword in ["자외선", "선크림", "선케어", "차단제", "sunscreen", "SPF", "spf"]
            )

            if message_has_sunscreen:
                if product_name and ingredient:
                    comments.append(
                        f"{product_name}에는 {ingredient} 성분이 포함되어 있어, "
                        "아침에 사용할 경우 자외선 차단제를 함께 사용하는 것을 권장해요. "
                        "처음 사용할 때는 피부 반응을 확인해 주세요."
                    )
                elif ingredient:
                    comments.append(
                        f"{ingredient} 성분은 아침에 사용할 경우 자외선 차단제를 함께 사용하는 것을 권장해요. "
                        "처음 사용할 때는 피부 반응을 확인해 주세요."
                    )
                elif product_name:
                    comments.append(
                        f"{product_name}은 아침에 사용할 경우 자외선 차단제를 함께 사용하는 것을 권장해요. "
                        "처음 사용할 때는 피부 반응을 확인해 주세요."
                    )
                else:
                    comments.append(
                        "아침에 사용할 경우 자외선 차단제를 함께 사용하는 것이 권장된 제품이 있어요. "
                        "처음 사용할 때는 피부 반응을 확인해 주세요."
                    )

            elif product_name and ingredient:
                comments.append(
                    f"{product_name}에는 {ingredient} 성분 특성상 아침 사용을 권장해요. "
                    "처음 사용할 때는 피부 반응을 확인해 주세요."
                )
            elif ingredient:
                comments.append(
                    f"{ingredient} 성분 특성상 아침 사용을 권장해요. "
                    "처음 사용할 때는 피부 반응을 확인해 주세요."
                )
            elif product_name:
                comments.append(
                    f"{product_name}은 아침 사용이 권장된 제품이에요. "
                    "처음 사용할 때는 피부 반응을 확인해 주세요."
                )
            else:
                comments.append(
                    "아침 사용이 권장된 제품이 있어요. "
                    "처음 사용할 때는 피부 반응을 확인해 주세요."
                )

        elif warning_type == "avoid_combination":
            if product_name and ingredient:
                comments.append(
                    f"{product_name}의 {ingredient} 성분 조합은 함께 사용할 때 확인이 필요해요. "
                    "처음에는 같은 루틴에서 한 번에 많이 겹치지 않게 조절해 주세요."
                )
            elif ingredient:
                comments.append(
                    f"{ingredient} 성분 조합은 함께 사용할 때 확인이 필요해요. "
                    "처음에는 같은 루틴에서 한 번에 많이 겹치지 않게 조절해 주세요."
                )
            elif product_name:
                comments.append(
                    f"{product_name}은 다른 제품과 함께 사용할 때 확인이 필요해요. "
                    "처음에는 같은 루틴에서 한 번에 많이 겹치지 않게 조절해 주세요."
                )
            else:
                comments.append(
                    "함께 사용할 때 확인이 필요한 조합이 있어요. "
                    "처음에는 같은 루틴에서 한 번에 많이 겹치지 않게 조절해 주세요."
                )

        elif message:
            if product_name and ingredient:
                comments.append(
                    f"{product_name}에는 {ingredient} 성분이 포함되어 있어 사용 전 확인이 필요해요. "
                    "처음 사용할 때는 소량으로 시작해 피부 반응을 확인해 주세요."
                )
            elif product_name:
                comments.append(
                    f"{product_name}은 사용 전 확인이 필요한 제품이에요. "
                    "처음 사용할 때는 소량으로 시작해 피부 반응을 확인해 주세요."
                )
            elif ingredient:
                comments.append(
                    f"{ingredient} 성분은 사용 전 확인이 필요해요. "
                    "처음 사용할 때는 소량으로 시작해 피부 반응을 확인해 주세요."
                )
            else:
                comments.append(
                    f"{message}. 처음 사용할 때는 소량으로 시작해 피부 반응을 확인해 주세요."
                )

    if not comments:
        return "큰 주의 요소는 적지만, 처음 사용할 때는 피부 반응을 확인해 주세요."

    return " ".join(comments)


def clean_warning_comment_text(text: Any, brand_names: set[str]) -> str:
    cleaned = clean_output_text(text, brand_names=brand_names)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned and not cleaned.endswith((".", "요", "요.", "다")):
        cleaned += "."

    return cleaned


def validate_score_range(value: Any, key: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key} 값은 숫자여야 합니다: {value}") from e

    if not 0 <= score <= 1:
        raise ValueError(f"{key} 값은 0~1 범위여야 합니다: {score}")

    return score


def validate_optional_number(value: Any, key: str) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key} 값은 숫자 또는 null이어야 합니다: {value}") from e


def validate_optional_int(value: Any, key: str) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key} 값은 정수 또는 null이어야 합니다: {value}") from e


def normalize_tag_list(
    tags: Any,
    allowed_tags: set[str],
    max_count: int,
    key_name: str,
) -> list[str]:
    if tags is None:
        return []

    if not isinstance(tags, list):
        raise ValueError(f"{key_name}는 list 형식이어야 합니다.")

    normalized_tags = []

    for tag in tags:
        tag_value = str(tag).strip()

        if not tag_value:
            continue

        if tag_value not in allowed_tags:
            raise ValueError(
                f"{key_name}에 지원하지 않는 tag가 있습니다: {tag_value}. "
                f"허용값: {sorted(allowed_tags)}"
            )

        normalized_tags.append(tag_value)

    if len(normalized_tags) > max_count:
        raise ValueError(
            f"{key_name}는 최대 {max_count}개까지 허용됩니다: {normalized_tags}"
        )

    return normalized_tags


def get_reason_tag_texts(reason_tags: list[str]) -> list[str]:
    return [
        REASON_TAG_TEXT[tag]
        for tag in reason_tags
        if tag in REASON_TAG_TEXT
    ]


def get_caution_tag_texts(caution_tags: list[str]) -> list[str]:
    return [
        CAUTION_TAG_TEXT[tag]
        for tag in caution_tags
        if tag in CAUTION_TAG_TEXT
    ]


def get_recommend_action_text(recommend_action: str) -> str:
    return RECOMMEND_ACTION_TEXT.get(recommend_action, "입력 action 따름")


def get_category_usage_guide(category: str) -> str:
    return CATEGORY_USAGE_GUIDE.get(category, "얼굴에 얇게 펴 바르고 피부에 충분히 흡수")


def get_category_role_text(category: str) -> str:
    return CATEGORY_ROLE_TEXT.get(category, "현재 루틴에서 필요한 보완 역할")


def get_skin_type_text(skin_type: Any) -> str:
    skin_type_value = str(skin_type or "").strip()
    return SKIN_TYPE_DISPLAY_TEXT.get(skin_type_value, skin_type_value)


def build_skin_concern_text(skin_concerns: Any) -> str:
    if not isinstance(skin_concerns, list):
        return ""

    converted = []

    for concern in skin_concerns:
        concern_value = str(concern).strip()
        if not concern_value:
            continue

        display_text = SKIN_CONCERN_DISPLAY_TEXT.get(concern_value, concern_value)

        if display_text not in converted:
            converted.append(display_text)

    return ", ".join(converted)


def build_score_focus_text(
    skin_scores: dict[str, Any],
    max_count: int = 2,
) -> str:
    sorted_scores = sorted(
        [
            (key, float(skin_scores.get(key, 0)))
            for key in SKIN_SCORE_KEYS
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    focus_items = []

    for key, score in sorted_scores:
        if score <= 0:
            continue

        display_text = SKIN_SCORE_DISPLAY_TEXT.get(key, key)

        if display_text and display_text not in focus_items:
            focus_items.append(display_text)

        if len(focus_items) >= max_count:
            break

    return ", ".join(focus_items)


def count_routine_sources(final_routine: list[dict[str, Any]]) -> tuple[int, int]:
    fixed_n = 0
    rec_n = 0

    for product in final_routine:
        source = str(product.get("source") or "").strip()

        if source == "vanity":
            fixed_n += 1
        elif source == "recommendation":
            rec_n += 1

    return fixed_n, rec_n


def compact_warning(warning: Union[str, dict[str, Any]]) -> Union[str, dict[str, Any]]:
    if isinstance(warning, str):
        parsed = parse_warning_string(warning)

        return {
            "type": parsed.get("warning_type"),
            "id": parsed.get("product_id"),
            "name": parsed.get("product_name"),
            "ing": parsed.get("ingredient"),
            "msg": parsed.get("message"),
        }

    return {
        "type": warning.get("warning_type"),
        "id": warning.get("product_id"),
        "name": warning.get("product_name"),
        "ing": warning.get("ingredient"),
        "msg": warning.get("message"),
    }


def build_compact_vanity_input(normalized_input: dict[str, Any]) -> dict[str, Any]:
    user_profile = normalized_input["user_profile"]

    skin_type_text = get_skin_type_text(user_profile.get("skin_type"))
    skin_concern_text = build_skin_concern_text(user_profile.get("skin_concern"))

    compact_match = []

    for product in normalized_input["product_match_results"]:
        compact_match.append(
            {
                "id": product["product_id"],
                "cat": product["category"],
                "fit": product["fit_label_display"],
                "why": product["reason_tag_texts"],
                "caution": product["caution_tag_texts"],
            }
        )

    routine_results = normalized_input["routine_recommendation_results"]
    final_routine = routine_results["final_routine"]

    compact_steps = []

    for product in final_routine:
        source = str(product["source"]).strip()

        if source == "vanity":
            source_text = "고정"
        elif source == "recommendation":
            source_text = "추천"
        else:
            source_text = source

        compact_steps.append(
            {
                "order": product["slot_order"],
                "id": product["product_id"],
                "cat": product["category"],
                "src": source_text,
                "role": get_category_role_text(product["category"]),
                "use": product["usage_guide"],
            }
        )

    return {
        "skin": {
            "type": skin_type_text,
            "concerns": skin_concern_text,
        },
        "match": compact_match,
        "routine": {
            "steps": compact_steps,
        },
    }


def validate_vanity_input(llm_input: dict[str, Any]) -> None:
    missing = [key for key in VANITY_REQUIRED_KEYS if key not in llm_input]

    if missing:
        raise ValueError(f"Vanity LLM input에 필요한 key가 없습니다: {missing}")

    user_profile = llm_input.get("user_profile")
    if not isinstance(user_profile, dict):
        raise ValueError("user_profile은 dict 형식이어야 합니다.")

    skin_result = llm_input.get("skin_analysis_result")
    if not isinstance(skin_result, dict):
        raise ValueError("skin_analysis_result는 dict 형식이어야 합니다.")

    missing_scores = [key for key in SKIN_SCORE_KEYS if key not in skin_result]
    if missing_scores:
        raise ValueError(
            f"skin_analysis_result에 필요한 score key가 없습니다: {missing_scores}"
        )

    for key in SKIN_SCORE_KEYS:
        validate_score_range(skin_result[key], key)

    validate_product_match_results(llm_input.get("product_match_results"))

    validate_routine_recommendation_results(
        llm_input.get("routine_recommendation_results")
    )


def validate_product_match_results(products: Any) -> None:
    if not isinstance(products, list) or not products:
        raise ValueError("product_match_results는 비어 있지 않은 list여야 합니다.")

    for idx, product in enumerate(products):
        if not isinstance(product, dict):
            raise ValueError(f"product_match_results[{idx}]는 dict 형식이어야 합니다.")

        missing_product_keys = [
            key for key in PRODUCT_MATCH_REQUIRED_KEYS if key not in product
        ]

        if missing_product_keys:
            raise ValueError(
                f"product_match_results[{idx}]에 필요한 key가 없습니다: "
                f"{missing_product_keys}"
            )

        validate_score_range(
            product["vanity_fit_score"],
            f"product_match_results[{idx}].vanity_fit_score",
        )

        fit_label = str(product["fit_label"]).strip()

        if fit_label not in ALLOWED_FIT_LABELS:
            raise ValueError(
                f"product_match_results[{idx}].fit_label 값이 올바르지 않습니다: "
                f"{fit_label}. 허용값: {sorted(ALLOWED_FIT_LABELS)}"
            )

        recommend_action = str(product["recommend_action"]).strip()

        if recommend_action not in ALLOWED_RECOMMEND_ACTIONS:
            raise ValueError(
                f"product_match_results[{idx}].recommend_action 값이 올바르지 않습니다: "
                f"{recommend_action}. 허용값: {sorted(ALLOWED_RECOMMEND_ACTIONS)}"
            )

        normalize_tag_list(
            tags=product.get("reason_tags"),
            allowed_tags=ALLOWED_REASON_TAGS,
            max_count=3,
            key_name=f"product_match_results[{idx}].reason_tags",
        )

        normalize_tag_list(
            tags=product.get("caution_tags"),
            allowed_tags=ALLOWED_CAUTION_TAGS,
            max_count=2,
            key_name=f"product_match_results[{idx}].caution_tags",
        )


def validate_final_routine_item(product: dict[str, Any], idx: int) -> None:
    missing = [key for key in FINAL_ROUTINE_REQUIRED_KEYS if key not in product]

    if missing:
        raise ValueError(f"final_routine[{idx}]에 필요한 key가 없습니다: {missing}")

    validate_optional_int(product.get("slot_order"), f"final_routine[{idx}].slot_order")
    validate_optional_number(product.get("product_score"), f"final_routine[{idx}].product_score")
    validate_optional_int(product.get("price"), f"final_routine[{idx}].price")

    source = str(product["source"]).strip()

    if source not in ALLOWED_SOURCES:
        raise ValueError(
            f"final_routine[{idx}].source는 {sorted(ALLOWED_SOURCES)} 중 하나여야 합니다: "
            f"{source}"
        )


def validate_warning_item(warning: Any, idx: int) -> None:
    if isinstance(warning, str):
        return

    if not isinstance(warning, dict):
        raise ValueError(f"warnings[{idx}]는 dict 또는 string 형식이어야 합니다.")

    if "message" not in warning and "warning_type" not in warning:
        raise ValueError(
            f"warnings[{idx}]에는 message 또는 warning_type 중 하나가 필요합니다."
        )


def validate_routine_recommendation_results(results: Any) -> None:
    if not isinstance(results, dict):
        raise ValueError("routine_recommendation_results는 dict 형식이어야 합니다.")

    missing = [key for key in ROUTINE_RESULT_REQUIRED_KEYS if key not in results]

    if missing:
        raise ValueError(
            "routine_recommendation_results에 필요한 key가 없습니다: "
            f"{missing}"
        )

    if not isinstance(results["final_routine"], list) or not results["final_routine"]:
        raise ValueError(
            "routine_recommendation_results.final_routine은 비어 있지 않은 list여야 합니다."
        )

    if not isinstance(results["warnings"], list):
        raise ValueError("routine_recommendation_results.warnings는 list여야 합니다.")

    for idx, product in enumerate(results["final_routine"]):
        if not isinstance(product, dict):
            raise ValueError(f"final_routine[{idx}]는 dict 형식이어야 합니다.")

        validate_final_routine_item(product, idx)

    for idx, warning in enumerate(results["warnings"]):
        validate_warning_item(warning, idx)

    validate_optional_int(results.get("total_price"), "total_price")


def normalize_product_match_results(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_products = []

    for product in products:
        reason_tags = normalize_tag_list(
            tags=product.get("reason_tags"),
            allowed_tags=ALLOWED_REASON_TAGS,
            max_count=3,
            key_name="reason_tags",
        )

        caution_tags = normalize_tag_list(
            tags=product.get("caution_tags"),
            allowed_tags=ALLOWED_CAUTION_TAGS,
            max_count=2,
            key_name="caution_tags",
        )

        fit_label = str(product["fit_label"]).strip()
        recommend_action = str(product["recommend_action"]).strip()

        normalized_products.append(
            {
                "product_id": product["product_id"],
                "category": str(product["category"]),
                "brand_name": str(product["brand_name"]),
                "product_name": str(product["product_name"]),
                "vanity_fit_score": round(float(product["vanity_fit_score"]), 4),
                "fit_label": fit_label,
                "fit_label_display": FIT_LABEL_DISPLAY_TEXT[fit_label],
                "recommend_action": recommend_action,
                "recommend_action_text": get_recommend_action_text(recommend_action),
                "reason_tags": reason_tags,
                "caution_tags": caution_tags,
                "reason_tag_texts": get_reason_tag_texts(reason_tags),
                "caution_tag_texts": get_caution_tag_texts(caution_tags),
            }
        )

    return normalized_products


def normalize_final_routine_item(product: dict[str, Any]) -> dict[str, Any]:
    category = str(product["category"])

    return {
        "slot_order": int(product["slot_order"]),
        "product_id": product["product_id"],
        "category": category,
        "brand_name": str(product["brand_name"]),
        "product_name": str(product["product_name"]),
        "source": str(product["source"]).strip(),
        "product_score": (
            None
            if product.get("product_score") is None
            else round(float(product["product_score"]), 4)
        ),
        "price": (
            None
            if product.get("price") is None
            else int(product["price"])
        ),
        "usage_guide": get_category_usage_guide(category),
    }


def normalize_warning(warning: Union[str, dict[str, Any]]) -> Union[str, dict[str, Any]]:
    if isinstance(warning, str):
        return warning

    return {
        "warning_type": (
            None
            if warning.get("warning_type") is None
            else str(warning["warning_type"])
        ),
        "product_id": warning.get("product_id"),
        "product_name": (
            None
            if warning.get("product_name") is None
            else str(warning["product_name"])
        ),
        "ingredient": (
            None
            if warning.get("ingredient") is None
            else str(warning["ingredient"])
        ),
        "message": (
            ""
            if warning.get("message") is None
            else str(warning["message"])
        ),
    }


def normalize_routine_recommendation_results(
    results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "final_routine": [
            normalize_final_routine_item(product)
            for product in results["final_routine"]
        ],
        "warnings": [
            normalize_warning(warning)
            for warning in results["warnings"]
        ],
        "total_price": (
            None
            if results.get("total_price") is None
            else int(results["total_price"])
        ),
    }


def normalize_vanity_input(llm_input: dict[str, Any]) -> dict[str, Any]:
    validate_vanity_input(llm_input)

    return {
        "user_id": llm_input["user_id"],
        "result_id": llm_input["result_id"],
        "user_profile": llm_input["user_profile"],
        "skin_analysis_result": {
            key: round(float(llm_input["skin_analysis_result"][key]), 4)
            for key in SKIN_SCORE_KEYS
        },
        "product_match_results": normalize_product_match_results(
            llm_input["product_match_results"]
        ),
        "routine_recommendation_results": normalize_routine_recommendation_results(
            llm_input["routine_recommendation_results"]
        ),
    }


def build_vanity_user_prompt(llm_input: dict[str, Any]) -> str:
    normalized_input = normalize_vanity_input(llm_input)
    compact_input = build_compact_vanity_input(normalized_input)

    return json.dumps(
        compact_input,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_skin_match_response(
    normalized_input: dict[str, Any],
    skin_match_response: dict[str, Any],
) -> None:
    if "overall_summary" not in skin_match_response:
        raise ValueError("skin_match 응답에 overall_summary가 없습니다.")

    comments = skin_match_response.get("product_comments")

    if not isinstance(comments, list):
        raise ValueError("skin_match.product_comments는 list 형식이어야 합니다.")

    input_products = normalized_input["product_match_results"]

    if len(comments) != len(input_products):
        raise ValueError(
            "skin_match.product_comments 개수가 입력 product_match_results 개수와 다릅니다: "
            f"input={len(input_products)}, output={len(comments)}"
        )

    required_comment_keys = [
        "product_id",
        "fit_reason",
        "caution_comment",
    ]

    for idx, comment in enumerate(comments):
        if not isinstance(comment, dict):
            raise ValueError(f"product_comments[{idx}]는 dict 형식이어야 합니다.")

        missing = [key for key in required_comment_keys if key not in comment]

        if missing:
            raise ValueError(f"product_comments[{idx}]에 필요한 key가 없습니다: {missing}")

        input_product = input_products[idx]

        if comment["product_id"] != input_product["product_id"]:
            raise ValueError(
                f"product_comments[{idx}].product_id가 입력값과 다릅니다: "
                f"expected={input_product['product_id']}, output={comment['product_id']}"
            )


def validate_vanity_routine_response(
    normalized_input: dict[str, Any],
    vanity_routine_response: dict[str, Any],
) -> None:
    required_keys = [
        "overall_summary",
        "step_comments",
        "warning_comment",
    ]

    missing = [key for key in required_keys if key not in vanity_routine_response]

    if missing:
        raise ValueError(f"vanity_routine 응답에 필요한 key가 없습니다: {missing}")

    comments = vanity_routine_response.get("step_comments")

    if not isinstance(comments, list):
        raise ValueError("vanity_routine.step_comments는 list 형식이어야 합니다.")

    final_routine = normalized_input["routine_recommendation_results"]["final_routine"]

    if len(comments) != len(final_routine):
        raise ValueError(
            "vanity_routine.step_comments 개수가 입력 final_routine 개수와 다릅니다: "
            f"input={len(final_routine)}, output={len(comments)}"
        )

    required_comment_keys = ["slot_order", "product_id", "comment"]

    for idx, comment in enumerate(comments):
        if not isinstance(comment, dict):
            raise ValueError(f"step_comments[{idx}]는 dict 형식이어야 합니다.")

        missing_comment_keys = [
            key for key in required_comment_keys if key not in comment
        ]

        if missing_comment_keys:
            raise ValueError(
                f"step_comments[{idx}]에 필요한 key가 없습니다: "
                f"{missing_comment_keys}"
            )

        input_product = final_routine[idx]

        if comment["slot_order"] != input_product["slot_order"]:
            raise ValueError(
                f"step_comments[{idx}].slot_order가 입력 final_routine과 다릅니다: "
                f"expected={input_product['slot_order']}, output={comment['slot_order']}"
            )

        if comment["product_id"] != input_product["product_id"]:
            raise ValueError(
                f"step_comments[{idx}].product_id가 입력 final_routine과 다릅니다: "
                f"expected={input_product['product_id']}, output={comment['product_id']}"
            )




def build_skin_match_product_comment_fallback(
    product: dict[str, Any],
    normalized_input: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    product_id = product.get("product_id")
    category_text = get_category_overall_display_text(product.get("category"), count=1)
    category_particle = topic_particle(category_text)
    fit_label = str(product.get("fit_label") or "").strip()
    recommend_action = str(product.get("recommend_action") or "").strip()
    reason_texts = product.get("reason_tag_texts") or []
    caution_texts = product.get("caution_tag_texts") or []

    if fit_label in {"excellent_match", "good_match"}:
        summary = (
            f"{category_text}{category_particle} 현재 피부 상태와 비교적 잘 맞는 편이에요. "
            "피부 고민을 보완하는 방향이 맞아 꾸준히 사용해도 좋아요."
        )
    elif fit_label == "so_so":
        summary = (
            f"{category_text}{category_particle} 현재 피부 고민을 가볍게 보조하는 정도로 활용하기 좋아요. "
            "다만 핵심 제품처럼 강하게 기대하기보다는 피부 반응을 보며 비교해 주세요."
        )
    elif fit_label == "weak_match":
        summary = (
            f"{category_text}{category_particle} 현재 피부 고민과의 맞물림이 조금 약한 편이에요. "
            "사용량과 빈도를 조절하면서 실제 체감을 확인하는 방향이 좋아요."
        )
    else:
        summary = (
            f"{category_text}{category_particle} 현재 피부 고민을 직접적으로 보완하는 힘이 상대적으로 약할 수 있어요. "
            "다른 제품과 비교하거나 교체 후보로 함께 검토해 보세요."
        )

    fit_reason = build_skin_match_fit_reason_fallback(
        product=product,
        normalized_input=normalized_input,
    )

    if caution_texts:
        caution_comment = " ".join(str(item).strip() for item in caution_texts if str(item).strip())
        caution_comment = expand_short_caution_text(caution_comment)
    else:
        caution_comment = "큰 주의 요소는 적지만, 사용 중 피부가 예민해지면 양을 조절해 주세요."

    if recommend_action in {"strong_keep", "keep"}:
        action_comment = "현재 제품은 유지해도 좋아요. 사용 중 건조감이나 자극감이 달라지면 양만 가볍게 조절해 주세요."
    elif recommend_action == "neutral":
        action_comment = "가볍게 사용하면서 피부 반응을 확인해 주세요. 더 잘 맞는 제품이 있다면 함께 비교해 보는 방향이 좋아요."
    elif recommend_action == "caution":
        action_comment = "처음에는 사용량과 빈도를 낮춰 시작해 주세요. 피부가 편안하게 반응하는 범위에서만 천천히 조절하는 편이 좋아요."
    elif recommend_action == "replace":
        action_comment = "현재 제품만 고집하기보다는 다른 제품과 비교해 보세요. 기대한 보완감이 낮다면 교체를 고려하는 방향이 좋아요."
    else:
        action_comment = "피부 반응을 확인하면서 유지 여부를 결정해 주세요."

    return {
        "product_id": product_id,
        "summary": summary,
        "fit_reason": fit_reason,
        "caution_comment": caution_comment,
        "action_comment": action_comment,
    }


def build_step_comment_fallback(product: dict[str, Any]) -> dict[str, Any]:
    slot_order = product.get("slot_order")
    product_id = product.get("product_id")
    category = str(product.get("category") or "").strip()
    source = str(product.get("source") or "").strip()
    role_text = get_category_role_text(category)
    usage_text = get_category_usage_guide(category)

    if source == "vanity":
        prefix = "이미 가지고 있는 제품을 유지한 항목으로"
    elif source == "recommendation":
        prefix = "새로 추천된 제품으로"
    else:
        prefix = "루틴에 포함된 제품으로"

    comment = (
        f"{prefix}, {role_text}을 맡아요. "
        f"{usage_text} 방식으로 사용하고, 처음에는 피부 반응을 보며 양을 조절해 주세요."
    )

    return {
        "slot_order": slot_order,
        "product_id": product_id,
        "comment": comment,
    }


def repair_skin_match_response(
    normalized_input: dict[str, Any],
    skin_match_response: Any,
) -> dict[str, Any]:
    if not isinstance(skin_match_response, dict):
        skin_match_response = {}

    input_products = normalized_input["product_match_results"]
    raw_comments = skin_match_response.get("product_comments")
    if not isinstance(raw_comments, list):
        raw_comments = []

    required_keys = ["fit_reason", "caution_comment"]
    comments_by_id = {}

    for comment in raw_comments:
        if not isinstance(comment, dict):
            continue
        try:
            comment_product_id = int(comment.get("product_id"))
        except (TypeError, ValueError):
            continue
        comments_by_id[comment_product_id] = comment

    repaired_comments = []

    for product in input_products:
        product_id = product.get("product_id")
        try:
            lookup_id = int(product_id)
        except (TypeError, ValueError):
            lookup_id = product_id

        fallback = build_skin_match_product_comment_fallback(product, normalized_input)
        comment = comments_by_id.get(lookup_id, {})

        if not isinstance(comment, dict):
            comment = {}

        repaired = {"product_id": product_id}
        for key in required_keys:
            value = comment.get(key)
            if value is None or str(value).strip() == "":
                value = fallback[key]
            repaired[key] = value

        repaired_comments.append(repaired)

    return {
        "overall_summary": skin_match_response.get("overall_summary") or "",
        "product_comments": repaired_comments,
    }


def repair_vanity_routine_response(
    normalized_input: dict[str, Any],
    vanity_routine_response: Any,
) -> dict[str, Any]:
    if not isinstance(vanity_routine_response, dict):
        vanity_routine_response = {}

    final_routine = normalized_input["routine_recommendation_results"]["final_routine"]
    raw_comments = vanity_routine_response.get("step_comments")
    if not isinstance(raw_comments, list):
        raw_comments = []

    comments_by_key = {}
    for comment in raw_comments:
        if not isinstance(comment, dict):
            continue
        try:
            key = (int(comment.get("slot_order")), int(comment.get("product_id")))
        except (TypeError, ValueError):
            continue
        comments_by_key[key] = comment

    repaired_comments = []

    for product in final_routine:
        fallback = build_step_comment_fallback(product)
        try:
            lookup_key = (int(product.get("slot_order")), int(product.get("product_id")))
        except (TypeError, ValueError):
            lookup_key = (product.get("slot_order"), product.get("product_id"))

        comment = comments_by_key.get(lookup_key, {})
        if not isinstance(comment, dict):
            comment = {}

        repaired_comments.append(
            {
                "slot_order": product.get("slot_order"),
                "product_id": product.get("product_id"),
                "comment": comment.get("comment") or fallback["comment"],
            }
        )

    return {
        "overall_summary": vanity_routine_response.get("overall_summary") or "",
        "step_comments": repaired_comments,
        "warning_comment": vanity_routine_response.get("warning_comment") or "",
    }


def repair_vanity_response(
    normalized_input: dict[str, Any],
    llm_response: Any,
) -> dict[str, Any]:
    if not isinstance(llm_response, dict):
        llm_response = {}

    return {
        "skin_match": repair_skin_match_response(
            normalized_input=normalized_input,
            skin_match_response=llm_response.get("skin_match"),
        ),
        "vanity_routine": repair_vanity_routine_response(
            normalized_input=normalized_input,
            vanity_routine_response=llm_response.get("vanity_routine"),
        ),
    }

def validate_vanity_response(
    normalized_input: dict[str, Any],
    llm_response: dict[str, Any],
) -> None:
    required_keys = ["skin_match", "vanity_routine"]

    missing = [key for key in required_keys if key not in llm_response]

    if missing:
        raise ValueError(f"LLM 응답에 필요한 key가 없습니다: {missing}")

    if not isinstance(llm_response["skin_match"], dict):
        raise ValueError("skin_match 응답은 dict 형식이어야 합니다.")

    if not isinstance(llm_response["vanity_routine"], dict):
        raise ValueError("vanity_routine 응답은 dict 형식이어야 합니다.")

    validate_skin_match_response(
        normalized_input=normalized_input,
        skin_match_response=llm_response["skin_match"],
    )

    validate_vanity_routine_response(
        normalized_input=normalized_input,
        vanity_routine_response=llm_response["vanity_routine"],
    )


def build_final_vanity_result(
    llm_input: dict[str, Any],
    llm_response: dict[str, Any],
) -> dict[str, Any]:
    normalized_input = normalize_vanity_input(llm_input)
    llm_response = repair_vanity_response(normalized_input, llm_response)

    validate_vanity_response(normalized_input, llm_response)

    brand_names = collect_brand_names(normalized_input)

    product_comments = []

    for idx, comment in enumerate(llm_response["skin_match"]["product_comments"]):
        input_product = normalized_input["product_match_results"][idx]

        caution_fallback = (
            "큰 주의 요소는 적지만, 사용 중 피부가 예민해지면 양을 조절해 주세요."
        )

        product_comments.append(
            {
                "product_id": input_product["product_id"],
                "summary": clean_skin_match_comment_text(
                    build_skin_match_summary_text(input_product, normalized_input),
                    brand_names=brand_names,
                    product_name=input_product["product_name"],
                    hide_product_name=False,
                ),
                "fit_reason": enforce_skin_match_field_meaning(
                    clean_skin_match_comment_text(
                        comment["fit_reason"],
                        brand_names=brand_names,
                        product_name=input_product["product_name"],
                        hide_product_name=False,
                    ),
                    field_name="fit_reason",
                    product=input_product,
                    normalized_input=normalized_input,
                ),
                "caution_comment": enforce_skin_match_field_meaning(
                    clean_skin_match_comment_text(
                        comment["caution_comment"],
                        brand_names=brand_names,
                        product_name=input_product["product_name"],
                        hide_product_name=False,
                        fallback=caution_fallback,
                    ),
                    field_name="caution_comment",
                    product=input_product,
                ),
                "action_comment": clean_skin_match_comment_text(
                    build_skin_match_action_comment_text(input_product, normalized_input),
                    brand_names=brand_names,
                    product_name=input_product["product_name"],
                    hide_product_name=False,
                ),
            }
        )

    step_comments = []
    final_routine = normalized_input["routine_recommendation_results"]["final_routine"]

    for idx, comment in enumerate(llm_response["vanity_routine"]["step_comments"]):
        input_product = final_routine[idx]

        cleaned_step_comment = clean_step_comment_text(
            comment["comment"],
            brand_names=brand_names,
            product_name=input_product["product_name"],
        )

        cleaned_step_comment = ensure_step_source_prefix(
            comment=cleaned_step_comment,
            product=input_product,
        )

        step_comments.append(
            {
                "slot_order": input_product["slot_order"],
                "product_id": input_product["product_id"],
                "comment": cleaned_step_comment,
            }
        )

    input_warning_comment = build_warning_comment_from_input(
        warnings=normalized_input["routine_recommendation_results"]["warnings"],
        final_routine=final_routine,
    )

    return {
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "skin_match": {
            "overall_summary": clean_skin_match_overall_text(
                build_skin_match_overall_summary(normalized_input),
                brand_names=brand_names,
            ),
            "product_comments": product_comments,
        },
        "vanity_routine": {
            "overall_summary": clean_routine_overall_text(
                llm_response["vanity_routine"]["overall_summary"],
                brand_names=brand_names,
                normalized_input=normalized_input,
                final_routine=final_routine,
            ),
            "step_comments": step_comments,
            "warning_comment": clean_warning_comment_text(
                input_warning_comment
                if input_warning_comment
                else llm_response["vanity_routine"]["warning_comment"],
                brand_names=brand_names,
            ),
        },
    }


def generate_vanity_llm_result(
    llm_input: dict[str, Any],
    output_dir: str = OUTPUT_DIR,
) -> dict[str, Any]:
    llm_response = call_llm_json(
        system_prompt=VANITY_SYSTEM_PROMPT,
        user_prompt=build_vanity_user_prompt(llm_input),
    )

    final_result = build_final_vanity_result(
        llm_input=llm_input,
        llm_response=llm_response,
    )

    saved_path = save_json(
        data=final_result,
        output_dir=output_dir,
        file_name=(
            f"vanity_user_{llm_input['user_id']}"
            f"_result_{llm_input['result_id']}.json"
        ),
    )

    final_result["saved_path"] = saved_path

    return final_result
