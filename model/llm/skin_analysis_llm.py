# skin_analysis_llm.py

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_client import call_llm_json


PROMPT_VERSION = "skin_v8"
OUTPUT_DIR = "outputs/llm_results/skin_analysis"


SKIN_ANALYSIS_SYSTEM_PROMPT = """
AI 피부 score를 사용자용 JSON으로 바꾼다. score는 0~1, 높을수록 고민이 두드러진다.
지표: acne=트러블, dryness=건조, sagging=처짐, pore=모공, pigmentation=색소침착, wrinkle=주름.

공통 규칙:
- 입력 score, priority_info만 사용. 없는 상태/제품/성분/피부타입 생성 금지.
- 진단/치료/완치/질환/증상, score/등급/수치 표현 금지.
- acne는 "트러블"로 작성.
- 없는 합성어 생성 금지. 특히 "잔탄력" 절대 금지.
- JSON만 출력. 마크다운 금지.

summary_comment:
- 2문장 이내.
- top_indicators는 더 신경 쓸 고민, stable_indicators는 안정적인 고민으로 설명.
- 관리 방향은 수분 유지/장벽 보호/자극 완화/피부결 정돈/기본 관리 유지 중 선택.
- 지표 나열은 쉼표만 사용. "·" 금지.

indicator_comments:
- 6개 지표별 1문장, 70~110자.
- 지표명으로 시작 금지.
- 현재 상태 해석 + 피부에 나타날 수 있는 변화 + 관리 방향 포함.
- top은 관리 강화, stable은 유지·기본 관리 중심.
- 자연스러운 사용자 말투. 딱딱한 보고서 말투 금지.
- 다양한 표현은 가능하지만 실제로 쓰는 자연스러운 단어만 사용.

수준별 표현:
- 높음: "쉽게 부족해질 수 있어", "조금 더 신경 써서", "관리 비중을 높여"
- 중간: "함께 관리하면", "꾸준히 정돈하면", "균형을 맞추면"
- 낮음: "크게 두드러지지 않아", "현재 상태를 유지하면", "기본 관리를 이어가면"

금지 표현:
영향이 뚜렷해, 영향이 비교적 뚜렷해, 신호는 낮은 편, 중간 수준으로 나타나, 비교적 낮은 편, 상대적으로 두드러져, 예방 중심 접근, 현재 수준, 항목, 지표, 흐름, 다스리다, 적합합니다, 안정적입니다, 관리가 필요합니다, 양호합니다, 잔탄력, 잔탄력 변화, 잔탄력 관리, 잔탄력 관찰.

대체 표현:
건조 높음=수분이 쉽게 부족해질 수 있어.
처짐 낮음=탄력 저하는 크게 두드러지지 않아.
처짐 표현=탄력, 탄력 저하, 탄력감 유지, 현재 상태 유지 중에서만 사용.
모공 중간=피지와 피부결 관리가 함께 필요해.
색소 낮음=톤 변화가 크게 두드러지지는 않아.
주름 낮음=잔주름은 크게 두드러지지 않아.
예방 중심=꾸준히 유지하는 관리.

지표별 방향:
acne=저자극 세안/진정/피지 밸런스
dryness=수분/보습막/장벽
sagging=보습/탄력 유지/탄력감 유지
pore=피지 밸런스/피부결 정돈
pigmentation=자외선 차단/피부톤 관리
wrinkle=수분 유지/탄력/잔주름 예방

좋은 문장 예시:
수분이 쉽게 부족해질 수 있어 피부가 당기거나 결이 거칠어 보일 수 있으니, 수분 공급 후 보습막으로 오래 잡아주는 관리가 우선이에요.
피지와 피부결 관리가 함께 필요해 번들거림이나 결 흐트러짐이 느껴질 수 있으니, 자극을 줄이면서 꾸준히 정돈하는 방향이 좋아요.
탄력 저하는 크게 두드러지지 않아 급하게 관리 강도를 높이기보다는, 보습과 생활 습관으로 탄력감을 유지하는 방향이 좋아요.
톤 변화가 크게 두드러지지는 않아 과한 미백 관리보다는, 자외선 차단과 기본 보습을 꾸준히 이어가는 방향이 좋아요.
잔주름은 크게 두드러지지 않아 자극적인 관리를 늘리기보다는, 수분 유지와 탄력 관리 습관을 이어가면 좋아요.

출력:
{"summary_comment":"...","indicator_comments":{"acne":"...","dryness":"...","sagging":"...","pore":"...","pigmentation":"...","wrinkle":"..."}}
"""

SCORE_KEYS = [
    "acne_score",
    "dryness_score",
    "sagging_score",
    "pore_score",
    "pigmentation_score",
    "wrinkle_score"
]

REQUIRED_INPUT_KEYS = [
    "result_id",
    "image_id",
    "user_id",
    "analyzed_at",
    *SCORE_KEYS
]

INDICATOR_KEYS = [
    "acne",
    "dryness",
    "sagging",
    "pore",
    "pigmentation",
    "wrinkle"
]

INDICATOR_LABELS = {
    "acne": "트러블",
    "dryness": "건조",
    "sagging": "처짐",
    "pore": "모공",
    "pigmentation": "색소침착",
    "wrinkle": "주름"
}


FORBIDDEN_MENTIONS = {
    "acne": [
        "트러블 항목은 ", "트러블 항목이 ", "트러블 지표는 ", "트러블 지표가 ",
        "트러블 관련 고민은 ", "트러블 관련 고민이 ",
        "트러블 고민은 ", "트러블 고민이 ",
        "트러블은 ", "트러블이 ", "트러블의 경우 ",
        "트러블 관리에서는 ", "트러블 관리의 경우 "
    ],
    "dryness": [
        "건조 항목은 ", "건조 항목이 ", "건조 지표는 ", "건조 지표가 ",
        "건조 관련 고민은 ", "건조 관련 고민이 ",
        "건조 고민은 ", "건조 고민이 ",
        "건조는 ", "건조가 ", "건조의 경우 ",
        "건조 관리에서는 ", "건조 관리의 경우 "
    ],
    "sagging": [
        "처짐 항목은 ", "처짐 항목이 ", "처짐 지표는 ", "처짐 지표가 ",
        "처짐 관련 고민은 ", "처짐 관련 고민이 ",
        "처짐 고민은 ", "처짐 고민이 ",
        "처짐은 ", "처짐이 ", "처짐의 경우 ",
        "처짐 관리에서는 ", "처짐 관리의 경우 "
    ],
    "pore": [
        "모공 항목은 ", "모공 항목이 ", "모공 지표는 ", "모공 지표가 ",
        "모공 관련 고민은 ", "모공 관련 고민이 ",
        "모공 고민은 ", "모공 고민이 ",
        "모공은 ", "모공이 ", "모공의 경우 ",
        "모공 관리에서는 ", "모공 관리의 경우 "
    ],
    "pigmentation": [
        "색소침착 항목은 ", "색소침착 항목이 ", "색소침착 지표는 ", "색소침착 지표가 ",
        "색소침착 관련 고민은 ", "색소침착 관련 고민이 ",
        "색소침착 고민은 ", "색소침착 고민이 ",
        "색소침착은 ", "색소침착이 ", "색소침착의 경우 ",
        "색소침착 관리에서는 ", "색소침착 관리의 경우 "
    ],
    "wrinkle": [
        "주름 항목은 ", "주름 항목이 ", "주름 지표는 ", "주름 지표가 ",
        "주름 관련 고민은 ", "주름 관련 고민이 ",
        "주름 고민은 ", "주름 고민이 ",
        "주름은 ", "주름이 ", "주름의 경우 ",
        "주름 관리에서는 ", "주름 관리의 경우 "
    ]
}


STYLE_REPLACEMENTS = {
    # 없는 단어/비자연어 최우선 보정
    "잔탄력 변화를 덜 느끼게 해주는 쪽이 좋아요": "탄력감을 유지하는 방향이 좋아요",
    "잔탄력 변화를 덜 느끼게": "탄력감을 유지할 수 있게",
    "잔탄력 변화가 덜 느껴지게": "탄력감을 유지할 수 있게",
    "잔탄력 변화": "탄력 변화",
    "잔탄력 관리": "탄력 관리",
    "잔탄력 관찰": "탄력 변화 관찰",
    "잔탄력": "탄력",
    "탄력 변화를 덜 느끼게 해주는 쪽이 좋아요": "탄력감을 유지하는 방향이 좋아요",
    "탄력 변화를 덜 느끼게": "탄력감을 유지할 수 있게",

    # 기존 어색한 표현
    "잔크게 두드러지지 않아": "크게 두드러지지 않아",
    "잔자리잡지 않도록": "잔주름이 자리 잡지 않도록",
    "잔자리 잡지 않도록": "잔주름이 자리 잡지 않도록",
    "잔주름이 자리잡지 않도록": "잔주름이 자리 잡지 않도록",
    "살피고,": "살펴야 할 것으로 보이고,",
    "흐름이 좋습니다": "관리 방향이 좋습니다",
    "흐름이 좋아요": "관리 방향이 좋아요",
    "흐름이 편해질 거예요": "피부 컨디션을 더 편안하게 유지하는 데 도움이 돼요",
    "흐름이 편해질": "관리 방향이 자연스러워질",
    "피부 흐름": "관리 방향",
    "흐름": "관리 방향",
    "·": ", ",

    # 다스리다 계열 제거
    "피지와 피부결을 차분히 다스리면": "피지 밸런스와 피부결을 함께 정돈하면",
    "피지와 피부결을 다스리면": "피지 밸런스와 피부결을 함께 정돈하면",
    "피지를 다스리면": "피지 밸런스를 맞추면",
    "다스리면": "정돈하면",
    "다스리는": "정돈하는",
    "다스려": "정돈해",

    # 영향/신호/수준 계열 제거
    "영향이 비교적 뚜렷해": "관리 비중을 조금 더 높여",
    "영향이 뚜렷해": "관리 비중을 높여",
    "영향이 있는 편이라": "조금 더 신경 써서",
    "영향이 있는 편": "조금 더 신경 쓸 부분",
    "영향이 낮은 편이라": "크게 두드러지지 않아",
    "영향은 낮은 편이라": "크게 두드러지지 않아",

    "신호는 낮은 편이라": "크게 두드러지지 않아",
    "신호가 낮은 편이라": "크게 두드러지지 않아",
    "신호는 낮은 편": "크게 두드러지지 않음",
    "신호가 낮은 편": "크게 두드러지지 않음",
    "신호가 적어": "크게 두드러지지 않아",
    "신호는 적어": "크게 두드러지지 않아",

    "중간 수준으로 나타나": "꾸준히 관리할 필요가 있어",
    "중간 정도로 나타나": "꾸준히 관리할 필요가 있어",
    "중간 정도라": "함께 관리하면 좋아",
    "중간 수준이라": "함께 관리하면 좋아",
    "중간 수준입니다": "꾸준히 관리하면 좋아요",

    "비교적 낮은 편이라": "크게 두드러지지 않아",
    "비교적 낮은 편": "크게 두드러지지 않음",
    "낮은 편이라": "크게 두드러지지 않아",
    "낮은 편입니다": "크게 두드러지지 않아요",
    "낮게 나타나": "크게 두드러지지 않아",

    "상대적으로 두드러져": "조금 더 신경 써서 살펴야 해",
    "상대적으로 두드러지는 편이라": "조금 더 신경 써서 살펴야 해",
    "두드러지는 편이라": "조금 더 신경 써서 살펴야 해",
    "두드러집니다": "조금 더 신경 써서 살펴야 해요",

    # 예방 중심 표현 보정
    "예방 중심 접근이 적합해요": "꾸준히 유지하는 관리가 좋아요",
    "예방 중심 접근이 좋아요": "꾸준히 유지하는 관리가 좋아요",
    "예방 중심 접근": "꾸준히 유지하는 관리",
    "예방 중심으로 관리해": "꾸준히 관리해",
    "예방 중심으로 접근해": "꾸준히 관리해",

    # 딱딱한 보고서 말투 완화
    "현재 수준": "현재 상태",
    "해당 고민": "이 부분",
    "해당 부위": "이 부분",
    "관리하는 것이 적합합니다": "관리하면 좋아요",
    "관리하는 것이 좋습니다": "관리하면 좋아요",
    "관리가 필요합니다": "관리해 주세요",
    "관리 필요성이 있습니다": "관리해 주세요",
    "적합합니다": "좋아요",
    "안정적입니다": "안정적으로 보입니다",
    "양호합니다": "크게 두드러지지 않습니다",
    "권장됩니다": "좋아요",
    "도움이 됩니다": "도움이 돼요",
    "유지하는 것이 좋습니다": "유지하면 좋아요",
    "이어가는 것이 좋습니다": "이어가면 좋아요",

    # 문장 톤 통일
    "합니다.": "해요.",
    "됩니다.": "돼요.",
    "좋습니다.": "좋아요.",
    "필요합니다.": "필요해요.",
    "보입니다.": "보여요.",
    "있습니다.": "있어요.",
    "없습니다.": "없어요."
}


def save_json(data: dict[str, Any], output_dir: str, file_name: str) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / file_name

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(file_path)


def validate_skin_llm_input(llm_input: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_INPUT_KEYS if key not in llm_input]

    if missing:
        raise ValueError(f"Skin LLM input에 필요한 key가 없습니다: {missing}")

    for key in SCORE_KEYS:
        score = float(llm_input[key])

        if not 0 <= score <= 1:
            raise ValueError(
                f"{key} 값이 올바르지 않습니다: {score}. "
                "score는 0~1 범위여야 합니다."
            )


def get_score_map(llm_input: dict[str, Any]) -> dict[str, float]:
    return {
        "acne": float(llm_input["acne_score"]),
        "dryness": float(llm_input["dryness_score"]),
        "sagging": float(llm_input["sagging_score"]),
        "pore": float(llm_input["pore_score"]),
        "pigmentation": float(llm_input["pigmentation_score"]),
        "wrinkle": float(llm_input["wrinkle_score"])
    }


def get_priority_info(llm_input: dict[str, Any], top_n: int = 3) -> dict[str, Any]:
    sorted_items = sorted(
        get_score_map(llm_input).items(),
        key=lambda item: item[1],
        reverse=True
    )

    return {
        "top_indicators": [
            {
                "key": key,
                "label": INDICATOR_LABELS[key],
                "score": score
            }
            for key, score in sorted_items[:top_n]
        ],
        "stable_indicators": [
            {
                "key": key,
                "label": INDICATOR_LABELS[key],
                "score": score
            }
            for key, score in sorted_items[top_n:]
        ]
    }


def build_skin_user_prompt(llm_input: dict[str, Any]) -> str:
    validate_skin_llm_input(llm_input)

    clean_input = {
        key: float(llm_input[key]) if key in SCORE_KEYS else llm_input[key]
        for key in REQUIRED_INPUT_KEYS
    }

    prompt_input = {
        "priority_info": get_priority_info(clean_input),
        "llm_input": clean_input
    }

    return "JSON input:\n" + json.dumps(
        prompt_input,
        ensure_ascii=False,
        separators=(",", ":")
    )


def validate_skin_llm_response(llm_response: dict[str, Any]) -> None:
    if "summary_comment" not in llm_response:
        raise ValueError("LLM 응답에 summary_comment가 없습니다.")

    comments = llm_response.get("indicator_comments")

    if not isinstance(comments, dict):
        raise ValueError("indicator_comments는 dict 형식이어야 합니다.")

    missing = [key for key in INDICATOR_KEYS if key not in comments]

    if missing:
        raise ValueError(f"indicator_comments에 필요한 key가 없습니다: {missing}")


def clean_text(text: Any) -> str:
    cleaned = str(text).strip()

    # 여러 단계 치환이 필요한 경우가 있어 2회 반복
    for _ in range(2):
        for before, after in STYLE_REPLACEMENTS.items():
            cleaned = cleaned.replace(before, after)

    cleaned = cleaned.replace(" ,", ",")
    cleaned = cleaned.replace(",,", ",")
    cleaned = cleaned.replace(",  ", ", ")
    cleaned = cleaned.replace(" .", ".")
    cleaned = cleaned.replace("..", ".")
    cleaned = cleaned.replace("요..", "요.")
    cleaned = cleaned.replace("니다..", "니다.")
    cleaned = cleaned.replace("  ", " ")

    cleaned = cleaned.replace("않음, ", "않아, ")
    cleaned = cleaned.replace("않음.", "않아요.")
    cleaned = cleaned.replace("좋아.", "좋아요.")
    cleaned = cleaned.replace("필요해.", "필요해요.")
    cleaned = cleaned.replace("우선이에요.", "우선이에요.")

    return " ".join(cleaned.split())


def clean_indicator_comments(
    indicator_comments: dict[str, Any]
) -> dict[str, str]:
    cleaned_comments = {}

    for indicator in INDICATOR_KEYS:
        text = clean_text(indicator_comments[indicator])

        for mention in FORBIDDEN_MENTIONS.get(indicator, []):
            if text.startswith(mention):
                text = text.replace(mention, "", 1)
            else:
                text = text.replace(mention, "")

        text = clean_text(text)
        text = text.lstrip(" ,.-")

        if text and not text.endswith(("요.", "다.", ".")):
            text += "."

        cleaned_comments[indicator] = clean_text(text)

    return cleaned_comments


def build_final_skin_result(llm_response: dict[str, Any]) -> dict[str, Any]:
    validate_skin_llm_response(llm_response)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_comment": clean_text(llm_response["summary_comment"]),
        "indicator_comments": clean_indicator_comments(
            llm_response["indicator_comments"]
        )
    }


def generate_skin_llm_result(
    llm_input: dict[str, Any],
    output_dir: str = OUTPUT_DIR
) -> dict[str, Any]:
    llm_response = call_llm_json(
        system_prompt=SKIN_ANALYSIS_SYSTEM_PROMPT,
        user_prompt=build_skin_user_prompt(llm_input)
    )

    final_result = build_final_skin_result(llm_response)

    saved_path = save_json(
        data=final_result,
        output_dir=output_dir,
        file_name=(
            f"skin_result_{llm_input['result_id']}"
            f"_user_{llm_input['user_id']}"
            f"_image_{llm_input['image_id']}.json"
        )
    )

    final_result["saved_path"] = saved_path

    return final_result
