from __future__ import annotations

from datetime import UTC, datetime


def _score_to_grade(name: str, value: int) -> int:
    if name == "acne":
        if value < 20:
            return 0
        if value < 35:
            return 1
        if value < 60:
            return 2
        return 3
    if name == "dryness":
        if value < 20:
            return 0
        if value < 40:
            return 1
        if value < 60:
            return 2
        if value < 80:
            return 3
        return 4
    if name in {"sagging", "pore"}:
        if value < 20:
            return 0
        if value < 35:
            return 1
        if value < 50:
            return 2
        if value < 65:
            return 3
        if value < 80:
            return 4
        return 5
    if name == "pigmentation":
        if value < 15:
            return 0
        if value < 30:
            return 1
        if value < 45:
            return 2
        if value < 60:
            return 3
        if value < 75:
            return 4
        return 5
    if name == "wrinkle":
        if value < 15:
            return 0
        if value < 30:
            return 1
        if value < 45:
            return 2
        if value < 55:
            return 3
        if value < 70:
            return 4
        if value < 85:
            return 5
        return 6
    return 0


def build_skin_scores(image_id: int, user_id: int) -> dict:
    base = (image_id * 7 + user_id * 13) % 100
    display_scores = {
        "acne": 20 + (base % 20),
        "dryness": 30 + (base % 25),
        "sagging": 10 + (base % 18),
        "pore": 35 + (base % 20),
        "pigmentation": 18 + (base % 15),
        "wrinkle": 16 + (base % 14),
    }
    raw_metrics = {name: _score_to_grade(name, value) for name, value in display_scores.items()}
    return {
        "display_scores": display_scores,
        "raw_metrics": raw_metrics,
        "analyzed_at": datetime.now(UTC).isoformat(),
        "model_version": "mock-skin-analysis-v1",
        "analysis_status": "SUCCESS",
    }


def build_skin_summary(result: dict) -> dict:
    display_scores = result["display_scores"]
    sorted_metrics = sorted(display_scores.items(), key=lambda item: item[1], reverse=True)
    top1, top2 = sorted_metrics[0][0], sorted_metrics[1][0]
    name_map = {
        "acne": "트러블",
        "dryness": "건조",
        "sagging": "처짐",
        "pore": "모공",
        "pigmentation": "색소침착",
        "wrinkle": "주름",
    }
    indicator_comments = {
        "acne": "트러블 지표는 비교적 낮은 편이라 저자극 세안과 진정 케어로 유지하는 방향이 적합합니다.",
        "dryness": "건조 지표는 살짝 신경 쓰이는 수준이므로 수분 공급과 보습막 형성을 꾸준히 챙겨주세요.",
        "sagging": "처짐 지표는 낮아 기본적인 보습과 생활 관리면 충분합니다.",
        "pore": "모공 지표가 도드라져 피지 조절과 모공 케어를 함께 해주세요.",
        "pigmentation": "색소침착은 낮지만 자외선 차단을 꾸준히 이어가세요.",
        "wrinkle": "주름 지표는 낮아 예방 중심으로 보습 관리를 유지하세요.",
    }
    return {
        "llm_model": "mock-llm-analysis-v1",
        "prompt_version": "skin_v1",
        "summary_comment": f"전반적으로 {name_map[top1]}과 {name_map[top2]} 지표가 상대적으로 조금 더 눈에 띄어 관련 관리에 신경 쓰는 방향이 적합합니다.",
        "indicator_comments": indicator_comments,
        "generated_at": datetime.now(UTC).isoformat(),
    }
