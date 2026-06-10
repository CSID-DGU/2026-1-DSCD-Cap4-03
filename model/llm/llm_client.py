# llm_client.py

import os
import json
import re
from typing import Any, Optional, Dict

from dotenv import load_dotenv
from openai import OpenAI


# 1. env 로드
load_dotenv()

def _get_env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


# 2. 기본 설정
DGU_LLM_API_KEY = _get_env_str("DGU_LLM_API_KEY")
DGU_LLM_BASE_URL = _get_env_str(
    "DGU_LLM_BASE_URL",
    "https://factchat-cloud.mindlogic.ai/v1/gateway"
)
DGU_LLM_MODEL = _get_env_str("DGU_LLM_MODEL", "gpt-5.4-mini")

DEFAULT_TEMPERATURE = _get_env_float("DGU_LLM_TEMPERATURE", 0.6)
DEFAULT_MAX_TOKENS = _get_env_int("DGU_LLM_MAX_TOKENS", 800)


if not DGU_LLM_API_KEY:
    raise RuntimeError(
        "DGU_LLM_API_KEY가 설정되지 않았습니다. "
        ".env 파일에 DGU_LLM_API_KEY를 추가하세요."
    )

if not DGU_LLM_BASE_URL:
    raise RuntimeError(
        "DGU_LLM_BASE_URL이 설정되지 않았습니다. "
        ".env 파일에 DGU_LLM_BASE_URL을 추가하세요."
    )


# 3. OpenAI SDK Client
_client = OpenAI(
    api_key=DGU_LLM_API_KEY,
    base_url=DGU_LLM_BASE_URL
)


def get_llm_client() -> OpenAI:
    """
    다른 파일에서 OpenAI client가 직접 필요할 때 사용.
    """
    return _client


# 4. 토큰 사용량 출력 유틸
def _get_usage_value(usage: Any, key: str) -> Any:
    """
    usage가 dict이거나 OpenAI SDK 객체일 수 있으므로 둘 다 대응.
    """

    if usage is None:
        return None

    if isinstance(usage, dict):
        return usage.get(key)

    return getattr(usage, key, None)


def print_token_usage(response: Any) -> None:
    """
    Gateway 응답의 usage 정보가 있으면 입력/출력/총 토큰을 출력한다.
    동국 Gateway가 usage 정보를 내려주지 않으면 안내 문구만 출력한다.

    .env:
        DGU_LLM_PRINT_USAGE=true  -> 출력
        DGU_LLM_PRINT_USAGE=false -> 출력 안 함
    """

    should_print = _get_env_str("DGU_LLM_PRINT_USAGE", "true").lower()

    if should_print not in ["true", "1", "yes", "y"]:
        return

    usage = getattr(response, "usage", None)

    if usage is None:
        print("\n[LLM Token Usage]")
        print("토큰 사용량: 응답에 usage 정보 없음")
        return

    input_tokens = (
        _get_usage_value(usage, "prompt_tokens")
        or _get_usage_value(usage, "input_tokens")
    )

    output_tokens = (
        _get_usage_value(usage, "completion_tokens")
        or _get_usage_value(usage, "output_tokens")
    )

    total_tokens = _get_usage_value(usage, "total_tokens")

    print("\n[LLM Token Usage]")
    print(f"입력 토큰: {input_tokens}")
    print(f"출력 토큰: {output_tokens}")
    print(f"총 토큰: {total_tokens}")


# 5. JSON 파싱 유틸
def _remove_markdown_code_block(text: str) -> str:
    """
    LLM이 실수로 ```json ... ``` 형태로 출력했을 때 제거.
    """

    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned.strip()


def _extract_json_substring(text: str) -> str:
    """
    JSON 앞뒤로 불필요한 문장이 붙었을 때 dict 부분만 추출.
    """

    cleaned = _remove_markdown_code_block(text)

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if not match:
        raise RuntimeError(
            "LLM 응답에서 JSON 객체를 찾을 수 없습니다.\n"
            f"원본 응답:\n{text}"
        )

    return match.group(0).strip()


def parse_llm_json(text: str) -> Dict[str, Any]:
    """
    LLM 응답 문자열을 dict JSON으로 변환.
    """

    json_text = _extract_json_substring(text)

    try:
        data = json.loads(json_text)

    except json.JSONDecodeError as e:
        raise RuntimeError(
            "LLM 응답을 JSON으로 파싱할 수 없습니다.\n"
            f"원본 응답:\n{text}"
        ) from e

    if not isinstance(data, dict):
        raise RuntimeError(
            "LLM 응답 JSON이 dict 형식이 아닙니다.\n"
            f"원본 응답:\n{text}"
        )

    return data


# 6. 공통 LLM 호출 함수
def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    동국 AI CHAT Gateway를 통해 LLM을 호출하고 JSON dict로 반환.

    사용 예:
        result = call_llm_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt
        )
    """

    selected_model = model_name or DGU_LLM_MODEL
    selected_temperature = (
        DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    )
    selected_max_tokens = (
        DEFAULT_MAX_TOKENS if max_tokens is None else int(max_tokens)
    )

    try:
        response = _client.chat.completions.create(
            model=selected_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=selected_temperature,
            max_tokens=selected_max_tokens,
            response_format={
                "type": "json_object"
            }
        )

        print_token_usage(response)

    except Exception as e:
        raise RuntimeError(
            f"LLM API 호출 실패: {e}"
        ) from e

    try:
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            raise RuntimeError(
                "LLM 응답이 max_tokens 제한으로 중간에 잘렸습니다. "
                "DGU_LLM_MAX_TOKENS 값을 늘려주세요."
            )

        text = response.choices[0].message.content

    except RuntimeError:
        raise

    except Exception as e:
        raise RuntimeError(
            f"LLM 응답 형식을 읽을 수 없습니다.\n응답 객체:\n{response}"
        ) from e

    if not text or not text.strip():
        raise RuntimeError("LLM 응답이 비어 있습니다.")

    return parse_llm_json(text)


def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> str:
    """
    JSON이 아닌 일반 텍스트 응답이 필요할 때 사용.
    현재 프로젝트에서는 주로 call_llm_json 사용 권장.
    """

    selected_model = model_name or DGU_LLM_MODEL
    selected_temperature = (
        DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    )
    selected_max_tokens = (
        DEFAULT_MAX_TOKENS if max_tokens is None else int(max_tokens)
    )

    try:
        response = _client.chat.completions.create(
            model=selected_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=selected_temperature,
            max_tokens=selected_max_tokens
        )

        print_token_usage(response)

    except Exception as e:
        raise RuntimeError(
            f"LLM API 호출 실패: {e}"
        ) from e

    try:
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            raise RuntimeError(
                "LLM 응답이 max_tokens 제한으로 중간에 잘렸습니다. "
                "DGU_LLM_MAX_TOKENS 값을 늘려주세요."
            )

        text = response.choices[0].message.content

    except RuntimeError:
        raise

    except Exception as e:
        raise RuntimeError(
            f"LLM 응답 형식을 읽을 수 없습니다.\n응답 객체:\n{response}"
        ) from e

    if not text or not text.strip():
        raise RuntimeError("LLM 응답이 비어 있습니다.")

    return text.strip()