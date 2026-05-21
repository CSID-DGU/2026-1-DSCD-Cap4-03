from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_backend_env() -> None:
    env_path = PROJECT_ROOT / "backend" / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)
        if key.startswith("MYSQL_"):
            os.environ.setdefault(f"ROUPLE_{key}", value)


load_backend_env()

from model.vanity.schemas import VanityPipelineInput
from model.vanity.skin_match import run_skin_match


def parse_product_ids(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run My Vanity Skin Match with USER_VANITY products."
    )
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument(
        "--result-id",
        type=int,
        default=None,
        help="Skin analysis result_id. Uses latest result if omitted.",
    )
    parser.add_argument(
        "--product-ids",
        type=str,
        default=None,
        help="Comma-separated product ids. Uses USER_VANITY products if omitted.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results into VANITY_MATCH_SESSION / VANITY_MATCH_ITEM.",
    )
    args = parser.parse_args()

    pipeline_input = VanityPipelineInput(
        user_id=args.user_id,
        result_id=args.result_id,
        vanity_product_ids=parse_product_ids(args.product_ids),
    )

    result = run_skin_match(
        pipeline_input=pipeline_input,
        save_result=args.save,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
