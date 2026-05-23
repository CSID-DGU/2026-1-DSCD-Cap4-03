from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_backend_env() -> None:
    env_path = PROJECT_ROOT / "backend" / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        if key.startswith("MYSQL_"):
            os.environ.setdefault(f"ROUPLE_{key}", value)


load_backend_env()

from model.vanity.pipeline import run_vanity_pipeline
from model.vanity.schemas import VanityPipelineInput


def parse_ids(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vanity-Based Routine test.")
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--result-id", type=int, default=None)
    parser.add_argument("--vanity-product-ids", type=str, default=None)
    parser.add_argument("--fixed-product-ids", type=str, required=True)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--save-skin-match", action="store_true")
    parser.add_argument("--save-routine", action="store_true")
    args = parser.parse_args()

    pipeline_input = VanityPipelineInput(
        user_id=args.user_id,
        result_id=args.result_id,
        vanity_product_ids=parse_ids(args.vanity_product_ids),
        fixed_product_ids=parse_ids(args.fixed_product_ids),
        budget=args.budget,
    )
    result = run_vanity_pipeline(
        pipeline_input=pipeline_input,
        save_skin_match=args.save_skin_match,
        save_routine=args.save_routine,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
