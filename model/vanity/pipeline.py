from __future__ import annotations

from dataclasses import asdict
from typing import Any

from model.vanity.routine_builder import build_vanity_routine
from model.vanity.schemas import VanityPipelineInput
from model.vanity.skin_match import run_skin_match


def run_vanity_pipeline(
    pipeline_input: VanityPipelineInput,
    candidate_products: list[dict[str, Any]] | None = None,
    save_skin_match: bool = False,
) -> dict[str, Any]:
    skin_match_result = run_skin_match(
        pipeline_input=pipeline_input,
        save_result=save_skin_match,
    )

    routine_result = None
    if pipeline_input.fixed_product_ids:
        routine = build_vanity_routine(
            user_id=pipeline_input.user_id,
            fixed_product_ids=pipeline_input.fixed_product_ids,
            candidate_products=candidate_products or [],
        )
        routine_result = asdict(routine)

    return {
        "user_id": pipeline_input.user_id,
        "result_id": skin_match_result["result_id"],
        "product_match_results": skin_match_result["product_match_results"],
        "routine_recommendation_results": routine_result,
    }


if __name__ == "__main__":
    sample_input = VanityPipelineInput(
        user_id=1,
        result_id=None,
        vanity_product_ids=None,
        fixed_product_ids=None,
        budget=None,
    )
    print(run_vanity_pipeline(sample_input))

