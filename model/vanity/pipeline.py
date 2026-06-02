from __future__ import annotations

from dataclasses import asdict
from typing import Any

from model.vanity.candidate_generator import prepare_vanity_candidates
from model.vanity.data_loader import load_products
from model.vanity.routine_builder import build_vanity_routine
from model.vanity.routine_saver import save_vanity_routine_result
from model.vanity.schemas import VanityPipelineInput
from model.vanity.skin_match import run_skin_match


def _total_budget_max(pipeline_input: VanityPipelineInput) -> int | None:
    return pipeline_input.total_budget_max if pipeline_input.total_budget_max is not None else pipeline_input.budget


def _remaining_budget_max(pipeline_input: VanityPipelineInput) -> int | None:
    total_budget_max = _total_budget_max(pipeline_input)
    if total_budget_max is None or not pipeline_input.fixed_product_ids:
        return total_budget_max
    fixed_products = load_products(pipeline_input.fixed_product_ids)
    fixed_total = sum(int(product.price or 0) for product in fixed_products)
    return max(0, int(total_budget_max) - fixed_total)


def run_vanity_pipeline(
    pipeline_input: VanityPipelineInput,
    candidate_products: list[dict[str, Any]] | None = None,
    save_skin_match: bool = False,
    save_routine: bool = False,
) -> dict[str, Any]:
    skin_match_result = run_skin_match(
        pipeline_input=pipeline_input,
        save_result=save_skin_match,
    )

    routine_result = None
    recommendation_session_id = None
    if pipeline_input.fixed_product_ids:
        if candidate_products is None:
            candidate_products = prepare_vanity_candidates(
                user_id=pipeline_input.user_id,
                result_id=skin_match_result["result_id"],
                budget=pipeline_input.budget,
                total_budget_min=pipeline_input.total_budget_min,
                total_budget_max=_remaining_budget_max(pipeline_input),
                slot_budget_min_map=pipeline_input.slot_budget_min_map,
                slot_budget_max_map=pipeline_input.slot_budget_max_map,
            )
        routine = build_vanity_routine(
            user_id=pipeline_input.user_id,
            fixed_product_ids=pipeline_input.fixed_product_ids,
            candidate_products=candidate_products,
            total_budget_min=pipeline_input.total_budget_min,
            total_budget_max=_total_budget_max(pipeline_input),
            slot_budget_min_map=pipeline_input.slot_budget_min_map,
            slot_budget_max_map=pipeline_input.slot_budget_max_map,
        )
        if save_routine:
            recommendation_session_id = save_vanity_routine_result(
                user_id=pipeline_input.user_id,
                result_id=skin_match_result["result_id"],
                routine=routine,
                budget=_total_budget_max(pipeline_input),
                total_budget_min=pipeline_input.total_budget_min,
                slot_budget_min_map=pipeline_input.slot_budget_min_map,
                slot_budget_max_map=pipeline_input.slot_budget_max_map,
            )
        routine_result = asdict(routine)

    return {
        "user_id": pipeline_input.user_id,
        "result_id": skin_match_result["result_id"],
        "recommendation_session_id": recommendation_session_id,
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

