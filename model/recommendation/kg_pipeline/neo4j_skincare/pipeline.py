from typing import Any
import json
from pathlib import Path
import sys
import time
import pandas as pd
import pymysql

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# config.py
from model.recommendation.kg_pipeline.neo4j_skincare.config import OUTPUT_DIR, driver
from model.recommendation.kg_pipeline.neo4j_skincare.graph.load_graph import create_user_session
from model.recommendation.kg_pipeline.neo4j_skincare.rerank.hard_filter import hard_filter
from model.recommendation.kg_pipeline.neo4j_skincare.rerank.soft_score import soft_score
from model.recommendation.kg_pipeline.neo4j_skincare.routine.routine_builder import build_routines, build_value_routines
from model.recommendation.kg_pipeline.neo4j_skincare.services.user_data import (
    _load_candidates_from_embedding,
    _load_product_catalog,
    _load_user_context,
    _mysql_connect,
    _norm_brand_name_key,
    _print_run_context,
)
from model.recommendation.kg_pipeline.neo4j_skincare.services.reco_policy import (
    _attach_all_in_one_to_routines,
    _build_price_map,
)
# MySQL connection

# Normalization utilities




# Data Loading (MySQL)

# Load user context from MySQL (profile, skin analysis, allergies, wishlist, etc.)




def _insert_recommendation_results(
    user_id: int,
    session_meta: dict[str, Any],
    candidates: pd.DataFrame,
    routines: list[dict[str, Any]],
    reranked: pd.DataFrame,
    total_budget_min: float | None,
    total_budget_max: float | None,
    slot_budget_min_map: dict[str, float] | None,
    slot_budget_max_map: dict[str, float] | None,
    session_status: str,
    failure_reason: str | None,
    budget_check_passed: bool,
) -> int:
    print(f"[save] start session insert: routines={len(routines)}, reranked_rows={len(reranked)}, candidate_rows={len(candidates)}")
    conn = _mysql_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO RECOMMENDATION_SESSION (
                    user_id, image_id, result_id, strict_budget,
                    total_budget_min, total_budget_max, slot_budget_min_json, slot_budget_max_json,
                    budget_check_passed, session_status, failure_reason
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    session_meta["image_id"],
                    session_meta["result_id"],
                    0,
                    int(total_budget_min) if total_budget_min is not None else None,
                    int(total_budget_max) if total_budget_max is not None else None,
                    json.dumps(slot_budget_min_map, ensure_ascii=False) if slot_budget_min_map else None,
                    json.dumps(slot_budget_max_map, ensure_ascii=False) if slot_budget_max_map else None,
                    1 if budget_check_passed else 0,
                    session_status,
                    failure_reason,
                ),
            )
            rec_session_id = int(cur.lastrowid)
            print(f"[save] session row inserted: rec_session_id={rec_session_id}")
            key_map: dict[str, int] = {}
            key_map_norm: dict[str, int] = {}
            if "product_id" in reranked.columns:
                rr = reranked.copy()
                rr["product_id_num"] = pd.to_numeric(rr["product_id"], errors="coerce")
                rr = rr[rr["product_id_num"].notna()]
                for _, r in rr.iterrows():
                    brand = str(r["Brand"])
                    name = str(r["product_name"])
                    pid = int(r["product_id_num"])
                    key_map[f"{brand}::{name}"] = pid
                    key_map_norm[_norm_brand_name_key(brand, name)] = pid
            # Resolve missing product_id from PRODUCT table using normalized brand/name.
            product_catalog = _load_product_catalog()
            union_df = pd.concat([candidates, reranked], ignore_index=True) if not reranked.empty else candidates
            key_candidates = [
                (str(r.get("Brand")), str(r.get("product_name")))
                for _, r in union_df.iterrows()
                if pd.notna(r.get("Brand")) and pd.notna(r.get("product_name"))
            ]
            for brand, name in set(key_candidates):
                k = f"{brand}::{name}"
                if k in key_map:
                    continue
                hit = product_catalog.get(_norm_brand_name_key(brand, name))
                if hit:
                    pid = int(hit[0])
                    key_map[k] = pid
                    key_map_norm[_norm_brand_name_key(brand, name)] = pid
            print(f"[save] inserting routines/items: routine_count={len(routines)}")
            for ridx, routine in enumerate(routines, start=1):
                rule_conflicts = routine.get("rule_conflict_log", [])
                smiles_conflicts = routine.get("smiles_conflict_log", [])
                conflict_notes = []
                conflict_notes.extend([f"[RULE] {line}" for line in rule_conflicts])
                conflict_notes.extend([f"[SMILES] {line}" for line in smiles_conflicts])
                cur.execute(
                    """
                    INSERT INTO RECOMMENDATION_ROUTINE (
                        session_id, routine_rank, routine_label, ampm_mode, routine_score,
                        has_conflict, conflict_pairs
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        rec_session_id,
                        ridx,
                        str(routine.get("routine_label") or "").replace(" Routine", "") or None,
                        routine.get("am_pm_label"),
                        float(routine.get("total_score", 0.0)),
                        1 if rule_conflicts else 0,
                        "; ".join(conflict_notes) if conflict_notes else None,
                    ),
                )
                routine_id = int(cur.lastrowid)
                print(f"[save] routine inserted: rank={ridx}, routine_id={routine_id}, item_count={len(routine.get('products', []))}")
                for sidx, p in enumerate(routine.get("products", []), start=1):
                    brand = p.get("brand")
                    name = p.get("name")
                    product_key = f"{brand}::{name}"
                    product_id = key_map.get(product_key)
                    if product_id is None:
                        product_id = key_map_norm.get(_norm_brand_name_key(brand, name))
                    if product_id is None:
                        hit = product_catalog.get(_norm_brand_name_key(brand, name))
                        if hit:
                            product_id = int(hit[0])
                    cur.execute(
                        """
                        INSERT INTO RECOMMENDATION_ITEM (
                            routine_id, slot_order, category, product_id, product_score
                        ) VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            routine_id,
                            sidx,
                            p.get("category"),
                            product_id,
                            float(p.get("S_rerank", 0.0)),
                        ),
                    )
    finally:
        conn.close()
    print(f"[save] completed: rec_session_id={rec_session_id}")
    return rec_session_id


def _build_value_with_expand(
    reranked_for_routine: pd.DataFrame,
    gender: str,
    session_id: str,
    top_n: int = 20,
    start_beam: int = 500,
    step: int = 100,
    max_beam: int = 1500,
    total_budget_min: float | None = None,
    total_budget_max: float | None = None,
    slot_budget_min_map: dict[str, float] | None = None,
    slot_budget_max_map: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    beam = max(start_beam, 1)
    while beam <= max_beam:
        value_candidates = build_value_routines(
            reranked_for_routine,
            gender,
            session_id,
            top_n=top_n,
            beam_width=beam,
            total_budget_min=total_budget_min,
            total_budget_max=total_budget_max,
            slot_budget_min_map=slot_budget_min_map,
            slot_budget_max_map=slot_budget_max_map,
        )
        if value_candidates:
            if beam > start_beam:
                print(f"[info][value] beam expanded to {beam} and found {len(value_candidates)} routines")
            return value_candidates
        beam += step
    print(f"[warn][value] no value routine found up to beam={max_beam}")
    return []


def _safe_price(value: Any) -> float | None:
    if value is None or str(value) == "nan":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_category_name(value: Any) -> str:
    base = " ".join(str(value or "").strip().lower().split())
    aliases = {
        "toner": "toner+toner pads",
        "toners": "toner+toner pads",
        "toner pads": "toner+toner pads",
        "toner pad": "toner+toner pads",
        "toner + toner pads": "toner+toner pads",
        "essence": "essences/ampoules/serums",
        "essences": "essences/ampoules/serums",
        "ampoule": "essences/ampoules/serums",
        "ampoules": "essences/ampoules/serums",
        "serum": "essences/ampoules/serums",
        "serums": "essences/ampoules/serums",
        "essence/ampoule/serum": "essences/ampoules/serums",
        "face moisturizers": "cream/gel",
        "all in one": "all-in-one",
    }
    return aliases.get(base, base)


def _resolve_slot_budget_value(budget_map: dict[str, float] | None, category: Any) -> float | None:
    if not budget_map:
        return None
    category_norm = _norm_category_name(category)
    for k, v in budget_map.items():
        if _norm_category_name(k) == category_norm:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _routine_budget_ok(
    routine: dict[str, Any],
    price_map: dict[str, Any],
    total_budget_min: float | None,
    total_budget_max: float | None,
    slot_budget_min_map: dict[str, float] | None,
    slot_budget_max_map: dict[str, float] | None,
) -> bool:
    total_price = 0.0
    for p in routine.get("products", []):
        price = _safe_price(p.get("price"))
        if price is None:
            price = _safe_price(price_map.get(_norm_brand_name_key(p.get("brand"), p.get("name"))))
        if price is None:
            return not any([total_budget_min, total_budget_max, slot_budget_min_map, slot_budget_max_map])

        slot_min = _resolve_slot_budget_value(slot_budget_min_map, p.get("category"))
        slot_max = _resolve_slot_budget_value(slot_budget_max_map, p.get("category"))
        if slot_min is not None and price < slot_min:
            return False
        if slot_max is not None and price > slot_max:
            return False
        total_price += price

    if total_budget_min is not None and total_price < float(total_budget_min):
        return False
    if total_budget_max is not None and total_price > float(total_budget_max):
        return False
    return True


def _format_price_text(price: Any) -> str:
    if price is None or str(price) == "nan":
        return "N/A"
    return f"{int(float(price)):,}원"


def _print_am_pm_details(routine: dict[str, Any]) -> None:
    label = routine.get("am_pm_label")
    if label == "am+pm":
        return

    if label in {"pm_only", "check_required"}:
        pm_hits = routine.get("am_hit_details", [])
        if pm_hits:
            print("  pm_only products:")
            for hit in pm_hits:
                print(f"    - {hit['brand']} - {hit['product_name']} | ingredient={hit['ingredient']}")

    if label in {"am_only", "check_required"}:
        am_hits = routine.get("pm_hit_details", [])
        if am_hits:
            print("  am_only products:")
            for hit in am_hits:
                print(f"    - {hit['brand']} - {hit['product_name']} | ingredient={hit['ingredient']}")


def _print_conflict_details(routine: dict[str, Any]) -> None:
    rule_logs = routine.get("rule_conflict_log", [])
    smiles_logs = routine.get("smiles_conflict_log", [])

    if rule_logs:
        print("  rule conflicts:")
        for line in rule_logs:
            print(f"    - {line}")

    if smiles_logs:
        print("  smiles warnings:")
        for line in smiles_logs:
            print(f"    - {line}")










def run_pipeline(
    user_id: int,
    image_id: int | None = None,
    top_n: int | None = None,
    total_budget: float | None = None,
    slot_budget_map: dict[str, float] | None = None,
    total_budget_min: float | None = None,
    total_budget_max: float | None = None,
    slot_budget_min_map: dict[str, float] | None = None,
    slot_budget_max_map: dict[str, float] | None = None,
) -> list[dict]:
    ctx = _load_user_context(user_id, image_id)
    candidates = _load_candidates_from_embedding(ctx.get("image_id"), ctx.get("image_name"), ctx["gender"])
    _print_run_context(ctx, user_id=user_id, candidates=candidates)
    session_id = f"user::{user_id}::image::{ctx['image_id']}"
    effective_total_budget_max = total_budget_max if total_budget_max is not None else total_budget
    effective_slot_budget_max_map = slot_budget_max_map if slot_budget_max_map is not None else slot_budget_map
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with driver.session() as s:
        s.execute_write(
            create_user_session,
            session_id,
            ctx["skin_data"],
            ctx["gender"],
            ctx["allergies"],
            ctx.get("wishlist_product_keys", []),
            (ctx.get("profile") or {}).get("skin_type"),
        )
    filtered, drop_log = hard_filter(
        candidates,
        session_id,
        gender=ctx["gender"],
        total_budget=total_budget,
        slot_budget_map=slot_budget_map,
        total_budget_min=total_budget_min,
        total_budget_max=total_budget_max,
        slot_budget_min_map=slot_budget_min_map,
        slot_budget_max_map=slot_budget_max_map,
    )
    print("\n=== Rountine Result ===")
    print(f"Hard Filter: {len(candidates)} -> {len(filtered)} (dropped={len(drop_log)})")
    if not drop_log.empty:
        drop_log.to_csv(OUTPUT_DIR / f"drop_log_user_{user_id}.csv", index=False)
        print(f"Drop reasons: {drop_log['drop_reason'].value_counts().to_dict()}")
    all_in_one_pick = None
    if filtered.empty:
        reranked = pd.DataFrame(
            columns=[
                "Brand",
                "product_name",
                "query_category",
                "score",
                "product_key",
                "vector_score",
                "concern_match_score",
                "skin_type_bonus",
                "wishlist_bonus",
                "irritation_penalty",
                "review_score",
                "S_rerank",
            ]
        )
        routines = []
    else:
        scored_rows = []
        for _, row in filtered.iterrows():
            scores = soft_score(row["product_key"], session_id, float(row["score"]), user_id=user_id)
            scored_rows.append({**row.to_dict(), **scores})
        reranked = pd.DataFrame(scored_rows).sort_values("S_rerank", ascending=False)
        # All-In-One is treated as a standalone recommendation (not routine-combination slot)
        all_in_one_df = reranked[
            reranked["query_category"].astype(str).str.lower().str.replace(" ", "", regex=False).isin(["all-in-one", "allinone"])
        ].copy()
        all_in_one_pick = all_in_one_df.iloc[0].to_dict() if not all_in_one_df.empty else None
        reranked_for_routine = reranked[
            ~reranked["query_category"].astype(str).str.lower().str.replace(" ", "", regex=False).isin(["all-in-one", "allinone"])
        ].copy()
        beam_top_n = 8 if top_n is None else max(int(top_n), 2)
        best_candidates = build_routines(
            reranked_for_routine,
            ctx["gender"],
            session_id,
            top_n=beam_top_n,
            beam_width=500,
            total_budget_min=total_budget_min,
            total_budget_max=effective_total_budget_max,
            slot_budget_min_map=slot_budget_min_map,
            slot_budget_max_map=effective_slot_budget_max_map,
        )
        value_candidates = _build_value_with_expand(
            reranked_for_routine,
            ctx["gender"],
            session_id,
            top_n=20,
            start_beam=500,
            step=100,
            max_beam=1500,
            total_budget_min=total_budget_min,
            total_budget_max=effective_total_budget_max,
            slot_budget_min_map=slot_budget_min_map,
            slot_budget_max_map=effective_slot_budget_max_map,
        )
        routines = []
        if best_candidates:
            b = dict(best_candidates[0])
            b["routine_label"] = "Best Routine"
            routines.append(b)
        if value_candidates:
            best_key = str(routines[0].get("products")) if routines else ""
            pick_v = None
            for cand in value_candidates:
                if str(cand.get("products")) != best_key:
                    pick_v = cand
                    break
            if pick_v is None and len(value_candidates) > 0:
                pick_v = value_candidates[0]
            if pick_v is not None:
                v = dict(pick_v)
                v["routine_label"] = "Value Routine"
                # if exact duplicate, try best_candidates second item as fallback
                if routines and str(v.get("products")) == str(routines[0].get("products")) and len(best_candidates) > 1:
                    v = dict(best_candidates[1])
                    v["routine_label"] = "Value Routine"
                routines.append(v)
    reranked.to_csv(OUTPUT_DIR / f"reranked_user_{user_id}.csv", index=False)

    price_map = _build_price_map(reranked)

    # Case 3: budget provided but no routine -> fallback to no-budget pipeline result
    fallback_applied = False
    has_budget_input = any([
        total_budget is not None,
        total_budget_min is not None,
        total_budget_max is not None,
        bool(slot_budget_map),
        bool(slot_budget_min_map),
        bool(slot_budget_max_map),
    ])

    if has_budget_input and (filtered.empty or len(routines) == 0):
        filtered_fb, _ = hard_filter(
            candidates,
            session_id,
            gender=ctx["gender"],
            total_budget=None,
            slot_budget_map=None,
            total_budget_min=None,
            total_budget_max=None,
            slot_budget_min_map=None,
            slot_budget_max_map=None,
        )

        if not filtered_fb.empty:
            scored_fb = []
            for _, row in filtered_fb.iterrows():
                scores = soft_score(row["product_key"], session_id, float(row["score"]), user_id=user_id)
                scored_fb.append({**row.to_dict(), **scores})

            reranked_fb = pd.DataFrame(scored_fb).sort_values("S_rerank", ascending=False)
            all_in_one_fb = reranked_fb[
                reranked_fb["query_category"].astype(str).str.lower().str.replace(" ", "", regex=False).isin(["all-in-one", "allinone"])
            ].copy()
            if not all_in_one_fb.empty:
                all_in_one_pick = all_in_one_fb.iloc[0].to_dict()
            reranked_for_routine_fb = reranked_fb[
                ~reranked_fb["query_category"].astype(str).str.lower().str.replace(" ", "", regex=False).isin(["all-in-one", "allinone"])
            ].copy()
            beam_top_n = 8 if top_n is None else max(int(top_n), 2)
            best_fb = build_routines(
                reranked_for_routine_fb,
                ctx["gender"],
                session_id,
                top_n=beam_top_n,
                beam_width=500,
            )
            value_fb = _build_value_with_expand(
                reranked_for_routine_fb,
                ctx["gender"],
                session_id,
                top_n=20,
                start_beam=500,
                step=100,
                max_beam=1500,
            )
            routines_fb = []
            if best_fb:
                b = dict(best_fb[0]); b["routine_label"] = "Best Routine"; routines_fb.append(b)
            if value_fb:
                best_key = str(routines_fb[0].get("products")) if routines_fb else ""
                pick_v = None
                for cand in value_fb:
                    if str(cand.get("products")) != best_key:
                        pick_v = cand
                        break
                if pick_v is None and len(value_fb) > 0:
                    pick_v = value_fb[0]
                if pick_v is not None:
                    v = dict(pick_v); v["routine_label"] = "Value Routine"
                    if routines_fb and str(v.get("products")) == str(routines_fb[0].get("products")) and len(best_fb) > 1:
                        v = dict(best_fb[1]); v["routine_label"] = "Value Routine"
                    routines_fb.append(v)

            if routines_fb:
                routines = routines_fb
                reranked = reranked_fb
                price_map = _build_price_map(reranked)
                fallback_applied = True
                print("[fallback] 예산 조건에 맞는 추천이 없어, 예산 제한 없이 가장 유사한 루틴을 제공합니다.")

    # Best/Value are built by separate engines above
    if ctx.get("gender") == "male":
        routines = _attach_all_in_one_to_routines(routines, all_in_one_pick)
    if len(routines) >= 2 and str(routines[0].get("products")) == str(routines[1].get("products")):
        routines = routines[:1]
    if has_budget_input and routines and not fallback_applied:
        before_budget_check = len(routines)
        routines = [
            r
            for r in routines
            if _routine_budget_ok(
                r,
                price_map=price_map,
                total_budget_min=total_budget_min,
                total_budget_max=effective_total_budget_max,
                slot_budget_min_map=slot_budget_min_map,
                slot_budget_max_map=effective_slot_budget_max_map,
            )
        ]
        dropped_budget_routines = before_budget_check - len(routines)
        if dropped_budget_routines > 0:
            print(f"[budget] dropped {dropped_budget_routines} routine(s) exceeding strict budget")
    failure_reason = None
    session_status = "SUCCESS"
    if len(routines) == 0:
        if fallback_applied:
            # defensive: fallback applied but still no routine
            failure_reason = "NO_ROUTINE_AFTER_FALLBACK"
        elif filtered.empty:
            if not drop_log.empty and (drop_log["drop_reason"] == "HF5_CORE_SLOT_EMPTY").any():
                failure_reason = "HF5_CORE_SLOT_EMPTY"
            else:
                failure_reason = "NO_CANDIDATE_AFTER_HARD_FILTER"
        else:
            failure_reason = "NO_ROUTINE_AFTER_BUILD"
        session_status = "FAILED"
    budget_check_passed = session_status == "SUCCESS" and not fallback_applied
    rerank_changed = len(drop_log) > 0
    rec_session_id = None
    save_error = None
    for attempt in range(1, 4):
        try:
            rec_session_id = _insert_recommendation_results(
                user_id=user_id,
                session_meta=ctx,
                candidates=candidates,
                routines=routines,
                reranked=reranked,
                total_budget_min=total_budget_min,
                total_budget_max=total_budget_max,
                slot_budget_min_map=slot_budget_min_map,
                slot_budget_max_map=effective_slot_budget_max_map,
                session_status=session_status,
                failure_reason=failure_reason,
                budget_check_passed=budget_check_passed,
            )
            print(f"Saved recommendation session_id={rec_session_id}")
            break
        except pymysql.err.OperationalError as e:
            save_error = e
            if e.args and e.args[0] in (1205, 1213) and attempt < 3:
                wait_sec = attempt
                print(f"[warn][save] transient DB lock error ({e.args[0]}), retry {attempt}/3 after {wait_sec}s")
                time.sleep(wait_sec)
                continue
            print(f"[warn][save] DB save failed: {e}")
            break
        except pymysql.MySQLError as e:
            save_error = e
            print(f"[warn][save] DB save failed: {e}")
            break

    if session_status != "SUCCESS":
        if save_error is not None:
            print(f"Recommendation failed: {failure_reason} (session save skipped)")
        else:
            print(f"Recommendation failed: {failure_reason}")
        return []
    name_kor_map = {}
    brand_kor_map = {}
    if not reranked.empty:
        for _, rr in reranked.iterrows():
            pk = str(rr.get("product_key") or "").strip().lower()
            nk = rr.get("product_name_kor")
            bk = rr.get("brand_name_kor")
            if pk and nk is not None and str(nk).strip() != "":
                name_kor_map[pk] = str(nk)
            if pk and bk is not None and str(bk).strip() != "":
                brand_kor_map[pk] = str(bk)

    if ctx.get("gender") == "male" and all_in_one_pick is not None:
        ai_brand = all_in_one_pick.get("Brand")
        ai_pk = str(all_in_one_pick.get("product_key") or "").strip().lower()
        ai_name = name_kor_map.get(ai_pk, all_in_one_pick.get("product_name"))
        ai_brand = brand_kor_map.get(ai_pk, all_in_one_pick.get("Brand"))
        ai_score = all_in_one_pick.get("S_rerank")
        ai_price = all_in_one_pick.get("price")
        ai_price_txt = _format_price_text(ai_price)
        print("\n[All-In-One Standalone]")
        print(f"  All-In-One           -> {ai_brand} - {ai_name} | S={ai_score} | price={ai_price_txt} (included in routines)")
    for i, r in enumerate(routines, 1):
        total_price = 0.0
        price_missing = False
        for p in r["products"]:
            k = _norm_brand_name_key(p.get("brand"), p.get("name"))
            pv = price_map.get(k)
            if pv is None or str(pv) == "nan":
                price_missing = True
            else:
                total_price += float(pv)

        total_price_txt = "N/A" if price_missing else f"{int(total_price):,}원"
        if price_missing:
            budget_state = "N/A"
        elif total_budget_min is not None and total_price < float(total_budget_min):
            budget_state = "UNDER_MIN"
        elif total_budget_max is not None and total_price > float(total_budget_max):
            budget_state = "OVER_MAX"
        elif total_budget_min is not None or total_budget_max is not None:
            budget_state = "WITHIN_RANGE"
        else:
            budget_state = "NO_LIMIT"

        label = r.get("routine_label", f"Routine {i}")
        print(f"\n[{label}] score={r['total_score']} ({r['am_pm_label']}) | total_price={total_price_txt} | budget={budget_state}")
        for p in r["products"]:
            k = _norm_brand_name_key(p.get("brand"), p.get("name"))
            price = price_map.get(k)
            price_txt = _format_price_text(price)
            pk = str(p.get("product_key") or "").strip().lower()
            disp_name = name_kor_map.get(pk, p.get("name"))
            disp_brand = brand_kor_map.get(pk, p.get("brand"))
            print(f"  {p['category']:20s} -> {disp_brand} - {disp_name} | S={p['S_rerank']} | price={price_txt}")
        _print_conflict_details(r)
        _print_am_pm_details(r)
    return routines

if __name__ == "__main__":
    # First setup only: run `python -u -B -m ...graph.load_graph --mode static`
    # Dynamic user-session graph data is created per recommendation request.
    run_pipeline(user_id=1, image_id=None, top_n=3)
