# corpus_builder.py
from __future__ import annotations

import pandas as pd

from .config import (
    MAX_QUERY_METRICS,
    METRICS,
    METRIC_TO_FUNCTIONS_M2,
    MIN_METRIC_SCORE,
)

# Text normalization helper
def normalize_text(v: str | None) -> str:
    if v is None:
        return ""
    return " ".join(str(v).strip().split())


def select_top_ingredients(raw: object, topn: int = 10) -> str:
    text = normalize_text(raw)
    if not text:
        return ""
    for sep in ["/", "|", ";"]:
        text = text.replace(sep, ",")
    items = [x.strip() for x in text.split(",") if x.strip()]
    return ", ".join(items[:topn])

# Skin Query metric-function profile builder
def build_query_profile(row: pd.Series) -> dict:
    scores = {}
    for m in METRICS:
        raw = row.get(f"{m}_score", 0.0)
        val = 0.0 if raw is None else float(raw)
        val = val / 100.0 if val > 1.0 else val
        scores[m] = max(0.0, min(1.0, val))

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [(m, s) for m, s in ordered if s >= MIN_METRIC_SCORE][:MAX_QUERY_METRICS]
    if not selected:
        selected = ordered[:1]
    function_weights: dict[str, float] = {}
    for metric, sev in selected:
        cfg = METRIC_TO_FUNCTIONS_M2.get(metric, {})
        for f in cfg.get("primary", []):
            function_weights[f] = function_weights.get(f, 0.0) + sev * 1.0
        for f in cfg.get("secondary", []):
            function_weights[f] = function_weights.get(f, 0.0) + sev * 0.6

    return {
        "selected_metrics": selected,
        "function_weights": function_weights,
        "dominant_metric": selected[0][0] if selected else None,
    }


# Skin Query Text Builder
def build_query_text(row: pd.Series) -> str:
    profile = build_query_profile(row)
    ordered = profile["selected_metrics"]
    parts: list[str] = []

    for metric, s in ordered:
        cfg = METRIC_TO_FUNCTIONS_M2.get(metric, {})
        primary = ", ".join(cfg.get("primary", []))
        secondary = ", ".join(cfg.get("secondary", []))
        parts.append(
            f"indicator={metric} | score={s:.2f} | primary={primary} | secondary={secondary}"
        )

    return "; ".join(parts)


# Cosmetic Text Builder
def build_cosmetic_text(row: pd.Series) -> str:
    top_ingredients = select_top_ingredients(row.get("ingredients"), topn=10)
    return " | ".join(
        [
            f"brand={row.get('Brand','')}",
            f"name={row.get('product_name','')}",
            f"category={row.get('Category','')}",
            f"function={row.get('Function','')}",
            f"ingredients={top_ingredients}",
        ]
    )


# Backward-compatible alias for retriever.
def build_product_doc_by_style(row: pd.Series) -> str:
    return build_cosmetic_text(row)
