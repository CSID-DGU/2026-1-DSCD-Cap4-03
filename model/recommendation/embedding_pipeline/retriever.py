# retriever.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch


from .config import (
    BATCH_SIZE,
    FUNCTION_WEIGHT,
    MAX_PER_BRAND_IN_CATEGORY,
    MODEL_NAME,
    SEMANTIC_WEIGHT,
)
from .corpus_builder import build_product_doc_by_style, build_query_profile, build_query_text


def load_or_create_embeddings(
    model,
    corpus_df,
    batch_size,
    emb_path: Path,
):
    if emb_path.exists():
        print(f"Load embeddings from: {emb_path}")
        return np.load(emb_path)

    print("Creating embeddings...")
    corpus_df["doc_text"] = corpus_df.apply(
        lambda r: build_product_doc_by_style(r), axis=1
    )

    doc_vec = model.encode(
        corpus_df["doc_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )

    doc_vec = np.array(doc_vec)
    np.save(emb_path, doc_vec)

    print(f"Saved embeddings to: {emb_path}")
    return doc_vec


def _parse_function_set(raw) -> set[str]:
    if raw is None:
        return set()
    text = str(raw)
    for sep in ["/", "|", ";"]:
        text = text.replace(sep, ",")
    return {x.strip() for x in text.split(",") if x.strip()}


def _function_match_score(product_functions: set[str], function_weights: dict[str, float]) -> float:
    if not function_weights:
        return 0.0
    total = sum(function_weights.values())
    if total <= 0:
        return 0.0
    hit = sum(w for f, w in function_weights.items() if f in product_functions)
    return float(hit / total)


def run_retrieval(
    corpus_df: pd.DataFrame,
    skin_df: pd.DataFrame,
    topk_per_category: int,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    emb_path: Path | None = None,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[retriever] device={device}")
    model = SentenceTransformer(model_name, device=device)

    corpus_df = corpus_df.copy()

    if emb_path is None:
        emb_path = Path("doc_embeddings.npy")

    doc_vec = load_or_create_embeddings(
        model,
        corpus_df,
        batch_size,
        emb_path,
    )

    cat_to_idx = {}
    for idx, cat in enumerate(corpus_df["Category"].tolist()):
        cat_to_idx.setdefault(cat, []).append(idx)

    product_function_sets = [
        _parse_function_set(v) for v in corpus_df.get("Function", pd.Series([""] * len(corpus_df))).tolist()
    ]

    results = []

    for _, srow in skin_df.iterrows():
        query_text = build_query_text(srow)
        profile = build_query_profile(srow)

        qv = model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        image_id = int(srow["image_id"])
        user_id = int(srow["user_id"]) if "user_id" in srow and pd.notna(srow["user_id"]) else None

        for cat, idxs in cat_to_idx.items():
            mat = doc_vec[idxs]
            semantic_scores = mat @ qv

            mixed_rows = []
            for local_i, semantic in enumerate(semantic_scores):
                ridx = idxs[local_i]
                fscore = _function_match_score(product_function_sets[ridx], profile["function_weights"])
                final_score = SEMANTIC_WEIGHT * float(semantic) + FUNCTION_WEIGHT * float(fscore)
                mixed_rows.append((ridx, float(semantic), float(fscore), float(final_score)))

            mixed_rows.sort(key=lambda x: x[3], reverse=True)

            selected = []
            brand_counts: dict[str, int] = {}
            for ridx, sem, fsc, fin in mixed_rows:
                prow = corpus_df.iloc[ridx]
                brand = str(prow.get("Brand", "")).strip()
                if brand_counts.get(brand, 0) >= MAX_PER_BRAND_IN_CATEGORY:
                    continue
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
                selected.append((ridx, sem, fsc, fin))
                if len(selected) >= topk_per_category:
                    break

            for rank, (ridx, sem, fsc, fin) in enumerate(selected, start=1):
                prow = corpus_df.iloc[ridx]
                results.append(
                    {
                        "image_id": image_id,
                        "user_id": user_id,
                        "query_category": cat,
                        "product_id": prow["product_id"],
                        "Category": prow["Category"],
                        "Brand": prow["Brand"],
                        "Function": prow.get("Function"),
                        "product_name": prow["product_name"],
                        "semantic_score": sem,
                        "function_match_score": fsc,
                        "score": fin,
                        "rank_in_category": rank,
                        "dominant_metric": profile["dominant_metric"],
                    }
                )

    return pd.DataFrame(results)
