# run_embedding.py
from __future__ import annotations

from pathlib import Path

from .config import AUTO_UPLOAD_RECOMMENDATION_CANDIDATE, DEFAULT_OUTPUT_DIR, MODEL_NAME, TOPK_PER_CATEGORY
from .corpus_builder import build_product_doc_by_style, build_query_text
from .data_loader import load_corpus_from_db, load_skin_queries_from_db
from .db_uploader import upload_recommendation_candidates
from .retriever import run_retrieval


def build_load_output(df):
    return df.rename(
        columns={
            "Brand": "brand",
            "Category": "category",
            "Function": "function",
        }
    )[
        [
            "image_id",
            "user_id",
            "rank_in_category",
            "product_id",
            "query_category",
            "brand",
            "product_name",
            "category",
            "function",
            "score",
        ]
    ]


def main():
    print(f"[config] model={MODEL_NAME}, topk={TOPK_PER_CATEGORY}")

    emb_path = DEFAULT_OUTPUT_DIR / f"cosmetic_emb_{MODEL_NAME.replace('/', '_')}.npy"

    print("[1/4] Load corpus...")
    corpus_df = load_corpus_from_db()
    print(f"products: {len(corpus_df)}")

    print("[2/4] Load skin data...")
    skin_df = load_skin_queries_from_db()
    print(f"images: {len(skin_df)}")

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # TEMP(one-time check): save cosmetic doc_text corpus, then remove this block.
    cosmetic_snapshot = corpus_df.copy()
    cosmetic_snapshot["doc_text"] = cosmetic_snapshot.apply(build_product_doc_by_style, axis=1)
    cosmetic_path = DEFAULT_OUTPUT_DIR / "cosmetic_doc_text_corpus_top.csv"
    cosmetic_snapshot.to_csv(cosmetic_path, index=False, encoding="utf-8-sig")
    print(f"Saved temp cosmetic corpus: {cosmetic_path}")

    # TEMP(one-time check): save skin query_text corpus, then remove this block.
    skin_snapshot = skin_df.copy()
    skin_snapshot["query_text"] = skin_snapshot.apply(build_query_text, axis=1)
    query_path = DEFAULT_OUTPUT_DIR / "skin_query_text_corpus.csv"
    skin_snapshot.to_csv(query_path, index=False, encoding="utf-8-sig")
    print(f"Saved temp skin query corpus: {query_path}")
    # ---------------------------------------------------------------------
    
    print("[3/4] Run retrieval...")
    result_df = run_retrieval(
        corpus_df=corpus_df,
        skin_df=skin_df,
        topk_per_category=TOPK_PER_CATEGORY,
        model_name=MODEL_NAME,
        emb_path=emb_path,
    )

    if result_df.empty:
        raise ValueError("No results generated.")

    print("[4/4] Save results...")

    test_path = DEFAULT_OUTPUT_DIR / "results_detail.csv"      # (세부사항) analysis/detail output
    load_path = DEFAULT_OUTPUT_DIR / "results_candidate.csv"   # (최종) DB load candidate output

    result_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    print(f"Saved test: {test_path}")

    load_df = build_load_output(result_df)
    load_df.to_csv(load_path, index=False, encoding="utf-8-sig")
    print(f"Saved load: {load_path}")

    if AUTO_UPLOAD_RECOMMENDATION_CANDIDATE:
        print("[5/5] Upload results to MySQL RECOMMENDATION_CANDIDATE...")
        upload_recommendation_candidates(load_df)
    else:
        print("[5/5] Skip MySQL upload (EMBED_AUTO_UPLOAD_CANDIDATE=0)")

    print(f"Total rows: {len(result_df)}")


if __name__ == "__main__":
    main()
