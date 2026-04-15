from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
from sentence_transformers import SentenceTransformer


# -----------------------------
# Config
# -----------------------------
MYSQL_HOST = os.getenv("ROUPLE_MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("ROUPLE_MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("ROUPLE_MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("ROUPLE_MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("ROUPLE_MYSQL_DB", "Rouple_db")

MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
TOPK_PER_CATEGORY = int(os.getenv("EMBED_TOPK_PER_CATEGORY", "15"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "4_DB" / "RECOMMENDATION_EMBEDDING_TOP.csv"

METRICS = ["dryness", "pore", "wrinkle", "pigmentation", "sagging", "acne"]
METRIC_TO_FUNCTIONS = {
    "dryness": ["Hydration", "Moisturizing", "Soothing"],
    "pore": ["Pores", "Exfoliation"],
    "wrinkle": ["Anti-Aging"],
    "pigmentation": ["Brightening"],
    "sagging": ["Anti-Aging", "Firming"],
    "acne": ["Blemishes", "Soothing", "Exfoliation", "Pores"],
}


def _connect():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def _normalize(v: str | None) -> str:
    if v is None:
        return ""
    return " ".join(str(v).strip().split())


def load_corpus_from_db() -> pd.DataFrame:
    query = """
    SELECT
        p.product_id,
        p.hwahae_brand AS Brand,
        p.hwahae_product_name AS product_name,
        p.category AS Category,
        p.`function` AS `Function`,
        p.price_en AS price,
        pr.pros_text AS pros,
        pr.cons_text AS cons,
        GROUP_CONCAT(DISTINCT i.ingredient_name ORDER BY i.ingredient_name SEPARATOR ', ') AS ingredients
    FROM PRODUCT p
    LEFT JOIN PRODUCT_REVIEW pr ON pr.product_id = p.product_id
    LEFT JOIN PRODUCT_INGREDIENT pi ON pi.product_id = p.product_id
    LEFT JOIN INGREDIENT i ON i.ingredient_id = pi.ingredient_id
    WHERE p.hwahae_brand IS NOT NULL
      AND p.hwahae_product_name IS NOT NULL
      AND p.category IS NOT NULL
    GROUP BY
        p.product_id, p.hwahae_brand, p.hwahae_product_name, p.category, p.`function`,
        p.price_en, pr.pros_text, pr.cons_text
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("PRODUCT corpus is empty in DB.")

    for c in ["Brand", "product_name", "Category", "Function", "pros", "cons", "ingredients"]:
        if c in df.columns:
            df[c] = df[c].map(_normalize)
    return df


def load_skin_queries_from_db() -> pd.DataFrame:
    query = """
    SELECT
        ui.image_id,
        ui.storage_url,
        sar.dryness_score,
        sar.pore_score,
        sar.wrinkle_score,
        sar.pigmentation_score,
        sar.sagging_score,
        sar.acne_score
    FROM USER_IMAGE ui
    JOIN (
        SELECT image_id, MAX(result_id) AS result_id
        FROM SKIN_ANALYSIS_RESULT
        GROUP BY image_id
    ) latest ON latest.image_id = ui.image_id
    JOIN SKIN_ANALYSIS_RESULT sar ON sar.result_id = latest.result_id
    ORDER BY ui.image_id
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("SKIN_ANALYSIS_RESULT join USER_IMAGE is empty in DB.")
    return df


def build_query_text(row: pd.Series) -> str:
    severities = {}
    for m in METRICS:
        raw = row.get(f"{m}_score", 0.0)
        val = 0.0 if raw is None else float(raw)
        val = val / 100.0 if val > 1.0 else val
        severities[m] = val

    ordered = sorted(severities.items(), key=lambda x: x[1], reverse=True)
    parts = []
    for metric, s in ordered:
        funcs = ", ".join(METRIC_TO_FUNCTIONS.get(metric, []))
        parts.append(f"{metric}({s:.4f}): {funcs}")
    return "user skin need profile (severity -> auxiliary functions, non-binding): " + "; ".join(parts) + "."


def build_product_doc(row: pd.Series) -> str:
    return " | ".join(
        [
            f"brand={row.get('Brand','')}",
            f"name={row.get('product_name','')}",
            f"category={row.get('Category','')}",
            f"function={row.get('Function','')}",
            f"ingredients={row.get('ingredients','')}",
            f"pros={row.get('pros','')}",
            f"cons={row.get('cons','')}",
        ]
    )


def run_retrieval(
    corpus_df: pd.DataFrame,
    skin_df: pd.DataFrame,
    topk_per_category: int,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    model = SentenceTransformer(model_name)

    corpus_df = corpus_df.copy()
    corpus_df["doc_text"] = corpus_df.apply(build_product_doc, axis=1)

    doc_vec = model.encode(corpus_df["doc_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    if not isinstance(doc_vec, np.ndarray):
        doc_vec = np.array(doc_vec)

    # category -> row indices in corpus
    cat_to_idx = {}
    for idx, cat in enumerate(corpus_df["Category"].tolist()):
        cat_to_idx.setdefault(cat, []).append(idx)

    results = []
    for _, srow in skin_df.iterrows():
        query_text = build_query_text(srow)
        qv = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
        if not isinstance(qv, np.ndarray):
            qv = np.array(qv)
        qv = qv[0]

        image_id = int(srow["image_id"])
        image_name = Path(str(srow.get("storage_url", ""))).name

        for cat, idxs in cat_to_idx.items():
            if not idxs:
                continue
            mat = doc_vec[idxs]
            scores = mat @ qv
            order = np.argsort(-scores)[:topk_per_category]

            for local_rank in order:
                ridx = idxs[int(local_rank)]
                prow = corpus_df.iloc[ridx]
                score = float(scores[int(local_rank)])
                results.append(
                    {
                        "image_id": image_id,
                        "image": image_name,
                        "query_category": prow["Category"],
                        "product_id": prow.get("product_id"),
                        "Brand": prow["Brand"],
                        "product_name": prow["product_name"],
                        "Category": prow["Category"],
                        "Function": prow["Function"],
                        "score": score,
                        "price": prow.get("price"),
                        "pros": prow.get("pros"),
                        "cons": prow.get("cons"),
                        "query_text": query_text,
                    }
                )

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values(["image_id", "query_category", "score"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def main(
    output_csv: Path = DEFAULT_OUTPUT,
    model_name: str = MODEL_NAME,
    topk_per_category: int = TOPK_PER_CATEGORY,
):
    print(f"[config] model={model_name}, topk_per_category={topk_per_category}")
    print("[1/4] Load corpus from DB...")
    corpus_df = load_corpus_from_db()
    print(f"  products: {len(corpus_df)}")

    print("[2/4] Load skin query rows from DB...")
    skin_df = load_skin_queries_from_db()
    print(f"  images: {len(skin_df)}")

    print("[3/4] Embed + retrieve...")
    out = run_retrieval(
        corpus_df=corpus_df,
        skin_df=skin_df,
        topk_per_category=topk_per_category,
        model_name=model_name,
    )
    if out.empty:
        raise ValueError("No retrieval rows produced.")

    print("[4/4] Save CSV...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_csv}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed skin-query/product corpus and export top-k retrieval results.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument(
        "--topk-per-category",
        type=int,
        default=TOPK_PER_CATEGORY,
        help="Number of products per category for each image query",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path for retrieval results",
    )
    args = parser.parse_args()
    main(
        output_csv=Path(args.output_csv),
        model_name=args.model_name,
        topk_per_category=args.topk_per_category,
    )
