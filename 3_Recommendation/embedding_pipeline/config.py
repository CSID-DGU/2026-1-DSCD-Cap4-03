# config.py
from __future__ import annotations

import os
from pathlib import Path

# Database Configuration
MYSQL_HOST = os.getenv("ROUPLE_MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("ROUPLE_MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("ROUPLE_MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("ROUPLE_MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("ROUPLE_MYSQL_DB", "Rouple_db")

# ---------------------------------------------------
# Final Model Configuration
# - Cosmetic corpus text style: B_function_ingredient
# - Skin query mapping style: M2 (primary/secondary)
# - Similarity metric: cosine (via normalized embeddings dot product)
# ---------------------------------------------------
MODEL_NAME = "BAAI/bge-m3"
TOPK_PER_CATEGORY = 20

# --------------------------------------------------
# Hybrid ranking & query limit
# --------------------------------------------------
SEMANTIC_WEIGHT = float(os.getenv("EMBED_SEMANTIC_WEIGHT", "0.75"))
FUNCTION_WEIGHT = float(os.getenv("EMBED_FUNCTION_WEIGHT", "0.25"))
MAX_PER_BRAND_IN_CATEGORY = int(os.getenv("EMBED_MAX_PER_BRAND_IN_CATEGORY", "2"))
MIN_METRIC_SCORE = float(os.getenv("EMBED_MIN_METRIC_SCORE", "0.15"))
MAX_QUERY_METRICS = int(os.getenv("EMBED_MAX_QUERY_METRICS", "4"))

# CPU 실행 시 발열/쓰로틀링을 줄이기 위한 기본 배치 크기
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "3_Recomendation" / "embedding_final" / "embedding_results"

# DB upload
AUTO_UPLOAD_RECOMMENDATION_CANDIDATE = os.getenv("EMBED_AUTO_UPLOAD_CANDIDATE", "1") == "1" # ON

# METRICS: Skin Indicator - Function Mapping
METRICS = ["dryness", "pore", "wrinkle", "pigmentation", "sagging", "acne"]

# Skin Query mapping (M2): primary - secondary
METRIC_TO_FUNCTIONS_M2 = {
    "dryness": {"primary": ["Hydration", "Moisturizing"], "secondary": ["Soothing"]},
    "pore": {"primary": ["Pores"], "secondary": ["Exfoliation"]},
    "wrinkle": {"primary": ["Anti-Aging"], "secondary": ["Soothing"]},
    "pigmentation": {"primary": ["Brightening"], "secondary": ["Soothing"]},
    "sagging": {"primary": ["Anti-Aging"], "secondary": ["Hydration"]},
    "acne": {"primary": ["Blemishes"], "secondary": ["Soothing", "Exfoliation", "Pores"]},
}
