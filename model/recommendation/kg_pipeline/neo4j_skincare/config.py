from pathlib import Path
from neo4j import GraphDatabase
import os

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", os.getenv("NEO4J_PASS", "cap4cap4"))

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

KG_ROOT = Path(__file__).resolve().parent
KG_PIPELINE_ROOT = KG_ROOT.parent
RECOMMENDATION_ROOT = KG_PIPELINE_ROOT.parent
MODEL_ROOT = RECOMMENDATION_ROOT.parent
PROJECT_ROOT = MODEL_ROOT.parent

# Data directories (local paths)
EMBEDDING_RESULT_DIR = RECOMMENDATION_ROOT / "embedding_pipeline" / "embedding_results"
CRAWLING_DATA_DIR = PROJECT_ROOT / "cosmetic_crawling" / "data"
OUTPUT_DIR = KG_PIPELINE_ROOT / "output"

# 성별에 따른 스킨케어 루틴 SLOT
SLOT_ORDER = {
    "female": [
        ("optional", ["Face Mists"]),
        ("core", ["Toner", "Toner Pads"]),
        ("core", ["Emulsions"]),
        ("core", ["Essences/Ampoules/Serums"]),
        ("core", ["Cream/Gel", "Face Moisturizers"]),
        ("optional", ["Balms/Multi-balms", "Facial Oils", "Eye Treatments"]),
    ],
    "male": [
        ("optional", ["Shaving Products"]),
        ("core", ["Toner", "Toner Pads"]),
        ("core", ["Emulsions"]),
        ("core", ["Essences/Ampoules/Serums"]),
        ("core", ["Cream/Gel", "Face Moisturizers"]),
        ("optional", ["All-in-One"]),
    ],
}

# AM/PM avoid ingredients
AM_AVOID_INGREDIENTS = [
    "Retinol",
    "Retinyl Palmitate",
    "Glycolic Acid",
    "Lactic Acid",
    "Salicylic Acid",
]
PM_AVOID_INGREDIENTS = ["Oxybenzone", "Avobenzone"]


# MySQL source DB (4_DB/schema.sql)
MYSQL_HOST = os.getenv("ROUPLE_MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("ROUPLE_MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("ROUPLE_MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("ROUPLE_MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("ROUPLE_MYSQL_DB", "Rouple_db")

# Rerank weights
RETRIEVAL_TOPK_PER_CATEGORY = int(os.getenv("KG_RETRIEVAL_TOPK_PER_CATEGORY", "20"))
RERANK_VECTOR_WEIGHT = float(os.getenv("KG_RERANK_VECTOR_WEIGHT", "0.50"))
RERANK_CONCERN_WEIGHT = float(os.getenv("KG_RERANK_CONCERN_WEIGHT", "0.22"))
RERANK_SKIN_WEIGHT = float(os.getenv("KG_RERANK_SKIN_WEIGHT", "0.15"))
RERANK_WISHLIST_WEIGHT = float(os.getenv("KG_RERANK_WISHLIST_WEIGHT", "0.05"))
RERANK_REVIEW_WEIGHT = float(os.getenv("KG_RERANK_REVIEW_WEIGHT", "0.08"))
RERANK_IRRITATION_PENALTY_SCALE = float(os.getenv("KG_RERANK_IRRITATION_PENALTY_SCALE", "0.05"))
