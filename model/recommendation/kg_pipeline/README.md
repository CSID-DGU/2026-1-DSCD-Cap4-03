# kg_pipeline guide

This folder is the post-retrieval recommendation stage (KG rerank + routine build).

## Active paths
- `neo4j_skincare/pipeline.py`: main KG pipeline
- `neo4j_skincare/config.py`: Neo4j/MySQL and weight settings
- `neo4j_skincare/graph/`: graph load and Cypher schema
- `neo4j_skincare/rerank/`: hard and soft rerank logic
- `neo4j_skincare/routine/`: routine composition and conflict checks
- `neo4j_skincare/tools/test_connection.py`: Neo4j connection check script

## Current structure policy
- Run and edit only from `pipeline.py`
- Keep outputs under `kg_pipeline/output/`
- Ignore cache/output artifacts via `.gitignore`
