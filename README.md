# ROUPLE

AI-based personalized skincare routine recommendation service.

ROUPLE analyzes a user's facial image, predicts skin condition scores, recommends skincare routines with embedding and knowledge graph reasoning, and evaluates the fit of products already owned by the user.

## Overview

ROUPLE connects three core experiences into one skincare workflow.

| Feature | Description |
| --- | --- |
| Skin Report | Analyzes a face image and predicts six skin indicators. |
| Routine Report | Recommends best/value skincare routines based on skin scores, profile, budget, and ingredient rules. |
| Vanity Report | Evaluates owned cosmetics with Skin Match and recommends a routine using fixed owned products. |

## Key Features

- Face image upload and skin analysis
- Six skin indicators: acne, dryness, sagging, pore, pigmentation, wrinkle
- BGE-M3 embedding-based candidate retrieval
- Neo4j knowledge graph re-ranking
- Rule-based ingredient conflict filtering
- Beam Search routine generation
- Owned product Skin Match scoring
- LLM-generated explanations for skin reports, routines, and vanity results
- JSON cache for LLM outputs to reduce repeated calls
- Docker-based deployment on AWS EC2

## System Architecture

```text
User Image
   -> S3 Upload
   -> FastAPI Backend
   -> Skin Analysis Model
   -> Six Skin Scores
   -> Embedding Candidate Retrieval
   -> Knowledge Graph Re-ranking
   -> Rule Filtering + Beam Search
   -> Routine Recommendation
   -> LLM Explanation
```

```text
Owned Products + Six Skin Scores
   -> Skin Match Scoring
   -> Fixed Product Selection
   -> Missing Slot Detection
   -> Embedding + KG Recommendation
   -> Vanity Routine
   -> LLM Explanation
```

## Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | React, Vite, TypeScript |
| Backend | FastAPI, SQLAlchemy |
| Database | MySQL |
| Knowledge Graph | Neo4j |
| Image Storage | AWS S3 |
| Deployment | Docker, Docker Compose, AWS EC2 |
| Skin Analysis | Swin-Tiny Transformer based multitask model |
| Recommendation | BGE-M3 embedding, Knowledge Graph, Beam Search |
| LLM | Report and recommendation explanation generation |

## Repository Structure

```text
.
├── backend/                 # FastAPI backend
├── frontend/                # React frontend
├── model/                   # Skin analysis, recommendation, vanity, LLM modules
├── DB/                      # Local database loading resources
├── scripts/                 # Utility scripts
├── docker-compose.yml       # Deployment compose file
└── DEPLOY_DOCKER.md         # Docker deployment guide
```

## Local Development

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

## Docker Deployment

Build and push images from the project root.

```powershell
docker build -f backend/Dockerfile -t myhyun1123/rouple-backend:latest .
docker push myhyun1123/rouple-backend:latest

docker build -f frontend/Dockerfile -t myhyun1123/rouple-frontend:latest .
docker push myhyun1123/rouple-frontend:latest
```

On EC2:

```bash
cd ~/rouple-deploy
docker compose pull
docker compose up -d
docker compose ps
```

## Environment Variables

The backend requires environment variables for database, Neo4j, S3, model paths, and LLM API settings.

Example categories:

```text
MYSQL_HOST
MYSQL_PORT
MYSQL_USER
MYSQL_PASSWORD
MYSQL_DB

NEO4J_URI
NEO4J_USER
NEO4J_PASSWORD

S3_BUCKET
S3_REGION
S3_PREFIX

SKIN_MODEL_CHECKPOINT
EMBED_LOCAL_FILES_ONLY
ROUPLE_CACHE_DIR
```

Do not commit real `.env`, API keys, database dumps, model weights, or private key files.

## LLM Cache

LLM outputs are persisted as JSON files so that repeated page visits do not trigger unnecessary LLM calls.

Cached outputs include:

- Skin analysis summaries
- Routine recommendation explanations
- Vanity skin match explanations
- Vanity routine explanations

## Notes

- `schema.sql` is treated as a local-only file and ignored by Git.
- Large seed dumps such as `rouple_seed.sql` should not be committed.
- AWS credentials and PEM key files must remain local only.

