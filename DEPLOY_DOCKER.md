# ROUPLE Docker Deployment

This deployment uses lightweight Docker images.

- Backend and frontend are built as Docker images.
- MySQL is expected to run on AWS RDS.
- Neo4j runs as a Docker Compose service on the EC2 host.
- S3 and the DGU LLM API are external services.
- Large model artifacts are not copied into the backend image. They are mounted from the EC2 host.

## 1. EC2 Artifact Layout

Create the artifact directories on EC2:

```bash
sudo mkdir -p /opt/rouple/model_artifacts/skin_analysis
sudo mkdir -p /opt/rouple/model_artifacts/embedding_results
sudo mkdir -p /opt/rouple/huggingface_cache
sudo chown -R $USER:$USER /opt/rouple
```

Place files as follows:

```text
/opt/rouple/model_artifacts/skin_analysis/best_260507_21.pt
/opt/rouple/model_artifacts/embedding_results/cosmetic_emb_BAAI_bge-m3.npy
/opt/rouple/model_artifacts/embedding_results/cosmetic_doc_text_corpus_top.csv
/opt/rouple/huggingface_cache/
```

The backend container mounts these paths into:

```text
/app/model/skin_analysis/best_260507_21.pt
/app/model/recommendation/embedding_pipeline/embedding_results
/root/.cache/huggingface
```

## 2. Backend Environment

Create `backend/.env` on EC2. Do not commit this file.

```env
USE_LOCAL_SQLITE=false

MYSQL_HOST=your-rds-endpoint
MYSQL_PORT=3306
MYSQL_USER=your-user
MYSQL_PASSWORD=your-password
MYSQL_DB=Rouple_db

NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=cap4cap4

S3_BUCKET=your-bucket
S3_REGION=ap-northeast-2
S3_PREFIX=users
S3_PRESIGN_EXPIRE_SECONDS=900

AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

DGU_LLM_API_KEY=your-key
DGU_LLM_BASE_URL=https://factchat-cloud.mindlogic.ai/v1/gateway
DGU_LLM_MODEL=gpt-5.4-nano
DGU_LLM_TEMPERATURE=0.6
DGU_LLM_MAX_TOKENS=5000
DGU_LLM_PRINT_USAGE=true

FRONTEND_ORIGINS=http://your-ec2-public-ip,http://your-domain
```

## 3. Build Images

```bash
docker compose build
```

## 4. Run

```bash
docker compose up -d
```

Check logs:

```bash
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f neo4j
```

## 5. Load Neo4j Knowledge Graph

Run this once after MySQL cosmetic data is loaded and the Neo4j container is healthy:

```bash
docker compose exec backend python -u -B -m model.recommendation.kg_pipeline.neo4j_skincare.graph.load_graph --mode static
```

You can check Neo4j from the EC2 host:

```bash
docker compose exec backend python -u -B -m model.recommendation.kg_pipeline.neo4j_skincare.tools.test_connection
```

Neo4j browser is available only if port `7474` is opened in the EC2 security group:

```text
http://EC2_PUBLIC_IP:7474
```

For normal service use, only the backend needs internal access to `bolt://neo4j:7687`.

## 6. Test

```text
Frontend: http://EC2_PUBLIC_IP
Backend docs: http://EC2_PUBLIC_IP:8000/docs
Neo4j browser: http://EC2_PUBLIC_IP:7474
```

Recommended test order:

```text
1. Sign up / login
2. S3 image upload
3. Skin analysis
4. Skin analysis summary
5. Routine recommendation
6. Recommendation explanation
7. Vanity product add
8. Skin match
9. Vanity routine
```
