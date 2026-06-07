FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBED_LOCAL_FILES_ONLY=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt

# Install CPU PyTorch first to avoid pulling large CUDA wheels.
RUN python -m pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r /app/backend/requirements.txt

COPY backend/app /app/backend/app
COPY model/__init__.py /app/model/__init__.py
COPY model/skin_analysis /app/model/skin_analysis
COPY model/llm /app/model/llm
COPY model/recommendation/embedding_pipeline /app/model/recommendation/embedding_pipeline
COPY model/recommendation/kg_pipeline /app/model/recommendation/kg_pipeline
COPY model/vanity /app/model/vanity

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
