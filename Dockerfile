FROM python:3.11-slim

WORKDIR /app

# System deps for torch/sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models at build time (optional, reduces cold start)
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers
RUN python -c "\
from transformers import pipeline; \
pipeline('text-classification', model='unitary/toxic-bert', device=-1); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
" || true

COPY . .

RUN mkdir -p data/chroma

ENV PYTHONPATH=/app
ENV APP_ENV=production
ENV CHROMA_PERSIST_DIR=/app/data/chroma
ENV POLICY_DATA_PATH=/app/data/policies/policy_chunks.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
