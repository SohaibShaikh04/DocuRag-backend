FROM python:3.11-slim

WORKDIR /app

# ── System deps for PyMuPDF ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download embedding model so cold starts are fast ─────────────────────
# The model (~80 MB) is baked into the image, not fetched at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

EXPOSE 8000

# Use $PORT if the platform injects it (Railway does), fallback to 8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
