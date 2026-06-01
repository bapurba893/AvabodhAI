# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim — small size, production ready
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="Apurba"
LABEL description="Avabodh API — Document Intelligence Platform"
LABEL version="1.0.0"

# ── Environment variables ─────────────────────────────────────────────────────
# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install system dependencies ───────────────────────────────────────────────
# These are needed for psycopg2, pdfminer, and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first (Docker cache optimization) ───────────────────────
# If requirements.txt doesn't change, Docker reuses cached layer
# This makes rebuilds much faster
COPY requirements.txt .

# ── Install Python dependencies ───────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ── Create necessary directories ──────────────────────────────────────────────
RUN mkdir -p uploaded_files documents

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
# Docker checks this every 30s — marks container unhealthy if it fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/')" || exit 1

# ── Start command ─────────────────────────────────────────────────────────────
# 0.0.0.0 makes it accessible from outside the container
# workers=2 handles multiple requests simultaneously
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]