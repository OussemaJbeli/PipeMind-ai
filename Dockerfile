FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libpq5 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY pyproject.toml README.md ./
COPY app ./app

# Base image only. The `ml` extra pulls torch + transformers (~2 GB) and is
# added in roadmaps/09, when embeddings actually arrive — installing it now
# would quadruple the image for code that does not exist yet.
RUN pip install --no-cache-dir .

COPY scripts ./scripts

EXPOSE 8001

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
