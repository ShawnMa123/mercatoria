# mercatoria test image: corpus and the active SQLite index are baked in.
#
# The index is tied to the embedding model recorded in its metadata
# (my-team-2k/qwen3.7-text-embedding, 1024 dims). Rebuild it when the
# embedding model or chunking config changes:
#   docker run --rm -e MERCATORIA_FORCE_REBUILD=1 ... mercatoria:test
FROM python:3.11-slim AS index

WORKDIR /stage
COPY data/index /stage/index
RUN set -eu; \
    keep="$(cat /stage/index/current.txt)"; \
    mkdir -p /stage/out; \
    cp "/stage/index/$keep" /stage/out/; \
    cp /stage/index/current.txt /stage/out/

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

RUN pip install --no-cache-dir \
    "numpy>=1.26" "fastapi>=0.110" "uvicorn>=0.29" \
    "pymupdf>=1.24" "python-docx>=1.1" "python-multipart>=0.0.9"

COPY mercatoria ./mercatoria
COPY corpus ./corpus
COPY config.example.json ./config.json
COPY --from=index /stage/out ./data/index
COPY docker/entrypoint.sh /usr/local/bin/mercatoria-entrypoint

RUN chmod +x /usr/local/bin/mercatoria-entrypoint

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=30 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"

ENTRYPOINT ["mercatoria-entrypoint"]
