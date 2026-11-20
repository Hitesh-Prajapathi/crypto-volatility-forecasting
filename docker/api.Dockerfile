# syntax=docker/dockerfile:1.7
# volforecast API image (FastAPI + uvicorn). Build from the repo root:
#   docker build -f docker/api.Dockerfile -t volforecast-api:dev .

# ---- uv binary (pinned) ------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.13.8 AS uv

# ---- builder: resolve + install into a relocatable venv ----------------------
FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /uvx /bin/
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app

# 1) dependencies only (cached until pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra serve

# 2) the project itself, non-editable, so /app is not needed at runtime
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra serve \
    # xgboost pulls ~200 MB of CUDA/NCCL wheels on linux that CPU inference never loads
    && rm -rf /opt/venv/lib/python3.12/site-packages/nvidia \
    && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} +

# ---- runtime ------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ARG VERSION=0.1.0
ARG REVISION=unknown
ARG SOURCE=https://github.com/Hitesh-Prajapathi/crypto-volatility-forecasting
LABEL org.opencontainers.image.title="volforecast-api" \
      org.opencontainers.image.description="Realized-volatility forecast API" \
      org.opencontainers.image.source="${SOURCE}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="MIT"

# libgomp1: OpenMP runtime required by lightgbm / xgboost wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown 10001:10001 /data

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    VOLF_DATA_DIR=/data \
    WEB_CONCURRENCY=2

USER 10001:10001
WORKDIR /tmp
EXPOSE 8000

# Read-only root fs: the process only writes to /tmp and the /data volume.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"]

# Exec form: the Python entrypoint is PID 1 and receives SIGTERM, so uvicorn drains in-flight
# requests and runs the lifespan shutdown. WEB_CONCURRENCY sets the worker count; with more than
# one worker the entrypoint enables Prometheus multiprocess mode in /tmp/volforecast-prom.
CMD ["python", "-m", "volforecast.serving"]
