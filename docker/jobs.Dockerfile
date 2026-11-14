# syntax=docker/dockerfile:1.7
# volforecast batch-jobs image: `python -m volforecast.flows.<flow>`.
#   docker build -f docker/jobs.Dockerfile -t volforecast-jobs:dev .
# Flows that need extra runtime deps (prefect, mlflow, boto3) get them via a pyproject
# extra: docker build --build-arg UV_EXTRAS="--extra jobs" ...

# ---- uv binary (pinned) ------------------------------------------------------
FROM ghcr.io/astral-sh/uv:0.9.25 AS uv

# ---- builder -------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /uvx /bin/
ARG UV_EXTRAS=""
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${UV_EXTRAS}

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${UV_EXTRAS} \
    # xgboost pulls ~200 MB of CUDA/NCCL wheels on linux that CPU inference never loads
    && rm -rf /opt/venv/lib/python3.12/site-packages/nvidia \
    && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} +

# ---- runtime ------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ARG VERSION=0.1.0
ARG REVISION=unknown
ARG SOURCE=https://github.com/OWNER/crypto-volatility-forecasting
LABEL org.opencontainers.image.title="volforecast-jobs" \
      org.opencontainers.image.description="Batch jobs: ingest, forecast, evaluate, retrain" \
      org.opencontainers.image.source="${SOURCE}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="MIT"

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
    MPLBACKEND=Agg \
    VOLF_DATA_DIR=/data

USER 10001:10001
WORKDIR /tmp

# Default flow; CronJobs / compose override the command per flow.
CMD ["python", "-m", "volforecast.flows.forecast"]
