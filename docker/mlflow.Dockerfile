# syntax=docker/dockerfile:1.7
# MLflow tracking server with the Postgres + S3 (MinIO) drivers the stock image lacks.
FROM python:3.12-slim-bookworm

ARG MLFLOW_VERSION=3.18.0
RUN pip install --no-cache-dir \
        "mlflow==${MLFLOW_VERSION}" \
        "psycopg2-binary==2.9.10" "boto3==1.40.0" \
    && groupadd --gid 10001 mlflow \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin mlflow

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
USER 10001:10001
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
