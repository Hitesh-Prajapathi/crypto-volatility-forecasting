#!/bin/sh
# Creates the extra databases on first boot of the postgres container (runs as POSTGRES_USER).
set -eu
for db in mlflow prefect; do
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "CREATE DATABASE ${db} OWNER ${POSTGRES_USER}"
done
