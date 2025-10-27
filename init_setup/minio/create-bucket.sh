#!/bin/sh
set -e

# небольшой ретрай на случай гонки после healthcheck
i=0
until mc alias set local http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1; do
  i=$((i+1))
  echo "Waiting for MinIO... ($i)"
  [ $i -ge 30 ] && echo "MinIO is not ready" && exit 1
  sleep 2
done

# создаём бакет под артефакты (идемпотентно)
mc mb -p local/mlflow-artifacts || true
mc version enable local/mlflow-artifacts || true

echo "Bucket 'mlflow-artifacts' ready."