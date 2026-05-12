#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_PATH="${ROOT_DIR}/cmd/api/default.pgo"
BENCH_TIME="${BENCH_TIME:-20s}"
CACHE_DIR="${GOCACHE:-/tmp/rinha-go-cache}"

echo "generating PGO profile at ${PROFILE_PATH}"
rm -f "${PROFILE_PATH}"

cd "${ROOT_DIR}"
GOCACHE="${CACHE_DIR}" \
go test ./internal/fraud \
  -run '^$' \
  -bench '^BenchmarkFraudHotPath$' \
  -benchtime="${BENCH_TIME}" \
  -count=1 \
  -cpuprofile "${PROFILE_PATH}"

ls -lh "${PROFILE_PATH}"
echo "PGO profile ready"
