#!/usr/bin/env bash
# ==============================================================================
# VitalPulse Physiological Stress API Server Startup Script
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default environment configuration
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export CORS_ORIGINS="${CORS_ORIGINS:-http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001}"

echo "==================================================================="
echo "Starting VitalPulse Physiological Stress API"
echo "Host:         http://${HOST}:${PORT}"
echo "API Docs:     http://${HOST}:${PORT}/docs"
echo "CORS Origins: ${CORS_ORIGINS}"
echo "==================================================================="

# Execute via project virtual environment
exec ./venv/bin/uvicorn backend.physiology_api:app --host "$HOST" --port "$PORT" --reload

