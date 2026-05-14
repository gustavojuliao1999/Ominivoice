#!/bin/bash
set -e

if [ -n "$RUNPOD_WEBHOOK_GET_JOB" ]; then
    echo "Modo: Serverless RunPod"
    exec python handler.py
else
    echo "Modo: API Pod (FastAPI)"
    exec uvicorn server:app --host 0.0.0.0 --port 8000
fi
