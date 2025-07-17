#!/bin/bash
echo "🚀 Starting backend server..."

# Activate conda
source /opt/conda/etc/profile.d/conda.sh
conda activate thyroid-classifier

# Run uvicorn with unbuffered logging and auto-reload disabled for production
exec python -u -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug
