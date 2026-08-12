#!/usr/bin/env bash
# scripts/start_sglang_server.sh
#
# Starts the SGLang server in a detached tmux session, so it keeps running after you
# disconnect and your training script (run against http://localhost:30000) can reach it.
#
# Usage (on the RunPod pod, after `pip install -r requirements.txt -r requirements-cloud.txt`):
#   bash scripts/start_sglang_server.sh /workspace/models/phi-4-mini
#
# Check it's actually up before starting training:
#   curl http://localhost:30000/health
#
# Watch its logs:
#   tmux attach -t sglang
# (detach again with Ctrl+B then D - this does NOT stop the server)
#
# Stop it:
#   tmux kill-session -t sglang

set -euo pipefail

MODEL_PATH="${1:?Usage: start_sglang_server.sh <model_path> [port]}"
PORT="${2:-30000}"

if tmux has-session -t sglang 2>/dev/null; then
    echo "A tmux session named 'sglang' already exists. Run 'tmux kill-session -t sglang' first if you want to restart it."
    exit 1
fi

tmux new-session -d -s sglang \
    "python -m sglang.launch_server --model-path '${MODEL_PATH}' --port ${PORT} --host 0.0.0.0"

echo "SGLang server starting in tmux session 'sglang' on port ${PORT}."
echo "It can take 30-90s to finish loading the model before /health responds - poll with:"
echo "  curl http://localhost:${PORT}/health"
