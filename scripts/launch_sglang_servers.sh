#!/usr/bin/env bash
# scripts/launch_sglang_servers.sh
#
# Phase 4 (Stage 3) - launches the two SGLang servers this project needs: the agent
# model (Phi-4-mini, serves OAA/SA/TEA calls + MCTS rollouts) and the judge model
# (Llama-3.1-8B-Instruct). Reads config/inference.yaml's sglang.launch block.
# CUDA-only; will not run on the local Mac (no SGLang without NVIDIA GPU).
#
# Usage:
#   bash scripts/launch_sglang_servers.sh
#   bash scripts/launch_sglang_servers.sh stop      # kill both servers

set -euo pipefail

CONFIG_FILE="config/inference.yaml"
LOG_DIR="logs/sglang"
mkdir -p "$LOG_DIR"

# --- tiny YAML value extraction (avoids adding a bash YAML dependency) ---
_yaml_get() {
  python3 -c "
import sys
sys.path.insert(0, '.')
from src.common.config_loader import load_config
cfg = load_config('$CONFIG_FILE')
node = cfg
for part in '$1'.split('.'):
    node = node[part]
print(node)
"
}

if [[ "${1:-}" == "stop" ]]; then
  echo "Stopping SGLang servers..."
  pkill -f "sglang.launch_server.*--port $(_yaml_get sglang.launch.agent_port)" || true
  pkill -f "sglang.launch_server.*--port $(_yaml_get sglang.launch.judge_port)" || true
  echo "Done."
  exit 0
fi

AGENT_MODEL=$(_yaml_get sglang.launch.agent_model_path)
AGENT_PORT=$(_yaml_get sglang.launch.agent_port)
AGENT_MEM=$(_yaml_get sglang.launch.agent_mem_fraction_static)
JUDGE_MODEL=$(_yaml_get sglang.launch.judge_model_path)
JUDGE_PORT=$(_yaml_get sglang.launch.judge_port)
JUDGE_MEM=$(_yaml_get sglang.launch.judge_mem_fraction_static)
HOST=$(_yaml_get sglang.launch.host)
TIMEOUT_S=$(_yaml_get sglang.launch.startup_timeout_s)

echo "Launching agent server: $AGENT_MODEL on port $AGENT_PORT (mem_fraction=$AGENT_MEM)"
nohup python3 -m sglang.launch_server \
    --model-path "$AGENT_MODEL" \
    --host "$HOST" --port "$AGENT_PORT" \
    --mem-fraction-static "$AGENT_MEM" \
    > "$LOG_DIR/agent_server.log" 2>&1 &
echo $! > "$LOG_DIR/agent_server.pid"

echo "Launching judge server: $JUDGE_MODEL on port $JUDGE_PORT (mem_fraction=$JUDGE_MEM)"
nohup python3 -m sglang.launch_server \
    --model-path "$JUDGE_MODEL" \
    --host "$HOST" --port "$JUDGE_PORT" \
    --mem-fraction-static "$JUDGE_MEM" \
    > "$LOG_DIR/judge_server.log" 2>&1 &
echo $! > "$LOG_DIR/judge_server.pid"

echo "Waiting for both servers to become healthy (timeout ${TIMEOUT_S}s)..."
for port in "$AGENT_PORT" "$JUDGE_PORT"; do
  waited=0
  until curl -sf "http://${HOST}:${port}/health" > /dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if [[ $waited -ge $TIMEOUT_S ]]; then
      echo "ERROR: server on port $port did not become healthy within ${TIMEOUT_S}s."
      echo "Check $LOG_DIR/*.log for details."
      exit 1
    fi
  done
  echo "Port $port healthy."
done

echo "Both SGLang servers are up."
echo "  agent:  http://${HOST}:${AGENT_PORT}"
echo "  judge:  http://${HOST}:${JUDGE_PORT}"
echo "Logs: $LOG_DIR/agent_server.log, $LOG_DIR/judge_server.log"
echo "Stop with: bash scripts/launch_sglang_servers.sh stop"