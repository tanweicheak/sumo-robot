#!/usr/bin/env bash
# scripts/launch_sglang_servers.sh
#
# Starts BOTH SGLang servers inference.yaml expects, each in its own detached tmux
# session, so they keep running after you disconnect:
#   - agent server (port 30000): Phi-4-mini - only needed for LIVE match inference
#     (OAA/SA/TEA via LangGraph), NOT needed for Stage 3 SBSO training itself, which
#     uses MacroStrategyExecutor instead of live SLM calls during MCTS.
#   - judge server (port 30001): Llama-3.1-8B-Instruct - THIS is the one
#     run_phase4_pilot.py actually needs.
#
# Usage (on the RunPod pod, after pip install -r requirements.txt -r requirements-cloud.txt):
#   bash scripts/launch_sglang_servers.sh /workspace/models/phi-4-mini-hf /workspace/models/llama-3.1-8b-instruct-hf
#
# Training-only run? You can skip the agent server entirely - see launch_judge_only below.
#
# Check both are up:
#   curl http://localhost:30000/health
#   curl http://localhost:30001/health
#
# Watch logs:
#   tmux attach -t sglang-agent    (Ctrl+B then D to detach again)
#   tmux attach -t sglang-judge
#
# Stop:
#   tmux kill-session -t sglang-agent
#   tmux kill-session -t sglang-judge

set -euo pipefail

AGENT_MODEL_PATH="${1:-}"
JUDGE_MODEL_PATH="${2:?Usage: launch_sglang_servers.sh [agent_model_path] <judge_model_path>}"
AGENT_PORT="${3:-30000}"
JUDGE_PORT="${4:-30001}"
MEM_FRACTION="${5:-0.45}"   # inference.yaml default - each server gets ~45% of GPU memory

_start_one() {
    local name="$1" model_path="$2" port="$3"
    if tmux has-session -t "$name" 2>/dev/null; then
        echo "tmux session '$name' already exists - run 'tmux kill-session -t $name' first to restart it."
        return 0
    fi
    tmux new-session -d -s "$name" \
        "python -m sglang.launch_server --model-path '${model_path}' --port ${port} --host 0.0.0.0 --mem-fraction-static ${MEM_FRACTION}"
    echo "Starting '$name' on port ${port} (model: ${model_path})"
}

if [ -n "$AGENT_MODEL_PATH" ]; then
    _start_one "sglang-agent" "$AGENT_MODEL_PATH" "$AGENT_PORT"
else
    echo "No agent_model_path given - skipping the agent server (fine for training-only runs;"
    echo "run_phase4_pilot.py only needs the judge server)."
fi
_start_one "sglang-judge" "$JUDGE_MODEL_PATH" "$JUDGE_PORT"

echo
echo "Both can take 30-90s to finish loading before /health responds."