"""One-shot fix for scripts/run_stress_test.py.

Adds --agent-model-path CLI flag and threads it through to
load_benchmark_opponent(), matching the model_path parameter just added to that
function (see fix_match_runner_model_path.py, applied first). Without this,
--attacker benchmark1/benchmark2 crashes with a missing model_path TypeError.

Run once on the pod, from the repo root:
    python fix_run_stress_test_model_path.py
"""

from pathlib import Path

target = Path("scripts/run_stress_test.py")
content = target.read_text()

# --- Fix 1: add the CLI flag, right after --sglang-agent-url's existing block ---

old_arg = '''    p.add_argument("--sglang-agent-url", default=None,
                    help="Required for --attacker benchmark1/benchmark2, e.g. http://localhost:30000")'''

new_arg = '''    p.add_argument("--sglang-agent-url", default=None,
                    help="Required for --attacker benchmark1/benchmark2, e.g. http://localhost:30000")
    p.add_argument("--agent-model-path", default=None,
                    help="Required for --attacker benchmark1/benchmark2 - the HF model directory "
                         "the sglang-agent-url server was launched with, e.g. "
                         "/workspace/export/benchmark2_full_sbso/merged_fp16")'''

if old_arg not in content:
    print("ERROR: --sglang-agent-url arg anchor not found - paste current args block.")
    raise SystemExit(1)
content = content.replace(old_arg, new_arg)

# --- Fix 2: require it alongside --sglang-agent-url, and pass it through --------

old_call = '''        return load_benchmark_opponent(
            args.attacker,
            sglang_agent_url=args.sglang_agent_url,
            prompt_history_path=args.prompt_history_path,
        )'''

new_call = '''        if not args.agent_model_path:
            raise SystemExit(
                f"--attacker {args.attacker} requires --agent-model-path (the HF "
                "directory the sglang-agent-url server was launched with)."
            )
        return load_benchmark_opponent(
            args.attacker,
            sglang_agent_url=args.sglang_agent_url,
            agent_model_path=args.agent_model_path,
            prompt_history_path=args.prompt_history_path,
        )'''

if old_call not in content:
    print("ERROR: load_benchmark_opponent call anchor not found.")
    raise SystemExit(1)
content = content.replace(old_call, new_call)

target.write_text(content)
print("Both replacements applied.")

import ast
ast.parse(target.read_text())
print("Syntax OK.")
