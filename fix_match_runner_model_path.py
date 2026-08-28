"""One-shot fix for src/evaluation/match_runner.py's load_benchmark_opponent.

Same gap as every other SGLangSLMClient call site fixed tonight - built before the
chat-template fix added model_path as a required constructor argument, so this one
call site was missed (match_runner.py isn't in project knowledge, was never
re-uploaded for re-checking after that fix landed).

Run once on the pod, from the repo root:
    python fix_match_runner_model_path.py
"""

from pathlib import Path

target = Path("src/evaluation/match_runner.py")
content = target.read_text()

old_sig = '''def load_benchmark_opponent(
    benchmark: Literal["benchmark1", "benchmark2"],
    sglang_agent_url: str,
    *,
    perception_config_path: str | None = None,
    prompt_history_path: str | None = None,
) -> SLMPolicyOpponent:'''

new_sig = '''def load_benchmark_opponent(
    benchmark: Literal["benchmark1", "benchmark2"],
    sglang_agent_url: str,
    agent_model_path: str,
    *,
    perception_config_path: str | None = None,
    prompt_history_path: str | None = None,
) -> SLMPolicyOpponent:'''

if old_sig not in content:
    print("ERROR: signature anchor not found - paste current file content, do not guess.")
    raise SystemExit(1)
content = content.replace(old_sig, new_sig)

old_call = "SGLangSLMClient(server_url=sglang_agent_url)"
new_call = "SGLangSLMClient(server_url=sglang_agent_url, model_path=agent_model_path)"

if old_call not in content:
    print("ERROR: SGLangSLMClient call anchor not found.")
    raise SystemExit(1)
content = content.replace(old_call, new_call)

target.write_text(content)
print("Both replacements applied.")

import ast
ast.parse(target.read_text())
print("Syntax OK.")
