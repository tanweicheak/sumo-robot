"""One-shot fix for src/finetuning/lora_finetune.py's evaluate_strategy_accuracy.

Bug (confirmed via direct raw-generation inspection against the flawed checkpoint):
exact string equality (generated == target) fails on correct-but-over-generated
output, e.g. generated='evade_edge\nmom=still\n' vs target='evade_edge' - the model
got the content right but kept generating past the strategy word (no chat-template
turn boundary to signal "stop here" - see 3.6.3 in the report reference doc). This
produced a false 0.0 accuracy on a checkpoint that had partially learned something
real, alongside a genuine, separate collapse toward one dominant answer for some
inputs regardless of target - the fix below distinguishes the two rather than
conflating them into one wrong number.

Fix: accept a match if the target appears as the first line / a prefix of the
generation (tolerant of trailing hallucinated continuation), not only byte-exact
equality. Also returns a `dominant_output_rate` diagnostic - how often the SAME
generated string appears regardless of target - so the retrained checkpoint's real
generalization (not just accuracy) can be checked directly.

Run once on the pod, from the repo root:
    python fix_evaluate_strategy_accuracy.py
"""

from pathlib import Path

target_file = Path("src/finetuning/lora_finetune.py")
content = target_file.read_text()

old = '''    model.eval()
    correct = 0
    for r in records:
        inputs = tokenizer(r["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if generated == r["completion"].strip():
            correct += 1

    n = len(records)
    return {"n": n, "correct": correct, "accuracy": round(correct / n, 4) if n else None}'''

new = '''    from collections import Counter

    model.eval()
    correct = 0
    outputs_seen = Counter()
    for r in records:
        inputs = tokenizer(r["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        target = r["completion"].strip()
        # Accept the target as the first line OR a leading prefix of generation -
        # tolerant of trailing hallucinated continuation past the strategy word
        # (no chat-template turn boundary to signal "stop here"), NOT just
        # byte-exact equality, which previously produced false negatives on
        # correct-but-over-generated output (e.g. 'evade_edge\\nmom=still\\n').
        first_line = generated.split("\\n")[0].strip()
        is_match = (generated == target) or (first_line == target) or generated.startswith(target)
        if is_match:
            correct += 1
        outputs_seen[first_line] += 1

    n = len(records)
    dominant_output, dominant_count = outputs_seen.most_common(1)[0] if outputs_seen else (None, 0)
    return {
        "n": n, "correct": correct, "accuracy": round(correct / n, 4) if n else None,
        # Collapse diagnostic - separate from accuracy on purpose. A checkpoint can
        # score well on accuracy AND still show a high dominant_output_rate if the
        # dominant output happens to be correct often (e.g. because that strategy
        # genuinely IS the right answer often) - this number alone doesn't prove
        # collapse, but a HIGH dominant_output_rate paired with LOW accuracy is the
        # collapse signature to watch for, distinct from "the metric was wrong."
        "dominant_output": dominant_output,
        "dominant_output_count": dominant_count,
        "dominant_output_rate": round(dominant_count / n, 4) if n else None,
    }'''

if old not in content:
    print("ERROR: exact function body not found - file may differ from what this script expects.")
    raise SystemExit(1)

content = content.replace(old, new)
target_file.write_text(content)
print("Fix applied: evaluate_strategy_accuracy now uses prefix-tolerant matching + reports dominant_output_rate.")

import ast
ast.parse(target_file.read_text())
print("Syntax check passed - file parses correctly.")
