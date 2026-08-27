"""
One-off diagnostic: calls the agent SGLang server's OpenAI-compatible
/v1/chat/completions endpoint directly with a plain `openai` client - no
DSPy, no litellm in the path at all. Isolates whether "input_ids should be
a list of lists for batch processing" is:
  (a) a general SGLang bug on this endpoint/version - this script fails too, or
  (b) specific to how litellm formats DSPy's request - this script succeeds,
      narrowing the problem to litellm's request shape, not SGLang itself.

Run on the pod, with the agent server already up:
    python repro_sglang_chat.py
"""

from openai import OpenAI

# Match run_phase4_pilot.py's actual construction: f"{sg['agent_server_url']}/v1"
client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

print("=== Test 1: single chat completion, n=1 (default) ===")
try:
    resp = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "State: opp=mid,dir=FC;edge=safe;mom=still\nStrategy:"}],
        max_tokens=8,
        temperature=0.0,
    )
    print("OK:", resp.choices[0].message.content)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\n=== Test 2: explicit n=1 passed (DSPy/litellm may set this explicitly) ===")
try:
    resp = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "State: opp=mid,dir=FC;edge=safe;mom=still\nStrategy:"}],
        max_tokens=8,
        temperature=0.0,
        n=1,
    )
    print("OK:", resp.choices[0].message.content)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\n=== Test 3: system + user messages (DSPy predictors often add a system prompt) ===")
try:
    resp = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a sumo robot strategist."},
            {"role": "user", "content": "State: opp=mid,dir=FC;edge=safe;mom=still\nStrategy:"},
        ],
        max_tokens=8,
        temperature=0.0,
    )
    print("OK:", resp.choices[0].message.content)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\n=== Test 4: two rapid sequential calls (not concurrent - just back-to-back) ===")
try:
    for i in range(2):
        resp = client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": f"State: opp=near,dir=FC;edge=safe;mom=still ({i})\nStrategy:"}],
            max_tokens=8,
            temperature=0.0,
        )
        print(f"OK (call {i}):", resp.choices[0].message.content)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("\nDone. Paste this whole output back - which tests failed (if any) narrows the cause.")
