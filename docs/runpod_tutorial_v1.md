# Sumo-SBSO on RunPod — Complete Beginner Tutorial

Written for someone who has never used RunPod before. Uses every fix/script
built and verified this session. Follow in order — don't skip the cheap steps
to save time on the expensive ones.

---

## 0. Before you touch RunPod at all — do this on your Mac first

These cost nothing and catch problems before you're paying for a GPU:

```bash
# From your repo root, on your Mac
python -m scripts.preflight_static_check
```

This must show `0 failed` before you spend a cent. If it doesn't, fix what
it flags first — it's free here, it's not free on RunPod.

**Also verify `requirements-cloud.txt` right now** (open it in an editor) —
confirm it includes `transformers`, `peft`, `datasets`, `sglang`, and this
session's additions (`wandb`, `huggingface_hub`). This file wasn't available
for audit this session; if any of those are missing, add them now, before
you're on the clock.

---

## 1. Create a RunPod account

1. Go to **runpod.io**, click **Sign Up**, verify your email.
2. Set up two-factor authentication (recommended).
3. Add credit under **Billing** — $15–20 is plenty to get through the pilot
   and a first real run at the rates below. You need at least a small
   balance on file before you can deploy anything.

---

## 2. (Recommended) Test the cheap way first — a CPU-only pod

You do **not** need a GPU to run the static preflight check or `pytest` again
against Linux instead of your Mac. A CPU pod costs pennies per hour.

1. In the console, click **+ New** (top right) → **Pod**, or click **Pods**
   in the left sidebar → **Deploy**.
2. Choose **Community Cloud** (cheaper) or **Secure Cloud** (more reliable,
   pricier) — Community is fine for this step.
3. Pick a **CPU-only** instance from the GPU list (filter for CPU, or pick
   the smallest/cheapest GPU option if no pure-CPU option is offered —
   either way, don't touch a 4090 for this step).
4. Click **Customize Deployment** → bump **Container Disk** to at least
   20 GB (room for the repo + Python packages).
5. Click **Deploy On-Demand**.
6. Once the pod shows **Running**, click **Connect** → open the web terminal
   (or use the SSH command RunPod shows you — see §5 below for the pattern).
7. On the pod:
   ```bash
   git clone <your-repo-url> sumo-sbso
   cd sumo-sbso
   pip install -r requirements.txt --break-system-packages
   python -m scripts.preflight_static_check
   pytest tests/unit/ -v
   ```
8. **Terminate this pod once done** — it's not needed again. Click the pod
   in **Pods**, then **Stop**/**Terminate**.

---

## 3. Create a Network Volume (do this before the real GPU pod)

Without this, everything you produce vanishes if the pod is ever stopped or
terminated — this is the single most important step for not losing work.

1. In the console, go to **Storage** → **Network Volumes** → **New Network
   Volume**.
2. Give it a name (e.g. `sumo-sbso-storage`), pick a size — **100 GB minimum**
   (models + checkpoints + training logs add up fast; Phi-4-mini and
   Llama-3.1-8B HF checkpoints alone are several GB each, times two backends).
3. Pick a **region** — remember this, your GPU pod must be deployed in the
   same region to attach this volume.
4. Create it. Cost is roughly $0.07/GB/month, billed whether or not it's
   attached to a running pod — don't leave it idle indefinitely once you're
   done with the project.

---

## 4. Deploy the real GPU pod

1. **Pods** → **Deploy**.
2. Choose **Community Cloud** for the cheaper rate, or **Secure Cloud** for
   fewer availability/reliability hiccups — either works for this project.
3. Use the **Network volume filter** to narrow the GPU list to your region
   (or select your volume directly if the flow offers that) — this ensures
   compatibility.
4. Select an **RTX 4090** (24 GB VRAM — enough for both SGLang servers at
   `mem_fraction_static: 0.45` each, per `inference.yaml`).
5. **Customize Deployment**:
   - Attach your Network Volume, mount path e.g. `/workspace`.
   - **Container Disk**: 20–30 GB is enough (the volume holds the big stuff).
   - **Exposed Ports**: add `30000` and `30001` (SGLang agent + judge server
     ports, per `inference.yaml`'s `launch:` block) if you want to `curl`
     them from outside the pod — optional, you can also just check from
     inside via SSH.
6. Choose **On-Demand**, not Spot — non-interruptible, per this project's
   earlier decision (see `session_record.md`). Costs more per hour than
   Spot but removes provider-side preemption risk entirely.
7. Review pricing (RTX 4090 Community Cloud is currently in the
   ~$0.30–0.35/hr range — confirm the exact number shown before deploying,
   rates change) and click **Deploy**.
8. Wait for **Running** status in **Pods**.

---

## 5. Connect and set up `tmux` immediately

1. Click **Connect** on your running pod → either open the web terminal, or
   copy the SSH command RunPod shows you (format:
   `ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa`).
2. **The very first thing you do, before anything else**:
   ```bash
   tmux new -s training
   ```
   Everything from here on happens inside this `tmux` session. If your SSH
   connection drops, the pod keeps running and so does whatever's inside
   `tmux` — reconnect later with `tmux attach -t training`. Skipping this
   step is the single most common way people accidentally lose a long
   training run to a dropped laptop connection, unrelated to RunPod itself.

---

## 6. Get the code onto the pod

```bash
cd /workspace   # your attached Network Volume
git clone <your-repo-url> sumo-sbso
cd sumo-sbso
```

If your models aren't in the repo (they shouldn't be — too large for git),
download them now (see §8) or `scp`/`rsync` them from your Mac into
`/workspace/models/`.

---

## 7. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
pip install -r requirements-cloud.txt --break-system-packages
```

**Verify, don't assume**, right after:
```bash
python -c "import transformers, peft, datasets, sglang, wandb, huggingface_hub; print('all present')"
```
If this errors, install whatever's missing explicitly before continuing —
this exact gap (a package silently missing from a requirements file) is
called out as an unverified risk in this session's audit.

---

## 8. Download the models

You need three model directories under `/workspace/models/` (adjust paths
to match your actual `config/inference.yaml` — these are the defaults):

```bash
mkdir -p /workspace/models
cd /workspace/models

# Phi-4-mini, HF format - SGLang agent server + LoRA fine-tuning base
huggingface-cli download microsoft/Phi-4-mini-instruct --local-dir phi-4-mini-hf

# Llama-3.1-8B-Instruct, HF format - SGLang judge server
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir llama-3.1-8b-instruct-hf
```

(Exact HF repo IDs depend on which checkpoints your report specifies —
confirm against `config/inference.yaml`'s `sglang.launch.*_model_path`
values before downloading; adjust the `huggingface-cli download` targets to
match.)

If you also want the local GGUF judge for anything (not needed for the real
pilot/full run, only for local Mac-style testing), that's a separate
download — skip it here, it's not part of the RunPod path.

---

## 9. (Optional) Log into wandb

```bash
wandb login
```
Paste your API key from wandb.ai/authorize. Skip this entirely if you're not
using `--use-wandb` — every script this session added it to treats it as
fully optional.

---

## 10. Run the static config check, one more time, on this exact machine

```bash
python -m scripts.preflight_static_check
```
Must show `0 failed`. This is the same check from §0 — running it again
here catches anything specific to this pod's config/environment (e.g. if
your GPU-rate placeholder in `phase4_pilot.yaml` still needs updating to
match the real rate you saw in §4).

---

## 11. Launch the SGLang servers

```bash
bash scripts/launch_sglang_servers.sh /workspace/models/phi-4-mini-hf /workspace/models/llama-3.1-8b-instruct-hf
```

This starts two detached `tmux` sessions (`sglang-agent`, `sglang-judge`) —
they keep running independently of your `training` session. Give them
30–90 seconds to load, then verify:

```bash
curl http://localhost:30000/health
curl http://localhost:30001/health
```
Both should return a healthy response before continuing.

---

## 12. Run the RunPod-only runtime preflight check

```bash
python -m scripts.preflight_runpod_check
```

This is the one that genuinely needs this exact machine — checks real GPU
visibility, both SGLang servers actually reachable, constrained decoding
schema-valid under real load, and the full `pytest` suite against this
image's real dependency versions. Must show `0 failed` before spending any
real training time.

---

## 13. Run the pilot — your first real spend

Back in your `training` tmux session:

```bash
python scripts/run_phase4_pilot.py --config config/phase4_pilot.yaml --episodes-override 100 --use-wandb
```

(`--episodes-override 100` keeps this short and cheap — a genuine
calibration run, not the full 500. Drop `--use-wandb` if you skipped §9.)

Watch the output — both live in your terminal and permanently in
`checkpoints/<run_id>/run.log`. When it finishes, check:

```bash
cat checkpoints/*/progress.json
python -m scripts.report_strategy_distribution --training-pairs checkpoints/*/training_pairs.jsonl --prompt-history checkpoints/*/prompt_history.jsonl
```

This tells you real wall-clock cost per episode and whether DSPy actually
recompiled sensibly — decide now whether `k_rollout_batches`/`W`/`delta`
need any final adjustment before the real run (per the open item in
`session_record.md`).

---

## 14. Run the real full training — 5 separate runs

One command per variant, same script, different config each time:

```bash
python scripts/run_phase4_pilot.py --config config/phase4_full_sbso.yaml --use-wandb
python scripts/run_phase4_pilot.py --config config/phase4_ablation_no_sa.yaml --use-wandb
python scripts/run_phase4_pilot.py --config config/phase4_ablation_no_mcts.yaml --use-wandb
python scripts/run_phase4_pilot.py --config config/phase4_ablation_no_dspy.yaml --use-wandb
python scripts/run_phase4_pilot.py --config config/phase4_ablation_no_judge.yaml --use-wandb
```

Run these sequentially, not in parallel — they'd contend for the same GPU
and SGLang servers. Each writes to its own `checkpoint_output_dir` (set in
each YAML). This is the expensive part — the pilot's cost projection
(printed at the end of §13, or check `wandb` if enabled) told you roughly
what to expect for each.

---

## 15. After training — export the model (Benchmark 2 only, not the ablations)

```bash
python scripts/run_export_pipeline.py --config config/export_pipeline.yaml
```

This runs merge → {GPTQ INT4, GGUF-f16 → Q4_K_M} per the two-branch export
in the pipeline report. Verify the result:

```bash
python -m scripts.validate_quantization_quality \
    --fp16-gguf-path checkpoints/benchmark2_full_sbso/gguf/model-f16.gguf \
    --quantized-gguf-path checkpoints/benchmark2_full_sbso/gguf/model-Q4_K_M.gguf \
    --validation-prompts checkpoints/benchmark2_full_sbso/training_pairs.jsonl
```

---

## 16. Back up what you built

```bash
huggingface-cli login   # once, if not already
python -m scripts.backup_to_hub \
    --repo-id <your-username>/sumo-sbso-benchmark2 \
    --adapter-dir checkpoints/benchmark2_full_sbso/lora_run/lora_adapters \
    --merged-dir checkpoints/benchmark2_full_sbso/merged_fp16 \
    --gguf-dir checkpoints/benchmark2_full_sbso/gguf
```

This is redundancy *on top of* the Network Volume (§3), which already keeps
everything safe as long as the pod is attached to it — this step protects
against a Network Volume-level problem specifically, for your most
expensive-to-regenerate artifacts.

---

## 17. Shut down and stop paying

```bash
tmux kill-session -t sglang-agent
tmux kill-session -t sglang-judge
exit   # leaves the training tmux session running in background if still needed
```

Then in the console: **Pods** → your pod → **Stop** (keeps the Network
Volume and its contents, stops GPU billing) or **Terminate** (also fine,
since everything durable is on the Network Volume + HF Hub backup, not the
pod's local disk). Don't leave a Running pod idle — billing is per-second
whether you're using it or not.

---

## Troubleshooting quick-reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'mcts'` on startup | Stale local copy of `_shared_defaults.yaml` predating this session's fix | Re-pull the fixed config file |
| `ImportError` for `transformers`/`peft`/`wandb`/etc. | `requirements-cloud.txt` genuinely missing a package | Add it, `pip install`, retry — see §0/§7 |
| SGLang server `/health` never responds | Servers still loading (can take 90s), or wrong model path | Wait, then `tmux attach -t sglang-judge` to see the real error |
| Training stopped when you closed your laptop | Ran outside `tmux` | Always `tmux new -s training` first — §5 |
| Everything gone after pod restart | No Network Volume attached, or `checkpoint_output_dir` pointed outside it | Confirm §3 was done and configs point at the mounted path |
| `preflight_runpod_check.py` fails on constrained decoding | Real load issue with the grammar/regex constraint — see pipeline report Phase 3 Critical Remarks | Check `run.log`, consider the silent-fallback instrumentation discussed earlier in this session |
