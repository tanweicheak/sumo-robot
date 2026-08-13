# RunPod Tutorial: Sumo-SBSO Stage 3 Training, Step by Step

This assumes zero prior RunPod experience. Every step says exactly which file to run,
what to expect back, and how to tell it worked before moving to the next step.

---

## Part A — One-time account setup

### A1. Create an account and add funds
Go to runpod.io, sign up, add credit under **Billing**. Billed per second the pod runs,
not a subscription.

**Input:** your credit card. **Output:** account with a balance. **Expect:** $20-30 is
plenty to get through Parts B-D below; the full training run (Part E) will cost more -
you'll know roughly how much once you finish Part D's timing measurement.

### A2. Create a Network Volume (do this before renting anything)
**RunPod dashboard → Storage → Network Volumes → Create.** Size: 50GB is comfortable
(model weights + checkpoints + training_pairs.jsonl).

**Why this matters:** a Network Volume survives even after you terminate a pod. Skip
this and use only pod-local storage, and terminating the pod (or it getting reclaimed)
deletes everything - checkpoints, training data, all of it.

**Expect:** a volume shown in your dashboard with a name and size. Note the name -
you'll attach it when creating the pod.

### A3. Add your SSH key
**Settings → SSH Keys → Add.** If you don't have one: `ssh-keygen -t ed25519` on your
Mac, then paste the contents of `~/.ssh/id_ed25519.pub`.

**Expect:** key listed in your account settings. This lets you SSH in instead of using
the (slower, more limited) web terminal for every session.

---

## Part B — Rent and connect to a pod

### B1. Deploy a pod
**Pods → Deploy.** Choose:
- GPU: RTX 4090
- Template: a PyTorch template (comes with CUDA pre-installed)
- Network Volume: the one from A2
- Container Disk: 20-30GB (separate, temporary scratch space)

Click Deploy. **Expect:** pod running within 1-2 minutes, shown in your Pods list.

### B2. Connect
Click **Connect** on the pod card → copy the SSH command shown (includes your pod's IP
and port). Run it from your Mac's terminal.

**Expect:** a shell prompt on the pod, e.g. `root@<pod-id>:/workspace#`. Your network
volume is mounted at `/workspace` - confirm with `df -h /workspace`.

---

## Part C — Get your code and dependencies onto the pod

### C1. Copy your project
From your Mac (a new terminal, not the SSH session):
```bash
scp -P <port> -r /Users/weicheaktan/Documents/UM/sumo_robot/source_code root@<pod-ip>:/workspace/
```
(Use the exact port/IP RunPod showed you in B2. `git clone` is faster if your repo is
on GitHub/GitLab - use whichever you have.)

**Expect:** your full project folder now under `/workspace/source_code` on the pod.
Verify: `ls /workspace/source_code` (back in the SSH session) should show `scripts/`,
`src/`, `config/`, etc.

### C2. Install dependencies
```bash
cd /workspace/source_code
pip install -r requirements.txt -r requirements-cloud.txt
```
**Input:** `requirements.txt` + `requirements-cloud.txt` (built earlier this project).
**Expect:** no errors. `sglang`, `gptqmodel`, `peft`, `pybullet` etc. all installed.
Sanity check: `python -c "import sglang, pybullet, torch; print(torch.cuda.is_available())"`
should print `True` for CUDA availability - if `False`, something's wrong with the pod's
GPU setup before you go any further.

### C3. Download model weights
Pull directly from HuggingFace on the pod (much faster than uploading from your Mac):
```bash
huggingface-cli download microsoft/Phi-4-mini-instruct --local-dir /workspace/models/phi-4-mini-hf
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir /workspace/models/llama-3.1-8b-instruct-hf
```
(Exact model repo names to confirm against your report/config if these differ.)

**Expect:** two populated model directories under `/workspace/models/`. Verify with
`ls /workspace/models/llama-3.1-8b-instruct-hf` - should show `config.json`, safetensors
files, tokenizer files.

---

## Part D — Smoke test everything before spending real time/money

### D1. Start the SGLang servers
**Input:** `scripts/launch_sglang_servers.sh` (built this project - launches TWO
servers: agent on 30000, judge on 30001, matching `inference.yaml`).
```bash
bash scripts/launch_sglang_servers.sh /workspace/models/phi-4-mini-hf /workspace/models/llama-3.1-8b-instruct-hf
```
For a training-only run you can omit the agent model path (training doesn't need it -
only the judge server does):
```bash
bash scripts/launch_sglang_servers.sh "" /workspace/models/llama-3.1-8b-instruct-hf
```

**Expect:** two `tmux` sessions started (`sglang-agent` if requested, `sglang-judge`).
Wait 30-90s, then confirm:
```bash
curl http://localhost:30001/health
```
**Expect:** a healthy response (not a connection error). If it hangs or errors, check
logs with `tmux attach -t sglang-judge` (detach again with `Ctrl+B` then `D` - this does
NOT stop the server).

### D2. Connectivity + pipeline smoke test
**Input:** `scripts/run_phase4_stage3_cloud.py`, deliberately tiny settings.
```bash
python scripts/run_phase4_stage3_cloud.py \
    --config config/phase4_full_sbso.yaml \
    --judge-server-url http://127.0.0.1:30001 \
    --episodes-override 2 --sim-budget 5 --horizon 2 --cycles-per-node 3
```
**Output:** `checkpoints/benchmark2_full_sbso/progress.json` and
`.../training_pairs.jsonl` (paths come from the config's `checkpoint_output_dir`), plus
console output ending in a `summary={...}` line.

**Expect this to finish in a few minutes, not hours** - this is deliberately weak/fast
settings, purely to confirm the whole chain (PyBullet → real Judge over SGLang → MCTS →
match-runner → checkpoint files) works end to end on this pod. **Do not use this run's
numbers for anything except "did it work."**

**If it fails:** the error will point at whichever piece broke - a connection error means
D1 didn't actually come up; a Python traceback means something in the pipeline itself.
Fix before proceeding - don't skip ahead with a broken pipeline.

### D3. Real-fidelity timing sample (this is plan item #4)
Same script, small episode **count** but real-fidelity settings - this is what actually
feeds `cost_projection.py`.
```bash
python scripts/run_phase4_stage3_cloud.py \
    --config config/phase4_full_sbso.yaml \
    --judge-server-url http://127.0.0.1:30001 \
    --episodes-override 3 --sim-budget 30 --horizon 4 --cycles-per-node 3
```
(Confirm 30/4/3 are your actual intended production values before running this -
flagged earlier as worth double-checking against the report.)

**Output:** console line `[stage3-cloud] done in {wall_clock_s}s`, plus
`judge.call_count`. **Expect:** noticeably slower than D2 (this is real MCTS depth
against a real 8B judge model) - this is the number to report back for
`cost_projection.py` to turn into a real dollar estimate for the full run.

---

## Part E — The real training run

### E1. Confirm the DSPy trigger values are real, not placeholders
```bash
grep -A3 "dspy_recompilation" config/_shared_defaults.yaml
```
**Expect:** real numbers, not `null`. If still `null`, either fix the file or pass
`--k-episodes`/`--window-w`/`--delta` explicitly in E2 - the script refuses to guess.

### E2. Launch the full run
**Input:** `scripts/run_phase4_stage3_cloud.py`, no episode override (uses the config's
real `episodes_total: 5000`). Run inside `tmux` so it survives disconnects:
```bash
tmux new -s training
cd /workspace/source_code
python scripts/run_phase4_stage3_cloud.py \
    --config config/phase4_full_sbso.yaml \
    --judge-server-url http://127.0.0.1:30001 \
    --sim-budget 30 --horizon 4 --cycles-per-node 3
```
Detach with `Ctrl+B` then `D` once it's running - training continues in the background.
Reattach any time with `tmux attach -t training`.

**Output (produced incrementally, not just at the end):**
- `checkpoints/benchmark2_full_sbso/progress.json` - updated every checkpoint interval; latest episode, prompt program, win history
- `checkpoints/benchmark2_full_sbso/training_pairs.jsonl` - every decision made, appended live
- `checkpoints/benchmark2_full_sbso/gptq_calibration_texts.txt` - written once, at the very end

**Expect:** this takes a genuinely long time (hours, scaled from D3's per-episode
timing × 5000). Check progress periodically with
`cat checkpoints/benchmark2_full_sbso/progress.json` without needing to interrupt
anything. If the pod gets interrupted, you lose at most one checkpoint-interval's worth
of progress, not the whole run - but note the run does NOT auto-resume; restarting means
starting over from episode 0 with what's already saved kept on disk for reference.

### E3. Repeat for the four ablations
Same command, four more times, pointing `--config` at each ablation's yaml
(`config/ablation_*.yaml` or similar - whatever your actual ablation config filenames
are) and a distinct `checkpoint_output_dir` per variant so they don't overwrite each other.

---

## Part F — Export deployable artifacts

Once a variant's training run finishes:

```bash
python scripts/run_export_pipeline.py \
    --base-model-path /workspace/models/phi-4-mini-hf \
    --adapter-path checkpoints/benchmark2_full_sbso/lora_adapter \
    --output-dir /workspace/export/benchmark2_full_sbso \
    --llama-cpp-convert-script /path/to/llama.cpp/convert_hf_to_gguf.py \
    --calibration-texts-file checkpoints/benchmark2_full_sbso/gptq_calibration_texts.txt
```
**Input:** the trained adapter (from a LoRA fine-tuning step on `training_pairs.jsonl` -
not covered in this tutorial, a separate step) + the auto-generated calibration file
from E2.
**Output:** `/workspace/export/benchmark2_full_sbso/model-q4_k_m.gguf` (edge/llama.cpp)
and `.../gptq_int4/` (cloud/SGLang serving) - two independent artifacts, per the D4 fix.

**Expect:** console output showing `[1/3]`, `[2/3]`, `[3/3]` steps completing, ending
with both artifact paths printed.

---

## Part G — Get results back and shut down

### G1. Copy results to your Mac
```bash
scp -P <port> -r root@<pod-ip>:/workspace/source_code/checkpoints ./local-results
scp -P <port> -r root@<pod-ip>:/workspace/export ./local-export
```

### G2. Terminate the pod
**Pods → Terminate.** Stops GPU billing immediately. Your Network Volume (and
everything on it) persists independently - nothing is lost by terminating the pod
itself, only by deleting the volume.

**Expect:** pod removed from your active list; billing for it stops; the network volume
still shows in Storage, available to attach to a future pod if needed.
