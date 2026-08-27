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

**Also verify `requirements_runpod.txt` right now** (open it in an editor) —
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
   either way, don't touch a 4090 for this step). Also pick a **minimal
   Pod Template** (plain Ubuntu / base Python image) rather than a
   preloaded "PyTorch"/CUDA template — those templates ship with several
   GB of CUDA runtime already on disk before you install anything, and
   you don't need any of it for this CPU-only step.
4. Click **Customize Deployment** → set **Container Disk** to at least
   20 GB. That's enough *if* you (a) picked a minimal template in step 3,
   not a preloaded CUDA/PyTorch one, and (b) install the CPU-only `torch`
   wheel explicitly in step 7 below — an unpinned `torch>=2.2` otherwise
   pulls PyPI's default CUDA build (`torch` itself plus
   `nvidia_cudnn_cu13`, `nvidia_cusparselt_cu13`, `cuda_toolkit` — over
   1 GB combined) even though this pod has no GPU. If you can get more
   than 20 GB, 30–50 GB gives more margin for error; 20 GB works with
   both mitigations followed.
5. Click **Deploy On-Demand**.
6. Once the pod shows **Running**, click **Connect** → open the web terminal
   (or use the SSH command RunPod shows you — see §5 below for the pattern).
7. On the pod, set up Python 3.12 the same reliable way as §7 below — don't
   just check `python3 --version` and hope, and don't assume `python3.12`
   exists on this image just because the command name matches one you've
   used before. Confirm and install explicitly, every time:
   ```bash
   which python3.12 || (apt-get update && apt-get install -y python3.12 python3.12-venv)
   # if that package isn't found on your image's default repos:
   #   apt-get install -y software-properties-common
   #   add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.12 python3.12-venv

   which cmake || apt-get install -y cmake build-essential   # llama-cpp-python
   # falls back to a source build without a compiler + cmake present, and
   # that source build failing can abort the entire requirements.txt install

   git clone <your-repo-url> sumo-sbso
   cd sumo-sbso
   python3.12 -m venv /workspace/venv && source /workspace/venv/bin/activate
   python --version   # confirm 3.12.x - a mismatch here is why pybullet
                       # would try to build from source and may fail

   df -h /workspace   # note free space now, as a baseline

   # CPU-only torch FIRST, and --no-cache-dir throughout: pip's cache
   # keeps a second copy of every downloaded wheel on disk in addition to
   # the installed package, which wastes real space on a tight 20 GB
   # budget for a pod you'll terminate anyway. An unpinned torch>=2.2
   # would otherwise pull PyPI's default CUDA build (~1 GB+ of
   # nvidia_cudnn/cusparselt/cuda_toolkit) even on a pod with no GPU, and
   # can exhaust a small Container Disk mid-install - taking pyyaml,
   # pytest, and everything else in the same requirements.txt down with
   # it, since pip resolves the whole file before installing anything.
   python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

   df -h /workspace   # checkpoint: confirm this didn't eat more than ~1-1.5 GB

   python -m pip install --no-cache-dir -r requirements.txt

   df -h /workspace   # checkpoint: total usage should still be well under 20 GB

   python -c "import yaml, pytest, torch; print(torch.__version__, torch.cuda.is_available())"
   # last value should print False here - correct on this CPU-only pod

   python -m scripts.preflight_static_check
   python -m pytest tests/unit/ -v   # not bare `pytest` - see troubleshooting table
   ```
8. **Terminate this pod once done** — it's not needed again. Click the pod
   in **Pods**, then **Stop**/**Terminate**.

---

## 3. Create a Network Volume (do this before the real GPU pod) — or read the alternative below first

Without this, everything you produce vanishes if the pod is ever
**terminated** — this is the single most important step for not losing work.

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

**Running the 5 variants (§14) in parallel, on 5 separate pods:** one
Network Volume, attached to all 5 pods at the same time — this is fully
supported, it's not "detach from pod 1, reattach to pod 2." Download both
base models once, from whichever pod you deploy first (§8); the other 4
pods see those files immediately at the same mount path, no re-download.
Each variant's distinct `checkpoint_output_dir` (already set per-YAML)
keeps their outputs from colliding on the shared volume. One constraint:
confirm your chosen GPU type is available **in this exact volume's
datacenter** for all 5 deployments — if your account can only get fewer
than 5 of that GPU type in that datacenter at once, deploy in waves rather
than all simultaneously (doesn't change the storage plan, just the
wall-clock schedule).

### Alternative: no Network Volume available for your GPU/region

Some GPU types aren't offered in every Network-Volume-enabled datacenter —
if the console won't let you pair your chosen GPU with a volume, skip this
section and use **Container/Volume Disk only**, with stricter manual
discipline in its place. Go to §4's disk-sizing note and §16's backup
timing — both have a Volume-Disk-specific path. The short version: nothing
persists off a single pod automatically, so backups become mandatory and
time-sensitive rather than a nice-to-have, and each of the 5 pods (if
running in parallel) needs its own separate disk and its own separate
backup, since there's no shared storage between them at all.

---

## 4. Deploy the real GPU pod

1. **Pods** → **Deploy**.
2. Choose **Community Cloud** for the cheaper rate, or **Secure Cloud** for
   fewer availability/reliability hiccups — either works for this project.
3. **If using a Network Volume:** use the **Network volume filter** to
   narrow the GPU list to your region (or select your volume directly if
   the flow offers that) — this ensures compatibility. **If not:** skip
   this filter, any region works.
4. Select a GPU with **at least 32GB VRAM, 48GB strongly preferred**.
   Combined weights for both SGLang servers (Phi-4-mini ~7.6GB + Llama-3.1-8B
   ~16GB, both bf16) already total ~23.6GB before any KV cache — 24GB cards
   are **not enough** (confirmed: `run_phase4_pilot.py` needs both the agent
   *and* judge servers running together, not judge-only as earlier assumed).
   32GB (RTX 5090 $0.99/hr, RTX PRO 4500 $0.72/hr) works but is tight —
   plan to set `mem_fraction_static` conservatively (~0.35-0.4 each) and
   watch `nvidia-smi` on your first pilot run. **A40 (48GB, $0.44/hr)**
   remains the best-value comfortable choice — see `session_record.md`.
5. **Customize Deployment**:
   - **If using a Network Volume:** attach it, mount path e.g. `/workspace`.
     **Container Disk**: 100GB+ is still worth keeping even with a volume
     attached, as scratch space — see the disk-exhaustion issue in §2 for
     why undersizing this specifically (not the volume) still bites.
   - **If using Volume Disk only (no Network Volume):** there's no
     separate "attach a volume" step — just size **Container Disk**
     per what this specific pod will hold:
     - The pod running `phase4_full_sbso.yaml` (Benchmark 2): **100GB** —
       it's the only variant that goes through export (§15), where merged
       fp16 + GGUF coexist with everything else.
     - Each of the 4 ablation-only pods: **50-60GB** is enough — same base
       models, but no export-stage artifacts to hold.
     - Remember: this disk is **per pod**, not shared. Running 5 pods in
       parallel without a Network Volume means 5 separate downloads of both
       base models (~24GB each, unavoidable, but only a few minutes per pod)
       and 5 separately-sized disks per the split above.
   - **Exposed Ports**: add `30000` and `30001` (SGLang agent + judge server
     ports, per `inference.yaml`'s `launch:` block) if you want to `curl`
     them from outside the pod — optional, you can also just check from
     inside via SSH.
6. Choose **On-Demand**, not Spot — non-interruptible, per this project's
   earlier decision (see `session_record.md`). Costs more per hour than
   Spot but removes provider-side preemption risk entirely.
7. Review pricing (A40 Community Cloud is currently in the ~$0.44/hr
   range — confirm the exact number shown before deploying, rates change)
   and click **Deploy**.
8. Wait for **Running** status in **Pods**. If running 5 in parallel,
   repeat this whole section per pod — same base steps whether or not you
   have a shared volume, just different disk numbers per the branch above.

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
git clone https://github.com/tanweicheak/sumo-robot sumo-sbso
cd sumo-sbso
```

If your models aren't in the repo (they shouldn't be — too large for git),
download them now (see §8) or `scp`/`rsync` them from your Mac into
`/workspace/models/`.

---

## 7. Set up Python 3.12 and install dependencies

**Do not skip this — check your pod's default Python version first.** Plain
`pybullet` on PyPI has no prebuilt wheel for Python 3.13 (only up to `cp312`),
and Python 3.13 also changed CPython's C API in ways that can break
`pybullet`'s extension even if you force a source build. Some RunPod base
images also have multiple Python installs where `pip` and `python3` silently
point at *different* interpreters — packages "install successfully" into one
and are invisible to the other. Avoid both problems with one clean venv:

```bash
which python3.12 || (apt-get update && apt-get install -y python3.12 python3.12-venv)
# if that package isn't found on your image's default repos:
#   apt-get install -y software-properties-common
#   add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.12 python3.12-venv

python3.12 -m venv /workspace/venv
source /workspace/venv/bin/activate
python --version   # confirm 3.12.x
```

Every command below assumes this venv is activated (if you reconnect later,
re-run `source /workspace/venv/bin/activate` first).

```bash
pip install -r requirements.txt
pip install -r requirements_runpod.txt
```

**Verify, don't assume**, right after:
```bash
python -c "import pybullet; print('pybullet OK - real wheel, not a source build')"
python -c "import transformers, peft, datasets, sglang, wandb, huggingface_hub; print('all present')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```
If any of these errors, install whatever's missing explicitly before continuing.

---

## 8. Download the models

You need three model directories under `/workspace/models/` (adjust paths
to match your actual `config/inference.yaml` — these are the defaults):

```bash
mkdir -p /workspace/models
cd /workspace/models

# 1. Authenticate (required for gated models like Meta-Llama)
hf auth login

# 2. Download Phi-4-mini-instruct
hf download microsoft/Phi-4-mini-instruct --local-dir /workspace/models/phi-4-mini-hf

# 3. Download Llama-3.1-8B-Instruct
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir /workspace/models/llama-3.1-8b-instruct-hf


(Exact HF repo IDs depend on which checkpoints your report specifies —
confirm against `config/inference.yaml`'s `sglang.launch.*_model_path`
values before downloading; adjust the `huggingface-cli download` targets to
match.)

If you also want the local GGUF judge for anything (not needed for the real
pilot/full run, only for local Mac-style testing), that's a separate
download — skip it here, it's not part of the RunPod path.

---

## 9. (Optional but recommended) Log into wandb

```bash
wandb login
api key: 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
```

Paste your API key from wandb.ai/authorize. **Update:** an earlier version
of this tutorial said `--use-wandb` wasn't a real flag — that was correct
at the time, but wandb tracking has since been genuinely wired into four
scripts: `run_phase1_baselines.py` (PPO, `sync_tensorboard=True` off the
existing `tensorboard_log` param), `lora_finetune.py`/`run_phase4_stage2_token_run.py`
(LoRA, `report_to=["wandb"] if self.use_wandb else []`), `run_phase4_pilot.py`
(the real RunPod SBSO entrypoint — `wandb.init()` right after
`setup_logging()`, per-episode `wandb.log()` including live cost tracking
and GPU stats, final `wandb.log({"final/"..., "projection/"...})` +
`wandb.finish()`), and `run_phase4_stage3_local.py` (same pattern, minus
live cost tracking). Do this section now if you want tracking for §13/§14.

---

## 10. Run the static config check, one more time, on this exact machine

```bash
cd sumo-sbso
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

**Confirm the config path first — two references in this repo disagree.**
`run_phase4_pilot.py`'s own docstring says `config/training/phase4_pilot.yaml`,
and `config_loader.py`'s comment references `config/training/_shared_defaults.yaml`,
but `inference.yaml`/`stage3_local.yaml` are loaded from a flat `config/`. Resolve
it once, on this pod, before running anything:
```bash
find . -iname "phase4_pilot.yaml" -o -iname "_shared_defaults.yaml"
```
Use whatever path that actually returns in every command below and in §14.

Back in your `training` tmux session:

```bash
python -m scripts.run_phase4_pilot --config ./config/training/phase4_pilot.yaml --episodes-override 100 --use-wandb
```

(`--episodes-override 100` keeps this short and cheap — a genuine
calibration run, not the full 500. `--use-wandb` is real now — see §9 —
drop it if you skipped logging in.)

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

**Before running these:** `run_phase4_pilot.py` as originally written only
supported `phase4_pilot.yaml`'s reduced opponent scope — pointing it at
`phase4_full_sbso.yaml` or any ablation config raised
`KeyError: 'opponent_pool'/'pilot_scope'` immediately. This has been fixed
and verified merged with the wandb integration (both changes now coexist
in one file — the fix was briefly lost when the wandb code was reinstated
from a separate session, then re-applied on top and re-verified). Make sure
the version of `scripts/run_phase4_pilot.py` on this pod is the merged one
from your project notes, not an older copy from either branch — confirm
with `grep -n "self_checkpoint_manager=checkpoint_mgr" scripts/run_phase4_pilot.py`
(should return a match; empty means you have the pre-fix version). See also
the noted limitation there: self-checkpoint opponents are correctly used
during MCTS search after the patch, but still fall back to a rule-based
policy for the real match continuation — a documented, pre-existing gap,
not something this patch introduces or fixes.

One command per variant, same script, different config each time:

```bash
python -m scripts.run_phase4_pilot --config <verified-path>/phase4_full_sbso.yaml --use-wandb
python -m scripts.run_phase4_pilot --config <verified-path>/phase4_ablation_no_sa.yaml --use-wandb
python -m scripts.run_phase4_pilot --config <verified-path>/phase4_ablation_no_mcts.yaml --use-wandb
python -m scripts.run_phase4_pilot --config <verified-path>/phase4_ablation_no_dspy.yaml --use-wandb
python -m scripts.run_phase4_pilot --config <verified-path>/phase4_ablation_no_judge.yaml --use-wandb
```

Run these sequentially, not in parallel — they'd contend for the same GPU
and SGLang servers. Each writes to its own `checkpoint_output_dir` (set in
each YAML). This is the expensive part — the pilot's cost projection
(printed at the end of §13, and in your wandb dashboard's `projection/*`
fields if you logged in for §9) told you roughly what to expect for each.

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

**If using a Network Volume:** this is redundancy *on top of* it, for your
most expensive-to-regenerate artifacts specifically — run once after
training completes:
```bash
huggingface-cli login   # once, if not already
python -m scripts.backup_to_hub \
    --repo-id <your-username>/sumo-sbso-benchmark2 \
    --adapter-dir checkpoints/benchmark2_full_sbso/lora_run/lora_adapters \
    --merged-dir checkpoints/benchmark2_full_sbso/merged_fp16 \
    --gguf-dir checkpoints/benchmark2_full_sbso/gguf
```

**If using Volume Disk only (no Network Volume):** this step is not
optional redundancy — it's the *only* thing standing between your results
and permanent loss the moment a pod is terminated. Two changes from the
Network-Volume version:

- **Do it per pod, as each variant finishes** — don't wait for all 5. Each
  pod is independent; if pod 3 finishes hours before pod 5, back it up then,
  not later.
- **Include `--extra-dir` for the JSONL logs.** `backup_to_hub.py` only
  covers model artifacts by design (its own docstring: the JSONL logs are
  "better served by... Network Volume" — which you don't have here), so
  without this flag `training_pairs.jsonl`/`prompt_history.jsonl`/
  `mcts_calibration.jsonl` are lost on Terminate along with everything
  else, even though the model artifacts are safe:
  ```bash
  python -m scripts.backup_to_hub \
      --repo-id <your-username>/sumo-sbso-<variant-name> \
      --adapter-dir checkpoints/<variant>/lora_run/lora_adapters \
      --merged-dir checkpoints/<variant>/merged_fp16 \
      --gguf-dir checkpoints/<variant>/gguf \
      --extra-dir checkpoints/<variant>:jsonl_logs
  ```
  Use a **different `--repo-id` per variant** (5 separate repos, or 5
  distinct paths in one repo) so the 5 pods' backups don't overwrite each
  other.
- **Confirm the upload actually completed** (check the HF Hub repo in a
  browser, or watch for `backup_to_hub`'s own per-directory success prints)
  **before** touching that pod's Stop/Terminate button in §17. There's no
  volume to fall back on if the backup silently failed.
- Keep a simple manual checklist across the 5 pods — variant name / pod
  status / backed up Y-N — since there's no shared dashboard showing all 5
  at once the way a Network Volume gives you.

---

## 17. Shut down and stop paying

```bash
tmux kill-session -t sglang-agent
tmux kill-session -t sglang-judge
exit   # leaves the training tmux session running in background if still needed
```

Then in the console: **Pods** → your pod → **Stop** or **Terminate**.

- **If using a Network Volume:** either is fine — everything durable is on
  the volume + HF Hub backup, not the pod's local disk. Terminate freely
  once you've confirmed §16.
- **If using Volume Disk only:** **Terminate deletes this pod's disk
  permanently.** Only terminate a given pod after its §16 backup for that
  specific variant is confirmed uploaded — check the HF Hub repo, don't
  just trust the command exited without an error. If in doubt, **Stop**
  instead (keeps the disk, small storage-only charge, GPU billing paused) —
  reversible, Terminate is not.
- **Running 5 pods in parallel:** repeat this per pod as each one finishes
  its backup — don't wait for all 5 to be done before starting to shut the
  earlier-finishing ones down, that just leaves them billing idle.

Don't leave a Running pod idle either way — billing is per-second whether
you're using it or not.

---

## Troubleshooting quick-reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `KeyError: 'mcts'` on startup | Stale local copy of `_shared_defaults.yaml` predating this session's fix | Re-pull the fixed config file |
| `ImportError` for `transformers`/`peft`/`wandb`/etc. | `requirements_runpod.txt` genuinely missing a package | Add it, `pip install`, retry — see §0/§7 |
| `pip install` succeeds but `python -m scripts.preflight_static_check` still says `ModuleNotFoundError` | `pip`/`python` resolve to different interpreters (multi-Python image) | Use the venv from §7, or diagnose with `pip --version` vs `python --version` |
| `pytest: command not found` despite `pytest` installing fine | `pytest`'s console-script entry point isn't on `$PATH` | `python -m pytest` instead of bare `pytest` — doesn't need `$PATH`, only needs the package importable |
| `pybullet-3.2.7.tar.gz` downloading as source instead of a `.whl`, slow/fails to build | Wrong Python version — plain `pybullet` has no prebuilt wheel past `cp312` | Use the Python 3.12 venv from §7, not 3.13 or newer |
| SGLang server `/health` never responds | Servers still loading (can take 90s), or wrong model path | Wait, then `tmux attach -t sglang-judge` to see the real error |
| Training stopped when you closed your laptop | Ran outside `tmux` | Always `tmux new -s training` first — §5 |
| Everything gone after pod restart | No Network Volume attached, or `checkpoint_output_dir` pointed outside it | Confirm §3 was done and configs point at the mounted path |
| `preflight_runpod_check.py` fails on constrained decoding | Real load issue with the grammar/regex constraint — see pipeline report Phase 3 Critical Remarks | Check `run.log`, consider the silent-fallback instrumentation discussed earlier in this session |
| `pip install -r requirements.txt` errors `OSError: [Errno 28] No space left on device`, and afterward *everything* in that file (even `pyyaml`, `pytest`) reports `ModuleNotFoundError` | Unpinned `torch>=2.2` pulled PyPI's default CUDA build (~1 GB+ across `torch`/`nvidia_cudnn_cu13`/`nvidia_cusparselt_cu13`/`cuda_toolkit`) and filled the disk mid-download; pip resolves the whole file before installing anything, so nothing got installed, not just torch | Install `torch --index-url https://download.pytorch.org/whl/cpu` with `--no-cache-dir` before `python -m pip install --no-cache-dir -r requirements.txt` — see §2 step 7 / §7. A minimal (non-CUDA-preloaded) pod template plus these flags makes even a 20 GB Container Disk workable; if you still hit this, `df -h` after each install line (per §2 step 7) to find which package actually filled it |
