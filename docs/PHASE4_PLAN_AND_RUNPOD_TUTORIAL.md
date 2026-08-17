# Phase 4 — Current Plan + RunPod Tutorial (Beginner)

## Part A — Current file table (revised)

| # | File | What it does | Run where | Status |
|---|---|---|---|---|
| 1 | `scripts/verify_a1_a2_real_pybullet.py` | Sanity-checks core physics/MCTS fixes | Mac | Pending re-run |
| 2 | `python -m pytest tests/unit/ -v` | Full test suite | Mac | Pending re-run |
| 3 | `scripts/run_phase4_training.py` | Stage 1, mock loop | Mac | Confirmed working (earlier) |
| 4 | `scripts/run_phase4_stage2_local.py --config config/stage2_local.yaml` | Stage 2, one decision/episode, real physics | Mac | Built, needs a fresh smoke test |
| 5 | `scripts/run_phase4_stage3_local.py --config config/stage3_local.yaml` | Stage 3, full matches, real physics, local LlamaCpp, **now uses `MatchLevelSBSOTrainer`** | Mac | **Rewritten since last test — needs first real run** |
| 6 | `bash scripts/launch_sglang_servers.sh` | Starts both SGLang servers | RunPod | Never run |
| 7 | `scripts/run_phase4_pilot.py --config ... --episodes-override N` | Stage 3, full matches, real SGLang — **the one canonical script**, pilot AND full runs, ablations too | RunPod | Never run |
| 8 | `scripts/run_export_pipeline.py --config config/export_pipeline.yaml` | Merge + GGUF + GPTQ export, after training | RunPod (GPTQ) / Mac (GGUF) | Never run |

**Deleted this session, do not look for these:** `match_runner.py`, `run_phase4_cloud.py`, `scripts/stage2_wiring.py`, `run_phase4_stage3_cloud.py`.

## Part B — Revised next steps

1. Run items 1–3 on your Mac, confirm still green (many Phase 0-3 files changed since you last ran these — `sumo_env.py`, `arena.py`, `robot.py`, `opponent_analysis_agent.py`, etc.)
2. Run item 5 — **first real test of the `MatchLevelSBSOTrainer` rewrite**, small `--episodes`
3. Run item 4 similarly, small `--episodes`
4. Fill in real values in `config/stage2_local.yaml`, `config/stage3_local.yaml`, `config/export_pipeline.yaml` (still placeholders)
5. Confirm `_shared_defaults.yaml`'s DSPy trigger values are real, not `null`
6. RunPod: items 6–7 with `--episodes-override 2` (connectivity smoke test — see tutorial below)
7. RunPod: item 7 again, real `sim_budget`/`horizon`/`decision_cycles`, small `--episodes-override` (timing sample → `cost_projection.py`)
8. Full production run (item 7, no override, real config) — Benchmark 2, then the 4 ablations
9. Item 8 per trained variant
10. Finish Phase 4 audit (`ablation_strategies.py`, `cost_projection.py`, `lora_finetune.py`, `merge_adapters.py` — never read yet) — recommend doing this **before** step 8 above, not after

---

## Part C — RunPod Tutorial (beginner, full commands)

### Step 1: Account + storage (one-time)
1. Sign up at runpod.io, add credit under **Billing**.
2. **Storage → Network Volumes → Create.** 50GB. This survives pod termination — your training checkpoints live here, not on the pod itself.
3. **Settings → SSH Keys → Add** your public key (`cat ~/.ssh/id_ed25519.pub` on your Mac; generate with `ssh-keygen -t ed25519` if you don't have one).

### Step 2: Rent a pod
**Pods → Deploy.** GPU: RTX 4090. Template: a PyTorch template. Attach your Network Volume. Deploy.

### Step 3: Connect
Click **Connect** on the pod card, copy the SSH command, run it from your Mac terminal:
```bash
ssh -p <port> root@<pod-ip>
```

### Step 4: Get your code onto the pod
From your Mac, in a **new** terminal (not the SSH session):
```bash
scp -P <port> -r /Users/weicheaktan/Documents/UM/sumo_robot/source_code root@<pod-ip>:/workspace/
```
(If your repo is on GitHub/GitLab, `git clone <your-repo-url>` inside the SSH session instead — faster.)

### Step 5: Install dependencies
Back in the SSH session:
```bash
cd /workspace/source_code
pip install -r requirements.txt -r requirements-cloud.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

### Step 6: Clone llama.cpp (needed for the GGUF export branch only, not training)
```bash
cd /workspace
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && pip install -r requirements.txt
cd /workspace/source_code
```
Point `config/export_pipeline.yaml`'s `llama_cpp_convert_script` at `/workspace/llama.cpp/convert_hf_to_gguf.py`.

### Step 7: Download model weights
```bash
huggingface-cli download microsoft/Phi-4-mini-instruct --local-dir /workspace/models/phi-4-mini-hf
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir /workspace/models/llama-3.1-8b-instruct-hf
```
(Confirm these are the exact real repo names against your report before running — not independently verified this session.)

### Step 8: Start the SGLang servers (table item 6)
```bash
bash scripts/launch_sglang_servers.sh /workspace/models/phi-4-mini-hf /workspace/models/llama-3.1-8b-instruct-hf
curl http://localhost:30001/health   # wait for a healthy response, 30-90s
```

### Step 9: Connectivity smoke test (plan step 6)
```bash
python -m scripts.run_phase4_pilot --config config/phase4_full_sbso.yaml \
    --episodes-override 2 --results-dir results/smoke_test
```

### Step 10: Real-fidelity timing sample (plan step 7)
```bash
python -m scripts.run_phase4_pilot --config config/phase4_full_sbso.yaml \
    --episodes-override 3 --results-dir results/timing_sample
```
Feed the printed `PROJECTED FULL RUN` line's numbers forward.

### Step 11: Full production run (plan step 8), inside `tmux` so it survives disconnects
```bash
tmux new -s training
cd /workspace/source_code
python -m scripts.run_phase4_pilot --config config/phase4_full_sbso.yaml --results-dir results/benchmark2
```
`Ctrl+B` then `D` to detach. Reattach: `tmux attach -t training`. Repeat for each of the 4 ablation configs.

### Step 12: Export (plan step 9)
```bash
python -m scripts.run_export_pipeline --config config/export_pipeline.yaml \
    --adapter-path checkpoints/benchmark2_full_sbso/lora_adapter \
    --output-dir /workspace/export/benchmark2 \
    --calibration-texts-file checkpoints/benchmark2_full_sbso/gptq_calibration_texts.txt
```

### Step 13: Get results back, shut down
```bash
scp -P <port> -r root@<pod-ip>:/workspace/source_code/checkpoints ./local-results
scp -P <port> -r root@<pod-ip>:/workspace/export ./local-export
```
**Pods → Terminate.** Billing stops immediately; your Network Volume persists.
