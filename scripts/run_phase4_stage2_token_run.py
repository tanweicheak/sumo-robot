"""
scripts.run_stage2_token_run

Phase: Phase 4 (Stage 2)
Purpose: Local token-run proving every Stage 2 code path executes for real, at tiny
    scale, before renting a GPU. Runs a few MCTS decisions against the REAL PyBullet
    physics backend (not the mock), optionally with the real Judge and real DSPy, then
    optionally a tiny real LoRA fine-tune + merge. Stops before GPTQ/GGUF export
    (cloud-only, see src/finetuning/quantize_gptq.py).

Stages (each additive, gate with flags to control cost/latency):
    1. PyBulletSimulationBackend + MockJudge   (default - fast, proves real physics wiring)
    2. + real LlamaCppJudge                    (--use-real-judge; slow, ~9s/call)
    3. + real DSPy recompilation                (--use-real-dspy; needs dspy-ai)
    4. + real LoRA fine-tune + merge             (--run-lora; needs --hf-model-path)

Examples:
    # Cheapest: real physics + real MCTS, mock judge, mock dspy. Seconds.
    python -m scripts.run_phase4_stage2_token_run --episodes 2 --sim-budget 5

    # Add the real judge (slow - expect minutes for a handful of calls).
    python -m scripts.run_phase4_stage2_token_run --episodes 1 --sim-budget 3 \
        --use-real-judge --judge-model-path models/llama-3.1-8b-instruct-Q4_K_M.gguf

    # Full local proof including a tiny real LoRA fine-tune.
    python -m scripts.run_phase4_stage2_token_run --episodes 1 --sim-budget 3 \
        --use-real-judge --judge-model-path models/llama-3.1-8b-instruct-Q4_K_M.gguf \
        --run-lora --hf-model-path models/phi-4-mini-hf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.schemas import MacroStrategy
from src.baselines.rule_based_controller import make_rule_based_policy
from src.common.config_loader import load_config
from src.sbso.ablation_strategies import AblationConfig
from src.sbso.dspy_compiler import MockDSPyCompiler
from src.sbso.judge import MockJudge
from src.sbso.macro_executor import MacroStrategyExecutor
from src.sbso.mcts import MCTS
from src.sbso.opponent_pool import OpponentPool
from src.sbso.recompilation_scheduler import RecompilationScheduler
from src.sbso.self_checkpoint_manager import SelfCheckpointManager
from src.sbso.training_loop import SBSOTrainer
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv

STRATEGIES = list(MacroStrategy)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 local token-run.")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--sim-budget", type=int, default=5)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--cycles-per-node", type=int, default=2)
    p.add_argument("--use-real-judge", action="store_true")
    p.add_argument("--judge-model-path", default=None)
    p.add_argument("--use-real-dspy", action="store_true")
    p.add_argument("--run-lora", action="store_true")
    p.add_argument("--hf-model-path", default=None)
    p.add_argument("--lora-device", default="mps")
    p.add_argument("--finetuning-config", default="config/finetuning.yaml",
                    help="Stable LoRA hyperparams (rank/alpha/target_modules/learning_rate). "
                         "epochs and device stay CLI-controlled for this token-run.")
    p.add_argument("--output-dir", default="checkpoints/stage2_token_run")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[stage2] building real PyBullet env (rule-based opponent)...")
    env_cfg = EnvConfig.from_config(use_gui=False, enable_reward_shaping=False)
    env = PyBulletSumoEnv(env_config=env_cfg, opponent_policy=make_rule_based_policy())
    env.reset()

    executor = MacroStrategyExecutor()

    if args.use_real_judge:
        if not args.judge_model_path:
            raise SystemExit("--use-real-judge requires --judge-model-path")
        print(f"[stage2] loading REAL judge from {args.judge_model_path} (slow, be patient)...")
        from src.sbso.judge import LlamaCppJudge
        judge = LlamaCppJudge(model_path=args.judge_model_path)
    else:
        print("[stage2] using MockJudge (fast; pass --use-real-judge for the real thing)")
        judge = MockJudge(seed=0)

    from src.sbso.simulation_backend import PyBulletSimulationBackend
    backend = PyBulletSimulationBackend(
        env=env, executor=executor, judge=judge, cycles_per_node=args.cycles_per_node,
    )

    mcts = MCTS(
        backend, STRATEGIES, sim_budget=args.sim_budget, horizon=args.horizon,
        judge_prune_threshold=0.3,
    )

    if args.use_real_dspy:
        if not args.hf_model_path:
            raise SystemExit("--use-real-dspy needs a llama.cpp GGUF path via --judge-model-path's "
                             "sibling flag is not required here; pass the SA model via config if needed.")
        print("[stage2] using REAL DSPy compiler (needs `dspy-ai` installed)...")
        from src.sbso.dspy_compiler import RealDSPyCompiler
        # Reuses the same GGUF the TEA/SA agents run on (Phase 3 model), not the judge model.
        dspy_compiler = RealDSPyCompiler(llama_model_path="models/phi-4-mini-Q4_K_M.gguf")
    else:
        print("[stage2] using MockDSPyCompiler (fast; pass --use-real-dspy for the real thing)")
        dspy_compiler = MockDSPyCompiler()

    def root_state_builder(episode: int, opponent: str):
        # Build LSSD text from the env's CURRENT live sensor state, then snapshot physics.
        obs = env.agent_sensors.read()
        from src.data.lssd_encoder import LSSDEncoder
        lssd = LSSDEncoder.from_config()
        ego = {"fwd": 0.0, "turn": 0.0}  # neutral; real ego-motion comes from PA in full pipeline
        enc = lssd.encode(obs["tof"], approach_rate=0.0, ego=ego)
        return backend.root_state(enc["lssd_text"], opponent_behavior="unknown")

    trainer = SBSOTrainer(
        ablation=AblationConfig(),
        mcts=mcts,
        opponent_pool=OpponentPool(warmup_episodes=max(1, args.episodes // 2), total_episodes=args.episodes),
        scheduler=RecompilationScheduler(k_episodes=max(1, args.episodes // 2), window_w=max(2, args.episodes), delta=0.05),
        checkpoint_mgr=SelfCheckpointManager(interval=max(1, args.episodes // 2)),
        dspy_compiler=dspy_compiler,
        strategies=STRATEGIES,
        episodes=args.episodes,
        root_state_builder=root_state_builder,
    )

    print(f"[stage2] running {args.episodes} episode(s), sim_budget={args.sim_budget}, "
          f"horizon={args.horizon}, cycles_per_node={args.cycles_per_node}...")
    summary = trainer.run()
    print(f"[stage2] SBSO loop summary: {summary}")
    env.close()

    if not args.run_lora:
        print("[stage2] --run-lora not set; stopping after training-pair collection.")
        print("[stage2] STAGE 2 (MCTS/Judge/DSPy plumbing) TOKEN-RUN COMPLETE.")
        return

    if not args.hf_model_path:
        raise SystemExit("--run-lora requires --hf-model-path (a HuggingFace-format Phi-4-mini "
                         "checkpoint, NOT the GGUF - see config/finetuning.yaml).")

    print("[stage2] building SFT dataset from collected training pairs...")
    from src.finetuning.lora_finetune import LoRAFineTuner, build_sft_dataset

    sft_records = build_sft_dataset(
        trainer.training_pairs, prompt_template="State: {state}\nStrategy:",
    )
    print(f"[stage2] {len(sft_records)} SFT record(s). Running a tiny real LoRA fine-tune "
          f"(device={args.lora_device})...")

    ft_config = load_config(args.finetuning_config)
    lora_cfg = ft_config.get("lora", {})
    tuner = LoRAFineTuner(
        base_model_path=args.hf_model_path, output_dir=out_dir / "lora_run",
        rank=int(lora_cfg.get("rank", 16)),
        alpha=int(lora_cfg.get("alpha", 32)),
        target_modules=lora_cfg.get("target_modules"),
        learning_rate=float(lora_cfg.get("learning_rate", 2e-4)),
        epochs=1, device=args.lora_device,
    )
    adapter_path = tuner.run(sft_records)
    print(f"[stage2] LoRA adapters saved -> {adapter_path}")

    print("[stage2] merging LoRA adapters into base model (fp16)...")
    from src.finetuning.merge_adapters import merge_lora_adapters

    merged_path = merge_lora_adapters(
        base_model_path=args.hf_model_path, adapter_path=adapter_path,
        output_dir=out_dir / "merged_fp16",
    )
    print(f"[stage2] merged model saved -> {merged_path}")
    print("[stage2] STOPPING before GPTQ quantization / GGUF export - those are CLOUD-ONLY.")
    print("[stage2] FULL STAGE 2 TOKEN-RUN COMPLETE: real physics, real training-pair "
          "collection, real LoRA fine-tune, real merge - all proven end-to-end.")


if __name__ == "__main__":
    main()