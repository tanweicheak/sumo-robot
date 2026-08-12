# Architecture Diagram (provisional)

This is a prose/ASCII reconstruction of the intended system architecture (report.md
Diagram 1). It is **provisional**: the original `research_phase.pdf` was not available
when this was written, so verify against it once re-attached.

## Decision cycle (Benchmark 1 / Benchmark 2)

```
                          +---------------------------+
   raw sensor arrays      |  Perception Agent (PA)    |
   (ToF / IR / encoder) ->|  deterministic LSSD map   |-> semantic state text
                          +---------------------------+
                                       |
                                       v
                          +---------------------------+
                          |  Opponent Analysis (OAA)  |  Phi-4-mini SLM
                          |  -> OpponentBehavior      |
                          +---------------------------+
                                       |
                       (No-SA ablation | bypasses SA)
                                       v
                          +---------------------------+
                          |  Strategy Agent (SA)      |  Phi-4-mini SLM
                          |  -> MacroStrategy         |
                          +---------------------------+
                                       |
                                       v
                          +---------------------------+
                          |  Tactical Execution (TEA) |  Phi-4-mini SLM
                          |  Outlines-constrained     |
                          |  -> TacticalCommand JSON  |
                          +---------------------------+
                                       |
                                       v
                          +---------------------------+
                          |  Actuator Bridge          |  deterministic
                          |  -> PWM (left/right)      |
                          +---------------------------+
                                       |
                                       v
                                 PyBullet sim

   HCMA (deterministic) subscribes to the HEL log stream out-of-band and adjusts
   TEA's token budget / triggers emergency SA-bypass to protect the 50ms deadline.
```

Baseline 3 collapses OAA + SA + TEA into a single Monolithic Reasoning Agent (MRA):
one Phi-4-mini call replaces the three specialized calls. PA and the Actuator Bridge
are identical across all three architectures.

## Offline SBSO training loop (Phase 4, Benchmark 2)

```
   MCTS rollout generation
        |  (state-macro-strategy pairs)
        v
   LLM-as-a-Judge filtering  ----(prune low-quality branches)
        |
        v
   DSPy recompilation  <---- triggered every K batches OR on reward drop > delta
        |  (surviving high-reward trajectories -> refined prompt program)
        v
   accumulate prompt-and-tactic pairs across the full run
        |
        v
   LoRA fine-tuning -> merge (fp16) -> GPTQ INT4 -> GGUF export
        |
        v
   one frozen checkpoint (Benchmark 2, or one ablation variant)
```

Ablations remove exactly one component: No-SA (OAA feeds TEA directly), No-MCTS
(single-pass direct policy sampling), No-DSPy (prompt frozen at initial state),
No-Judge (branches unfiltered).
