"""
tests.unit.test_project_configs

Phase: Phase 0
Purpose: Smoke-test that the actual shipped config files load, resolve `extends`,
    and contain the keys downstream phases depend on. Guards against a malformed
    YAML file reaching Phase 1+ code. This is the Phase 0 exit-criterion test for
    "config/arena_config.yaml loads and validates".
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.common.config_loader import load_config, require_keys

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class TestProjectConfigs(unittest.TestCase):
    def test_project_config(self) -> None:
        cfg = load_config(CONFIG_DIR / "project_config.yaml")
        require_keys(cfg, ["project.name", "decision_window_ms", "sensor_channels"])
        self.assertEqual(cfg["decision_window_ms"], 50)
        # Exactly three sensor channels, no IMU (plan Assumption A2).
        self.assertEqual(len(cfg["sensor_channels"]), 3)
        self.assertNotIn("imu", [c.lower() for c in cfg["sensor_channels"]])

    def test_arena_config(self) -> None:
        cfg = load_config(CONFIG_DIR / "arena_config.yaml")
        require_keys(
            cfg,
            [
                "arena.radius_m",
                "arena.boundary_line_width_m",
                "robot.chassis_mass_kg",
                "lssd_thresholds.distance_cm.near_max",
            ],
            context="arena_config",
        )
        self.assertEqual(cfg["arena"]["radius_m"], 1.5)
        self.assertEqual(cfg["robot"]["chassis_mass_kg"], 1.0)

    def test_hcma_policy(self) -> None:
        cfg = load_config(CONFIG_DIR / "hcma_policy.yaml")
        require_keys(
            cfg,
            [
                "token_budget_governor.enabled",
                "token_budget_governor.min_max_new_tokens.tea",
                "emergency_bypass.trigger_consumed_ratio",
            ],
            context="hcma_policy",
        )
        # Recommended default (plan Section 6.5): governor + bypass both on.
        self.assertTrue(cfg["token_budget_governor"]["enabled"])
        self.assertTrue(cfg["emergency_bypass"]["enabled"])

    def test_all_phase4_training_configs_resolve(self) -> None:
        training_dir = CONFIG_DIR / "training"
        # episodes_total diverges by design (reduced-scope ablation study - see
        # reduced_scope_ablation_note.md): Benchmark 2 runs the full 5000-episode
        # schedule; the 4 ablations run a reduced 1800 to manage compute cost while
        # preserving the Phase 5b effect-size comparison. self_checkpoint_interval_episodes
        # was deliberately left inherited (500) for every variant, ablations included -
        # NOT reduced to match the smaller episode count - so that assertion stays flat.
        expected_episodes_total = {
            "phase4_full_sbso.yaml": 5000,
            "phase4_ablation_no_sa.yaml": 1800,
            "phase4_ablation_no_mcts.yaml": 1800,
            "phase4_ablation_no_dspy.yaml": 1800,
            "phase4_ablation_no_judge.yaml": 1800,
        }
        seen_variants = set()
        for fname, expected_total in expected_episodes_total.items():
            cfg = load_config(training_dir / fname)
            # extends: _shared_defaults.yaml must have merged in the episode budget.
            require_keys(
                cfg,
                ["variant_name", "ablation.strategy", "episodes_total", "opponent_pool.warmup_episodes"],
                context=fname,
            )
            self.assertEqual(cfg["episodes_total"], expected_total, msg=fname)
            self.assertEqual(cfg["self_checkpoint_interval_episodes"], 500, msg=fname)
            seen_variants.add(cfg["variant_name"])
        self.assertEqual(len(seen_variants), 5)  # all five variants distinct

    def test_ablation_component_flags_consistent(self) -> None:
        training_dir = CONFIG_DIR / "training"
        # Each ablation must disable exactly the component it names.
        expected = {
            "phase4_ablation_no_mcts.yaml": ("mcts", "disabled"),
            "phase4_ablation_no_dspy.yaml": ("dspy", "disabled"),
            "phase4_ablation_no_judge.yaml": ("llm_as_a_judge", "disabled"),
        }
        for fname, (component, state) in expected.items():
            cfg = load_config(training_dir / fname)
            self.assertEqual(cfg["components"][component], state, msg=fname)

    def test_phase5_eval_configs_resolve(self) -> None:
        eval_dir = CONFIG_DIR / "eval"
        a = load_config(eval_dir / "phase5a_blocks.yaml")
        self.assertEqual(a["total_matches"], 3500)
        b = load_config(eval_dir / "phase5b_ablations.yaml")
        self.assertEqual(b["total_matches"], 2000)
        self.assertEqual(len(b["scenarios"]), 4)
        c = load_config(eval_dir / "phase5c_case_5c1.yaml")
        # Phase 5c thresholds: two fixed, two descriptive (null) - plan Section 7.
        self.assertEqual(c["kpis"]["schema_validity_rate_under_noise"]["threshold_min_percent"], 99)
        self.assertEqual(c["kpis"]["latency_compliance_rate_under_noise"]["threshold_min_percent"], 95)
        self.assertIsNone(c["kpis"]["sensor_robustness_index"]["threshold"])

    def test_arena_config_has_physics_blocks(self):
        cfg = load_config(CONFIG_DIR / "arena_config.yaml")
        require_keys(
            cfg,
            [
                "robot.max_wheel_torque_nm",
                "robot.max_wheel_speed_rad_s",
                "episode.control_dt_s",
                "episode.sim_timestep_s",
                "reward_shaping.push_weight",
            ],
            context="arena_config physics",
        )
        substeps = cfg["episode"]["control_dt_s"] / cfg["episode"]["sim_timestep_s"]
        self.assertAlmostEqual(substeps, round(substeps), places=3)
        self.assertEqual(round(substeps), 12)

if __name__ == "__main__":
    unittest.main()