"""
src.simulation.sumo_env

Phase: Phase 1
Purpose: PyBulletSumoEnv - a Gymnasium environment for one-on-one autonomous sumo.
    Controls the agent robot; the opponent is driven by an injectable policy callable
    (rule-based / PPO / self-checkpoint in later phases). One env step is one 50 ms
    decision cycle (12 PyBullet substeps at 1/240 s). Observation is the agent's raw
    three-channel sensor snapshot; termination is decided by ring-boundary tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from src.common.config_loader import load_config
from src.simulation.arena import Dohyo
from src.simulation.robot import RobotSpec, SumoRobot, build_robot
from src.simulation.sensors import SensorSpec, SensorSuite

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ARENA_CONFIG = _REPO_ROOT / "config" / "arena_config.yaml"

# Opponent policy: maps the opponent's own observation dict to (left_pwm, right_pwm).
OpponentPolicy = Callable[[dict[str, np.ndarray]], tuple[float, float]]


def _idle_opponent(_obs: dict[str, np.ndarray]) -> tuple[float, float]:
    return 0.0, 0.0


@dataclass
class EnvConfig:
    sim_timestep_s: float = 1.0 / 240.0
    control_dt_s: float = 0.05            # 50 ms decision window
    max_episode_seconds: float = 15.0     # per report: avg match ~15 s
    spawn_offset_m: float = 0.6           # each robot this far from center
    use_gui: bool = False
    seed: int | None = None
    # Optional dense shaping (privileged geometry; policy observation stays sensor-only).
    enable_reward_shaping: bool = False
    shaping_contact_dist_m: float = 0.30    # distance below which robots count as in contact
    shaping_fwd_weight: float = 0.03        # reward forward motion
    shaping_push_weight: float = 0.50       # reward front-on pushing
    shaping_edge_bonus: float = 0.10        # bonus for pushing opponent toward rim
    shaping_track_weight: float = 0.02      # reward facing/approaching opponent
    shaping_glance_penalty: float = 0.15    # penalty for side/glancing contact
    shaping_spin_penalty: float = 0.10      # penalty for spinning without progress
    shaping_selfeject_penalty: float = 0.40 # penalty for facing own edge near rim
    shaping_time_penalty: float = 0.03      # per-step time cost

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = _DEFAULT_ARENA_CONFIG,
        *,
        use_gui: bool = False,
        enable_reward_shaping: bool = False,
        seed: int | None = None,
    ) -> "EnvConfig":
        cfg: dict[str, Any] = load_config(config_path)
        ep = cfg["episode"]
        shaping = cfg.get("reward_shaping", {})
        return cls(
            sim_timestep_s=float(ep["sim_timestep_s"]),
            control_dt_s=float(ep["control_dt_s"]),
            max_episode_seconds=float(ep["max_episode_seconds"]),
            spawn_offset_m=float(ep["spawn_offset_m"]),
            use_gui=use_gui,
            seed=seed,
            enable_reward_shaping=enable_reward_shaping,
            shaping_contact_dist_m=float(shaping.get("contact_dist_m", 0.30)),
            shaping_fwd_weight=float(shaping.get("fwd_weight", 0.03)),
            shaping_push_weight=float(shaping.get("push_weight", 0.50)),
            shaping_edge_bonus=float(shaping.get("edge_bonus", 0.10)),
            shaping_track_weight=float(shaping.get("track_weight", 0.02)),
            shaping_glance_penalty=float(shaping.get("glance_penalty", 0.15)),
            shaping_spin_penalty=float(shaping.get("spin_penalty", 0.10)),
            shaping_selfeject_penalty=float(shaping.get("selfeject_penalty", 0.40)),
            shaping_time_penalty=float(shaping.get("time_penalty", 0.03)),
        )


class PyBulletSumoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        arena_config_path: str | None = None,
        opponent_policy: OpponentPolicy | None = None,
    ) -> None:
        super().__init__()
        self.cfg = env_config or EnvConfig()
        self.arena_config_path = arena_config_path
        self.opponent_policy = opponent_policy or _idle_opponent

        self.substeps = max(1, round(self.cfg.control_dt_s / self.cfg.sim_timestep_s))
        self.max_steps = round(self.cfg.max_episode_seconds / self.cfg.control_dt_s)

        self.client_id: int = -1
        self.dohyo: Dohyo | None = None
        self.agent: SumoRobot | None = None
        self.opponent: SumoRobot | None = None
        self.agent_sensors: SensorSuite | None = None
        self.opponent_sensors: SensorSuite | None = None
        self._step_count = 0

        sensor_spec = SensorSpec()
        self.observation_space = spaces.Dict(
            {
                "tof": spaces.Box(0.0, sensor_spec.tof_max_range_m, (sensor_spec.tof_num_rays,), np.float32),
                "ir": spaces.Box(0.0, 1.0, (2,), np.float32),
                "encoder": spaces.Box(-np.inf, np.inf, (4,), np.float32),
            }
        )
        # Action: normalized left/right PWM.
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

    # -- lifecycle -----------------------------------------------------------

    def _connect(self) -> None:
        if self.client_id >= 0:
            return
        mode = p.GUI if self.cfg.use_gui else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed if seed is not None else self.cfg.seed)
        self._connect()
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.cfg.sim_timestep_s, physicsClientId=self.client_id)

        self.dohyo = (
            Dohyo.from_config(self.client_id, self.arena_config_path)
            if self.arena_config_path
            else Dohyo.from_config(self.client_id)
        )
        self.dohyo.build()

        spawn_z = self.dohyo.spec.platform_top_z + 0.12
        spec_source = self.arena_config_path or _DEFAULT_ARENA_CONFIG
        agent_spec = RobotSpec.from_config(spec_source)
        opp_spec = RobotSpec.from_config(spec_source)

        # Agent at -x facing +x (yaw 0); opponent at +x facing -x (yaw pi). They face off.
        self.agent = build_robot(
            self.client_id, agent_spec,
            base_position=(-self.cfg.spawn_offset_m, 0.0, spawn_z),
            base_yaw_rad=0.0, color=(0.2, 0.5, 0.9, 1.0),
        )
        self.opponent = build_robot(
            self.client_id, opp_spec,
            base_position=(self.cfg.spawn_offset_m, 0.0, spawn_z),
            base_yaw_rad=math.pi, color=(0.9, 0.4, 0.2, 1.0),
        )

        self.agent_sensors = SensorSuite(
            self.agent, self.client_id, self_body_id=self.agent.body_id,
            spec=SensorSpec(chassis_half_extents_z=agent_spec.chassis_half_extents[2]),
        )
        self.opponent_sensors = SensorSuite(
            self.opponent, self.client_id, self_body_id=self.opponent.body_id,
            spec=SensorSpec(chassis_half_extents_z=opp_spec.chassis_half_extents[2]),
        )

        # Settle onto the platform, then zero velocities so the first sensor read is
        # taken from rest (avoids a spurious edge-trigger from spawn-drop bounce).
        for _ in range(120):
            p.stepSimulation(physicsClientId=self.client_id)
        for body in (self.agent.body_id, self.opponent.body_id):
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0], physicsClientId=self.client_id)
        self.agent.apply_pwm(0.0, 0.0)
        self.opponent.apply_pwm(0.0, 0.0)

        self._step_count = 0
        obs = self.agent_sensors.read()
        return obs, {}

    # -- stepping ------------------------------------------------------------

    def step(self, action: np.ndarray):
        assert self.agent and self.opponent and self.agent_sensors and self.opponent_sensors and self.dohyo

        # Agent drive.
        a_left, a_right = float(action[0]), float(action[1])
        self.agent.apply_pwm(a_left, a_right)

        # Opponent decision + drive. A commanded (0, 0) means "no drive" -> coast
        # (zero motor force) so the opponent can be pushed rather than brake-anchored.
        opp_obs = self.opponent_sensors.read()
        o_left, o_right = self.opponent_policy(opp_obs)
        if abs(o_left) < 1e-9 and abs(o_right) < 1e-9:
            self.opponent.coast()
        else:
            self.opponent.apply_pwm(o_left, o_right)

        # Advance one 50 ms decision cycle.
        for _ in range(self.substeps):
            p.stepSimulation(physicsClientId=self.client_id)

        self._step_count += 1
        obs = self.agent_sensors.read()

        agent_pos, agent_orn = self.agent.base_pose()
        opp_pos, opp_orn = self.opponent.base_pose()

        def _out_reason(pos, orn) -> str | None:
            if self.dohyo.is_outside(pos):
                return "pushed_out"
            if self.dohyo.has_fallen(pos):
                return "fell_off_edge"
            if self.dohyo.has_capsized(orn):
                return "capsized"
            return None

        agent_out_reason = _out_reason(agent_pos, agent_orn)
        opp_out_reason = _out_reason(opp_pos, opp_orn)
        agent_out = agent_out_reason is not None
        opp_out = opp_out_reason is not None

        terminated = agent_out or opp_out
        truncated = self._step_count >= self.max_steps

        reward, outcome = self._compute_reward(agent_pos, opp_pos, agent_out, opp_out, truncated)
        info = {
            "outcome": outcome, "agent_pos": agent_pos, "opponent_pos": opp_pos, "steps": self._step_count,
            "agent_out_reason": agent_out_reason, "opponent_out_reason": opp_out_reason,
        }
        return obs, reward, terminated, truncated, info
    def _compute_reward(self, agent_pos, opp_pos, agent_out, opp_out, truncated):
        # Terminal outcomes dominate (large magnitude vs. per-step shaping).
        if agent_out and opp_out:
            return 0.0, "draw"
        if opp_out:
            return 10.0, "win"
        if agent_out:
            return -10.0, "loss"
        if truncated:
            return 0.0, "draw"

        if not self.cfg.enable_reward_shaping:
            return 0.0, "ongoing"

        # --- Privileged geometry (from ground-truth poses; NOT seen by the policy) ---
        _, agent_orn = self.agent.base_pose()
        agent_yaw = p.getEulerFromQuaternion(agent_orn)[2]
        # Agent forward unit vector (local +x in world frame).
        fwd = np.array([math.cos(agent_yaw), math.sin(agent_yaw)])

        to_opp = np.array([opp_pos[0] - agent_pos[0], opp_pos[1] - agent_pos[1]])
        dist_opp = float(np.linalg.norm(to_opp))
        to_opp_unit = to_opp / (dist_opp + 1e-8)
        cos_to_opp = float(np.dot(fwd, to_opp_unit))  # 1 = facing opponent dead-on

        agent_r = math.hypot(agent_pos[0], agent_pos[1])
        opp_r = math.hypot(opp_pos[0], opp_pos[1])
        ring_r = self.dohyo.spec.radius_m
        agent_edge_norm = max(0.0, (ring_r - agent_r) / ring_r)  # 1 center, 0 at rim
        # Agent facing toward center? cos between forward and (-position) direction.
        to_center = np.array([-agent_pos[0], -agent_pos[1]])
        to_center_unit = to_center / (agent_r + 1e-8)
        cos_to_center = float(np.dot(fwd, to_center_unit))

        # Forward speed along the chassis heading.
        lin_vel, _ = self.agent.base_velocity()
        v_world = np.array([lin_vel[0], lin_vel[1]])
        v_fwd = float(np.dot(v_world, fwd))

        in_contact = dist_opp < self.cfg.shaping_contact_dist_m

        r = 0.0
        p_ = self.cfg  # shaping params

        # Reward moving forward at all (discourage passive stalling / draws).
        if v_fwd > 0.2:
            r += p_.shaping_fwd_weight * v_fwd

        if in_contact:
            if cos_to_opp < 0.6:
                # Glancing / side contact -> anti-dancing penalty.
                r -= p_.shaping_glance_penalty
            else:
                # Front-on push: reward driving forward into the opponent.
                r += p_.shaping_push_weight * max(0.0, v_fwd)
                # Extra reward when opponent is near the rim and we're aligned.
                if opp_r > 0.5 * ring_r and cos_to_opp > 0.9:
                    r += p_.shaping_edge_bonus * (opp_r / ring_r)
        else:
            # Not in contact: reward facing/approaching the opponent.
            if cos_to_opp > 0.8:
                r += p_.shaping_track_weight * cos_to_opp

        # Penalize spinning in place (rotation without forward progress).
        _, ang_vel = self.agent.base_velocity()
        omega = abs(ang_vel[2])
        if omega > 0.5 and v_fwd < 0.2:
            r -= p_.shaping_spin_penalty * omega

        # Anti-self-ejection: near the rim AND facing outward -> strong penalty.
        if agent_edge_norm < 0.25:
            danger = (0.25 - agent_edge_norm) / 0.25
            if cos_to_center < 0.0:
                r -= p_.shaping_selfeject_penalty * danger

        # Small time penalty to encourage decisive victories over stalling.
        r -= p_.shaping_time_penalty

        return float(r), "ongoing"

    def render(self):
        # GUI mode is selected via EnvConfig.use_gui at construction time.
        return None

    def close(self):
        if self.client_id >= 0:
            p.disconnect(physicsClientId=self.client_id)
            self.client_id = -1