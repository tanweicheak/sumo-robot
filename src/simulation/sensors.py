"""
src.simulation.sensors

Phase: Phase 1
Purpose: Simulated sensor emission for one robot: exactly three channels, no IMU.
    - ToF (distance/object): forward ray fan detecting the opponent / obstacles.
    - IR (edge): downward rays at the front corners; miss = over the ring edge.
    - Motor encoder: wheel joint position/velocity.
    Emits raw readings only; LSSD semantic mapping happens in the Perception Agent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pybullet as p

from src.simulation.robot import SumoRobot

@dataclass
class SensorSpec:
    tof_num_rays: int = 7
    tof_fan_deg: float = 60.0
    tof_max_range_m: float = 1.5
    tof_mount_forward_m: float = 0.11
    tof_mount_height_m: float = 0.02
    ir_max_range_m: float = 0.30
    ir_mount_forward_m: float = 0.13
    ir_mount_side_m: float = 0.08
    chassis_half_extents_z: float = 0.04   # must match RobotSpec chassis z half-extent


def _local_to_world(
    base_pos: tuple[float, float, float],
    base_orn: tuple[float, float, float, float],
    local: tuple[float, float, float],
) -> np.ndarray:
    rot = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
    return np.array(base_pos) + rot @ np.array(local)


def _local_dir_to_world(base_orn: tuple[float, float, float, float], local_dir: tuple[float, float, float]) -> np.ndarray:
    rot = np.array(p.getMatrixFromQuaternion(base_orn)).reshape(3, 3)
    return rot @ np.array(local_dir)


@dataclass
class SensorSuite:
    robot: SumoRobot
    client_id: int
    spec: SensorSpec = field(default_factory=SensorSpec)
    self_body_id: int = -1

    def read(self, debug_draw: bool = False) -> dict[str, np.ndarray]:
        base_pos, base_orn = self.robot.base_pose()
        return {
            "tof": self._read_tof(base_pos, base_orn, debug_draw),
            "ir": self._read_ir(base_pos, base_orn, debug_draw),
            "encoder": self._read_encoder(),
        }

    def _read_tof(self, base_pos, base_orn, debug_draw: bool = False) -> np.ndarray:
        origin = _local_to_world(
            base_pos, base_orn,
            (self.spec.tof_mount_forward_m, 0.0, self.spec.tof_mount_height_m),
        )
        half = math.radians(self.spec.tof_fan_deg) / 2.0
        n = self.spec.tof_num_rays
        angles = np.linspace(-half, half, n) if n > 1 else np.array([0.0])

        froms, tos = [], []
        for a in angles:
            local_dir = (math.cos(a), math.sin(a), 0.0)
            world_dir = _local_dir_to_world(base_orn, local_dir)
            froms.append(origin.tolist())
            tos.append((origin + world_dir * self.spec.tof_max_range_m).tolist())

        results = p.rayTestBatch(froms, tos, physicsClientId=self.client_id)
        distances = np.full(n, self.spec.tof_max_range_m, dtype=np.float32)
        for i, r in enumerate(results):
            hit_id, hit_fraction = r[0], r[2]
            if hit_id >= 0 and hit_id != self.self_body_id:
                distances[i] = hit_fraction * self.spec.tof_max_range_m
            if debug_draw:
                hit = hit_id >= 0 and hit_id != self.self_body_id
                color = [1.0, 0.0, 0.0] if hit else [0.0, 1.0, 1.0]
                p.addUserDebugLine(froms[i], tos[i], color, lineWidth=1.5,
                                   lifeTime=0.06, physicsClientId=self.client_id)
        return distances

    def _read_ir(self, base_pos, base_orn, debug_draw: bool = False) -> np.ndarray:
        # Origin sits just BELOW the chassis bottom face so the downward ray skips the
        # robot's own hull and its first hit is the platform (on-ring) or nothing/void
        # (over the edge). Casting from above fails: the ray hits the chassis first and
        # the self-hit filter discards it, leaving no platform hit.
        mount_drop = -(self.spec.chassis_half_extents_z + 0.005)  # ~ -0.045 (below hull)
        probe_len = 0.12  # from ~z=0.135 down past the platform top (0.10)
        mounts = [
            (self.spec.ir_mount_forward_m, self.spec.ir_mount_side_m, mount_drop),
            (self.spec.ir_mount_forward_m, -self.spec.ir_mount_side_m, mount_drop),
        ]
        froms, tos = [], []
        for m in mounts:
            origin = _local_to_world(base_pos, base_orn, m)
            froms.append(origin.tolist())
            tos.append((origin + np.array([0.0, 0.0, -probe_len])).tolist())

        results = p.rayTestBatch(froms, tos, physicsClientId=self.client_id)
        # Default 1.0 = "over the edge" (no floor beneath). On-ring hits overwrite it
        # with the hit fraction (~0.29 at rest), which is well below the edge threshold.
        readings = np.ones(len(mounts), dtype=np.float32)
        for i, r in enumerate(results):
            hit_id, hit_fraction = r[0], r[2]
            if hit_id >= 0 and hit_id != self.self_body_id:
                readings[i] = np.float32(hit_fraction)
            if debug_draw:
                p.addUserDebugLine(froms[i], tos[i], [1.0, 0.0, 1.0], lineWidth=1.5,
                                   lifeTime=0.06, physicsClientId=self.client_id)
        return readings

    def _read_encoder(self) -> np.ndarray:
        e = self.robot.encoder_state()
        return np.array(
            [e["left_vel"], e["right_vel"], e["left_pos"], e["right_pos"]],
            dtype=np.float32,
        )