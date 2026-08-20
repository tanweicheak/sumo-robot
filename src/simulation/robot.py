"""
src.simulation.robot

Phase: Phase 1
Purpose: Differential (4-wheel skid-steer) sumo robot chassis, 1.0 kg total, built
    programmatically via createMultiBody. Normalized PWM (left/right) maps to wheel
    angular velocity under torque-limited velocity control, so collisions produce
    genuine frictional push. No caster needed: four driven wheels keep it upright.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pybullet as p

from src.common.config_loader import load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ARENA_CONFIG = _REPO_ROOT / "config" / "arena_config.yaml"

@dataclass
class RobotSpec:
    # NOTE: total_mass_kg is the WHOLE robot (chassis body + all 4 wheels), even though
    # it's read from config's `chassis_mass_kg` key - the key name is a bit misleading
    # in isolation, but build_robot()'s base_mass = total_mass_kg - 4*wheel_mass_kg makes
    # the actual semantics unambiguous in code.
    total_mass_kg: float = 1.0
    chassis_half_extents: tuple[float, float, float] = (0.10, 0.10, 0.04)
    wheel_radius_m: float = 0.04
    wheel_width_m: float = 0.02
    wheel_mass_kg: float = 0.10          # x4 = 0.4 kg; base carries the remainder
    max_wheel_speed_rad_s: float = 20.0  # pilot-calibrated
    max_wheel_torque_nm: float = 5.0     # pilot-calibrated; governs push strength
    wheel_lateral_friction: float = 1.2
    chassis_lateral_friction: float = 0.4

    @classmethod
    def from_config(cls, config_path: str | Path = _DEFAULT_ARENA_CONFIG) -> "RobotSpec":
        cfg: dict[str, Any] = load_config(config_path)
        r = cfg["robot"]
        return cls(
            total_mass_kg=float(r["chassis_mass_kg"]),
            chassis_half_extents=tuple(r["chassis_half_extents_m"]),
            wheel_radius_m=float(r["wheel_radius_m"]),
            wheel_width_m=float(r["wheel_width_m"]),
            wheel_mass_kg=float(r["wheel_mass_kg"]),
            max_wheel_speed_rad_s=float(r["max_wheel_speed_rad_s"]),
            max_wheel_torque_nm=float(r["max_wheel_torque_nm"]),
            wheel_lateral_friction=float(r["wheel_lateral_friction"]),
            chassis_lateral_friction=float(r["chassis_lateral_friction"]),
        )


@dataclass
class SumoRobot:
    body_id: int
    spec: RobotSpec
    client_id: int
    left_wheel_joints: list[int] = field(default_factory=list)
    right_wheel_joints: list[int] = field(default_factory=list)

    def apply_pwm(self, left_pwm: float, right_pwm: float) -> None:
        """Drive both wheel banks. PWM is clamped to [-1, 1]."""
        left = max(-1.0, min(1.0, left_pwm)) * self.spec.max_wheel_speed_rad_s
        right = max(-1.0, min(1.0, right_pwm)) * self.spec.max_wheel_speed_rad_s
        for joint in self.left_wheel_joints:
            p.setJointMotorControl2(
                self.body_id, joint, p.VELOCITY_CONTROL,
                targetVelocity=left, force=self.spec.max_wheel_torque_nm,
                physicsClientId=self.client_id,
            )
        for joint in self.right_wheel_joints:
            p.setJointMotorControl2(
                self.body_id, joint, p.VELOCITY_CONTROL,
                targetVelocity=right, force=self.spec.max_wheel_torque_nm,
                physicsClientId=self.client_id,
            )

    def coast(self) -> None:
        """Release the drive motors so the robot can be pushed freely (true idle).
        Uses velocity control with zero force rather than holding position, so a
        stationary robot offers only wheel/chassis friction, not active braking."""
        for joint in self.left_wheel_joints + self.right_wheel_joints:
            p.setJointMotorControl2(
                self.body_id, joint, p.VELOCITY_CONTROL,
                targetVelocity=0.0, force=0.0,
                physicsClientId=self.client_id,
            )

    def base_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        pos, orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.client_id)
        return pos, orn

    def encoder_state(self) -> dict[str, float]:
        """Average wheel position/velocity per bank (motor-encoder channel)."""
        def bank(joints: list[int]) -> tuple[float, float]:
            pos = vel = 0.0
            for j in joints:
                state = p.getJointState(self.body_id, j, physicsClientId=self.client_id)
                pos += state[0]
                vel += state[1]
            n = max(1, len(joints))
            return pos / n, vel / n
        lp, lv = bank(self.left_wheel_joints)
        rp, rv = bank(self.right_wheel_joints)
        return {"left_pos": lp, "left_vel": lv, "right_pos": rp, "right_vel": rv}

    def base_velocity(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return (linear_velocity, angular_velocity) of the chassis base, world frame."""
        lin, ang = p.getBaseVelocity(self.body_id, physicsClientId=self.client_id)
        return lin, ang


def build_robot(
    client_id: int,
    spec: RobotSpec,
    base_position: tuple[float, float, float],
    base_yaw_rad: float,
    color: tuple[float, float, float, float],
) -> SumoRobot:
    """Construct a 4-wheel skid-steer robot and return a SumoRobot handle."""
    hx, hy, hz = spec.chassis_half_extents
    base_mass = spec.total_mass_kg - 4 * spec.wheel_mass_kg

    chassis_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=client_id)
    chassis_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=list(color), physicsClientId=client_id)

    wheel_col = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=spec.wheel_radius_m, height=spec.wheel_width_m, physicsClientId=client_id
    )
    wheel_vis = p.createVisualShape(
        p.GEOM_CYLINDER, radius=spec.wheel_radius_m, length=spec.wheel_width_m,
        rgbaColor=[0.1, 0.1, 0.1, 1.0], physicsClientId=client_id,
    )

    # Cylinder default axis is local Z; orient each wheel so local Z lies along the
    # chassis Y (the axle), then spin about jointAxis = local Z.
    wheel_orn = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
    wheel_z = -hz  # wheels at chassis underside
    # (x, y) mounts: front/back x = +/-0.07, left/right y = +/-(hy + 0.01)
    mounts = [
        (0.07, hy + 0.01, wheel_z),    # front-left  -> left bank
        (0.07, -(hy + 0.01), wheel_z),  # front-right -> right bank
        (-0.07, hy + 0.01, wheel_z),   # back-left   -> left bank
        (-0.07, -(hy + 0.01), wheel_z),  # back-right -> right bank
    ]

    n = len(mounts)
    body_id = p.createMultiBody(
        baseMass=base_mass,
        baseCollisionShapeIndex=chassis_col,
        baseVisualShapeIndex=chassis_vis,
        basePosition=list(base_position),
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, base_yaw_rad]),
        linkMasses=[spec.wheel_mass_kg] * n,
        linkCollisionShapeIndices=[wheel_col] * n,
        linkVisualShapeIndices=[wheel_vis] * n,
        linkPositions=[list(m) for m in mounts],
        linkOrientations=[list(wheel_orn)] * n,
        linkInertialFramePositions=[[0, 0, 0]] * n,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
        linkParentIndices=[0] * n,               # all parented to the base
        linkJointTypes=[p.JOINT_REVOLUTE] * n,
        linkJointAxis=[[0, 0, -1]] * n,           # spin about the (reoriented) wheel axle
        physicsClientId=client_id,
    )

    # left_wheel_joints/right_wheel_joints are assigned by CONTROL-CONVENTION
    # ("which bank does apply_pwm's left_pwm arg drive"), not by mount geometry.
    # tests/unit/test_wheel_turn_direction.py empirically measured real yaw under
    # PyBulletSumoEnv's actual reset()/step() path (platform + settle + gravity,
    # not a floating robot) and found apply_pwm(left=-1, right=+1) produces a
    # strong, consistent, clearly-signed yaw in the OPPOSITE direction every
    # steering consumer (rule_based_controller.py's attack lead term,
    # adversarial_bait_controller.py's lure centering steer, any future
    # TacticalCommand PWM output) assumes. Straight-line motion (symmetric PWM)
    # is unaffected by which physical mount is labeled "left" vs "right", so this
    # swap is the correct, minimal fix - it does not touch jointAxis, wheel_orn,
    # or any caller.
    left_joints = [1, 3]   # front-right, back-right mounts -> drives as the LEFT PWM bank
    right_joints = [0, 2]  # front-left, back-left mounts -> drives as the RIGHT PWM bank

    for j in range(n):
        p.changeDynamics(
            body_id, j, lateralFriction=spec.wheel_lateral_friction, physicsClientId=client_id
        )
    p.changeDynamics(
        body_id, -1, lateralFriction=spec.chassis_lateral_friction, physicsClientId=client_id
    )

    robot = SumoRobot(
        body_id=body_id, spec=spec, client_id=client_id,
        left_wheel_joints=left_joints, right_wheel_joints=right_joints,
    )
    # Initialize motors to zero so the robot holds still until commanded.
    robot.apply_pwm(0.0, 0.0)
    return robot