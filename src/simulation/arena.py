"""
src.simulation.arena

Phase: Phase 1
Purpose: Dohyo geometry (radius, boundary line) and boundary-crossing detection. Reads config/arena_config.yaml.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pybullet as p
import pybullet_data

from src.common.config_loader import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARENA_CONFIG = REPO_ROOT / "config" / "arena_config.yaml"


@dataclass
class DohyoSpec:
    radius_m: float
    boundary_line_width_m: float
    platform_thickness_m: float = 0.10
    # Top surface height above the catch-plane (z=0). Robots ride on this surface.
    platform_top_z: float = 0.10

    @property
    def platform_center_z(self) -> float:
        return self.platform_top_z - self.platform_thickness_m / 2.0


class Dohyo:
    """The sumo ring. One instance per simulation client."""

    def __init__(self, client_id: int, spec: DohyoSpec) -> None:
        self.client_id = client_id
        self.spec = spec
        self.ground_id: int | None = None
        self.platform_id: int | None = None

    @classmethod
    def from_config(cls, client_id: int, config_path: str | Path = DEFAULT_ARENA_CONFIG) -> "Dohyo":
        cfg: dict[str, Any] = load_config(config_path)
        arena = cfg["arena"]
        spec = DohyoSpec(
            radius_m=float(arena["radius_m"]),
            boundary_line_width_m=float(arena["boundary_line_width_m"]),
        )
        return cls(client_id, spec)

    def build(self) -> None:
        """Create the catch-plane and the raised dohyo platform."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        # Catch-plane well-suited as the floor a fallen robot lands on.
        self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.spec.radius_m,
            height=self.spec.platform_thickness_m,
            physicsClientId=self.client_id,
        )
        # Play surface: dark. Outer boundary line rendered as a slightly larger white disk
        # sitting just beneath the play surface so its rim shows as a ring.
        vis_surface = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.spec.radius_m,
            length=self.spec.platform_thickness_m,
            rgbaColor=[0.15, 0.15, 0.18, 1.0],
            physicsClientId=self.client_id,
        )
        self.platform_id = p.createMultiBody(
            baseMass=0.0,  # static
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis_surface,
            basePosition=[0.0, 0.0, self.spec.platform_center_z],
            physicsClientId=self.client_id,
        )
        # White boundary ring (visual only) as a thin disk flush with the top surface.
        ring_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.spec.radius_m,
            length=0.002,
            rgbaColor=[0.95, 0.95, 0.95, 1.0],
            physicsClientId=self.client_id,
        )
        inner_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.spec.radius_m - self.spec.boundary_line_width_m,
            length=0.004,
            rgbaColor=[0.15, 0.15, 0.18, 1.0],
            physicsClientId=self.client_id,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=ring_vis,
            basePosition=[0.0, 0.0, self.spec.platform_top_z + 0.001],
            physicsClientId=self.client_id,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=inner_vis,
            basePosition=[0.0, 0.0, self.spec.platform_top_z + 0.002],
            physicsClientId=self.client_id,
        )
        # Platform friction: moderate, so pushing translates to sliding rather than grip-lock.
        p.changeDynamics(self.platform_id, -1, lateralFriction=0.6, physicsClientId=self.client_id)

    def is_outside(self, position: tuple[float, float, float]) -> bool:
        """True once the body center crosses the ring edge (radial > radius)."""
        x, y, _ = position
        return math.hypot(x, y) > self.spec.radius_m

    def has_fallen(self, position: tuple[float, float, float]) -> bool:
        """Backstop for the edge test: body dropped well below the platform top."""
        return position[2] < self.spec.platform_top_z - 0.05

    def has_capsized(self, orientation: tuple[float, float, float, float], max_tilt_rad: float = 1.0) -> bool:
        """True once the chassis has tipped past max_tilt_rad (default ~57 deg) from
        upright, whether or not it has left the platform or dropped in z. A robot
        pushed hard enough to tip onto its end/side while still ON the platform
        keeps roughly the same z-height and stays within the ring radius, so neither
        has_fallen() nor is_outside() ever catches it - confirmed via a real match
        where the agent visibly capsized (see GUI screenshot) and the episode then
        ran for hundreds more steps as "ongoing" with the agent sliding around
        incapacitated, timing out to a draw that reflects a physics event, not a
        tactical outcome. Real sumo rules treat a capsize as ending the match; this
        mirrors that instead of silently missing it. max_tilt_rad=1.0 (~57 deg) is a
        starting point - tune once you have a feel for how much lean is "still
        fighting" vs. "actually tipped over" from watching a few real matches."""
        roll, pitch, _yaw = p.getEulerFromQuaternion(orientation)
        return abs(roll) > max_tilt_rad or abs(pitch) > max_tilt_rad

    def distance_to_edge(self, position: tuple[float, float, float]) -> float:
        """Signed distance from body center to the ring edge (positive = inside)."""
        x, y, _ = position
        return self.spec.radius_m - math.hypot(x, y)