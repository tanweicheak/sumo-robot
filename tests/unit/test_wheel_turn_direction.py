"""
tests.unit.test_wheel_turn_direction

Phase: Phase 1 regression test (should have existed since Phase 1)
Purpose: robot.py mounts all 4 wheels on the SAME linkJointAxis ([0, 0, -1]) despite
    left/right banks being geometrically mirrored across the chassis Y axis. Driving
    both banks with the SAME PWM sign (straight-line motion) is not sufficient to
    verify wheel handedness is correct - a left/right axis-convention swap that's
    CONSISTENT between banks would still drive straight under symmetric PWM, but
    would silently invert every DIFFERENTIAL (turn) command project-wide.

    This test isolates that specific case: command a pure differential turn
    (left_pwm negative, right_pwm positive) and assert the chassis yaws the
    direction that convention implies. If this ever fails, every strategy label
    involving a turn (attack lead-term, BaitController's centering steer, any future
    TacticalCommand PWM output) is systematically mirrored in physical reality.

    v2: rebuilt on top of PyBulletSumoEnv.reset() instead of calling build_robot()
    directly at an arbitrary floating position. The original version spawned the
    robot with NO ground plane and NO platform, so wheel spin had nothing to push
    against - it measured "does an unsupported robot rotate" (always no) rather
    than "does this robot's differential drive turn the right way." Reusing the
    real production reset() path guarantees the same platform/spawn-height/settle
    sequence every actual training episode uses, so ground contact is never in
    question again.

Run: pytest tests/unit/test_wheel_turn_direction.py -v
    Requires pybullet (not available in the sandbox this was written in - written
    for you to run locally where PyBullet already works, per your GUI testing).
"""

from __future__ import annotations

import math

import numpy as np
import pybullet as p
import pytest

from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv


@pytest.fixture
def env():
    cfg = EnvConfig.from_config(use_gui=False)
    e = PyBulletSumoEnv(env_config=cfg)
    yield e
    e.close()


def _yaw(e: PyBulletSumoEnv) -> float:
    _, orn = e.agent.base_pose()
    return p.getEulerFromQuaternion(orn)[2]


def _drive_and_measure(e: PyBulletSumoEnv, left_pwm: float, right_pwm: float, cycles: int = 40):
    """Drive via env.step() (the real per-episode path, opponent left idle) and
    return (yaw_delta, linear_displacement_m)."""
    e.reset()
    pos_before, _ = e.agent.base_pose()
    yaw_before = _yaw(e)

    action = np.array([left_pwm, right_pwm], dtype=np.float32)
    for _ in range(cycles):
        e.step(action)

    pos_after, _ = e.agent.base_pose()
    yaw_after = _yaw(e)
    delta_yaw = math.atan2(math.sin(yaw_after - yaw_before), math.cos(yaw_after - yaw_before))
    displacement = math.hypot(pos_after[0] - pos_before[0], pos_after[1] - pos_before[1])
    return delta_yaw, displacement


def test_wheels_actually_produce_motion(env):
    """Precondition sanity check: confirm the robot is on a real surface and PWM
    produces SOME physical effect at all, before trusting any sign/direction result
    below. This is what the original (broken) test version silently lacked."""
    _, displacement = _drive_and_measure(env, 1.0, 1.0, cycles=40)
    assert displacement > 0.05, (
        f"Straight-line command produced only {displacement*100:.1f}cm of movement over "
        "40 control cycles (2s) - the robot may not be in contact with the platform, "
        "or motor torque/friction values in arena_config.yaml can't overcome static "
        "friction. Fix ground contact/torque before trusting any turn-direction result."
    )


def test_symmetric_pwm_drives_straight_not_spinning(env):
    """Same-sign PWM on both banks should NOT produce large yaw drift. If this
    fails, the more severe (both-banks-inverted) version of the bug is present and
    the turn-direction test below isn't even the right next check - fix
    straight-line motion first."""
    delta_yaw, displacement = _drive_and_measure(env, 1.0, 1.0, cycles=40)
    assert displacement > 0.05, "no motion detected - see test_wheels_actually_produce_motion"
    assert abs(delta_yaw) < 0.15, (
        f"Straight-line command (1.0, 1.0) produced {math.degrees(delta_yaw):.1f} deg "
        "of yaw drift while moving - the two wheel banks are not driving symmetrically."
    )


def test_differential_turn_direction_matches_convention(env):
    """The actual question: does (left_pwm=-1, right_pwm=+1) - which every piece of
    steering logic in this project (rule_based_controller._attack's lead term,
    BaitController._lure's centering steer) assumes turns the robot toward its own
    LEFT (counter-clockwise from above, yaw increasing in PyBullet's convention) -
    actually do that in the physics, or is it mirrored?"""
    delta_yaw, displacement = _drive_and_measure(env, -1.0, 1.0, cycles=40)
    assert displacement > 0.02, "no motion detected - see test_wheels_actually_produce_motion"
    assert delta_yaw > 0.05, (
        f"apply_pwm(left=-1.0, right=+1.0) produced a yaw change of {math.degrees(delta_yaw):.1f} deg "
        "(expected clearly POSITIVE / counter-clockwise). If this is negative, every "
        "turn-direction assumption in rule_based_controller.py, adversarial_bait_controller.py, "
        "and any future TacticalCommand output is mirrored versus physical reality - "
        "left_wheel_joints/right_wheel_joints or linkJointAxis in robot.py needs a sign fix, "
        "not the callers."
    )


def test_opposite_differential_turns_the_other_way(env):
    """Complement of the above: the mirror-image command should yaw the opposite way,
    confirming this isn't just a magnitude/noise artifact."""
    delta_yaw, displacement = _drive_and_measure(env, 1.0, -1.0, cycles=40)
    assert displacement > 0.02, "no motion detected - see test_wheels_actually_produce_motion"
    assert delta_yaw < -0.05, (
        f"apply_pwm(left=+1.0, right=-1.0) produced a yaw change of {math.degrees(delta_yaw):.1f} deg "
        "(expected clearly NEGATIVE). See test_differential_turn_direction_matches_convention "
        "for what a failure here means."
    )