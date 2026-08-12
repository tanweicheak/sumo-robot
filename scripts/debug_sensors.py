# """
# scripts.debug_sensors

# Phase: Phase 1 (diagnostic)
# Purpose: Static sensor inspector. Spawns two robots facing each other, draws the ToF
#     fan and IR probes as debug lines in the PyBullet GUI, and prints raw sensor values
#     so ray geometry can be calibrated visually. Self-terminates after a fixed number
#     of cycles. Not part of the training/eval pipeline.
# """

# from __future__ import annotations

# import time

# import numpy as np
# import pybullet as p

# from src.simulation.arena import Dohyo
# from src.simulation.robot import RobotSpec, build_robot
# from src.simulation.sensors import SensorSuite


# def main() -> None:
#     client = p.connect(p.GUI)
#     p.setGravity(0, 0, -9.81, physicsClientId=client)
#     p.setTimeStep(1.0 / 240.0, physicsClientId=client)
#     p.resetDebugVisualizerCamera(
#         cameraDistance=2.2, cameraYaw=50, cameraPitch=-35,
#         cameraTargetPosition=[0, 0, 0.1], physicsClientId=client,
#     )

#     dohyo = Dohyo.from_config(client)
#     dohyo.build()

#     spec = RobotSpec.from_config()
#     top = dohyo.spec.platform_top_z
#     agent = build_robot(client, spec, (-0.6, 0.0, top + 0.12), 0.0, (0.2, 0.5, 0.9, 1.0))
#     opponent = build_robot(client, spec, (0.6, 0.0, top + 0.12), np.pi, (0.9, 0.4, 0.2, 1.0))

#     # Settle fully before reading.
#     for _ in range(120):
#         p.stepSimulation(physicsClientId=client)
#     for body in (agent.body_id, opponent.body_id):
#         p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0], physicsClientId=client)

#     sensors = SensorSuite(agent, client, self_body_id=agent.body_id)

#     print("Watch the GUI. ToF rays = cyan/red, IR probes = magenta.")
#     print("The agent (blue) faces +x toward the opponent (orange).\n")

#     try:
#         for _ in range(200):
#             reading = sensors.read(debug_draw=True)
#             pos, _ = agent.base_pose()
#             print(
#                 f"agent_z={pos[2]:.3f} | "
#                 f"tof={np.round(reading['tof'], 3)} | "
#                 f"ir={np.round(reading['ir'], 3)} | "
#                 f"min_tof={float(np.min(reading['tof'])):.3f}",
#                 end="\r",
#             )
#             for _ in range(12):  # one 50 ms control cycle
#                 p.stepSimulation(physicsClientId=client)
#             time.sleep(0.05)
#     except KeyboardInterrupt:
#         print("\nstopping.")
#     finally:
#         print()  # newline so the final reading isn't overwritten
#         p.disconnect(physicsClientId=client)


# if __name__ == "__main__":
#     main()