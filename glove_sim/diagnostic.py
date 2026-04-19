"""Standalone glove diagnostic: exports GLBs for visual inspection without DynHaMR data.

Uses urdfpy for forward kinematics (correct URDF rpy handling) and trimesh for export.

Usage:
    python diagnostic.py
Outputs:
    outputs/diagnostic/glove_rest.glb          — all joints at 0 degrees
    outputs/diagnostic/glove_index_000.glb     — index tip at 0°
    outputs/diagnostic/glove_index_010.glb     — index tip at 10°
    ...
    outputs/diagnostic/glove_index_090.glb     — index tip at 90°
"""

import sys
import numpy as np
import mujoco
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.glove_ik import GloveSimulator
from src.urdfpy_vis import load_robot, get_glove_scene


def main():
    print("Loading glove into MuJoCo...")
    sim = GloveSimulator(cfg.URDF_PATH, cfg.MESH_DIR)
    print(f"  nq={sim.model.nq}, nv={sim.model.nv}, nbody={sim.model.nbody}")

    print("Loading URDF into urdfpy...")
    robot = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)
    print(f"  urdfpy: {len(robot.links)} links, {len(robot.actuated_joints)} actuated joints")

    out_dir = cfg.OUTPUT_DIR / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Place glove at world origin with identity rotation
    sim.set_base_pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mujoco.mj_forward(sim.model, sim.data)

    # --- Rest pose (all joints at 0) ---
    joint_cfg = sim.read_all_joint_angles()
    T_hm = sim.get_hand_mount_world_pose()
    sensor_pos = sim.get_sensor_world_positions()
    scene = get_glove_scene(robot, joint_cfg, T_hm, sensor_positions_ydown=sensor_pos,
                            sensor_sphere_radius=cfg.SENSOR_SPHERE_RADIUS)
    out_path = out_dir / "glove_rest.glb"
    scene.export(str(out_path))
    print(f"  Exported: {out_path}")

    # --- Index tip sweep: revolute_9_0 from 0° to 90° in 10° steps ---
    idx_tip_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "revolute_9_0")
    if idx_tip_jid < 0:
        print("[ERROR] Joint 'revolute_9_0' not found in model.")
        return
    idx_tip_qadr = sim.model.jnt_qposadr[idx_tip_jid]

    for angle_deg in range(0, 91, 10):
        mujoco.mj_resetData(sim.model, sim.data)
        sim.set_base_pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        sim.data.qpos[idx_tip_qadr] = np.radians(angle_deg)
        mujoco.mj_forward(sim.model, sim.data)

        joint_cfg = sim.read_all_joint_angles()
        T_hm = sim.get_hand_mount_world_pose()
        sensor_pos = sim.get_sensor_world_positions()
        scene = get_glove_scene(robot, joint_cfg, T_hm, sensor_positions_ydown=sensor_pos,
                                sensor_sphere_radius=cfg.SENSOR_SPHERE_RADIUS)
        out_path = out_dir / f"glove_index_{angle_deg:03d}.glb"
        scene.export(str(out_path))
        print(f"  Exported: {out_path}  (index_tip={angle_deg}°)")

    print(f"\nDone. {len(list(out_dir.glob('*.glb')))} GLBs in {out_dir}")
    print("View with: f3d outputs/diagnostic/glove_rest.glb")


if __name__ == "__main__":
    main()
