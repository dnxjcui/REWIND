"""Standalone glove diagnostic: exports GLBs for visual inspection without DynHaMR data.

Primary export uses MuJoCo forward kinematics and the same mesh placement as
`pipeline.py` (`get_geom_world_poses` + `export_glove_only_glb`), so link meshes
stay connected. Optional `--urdfpy` writes parallel GLBs for comparison only.

Usage:
    python diagnostic.py
    python diagnostic.py --urdfpy

Outputs (MuJoCo, default):
    outputs/diagnostic/glove_rest.glb          — all joints at 0 degrees
    outputs/diagnostic/glove_index_000.glb     — index tip at 0°
    ...
    outputs/diagnostic/glove_index_090.glb     — index tip at 90°

With --urdfpy:
    outputs/diagnostic/glove_rest_urdfpy.glb
    outputs/diagnostic/glove_index_000_urdfpy.glb
    ...
"""

import argparse
import sys
import numpy as np
import mujoco
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.glove_ik import GloveSimulator
from src.visualize import export_glove_only_glb


def _export_mujoco_glb(sim: GloveSimulator, out_path: Path) -> None:
    mujoco.mj_forward(sim.model, sim.data)
    geom_poses = sim.get_geom_world_poses()
    sensor_pos = sim.get_sensor_world_positions()
    export_glove_only_glb(
        geom_poses,
        out_path,
        mesh_dir=cfg.MESH_DIR,
        sensor_positions_ydown=sensor_pos,
    )


def main():
    parser = argparse.ArgumentParser(description="Glove diagnostic GLB export")
    parser.add_argument(
        "--urdfpy",
        action="store_true",
        help="Also export *_urdfpy.glb files (second FK path; for comparison only)",
    )
    args = parser.parse_args()

    print("Loading glove into MuJoCo...")
    sim = GloveSimulator(cfg.URDF_PATH, cfg.MESH_DIR)
    print(f"  nq={sim.model.nq}, nv={sim.model.nv}, nbody={sim.model.nbody}")

    robot = None
    if args.urdfpy:
        from src.urdfpy_vis import load_robot, get_glove_scene

        print("Loading URDF into urdfpy (optional comparison)...")
        robot = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)
        print(f"  urdfpy: {len(robot.links)} links, {len(robot.actuated_joints)} actuated joints")

    out_dir = cfg.OUTPUT_DIR / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    def export_urdfpy_if_requested(suffix: str) -> None:
        if robot is None:
            return
        joint_cfg = sim.read_all_joint_angles()
        T_hm = sim.get_hand_mount_world_pose()
        sensor_pos = sim.get_sensor_world_positions()
        scene = get_glove_scene(
            robot,
            joint_cfg,
            T_hm,
            sensor_positions_ydown=sensor_pos,
            sensor_sphere_radius=cfg.SENSOR_SPHERE_RADIUS,
        )
        scene.export(str(out_dir / suffix))

    # Place glove at world origin with identity rotation
    sim.set_base_pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mujoco.mj_forward(sim.model, sim.data)

    # --- Rest pose (all joints at 0) ---
    out_path = out_dir / "glove_rest.glb"
    _export_mujoco_glb(sim, out_path)
    print(f"  Exported (MuJoCo): {out_path}")
    export_urdfpy_if_requested("glove_rest_urdfpy.glb")

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
        out_path = out_dir / f"glove_index_{angle_deg:03d}.glb"
        _export_mujoco_glb(sim, out_path)
        print(f"  Exported (MuJoCo): {out_path}  (index_tip={angle_deg}°)")
        if robot is not None:
            export_urdfpy_if_requested(f"glove_index_{angle_deg:03d}_urdfpy.glb")

    n_glb = len(list(out_dir.glob("*.glb")))
    print(f"\nDone. {n_glb} GLB(s) in {out_dir}")
    print("View with: f3d outputs/diagnostic/glove_rest.glb")


if __name__ == "__main__":
    main()
