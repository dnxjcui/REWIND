"""Standalone glove diagnostic: exports GLBs for visual inspection without DynHaMR data.

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
from src.visualize import export_glove_only_glb


def main():
    print("Loading glove into MuJoCo...")
    sim = GloveSimulator(cfg.URDF_PATH, cfg.MESH_DIR)
    print(f"  nq={sim.model.nq}, nv={sim.model.nv}, nbody={sim.model.nbody}")

    out_dir = cfg.OUTPUT_DIR / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 90° rotation around Z axis so the glove lies flat in the XZ plane.
    # wxyz quaternion: [cos(45°), 0, 0, sin(45°)] = 90° around Z.
    # If the result appears mirrored, try [0.707, 0.0, 0.0, -0.707] instead.
    BASE_QUAT = np.array([0.707, 0.0, 0.0, 0.707])
    sim.set_base_pose(np.zeros(3), BASE_QUAT)
    mujoco.mj_forward(sim.model, sim.data)

    # --- Rest pose (all joints at 0) ---
    geom_poses = sim.get_geom_world_poses()
    sensor_pos = sim.get_sensor_world_positions()
    out_path = out_dir / "glove_rest.glb"
    export_glove_only_glb(geom_poses, out_path, sensor_positions_ydown=sensor_pos)
    print(f"  Exported: {out_path}")

    # --- Index tip sweep: revolute_9_0 from 0° to 90° in 10° steps ---
    idx_tip_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "revolute_9_0")
    if idx_tip_jid < 0:
        print("[ERROR] Joint 'revolute_9_0' not found in model.")
        return
    idx_tip_qadr = sim.model.jnt_qposadr[idx_tip_jid]

    for angle_deg in range(0, 91, 10):
        # Reset all joints to 0 first, then set only the index tip
        mujoco.mj_resetData(sim.model, sim.data)
        sim.set_base_pose(np.zeros(3), BASE_QUAT)
        sim.data.qpos[idx_tip_qadr] = np.radians(angle_deg)
        mujoco.mj_forward(sim.model, sim.data)

        geom_poses = sim.get_geom_world_poses()
        sensor_pos = sim.get_sensor_world_positions()
        out_path = out_dir / f"glove_index_{angle_deg:03d}.glb"
        export_glove_only_glb(geom_poses, out_path, sensor_positions_ydown=sensor_pos)
        print(f"  Exported: {out_path}  (index_tip={angle_deg}°)")

    print(f"\nDone. {len(list(out_dir.glob('*.glb')))} GLBs in {out_dir}")
    print("View with: f3d outputs/diagnostic/glove_rest.glb")


if __name__ == "__main__":
    main()
