"""
glove_sim pipeline — simulate AS5600 sensor data from DynHaMR knot_one_handed output.

Usage:
    python pipeline.py                        # full 1196-frame run
    python pipeline.py --frames 0 10          # quick test: frames 0-9
    python pipeline.py --no-vis               # skip GLB visualization export
    python pipeline.py --calibration outputs/calibration.npy
"""

import sys
import argparse
import numpy as np
import mujoco
from pathlib import Path
from tqdm import tqdm

# Add glove_sim root to path so `config` and `src` imports work
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.mano_loader import load_mano_trajectory
from src.calibration import default_calibration, load_calibration, save_calibration, pose_from_rotvec, mat_to_quat_wxyz
from src.glove_ik import GloveSimulator
from src.sensor_output import save_sensor_data
from src.visualize import export_frames


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", nargs=2, type=int, metavar=("START", "END"),
                   help="Process only frames [START, END) for quick testing")
    p.add_argument("--no-vis", action="store_true",
                   help="Skip per-frame GLB visualization export")
    p.add_argument("--calibration", type=str, default=None,
                   help="Path to calibration .npy file (4x4 T_wrist_to_base). "
                        "Uses default eyeballed calibration if not provided.")
    p.add_argument("--ik-iters", type=int, default=cfg.IK_ITERS)
    p.add_argument("--ik-tol", type=float, default=cfg.IK_TOL)
    p.add_argument("--zero-qpos", action="store_true",
                   help="Skip IK; hold all joints at 0 for rest-pose visualization debugging")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Step 1: Load MANO trajectory (right hand only)
    # ------------------------------------------------------------------
    print("Loading MANO trajectory from DynHaMR NPZ...")
    traj = load_mano_trajectory(cfg.NPZ_PATH, cfg.MANO_DIR)
    joints      = traj["joints"]       # (T, 16, 3) Y-down frame — 16 MANO pose joints
    vertices    = traj["vertices"]     # (T, 778, 3) Y-down frame — full mesh vertices
    root_orient = traj["root_orient"]  # (T, 3)
    trans       = traj["trans"]        # (T, 3)
    T_total     = traj["T"]
    print(f"  Loaded {T_total} frames, right-hand track mean is_right={traj['is_right_mean']:.4f}")

    # Frame range selection
    if args.frames:
        start, end = args.frames
        end = min(end, T_total)
        frame_indices = list(range(start, end))
    else:
        frame_indices = list(range(T_total))
    T = len(frame_indices)
    print(f"  Processing {T} frames (indices {frame_indices[0]}..{frame_indices[-1]})")

    # ------------------------------------------------------------------
    # Step 2: Calibration
    # ------------------------------------------------------------------
    if args.calibration and Path(args.calibration).exists():
        T_wrist_to_base = load_calibration(args.calibration)
        print(f"  Loaded calibration from {args.calibration}")
    else:
        T_wrist_to_base = default_calibration()
        save_calibration(T_wrist_to_base, cfg.CALIBRATION_PATH)
        print(f"  Using default calibration (saved to {cfg.CALIBRATION_PATH})")

    # ------------------------------------------------------------------
    # Step 3: Load MuJoCo glove model
    # ------------------------------------------------------------------
    print("Loading glove into MuJoCo...")
    sim = GloveSimulator(cfg.URDF_PATH, cfg.MESH_DIR)
    print("  MuJoCo model loaded successfully.")
    print(f"  nq={sim.model.nq}, nv={sim.model.nv}, nbody={sim.model.nbody}")

    # ------------------------------------------------------------------
    # Step 4: Per-frame IK loop
    # ------------------------------------------------------------------
    sensor_angles        = np.zeros((T, 5), dtype=np.float32)
    fingertip_positions  = np.zeros((T, 2, 3), dtype=np.float32)
    ik_residuals         = np.zeros(T, dtype=np.float32)
    link_poses_seq       = [] if not args.no_vis else None
    sensor_positions_seq = [] if not args.no_vis else None

    print("Running per-frame IK loop...")
    for seq_idx, t in enumerate(tqdm(frame_indices)):
        # Wrist world pose from MANO root_orient (axis-angle) + trans
        T_wrist = pose_from_rotvec(root_orient[t], trans[t])
        T_base  = T_wrist @ T_wrist_to_base

        pos_base  = T_base[:3, 3]
        quat_base = mat_to_quat_wxyz(T_base[:3, :3])

        # Fingertip targets from MANO mesh vertices (Y-down frame).
        # DynHaMR's pkl only provides 16 joints (no extra fingertip joints),
        # so we use known fingertip vertex indices on the 778-vertex MANO mesh.
        thumb_target = vertices[t, cfg.VERTEX_THUMB_TIP].astype(np.float64)
        index_target = vertices[t, cfg.VERTEX_INDEX_TIP].astype(np.float64)

        # Position glove base
        sim.set_base_pose(pos_base.astype(np.float64), quat_base)

        # Solve IK (warm-started: qpos carries over from previous frame)
        if not args.zero_qpos:
            residual = sim.solve_ik(
                thumb_target, index_target,
                n_iter=args.ik_iters, tol=args.ik_tol,
            )
        else:
            mujoco.mj_kinematics(sim.model, sim.data)
            residual = 0.0

        sensor_angles[seq_idx]       = sim.read_sensor_angles()
        fingertip_positions[seq_idx] = [thumb_target, index_target]
        ik_residuals[seq_idx]        = residual

        if link_poses_seq is not None:
            link_poses_seq.append(sim.get_geom_world_poses())
        if sensor_positions_seq is not None:
            sensor_positions_seq.append(sim.get_sensor_world_positions())

    # ------------------------------------------------------------------
    # Step 5: Save sensor data
    # ------------------------------------------------------------------
    print("Saving simulated sensor data...")
    save_sensor_data(
        sensor_angles=sensor_angles,
        fingertip_positions=fingertip_positions,
        ik_residuals=ik_residuals,
        calibration=T_wrist_to_base,
        out_path=cfg.SENSORS_NPZ_PATH,
    )

    # ------------------------------------------------------------------
    # Step 6: Export GLB visualizations
    # ------------------------------------------------------------------
    if not args.no_vis and link_poses_seq is not None:
        print(f"Exporting {T} GLB visualizations to {cfg.FRAMES_DIR} ...")
        export_frames(
            link_poses_seq=link_poses_seq,
            frame_indices=frame_indices,
            frames_dir=cfg.FRAMES_DIR,
            sensor_positions_seq=sensor_positions_seq,
        )
        print("Done.")
    else:
        print("Skipping GLB visualization export (--no-vis).")


if __name__ == "__main__":
    main()
