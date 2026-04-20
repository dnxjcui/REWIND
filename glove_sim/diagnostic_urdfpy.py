"""Standalone urdfpy-only diagnostic: exports GLBs without any MuJoCo involvement.

Uses urdfpy to perform forward kinematics directly from the URDF file.
urdfpy handles URDF rpy (extrinsic XYZ) correctly, serving as ground truth
for visual comparison against the MuJoCo-based diagnostic.py outputs.

Outputs:
    outputs/diagnostic_urdfpy/glove_rest.glb       — all joints at 0 degrees
    outputs/diagnostic_urdfpy/glove_index_000.glb  — revolute_9_0 at 0 deg
    ...
    outputs/diagnostic_urdfpy/glove_index_090.glb  — revolute_9_0 at 90 deg

Usage:
    python diagnostic_urdfpy.py
    python diagnostic_urdfpy.py --thumb   # sweep thumb tip instead of index
"""

import argparse
import sys
import numpy as np
import trimesh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.urdfpy_vis import load_robot

YDOWN_TO_YUP = np.array([
    [1,  0,  0,  0],
    [0,  0,  1,  0],
    [0, -1,  0,  0],
    [0,  0,  0,  1],
], dtype=float)


# Visual origin xyz offsets from URDF <visual><origin> — the mesh centroid relative
# to each link's joint-frame origin. Used to track the fingertip cap position.
_TIP_VISUAL_OFFSETS = {
    "part_3":   np.array([-0.125132,  0.004875, -0.0466837]),
    "part_3_1": np.array([-0.0674761, 0.004875, -0.0250619]),
}


def get_tip_position(robot, joint_cfg: dict, tip_link_name: str) -> np.ndarray:
    """Return the world-frame (Y-down) position of a fingertip cap's visual centre.

    robot.link_fk() returns {link: T_4x4} where T_4x4[:3,3] is the link's joint-frame
    origin — NOT the mesh position. Revolute joints rotate the link around that origin,
    so the origin position is invariant to joint angle. To get a point that actually moves,
    we apply the visual origin offset (from the URDF <visual><origin>) in link-local frame:

        world_pt = T[:3,:3] @ local_offset + T[:3,3]

    This is the foundation for urdfpy-based numerical IK (see diagnostics.md).
    """
    local_offset = _TIP_VISUAL_OFFSETS.get(tip_link_name, np.zeros(3))
    fk = robot.link_fk(cfg=joint_cfg)
    for link, T in fk.items():
        if link.name == tip_link_name:
            return (T[:3, :3] @ local_offset + T[:3, 3]).copy()
    raise ValueError(f"Link '{tip_link_name}' not found in FK result")


def export_urdfpy_glb(robot, joint_cfg: dict, out_path: Path) -> None:
    """Export a GLB of the glove at the given joint configuration."""
    fk = robot.visual_trimesh_fk(cfg=joint_cfg)
    meshes = []
    for mesh, T_from_root in fk.items():
        T_yu = YDOWN_TO_YUP @ T_from_root
        m = mesh.copy()
        m.apply_transform(T_yu)
        meshes.append(m)
    if not meshes:
        print(f"[WARNING] No meshes for {out_path}")
        return
    scene = trimesh.Scene(meshes)
    scene.export(str(out_path), file_type="glb")


def main():
    parser = argparse.ArgumentParser(description="urdfpy-only glove diagnostic")
    parser.add_argument("--thumb", action="store_true",
                        help="Sweep thumb tip (revolute_4_0) instead of index (revolute_9_0)")
    args = parser.parse_args()

    out_dir = cfg.OUTPUT_DIR / "diagnostic_urdfpy"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading URDF into urdfpy (no MuJoCo)...")
    robot = load_robot(cfg.URDF_PATH, cfg.MESH_DIR_SRC)
    actuated = [j.name for j in robot.actuated_joints]
    print(f"  {len(robot.links)} links, {len(actuated)} actuated joints")
    print(f"  Joints: {actuated}")

    # Rest pose — all joints at 0
    out_path = out_dir / "glove_rest.glb"
    export_urdfpy_glb(robot, {}, out_path)
    # Print tip positions as a sanity check
    thumb_pos = get_tip_position(robot, {}, "part_3")
    index_pos = get_tip_position(robot, {}, "part_3_1")
    print(f"  Exported: {out_path}")
    print(f"    part_3  (thumb tip) origin at: {np.round(thumb_pos, 4)}")
    print(f"    part_3_1 (index tip) origin at: {np.round(index_pos, 4)}")

    # Joint sweep
    sweep_joint = "revolute_4_0" if args.thumb else "revolute_9_0"
    sweep_label = "thumb" if args.thumb else "index"
    print(f"\nSweeping {sweep_joint} (0°→90°)...")

    for angle_deg in range(0, 91, 10):
        joint_cfg = {sweep_joint: np.radians(angle_deg)}
        out_path = out_dir / f"glove_{sweep_label}_{angle_deg:03d}.glb"
        export_urdfpy_glb(robot, joint_cfg, out_path)
        tip_link = "part_3" if args.thumb else "part_3_1"
        tip_pos = get_tip_position(robot, joint_cfg, tip_link)
        print(f"  {out_path.name}  tip={np.round(tip_pos, 4)}")

    n_glb = len(list(out_dir.glob("*.glb")))
    print(f"\nDone. {n_glb} GLB(s) in {out_dir}")
    print(f"View with: f3d {out_dir}/glove_rest.glb")


if __name__ == "__main__":
    main()
