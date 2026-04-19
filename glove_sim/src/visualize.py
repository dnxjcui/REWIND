"""Export per-frame GLB files combining the hand mesh and positioned glove mesh."""

import os
import numpy as np
import trimesh
from pathlib import Path

from config import GLB_DIR, MESH_DIR, LINK_TO_STL
from src.calibration import YDOWN_TO_YUP


def export_frame_glb(
    frame_idx: int,
    link_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    mesh_dir: Path = MESH_DIR,
    glb_dir: Path = GLB_DIR,
) -> None:
    """Combine the DynHaMR hand GLB with the glove mesh for one frame.

    Parameters
    ----------
    frame_idx  : 0-based frame index
    link_poses : {link_name: (pos_3, rotmat_3x3)} from GloveSimulator, Y-down frame
    out_path   : destination .glb file path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meshes = []

    # 1. Hand mesh from DynHaMR GLB (already Y-up)
    glb_file = glb_dir / f"{frame_idx:06d}_hands.glb"
    if glb_file.exists():
        try:
            hand_scene = trimesh.load(str(glb_file), force="scene")
            if isinstance(hand_scene, trimesh.Scene):
                for geom in hand_scene.geometry.values():
                    meshes.append(geom)
            else:
                meshes.append(hand_scene)
        except Exception as e:
            print(f"[WARNING] Could not load hand GLB for frame {frame_idx}: {e}")
    else:
        print(f"[WARNING] Hand GLB not found: {glb_file}")

    # 2. Glove meshes: apply world transform per link, convert Y-down → Y-up
    for link_name, stl_file in LINK_TO_STL.items():
        pose = link_poses.get(link_name)
        if pose is None:
            continue
        pos, rot = pose  # (3,), (3,3) in MuJoCo Y-down world frame

        stl_path = mesh_dir / stl_file
        if not stl_path.exists():
            continue
        try:
            mesh = trimesh.load(str(stl_path), force="mesh")
        except Exception:
            continue

        # Build 4x4 in Y-down frame
        T_yd = np.eye(4)
        T_yd[:3, :3] = rot
        T_yd[:3,  3] = pos

        # Convert to Y-up (GLB coordinate system)
        T_yu = YDOWN_TO_YUP @ T_yd @ np.linalg.inv(YDOWN_TO_YUP)

        mesh.apply_transform(T_yu)
        meshes.append(mesh)

    if not meshes:
        print(f"[WARNING] No meshes to export for frame {frame_idx}, skipping.")
        return

    scene = trimesh.Scene(meshes)
    scene.export(str(out_path), file_type="glb")


def export_frames(
    link_poses_seq: list[dict],
    frame_indices: list[int],
    frames_dir: Path,
) -> None:
    """Export GLBs for a list of frames.

    Parameters
    ----------
    link_poses_seq : list of per-frame link pose dicts from GloveSimulator
    frame_indices  : corresponding original frame indices (for GLB filename lookup)
    frames_dir     : output directory
    """
    for seq_idx, frame_idx in enumerate(frame_indices):
        out_path = frames_dir / f"{frame_idx:06d}_glove_overlay.glb"
        export_frame_glb(
            frame_idx=frame_idx,
            link_poses=link_poses_seq[seq_idx],
            out_path=out_path,
        )
        if seq_idx % 50 == 0:
            print(f"  Exported frame {frame_idx} ({seq_idx+1}/{len(frame_indices)})")
