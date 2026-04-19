"""Export per-frame GLB files combining the hand mesh and positioned glove mesh."""

import numpy as np
import trimesh
from pathlib import Path

from config import GLB_DIR, MESH_DIR, LINK_TO_STL, SENSOR_SPHERE_RADIUS
from src.calibration import YDOWN_TO_YUP


def _make_sensor_sphere(pos_yup: np.ndarray, radius: float = SENSOR_SPHERE_RADIUS) -> trimesh.Trimesh:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sphere.visual.face_colors = np.array([220, 20, 20, 255], dtype=np.uint8)
    T = np.eye(4)
    T[:3, 3] = pos_yup
    sphere.apply_transform(T)
    return sphere


def export_frame_glb(
    frame_idx: int,
    geom_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    mesh_dir: Path = MESH_DIR,
    glb_dir: Path = GLB_DIR,
    sensor_positions_ydown: np.ndarray | None = None,
) -> None:
    """Combine the DynHaMR hand GLB with the glove mesh for one frame.

    Parameters
    ----------
    frame_idx             : 0-based frame index
    geom_poses            : {body_name: (pos_3, rotmat_3x3)} from get_geom_world_poses(), Y-down frame
    out_path              : destination .glb file path
    sensor_positions_ydown: (4, 3) sensor world positions in Y-down frame, or None
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

    # 2. Glove meshes: apply geom-level world transform, convert Y-down → Y-up
    for link_name, stl_file in LINK_TO_STL.items():
        pose = geom_poses.get(link_name)
        if pose is None:
            continue
        pos, rot = pose  # (3,), (3,3) in MuJoCo Y-down world frame; geom_xpos/xmat

        stl_path = mesh_dir / stl_file
        if not stl_path.exists():
            continue
        try:
            mesh = trimesh.load(str(stl_path), force="mesh")
        except Exception:
            continue

        # Build 4x4 in Y-down frame (geom_xpos/xmat already compose body pose + geom offset)
        T_yd = np.eye(4)
        T_yd[:3, :3] = rot
        T_yd[:3,  3] = pos

        # Convert to Y-up (GLB coordinate system)
        T_yu = YDOWN_TO_YUP @ T_yd

        mesh.apply_transform(T_yu)
        meshes.append(mesh)

    # 3. Red sensor spheres
    if sensor_positions_ydown is not None:
        for pos_yd in sensor_positions_ydown:
            pos_h = np.append(pos_yd, 1.0)
            pos_yu = (YDOWN_TO_YUP @ pos_h)[:3]
            meshes.append(_make_sensor_sphere(pos_yu))

    if not meshes:
        print(f"[WARNING] No meshes to export for frame {frame_idx}, skipping.")
        return

    scene = trimesh.Scene(meshes)
    scene.export(str(out_path), file_type="glb")


def export_glove_only_glb(
    geom_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    mesh_dir: Path = MESH_DIR,
    sensor_positions_ydown: np.ndarray | None = None,
) -> None:
    """Export a GLB containing only the glove mesh (no hand overlay).

    Parameters
    ----------
    geom_poses            : {body_name: (pos_3, rotmat_3x3)} from get_geom_world_poses(), Y-down frame
    out_path              : destination .glb file path
    mesh_dir              : directory containing binary STL files
    sensor_positions_ydown: (4, 3) sensor world positions in Y-down frame, or None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meshes = []

    # Glove meshes: apply geom-level world transform, convert Y-down → Y-up
    for link_name, stl_file in LINK_TO_STL.items():
        pose = geom_poses.get(link_name)
        if pose is None:
            continue
        pos, rot = pose

        stl_path = mesh_dir / stl_file
        if not stl_path.exists():
            continue
        try:
            mesh = trimesh.load(str(stl_path), force="mesh")
        except Exception:
            continue

        T_yd = np.eye(4)
        T_yd[:3, :3] = rot
        T_yd[:3,  3] = pos

        T_yu = YDOWN_TO_YUP @ T_yd

        mesh.apply_transform(T_yu)
        meshes.append(mesh)

    # Red sensor spheres
    if sensor_positions_ydown is not None:
        for pos_yd in sensor_positions_ydown:
            pos_h = np.append(pos_yd, 1.0)
            pos_yu = (YDOWN_TO_YUP @ pos_h)[:3]
            meshes.append(_make_sensor_sphere(pos_yu))

    if not meshes:
        print(f"[WARNING] No meshes to export for {out_path}, skipping.")
        return

    scene = trimesh.Scene(meshes)
    scene.export(str(out_path), file_type="glb")


def export_frames(
    link_poses_seq: list[dict],
    frame_indices: list[int],
    frames_dir: Path,
    sensor_positions_seq: list[np.ndarray] | None = None,
) -> None:
    """Export GLBs for a list of frames.

    Parameters
    ----------
    link_poses_seq        : list of per-frame geom pose dicts from get_geom_world_poses()
    frame_indices         : corresponding original frame indices (for GLB filename lookup)
    frames_dir            : output directory
    sensor_positions_seq  : list of (4, 3) sensor position arrays in Y-down frame, or None
    """
    for seq_idx, frame_idx in enumerate(frame_indices):
        out_path = frames_dir / f"{frame_idx:06d}_glove_overlay.glb"
        sensor_pos = sensor_positions_seq[seq_idx] if sensor_positions_seq is not None else None
        export_frame_glb(
            frame_idx=frame_idx,
            geom_poses=link_poses_seq[seq_idx],
            out_path=out_path,
            sensor_positions_ydown=sensor_pos,
        )
        if seq_idx % 50 == 0:
            print(f"  Exported frame {frame_idx} ({seq_idx+1}/{len(frame_indices)})")
