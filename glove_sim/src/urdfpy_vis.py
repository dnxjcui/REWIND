"""URDF visualization via urdfpy.

Loads the glove URDF, performs forward kinematics at given joint angles,
and returns a positioned trimesh.Scene for GLB export. Replaces the manual
body-transform approach that was brittle against multi-axis joint RPY values.
"""

import re
import tempfile
import os
import numpy as np
import trimesh
from pathlib import Path

# urdfpy uses deprecated numpy aliases removed in NumPy 1.24; patch before import.
np.float = float   # noqa: NPY001
np.int = int       # noqa: NPY001
np.bool = bool     # noqa: NPY001
np.complex = complex
np.object = object
np.str = str

import xml.etree.ElementTree as ET

_robot_cache: dict = {}   # keyed by (urdf_path, mesh_dir) so reloads only when paths change


# Offset of hand_mount relative to URDF root link (from fixed_node_to_root_joint_0).
# Used to convert hand_mount world pose (from MuJoCo) into URDF-root world pose.
_ROOT_TO_HANDMOUNT_XYZ = np.array([-0.157876, 0.0663838, -0.0660817])


def load_robot(urdf_path: Path, mesh_dir: Path):
    """Load and cache a urdfpy URDF robot with paths and numpy compat fixed."""
    import urdfpy as _urdfpy

    key = (str(urdf_path), str(mesh_dir))
    if key in _robot_cache:
        return _robot_cache[key]

    mesh_dir_abs = str(mesh_dir.resolve())

    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://"):
            fn = fn.split("/", 2)[-1]
            fn = fn.split("/", 1)[-1]
            fn = fn.split("/", 1)[-1]
        mesh.set("filename", f"{mesh_dir_abs}/{fn}")

    for material in root.iter("material"):
        for tex in list(material.findall("texture")):
            material.remove(tex)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
        tree.write(f, encoding="unicode")
        tmp_path = f.name

    try:
        robot = _urdfpy.URDF.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    _robot_cache[key] = robot
    return robot


def get_glove_scene(
    robot,
    joint_cfg: dict[str, float],
    T_hand_mount_world: np.ndarray,
    sensor_positions_ydown: np.ndarray | None = None,
    sensor_sphere_radius: float = 0.005,
) -> trimesh.Scene:
    """Build a positioned trimesh.Scene for the glove at the given joint angles.

    Parameters
    ----------
    robot               : urdfpy.URDF robot object from load_robot()
    joint_cfg           : {joint_name: angle_rad} for all actuated joints
    T_hand_mount_world  : (4,4) world transform of hand_mount body, in MuJoCo Y-down frame
    sensor_positions_ydown : (N, 3) sensor dot positions in Y-down frame, or None
    sensor_sphere_radius   : radius of red sensor spheres in metres

    Returns
    -------
    trimesh.Scene with all glove meshes and optional sensor spheres, Y-up for GLB.
    """
    # URDF root → hand_mount is a fixed joint with xyz offset (rpy=0).
    # Invert to get hand_mount → root, then combine with hand_mount world pose.
    T_root_to_hm = np.eye(4)
    T_root_to_hm[:3, 3] = _ROOT_TO_HANDMOUNT_XYZ

    T_hm_to_root = np.eye(4)
    T_hm_to_root[:3, 3] = -_ROOT_TO_HANDMOUNT_XYZ  # pure translation, no rotation

    # World pose of URDF root (Y-down MuJoCo frame)
    T_root_world_yd = T_hand_mount_world @ T_hm_to_root

    # Y-down → Y-up coordinate flip for GLB export
    YDOWN_TO_YUP = np.array([
        [1,  0,  0,  0],
        [0,  0,  1,  0],
        [0, -1,  0,  0],
        [0,  0,  0,  1],
    ], dtype=float)

    # urdfpy FK: {trimesh_mesh: T_4x4_from_urdf_root}
    fk = robot.visual_trimesh_fk(cfg=joint_cfg)

    meshes = []
    for mesh, T_from_root in fk.items():
        # World transform in Y-down frame
        T_world_yd = T_root_world_yd @ T_from_root
        # Convert to Y-up
        T_world_yu = YDOWN_TO_YUP @ T_world_yd

        m = mesh.copy()
        m.apply_transform(T_world_yu)
        meshes.append(m)

    # Red sensor spheres
    if sensor_positions_ydown is not None:
        for pos_yd in sensor_positions_ydown:
            pos_h = np.array([pos_yd[0], pos_yd[1], pos_yd[2], 1.0])
            pos_yu = (YDOWN_TO_YUP @ pos_h)[:3]
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=sensor_sphere_radius)
            sphere.visual.face_colors = np.array([220, 20, 20, 255], dtype=np.uint8)
            T_s = np.eye(4)
            T_s[:3, 3] = pos_yu
            sphere.apply_transform(T_s)
            meshes.append(sphere)

    return trimesh.Scene(meshes)
