"""URDF visualization via urdfpy.

Loads the glove URDF, performs forward kinematics at given joint angles,
and returns a positioned trimesh.Scene for GLB export. Replaces the manual
body-transform approach that was brittle against multi-axis joint RPY values.
"""

import os
import tempfile
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

_robot_cache: dict = {}  # keyed by (urdf_path, mesh_dir)


# Offset of hand_mount relative to URDF root link (from fixed_node_to_root_joint_0).
# Used to convert hand_mount world pose (from MuJoCo) into URDF-root world pose.
_ROOT_TO_HANDMOUNT_XYZ = np.array([-0.157876, 0.0663838, -0.0660817])


def _package_mesh_to_basename(fn: str) -> str:
    """Strip ROS package://…/meshes/<file> down to <file>.stl."""
    if not fn.startswith("package://"):
        return fn
    rest = fn.split("/", 2)[-1]
    rest = rest.split("/", 1)[-1]
    return rest.split("/", 1)[-1]


def _prepare_urdf_tree_for_urdfpy(tree: ET.ElementTree, urdf_dir: Path, mesh_dir: Path) -> None:
    """Mutate tree so mesh filenames work with urdfpy.URDF.load (same as get_filename).

    urdfpy resolves mesh paths as ``os.path.join(urdf_directory, filename)`` when
    ``filename`` is not absolute (see ``urdfpy.utils.get_filename``). ROS ``package://``
    URIs are not supported upstream, so we rewrite them to paths relative to the URDF
    folder — the same layout ``URDF.save`` documents (relative to the .urdf file).

    Empty ``<texture/>`` placeholders (OnShape exports) are removed so Material parsing
    does not try to load a texture without a filename.
    """
    urdf_dir = urdf_dir.resolve()
    mesh_dir = mesh_dir.resolve()

    for mesh_el in tree.getroot().iter("mesh"):
        fn = mesh_el.get("filename", "")
        if not fn:
            continue
        if fn.startswith("package://"):
            base = _package_mesh_to_basename(fn)
            abs_mesh = mesh_dir / base
            rel = os.path.relpath(abs_mesh, start=urdf_dir)
            mesh_el.set("filename", rel.replace("\\", "/"))
        elif os.path.isabs(fn):
            rel = os.path.relpath(Path(fn).resolve(), start=urdf_dir)
            mesh_el.set("filename", rel.replace("\\", "/"))

    for material in tree.getroot().iter("material"):
        for tex in list(material.findall("texture")):
            if not tex.attrib or not tex.get("filename"):
                material.remove(tex)


def load_robot(urdf_path: Path, mesh_dir: Path):
    """Load urdfpy.URDF the same way ``URDF.load`` does, with ROS package paths fixed.

    The returned object is identical to what you get after saving a URDF whose mesh
    paths are relative to the ``.urdf`` directory. A short-lived temp file is written
    **next to** ``urdf_path`` so relative paths resolve like upstream urdfpy.

    For a checked-in URDF that already uses ``../meshes/...`` and valid materials,
    you can call ``urdfpy.URDF.load`` directly on
    ``rewind_glove_assembly/urdf/rewind_glove_for_urdfpy.urdf``.
    """
    import urdfpy as _urdfpy

    urdf_path = urdf_path.resolve()
    key = (str(urdf_path), str(mesh_dir.resolve()))
    if key in _robot_cache:
        return _robot_cache[key]

    urdf_dir = urdf_path.parent
    tree = ET.parse(str(urdf_path))
    _prepare_urdf_tree_for_urdfpy(tree, urdf_dir, mesh_dir)

    fd, tmp_path = tempfile.mkstemp(
        prefix="._urdfpy_load_",
        suffix=".urdf",
        dir=str(urdf_dir),
    )
    os.close(fd)
    try:
        tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
        robot = _urdfpy.URDF.load(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    _robot_cache[key] = robot
    return robot


def hand_mount_world_pose_for_root_identity() -> np.ndarray:
    """Return hand-mount world pose that places URDF root at identity (Y-down).

    `get_glove_scene` expects the hand-mount world pose as input. A common mistake
    is passing URDF-root identity directly, which visually detaches components.
    This helper provides the correct rest pose for root-at-origin comparisons.
    """
    t = np.eye(4, dtype=float)
    t[:3, 3] = _ROOT_TO_HANDMOUNT_XYZ
    return t


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
