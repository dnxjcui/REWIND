"""Calibration transform: MANO wrist frame → glove hand_mount frame."""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def default_calibration() -> np.ndarray:
    """Return a 4×4 homogeneous T_wrist_to_base transform.

    The hand_mount is strapped to the dorsum of the hand, slightly distal
    to the MANO wrist joint.  This default is an eyeballed estimate from
    the URDF geometry; tune it visually once GLB overlays are produced.

    Convention: T_base_world = T_wrist_world @ T_wrist_to_base
    """
    # Identity: glove base anchored at the MANO wrist joint origin.
    # Tune the translation once GLB overlays are visually inspected.
    return np.eye(4, dtype=np.float64)


def save_calibration(T: np.ndarray, path: str | Path) -> None:
    np.save(str(path), T)


def load_calibration(path: str | Path) -> np.ndarray:
    return np.load(str(path))


def pose_from_rotvec(rotvec: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4×4 homogeneous transform from axis-angle + translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3,  3] = translation
    return T


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion [w, x, y, z]."""
    q = Rotation.from_matrix(R).as_quat()  # returns [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


# Y-down (MuJoCo sim frame) → Y-up (glTF / GLB frame)
YDOWN_TO_YUP = np.diag([1.0, -1.0, -1.0, 1.0])
