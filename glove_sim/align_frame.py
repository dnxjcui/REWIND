"""Phase 1: urdfpy + scipy IK for glove alignment, test frames only."""

import sys
import numpy as np
import scipy.optimize
import trimesh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from src.urdfpy_vis import ROOT_TO_HANDMOUNT_XYZ

_THUMB_CHAIN = ['revolute_1_0', 'revolute_2_0', 'revolute_3_0', 'revolute_4_0']
_INDEX_CHAIN = ['revolute_5_0', 'revolute_6_0', 'revolute_7_0', 'revolute_8_0', 'revolute_9_0']
_THUMB_VISUAL_ORIGIN = np.array([-0.125132,  0.004875, -0.0466837])
_INDEX_VISUAL_ORIGIN = np.array([-0.0674761, 0.004875, -0.0250619])

# hand_mount → urdf root: negate the root→hand_mount offset (pure translation, no rotation)
_HM_TO_ROOT = np.eye(4, dtype=np.float64)
_HM_TO_ROOT[:3, 3] = -ROOT_TO_HANDMOUNT_XYZ


def _ik_finger(robot, chain_joints, tip_link_name, visual_origin, target_in_root, q0=None):
    """Solve IK for one finger chain using urdfpy FK + scipy LM.

    Returns (joint_cfg_dict, residual_metres).
    """
    if q0 is None:
        q0 = np.zeros(len(chain_joints))

    def residual(q_vec):
        fk = robot.link_fk(cfg=dict(zip(chain_joints, q_vec)))
        for link, T in fk.items():
            if link.name == tip_link_name:
                return T[:3, :3] @ visual_origin + T[:3, 3] - target_in_root
        return np.full(3, 1e6)

    result = scipy.optimize.least_squares(
        residual, q0, method='trf',
        ftol=cfg.IK_TOL, max_nfev=cfg.IK_MAX_NFEV,
    )
    return dict(zip(chain_joints, result.x)), float(np.linalg.norm(result.fun))


def solve_ik_frame(robot, T_wrist, thumb_tip_world, index_tip_world,
                   prev_q_thumb=None, prev_q_index=None):
    """Solve IK for both fingers in one frame.

    Parameters
    ----------
    robot           : urdfpy.URDF loaded via load_robot()
    T_wrist         : (4,4) wrist world transform, Y-down frame
    thumb_tip_world : (3,) thumb fingertip world position, Y-down frame
    index_tip_world : (3,) index fingertip world position, Y-down frame
    prev_q_thumb    : (4,) warm-start angles for thumb chain, or None
    prev_q_index    : (5,) warm-start angles for index chain, or None

    Returns
    -------
    joint_cfg      : dict[str, float] — all 9 joint angles in radians
    thumb_residual : float — Euclidean distance in metres
    index_residual : float — Euclidean distance in metres
    """
    T_root_world = T_wrist @ _HM_TO_ROOT
    inv_T = np.linalg.inv(T_root_world)

    thumb_target = (inv_T @ np.append(thumb_tip_world, 1.0))[:3]
    index_target = (inv_T @ np.append(index_tip_world, 1.0))[:3]

    thumb_cfg, thumb_res = _ik_finger(
        robot, _THUMB_CHAIN, 'part_3',   _THUMB_VISUAL_ORIGIN, thumb_target, prev_q_thumb)
    index_cfg, index_res = _ik_finger(
        robot, _INDEX_CHAIN, 'part_3_1', _INDEX_VISUAL_ORIGIN, index_target, prev_q_index)

    return {**thumb_cfg, **index_cfg}, thumb_res, index_res
