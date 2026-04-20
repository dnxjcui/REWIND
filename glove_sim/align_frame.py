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


def _assert_frame(joint_cfg, thumb_res, index_res, frame_idx):
    assert thumb_res < cfg.IK_RESIDUAL_THRESHOLD, (
        f"Frame {frame_idx} thumb IK: {thumb_res*1000:.1f}mm "
        f"> {cfg.IK_RESIDUAL_THRESHOLD*1000:.0f}mm tolerance"
    )
    assert index_res < cfg.IK_RESIDUAL_THRESHOLD, (
        f"Frame {frame_idx} index IK: {index_res*1000:.1f}mm "
        f"> {cfg.IK_RESIDUAL_THRESHOLD*1000:.0f}mm tolerance"
    )
    for name, angle in joint_cfg.items():
        assert -np.pi <= angle <= np.pi, (
            f"Frame {frame_idx} joint {name}: {np.degrees(angle):.1f}° "
            f"out of [-180°, 180°]"
        )


def _export_frame(robot, T_wrist, joint_cfg, frame_idx):
    """Export combined hand+glove GLB for one frame."""
    from src.urdfpy_vis import get_glove_scene

    glove_scene = get_glove_scene(robot, joint_cfg, T_wrist)
    meshes = list(glove_scene.geometry.values())

    hand_glb = cfg.GLB_DIR / f"{frame_idx:06d}_hands.glb"
    if hand_glb.exists():
        hand_scene = trimesh.load(str(hand_glb), force="scene")
        meshes.extend(hand_scene.geometry.values())
    else:
        print(f"[WARNING] Hand GLB not found: {hand_glb}")

    out_path = cfg.ALIGNED_DIR / f"{frame_idx:06d}_aligned.glb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Scene(meshes).export(str(out_path))
    return out_path


def main():
    import argparse
    from src.urdfpy_vis import load_robot
    from src.mano_io import load_frame

    parser = argparse.ArgumentParser(description="Phase 1 glove alignment — test frames only")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 300])
    args = parser.parse_args()

    robot = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)
    print(f"URDF loaded: {len(robot.links)} links, {len(robot.actuated_joints)} actuated joints")

    prev_q_thumb = None
    prev_q_index = None
    out_paths = []

    for frame_idx in args.frames:
        print(f"\n--- Frame {frame_idx} ---")
        frame = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, frame_idx)

        joint_cfg, thumb_res, index_res = solve_ik_frame(
            robot, frame["T_wrist"], frame["thumb_tip"], frame["index_tip"],
            prev_q_thumb=prev_q_thumb, prev_q_index=prev_q_index,
        )
        print(f"  Thumb IK residual : {thumb_res*1000:.2f} mm")
        print(f"  Index IK residual : {index_res*1000:.2f} mm")

        _assert_frame(joint_cfg, thumb_res, index_res, frame_idx)
        print(f"  Assertions passed.")

        out_path = _export_frame(robot, frame["T_wrist"], joint_cfg, frame_idx)
        out_paths.append(out_path)
        print(f"  GLB exported: {out_path}")

        prev_q_thumb = np.array([joint_cfg[j] for j in _THUMB_CHAIN])
        prev_q_index = np.array([joint_cfg[j] for j in _INDEX_CHAIN])

    print(f"\n{'='*60}")
    print(f"All assertions passed for frames {args.frames}.")
    print(f"\nGLBs written to:")
    for p in out_paths:
        print(f"  {p}")
    print(f"\nOpen with:  f3d {out_paths[0]}")
    print(f"\n*** STOP. Review the GLBs above visually. ***")
    print(f"*** Only run align_sequence.py after confirming the ***")
    print(f"*** glove alignment looks correct. ***")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
