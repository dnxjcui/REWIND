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
# Dome-tip centroid in link-local frame, computed from STL vertex data
# (centroid of vertices within 3 mm of the minimum-Z extreme = physical cap dome).
# The URDF <visual><origin> was the STL frame origin, which sits ~12 cm outside
# the actual mesh bounds and was causing the IK to drive a phantom point.
_THUMB_VISUAL_ORIGIN = np.array([-0.00047,  0.0, -0.03095])
_INDEX_VISUAL_ORIGIN = np.array([-0.00259,  0.0, -0.03085])

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
    # Apply calibrated fingertip offsets (wrist-local → world)
    R_w = T_wrist[:3, :3]
    thumb_tip_world = thumb_tip_world + R_w @ cfg.THUMB_TIP_OFFSET
    index_tip_world = index_tip_world + R_w @ cfg.INDEX_TIP_OFFSET

    # Apply calibrated wrist→hand_mount offset, then hand_mount→root
    T_root_world = T_wrist @ cfg.T_WRIST_TO_HM @ _HM_TO_ROOT
    inv_T = np.linalg.inv(T_root_world)

    # Note: MANO thumb (v745) → index chain (part_3_1); MANO index (v317) → thumb chain (part_3).
    # The glove's physical thumb cap is on the part_3_1/INDEX_CHAIN side and vice versa.
    thumb_target = (inv_T @ np.append(thumb_tip_world, 1.0))[:3]
    index_target = (inv_T @ np.append(index_tip_world, 1.0))[:3]

    thumb_cfg, thumb_res = _ik_finger(
        robot, _THUMB_CHAIN, 'part_3',   _THUMB_VISUAL_ORIGIN, index_target, prev_q_thumb)
    index_cfg, index_res = _ik_finger(
        robot, _INDEX_CHAIN, 'part_3_1', _INDEX_VISUAL_ORIGIN, thumb_target, prev_q_index)

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

    T_hand_mount = T_wrist @ cfg.T_WRIST_TO_HM
    glove_scene = get_glove_scene(robot, joint_cfg, T_hand_mount)
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


def _total_frames() -> int:
    data = np.load(str(cfg.NPZ_PATH), allow_pickle=False)
    return int(data["trans"].shape[1])


def main():
    import argparse
    from src.urdfpy_vis import load_robot
    from src.mano_io import load_frame

    parser = argparse.ArgumentParser(
        description="Align glove URDF onto DynHaMR hand frames and export GLBs."
    )
    parser.add_argument(
        "--frames", nargs="+", type=int, default=None,
        help="Explicit list of frame indices to process. Mutually exclusive with --frames-test.",
    )
    parser.add_argument(
        "--frames-test", action="store_true",
        help="Quick test: run 10 evenly-spaced frames with strict assertions (no skipping).",
    )
    args = parser.parse_args()

    if args.frames is not None and args.frames_test:
        parser.error("--frames and --frames-test are mutually exclusive.")

    n_total = _total_frames()

    if args.frames_test:
        indices = [int(round(i * (n_total - 1) / 9)) for i in range(10)]
        strict  = True
        print(f"Test mode: {len(indices)} evenly-spaced frames out of {n_total} total.")
    elif args.frames is not None:
        indices = args.frames
        strict  = True
        print(f"Explicit frames: {indices}")
    else:
        indices = list(range(n_total))
        strict  = False
        print(f"Full sequence: {n_total} frames — failures will be skipped.")

    robot = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)
    print(f"URDF loaded: {len(robot.links)} links, {len(robot.actuated_joints)} actuated joints")

    prev_q_thumb = None
    prev_q_index = None
    out_paths    = []
    failed       = []

    for frame_idx in indices:
        try:
            frame = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, frame_idx)

            joint_cfg, thumb_res, index_res = solve_ik_frame(
                robot, frame["T_wrist"], frame["thumb_tip"], frame["index_tip"],
                prev_q_thumb=prev_q_thumb, prev_q_index=prev_q_index,
            )

            if strict:
                _assert_frame(joint_cfg, thumb_res, index_res, frame_idx)
                print(f"  [{frame_idx:05d}] thumb {thumb_res*1000:.1f}mm  "
                      f"index {index_res*1000:.1f}mm  OK")
            else:
                _assert_frame(joint_cfg, thumb_res, index_res, frame_idx)
                if frame_idx % 100 == 0:
                    print(f"  [{frame_idx:05d}] thumb {thumb_res*1000:.1f}mm  "
                          f"index {index_res*1000:.1f}mm")

            out_path = _export_frame(robot, frame["T_wrist"], joint_cfg, frame_idx)
            out_paths.append(out_path)

            prev_q_thumb = np.array([joint_cfg[j] for j in _THUMB_CHAIN])
            prev_q_index = np.array([joint_cfg[j] for j in _INDEX_CHAIN])

        except Exception as exc:
            if strict:
                raise
            print(f"  [SKIP] Frame {frame_idx}: {exc}")
            failed.append(frame_idx)
            prev_q_thumb = None
            prev_q_index = None

    print(f"\n{'='*60}")
    print(f"Done. {len(out_paths)} frames exported, {len(failed)} skipped.")
    if failed:
        print(f"Failed frames: {failed}")
    if out_paths:
        print(f"\nFirst GLB: {out_paths[0]}")
        if strict:
            print(f"\n*** STOP. Review the GLBs above visually before processing the full sequence. ***")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
