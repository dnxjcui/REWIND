"""Align glove hand-mount to knot_one_handed right hand keyframes.

This pass focuses on base alignment only (no collision APF yet).
It optimizes T_wrist_to_base so hand-mount placement and XL linkage horn
position are consistent with right-hand MANO landmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import trimesh

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.calibration import mat_to_quat_wxyz, pose_from_rotvec
from src.glove_ik import GloveSimulator


def load_annotations(path: Path) -> dict:
    """Load and validate annotation JSON for manual flush alignment."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "reference_frame_idx", "hand_points", "glove_points"}
    missing = required.difference(data.keys())
    if missing:
        raise ValueError(f"Annotation file missing keys: {sorted(missing)}")
    hand = np.asarray(data["hand_points"], dtype=float)
    glove = np.asarray(data["glove_points"], dtype=float)
    if hand.ndim != 2 or hand.shape[1] != 3:
        raise ValueError("hand_points must be Nx3")
    if glove.ndim != 2 or glove.shape[1] != 3:
        raise ValueError("glove_points must be Nx3")
    if len(hand) < 3 or len(glove) < 3:
        raise ValueError("Need at least 3 points for each region")
    if len(hand) != len(glove):
        raise ValueError("hand_points and glove_points must have equal length")
    out = dict(data)
    out["hand_points"] = hand
    out["glove_points"] = glove
    return out


def validate_connectivity_graph(parents: dict[str, str | None], required_bodies: list[str]) -> None:
    """Validate each required body is connected to hand_mount in parent graph."""
    root = "hand_mount"
    for body in required_bodies:
        if body not in parents:
            raise ValueError(f"Required body missing from graph: {body}")
        seen = set()
        cur = body
        while cur is not None and cur not in seen:
            if cur == root:
                break
            seen.add(cur)
            cur = parents.get(cur)
        if cur != root:
            raise ValueError(f"Body not connected to {root}: {body}")


def _model_parent_graph(model) -> dict[str, str | None]:
    """Build body->parent_name map from MuJoCo model."""
    parents: dict[str, str | None] = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        pid = int(model.body_parentid[bid])
        pname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, pid) if pid >= 0 else None
        if name == "world":
            pname = None
        parents[name] = pname
    return parents


def _enforce_model_connectivity(sim: GloveSimulator) -> None:
    """Hard gate: all linkage bodies must remain connected to hand_mount."""
    parents = _model_parent_graph(sim.model)
    required = sorted(set(cfg.LINK_TO_STL.keys()))
    validate_connectivity_graph(parents, required_bodies=required)


def _kabsch_transform(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute rigid transform that maps src_points to dst_points."""
    src = np.asarray(src_points, dtype=float)
    dst = np.asarray(dst_points, dtype=float)
    c_src = np.mean(src, axis=0)
    c_dst = np.mean(dst, axis=0)
    xs = src - c_src
    yd = dst - c_dst
    h = xs.T @ yd
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = c_dst - r @ c_src
    out = np.eye(4, dtype=float)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def _compute_frame_metrics(
    hand_pos: np.ndarray,
    hand_z_axis: np.ndarray,
    horn_pos: np.ndarray,
    wrist: np.ndarray,
    index_knuckle: np.ndarray,
    pinky_knuckle: np.ndarray,
    contact_points_world: np.ndarray | None = None,
    hand_vertices: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute alignment metrics for one frame."""
    # Dorsum proxy plane from wrist/index_mcp/pinky_mcp triangle.
    v1 = index_knuckle - wrist
    v2 = pinky_knuckle - wrist
    normal = _safe_normalize(np.cross(v1, v2))
    dorsum_point = (wrist + index_knuckle + pinky_knuckle) / 3.0

    # Flip normal to face the hand-mount location for stable signed distance.
    if float(np.dot(hand_pos - dorsum_point, normal)) < 0.0:
        normal = -normal

    if contact_points_world is None or len(contact_points_world) == 0:
        signed_dist = float(np.dot(hand_pos - dorsum_point, normal))
        flush_distance = abs(signed_dist)
        flush_max = flush_distance
    else:
        signed_d = np.dot(contact_points_world - dorsum_point[None, :], normal)
        signed_dist = float(np.mean(signed_d))
        if hand_vertices is not None and len(hand_vertices) > 0:
            d = np.linalg.norm(
                contact_points_world[:, None, :] - np.asarray(hand_vertices, dtype=float)[None, :, :],
                axis=2,
            )
            nearest = np.min(d, axis=1)
            nearest = np.maximum(nearest - 0.002, 0.0)
            # Flush metric emphasizes the nearest underside contact point and
            # a robust spread (75th percentile) instead of single-point outliers.
            flush_distance = float(np.min(nearest))
            flush_max = float(np.percentile(nearest, 75))
        else:
            flush_distance = float(np.mean(np.abs(signed_d)))
            flush_max = float(np.max(np.abs(signed_d)))

    # Orientation target: hand mount "underside normal" ~ dorsum normal.
    # Use body local +Z as a practical reference in this model.
    z = _safe_normalize(hand_z_axis)
    orientation_error = 1.0 - abs(float(np.dot(z, normal)))

    horn_to_knuckle = float(np.linalg.norm(horn_pos - index_knuckle))
    return {
        "flush_distance_m": flush_distance,
        "flush_max_distance_m": flush_max,
        "signed_plane_distance_m": signed_dist,
        "orientation_error": orientation_error,
        "horn_to_knuckle_distance_m": horn_to_knuckle,
    }


def _sample_sparse_keyframes(n_total: int, n_keyframes: int) -> list[int]:
    """Deterministic sparse sampler biased away from unstable sequence ends."""
    if n_total <= 0:
        return []
    n_keyframes = max(1, min(int(n_keyframes), n_total))
    lo = int(0.1 * (n_total - 1))
    hi = int(0.9 * (n_total - 1))
    if hi <= lo:
        lo, hi = 0, n_total - 1
    return np.linspace(lo, hi, num=n_keyframes, dtype=int).tolist()


def _load_hand_mount_contact_points_local(mesh_dir: Path) -> np.ndarray:
    """Approximate underside contact patch points from Hand Mount STL."""
    stl = mesh_dir / "Hand Mount.stl"
    if not stl.is_file():
        return np.zeros((0, 3), dtype=float)
    mesh = trimesh.load(str(stl), force="mesh")
    verts = np.asarray(mesh.vertices, dtype=float)
    if verts.size == 0:
        return np.zeros((0, 3), dtype=float)
    # Contact patch is side opposite XL housing; empirically near max Y on this part.
    ymax = float(np.max(verts[:, 1]))
    sel = verts[verts[:, 1] >= (ymax - 0.0015)]
    if len(sel) < 16:
        sel = verts
    # Deterministic compact subset: centroid plus extrema on x/z.
    c = np.mean(sel, axis=0, keepdims=True)
    x_min = sel[np.argmin(sel[:, 0])][None, :]
    x_max = sel[np.argmax(sel[:, 0])][None, :]
    z_min = sel[np.argmin(sel[:, 2])][None, :]
    z_max = sel[np.argmax(sel[:, 2])][None, :]
    return np.vstack([c, x_min, x_max, z_min, z_max])


def _build_t_wrist_to_base(param: np.ndarray) -> np.ndarray:
    """Build 4x4 transform from 6-vector [tx,ty,tz,rx,ry,rz]."""
    t = np.eye(4, dtype=float)
    t[:3, 3] = param[:3]
    rot = pose_from_rotvec(param[3:], np.zeros(3, dtype=float))
    t[:3, :3] = rot[:3, :3]
    return t


def _resolve_npz(default_npz: Path) -> Path:
    if default_npz.is_file():
        return default_npz
    base = cfg.DYNHAMR_ROOT / "outputs/logs/video-custom"
    matches = sorted(base.rglob("knot_one_handed_*_world_results.npz"))
    if not matches:
        raise FileNotFoundError(f"No knot_one_handed world_results npz found under {base}")
    return matches[-1]


def _resolve_mano_dir(default_mano_dir: Path) -> Path:
    if default_mano_dir.is_dir():
        return default_mano_dir
    alt = cfg.DYNHAMR_ROOT / "_DATA/_DATA/data"
    if alt.is_dir():
        return alt
    raise FileNotFoundError(f"Could not find MANO data dir at {default_mano_dir} or {alt}")


def _resolve_hand_glb_dir() -> Path | None:
    base = cfg.DYNHAMR_ROOT / "outputs/logs/video-custom"
    candidates = sorted(base.rglob("unity_export/frames"))
    return candidates[-1] if candidates else None


def _extract_frame_metric(
    sim: GloveSimulator,
    traj: dict,
    frame_idx: int,
    t_wrist_to_base: np.ndarray,
    contact_points_local: np.ndarray,
) -> tuple[dict[str, float], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Set base pose from MANO frame and return alignment metrics + geom poses."""
    root_orient = traj["root_orient"][frame_idx]
    trans = traj["trans"][frame_idx]
    joints = traj["joints"][frame_idx]
    vertices = traj["vertices"][frame_idx]

    # MANO 16-joint ordering assumption: 0 wrist, 1 index_mcp, 7 pinky_mcp.
    wrist = joints[0].astype(np.float64)
    index_knuckle = joints[1].astype(np.float64)
    pinky_knuckle = joints[7].astype(np.float64)

    t_wrist = pose_from_rotvec(root_orient, trans)
    t_base = t_wrist @ t_wrist_to_base
    pos_base = t_base[:3, 3].astype(np.float64)
    quat_base = mat_to_quat_wxyz(t_base[:3, :3]).astype(np.float64)

    mujoco.mj_resetData(sim.model, sim.data)
    sim.set_base_pose(pos_base, quat_base)
    mujoco.mj_forward(sim.model, sim.data)

    hand_t = sim.get_hand_mount_world_pose()
    hand_pos = hand_t[:3, 3]
    hand_z = hand_t[:3, 2]
    if len(contact_points_local) > 0:
        cp_h = np.hstack([contact_points_local, np.ones((len(contact_points_local), 1), dtype=float)])
        cp_world = (hand_t @ cp_h.T).T[:, :3]
    else:
        cp_world = np.zeros((0, 3), dtype=float)

    horn_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "xl_linkage_horn_1")
    if horn_bid < 0:
        raise RuntimeError("Body 'xl_linkage_horn_1' not found in MuJoCo model")
    horn_pos = sim.data.xpos[horn_bid].copy()

    metrics = _compute_frame_metrics(
        hand_pos=hand_pos,
        hand_z_axis=hand_z,
        horn_pos=horn_pos,
        wrist=wrist,
        index_knuckle=index_knuckle,
        pinky_knuckle=pinky_knuckle,
        contact_points_world=cp_world,
        hand_vertices=vertices,
    )
    return metrics, sim.get_geom_world_poses()


def run_alignment(
    npz_path: Path,
    mano_dir: Path,
    out_dir: Path,
    max_frames: int = 5,
    max_iters: int = 120,
    annotations: Path | None = None,
) -> dict:
    """Run knot alignment optimization and emit report + overlays."""
    from src.mano_loader import load_mano_trajectory
    from src.visualize import export_frame_glb, export_glove_only_glb

    traj = load_mano_trajectory(npz_path=npz_path, mano_dir=mano_dir)
    sim = GloveSimulator(cfg.URDF_PATH, cfg.MESH_DIR)
    _enforce_model_connectivity(sim)
    contact_points_local = _load_hand_mount_contact_points_local(cfg.MESH_DIR)

    n_total = int(traj["T"])
    if n_total <= 0:
        raise RuntimeError("Trajectory has zero frames")
    frame_indices = _sample_sparse_keyframes(n_total=n_total, n_keyframes=max_frames)

    # Init with current default calibration.
    init_t = np.load(cfg.CALIBRATION_PATH) if cfg.CALIBRATION_PATH.is_file() else np.eye(4, dtype=float)
    if annotations is not None and Path(annotations).is_file():
        ann = load_annotations(Path(annotations))
        ref_idx = int(ann["reference_frame_idx"])
        ref_idx = max(0, min(ref_idx, n_total - 1))
        # Build current hand_mount world pose at reference frame.
        t_wrist_ref = pose_from_rotvec(traj["root_orient"][ref_idx], traj["trans"][ref_idx])
        t_base_ref = t_wrist_ref @ init_t
        sim.set_base_pose(t_base_ref[:3, 3], mat_to_quat_wxyz(t_base_ref[:3, :3]))
        mujoco.mj_forward(sim.model, sim.data)
        hand_t_ref = sim.get_hand_mount_world_pose()
        # Annotation-driven delta in world space maps glove patch -> hand patch.
        delta = _kabsch_transform(ann["glove_points"], ann["hand_points"])
        desired_hand_t = delta @ hand_t_ref
        init_t = np.linalg.inv(t_wrist_ref) @ desired_hand_t

    init = np.zeros(6, dtype=float)
    init[:3] = init_t[:3, 3]
    init[3:] = Rotation.from_matrix(init_t[:3, :3]).as_rotvec()

    def objective(param: np.ndarray) -> float:
        t_wrist_to_base = _build_t_wrist_to_base(param)
        total = 0.0
        for fi in frame_indices:
            m, _ = _extract_frame_metric(sim, traj, fi, t_wrist_to_base, contact_points_local)
            total += (
                120.0 * m["flush_distance_m"]
                + 500.0 * m["flush_max_distance_m"]
            )
        return float(total / len(frame_indices))

    res = minimize(objective, init, method="Powell", options={"maxiter": int(max_iters), "xtol": 1e-7, "ftol": 1e-7})
    opt_t = _build_t_wrist_to_base(res.x)

    out_dir.mkdir(parents=True, exist_ok=True)
    hand_glb_dir = _resolve_hand_glb_dir()
    per_frame = []
    for fi in frame_indices:
        m, geom_poses = _extract_frame_metric(sim, traj, fi, opt_t, contact_points_local)
        per_frame.append({"frame_idx": int(fi), **m})
        out_glb = out_dir / f"alignment_frame_{fi:06d}.glb"
        if hand_glb_dir is not None:
            export_frame_glb(frame_idx=fi, geom_poses=geom_poses, out_path=out_glb, mesh_dir=cfg.MESH_DIR, glb_dir=hand_glb_dir)
        else:
            export_glove_only_glb(geom_poses=geom_poses, out_path=out_glb, mesh_dir=cfg.MESH_DIR)

    flush = np.array([x["flush_distance_m"] for x in per_frame], dtype=float)
    orient = np.array([x["orientation_error"] for x in per_frame], dtype=float)
    horn = np.array([x["horn_to_knuckle_distance_m"] for x in per_frame], dtype=float)

    report = {
        "npz_path": str(npz_path),
        "mano_dir": str(mano_dir),
        "frames": per_frame,
        "optimized_t_wrist_to_base": opt_t.tolist(),
        "optimization": {"success": bool(res.success), "message": str(res.message), "fun": float(res.fun), "nit": int(res.nit)},
        "summary": {
            "flush_mean_m": float(flush.mean()),
            "flush_max_m": float(flush.max()),
            "flush_target_max_m": 0.005,
            "flush_target_ok": bool(float(flush.max()) <= 0.005),
            "orientation_mean": float(orient.mean()),
            "orientation_max": float(orient.max()),
            "horn_knuckle_mean_m": float(horn.mean()),
            "horn_knuckle_max_m": float(horn.max()),
        },
        "model_integrity": {
            "nbody": int(sim.model.nbody),
            "ngeom": int(sim.model.ngeom),
            "njnt": int(sim.model.njnt),
            "frame_count": int(len(frame_indices)),
            "connectivity_ok": True,
        },
    }
    if annotations is not None:
        report["annotations"] = str(Path(annotations))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Align glove hand-mount to knot_one_handed right hand.")
    parser.add_argument("--npz", type=Path, default=cfg.NPZ_PATH)
    parser.add_argument("--mano-dir", type=Path, default=cfg.MANO_DIR)
    parser.add_argument("--out-dir", type=Path, default=cfg.OUTPUT_DIR / "knot_alignment")
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--max-iters", type=int, default=1200)
    parser.add_argument("--annotations", type=Path, default=None, help="Path to annotation JSON from flush annotator")
    args = parser.parse_args()

    npz = _resolve_npz(args.npz)
    mano_dir = _resolve_mano_dir(args.mano_dir)
    report = run_alignment(
        npz_path=npz,
        mano_dir=mano_dir,
        out_dir=args.out_dir,
        max_frames=max(1, int(args.max_frames)),
        max_iters=max(10, int(args.max_iters)),
        annotations=args.annotations,
    )
    print(f"Alignment report: {args.out_dir / 'report.json'}")
    print(
        "Summary:",
        f"flush_mean={report['summary']['flush_mean_m']:.6f} m,",
        f"horn_knuckle_mean={report['summary']['horn_knuckle_mean_m']:.6f} m,",
        f"orientation_mean={report['summary']['orientation_mean']:.6f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
