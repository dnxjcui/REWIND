"""MuJoCo bring-up CLI for glove visualization + fingertip IK checks.

This is a quick operational entrypoint for device-side validation:
1) load MuJoCo model from project URDF/meshes,
2) export a rest-pose GLB for visual sanity checking,
3) run one-step IK toward fingertip targets (optional),
4) write a compact JSON report with residuals and solved joint angles.

Usage examples:
    python glove_sim/mujoco_device_setup.py
    python glove_sim/mujoco_device_setup.py --export-glb glove_sim/outputs/mujoco_setup/rest.glb
    python glove_sim/mujoco_device_setup.py --thumb-target "0.02,0.01,-0.04" --index-target "0.03,0.01,-0.05"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from urdfpy import URDF

# Keep script runnable from repo root or from glove_sim directory.
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.glove_ik import GloveSimulator
from src.visualize import export_glove_only_glb

ROOT = Path(__file__).resolve().parent.parent
URDFPY_NATIVE_URDF = ROOT / "rewind_glove_assembly/urdf/rewind_glove_for_urdfpy.urdf"


def parse_vec3(text: str) -> np.ndarray:
    """Parse CSV xyz triple into a float64 vector."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected 3 comma-separated numbers: x,y,z")
    try:
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric vector '{text}'") from exc


def _tip_world_positions(sim: GloveSimulator) -> tuple[np.ndarray, np.ndarray]:
    """Return (thumb_tip, index_tip) site positions in world frame."""
    mujoco.mj_kinematics(sim.model, sim.data)
    thumb = sim.data.site_xpos[sim.thumb_tip_site].copy()
    index = sim.data.site_xpos[sim.index_tip_site].copy()
    return thumb, index


def _export_rest_glb(sim: GloveSimulator, out_path: Path) -> None:
    """Export rest pose using the same mesh-placement path as diagnostic.py."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_forward(sim.model, sim.data)
    geom_poses = sim.get_geom_world_poses()
    export_glove_only_glb(
        geom_poses,
        out_path,
        mesh_dir=cfg.MESH_DIR,
    )


def _export_native_rest_glb(out_path: Path) -> None:
    """Export native URDFpy rest GLB using the notebook FK semantics."""
    robot = URDF.load(str(URDFPY_NATIVE_URDF))
    meshes = []
    for mesh, pose in robot.visual_trimesh_fk(cfg={}).items():
        m = mesh.copy()
        m.apply_transform(pose)
        meshes.append(m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Scene(meshes).export(str(out_path))


def _scene_metrics(scene: trimesh.Scene) -> dict:
    geoms = []
    for _, geom in scene.geometry.items():
        geoms.append(
            {
                "n_vertices": int(len(geom.vertices)),
                "n_faces": int(len(geom.faces)),
                "centroid": geom.centroid.tolist(),
                "extents": geom.extents.tolist(),
            }
        )
    bounds = scene.bounds if scene.bounds is not None else np.zeros((2, 3))
    return {"n_geom": len(geoms), "bounds_min": bounds[0].tolist(), "bounds_max": bounds[1].tolist(), "geom": geoms}


def _geom_buckets(metric: dict, round_decimals: int = 7) -> dict:
    buckets: dict[tuple[int, int, tuple[float, float, float]], list[dict]] = {}
    for g in metric["geom"]:
        key = (g["n_vertices"], g["n_faces"], tuple(np.round(np.array(g["extents"]), round_decimals).tolist()))
        buckets.setdefault(key, []).append(g)
    for vals in buckets.values():
        vals.sort(key=lambda x: tuple(np.round(np.array(x["centroid"]), 9).tolist()))
    return buckets


def _compare_scene_metrics(native_metric: dict, custom_metric: dict, atol: float) -> tuple[bool, dict]:
    diff: dict = {"ok": True, "messages": []}
    if native_metric["n_geom"] != custom_metric["n_geom"]:
        diff["ok"] = False
        diff["messages"].append(f"n_geom mismatch: {native_metric['n_geom']} vs {custom_metric['n_geom']}")
        return False, diff

    nmin = np.array(native_metric["bounds_min"])
    nmax = np.array(native_metric["bounds_max"])
    cmin = np.array(custom_metric["bounds_min"])
    cmax = np.array(custom_metric["bounds_max"])
    bdelta = max(float(np.max(np.abs(nmin - cmin))), float(np.max(np.abs(nmax - cmax))))
    diff["bounds_max_abs_delta"] = bdelta
    if bdelta > atol:
        diff["ok"] = False
        diff["messages"].append(f"bounds delta {bdelta:.3e} > atol {atol:.3e}")

    nb = _geom_buckets(native_metric)
    cb = _geom_buckets(custom_metric)
    if set(nb.keys()) != set(cb.keys()):
        diff["ok"] = False
        diff["messages"].append("geometry signature bucket keys mismatch")
        diff["native_bucket_keys"] = [str(k) for k in sorted(nb.keys())]
        diff["custom_bucket_keys"] = [str(k) for k in sorted(cb.keys())]
        return False, diff

    worst_centroid = 0.0
    for key in sorted(nb.keys()):
        if len(nb[key]) != len(cb[key]):
            diff["ok"] = False
            diff["messages"].append(f"bucket count mismatch for {key}: {len(nb[key])} vs {len(cb[key])}")
            continue
        for ng, cg in zip(nb[key], cb[key]):
            cdelta = float(np.max(np.abs(np.array(ng["centroid"]) - np.array(cg["centroid"]))))
            worst_centroid = max(worst_centroid, cdelta)
            if cdelta > atol:
                diff["ok"] = False
                diff["messages"].append(f"centroid delta {cdelta:.3e} > atol for bucket {key}")
                break
    diff["centroid_max_abs_delta"] = worst_centroid
    return diff["ok"], diff


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up and validate MuJoCo glove model for visualization + IK."
    )
    parser.add_argument(
        "--export-glb",
        type=Path,
        default=cfg.OUTPUT_DIR / "mujoco_setup" / "rest.glb",
        help="Output GLB path for rest pose export.",
    )
    parser.add_argument(
        "--thumb-target",
        type=parse_vec3,
        default=None,
        help="Optional fingertip target in world frame as 'x,y,z' (meters).",
    )
    parser.add_argument(
        "--index-target",
        type=parse_vec3,
        default=None,
        help="Optional fingertip target in world frame as 'x,y,z' (meters).",
    )
    parser.add_argument(
        "--ik-iters",
        type=int,
        default=cfg.IK_ITERS,
        help="Max IK iterations for one-shot solve.",
    )
    parser.add_argument(
        "--ik-tol",
        type=float,
        default=cfg.IK_TOL,
        help="IK convergence tolerance in meters.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=cfg.OUTPUT_DIR / "mujoco_setup" / "report.json",
        help="Where to write the setup report JSON.",
    )
    parser.add_argument(
        "--rest-atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for rest-scene parity checks.",
    )
    args = parser.parse_args()

    print("Loading MuJoCo glove model...")
    sim = GloveSimulator(URDFPY_NATIVE_URDF, cfg.MESH_DIR)
    print(f"Loaded: nq={sim.model.nq}, nv={sim.model.nv}, nbody={sim.model.nbody}, ngeom={sim.model.ngeom}")

    # Match native urdfpy FK export frame where URDF root is world identity.
    base_pos = np.array(cfg.ROOT_TO_HAND_MOUNT_XYZ, dtype=np.float64)
    sim.set_base_pose(base_pos, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    mujoco.mj_forward(sim.model, sim.data)

    print(f"Exporting rest GLBs -> {args.export_glb.parent}")
    mujoco_glb = args.export_glb
    native_glb = args.export_glb.parent / "native_rest.glb"
    _export_rest_glb(sim, mujoco_glb)
    _export_native_rest_glb(native_glb)

    thumb_curr, index_curr = _tip_world_positions(sim)
    thumb_target = args.thumb_target if args.thumb_target is not None else thumb_curr.copy()
    index_target = args.index_target if args.index_target is not None else index_curr.copy()

    print("Running one-shot IK solve...")
    residual = sim.solve_ik(
        thumb_target=thumb_target,
        index_target=index_target,
        n_iter=args.ik_iters,
        tol=args.ik_tol,
    )
    mujoco.mj_forward(sim.model, sim.data)
    thumb_after, index_after = _tip_world_positions(sim)

    report = {
        "ok": bool(residual <= args.ik_tol),
        "ik_residual_m": float(residual),
        "ik_tol_m": float(args.ik_tol),
        "model": {
            "nq": int(sim.model.nq),
            "nv": int(sim.model.nv),
            "nbody": int(sim.model.nbody),
            "ngeom": int(sim.model.ngeom),
        },
        "targets_world": {
            "thumb": thumb_target.tolist(),
            "index": index_target.tolist(),
        },
        "base_pose_world": {
            "xyz": base_pos.tolist(),
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "tips_before_world": {
            "thumb": thumb_curr.tolist(),
            "index": index_curr.tolist(),
        },
        "tips_after_world": {
            "thumb": thumb_after.tolist(),
            "index": index_after.tolist(),
        },
        "joint_cfg_rad": sim.read_all_joint_angles(),
        "mujoco_rest_glb": str(mujoco_glb.resolve()),
        "native_rest_glb": str(native_glb.resolve()),
    }

    loaded_native = trimesh.load(str(native_glb), force="scene")
    loaded_mujoco = trimesh.load(str(mujoco_glb), force="scene")
    native_metric = _scene_metrics(loaded_native)
    mujoco_metric = _scene_metrics(loaded_mujoco)
    parity_ok, parity_diff = _compare_scene_metrics(native_metric, mujoco_metric, atol=args.rest_atol)
    report["rest_parity"] = {
        "ok": bool(parity_ok),
        "atol": float(args.rest_atol),
        **parity_diff,
    }
    report["ok"] = bool(report["ok"] and parity_ok)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report -> {args.report}")
    print(f"IK residual: {residual:.6e} m")
    print(f"Rest parity ok: {parity_ok}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
