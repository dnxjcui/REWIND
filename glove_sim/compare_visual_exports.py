"""Export and compare native-vs-in-house urdfpy GLBs (rest-first).

GLB export follows the same steps as the notebook: ``visual_trimesh_fk``,
``mesh.copy()``, ``apply_transform(pose)``, ``trimesh.Scene(meshes).export``.

**URDF choice:** ``rewind_glove_for_urdfpy.urdf`` is the notebook / native
reference. The checked-in ``rewind_glove_assembly.urdf`` can differ (e.g. visual
``origin rpy``), so comparing native-on-for_urdfpy to ``load_robot`` on the
assembly file will not match until those files agree.

Usage:
    python glove_sim/compare_visual_exports.py
    python glove_sim/compare_visual_exports.py --only-rest
    python glove_sim/compare_visual_exports.py --rest-atol 1e-8 --sweep-atol 1e-7
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from urdfpy import URDF

ROOT = Path(__file__).resolve().parent.parent
URDFPY_NATIVE_URDF = ROOT / "rewind_glove_assembly/urdf/rewind_glove_for_urdfpy.urdf"
ASSEMBLY_MESH_DIR = ROOT / "rewind_glove_assembly/meshes"


def _scene_from_visual_fk(robot: URDF, cfg: dict[str, float] | None = None) -> trimesh.Scene:
    """Build GLB scene the same way as the notebook: visual_trimesh_fk + copy + apply_transform."""
    fk = robot.visual_trimesh_fk(cfg=cfg) if cfg is not None else robot.visual_trimesh_fk()
    meshes = []
    for mesh, pose in fk.items():
        m = mesh.copy()
        m.apply_transform(pose)
        meshes.append(m)
    return trimesh.Scene(meshes)


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


def _config_set(native_robot: URDF) -> list[tuple[str, dict[str, float]]]:
    base = {j.name: 0.0 for j in native_robot.actuated_joints}
    out: list[tuple[str, dict[str, float]]] = []
    out.append(("rest", dict(base)))
    c = dict(base)
    c["revolute_3_0"] = np.deg2rad(45.0)
    out.append(("thumb45", c))
    c = dict(base)
    c["revolute_9_0"] = np.deg2rad(45.0)
    out.append(("index45", c))
    c = dict(base)
    c["revolute_3_0"] = np.deg2rad(45.0)
    c["revolute_9_0"] = np.deg2rad(45.0)
    out.append(("combo45", c))
    for deg in range(0, 91, 15):
        c = dict(base)
        c["revolute_9_0"] = np.deg2rad(float(deg))
        out.append((f"index_sweep_{deg:03d}", c))
    return out


def run_parity_export_compare(
    out_dir: Path,
    rest_atol: float = 1e-8,
    sweep_atol: float = 1e-7,
    only_rest: bool = False,
) -> dict:
    """Run rest-first export/compare and return detailed report."""
    import sys

    sys.path.insert(0, str(ROOT / "glove_sim"))
    from src.urdfpy_vis import load_robot

    # Ground truth must use the exact same native loading path the notebook uses.
    if not URDFPY_NATIVE_URDF.is_file():
        raise FileNotFoundError(f"Native URDF not found: {URDFPY_NATIVE_URDF}")
    native = URDF.load(str(URDFPY_NATIVE_URDF))
    # Same tree as native so FK/GLB match; load_robot still exercises temp URDF + cache path.
    custom = load_robot(URDFPY_NATIVE_URDF, ASSEMBLY_MESH_DIR)

    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"ok": True, "rest_ok": None, "results": []}
    for name, cfg in _config_set(native):
        is_rest = name == "rest"
        atol = rest_atol if is_rest else sweep_atol

        case_dir = out_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        native_glb = case_dir / "native.glb"
        custom_glb = case_dir / "custom.glb"

        # Compare direct urdfpy visual FK export on both sides (same frame semantics).
        scene_native = _scene_from_visual_fk(native, cfg)
        scene_custom = _scene_from_visual_fk(custom, cfg)
        scene_native.export(str(native_glb))
        scene_custom.export(str(custom_glb))

        loaded_native = trimesh.load(str(native_glb), force="scene")
        loaded_custom = trimesh.load(str(custom_glb), force="scene")

        native_metric = _scene_metrics(loaded_native)
        custom_metric = _scene_metrics(loaded_custom)
        ok, diff = _compare_scene_metrics(native_metric, custom_metric, atol)
        diff["config"] = name
        diff["atol"] = atol
        diff["native_glb"] = str(native_glb)
        diff["custom_glb"] = str(custom_glb)
        report["results"].append(diff)

        (case_dir / "diff.json").write_text(json.dumps(diff, indent=2), encoding="utf-8")
        if is_rest:
            report["rest_ok"] = ok
        if not ok:
            report["ok"] = False
            if is_rest:
                report["fail_stage"] = "rest"
                break
            if only_rest:
                break
        if only_rest and is_rest:
            break

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Rest-first native vs in-house GLB export/compare.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "glove_sim/outputs/parity_exports")
    parser.add_argument("--rest-atol", type=float, default=1e-8)
    parser.add_argument("--sweep-atol", type=float, default=1e-7)
    parser.add_argument("--only-rest", action="store_true")
    args = parser.parse_args()

    report = run_parity_export_compare(
        out_dir=args.out_dir,
        rest_atol=args.rest_atol,
        sweep_atol=args.sweep_atol,
        only_rest=args.only_rest,
    )
    if report["rest_ok"] is False:
        print("FAIL: rest pose parity failed. Sweep was skipped.")
    elif report["ok"]:
        print("PASS: rest pose and small sweep parity checks are green.")
    else:
        print("FAIL: sweep mismatch after rest passed.")
    print(f"Report: {args.out_dir / 'report.json'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
