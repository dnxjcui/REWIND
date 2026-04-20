"""Quick native-vs-custom urdfpy visualization parity checker.

Usage:
    python glove_sim/compare_visual_parity.py
    python glove_sim/compare_visual_parity.py --transform-atol 1e-7 --snapshot-atol 1e-6
"""

from __future__ import annotations

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from urdfpy import URDF

ROOT = Path(__file__).resolve().parent.parent
ASSEMBLY_URDF = ROOT / "rewind_glove_assembly/urdf/rewind_glove_assembly.urdf"
ASSEMBLY_MESH_DIR = ROOT / "rewind_glove_assembly/meshes"


def _mesh_key(link_name: str, visual_idx: int, mesh_idx: int) -> str:
    return f"{link_name}:{visual_idx}:{mesh_idx}"


def _flatten_visual_fk(robot, cfg: dict[str, float]) -> dict[str, tuple[trimesh.Trimesh, np.ndarray]]:
    fk = robot.visual_trimesh_fk(cfg=cfg)
    out: dict[str, tuple[trimesh.Trimesh, np.ndarray]] = {}
    for link in sorted(robot.links, key=lambda l: l.name):
        for vidx, visual in enumerate(link.visuals):
            for midx, mesh in enumerate(visual.geometry.meshes):
                out[_mesh_key(link.name, vidx, midx)] = (mesh, fk[mesh])
    return out


def _scene_metrics(scene: trimesh.Scene) -> tuple[np.ndarray, np.ndarray]:
    bounds = scene.bounds if scene.bounds is not None else np.zeros((2, 3))
    return bounds[0], bounds[1]


def _prepare_native_oracle_from_canonical() -> URDF:
    """Native URDF.load with only minimal canonical-path sanitation."""
    tree = ET.parse(str(ASSEMBLY_URDF))
    urdf_dir = ASSEMBLY_URDF.parent.resolve()
    for mesh_el in tree.getroot().iter("mesh"):
        fn = mesh_el.get("filename", "")
        if fn.startswith("package://"):
            base = fn.split("/", 2)[-1]
            base = base.split("/", 1)[-1]
            base = base.split("/", 1)[-1]
            rel = os.path.relpath(ASSEMBLY_MESH_DIR / base, start=urdf_dir).replace("\\", "/")
            mesh_el.set("filename", rel)
    for material in tree.getroot().iter("material"):
        for tex in list(material.findall("texture")):
            if not tex.attrib or not tex.get("filename"):
                material.remove(tex)

    fd, tmp_path = tempfile.mkstemp(prefix="._urdfpy_native_", suffix=".urdf", dir=str(urdf_dir))
    os.close(fd)
    try:
        tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
        return URDF.load(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare native urdfpy and in-house visualization outputs.")
    parser.add_argument("--transform-atol", type=float, default=1e-8)
    parser.add_argument("--snapshot-atol", type=float, default=1e-7)
    args = parser.parse_args()

    # Import after path setup.
    import sys

    sys.path.insert(0, str(ROOT / "glove_sim"))
    from src.urdfpy_vis import get_glove_scene, load_robot

    # Use canonical URDF + minimal sanitation as oracle (same as parity tests).
    native = _prepare_native_oracle_from_canonical()
    custom = load_robot(ASSEMBLY_URDF, ASSEMBLY_MESH_DIR)

    cfg = {j.name: 0.0 for j in native.actuated_joints}
    cfg["revolute_3_0"] = np.deg2rad(45.0)
    cfg["revolute_9_0"] = np.deg2rad(45.0)

    nf = _flatten_visual_fk(native, cfg)
    cf = _flatten_visual_fk(custom, cfg)
    if set(nf.keys()) != set(cf.keys()):
        print("FAIL: mesh-key sets differ between native and custom FK.")
        return 1

    worst = 0.0
    worst_key = ""
    for key in sorted(nf):
        max_abs = float(np.max(np.abs(nf[key][1] - cf[key][1])))
        if max_abs > worst:
            worst = max_abs
            worst_key = key
    print(f"FK max abs transform delta: {worst:.3e} ({worst_key})")
    if worst > args.transform_atol:
        print(f"FAIL: FK delta exceeds --transform-atol ({args.transform_atol}).")
        return 1

    # Snapshot parity for scene output.
    t_hand_mount_world = np.eye(4)
    t_hand_mount_world[:3, 3] = np.array([-0.157876, 0.0663838, -0.0660817], dtype=float)
    scene_custom = get_glove_scene(custom, cfg, t_hand_mount_world)

    # Build native scene in same Y-up frame as get_glove_scene (root world is identity).
    ydown_to_yup = np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
    )
    meshes = []
    for _, (mesh, t_from_root) in sorted(nf.items()):
        m = mesh.copy()
        m.apply_transform(ydown_to_yup @ t_from_root)
        meshes.append(m)
    scene_native = trimesh.Scene(meshes)

    nmin, nmax = _scene_metrics(scene_native)
    cmin, cmax = _scene_metrics(scene_custom)
    bdelta = max(float(np.max(np.abs(nmin - cmin))), float(np.max(np.abs(nmax - cmax))))
    print(f"Scene bounds max delta: {bdelta:.3e}")
    if bdelta > args.snapshot_atol:
        print(f"FAIL: scene-bounds delta exceeds --snapshot-atol ({args.snapshot_atol}).")
        return 1

    print("PASS: native and in-house visualization outputs are within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

