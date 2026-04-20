import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from urdfpy import URDF


REPO_ROOT = Path(__file__).resolve().parents[2]
URDFPY_NATIVE_URDF = REPO_ROOT / "rewind_glove_assembly/urdf/rewind_glove_for_urdfpy.urdf"


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


class TestMuJoCoUrdfpyRestParity(unittest.TestCase):
    def test_mujoco_rest_glb_matches_native_urdfpy_rest(self) -> None:
        import sys

        sys.path.insert(0, str(REPO_ROOT / "glove_sim"))
        import config as cfg
        from src.glove_ik import GloveSimulator
        from src.visualize import export_glove_only_glb

        native_robot = URDF.load(str(URDFPY_NATIVE_URDF))
        native_meshes = []
        for mesh, pose in native_robot.visual_trimesh_fk(cfg={}).items():
            m = mesh.copy()
            m.apply_transform(pose)
            native_meshes.append(m)
        native_scene = trimesh.Scene(native_meshes)

        sim = GloveSimulator(URDFPY_NATIVE_URDF, cfg.MESH_DIR)
        sim.set_base_pose(
            np.array(cfg.ROOT_TO_HAND_MOUNT_XYZ, dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        mujoco.mj_forward(sim.model, sim.data)
        geom_poses = sim.get_geom_world_poses()

        with tempfile.TemporaryDirectory() as tmp:
            native_path = Path(tmp) / "native_rest.glb"
            out_path = Path(tmp) / "mujoco_rest.glb"
            native_scene.export(str(native_path))
            export_glove_only_glb(geom_poses=geom_poses, out_path=out_path, mesh_dir=cfg.MESH_DIR)
            native_scene = trimesh.load(str(native_path), force="scene")
            mujoco_scene = trimesh.load(str(out_path), force="scene")

        native_metric = _scene_metrics(native_scene)
        mujoco_metric = _scene_metrics(mujoco_scene)
        ok, diff = _compare_scene_metrics(native_metric, mujoco_metric, atol=1e-8)
        self.assertTrue(ok, f"MuJoCo rest export must match URDFpy native rest export: {diff}")


if __name__ == "__main__":
    unittest.main()
