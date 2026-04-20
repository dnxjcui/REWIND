import json
import os
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from urdfpy import URDF


REPO_ROOT = Path(__file__).resolve().parents[2]
GLOVE_SIM_ROOT = REPO_ROOT / "glove_sim"
ASSEMBLY_URDF = REPO_ROOT / "rewind_glove_assembly/urdf/rewind_glove_assembly.urdf"
ASSEMBLY_MESH_DIR = REPO_ROOT / "rewind_glove_assembly/meshes"

YDOWN_TO_YUP = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=float,
)
ROOT_TO_HAND_MOUNT = np.array([-0.157876, 0.0663838, -0.0660817], dtype=float)


def _tol(env: str, default: float) -> float:
    return float(os.environ.get(env, default))


TRANSFORM_ATOL = _tol("PARITY_TRANSFORM_ATOL", 1e-8)
GEOM_ATOL = _tol("PARITY_GEOM_ATOL", 1e-8)
SCENE_ATOL = _tol("PARITY_SCENE_ATOL", 1e-8)
SNAPSHOT_ATOL = _tol("PARITY_SNAPSHOT_ATOL", 1e-7)


def _mesh_key(link_name: str, visual_idx: int, mesh_idx: int) -> str:
    return f"{link_name}:{visual_idx}:{mesh_idx}"


def _flatten_visual_fk(robot, cfg: dict[str, float]) -> dict[str, tuple[trimesh.Trimesh, np.ndarray]]:
    """Deterministic map: link/visual/mesh index -> (mesh, transform)."""
    fk = robot.visual_trimesh_fk(cfg=cfg)
    out: dict[str, tuple[trimesh.Trimesh, np.ndarray]] = {}
    for link in sorted(robot.links, key=lambda l: l.name):
        for vidx, visual in enumerate(link.visuals):
            for midx, mesh in enumerate(visual.geometry.meshes):
                key = _mesh_key(link.name, vidx, midx)
                out[key] = (mesh, fk[mesh])
    return out


def _mesh_metrics(mesh: trimesh.Trimesh) -> dict[str, list[float] | int]:
    return {
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "centroid": mesh.centroid.tolist(),
        "extents": mesh.extents.tolist(),
        "bounds_min": mesh.bounds[0].tolist(),
        "bounds_max": mesh.bounds[1].tolist(),
    }


def _scene_metrics(scene: trimesh.Scene) -> dict:
    geoms = [_mesh_metrics(geom) for _, geom in scene.geometry.items()]
    bounds = scene.bounds if scene.bounds is not None else np.zeros((2, 3))
    return {
        "n_geom": len(scene.geometry),
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "geom": geoms,
    }


def _scene_geom_buckets(scene_metric: dict) -> dict[tuple[int, int, tuple[float, float, float]], list[dict]]:
    buckets: dict[tuple[int, int, tuple[float, float, float]], list[dict]] = {}
    for g in scene_metric["geom"]:
        key = (
            g["n_vertices"],
            g["n_faces"],
            tuple(np.round(np.array(g["extents"]), 7).tolist()),
        )
        buckets.setdefault(key, []).append(g)
    for vals in buckets.values():
        vals.sort(key=lambda x: tuple(np.round(np.array(x["centroid"]), 9).tolist()))
    return buckets


def _scene_from_fk_entries(
    fk_entries: dict[str, tuple[trimesh.Trimesh, np.ndarray]],
    t_root_world_yd: np.ndarray,
) -> trimesh.Scene:
    meshes = []
    for _, (mesh, t_from_root) in sorted(fk_entries.items()):
        t_world_yd = t_root_world_yd @ t_from_root
        t_world_yu = YDOWN_TO_YUP @ t_world_yd
        m = mesh.copy()
        m.apply_transform(t_world_yu)
        meshes.append(m)
    return trimesh.Scene(meshes)


def _assert_close_array(testcase: unittest.TestCase, a: np.ndarray, b: np.ndarray, atol: float, msg: str) -> None:
    max_abs = float(np.max(np.abs(a - b)))
    testcase.assertTrue(np.allclose(a, b, atol=atol, rtol=0.0), msg=f"{msg}; max_abs={max_abs:.3e}")


class TestURDFPyVisualizationParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import sys

        sys.path.insert(0, str(GLOVE_SIM_ROOT))
        from src.urdfpy_vis import get_glove_scene, load_robot

        cls.get_glove_scene_fn = staticmethod(get_glove_scene)
        cls.load_robot = load_robot
        cls.native_robot = cls._load_native_oracle_robot()
        cls.custom_robot = load_robot(ASSEMBLY_URDF, ASSEMBLY_MESH_DIR)
        cls.base_cfg = {j.name: 0.0 for j in cls.native_robot.actuated_joints}

    @classmethod
    def _load_native_oracle_robot(cls):
        """Native urdfpy load from canonical URDF with only path/texture sanitization.

        This avoids relying on a pre-generated URDF copy with possible transform drift.
        """
        tree = ET.parse(str(ASSEMBLY_URDF))
        root = tree.getroot()
        urdf_dir = ASSEMBLY_URDF.parent.resolve()
        for mesh_el in root.iter("mesh"):
            fn = mesh_el.get("filename", "")
            if fn.startswith("package://"):
                base = fn.split("/", 2)[-1]
                base = base.split("/", 1)[-1]
                base = base.split("/", 1)[-1]
                rel = os.path.relpath(ASSEMBLY_MESH_DIR / base, start=urdf_dir).replace("\\", "/")
                mesh_el.set("filename", rel)
        for material in root.iter("material"):
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

    def _cfgs(self) -> dict[str, dict[str, float]]:
        cfg = dict(self.base_cfg)
        out = {"rest": dict(cfg)}
        c = dict(cfg)
        c["revolute_3_0"] = np.deg2rad(45.0)
        out["thumb45"] = c
        c = dict(cfg)
        c["revolute_9_0"] = np.deg2rad(45.0)
        out["index45"] = c
        c = dict(cfg)
        c["revolute_3_0"] = np.deg2rad(45.0)
        c["revolute_9_0"] = np.deg2rad(45.0)
        out["combo45"] = c
        return out

    def _write_debug(self, name: str, payload: dict) -> None:
        out_dir = GLOVE_SIM_ROOT / "outputs/parity_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_layer_a_structure_and_fk_parity(self) -> None:
        self.assertEqual(len(self.native_robot.links), len(self.custom_robot.links))
        self.assertEqual(
            {j.name for j in self.native_robot.actuated_joints},
            {j.name for j in self.custom_robot.actuated_joints},
        )

        for cfg_name, cfg in self._cfgs().items():
            native_fk = _flatten_visual_fk(self.native_robot, cfg)
            custom_fk = _flatten_visual_fk(self.custom_robot, cfg)

            self.assertEqual(set(native_fk.keys()), set(custom_fk.keys()), msg=f"FK key mismatch for {cfg_name}")

            for key in sorted(native_fk.keys()):
                native_mesh, native_t = native_fk[key]
                custom_mesh, custom_t = custom_fk[key]
                _assert_close_array(self, native_t, custom_t, TRANSFORM_ATOL, f"{cfg_name}:{key} transform mismatch")

                n_metrics = _mesh_metrics(native_mesh)
                c_metrics = _mesh_metrics(custom_mesh)
                self.assertEqual(n_metrics["n_vertices"], c_metrics["n_vertices"], f"{cfg_name}:{key} vertices mismatch")
                self.assertEqual(n_metrics["n_faces"], c_metrics["n_faces"], f"{cfg_name}:{key} faces mismatch")
                _assert_close_array(
                    self,
                    np.array(n_metrics["extents"]),
                    np.array(c_metrics["extents"]),
                    GEOM_ATOL,
                    f"{cfg_name}:{key} extents mismatch",
                )

    def test_layer_b_and_c_snapshot_and_visualization_output_parity(self) -> None:
        # Hand-mount pose that yields URDF root at identity in get_glove_scene.
        t_hand_mount_world = np.eye(4)
        t_hand_mount_world[:3, 3] = ROOT_TO_HAND_MOUNT
        t_root_world_yd = np.eye(4)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for cfg_name, cfg in self._cfgs().items():
                native_fk = _flatten_visual_fk(self.native_robot, cfg)
                custom_fk = _flatten_visual_fk(self.custom_robot, cfg)

                native_scene = _scene_from_fk_entries(native_fk, t_root_world_yd)
                custom_scene = self.get_glove_scene_fn(self.custom_robot, cfg, t_hand_mount_world)

                # Layer C: compare transformed component placements.
                n_metrics = _scene_metrics(native_scene)
                c_metrics = _scene_metrics(custom_scene)
                self.assertEqual(n_metrics["n_geom"], c_metrics["n_geom"], f"{cfg_name}: scene geom count mismatch")
                _assert_close_array(
                    self,
                    np.array(n_metrics["bounds_min"]),
                    np.array(c_metrics["bounds_min"]),
                    SCENE_ATOL,
                    f"{cfg_name}: scene bounds_min mismatch",
                )
                _assert_close_array(
                    self,
                    np.array(n_metrics["bounds_max"]),
                    np.array(c_metrics["bounds_max"]),
                    SCENE_ATOL,
                    f"{cfg_name}: scene bounds_max mismatch",
                )

                nb = _scene_geom_buckets(n_metrics)
                cb = _scene_geom_buckets(c_metrics)
                self.assertEqual(set(nb.keys()), set(cb.keys()), f"{cfg_name}: geometry signature buckets mismatch")
                for key in sorted(nb.keys()):
                    self.assertEqual(len(nb[key]), len(cb[key]), f"{cfg_name}:{key} bucket count mismatch")
                    for idx, (ng, cg) in enumerate(zip(nb[key], cb[key])):
                        _assert_close_array(
                            self,
                            np.array(ng["centroid"]),
                            np.array(cg["centroid"]),
                            SCENE_ATOL,
                            f"{cfg_name}:{key}:{idx} centroid mismatch",
                        )

                # Layer B: export/reload GLBs and compare snapshot metrics.
                native_glb = tmp_dir / f"{cfg_name}_native.glb"
                custom_glb = tmp_dir / f"{cfg_name}_custom.glb"
                native_scene.export(str(native_glb))
                custom_scene.export(str(custom_glb))

                loaded_native = trimesh.load(str(native_glb), force="scene")
                loaded_custom = trimesh.load(str(custom_glb), force="scene")
                ln_metrics = _scene_metrics(loaded_native)
                lc_metrics = _scene_metrics(loaded_custom)
                try:
                    self.assertEqual(ln_metrics["n_geom"], lc_metrics["n_geom"], f"{cfg_name}: loaded geom count mismatch")
                    _assert_close_array(
                        self,
                        np.array(ln_metrics["bounds_min"]),
                        np.array(lc_metrics["bounds_min"]),
                        SNAPSHOT_ATOL,
                        f"{cfg_name}: loaded bounds_min mismatch",
                    )
                    _assert_close_array(
                        self,
                        np.array(ln_metrics["bounds_max"]),
                        np.array(lc_metrics["bounds_max"]),
                        SNAPSHOT_ATOL,
                        f"{cfg_name}: loaded bounds_max mismatch",
                    )
                    lnb = _scene_geom_buckets(ln_metrics)
                    lcb = _scene_geom_buckets(lc_metrics)
                    self.assertEqual(set(lnb.keys()), set(lcb.keys()), f"{cfg_name}: loaded geometry buckets mismatch")
                    for key in sorted(lnb.keys()):
                        self.assertEqual(len(lnb[key]), len(lcb[key]), f"{cfg_name}:{key} loaded bucket count mismatch")
                        for idx, (ng, cg) in enumerate(zip(lnb[key], lcb[key])):
                            _assert_close_array(
                                self,
                                np.array(ng["centroid"]),
                                np.array(cg["centroid"]),
                                SNAPSHOT_ATOL,
                                f"{cfg_name}:{key}:{idx} loaded centroid mismatch",
                            )
                except AssertionError:
                    self._write_debug(
                        f"{cfg_name}_snapshot_mismatch",
                        {"native": ln_metrics, "custom": lc_metrics},
                    )
                    raise


if __name__ == "__main__":
    unittest.main()
