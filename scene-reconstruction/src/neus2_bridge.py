"""
NeuS2 bridge: convert Dyn-HaMR cameras to NeuS2 transforms.json format,
manage frame preparation, invoke NeuS2 training, and extract the mesh.

NeuS2 uses NeRF-synthetic format (transforms.json) with:
  - camera_angle_x: horizontal FoV in radians
  - frames[]: list of {file_path, transform_matrix, near, far}
  - transform_matrix: 4x4 c2w in OpenGL convention (Y-up, Z-back camera)

Coordinate note:
  cameras.json stores c2w in Y-up world (same as OpenGL). No flip needed.
  Just assemble the 4x4 c2w matrix from R_json / t_json.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import trimesh


# ── Camera conversion ─────────────────────────────────────────────────────────

def build_neus2_transform_matrix(
    R_c2w: np.ndarray,
    t_c2w: np.ndarray,
) -> list:
    """
    Assemble a 4x4 c2w matrix for NeuS2's transforms.json.

    cameras.json R_c2w / t_c2w are already c2w in Y-up (OpenGL-compatible).
    NeuS2 expects c2w in OpenGL convention — no axis flip required.

    Returns Python nested list for json.dump() serialization.
    """
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R_c2w.astype(np.float64)
    m[:3,  3] = t_c2w.astype(np.float64)
    return m.tolist()


def cameras_to_transforms_json(
    Rs: np.ndarray,
    ts: np.ndarray,
    Ks: np.ndarray,
    frame_paths: list,
    img_w: int,
    img_h: int,
    near: float = 0.01,
    far: float = 10.0,
    stride: int = 1,
    start: int = 0,
    end: int = -1,
) -> dict:
    """
    Build the full transforms.json dict for NeuS2 / NeRF-synthetic format.

    Parameters
    ----------
    Rs, ts, Ks    : arrays from load_cameras()
    frame_paths   : sorted list of frame image paths
    img_w, img_h  : image dimensions in pixels
    near, far     : per-frame depth bounds (meters)
    stride, start, end : frame selection

    Returns
    -------
    dict ready for json.dump()
    """
    N = len(frame_paths)
    stop = N if (end < 0 or end > N) else end
    indices = list(range(start, stop, stride))

    # camera_angle_x from median fx
    median_fx = float(np.median(Ks[indices, 0]))
    camera_angle_x = 2.0 * math.atan(img_w / (2.0 * median_fx))

    frames = []
    for i in indices:
        # file_path: relative, no extension (NeRF-synthetic convention)
        rel_path = os.path.join("images", Path(frame_paths[i]).stem)
        frames.append({
            "file_path": rel_path,
            "transform_matrix": build_neus2_transform_matrix(Rs[i], ts[i]),
            "near": near,
            "far": far,
        })

    return {
        "camera_angle_x": camera_angle_x,
        "w": img_w,
        "h": img_h,
        "frames": frames,
    }


# ── NeuS2Bridge ───────────────────────────────────────────────────────────────

class NeuS2Bridge:
    """
    Orchestrates the full NeuS2 pipeline for a single Dyn-HaMR sequence.

    Parameters
    ----------
    neus2_repo_path : str|None  path to NeuS2 repo; None = auto-detect third-party/neus2
    n_steps         : int       training iterations
    near, far       : float     scene depth bounds (meters)
    aabb_min/max    : list[float] scene AABB for normalization
    mesh_resolution : int       marching cubes grid resolution
    """

    def __init__(
        self,
        neus2_repo_path: Optional[str] = None,
        n_steps: int = 20000,
        near: float = 0.01,
        far: float = 10.0,
        aabb_min: Optional[list] = None,
        aabb_max: Optional[list] = None,
        mesh_resolution: int = 512,
    ):
        self.neus2_repo_path = neus2_repo_path
        self.n_steps = n_steps
        self.near = near
        self.far = far
        self.aabb_min = aabb_min or [-1.0, -1.0, -1.0]
        self.aabb_max = aabb_max or [ 1.0,  1.0,  1.0]
        self.mesh_resolution = mesh_resolution

    def _resolve_neus2_repo(self) -> Path:
        if self.neus2_repo_path:
            return Path(self.neus2_repo_path)
        here = Path(__file__).resolve().parent.parent
        candidate = here / "third-party" / "neus2"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            "Cannot locate NeuS2 repo. Run:\n"
            "  cd scene-reconstruction\n"
            "  git submodule add https://github.com/19reborn/NeuS2 third-party/neus2\n"
            "or pass neus2_repo_path= explicitly."
        )

    def prepare_scene(
        self,
        Rs: np.ndarray,
        ts: np.ndarray,
        Ks: np.ndarray,
        frame_paths: list,
        output_dir: str,
        img_w: int,
        img_h: int,
        stride: int = 1,
        start: int = 0,
        end: int = -1,
    ) -> str:
        """
        Create the NeuS2 scene directory:
          <output_dir>/
            images/        symlinks (or copies) to source frames
            transforms.json

        Returns path to transforms.json.
        """
        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Build transforms.json
        transforms = cameras_to_transforms_json(
            Rs, ts, Ks, frame_paths, img_w, img_h,
            near=self.near, far=self.far,
            stride=stride, start=start, end=end,
        )

        N = len(frame_paths)
        stop = N if (end < 0 or end > N) else end
        indices = list(range(start, stop, stride))

        # Create symlinks (or copies) for selected frames
        for idx, i in enumerate(indices):
            src = os.path.abspath(frame_paths[i])
            ext = Path(src).suffix
            stem = Path(src).stem
            dst = os.path.join(images_dir, f"{stem}{ext}")
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)

        transforms_path = os.path.join(output_dir, "transforms.json")
        with open(transforms_path, "w") as f:
            json.dump(transforms, f, indent=2)
        print(f"Wrote transforms.json ({len(indices)} frames) -> {transforms_path}")
        return transforms_path

    def run_training(self, scene_dir: str) -> int:
        """
        Invoke NeuS2 training as a subprocess.
        Looks for scripts/run.py or run.py inside the NeuS2 repo.

        Returns subprocess returncode.
        """
        neus2_repo = self._resolve_neus2_repo()
        transforms_path = os.path.join(scene_dir, "transforms.json")

        # Find entry point
        entry_candidates = [
            neus2_repo / "scripts" / "run.py",
            neus2_repo / "run.py",
        ]
        entry = None
        for c in entry_candidates:
            if c.exists():
                entry = str(c)
                break
        if entry is None:
            raise FileNotFoundError(
                f"Cannot find NeuS2 entry point in {neus2_repo}. "
                "Check that the submodule is initialized."
            )

        cmd = [
            sys.executable, entry,
            "--scene", transforms_path,
            "--n_steps", str(self.n_steps),
            "--aabb_min", *[str(v) for v in self.aabb_min],
            "--aabb_max", *[str(v) for v in self.aabb_max],
        ]
        print("Running NeuS2:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(neus2_repo))
        return result.returncode

    def extract_mesh_from_neus2(
        self,
        scene_dir: str,
        output_glb: str,
    ) -> None:
        """
        After NeuS2 training, find the output mesh (.ply or .obj), convert to
        trimesh, and export as GLB in Y-up world frame.

        NeuS2 typically writes meshes to <scene_dir>/mesh/ or as mesh.ply.
        """
        # Search for output mesh
        candidates = []
        for root, _, files in os.walk(scene_dir):
            for fn in files:
                if fn.endswith(".ply") or fn.endswith(".obj"):
                    candidates.append(os.path.join(root, fn))

        if not candidates:
            raise FileNotFoundError(
                f"No .ply or .obj mesh found in {scene_dir} after NeuS2 training."
            )

        # Prefer mesh.ply if multiple
        mesh_path = sorted(candidates)[-1]
        print(f"Loading NeuS2 output mesh: {mesh_path}")

        mesh_raw = trimesh.load(mesh_path, process=False)
        if isinstance(mesh_raw, trimesh.Scene):
            mesh_raw = trimesh.util.concatenate(list(mesh_raw.geometry.values()))

        os.makedirs(os.path.dirname(os.path.abspath(output_glb)), exist_ok=True)
        scene = trimesh.Scene()
        scene.add_geometry(mesh_raw, node_name="scene_mesh")
        with open(output_glb, "wb") as f:
            f.write(trimesh.exchange.gltf.export_glb(scene, include_normals=True))
        print(f"Saved NeuS2 mesh -> {output_glb}")

    def run_full_pipeline(
        self,
        Rs: np.ndarray,
        ts: np.ndarray,
        Ks: np.ndarray,
        frame_paths: list,
        output_dir: str,
        img_w: int,
        img_h: int,
        stride: int = 1,
        start: int = 0,
        end: int = -1,
        mesh_filename: str = "scene_mesh.glb",
    ) -> str:
        """
        End-to-end: prepare → train → extract mesh.

        Returns path to output GLB.
        """
        self.prepare_scene(
            Rs, ts, Ks, frame_paths, output_dir, img_w, img_h,
            stride=stride, start=start, end=end,
        )

        ret = self.run_training(output_dir)
        if ret != 0:
            raise RuntimeError(f"NeuS2 training failed with return code {ret}")

        output_glb = os.path.join(output_dir, mesh_filename)
        self.extract_mesh_from_neus2(output_dir, output_glb)
        return output_glb
