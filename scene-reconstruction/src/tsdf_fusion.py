"""
Open3D VoxelBlockGrid TSDF integration and mesh export.

Coordinate convention:
  cameras.json stores c2w in Y-up world frame.
  Open3D extrinsic = w2c = inv(c2w_yup_4x4).
  No axis flip is needed — the TSDF volume is built in the Y-up world frame,
  so the extracted mesh is co-registered with Dyn-HaMR's hand GLBs directly.

Depth unit convention:
  DA2 outputs float32 in meters. Open3D integrate expects uint16 with
  depth_scale=1000 (millimeters). We multiply by 1000 before converting.
"""

from __future__ import annotations

import os
from typing import Generator, Iterable, Optional

import cv2
import numpy as np
import trimesh
from tqdm import tqdm

import open3d as o3d
import open3d.core as o3c


# ── Coordinate helpers ────────────────────────────────────────────────────────

def c2w_to_open3d_extrinsic(R_c2w: np.ndarray, t_c2w: np.ndarray) -> np.ndarray:
    """
    Convert c2w (Y-up, from cameras.json) to Open3D extrinsic (w2c 4x4 float64).

    cameras.json already stores c2w in Y-up frame (save_camera_json applies the
    Y-down → Y-up flip before writing). Open3D needs w2c. Simple matrix inverse.
    """
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_c2w.astype(np.float64)
    c2w[:3,  3] = t_c2w.astype(np.float64)
    return np.linalg.inv(c2w)


def build_intrinsic_tensor(
    fx: float, fy: float, cx: float, cy: float,
    device: "o3c.Device",
) -> "o3c.Tensor":
    """Return Open3D (3,3) float64 intrinsic tensor."""
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return o3c.Tensor(K, dtype=o3c.float64, device=device)


# ── TSDFFusion ────────────────────────────────────────────────────────────────

class TSDFFusion:
    """
    Manages an Open3D VoxelBlockGrid TSDF volume for multi-frame depth fusion.

    Parameters
    ----------
    voxel_size             : float  meters per voxel (default ~6 mm)
    block_resolution       : int    voxels per block edge
    block_count            : int    initial hash map capacity
    depth_scale            : float  depth_m * depth_scale → Open3D uint16 units
    depth_max              : float  max integration depth (meters)
    trunc_voxel_multiplier : float  truncation = voxel_size * this
    weight_threshold       : float  min weight for mesh extraction
    device_str             : str    Open3D device string, e.g. 'CUDA:0' or 'CPU:0'
    """

    def __init__(
        self,
        voxel_size: float = 0.006,
        block_resolution: int = 16,
        block_count: int = 50000,
        depth_scale: float = 1000.0,
        depth_max: float = 5.0,
        trunc_voxel_multiplier: float = 8.0,
        weight_threshold: float = 3.0,
        device_str: str = "CUDA:0",
    ):
        self.voxel_size = voxel_size
        self.block_resolution = block_resolution
        self.block_count = block_count
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.trunc_voxel_multiplier = trunc_voxel_multiplier
        self.weight_threshold = weight_threshold

        # Graceful CUDA fallback
        try:
            self.device = o3c.Device(device_str)
            # Test device is usable
            _ = o3c.Tensor([1.0], device=self.device)
        except Exception:
            print(f"WARNING: Open3D device {device_str!r} unavailable, falling back to CPU:0")
            self.device = o3c.Device("CPU:0")

        self.vbg: Optional[o3d.t.geometry.VoxelBlockGrid] = None

    def initialize(self) -> None:
        """Create the VoxelBlockGrid."""
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_resolution,
            block_count=self.block_count,
            device=self.device,
        )

    def integrate_frame(
        self,
        depth_m: np.ndarray,
        color_bgr: np.ndarray,
        extrinsic_4x4: np.ndarray,
        intrinsic_3x3: np.ndarray,
    ) -> None:
        """
        Integrate one frame into the TSDF volume.

        Parameters
        ----------
        depth_m        : (H, W) float32  metric depth in meters (hand-masked)
        color_bgr      : (H, W, 3) uint8
        extrinsic_4x4  : (4, 4) float64  w2c from c2w_to_open3d_extrinsic()
        intrinsic_3x3  : (3, 3) float64  K matrix
        """
        if self.vbg is None:
            self.initialize()

        H, W = depth_m.shape

        # Convert depth meters → uint16 millimeters
        depth_mm = (depth_m * self.depth_scale).clip(0, 65535).astype(np.uint16)

        # Wrap as Open3D images
        depth_img = o3d.t.geometry.Image(
            o3c.Tensor(depth_mm, dtype=o3c.uint16, device=self.device)
        )
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        color_img = o3d.t.geometry.Image(
            o3c.Tensor(color_rgb.astype(np.uint8), dtype=o3c.uint8, device=self.device)
        )

        # Open3D requires intrinsic/extrinsic on CPU regardless of VBG device.
        # Only depth/color images should be on the CUDA device.
        cpu = o3c.Device("CPU:0")
        intr_t = o3c.Tensor(intrinsic_3x3, dtype=o3c.float64, device=cpu)
        extr_t = o3c.Tensor(extrinsic_4x4, dtype=o3c.float64, device=cpu)

        # Activate voxel blocks that the depth map touches
        block_coords = self.vbg.compute_unique_block_coordinates(
            depth_img,
            intr_t,
            extr_t,
            depth_scale=self.depth_scale,
            depth_max=self.depth_max,
        )

        # Fuse
        self.vbg.integrate(
            block_coords,
            depth_img,
            color_img,
            intr_t,
            intr_t,
            extr_t,
            depth_scale=self.depth_scale,
            depth_max=self.depth_max,
            trunc_voxel_multiplier=self.trunc_voxel_multiplier,
        )

    def run(
        self,
        frame_paths: list,
        depth_generator: Iterable,
        mask_generator: Iterable,
        Rs: np.ndarray,
        ts: np.ndarray,
        Ks: np.ndarray,
        stride: int = 1,
        start: int = 0,
        end: int = -1,
        show_progress: bool = True,
    ) -> None:
        """
        Main integration loop: stream depth+masks, integrate each frame.

        Parameters
        ----------
        frame_paths      : list[str]   paths to color frames (1-indexed, sorted)
        depth_generator  : iterable    yields (H, W) float32 masked depth per frame
        mask_generator   : iterable    yields (H, W) uint8 mask per frame (for logging)
        Rs               : (N, 3, 3)  c2w rotations from cameras.json
        ts               : (N, 3)     c2w translations from cameras.json
        Ks               : (N, 4)     [fx,fy,cx,cy] from cameras.json
        stride, start, end : frame selection parameters
        """
        if self.vbg is None:
            self.initialize()

        # Frame index range
        N = len(frame_paths)
        stop = N if (end < 0 or end > N) else end
        indices = list(range(start, stop, stride))

        print(f"TSDF integration: {len(indices)} frames (stride={stride})")
        pbar = tqdm(indices) if show_progress else indices

        depth_iter = iter(depth_generator)
        mask_iter  = iter(mask_generator)

        for i in pbar:
            depth_masked = next(depth_iter)
            mask         = next(mask_iter)

            color_bgr = cv2.imread(frame_paths[i])
            if color_bgr is None:
                print(f"WARNING: Cannot read {frame_paths[i]}, skipping")
                continue

            # Resize color to match depth if needed
            dH, dW = depth_masked.shape[:2]
            cH, cW = color_bgr.shape[:2]
            if (cH, cW) != (dH, dW):
                color_bgr = cv2.resize(color_bgr, (dW, dH))

            # Build extrinsic (w2c) from c2w in cameras.json
            extrinsic = c2w_to_open3d_extrinsic(Rs[i], ts[i])

            # Build intrinsic K matrix
            fx, fy, cx, cy = Ks[i]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

            self.integrate_frame(depth_masked, color_bgr, extrinsic, K)

    def extract_mesh(self) -> trimesh.Trimesh:
        """
        Extract triangle mesh via marching cubes from the TSDF volume.

        Returns trimesh.Trimesh with vertex colors in Y-up world frame.
        """
        if self.vbg is None:
            raise RuntimeError("TSDF not initialized. Call run() first.")

        print(f"Extracting mesh (weight_threshold={self.weight_threshold})...")
        o3d_mesh = self.vbg.extract_triangle_mesh(
            weight_threshold=self.weight_threshold
        ).to_legacy()

        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_non_manifold_edges()

        verts  = np.asarray(o3d_mesh.vertices, dtype=np.float32)
        faces  = np.asarray(o3d_mesh.triangles, dtype=np.int32)
        colors = (np.asarray(o3d_mesh.vertex_colors) * 255).astype(np.uint8)

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_colors=colors,
            process=False,
        )
        print(f"  {len(verts):,} vertices, {len(faces):,} faces")
        return mesh

    def save_mesh_glb(self, out_path: str) -> None:
        """Export the TSDF mesh as a GLB file with vertex colors."""
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        mesh = self.extract_mesh()
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="scene_mesh")
        with open(out_path, "wb") as f:
            f.write(trimesh.exchange.gltf.export_glb(scene, include_normals=True))
        print(f"Saved scene mesh -> {out_path}")
