"""
scene-reconstruction: dense scene reconstruction co-registered with Dyn-HaMR hands.

Two pipelines:
  - TSDF + Depth Anything V2  (fast, dynamic-scene-capable)
  - NeuS2                     (high-quality SDF meshes, static scenes)

Both output scene_mesh.glb in the same Y-up right-handed coordinate frame as
the hand GLBs exported by Dyn-HaMR's export_for_unity.py.
"""

__version__ = "0.1.0"

from .io import load_cameras, load_world_results, get_frame_paths, DynHaMRData
from .masking import MANOMaskRenderer, project_vertices_to_image
from .depth import DepthAnythingV2Wrapper, apply_depth_mask
from .tsdf_fusion import TSDFFusion, c2w_to_open3d_extrinsic
from .neus2_bridge import NeuS2Bridge, cameras_to_transforms_json

__all__ = [
    "load_cameras", "load_world_results", "get_frame_paths", "DynHaMRData",
    "MANOMaskRenderer", "project_vertices_to_image",
    "DepthAnythingV2Wrapper", "apply_depth_mask",
    "TSDFFusion", "c2w_to_open3d_extrinsic",
    "NeuS2Bridge", "cameras_to_transforms_json",
]
