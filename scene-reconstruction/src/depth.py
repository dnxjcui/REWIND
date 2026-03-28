"""
Depth Anything V2 metric depth inference wrapper.

Expects the DA2 repo cloned at third-party/Depth-Anything-V2/ (default).
The metric_depth/ subdirectory is prepended to sys.path for import.

Checkpoint naming convention (Hypersim indoor metric):
  depth_anything_v2_metric_hypersim_vitl.pth   (vitl)
  depth_anything_v2_metric_hypersim_vitb.pth   (vitb)
  depth_anything_v2_metric_hypersim_vits.pth   (vits)

Download from: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────────

def apply_depth_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    depth_max: float = 5.0,
) -> np.ndarray:
    """
    Zero out hand-region pixels and pixels beyond depth_max.

    Parameters
    ----------
    depth    : (H, W) float32  metric depth in meters
    mask     : (H, W) uint8    255 = hand region (from MANOMaskRenderer)
    depth_max: float           clip depth beyond this value (meters)

    Returns
    -------
    masked_depth : (H, W) float32
    """
    out = depth.copy()
    out[mask > 0] = 0.0
    out[out > depth_max] = 0.0
    return out


# ── Model configs ─────────────────────────────────────────────────────────────

_MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "max_depth": 20,
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "max_depth": 20,
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "max_depth": 20,
    },
}

_CHECKPOINT_NAMES = {
    ("vitl", "hypersim"): "depth_anything_v2_metric_hypersim_vitl.pth",
    ("vitb", "hypersim"): "depth_anything_v2_metric_hypersim_vitb.pth",
    ("vits", "hypersim"): "depth_anything_v2_metric_hypersim_vits.pth",
    ("vitl", "vkitti"):   "depth_anything_v2_metric_vkitti_vitl.pth",
    ("vitb", "vkitti"):   "depth_anything_v2_metric_vkitti_vitb.pth",
    ("vits", "vkitti"):   "depth_anything_v2_metric_vkitti_vits.pth",
}


# ── Wrapper ───────────────────────────────────────────────────────────────────

class DepthAnythingV2Wrapper:
    """
    Wraps Depth Anything V2 metric depth inference.

    Parameters
    ----------
    model        : str   'vitl' | 'vitb' | 'vits'
    checkpoint   : str|None  path to .pth; None = auto-locate in da2_repo_path/checkpoints/
    dataset      : str   'hypersim' (indoor metric) | 'vkitti' (outdoor metric)
    max_depth    : float  model-level max depth (meters); also clips output
    input_size   : int   model input resolution (DA2 internal; output = original res)
    device       : str   'cuda' | 'cpu'
    da2_repo_path: str|None  path to DA2 repo root; None = auto-detect third-party/
    """

    def __init__(
        self,
        model: str = "vitl",
        checkpoint: Optional[str] = None,
        dataset: str = "hypersim",
        max_depth: float = 20.0,
        input_size: int = 518,
        device: str = "cuda",
        da2_repo_path: Optional[str] = None,
    ):
        self.model_variant = model
        self.checkpoint_path = checkpoint
        self.dataset = dataset
        self.max_depth = max_depth
        self.input_size = input_size
        self.device_str = device
        self.da2_repo_path = da2_repo_path
        self._model = None

    def _resolve_da2_path(self) -> Path:
        if self.da2_repo_path:
            return Path(self.da2_repo_path)
        # Auto-detect: look for third-party/Depth-Anything-V2 relative to this file
        here = Path(__file__).resolve().parent.parent
        candidates = [
            here / "third-party" / "Depth-Anything-V2",
            here / "third-party" / "Depth-Anything",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            "Cannot locate Depth-Anything-V2 repo. "
            "Either run:\n"
            "  cd scene-reconstruction\n"
            "  git submodule add https://github.com/LiheYoung/Depth-Anything "
            "third-party/Depth-Anything-V2\n"
            "or pass da2_repo_path= explicitly."
        )

    def load_model(self) -> None:
        """Lazy-load the DA2 model from checkpoint."""
        if self._model is not None:
            return

        da2_root = self._resolve_da2_path()
        metric_depth_dir = str(da2_root / "metric_depth")
        if metric_depth_dir not in sys.path:
            sys.path.insert(0, metric_depth_dir)

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as e:
            raise ImportError(
                f"Cannot import depth_anything_v2 from {metric_depth_dir}. "
                "Ensure the DA2 submodule is checked out and timm/transformers are installed."
            ) from e

        cfg = dict(_MODEL_CONFIGS[self.model_variant])
        cfg["max_depth"] = self.max_depth
        model = DepthAnythingV2(**cfg)

        ckpt_path = self.checkpoint_path
        if ckpt_path is None:
            # Auto-locate checkpoint
            name = _CHECKPOINT_NAMES.get((self.model_variant, self.dataset))
            if name is None:
                raise ValueError(
                    f"No known checkpoint for model={self.model_variant}, "
                    f"dataset={self.dataset}"
                )
            candidates = [
                da2_root / "checkpoints" / name,
                da2_root / name,
            ]
            for c in candidates:
                if c.exists():
                    ckpt_path = str(c)
                    break
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"Checkpoint {name} not found. Download from Hugging Face and place in "
                    f"{da2_root / 'checkpoints' / name}"
                )

        print(f"Loading DA2 {self.model_variant} from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)

        device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")
        self._model = model.to(device).eval()
        self._device = device
        print(f"DA2 model loaded on {self._device}")

    def infer_frame(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Run metric depth inference on a single BGR image.

        Parameters
        ----------
        img_bgr : (H, W, 3) uint8

        Returns
        -------
        depth : (H, W) float32  metric depth in meters at original resolution
        """
        self.load_model()
        with torch.no_grad():
            depth = self._model.infer_image(img_bgr, self.input_size)
        return depth.astype(np.float32)

    def infer_batch(
        self,
        frame_paths: list,
        show_progress: bool = True,
    ) -> Generator:
        """
        Generator: yield (H, W) float32 depth map for each frame path.
        Streams one frame at a time to avoid OOM on long sequences.

        Usage
        -----
        for depth in wrapper.infer_batch(frame_paths):
            # process depth ...
        """
        self.load_model()
        it = tqdm(frame_paths, desc="Depth inference") if show_progress else frame_paths
        for path in it:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Cannot read frame: {path}")
            yield self.infer_frame(img)
