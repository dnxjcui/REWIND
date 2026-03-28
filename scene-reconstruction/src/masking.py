"""
Hand masking utilities for scene reconstruction.

Two-tier approach:
  Tier 1 — MANO vertex projection + convex hull (fast, always available)
  Tier 2 — SAM2 refinement using MANO bboxes as video prompts (optional)

Coordinate note:
  project_vertices_to_image() uses cam_R / cam_t from world_results.npz,
  which are w2c in the Y-down optimization frame. MANO vertices from
  run_mano() are in the same Y-down frame. These are used together directly
  without any axis flip.
"""

from __future__ import annotations

import numpy as np
import cv2

try:
    from scipy.spatial import ConvexHull
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


# ── Low-level geometry ───────────────────────────────────────────────────────

def project_vertices_to_image(
    verts_world: np.ndarray,
    cam_R: np.ndarray,
    cam_t: np.ndarray,
    intrins: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Project 3D world-space vertices to 2D pixel coordinates.

    Parameters
    ----------
    verts_world : (V, 3)  3D points in Y-down optimization frame
    cam_R       : (3, 3)  w2c rotation (from .npz cam_R[b, t])
    cam_t       : (3,)    w2c translation (from .npz cam_t[b, t])
    intrins     : (4,)    [fx, fy, cx, cy]
    img_h, img_w : image dimensions (used only for bounds reference)

    Returns
    -------
    pts2d : (V, 2) float32 pixel coordinates (may be outside image bounds)
    """
    # Camera-space coordinates: x_cam = R @ x_world + t
    x_cam = (cam_R @ verts_world.T).T + cam_t  # (V, 3)

    # Only keep points in front of camera
    z = x_cam[:, 2]
    valid = z > 1e-4
    pts2d = np.full((len(verts_world), 2), -1.0, dtype=np.float32)

    if valid.any():
        fx, fy, cx, cy = intrins
        pts2d[valid, 0] = fx * x_cam[valid, 0] / z[valid] + cx
        pts2d[valid, 1] = fy * x_cam[valid, 1] / z[valid] + cy

    return pts2d


def convex_hull_mask(
    pts2d: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Fill the convex hull of 2D projected points in a boolean image mask.

    Parameters
    ----------
    pts2d : (V, 2) float32 pixel coordinates
    img_h, img_w : image dimensions

    Returns
    -------
    mask : (H, W) uint8  0/255
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Filter valid (in-frame) points
    valid = (
        (pts2d[:, 0] >= 0) & (pts2d[:, 0] < img_w) &
        (pts2d[:, 1] >= 0) & (pts2d[:, 1] < img_h) &
        (pts2d[:, 0] >= -1e3)  # -1.0 sentinel from project_vertices_to_image
    )
    pts_valid = pts2d[valid]

    if len(pts_valid) < 3:
        return mask

    if _SCIPY_OK:
        try:
            hull = ConvexHull(pts_valid)
            hull_pts = pts_valid[hull.vertices].astype(np.int32)
        except Exception:
            hull_pts = pts_valid.astype(np.int32)
    else:
        hull_pts = cv2.convexHull(pts_valid.astype(np.float32)).squeeze().astype(np.int32)

    cv2.fillConvexPoly(mask, hull_pts, 255)
    return mask


def dilate_mask(mask: np.ndarray, radius: int = 15) -> np.ndarray:
    """Morphological dilation with an elliptical kernel of given radius."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(mask, kernel)


def compute_bbox_from_mask(mask: np.ndarray):
    """
    Returns (x1, y1, x2, y2) bounding box of nonzero pixels, or None if empty.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def compute_bbox_from_pts2d(pts2d: np.ndarray, pad: int = 20):
    """
    Returns (x1, y1, x2, y2) bounding box around 2D projected points with padding.
    Filters out points with negative x (invalid projection sentinel).
    """
    valid = pts2d[:, 0] > 0
    if not valid.any():
        return None
    pts = pts2d[valid]
    x1 = max(0, int(pts[:, 0].min()) - pad)
    y1 = max(0, int(pts[:, 1].min()) - pad)
    x2 = int(pts[:, 0].max()) + pad
    y2 = int(pts[:, 1].max()) + pad
    return x1, y1, x2, y2


# ── MANOMaskRenderer ─────────────────────────────────────────────────────────

class MANOMaskRenderer:
    """
    Generates per-frame hand masks from pre-computed MANO world-space vertices.

    Parameters
    ----------
    img_h, img_w : int   image dimensions
    dilation_px  : int   elliptical dilation radius in pixels
    """

    def __init__(self, img_h: int, img_w: int, dilation_px: int = 15):
        self.img_h = img_h
        self.img_w = img_w
        self.dilation_px = dilation_px

    def render_frame_mask(
        self,
        verts_world_list: list,
        cam_R: np.ndarray,
        cam_t: np.ndarray,
        intrins: np.ndarray,
    ) -> np.ndarray:
        """
        Render combined mask for all hands in a single frame.

        Parameters
        ----------
        verts_world_list : list of (V, 3) arrays, one per hand/track
        cam_R  : (3, 3) w2c rotation for this frame (from .npz)
        cam_t  : (3,)   w2c translation for this frame (from .npz)
        intrins: (4,)   [fx, fy, cx, cy] from .npz

        Returns
        -------
        mask : (H, W) uint8  255 = hand region
        """
        combined = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        for verts in verts_world_list:
            pts2d = project_vertices_to_image(
                verts, cam_R, cam_t, intrins, self.img_h, self.img_w
            )
            hull_mask = convex_hull_mask(pts2d, self.img_h, self.img_w)
            combined = np.maximum(combined, hull_mask)

        if self.dilation_px > 0:
            combined = dilate_mask(combined, self.dilation_px)
        return combined

    def render_all_frames(
        self,
        verts_world: np.ndarray,
        cam_R_seq: np.ndarray,
        cam_t_seq: np.ndarray,
        intrins: np.ndarray,
    ) -> np.ndarray:
        """
        Render masks for all T frames.

        Parameters
        ----------
        verts_world : (B, T, V, 3)  MANO vertices, all tracks and frames
        cam_R_seq   : (T, 3, 3)     w2c rotations (averaged over tracks or track 0)
        cam_t_seq   : (T, 3)        w2c translations
        intrins     : (4,)          [fx, fy, cx, cy]

        Returns
        -------
        masks : (T, H, W) uint8
        """
        B, T = verts_world.shape[:2]
        masks = np.zeros((T, self.img_h, self.img_w), dtype=np.uint8)
        for t in range(T):
            verts_list = [verts_world[b, t] for b in range(B)]
            masks[t] = self.render_frame_mask(
                verts_list, cam_R_seq[t], cam_t_seq[t], intrins
            )
        return masks


# ── SAM2Refiner (optional) ───────────────────────────────────────────────────

class SAM2Refiner:
    """
    Refines MANO convex-hull masks using SAM2 video predictor.
    Only usable if `pip install segment-anything-2` has been run.

    Parameters
    ----------
    checkpoint  : str  path to SAM2 .pt checkpoint
    model_cfg   : str  path to SAM2 model config yaml
    """

    def __init__(self, checkpoint: str, model_cfg: str):
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not installed. Run:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything-2\n"
                "or set masking.use_sam2: false in your config."
            )
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    def refine_sequence(
        self,
        frames_dir: str,
        initial_masks: np.ndarray,
        prompt_interval: int = 30,
    ) -> np.ndarray:
        """
        Refine a sequence of MANO masks using SAM2 video propagation.

        Parameters
        ----------
        frames_dir      : str          directory of extracted frame images
        initial_masks   : (T, H, W) uint8  MANO convex-hull masks (255 = hand)
        prompt_interval : int          re-prompt SAM2 every N frames

        Returns
        -------
        refined_masks : (T, H, W) uint8
        """
        import torch

        T = len(initial_masks)
        state = self.predictor.init_state(frames_dir)
        self.predictor.reset_state(state)

        # Add box prompts at regular intervals using MANO bboxes
        for kf in range(0, T, prompt_interval):
            bbox = compute_bbox_from_mask(initial_masks[kf])
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            box_tensor = np.array([x1, y1, x2, y2], dtype=np.float32)
            self.predictor.add_new_points_or_box(
                state, frame_idx=kf, obj_id=1, box=box_tensor
            )

        # Propagate masks across all frames
        refined = np.zeros_like(initial_masks)
        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(state):
            if 1 in obj_ids:
                idx = list(obj_ids).index(1)
                mask_bin = (mask_logits[idx].squeeze() > 0.0).cpu().numpy()
                refined[frame_idx] = (mask_bin * 255).astype(np.uint8)

        return refined
