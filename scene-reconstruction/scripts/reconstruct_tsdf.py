#!/usr/bin/env python3
"""
TSDF + Depth Anything V2 scene reconstruction from Dyn-HaMR output.

Usage
-----
conda activate scene-recon
python scripts/reconstruct_tsdf.py \\
    --log_dir  ../Dyn-HaMR/outputs/logs/video-custom/2026-03-24/demo1-all-shot-0-0--1 \\
    --video    ../Dyn-HaMR/test/videos/demo1.mp4 \\
    --config   configs/tsdf.yaml \\
    --out_dir  outputs/demo1_tsdf/

The script reads cameras.json and the latest smooth_fit/*.npz from --log_dir,
estimates per-frame metric depth with Depth Anything V2, masks hand regions
using MANO vertex projection, fuses into an Open3D TSDF volume, and exports
scene_mesh.glb in the same Y-up coordinate frame as the hand GLBs.
"""

import argparse
import os
import sys
from pathlib import Path

# Patch removed numpy aliases before any chumpy-dependent imports (MANO pkl loader).
# chumpy uses np.bool/np.int/etc. which were removed in numpy 1.24.
import numpy as _np
for _alias in ("bool", "int", "float", "complex", "object", "str"):
    if not hasattr(_np, _alias):
        setattr(_np, _alias, getattr(__builtins__ if isinstance(__builtins__, dict)
                                     else __builtins__, _alias, None) or
                              {"bool": bool, "int": int, "float": float,
                               "complex": complex, "object": object, "str": str}[_alias])
if not hasattr(_np, "unicode"):
    _np.unicode = str

# ── ensure scene-reconstruction/src is importable ────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
import torch
import yaml

from src.io import load_cameras, load_world_results, find_latest_npz, \
    extract_frames_from_video, get_frame_paths
from src.masking import MANOMaskRenderer, SAM2Refiner, SAM2_AVAILABLE
from src.depth import DepthAnythingV2Wrapper, apply_depth_mask
from src.tsdf_fusion import TSDFFusion


# ── MANO forward pass helper ──────────────────────────────────────────────────

def _run_mano_forward(world_results: dict, mano_dir: str, device_str: str = "cpu"):
    """
    Minimal MANO forward pass replicating Dyn-HaMR's run_mano() logic.

    Applies the handedness X-flip: verts[:,:,:,0] *= (2*is_right - 1).
    See Dyn-HaMR/dyn-hamr/body_model/utils.py run_mano().

    Parameters
    ----------
    world_results : dict from load_world_results()
    mano_dir      : path to directory containing MANO_RIGHT.pkl
    device_str    : 'cpu' or 'cuda:N'

    Returns
    -------
    verts_world : (B, T, 778, 3) numpy float32  in Y-down optimization frame
    """
    import smplx

    device = torch.device(device_str)
    trans       = torch.tensor(world_results["trans"],       dtype=torch.float32)  # (B,T,3)
    root_orient = torch.tensor(world_results["root_orient"], dtype=torch.float32)  # (B,T,3)
    pose_body   = torch.tensor(world_results["pose_body"],   dtype=torch.float32)  # (B,T,45)
    betas       = torch.tensor(world_results["betas"],       dtype=torch.float32)  # (B,10)
    is_right    = torch.tensor(world_results["is_right"],    dtype=torch.float32)  # (B,T)

    B, T = trans.shape[:2]
    all_verts = []

    for b in range(B):
        # Determine handedness for this track (use majority vote)
        right_frac = float(is_right[b].mean())
        is_rhand = right_frac > 0.5

        # Dyn-HaMR only ships MANO_RIGHT.pkl; left hands are produced by X-flipping
        # right-hand vertices after the forward pass (same pattern as run_mano()).
        # Always load the right-hand model.
        model = smplx.create(
            mano_dir,
            model_type="mano",
            use_pca=False,
            num_pca_comps=45,
            is_rhand=True,          # always right; flip below for left tracks
            batch_size=T,
            flat_hand_mean=False,
        ).to(device)

        # pose_body may be (T, 15, 3) or (T, 45); smplx expects (T, 45)
        pb = pose_body[b]
        if pb.dim() == 3:
            pb = pb.reshape(T, -1)  # (T, 15, 3) → (T, 45)

        with torch.no_grad():
            out = model(
                global_orient=root_orient[b].to(device),
                hand_pose=pb.to(device),
                transl=trans[b].to(device),
                betas=betas[b].unsqueeze(0).expand(T, -1).to(device),
            )
        verts = out.vertices.cpu()  # (T, 778, 3)

        # Handedness X-flip (mirrors left hand from right-hand model)
        sign = (2.0 * is_right[b] - 1.0).view(T, 1, 1)  # +1 for right, -1 for left
        verts[:, :, 0] = sign.squeeze(-1) * verts[:, :, 0]

        all_verts.append(verts.numpy())  # (T, 778, 3)

    return np.stack(all_verts, axis=0)  # (B, T, 778, 3)


# ── Config loading ────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _merge_cli(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values with non-None CLI arguments."""
    if args.log_dir:
        cfg["dynhamr"]["log_dir"] = args.log_dir
    if args.video:
        cfg["dynhamr"]["video_path"] = args.video
    if args.frames_dir:
        cfg["dynhamr"]["frames_dir"] = args.frames_dir
    if args.out_dir:
        cfg["output"]["dir"] = args.out_dir
    if args.phase:
        cfg["dynhamr"]["phase"] = args.phase
    if args.stride is not None:
        cfg["frames"]["stride"] = args.stride
    if args.depth_model:
        cfg["depth"]["model"] = args.depth_model
    if args.depth_checkpoint:
        cfg["depth"]["checkpoint"] = args.depth_checkpoint
    if args.voxel_size is not None:
        cfg["tsdf"]["voxel_size"] = args.voxel_size
    if args.cpu:
        cfg["depth"]["device"] = "cpu"
        cfg["tsdf"]["device"] = "CPU:0"
    elif args.gpu is not None:
        cfg["depth"]["device"] = f"cuda:{args.gpu}"
        cfg["tsdf"]["device"] = f"CUDA:{args.gpu}"
    if args.save_intermediate:
        cfg["output"]["save_intermediate"] = True
    if args.no_sam2:
        cfg["masking"]["use_sam2"] = False
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TSDF scene reconstruction from Dyn-HaMR output"
    )
    # Required
    parser.add_argument("--log_dir", help="Dyn-HaMR run_opt output directory")
    parser.add_argument("--video",   help="Source video .mp4 (for frame extraction)")
    # Optional overrides
    parser.add_argument("--config",      default=str(_REPO_ROOT / "configs" / "tsdf.yaml"))
    parser.add_argument("--frames_dir",  help="Pre-extracted frames directory (skips ffmpeg)")
    parser.add_argument("--out_dir",     help="Output directory for scene_mesh.glb")
    parser.add_argument("--phase",       default=None, help="Optimization phase (default: smooth_fit)")
    parser.add_argument("--stride",      type=int, default=None)
    parser.add_argument("--depth_model", choices=["vitl", "vitb", "vits"], default=None)
    parser.add_argument("--depth_checkpoint", default=None)
    parser.add_argument("--voxel_size",  type=float, default=None)
    parser.add_argument("--gpu",         type=int, default=None)
    parser.add_argument("--cpu",         action="store_true")
    parser.add_argument("--save_intermediate", action="store_true")
    parser.add_argument("--no_sam2",     action="store_true")
    args = parser.parse_args()

    # Load + merge config
    cfg = _load_config(args.config)
    cfg = _merge_cli(cfg, args)

    log_dir = cfg["dynhamr"]["log_dir"]
    if not log_dir:
        parser.error("--log_dir is required (or set dynhamr.log_dir in config)")

    video_path  = cfg["dynhamr"]["video_path"]
    frames_dir  = cfg["dynhamr"]["frames_dir"]
    phase       = cfg["dynhamr"]["phase"]
    mano_dir    = cfg["dynhamr"].get("mano_dir") or _auto_find_mano(log_dir)

    out_dir = cfg["output"]["dir"] or os.path.join(log_dir, "scene_reconstruction")
    os.makedirs(out_dir, exist_ok=True)

    stride = cfg["frames"]["stride"]
    start  = cfg["frames"]["start"]
    end    = cfg["frames"]["end"]

    # ── Step 1: Load cameras ─────────────────────────────────────────────────
    cameras_json = os.path.join(log_dir, "cameras.json")
    print(f"\n[1/6] Loading cameras from {cameras_json}")
    Rs, ts, Ks = load_cameras(cameras_json)
    N = len(Rs)
    print(f"  {N} camera frames loaded")

    # ── Step 2: Load world results ───────────────────────────────────────────
    phase_dir = os.path.join(log_dir, phase)
    print(f"\n[2/6] Loading world results from {phase_dir}")
    world_results = load_world_results(phase_dir)
    B, T = world_results["trans"].shape[:2]

    # ── Step 3: MANO forward pass ────────────────────────────────────────────
    print(f"\n[3/6] Running MANO forward pass (B={B}, T={T})")
    device_str = "cuda:0" if cfg["depth"]["device"].startswith("cuda") else "cpu"
    verts_world = _run_mano_forward(world_results, mano_dir, device_str)
    print(f"  verts_world shape: {verts_world.shape}")

    # ── Step 4: Extract frames ───────────────────────────────────────────────
    print(f"\n[4/6] Preparing frames")
    if frames_dir:
        frame_paths = get_frame_paths(frames_dir)
        print(f"  Using pre-extracted frames: {len(frame_paths)} files from {frames_dir}")
    elif video_path:
        tmp_frames = os.path.join(out_dir, "_frames")
        frame_paths = extract_frames_from_video(video_path, tmp_frames)
        print(f"  Extracted {len(frame_paths)} frames to {tmp_frames}")
    else:
        parser.error("Either --video or --frames_dir is required")

    # Clamp frame range to what's available
    stop = len(frame_paths) if (end < 0 or end > len(frame_paths)) else end
    selected = frame_paths[start:stop:stride]
    print(f"  Processing {len(selected)} frames (start={start}, end={stop}, stride={stride})")

    # Read image dimensions from first frame
    img0 = cv2.imread(frame_paths[0])
    img_h, img_w = img0.shape[:2]

    # ── Step 5: Depth + masking ──────────────────────────────────────────────
    print(f"\n[5/6] Depth inference + masking")
    depth_wrapper = DepthAnythingV2Wrapper(
        model=cfg["depth"]["model"],
        checkpoint=cfg["depth"].get("checkpoint"),
        dataset=cfg["depth"]["dataset"],
        max_depth=cfg["depth"]["max_depth"],
        input_size=cfg["depth"]["input_size"],
        device=cfg["depth"]["device"],
    )

    mask_renderer = MANOMaskRenderer(img_h, img_w, cfg["masking"]["dilation_px"])

    # Cam_R/cam_t from .npz: use track 0 for per-frame projection
    # (cam_R is same across tracks since it's the camera, not the hand)
    cam_R_seq = world_results["cam_R"][0]   # (T, 3, 3) w2c in Y-down frame
    cam_t_seq = world_results["cam_t"][0]   # (T, 3)
    intrins   = world_results["intrins"]    # (4,)

    # Optional SAM2 init
    sam2_refiner = None
    if cfg["masking"]["use_sam2"]:
        if not SAM2_AVAILABLE:
            print("WARNING: SAM2 requested but not installed. Falling back to MANO masks.")
        else:
            sam2_refiner = SAM2Refiner(
                cfg["masking"]["sam2_checkpoint"],
                cfg["masking"]["sam2_model_cfg"],
            )

    save_intermediate = cfg["output"]["save_intermediate"]
    if save_intermediate:
        depth_dir = os.path.join(out_dir, "depth")
        mask_dir  = os.path.join(out_dir, "masks")
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(mask_dir,  exist_ok=True)

    # ── Step 6: TSDF integration ─────────────────────────────────────────────
    print(f"\n[6/6] TSDF integration")
    tsdf_cfg = cfg["tsdf"]
    fusion = TSDFFusion(
        voxel_size=tsdf_cfg["voxel_size"],
        block_resolution=tsdf_cfg["block_resolution"],
        block_count=tsdf_cfg["block_count"],
        depth_scale=tsdf_cfg["depth_scale"],
        depth_max=tsdf_cfg["depth_max"],
        trunc_voxel_multiplier=tsdf_cfg["trunc_voxel_multiplier"],
        weight_threshold=tsdf_cfg["weight_threshold"],
        device_str=tsdf_cfg["device"],
    )
    fusion.initialize()

    from tqdm import tqdm

    indices = list(range(start, stop, stride))
    for enum_i, frame_i in enumerate(tqdm(indices, desc="Fusing")):
        # Depth inference
        img_bgr = cv2.imread(frame_paths[frame_i])
        depth   = depth_wrapper.infer_frame(img_bgr)

        # MANO mask — use npz frame index (0-indexed)
        # frames_dir files are 1-indexed (000001.jpg), npz is 0-indexed
        # enum_i is the 0-based position in selected frames
        npz_t = min(enum_i, T - 1)  # clamp in case T < N
        verts_list = [verts_world[b, npz_t] for b in range(B)]
        mask = mask_renderer.render_frame_mask(
            verts_list, cam_R_seq[npz_t], cam_t_seq[npz_t], intrins
        )

        # Apply mask + depth clip
        depth_masked = apply_depth_mask(depth, mask, cfg["depth"]["max_depth"])

        if save_intermediate:
            np.save(os.path.join(depth_dir, f"{frame_i:06d}.npy"), depth_masked)
            cv2.imwrite(os.path.join(mask_dir,  f"{frame_i:06d}.png"), mask)

        # Build extrinsic + intrinsic for this frame
        from src.tsdf_fusion import c2w_to_open3d_extrinsic
        extrinsic = c2w_to_open3d_extrinsic(Rs[frame_i], ts[frame_i])
        fx, fy, cx, cy = Ks[frame_i]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Resize image to match depth resolution if needed
        dH, dW = depth_masked.shape[:2]
        if (img_bgr.shape[0], img_bgr.shape[1]) != (dH, dW):
            img_bgr = cv2.resize(img_bgr, (dW, dH))

        fusion.integrate_frame(depth_masked, img_bgr, extrinsic, K)

    # Optionally run SAM2 refinement (post-hoc: re-integrate with better masks)
    # This is a placeholder; for full SAM2 refinement, run a second pass.
    if sam2_refiner is not None:
        print("NOTE: SAM2 refinement was requested but requires a two-pass integration.")
        print("      The current output uses MANO convex-hull masks.")

    # Export mesh
    mesh_path = os.path.join(out_dir, cfg["output"]["mesh_filename"])
    fusion.save_mesh_glb(mesh_path)

    print(f"\nDone. Output: {out_dir}/")
    print(f"  {cfg['output']['mesh_filename']}")
    if save_intermediate:
        print(f"  depth/  (per-frame .npy)")
        print(f"  masks/  (per-frame .png)")


def _auto_find_mano(log_dir: str) -> str:
    """
    Try to find MANO model directory relative to Dyn-HaMR's _DATA folder.
    Falls back to expecting MANO_RIGHT.pkl in the path smplx uses.
    """
    # Walk up from log_dir to find Dyn-HaMR root
    path = Path(log_dir).resolve()
    for _ in range(10):
        # smplx.create(model_path, model_type='mano') looks for model_path/mano/MANO_*.pkl
        # so we need to return _DATA/data, not _DATA/data/mano
        candidate = path / "_DATA" / "data" / "mano"
        if candidate.exists():
            return str(path / "_DATA" / "data")
        path = path.parent
    # Fall back to smplx default
    return os.path.expanduser("~/.smplx")


if __name__ == "__main__":
    main()
