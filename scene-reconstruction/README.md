# scene-reconstruction

Dense scene reconstruction co-registered with Dyn-HaMR hand meshes, for the REWIND senior design project.

Outputs `scene_mesh.glb` in the same Y-up right-handed coordinate frame as the hand GLBs from `export_for_unity.py`. Import both in Unity via glTFast — no manual alignment needed.

---

## Two Pipelines

| Pipeline | Speed | Quality | Dynamic scenes | Requirements |
|---|---|---|---|---|
| **TSDF + Depth Anything V2** | ~5–15 min | Good | ✅ | `scene-recon` conda env |
| **NeuS2** | 30 min – 2 hr | High (SDF) | ❌ static only | + CUDA build |

---

## Setup

### 1. Create conda environment

```bash
cd scene-reconstruction
conda env create -f environment.yml
conda activate scene-recon
```

### 2. Clone Depth Anything V2 submodule

```bash
git submodule add https://github.com/LiheYoung/Depth-Anything third-party/Depth-Anything-V2
git submodule update --init third-party/Depth-Anything-V2
```

Download the metric checkpoint (Hypersim, indoor, vitl — recommended):
```
third-party/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth
```
Available from: https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large

### 3. (Optional) SAM2 mask refinement

```bash
conda activate scene-recon
pip install git+https://github.com/facebookresearch/segment-anything-2
```

Download SAM2 checkpoint and config from Meta's SAM2 repo.
Set `masking.use_sam2: true` and paths in `configs/tsdf.yaml`.

### 4. (Optional) NeuS2

```bash
git submodule add https://github.com/19reborn/NeuS2 third-party/neus2
git submodule update --init third-party/neus2
# Follow NeuS2's own build instructions (requires CUDA compilation)
```

---

## Usage

### TSDF Pipeline (recommended)

```bash
conda activate scene-recon
cd scene-reconstruction

python scripts/reconstruct_tsdf.py \
    --log_dir  ../Dyn-HaMR/outputs/logs/video-custom/2026-03-24/demo1-all-shot-0-0--1 \
    --video    ../Dyn-HaMR/test/videos/demo1.mp4 \
    --config   configs/tsdf.yaml \
    --out_dir  outputs/demo1_tsdf/
```

All parameters can be set in `configs/tsdf.yaml` or overridden via CLI:
```
--stride N          integrate every Nth frame (default: 1)
--depth_model       vitl | vitb | vits (default: vitl)
--voxel_size        meters per voxel (default: 0.006 = 6mm)
--gpu N             GPU index (default: 0)
--cpu               force CPU mode
--save_intermediate save per-frame depth .npy and mask .png
--frames_dir DIR    use pre-extracted frames (skip ffmpeg)
```

### NeuS2 Pipeline

```bash
python scripts/reconstruct_neus2.py \
    --log_dir  ../Dyn-HaMR/outputs/logs/video-custom/2026-03-24/demo1-all-shot-0-0--1 \
    --video    ../Dyn-HaMR/test/videos/demo1.mp4 \
    --config   configs/neus2.yaml \
    --out_dir  outputs/demo1_neus2/ \
    --stride   2 \
    --n_steps  20000
```

---

## Output Structure

```
outputs/<name>/
  scene_mesh.glb        scene mesh with vertex colors, Y-up frame
  depth/                per-frame depth .npy (if --save_intermediate)
  masks/                per-frame mask .png (if --save_intermediate)
```

---

## Coordinate Systems

All outputs use **right-handed Y-up** (glTF/OpenGL standard), matching Dyn-HaMR's `export_for_unity.py` output.

```
cameras.json  →  c2w in Y-up world  (save_camera_json applies diag(1,-1,-1) flip)
world_results.npz cam_R/cam_t  →  w2c in Y-down optimization frame  (masking only)
TSDF volume  →  Y-up world (extrinsic = inv(c2w_yup))
NeuS2 transforms.json  →  c2w in Y-up OpenGL (same as cameras.json, no flip)
scene_mesh.glb  →  Y-up right-handed  ← co-registered with hand GLBs
Unity (via glTFast)  →  auto-converts right-handed → left-handed
```

---

## Gotchas

1. **MANO model path**: the script auto-searches for `_DATA/data/mano/` walking up from `--log_dir`. Set `dynhamr.mano_dir` in config if needed.
2. **world_scale**: not present in most `.npz` files; defaults to 1.0 automatically.
3. **Frame indexing**: ffmpeg uses 1-indexed filenames (`000001.jpg`); `.npz` is 0-indexed. The script uses sorted file order, not numeric parsing.
4. **Depth units**: DA2 outputs meters; Open3D TSDF uses `depth_scale=1000` (mm).
5. **NeuS2 for static scenes only**: camera moves around a static scene. For dynamic sequences (hands moving objects), use TSDF.
6. **Memory**: long sequences (1000+ frames) stream depth one frame at a time to avoid OOM.

---

## Python API

```python
from src.io import load_cameras, load_world_results
from src.depth import DepthAnythingV2Wrapper, apply_depth_mask
from src.masking import MANOMaskRenderer
from src.tsdf_fusion import TSDFFusion, c2w_to_open3d_extrinsic
from src.neus2_bridge import NeuS2Bridge, cameras_to_transforms_json
```
