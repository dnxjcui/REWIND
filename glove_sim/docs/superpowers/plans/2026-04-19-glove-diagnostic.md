# Glove Visualization Fix & Diagnostic GLB Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Y-down→Y-up rotation bug that causes glove mesh parts to appear visually disconnected, then add a standalone diagnostic GLB exporter to verify the fix and tune sensor dot positions.

**Architecture:** Two changes to `visualize.py` (new `export_glove_only_glb` function + one-line transform fix applied to both export functions), plus a new `diagnostic.py` script that drives the glove simulator without DynHaMR data and exports static GLBs for visual inspection.

**Tech Stack:** Python 3.10, MuJoCo 3.x, trimesh, numpy — `/home/sybbure/miniconda3/envs/glove_sim/bin/python`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `glove_sim/src/visualize.py` | Modify | Fix `T_yu` formula; add `export_glove_only_glb()` |
| `glove_sim/diagnostic.py` | Create | CLI: load glove, export rest GLB + index-sweep GLBs |
| `glove_sim/config.py` | Modify (post-visual) | Tune `SENSOR_TIP_OFFSETS` after inspection |

---

## Task 1: Fix the Y-down→Y-up rotation bug in `export_frame_glb`

**Files:**
- Modify: `glove_sim/src/visualize.py:78`

The conjugation `YDOWN_TO_YUP @ T_yd @ inv(YDOWN_TO_YUP)` incorrectly pre-flips mesh-local vertices, producing rotation `P @ R @ P` instead of `P @ R`. The fix is to just left-multiply by `YDOWN_TO_YUP`.

- [ ] **Step 1: Open `glove_sim/src/visualize.py` and replace line 78**

Change:
```python
        T_yu = YDOWN_TO_YUP @ T_yd @ np.linalg.inv(YDOWN_TO_YUP)
```
To:
```python
        T_yu = YDOWN_TO_YUP @ T_yd
```

- [ ] **Step 2: Verify the change looks right**

Run:
```bash
grep -n "T_yu" /home/sybbure/Desktop/REWIND_SENIOR_DESIGN/glove_sim/src/visualize.py
```
Expected output:
```
78:        T_yu = YDOWN_TO_YUP @ T_yd
```

- [ ] **Step 3: Quick smoke test — pipeline still runs**

```bash
cd /home/sybbure/Desktop/REWIND_SENIOR_DESIGN/glove_sim
/home/sybbure/miniconda3/envs/glove_sim/bin/python pipeline.py --frames 0 2 --no-vis
```
Expected: completes with `Reliable frames: 2 (100.0%)`, no Python errors.

- [ ] **Step 4: Commit**

```bash
cd /home/sybbure/Desktop/REWIND_SENIOR_DESIGN/glove_sim
git add src/visualize.py
git commit -m "Fix Y-down→Y-up rotation: use P@T not P@T@P conjugation"
```

---

## Task 2: Add `export_glove_only_glb` to `visualize.py`

**Files:**
- Modify: `glove_sim/src/visualize.py` (append after `export_frame_glb`)

This function is identical to `export_frame_glb` but skips the hand GLB loading block. `diagnostic.py` will call it directly.

- [ ] **Step 1: Add the function to `visualize.py`**

Append after the closing of `export_frame_glb` (after line 95, before `export_frames`):

```python
def export_glove_only_glb(
    geom_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    mesh_dir: Path = MESH_DIR,
    sensor_positions_ydown: np.ndarray | None = None,
) -> None:
    """Export a GLB containing only the glove mesh (no hand overlay).

    Parameters
    ----------
    geom_poses            : {body_name: (pos_3, rotmat_3x3)} from get_geom_world_poses(), Y-down frame
    out_path              : destination .glb file path
    mesh_dir              : directory containing binary STL files
    sensor_positions_ydown: (4, 3) sensor world positions in Y-down frame, or None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meshes = []

    # Glove meshes: apply geom-level world transform, convert Y-down → Y-up
    for link_name, stl_file in LINK_TO_STL.items():
        pose = geom_poses.get(link_name)
        if pose is None:
            continue
        pos, rot = pose

        stl_path = mesh_dir / stl_file
        if not stl_path.exists():
            continue
        try:
            mesh = trimesh.load(str(stl_path), force="mesh")
        except Exception:
            continue

        T_yd = np.eye(4)
        T_yd[:3, :3] = rot
        T_yd[:3,  3] = pos

        T_yu = YDOWN_TO_YUP @ T_yd

        mesh.apply_transform(T_yu)
        meshes.append(mesh)

    # Red sensor spheres
    if sensor_positions_ydown is not None:
        for pos_yd in sensor_positions_ydown:
            pos_h = np.append(pos_yd, 1.0)
            pos_yu = (YDOWN_TO_YUP @ pos_h)[:3]
            meshes.append(_make_sensor_sphere(pos_yu))

    if not meshes:
        print(f"[WARNING] No meshes to export for {out_path}, skipping.")
        return

    scene = trimesh.Scene(meshes)
    scene.export(str(out_path), file_type="glb")
```

- [ ] **Step 2: Verify it's importable**

```bash
cd /home/sybbure/Desktop/REWIND_SENIOR_DESIGN/glove_sim
/home/sybbure/miniconda3/envs/glove_sim/bin/python -c "from src.visualize import export_glove_only_glb; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/visualize.py
git commit -m "Add export_glove_only_glb() for standalone diagnostic output"
```

---

## Task 3: Write `diagnostic.py`

**Files:**
- Create: `glove_sim/diagnostic.py`

The script:
1. Loads `GloveSimulator` at world origin (identity pose, no DynHaMR data)
2. Exports `outputs/diagnostic/glove_rest.glb` — all joints at 0°
3. Sweeps `revolute_9_0` from 0° to 90° in 10° steps, exporting one GLB per step

- [ ] **Step 1: Create `glove_sim/diagnostic.py`**

```python
"""Standalone glove diagnostic: exports GLBs for visual inspection without DynHaMR data.

Usage:
    python diagnostic.py
Outputs:
    outputs/diagnostic/glove_rest.glb          — all joints at 0 degrees
    outputs/diagnostic/glove_index_000.glb     — index tip at 0°
    outputs/diagnostic/glove_index_010.glb     — index tip at 10°
    ...
    outputs/diagnostic/glove_index_090.glb     — index tip at 90°
"""

import sys
import numpy as np
import mujoco
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.glove_ik import GloveSimulator
from src.visualize import export_glove_only_glb


def main():
    print("Loading glove into MuJoCo...")
    sim = GloveSimulator(cfg.URDF_PATH, cfg.MESH_DIR)
    print(f"  nq={sim.model.nq}, nv={sim.model.nv}, nbody={sim.model.nbody}")

    out_dir = cfg.OUTPUT_DIR / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Place glove at world origin with identity rotation
    sim.set_base_pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mujoco.mj_forward(sim.model, sim.data)

    # --- Rest pose (all joints at 0) ---
    geom_poses = sim.get_geom_world_poses()
    sensor_pos = sim.get_sensor_world_positions()
    out_path = out_dir / "glove_rest.glb"
    export_glove_only_glb(geom_poses, out_path, sensor_positions_ydown=sensor_pos)
    print(f"  Exported: {out_path}")

    # --- Index tip sweep: revolute_9_0 from 0° to 90° in 10° steps ---
    idx_tip_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "revolute_9_0")
    if idx_tip_jid < 0:
        print("[ERROR] Joint 'revolute_9_0' not found in model.")
        return
    idx_tip_qadr = sim.model.jnt_qposadr[idx_tip_jid]

    for angle_deg in range(0, 91, 10):
        # Reset all joints to 0 first, then set only the index tip
        mujoco.mj_resetData(sim.model, sim.data)
        sim.set_base_pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        sim.data.qpos[idx_tip_qadr] = np.radians(angle_deg)
        mujoco.mj_forward(sim.model, sim.data)

        geom_poses = sim.get_geom_world_poses()
        sensor_pos = sim.get_sensor_world_positions()
        out_path = out_dir / f"glove_index_{angle_deg:03d}.glb"
        export_glove_only_glb(geom_poses, out_path, sensor_positions_ydown=sensor_pos)
        print(f"  Exported: {out_path}  (index_tip={angle_deg}°)")

    print(f"\nDone. {len(list(out_dir.glob('*.glb')))} GLBs in {out_dir}")
    print("View with: f3d outputs/diagnostic/glove_rest.glb")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
cd /home/sybbure/Desktop/REWIND_SENIOR_DESIGN/glove_sim
/home/sybbure/miniconda3/envs/glove_sim/bin/python diagnostic.py
```
Expected output:
```
Loading glove into MuJoCo...
  nq=16, nv=15, nbody=22
  Exported: outputs/diagnostic/glove_rest.glb
  Exported: outputs/diagnostic/glove_index_000.glb  (index_tip=0°)
  ...
  Exported: outputs/diagnostic/glove_index_090.glb  (index_tip=90°)

Done. 12 GLBs in outputs/diagnostic
View with: f3d outputs/diagnostic/glove_rest.glb
```

- [ ] **Step 3: Verify output files exist**

```bash
ls -lh /home/sybbure/Desktop/REWIND_SENIOR_DESIGN/glove_sim/outputs/diagnostic/
```
Expected: 12 `.glb` files (glove_rest + glove_index_000 through glove_index_090).

- [ ] **Step 4: Commit**

```bash
git add diagnostic.py
git commit -m "Add diagnostic.py: standalone glove GLB exporter for visual inspection"
```

---

## Task 4: Visual inspection — open GLBs in F3D

This task is manual. No code changes.

- [ ] **Step 1: Open rest pose**

```bash
f3d /home/sybbure/Desktop/REWIND_SENIOR_DESIGN/glove_sim/outputs/diagnostic/glove_rest.glb
```
**Pass criteria:**
- All glove parts are visually connected at their joints (no floating pieces)
- 4 red dots visible (2 near the cap meshes, 2 near the base joints)
- The glove looks like a coherent mechanical assembly

**Fail criteria (and what to do):**
- Parts still floating → the geom transform from `_build_body_geom_map` may be mapping the wrong geom ID per body; flag for investigation
- Glove looks inside-out or mirrored → try removing the `YDOWN_TO_YUP @` prefix in `export_glove_only_glb` to see the raw MuJoCo frame

- [ ] **Step 2: Open index sweep to check joint motion**

```bash
f3d /home/sybbure/Desktop/REWIND_SENIOR_DESIGN/glove_sim/outputs/diagnostic/glove_index_090.glb
```
**Pass criteria:**
- The index fingertip cap (`part_3_1`) is visibly rotated ~90° relative to `glove_rest.glb`
- Rest of the glove is unchanged

- [ ] **Step 3: Check sensor dot positions**

Open `glove_rest.glb` in F3D and visually locate the 4 red dots:
- `thumb_base` dot: should be at the hinge between `part_1` and `part_2_1` (the proximal thumb linkage joint)
- `thumb_tip` dot: should be halfway along the thumb cap body (`part_3`)
- `index_base` dot: should be at the hinge between `part_1_1` and `part_2` (the proximal index linkage joint)
- `index_tip` dot: should be halfway along the index cap body (`part_3_1`)

Note which dots look wrong and what direction they need to move.

---

## Task 5: Tune `SENSOR_TIP_OFFSETS` in `config.py` (post-inspection)

**Files:**
- Modify: `glove_sim/config.py`

Run after Task 4 visual inspection. Adjust offsets based on what you saw in F3D.

The offsets are in the **body's local frame** (not world frame). For `part_3` (thumb cap), the long axis of the cap is approximately along `-X` in the body frame (the visual origin is at `(-0.125132, 0.004875, -0.0466837)`, pointing away from the joint). Halfway along the cap would be roughly half of that visual origin vector.

- [ ] **Step 1: Update `SENSOR_TIP_OFFSETS` in `config.py`**

Current values (estimated):
```python
SENSOR_TIP_OFFSETS = {
    "thumb_tip": ("part_3",   _np.array([-0.0626, 0.00244, -0.0233])),
    "index_tip": ("part_3_1", _np.array([-0.0337, 0.00244, -0.0125])),
}
```

Adjust based on visual inspection. For example, if the thumb tip dot appears too close to the joint, increase the magnitude:
```python
SENSOR_TIP_OFFSETS = {
    "thumb_tip": ("part_3",   _np.array([-0.08, 0.00244, -0.030])),   # example — tune to match
    "index_tip": ("part_3_1", _np.array([-0.045, 0.00244, -0.016])),  # example — tune to match
}
```

- [ ] **Step 2: Re-run diagnostic and re-inspect**

```bash
cd /home/sybbure/Desktop/REWIND_SENIOR_DESIGN/glove_sim
/home/sybbure/miniconda3/envs/glove_sim/bin/python diagnostic.py
f3d outputs/diagnostic/glove_rest.glb
```
Repeat until both tip dots sit visually halfway along their respective cap meshes.

- [ ] **Step 3: Commit tuned values**

```bash
git add config.py
git commit -m "Tune SENSOR_TIP_OFFSETS from visual inspection of diagnostic GLB"
```

---

## Task 6: Confirm fix carries through to full pipeline

**Files:** None modified — verification only.

- [ ] **Step 1: Run a short pipeline segment with visualization**

```bash
cd /home/sybbure/Desktop/REWIND_SENIOR_DESIGN/glove_sim
/home/sybbure/miniconda3/envs/glove_sim/bin/python pipeline.py --frames 0 5
```
Expected:
```
Reliable frames : 5 (100.0%)
IK residual max : < 1.0 mm
Exporting 5 GLB visualizations...
Done.
```

- [ ] **Step 2: Inspect one overlay GLB**

```bash
f3d /home/sybbure/Desktop/REWIND_SENIOR_DESIGN/glove_sim/outputs/frames/000000_glove_overlay.glb
```
**Pass criteria:**
- Glove parts are connected (same as diagnostic)
- Glove is positioned somewhere near the right hand (calibration may still be off — that is a separate task)
- Red sensor dots visible and on the correct parts

- [ ] **Step 3: Commit if any remaining tweaks were made**

```bash
git add -p  # stage only what changed
git commit -m "Verify pipeline GLB output with corrected transforms"
```
