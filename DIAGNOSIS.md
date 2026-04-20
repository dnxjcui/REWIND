# MuJoCo Pipeline Diagnosis and IK Guidance

## Purpose

This document describes the current MuJoCo pipeline in `glove_sim`, why rest GLB parity previously diverged from native URDFpy, what was changed to enforce parity, and how to use the uploaded Dyn-HaMR/MANO assets to run inverse kinematics for fingertip-cap alignment.

The intended outcomes are:

1. deterministic MuJoCo-vs-URDFpy rest visualization parity, and
2. a concrete path to optimize glove joint angles so glove fingertip caps align with MANO fingertip targets.

---

## 1) Current End-to-End Pipeline

## 1.1 Entry points

- `glove_sim/pipeline.py`:
  - frame-sequence pipeline for MANO trajectory ingestion, MuJoCo IK solve, sensor export, and per-frame GLB overlays.
- `glove_sim/diagnostic.py`:
  - standalone MuJoCo diagnostic exporter (rest and index sweep), optional URDFpy side-by-side export.
- `glove_sim/mujoco_device_setup.py`:
  - setup/bring-up tool for one-shot MuJoCo model validation, rest GLB export, and now native URDFpy rest parity checks.

## 1.2 Data path

1. Dyn-HaMR smooth-fit NPZ (`config.NPZ_PATH`) is loaded by `src/mano_loader.py`.
2. The right-hand track is selected (`is_right`-based).
3. MANO forward pass yields:
   - `joints`: `(T, 16, 3)`
   - `vertices`: `(T, 778, 3)`
   - `root_orient`, `trans`, `betas`
4. MuJoCo glove model is instantiated via `src/glove_ik.GloveSimulator`.
5. Per frame:
   - compute `T_wrist` from MANO root orientation + translation
   - apply `T_base = T_wrist @ T_wrist_to_base`
   - set freejoint base pose in MuJoCo
   - solve IK for thumb/index fingertip targets (from MANO vertices)
6. Save outputs:
   - sensor NPZ (`outputs/simulated_sensors.npz`)
   - GLB overlays (`outputs/frames/*_glove_overlay.glb`) unless `--no-vis`

---

## 2) Coordinate Frames and Transform Conventions

## 2.1 Internal conventions in current code

- Pipeline comments/documentation describe a MuJoCo-side "Y-down optimization frame."
- Native URDFpy rest export reference for parity is defined by notebook semantics:
  - `URDF.load(...)`
  - `fk = robot.visual_trimesh_fk()`
  - mesh copy + `apply_transform(pose)`
  - `scene.export(...)`

## 2.2 Important distinction

- `mujoco_device_setup.py` now uses a parity-oriented rest export path:
  - MuJoCo rest GLB and native URDFpy rest GLB are both exported and compared in the same run.
- Overlay export (`src/visualize.export_frame_glb`) still contains explicit conversion behavior tailored to mixed hand+glove visualization needs.

---

## 3) Model Construction in MuJoCo

`src/glove_ik.py` constructs MJCF from URDF, with:

- freejoint root (`base_free`) for glove base placement,
- recursive body/joint generation from URDF tree,
- mesh assets injected under `<asset>`,
- fingertip sites for stable Jacobian-based IK (`thumb_tip_site`, `index_tip_site`).

### Current parity-critical implementation details

1. URDF `rpy` is converted to explicit quaternions for MJCF body/geom tags.
2. Export-time mesh placement no longer depends on MuJoCo geom centroid normalization hacks.
3. Instead, STL world transforms are computed by composing:
   - MuJoCo body world transform
   - URDF visual local transform (`origin xyz/rpy`)

This matches `urdfpy.visual_trimesh_fk` semantics for mesh placement.

---

## 4) What Was Wrong Before, and What Was Fixed

## 4.1 Previously observed parity issues

- MuJoCo rest GLB and URDFpy rest GLB disagreed in bounds/geometry signatures.
- Visuals appeared detached or globally reoriented in some exports.

## 4.2 Root causes

1. Transform reconstruction from MuJoCo geoms included normalization assumptions that drifted from URDF visual semantics.
2. Euler/rotation handling ambiguity in URDF-to-MJCF conversion.
3. Export paths did not always compare apples-to-apples artifacts (native and MuJoCo rest GLBs in one report).

## 4.3 Fixes applied

- `src/glove_ik.py`:
  - added URDF visual transform extraction and composition with MuJoCo body transforms.
  - converted URDF rpy to explicit quaternions for MJCF body/geom tags.
- `src/visualize.py`:
  - glove-only export path now uses direct world transforms for parity workflows.
- `mujoco_device_setup.py`:
  - exports both:
    - `rest.glb` (MuJoCo)
    - `native_rest.glb` (URDFpy reference)
  - computes rest parity metrics into `report.json`.
- Added tests:
  - `glove_sim/tests/test_mujoco_urdfpy_rest_parity.py`
  - extended `glove_sim/tests/test_mujoco_device_setup.py` with integration-style report assertions.

---

## 5) Current Acceptance Signal for Rest Parity

`glove_sim/outputs/mujoco_setup/report.json` now includes:

- `mujoco_rest_glb`
- `native_rest_glb`
- `rest_parity.ok`
- `rest_parity.bounds_max_abs_delta`
- `rest_parity.centroid_max_abs_delta`

Expected pass condition:

- `ok == true`
- `rest_parity.ok == true`
- deltas near numeric tolerance (`~1e-8` order)

---

## 6) Uploaded Dyn-HaMR/MANO Assets Found in Repo

These assets are present and usable for IK experiments:

- MANO model:
  - `Dyn-HaMR/_DATA/_DATA/data/mano/MANO_RIGHT.pkl`
- Smooth-fit world results (example keyframe):
  - `Dyn-HaMR/outputs/logs/video-custom/2026-03-25/knot_one_handed-all-shot-0-0--1/knot_one_handed-all-shot-0-0--1/smooth_fit/knot_one_handed_000300_world_results.npz`
- Exported hand GLBs:
  - `Dyn-HaMR/outputs/logs/video-custom/2026-03-25/knot_one_handed-all-shot-0-0--1/knot_one_handed-all-shot-0-0--1/unity_export/frames/`

---

## 7) Practical IK Workflow for Fingertip-Cap Alignment

This section is the actionable path for aligning glove fingertip caps to MANO fingertips.

## 7.1 Targets to use

Use MANO vertices (already in code):

- thumb tip: `VERTEX_THUMB_TIP = 745`
- index tip: `VERTEX_INDEX_TIP = 317`

Reason:

- your Dyn-HaMR MANO output in this pipeline exposes 16 pose joints; fingertip surface vertices are more suitable for fingertip-cap alignment than relying on absent extra fingertip joints.

## 7.2 Calibration strategy

Current default calibration is identity (`default_calibration()`).
This is a bootstrap only.

Recommended staged calibration:

1. Use rest frame where glove physically aligned with hand.
2. Fit `T_wrist_to_base` translation first (orientation fixed).
3. Then fit orientation (small-angle refinement).
4. Re-run short frame subset and track residual drop.

## 7.3 Solver objective and thresholds

Current objective:

- solve thumb and index site positions to MANO targets with damped least squares.

Recommended quality gates:

- frame-level residual target: `<= 0.01 m` for reliability mask baseline
- preferred operating range for close alignment: `<= 0.003-0.005 m` on keyframes
- monitor:
  - median residual
  - p95 residual
  - max residual

## 7.4 Staged validation protocol

1. **Rest parity gate**
   - run `mujoco_device_setup.py`
   - require `rest_parity.ok == true`
2. **Sparse keyframe IK gate**
   - test 5-10 representative frames from knot sequence
   - inspect residual statistics + visual overlays
3. **Short sequence gate**
   - process a short contiguous window (for temporal behavior)
   - inspect continuity and saturation/clipping
4. **Full-sequence gate**
   - run complete frame set
   - report reliability percentage and residual distribution

## 7.5 Suggested next implementation for IK exploration

Add a "trajectory dry-run" mode (small new CLI or option) that:

- loads `knot_one_handed_000300_world_results.npz` (or configurable NPZ),
- runs first `N` frames (for example 60),
- writes:
  - per-frame residuals
  - summary stats (mean/median/p95/max)
  - optional sampled GLB overlays for visual audit.

This provides a fast readiness signal before long runs.

---

## 8) Known Risks and Limitations

1. Calibration is still user-tuned; systematic offsets can persist while residuals look acceptable.
2. Current IK is position-only for two fingertips (thumb/index), not full-hand/contact objective.
3. Solver settings (fixed damping/iterations) may underperform on sharp motion transitions.
4. Joint clipping bounds are broad (`[-2pi, 2pi]`) and may mask physically unrealistic poses.
5. Frame-convention assumptions must remain consistent across Dyn-HaMR outputs, MuJoCo model, and GLB visualization.

---

## 9) Operational Commands

## 9.1 Rest parity bring-up

```bash
conda run -n glove_sim python glove_sim/mujoco_device_setup.py
```

## 9.2 Core regression tests

```bash
conda run -n glove_sim python -m unittest glove_sim.tests.test_compare_visual_exports glove_sim.tests.test_urdfpy_visualization_parity glove_sim.tests.test_mujoco_device_setup glove_sim.tests.test_mujoco_urdfpy_rest_parity -v
```

## 9.3 Pipeline short run

```bash
conda run -n glove_sim python glove_sim/pipeline.py --frames 0 20 --no-vis
```

---

## 10) Summary

- Rest GLB parity between MuJoCo setup export and native URDFpy export is now explicitly tested and reported.
- The MuJoCo visual export path now composes body pose with URDF visual transforms, which is the key to matching URDFpy mesh placement.
- The repo contains all required Dyn-HaMR and MANO assets for immediate fingertip-cap IK exploration using the staged protocol above.

---

## 11) Current Knot Hand-Mount Alignment Pass

A first-pass alignment tool is now in place:

- `glove_sim/knot_alignment.py`

It performs a constrained optimization of `T_wrist_to_base` using keyframes sampled from `knot_one_handed` and emits:

- `glove_sim/outputs/knot_alignment/report.json`
- overlay GLBs `glove_sim/outputs/knot_alignment/alignment_frame_*.glb` (when hand GLB frames are found)

### Metrics used in this pass

For each sampled frame:

1. `flush_distance_m`
   - absolute signed distance from hand-mount position to a dorsum proxy plane built from MANO points:
     - wrist
     - index MCP proxy (`joints[1]`)
     - pinky MCP proxy (`joints[7]`)
2. `orientation_error`
   - `1 - |dot(hand_mount_z_axis, dorsum_normal)|`
3. `horn_to_knuckle_distance_m`
   - Euclidean distance between `xl_linkage_horn_1` body world position and index knuckle proxy.

The objective is a weighted sum of these metrics across sampled frames.

### Interpretation

This is an intentionally lightweight alignment pass to establish a measurable baseline with generated artifacts for review. It does not yet enforce collisions or anatomical constraints.

### Flush-only scope (current target)

This pass is now intentionally scoped to:

- get glove base flush against the back of the hand first,
- keep linkage connectivity intact,
- defer fingertip IK targeting and APF collision terms to a later phase.

Locked gate used in tests:

- sparse 6 keyframes
- `summary.flush_max_m <= 0.005`

Implementation notes for this scope:

- objective prioritizes flush contact terms only,
- deterministic sparse keyframe sampling is used,
- model integrity counters are written in report (`nbody`, `ngeom`, `njnt`),
- a small contact-distance offset is applied in the metric to account for MANO skin/proxy mismatch.

---

## 12) Collision Integration Roadmap (Artificial Potential Fields)

You requested APF-style collision avoidance rather than full rigid-body collision simulation in the IK loop. The recommended implementation pattern is:

1. **Proxy geometries**
   - Add a few simple geoms (capsules/cylinders) as hand volume proxies:
     - dorsum proxy,
     - thumb corridor proxy,
     - index corridor proxy.
2. **Distance query**
   - During each IK iteration, query distances between glove linkage geoms and proxy geoms (`mujoco.mj_geomDistance`).
3. **Soft penalty**
   - When distance falls below a safety radius (for example 5 mm), add a smooth repulsive term to IK objective:
     - zero outside radius,
     - rapidly increasing inside radius.
4. **Combined objective**
   - Minimize:
     - fingertip tracking error
     - + weighted APF collision penalty
5. **Validation**
   - Compare:
     - baseline IK residuals
     - APF-on residuals
     - penetration/near-contact rates from distance traces.

This gives the solver a differentiable avoidance field that encourages linkage motion around the hand volume while keeping runtime tractable.

## 13) Deferred after flush-only gate

After flush-only is accepted visually, the next steps are:

1. reintroduce index-knuckle placement objective (`XL Linkage Horn 1` hovering above index knuckle),
2. add APF collision proxy geoms and penalties inside IK iterations,
3. then move to full fingertip-target IK quality optimization.
