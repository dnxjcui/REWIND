# URDFPy Rest-First Export/Compare Design

## Goal
Add a deterministic Python tool and corresponding tests that:
- exports GLBs from native `urdfpy` and in-house `urdfpy_vis`,
- compares visualization outputs directly,
- enforces a **rest-pose-first** pass gate before running a small sweep.

This is focused on visualization correctness only (not IK).

## Scope
- In scope:
  - New script to export and compare native vs in-house GLBs.
  - Rest-first comparison workflow.
  - Small sweep comparison after rest passes.
  - Test updates that use the same export/compare path and fail on mismatches.
- Out of scope:
  - Viewer/OpenGL runtime issues in `pyglet`/`pyrender`.
  - MuJoCo IK solving behavior.

## Problem Statement
Notebook/manual exports can look disconnected even when some parity checks pass. This suggests mismatch risk in visualization output flow (pose framing and exported scene placement), not only URDF loading.

Need: one authoritative CLI/export pipeline and tests that validate **actual exported GLBs**.

## Design Overview

### 1) New Script: `glove_sim/compare_visual_exports.py`
Primary behavior:
1. Build native oracle robot from canonical URDF with minimal sanitation:
   - `package://...` mesh path rewrite to URDF-relative paths.
   - remove invalid empty `<texture/>`.
2. Build in-house robot via `load_robot(...)`.
3. Export native and custom GLBs for **rest pose**.
4. Compare rest outputs. If fail:
   - write diagnostics + artifacts,
   - return non-zero,
   - do not continue.
5. If rest passes, run small sweep exports and compare all.

CLI options:
- `--out-dir` (default: `glove_sim/outputs/parity_exports`)
- `--rest-atol` (strict default)
- `--sweep-atol` (default same as rest unless provided)
- `--only-rest` (optional debug flag)

Exit behavior:
- `0` only if all required comparisons pass.
- non-zero on first rest failure or any sweep failure.

### 2) Comparison Metrics
For each config:
- exact `n_geom` equality.
- scene bounds comparison (`bounds_min`, `bounds_max`) using tolerance.
- geometry signature bucket parity:
  - key: `(n_vertices, n_faces, rounded extents)`.
  - within bucket: sorted centroids compared by tolerance.
- optional strict mode for per-geometry bounds (internal helper, default on if stable).

These metrics are chosen to detect detached/shifted components and chain disconnections while being robust to geometry iteration order.

### 3) Config Set (Small Sweep)
- `rest`
- `thumb45` (`revolute_3_0 = 45°`)
- `index45` (`revolute_9_0 = 45°`)
- `combo45` (both above)
- `index_sweep`: 0..90° step 15° for `revolute_9_0`

## Rest-First Fix Workflow
1. Run script (rest gate first).
2. If rest fails, patch visualization logic first (expected area: frame/pose conversion in `get_glove_scene` usage and helper assumptions).
3. Re-run until rest passes.
4. Then validate sweep cases.

This prevents noisy cascade failures and keeps debugging focused.

### Mandatory Iteration Rule
Implementation is not considered done until in-house loader/visualizer output is in complete parity with native urdfpy visualizer/loader for the defined config suite and tolerances. Do not stop at partial improvements; continue editing and re-running parity checks until all gates pass.

## Test Updates
Update/add tests in `glove_sim/tests/` to:
- call export/compare helper logic used by the script (single source of truth).
- include explicit rest-first assertion behavior.
- verify both:
  - config-level parity results,
  - artifact creation on mismatch.

Test pass criteria:
- rest comparison passes.
- all small-sweep comparisons pass.
- no tolerance violations in any checked metric.

## Diagnostics and Artifacts
On mismatch, write:
- `native.glb`, `custom.glb` per failing config under output subfolder.
- `diff.json` containing:
  - failing metric,
  - observed delta(s),
  - tolerance used,
  - config/joint values.

Retain only latest run’s artifacts by default to avoid clutter.

## Risks and Mitigations
- Risk: false mismatches from geometry order.
  - Mitigation: signature-bucket matching and centroid sorting.
- Risk: over-loose tolerance hides real bugs.
  - Mitigation: strict defaults + explicit CLI overrides.
- Risk: duplicate signatures in rare meshes.
  - Mitigation: include secondary tie-breakers and report ambiguous buckets in diagnostics.

## Acceptance Criteria
- Script can be run in one command and produces native/custom GLBs.
- Rest pose must pass before sweep is evaluated.
- Small sweep passes within configured tolerances.
- Tests fail when outputs diverge and provide actionable diagnostics.
- Workflow identifies visualization-placement issues directly from exported artifacts.
- Work continues iteratively until parity gates are fully green (no early stop on partial fixes).
