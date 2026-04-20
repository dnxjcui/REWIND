# URDFPy Visualization Parity Design

## Goal
Establish strict visualization parity between native `urdfpy` and the in-house loader/visualizer in `glove_sim/src/urdfpy_vis.py`, so the custom path reproduces native visualization behavior exactly (within configurable numeric tolerance), with clear diagnostics for any mismatch.

## Scope
- In scope:
  - Native vs custom parity for URDF loading + visual FK output.
  - Native vs custom parity for exported visualization snapshots.
  - Tolerance-tunable checks and debug artifact generation.
- Out of scope:
  - MuJoCo IK correctness.
  - Live interactive viewer/OpenGL driver reliability (`pyglet`/`pyrender` runtime driver issues).

## Context
Current custom logic in `urdfpy_vis.py`:
- Normalizes ROS-style `package://` mesh paths to paths urdfpy can resolve.
- Removes invalid empty `<texture/>` entries.
- Loads with `URDF.load` from a temp file located beside the source URDF to preserve native path semantics.

Native urdfpy expectations:
- Mesh filename resolution follows `urdfpy.utils.get_filename(base_path, file_path)`:
  - absolute path -> use directly
  - relative path -> `os.path.join(urdf_dir, filename)`

Primary parity target:
- Visualization equivalence (angles, geometry, transforms), not runtime viewer behavior.

## Test Strategy

### Layer A: Behavioral Parity (Oracle = native urdfpy)
Compare outputs from:
- Native path: `URDF.load(rewind_glove_for_urdfpy.urdf)`
- Custom path: `load_robot(rewind_glove_assembly.urdf, mesh_dir)`

For each test config:
- Compute `visual_trimesh_fk(cfg)` on both.
- Match meshes deterministically via stable signature.
- Compare:
  - Transform matrices (elementwise, translation norm, rotation norm).
  - Geometry signatures (vertex count, face count, extents/centroid summaries).

### Layer B: Snapshot Parity (Tolerance-based)
For selected configs (rest + bent + short sweep):
- Export native and custom scenes to GLB.
- Reload both and compare scene-derived metrics:
  - Per-geometry extents and centroid.
  - Per-node transform summaries.
  - Scene bounds.

Byte-level GLB equality is optional and off by default (too brittle); metric parity is the authoritative snapshot criterion.

### Layer C: Visualization Output Parity (Primary focus)
This layer directly targets what is rendered, not just what is loaded.

For each config/frame, compare the visualized component placement between methods:
- Native visualization path:
  - `URDF.load(...).visual_trimesh_fk(cfg)` -> transformed meshes in world frame.
- Custom visualization path:
  - `load_robot(...).visual_trimesh_fk(cfg)` and `get_glove_scene(...)`.

Checks (component-level and scene-level):
- Per-component world transform parity:
  - Translation delta per component.
  - Rotation delta per component (angle-axis or quaternion distance).
- Per-component geometric placement parity:
  - Centroid in world frame.
  - Axis-aligned bounds in world frame.
- Whole-scene parity:
  - Scene centroid and bounds.
  - Inter-component distance matrix for key parts (e.g., finger links) to catch chain disconnections.

This is the decisive parity layer for the suspected failure mode ("wrong 3D placement despite successful loading").

## Config Set
Minimum config suite:
- `rest` (all actuated joints = 0)
- `thumb45`
- `index45`
- `combo45`
- `index_sweep` (0 to 90 degrees in 15 degree steps)

All configs use the same joint names as native urdfpy FK calls.

## Matching and Canonicalization
Because `visual_trimesh_fk` maps `Trimesh` objects (identity-based keys), matching uses deterministic mesh signatures:
- vertex count
- face count
- rounded extents
- rounded centroid
- fallback index for rare collisions

Canonical ordering is applied before pairwise comparisons to avoid iteration-order noise.

## Tolerance Model
All tolerances centralized in test constants (and optionally env overrides):
- `TRANSFORM_ATOL` for matrix elementwise checks
- `GEOM_ATOL` for geometry metrics
- `SCENE_ATOL` for scene bounds
- `SNAPSHOT_ATOL` for snapshot metric parity

Defaults are strict; user can tune for platform-specific floating point variance.

## Failure Diagnostics
On mismatch, tests report:
- config/frame id
- mesh signature and index
- max absolute matrix delta and element location
- translation and rotation error norms
- geometry delta summaries
- snapshot metric deltas
- visualization parity deltas (component-level transforms/centroids/bounds and key inter-component distances)

Persist debug artifacts for failing cases:
- native GLB
- custom GLB
- JSON diff summary

Keep only latest failure artifacts to avoid output bloat.

## Iteration Workflow
1. Run full parity test suite.
2. If failing, inspect diagnostics and adjust `urdfpy_vis` logic.
3. Re-run suite.
4. Repeat until all parity checks pass.

## Design Constraints
- Preserve native urdfpy behavior as oracle.
- Keep custom logic minimal (path normalization + invalid texture cleanup + same load semantics).
- Keep tests deterministic and platform-aware.

## Risks and Mitigations
- Risk: nondeterministic geometry ordering.
  - Mitigation: signature-based matching + canonical sorting.
- Risk: tiny numeric differences across environments.
  - Mitigation: adjustable tolerance model + strict defaults.
- Risk: snapshot brittleness.
  - Mitigation: compare derived scene metrics, not raw bytes.

## Acceptance Criteria
Design is complete when:
- Layer A and Layer B tests pass for the full config suite.
- Layer C visualization parity checks pass for all tested configs/frames.
- Any parity regression produces actionable diagnostics.
- Tolerances are adjustable without editing test logic deeply.
- Custom visualization behavior is reproducible against native urdfpy for the specified cases.
