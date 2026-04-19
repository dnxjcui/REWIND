# Glove Visualization Fix & Diagnostic GLB Export

**Date:** 2026-04-19  
**Status:** Approved

---

## Context

The `glove_sim` pipeline produces per-frame GLB files overlaying the simulated glove on the DynHaMR hand mesh. Three bugs were identified:

1. Glove mesh parts appear visually disconnected (float separately at the joints)
2. Glove base is not aligned with the dorsum of the hand (calibration gap)
3. Red sensor dots are not at the correct physical locations

The fix proceeds in a diagnostic-first order: fix the transform bug, verify the glove looks self-consistent in a standalone GLB, then tune sensor positions visually.

---

## Design

### Phase 1 — `diagnostic.py` (standalone glove GLB exporter)

New script at `glove_sim/diagnostic.py`:

- Loads `GloveSimulator` using the same `cfg.URDF_PATH` / `cfg.MESH_DIR` as `pipeline.py`
- Sets freejoint base pose to world origin, identity quaternion — no DynHaMR data needed
- Runs `mj_forward` to propagate kinematics through the full chain
- Exports `outputs/diagnostic/glove_rest.glb` — all joints at 0°, red sensor dots included
- Sweeps `revolute_9_0` (index fingertip cap joint) from 0° to 90° in 10° steps, exporting `outputs/diagnostic/glove_index_000.glb` … `glove_index_090.glb`
- Viewable with F3D (same format as existing frame exports)

`visualize.py` gains `export_glove_only_glb(geom_poses, sensor_positions_ydown, out_path)` — identical to `export_frame_glb` but without the hand GLB loading block. `diagnostic.py` calls this directly.

### Phase 2 — Transform fix in `visualize.py`

One-line change in both `export_frame_glb` and `export_glove_only_glb`:

```python
# Wrong: conjugation formula — pre-flips mesh-local vertices, producing P @ R @ P rotation
T_yu = YDOWN_TO_YUP @ T_yd @ np.linalg.inv(YDOWN_TO_YUP)

# Correct: flip world-space output only — produces P @ R rotation
T_yu = YDOWN_TO_YUP @ T_yd
```

Both formulas give the same translation (`P @ t`). The conjugation gives `P @ R @ P` for rotation, which mirrors each mesh piece's orientation in the Y-Z plane and breaks the visual seam between adjacent parts.

### Phase 3 — Sensor dot position tuning

- **Base sensors** (`thumb_base` → `part_2_1`, `index_base` → `part_2`): use `data.xpos[body_id]`. This equals the revolute joint hinge location by MuJoCo convention. No change required; verify visually.
- **Tip sensors** (`part_3`, `part_3_1`): offset from body origin along the cap's local frame. Current values (`[-0.0626, 0.00244, -0.0233]` for thumb, `[-0.0337, 0.00244, -0.0125]` for index) were estimated. Adjust `SENSOR_TIP_OFFSETS` in `config.py` based on visual inspection of the diagnostic GLB in F3D.

---

## Files Modified

| File | Change |
|------|--------|
| `glove_sim/diagnostic.py` | New script |
| `glove_sim/src/visualize.py` | Add `export_glove_only_glb()`; fix `T_yu` formula in both export functions |
| `glove_sim/config.py` | Tune `SENSOR_TIP_OFFSETS` (post-diagnostic, values TBD) |

---

## Verification

1. `python diagnostic.py` — should produce `outputs/diagnostic/glove_rest.glb` and 10 index-sweep GLBs
2. Open in F3D — all glove parts should be visually connected at their joints; red dots visible
3. Open `glove_index_090.glb` — index fingertip cap should be visibly flexed ~90° from rest
4. If parts still look disconnected after the transform fix, the URDF joint parsing in `_recursive_body` needs further investigation
5. Adjust `SENSOR_TIP_OFFSETS` until red dots sit halfway along the cap meshes
6. Re-run `python pipeline.py --frames 0 5` to confirm the fix carries through to the full pipeline
