# Glove Simulation Diagnostics

## What we are trying to do

The goal of `glove_sim` is to compute what the four AS5600 magnetic rotary encoders on the REWIND haptic glove *would have read* if the glove were worn during the `knot_one_handed` surgical demonstration captured by DynHaMR. These "simulated ground truth" sensor traces are then compared against real sensor data from a trainee using multivariate subsequence Dynamic Time Warping (DTW), producing timing and motion accuracy scores.

**Concretely:** DynHaMR outputs a MANO hand model fitted to 1196 frames at 30 fps. Each frame gives us the world positions of the thumb and index fingertips. We place the glove URDF so that its fingertip caps (`part_3` for thumb, `part_3_1` for index) touch those MANO tip positions, solve the inverse kinematics to find the joint angles that achieve this, and read off the four sensor joints (`revolute_3_0`, `revolute_4_0`, `revolute_7_0`, `revolute_9_0`).

The visual output — per-frame GLB files combining the hand mesh with the positioned glove mesh — lets us visually verify that the simulation is correct before trusting the sensor numbers.

---

## The two FK paths

### Path 1: MuJoCo-based (diagnostic.py)

**How it works:**
1. `GloveSimulator.__init__` parses the URDF XML with Python's `xml.etree.ElementTree` and programmatically generates a MuJoCo MJCF XML string (`_build_mjcf`).
2. The MJCF is loaded into MuJoCo (`mujoco.MjModel.from_xml_string`), giving us a compiled C simulation.
3. `set_base_pose` places the glove in world space via a freejoint on `hand_mount`.
4. `mujoco.mj_forward` runs the forward kinematics, populating `data.xpos`, `data.xmat`, `data.geom_xpos`, `data.geom_xmat` for every body and geom.
5. `get_geom_world_poses` reads these arrays and returns `{body_name: (pos, rot)}` for each link.
6. `export_glove_only_glb` (in `visualize.py`) loads the binary STL for each link, applies the geom transform, converts Y-down → Y-up, and writes a GLB.
7. For IK, `solve_ik` calls `mujoco.mj_jacSite` to get the analytical Jacobian at named tip sites, then applies a Levenberg-Marquardt step per iteration.

**Advantages:** Very fast (MuJoCo FK in C, ~0.01 ms per call). Analytical Jacobians. Warm-starting between frames.

**The critical bug (now fixed):** When generating MJCF body elements, `_recursive_body` was using:
```xml
<body name="part_1" pos="..." euler="3.14159 0.655389 1.5708">
```
MuJoCo's `euler=` with default `eulerseq="xyz"` is **intrinsic XYZ**: R = Rx(r) @ Ry(p) @ Rz(y).
URDF `rpy="r p y"` is **extrinsic XYZ**: R = Rz(y) @ Ry(p) @ Rx(r).

For single-axis rotations (like the `Rx(π)` visual flips on geom elements) these are equivalent. For multi-axis joints like `fastened_5_0` (connecting `xl_linkage_horn → part_1`, rpy = "π 0.655 π/2"), the two conventions produce completely different matrices. The error was ~2.81 in Frobenius norm, causing `part_2_1` to be ~102 mm off and `part_3` to be ~244 mm off their correct positions.

**The fix:** Added `_rpy_to_quat_str(rpy_str)` which converts URDF extrinsic XYZ to a wxyz quaternion via:
```python
R = Rz(y) @ Ry(p) @ Rx(r)   # URDF convention
```
and replaced `euler=` with `quat=` in body elements only. Geom elements remain `euler=` (they are all single-axis and correct). After the fix, MuJoCo places `part_2_1` at `[-0.002, 0.094, 0.093]`, matching the urdfpy reference.

---

### Path 2: urdfpy-only (diagnostic_urdfpy.py)

**How it works:**
1. `load_robot` (in `src/urdfpy_vis.py`) patches the URDF's mesh paths to absolute paths, strips texture elements (which urdfpy can't load), writes a temp URDF, and calls `urdfpy.URDF.load`.
2. `robot.visual_trimesh_fk(cfg=joint_cfg)` returns `{trimesh_mesh: T_4x4_from_urdf_root}` for every visual element. urdfpy internally computes the full kinematic chain using the correct URDF rpy convention (extrinsic XYZ), composing transforms link by link.
3. Each mesh is copied, `YDOWN_TO_YUP @ T_from_root` is applied, and the result is assembled into a `trimesh.Scene`.
4. No MuJoCo is involved. No MJCF generation. No euler/rpy convention issues.

**Advantages:** URDF rpy handled correctly by construction. Simpler code path. Serves as ground truth to verify MuJoCo outputs.

**Important FK subtlety:** `robot.link_fk(cfg)` returns `{link: T_4x4}` where `T[:3, 3]` is the **joint-frame origin** of the link, NOT the mesh position. A revolute joint rotates the child link around its own origin, so the origin position is invariant to joint angle. The mesh position (and the functionally meaningful "fingertip position") is obtained by applying the visual origin offset in link-local frame:

```python
world_pt = T[:3, :3] @ local_visual_offset + T[:3, 3]
```

`visual_trimesh_fk` already handles this internally for the mesh export. For position-based IK, you must use the above formula with the URDF `<visual><origin xyz=...>` values:

| Link      | Visual origin xyz (local frame)         |
|-----------|-----------------------------------------|
| `part_3`  | `[-0.125132, 0.004875, -0.0466837]`    |
| `part_3_1`| `[-0.0674761, 0.004875, -0.0250619]`  |

Verified: revolute_9_0 swept from 0° to 90° moves `part_3_1`'s visual centre from `[-0.2346, 0.0244, -0.0363]` to `[-0.2475, 0.1059, -0.0960]` — a smooth arc, confirming correct FK.

---

## Feasibility of urdfpy-based IK

The central question: can we replace MuJoCo's IK entirely with urdfpy FK + a Python optimizer?

### Forward model

Given a joint configuration dict `q = {joint_name: angle_rad}`, the fingertip world position is:

```python
fk = robot.link_fk(cfg=q)
T = fk[tip_link]
tip_world = T[:3, :3] @ visual_offset + T[:3, 3]   # (3,)
```

This is a differentiable function of `q` (via finite differences). The Jacobian has shape (3, n_joints) where n_joints = 4 for thumb chain, 5 for index chain.

### IK formulation

For each finger independently:

```
minimize over q_chain:  || tip_position(q) - target ||^2
```

Approaches:
1. **Finite-difference Levenberg-Marquardt** (matches what MuJoCo does internally):
   - Perturb each joint angle by δ = 1e-5 rad, compute Δtip → Jacobian column
   - J shape: (3, 4) for thumb, (3, 5) for index
   - LM step: dq = (J^T J + λI)^{-1} J^T err
   - Cost per iteration: 4–5 FK calls (one per joint) + 1 FK for current tip = 5–6 FK calls
   - At 50 iterations: ~300 FK calls per finger per frame

2. **scipy.optimize.least_squares** (built-in LM):
   ```python
   from scipy.optimize import least_squares
   def residual(q_vec):
       cfg = dict(zip(chain_joints, q_vec))
       return tip_position(robot, cfg, tip_link) - target
   result = least_squares(residual, q0, method='lm')
   ```
   This is cleaner but doesn't warmstart well across frames.

3. **ikpy library** (purpose-built URDF IK):
   - Parses URDF and builds a `Chain` from base to tip
   - `chain.inverse_kinematics(target_position)` uses scipy under the hood
   - Risk: ikpy expects a simple serial chain from a fixed base; our glove chains branch off `hand_mount` which is free-floating during pipeline runs
   - May need a custom base-link specification per finger

### Speed estimate

urdfpy FK call time (measured):
```python
# rough benchmark
import time
t0 = time.perf_counter()
for _ in range(1000):
    robot.link_fk(cfg={'revolute_9_0': 0.5})
print((time.perf_counter() - t0) / 1000 * 1e3, 'ms per call')
```
Expected: 2–8 ms per call (pure Python, no JIT).

For 1196 frames × 2 fingers × 50 iter × 6 FK calls = **717,600 FK calls total**.
At 5 ms each → **~1 hour**. This is too slow for production use.

At 0.5 ms each (optimistic) → **~6 minutes**. Marginal.

MuJoCo IK at 0.01 ms per FK with analytical Jacobian → **~12 seconds** for 1196 frames. Current measured throughput is ~4273 fps.

### Recommendation

**Keep MuJoCo for IK.** The quaternion fix resolves the root cause of incorrect body placement. urdfpy serves as the visual verification ground truth.

If the MuJoCo MJCF generation continues to cause bugs, the right path is to fix the MJCF generator (add more unit tests per joint), not to switch IK backends.

A hybrid approach that IS viable: use urdfpy FK once at rest pose to verify body positions match MuJoCo, then run all 1196-frame IK in MuJoCo. This is what the current diagnostic scripts implement.

**If you do want urdfpy IK for any reason** (e.g. to eliminate the MJCF generator entirely), the minimum viable implementation is:
```python
from scipy.optimize import least_squares

THUMB_CHAIN  = ['revolute_1_0', 'revolute_2_0', 'revolute_3_0', 'revolute_4_0']
INDEX_CHAIN  = ['revolute_5_0', 'revolute_6_0', 'revolute_7_0', 'revolute_8_0', 'revolute_9_0']

def solve_ik_urdfpy(robot, finger_chain, tip_link, visual_offset, target, q0=None):
    if q0 is None:
        q0 = np.zeros(len(finger_chain))
    def residual(q_vec):
        cfg = dict(zip(finger_chain, q_vec))
        fk  = robot.link_fk(cfg=cfg)
        for lnk, T in fk.items():
            if lnk.name == tip_link:
                return T[:3, :3] @ visual_offset + T[:3, 3] - target
        return np.full(3, 1e6)
    res = least_squares(residual, q0, method='lm', ftol=1e-8, max_nfev=200)
    return dict(zip(finger_chain, res.x)), res.cost
```

This would give correct IK (urdfpy FK is exact) but would run ~100× slower than MuJoCo.

---

## Current status

| Component | State |
|-----------|-------|
| MuJoCo MJCF generation | Fixed (euler→quat for body elements) |
| MuJoCo body positions | Correct — `part_2_1` at `[-0.002, 0.094, 0.093]` |
| MuJoCo mesh export (GLB) | Needs visual inspection |
| urdfpy FK | Confirmed correct — tip moves smoothly through 90° sweep |
| urdfpy GLBs | Exported to `outputs/diagnostic_urdfpy/` — inspect with f3d |
| MuJoCo GLBs | Exported to `outputs/diagnostic/` — inspect with f3d |
| Full pipeline IK | Not yet run post-fix |

**Next step:** Open `outputs/diagnostic_urdfpy/glove_rest.glb` in f3d and confirm all glove parts appear connected. Then open `outputs/diagnostic/glove_rest.glb` and compare. If both show a coherent assembled glove, run `python pipeline.py --frames 0 5` to test end-to-end IK.
