"""Glove alignment: urdfpy FK + scipy IK with joint limits, proxy-plane penalty,
exact trimesh collision gating, golden-seed initialization, and pluggable optimizer
backends (local TRF, basin-hopping, differential evolution)."""

import json
import sys
import numpy as np
import scipy.optimize
import trimesh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from src.urdfpy_vis import ROOT_TO_HANDMOUNT_XYZ

_THUMB_CHAIN = ['revolute_1_0', 'revolute_2_0', 'revolute_3_0', 'revolute_4_0']
_INDEX_CHAIN = ['revolute_5_0', 'revolute_6_0', 'revolute_7_0', 'revolute_8_0', 'revolute_9_0']

# Dome-tip centroid in link-local frame, computed from STL vertex data.
_THUMB_VISUAL_ORIGIN = np.array([-0.00047,  0.0, -0.03095])
_INDEX_VISUAL_ORIGIN = np.array([-0.00259,  0.0, -0.03085])

# hand_mount → urdf root (pure translation, no rotation)
_HM_TO_ROOT = np.eye(4, dtype=np.float64)
_HM_TO_ROOT[:3, 3] = -ROOT_TO_HANDMOUNT_XYZ

_MANO_TO_GLB = np.diag([1.0, -1.0, -1.0, 1.0])

# ---------------------------------------------------------------------------
# Joint bounds [lower, upper] in radians.
# ---------------------------------------------------------------------------
_JOINT_BOUNDS = {j: (-np.pi, np.pi) for j in
                 ['revolute_1_0', 'revolute_2_0', 'revolute_3_0', 'revolute_4_0',
                  'revolute_5_0', 'revolute_6_0', 'revolute_7_0', 'revolute_8_0',
                  'revolute_9_0']}

# ---------------------------------------------------------------------------
# Proxy-plane collision constants (fast, used inside TRF inner loop).
# ---------------------------------------------------------------------------
_THUMB_COLLISION_LINKS = ['xl330_m077_t_1', 'xl_linkage_horn', 'part_2_1']
_INDEX_COLLISION_LINKS = ['part_6', 'xl_housing_1', 'part_2', 'xl_linkage_horn_1']

_COLLISION_PROXIES = {
    'xl330_m077_t_1':    np.array([ 0.00900, -0.02450, -0.00320]),
    'xl_linkage_horn':   np.array([ 0.00300, -0.00297,  0.00716]),
    'part_2_1':          np.array([-0.00496,  0.00250,  0.00061]),
    'part_6':            np.array([-0.00832, -0.00090, -0.05431]),
    'xl_housing_1':      np.array([ 0.01215, -0.02665,  0.00000]),
    'part_2':            np.array([-0.00435,  0.00250,  0.00247]),
    'xl_linkage_horn_1': np.array([ 0.01550, -0.00625,  0.00434]),
}

_COLLISION_MARGIN         = 0.003   # 3 mm
_COLLISION_WEIGHT         = 2000.0  # quadratic: WEIGHT * penetration^2
_COLLISION_LATERAL_CUTOFF = 0.200   # 200 mm — finger links extend up to ~120mm from centroid

# ---------------------------------------------------------------------------
# Exact mesh collision constants.
# Danger links: finger linkages that can contort into the hand.
# Excludes the glove base (hand_mount, xl330_m077_t, xl_housing) which sit
# on the dorsal surface and are protected by the proxy plane.
# ---------------------------------------------------------------------------
_DANGER_LINKS = frozenset({
    'part_2', 'part_3', 'xl_linkage_horn_1', 'part_6',  # index chain
    'part_2_1', 'part_3_1', 'xl_linkage_horn',          # thumb chain
})

# Per-joint direction map: empirically measured with +0.8 rad applied individually.
_JOINT_DORSAL_DIRECTION = {
    'revolute_1_0': 'na',
    'revolute_2_0': 'towards',
    'revolute_3_0': 'away',
    'revolute_4_0': 'away',
    'revolute_5_0': 'away',
    'revolute_6_0': 'towards',
    'revolute_7_0': 'away',
    'revolute_8_0': 'towards',
    'revolute_9_0': 'away',
}

# Hover offset: currently disabled — re-enable only after re-running annotate_fingertips.py.
_HOVER_DIST = 0.0


# ---------------------------------------------------------------------------
# Exact mesh collision: build a per-frame CollisionManager from the hand GLB.
# ---------------------------------------------------------------------------

def _build_hand_manager(hand_glb_path: Path) -> "trimesh.collision.CollisionManager | None":
    """Load the MANO hand mesh for one frame and return a trimesh CollisionManager.

    The manager is in GLB (Y-up) space, matching the DynHaMR hand GLB files.
    Returns None if the file is missing or unparsable.
    """
    if not hand_glb_path.exists():
        return None
    try:
        hand_scene = trimesh.load(str(hand_glb_path), force="scene")
    except Exception:
        return None

    # MANO hand mesh has exactly 778 vertices.
    hand_meshes = [m for m in hand_scene.geometry.values()
                   if isinstance(m, trimesh.Trimesh) and len(m.vertices) == 778]
    if not hand_meshes:
        hand_meshes = [m for m in hand_scene.geometry.values()
                       if isinstance(m, trimesh.Trimesh)]
    if not hand_meshes:
        return None

    manager = trimesh.collision.CollisionManager()
    manager.add_object('mano_hand', hand_meshes[0])
    return manager


def _get_visual_mesh(visual) -> "trimesh.Trimesh | None":
    """Extract a trimesh.Trimesh from a urdfpy Visual, or None."""
    geom = getattr(visual, 'geometry', None)
    if geom is None:
        return None
    mesh = getattr(geom, 'mesh', None)
    if isinstance(mesh, trimesh.Trimesh):
        return mesh
    for m in getattr(geom, 'meshes', []):
        if isinstance(m, trimesh.Trimesh):
            return m
    return None


# ---------------------------------------------------------------------------
# Multi-start candidate generation
# ---------------------------------------------------------------------------

def _build_candidates(chain_joints):
    """Generate sign-aware multi-start seeds for one finger chain."""
    n = len(chain_joints)
    signs = np.array([
        +1.0 if _JOINT_DORSAL_DIRECTION[j] == 'away' else
        -1.0 if _JOINT_DORSAL_DIRECTION[j] == 'towards' else
         0.0
        for j in chain_joints
    ])
    active = [i for i, s in enumerate(signs) if s != 0.0]
    twist  = [i for i, s in enumerate(signs) if s == 0.0]

    Q, H = np.pi / 4, np.pi / 2
    cands = []

    for k in range(1, len(active) + 1):
        q = np.zeros(n)
        for i in active[:k]: q[i] = signs[i] * Q
        cands.append(q.copy())

    for k in range(1, len(active) + 1):
        q = np.zeros(n)
        for i in active[:k]: q[i] = signs[i] * H
        cands.append(q.copy())

    q_base = signs * Q
    for i in twist:
        for tv in (H, -H):
            q = q_base.copy(); q[i] = tv; cands.append(q)

    cands.append(np.zeros(n))
    cands.append(-signs * Q)
    cands.append(-signs * H)
    return cands


_INDEX_Q0_CANDIDATES = _build_candidates(_INDEX_CHAIN)
_THUMB_Q0_CANDIDATES = _build_candidates(_THUMB_CHAIN)


# ---------------------------------------------------------------------------
# Kabsch algorithm
# ---------------------------------------------------------------------------

def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Optimal rotation + translation mapping P → Q (both (N,3))."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    p_c = P.mean(axis=0)
    q_c = Q.mean(axis=0)
    H = (P - p_c).T @ (Q - q_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = q_c - R @ p_c
    return R, t


# ---------------------------------------------------------------------------
# Per-frame root-world transform
# ---------------------------------------------------------------------------

def _compute_T_root_world(T_wrist: np.ndarray, vertices: "np.ndarray | None") -> np.ndarray:
    """4×4 world transform of the URDF root link in MANO world space."""
    if (vertices is not None
            and cfg.ANCHOR_VERTEX_IDS is not None
            and cfg.GLOVE_ANCHOR_PTS_HM is not None):
        skin_pts = vertices[cfg.ANCHOR_VERTEX_IDS]
        R, t = _kabsch(cfg.GLOVE_ANCHOR_PTS_HM, skin_pts)
        T_hm_world = np.eye(4, dtype=np.float64)
        T_hm_world[:3, :3] = R
        T_hm_world[:3,  3] = t
        return T_hm_world @ _HM_TO_ROOT
    return T_wrist @ cfg.T_WRIST_TO_HM @ _HM_TO_ROOT


# ---------------------------------------------------------------------------
# Golden seed I/O
# ---------------------------------------------------------------------------

def load_golden_seed() -> "tuple[np.ndarray, np.ndarray] | tuple[None, None]":
    """Load golden seed joint angles from golden_seed.json.

    Returns (q_thumb (4,), q_index (5,)) or (None, None) if not found.
    """
    if not cfg.GOLDEN_SEED_PATH.exists():
        return None, None
    data = json.loads(cfg.GOLDEN_SEED_PATH.read_text())
    jcfg = data["joint_cfg"]
    q_thumb = np.array([jcfg[j] for j in _THUMB_CHAIN])
    q_index = np.array([jcfg[j] for j in _INDEX_CHAIN])
    return q_thumb, q_index


def save_golden_seed(frame_idx: int, joint_cfg: dict):
    """Persist joint_cfg for frame_idx to golden_seed.json."""
    cfg.GOLDEN_SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "frame_idx": frame_idx,
        "joint_cfg": {k: float(v) for k, v in joint_cfg.items()},
    }
    cfg.GOLDEN_SEED_PATH.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# IK solver (one finger chain) — stratified collision pipeline
# ---------------------------------------------------------------------------

def _ik_finger(robot, chain_joints, tip_link_name, visual_origin,
               target_in_root, collision_links,
               plane_normal_root, plane_point_root, q0=None,
               q0_candidates=None, optimizer='local',
               T_root_world=None, hand_manager=None):
    """Solve IK for one finger chain with stratified exact collision gating.

    Collision strata (outermost TRF loop is NOT affected):
      Stratum 1 — Warm-start gate: exact trimesh check on the warm-start result.
                  If it intersects the hand, fall through to multi-start.
      Stratum 2 — Multi-start filter: all TRF candidates are filtered by exact
                  collision before selecting the best.  If ALL candidates intersect,
                  returns (None, None) → Stratum 3.
      Stratum 3 — Auto-DE fallback: differential evolution with the exact collision
                  penalty injected into the scalar objective.  Triggered only when
                  every TRF candidate failed the exact check.

    Parameters
    ----------
    optimizer    : 'local' | 'global_de' | 'global_bh'
    T_root_world : (4,4) URDF root in MANO world space — required for exact collision.
    hand_manager : trimesh.collision.CollisionManager pre-loaded with this frame's
                   MANO hand mesh in GLB space.  Pass None to skip exact checks.

    Returns (joint_cfg_dict, ik_residual_metres).
    """
    lower_arr = np.array([_JOINT_BOUNDS[j][0] for j in chain_joints])
    upper_arr = np.array([_JOINT_BOUNDS[j][1] for j in chain_joints])
    n         = len(chain_joints)
    n_penalty = len(collision_links)

    # ------------------------------------------------------------------
    # Residual vector: [ik_err(3), proxy_plane_penalty(n_penalty)]
    # Used inside TRF — NO exact collision here (preserves smooth gradients).
    # ------------------------------------------------------------------
    def residual(q_vec):
        fk = robot.link_fk(cfg=dict(zip(chain_joints, q_vec)))

        ik_err = np.full(3, 1e6)
        link_T = {}
        for link, T in fk.items():
            if link.name == tip_link_name:
                ik_err = T[:3, :3] @ visual_origin + T[:3, 3] - target_in_root
            elif link.name in collision_links:
                link_T[link.name] = T

        penalty = np.zeros(n_penalty)
        if plane_normal_root is not None:
            for i, lname in enumerate(collision_links):
                T = link_T.get(lname)
                if T is None:
                    continue
                proxy_local = _COLLISION_PROXIES.get(lname, np.zeros(3))
                proxy_pos = T[:3, :3] @ proxy_local + T[:3, 3]
                v = proxy_pos - plane_point_root
                dist = float(np.dot(v, plane_normal_root))
                if dist >= _COLLISION_MARGIN:
                    continue
                lateral = float(np.linalg.norm(v - dist * plane_normal_root))
                if lateral > _COLLISION_LATERAL_CUTOFF:
                    continue
                penetration = _COLLISION_MARGIN - dist
                penalty[i] = _COLLISION_WEIGHT * (penetration ** 2)

        return np.concatenate([ik_err, penalty])

    def cost(q_vec):
        r = residual(np.clip(q_vec, lower_arr, upper_arr))
        return float(np.dot(r, r))

    def _ik_res_of(q_vec):
        return float(np.linalg.norm(residual(q_vec)[:3]))

    # ------------------------------------------------------------------
    # Backend: local TRF
    # ------------------------------------------------------------------
    def _run_local(start):
        start = np.clip(start, lower_arr, upper_arr)
        res = scipy.optimize.least_squares(
            residual, start,
            bounds=(lower_arr, upper_arr),
            method='trf',
            ftol=cfg.IK_TOL,
            max_nfev=cfg.IK_MAX_NFEV,
        )
        return res.x, float(np.linalg.norm(res.fun[:3]))

    # ------------------------------------------------------------------
    # Exact mesh collision check (Strata 1 & 2 gate, DE penalty).
    # Called OUTSIDE the TRF inner loop only.
    # Transforms each danger-link visual mesh from root-local → GLB space
    # and queries the pre-built MANO hand CollisionManager.
    # ------------------------------------------------------------------
    def _exact_collision(q_vec) -> bool:
        """True if any DANGER_LINK visual mesh intersects the MANO hand mesh."""
        if hand_manager is None or T_root_world is None:
            return False
        fk = robot.link_fk(cfg=dict(zip(chain_joints, q_vec)))
        # root-local → MANO world → GLB (Y-up)
        T_to_glb = _MANO_TO_GLB @ T_root_world
        for link, T_link in fk.items():
            if link.name not in _DANGER_LINKS:
                continue
            for visual in link.visuals:
                mesh = _get_visual_mesh(visual)
                if mesh is None:
                    continue
                # visual.origin: 4×4 offset from link frame to visual mesh frame
                origin = visual.origin if visual.origin is not None else np.eye(4)
                T_vis_glb = T_to_glb @ T_link @ origin
                try:
                    _r = hand_manager.in_collision_single(mesh, transform=T_vis_glb)
                    collides = _r[0] if isinstance(_r, tuple) else bool(_r)
                    if collides:
                        return True
                except Exception:
                    pass
        return False

    # ------------------------------------------------------------------
    # Backend: differential evolution (global, gradient-free).
    # When use_exact_collision=True, injects exact trimesh penalty into the
    # scalar objective — only evaluated for low-IK-error candidates to avoid
    # spending budget on obviously-wrong configurations.
    # ------------------------------------------------------------------
    def _run_de(seed_x=None, use_exact_collision=False):
        n_pop = max(7 * n, 15)
        rng   = np.random.default_rng(0)
        init_pop = rng.uniform(lower_arr, upper_arr, (n_pop, n))
        if seed_x is not None:
            init_pop[0] = np.clip(seed_x, lower_arr, upper_arr)

        _ik_sq_threshold = (3.0 * cfg.IK_RESIDUAL_THRESHOLD) ** 2

        def cost_fn(q_vec):
            q = np.clip(q_vec, lower_arr, upper_arr)
            r = residual(q)
            c = float(np.dot(r, r))
            if use_exact_collision:
                # Only pay for exact check when IK error is low enough to matter.
                if float(np.dot(r[:3], r[:3])) < _ik_sq_threshold:
                    if _exact_collision(q):
                        c += 1.0   # 1.0 m² = ~1000 mm penalty (dominates IK cost)
            return c

        res = scipy.optimize.differential_evolution(
            cost_fn,
            list(zip(lower_arr.tolist(), upper_arr.tolist())),
            init=init_pop,
            updating='deferred',
            maxiter=150,
            tol=1e-10,
            seed=0,
            # Skip auto-polish when exact collision is in the objective;
            # L-BFGS-B (used by polish) doesn't handle the discontinuous penalty.
            polish=(not use_exact_collision),
        )
        x = np.clip(res.x, lower_arr, upper_arr)

        # Manual TRF polish after exact-collision DE: refine with smooth residual.
        if use_exact_collision:
            xp, rp = _run_local(x)
            if not _exact_collision(xp) and not _is_clipping(xp):
                x = xp

        return x, _ik_res_of(x)

    # ------------------------------------------------------------------
    # Backend: basin hopping (global kicks + local TRF).
    # Each hop's result is exact-collision-gated via accept_test so that
    # intersecting configurations can never become the next starting point.
    # ------------------------------------------------------------------
    def _run_bh(start):
        start = np.clip(start, lower_arr, upper_arr)

        def _local_trf(fun, x0, args=(), **options):
            # Clip after BH random step (step can push outside bounds).
            x0 = np.clip(x0, lower_arr, upper_arr)
            r = scipy.optimize.least_squares(
                residual, x0,
                bounds=(lower_arr, upper_arr),
                method='trf',
                ftol=cfg.IK_TOL,
                max_nfev=cfg.IK_MAX_NFEV,
            )
            class _Result:
                x   = r.x
                fun = float(np.dot(r.fun, r.fun))
                success = True
                message = 'TRF'
            return _Result()

        def _hop_accept(f_new, x_new, f_old, x_old):
            """Reject any hop that causes exact mesh intersection.
            Called by basinhopping AFTER the local minimizer converges.
            Returning False keeps x_old as the current state; basinhopping
            then tries a different random step next iteration.
            """
            if _exact_collision(np.clip(x_new, lower_arr, upper_arr)):
                return False
            return True   # standard Metropolis criterion applies

        res = scipy.optimize.basinhopping(
            cost, start,
            niter=10,
            minimizer_kwargs={"method": _local_trf},
            accept_test=_hop_accept,
            seed=0,
        )
        x = np.clip(res.x, lower_arr, upper_arr)
        return x, _ik_res_of(x)

    # ------------------------------------------------------------------
    # Proxy-plane clipping detection
    # ------------------------------------------------------------------
    def _is_clipping(q_vec) -> bool:
        if plane_normal_root is None:
            return False
        fk = robot.link_fk(cfg=dict(zip(chain_joints, q_vec)))
        for link, T in fk.items():
            if link.name not in collision_links:
                continue
            proxy_local = _COLLISION_PROXIES.get(link.name, np.zeros(3))
            pos = T[:3, :3] @ proxy_local + T[:3, 3]
            v = pos - plane_point_root
            dist = float(np.dot(v, plane_normal_root))
            if dist < 0:
                lateral = float(np.linalg.norm(v - dist * plane_normal_root))
                if lateral <= _COLLISION_LATERAL_CUTOFF:
                    return True
        return False

    # ------------------------------------------------------------------
    # Best-of selection with stratified filtering.
    #
    # Priority: non-clipping AND non-intersecting (exact) → best IK residual.
    # Fallback when no exact checker: proxy-only filter.
    # Returns (None, None) when EVERY candidate intersects — signals DE fallback.
    # ------------------------------------------------------------------
    def _best_of(results):
        # Tier 1: passes both proxy plane AND exact mesh check.
        tier1 = [(x, r) for x, r in results
                 if not _is_clipping(x) and not _exact_collision(x)]
        if tier1:
            return min(tier1, key=lambda t: t[1])

        # Tier 2: no exact checker available → proxy-only filter (legacy behaviour).
        if hand_manager is None:
            proxy_ok = [(x, r) for x, r in results if not _is_clipping(x)]
            pool = proxy_ok if proxy_ok else results
            return min(pool, key=lambda t: t[1])

        # All candidates intersect the hand mesh → signal DE fallback.
        return None, None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    if optimizer == 'global_de':
        best_x, best_res = _run_de(
            seed_x=q0,
            use_exact_collision=(hand_manager is not None),
        )
        return dict(zip(chain_joints, best_x)), best_res

    if optimizer == 'global_bh':
        start = q0 if q0 is not None else np.zeros(n)
        best_x, best_res = _run_bh(start)
        # Exact collision gate on BH result; fall back to DE if it intersects.
        if hand_manager is not None and _exact_collision(best_x):
            best_x, best_res = _run_de(seed_x=best_x, use_exact_collision=True)
        return dict(zip(chain_joints, best_x)), best_res

    # ------------------------------------------------------------------
    # 'local': warm-start fast path → multi-start → auto-DE fallback
    # ------------------------------------------------------------------
    starts = list(q0_candidates) if q0_candidates else [np.zeros(n)]

    if q0 is not None:
        # Stratum 1: try warm-start.
        best_x, best_res = _run_local(q0)

        warm_ok = (
            not _is_clipping(best_x)
            and best_res < cfg.IK_RESIDUAL_THRESHOLD
            and not _exact_collision(best_x)  # exact gate
        )
        if warm_ok:
            return dict(zip(chain_joints, best_x)), best_res

        # Warm-start failed one of the gates → run all candidates.
        candidates_results = [(best_x, best_res)]
        for start in starts:
            candidates_results.append(_run_local(start))

    else:
        # No warm-start: exhaustive multi-start over all sign-aware candidates.
        candidates_results = [_run_local(s) for s in starts]

    # Stratum 2: filter candidates by proxy plane + exact mesh.
    best_x, best_res = _best_of(candidates_results)

    # Stratum 3: all TRF candidates intersect → global DE with exact penalty.
    if best_x is None:
        best_x, best_res = _run_de(seed_x=q0, use_exact_collision=True)

    return dict(zip(chain_joints, best_x)), best_res


# ---------------------------------------------------------------------------
# Per-frame solve
# ---------------------------------------------------------------------------

def solve_ik_frame(robot, T_wrist, thumb_tip_world, index_tip_world,
                   prev_q_thumb=None, prev_q_index=None, vertices=None,
                   vertex_normals=None, optimizer='local', hybrid_threshold=None,
                   hand_manager=None):
    """Solve IK for both fingers in one frame.

    Parameters
    ----------
    optimizer        : 'local' | 'global_de' | 'global_bh'
    hybrid_threshold : float | None.  When set and optimizer='local', any finger
                       whose local-TRF residual exceeds this value (metres) is
                       automatically re-solved with global_de.
    hand_manager     : trimesh.collision.CollisionManager built from this frame's
                       MANO hand GLB (GLB space).  Pass None to skip exact checks.

    Returns (joint_cfg dict, thumb_residual m, index_residual m).
    """
    if vertex_normals is not None:
        thumb_tip_world = thumb_tip_world + _HOVER_DIST * vertex_normals[745]
        index_tip_world = index_tip_world + _HOVER_DIST * vertex_normals[317]

    R_w = T_wrist[:3, :3]
    thumb_tip_world = thumb_tip_world + R_w @ cfg.THUMB_TIP_OFFSET
    index_tip_world = index_tip_world + R_w @ cfg.INDEX_TIP_OFFSET

    T_root_world = _compute_T_root_world(T_wrist, vertices)
    inv_T = np.linalg.inv(T_root_world)

    centroid_mano = (_MANO_TO_GLB @ np.append(cfg.HAND_DORSAL_CENTROID_GLB, 1.0))[:3]
    normal_mano   = _MANO_TO_GLB[:3, :3] @ cfg.HAND_DORSAL_NORMAL_GLB
    plane_centroid_root = (inv_T @ np.append(centroid_mano, 1.0))[:3]
    plane_normal_root   = inv_T[:3, :3] @ normal_mano
    if float(np.dot(-plane_centroid_root, plane_normal_root)) < 0:
        plane_normal_root = -plane_normal_root

    # Cap swap: MANO thumb (v745) → index chain; MANO index (v317) → thumb chain.
    thumb_target = (inv_T @ np.append(thumb_tip_world, 1.0))[:3]
    index_target = (inv_T @ np.append(index_tip_world, 1.0))[:3]

    # Common kwargs forwarded to both _ik_finger calls.
    ik_kwargs = dict(
        optimizer=optimizer,
        T_root_world=T_root_world,
        hand_manager=hand_manager,
    )

    thumb_cfg, thumb_res = _ik_finger(
        robot, _THUMB_CHAIN, 'part_3', _THUMB_VISUAL_ORIGIN, index_target,
        _THUMB_COLLISION_LINKS, plane_normal_root, plane_centroid_root,
        q0=prev_q_thumb, q0_candidates=_THUMB_Q0_CANDIDATES, **ik_kwargs)

    # Hybrid IK-residual fallback (separate from the exact-collision auto-DE
    # stratum inside _ik_finger): if TRF missed the target by a lot, re-solve
    # with global DE even when no exact collision was detected.
    if hybrid_threshold is not None and optimizer == 'local' and thumb_res > hybrid_threshold:
        c, r = _ik_finger(
            robot, _THUMB_CHAIN, 'part_3', _THUMB_VISUAL_ORIGIN, index_target,
            _THUMB_COLLISION_LINKS, plane_normal_root, plane_centroid_root,
            optimizer='global_de', T_root_world=T_root_world, hand_manager=hand_manager)
        if r < thumb_res:
            thumb_cfg, thumb_res = c, r

    index_cfg, index_res = _ik_finger(
        robot, _INDEX_CHAIN, 'part_3_1', _INDEX_VISUAL_ORIGIN, thumb_target,
        _INDEX_COLLISION_LINKS, plane_normal_root, plane_centroid_root,
        q0=prev_q_index, q0_candidates=_INDEX_Q0_CANDIDATES, **ik_kwargs)

    if hybrid_threshold is not None and optimizer == 'local' and index_res > hybrid_threshold:
        c, r = _ik_finger(
            robot, _INDEX_CHAIN, 'part_3_1', _INDEX_VISUAL_ORIGIN, thumb_target,
            _INDEX_COLLISION_LINKS, plane_normal_root, plane_centroid_root,
            optimizer='global_de', T_root_world=T_root_world, hand_manager=hand_manager)
        if r < index_res:
            index_cfg, index_res = c, r

    return {**thumb_cfg, **index_cfg}, thumb_res, index_res


def _assert_frame(joint_cfg, thumb_res, index_res, frame_idx):
    assert thumb_res < cfg.IK_RESIDUAL_THRESHOLD, (
        f"Frame {frame_idx} thumb IK: {thumb_res*1000:.1f}mm "
        f"> {cfg.IK_RESIDUAL_THRESHOLD*1000:.0f}mm tolerance"
    )
    assert index_res < cfg.IK_RESIDUAL_THRESHOLD, (
        f"Frame {frame_idx} index IK: {index_res*1000:.1f}mm "
        f"> {cfg.IK_RESIDUAL_THRESHOLD*1000:.0f}mm tolerance"
    )
    for name, angle in joint_cfg.items():
        assert -np.pi <= angle <= np.pi, (
            f"Frame {frame_idx} joint {name}: {np.degrees(angle):.1f}° "
            f"out of [-180°, 180°]"
        )


def _export_frame(robot, T_wrist, joint_cfg, frame_idx, vertices=None):
    """Export combined hand+glove GLB for one frame."""
    from src.urdfpy_vis import get_glove_scene

    T_root_world = _compute_T_root_world(T_wrist, vertices)
    T_root_to_hm = np.eye(4)
    T_root_to_hm[:3, 3] = ROOT_TO_HANDMOUNT_XYZ
    T_hand_mount = T_root_world @ T_root_to_hm
    glove_scene = get_glove_scene(robot, joint_cfg, T_hand_mount)
    meshes = list(glove_scene.geometry.values())

    hand_glb = cfg.GLB_DIR / f"{frame_idx:06d}_hands.glb"
    if hand_glb.exists():
        hand_scene = trimesh.load(str(hand_glb), force="scene")
        meshes.extend(hand_scene.geometry.values())
    else:
        print(f"[WARNING] Hand GLB not found: {hand_glb}")

    out_path = cfg.GLB_FRAMES_DIR / f"{frame_idx:06d}_aligned.glb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Scene(meshes).export(str(out_path))
    return out_path


def export_frame_urdf(robot, joint_cfg, T_root_world, frame_idx) -> Path:
    """Write a snapshot URDF for a single solved frame."""
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation as Rot

    fk = robot.link_fk(cfg=joint_cfg)
    link_T = {link.name: T for link, T in fk.items()}
    joint_parents = {j.name: (j.parent, j.child) for j in robot.joints}

    def mat_to_xyzrpy(T):
        xyz = T[:3, 3]
        rpy = Rot.from_matrix(T[:3, :3]).as_euler('xyz')
        return (f"{xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}",
                f"{rpy[0]:.8f} {rpy[1]:.8f} {rpy[2]:.8f}")

    tree = ET.parse(str(cfg.URDF_PATH))
    root_elem = tree.getroot()
    root_elem.set('name', f'rewind_glove_frame_{frame_idx:06d}')

    mesh_dir_abs = str(cfg.MESH_DIR.resolve())
    for mesh_elem in root_elem.iter('mesh'):
        fname = mesh_elem.get('filename', '')
        if fname.startswith('../meshes/'):
            abs_fname = mesh_dir_abs + '/' + fname[len('../meshes/'):]
            mesh_elem.set('filename', abs_fname)

    for joint_elem in root_elem.findall('joint'):
        jtype = joint_elem.get('type')
        jname = joint_elem.get('name')
        if jtype not in ('revolute', 'continuous'):
            continue
        parent_name, child_name = joint_parents[jname]
        T_parent = link_T.get(parent_name, np.eye(4))
        T_child  = link_T.get(child_name,  np.eye(4))
        T_rel    = np.linalg.inv(T_parent) @ T_child
        xyz_str, rpy_str = mat_to_xyzrpy(T_rel)

        joint_elem.set('type', 'fixed')
        origin_elem = joint_elem.find('origin')
        if origin_elem is None:
            origin_elem = ET.SubElement(joint_elem, 'origin')
        origin_elem.set('xyz', xyz_str)
        origin_elem.set('rpy', rpy_str)
        for tag in ('axis', 'limit', 'dynamics', 'safety_controller'):
            for sub in joint_elem.findall(tag):
                joint_elem.remove(sub)

    base_link_name = robot.base_link.name
    world_link_elem = ET.Element('link')
    world_link_elem.set('name', 'world')
    root_elem.insert(0, world_link_elem)

    xyz_str, rpy_str = mat_to_xyzrpy(T_root_world)
    world_joint_elem = ET.Element('joint')
    world_joint_elem.set('name', 'world_to_glove')
    world_joint_elem.set('type', 'fixed')
    ET.SubElement(world_joint_elem, 'parent').set('link', 'world')
    ET.SubElement(world_joint_elem, 'child').set('link', base_link_name)
    origin_elem = ET.SubElement(world_joint_elem, 'origin')
    origin_elem.set('xyz', xyz_str)
    origin_elem.set('rpy', rpy_str)
    root_elem.append(world_joint_elem)

    out_dir = cfg.URDF_FRAMES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'rewind_glove_{frame_idx:06d}.urdf'
    tree.write(str(out_path), encoding='unicode', xml_declaration=True)
    return out_path


def _total_frames() -> int:
    data = np.load(str(cfg.NPZ_PATH), allow_pickle=False)
    return int(data["trans"].shape[1])


def main():
    import argparse
    from src.urdfpy_vis import load_robot
    from src.mano_io import load_frame

    parser = argparse.ArgumentParser(
        description="Align glove URDF onto DynHaMR hand frames and export GLBs."
    )
    parser.add_argument(
        "--frames", nargs="+", type=int, default=None,
        help="Explicit list of frame indices to process.",
    )
    parser.add_argument(
        "--frames-test", action="store_true",
        help="Quick test: 10 evenly-spaced frames from SEQUENCE_START with strict assertions.",
    )
    parser.add_argument(
        "--urdf", action="store_true",
        help="Also export per-frame snapshot URDFs.",
    )
    parser.add_argument(
        "--optimizer", choices=['local', 'global_de', 'global_bh'], default='local',
        help=(
            "'local' = TRF (fast).  "
            "'global_de' = differential evolution (global, ~25× slower).  "
            "'global_bh' = basin hopping with TRF steps (~10× slower)."
        ),
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Run local TRF first; auto-escalate to global_de if IK residual > threshold.",
    )
    parser.add_argument(
        "--hybrid-threshold", type=float, default=cfg.IK_RESIDUAL_THRESHOLD,
        metavar="METRES",
        help="IK residual (m) above which hybrid triggers DE (default: %(default)s).",
    )
    parser.add_argument(
        "--save-golden", action="store_true",
        help=f"Save the last processed frame's angles to {cfg.GOLDEN_SEED_PATH}.",
    )
    args = parser.parse_args()

    if args.frames is not None and args.frames_test:
        parser.error("--frames and --frames-test are mutually exclusive.")

    hybrid_threshold = args.hybrid_threshold if args.hybrid else None

    n_total = _total_frames()
    start   = cfg.SEQUENCE_START

    if args.frames_test:
        span    = n_total - 1 - start
        indices = [start + int(round(i * span / 9)) for i in range(10)]
        strict  = True
        print(f"Test mode: {len(indices)} evenly-spaced frames [{indices[0]}…{indices[-1]}].")
    elif args.frames is not None:
        indices = args.frames
        strict  = True
        print(f"Explicit frames: {indices}")
    else:
        indices = list(range(start, n_total))
        strict  = False
        print(f"Full sequence: frames {start}–{n_total-1} ({len(indices)} frames) — failures skipped.")

    print(f"Optimizer: {args.optimizer}"
          + (f"  [hybrid threshold={hybrid_threshold*1000:.1f}mm]" if hybrid_threshold else ""))

    # ------------------------------------------------------------------
    # Golden seed: joint angles only — base is always placed from live wrist.
    # ------------------------------------------------------------------
    golden_q_thumb, golden_q_index = load_golden_seed()
    if golden_q_thumb is not None:
        seed_frame = json.loads(cfg.GOLDEN_SEED_PATH.read_text()).get('frame_idx', '?')
        print(f"Golden seed loaded from frame {seed_frame}: "
              f"thumb={np.degrees(golden_q_thumb).round(1).tolist()}°  "
              f"index={np.degrees(golden_q_index).round(1).tolist()}°")
        if indices:
            first = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, indices[0])
            T_root_first = _compute_T_root_world(first["T_wrist"], first.get("vertices"))
            print(f"  Glove base origin frame {indices[0]}: "
                  f"{T_root_first[:3, 3].round(4).tolist()}  "
                  f"(from live wrist, not seed)")
    else:
        print("No golden seed found — first frame uses cold multi-start.")

    robot = load_robot(cfg.URDF_PATH, cfg.MESH_DIR)
    print(f"URDF loaded: {len(robot.links)} links, {len(robot.actuated_joints)} actuated joints")

    prev_q_thumb = golden_q_thumb
    prev_q_index = golden_q_index
    out_paths  = []
    urdf_paths = []
    failed     = []
    last_good  = None

    for frame_idx in indices:
        try:
            frame = load_frame(cfg.NPZ_PATH, cfg.MANO_DIR, frame_idx)

            # Build exact collision manager for this frame (GLB space).
            hand_glb     = cfg.GLB_DIR / f"{frame_idx:06d}_hands.glb"
            hand_manager = _build_hand_manager(hand_glb)

            joint_cfg, thumb_res, index_res = solve_ik_frame(
                robot, frame["T_wrist"], frame["thumb_tip"], frame["index_tip"],
                prev_q_thumb=prev_q_thumb,
                prev_q_index=prev_q_index,
                vertices=frame.get("vertices"),
                vertex_normals=frame.get("vertex_normals"),
                optimizer=args.optimizer,
                hybrid_threshold=hybrid_threshold,
                hand_manager=hand_manager,
            )

            if strict:
                _assert_frame(joint_cfg, thumb_res, index_res, frame_idx)
                print(f"  [{frame_idx:05d}] thumb {thumb_res*1000:.1f}mm  "
                      f"index {index_res*1000:.1f}mm  OK")
            else:
                _assert_frame(joint_cfg, thumb_res, index_res, frame_idx)
                if frame_idx % 100 == 0:
                    print(f"  [{frame_idx:05d}] thumb {thumb_res*1000:.1f}mm  "
                          f"index {index_res*1000:.1f}mm")

            T_root_world = _compute_T_root_world(frame["T_wrist"], frame.get("vertices"))

            out_path = _export_frame(robot, frame["T_wrist"], joint_cfg, frame_idx,
                                     vertices=frame.get("vertices"))
            out_paths.append(out_path)

            if args.urdf:
                urdf_path = export_frame_urdf(robot, joint_cfg, T_root_world, frame_idx)
                urdf_paths.append(urdf_path)

            prev_q_thumb = np.array([joint_cfg[j] for j in _THUMB_CHAIN])
            prev_q_index = np.array([joint_cfg[j] for j in _INDEX_CHAIN])
            last_good = (frame_idx, dict(joint_cfg))

        except Exception as exc:
            if strict:
                raise
            print(f"  [SKIP] Frame {frame_idx}: {exc}")
            failed.append(frame_idx)
            prev_q_thumb = None
            prev_q_index = None

    if args.save_golden:
        if last_good is not None:
            fi, jcfg = last_good
            save_golden_seed(fi, jcfg)
            print(f"\nGolden seed saved (frame {fi}) → {cfg.GOLDEN_SEED_PATH}")
        else:
            print("\n[WARNING] --save-golden: no successful frames to save.")

    print(f"\n{'='*60}")
    print(f"Done. {len(out_paths)} GLBs exported to {cfg.GLB_FRAMES_DIR}")
    if args.urdf:
        print(f"      {len(urdf_paths)} URDFs exported to {cfg.URDF_FRAMES_DIR}")
    print(f"      {len(failed)} frames skipped.")
    if failed:
        print(f"Failed frames: {failed}")
    if out_paths and strict:
        print(f"\n*** STOP. Review GLBs visually before processing the full sequence. ***")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
