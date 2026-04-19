"""MuJoCo-based glove simulator with per-frame IK."""

import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from pathlib import Path

from config import (
    THUMB_TIP_BODY, INDEX_TIP_BODY,
    THUMB_CHAIN, INDEX_CHAIN,
    SENSOR_JOINTS, IK_TOL, IK_ITERS,
)
from src.calibration import mat_to_quat_wxyz


def _build_mjcf(urdf_path: Path, mesh_dir: Path) -> str:
    """Parse the URDF and return a MuJoCo XML string.

    Modifications vs. the raw URDF:
      - The virtual `root` link and its fixed joint to `hand_mount` are removed.
      - A <freejoint name="base_free"/> is injected into `hand_mount` so the
        glove can be repositioned per-frame without rebuilding the model.
      - STL mesh paths are resolved to absolute paths.
    """
    tree = ET.parse(str(urdf_path))
    robot = tree.getroot()

    # Collect all joint → child mappings
    joint_parent_child = {
        j.find("child").get("link"): j
        for j in robot.findall("joint")
    }

    # Build MJCF by hand — MuJoCo's URDF loader strips the freejoint we need,
    # so we generate a minimal MJCF that preserves all URDF joints/links.
    # Simpler: use mujoco.MjModel.from_xml_string with a URDF wrapped in <mujoco>.
    # MuJoCo 3.x can load URDF via <include> or directly; we patch the tree.

    # Remove the fixed joint from root → hand_mount
    for j in robot.findall("joint"):
        if (j.find("parent").get("link") == "root" and
                j.find("child").get("link") == "hand_mount"):
            robot.remove(j)
            break

    # Remove the virtual root link (empty)
    for lnk in robot.findall("link"):
        if lnk.get("name") == "root":
            robot.remove(lnk)
            break

    # Fix STL mesh paths to absolute
    mesh_dir_abs = str(mesh_dir.resolve())
    for mesh in robot.iter("mesh"):
        fn = mesh.get("filename", "")
        # strip ROS package:// prefix
        if fn.startswith("package://"):
            fn = fn.split("/", 2)[-1]           # "rewind_glove_assembly/meshes/Foo.stl"
            fn = fn.split("/", 1)[-1]            # "meshes/Foo.stl"
            fn = fn.split("/", 1)[-1]            # "Foo.stl"
        mesh.set("filename", str(Path(mesh_dir_abs) / fn))

    urdf_str = ET.tostring(robot, encoding="unicode")

    # Wrap in <mujoco> with compiler settings and inject the freejoint
    mjcf = f"""
<mujoco model="rewind_glove">
  <compiler angle="radian" meshdir="{mesh_dir_abs}/" />
  <option gravity="0 0 0" />
  <worldbody>
    <body name="hand_mount_world">
      <freejoint name="base_free"/>
      {_urdf_body_to_mjcf(urdf_path, mesh_dir)}
    </body>
  </worldbody>
</mujoco>
"""
    return mjcf


def _urdf_body_to_mjcf(urdf_path: Path, mesh_dir: Path) -> str:
    """Convert the URDF kinematic tree (rooted at hand_mount) to MJCF body XML.

    Returns the inner XML string of child bodies/joints to be nested inside
    the hand_mount_world body.
    """
    tree = ET.parse(str(urdf_path))
    robot = tree.getroot()

    links = {lnk.get("name"): lnk for lnk in robot.findall("link")}
    joints = {}
    for j in robot.findall("joint"):
        child_name = j.find("child").get("link")
        joints[child_name] = j

    mesh_dir_abs = str(mesh_dir.resolve())

    def _rpy_to_euler(rpy_str: str) -> str:
        return rpy_str  # MJCF uses euler in radians same as URDF rpy

    def _body_xml(link_name: str) -> str:
        lnk = links.get(link_name)
        if lnk is None:
            return ""

        # Find the joint that connects parent → this link
        jnt = joints.get(link_name)
        origin_xyz = "0 0 0"
        origin_rpy = "0 0 0"
        joint_xml = ""

        if jnt is not None:
            origin = jnt.find("origin")
            if origin is not None:
                origin_xyz = origin.get("xyz", "0 0 0")
                origin_rpy = origin.get("rpy", "0 0 0")
            jtype = jnt.get("type", "fixed")
            jname = jnt.get("name")
            axis_el = jnt.find("axis")
            axis_xyz = axis_el.get("xyz", "0 0 1") if axis_el is not None else "0 0 1"
            if jtype == "continuous" or jtype == "revolute":
                joint_xml = f'<joint name="{jname}" type="hinge" axis="{axis_xyz}"/>'
            # fixed joints → no joint element in MJCF (body is rigidly attached)

        # Visual geometry → geom
        geom_xml = ""
        visual = lnk.find("visual")
        if visual is not None:
            geom_origin = visual.find("origin")
            geom_xyz = "0 0 0"
            geom_rpy = "0 0 0"
            if geom_origin is not None:
                geom_xyz = geom_origin.get("xyz", "0 0 0")
                geom_rpy = geom_origin.get("rpy", "0 0 0")
            mesh_el = visual.find(".//mesh")
            if mesh_el is not None:
                fn = mesh_el.get("filename", "")
                if fn.startswith("package://"):
                    fn = fn.split("/", 2)[-1]
                    fn = fn.split("/", 1)[-1]
                    fn = fn.split("/", 1)[-1]
                abs_path = str(Path(mesh_dir_abs) / fn)
                geom_xml = (
                    f'<geom type="mesh" mesh="{abs_path}" '
                    f'pos="{geom_xyz}" euler="{geom_rpy}" contype="0" conaffinity="0"/>'
                )

        # Inertial
        inertia_xml = ""
        inertial = lnk.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            mass = mass_el.get("value", "0.001") if mass_el is not None else "0.001"
            orig = inertial.find("origin")
            i_xyz = orig.get("xyz", "0 0 0") if orig is not None else "0 0 0"
            inertia_el = inertial.find("inertia")
            if inertia_el is not None:
                ixx = inertia_el.get("ixx", "1e-9")
                iyy = inertia_el.get("iyy", "1e-9")
                izz = inertia_el.get("izz", "1e-9")
                ixy = inertia_el.get("ixy", "0")
                ixz = inertia_el.get("ixz", "0")
                iyz = inertia_el.get("iyz", "0")
                inertia_xml = (
                    f'<inertial pos="{i_xyz}" mass="{mass}" '
                    f'fullinertia="{ixx} {iyy} {izz} {ixy} {ixz} {iyz}"/>'
                )

        # Children
        children_xml = ""
        for child_name, child_joint in joints.items():
            if child_joint.find("parent").get("link") == link_name:
                children_xml += _body_xml(child_name)

        return (
            f'<body name="{link_name}" pos="{origin_xyz}" euler="{origin_rpy}">'
            f"  {inertia_xml}"
            f"  {joint_xml}"
            f"  {geom_xml}"
            f"  {children_xml}"
            f"</body>"
        )

    # Build children of hand_mount
    out = ""
    for child_name, jnt in joints.items():
        if jnt.find("parent").get("link") == "hand_mount":
            out += _body_xml(child_name)

    # Also add the hand_mount inertial/geom itself
    lnk = links.get("hand_mount")
    inertia_xml = ""
    geom_xml = ""
    if lnk is not None:
        inertial = lnk.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            mass = mass_el.get("value", "0.001") if mass_el is not None else "0.001"
            orig = inertial.find("origin")
            i_xyz = orig.get("xyz", "0 0 0") if orig is not None else "0 0 0"
            inertia_el = inertial.find("inertia")
            if inertia_el is not None:
                ixx = inertia_el.get("ixx", "1e-9")
                iyy = inertia_el.get("iyy", "1e-9")
                izz = inertia_el.get("izz", "1e-9")
                ixy = inertia_el.get("ixy", "0")
                ixz = inertia_el.get("ixz", "0")
                iyz = inertia_el.get("iyz", "0")
                inertia_xml = (
                    f'<inertial pos="{i_xyz}" mass="{mass}" '
                    f'fullinertia="{ixx} {iyy} {izz} {ixy} {ixz} {iyz}"/>'
                )
        visual = lnk.find("visual")
        if visual is not None:
            geom_origin = visual.find("origin")
            geom_xyz = "0 0 0"
            geom_rpy = "0 0 0"
            if geom_origin is not None:
                geom_xyz = geom_origin.get("xyz", "0 0 0")
                geom_rpy = geom_origin.get("rpy", "0 0 0")
            mesh_el = visual.find(".//mesh")
            if mesh_el is not None:
                fn = mesh_el.get("filename", "")
                if fn.startswith("package://"):
                    fn = fn.split("/", 2)[-1]
                    fn = fn.split("/", 1)[-1]
                    fn = fn.split("/", 1)[-1]
                abs_path = str(Path(mesh_dir_abs) / fn)
                geom_xml = (
                    f'<geom type="mesh" mesh="{abs_path}" '
                    f'pos="{geom_xyz}" euler="{geom_rpy}" contype="0" conaffinity="0"/>'
                )

    return inertia_xml + geom_xml + out


class GloveSimulator:
    """MuJoCo-based glove kinematic simulator.

    Positions the glove base per-frame and solves IK to place
    the thumb/index fingertip caps at the MANO fingertip targets.
    """

    def __init__(self, urdf_path: Path, mesh_dir: Path):
        mjcf_str = self._build_model_xml(urdf_path, mesh_dir)
        try:
            self.model = mujoco.MjModel.from_xml_string(mjcf_str)
        except Exception as e:
            # Save the MJCF for debugging
            debug_path = urdf_path.parent / "debug_glove.xml"
            debug_path.write_text(mjcf_str)
            raise RuntimeError(
                f"MuJoCo failed to load MJCF: {e}\nDebug XML saved to {debug_path}"
            )
        self.data = mujoco.MjData(self.model)

        # Cache site IDs for IK (sites have offsets from joint origins → non-zero Jacobians)
        self.thumb_tip_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "thumb_tip_site"
        )
        self.index_tip_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "index_tip_site"
        )
        if self.thumb_tip_site < 0:
            raise RuntimeError("Site 'thumb_tip_site' not found in model")
        if self.index_tip_site < 0:
            raise RuntimeError("Site 'index_tip_site' not found in model")

        # Also cache body IDs for world-pose export
        self.thumb_tip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, THUMB_TIP_BODY
        )
        self.index_tip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, INDEX_TIP_BODY
        )
        if self.thumb_tip_id < 0:
            raise RuntimeError(f"Body '{THUMB_TIP_BODY}' not found in model")
        if self.index_tip_id < 0:
            raise RuntimeError(f"Body '{INDEX_TIP_BODY}' not found in model")

        # Cache sensor joint qpos addresses
        self._sensor_qadr = {}
        for label, jname in SENSOR_JOINTS.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"Sensor joint '{jname}' not found in model")
            self._sensor_qadr[label] = self.model.jnt_qposadr[jid]

        # Cache IK chain dof addresses
        self._thumb_dof_addrs = self._chain_dof_addrs(THUMB_CHAIN)
        self._index_dof_addrs = self._chain_dof_addrs(INDEX_CHAIN)

        # Free joint qpos address (7 values: xyz + quat wxyz)
        free_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
        self._base_qadr = self.model.jnt_qposadr[free_jid]

        # Initialize to neutral
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _build_model_xml(self, urdf_path: Path, mesh_dir: Path) -> str:
        """Build the full MJCF XML string from the URDF."""
        mesh_dir_abs = str(mesh_dir.resolve())
        inner = _urdf_body_to_mjcf(urdf_path, mesh_dir)
        lnk_inertia, lnk_geom, lnk_children = self._hand_mount_parts(urdf_path, mesh_dir)

        # Also collect all mesh assets referenced
        assets = self._collect_mesh_assets(urdf_path, mesh_dir)

        # Site offsets = visual origin of the fingertip cap links (from URDF visual elements).
        # These are the points with meaningful moment arms from the tip joints,
        # necessary for non-degenerate IK Jacobians.
        thumb_site_pos  = "-0.125132 0.004875 -0.0466837"  # part_3 visual origin
        index_site_pos  = "-0.0674761 0.004875 -0.0250619"  # part_3_1 visual origin

        return f"""
<mujoco model="rewind_glove">
  <compiler angle="radian" meshdir="{mesh_dir_abs}/"/>
  <option gravity="0 0 0" integrator="RK4"/>
  <asset>
    {assets}
  </asset>
  <worldbody>
    <body name="hand_mount" pos="0 0 0">
      <freejoint name="base_free"/>
      {lnk_inertia}
      {lnk_geom}
      {lnk_children}
    </body>
  </worldbody>
</mujoco>
"""

    def _hand_mount_parts(self, urdf_path: Path, mesh_dir: Path):
        """Return (inertia_xml, geom_xml, children_xml) for hand_mount."""
        tree = ET.parse(str(urdf_path))
        robot = tree.getroot()
        links = {lnk.get("name"): lnk for lnk in robot.findall("link")}
        joints_map = {
            j.find("child").get("link"): j for j in robot.findall("joint")
        }
        mesh_dir_abs = str(mesh_dir.resolve())
        lnk = links["hand_mount"]
        inertia_xml = _inertial_xml(lnk)
        geom_xml = _visual_geom_xml(lnk, mesh_dir_abs)
        children_xml = ""
        for child_name, jnt in joints_map.items():
            if jnt.find("parent").get("link") == "hand_mount":
                children_xml += _recursive_body(child_name, links, joints_map, mesh_dir_abs)
        return inertia_xml, geom_xml, children_xml

    def _collect_mesh_assets(self, urdf_path: Path, mesh_dir: Path) -> str:
        """Return <mesh> asset declarations for all unique STL files."""
        tree = ET.parse(str(urdf_path))
        robot = tree.getroot()
        mesh_dir_abs = str(mesh_dir.resolve())
        seen = set()
        out = []
        for mesh_el in robot.iter("mesh"):
            fn = mesh_el.get("filename", "")
            if fn.startswith("package://"):
                fn = fn.split("/", 2)[-1]
                fn = fn.split("/", 1)[-1]
                fn = fn.split("/", 1)[-1]
            abs_path = str(Path(mesh_dir_abs) / fn)
            name = Path(fn).stem.replace(" ", "_")
            key = abs_path
            if key not in seen:
                seen.add(key)
                out.append(f'<mesh name="{name}" file="{abs_path}"/>')
        return "\n    ".join(out)

    def _chain_dof_addrs(self, joint_names: list[str]) -> list[int]:
        addrs = []
        for jname in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                addrs.append(self.model.jnt_dofadr[jid])
        return addrs

    def set_base_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        """Set the glove hand_mount world pose via the freejoint."""
        adr = self._base_qadr
        self.data.qpos[adr:adr+3] = pos
        self.data.qpos[adr+3:adr+7] = quat_wxyz

    def solve_ik(
        self,
        thumb_target: np.ndarray,
        index_target: np.ndarray,
        n_iter: int = IK_ITERS,
        tol: float = IK_TOL,
    ) -> float:
        """Solve IK for both fingertip caps. Returns max residual in meters."""
        residuals = []
        for site_id, target, dof_addrs in [
            (self.thumb_tip_site, thumb_target, self._thumb_dof_addrs),
            (self.index_tip_site, index_target, self._index_dof_addrs),
        ]:
            if not dof_addrs:
                continue
            n = len(dof_addrs)
            for _ in range(n_iter):
                mujoco.mj_kinematics(self.model, self.data)
                mujoco.mj_comPos(self.model, self.data)
                # Use site position (offset from joint → non-degenerate Jacobian for tip joints)
                site_pos = self.data.site_xpos[site_id]
                err = target - site_pos
                if np.linalg.norm(err) < tol:
                    break
                jacp = np.zeros((3, self.model.nv))
                # mj_jacSite: Jacobian at the named site
                mujoco.mj_jacSite(self.model, self.data, jacp, None, site_id)
                J = jacp[:, dof_addrs]
                # Damped least-squares (Levenberg-Marquardt)
                lam = 1e-4
                dq = np.linalg.solve(J.T @ J + lam * np.eye(n), J.T @ err)
                for i, dof_adr in enumerate(dof_addrs):
                    qadr = self._dof_to_qadr(dof_adr)
                    if qadr is not None:
                        self.data.qpos[qadr] += dq[i]
                        # Clamp to [-2π, 2π] to prevent unbounded IK drift
                        self.data.qpos[qadr] = np.clip(
                            self.data.qpos[qadr], -2 * np.pi, 2 * np.pi
                        )

            mujoco.mj_kinematics(self.model, self.data)
            residuals.append(float(np.linalg.norm(target - self.data.site_xpos[site_id])))

        return max(residuals) if residuals else 0.0

    def _dof_to_qadr(self, dof_adr: int) -> int | None:
        """Map a dof address to its qpos address (1:1 for hinge joints)."""
        for j in range(self.model.njnt):
            if self.model.jnt_dofadr[j] == dof_adr:
                return self.model.jnt_qposadr[j]
        return None

    def read_sensor_angles(self) -> np.ndarray:
        """Return the 4 sensor joint angles [thumb_base, thumb_tip, index_base, index_tip]."""
        return np.array([self.data.qpos[adr] for adr in self._sensor_qadr.values()])

    def get_link_world_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return {link_name: (pos_3, rotmat_3x3)} in MuJoCo world frame."""
        poses = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name is None:
                continue
            pos = self.data.xpos[i].copy()
            rot = self.data.xmat[i].reshape(3, 3).copy()
            poses[name] = (pos, rot)
        return poses


# ---------------------------------------------------------------------------
# MJCF generation helpers (module-level so _urdf_body_to_mjcf can call them)
# ---------------------------------------------------------------------------

def _inertial_xml(lnk: ET.Element) -> str:
    inertial = lnk.find("inertial")
    if inertial is None:
        return ""
    mass_el = inertial.find("mass")
    mass = mass_el.get("value", "0.001") if mass_el is not None else "0.001"
    orig = inertial.find("origin")
    i_xyz = orig.get("xyz", "0 0 0") if orig is not None else "0 0 0"
    inertia_el = inertial.find("inertia")
    if inertia_el is None:
        return f'<inertial pos="{i_xyz}" mass="{mass}"/>'
    ixx = inertia_el.get("ixx", "1e-9")
    iyy = inertia_el.get("iyy", "1e-9")
    izz = inertia_el.get("izz", "1e-9")
    ixy = inertia_el.get("ixy", "0")
    ixz = inertia_el.get("ixz", "0")
    iyz = inertia_el.get("iyz", "0")
    return (
        f'<inertial pos="{i_xyz}" mass="{mass}" '
        f'fullinertia="{ixx} {iyy} {izz} {ixy} {ixz} {iyz}"/>'
    )


def _visual_geom_xml(lnk: ET.Element, mesh_dir_abs: str) -> str:
    visual = lnk.find("visual")
    if visual is None:
        return ""
    geom_origin = visual.find("origin")
    geom_xyz = "0 0 0"
    geom_rpy = "0 0 0"
    if geom_origin is not None:
        geom_xyz = geom_origin.get("xyz", "0 0 0")
        geom_rpy = geom_origin.get("rpy", "0 0 0")
    mesh_el = visual.find(".//mesh")
    if mesh_el is None:
        return ""
    fn = mesh_el.get("filename", "")
    if fn.startswith("package://"):
        fn = fn.split("/", 2)[-1]
        fn = fn.split("/", 1)[-1]
        fn = fn.split("/", 1)[-1]
    mesh_name = Path(fn).stem.replace(" ", "_")
    return (
        f'<geom type="mesh" mesh="{mesh_name}" '
        f'pos="{geom_xyz}" euler="{geom_rpy}" contype="0" conaffinity="0"/>'
    )


def _recursive_body(
    link_name: str,
    links: dict,
    joints_map: dict,
    mesh_dir_abs: str,
) -> str:
    lnk = links.get(link_name)
    if lnk is None:
        return ""

    jnt = joints_map.get(link_name)
    origin_xyz = "0 0 0"
    origin_rpy = "0 0 0"
    joint_xml = ""

    if jnt is not None:
        origin = jnt.find("origin")
        if origin is not None:
            origin_xyz = origin.get("xyz", "0 0 0")
            origin_rpy = origin.get("rpy", "0 0 0")
        jtype = jnt.get("type", "fixed")
        jname = jnt.get("name")
        if jtype in ("continuous", "revolute"):
            axis_el = jnt.find("axis")
            axis_xyz = axis_el.get("xyz", "0 0 1") if axis_el is not None else "0 0 1"
            joint_xml = f'<joint name="{jname}" type="hinge" axis="{axis_xyz}"/>'

    inertia_xml = _inertial_xml(lnk)
    geom_xml = _visual_geom_xml(lnk, mesh_dir_abs)

    # Add a site at the visual center of the fingertip cap bodies.
    # These sites have a non-zero offset from the joint origin, giving a non-degenerate
    # Jacobian for the tip joints (revolute_4_0 and revolute_9_0).
    _FINGERTIP_SITES = {
        "part_3":   ("thumb_tip_site",  "-0.125132 0.004875 -0.0466837"),
        "part_3_1": ("index_tip_site",  "-0.0674761 0.004875 -0.0250619"),
    }
    site_xml = ""
    if link_name in _FINGERTIP_SITES:
        site_name, site_pos = _FINGERTIP_SITES[link_name]
        site_xml = f'<site name="{site_name}" pos="{site_pos}" size="0.004"/>'

    children_xml = ""
    for child_name, child_jnt in joints_map.items():
        if child_jnt.find("parent").get("link") == link_name:
            children_xml += _recursive_body(child_name, links, joints_map, mesh_dir_abs)

    return (
        f'<body name="{link_name}" pos="{origin_xyz}" euler="{origin_rpy}">'
        f"  {inertia_xml}"
        f"  {joint_xml}"
        f"  {site_xml}"
        f"  {geom_xml}"
        f"  {children_xml}"
        f"</body>"
    )
