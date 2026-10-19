"""Convert the WF-TRON1B MJCF to the single-file USD the task config loads.

The Isaac Sim 5.1 MJCF importer nests every link under a ``/robot/base_Link`` container,
adds an empty second articulation root (``/robot/worldBody``), instances the collision
prims and leaves them renderable. After converting, this script

1. flattens the layered output into one USD with links at ``/robot/<link>`` -- the layout
   the sensor prim paths assume -- and a single articulation root on ``base_Link``,
2. hides the collision primitives from rendering: mjlab's depth camera only sees visual
   geoms, so a camera here must not see the collision cylinders either,
3. moves each knee's wheel-motor housing collider onto its own 1 g link, fixed to the knee:
   mjlab tells contacts apart per geom and ignores the housing's, Isaac Lab senses contact per
   body, so the housing must not share the knee's body,
4. spawns the articulation, checks it has exactly the MJCF's joints and bodies (plus the two
   housing links), and writes the order PhysX resolves to ``joint_order.txt`` so a
   reordering shows up in review.

Usage:
    python scripts/tools/convert_wf_tron1b_mjcf.py --headless
"""

import argparse
import os
import tempfile
import traceback
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../exts/bipedal_locomotion/bipedal_locomotion/assets")
MJCF_PATH = os.path.abspath(os.path.join(_ASSETS, "mjcf/WF_TRON1B/xml/robot.xml"))
USD_DIR = os.path.abspath(os.path.join(_ASSETS, "usd/WF_TRON1B"))

parser = argparse.ArgumentParser(description="Convert the WF-TRON1B MJCF to USD.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.importer.mjcf")  # not part of the headless experience

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# Housing link -> (knee it is fixed to, the knee's parent and child links it must not collide with).
WHEEL_MOTOR_LINKS = {
    f"wheel_motor_{side}_Link": (f"knee_{side}_Link", f"hip_{side}_Link", f"wheel_{side}_Link") for side in "LR"
}
WHEEL_MOTOR_MASS = 1.0e-3  # kg; the knee's MJCF inertial already accounts for the housing


def import_mjcf(out_dir: str) -> str:
    cfg = MjcfConverterCfg(
        asset_path=MJCF_PATH,
        usd_dir=out_dir,
        usd_file_name="robot.usd",
        force_usd_conversion=True,
        fix_base=False,
        import_inertia_tensor=True,
        self_collision=True,  # MuJoCo collides non-adjacent links
        make_instanceable=False,
    )
    return MjcfConverter(cfg).usd_path


def flatten(raw_usd: str, out_usd: str) -> None:
    """Write ``raw_usd`` as one layer with the links moved up to ``/robot/<link>``."""
    stage = Usd.Stage.Open(raw_usd)
    for prim in stage.Traverse():
        if prim.IsInstance():
            prim.SetInstanceable(False)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
    layer = stage.Flatten()
    layer.documentation = "Generated from robot.xml by scripts/tools/convert_wf_tron1b_mjcf.py"  # not the temp path

    root = stage.GetDefaultPrim().GetPath()
    container = root.AppendChild("base_Link")  # holds every link, including a base_Link

    # Sdf namespace edits do not rewrite relationship targets (joint bodies, filtered pairs).
    def retarget(path: Sdf.Path) -> None:
        rel = layer.GetObjectAtPath(path)
        if isinstance(rel, Sdf.RelationshipSpec):
            targets = list(rel.targetPathList.GetAddedOrExplicitItems())
            rel.targetPathList.ClearEdits()
            rel.targetPathList.explicitItems = [
                t.ReplacePrefix(container, root) if t.HasPrefix(container) and t != container else t for t in targets
            ]

    layer.Traverse(root, retarget)
    links = root.AppendChild("_links")
    _apply(layer, [Sdf.NamespaceEdit.Rename(container, links.name)])
    _apply(layer, [Sdf.NamespaceEdit.Reparent(link.path, root, -1) for link in layer.GetPrimAtPath(links).nameChildren])
    for name in (links.name, "worldBody"):
        del layer.GetPrimAtPath(root).nameChildren[name]
    split_wheel_motors(layer, root)
    layer.Export(out_usd)


def split_wheel_motors(layer: Sdf.Layer, root: Sdf.Path) -> None:
    """Move each knee's wheel-motor housing collider onto its own link, fixed to the knee at the knee's pose."""
    stage = Usd.Stage.Open(layer)
    for name, (knee, parent, child) in WHEEL_MOTOR_LINKS.items():
        knee_path, link_path = root.AppendChild(knee), root.AppendChild(name)
        geom = name.removesuffix("_Link") + "_collision"
        link = UsdGeom.Xform.Define(stage, link_path).GetPrim()
        for attr in ("xformOp:translate", "xformOp:orient", "xformOp:scale", "xformOpOrder"):
            Sdf.CopySpec(layer, knee_path.AppendProperty(attr), layer, link_path.AppendProperty(attr))
        collisions = link_path.AppendChild("collisions")
        UsdGeom.Xform.Define(stage, collisions).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        _apply(layer, [Sdf.NamespaceEdit.Reparent(knee_path.AppendPath(f"collisions/{geom}"), collisions, -1)])

        # A light solid cylinder (axis along the link's y) centred on the housing.
        frame = stage.GetPrimAtPath(collisions.AppendChild(geom))
        shape = UsdGeom.Cylinder(frame.GetChild(geom))
        r, h, m = shape.GetRadiusAttr().Get(), shape.GetHeightAttr().Get(), WHEEL_MOTOR_MASS
        UsdPhysics.RigidBodyAPI.Apply(link)
        mass = UsdPhysics.MassAPI.Apply(link)
        mass.CreateMassAttr(m)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(frame.GetAttribute("xformOp:translate").Get()))
        transverse = m * (3 * r * r + h * h) / 12
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(transverse, m * r * r / 2, transverse))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
        # MuJoCo never collides a geom with its body's parent or child; those were the knee's, keep them filtered.
        pairs = UsdPhysics.FilteredPairsAPI.Apply(link).CreateFilteredPairsRel()
        pairs.SetTargets([root.AppendChild(parent), root.AppendChild(child)])

        joint = UsdPhysics.FixedJoint.Define(stage, root.AppendPath(f"joints/{name.removesuffix('_Link')}_Joint"))
        joint.CreateBody0Rel().SetTargets([knee_path])
        joint.CreateBody1Rel().SetTargets([link_path])


def _apply(layer: Sdf.Layer, edits: list) -> None:
    batch = Sdf.BatchNamespaceEdit()
    for edit in edits:
        batch.Add(edit)
    if not layer.Apply(batch):
        raise RuntimeError(f"Namespace edit failed: {edits}")


def audit(usd_path: str) -> None:
    """Spawn the articulation, compare names with the MJCF and record the resolved order."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    robot = Articulation(
        ArticulationCfg(prim_path="/World/Robot", spawn=sim_utils.UsdFileCfg(usd_path=usd_path), actuators={})
    )
    sim.reset()

    mjcf = ET.parse(MJCF_PATH).getroot()
    mjcf_joints = [j.get("name") for j in mjcf.iter("joint") if j.get("name")]  # the free joint is unnamed
    mjcf_bodies = [b.get("name") for b in mjcf.iter("body")] + list(WHEEL_MOTOR_LINKS)
    if sorted(robot.joint_names) != sorted(mjcf_joints) or sorted(robot.body_names) != sorted(mjcf_bodies):
        raise AssertionError(f"USD {robot.joint_names} / {robot.body_names} != MJCF {mjcf_joints} / {mjcf_bodies}")

    limits = robot.data.joint_pos_limits[0].cpu()
    lines = [
        "# Generated by scripts/tools/convert_wf_tron1b_mjcf.py -- do not edit by hand.",
        "# Joint and body order PhysX resolves. The task config never relies on it: it passes",
        "# explicit ordered name lists with preserve_order=True. The wheel_motor_* links are fixed",
        "# to the knees and only carry the wheel-motor housing colliders.",
        "",
        "[joints] (index, name, lower limit, upper limit [rad])",
        *(f"{i}\t{n}\t{limits[i, 0]:.6f}\t{limits[i, 1]:.6f}" for i, n in enumerate(robot.joint_names)),
        "",
        "[bodies] (index, name)",
        *(f"{i}\t{n}" for i, n in enumerate(robot.body_names)),
        "",
    ]
    with open(os.path.join(USD_DIR, "joint_order.txt"), "w") as handle:
        handle.write("\n".join(lines))
    print(f"[INFO] joints: {robot.joint_names}\n[INFO] bodies: {robot.body_names}")
    # A live simulation context makes SimulationApp.close() hang.
    sim.clear_all_callbacks()
    sim.clear_instance()


def main() -> None:
    os.makedirs(USD_DIR, exist_ok=True)
    usd_path = os.path.join(USD_DIR, "robot.usd")
    with tempfile.TemporaryDirectory() as tmp:
        flatten(import_mjcf(tmp), usd_path)
    audit(usd_path)
    print(f"[INFO] Wrote {usd_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - printed, then exit code 1
        # SimulationApp.close() terminates the process, which would swallow the traceback.
        traceback.print_exc()
        os._exit(1)
    simulation_app.close()
