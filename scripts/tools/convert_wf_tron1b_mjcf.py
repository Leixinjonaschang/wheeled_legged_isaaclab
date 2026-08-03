"""Convert the WF-TRON1B MJCF description to USD and audit the result.

The task configuration in ``bipedal_locomotion.tasks.locomotion.cfg.WF_TRON1B`` addresses
joints and bodies by *name* (``abad_L_Joint``, ``wheel_R_Link``, ...) and depends on the
articulation's joint *order* for the action vector and the exported deployment metadata.
Neither survives a conversion automatically, so this script does three things:

1. converts ``assets/mjcf/WF_TRON1B/xml/robot.xml`` with :class:`MjcfConverter`,
2. asserts every expected joint and body name is present in the generated USD,
3. spawns the articulation once and writes the resolved joint/body order to
   ``joint_order.txt`` next to the USD, so the ordering is reviewable in a diff.

Step 3 is the guard against the MuJoCo depth-first / PhysX breadth-first ordering mismatch:
MJCF declares ``abad_L, hip_L, knee_L, wheel_L, abad_R, ...`` but PhysX is free to choose
another order, which would silently permute actions and observations.

Usage:
    ${ISAACLAB}/isaaclab.sh -p scripts/tools/convert_wf_tron1b_mjcf.py --headless
"""

import argparse
import dataclasses
import os
import sys

from isaaclab.app import AppLauncher

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../exts/bipedal_locomotion/bipedal_locomotion"))
_MJCF_PATH = os.path.join(_EXT_ROOT, "assets/mjcf/WF_TRON1B/xml/robot.xml")
_USD_DIR = os.path.join(_EXT_ROOT, "assets/usd/WF_TRON1B")
_USD_NAME = "robot.usd"

# Expected names, taken from the MJCF. Kept here rather than imported from the task config so
# that the audit stays independent of the configuration it is meant to protect.
EXPECTED_JOINTS = (
    "abad_L_Joint",
    "hip_L_Joint",
    "knee_L_Joint",
    "wheel_L_Joint",
    "abad_R_Joint",
    "hip_R_Joint",
    "knee_R_Joint",
    "wheel_R_Joint",
)
EXPECTED_BODIES = (
    "base_Link",
    "abad_L_Link",
    "hip_L_Link",
    "knee_L_Link",
    "wheel_L_Link",
    "abad_R_Link",
    "hip_R_Link",
    "knee_R_Link",
    "wheel_R_Link",
)
# MJCF joint declaration order (depth-first). The USD/PhysX order is compared against this.
MJCF_JOINT_ORDER = EXPECTED_JOINTS

parser = argparse.ArgumentParser(description="Convert the WF-TRON1B MJCF to USD and audit it.")
parser.add_argument("--mjcf", type=str, default=_MJCF_PATH, help="Source MJCF file.")
parser.add_argument("--usd-dir", type=str, default=_USD_DIR, help="Destination USD directory.")
parser.add_argument("--usd-name", type=str, default=_USD_NAME, help="Destination USD file name.")
parser.add_argument("--no-audit", action="store_true", default=False, help="Skip spawning the articulation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg


def build_converter_cfg(mjcf_path: str, usd_dir: str, usd_name: str) -> MjcfConverterCfg:
    """Build the converter config, tolerating field renames across Isaac Lab versions.

    ``MjcfConverterCfg`` has gained and lost fields between releases. Rather than passing a
    fixed keyword set and either crashing or -- worse -- silently accepting a default we did
    not intend, introspect the dataclass and report exactly which options were unavailable.
    """
    desired = {
        "asset_path": mjcf_path,
        "usd_dir": usd_dir,
        "usd_file_name": usd_name,
        "force_usd_conversion": True,
        # Floating base: this is a legged robot, not a fixed manipulator. The field has no
        # default in MjcfConverterCfg, so it must be supplied.
        "fix_base": False,
        # The MJCF carries explicit <inertial> tags; without this the converter would
        # substitute identity tensors and the robot's rotational dynamics would be wrong.
        "import_inertia_tensor": True,
        # Keeps the `imu` site, which is harmless and useful when sanity-checking frames.
        "import_sites": True,
        # Matches `enabled_self_collisions=True` on the articulation config; MuJoCo has
        # self-collision active for these bodies.
        "self_collision": True,
        # Instanceable assets share a single prototype prim, which breaks the
        # per-environment mass / COM / material randomisation the task relies on.
        "make_instanceable": False,
    }
    available = {f.name for f in dataclasses.fields(MjcfConverterCfg)}
    accepted = {k: v for k, v in desired.items() if k in available}
    dropped = sorted(set(desired) - available)
    if dropped:
        print(f"[WARN] MjcfConverterCfg has no field(s) {dropped}; conversion will use its defaults.")
        print(f"[WARN] Available fields: {sorted(available)}")
    return MjcfConverterCfg(**accepted)


def audit_usd_names(usd_path: str) -> None:
    """Assert the generated USD carries every joint and body name the task config addresses."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open generated USD: {usd_path}")

    joint_names, body_names, collider_names = set(), set(), set()
    for prim in stage.Traverse():
        name = prim.GetName()
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint_names.add(name)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_names.add(name)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collider_names.add(name)

    missing_joints = sorted(set(EXPECTED_JOINTS) - joint_names)
    missing_bodies = sorted(set(EXPECTED_BODIES) - body_names)
    if missing_joints:
        raise AssertionError(f"Generated USD is missing revolute joints: {missing_joints}\nfound: {sorted(joint_names)}")
    if missing_bodies:
        raise AssertionError(f"Generated USD is missing rigid bodies: {missing_bodies}\nfound: {sorted(body_names)}")

    print(f"[INFO] Joints found ({len(joint_names)}): {sorted(joint_names)}")
    print(f"[INFO] Bodies found ({len(body_names)}): {sorted(body_names)}")
    print(f"[INFO] Colliders found ({len(collider_names)}): {sorted(collider_names)}")
    # The MJCF marks visual meshes contype=0/conaffinity=0 via a default class. If the importer
    # ignored that, every visual mesh becomes a collider and `illegal_ground_contact` fires on
    # flat ground. Warn loudly; the smoke rollout's episode length is the other detector.
    suspicious = sorted(n for n in collider_names if "collision" not in n.lower())
    if suspicious:
        print(f"[WARN] Collider prims whose names do not contain 'collision': {suspicious}")
        print("[WARN] If these are visual meshes, contact-based terminations will misfire.")


def audit_articulation_order(usd_path: str, usd_dir: str) -> None:
    """Spawn the articulation once and record the joint/body order PhysX actually resolves."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.966)),
        actuators={},
    )
    robot = Articulation(robot_cfg)
    sim.reset()

    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)

    print(f"[INFO] Resolved joint order ({len(joint_names)}): {joint_names}")
    print(f"[INFO] Resolved body order  ({len(body_names)}): {body_names}")
    if joint_names != list(MJCF_JOINT_ORDER):
        print("[WARN] Resolved joint order DIFFERS from the MJCF declaration order.")
        print(f"[WARN]   MJCF: {list(MJCF_JOINT_ORDER)}")
        print(f"[WARN]   USD : {joint_names}")
        print("[WARN] This is expected (PhysX may reorder). The task config must therefore pass")
        print("[WARN] explicit ordered joint name lists with preserve_order=True everywhere.")

    limits = robot.data.joint_pos_limits[0].cpu() if hasattr(robot.data, "joint_pos_limits") else None
    report = [
        "# Generated by scripts/tools/convert_wf_tron1b_mjcf.py -- do not edit by hand.",
        "# Records the joint order PhysX resolves for the WF-TRON1B articulation.",
        "# The task config must not rely on this order implicitly; it passes explicit",
        "# ordered name lists with preserve_order=True. This file exists so that a change",
        "# in ordering is visible in review.",
        "",
        "[mjcf_declaration_order]",
        *(f"{i}\t{n}" for i, n in enumerate(MJCF_JOINT_ORDER)),
        "",
        "[usd_resolved_joint_order]",
        *(f"{i}\t{n}" for i, n in enumerate(joint_names)),
        "",
        "[usd_resolved_body_order]",
        *(f"{i}\t{n}" for i, n in enumerate(body_names)),
    ]
    if limits is not None:
        report += ["", "[joint_position_limits] (resolved order, radians)"]
        report += [f"{n}\t{limits[i, 0]:.6f}\t{limits[i, 1]:.6f}" for i, n in enumerate(joint_names)]
    report.append("")

    out_path = os.path.join(usd_dir, "joint_order.txt")
    with open(out_path, "w") as handle:
        handle.write("\n".join(report))
    print(f"[INFO] Wrote {out_path}")

    del robot
    torch.cuda.empty_cache()


def main() -> None:
    if not os.path.isfile(args_cli.mjcf):
        raise FileNotFoundError(f"MJCF not found: {args_cli.mjcf}")

    cfg = build_converter_cfg(args_cli.mjcf, args_cli.usd_dir, args_cli.usd_name)
    print(f"[INFO] Converting {args_cli.mjcf}")
    print(f"[INFO] Converter config: {cfg.to_dict() if hasattr(cfg, 'to_dict') else cfg}")

    converter = MjcfConverter(cfg)
    usd_path = converter.usd_path
    print(f"[INFO] Generated USD: {usd_path}")

    audit_usd_names(usd_path)
    if not args_cli.no_audit:
        audit_articulation_order(usd_path, args_cli.usd_dir)
    print("[INFO] Conversion and audit complete.")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    sys.exit(0)
