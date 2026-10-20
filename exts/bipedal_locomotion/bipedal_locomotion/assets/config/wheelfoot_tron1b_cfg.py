"""Articulation config for the LimX WF-TRON1B wheeled biped.

Every number here is taken from ``assets/mjcf/WF_TRON1B/xml/robot.xml`` so that the
Isaac Lab model matches the MuJoCo one the policy was originally shaped against. The USD
is generated from that same MJCF by ``scripts/tools/convert_wf_tron1b_mjcf.py``.

MuJoCo terms map one-to-one onto the PhysX joint model (Isaac Sim >= 5.0):
actuator ``kp`` / ``kv`` -> drive stiffness / damping, joint ``damping`` (passive) ->
``viscous_friction``, joint ``frictionloss`` -> static and dynamic friction effort,
``forcerange`` -> ``effort_limit_sim``. MuJoCo has no joint velocity limit, so none is set.

Besides the MJCF's nine bodies, the USD has two 1 g ``wheel_motor_[LR]_Link`` bodies fixed to
the knees. They carry the wheel-motor housing colliders, which mjlab's geom-level contact groups
leave out; per-body contact sensing can only do the same if the housings have their own bodies.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_CURRENT_DIR = os.path.dirname(__file__)
USD_PATH = os.path.join(_CURRENT_DIR, "../usd/WF_TRON1B/robot.usd")
MESH_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "../mjcf/WF_TRON1B/meshes"))
"""Visual meshes, ``<link>.STL`` in each link's frame (the MJCF visual geoms have no offset)."""

WHEELFOOT_TRON1B_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            # In deg/s. The usual 1000 deg/s (17.5 rad/s) would cap the wheels below the
            # 20 rad/s the action range commands; MuJoCo has no such cap.
            max_angular_velocity=math.degrees(100.0),
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            # MJCF declares iterations=10; matching it keeps contact-rich stepping from
            # resolving differently between the two simulators.
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # MJCF spawns the base at 0.92 m; 0.966 m is the mjlab task's spawn height
        # (0.8 nominal stance + 0.166 wheel offset), which starts the robot standing
        # rather than dropping it.
        pos=(0.0, 0.0, 0.966),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    # The MJCF applies no soft-limit shrink; a factor below 1 would make `leg_joint_pos_limits`
    # penalise inside the true limit.
    soft_joint_pos_limit_factor=1.0,
    actuators={
        # MJCF: <position kp="40.0" kv="1.8" forcerange="-80 80">, joint damping 0.01.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["abad_[LR]_Joint", "hip_[LR]_Joint", "knee_[LR]_Joint"],
            stiffness=40.0,
            damping=1.8,
            armature=0.01,
            effort_limit_sim=80.0,
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=0.01,
        ),
        # MJCF: <velocity kv="0.5" forcerange="-40 40">, joint damping 0.01, frictionloss
        # 0.01. Zero stiffness is what makes this a velocity servo rather than a position one.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_[LR]_Joint"],
            stiffness=0.0,
            damping=0.5,
            armature=0.01,
            effort_limit_sim=40.0,
            friction=0.01,
            dynamic_friction=0.01,
            viscous_friction=0.01,
        ),
    },
)
"""WF-TRON1B: position-actuated legs, velocity-actuated wheels."""
