"""Base environment configuration for the WF-TRON1B velocity task.

This class is the **rough-terrain superset**. The flat variant subclasses it and sets the
terrain-dependent sensors, rewards, observations and terminations to ``None``.

That direction matters: Isaac Lab's observation manager iterates a group's fields in
declaration order and skips the ``None`` ones, so nulling from a superset preserves the
relative order of everything that remains. Declaring flat as the base and *adding* rough
terms would append them in a different order than the mjlab reference, silently permuting
the critic's input vector.

Group widths (asserted by ``tests/`` and by the smoke script):

===================  ======  ======
group                flat    rough
===================  ======  ======
``actor``            31      31
``actor_history``    5 x 31  5 x 31
``critic``           42      166
``dynamics_context`` 13      13
===================  ======  ======
"""

import math

from bipedal_locomotion.assets.config.wheelfoot_tron1b_cfg import WHEELFOOT_TRON1B_CFG
from bipedal_locomotion.tasks.locomotion import mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from . import constants as C
from .terrain_cfg import WF_TRON1B_TERRAINS_CFG

# Frequently reused entity selections. `preserve_order=True` everywhere: see constants.py.
_LEG_JOINTS = SceneEntityCfg("robot", joint_names=list(C.LEG_JOINT_NAMES), preserve_order=True)
_WHEEL_JOINTS = SceneEntityCfg("robot", joint_names=list(C.WHEEL_JOINT_NAMES), preserve_order=True)
_ALL_JOINTS = SceneEntityCfg("robot", joint_names=list(C.ALL_JOINT_NAMES), preserve_order=True)
_WHEEL_BODIES = SceneEntityCfg("robot", body_names=list(C.WHEEL_BODY_NAMES), preserve_order=True)
_BASE_BODY = SceneEntityCfg("robot", body_names=[C.BASE_BODY])
_WHEEL_CONTACTS = SceneEntityCfg(C.WHEEL_CONTACT_SENSOR, body_names=list(C.WHEEL_BODY_NAMES), preserve_order=True)
_ILLEGAL_CONTACTS = SceneEntityCfg(C.ILLEGAL_CONTACT_SENSOR, body_names=list(C.NON_WHEEL_BODY_NAMES))
_TERRAIN_SCAN = SceneEntityCfg(C.TERRAIN_SCAN_SENSOR)

_ROUGHNESS_KWARGS = {
    "roughness_sensor_names": C.WHEEL_HEIGHT_SENSORS,
    "wheel_radius": C.WHEEL_RADIUS,
    "gate_min": C.ROUGHNESS_GATE_MIN,
    "gate_max": C.ROUGHNESS_GATE_MAX,
    "grid_shape": C.WHEEL_HEIGHT_GRID_SHAPE,
    "roughness_gate_threshold": C.ROUGHNESS_GATE_THRESHOLD_INITIAL,
    "roughness_gate_threshold_final": C.ROUGHNESS_GATE_THRESHOLD_FINAL,
    "roughness_gate_threshold_ramp_steps": C.ROUGHNESS_GATE_THRESHOLD_RAMP_STEPS,
}


##
# Scene
##


@configclass
class WF_TRON1B_SceneCfg(InteractiveSceneCfg):
    """Terrain, robot, height scans and contact sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=WF_TRON1B_TERRAINS_CFG,
        max_init_terrain_level=8,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot = WHEELFOOT_TRON1B_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Base-mounted scan: the critic's terrain map and, with the per-wheel scans, the
    # input to the roughness gate. Yaw-aligned so the map turns with the robot but does
    # not tilt with it -- a pitching robot should not see the ground swing away.
    terrain_scan = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + C.BASE_BODY,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=C.TERRAIN_SCAN_RESOLUTION, size=C.TERRAIN_SCAN_SIZE),
        max_distance=C.TERRAIN_SCAN_MAX_DISTANCE,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    # Per-wheel scans. Isaac Lab's RayCaster binds one prim per environment (a regex in
    # the leaf path is rejected), so the two wheels need two sensors; sensor_utils stacks
    # them back into the [B, F, N] layout the roughness and clearance kernels expect.
    #
    # `ray_alignment="world"`, not "base": these measure how far the ground is *below*
    # the wheel, so the rays must point straight down in world coordinates. Base-aligned
    # rays would tilt with the robot and report a longer distance the more it leans,
    # which the clearance reward would read as the wheel having been lifted.
    wheel_height_scan_L = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wheel_L_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=C.WHEEL_HEIGHT_SCAN_RESOLUTION, size=C.WHEEL_HEIGHT_SCAN_SIZE
        ),
        max_distance=C.WHEEL_HEIGHT_MAX_DISTANCE,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    wheel_height_scan_R = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wheel_R_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=C.WHEEL_HEIGHT_SCAN_RESOLUTION, size=C.WHEEL_HEIGHT_SCAN_SIZE
        ),
        max_distance=C.WHEEL_HEIGHT_MAX_DISTANCE,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    # Two contact sensors rather than one catch-all: history_length is a sensor-wide
    # setting, and only the illegal-contact term needs sub-step history.
    wheel_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wheel_.*_Link",
        history_length=0,
        track_air_time=True,
        force_threshold=C.CONTACT_FORCE_THRESHOLD,
        update_period=0.0,
    )
    illegal_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(base|abad_[LR]|hip_[LR]|knee_[LR])_Link",
        history_length=4,
        track_air_time=False,
        update_period=0.0,
    )

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP
##


@configclass
class CommandsCfg:
    twist = mdp.WheeledUniformVelocityCommandCfg(
        asset_name=C.ROBOT_ENTITY,
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        rel_forward_envs=0.2,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=False,
        ranges=mdp.WheeledUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-math.pi / 2, math.pi / 2),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Exactly two terms, in this order: the 8-vector is ``[leg_pos x6, wheel_vel x2]``.

    Splitting into four L/R terms (as the TRON2A configs do) would produce a different
    layout and silently invalidate the exported deployment metadata.
    """

    leg_pos = mdp.DelayedJointPositionActionCfg(
        asset_name=C.ROBOT_ENTITY,
        joint_names=list(C.LEG_JOINT_NAMES),
        preserve_order=True,
        scale=C.LEG_ACTION_SCALE,
        use_default_offset=True,
        delay_range_s=C.ACTION_DELAY_RANGE_S,
        resampling_time_s=C.ACTION_DELAY_RESAMPLING_TIME_S,
        delay_group=C.ACTION_DELAY_GROUP,
    )
    wheel_vel = mdp.DelayedJointVelocityActionCfg(
        asset_name=C.ROBOT_ENTITY,
        joint_names=list(C.WHEEL_JOINT_NAMES),
        preserve_order=True,
        scale=C.WHEEL_ACTION_SCALE,
        use_default_offset=False,
        delay_range_s=C.ACTION_DELAY_RANGE_S,
        resampling_time_s=C.ACTION_DELAY_RESAMPLING_TIME_S,
        delay_group=C.ACTION_DELAY_GROUP,
    )


@configclass
class ObservationsCfg:
    @configclass
    class ProprioCfg(ObsGroup):
        """Everything the deployed robot can measure. Term order is the ONNX contract."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-C.NOISE_BASE_ANG_VEL, n_max=C.NOISE_BASE_ANG_VEL))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-C.NOISE_PROJECTED_GRAVITY, n_max=C.NOISE_PROJECTED_GRAVITY),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _LEG_JOINTS},
            noise=Unoise(n_min=-C.NOISE_JOINT_POS, n_max=C.NOISE_JOINT_POS),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _LEG_JOINTS},
            noise=Unoise(n_min=-C.NOISE_JOINT_VEL, n_max=C.NOISE_JOINT_VEL),
            scale=C.SCALE_JOINT_VEL,
        )
        # Wheel *position* is deliberately absent: it is unbounded and carries no
        # information the policy can act on. Only wheel velocity is observed.
        wheel_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _WHEEL_JOINTS},
            noise=Unoise(n_min=-C.NOISE_WHEEL_VEL, n_max=C.NOISE_WHEEL_VEL),
            scale=C.SCALE_WHEEL_VEL,
        )
        actions = ObsTerm(func=mdp.last_action)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": C.COMMAND_NAME})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ProprioHistoryCfg(ProprioCfg):
        """The same terms, corrupted and stacked over time.

        Unflattened so the shape stays ``[B, T, F]``: the representation models consume
        the time axis directly, and the exported metadata declares the frame order as
        oldest-to-newest.
        """

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = True
            self.history_length = C.HISTORY_LENGTH
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations. Never corrupted -- the critic is not deployed."""

        base_lin_vel = ObsTerm(func=mdp.base_link_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _LEG_JOINTS})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _LEG_JOINTS}, scale=C.SCALE_JOINT_VEL)
        wheel_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _WHEEL_JOINTS}, scale=C.SCALE_WHEEL_VEL)
        actions = ObsTerm(func=mdp.last_action)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": C.COMMAND_NAME})
        wheel_contact = ObsTerm(func=mdp.wheel_contact, params={"sensor_cfg": _WHEEL_CONTACTS})
        wheel_contact_forces = ObsTerm(func=mdp.wheel_contact_forces, params={"sensor_cfg": _WHEEL_CONTACTS})
        # Terrain-dependent; nulled by the flat config.
        height_scan = ObsTerm(
            func=mdp.terrain_height_scan,
            params={"sensor_cfg": _TERRAIN_SCAN, "max_distance": C.TERRAIN_SCAN_MAX_DISTANCE},
            scale=C.SCALE_HEIGHT_SCAN,
        )
        wheel_height = ObsTerm(
            func=mdp.wheel_height,
            params={"sensor_names": C.WHEEL_HEIGHT_SENSORS, "max_distance": C.WHEEL_HEIGHT_MAX_DISTANCE},
        )
        roughness_indicator = ObsTerm(
            func=mdp.terrain_roughness_indicator,
            params={
                "sensor_names": C.WHEEL_HEIGHT_SENSORS,
                "wheel_radius": C.WHEEL_RADIUS,
                "gate_min": C.ROUGHNESS_GATE_MIN,
                "gate_max": C.ROUGHNESS_GATE_MAX,
                "grid_shape": C.WHEEL_HEIGHT_GRID_SHAPE,
                "max_distance": C.WHEEL_HEIGHT_MAX_DISTANCE,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DynamicsContextCfg(ObsGroup):
        """What was randomised this episode. Consumed by the teacher-side encoder."""

        domain_randomization_delta_quantity = ObsTerm(func=mdp.domain_randomization_delta_quantity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    actor: ProprioCfg = ProprioCfg()
    actor_history: ProprioHistoryCfg = ProprioHistoryCfg()
    critic: CriticCfg = CriticCfg()
    dynamics_context: DynamicsContextCfg = DynamicsContextCfg()


@configclass
class EventsCfg:
    # -- startup
    wheel_friction = EventTerm(
        func=mdp.randomize_wheel_friction,
        mode="startup",
        params={"asset_cfg": _WHEEL_BODIES, "friction_range": C.WHEEL_FRICTION_RANGE},
    )
    encoder_bias = EventTerm(
        func=mdp.randomize_encoder_bias,
        mode="startup",
        params={"asset_cfg": _ALL_JOINTS, "bias_range": C.ENCODER_BIAS_RANGE},
    )
    base_com = EventTerm(
        func=mdp.randomize_base_com,
        mode="startup",
        params={"asset_cfg": _BASE_BODY, "com_range": C.BASE_COM_RANGE},
    )

    # -- reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.01, 0.05), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.3, 0.5), "velocity_range": (-0.2, 0.2), "asset_cfg": _LEG_JOINTS},
    )
    # Wheels reset to zero position: a wheel angle is meaningless, and randomising it
    # would only scramble the (unobserved) joint state.
    reset_wheel_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (-0.1, 0.1), "asset_cfg": _WHEEL_JOINTS},
    )

    # -- interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(15.0, 15.5),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-0.5, 0.5),
            }
        },
    )


@configclass
class RewardsCfg:
    # -- task
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    track_linear_velocity = RewTerm(
        func=mdp.track_linear_velocity,
        weight=4.0,
        params={"command_name": C.COMMAND_NAME, "std": math.sqrt(0.25)},
    )
    track_angular_velocity = RewTerm(
        func=mdp.track_angular_velocity,
        weight=1.0,
        params={"command_name": C.COMMAND_NAME, "std": math.sqrt(0.25)},
    )
    track_heading = RewTerm(
        func=mdp.track_heading,
        weight=0.5,
        params={"command_name": C.COMMAND_NAME, "std": math.sqrt(0.20), "command_norm_threshold": 0.05},
    )
    # Disabled, matching the mjlab reference. Both RewardManagers skip zero-weight terms
    # entirely, so this term is not merely scaled to nothing -- it is never called, and
    # its internal per-step state never advances. Raise the weight to enable it.
    heading_progress = RewTerm(
        func=mdp.HeadingProgress,
        weight=0.0,
        params={"command_name": C.COMMAND_NAME, "max_progress": 0.05},
    )
    stand_still = RewTerm(
        func=mdp.stand_still_world,
        weight=-2.0,
        params={"command_name": C.COMMAND_NAME},
    )

    # -- base regulation
    upright = RewTerm(
        func=mdp.Upright,
        weight=1.0,
        params={"std": math.sqrt(0.2), "asset_cfg": _BASE_BODY, "terrain_sensor_cfg": _TERRAIN_SCAN},
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    base_ang_vel_xy = RewTerm(
        func=mdp.base_ang_vel_xy_l2,
        weight=-0.15,
        params={"roll_weight": 1.0, "pitch_weight": 2.0},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={
            "target_height": C.BASE_HEIGHT_TARGET,
            "deadband": C.BASE_HEIGHT_DEADBAND,
            "sensor_cfg": _TERRAIN_SCAN,
        },
    )

    # -- posture
    pose = RewTerm(
        func=mdp.VariablePosture,
        weight=0.5,
        params={
            "asset_cfg": _LEG_JOINTS,
            "command_name": C.COMMAND_NAME,
            "target_joint_pos": C.POSE_TARGET_JOINT_POS,
            "std_standing": C.POSE_STD_STANDING,
            "std_walking": C.POSE_STD_WALKING,
            "std_running": C.POSE_STD_RUNNING,
            "walking_threshold": C.POSE_WALKING_THRESHOLD,
            "running_threshold": C.POSE_RUNNING_THRESHOLD,
        },
    )
    wheel_distance = RewTerm(
        func=mdp.wheel_distance,
        weight=-5.0,
        params={
            "asset_cfg": _WHEEL_BODIES,
            "min_distance": C.WHEEL_DISTANCE_RANGE[0],
            "max_distance": C.WHEEL_DISTANCE_RANGE[1],
        },
    )

    # -- joint regulation
    leg_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": _LEG_JOINTS})
    leg_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.02, params={"asset_cfg": _LEG_JOINTS})
    leg_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": _LEG_JOINTS})
    wheel_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.002, params={"asset_cfg": _WHEEL_JOINTS})
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7, params={"asset_cfg": _ALL_JOINTS})
    # `joint_powers_l1` (a local term predating this task) sums over *every* joint and
    # ignores asset_cfg.joint_ids. That is numerically correct here only because
    # _ALL_JOINTS is all eight joints; adding a joint to the robot would change what this
    # penalises without changing this line.
    joint_power = RewTerm(func=mdp.joint_powers_l1, weight=-5.0e-5, params={"asset_cfg": _ALL_JOINTS})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # -- contact safety
    illegal_ground_contact = RewTerm(
        func=mdp.contact_force_cost,
        weight=-1.0,
        params={"sensor_cfg": _ILLEGAL_CONTACTS, "force_threshold": C.ILLEGAL_CONTACT_FORCE_THRESHOLD},
    )
    soft_landing = RewTerm(
        func=mdp.soft_landing,
        weight=-1.0e-5,
        params={"sensor_cfg": _WHEEL_CONTACTS, "command_name": C.COMMAND_NAME, "command_threshold": 0.05},
    )
    wheel_air_time_balance = RewTerm(
        func=mdp.WheelAirTimeBalance,
        weight=-4.0,
        params={"sensor_cfg": _WHEEL_CONTACTS, "min_total_air_time": 1.0, "balance_tolerance": 0.2},
    )

    # -- rough-terrain gated (nulled by the flat config)
    rough_wheel_usage = RewTerm(
        func=mdp.rough_wheel_usage,
        weight=-3.0e-2,
        params={**_ROUGHNESS_KWARGS, "asset_cfg": _WHEEL_JOINTS},
    )
    rough_wheel_foot_clearance = RewTerm(
        func=mdp.rough_wheel_foot_clearance,
        weight=2.0,
        params={
            **_ROUGHNESS_KWARGS,
            "clearance_sensor_names": C.WHEEL_HEIGHT_SENSORS,
            "clearance_grid_shape": C.WHEEL_HEIGHT_GRID_SHAPE,
            "contact_sensor_cfg": _WHEEL_CONTACTS,
            "command_name": C.COMMAND_NAME,
            "base_target_height": 0.03,
            "range_scale": 0.5,
            "max_target_height": 0.18,
            "target_std": 0.04,
            "command_threshold": 0.05,
        },
    )
    rough_contact_pattern = RewTerm(
        func=mdp.rough_contact_pattern,
        weight=0.3,
        params={
            **_ROUGHNESS_KWARGS,
            "contact_sensor_cfg": _WHEEL_CONTACTS,
            "command_name": C.COMMAND_NAME,
            "command_threshold": 0.05,
        },
    )
    non_rough_wheel_lateral_symmetry = RewTerm(
        func=mdp.non_rough_wheel_lateral_symmetry,
        weight=0.5,
        params={**_ROUGHNESS_KWARGS, "asset_cfg": _WHEEL_BODIES, "std": math.sqrt(0.5)},
    )
    non_rough_wheel_x_alignment = RewTerm(
        func=mdp.non_rough_wheel_x_alignment,
        weight=-50.0,
        params={**_ROUGHNESS_KWARGS, "asset_cfg": _WHEEL_BODIES},
    )
    standing_forward_wheel_air_time = RewTerm(
        func=mdp.standing_forward_wheel_air_time,
        weight=-4.0,
        params={
            **_ROUGHNESS_KWARGS,
            "contact_sensor_cfg": _WHEEL_CONTACTS,
            "command_name": C.COMMAND_NAME,
            "max_time": 0.5,
            "air_time_offset": 0.05,
            "standing_scale": 2.5,
            "forward_scale": 2.0,
        },
    )


@configclass
class TerminationsCfg:
    non_finite_physics = DoneTerm(func=mdp.non_finite_physics)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": C.FELL_OVER_LIMIT_ANGLE_INITIAL},
    )
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": _ILLEGAL_CONTACTS, "threshold": C.ILLEGAL_CONTACT_FORCE_THRESHOLD},
    )
    # Leaving the terrain is not a failure, so it truncates rather than terminates.
    out_of_terrain_bounds = DoneTerm(func=mdp.out_of_terrain_bounds, params={"margin": 0.05}, time_out=True)


@configclass
class CurriculumCfg:
    fell_over_limit_angle = CurrTerm(
        func=mdp.fell_over_limit_angle,
        params={
            "termination_term_name": "fell_over",
            "initial_limit_angle": C.FELL_OVER_LIMIT_ANGLE_INITIAL,
            "final_limit_angle": C.FELL_OVER_LIMIT_ANGLE_FINAL,
            "ramp_steps": C.FELL_OVER_LIMIT_ANGLE_RAMP_STEPS,
        },
    )
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel, params={"command_name": C.COMMAND_NAME})


##
# Environment
##


@configclass
class WF_TRON1B_BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Rough-terrain WF-TRON1B velocity tracking. Flat is a nulled-down subclass."""

    scene: WF_TRON1B_SceneCfg = WF_TRON1B_SceneCfg(num_envs=C.NUM_ENVS, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = C.DECIMATION
        self.episode_length_s = C.EPISODE_LENGTH_S
        self.sim.dt = C.SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.seed = 0

        if self.scene.terrain_scan is not None:
            self.scene.terrain_scan.update_period = self.decimation * self.sim.dt
        if self.scene.wheel_height_scan_L is not None:
            self.scene.wheel_height_scan_L.update_period = self.decimation * self.sim.dt
        if self.scene.wheel_height_scan_R is not None:
            self.scene.wheel_height_scan_R.update_period = self.decimation * self.sim.dt
        if self.scene.wheel_contact_forces is not None:
            self.scene.wheel_contact_forces.update_period = self.sim.dt
        if self.scene.illegal_contact_forces is not None:
            self.scene.illegal_contact_forces.update_period = self.sim.dt

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        self.viewer.origin_type = "env"

        self._assert_deployment_contract()

    def _assert_deployment_contract(self) -> None:
        """Fail loudly on the two mistakes that would otherwise pass every test.

        Both are invisible in training curves and only surface on hardware, so they are
        checked at config time rather than left to review.
        """
        assert self.scene.robot.soft_joint_pos_limit_factor == 1.0, (
            "WF-TRON1B expects no soft joint-limit shrink (the MJCF applies none). A "
            "non-unity factor silently changes what `leg_joint_pos_limits` penalises."
        )
        assert list(vars(self.actions).keys()) == ["leg_pos", "wheel_vel"], (
            "The action vector layout [leg_pos x6, wheel_vel x2] is part of the deployment "
            f"contract recorded in the exported ONNX metadata; got {list(vars(self.actions).keys())}."
        )
