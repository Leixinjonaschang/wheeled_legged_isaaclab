"""Base environment configuration for the WF-TRON1B velocity tasks: mjlab's rough-terrain
``rep_ts_lin_vel`` environment (``make_env_cfg(rough=True, lin_vel_representation=True)``).

Observation groups:

=======================  ===========  =============================================
group                    width        consumer
=======================  ===========  =============================================
``proprio_history``      5 x 28       student encoder (oldest to newest), deployed
``actor_command``        3            actor, deployed
``lin_vel_target``       3            student velocity-regression target
``critic``               166          critic (+ ``dynamics_context``)
``privileged_encoder``   160          teacher encoder (+ ``dynamics_context``)
``wheel_roughness``      2            depth model's roughness head
``dynamics_context``     87           critic / teacher encoder
``depth_camera``         1 x 30 x 45  depth student encoder (depth variant only)
=======================  ===========  =============================================
"""

import math

from bipedal_locomotion.assets.config.wheelfoot_tron1b_cfg import MESH_DIR, WHEELFOOT_TRON1B_CFG
from bipedal_locomotion.sensors import SelfOccludingRayCasterCameraCfg
from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.terrains import ProportionalTerrainImporter
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
_WHEEL_CONTACTS = SceneEntityCfg(C.CONTACT_SENSOR, body_names=list(C.WHEEL_BODY_NAMES), preserve_order=True)
_ILLEGAL_CONTACTS = SceneEntityCfg(C.CONTACT_SENSOR, body_names=list(C.NON_WHEEL_BODY_NAMES))
_ROBOT_CONTACTS = SceneEntityCfg(C.CONTACT_SENSOR)
_TERRAIN_SCAN = SceneEntityCfg(C.TERRAIN_SCAN_SENSOR)

# Shared by every terrain-gated reward: the gate is computed from the base scan, as in mjlab.
_ROUGHNESS_KWARGS = {
    "roughness_sensor_names": (C.TERRAIN_SCAN_SENSOR,),
    "roughness_max_distance": C.TERRAIN_SCAN_MAX_DISTANCE,
    "wheel_radius": C.WHEEL_RADIUS,
    "gate_min": C.ROUGHNESS_GATE_MIN,
    "gate_max": C.ROUGHNESS_GATE_MAX,
    "grid_shape": C.TERRAIN_SCAN_GRID_SHAPE,
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
        class_type=ProportionalTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=WF_TRON1B_TERRAINS_CFG,
        max_init_terrain_level=8,
        collision_group=-1,
        # MuJoCo combines the friction of two geoms by taking the max; "max" makes PhysX
        # do the same, so the randomised wheel friction only matters where it exceeds 1.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot = WHEELFOOT_TRON1B_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Base-mounted scan: the critic's terrain map and the input to the roughness gate.
    # Yaw-aligned so the map turns with the robot but does not tilt with it.
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

    # Every robot body, filtered on the terrain mesh: the force matrix is the terrain contact
    # (mjlab's wheel / illegal-ground sensors), net minus matrix the robot-on-robot contact (its
    # self-collision sensor). The history spans the physics substeps of one control step.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_Link",
        filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        history_length=C.DECIMATION,
        track_air_time=True,
        force_threshold=C.CONTACT_FORCE_THRESHOLD,
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
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        rel_forward_envs=0.2,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=False,
        ranges=mdp.WheeledUniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0),
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
    class ProprioHistoryCfg(ObsGroup):
        """What the deployed robot measures, corrupted and stacked oldest to newest. Term order is the ONNX contract."""

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
        wheel_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _WHEEL_JOINTS},
            noise=Unoise(n_min=-C.NOISE_WHEEL_VEL, n_max=C.NOISE_WHEEL_VEL),
            scale=C.SCALE_WHEEL_VEL,
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = C.HISTORY_LENGTH
            self.flatten_history_dim = False

    @configclass
    class ActorCommandCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": C.COMMAND_NAME})

    @configclass
    class LinVelTargetCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_link_lin_vel)

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
                "sensor_names": (C.TERRAIN_SCAN_SENSOR,),
                "wheel_radius": C.WHEEL_RADIUS,
                "gate_min": C.ROUGHNESS_GATE_MIN,
                "gate_max": C.ROUGHNESS_GATE_MAX,
                "grid_shape": C.TERRAIN_SCAN_GRID_SHAPE,
                "max_distance": C.TERRAIN_SCAN_MAX_DISTANCE,
            },
        )

    @configclass
    class PrivilegedEncoderCfg(CriticCfg):
        """The critic terms minus what the teacher latent must not shortcut: velocity and command."""

        def __post_init__(self):
            self.base_lin_vel = None
            self.command = None

    @configclass
    class WheelRoughnessCfg(ObsGroup):
        wheel_roughness = ObsTerm(
            func=mdp.wheel_roughness_gate,
            params={
                "sensor_names": C.WHEEL_HEIGHT_SENSORS,
                "wheel_radius": C.WHEEL_RADIUS,
                "gate_min": C.ROUGHNESS_GATE_MIN,
                "gate_max": C.ROUGHNESS_GATE_MAX,
                "grid_shape": C.WHEEL_HEIGHT_GRID_SHAPE,
                "max_distance": C.WHEEL_HEIGHT_MAX_DISTANCE,
            },
        )

    @configclass
    class DynamicsContextCfg(ObsGroup):
        """What was randomised at startup (87 values)."""

        domain_randomization_delta_quantity = ObsTerm(func=mdp.domain_randomization_delta_quantity)

    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()
    actor_command: ActorCommandCfg = ActorCommandCfg()
    lin_vel_target: LinVelTargetCfg = LinVelTargetCfg()
    critic: CriticCfg = CriticCfg()
    privileged_encoder: PrivilegedEncoderCfg = PrivilegedEncoderCfg()
    wheel_roughness: WheelRoughnessCfg = WheelRoughnessCfg()
    dynamics_context: DynamicsContextCfg = DynamicsContextCfg()


# MJCF-ordered selections: the layout of mjlab's privileged DR observation.
_MJCF_JOINTS = SceneEntityCfg("robot", joint_names=list(C.MJCF_JOINT_NAMES), preserve_order=True)
_MJCF_BODIES = SceneEntityCfg("robot", body_names=list(C.MJCF_BODY_NAMES), preserve_order=True)
_MJCF_LINKS = SceneEntityCfg("robot", body_names=list(C.MJCF_BODY_NAMES[1:]), preserve_order=True)


@configclass
class EventsCfg:
    # -- startup: mjlab's domain randomisation, in mjlab's order. `domain_randomization_delta_quantity`
    # reads the ranges and body / joint orders back from these terms.
    wheel_friction = EventTerm(
        func=mdp.randomize_wheel_friction,
        mode="startup",
        params={
            "asset_cfg": _WHEEL_BODIES,
            "friction_range": C.WHEEL_FRICTION_RANGE,
            "difference_range": C.WHEEL_FRICTION_DIFFERENCE_RANGE,
        },
    )
    body_mass_inertia = EventTerm(
        func=mdp.randomize_mass_inertia,
        mode="startup",
        params={"asset_cfg": _MJCF_BODIES, "alpha_range": C.MASS_INERTIA_ALPHA_RANGE},
    )
    encoder_bias = EventTerm(
        func=mdp.randomize_encoder_bias,
        mode="startup",
        params={"asset_cfg": _MJCF_JOINTS, "bias_range": C.ENCODER_BIAS_RANGE},
    )
    pd_gains = EventTerm(
        func=mdp.randomize_pd_gains,
        mode="startup",
        params={
            "asset_cfg": _ALL_JOINTS,
            "stiffness_scale_range": C.STIFFNESS_SCALE_RANGE,
            "damping_scale_range": C.DAMPING_SCALE_RANGE,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_com_offset,
        mode="startup",
        params={"asset_cfg": _BASE_BODY, "com_range": C.BASE_COM_RANGE},
    )
    link_com = EventTerm(
        func=mdp.randomize_com_offset,
        mode="startup",
        params={"asset_cfg": _MJCF_LINKS, "com_range": C.LINK_COM_RANGE},
    )

    # -- reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.01, 0.05), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.2, 0.2),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.2, 0.2),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.3, 0.3), "velocity_range": (-0.2, 0.2), "asset_cfg": _LEG_JOINTS},
    )
    reset_wheel_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (-0.5, 0.5), "asset_cfg": _WHEEL_JOINTS},
    )

    # -- interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.6, 0.6),
                "z": (-0.2, 0.2),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-0.5, 0.5),
            }
        },
    )


@configclass
class RewardsCfg:
    """mjlab ``make_rewards(rough=True, roughness_conditioned_rewards=True)``, same order and weights."""

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
    base_ang_vel_xy = RewTerm(
        func=mdp.base_ang_vel_xy_l2,
        weight=-0.15,
        params={"roll_weight": 2.0, "pitch_weight": 1.0},
    )
    track_heading = RewTerm(
        func=mdp.track_heading,
        weight=0.5,
        params={"command_name": C.COMMAND_NAME, "std": math.sqrt(0.20), "command_norm_threshold": 0.05},
    )
    upright = RewTerm(
        func=mdp.upright,
        weight=1.0,
        params={"std": math.sqrt(0.2), "asset_cfg": _BASE_BODY, "terrain_sensor_cfg": _TERRAIN_SCAN},
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-50.0,
        params={
            "target_height": C.BASE_HEIGHT_TARGET,
            "deadband": C.BASE_HEIGHT_DEADBAND,
            "sensor_cfg": _TERRAIN_SCAN,
            "terrain_quantile": C.BASE_HEIGHT_TERRAIN_QUANTILE,
        },
    )
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
    stand_still = RewTerm(func=mdp.stand_still_world, weight=-2.0, params={"command_name": C.COMMAND_NAME})
    wheel_distance = RewTerm(
        func=mdp.wheel_distance,
        weight=-5.0,
        params={
            "asset_cfg": _WHEEL_BODIES,
            "min_distance": C.WHEEL_DISTANCE_RANGE[0],
            "max_distance": C.WHEEL_DISTANCE_RANGE[1],
        },
    )
    leg_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": _LEG_JOINTS})
    leg_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.015, params={"asset_cfg": _LEG_JOINTS})
    # mjlab's joint-name selection leaves its actuator ids unresolved, so this sums over all
    # eight actuators, wheels included (decision D4).
    leg_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": _ALL_JOINTS})
    wheel_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005, params={"asset_cfg": _WHEEL_JOINTS})
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7, params={"asset_cfg": _ALL_JOINTS})
    joint_power = RewTerm(func=mdp.joint_power_l1, weight=-5.0e-5, params={"asset_cfg": _ALL_JOINTS})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    self_collisions = RewTerm(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_cfg": _ROBOT_CONTACTS, "force_threshold": C.ILLEGAL_CONTACT_FORCE_THRESHOLD},
    )
    illegal_ground_contact = RewTerm(
        func=mdp.illegal_contact_cost,
        weight=-1.0,
        params={"sensor_cfg": _ILLEGAL_CONTACTS, "force_threshold": C.ILLEGAL_CONTACT_FORCE_THRESHOLD},
    )
    soft_landing = RewTerm(
        func=mdp.soft_landing,
        weight=-3.0e-5,
        params={"sensor_cfg": _WHEEL_CONTACTS, "command_name": C.COMMAND_NAME, "command_threshold": 0.05},
    )
    wheel_air_time_balance = RewTerm(
        func=mdp.WheelAirTimeBalance,
        weight=-4.0,
        params={"sensor_cfg": _WHEEL_CONTACTS, "min_total_air_time": 1.0, "balance_tolerance": 0.2},
    )

    # -- terrain-gated: legged motion on rough terrain
    rough_wheel_usage = RewTerm(
        func=mdp.rough_wheel_usage,
        weight=-1.5e-2,
        params={**_ROUGHNESS_KWARGS, "asset_cfg": _WHEEL_JOINTS},
    )
    rough_wheel_foot_clearance = RewTerm(
        func=mdp.rough_wheel_foot_clearance,
        weight=2.0,
        params={
            **_ROUGHNESS_KWARGS,
            "clearance_sensor_names": C.WHEEL_HEIGHT_SENSORS,
            "clearance_grid_shape": C.WHEEL_HEIGHT_GRID_SHAPE,
            "clearance_max_distance": C.WHEEL_HEIGHT_MAX_DISTANCE,
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
    # -- terrain-gated: wheeled motion elsewhere
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
    non_rough_flat_orientation = RewTerm(
        func=mdp.non_rough_flat_orientation,
        weight=-10.0,
        params={**_ROUGHNESS_KWARGS, "roll_weight": 2.0, "pitch_weight": 1.0},
    )
    standing_forward_wheel_air_time = RewTerm(
        func=mdp.standing_forward_wheel_air_time,
        weight=-10.0,
        params={
            **_ROUGHNESS_KWARGS,
            "contact_sensor_cfg": _WHEEL_CONTACTS,
            "command_name": C.COMMAND_NAME,
            "max_time": 0.5,
            "air_time_offset": 0.05,
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
        func=mdp.illegal_terrain_contact,
        params={"sensor_cfg": _ILLEGAL_CONTACTS, "force_threshold": C.ILLEGAL_CONTACT_FORCE_THRESHOLD},
    )
    world_command_tracking_failure = DoneTerm(
        func=mdp.WorldCommandTrackingFailure,
        params={
            "command_name": C.COMMAND_NAME,
            "contact_sensor_cfg": _WHEEL_CONTACTS,
            "activation_step": C.WORLD_COMMAND_TRACKING_ACTIVATION_STEPS,
            "command_norm_threshold": 0.35,
            "progress_deficit_threshold": 0.45,
            "min_progress_ratio": 0.55,
            "progress_duration_s": 0.4,
            "actual_speed_threshold": 0.2,
            "direction_angle_threshold_deg": 70.0,
            "direction_duration_s": 0.3,
            "heading_error_threshold_deg": 45.0,
            "heading_duration_s": 0.6,
            "heading_alignment_gate_deg": 45.0,
            "command_grace_s": 2.5,
        },
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
# Depth camera (mjlab ``make_*(depth=True, async_depth=True)``)
##

_DEPTH_CAMERA = SceneEntityCfg(C.DEPTH_CAMERA_SENSOR)
_DEPTH_FOCAL = 0.5 * C.DEPTH_CAMERA_HEIGHT / math.tan(math.radians(C.DEPTH_CAMERA_FOVY_DEG) / 2)


@configclass
class WF_TRON1B_DepthSceneCfg(WF_TRON1B_SceneCfg):
    """Adds the MJCF's ``d435`` on base_Link. It also sees the robot's wheels and knees, as mjlab's camera does."""

    depth_camera = SelfOccludingRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + C.BASE_BODY,
        mesh_prim_paths=["/World/ground"],
        offset=SelfOccludingRayCasterCameraCfg.OffsetCfg(
            pos=C.DEPTH_CAMERA_POS, rot=C.DEPTH_CAMERA_ROT, convention="opengl"
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            [_DEPTH_FOCAL, 0.0, C.DEPTH_CAMERA_WIDTH / 2, 0.0, _DEPTH_FOCAL, C.DEPTH_CAMERA_HEIGHT / 2, 0.0, 0.0, 1.0],
            C.DEPTH_CAMERA_WIDTH,
            C.DEPTH_CAMERA_HEIGHT,
        ),
        data_types=["distance_to_image_plane"],
        self_occlusion_links=C.DEPTH_CAMERA_VISIBLE_LINKS,
        mesh_dir=MESH_DIR,
    )


@configclass
class DepthObservationsCfg(ObservationsCfg):
    @configclass
    class DepthCameraCfg(ObsGroup):
        """Normalised depth on a 30 Hz capture clock with a per-env system delay, ``[B, 1, 30, 45]``."""

        depth_camera = ObsTerm(
            func=mdp.AsyncDepthBuffer,
            params={
                "sensor_cfg": _DEPTH_CAMERA,
                "capture_frequency_hz": C.DEPTH_CAPTURE_FREQUENCY_HZ,
                "system_delay_range_s": C.DEPTH_SYSTEM_DELAY_RANGE_S,
                "left_crop": C.DEPTH_LEFT_CROP,
                "depth_min_m": C.DEPTH_MIN_M,
                "depth_max_m": C.DEPTH_MAX_M,
            },
        )

    depth_camera: DepthCameraCfg = DepthCameraCfg()


@configclass
class DepthEventsCfg(EventsCfg):
    # mjlab's cam_pos / cam_pitch / cam_fovy: per-reset camera calibration error.
    depth_camera_calibration = EventTerm(
        func=mdp.randomize_depth_camera,
        mode="reset",
        params={
            "sensor_cfg": _DEPTH_CAMERA,
            "position_range": C.DEPTH_CAMERA_POSITION_DELTA_RANGE_M,
            "pitch_range": C.DEPTH_CAMERA_PITCH_DELTA_RANGE_RAD,
            "fovy_range_deg": C.DEPTH_CAMERA_FOVY_DELTA_RANGE_DEG,
        },
    )


##
# Environment
##


@configclass
class WF_TRON1B_BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Rough-terrain WF-TRON1B velocity tracking with the representation observation groups."""

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

        for scan in (self.scene.terrain_scan, self.scene.wheel_height_scan_L, self.scene.wheel_height_scan_R):
            scan.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
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
