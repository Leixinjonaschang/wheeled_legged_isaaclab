"""Concrete WF-TRON1B environment configs: flat / rough, train / play.

The rough config is the base (see ``wf_tron1b_base_env_cfg``); flat nulls the
terrain-dependent machinery. Play configs relax whatever exists only to shape training.
"""

import math

from bipedal_locomotion.tasks.locomotion.cfg.WF_TRON1B import constants as C
from bipedal_locomotion.tasks.locomotion.cfg.WF_TRON1B.wf_tron1b_base_env_cfg import WF_TRON1B_BaseEnvCfg
from bipedal_locomotion.tasks.locomotion.mdp import wheeled_events
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass


def _disable_action_delay(cfg: WF_TRON1B_BaseEnvCfg) -> None:
    """Swap the delayed action terms for plain ones.

    Actuation delay is a domain-randomisation device. During evaluation it only adds
    variance to what is meant to be a repeatable rollout, so play runs the undelayed
    terms. The wrapper's ``applied_action`` lookup falls back to ``raw_actions``, so
    nothing downstream needs to know.
    """
    leg, wheel = cfg.actions.leg_pos, cfg.actions.wheel_vel
    cfg.actions.leg_pos = JointPositionActionCfg(
        asset_name=leg.asset_name,
        joint_names=leg.joint_names,
        preserve_order=leg.preserve_order,
        scale=leg.scale,
        use_default_offset=leg.use_default_offset,
    )
    cfg.actions.wheel_vel = JointVelocityActionCfg(
        asset_name=wheel.asset_name,
        joint_names=wheel.joint_names,
        preserve_order=wheel.preserve_order,
        scale=wheel.scale,
        use_default_offset=wheel.use_default_offset,
    )


def _apply_play_overrides(cfg: WF_TRON1B_BaseEnvCfg) -> None:
    """Make a rollout long, quiet and repeatable enough to actually watch."""
    cfg.scene.num_envs = 32
    # Long enough that the episode does not reset while an operator is driving it.
    cfg.episode_length_s = 1.0e9

    cfg.observations.actor.enable_corruption = False
    cfg.observations.actor_history.enable_corruption = False
    _disable_action_delay(cfg)

    cfg.events.push_robot = None

    # Curricula are training schedules; disable them and pin the terms they drive at
    # their final values, so play shows the trained-for behaviour rather than the start
    # of a ramp.
    cfg.curriculum.fell_over_limit_angle = None
    cfg.curriculum.terrain_levels = None
    cfg.terminations.fell_over.params["limit_angle"] = C.FELL_OVER_LIMIT_ANGLE_FINAL
    cfg.terminations.world_command_tracking_failure = None
    for term in vars(cfg.rewards).values():
        params = getattr(term, "params", None)
        if params and "roughness_gate_threshold_ramp_steps" in params:
            params["roughness_gate_threshold"] = C.ROUGHNESS_GATE_THRESHOLD_FINAL
            params["roughness_gate_threshold_ramp_steps"] = 0

    cfg.commands.twist.ranges.lin_vel_x = (-1.0, 1.0)
    cfg.commands.twist.ranges.lin_vel_y = (-1.0, 1.0)
    cfg.commands.twist.ranges.ang_vel_z = (-math.pi / 2, math.pi / 2)


##
# Rough terrain
##


@configclass
class WF_TRON1B_RoughEnvCfg(WF_TRON1B_BaseEnvCfg):
    """Velocity tracking over generated rough terrain.

    The base config is already the rough superset, so this adds nothing but a name for
    the registered task to point at.
    """


@configclass
class WF_TRON1B_RoughEnvCfg_PLAY(WF_TRON1B_RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)
        self.terminations.out_of_terrain_bounds = None
        # Without the terrain-level curriculum every environment would sit on whichever
        # patch it spawned on, so a play session would show a single difficulty.
        self.events.randomize_terrain = EventTerm(func=wheeled_events.randomize_terrain, mode="reset", params={})
        generator = self.scene.terrain.terrain_generator
        if generator is not None:
            generator.curriculum = True
            generator.num_rows = 5
            generator.num_cols = len(generator.sub_terrains)
            generator.border_width = 5.0


##
# Flat terrain
##


@configclass
class WF_TRON1B_FlatEnvCfg(WF_TRON1B_BaseEnvCfg):
    """Velocity tracking on a plane.

    Everything that reads the terrain is removed rather than left pointing at a flat
    scan: with no height variation the roughness gate is identically zero, so the gated
    rewards would contribute nothing while still costing two ray-casts per environment
    per step.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain_scan = None
        self.scene.wheel_height_scan_L = None
        self.scene.wheel_height_scan_R = None
        self.scene.contact_forces.filter_prim_paths_expr = ["/World/ground/terrain/GroundPlane/CollisionPlane"]

        self.observations.critic.height_scan = None
        self.observations.critic.wheel_height = None
        self.observations.critic.roughness_indicator = None

        self.rewards.rough_wheel_usage = None
        self.rewards.rough_wheel_foot_clearance = None
        self.rewards.rough_contact_pattern = None
        self.rewards.non_rough_wheel_lateral_symmetry = None
        self.rewards.non_rough_wheel_x_alignment = None
        self.rewards.non_rough_flat_orientation = None
        self.rewards.standing_forward_wheel_air_time = None

        # These two fall back to their world-frame / absolute definitions, which are
        # correct on a plane.
        self.rewards.upright.params["terrain_sensor_cfg"] = None
        self.rewards.base_height.params["sensor_cfg"] = None

        self.terminations.out_of_terrain_bounds = None
        self.curriculum.terrain_levels = None


@configclass
class WF_TRON1B_FlatEnvCfg_PLAY(WF_TRON1B_FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)
