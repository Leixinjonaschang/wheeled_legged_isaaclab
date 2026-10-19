"""Concrete WF-TRON1B environment configs, train and play (mjlab ``env_cfgs.py`` factories).

Play configs relax whatever exists only to shape training (mjlab ``apply_play_overrides``).
"""

import math

from bipedal_locomotion.tasks.locomotion.cfg.WF_TRON1B import constants as C
from bipedal_locomotion.tasks.locomotion.cfg.WF_TRON1B.wf_tron1b_base_env_cfg import WF_TRON1B_BaseEnvCfg
from bipedal_locomotion.tasks.locomotion.mdp import wheeled_events
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass


def _disable_action_delay(cfg: WF_TRON1B_BaseEnvCfg) -> None:
    """Swap the delayed action terms for plain ones (mjlab ``make_actions(action_delay=False)``)."""
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
    """Make a rollout long, quiet and repeatable enough to watch."""
    cfg.scene.num_envs = 32
    cfg.episode_length_s = 1.0e9
    cfg.observations.proprio_history.enable_corruption = False
    _disable_action_delay(cfg)
    cfg.events.push_robot = None

    # Curricula are training schedules: disable them and pin what they drive at the final values.
    cfg.curriculum.fell_over_limit_angle = None
    cfg.curriculum.terrain_levels = None
    cfg.terminations.fell_over.params["limit_angle"] = C.FELL_OVER_LIMIT_ANGLE_FINAL
    cfg.terminations.world_command_tracking_failure = None
    cfg.terminations.out_of_terrain_bounds = None
    for term in vars(cfg.rewards).values():
        params = getattr(term, "params", None)
        if params and "roughness_gate_threshold_ramp_steps" in params:
            params["roughness_gate_threshold"] = C.ROUGHNESS_GATE_THRESHOLD_FINAL
            params["roughness_gate_threshold_ramp_steps"] = 0

    cfg.commands.twist.ranges.lin_vel_x = (-1.0, 1.0)
    cfg.commands.twist.ranges.lin_vel_y = (-1.0, 1.0)
    cfg.commands.twist.ranges.ang_vel_z = (-math.pi / 2, math.pi / 2)

    # Without the terrain-level curriculum every environment would stay on the patch it spawned
    # on; respawn on a random one instead, over a smaller grid (one column per sub-terrain).
    cfg.events.randomize_terrain = EventTerm(func=wheeled_events.randomize_terrain, mode="reset", params={})
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows = 5
    generator.border_width = 5.0


@configclass
class WF_TRON1B_RepTSLinVelEnvCfg(WF_TRON1B_BaseEnvCfg):
    """mjlab ``wf_tron1b_rough_rep_ts_lin_vel_env_cfg()``: the base config as it stands."""


@configclass
class WF_TRON1B_RepTSLinVelEnvCfg_PLAY(WF_TRON1B_RepTSLinVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_overrides(self)
