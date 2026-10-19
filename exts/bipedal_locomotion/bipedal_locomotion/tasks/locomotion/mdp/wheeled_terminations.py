"""Termination terms for the WF-TRON1B velocity task (ports of mjlab ``tasks/velocity/mdp/terminations.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from bipedal_locomotion.utils.math import wrap_to_pi
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .kernels import CommandTrackingMonitor, contact_substep_count
from .sensor_utils import contact_state, terrain_forces
from .wheeled_rewards import log_metric

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ROBOT = SceneEntityCfg("robot")

__all__ = ["WorldCommandTrackingFailure", "illegal_terrain_contact", "non_finite_physics", "out_of_terrain_bounds"]


def non_finite_physics(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Terminate environments whose articulation state or contact forces have gone non-finite."""
    asset = env.scene[asset_cfg.name]
    bad = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for tensor in (
        asset.data.root_state_w,
        asset.data.joint_pos,
        asset.data.joint_vel,
        asset.data.joint_acc,
        asset.data.applied_torque,
    ):
        bad |= ~torch.isfinite(tensor).reshape(env.num_envs, -1).all(dim=1)
    for sensor in env.scene.sensors.values():
        forces = getattr(sensor.data, "net_forces_w", None)
        if forces is not None:
            bad |= ~torch.isfinite(forces).reshape(env.num_envs, -1).all(dim=1)
    return bad


def illegal_terrain_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 10.0
) -> torch.Tensor:
    """Terminate when a selected body pressed on the terrain harder than ``force_threshold`` in any
    substep of the last control step (mjlab ``illegal_contact``)."""
    return contact_substep_count(terrain_forces(env, sensor_cfg, history=True), force_threshold) > 0


class WorldCommandTrackingFailure(ManagerTermBase):
    """Terminate sustained world-frame command tracking failures (see :class:`~.kernels.CommandTrackingMonitor`).

    Inactive until ``activation_step`` global policy steps have elapsed.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._monitor = CommandTrackingMonitor(env.num_envs, env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        contact_sensor_cfg: SceneEntityCfg,
        activation_step: int = 120_000,
        command_norm_threshold: float = 0.35,
        progress_deficit_threshold: float = 0.45,
        min_progress_ratio: float = 0.55,
        progress_duration_s: float = 0.4,
        actual_speed_threshold: float = 0.2,
        direction_angle_threshold_deg: float = 70.0,
        direction_duration_s: float = 0.3,
        heading_error_threshold_deg: float = 55.0,
        heading_duration_s: float = 0.6,
        heading_alignment_gate_deg: float = 45.0,
        command_grace_s: float = 2.5,
        asset_cfg: SceneEntityCfg = _ROBOT,
    ) -> torch.Tensor:
        term = env.command_manager.get_term(command_name)
        done, metrics = self._monitor.update(
            command_counter=term.command_counter,
            command_xy=term.command_w[:, :2],
            actual_xy=env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, :2],
            heading_error=wrap_to_pi(term.heading_target - term.robot.data.heading_w),
            grounded=contact_state(env, contact_sensor_cfg).any(dim=-1),
            is_heading_env=term.is_heading_env,
            is_standing_env=term.is_standing_env,
            enabled=env.common_step_counter >= activation_step,
            step_dt=env.step_dt,
            command_norm_threshold=command_norm_threshold,
            progress_deficit_threshold=progress_deficit_threshold,
            min_progress_ratio=min_progress_ratio,
            progress_duration_s=progress_duration_s,
            actual_speed_threshold=actual_speed_threshold,
            direction_angle_threshold_deg=direction_angle_threshold_deg,
            direction_duration_s=direction_duration_s,
            heading_error_threshold_deg=heading_error_threshold_deg,
            heading_duration_s=heading_duration_s,
            heading_alignment_gate_deg=heading_alignment_gate_deg,
            command_grace_s=command_grace_s,
        )
        for key, value in metrics.items():
            log_metric(env, f"Metrics/world_command_tracking_{key}", value)
        return done

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._monitor.reset(slice(None) if env_ids is None else env_ids)


def out_of_terrain_bounds(
    env: ManagerBasedRLEnv, margin: float = 0.3, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """Truncate once the robot has left the generated sub-terrain grid by more than ``margin``.

    Leaving the terrain is not a failure, so configure it with ``time_out=True``.
    """
    terrain = env.scene.terrain
    size = terrain.cfg.terrain_generator.size
    # The realised grid shape: curriculum mode can override the configured column count.
    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    root_xy_w = env.scene[asset_cfg.name].data.root_link_pos_w[:, :2]
    limit_x = 0.5 * num_rows * size[0] + margin
    limit_y = 0.5 * num_cols * size[1] + margin
    return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)
