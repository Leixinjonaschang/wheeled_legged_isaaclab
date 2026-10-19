"""Reward terms for the WF-TRON1B velocity task (ports of mjlab ``tasks/velocity/mdp/rewards.py``).

Terrain-gated terms are multiplied by (or by one minus) a thresholded smoothstep of the terrain
roughness under the robot, so one policy is paid for rolling on smooth ground and for stepping on
rough ground; the threshold ramps over training. See
:func:`~.kernels.terrain_roughness_from_height_samples`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from bipedal_locomotion.utils.math import quat_apply_inverse, wrap_to_pi
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.string import resolve_matching_names_values

from .kernels import (
    air_time_balance_cost,
    base_height_error_sq,
    body_positions_in_base_frame,
    contact_substep_count,
    ground_height_quantile,
    posture_reward,
    posture_std,
    roughness_gate_active,
    roughness_gate_inactive,
    scheduled_gate_threshold,
    terrain_normal_from_points,
    terrain_roughness_from_height_samples,
    track_angular_velocity_reward,
    track_linear_velocity_reward,
    wheel_distance_cost,
    wheel_lateral_symmetry,
    wheel_x_alignment,
)
from .sensor_utils import (
    contact_state,
    hit_points_with_mask,
    self_contact_forces,
    stack_clearance_samples,
    terrain_forces,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ROBOT = SceneEntityCfg("robot")

__all__ = [
    "VariablePosture",
    "WheelAirTimeBalance",
    "base_ang_vel_xy_l2",
    "base_height_l2",
    "illegal_contact_cost",
    "joint_power_l1",
    "log_metric",
    "non_rough_flat_orientation",
    "non_rough_wheel_lateral_symmetry",
    "non_rough_wheel_x_alignment",
    "rough_contact_pattern",
    "rough_wheel_foot_clearance",
    "rough_wheel_usage",
    "self_collision_cost",
    "soft_landing",
    "stand_still_world",
    "standing_forward_wheel_air_time",
    "track_angular_velocity",
    "track_heading",
    "track_linear_velocity",
    "upright",
    "wheel_distance",
]


def log_metric(env: ManagerBasedRLEnv, key: str, value) -> None:
    """Stage a per-step metric for ``extras["log"]``.

    Isaac Lab rebuilds ``extras["log"]`` when any environment resets (mjlab only clears it at the start
    of ``step``), which would drop metrics written by reward and termination terms; the RSL-RL wrapper
    merges the staged ones after the step.
    """
    env.__dict__.setdefault("_step_metrics", {})[key] = value


def _roughness_gate(
    env: ManagerBasedRLEnv,
    roughness_sensor_names: tuple[str, ...],
    roughness_max_distance: float,
    wheel_radius: float,
    gate_min: float,
    gate_max: float,
    grid_shape: tuple[int, int] | None,
    roughness_gate_threshold: float,
    roughness_gate_threshold_final: float | None,
    roughness_gate_threshold_ramp_steps: int,
) -> tuple[torch.Tensor, float]:
    """Roughness gate of the scanned terrain and the currently scheduled activation threshold."""
    samples = stack_clearance_samples(env, roughness_sensor_names, roughness_max_distance)
    stats = terrain_roughness_from_height_samples(
        samples, wheel_radius=wheel_radius, gate_min=gate_min, gate_max=gate_max, grid_shape=grid_shape
    )
    threshold, progress = scheduled_gate_threshold(
        int(env.common_step_counter),
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    log_metric(env, "Metrics/roughness_mean", stats.foot_roughness.mean())
    log_metric(env, "Metrics/roughness_max_mean", stats.robot_roughness.mean())
    log_metric(env, "Metrics/roughness_lambda_mean", stats.gate.mean())
    log_metric(env, "Metrics/roughness_jump_over_R_mean", (stats.jump / wheel_radius).mean())
    log_metric(env, "Metrics/roughness_curvature_over_R_mean", (stats.curvature / wheel_radius).mean())
    log_metric(env, "Metrics/roughness_gate_threshold", threshold)
    log_metric(env, "Metrics/roughness_gate_threshold_schedule_progress", progress)
    return stats.gate, threshold


def _command_active(env: ManagerBasedRLEnv, command_name: str, command_threshold: float) -> torch.Tensor:
    """1.0 where the commanded motion is above a dead-command threshold."""
    command = env.command_manager.get_term(command_name).command_w
    return ((torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > command_threshold).float()


def _wheel_pos_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return body_positions_in_base_frame(
        asset.data.body_link_pos_w[:, asset_cfg.body_ids, :],
        asset.data.root_link_pos_w,
        asset.data.root_link_quat_w,
    )


# --------------------------------------------------------------------------------------
# Velocity and heading tracking
# --------------------------------------------------------------------------------------


def track_linear_velocity(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """Track the commanded linear velocity in the world frame."""
    command_w = env.command_manager.get_term(command_name).command_w
    return track_linear_velocity_reward(command_w[:, :2], env.scene[asset_cfg.name].data.root_link_lin_vel_w, std)


def track_angular_velocity(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """Track the commanded yaw rate."""
    command = env.command_manager.get_command(command_name)
    return track_angular_velocity_reward(command[:, 2], env.scene[asset_cfg.name].data.root_link_ang_vel_b, std)


def track_heading(
    env: ManagerBasedRLEnv, std: float, command_name: str, command_norm_threshold: float = 0.2
) -> torch.Tensor:
    """Reward aligning the base heading with the commanded world heading while moving."""
    term = env.command_manager.get_term(command_name)
    active = torch.norm(term.command_w[:, :2], dim=1) > command_norm_threshold
    heading_error = wrap_to_pi(term.heading_target - term.robot.data.heading_w)
    return torch.exp(-torch.square(heading_error) / std**2) * active.float()


def stand_still_world(
    env: ManagerBasedRLEnv,
    command_name: str,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Penalise world-frame drift when the command is (near) zero (mjlab ``stand_still``)."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    still = (torch.norm(command[:, :2], dim=1) < lin_threshold) & (torch.abs(command[:, 2]) < ang_threshold)
    lin_drift = torch.sum(torch.abs(asset.data.root_link_lin_vel_w[:, :2]), dim=1)
    yaw_drift = torch.abs(asset.data.root_link_ang_vel_w[:, 2])
    return (lin_drift + yaw_drift) * still.float()


# --------------------------------------------------------------------------------------
# Base regulation
# --------------------------------------------------------------------------------------


def base_ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT, roll_weight: float = 1.0, pitch_weight: float = 1.0
) -> torch.Tensor:
    """Penalise base roll and pitch rate with independent weights."""
    ang_vel_xy_sq = torch.square(env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, :2])
    return roll_weight * ang_vel_xy_sq[:, 0] + pitch_weight * ang_vel_xy_sq[:, 1]


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
    sensor_cfg: SceneEntityCfg | None = None,
    terrain_quantile: float = 0.5,
    deadband: float = 0.0,
) -> torch.Tensor:
    """Penalise base-height error outside a deadband, relative to a height-scan quantile when given.

    The quantile (mjlab ``terrain_sample="quantile"``) is robust to sparse deep holes such as
    stepping-stone pits.
    """
    base_height = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2]
    if sensor_cfg is not None:
        hit_z = env.scene[sensor_cfg.name].data.ray_hits_w[..., 2]
        base_height = base_height - ground_height_quantile(hit_z, torch.isfinite(hit_z), terrain_quantile)
    return base_height_error_sq(base_height, target_height, deadband)


def upright(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
    terrain_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward keeping the base upright: aligned with the terrain normal fitted to a height scan if
    given, else with world up."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    body_quat_w = asset.data.root_link_quat_w if isinstance(ids, slice) else asset.data.body_link_quat_w[:, ids[0]]
    if terrain_sensor_cfg is None:
        up_w = torch.zeros_like(body_quat_w[:, 1:])
        up_w[:, 2] = 1.0
    else:
        up_w = terrain_normal_from_points(*hit_points_with_mask(env, (terrain_sensor_cfg.name,)))
    xy_squared = torch.sum(torch.square(quat_apply_inverse(body_quat_w, up_w)[:, :2]), dim=1)
    return torch.exp(-xy_squared / std**2)


class VariablePosture(ManagerTermBase):
    """Reward holding a reference leg pose with a per-joint tolerance that widens with commanded speed."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        joint_names = asset.joint_names if isinstance(joint_ids, slice) else [asset.joint_names[i] for i in joint_ids]

        target = cfg.params.get("target_joint_pos")
        if target is None:
            self.target_joint_pos = asset.data.default_joint_pos[:, joint_ids]
        else:
            _, _, values = resolve_matching_names_values(target, joint_names)
            self.target_joint_pos = torch.tensor(values, device=env.device, dtype=torch.float32).unsqueeze(0)

        def _std(key: str) -> torch.Tensor:
            _, _, values = resolve_matching_names_values(cfg.params[key], joint_names)
            return torch.tensor(values, device=env.device, dtype=torch.float32)

        self.std_standing = _std("std_standing")
        self.std_walking = _std("std_walking")
        self.std_running = _std("std_running")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        asset_cfg: SceneEntityCfg,
        command_name: str,
        target_joint_pos: dict[str, float] | None = None,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running, target_joint_pos  # resolved in __init__
        command = env.command_manager.get_command(command_name)
        total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        std = posture_std(
            total_speed, self.std_standing, self.std_walking, self.std_running, walking_threshold, running_threshold
        )
        joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
        return posture_reward(joint_pos, self.target_joint_pos, std)


# --------------------------------------------------------------------------------------
# Joints and geometry
# --------------------------------------------------------------------------------------


def joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Penalise mechanical joint power with an L1 kernel."""
    asset = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(power), dim=1)


def wheel_distance(
    env: ManagerBasedRLEnv, min_distance: float, max_distance: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalise a horizontal wheel separation outside ``[min_distance, max_distance]``."""
    return wheel_distance_cost(_wheel_pos_b(env, asset_cfg), min_distance, max_distance)


# --------------------------------------------------------------------------------------
# Contact
# --------------------------------------------------------------------------------------


def illegal_contact_cost(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 10.0
) -> torch.Tensor:
    """Substeps in which a selected body pressed on the terrain harder than ``force_threshold``."""
    return contact_substep_count(terrain_forces(env, sensor_cfg, history=True), force_threshold)


def self_collision_cost(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 10.0
) -> torch.Tensor:
    """Substeps in which a selected body touched the robot itself harder than ``force_threshold``."""
    return contact_substep_count(self_contact_forces(env, sensor_cfg), force_threshold)


def soft_landing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalise the terrain impact force at touchdown."""
    sensor = env.scene[sensor_cfg.name]
    force = torch.norm(torch.nan_to_num(terrain_forces(env, sensor_cfg), nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].float()
    landing_impact = force * first_contact
    log_metric(env, "Metrics/landing_force_mean", landing_impact.sum() / first_contact.sum().clamp(min=1))
    cost = landing_impact.sum(dim=1)
    if command_name is not None:
        cost = cost * _command_active(env, command_name, command_threshold)
    return cost


class WheelAirTimeBalance(ManagerTermBase):
    """Penalise an episode-level left/right imbalance in accumulated wheel air time."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._cumulative_air_time = torch.zeros(env.num_envs, 2, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        min_total_air_time: float = 1.0,
        balance_tolerance: float = 0.2,
    ) -> torch.Tensor:
        in_contact = contact_state(env, sensor_cfg)
        if in_contact.shape[1] != 2:
            raise ValueError(f"WheelAirTimeBalance expects exactly two wheel bodies, got {in_contact.shape[1]}")
        self._cumulative_air_time += (~in_contact).float() * env.step_dt
        cost = air_time_balance_cost(self._cumulative_air_time, min_total_air_time, balance_tolerance)
        log_metric(env, "Metrics/wheel_episode_air_time_left_mean", self._cumulative_air_time[:, 0].mean())
        log_metric(env, "Metrics/wheel_episode_air_time_right_mean", self._cumulative_air_time[:, 1].mean())
        log_metric(env, "Metrics/wheel_air_time_balance_cost_mean", cost.mean())
        return cost

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._cumulative_air_time[slice(None) if env_ids is None else env_ids] = 0.0


# --------------------------------------------------------------------------------------
# Terrain-gated terms. Every one takes the same roughness parameters (see `_roughness_gate`).
# --------------------------------------------------------------------------------------


def rough_wheel_usage(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    roughness_sensor_names: tuple[str, ...],
    roughness_max_distance: float = 10.0,
    wheel_radius: float = 0.127,
    gate_min: float = 0.10,
    gate_max: float = 0.40,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Penalise wheel speed where the terrain is rough."""
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        roughness_max_distance,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    wheel_vel = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return roughness_gate_active(gate, threshold) * torch.sum(torch.square(wheel_vel), dim=1)


def rough_wheel_foot_clearance(
    env: ManagerBasedRLEnv,
    clearance_sensor_names: tuple[str, ...],
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str,
    roughness_sensor_names: tuple[str, ...],
    roughness_max_distance: float = 10.0,
    wheel_radius: float = 0.127,
    gate_min: float = 0.10,
    gate_max: float = 0.40,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
    clearance_grid_shape: tuple[int, int] | None = None,
    clearance_max_distance: float = 1.0,
    base_target_height: float = 0.06,
    range_scale: float = 0.5,
    max_target_height: float = 0.18,
    target_std: float = 0.04,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward swing-wheel clearance toward a target set by the local terrain height range."""
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        roughness_max_distance,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    samples = stack_clearance_samples(env, clearance_sensor_names, clearance_max_distance)  # [B, F, N]
    if clearance_grid_shape is not None and clearance_grid_shape[0] * clearance_grid_shape[1] != samples.shape[-1]:
        raise ValueError(f"clearance_grid_shape={clearance_grid_shape} does not match {samples.shape[-1]} samples")
    local_height_range = samples.amax(dim=-1) - samples.amin(dim=-1)
    wheel_foot_clearance = torch.clamp(samples.amin(dim=-1) - wheel_radius, min=0.0)
    in_air = (~contact_state(env, contact_sensor_cfg)).float()

    target = torch.clamp(base_target_height + range_scale * local_height_range, max=max_target_height)
    reward_per_foot = torch.exp(-torch.square(wheel_foot_clearance - target) / target_std**2)
    reward = (
        torch.sum(reward_per_foot * in_air, dim=1)
        * roughness_gate_active(gate, threshold)
        * _command_active(env, command_name, command_threshold)
    )
    log_metric(env, "Metrics/rough_wheel_foot_clearance_mean", wheel_foot_clearance.mean())
    log_metric(env, "Metrics/rough_wheel_foot_clearance_target_mean", target.mean())
    return reward


def rough_contact_pattern(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str,
    roughness_sensor_names: tuple[str, ...],
    roughness_max_distance: float = 10.0,
    wheel_radius: float = 0.127,
    gate_min: float = 0.10,
    gate_max: float = 0.40,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """On rough terrain, -1 for double support or flight while commanded to move."""
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        roughness_max_distance,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    in_contact = contact_state(env, contact_sensor_cfg)
    contact_count = torch.sum(in_contact.float(), dim=1)
    double_contact = (contact_count == in_contact.shape[1]).float()
    no_contact = (contact_count == 0).float()
    active = _command_active(env, command_name, command_threshold)
    log_metric(env, "Metrics/rough_double_contact_mean", double_contact.mean())
    log_metric(env, "Metrics/rough_no_contact_mean", no_contact.mean())
    log_metric(env, "Metrics/rough_single_contact_mean", (contact_count == 1).float().mean())
    return -(double_contact + no_contact) * roughness_gate_active(gate, threshold) * active


def non_rough_wheel_lateral_symmetry(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    roughness_sensor_names: tuple[str, ...],
    roughness_max_distance: float = 10.0,
    wheel_radius: float = 0.127,
    gate_min: float = 0.10,
    gate_max: float = 0.40,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Reward a laterally symmetric wheel stance where the terrain is not rough."""
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        roughness_max_distance,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    return roughness_gate_inactive(gate, threshold) * wheel_lateral_symmetry(_wheel_pos_b(env, asset_cfg), std)


def non_rough_wheel_x_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    roughness_sensor_names: tuple[str, ...],
    roughness_max_distance: float = 10.0,
    wheel_radius: float = 0.127,
    gate_min: float = 0.10,
    gate_max: float = 0.40,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Penalise fore-aft wheel offset where the terrain is not rough."""
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        roughness_max_distance,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    return roughness_gate_inactive(gate, threshold) * wheel_x_alignment(_wheel_pos_b(env, asset_cfg))


def non_rough_flat_orientation(
    env: ManagerBasedRLEnv,
    roughness_sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = _ROBOT,
    roll_weight: float = 1.0,
    pitch_weight: float = 1.0,
    roughness_max_distance: float = 10.0,
    wheel_radius: float = 0.127,
    gate_min: float = 0.10,
    gate_max: float = 0.40,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Penalise static base roll/pitch tilt where the terrain is not rough."""
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        roughness_max_distance,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    # projected gravity: x carries the pitch tilt, y the roll tilt.
    tilt_sq = torch.square(env.scene[asset_cfg.name].data.projected_gravity_b[:, :2])
    return roughness_gate_inactive(gate, threshold) * (roll_weight * tilt_sq[:, 1] + pitch_weight * tilt_sq[:, 0])


def standing_forward_wheel_air_time(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str,
    roughness_sensor_names: tuple[str, ...],
    roughness_max_distance: float = 10.0,
    wheel_radius: float = 0.127,
    gate_min: float = 0.10,
    gate_max: float = 0.40,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
    max_time: float = 0.5,
    air_time_offset: float = 0.05,
) -> torch.Tensor:
    """Penalise wheel air time when standing (anywhere) and when driving straight on non-rough terrain."""
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        roughness_max_distance,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    current_air_time = env.scene[contact_sensor_cfg.name].data.current_air_time[:, contact_sensor_cfg.body_ids]
    in_air = (~contact_state(env, contact_sensor_cfg)).float()
    air_time = torch.sum((torch.clamp(current_air_time, max=max_time) + air_time_offset) * in_air, dim=1)

    term = env.command_manager.get_term(command_name)
    standing = term.is_standing_env
    forward = term.is_forward_env & ~standing
    standing_cost = air_time * standing.float()
    forward_cost = air_time * forward.float() * roughness_gate_inactive(gate, threshold)
    cost = standing_cost + forward_cost
    log_metric(env, "Metrics/standing_forward_wheel_air_time_mean", cost.mean())
    log_metric(env, "Metrics/standing_wheel_air_time_mean", standing_cost.mean())
    log_metric(env, "Metrics/non_rough_forward_wheel_air_time_mean", forward_cost.mean())
    return cost
