"""Reward terms for the wheeled-legged WF-TRON1B velocity task.

Two families live here:

* Terrain-agnostic terms -- velocity tracking, posture, base regulation, contact safety.
* Terrain-*gated* terms, which are multiplied by (or by one minus) a smoothstep of the
  local terrain roughness. A wheeled biped should roll on smooth ground and step on
  rough ground, and those two behaviours want opposite rewards; the gate is what lets a
  single policy be paid for both without the reward flipping discontinuously at the
  boundary. See :func:`~.kernels.terrain_roughness_from_height_samples`.

Stateful terms subclass ``ManagerTermBase`` so the manager resets their buffers per
environment on episode reset.
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
from .sensor_utils import hit_points_with_mask, stack_clearance_samples

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ROBOT = SceneEntityCfg("robot")

__all__ = [
    "HeadingProgress",
    "Upright",
    "VariablePosture",
    "WheelAirTimeBalance",
    "base_ang_vel_xy_l2",
    "base_height_l2",
    "contact_force_cost",
    "non_rough_wheel_lateral_symmetry",
    "non_rough_wheel_x_alignment",
    "rough_contact_pattern",
    "rough_wheel_foot_clearance",
    "rough_wheel_usage",
    "soft_landing",
    "stand_still_world",
    "standing_forward_wheel_air_time",
    "track_angular_velocity",
    "track_heading",
    "track_linear_velocity",
    "wheel_distance",
]


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _log(env: ManagerBasedRLEnv, key: str, value) -> None:
    env.extras.setdefault("log", {})[key] = value


def _roughness_gate(
    env: ManagerBasedRLEnv,
    roughness_sensor_names: tuple[str, ...],
    wheel_radius: float,
    gate_min: float,
    gate_max: float,
    grid_shape: tuple[int, int] | None,
    roughness_gate_threshold: float,
    roughness_gate_threshold_final: float | None,
    roughness_gate_threshold_ramp_steps: int,
    max_distance: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Compute the roughness gate and the currently scheduled activation threshold."""
    samples = stack_clearance_samples(env, roughness_sensor_names, max_distance)
    stats = terrain_roughness_from_height_samples(
        samples, wheel_radius=wheel_radius, gate_min=gate_min, gate_max=gate_max, grid_shape=grid_shape
    )
    threshold, progress = scheduled_gate_threshold(
        int(env.common_step_counter),
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    _log(env, "Metrics/roughness_mean", stats.foot_roughness.mean())
    _log(env, "Metrics/roughness_max_mean", stats.robot_roughness.mean())
    _log(env, "Metrics/roughness_lambda_mean", stats.gate.mean())
    _log(env, "Metrics/roughness_gate_threshold", threshold)
    _log(env, "Metrics/roughness_gate_threshold_schedule_progress", progress)
    return stats.gate, threshold


def _command_term(env: ManagerBasedRLEnv, command_name: str):
    term = env.command_manager.get_term(command_name)
    assert term is not None, f"Command '{command_name}' not found."
    return term


def _command_active(env: ManagerBasedRLEnv, command_name: str, command_threshold: float) -> torch.Tensor:
    """1.0 where the commanded motion is above a dead-command threshold."""
    command = _command_term(env, command_name).command_w
    return ((torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])) > command_threshold).float()


# --------------------------------------------------------------------------------------
# Velocity and heading tracking
# --------------------------------------------------------------------------------------


def track_linear_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Track the commanded linear velocity in the world frame."""
    asset = env.scene[asset_cfg.name]
    command_w = _command_term(env, command_name).command_w
    return track_linear_velocity_reward(command_w[:, :2], asset.data.root_link_lin_vel_w, std)


def track_angular_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Track the commanded yaw rate."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return track_angular_velocity_reward(command[:, 2], asset.data.root_link_ang_vel_b, std)


def track_heading(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    command_norm_threshold: float = 0.2,
) -> torch.Tensor:
    """Reward aligning the base heading with the commanded world heading.

    Inactive when the commanded speed is near zero, where "the heading to travel in" is
    not defined and holding an arbitrary yaw would be an unearned reward.
    """
    term = _command_term(env, command_name)
    command_speed = torch.norm(term.command_w[:, :2], dim=1)
    active = command_speed > command_norm_threshold
    heading_error = wrap_to_pi(term.heading_target - term.robot.data.heading_w)
    return torch.exp(-torch.square(heading_error) / std**2) * active.float()


class HeadingProgress(ManagerTermBase):
    """Reward *reducing* absolute heading error between consecutive steps.

    A pure heading-error reward pays nothing until the robot is already aligned, which
    gives no gradient when it starts facing backwards. This term pays for the turn
    itself. It is clamped so that spinning fast cannot farm it, and it is disabled on the
    step a new command is sampled (where the error jumps for reasons the policy did not
    cause) and for standing environments.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_abs_heading_error = torch.zeros(env.num_envs, device=env.device)
        self._has_prev = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._last_command_counter = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        max_progress: float = 0.05,
    ) -> torch.Tensor:
        term = _command_term(env, command_name)
        heading_error = wrap_to_pi(term.heading_target - term.robot.data.heading_w)
        abs_heading_error = torch.abs(heading_error)

        command_changed = term.command_counter != self._last_command_counter
        active = term.is_heading_env & ~term.is_standing_env
        valid = self._has_prev & ~command_changed & active

        progress_scale = max(max_progress, 1.0e-6)
        progress = self._prev_abs_heading_error - abs_heading_error
        reward = torch.clamp(progress, min=0.0, max=progress_scale) / progress_scale
        reward = reward * valid.float()

        _log(env, "Metrics/heading_progress_mean", reward.mean())

        self._prev_abs_heading_error = abs_heading_error.detach()
        self._has_prev[:] = True
        self._last_command_counter = term.command_counter.clone()
        return reward

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_abs_heading_error[env_ids] = 0.0
        self._has_prev[env_ids] = False
        self._last_command_counter[env_ids] = -1


def stand_still_world(
    env: ManagerBasedRLEnv,
    command_name: str,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Penalise world-frame drift when the command is (near) zero.

    Wheels make drifting nearly free, so without this the robot has no incentive to
    actually stop. Measured in the world frame because the failure is displacement, not
    body-frame velocity.
    """
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
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = _ROBOT,
    roll_weight: float = 1.0,
    pitch_weight: float = 1.0,
) -> torch.Tensor:
    """Penalise base roll and pitch rate with independent weights.

    Pitch is weighted harder than roll: for a wheeled biped, pitch oscillation is the
    balance failure mode, whereas some roll is unavoidable on uneven ground.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_xy_sq = torch.square(asset.data.root_link_ang_vel_b[:, :2])
    return roll_weight * ang_vel_xy_sq[:, 0] + pitch_weight * ang_vel_xy_sq[:, 1]


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
    sensor_cfg: SceneEntityCfg | None = None,
    deadband: float = 0.0,
) -> torch.Tensor:
    """Penalise base-height error outside a deadband, terrain-relative when a scan exists.

    On generated terrain the world-frame height is meaningless -- standing on a 15 cm
    block is not the same failure as crouching 15 cm. With ``sensor_cfg`` the height is
    measured against the terrain sample directly beneath the base.
    """
    asset = env.scene[asset_cfg.name]
    base_z = asset.data.root_link_pos_w[:, 2]
    if sensor_cfg is None:
        base_height = base_z
    else:
        sensor = env.scene[sensor_cfg.name]
        hit_z = sensor.data.ray_hits_w[..., 2]
        centre = hit_z.shape[1] // 2
        ground_z = hit_z[:, centre]
        ground_z = torch.where(torch.isfinite(ground_z), ground_z, torch.zeros_like(ground_z))
        base_height = base_z - ground_z
    return base_height_error_sq(base_height, target_height, deadband)


class Upright(ManagerTermBase):
    """Reward keeping the base upright, relative to the terrain surface when known.

    On a slope, "upright" in the world sense means leaning into the hill. Fitting a plane
    through the height-scan hits and measuring tilt against *that* normal lets the robot
    stand perpendicular to the ground it is on, which is what keeps the wheels loaded.
    Without a terrain sensor this degrades to the world-up definition, which is correct
    on flat ground.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        asset_cfg: SceneEntityCfg = _ROBOT,
        terrain_sensor_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        body_ids = asset_cfg.body_ids
        if body_ids is not None and not isinstance(body_ids, slice) and len(body_ids) > 0:
            body_quat_w = asset.data.body_link_quat_w[:, body_ids, :].squeeze(1)
        else:
            body_quat_w = asset.data.root_link_quat_w

        if terrain_sensor_cfg is not None:
            points, valid = hit_points_with_mask(env, (terrain_sensor_cfg.name,))
            normal_w = terrain_normal_from_points(points, valid)
            target_b = quat_apply_inverse(body_quat_w, normal_w)
            xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
        else:
            gravity_w = asset.data.GRAVITY_VEC_W
            projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
            xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

        return torch.exp(-xy_squared / std**2)


class VariablePosture(ManagerTermBase):
    """Reward holding a reference leg pose, with a speed-dependent tolerance.

    A fixed posture penalty either cripples fast motion or fails to keep the robot from
    collapsing when standing. Instead the tolerance -- per joint -- widens across three
    commanded-speed regimes, so the same term stabilises a stand and still permits the
    leg travel that running needs.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            joint_names = list(asset.joint_names)
        else:
            joint_names = [asset.joint_names[i] for i in joint_ids]

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

        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        total_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])

        std = posture_std(
            total_speed,
            self.std_standing,
            self.std_walking,
            self.std_running,
            walking_threshold,
            running_threshold,
        )
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        return posture_reward(joint_pos, self.target_joint_pos, std)


# --------------------------------------------------------------------------------------
# Wheel geometry
# --------------------------------------------------------------------------------------


def wheel_distance(
    env: ManagerBasedRLEnv,
    min_distance: float,
    max_distance: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise a wheel separation outside the mechanically sensible band."""
    asset = env.scene[asset_cfg.name]
    wheel_pos_b = body_positions_in_base_frame(
        asset.data.body_link_pos_w[:, asset_cfg.body_ids, :],
        asset.data.root_link_pos_w,
        asset.data.root_link_quat_w,
    )
    return wheel_distance_cost(wheel_pos_b, min_distance, max_distance)


def _wheel_pos_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return body_positions_in_base_frame(
        asset.data.body_link_pos_w[:, asset_cfg.body_ids, :],
        asset.data.root_link_pos_w,
        asset.data.root_link_quat_w,
    )


def non_rough_wheel_lateral_symmetry(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    roughness_sensor_names: tuple[str, ...],
    wheel_radius: float = 0.127,
    gate_min: float = 0.0,
    gate_max: float = 0.5,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Reward a symmetric wheel stance -- but only on ground smooth enough to roll on."""
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
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
    wheel_radius: float = 0.127,
    gate_min: float = 0.0,
    gate_max: float = 0.5,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Penalise fore-aft wheel offset on smooth ground, where stepping is not needed."""
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    return roughness_gate_inactive(gate, threshold) * wheel_x_alignment(_wheel_pos_b(env, asset_cfg))


# --------------------------------------------------------------------------------------
# Contact
# --------------------------------------------------------------------------------------


def contact_force_cost(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Count contact-history substeps where any selected body was loaded past a threshold.

    Counting substeps rather than sampling once per control step means a brief impact
    between control steps still registers, which is exactly when illegal contacts happen.
    """
    sensor = env.scene[sensor_cfg.name]
    history = sensor.data.net_forces_w_history  # [B, H, N, 3]
    history = torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)
    force_mag = torch.norm(history[:, :, sensor_cfg.body_ids], dim=-1)  # [B, H, N]
    return (force_mag > force_threshold).any(dim=-1).sum(dim=-1).float()


def soft_landing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalise impact force at the moment of touchdown.

    Scored only on the first contact step so it costs nothing to *stay* on the ground --
    the term shapes how the robot lands, not how much weight it carries.
    """
    sensor = env.scene[sensor_cfg.name]
    forces = torch.nan_to_num(sensor.data.net_forces_w[:, sensor_cfg.body_ids], nan=0.0, posinf=0.0, neginf=0.0)
    force_magnitude = torch.norm(forces, dim=-1)
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    landing_impact = force_magnitude * first_contact.float()
    cost = torch.sum(landing_impact, dim=1)

    num_landings = torch.sum(first_contact.float())
    _log(env, "Metrics/landing_force_mean", torch.sum(landing_impact) / torch.clamp(num_landings, min=1))

    if command_name is not None:
        cost = cost * _command_active(env, command_name, command_threshold)
    return cost


class WheelAirTimeBalance(ManagerTermBase):
    """Penalise an episode-long left/right imbalance in accumulated wheel air time.

    Catches the degenerate gait where the robot props itself on one wheel and hops on the
    other: every per-step contact term is satisfied, but the behaviour is useless and
    does not transfer. Only an episode-level accumulator can see it.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._cumulative_air_time = torch.zeros(env.num_envs, 2, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        min_total_air_time: float = 1.0,
        balance_tolerance: float = 0.2,
        force_threshold: float = 1.0,
    ) -> torch.Tensor:
        sensor = env.scene[sensor_cfg.name]
        forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
        if forces.shape[1] != 2:
            raise ValueError(f"WheelAirTimeBalance expects exactly two wheel bodies, got {forces.shape[1]}")
        in_air = (torch.norm(forces, dim=-1) <= force_threshold).float()
        self._cumulative_air_time += in_air * env.step_dt

        cost = air_time_balance_cost(self._cumulative_air_time, min_total_air_time, balance_tolerance)

        _log(env, "Metrics/wheel_episode_air_time_left_mean", self._cumulative_air_time[:, 0].mean())
        _log(env, "Metrics/wheel_episode_air_time_right_mean", self._cumulative_air_time[:, 1].mean())
        _log(env, "Metrics/wheel_air_time_balance_cost_mean", cost.mean())
        return cost

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._cumulative_air_time[env_ids] = 0.0


# --------------------------------------------------------------------------------------
# Rough-terrain gated terms
# --------------------------------------------------------------------------------------


def rough_wheel_usage(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    roughness_sensor_names: tuple[str, ...],
    wheel_radius: float = 0.127,
    gate_min: float = 0.0,
    gate_max: float = 0.5,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Penalise spinning the wheels on rough ground.

    Wheel speed is the cheapest way to make progress and the wrong one over obstacles,
    where it just slips. Suppressing it where the terrain is rough is what forces the
    legs to do the work.
    """
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    asset = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return roughness_gate_active(gate, threshold) * torch.sum(torch.square(wheel_vel), dim=1)


def rough_wheel_foot_clearance(
    env: ManagerBasedRLEnv,
    clearance_sensor_names: tuple[str, ...],
    roughness_sensor_names: tuple[str, ...],
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str,
    wheel_radius: float = 0.127,
    gate_min: float = 0.0,
    gate_max: float = 0.5,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
    clearance_grid_shape: tuple[int, int] | None = None,
    base_target_height: float = 0.03,
    range_scale: float = 0.5,
    max_target_height: float = 0.18,
    target_std: float = 0.04,
    command_threshold: float = 0.05,
    max_distance: float = 1.0,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward lifting a swinging wheel to a clearance that suits the local terrain.

    The target is not a constant: it scales with how much height variation the wheel's
    own scan sees, so the robot steps just high enough for the obstacle in front of it
    rather than marching with a fixed, energetically wasteful gait.
    """
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )

    samples = stack_clearance_samples(env, clearance_sensor_names, max_distance)  # [B, F, N]
    if clearance_grid_shape is not None:
        rows, cols = clearance_grid_shape
        if rows * cols != samples.shape[-1]:
            raise ValueError(f"clearance_grid_shape={clearance_grid_shape} does not match {samples.shape[-1]} samples")
    local_height_range = samples.max(dim=-1).values - samples.min(dim=-1).values
    wheel_centre_clearance = samples.amin(dim=-1)
    wheel_foot_clearance = torch.clamp(wheel_centre_clearance - wheel_radius, min=0.0)

    contact_sensor = env.scene[contact_sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids]
    in_air = (torch.norm(forces, dim=-1) <= force_threshold).float()
    active = _command_active(env, command_name, command_threshold)

    target = torch.clamp(base_target_height + range_scale * local_height_range, max=max_target_height)
    reward_per_foot = torch.exp(-torch.square(wheel_foot_clearance - target) / target_std**2)
    reward = torch.sum(reward_per_foot * in_air, dim=1) * roughness_gate_active(gate, threshold) * active

    _log(env, "Metrics/rough_wheel_foot_clearance_mean", wheel_foot_clearance.mean())
    _log(env, "Metrics/rough_wheel_foot_clearance_target_mean", target.mean())
    return reward


def rough_contact_pattern(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    roughness_sensor_names: tuple[str, ...],
    command_name: str,
    wheel_radius: float = 0.127,
    gate_min: float = 0.0,
    gate_max: float = 0.5,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
    command_threshold: float = 0.05,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Push toward single support on rough ground by penalising the two static patterns.

    Both wheels down means the robot is not stepping; neither down means it is airborne.
    Penalising both leaves alternating single support as the only unpunished pattern,
    without prescribing a gait frequency or phase.
    """
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    sensor = env.scene[contact_sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids]
    in_contact = torch.norm(forces, dim=-1) > force_threshold
    contact_count = torch.sum(in_contact.float(), dim=1)
    num_contacts = in_contact.shape[1]

    double_contact = (contact_count == num_contacts).float()
    no_contact = (contact_count == 0).float()
    active = _command_active(env, command_name, command_threshold)
    reward = -(double_contact + no_contact) * roughness_gate_active(gate, threshold) * active

    _log(env, "Metrics/rough_double_contact_mean", double_contact.mean())
    _log(env, "Metrics/rough_no_contact_mean", no_contact.mean())
    _log(env, "Metrics/rough_single_contact_mean", (contact_count == 1).float().mean())
    return reward


def standing_forward_wheel_air_time(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    roughness_sensor_names: tuple[str, ...],
    command_name: str,
    wheel_radius: float = 0.127,
    gate_min: float = 0.0,
    gate_max: float = 0.5,
    roughness_gate_threshold: float = 0.2,
    roughness_gate_threshold_final: float | None = None,
    roughness_gate_threshold_ramp_steps: int = 0,
    grid_shape: tuple[int, int] | None = None,
    max_time: float = 0.5,
    air_time_offset: float = 0.05,
    standing_scale: float = 2.5,
    forward_scale: float = 2.0,
) -> torch.Tensor:
    """Penalise lifting a wheel when there is no reason to.

    Standing still is penalised everywhere -- a stationary robot has nothing to step
    over. Driving straight is penalised only on smooth ground, where rolling is
    strictly better than stepping; on rough ground the same motion is exactly what the
    clearance reward is asking for.
    """
    gate, threshold = _roughness_gate(
        env,
        roughness_sensor_names,
        wheel_radius,
        gate_min,
        gate_max,
        grid_shape,
        roughness_gate_threshold,
        roughness_gate_threshold_final,
        roughness_gate_threshold_ramp_steps,
    )
    sensor = env.scene[contact_sensor_cfg.name]
    current_air_time = sensor.data.current_air_time[:, contact_sensor_cfg.body_ids]
    # `current_air_time` is reset to zero on touchdown and accumulates while airborne, so
    # a strictly positive value is the definition of "in air" -- the same test Isaac Lab's
    # and mjlab's `compute_first_contact` use internally. Requires track_air_time=True.
    in_air = current_air_time > 0.0
    air_time = torch.sum((torch.clamp(current_air_time, max=max_time) + air_time_offset) * in_air.float(), dim=1)

    term = _command_term(env, command_name)
    standing = term.is_standing_env
    forward = term.is_forward_env & ~standing

    standing_cost = air_time * standing.float() * standing_scale
    forward_cost = air_time * forward.float() * forward_scale * roughness_gate_inactive(gate, threshold)
    cost = standing_cost + forward_cost

    _log(env, "Metrics/standing_wheel_air_time_mean", standing_cost.mean())
    _log(env, "Metrics/non_rough_forward_wheel_air_time_mean", forward_cost.mean())
    return cost


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
#
# mjlab's `mean_action_acc` diagnostic is deliberately NOT ported.
#
# It lived in mjlab's separate MetricsManager precisely because mjlab's RewardManager
# skips zero-weight terms. Isaac Lab has no metrics manager and its RewardManager skips
# zero-weight terms too (`if term_cfg.weight == 0.0: continue`), so registering the
# metric as a weight-0 reward would produce a term that is never called -- it would log
# nothing while looking like it was wired up.
#
# The same information is available from `Loss/*` and from the `action_rate_l2` reward,
# which is active. If a dedicated metric is wanted later, it needs a home that is
# evaluated unconditionally (an observation term, or a termination term that always
# returns False), not a zero-weight reward.
