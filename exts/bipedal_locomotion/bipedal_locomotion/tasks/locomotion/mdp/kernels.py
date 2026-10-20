"""Pure-tensor math shared by the WF-TRON1B MDP terms.

Nothing in this module imports ``isaaclab``. Every function takes plain tensors
and returns plain tensors, so the whole file is exercisable by ``pytest`` on a
machine with no Isaac Sim installed -- which is what makes it possible to check
this port's arithmetic against the mjlab original before ever launching a
simulator.

The manager-facing terms in the ``wheeled_*.py`` modules are thin adapters: they
pull tensors out of ``env.scene[...]`` and call in here.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

__all__ = [
    "CommandTrackingMonitor",
    "TerrainRoughnessStats",
    "air_time_balance_cost",
    "base_height_error_sq",
    "body_positions_in_base_frame",
    "contact_substep_count",
    "delay_step_bounds",
    "encode_contact_forces",
    "ground_height_quantile",
    "heights_from_raycast",
    "normalize_to_unit_range",
    "posture_reward",
    "posture_std",
    "resolve_grid_shape",
    "roughness_gate_active",
    "roughness_gate_inactive",
    "scheduled_gate_threshold",
    "smoothstep_gate",
    "terrain_normal_from_points",
    "terrain_roughness_from_height_samples",
    "track_angular_velocity_reward",
    "track_linear_velocity_reward",
    "wheel_distance_cost",
    "wheel_lateral_symmetry",
    "wheel_x_alignment",
]


# --------------------------------------------------------------------------------------
# Terrain roughness
# --------------------------------------------------------------------------------------


class TerrainRoughnessStats(NamedTuple):
    """Per-foot and whole-robot terrain roughness, plus the smoothstep gate."""

    jump: torch.Tensor
    """Largest first difference between adjacent height samples, per foot. [B, F]"""
    curvature: torch.Tensor
    """Largest second difference between adjacent height samples, per foot. [B, F]"""
    foot_roughness: torch.Tensor
    """``max(jump, curvature) / wheel_radius``, per foot. [B, F]"""
    robot_roughness: torch.Tensor
    """Worst per-foot roughness. [B]"""
    gate: torch.Tensor
    """Smoothstep of ``robot_roughness`` over ``[gate_min, gate_max]``, in [0, 1]. [B]"""


def resolve_grid_shape(num_samples: int, grid_shape: tuple[int, int] | None) -> tuple[int, int]:
    """Resolve a (rows, cols) height-sample grid, defaulting to a square."""
    if grid_shape is not None:
        rows, cols = grid_shape
        if rows * cols != num_samples:
            raise ValueError(f"grid_shape={grid_shape} does not match {num_samples} height samples")
        return rows, cols

    side = int(num_samples**0.5)
    if side * side != num_samples:
        raise ValueError(
            f"Cannot infer a square grid from {num_samples} height samples; pass grid_shape explicitly."
        )
    return side, side


def terrain_roughness_from_height_samples(
    height_samples: torch.Tensor,
    *,
    wheel_radius: float,
    gate_min: float,
    gate_max: float,
    grid_shape: tuple[int, int] | None,
) -> TerrainRoughnessStats:
    """Score how rough the ground under each wheel is, on a scale set by the wheel radius.

    A wheel rolls over an obstacle it can climb and has to be lifted over one it cannot,
    so the natural length scale for "rough" is the wheel radius. Both the largest step
    (first difference) and the largest ridge or trough (second difference) in the local
    height grid are measured and divided by that radius; the worse of the two defines
    the foot's roughness.

    The whole-robot roughness is the worst foot, passed through a smoothstep so that
    rewards gated on it fade in and out rather than switching, which would otherwise put
    a discontinuity in the reward exactly where the terrain transitions.

    Args:
        height_samples: Unreduced local clearance samples, shape ``[B, F, N]``.
        wheel_radius: Wheel radius in metres; the roughness scale.
        gate_min: Roughness at which the gate starts opening.
        gate_max: Roughness at which the gate is fully open.
        grid_shape: ``(rows, cols)`` layout of the ``N`` samples. Inferred as a square
            when ``None``.

    Returns:
        The per-foot and whole-robot statistics, plus the gate.
    """
    if height_samples.ndim != 3:
        raise ValueError(
            "terrain roughness requires unreduced height samples with shape [B, F, N], "
            f"got {tuple(height_samples.shape)}"
        )
    if wheel_radius <= 0.0:
        raise ValueError(f"wheel_radius ({wheel_radius}) must be positive")
    if gate_max <= gate_min:
        raise ValueError(f"gate_max ({gate_max}) must be greater than gate_min ({gate_min})")

    height_samples = torch.nan_to_num(height_samples, nan=0.0, posinf=0.0, neginf=0.0)
    rows, cols = resolve_grid_shape(height_samples.shape[-1], grid_shape)
    grid = height_samples.view(height_samples.shape[0], height_samples.shape[1], rows, cols)
    zeros = torch.zeros_like(height_samples[..., 0])

    if cols > 1:
        jump_x = torch.abs(grid[..., 1:] - grid[..., :-1]).amax(dim=(-1, -2))
    else:
        jump_x = zeros
    if rows > 1:
        jump_y = torch.abs(grid[..., 1:, :] - grid[..., :-1, :]).amax(dim=(-1, -2))
    else:
        jump_y = zeros
    jump = torch.maximum(jump_x, jump_y)

    if cols > 2:
        curvature_x = torch.abs(grid[..., :, 2:] - 2.0 * grid[..., :, 1:-1] + grid[..., :, :-2]).amax(dim=(-1, -2))
    else:
        curvature_x = zeros
    if rows > 2:
        curvature_y = torch.abs(grid[..., 2:, :] - 2.0 * grid[..., 1:-1, :] + grid[..., :-2, :]).amax(dim=(-1, -2))
    else:
        curvature_y = zeros
    curvature = torch.maximum(curvature_x, curvature_y)

    foot_roughness = torch.maximum(jump / wheel_radius, curvature / wheel_radius)
    robot_roughness = foot_roughness.max(dim=1).values
    gate = smoothstep_gate(robot_roughness, gate_min, gate_max)
    return TerrainRoughnessStats(
        jump=jump,
        curvature=curvature,
        foot_roughness=foot_roughness,
        robot_roughness=robot_roughness,
        gate=gate,
    )


def smoothstep_gate(roughness: torch.Tensor, gate_min: float, gate_max: float) -> torch.Tensor:
    """Smoothstep of ``roughness`` over ``[gate_min, gate_max]``, in [0, 1]."""
    u = torch.clamp((roughness - gate_min) / (gate_max - gate_min), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def roughness_gate_active(gate: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
    """1.0 where the terrain counts as rough."""
    return (gate > threshold).float()


def roughness_gate_inactive(gate: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
    """1.0 where the terrain counts as smooth."""
    return 1.0 - roughness_gate_active(gate, threshold)


def scheduled_gate_threshold(
    step: int,
    threshold: float,
    threshold_final: float | None,
    ramp_steps: int,
) -> tuple[float, float]:
    """Linearly ramp the roughness activation threshold over training.

    Starting at a low threshold makes almost all terrain count as rough, so the
    legged-motion rewards are active from the beginning and the robot learns to lift its
    wheels at all. Raising it later narrows "rough" to terrain that genuinely needs
    stepping, letting the wheeled-motion rewards take over elsewhere.

    Returns:
        ``(current_threshold, progress)`` where progress is in [0, 1].
    """
    if ramp_steps <= 0 or threshold_final is None:
        return threshold, 0.0
    progress = min(max(float(step) / ramp_steps, 0.0), 1.0)
    return threshold + progress * (threshold_final - threshold), progress


# --------------------------------------------------------------------------------------
# Height-scan geometry
# --------------------------------------------------------------------------------------


def heights_from_raycast(
    frame_z: torch.Tensor,
    hit_z: torch.Tensor,
    miss_mask: torch.Tensor,
    max_distance: float,
) -> torch.Tensor:
    """Convert ray hit positions into clearance below each sensor frame.

    Args:
        frame_z: World-frame z of the sensor origin, shape ``[B, F, 1]`` or ``[B, 1]``.
        hit_z: World-frame z of each ray hit, shape ``[B, F, N]``.
        miss_mask: True where the ray hit nothing, shape ``[B, F, N]``.
        max_distance: Clearance reported for a ray that hit nothing.

    Returns:
        Clearance ``frame_z - hit_z``, shape ``[B, F, N]``.

    A ray that misses would otherwise produce an infinite or garbage hit position. It is
    reported as ``max_distance`` -- "at least this far down" -- which keeps the roughness
    statistics finite and makes a missed ray look like a drop-off rather than a wall.
    """
    heights = frame_z - hit_z
    return torch.where(miss_mask, torch.full_like(heights, max_distance), heights)


def ground_height_quantile(hit_z: torch.Tensor, valid: torch.Tensor, quantile: float) -> torch.Tensor:
    """Quantile of the valid ray-hit heights per environment, 0 where no ray hit (mjlab ``base_height_l2``).

    Args:
        hit_z: World-frame z of each ray hit, shape ``[B, N]``.
        valid: True where the ray hit something, shape ``[B, N]``.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    ground_z = torch.nanquantile(torch.where(valid, hit_z, torch.full_like(hit_z, torch.nan)), quantile, dim=1)
    return torch.nan_to_num(ground_z, nan=0.0)


def terrain_normal_from_points(
    points: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fit a plane normal to a batch of world-frame points by eigendecomposition.

    Args:
        points: Ray hit positions, shape ``[B, N, 3]``.
        valid_mask: True where the point is a real hit, shape ``[B, N]``. Rays that hit
            nothing must be masked out rather than passed in with a substituted position:
            a missed ray replaced by the origin is an extreme outlier that would drag the
            fit by more than the terrain ever does.

    Returns:
        Unit normals oriented upward (positive z), shape ``[B, 3]``.

    Uses the eigenvector of the smallest eigenvalue of the centred covariance, not a
    least-squares ``z = ax + by + c`` fit. The regression form cannot represent a plane
    steeper than vertical and is badly conditioned well before that, which is exactly the
    regime -- stairs and steep slopes -- where this reward matters.

    Falls back to world up when the fit is not trustworthy: fewer than three valid points,
    a point cloud that is line-like rather than plane-like, or one with no spread at all.
    Returning up is the same answer flat ground gives, so a degenerate scan degrades to
    the world-frame definition instead of injecting a random normal.
    """
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"terrain_normal_from_points expects [B, N, 3], got {tuple(points.shape)}")

    batch = points.shape[0]
    up = torch.zeros(batch, 3, device=points.device, dtype=points.dtype)
    up[:, 2] = 1.0

    if valid_mask is None:
        valid_mask = torch.ones(points.shape[:2], device=points.device, dtype=torch.bool)
    # A non-finite coordinate would make the covariance NaN and poison the whole batch.
    valid_mask = valid_mask & torch.isfinite(points).all(dim=-1)

    count = valid_mask.sum(dim=1)
    enough = count >= 3

    mask_f = valid_mask.unsqueeze(-1).to(points.dtype)
    safe_points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    centroid = (safe_points * mask_f).sum(dim=1) / count.clamp(min=1).to(points.dtype).unsqueeze(-1)
    centered = (safe_points - centroid.unsqueeze(1)) * mask_f

    cov = torch.einsum("bni,bnj->bij", centered, centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    normal = eigenvectors[:, :, 0]  # smallest eigenvalue -> plane normal
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1.0e-8)
    normal = torch.where((normal[:, 2] < 0).unsqueeze(-1), -normal, normal)

    # A genuine plane has one small eigenvalue and two materially larger ones. A line-like
    # or point-like cloud does not, and its "normal" is arbitrary within a subspace.
    eps = torch.finfo(eigenvalues.dtype).eps
    plane_like = (eigenvalues[:, 0] / eigenvalues[:, 1].clamp(min=eps)) < 0.1
    has_spread = eigenvalues[:, 1] > eigenvalues[:, 2].clamp(min=eps) * 1.0e-6
    reliable = enough & plane_like & has_spread & torch.isfinite(normal).all(dim=-1)

    return torch.where(reliable.unsqueeze(-1), normal, up)


# --------------------------------------------------------------------------------------
# Wheel geometry
# --------------------------------------------------------------------------------------


def body_positions_in_base_frame(
    body_pos_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Express body positions in the base frame.

    Args:
        body_pos_w: World-frame body positions, shape ``[B, N, 3]``.
        base_pos_w: World-frame base position, shape ``[B, 3]``.
        base_quat_w: World-frame base orientation (w, x, y, z), shape ``[B, 4]``.

    Returns:
        Body positions in the base frame, shape ``[B, N, 3]``.
    """
    from bipedal_locomotion.utils.math import quat_apply_inverse

    offset = body_pos_w - base_pos_w[:, None, :]
    quat = base_quat_w[:, None, :].expand(-1, body_pos_w.shape[1], -1)
    return quat_apply_inverse(quat, offset)


def wheel_lateral_symmetry(wheel_pos_b: torch.Tensor, std: float) -> torch.Tensor:
    """Reward the two wheels for sitting equally far to either side of the base.

    Compares |y| rather than y, so the term measures stance symmetry and is blind to a
    sideways shift of both wheels together.

    Args:
        wheel_pos_b: Wheel positions in the base frame, shape ``[B, 2, 3]``.
        std: Tolerance on the lateral asymmetry, in metres.
    """
    lateral_error = torch.abs(wheel_pos_b[:, 0, 1]) - torch.abs(wheel_pos_b[:, 1, 1])
    return torch.exp(-torch.square(lateral_error) / std**2)


def wheel_x_alignment(wheel_pos_b: torch.Tensor) -> torch.Tensor:
    """Fore-aft offset between the two wheels, in metres. Use with a negative weight."""
    return torch.abs(wheel_pos_b[:, 0, 0] - wheel_pos_b[:, 1, 0])


def wheel_distance_cost(
    wheel_pos_b: torch.Tensor,
    min_distance: float,
    max_distance: float,
) -> torch.Tensor:
    """Penalise a wheel separation outside ``[min_distance, max_distance]``.

    Zero inside the band, then linear in the violation, so the term is silent for any
    stance width the robot is allowed to choose.

    Args:
        wheel_pos_b: Wheel positions in the base frame, shape ``[B, 2, 3]``.
    """
    distance = torch.norm(wheel_pos_b[:, 0, :2] - wheel_pos_b[:, 1, :2], dim=1)
    return torch.clip(min_distance - distance, min=0.0) + torch.clip(distance - max_distance, min=0.0)


# --------------------------------------------------------------------------------------
# Velocity tracking
# --------------------------------------------------------------------------------------


def track_linear_velocity_reward(
    command_xy_w: torch.Tensor,
    lin_vel_w: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Reward tracking the commanded linear velocity, in the **world** frame.

    Args:
        command_xy_w: Commanded world-frame planar velocity, shape ``[B, 2]``.
        lin_vel_w: Measured world-frame linear velocity of the base *link*, ``[B, 3]``.
        std: Tolerance in m/s.

    The command is held fixed in the world frame while the robot turns to face it, so
    both sides of the comparison must be world-frame. Comparing a body-frame command
    against a body-frame velocity would let the robot satisfy the term by yawing, which
    is exactly the behaviour the paired heading reward exists to prevent.

    The z velocity is included in the error with an implied command of zero.
    """
    xy_error = torch.sum(torch.square(command_xy_w - lin_vel_w[:, :2]), dim=1)
    z_error = torch.square(lin_vel_w[:, 2])
    return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity_reward(
    command_yaw_rate: torch.Tensor,
    ang_vel_b: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Reward tracking the commanded yaw rate.

    Args:
        command_yaw_rate: Commanded yaw rate, shape ``[B]``.
        ang_vel_b: Measured body-frame angular velocity, shape ``[B, 3]``.
        std: Tolerance in rad/s.
    """
    z_error = torch.square(command_yaw_rate - ang_vel_b[:, 2])
    return torch.exp(-z_error / std**2)


def base_height_error_sq(
    base_height: torch.Tensor,
    target_height: float,
    deadband: float,
) -> torch.Tensor:
    """Squared base-height error outside a symmetric deadband.

    The deadband keeps the (heavily weighted) height term from fighting the small
    vertical motion that walking over rough ground necessarily produces.
    """
    error = torch.clamp(torch.abs(base_height - target_height) - deadband, min=0.0)
    return torch.square(error)


# --------------------------------------------------------------------------------------
# Posture
# --------------------------------------------------------------------------------------


def posture_std(
    total_speed: torch.Tensor,
    std_standing: torch.Tensor,
    std_walking: torch.Tensor,
    std_running: torch.Tensor,
    walking_threshold: float,
    running_threshold: float,
) -> torch.Tensor:
    """Select a per-joint posture tolerance from the commanded speed.

    Args:
        total_speed: Commanded linear + angular speed, shape ``[B]``.
        std_standing / std_walking / std_running: Per-joint tolerances, shape ``[J]``.

    Returns:
        Per-environment, per-joint tolerance, shape ``[B, J]``.

    Holding a pose and running need different amounts of freedom, so the reference-pose
    tolerance widens with the commanded speed rather than the term being switched off.
    """
    standing = (total_speed < walking_threshold).float().unsqueeze(1)
    walking = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float().unsqueeze(1)
    running = (total_speed >= running_threshold).float().unsqueeze(1)
    return std_standing * standing + std_walking * walking + std_running * running


def posture_reward(
    joint_pos: torch.Tensor,
    target_joint_pos: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """``exp(-mean(error^2 / std^2))`` over the selected joints."""
    error_squared = torch.square(joint_pos - target_joint_pos)
    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


# --------------------------------------------------------------------------------------
# Contact / air time
# --------------------------------------------------------------------------------------


def encode_contact_forces(forces_b: torch.Tensor) -> torch.Tensor:
    """Flatten and log-compress body-frame contact forces for use as an observation.

    Args:
        forces_b: Body-frame contact forces, shape ``[B, N, 3]``.

    Returns:
        ``sign(f) * log1p(|f|)`` flattened to ``[B, N * 3]``.

    Contact forces span roughly three orders of magnitude between light ground contact
    and a hard landing. Feeding them in raw would let the impacts dominate the network's
    input scaling; the signed log keeps the sign and the ordering while compressing the
    range.
    """
    flat = torch.nan_to_num(forces_b, nan=0.0, posinf=0.0, neginf=0.0).flatten(start_dim=1)
    return torch.sign(flat) * torch.log1p(torch.abs(flat))


def contact_substep_count(force_history: torch.Tensor, threshold: float) -> torch.Tensor:
    """Number of history substeps in which any body's contact force exceeds ``threshold``.

    mjlab's ``self_collision_cost``; ``> 0`` is its ``illegal_contact`` termination.

    Args:
        force_history: Contact forces, shape ``[B, H, N, 3]``.
    """
    force = torch.nan_to_num(force_history, nan=0.0, posinf=0.0, neginf=0.0)
    return (torch.norm(force, dim=-1) > threshold).any(dim=-1).sum(dim=-1).float()


def air_time_balance_cost(
    cumulative_air_time: torch.Tensor,
    min_total_air_time: float,
    balance_tolerance: float,
) -> torch.Tensor:
    """Penalise an episode-level left/right imbalance in accumulated air time.

    Args:
        cumulative_air_time: Per-wheel air time accumulated this episode, ``[B, 2]``.
        min_total_air_time: Total air time at which the penalty reaches full strength.
        balance_tolerance: Imbalance ratio tolerated before any penalty applies, in [0, 1).

    A robot that props itself up on one wheel and hops on the other scores well on most
    per-step terms while being useless. Measuring the imbalance over the whole episode
    catches that, where a per-step contact term cannot.

    The tolerance keeps normal asymmetric gaits unpenalised, and the activation ramp
    stops the very first swing of an episode -- when the ratio is trivially 1.0 -- from
    registering as a violation.
    """
    if min_total_air_time <= 0.0:
        raise ValueError("min_total_air_time must be positive")
    if not 0.0 <= balance_tolerance < 1.0:
        raise ValueError("balance_tolerance must be in [0, 1)")

    total_air_time = cumulative_air_time.sum(dim=1)
    air_time_diff = torch.abs(cumulative_air_time[:, 0] - cumulative_air_time[:, 1])
    imbalance_ratio = air_time_diff / torch.clamp(total_air_time, min=1.0e-6)

    excess_ratio = torch.clamp(
        (imbalance_ratio - balance_tolerance) / (1.0 - balance_tolerance),
        min=0.0,
        max=1.0,
    )
    activation = torch.clamp(total_air_time / min_total_air_time, 0.0, 1.0)
    return activation * torch.square(excess_ratio)


# --------------------------------------------------------------------------------------
# Domain randomisation / actuation delay
# --------------------------------------------------------------------------------------


def normalize_to_unit_range(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Map ``[lower, upper]`` onto ``[-1, 1]``, clamped."""
    scaled = 2.0 * (value - lower) / (upper - lower) - 1.0
    return torch.clamp(scaled, -1.0, 1.0)


def delay_step_bounds(delay_range_s: tuple[float, float], physics_dt: float) -> tuple[int, int]:
    """Convert a delay range in seconds to inclusive bounds in physics steps.

    Only delays that land on a physics step are representable, so the bounds are rounded
    inward: ``ceil`` for the minimum, ``floor`` for the maximum. A range that contains no
    representable step is an error rather than being silently widened -- otherwise a
    typo in the config would quietly change the delay distribution.
    """
    delay_min_s, delay_max_s = delay_range_s
    if delay_min_s < 0.0 or delay_max_s < 0.0:
        raise ValueError(f"delay_range_s must be non-negative: {delay_range_s}")
    if delay_min_s > delay_max_s:
        raise ValueError(f"delay_range_s must be ordered min <= max: {delay_range_s}")
    if physics_dt <= 0.0:
        raise ValueError(f"physics_dt must be positive, got {physics_dt}")

    tolerance = max(abs(physics_dt) * 1.0e-9, 1.0e-12)
    min_steps = max(0, math.ceil((delay_min_s - tolerance) / physics_dt))
    max_steps = math.floor((delay_max_s + tolerance) / physics_dt)
    if min_steps > max_steps:
        raise ValueError(
            "delay_range_s must include at least one representable physics-step delay, "
            f"got delay_range_s={delay_range_s} and physics_dt={physics_dt}."
        )
    return min_steps, max_steps


# --------------------------------------------------------------------------------------
# Command tracking failure
# --------------------------------------------------------------------------------------


class CommandTrackingMonitor:
    """Per-environment counters of mjlab's ``world_command_tracking_failure`` termination.

    Terminates when world-frame progress, velocity direction or heading stays bad for long
    enough. Checks start after a grace period following each new command and are skipped while
    both wheels are airborne.
    """

    def __init__(self, num_envs: int, device) -> None:
        self.progress_bad_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.direction_bad_steps = torch.zeros_like(self.progress_bad_steps)
        self.heading_bad_steps = torch.zeros_like(self.progress_bad_steps)
        self.command_age_steps = torch.zeros_like(self.progress_bad_steps)
        self.last_command_counter = torch.full_like(self.progress_bad_steps, -1)

    def update(
        self,
        *,
        command_counter: torch.Tensor,
        command_xy: torch.Tensor,
        actual_xy: torch.Tensor,
        heading_error: torch.Tensor,
        grounded: torch.Tensor,
        is_heading_env: torch.Tensor,
        is_standing_env: torch.Tensor,
        enabled: bool,
        step_dt: float,
        command_norm_threshold: float,
        progress_deficit_threshold: float,
        min_progress_ratio: float,
        progress_duration_s: float,
        actual_speed_threshold: float,
        direction_angle_threshold_deg: float,
        direction_duration_s: float,
        heading_error_threshold_deg: float,
        heading_duration_s: float,
        heading_alignment_gate_deg: float,
        command_grace_s: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Advance one policy step. Returns the termination flags and logging metrics.

        Args:
            command_xy: World-frame linear command, ``[B, 2]``.
            actual_xy: World-frame base linear velocity, ``[B, 2]``.
            heading_error: Wrapped heading-target minus heading, ``[B]``.
            grounded: True where at least one wheel touches the terrain, ``[B]``.
            enabled: Whether the training step has reached the activation step.
        """
        command_changed = command_counter != self.last_command_counter
        self.command_age_steps = torch.where(command_changed, 0, self.command_age_steps + 1)
        self.last_command_counter = command_counter.clone()

        command_norm = torch.norm(command_xy, dim=1)
        actual_speed = torch.norm(actual_xy, dim=1)
        command_direction = command_xy / command_norm.unsqueeze(1).clamp_min(1.0e-6)
        progress_velocity = torch.sum(actual_xy * command_direction, dim=1)
        progress_deficit = torch.clamp(command_norm - progress_velocity, min=0.0)
        progress_ratio = progress_velocity / command_norm.clamp_min(1.0e-6)
        cos_angle = torch.clamp(progress_velocity / (actual_speed + 1.0e-6), -1.0, 1.0)
        heading_error = torch.abs(heading_error)

        command_active = (
            enabled
            & (command_norm > command_norm_threshold)
            & (self.command_age_steps >= max(0, math.ceil(command_grace_s / step_dt)))
        )
        heading_aligned = heading_error < math.radians(heading_alignment_gate_deg)
        linear_check_active = command_active & grounded & heading_aligned
        heading_check_active = command_active & grounded & is_heading_env & ~is_standing_env

        bad_progress = linear_check_active & (
            (progress_deficit > progress_deficit_threshold) | (progress_ratio < min_progress_ratio)
        )
        bad_direction = (
            linear_check_active
            & (actual_speed > actual_speed_threshold)
            & (cos_angle < math.cos(math.radians(direction_angle_threshold_deg)))
        )
        bad_heading = heading_check_active & (heading_error > math.radians(heading_error_threshold_deg))
        self.progress_bad_steps = torch.where(bad_progress, self.progress_bad_steps + 1, 0)
        self.direction_bad_steps = torch.where(bad_direction, self.direction_bad_steps + 1, 0)
        self.heading_bad_steps = torch.where(bad_heading, self.heading_bad_steps + 1, 0)

        done = (
            (self.progress_bad_steps >= max(1, math.ceil(progress_duration_s / step_dt)))
            | (self.direction_bad_steps >= max(1, math.ceil(direction_duration_s / step_dt)))
            | (self.heading_bad_steps >= max(1, math.ceil(heading_duration_s / step_dt)))
        )
        metrics = {
            "progress_deficit_mean": progress_deficit.mean(),
            "progress_bad_frac": bad_progress.float().mean(),
            "direction_angle_mean": torch.rad2deg(torch.acos(cos_angle)).mean(),
            "direction_bad_frac": bad_direction.float().mean(),
            "heading_error_deg_mean": torch.rad2deg(heading_error).mean(),
            "heading_bad_frac": bad_heading.float().mean(),
            "airborne_frac": (~grounded).float().mean(),
            "active": torch.tensor(float(enabled), device=command_xy.device),
        }
        return done, metrics

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        self.progress_bad_steps[env_ids] = 0
        self.direction_bad_steps[env_ids] = 0
        self.heading_bad_steps[env_ids] = 0
        self.command_age_steps[env_ids] = 0
        self.last_command_counter[env_ids] = -1
