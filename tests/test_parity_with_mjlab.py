"""Differential test against the mjlab reference implementations.

The kernels in ``mdp/kernels.py`` were ported from the mjlab task package. Unit tests
pin their *behaviour*, but behaviour tests are written from the same understanding that
produced the port, so they cannot catch a shared misreading of the original.

This module transcribes the reference implementations verbatim and asserts the ported
kernels agree on randomised inputs. The transcriptions are byte-for-byte copies of the
cited functions with only the mjlab-specific type annotations dropped; a reviewer should
diff them against the cited source rather than trust this docstring.

Reference (sibling repository ``wheeled_legged_mjlab``):
    src/wheeled_legged_mjlab/tasks/velocity/mdp/rewards.py
    src/wheeled_legged_mjlab/tasks/velocity/mdp/observations.py
"""

import math

import pytest
import torch

from _loader import kernels

# ======================================================================================
# Verbatim transcriptions of the mjlab reference implementations.
# ======================================================================================


def _ref_resolve_grid_shape(num_samples, grid_shape):
    """rewards.py::_resolve_grid_shape"""
    if grid_shape is not None:
        rows, cols = grid_shape
        if rows * cols != num_samples:
            raise ValueError(f"grid_shape={grid_shape} does not match {num_samples} height samples")
        return rows, cols

    side = int(num_samples**0.5)
    if side * side != num_samples:
        raise ValueError(f"Cannot infer a square grid from {num_samples} height samples; pass grid_shape explicitly.")
    return side, side


def _ref_terrain_roughness_from_height_samples(height_samples, *, wheel_radius, gate_min, gate_max, grid_shape):
    """rewards.py::_terrain_roughness_from_height_samples"""
    height_samples = torch.nan_to_num(height_samples, nan=0.0, posinf=0.0, neginf=0.0)
    rows, cols = _ref_resolve_grid_shape(height_samples.shape[-1], grid_shape)
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
    u = torch.clamp((robot_roughness - gate_min) / (gate_max - gate_min), 0.0, 1.0)
    gate = u * u * (3.0 - 2.0 * u)
    return jump, curvature, foot_roughness, robot_roughness, gate


def _ref_scheduled_roughness_gate_threshold(step, threshold, threshold_final, ramp_steps):
    """rewards.py::_scheduled_roughness_gate_threshold (logging side effects removed)"""
    if ramp_steps <= 0 or threshold_final is None:
        current = threshold
        progress = 0.0
    else:
        progress = min(max(float(step) / ramp_steps, 0.0), 1.0)
        current = threshold + progress * (threshold_final - threshold)
    return current, progress


def _ref_encode_contact_forces(forces_b):
    """observations.py::foot_contact_forces, from the body-frame rotation onward"""
    forces_flat = forces_b.flatten(start_dim=1)
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def _ref_wheel_distance(wheel_pos_b, min_distance, max_distance):
    """rewards.py::wheel_distance"""
    distance = torch.norm(wheel_pos_b[:, 0, :2] - wheel_pos_b[:, 1, :2], dim=1)
    return torch.clip(min_distance - distance, min=0.0) + torch.clip(distance - max_distance, min=0.0)


def _ref_wheel_lateral_symmetry(wheel_pos_b, std):
    """rewards.py::wheel_lateral_symmetry"""
    lateral_error = torch.abs(wheel_pos_b[:, 0, 1]) - torch.abs(wheel_pos_b[:, 1, 1])
    return torch.exp(-torch.square(lateral_error) / std**2)


def _ref_wheel_x_alignment(wheel_pos_b):
    """rewards.py::wheel_x_alignment"""
    return torch.abs(wheel_pos_b[:, 0, 0] - wheel_pos_b[:, 1, 0])


def _ref_base_height_l2(base_height, target_height, deadband):
    """rewards.py::base_height_l2, error computation only"""
    height_error = torch.clamp(torch.abs(base_height - target_height) - deadband, min=0.0)
    return torch.square(height_error)


def _ref_track_linear_velocity(command_xy, actual, std):
    """rewards.py::track_linear_velocity, UniformVelocityCommand branch"""
    xy_error = torch.sum(torch.square(command_xy - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + z_error
    return torch.exp(-lin_vel_error / std**2)


def _ref_track_angular_velocity(command, actual, std):
    """rewards.py::track_angular_velocity"""
    z_error = torch.square(command - actual[:, 2])
    return torch.exp(-z_error / std**2)


def _ref_air_time_balance(cumulative_air_time, min_total_air_time, balance_tolerance):
    """rewards.py::wheel_air_time_balance.__call__, cost computation only"""
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


def _ref_fit_terrain_normal(points, valid_mask):
    """terrain_utils.py::fit_terrain_normal"""
    B = points.shape[0]
    device = points.device

    count = valid_mask.sum(dim=1)
    enough = count >= 3

    mask_f = valid_mask.float().unsqueeze(-1)
    masked_points = points * mask_f
    count_clamped = count.clamp(min=1).float().unsqueeze(-1)
    centroid = masked_points.sum(dim=1) / count_clamped
    centered = (points - centroid.unsqueeze(1)) * mask_f

    cov = torch.einsum("bni,bnj->bij", centered, centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    normal = eigenvectors[:, :, 0]
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normal = torch.where((normal[:, 2] < 0).unsqueeze(-1), -normal, normal)

    eps = torch.finfo(eigenvalues.dtype).eps
    plane_like = (eigenvalues[:, 0] / eigenvalues[:, 1].clamp(min=eps)) < 0.1
    has_spread = eigenvalues[:, 1] > eigenvalues[:, 2].clamp(min=eps) * 1e-6
    reliable = enough & plane_like & has_spread

    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, 3)
    return torch.where(reliable.unsqueeze(-1), normal, up)


def _ref_normalize_to_unit_range(value, lower, upper):
    """observations.py::_normalize_to_unit_range"""
    scaled = 2.0 * (value - lower) / (upper - lower) - 1.0
    return torch.clamp(scaled, -1.0, 1.0)


def _ref_delay_step_bounds(delay_range_s, physics_dt):
    """actions.py::_delay_step_bounds"""
    delay_min_s, delay_max_s = delay_range_s
    tolerance = max(abs(physics_dt) * 1.0e-9, 1.0e-12)
    min_steps = max(0, math.ceil((delay_min_s - tolerance) / physics_dt))
    max_steps = math.floor((delay_max_s + tolerance) / physics_dt)
    if min_steps > max_steps:
        raise ValueError("delay_range_s must include at least one representable physics-step delay")
    return min_steps, max_steps


# ======================================================================================
# Parity assertions
# ======================================================================================

ATOL = 1.0e-6


@pytest.fixture(autouse=True)
def _seeded():
    torch.manual_seed(20260731)


class TestRoughnessParity:
    @pytest.mark.parametrize("grid_shape", [(5, 5), (11, 11)])
    def test_matches_on_random_terrain(self, grid_shape):
        rows, cols = grid_shape
        samples = torch.randn(64, 2, rows * cols) * 0.05 + 0.4

        ref = _ref_terrain_roughness_from_height_samples(
            samples, wheel_radius=0.127, gate_min=0.0, gate_max=0.5, grid_shape=grid_shape
        )
        got = kernels.terrain_roughness_from_height_samples(
            samples, wheel_radius=0.127, gate_min=0.0, gate_max=0.5, grid_shape=grid_shape
        )

        assert torch.allclose(got.jump, ref[0], atol=ATOL)
        assert torch.allclose(got.curvature, ref[1], atol=ATOL)
        assert torch.allclose(got.foot_roughness, ref[2], atol=ATOL)
        assert torch.allclose(got.robot_roughness, ref[3], atol=ATOL)
        assert torch.allclose(got.gate, ref[4], atol=ATOL)

    def test_matches_on_structured_obstacles(self):
        """Steps and ridges, where jump and curvature disagree, are the interesting case."""
        samples = torch.zeros(8, 2, 25)
        samples[0, :, 10:] = 0.06  # step
        samples[1, :, 12] = 0.03  # single-cell ridge
        samples[2, :, :] = torch.linspace(0.0, 0.2, 25)  # ramp
        samples[3, :, ::2] = 0.05  # comb
        samples[4:, :, :] = torch.rand(4, 2, 25) * 0.1

        ref = _ref_terrain_roughness_from_height_samples(
            samples, wheel_radius=0.127, gate_min=0.0, gate_max=0.5, grid_shape=(5, 5)
        )
        got = kernels.terrain_roughness_from_height_samples(
            samples, wheel_radius=0.127, gate_min=0.0, gate_max=0.5, grid_shape=(5, 5)
        )
        assert torch.allclose(got.gate, ref[4], atol=ATOL)
        assert torch.allclose(got.foot_roughness, ref[2], atol=ATOL)

    @pytest.mark.parametrize("step", [0, 1, 5_000, 60_000, 120_000, 500_000])
    def test_scheduled_threshold_matches(self, step):
        ref = _ref_scheduled_roughness_gate_threshold(step, 0.0, 0.25, 5_000 * 24)
        got = kernels.scheduled_gate_threshold(step, 0.0, 0.25, 5_000 * 24)
        assert got[0] == pytest.approx(ref[0], abs=ATOL)
        assert got[1] == pytest.approx(ref[1], abs=ATOL)


class TestRewardKernelParity:
    def test_contact_force_encoding_matches(self):
        forces_b = torch.randn(64, 2, 3) * 50.0
        assert torch.allclose(kernels.encode_contact_forces(forces_b), _ref_encode_contact_forces(forces_b), atol=ATOL)

    def test_wheel_geometry_matches(self):
        wheel_pos_b = torch.randn(64, 2, 3) * 0.3
        assert torch.allclose(
            kernels.wheel_distance_cost(wheel_pos_b, 0.28, 0.55),
            _ref_wheel_distance(wheel_pos_b, 0.28, 0.55),
            atol=ATOL,
        )
        std = math.sqrt(0.5)
        assert torch.allclose(
            kernels.wheel_lateral_symmetry(wheel_pos_b, std),
            _ref_wheel_lateral_symmetry(wheel_pos_b, std),
            atol=ATOL,
        )
        assert torch.allclose(
            kernels.wheel_x_alignment(wheel_pos_b), _ref_wheel_x_alignment(wheel_pos_b), atol=ATOL
        )

    def test_base_height_error_matches(self):
        base_height = torch.rand(64) * 0.6 + 0.5
        assert torch.allclose(
            kernels.base_height_error_sq(base_height, 0.82, 0.04),
            _ref_base_height_l2(base_height, 0.82, 0.04),
            atol=ATOL,
        )

    def test_velocity_tracking_matches(self):
        command_xy = torch.randn(64, 2)
        lin_vel_w = torch.randn(64, 3)
        std = math.sqrt(0.25)
        assert torch.allclose(
            kernels.track_linear_velocity_reward(command_xy, lin_vel_w, std),
            _ref_track_linear_velocity(command_xy, lin_vel_w, std),
            atol=ATOL,
        )

        command_yaw = torch.randn(64)
        ang_vel_b = torch.randn(64, 3)
        assert torch.allclose(
            kernels.track_angular_velocity_reward(command_yaw, ang_vel_b, std),
            _ref_track_angular_velocity(command_yaw, ang_vel_b, std),
            atol=ATOL,
        )

    def test_air_time_balance_matches(self):
        cumulative = torch.rand(64, 2) * 5.0
        assert torch.allclose(
            kernels.air_time_balance_cost(cumulative, 1.0, 0.2),
            _ref_air_time_balance(cumulative, 1.0, 0.2),
            atol=ATOL,
        )


class TestTerrainNormalParity:
    """The plane fit is the one kernel where the port originally used a different estimator."""

    def _planes(self):
        """A batch mixing flat, gentle, steep and degenerate point clouds."""
        xs = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5, 0.25, 0.75, 0.5])
        ys = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5, 0.75, 0.25, 0.0])
        clouds, masks = [], []
        for slope in (0.0, 0.2, 0.5, 1.0, 2.75, -0.6):
            clouds.append(torch.stack([xs, ys, slope * xs], dim=-1))
            masks.append(torch.ones(len(xs), dtype=torch.bool))
        # Collinear -> must fall back to up in both implementations.
        line = torch.stack([xs, torch.zeros_like(xs), xs], dim=-1)
        clouds.append(line)
        masks.append(torch.ones(len(xs), dtype=torch.bool))
        return torch.stack(clouds), torch.stack(masks)

    def test_matches_on_planes_of_varying_steepness(self):
        points, valid = self._planes()
        assert torch.allclose(
            kernels.terrain_normal_from_points(points, valid),
            _ref_fit_terrain_normal(points, valid),
            atol=1e-5,
        )

    def test_matches_with_partial_masks(self):
        points, valid = self._planes()
        valid = valid.clone()
        valid[:, -2:] = False  # drop two rays per cloud, as a miss would
        assert torch.allclose(
            kernels.terrain_normal_from_points(points, valid),
            _ref_fit_terrain_normal(points, valid),
            atol=1e-5,
        )

    def test_matches_on_random_clouds(self):
        points = torch.randn(32, 24, 3)
        valid = torch.rand(32, 24) > 0.2
        assert torch.allclose(
            kernels.terrain_normal_from_points(points, valid),
            _ref_fit_terrain_normal(points, valid),
            atol=1e-5,
        )


class TestDomainRandomizationParity:
    def test_normalization_matches(self):
        value = torch.randn(64) * 2.0
        assert torch.allclose(
            kernels.normalize_to_unit_range(value, 0.3, 1.2),
            _ref_normalize_to_unit_range(value, 0.3, 1.2),
            atol=ATOL,
        )

    @pytest.mark.parametrize(
        "delay_range, dt",
        [((0.0, 0.02), 0.005), ((0.007, 0.018), 0.005), ((0.01, 0.01), 0.005), ((0.0, 0.0), 0.005), ((0.015, 0.015), 0.005)],
    )
    def test_delay_step_bounds_match(self, delay_range, dt):
        assert kernels.delay_step_bounds(delay_range, dt) == _ref_delay_step_bounds(delay_range, dt)
