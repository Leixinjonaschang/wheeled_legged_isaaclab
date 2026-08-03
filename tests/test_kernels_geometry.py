"""Wheel geometry, height-scan semantics, velocity tracking and posture kernels."""

import math

import pytest
import torch

from _loader import kernels

YAW_90 = torch.tensor([[math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]])
IDENTITY = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


class TestBodyPositionsInBaseFrame:
    def test_identity_orientation_is_a_pure_translation(self):
        body_pos_w = torch.tensor([[[1.0, 0.2, 0.0], [1.0, -0.2, 0.0]]])
        base_pos_w = torch.tensor([[1.0, 0.0, 0.8]])
        out = kernels.body_positions_in_base_frame(body_pos_w, base_pos_w, IDENTITY)
        expected = torch.tensor([[[0.0, 0.2, -0.8], [0.0, -0.2, -0.8]]])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_base_yaw_is_removed(self):
        """A robot yawed +90 degrees sees its world +y offset as body +x."""
        body_pos_w = torch.tensor([[[0.0, 0.3, 0.0]]])
        base_pos_w = torch.tensor([[0.0, 0.0, 0.0]])
        out = kernels.body_positions_in_base_frame(body_pos_w, base_pos_w, YAW_90)
        assert torch.allclose(out, torch.tensor([[[0.3, 0.0, 0.0]]]), atol=1e-6)


class TestWheelDistanceCost:
    MIN, MAX = 0.28, 0.55

    def _cost(self, separation):
        wheel_pos_b = torch.tensor([[[0.0, separation / 2.0, 0.0], [0.0, -separation / 2.0, 0.0]]])
        return kernels.wheel_distance_cost(wheel_pos_b, self.MIN, self.MAX).item()

    @pytest.mark.parametrize("separation", [0.28, 0.35, 0.425, 0.55])
    def test_no_cost_inside_the_allowed_band(self, separation):
        assert self._cost(separation) == pytest.approx(0.0, abs=1e-6)

    def test_cost_is_linear_in_the_violation_when_too_narrow(self):
        assert self._cost(0.20) == pytest.approx(self.MIN - 0.20, abs=1e-6)

    def test_cost_is_linear_in_the_violation_when_too_wide(self):
        assert self._cost(0.70) == pytest.approx(0.70 - self.MAX, abs=1e-6)

    def test_only_the_horizontal_separation_counts(self):
        """Lifting one wheel must not read as a stance-width violation."""
        flat = torch.tensor([[[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]])
        lifted = torch.tensor([[[0.0, 0.2, 0.15], [0.0, -0.2, 0.0]]])
        assert kernels.wheel_distance_cost(flat, self.MIN, self.MAX).item() == pytest.approx(
            kernels.wheel_distance_cost(lifted, self.MIN, self.MAX).item(), abs=1e-6
        )


class TestWheelSymmetry:
    def test_symmetric_stance_scores_one(self):
        wheel_pos_b = torch.tensor([[[0.0, 0.21, 0.0], [0.0, -0.21, 0.0]]])
        assert kernels.wheel_lateral_symmetry(wheel_pos_b, std=math.sqrt(0.5)).item() == pytest.approx(1.0, abs=1e-6)

    def test_a_common_lateral_shift_is_not_penalised(self):
        """The term measures stance symmetry, not where the stance sits."""
        centred = torch.tensor([[[0.0, 0.21, 0.0], [0.0, -0.21, 0.0]]])
        shifted = torch.tensor([[[0.0, 0.31, 0.0], [0.0, -0.31, 0.0]]])
        std = math.sqrt(0.5)
        assert kernels.wheel_lateral_symmetry(centred, std).item() == pytest.approx(
            kernels.wheel_lateral_symmetry(shifted, std).item(), abs=1e-6
        )

    def test_asymmetric_stance_scores_below_one(self):
        wheel_pos_b = torch.tensor([[[0.0, 0.30, 0.0], [0.0, -0.10, 0.0]]])
        std = math.sqrt(0.5)
        expected = math.exp(-((0.30 - 0.10) ** 2) / 0.5)
        assert kernels.wheel_lateral_symmetry(wheel_pos_b, std).item() == pytest.approx(expected, abs=1e-6)

    def test_x_alignment_is_the_absolute_fore_aft_offset(self):
        wheel_pos_b = torch.tensor([[[0.05, 0.2, 0.0], [-0.03, -0.2, 0.0]]])
        assert kernels.wheel_x_alignment(wheel_pos_b).item() == pytest.approx(0.08, abs=1e-6)


class TestHeightsFromRaycast:
    def test_clearance_is_frame_height_minus_hit_height(self):
        frame_z = torch.tensor([[[0.8]]])
        hit_z = torch.tensor([[[0.0, 0.1, -0.2]]])
        miss = torch.zeros(1, 1, 3, dtype=torch.bool)
        out = kernels.heights_from_raycast(frame_z, hit_z, miss, max_distance=1.0)
        assert torch.allclose(out, torch.tensor([[[0.8, 0.7, 1.0]]]), atol=1e-6)

    def test_missed_rays_report_max_distance(self):
        frame_z = torch.tensor([[[0.8]]])
        hit_z = torch.tensor([[[float("inf"), 0.1]]])
        miss = torch.tensor([[[True, False]]])
        out = kernels.heights_from_raycast(frame_z, hit_z, miss, max_distance=1.0)
        assert out[0, 0, 0].item() == pytest.approx(1.0)
        assert out[0, 0, 1].item() == pytest.approx(0.7)
        assert torch.isfinite(out).all()


class TestTerrainNormal:
    def test_flat_ground_gives_world_up(self):
        points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])
        normal = kernels.terrain_normal_from_points(points)
        assert torch.allclose(normal, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-5)

    def test_a_slope_tilts_the_normal(self):
        """z = 0.5x has normal proportional to (-0.5, 0, 1)."""
        xs = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5])
        ys = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5])
        zs = 0.5 * xs
        points = torch.stack([xs, ys, zs], dim=-1).unsqueeze(0)
        normal = kernels.terrain_normal_from_points(points)
        expected = torch.tensor([-0.5, 0.0, 1.0])
        expected = expected / torch.linalg.norm(expected)
        assert torch.allclose(normal[0], expected, atol=1e-5)

    def test_too_few_points_falls_back_to_up(self):
        points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]])
        assert torch.allclose(kernels.terrain_normal_from_points(points), torch.tensor([[0.0, 0.0, 1.0]]))

    def test_collinear_points_fall_back_to_up_instead_of_nan(self):
        xs = torch.linspace(0.0, 1.0, 5)
        points = torch.stack([xs, torch.zeros_like(xs), xs], dim=-1).unsqueeze(0)
        normal = kernels.terrain_normal_from_points(points)
        assert torch.isfinite(normal).all()
        assert torch.allclose(normal, torch.tensor([[0.0, 0.0, 1.0]]))

    def test_normals_are_unit_length(self):
        torch.manual_seed(0)
        points = torch.randn(4, 16, 3)
        normal = kernels.terrain_normal_from_points(points)
        assert torch.allclose(torch.linalg.norm(normal, dim=-1), torch.ones(4), atol=1e-5)

    def test_normal_is_always_oriented_upward(self):
        """The eigenvector's sign is arbitrary; the reward needs a consistent hemisphere."""
        xs = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5])
        ys = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5])
        for slope in (-0.8, -0.3, 0.0, 0.3, 0.8):
            points = torch.stack([xs, ys, slope * xs], dim=-1).unsqueeze(0)
            normal = kernels.terrain_normal_from_points(points)
            assert normal[0, 2].item() > 0.0, f"normal points downward for slope {slope}"

    def test_masked_out_rays_do_not_influence_the_fit(self):
        """A missed ray must be excluded, not substituted.

        Isaac Lab reports a miss as an infinite hit position. Replacing that with the
        origin -- or letting it through -- puts a point tens of metres from the terrain
        into the fit and tilts the normal far more than the ground ever does.
        """
        # A robot standing on a slope 10 m from the world origin. The last entry stands in
        # for a missed ray that a naive nan_to_num would have placed at the origin.
        # Real points lie on z = 0.5x + 2; the substituted origin does not, and sits 10 m
        # away from the rest of the cloud.
        xs = torch.tensor([10.0, 11.0, 10.0, 11.0, 10.5, 0.0])
        ys = torch.tensor([5.0, 5.0, 6.0, 6.0, 5.5, 0.0])
        zs = torch.tensor([7.0, 7.5, 7.0, 7.5, 7.25, 0.0])
        points = torch.stack([xs, ys, zs], dim=-1).unsqueeze(0)

        expected = torch.tensor([-0.5, 0.0, 1.0])
        expected = expected / torch.linalg.norm(expected)

        valid = torch.tensor([[True, True, True, True, True, False]])
        masked = kernels.terrain_normal_from_points(points, valid)
        assert torch.allclose(masked[0], expected, atol=1e-5)

        # Letting the substituted point through changes the answer materially.
        unmasked = kernels.terrain_normal_from_points(points)
        angle_error = torch.arccos(torch.clamp(torch.dot(unmasked[0], expected), -1.0, 1.0))
        assert angle_error.item() > math.radians(5.0), (
            f"the bogus point shifted the normal by only {math.degrees(angle_error.item()):.2f} deg; "
            "pick a construction where masking demonstrably matters"
        )

    def test_infinite_hits_are_rejected_even_without_an_explicit_mask(self):
        xs = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5, 0.2])
        ys = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5, 0.7])
        zs = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, float("inf")])
        points = torch.stack([xs, ys, zs], dim=-1).unsqueeze(0)
        normal = kernels.terrain_normal_from_points(points)
        assert torch.isfinite(normal).all()
        assert torch.allclose(normal[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)

    def test_too_few_valid_points_falls_back_to_up(self):
        points = torch.randn(1, 8, 3)
        valid = torch.zeros(1, 8, dtype=torch.bool)
        valid[0, :2] = True
        assert torch.allclose(kernels.terrain_normal_from_points(points, valid), torch.tensor([[0.0, 0.0, 1.0]]))

    def test_a_steep_plane_is_handled(self):
        """A 70-degree slope is where a z = ax + by + c regression falls apart."""
        angle = math.radians(70.0)
        xs = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5])
        ys = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5])
        zs = math.tan(angle) * xs
        points = torch.stack([xs, ys, zs], dim=-1).unsqueeze(0)

        normal = kernels.terrain_normal_from_points(points)
        expected = torch.tensor([-math.sin(angle), 0.0, math.cos(angle)])
        assert torch.allclose(normal[0], expected, atol=1e-4)


class TestVelocityTracking:
    def test_perfect_tracking_scores_one(self):
        command = torch.tensor([[1.0, 0.0]])
        lin_vel_w = torch.tensor([[1.0, 0.0, 0.0]])
        assert kernels.track_linear_velocity_reward(command, lin_vel_w, std=0.5).item() == pytest.approx(1.0, abs=1e-6)

    def test_vertical_velocity_is_penalised_with_an_implied_zero_command(self):
        command = torch.tensor([[1.0, 0.0]])
        still = torch.tensor([[1.0, 0.0, 0.0]])
        bouncing = torch.tensor([[1.0, 0.0, 0.4]])
        std = 0.5

        still_reward = kernels.track_linear_velocity_reward(command, still, std).item()
        bouncing_reward = kernels.track_linear_velocity_reward(command, bouncing, std).item()

        assert still_reward == pytest.approx(1.0, abs=1e-6)
        assert bouncing_reward < still_reward
        assert bouncing_reward == pytest.approx(math.exp(-(0.4**2) / std**2), abs=1e-6)

    def test_error_uses_the_world_frame_command_verbatim(self):
        command = torch.tensor([[0.0, 1.0]])
        lin_vel_w = torch.tensor([[0.0, 0.5, 0.0]])
        std = math.sqrt(0.25)
        expected = math.exp(-(0.5**2) / 0.25)
        assert kernels.track_linear_velocity_reward(command, lin_vel_w, std).item() == pytest.approx(expected, abs=1e-6)

    def test_angular_tracking_uses_the_body_frame_yaw_rate(self):
        command = torch.tensor([0.8])
        ang_vel_b = torch.tensor([[0.0, 0.0, 0.3]])
        std = math.sqrt(0.25)
        expected = math.exp(-((0.8 - 0.3) ** 2) / 0.25)
        assert kernels.track_angular_velocity_reward(command, ang_vel_b, std).item() == pytest.approx(expected, abs=1e-6)


class TestBaseHeightError:
    TARGET = 0.82
    DEADBAND = 0.04

    @pytest.mark.parametrize("height", [0.82, 0.80, 0.84, 0.78001, 0.85999])
    def test_no_cost_inside_the_deadband(self, height):
        out = kernels.base_height_error_sq(torch.tensor([height]), self.TARGET, self.DEADBAND)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_cost_is_squared_in_the_excess_beyond_the_deadband(self):
        height = 0.72  # 0.10 below target, 0.06 beyond the deadband
        out = kernels.base_height_error_sq(torch.tensor([height]), self.TARGET, self.DEADBAND)
        assert out.item() == pytest.approx(0.06**2, abs=1e-7)

    def test_cost_is_symmetric_about_the_target(self):
        low = kernels.base_height_error_sq(torch.tensor([self.TARGET - 0.1]), self.TARGET, self.DEADBAND)
        high = kernels.base_height_error_sq(torch.tensor([self.TARGET + 0.1]), self.TARGET, self.DEADBAND)
        assert low.item() == pytest.approx(high.item(), abs=1e-7)


class TestPosture:
    STD_STANDING = torch.tensor([0.05, 0.08, 0.10])
    STD_WALKING = torch.tensor([0.15, 0.25, 0.35])
    STD_RUNNING = torch.tensor([0.20, 0.35, 0.50])
    WALK, RUN = 0.05, 0.5

    def _std(self, speeds):
        return kernels.posture_std(
            torch.tensor(speeds), self.STD_STANDING, self.STD_WALKING, self.STD_RUNNING, self.WALK, self.RUN
        )

    def test_each_speed_regime_selects_its_own_tolerance(self):
        out = self._std([0.0, 0.2, 1.0])
        assert torch.allclose(out[0], self.STD_STANDING)
        assert torch.allclose(out[1], self.STD_WALKING)
        assert torch.allclose(out[2], self.STD_RUNNING)

    def test_regime_boundaries_are_inclusive_at_the_lower_edge(self):
        out = self._std([self.WALK, self.RUN])
        assert torch.allclose(out[0], self.STD_WALKING)
        assert torch.allclose(out[1], self.STD_RUNNING)

    def test_tolerance_widens_monotonically_with_speed(self):
        out = self._std([0.0, 0.2, 1.0])
        assert torch.all(out[0] <= out[1])
        assert torch.all(out[1] <= out[2])

    def test_matching_the_reference_pose_scores_one(self):
        joint_pos = torch.tensor([[0.0, 0.2, 0.48]])
        reward = kernels.posture_reward(joint_pos, joint_pos.clone(), self.STD_STANDING.unsqueeze(0))
        assert reward.item() == pytest.approx(1.0, abs=1e-6)

    def test_deviation_is_scored_relative_to_the_per_joint_tolerance(self):
        target = torch.tensor([[0.0, 0.0, 0.0]])
        joint_pos = torch.tensor([[0.05, 0.0, 0.0]])
        std = self.STD_STANDING.unsqueeze(0)
        expected = math.exp(-((0.05**2) / (0.05**2)) / 3.0)
        assert kernels.posture_reward(joint_pos, target, std).item() == pytest.approx(expected, abs=1e-6)

    def test_a_looser_tolerance_forgives_the_same_deviation(self):
        target = torch.tensor([[0.0, 0.0, 0.0]])
        joint_pos = torch.tensor([[0.1, 0.1, 0.1]])
        tight = kernels.posture_reward(joint_pos, target, self.STD_STANDING.unsqueeze(0))
        loose = kernels.posture_reward(joint_pos, target, self.STD_RUNNING.unsqueeze(0))
        assert loose.item() > tight.item()
