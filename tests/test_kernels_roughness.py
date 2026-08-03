"""Terrain-roughness gate: the switch that arbitrates legged vs wheeled motion.

Six rough-terrain reward terms are multiplied by this gate, so an error here silently
re-weights most of the rough task rather than raising.
"""

import pytest
import torch

from _loader import kernels

WHEEL_RADIUS = 0.127
GATE_MIN = 0.0
GATE_MAX = 0.5
GRID = (5, 5)


def _stats(grid_values, **overrides):
    """Build [1, 1, 25] samples from a 5x5 nested list."""
    samples = torch.tensor(grid_values, dtype=torch.float32).reshape(1, 1, 25)
    kwargs = dict(wheel_radius=WHEEL_RADIUS, gate_min=GATE_MIN, gate_max=GATE_MAX, grid_shape=GRID)
    kwargs.update(overrides)
    return kernels.terrain_roughness_from_height_samples(samples, **kwargs)


def _smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def test_flat_ground_produces_no_roughness_and_a_closed_gate():
    stats = _stats([[0.4] * 5 for _ in range(5)])
    assert stats.jump.item() == pytest.approx(0.0, abs=1e-7)
    assert stats.curvature.item() == pytest.approx(0.0, abs=1e-7)
    assert stats.foot_roughness.item() == pytest.approx(0.0, abs=1e-7)
    assert stats.gate.item() == pytest.approx(0.0, abs=1e-7)


def test_a_single_step_opens_the_gate_by_the_step_over_wheel_radius():
    """A 6 cm step under a 12.7 cm wheel is most of the way to 'rough'."""
    step = 0.06
    grid = [[0.0, 0.0, step, step, step] for _ in range(5)]
    stats = _stats(grid)

    assert stats.jump.item() == pytest.approx(step, abs=1e-6)
    # A single monotone step has no second difference along the step direction other
    # than at its edges; here the profile 0,0,s,s,s gives |0-2s+s| = s.
    expected_roughness = step / WHEEL_RADIUS
    assert stats.foot_roughness.item() == pytest.approx(expected_roughness, abs=1e-5)

    expected_gate = _smoothstep((expected_roughness - GATE_MIN) / (GATE_MAX - GATE_MIN))
    assert stats.gate.item() == pytest.approx(expected_gate, abs=1e-6)


def test_a_constant_slope_has_jump_but_no_curvature():
    slope = 0.02
    grid = [[col * slope for col in range(5)] for _ in range(5)]
    stats = _stats(grid)
    assert stats.jump.item() == pytest.approx(slope, abs=1e-6)
    assert stats.curvature.item() == pytest.approx(0.0, abs=1e-6)


def test_curvature_dominates_a_narrow_ridge():
    """A one-cell ridge is a worse obstacle than its height alone suggests."""
    height = 0.03
    grid = [[0.0, 0.0, height, 0.0, 0.0] for _ in range(5)]
    stats = _stats(grid)
    assert stats.jump.item() == pytest.approx(height, abs=1e-6)
    # Second difference across the ridge: |0 - 2h + 0| = 2h.
    assert stats.curvature.item() == pytest.approx(2.0 * height, abs=1e-6)
    assert stats.foot_roughness.item() == pytest.approx(2.0 * height / WHEEL_RADIUS, abs=1e-5)


def test_robot_roughness_is_the_worst_foot():
    flat = [0.4] * 25
    rough = [0.4] * 25
    rough[12] = 0.55
    samples = torch.tensor([flat, rough], dtype=torch.float32).reshape(1, 2, 25)
    stats = kernels.terrain_roughness_from_height_samples(
        samples, wheel_radius=WHEEL_RADIUS, gate_min=GATE_MIN, gate_max=GATE_MAX, grid_shape=GRID
    )
    assert stats.foot_roughness[0, 0].item() == pytest.approx(0.0, abs=1e-7)
    assert stats.foot_roughness[0, 1].item() > 0.0
    assert stats.robot_roughness.item() == pytest.approx(stats.foot_roughness[0, 1].item(), abs=1e-7)


def test_gate_saturates_at_one_beyond_gate_max():
    grid = [[0.0, 0.0, 1.0, 1.0, 1.0] for _ in range(5)]
    stats = _stats(grid)
    assert stats.gate.item() == pytest.approx(1.0, abs=1e-6)


def test_non_finite_samples_are_scrubbed_rather_than_propagated():
    grid = [[0.4] * 5 for _ in range(5)]
    samples = torch.tensor(grid, dtype=torch.float32).reshape(1, 1, 25)
    samples[0, 0, 7] = float("nan")
    samples[0, 0, 9] = float("inf")
    stats = kernels.terrain_roughness_from_height_samples(
        samples, wheel_radius=WHEEL_RADIUS, gate_min=GATE_MIN, gate_max=GATE_MAX, grid_shape=GRID
    )
    assert torch.isfinite(stats.gate).all()
    assert torch.isfinite(stats.foot_roughness).all()


def test_reduced_samples_are_rejected():
    with pytest.raises(ValueError, match=r"\[B, F, N\]"):
        kernels.terrain_roughness_from_height_samples(
            torch.zeros(1, 2), wheel_radius=WHEEL_RADIUS, gate_min=0.0, gate_max=0.5, grid_shape=None
        )


def test_mismatched_grid_shape_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        _stats([[0.0] * 5 for _ in range(5)], grid_shape=(4, 4))


def test_non_square_sample_count_without_a_grid_shape_is_rejected():
    with pytest.raises(ValueError, match="Cannot infer a square grid"):
        kernels.resolve_grid_shape(24, None)


def test_degenerate_gate_range_is_rejected():
    with pytest.raises(ValueError, match="gate_max"):
        _stats([[0.0] * 5 for _ in range(5)], gate_min=0.5, gate_max=0.5)


def test_non_positive_wheel_radius_is_rejected():
    with pytest.raises(ValueError, match="wheel_radius"):
        _stats([[0.0] * 5 for _ in range(5)], wheel_radius=0.0)


class TestScheduledGateThreshold:
    """The threshold ramp: rough-everything early, rough-where-it-matters later."""

    INITIAL = 0.0
    FINAL = 0.25
    RAMP = 5_000 * 24

    def test_starts_at_the_initial_threshold(self):
        value, progress = kernels.scheduled_gate_threshold(0, self.INITIAL, self.FINAL, self.RAMP)
        assert value == pytest.approx(self.INITIAL)
        assert progress == pytest.approx(0.0)

    def test_interpolates_linearly_at_the_midpoint(self):
        value, progress = kernels.scheduled_gate_threshold(self.RAMP // 2, self.INITIAL, self.FINAL, self.RAMP)
        assert progress == pytest.approx(0.5)
        assert value == pytest.approx(0.125)

    def test_clamps_at_the_final_threshold(self):
        value, progress = kernels.scheduled_gate_threshold(10 * self.RAMP, self.INITIAL, self.FINAL, self.RAMP)
        assert progress == pytest.approx(1.0)
        assert value == pytest.approx(self.FINAL)

    def test_zero_ramp_pins_the_initial_threshold(self):
        """Play configs zero the ramp after setting the threshold to its final value."""
        value, progress = kernels.scheduled_gate_threshold(999_999, self.FINAL, self.FINAL, 0)
        assert value == pytest.approx(self.FINAL)
        assert progress == pytest.approx(0.0)

    def test_absent_final_threshold_disables_the_ramp(self):
        value, progress = kernels.scheduled_gate_threshold(999, 0.2, None, self.RAMP)
        assert value == pytest.approx(0.2)
        assert progress == pytest.approx(0.0)


class TestGateMasks:
    def test_active_and_inactive_masks_are_complementary(self):
        gate = torch.tensor([0.0, 0.1, 0.25, 0.5, 1.0])
        active = kernels.roughness_gate_active(gate, threshold=0.25)
        inactive = kernels.roughness_gate_inactive(gate, threshold=0.25)
        assert torch.allclose(active + inactive, torch.ones_like(gate))
        # Strictly greater than: a gate exactly at the threshold counts as smooth.
        assert torch.allclose(active, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))
