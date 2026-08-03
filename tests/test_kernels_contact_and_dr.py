"""Contact encoding, air-time balance, domain-randomisation scaling and delay bounds."""

import math

import pytest
import torch

from _loader import kernels


class TestEncodeContactForces:
    def test_signed_log_compression_and_flattening(self):
        forces_b = torch.tensor([[[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]]])
        out = kernels.encode_contact_forces(forces_b)
        expected = torch.tensor([[math.log1p(1.0), 0.0, 0.0, 0.0, -math.log1p(2.0), 0.0]])
        assert out.shape == (1, 6)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_sign_is_preserved(self):
        forces_b = torch.tensor([[[-5.0, 5.0, 0.0]]])
        out = kernels.encode_contact_forces(forces_b)
        assert out[0, 0].item() < 0.0
        assert out[0, 1].item() > 0.0
        assert out[0, 0].item() == pytest.approx(-out[0, 1].item(), abs=1e-6)

    def test_compression_is_monotonic_but_sub_linear(self):
        """A 100x larger force must rank higher without dominating the input scale."""
        small = kernels.encode_contact_forces(torch.tensor([[[10.0, 0.0, 0.0]]]))[0, 0].item()
        large = kernels.encode_contact_forces(torch.tensor([[[1000.0, 0.0, 0.0]]]))[0, 0].item()
        assert large > small
        assert large < 100.0 * small

    def test_zero_force_maps_to_zero(self):
        out = kernels.encode_contact_forces(torch.zeros(1, 2, 3))
        assert torch.allclose(out, torch.zeros(1, 6))

    def test_non_finite_forces_are_scrubbed(self):
        forces_b = torch.tensor([[[float("nan"), float("inf"), float("-inf")]]])
        out = kernels.encode_contact_forces(forces_b)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.zeros(1, 3))


class TestAirTimeBalanceCost:
    MIN_TOTAL = 1.0
    TOLERANCE = 0.2

    def _cost(self, left, right, **overrides):
        kwargs = dict(min_total_air_time=self.MIN_TOTAL, balance_tolerance=self.TOLERANCE)
        kwargs.update(overrides)
        return kernels.air_time_balance_cost(torch.tensor([[left, right]]), **kwargs).item()

    def test_perfectly_balanced_air_time_is_free(self):
        assert self._cost(2.0, 2.0) == pytest.approx(0.0, abs=1e-7)

    def test_imbalance_within_tolerance_is_free(self):
        # ratio = 0.4 / 4.0 = 0.1, below the 0.2 tolerance.
        assert self._cost(2.2, 1.8) == pytest.approx(0.0, abs=1e-7)

    def test_hopping_on_one_wheel_is_maximally_penalised(self):
        """The failure mode this term exists to catch: one wheel planted, one hopping."""
        assert self._cost(4.0, 0.0) == pytest.approx(1.0, abs=1e-6)

    def test_cost_grows_with_imbalance(self):
        assert self._cost(3.0, 1.0) > self._cost(2.5, 1.5)

    def test_the_first_swing_of_an_episode_is_not_penalised(self):
        """Total air time far below min_total_air_time ramps the cost toward zero."""
        early = self._cost(0.02, 0.0)
        late = self._cost(2.0, 0.0)
        assert early < late
        assert early == pytest.approx(0.02 * 1.0, abs=1e-6)

    def test_zero_air_time_is_free_rather_than_a_division_by_zero(self):
        cost = self._cost(0.0, 0.0)
        assert math.isfinite(cost)
        assert cost == pytest.approx(0.0, abs=1e-7)

    def test_cost_is_symmetric_between_left_and_right(self):
        assert self._cost(3.0, 1.0) == pytest.approx(self._cost(1.0, 3.0), abs=1e-7)

    @pytest.mark.parametrize("bad_tolerance", [-0.1, 1.0, 1.5])
    def test_out_of_range_tolerance_is_rejected(self, bad_tolerance):
        with pytest.raises(ValueError, match="balance_tolerance"):
            self._cost(1.0, 1.0, balance_tolerance=bad_tolerance)

    def test_non_positive_activation_window_is_rejected(self):
        with pytest.raises(ValueError, match="min_total_air_time"):
            self._cost(1.0, 1.0, min_total_air_time=0.0)


class TestNormalizeToUnitRange:
    def test_endpoints_map_to_minus_one_and_one(self):
        out = kernels.normalize_to_unit_range(torch.tensor([0.3, 1.2]), 0.3, 1.2)
        assert torch.allclose(out, torch.tensor([-1.0, 1.0]), atol=1e-6)

    def test_midpoint_maps_to_zero(self):
        out = kernels.normalize_to_unit_range(torch.tensor([0.75]), 0.3, 1.2)
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_values_outside_the_range_are_clamped(self):
        out = kernels.normalize_to_unit_range(torch.tensor([-5.0, 5.0]), 0.3, 1.2)
        assert torch.allclose(out, torch.tensor([-1.0, 1.0]), atol=1e-6)

    def test_symmetric_range_maps_zero_to_zero(self):
        out = kernels.normalize_to_unit_range(torch.tensor([0.0]), -0.015, 0.015)
        assert out.item() == pytest.approx(0.0, abs=1e-6)


class TestDelayStepBounds:
    """Actuation delay is quantised to physics steps; the bounds round inward."""

    def test_full_range_over_four_steps(self):
        assert kernels.delay_step_bounds((0.0, 0.02), 0.005) == (0, 4)

    def test_bounds_round_inward(self):
        # 0.007s -> ceil(1.4) = 2 steps; 0.018s -> floor(3.6) = 3 steps.
        assert kernels.delay_step_bounds((0.007, 0.018), 0.005) == (2, 3)

    def test_a_representable_fixed_delay_collapses_to_one_value(self):
        assert kernels.delay_step_bounds((0.01, 0.01), 0.005) == (2, 2)

    def test_zero_delay(self):
        assert kernels.delay_step_bounds((0.0, 0.0), 0.005) == (0, 0)

    def test_a_range_containing_no_representable_step_is_rejected(self):
        """Silently widening this would change the delay distribution behind the config."""
        with pytest.raises(ValueError, match="representable physics-step"):
            kernels.delay_step_bounds((0.006, 0.009), 0.005)

    def test_floating_point_tolerance_admits_exact_multiples(self):
        # 0.015 / 0.005 is 2.9999999999999996 in binary floating point.
        assert kernels.delay_step_bounds((0.015, 0.015), 0.005) == (3, 3)

    def test_negative_delay_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            kernels.delay_step_bounds((-0.001, 0.02), 0.005)

    def test_reversed_range_is_rejected(self):
        with pytest.raises(ValueError, match="ordered"):
            kernels.delay_step_bounds((0.02, 0.01), 0.005)

    def test_non_positive_physics_dt_is_rejected(self):
        with pytest.raises(ValueError, match="physics_dt"):
            kernels.delay_step_bounds((0.0, 0.02), 0.0)
