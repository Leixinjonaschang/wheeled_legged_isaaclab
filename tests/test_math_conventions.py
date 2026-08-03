"""Pin the quaternion and angle conventions the whole MDP layer relies on.

Isaac Lab's ``ArticulationData`` reports quaternions real-part-first (w, x, y, z),
matching MuJoCo. USD and PhysX use (x, y, z, w). A silent swap here would rotate contact
forces, terrain normals and camera frames without raising anything, so the convention is
asserted rather than assumed.
"""

import math

import pytest
import torch

from _loader import math_utils

IDENTITY = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
# +90 degrees about z, real part first.
YAW_90 = torch.tensor([[math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]])


def test_identity_quaternion_is_a_no_op():
    vec = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(math_utils.quat_apply(IDENTITY, vec), vec, atol=1e-6)
    assert torch.allclose(math_utils.quat_apply_inverse(IDENTITY, vec), vec, atol=1e-6)


def test_yaw_90_rotates_x_axis_to_y_axis():
    x_axis = torch.tensor([[1.0, 0.0, 0.0]])
    rotated = math_utils.quat_apply(YAW_90, x_axis)
    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_inverse_yaw_90_rotates_x_axis_to_negative_y_axis():
    """World +x seen from a frame yawed +90 degrees points along the frame's -y."""
    x_axis = torch.tensor([[1.0, 0.0, 0.0]])
    rotated = math_utils.quat_apply_inverse(YAW_90, x_axis)
    assert torch.allclose(rotated, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6)


def test_apply_then_apply_inverse_round_trips():
    torch.manual_seed(0)
    quat = torch.randn(8, 4)
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
    vec = torch.randn(8, 3)
    round_tripped = math_utils.quat_apply_inverse(quat, math_utils.quat_apply(quat, vec))
    assert torch.allclose(round_tripped, vec, atol=1e-5)


def test_quat_apply_broadcasts_over_a_body_dimension():
    quat = YAW_90[:, None, :].expand(1, 3, 4)
    vecs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    rotated = math_utils.quat_apply(quat, vecs)
    expected = torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    assert torch.allclose(rotated, expected, atol=1e-6)


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0.0, 0.0),
        (math.pi / 2, math.pi / 2),
        (math.pi, math.pi),
        (3.0 * math.pi / 2, -math.pi / 2),
        (-3.0 * math.pi / 2, math.pi / 2),
        (2.0 * math.pi, 0.0),
        (5.0 * math.pi, math.pi),
    ],
)
def test_wrap_to_pi(angle, expected):
    wrapped = math_utils.wrap_to_pi(torch.tensor([angle]))
    assert wrapped.item() == pytest.approx(expected, abs=1e-6)
