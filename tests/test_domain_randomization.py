"""The privileged DR observation: 87 values laid out and normalised exactly like mjlab's."""

import math
from types import SimpleNamespace

import torch

from _loader import domain_randomization as dr

MJCF_JOINTS = (
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "wheel_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "wheel_R_Joint",
)  # fmt: skip
MJCF_BODIES = (
    "base_Link", "abad_L_Link", "hip_L_Link", "knee_L_Link", "wheel_L_Link",
    "abad_R_Link", "hip_R_Link", "knee_R_Link", "wheel_R_Link",
)  # fmt: skip
# mjlab's actuator order: the leg position servos, then the wheel velocity servos.
ACTUATOR_JOINTS = MJCF_JOINTS[:3] + MJCF_JOINTS[4:7] + (MJCF_JOINTS[3], MJCF_JOINTS[7])
# A breadth-first articulation order, as PhysX may resolve it, so that reordering is exercised.
SIM_JOINTS = tuple(MJCF_JOINTS[i] for i in (0, 4, 1, 5, 2, 6, 3, 7))
SIM_BODIES = ("base_Link",) + tuple(MJCF_BODIES[i] for i in (1, 5, 2, 6, 3, 7, 4, 8))

ALPHA_RANGE = (0.5 * math.log(0.8), 0.5 * math.log(1.2))
FRICTION_RANGE, DIFFERENCE_RANGE = (0.45, 1.1), (-0.04, 0.04)
BIAS_RANGE, BASE_COM_RANGE, LINK_COM_RANGE, GAIN_RANGE = (-0.015, 0.015), (-0.03, 0.03), (-0.015, 0.015), (0.8, 1.2)
NUM_ENVS = 5


def _cfg(joints=None, bodies=None):
    """A resolved SceneEntityCfg stand-in (preserve_order=True)."""
    return SimpleNamespace(
        name="robot",
        joint_ids=[SIM_JOINTS.index(n) for n in joints] if joints else slice(None),
        body_ids=[SIM_BODIES.index(n) for n in bodies] if bodies else slice(None),
    )


EVENT_PARAMS = {
    "wheel_friction": {"friction_range": FRICTION_RANGE, "difference_range": DIFFERENCE_RANGE},
    "encoder_bias": {"asset_cfg": _cfg(joints=MJCF_JOINTS), "bias_range": BIAS_RANGE},
    "base_com": {"asset_cfg": _cfg(bodies=MJCF_BODIES[:1]), "com_range": BASE_COM_RANGE},
    "link_com": {"asset_cfg": _cfg(bodies=MJCF_BODIES[1:]), "com_range": LINK_COM_RANGE},
    "body_mass_inertia": {"asset_cfg": _cfg(bodies=MJCF_BODIES), "alpha_range": ALPHA_RANGE},
    "pd_gains": {
        "asset_cfg": _cfg(joints=ACTUATOR_JOINTS),
        "stiffness_scale_range": GAIN_RANGE,
        "damping_scale_range": GAIN_RANGE,
    },
}


def _make_env():
    kp = torch.tensor([[0.0 if "wheel" in name else 40.0 for name in SIM_JOINTS]]).expand(NUM_ENVS, -1)
    robot = SimpleNamespace(num_joints=8, num_bodies=9, data=SimpleNamespace(default_joint_stiffness=kp))
    events = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(params=EVENT_PARAMS[name]))
    return SimpleNamespace(num_envs=NUM_ENVS, device="cpu", scene={"robot": robot}, event_manager=events)


def _uniform(shape, bounds, generator):
    return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(shape, generator=generator)


# --- mjlab reference (observations.py), with the MuJoCo model fields passed in ---------------------------


def _ref_normalize_to_unit_range(value, lower, upper):
    scaled = 2.0 * (value - lower) / (upper - lower) - 1.0
    return torch.clamp(scaled, -1.0, 1.0)


def _ref_normalized_ratio_to_default(current, default, scale_range):
    ratio = current / default
    return _ref_normalize_to_unit_range(ratio, *scale_range)


def _ref_domain_randomization_delta_quantity(
    wheel_friction, encoder_bias, com_deltas, body_mass, default_mass, body_inertia, default_inertia, kp, kp0, kd, kd0
):
    wheel_friction_range = (FRICTION_RANGE[0] + DIFFERENCE_RANGE[0], FRICTION_RANGE[1] + DIFFERENCE_RANGE[1])
    wheel_friction = _ref_normalize_to_unit_range(wheel_friction, wheel_friction_range[0], wheel_friction_range[1])
    encoder_bias = _ref_normalize_to_unit_range(encoder_bias, BIAS_RANGE[0], BIAS_RANGE[1])
    com_out = []
    for com_delta, com_range in zip(com_deltas, (BASE_COM_RANGE, LINK_COM_RANGE)):
        com_ranges = {0: com_range, 1: com_range, 2: com_range}
        com_out.append(
            torch.stack(
                [_ref_normalize_to_unit_range(com_delta[..., axis], *com_ranges[axis]) for axis in range(3)], dim=-1
            ).reshape(NUM_ENVS, -1)
        )
    mass_inertia_scale_range = tuple(math.exp(2.0 * alpha) for alpha in ALPHA_RANGE)
    body_mass_scale = _ref_normalized_ratio_to_default(body_mass, default_mass, mass_inertia_scale_range)
    body_inertia_scale = _ref_normalized_ratio_to_default(
        body_inertia.sort(dim=-1).values, default_inertia.sort(dim=-1).values, mass_inertia_scale_range
    ).reshape(NUM_ENVS, -1)
    stiffness_scales = _ref_normalized_ratio_to_default(kp, kp0, GAIN_RANGE)
    damping_scales = _ref_normalized_ratio_to_default(kd, kd0, GAIN_RANGE)
    return torch.cat(
        (wheel_friction, encoder_bias, *com_out, body_mass_scale, body_inertia_scale, stiffness_scales, damping_scales),
        dim=1,
    )


# ---------------------------------------------------------------------------------------------------------


def test_matches_mjlab_layout_and_normalisation():
    g = torch.Generator().manual_seed(0)
    env = _make_env()
    state = dr.get_dr_state(env)

    # One realised randomisation, recorded in articulation order (stretched slightly past the
    # ranges so that clamping is exercised too).
    state.wheel_friction[:] = _uniform((NUM_ENVS, 2), (0.38, 1.17), g)
    state.encoder_bias[:] = _uniform((NUM_ENVS, 8), (-0.016, 0.016), g)
    state.com_offset[:] = _uniform((NUM_ENVS, 9, 3), (-0.03, 0.03), g)
    state.mass_scale[:] = torch.exp(2.0 * _uniform((NUM_ENVS, 9), ALPHA_RANGE, g))
    kp_scale, kd_scale = _uniform((NUM_ENVS, 8), GAIN_RANGE, g), _uniform((NUM_ENVS, 8), GAIN_RANGE, g)
    state.stiffness_scale[:] = torch.where(torch.tensor(["wheel" in n for n in SIM_JOINTS]), 1.0, kp_scale)
    state.damping_scale[:] = kd_scale

    # The same randomisation as mjlab's MuJoCo model fields, in MJCF order.
    joints = [SIM_JOINTS.index(n) for n in MJCF_JOINTS]
    bodies = [SIM_BODIES.index(n) for n in MJCF_BODIES]
    actuators = [SIM_JOINTS.index(n) for n in ACTUATOR_JOINTS]
    default_mass = _uniform((9,), (1.0, 10.0), g)
    default_inertia = _uniform((9, 3), (0.001, 0.2), g)
    mass_scale = state.mass_scale[:, bodies]
    # pseudo_inertia re-diagonalises and may permute the principal axes; mjlab sorts before comparing.
    body_inertia = (default_inertia * mass_scale[..., None])[..., torch.tensor([2, 0, 1])]
    kp0 = torch.full((6,), 40.0)
    kd0 = torch.tensor([1.8] * 6 + [0.5] * 2)
    reference = _ref_domain_randomization_delta_quantity(
        wheel_friction=state.wheel_friction,
        encoder_bias=state.encoder_bias[:, joints],
        com_deltas=(state.com_offset[:, bodies[:1]], state.com_offset[:, bodies[1:]]),
        body_mass=default_mass * mass_scale,
        default_mass=default_mass,
        body_inertia=body_inertia,
        default_inertia=default_inertia,
        kp=kp0 * kp_scale[:, actuators[:6]],
        kp0=kp0,
        kd=kd0 * kd_scale[:, actuators],
        kd0=kd0,
    )

    out = dr.domain_randomization_delta_quantity(env)
    assert out.shape == (NUM_ENVS, 87)
    assert torch.allclose(out, reference, atol=1e-5)
    assert out.abs().max() <= 1.0


def test_nominal_state_reports_the_centre_of_every_symmetric_range():
    """Before any event runs, everything but friction is nominal: zero offsets and unit scales."""
    out = dr.domain_randomization_delta_quantity(_make_env())
    assert out.shape == (NUM_ENVS, 87)
    assert torch.allclose(out[:, 2:], torch.zeros(NUM_ENVS, 85), atol=1e-6)
