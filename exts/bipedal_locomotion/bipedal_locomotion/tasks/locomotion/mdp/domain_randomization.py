"""Per-environment domain-randomisation state and the privileged observation that reads it back.

Isaac Lab's stock randomisation writes into PhysX and keeps no record. The teacher-student
tasks feed the realised parameters to a privileged encoder, so the startup events in
``wheeled_events.py`` record what they sampled here, and ``domain_randomization_delta_quantity``
reports it without reading PhysX back every step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .kernels import normalize_to_unit_range

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["domain_randomization_delta_quantity"]

_ATTR = "_wf_tron1b_dr_state"


@dataclass
class DomainRandomizationState:
    """Raw (un-normalised) randomised quantities, one row per environment, articulation order."""

    wheel_friction: torch.Tensor  # [N, 2] friction coefficient per wheel, in the friction event's order
    encoder_bias: torch.Tensor  # [N, joints] rad
    com_offset: torch.Tensor  # [N, bodies, 3] m from the nominal COM, body frame
    mass_scale: torch.Tensor  # [N, bodies]; the inertia scales by the same factor
    stiffness_scale: torch.Tensor  # [N, joints]
    damping_scale: torch.Tensor  # [N, joints]
    default_coms: torch.Tensor | None = None  # PhysX COM poses before randomisation (CPU)
    position_servo_ids: list[int] | None = None  # cached for the observation


def get_dr_state(env: ManagerBasedRLEnv) -> DomainRandomizationState:
    """Return the environment's randomisation state, creating a nominal one on first access."""
    state = getattr(env, _ATTR, None)
    if state is None:
        robot = env.scene["robot"]
        num_envs, num_joints, num_bodies, device = env.num_envs, robot.num_joints, robot.num_bodies, env.device
        state = DomainRandomizationState(
            wheel_friction=torch.zeros(num_envs, 2, device=device),
            encoder_bias=torch.zeros(num_envs, num_joints, device=device),
            com_offset=torch.zeros(num_envs, num_bodies, 3, device=device),
            mass_scale=torch.ones(num_envs, num_bodies, device=device),
            stiffness_scale=torch.ones(num_envs, num_joints, device=device),
            damping_scale=torch.ones(num_envs, num_joints, device=device),
        )
        setattr(env, _ATTR, state)
    return state


def _position_servo_ids(env: ManagerBasedRLEnv, asset_cfg) -> list[int]:
    """The selected joints, in order, that are driven by a position servo (nominal Kp > 0)."""
    state = get_dr_state(env)
    if state.position_servo_ids is None:
        robot = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            joint_ids = range(robot.num_joints)
        stiffness = robot.data.default_joint_stiffness[0].tolist()
        state.position_servo_ids = [i for i in joint_ids if stiffness[i] > 0.0]
    return state.position_servo_ids


def domain_randomization_delta_quantity(
    env: ManagerBasedRLEnv,
    wheel_friction_event: str = "wheel_friction",
    encoder_bias_event: str = "encoder_bias",
    base_com_event: str = "base_com",
    link_com_event: str = "link_com",
    mass_inertia_event: str = "body_mass_inertia",
    pd_gains_event: str = "pd_gains",
) -> torch.Tensor:
    """The realised randomisation, normalised to [-1, 1] against each event's own ranges.

    Laid out like mjlab: wheel friction (2), encoder bias (8), base COM offset (3), link COM
    offsets (24), body mass scale (9), principal-inertia scale (27), leg Kp scale (6) and
    leg Kd plus wheel Kv scale (8) -- 87 values. Body and joint orders follow each event's
    ``asset_cfg``.
    """
    state = get_dr_state(env)

    def params(name: str) -> dict:
        return env.event_manager.get_term_cfg(name).params

    friction = params(wheel_friction_event)
    friction_range = [a + b for a, b in zip(friction["friction_range"], friction["difference_range"])]
    bias = params(encoder_bias_event)
    pieces = [
        (state.wheel_friction, friction_range),
        (state.encoder_bias[:, bias["asset_cfg"].joint_ids], bias["bias_range"]),
    ]
    for name in (base_com_event, link_com_event):
        com = params(name)
        pieces.append((state.com_offset[:, com["asset_cfg"].body_ids], com["com_range"]))

    mass = params(mass_inertia_event)
    scale_range = [math.exp(2.0 * alpha) for alpha in mass["alpha_range"]]
    mass_scale = state.mass_scale[:, mass["asset_cfg"].body_ids]
    # Only the density term is randomised, so all three principal moments scale like the mass.
    pieces += [(mass_scale, scale_range), (mass_scale.repeat_interleave(3, dim=1), scale_range)]

    gains = params(pd_gains_event)
    pieces += [
        (state.stiffness_scale[:, _position_servo_ids(env, gains["asset_cfg"])], gains["stiffness_scale_range"]),
        (state.damping_scale[:, gains["asset_cfg"].joint_ids], gains["damping_scale_range"]),
    ]
    return torch.cat([normalize_to_unit_range(value, *bounds).flatten(1) for value, bounds in pieces], dim=1)
