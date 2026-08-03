"""Observation terms specific to the wheeled-legged WF-TRON1B task.

Kept separate from ``observations.py`` (which serves the sole-foot / wheel-foot TRON2A
configs) because these terms assume a two-wheel robot with a mixed position/velocity
actuation split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from bipedal_locomotion.utils.math import quat_apply_inverse
from isaaclab.managers import SceneEntityCfg

from .domain_randomization import get_dr_state
from .kernels import encode_contact_forces, normalize_to_unit_range, terrain_roughness_from_height_samples
from .sensor_utils import stack_clearance_samples

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "base_link_lin_vel",
    "domain_randomization_delta_quantity",
    "terrain_height_scan",
    "terrain_roughness_indicator",
    "wheel_contact",
    "wheel_contact_forces",
    "wheel_height",
]


def base_link_lin_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base *link*-frame linear velocity.

    Deliberately not ``mdp.base_lin_vel``, which reports the velocity of the base's
    centre of mass. On WF-TRON1B the base COM sits 16 cm below the link origin, so the
    two differ by ``omega x r`` -- up to about 0.4 m/s at realistic pitch rates. The
    critic and the linear-velocity regression target must both refer to the link frame,
    which is what a real IMU at the base origin measures.
    """
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b


def wheel_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """1.0 per wheel body currently in contact with anything. Shape ``[B, num_wheels]``.

    Uses the net contact force rather than a ground-filtered force matrix. On this task
    the wheels only ever touch the terrain, so the two agree, and the net force avoids
    depending on filtered contact reporting working against instanced terrain meshes.
    """
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # [B, N, 3]
    return (torch.norm(forces, dim=-1) > threshold).float()


def wheel_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Wheel contact forces in the base frame, log-compressed. Shape ``[B, num_wheels * 3]``.

    Expressed in the base frame so the signal is invariant to the robot's heading, and
    log-compressed because impact forces are orders of magnitude larger than the
    steady-state load.
    """
    sensor = env.scene[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    forces_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # [B, N, 3]
    root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(-1, forces_w.shape[1], -1)
    forces_b = quat_apply_inverse(root_quat_w, forces_w)
    return encode_contact_forces(forces_b)


def terrain_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.0,
    max_distance: float = 10.0,
) -> torch.Tensor:
    """Height of the base above the terrain samples under it. Shape ``[B, num_rays]``.

    Equivalent to Isaac Lab's ``mdp.height_scan`` except that rays which hit nothing
    report ``max_distance`` instead of infinity. An infinite entry would propagate
    through observation normalisation and poison every other term in the group.
    """
    sensor = env.scene[sensor_cfg.name]
    hit_z = sensor.data.ray_hits_w[..., 2]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - hit_z - offset
    return torch.where(torch.isfinite(heights), heights, torch.full_like(heights, max_distance))


def wheel_height(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    max_distance: float = 1.0,
) -> torch.Tensor:
    """Smallest clearance under each wheel. Shape ``[B, num_wheels]``.

    The minimum over the local grid rather than the centre sample, so a wheel resting on
    the lip of a step reports the height of the lip, not of the gap beside it.
    """
    samples = stack_clearance_samples(env, sensor_names, max_distance)  # [B, F, N]
    return samples.amin(dim=-1)


def terrain_roughness_indicator(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    wheel_radius: float = 0.127,
    gate_min: float = 0.0,
    gate_max: float = 0.5,
    grid_shape: tuple[int, int] | None = None,
    max_distance: float = 1.0,
) -> torch.Tensor:
    """Scalar in [0, 1] telling the critic how rough the ground under the robot is.

    Same quantity that gates the rough-terrain rewards, exposed as an observation so the
    critic can attribute reward changes to the terrain rather than to the policy.
    """
    samples = stack_clearance_samples(env, sensor_names, max_distance)
    stats = terrain_roughness_from_height_samples(
        samples,
        wheel_radius=wheel_radius,
        gate_min=gate_min,
        gate_max=gate_max,
        grid_shape=grid_shape,
    )
    return stats.gate.unsqueeze(-1)


def domain_randomization_delta_quantity(
    env: ManagerBasedRLEnv,
    wheel_friction_event: str = "wheel_friction",
    encoder_bias_event: str = "encoder_bias",
    base_com_event: str = "base_com",
) -> torch.Tensor:
    """The realised randomisation, normalised to [-1, 1]. Shape ``[B, 2 + num_joints + 3]``.

    This is the privileged "what robot am I today" signal. Each quantity is normalised
    against the range declared on its own event term rather than against a hard-coded
    constant, so widening a randomisation range in the config cannot silently rescale
    this observation out from under a trained encoder.
    """
    state = get_dr_state(env)

    friction_range = env.event_manager.get_term_cfg(wheel_friction_event).params["friction_range"]
    bias_range = env.event_manager.get_term_cfg(encoder_bias_event).params["bias_range"]
    com_ranges = env.event_manager.get_term_cfg(base_com_event).params["com_range"]

    friction = normalize_to_unit_range(state.wheel_friction, friction_range[0], friction_range[1])
    encoder_bias = normalize_to_unit_range(state.encoder_bias, bias_range[0], bias_range[1])
    com_delta = torch.stack(
        [normalize_to_unit_range(state.base_com_delta[:, axis], *com_ranges[axis]) for axis in range(3)],
        dim=1,
    )
    return torch.cat((friction, encoder_bias, com_delta), dim=1)
