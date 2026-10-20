"""Observation terms for the WF-TRON1B task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from bipedal_locomotion.utils.math import quat_apply_inverse
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

from .depth import DelayedFrameBuffer, normalize_depth
from .kernels import encode_contact_forces, smoothstep_gate, terrain_roughness_from_height_samples
from .sensor_utils import contact_state, stack_clearance_samples, terrain_forces

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "AsyncDepthBuffer",
    "base_link_lin_vel",
    "terrain_height_scan",
    "terrain_roughness_indicator",
    "wheel_contact",
    "wheel_contact_forces",
    "wheel_height",
    "wheel_roughness_gate",
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


def wheel_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """1.0 per wheel touching the terrain, ``[B, num_wheels]`` (mjlab ``foot_contact``)."""
    return contact_state(env, sensor_cfg).float()


def wheel_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Log-compressed wheel-terrain contact forces in the base frame, ``[B, num_wheels * 3]``."""
    forces_w = terrain_forces(env, sensor_cfg)  # [B, N, 3]
    root_quat_w = env.scene[asset_cfg.name].data.root_link_quat_w[:, None, :]
    return encode_contact_forces(quat_apply_inverse(root_quat_w, forces_w))


def terrain_height_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.0) -> torch.Tensor:
    """Height of the base above the terrain samples under it. Shape ``[B, num_rays]``.

    Equivalent to Isaac Lab's ``mdp.height_scan`` except that rays which hit nothing report
    the sensor's ``max_distance`` instead of infinity. An infinite entry would propagate
    through observation normalisation and poison every other term in the group.
    """
    sensor = env.scene[sensor_cfg.name]
    hit_z = sensor.data.ray_hits_w[..., 2]
    heights = sensor.data.pos_w[:, 2].unsqueeze(1) - hit_z - offset
    return torch.where(torch.isfinite(heights), heights, torch.full_like(heights, sensor.cfg.max_distance))


def wheel_height(env: ManagerBasedRLEnv, sensor_names: tuple[str, ...]) -> torch.Tensor:
    """Smallest clearance under each wheel. Shape ``[B, num_wheels]``.

    The minimum over the local grid rather than the centre sample, so a wheel resting on
    the lip of a step reports the height of the lip, not of the gap beside it.
    """
    samples = stack_clearance_samples(env, sensor_names)  # [B, F, N]
    return samples.amin(dim=-1)


def terrain_roughness_indicator(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    wheel_radius: float = 0.127,
    gate_min: float = 0.10,
    gate_max: float = 0.40,
    grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Roughness gate of the scanned terrain, in [0, 1], ``[B, 1]``: the quantity gating the rough-terrain rewards."""
    stats = terrain_roughness_from_height_samples(
        stack_clearance_samples(env, sensor_names),
        wheel_radius=wheel_radius,
        gate_min=gate_min,
        gate_max=gate_max,
        grid_shape=grid_shape,
    )
    return stats.gate.unsqueeze(-1)


def wheel_roughness_gate(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    wheel_radius: float = 0.127,
    gate_min: float = 0.10,
    gate_max: float = 0.40,
    grid_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Per-wheel roughness gates of the wheel height scans, ``[B, num_wheels]`` ordered like ``sensor_names``."""
    stats = terrain_roughness_from_height_samples(
        stack_clearance_samples(env, sensor_names),
        wheel_radius=wheel_radius,
        gate_min=gate_min,
        gate_max=gate_max,
        grid_shape=grid_shape,
    )
    return smoothstep_gate(stats.foot_roughness, gate_min, gate_max)


class AsyncDepthBuffer(ManagerTermBase):
    """mjlab's ``async_depth_buffer``: the depth camera on a capture clock with a per-env system delay.

    Frames are post-processed like mjlab's ``depth_image`` (left crop, misses read as far, clamp, scale to
    [0, 1]); shape ``[B, 1, H, W - left_crop]``. The camera is only read on capture and reset-refill steps.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._buffer = DelayedFrameBuffer()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        capture_frequency_hz: float = 30.0,
        system_delay_range_s: tuple[float, float] = (0.0, 0.0),
        left_crop: int = 0,
        depth_min_m: float = 0.2,
        depth_max_m: float = 2.0,
    ) -> torch.Tensor:
        def capture() -> torch.Tensor:
            depth = env.scene[sensor_cfg.name].data.output["distance_to_image_plane"][..., 0]
            return normalize_depth(depth, left_crop, depth_min_m, depth_max_m).unsqueeze(1)

        current_time_s = int(env.common_step_counter) * env.step_dt
        return self._buffer(current_time_s, capture, capture_frequency_hz, system_delay_range_s)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._buffer.reset(env_ids)
