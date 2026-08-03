"""Adapters from Isaac Lab ray-caster output to the ``[B, F, N]`` clearance layout.

mjlab's ``TerrainHeightSensor`` binds several frames (one per wheel) to a single sensor
and reports ``[B, F, N]`` clearance samples directly. Isaac Lab's :class:`RayCaster`
binds one prim per environment and reports ``[B, N, 3]`` world-frame hit positions, so
the per-wheel scans are configured as *separate* sensors and stacked back together here.

The clearance convention matches mjlab: ``frame_z - hit_z``, i.e. how far below the
sensor origin the ground is, positive downward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .kernels import heights_from_raycast

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["clearance_samples", "hit_points_with_mask", "stack_clearance_samples"]

# Cached subsample indices, keyed by (total, max_points, device). Recomputing a linspace
# every step for every environment shows up in the profile at 2048 environments.
_SUBSAMPLE_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _subsample_indices(total: int, max_points: int, device) -> torch.Tensor:
    key = (total, max_points, str(device))
    if key not in _SUBSAMPLE_CACHE:
        _SUBSAMPLE_CACHE[key] = torch.linspace(0, total - 1, max_points, device=device).long()
    return _SUBSAMPLE_CACHE[key]


def hit_points_with_mask(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    max_points: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ray hit positions and a validity mask from one or more ray-casters.

    Args:
        max_points: Per-sensor cap on the number of points, evenly subsampled. A plane fit
            gains nothing from 121 coplanar points over 32, and the covariance is the one
            per-step cost here that scales with ray count.

    Returns:
        ``(points [B, N, 3], valid [B, N])``. Rays that hit nothing are marked invalid
        rather than given a substitute position -- see
        :func:`~.kernels.terrain_normal_from_points`.
    """
    all_points, all_valid = [], []
    for name in sensor_names:
        sensor = env.scene[name]
        hits = sensor.data.ray_hits_w  # [B, N, 3]
        valid = torch.isfinite(hits).all(dim=-1)
        num_rays = hits.shape[1]
        if num_rays > max_points:
            idx = _subsample_indices(num_rays, max_points, hits.device)
            hits = hits[:, idx]
            valid = valid[:, idx]
        all_points.append(hits)
        all_valid.append(valid)
    return torch.cat(all_points, dim=1), torch.cat(all_valid, dim=1)


def clearance_samples(env: ManagerBasedRLEnv, sensor_name: str, max_distance: float) -> torch.Tensor:
    """Clearance below one ray-caster, shape ``[B, 1, N]``.

    Isaac Lab reports a missed ray as a non-finite hit position. Those are replaced by
    ``max_distance`` so downstream statistics stay finite; see
    :func:`~.kernels.heights_from_raycast`.
    """
    sensor = env.scene[sensor_name]
    hits_w = sensor.data.ray_hits_w  # [B, N, 3]
    frame_z = sensor.data.pos_w[:, 2].view(-1, 1, 1)  # [B, 1, 1]
    hit_z = hits_w[..., 2].unsqueeze(1)  # [B, 1, N]
    miss = ~torch.isfinite(hit_z)
    return heights_from_raycast(frame_z, torch.nan_to_num(hit_z, nan=0.0, posinf=0.0, neginf=0.0), miss, max_distance)


def stack_clearance_samples(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    max_distance: float,
) -> torch.Tensor:
    """Clearance below several ray-casters, stacked into ``[B, F, N]``.

    ``sensor_names`` order defines the frame order, so it must match the wheel order the
    contact and reward terms assume (left, then right).
    """
    return torch.cat([clearance_samples(env, name, max_distance) for name in sensor_names], dim=1)
