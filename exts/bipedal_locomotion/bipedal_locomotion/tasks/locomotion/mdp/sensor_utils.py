"""Adapters from Isaac Lab sensor data to the layouts the mjlab-derived terms expect.

Height scans: mjlab's ``TerrainHeightSensor`` binds several frames (one per wheel) to one sensor and
reports ``[B, F, N]`` clearance samples; Isaac Lab's :class:`RayCaster` binds one prim per
environment, so each frame is a separate sensor and they are stacked back here. Clearance is
``frame_z - hit_z`` (positive downward), with missed rays reported as ``max_distance``.

Contacts: mjlab's contact sensors filter contacts by partner (terrain / robot subtree). Here one
robot-wide :class:`ContactSensor` is filtered on the terrain mesh: its force matrix is the terrain
contact and the remainder of the net force is robot-on-robot contact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .kernels import heights_from_raycast

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

__all__ = [
    "clearance_samples",
    "contact_state",
    "hit_points_with_mask",
    "self_contact_forces",
    "stack_clearance_samples",
    "terrain_forces",
]


def terrain_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, history: bool = False) -> torch.Tensor:
    """World-frame terrain contact force per selected body, ``[B, N, 3]`` (``[B, H, N, 3]`` with history)."""
    data = env.scene[sensor_cfg.name].data
    forces = data.force_matrix_w_history if history else data.force_matrix_w
    return forces.sum(dim=-2)[..., sensor_cfg.body_ids, :]


def self_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """History of the non-terrain (robot-on-robot) contact force per selected body, ``[B, H, N, 3]``."""
    data = env.scene[sensor_cfg.name].data
    return (data.net_forces_w_history - data.force_matrix_w_history.sum(dim=-2))[..., sensor_cfg.body_ids, :]


def contact_state(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terrain contact per selected body (mjlab's ``found > 0``), ``[B, N]``.

    Uses the sensor's own ``force_threshold``, the one it maintains ``current_air_time`` with, so
    every contact-based term agrees with the sensor's air-time bookkeeping.
    """
    threshold = env.scene[sensor_cfg.name].cfg.force_threshold
    return torch.norm(terrain_forces(env, sensor_cfg), dim=-1) > threshold


# Cached subsample indices, keyed by (total, max_points, device).
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
    """Ray hit positions ``[B, N, 3]`` and validity ``[B, N]``, at most ``max_points`` per sensor."""
    all_points, all_valid = [], []
    for name in sensor_names:
        hits = env.scene[name].data.ray_hits_w  # [B, N, 3]
        valid = torch.isfinite(hits).all(dim=-1)
        if hits.shape[1] > max_points:
            idx = _subsample_indices(hits.shape[1], max_points, hits.device)
            hits, valid = hits[:, idx], valid[:, idx]
        all_points.append(hits)
        all_valid.append(valid)
    return torch.cat(all_points, dim=1), torch.cat(all_valid, dim=1)


def clearance_samples(env: ManagerBasedRLEnv, sensor_name: str, max_distance: float) -> torch.Tensor:
    """Clearance below one ray-caster, ``[B, 1, N]``. Missed rays (non-finite hits) read ``max_distance``."""
    data = env.scene[sensor_name].data
    frame_z = data.pos_w[:, 2].view(-1, 1, 1)
    hit_z = data.ray_hits_w[..., 2].unsqueeze(1)
    miss = ~torch.isfinite(hit_z)
    return heights_from_raycast(frame_z, torch.nan_to_num(hit_z, nan=0.0, posinf=0.0, neginf=0.0), miss, max_distance)


def stack_clearance_samples(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    max_distance: float,
) -> torch.Tensor:
    """Clearance below several ray-casters, ``[B, F, N]``; ``sensor_names`` fixes the frame order."""
    return torch.cat([clearance_samples(env, name, max_distance) for name in sensor_names], dim=1)
