"""Termination terms for the wheeled-legged WF-TRON1B velocity task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ROBOT = SceneEntityCfg("robot")

__all__ = ["non_finite_physics", "out_of_terrain_bounds"]


def non_finite_physics(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Terminate environments whose physics state has gone non-finite.

    A single NaN in one environment propagates through the shared reward and advantage
    statistics and destroys the whole update, so a diverged environment has to be cut out
    of the batch immediately rather than left to be caught by a downstream check.

    Checks the articulation root and joint state plus every contact sensor's forces --
    the places divergence has actually shown up on stiff terrain.
    """
    asset = env.scene[asset_cfg.name]
    bad = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    for tensor in (
        asset.data.root_state_w,
        asset.data.joint_pos,
        asset.data.joint_vel,
        asset.data.joint_acc,
        asset.data.applied_torque,
    ):
        if tensor is not None:
            bad |= ~torch.isfinite(tensor).reshape(env.num_envs, -1).all(dim=1)

    for sensor in env.scene.sensors.values():
        forces = getattr(sensor.data, "net_forces_w", None)
        if forces is not None:
            bad |= ~torch.isfinite(forces).reshape(env.num_envs, -1).all(dim=1)

    return bad


def out_of_terrain_bounds(
    env: ManagerBasedRLEnv,
    margin: float = 0.05,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Truncate once the robot has driven off the generated terrain patch grid.

    Registered as a time-out rather than a failure: leaving the terrain is not the
    robot's mistake, and bootstrapping through it avoids teaching the policy that the
    edge of the world is a bad place to be.

    Returns all-false for non-generated terrain, where there is no grid to leave.
    """
    terrain = env.scene.terrain
    zeros = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    if terrain is None or getattr(terrain, "terrain_origins", None) is None:
        return zeros

    generator_cfg = getattr(terrain.cfg, "terrain_generator", None)
    if generator_cfg is None:
        return zeros

    asset = env.scene[asset_cfg.name]
    root_xy_w = asset.data.root_link_pos_w[:, :2]

    # The realised grid shape, not the configured one: curriculum mode overrides
    # num_cols with the number of sub-terrain entries.
    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    limit_x = 0.5 * (num_rows * generator_cfg.size[0]) + margin
    limit_y = 0.5 * (num_cols * generator_cfg.size[1]) + margin
    return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)
