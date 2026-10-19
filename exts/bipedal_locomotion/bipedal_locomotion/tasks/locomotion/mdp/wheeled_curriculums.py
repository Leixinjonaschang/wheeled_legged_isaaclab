"""Curriculum terms for the WF-TRON1B velocity task (ports of mjlab ``tasks/velocity/mdp/curriculums.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["fell_over_limit_angle", "terrain_levels_vel"]


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Promote environments that crossed half a patch, demote those that covered less than half the
    commanded distance. Unlike Isaac Lab's version, the command name is configurable.

    Returns the mean / max level and, when every sub-terrain owns one column, the mean level per
    terrain type (repeated ``name__i`` columns share one entry).
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    generator_cfg = terrain.cfg.terrain_generator
    command = env.command_manager.get_command(command_name)

    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > generator_cfg.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # The initial reset has no completed episode to evaluate.
    if env.common_step_counter > 0:
        terrain.update_env_origins(env_ids, move_up, move_down)

    levels = terrain.terrain_levels.float()
    result = {"mean": levels.mean(), "max": levels.max()}
    names = list(generator_cfg.sub_terrains)
    if terrain.terrain_origins.shape[1] == len(names):
        for logical_name in dict.fromkeys(name.split("__")[0] for name in names):
            columns = [i for i, name in enumerate(names) if name.split("__")[0] == logical_name]
            mask = torch.isin(terrain.terrain_types, torch.tensor(columns, device=env.device))
            if mask.any():
                result[logical_name] = levels[mask].mean()
    return result


def fell_over_limit_angle(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    termination_term_name: str,
    initial_limit_angle: float,
    final_limit_angle: float,
    ramp_steps: int,
) -> torch.Tensor:
    """Linearly relax the shared fall-over angle limit over global policy steps."""
    del env_ids  # the threshold is shared by all environments
    progress = min(max(float(env.common_step_counter) / max(ramp_steps, 1), 0.0), 1.0)
    limit_angle = initial_limit_angle + progress * (final_limit_angle - initial_limit_angle)
    term_cfg = env.termination_manager.get_term_cfg(termination_term_name)
    term_cfg.params["limit_angle"] = limit_angle
    return torch.tensor(limit_angle, device=env.device)
