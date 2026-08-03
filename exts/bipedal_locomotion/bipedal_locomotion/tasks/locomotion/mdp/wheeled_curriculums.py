"""Curriculum terms for the wheeled-legged WF-TRON1B velocity task."""

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
) -> torch.Tensor:
    """Promote or demote each environment's terrain difficulty by distance travelled.

    Deliberately not Isaac Lab's ``mdp.terrain_levels_vel``: that one hard-codes the
    command name ``"base_velocity"``, and this task's command is named ``twist``. Reusing
    it would raise at the first curriculum step.

    Promotion needs the robot to have crossed half a patch; demotion triggers when it
    covered less than half of what the command asked for, so an environment that is
    simply commanded slowly is not punished for it.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command(command_name)

    distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def fell_over_limit_angle(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    termination_term_name: str,
    initial_limit_angle: float,
    final_limit_angle: float,
    ramp_steps: int,
) -> torch.Tensor:
    """Relax the fall-over angle limit as training progresses.

    Early on, a tight limit ends an episode as soon as the robot tips, which keeps the
    rollout buffer full of recoverable states instead of long tails of a robot lying on
    its side. Later the limit is relaxed so the policy can learn to recover from
    attitudes that would previously have been cut short.

    The threshold is global rather than per-environment, so ``env_ids`` is ignored.

    Note that this mutates the live termination term's ``params`` dict. That only works
    because the termination manager reads ``params`` at call time; the returned value is
    logged so a regression to snapshot-at-startup behaviour shows up as the logged angle
    ramping while behaviour does not change.
    """
    del env_ids  # the threshold is shared by all environments

    progress = min(max(float(env.common_step_counter) / max(ramp_steps, 1), 0.0), 1.0)
    limit_angle = initial_limit_angle + progress * (final_limit_angle - initial_limit_angle)

    term_cfg = env.termination_manager.get_term_cfg(termination_term_name)
    term_cfg.params["limit_angle"] = limit_angle

    # Read back rather than returning the value we just computed: if a future Isaac Lab
    # binds term params at startup, these two diverge and the log makes that visible.
    return torch.tensor(term_cfg.params["limit_angle"], device=env.device)
