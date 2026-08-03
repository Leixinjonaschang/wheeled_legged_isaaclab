"""RSL-RL ``VecEnv`` adapter for Isaac Lab's :class:`ManagerBasedRLEnv`.

This replaces ``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper``. That wrapper is written
against whichever ``rsl-rl-lib`` the installed Isaac Lab pins and returns a plain
dict of observations; the vendored 5.3.0 fork expects a single
:class:`~tensordict.TensorDict` of named observation groups plus ``env.cfg``,
``env.num_actions`` and a writable ``episode_length_buf``.

Isaac Lab's ``ManagerBasedRLEnv.step()`` returns the same
``(obs_dict, rew, terminated, truncated, extras)`` 5-tuple that the mjlab
environment does, so the group plumbing is a direct correspondence rather than a
translation.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ActionManager


def get_applied_actions(action_manager: ActionManager) -> torch.Tensor:
    """Concatenate the policy-space action each term actually applied this step.

    Terms that model actuation delay expose ``applied_action``: the mean, over the
    physics substeps of one environment step, of the *raw* (pre-scale) action that
    was really sent to the simulator. Terms without delay applied their raw action
    unchanged, so ``raw_actions`` is the correct fallback.

    Concatenation follows ``action_manager.active_terms`` order, matching the layout
    of the action vector the policy produces.
    """
    chunks = []
    for name in action_manager.active_terms:
        term = action_manager.get_term(name)
        chunks.append(getattr(term, "applied_action", term.raw_actions))
    return torch.cat(chunks, dim=-1)


class RslRlVecEnvWrapper(VecEnv):
    """Wraps a :class:`ManagerBasedRLEnv` for the vendored RSL-RL runner."""

    def __init__(self, env: ManagerBasedRLEnv, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions

        self.num_envs = self.unwrapped.num_envs
        self.device = torch.device(self.unwrapped.device)
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self._modify_action_space()

        # Reset at the start since rsl_rl does not call reset().
        self.env.reset()

    @property
    def cfg(self) -> ManagerBasedRLEnvCfg:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        # rsl_rl writes this directly when ``init_at_random_ep_len`` is enabled.
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def get_observations(self) -> TensorDict:
        obs_dict = self.unwrapped.observation_manager.compute()
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # Only expose time-outs when the task is treated as infinite-horizon, i.e. when a
        # time-out is an artifact of episode chunking and the value function should
        # bootstrap through it rather than see a terminal state.
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self) -> None:
        return self.env.close()

    def _modify_action_space(self) -> None:
        if self.clip_actions is None:
            return
        self.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.unwrapped.action_space = gym.vector.utils.batch_space(
            self.unwrapped.single_action_space, self.num_envs
        )


class WheeledLeggedRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Adds ``extras["applied_actions"]`` for the latent-dynamics predictor variants.

    ``RepresentationVelocityPredictorTeacherStudentPPO`` conditions its latent-dynamics
    model on the action that was *actually* executed, which -- with per-environment
    actuation delay -- is not the action the policy emitted this step. The algorithm
    raises rather than silently training on the wrong signal, so this key must be
    present for those tasks.
    """

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rew, dones, extras = super().step(actions)
        extras["applied_actions"] = get_applied_actions(self.unwrapped.action_manager)
        return obs, rew, dones, extras
