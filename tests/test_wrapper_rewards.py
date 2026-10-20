"""The RSL-RL wrapper zeroes non-finite rewards (mjlab's reward manager zeroes non-finite terms)."""

import math
from types import SimpleNamespace

import torch

from _loader import load_wrapper

wrapper = load_wrapper("vecenv_wrapper")


def test_non_finite_rewards_are_zeroed():
    rew = torch.tensor([1.5, math.nan, math.inf, -math.inf])
    done = torch.zeros(4, dtype=torch.bool)
    env = SimpleNamespace(cfg=SimpleNamespace(is_finite_horizon=False))
    env.step = lambda actions: ({"policy": actions}, rew, done, done, {})
    env.unwrapped = env
    vec_env = object.__new__(wrapper.RslRlVecEnvWrapper)  # skip __init__, which resets a real env
    vec_env.env, vec_env.clip_actions, vec_env.num_envs = env, None, 4

    _, out, _, _ = vec_env.step(torch.zeros(4, 1))
    assert torch.equal(out, torch.tensor([1.5, 0.0, 0.0, 0.0]))
