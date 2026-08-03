"""Per-environment domain-randomisation state, readable back as an observation.

Isaac Lab's stock randomisation events write into the simulator and forget what they
wrote. The teacher-student tasks need the opposite: a privileged encoder is fed the
*actual* randomised parameters so it can learn to infer them from proprioception alone.
So the sampled values are also recorded here, on the environment, and read back by the
``domain_randomization_delta_quantity`` observation.

The buffers are allocated lazily on first use so that a task which registers no
randomisation events pays nothing and reports zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_ATTR = "_wf_tron1b_dr_state"


@dataclass
class DomainRandomizationState:
    """Raw (un-normalised) randomised quantities, one row per environment."""

    wheel_friction: torch.Tensor
    """Static friction coefficient sampled per wheel body. [num_envs, num_wheels]"""
    encoder_bias: torch.Tensor
    """Additive joint-position measurement bias, in radians. [num_envs, num_joints]"""
    base_com_delta: torch.Tensor
    """Base centre-of-mass offset from nominal, in metres. [num_envs, 3]"""

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.wheel_friction[env_ids] = 0.0
        self.encoder_bias[env_ids] = 0.0
        self.base_com_delta[env_ids] = 0.0


def get_dr_state(env: ManagerBasedRLEnv, num_wheels: int = 2) -> DomainRandomizationState:
    """Return the environment's randomisation state, creating it on first access."""
    state = getattr(env, _ATTR, None)
    if state is None:
        num_joints = env.scene["robot"].num_joints
        state = DomainRandomizationState(
            wheel_friction=torch.zeros(env.num_envs, num_wheels, device=env.device),
            encoder_bias=torch.zeros(env.num_envs, num_joints, device=env.device),
            base_com_delta=torch.zeros(env.num_envs, 3, device=env.device),
        )
        setattr(env, _ATTR, state)
    return state
