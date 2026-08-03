"""Wrappers and configuration for the vendored RSL-RL (``rsl-rl-lib`` 5.3.0).

``isaaclab_rl.rsl_rl`` is deliberately **not** re-exported here. Its config
dataclasses and env wrapper target whichever ``rsl-rl-lib`` the installed Isaac
Lab pins, which uses a ``policy:`` / ``empirical_normalization`` layout and a
plain-dict observation contract. The vendored 5.3.0 fork uses ``actor:`` /
``critic:`` / ``obs_groups:`` and a TensorDict of named observation groups.
Importing both makes it easy to build the wrong one and only find out deep
inside ``construct_algorithm``.
"""

from .exporter import attach_metadata_to_onnx, get_wheeled_legged_metadata
from .rl_cfg import (
    RslRlBaseRunnerCfg,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from .runner import WheeledLeggedVelocityOnPolicyRunner
from .vecenv_wrapper import (
    RslRlVecEnvWrapper,
    WheeledLeggedRslRlVecEnvWrapper,
    get_applied_actions,
)

__all__ = [
    "RslRlBaseRunnerCfg",
    "RslRlModelCfg",
    "RslRlOnPolicyRunnerCfg",
    "RslRlPpoAlgorithmCfg",
    "RslRlVecEnvWrapper",
    "WheeledLeggedRslRlVecEnvWrapper",
    "WheeledLeggedVelocityOnPolicyRunner",
    "attach_metadata_to_onnx",
    "get_applied_actions",
    "get_wheeled_legged_metadata",
]
