"""Gym registrations for every locomotion task in this extension.

Both entry points are registered as ``"module.path:ClassName"`` strings rather
than as pre-built objects. ``load_cfg_from_registry`` then instantiates a fresh
config per ``gym.make``, so a task and its ``-Play-v0`` twin cannot mutate each
other's configuration -- which they did when the runner cfg was shared as a
single module-level instance.
"""

import gymnasium as gym

from bipedal_locomotion.tasks.locomotion import agents

from . import wf_tron1b_env_cfg  # noqa: F401  (imported for side-effect-free discoverability)

_ENV_CFG = wf_tron1b_env_cfg.__name__
_AGENTS = agents.__name__

##
# WF-TRON1B velocity tracking -- flat terrain
##

gym.register(
    id="Isaac-Limx-WF-TRON1B-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:WF_TRON1B_FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:WFTron1BPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Limx-WF-TRON1B-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:WF_TRON1B_FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:WFTron1BPPORunnerCfg",
    },
)

##
# WF-TRON1B velocity tracking -- generated rough terrain
##

gym.register(
    id="Isaac-Limx-WF-TRON1B-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:WF_TRON1B_RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:WFTron1BPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Limx-WF-TRON1B-Velocity-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV_CFG}:WF_TRON1B_RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:WFTron1BPPORunnerCfg",
    },
)
