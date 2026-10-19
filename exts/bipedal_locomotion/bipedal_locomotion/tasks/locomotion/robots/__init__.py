"""Gym registrations for the WF-TRON1B locomotion tasks (mjlab's ``register_mjlab_task`` calls).

Entry points are ``"module:attr"`` strings, so ``load_cfg_from_registry`` builds a fresh config
per ``gym.make`` and a task cannot mutate its ``-Play-v0`` twin's configuration.
"""

import gymnasium as gym

from bipedal_locomotion.tasks.locomotion import agents

from . import wf_tron1b_env_cfg

_ENV_CFG = wf_tron1b_env_cfg.__name__
_AGENTS = agents.__name__


def _register(task_id: str, env_cfg: str, rsl_rl_cfg: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{_ENV_CFG}:{env_cfg}", "rsl_rl_cfg_entry_point": f"{_AGENTS}:{rsl_rl_cfg}"},
    )


# mjlab: Mjlab-Velocity-Rough-WF-Tron1B-RepTS-LinVel-BlindGP
_register(
    "Isaac-Limx-WF-TRON1B-Velocity-Rough-RepTS-LinVel-BlindGP-v0",
    "WF_TRON1B_RepTSLinVelEnvCfg",
    "wf_tron1b_rep_ts_lin_vel_runner_cfg",
)
_register(
    "Isaac-Limx-WF-TRON1B-Velocity-Rough-RepTS-LinVel-BlindGP-Play-v0",
    "WF_TRON1B_RepTSLinVelEnvCfg_PLAY",
    "wf_tron1b_rep_ts_lin_vel_runner_cfg",
)
