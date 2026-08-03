"""Shared RSL-RL command-line handling."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add the RSL-RL arguments shared by train and play."""
    group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agents.")
    group.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment log directory.")
    group.add_argument("--run_name", type=str, default=None, help="Suffix appended to the run directory name.")
    group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    group.add_argument("--load_run", type=str, default=None, help="Run directory to load. Supports regex.")
    group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load. Supports regex.")
    group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger backend."
    )
    group.add_argument("--log_project_name", type=str, default=None, help="W&B / Neptune project name.")


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Load the registered runner config for a task and apply CLI overrides."""
    from isaaclab_tasks.utils import load_cfg_from_registry

    rslrl_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    return update_rsl_rl_cfg(rslrl_cfg, args_cli)


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Apply the CLI overrides that were actually provided."""
    if getattr(args_cli, "seed", None) is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if getattr(args_cli, "max_iterations", None) is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if getattr(args_cli, "save_interval", None) is not None:
        agent_cfg.save_interval = args_cli.save_interval
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
    return agent_cfg
