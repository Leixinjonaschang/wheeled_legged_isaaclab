"""Train an RL agent with the vendored RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# The vendored fork must win over any pip-installed rsl_rl that Isaac Lab pulled in.
VENDORED_RSL_RL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rsl_rl"))
sys.path.insert(0, VENDORED_RSL_RL)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=24000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of iterations to train.")
parser.add_argument("--save_interval", type=int, default=None, help="Number of iterations between saves.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Explicit checkpoint file to resume from.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dataclasses
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch

import rsl_rl

# Guard against Isaac Lab having already bound a different rsl_rl: the 5.3.0 config
# layout is incompatible with the older forks, and the mismatch would otherwise surface
# as a confusing TypeError deep inside construct_algorithm.
_RESOLVED_RSL_RL = Path(rsl_rl.__file__).resolve()
assert str(_RESOLVED_RSL_RL).startswith(str(Path(VENDORED_RSL_RL).resolve())), (
    f"Expected the vendored rsl_rl under {VENDORED_RSL_RL}, but imported {_RESOLVED_RSL_RL}."
)

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import bipedal_locomotion  # noqa: F401  (importing registers the gym tasks)
from bipedal_locomotion.utils.wrappers.rsl_rl import (
    WheeledLeggedRslRlVecEnvWrapper,
    WheeledLeggedVelocityOnPolicyRunner,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main() -> None:
    env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, run_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment to: {log_dir}")

    env_cfg.seed = agent_cfg.seed
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), dataclasses.asdict(agent_cfg))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # extras["applied_actions"] feeds the latent-dynamics predictor (Depth-Predict-OursGP); BlindGP ignores it.
    env = WheeledLeggedRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = WheeledLeggedVelocityOnPolicyRunner(
        env, dataclasses.asdict(agent_cfg), log_dir=log_dir, device=args_cli.device
    )

    if agent_cfg.resume:
        resume_path = args_cli.checkpoint_path or get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
