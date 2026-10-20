"""Play a trained RL agent with the vendored RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

VENDORED_RSL_RL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rsl_rl"))
sys.path.insert(0, VENDORED_RSL_RL)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Explicit checkpoint file to load.")
parser.add_argument("--steps", type=int, default=None, help="Stop after this many steps instead of running forever.")
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export ONNX and exit.")
parser.add_argument(
    "--policy_role",
    type=str,
    default="student",
    choices=("student", "teacher"),
    help=(
        "Which path of a teacher-student policy to run. The student sees only "
        "proprioception and is what deploys; the teacher sees privileged observations "
        "and is useful for checking how much the distillation gave up."
    ),
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import dataclasses
from pathlib import Path

import gymnasium as gym
import torch

import rsl_rl

_RESOLVED_RSL_RL = Path(rsl_rl.__file__).resolve()
assert str(_RESOLVED_RSL_RL).startswith(str(Path(VENDORED_RSL_RL).resolve())), (
    f"Expected the vendored rsl_rl under {VENDORED_RSL_RL}, but imported {_RESOLVED_RSL_RL}."
)

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import bipedal_locomotion  # noqa: F401  (importing registers the gym tasks)
from bipedal_locomotion.utils.wrappers.rsl_rl import (
    WheeledLeggedRslRlVecEnvWrapper,
    WheeledLeggedVelocityOnPolicyRunner,
)


def _select_policy(policy, policy_role: str):
    """The student (deployed) or teacher (privileged) path of the teacher-student policy."""
    if policy_role == "teacher":
        return lambda obs: policy.act_teacher(obs, stochastic_output=False)
    return policy


def main() -> None:
    env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = args_cli.checkpoint_path or get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(os.path.dirname(resume_path), "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = WheeledLeggedRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = WheeledLeggedVelocityOnPolicyRunner(
        env, dataclasses.asdict(agent_cfg), log_dir=None, device=args_cli.device
    )
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    if args_cli.export_onnx:
        onnx_path = runner.export_onnx_with_metadata(os.path.join(os.path.dirname(resume_path), "exported"))
        print(f"[INFO] Exported ONNX policy with deployment metadata to: {onnx_path}")
        env.close()
        return

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    act = _select_policy(policy, args_cli.policy_role)

    obs = env.get_observations()
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = act(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)  # recurrent (depth) policies drop the hidden state of finished episodes
        step += 1
        if args_cli.steps is not None and step >= args_cli.steps:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
