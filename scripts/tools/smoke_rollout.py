"""Random-action rollout that checks the things unit tests cannot.

The Isaac-Sim-free suite in ``tests/`` pins the maths. This script pins the wiring: that
the observation groups have the widths the runner config assumes, that nothing is
NaN, and -- the one most likely to be silently wrong -- that a standing robot actually
reports wheel contact. If contact reporting returns zeros, every contact-based reward
still computes, the run still trains, and the resulting policy is shaped by a robot that
believes it is permanently airborne.

Usage:
    ${ISAACLAB}/isaaclab.sh -p scripts/tools/smoke_rollout.py \
        --task Isaac-Limx-WF-TRON1B-Velocity-Flat-v0 --num_envs 16 --steps 100 --headless
"""

import argparse
import os
import sys

VENDORED_RSL_RL = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rsl_rl"))
sys.path.insert(0, VENDORED_RSL_RL)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random-action smoke rollout.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--steps", type=int, default=100, help="Number of environment steps.")
parser.add_argument(
    "--expect-critic-dim",
    type=int,
    default=None,
    help="Assert the critic group width (42 for flat, 166 for rough).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import bipedal_locomotion  # noqa: F401  (importing registers the gym tasks)
from bipedal_locomotion.utils.wrappers.rsl_rl import WheeledLeggedRslRlVecEnvWrapper

# Widths the runner configs and the exported deployment metadata assume.
EXPECTED_ACTOR_DIM = 31
EXPECTED_DYNAMICS_CONTEXT_DIM = 87
EXPECTED_HISTORY_LENGTH = 5
EXPECTED_NUM_ACTIONS = 8


def main() -> int:
    env_cfg = parse_env_cfg(task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = WheeledLeggedRslRlVecEnvWrapper(env)

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    obs = env.get_observations()
    print("\n=== observation groups ===")
    for key in sorted(obs.keys()):
        print(f"  {key:<20} {tuple(obs[key].shape)}")

    print("\n=== action manager ===")
    action_terms = list(env.unwrapped.action_manager.active_terms)
    print(f"  terms: {action_terms}")
    print(f"  total_action_dim: {env.num_actions}")

    print("\n=== observation term order ===")
    for group, terms in env.unwrapped.observation_manager.active_terms.items():
        print(f"  {group}: {list(terms)}")

    obs_terms = env.unwrapped.observation_manager.active_terms
    # The history group is declared as a subclass of the proprio group so the two cannot
    # drift apart. Assert it, because a configclass inheritance change would silently
    # give the student a different observation layout than the actor.
    check(
        list(obs_terms["actor"]) == list(obs_terms["actor_history"]),
        f"actor terms {list(obs_terms['actor'])} != actor_history terms {list(obs_terms['actor_history'])}",
    )
    check(action_terms == ["leg_pos", "wheel_vel"], f"action term order is {action_terms}, expected ['leg_pos', 'wheel_vel']")
    check(env.num_actions == EXPECTED_NUM_ACTIONS, f"num_actions is {env.num_actions}, expected {EXPECTED_NUM_ACTIONS}")
    check(obs["actor"].shape[-1] == EXPECTED_ACTOR_DIM, f"actor width is {obs['actor'].shape[-1]}, expected {EXPECTED_ACTOR_DIM}")
    check(
        tuple(obs["actor_history"].shape[1:]) == (EXPECTED_HISTORY_LENGTH, EXPECTED_ACTOR_DIM),
        f"actor_history shape is {tuple(obs['actor_history'].shape[1:])}, "
        f"expected ({EXPECTED_HISTORY_LENGTH}, {EXPECTED_ACTOR_DIM})",
    )
    check(
        obs["dynamics_context"].shape[-1] == EXPECTED_DYNAMICS_CONTEXT_DIM,
        f"dynamics_context width is {obs['dynamics_context'].shape[-1]}, expected {EXPECTED_DYNAMICS_CONTEXT_DIM}",
    )
    if args_cli.expect_critic_dim is not None:
        check(
            obs["critic"].shape[-1] == args_cli.expect_critic_dim,
            f"critic width is {obs['critic'].shape[-1]}, expected {args_cli.expect_critic_dim}",
        )

    print(f"\n=== rolling out {args_cli.steps} steps with random actions ===")
    contact_fractions = []
    episode_lengths = []
    for step in range(args_cli.steps):
        actions = torch.rand(env.num_envs, env.num_actions, device=env.device) * 2.0 - 1.0
        obs, rew, dones, extras = env.step(actions)

        for key, value in obs.items():
            check(bool(torch.isfinite(value).all()), f"non-finite values in observation group '{key}' at step {step}")
        check(bool(torch.isfinite(rew).all()), f"non-finite reward at step {step}")
        check("applied_actions" in extras, f"extras is missing 'applied_actions' at step {step}")

        sensor = env.unwrapped.scene["contact_forces"]
        wheel_ids, _ = sensor.find_bodies(["wheel_L_Link", "wheel_R_Link"], preserve_order=True)
        forces = torch.norm(sensor.data.net_forces_w[:, wheel_ids], dim=-1)
        contact_fractions.append((forces > 1.0).float().mean().item())
        episode_lengths.append(env.unwrapped.episode_length_buf.float().mean().item())

    mean_contact = sum(contact_fractions) / len(contact_fractions)
    mean_episode_length = sum(episode_lengths) / len(episode_lengths)
    print(f"  mean wheel-contact fraction: {mean_contact:.3f}")
    print(f"  mean episode length (steps): {mean_episode_length:.1f}")

    # Under random actions the robot falls over often, so this is a floor rather than a
    # target: a value near zero means contact reporting is broken, not that the policy is bad.
    check(mean_contact > 0.05, f"mean wheel-contact fraction is {mean_contact:.3f}; contact reporting is likely broken")
    # Visual meshes imported as colliders would make illegal-contact fire immediately.
    check(mean_episode_length > 5.0, f"mean episode length is {mean_episode_length:.1f}; terminations fire far too early")

    env.close()

    print()
    if failures:
        print(f"=== FAILED ({len(failures)}) ===")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("=== all smoke checks passed ===")
    return 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    sys.exit(exit_code)
