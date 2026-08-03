"""Velocity command that is fixed in the **world** frame.

Isaac Lab's stock ``UniformVelocityCommand`` samples and stores a body-frame command, so
"go at 1 m/s forward" means "forward from wherever you happen to be pointing right now".
A robot can then satisfy it while continuously turning.

This term samples the linear velocity in the *commanded heading* frame, converts it once
into a fixed world-frame vector, and exposes the body-frame projection to the policy.
Tracking is scored in the world frame, so the robot has to turn to face the direction of
travel and then hold it -- which is what makes the paired heading rewards meaningful and
what a navigation stack issuing world-frame goals actually needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from bipedal_locomotion.utils.math import wrap_to_pi
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import WheeledUniformVelocityCommandCfg


class WheeledUniformVelocityCommand(CommandTerm):
    """Uniform velocity command stored in the world frame."""

    cfg: WheeledUniformVelocityCommandCfg

    def __init__(self, cfg: WheeledUniformVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        if cfg.heading_command and cfg.ranges.heading is None:
            raise ValueError("heading_command=True but ranges.heading is None.")
        if cfg.ranges.heading is not None and not cfg.heading_command:
            raise ValueError("ranges.heading is set but heading_command=False.")

        self.robot: Articulation = env.scene[cfg.asset_name]

        # Linear velocity is sampled in the target-heading frame, stored in the world
        # frame, and exposed to the policy in the body frame. The third column of the
        # world command carries the yaw rate derived from the heading error.
        self.vel_command_h = torch.zeros(self.num_envs, 2, device=self.device)
        self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, device=self.device)

        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.is_forward_env = torch.zeros_like(self.is_heading_env)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"WheeledUniformVelocityCommand: resampling every {self.cfg.resampling_time_range} s"

    @property
    def command(self) -> torch.Tensor:
        """Body-frame command exposed to the policy: ``[vx_b, vy_b, yaw_rate]``."""
        return self.vel_command_b

    @property
    def command_w(self) -> torch.Tensor:
        """World-frame linear command plus the yaw-rate command."""
        return self.vel_command_w

    def _update_metrics(self) -> None:
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_w[:, :2] - self.robot.data.root_link_lin_vel_w[:, :2], dim=-1)
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        r = torch.empty(len(env_ids), device=self.device)

        linear_command = self.vel_command_h if self.cfg.heading_command else self.vel_command_w
        linear_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        linear_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        if self.cfg.heading_command:
            assert self.cfg.ranges.heading is not None
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # A fraction of environments get a clean, axis-aligned, straight-line command.
        # Uniformly sampled commands almost never contain "drive straight along +x with
        # no yaw", which is both the most common real request and the case where a wheeled
        # robot should stop stepping and just roll -- so it is sampled explicitly.
        self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
        fwd_ids = env_ids[self.is_forward_env[env_ids]]
        if len(fwd_ids) > 0:
            speed = linear_command[fwd_ids, 0].abs().clamp(min=0.3)
            direction = torch.empty(len(fwd_ids), device=self.device).uniform_(0.0, 1.0)
            positive_x = direction < 0.5
            positive_y = (direction >= 0.5) & (direction < 0.75)

            if self.cfg.heading_command:
                self.vel_command_h[fwd_ids, 0] = speed
                self.vel_command_h[fwd_ids, 1] = 0.0
                self.heading_target[fwd_ids] = torch.where(
                    positive_x,
                    torch.zeros_like(speed),
                    torch.where(positive_y, torch.full_like(speed, torch.pi / 2), torch.full_like(speed, -torch.pi / 2)),
                )
            else:
                self.vel_command_w[fwd_ids, 0] = torch.where(positive_x, speed, torch.zeros_like(speed))
                self.vel_command_w[fwd_ids, 1] = torch.where(
                    positive_y, speed, torch.where(positive_x, torch.zeros_like(speed), -speed)
                )
            self.vel_command_w[fwd_ids, 2] = 0.0

        self._update_command()

        # Optionally start the episode already moving at the commanded velocity. Without
        # this every episode begins with an acceleration transient, and the policy spends
        # most of its samples on a regime it will rarely be in during deployment.
        if self.cfg.init_velocity_prob > 0.0:
            init_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
            init_ids = env_ids[init_mask]
            if len(init_ids) > 0:
                root_state = self.robot.data.root_state_w[init_ids].clone()
                root_state[:, 7:9] = self.vel_command_w[init_ids, :2]
                root_state[:, 12] = self.vel_command_b[init_ids, 2]
                self.robot.write_root_state_to_sim(root_state, init_ids)

    def _update_command(self) -> None:
        self.vel_command_b[:, 2] = self.vel_command_w[:, 2]

        if self.cfg.heading_command:
            # Rotate the heading-frame sample into a world-frame vector, once. From here
            # on the command is fixed in the world and the robot must turn to meet it.
            cos_target = torch.cos(self.heading_target)
            sin_target = torch.sin(self.heading_target)
            vx_h = self.vel_command_h[:, 0]
            vy_h = self.vel_command_h[:, 1]
            self.vel_command_w[:, 0] = cos_target * vx_h - sin_target * vy_h
            self.vel_command_w[:, 1] = sin_target * vx_h + cos_target * vy_h

            self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            yaw_rate_cmd = torch.clip(
                self.cfg.heading_control_stiffness * self.heading_error[env_ids],
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
            self.vel_command_w[env_ids, 2] = yaw_rate_cmd
            self.vel_command_b[env_ids, 2] = yaw_rate_cmd

        # Project the fixed world command into the robot's current yaw frame.
        heading = self.robot.data.heading_w
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        vx_w = self.vel_command_w[:, 0]
        vy_w = self.vel_command_w[:, 1]
        self.vel_command_b[:, 0] = cos_h * vx_w + sin_h * vy_w
        self.vel_command_b[:, 1] = -sin_h * vx_w + cos_h * vy_w

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_h[standing_env_ids, :] = 0.0
        self.vel_command_b[standing_env_ids, :] = 0.0
        self.vel_command_w[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        # Visualisation is not ported; the world-frame command is inspectable from the
        # logged metrics and the play script's overlay.
        pass

    def _debug_vis_callback(self, event) -> None:
        pass
