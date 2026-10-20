"""Velocity command that is fixed in the **world** frame (port of mjlab's ``UniformVelocityCommand``).

Isaac Lab's stock ``UniformVelocityCommand`` stores a body-frame command, which a robot can satisfy
while continuously turning. This term samples the linear velocity in the commanded-heading frame,
converts it once into a fixed world-frame vector, and exposes its body-frame projection to the
policy. Tracking is scored in the world frame, so the robot has to turn to the heading and hold it.
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

        # Linear velocity is sampled in the target-heading frame, stored in the world frame and
        # exposed in the body frame. The third world column holds the heading-derived yaw rate.
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
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        # Straight-line envs face +world-x (50%), +world-y (25%) or -world-y (25%) and get a
        # positive forward command in that heading frame.
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
                    positive_x, 0.0, torch.where(positive_y, torch.pi / 2, -torch.pi / 2)
                )
            else:
                self.vel_command_w[fwd_ids, 0] = torch.where(positive_x, speed, 0.0)
                self.vel_command_w[fwd_ids, 1] = torch.where(positive_y, speed, torch.where(positive_x, 0.0, -speed))
            self.vel_command_w[fwd_ids, 2] = 0.0

        self._update_command(env_ids)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.vel_command_b[ids, 2] = self.vel_command_w[ids, 2]
        if self.cfg.heading_command:
            # Rotate the heading-frame sample into one fixed world-frame vector.
            cos_target = torch.cos(self.heading_target[ids])
            sin_target = torch.sin(self.heading_target[ids])
            vx_h = self.vel_command_h[ids, 0]
            vy_h = self.vel_command_h[ids, 1]
            self.vel_command_w[ids, 0] = cos_target * vx_h - sin_target * vy_h
            self.vel_command_w[ids, 1] = sin_target * vx_h + cos_target * vy_h

            self.heading_error[ids] = wrap_to_pi(self.heading_target[ids] - self.robot.data.heading_w[ids])
            heading_ids = (
                self.is_heading_env.nonzero(as_tuple=False).flatten()
                if env_ids is None
                else env_ids[self.is_heading_env[env_ids]]
            )
            yaw_rate_cmd = torch.clip(
                self.cfg.heading_control_stiffness * self.heading_error[heading_ids],
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
            self.vel_command_w[heading_ids, 2] = yaw_rate_cmd
            self.vel_command_b[heading_ids, 2] = yaw_rate_cmd

        # Project the fixed world command into the robot's yaw frame for the policy.
        heading = self.robot.data.heading_w[ids]
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        vx_w = self.vel_command_w[ids, 0]
        vy_w = self.vel_command_w[ids, 1]
        self.vel_command_b[ids, 0] = cos_h * vx_w + sin_h * vy_w
        self.vel_command_b[ids, 1] = -sin_h * vx_w + cos_h * vy_w

        standing_ids = (
            self.is_standing_env.nonzero(as_tuple=False).flatten()
            if env_ids is None
            else env_ids[self.is_standing_env[env_ids]]
        )
        self.vel_command_h[standing_ids, :] = 0.0
        self.vel_command_b[standing_ids, :] = 0.0
        self.vel_command_w[standing_ids, :] = 0.0
