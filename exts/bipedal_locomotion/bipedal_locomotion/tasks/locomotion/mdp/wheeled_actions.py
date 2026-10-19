"""Joint action terms with per-environment actuation delay.

Real actuators do not execute a command the instant it is issued: there is bus latency,
firmware scheduling and driver lag between the policy emitting an action and the joint
starting to move. Randomising that lag per environment is what makes a policy tolerate
the real robot's timing instead of exploiting the simulator's zero-latency response.

The delay is applied here, at the *action term*, rather than inside a delayed actuator,
for three reasons:

1. The latent-dynamics predictor needs ``applied_action`` -- the policy-space action
   actually executed, averaged over the physics substeps of one control step. An
   actuator delays already-scaled joint targets, from which the policy-space value
   cannot be recovered.
2. Legs and wheels must share one delay draw. They are on the same bus on the real
   robot; letting them drift apart by up to four physics steps would be a different
   machine. Delay groups make the sharing explicit.
3. Encoder bias belongs on the commanded target, not inside the actuator's PD loop: it
   is a measurement error, so the *command* should be wrong by the amount the encoder is
   wrong by, while the true joint state stays clean.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.utils import configclass

from .domain_randomization import get_dr_state
from .kernels import delay_step_bounds

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_GROUPS_ATTR = "_wf_tron1b_action_delay_groups"

__all__ = [
    "DelayedJointPositionAction",
    "DelayedJointPositionActionCfg",
    "DelayedJointVelocityAction",
    "DelayedJointVelocityActionCfg",
]


def _policy_step(env: ManagerBasedRLEnv) -> int:
    return int(getattr(env, "common_step_counter", 0))


def _env_ids_tensor(env_ids, num_envs: int, device) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, device=device)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)[env_ids]
    return env_ids.to(device=device, dtype=torch.long)


class _DelayGroupState:
    """One shared per-environment delay draw for every action term in a group."""

    def __init__(
        self,
        *,
        delay_range_s: tuple[float, float],
        resampling_time_s: float,
        physics_dt: float,
        step_dt: float,
        num_envs: int,
        device,
    ) -> None:
        if resampling_time_s <= 0.0:
            raise ValueError(f"resampling_time_s must be positive, got {resampling_time_s}")
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, got {step_dt}")

        self.delay_range_s = (float(delay_range_s[0]), float(delay_range_s[1]))
        self.resampling_time_s = float(resampling_time_s)
        self.physics_dt = float(physics_dt)
        self.step_dt = float(step_dt)
        self.num_envs = int(num_envs)
        self.device = device
        self.min_delay_steps, self.max_delay_steps = delay_step_bounds(self.delay_range_s, self.physics_dt)
        self.resampling_steps = max(1, round(resampling_time_s / step_dt))

        self.delay_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._next_resample_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._last_prepared_step: int | None = None
        self.resample(slice(None), current_policy_step=0)

    def validate(self, **kwargs) -> None:
        mismatched = [
            key
            for key, value in kwargs.items()
            if getattr(self, key) != (tuple(float(v) for v in value) if key == "delay_range_s" else value)
        ]
        if mismatched:
            raise ValueError(
                f"Action terms sharing a delay_group must agree on {sorted(mismatched)}; "
                "otherwise the shared draw means different things to different terms."
            )

    def prepare_for_policy_step(self, env: ManagerBasedRLEnv) -> None:
        """Resample the environments whose delay has aged out. Idempotent per step."""
        current_step = _policy_step(env)
        if self._last_prepared_step == current_step:
            return
        self._last_prepared_step = current_step

        due = (self._next_resample_step <= current_step).nonzero(as_tuple=False).squeeze(-1)
        if due.numel() > 0:
            self.resample(due, current_policy_step=current_step)

    def resample(self, env_ids, *, current_policy_step: int) -> None:
        ids = _env_ids_tensor(env_ids, self.num_envs, self.device)
        if ids.numel() == 0:
            return
        self.delay_steps[ids] = torch.randint(
            self.min_delay_steps, self.max_delay_steps + 1, (ids.numel(),), device=self.device
        )
        self._next_resample_step[ids] = current_policy_step + self.resampling_steps


def _get_delay_group_state(env: ManagerBasedRLEnv, *, delay_group: str, **kwargs) -> tuple[_DelayGroupState, bool]:
    groups = getattr(env, _GROUPS_ATTR, None)
    if groups is None:
        groups = {}
        setattr(env, _GROUPS_ATTR, groups)

    if delay_group not in groups:
        groups[delay_group] = _DelayGroupState(**kwargs)
        return groups[delay_group], True

    groups[delay_group].validate(**kwargs)
    return groups[delay_group], False


class _DelayedActionMixin:
    """FIFO of processed actions, read at a per-environment offset."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg=cfg, env=env)
        self._delay_state, self._owns_delay_state = _get_delay_group_state(
            env,
            delay_group=cfg.delay_group,
            delay_range_s=cfg.delay_range_s,
            resampling_time_s=cfg.resampling_time_s,
            physics_dt=float(env.physics_dt),
            step_dt=float(env.step_dt),
            num_envs=env.num_envs,
            device=env.device,
        )
        depth = self._delay_state.max_delay_steps + 1
        self._processed_action_fifo = torch.zeros(self.num_envs, depth, self.action_dim, device=self.device)
        self._raw_action_fifo = torch.zeros_like(self._processed_action_fifo)
        self._applied_action_sum = torch.zeros_like(self._raw_actions)
        self._applied_action_substeps = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.long)
        self._terminal_applied_action = torch.zeros_like(self._raw_actions)
        self._reset_delay_buffer(slice(None))

    @property
    def delay_steps(self) -> torch.Tensor:
        """Per-environment actuation delay, in physics steps."""
        return self._delay_state.delay_steps

    @property
    def applied_action(self) -> torch.Tensor:
        """Mean policy-space action actually executed during the current control step."""
        return self._applied_action_sum / self._applied_action_substeps.clamp_min(1)

    @property
    def terminal_applied_action(self) -> torch.Tensor:
        """``applied_action`` captured just before the last reset of each environment."""
        return self._terminal_applied_action

    def process_actions(self, actions: torch.Tensor) -> None:
        self._delay_state.prepare_for_policy_step(self._env)
        super().process_actions(actions)
        self._applied_action_sum.zero_()
        self._applied_action_substeps.zero_()

    def _delayed_actions_for_substep(self) -> torch.Tensor:
        """Advance the FIFO one physics step and read each environment's delayed action."""
        self._processed_action_fifo = torch.roll(self._processed_action_fifo, shifts=1, dims=1)
        self._raw_action_fifo = torch.roll(self._raw_action_fifo, shifts=1, dims=1)
        self._processed_action_fifo[:, 0, :] = self._processed_actions
        self._raw_action_fifo[:, 0, :] = self._raw_actions

        env_ids = torch.arange(self.num_envs, device=self.device)
        delayed_processed = self._processed_action_fifo[env_ids, self._delay_state.delay_steps]
        delayed_raw = self._raw_action_fifo[env_ids, self._delay_state.delay_steps]

        self._applied_action_sum += delayed_raw
        self._applied_action_substeps += 1
        return delayed_processed

    def _reset_processed_actions(self, env_ids) -> None:
        """Recompute the processed action from the (just-zeroed) raw action.

        ``JointAction.reset`` zeroes ``_raw_actions`` but leaves ``_processed_actions``
        holding the last pre-reset target. Filling the FIFO from that stale value would
        make a fresh episode open by commanding the *previous* episode's final pose for
        up to ``max_delay_steps`` physics steps.
        """
        raw_actions = self._raw_actions[env_ids]
        scale = self._scale[env_ids] if isinstance(self._scale, torch.Tensor) else self._scale
        offset = self._offset[env_ids] if isinstance(self._offset, torch.Tensor) else self._offset
        processed_actions = raw_actions * scale + offset
        if self.cfg.clip is not None:
            processed_actions = torch.clamp(
                processed_actions,
                min=self._clip[env_ids, :, 0],
                max=self._clip[env_ids, :, 1],
            )
        self._processed_actions[env_ids] = processed_actions

    def _reset_delay_buffer(self, env_ids) -> None:
        """Fill the FIFO with the current action so a reset does not replay stale commands."""
        self._processed_action_fifo[env_ids] = self._processed_actions[env_ids].unsqueeze(1)
        self._raw_action_fifo[env_ids] = self._raw_actions[env_ids].unsqueeze(1)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._terminal_applied_action[env_ids] = self.applied_action[env_ids]
        super().reset(env_ids)
        self._reset_processed_actions(env_ids)
        self._reset_delay_buffer(env_ids)
        self._applied_action_sum[env_ids] = 0.0
        self._applied_action_substeps[env_ids] = 0
        # Only the term that created the group resamples, so a shared group is not
        # resampled once per member term on the same reset.
        if self._owns_delay_state:
            self._delay_state.resample(env_ids, current_policy_step=_policy_step(self._env))


class DelayedJointPositionAction(_DelayedActionMixin, JointPositionAction):
    """Position action applied through a delay FIFO, with encoder bias on the target."""

    def apply_actions(self) -> None:
        target = self._delayed_actions_for_substep()
        encoder_bias = get_dr_state(self._env).encoder_bias[:, self._joint_ids]
        self._asset.set_joint_position_target(target - encoder_bias, joint_ids=self._joint_ids)


class DelayedJointVelocityAction(_DelayedActionMixin, JointVelocityAction):
    """Velocity action applied through a delay FIFO."""

    def apply_actions(self) -> None:
        self._asset.set_joint_velocity_target(self._delayed_actions_for_substep(), joint_ids=self._joint_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action with per-environment physics-step actuation delay."""

    class_type: type = DelayedJointPositionAction

    delay_range_s: tuple[float, float] = (0.0, 0.02)
    """Inclusive range of representable physics-step delays, in seconds."""
    resampling_time_s: float = 5.0
    """Simulated time between per-environment delay resamples. Independent of episode
    boundaries, so a long episode still sees several different latencies."""
    delay_group: str = "actions"
    """Terms sharing a group share one delay draw per environment."""


@configclass
class DelayedJointVelocityActionCfg(JointVelocityActionCfg):
    """Joint velocity action with per-environment physics-step actuation delay."""

    class_type: type = DelayedJointVelocityAction

    delay_range_s: tuple[float, float] = (0.0, 0.02)
    """Inclusive range of representable physics-step delays, in seconds."""
    resampling_time_s: float = 5.0
    """Simulated time between per-environment delay resamples."""
    delay_group: str = "actions"
    """Terms sharing a group share one delay draw per environment."""
