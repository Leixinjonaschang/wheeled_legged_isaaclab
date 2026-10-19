"""Depth observation core: mjlab's ``depth_image`` post-processing and ``AsyncDepthBuffer`` clock.

Pure torch (no Isaac Lab import), so the tests can check it against mjlab without a simulator.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

__all__ = ["DelayedFrameBuffer", "normalize_depth"]


def normalize_depth(depth: torch.Tensor, left_crop: int, depth_min_m: float, depth_max_m: float) -> torch.Tensor:
    """Planar depth ``[B, H, W]`` -> ``[B, H, W - left_crop]`` in [0, 1]; misses (non-finite or <= 0) read as far."""
    if depth_min_m < 0.0:
        raise ValueError(f"depth_min_m must be >= 0, got {depth_min_m}")
    if depth_max_m <= depth_min_m:
        raise ValueError(f"depth_max_m must be greater than depth_min_m, got {depth_max_m}")
    if not 0 <= left_crop < depth.shape[-1]:
        raise ValueError(f"left_crop must be in [0, {depth.shape[-1] - 1}], got {left_crop}")
    depth = torch.nan_to_num(depth[..., left_crop:], nan=depth_max_m, posinf=depth_max_m, neginf=depth_max_m)
    depth = torch.where(depth <= 0.0, depth_max_m, depth).clamp(min=depth_min_m, max=depth_max_m)
    return (depth - depth_min_m) / (depth_max_m - depth_min_m)


class DelayedFrameBuffer:
    """Frames captured on a fixed clock; each environment reads the newest frame at least its own delay old.

    mjlab's ``AsyncDepthBuffer``: capture at ``capture_frequency_hz`` on the global clock, a per-environment
    system delay ~ U(``system_delay_range_s``) drawn at initialisation and at each environment's reset, a
    history just long enough for the largest delay, and reset environments refilled with the current frame.
    ``capture`` is only called on initialisation, capture and refill steps.
    """

    def __init__(self) -> None:
        self.reset()

    def __call__(
        self,
        current_time_s: float,
        capture: Callable[[], torch.Tensor],
        capture_frequency_hz: float,
        system_delay_range_s: tuple[float, float],
    ) -> torch.Tensor:
        if capture_frequency_hz <= 0.0:
            raise ValueError(f"capture_frequency_hz must be positive, got {capture_frequency_hz}")
        delay_min_s, delay_max_s = float(system_delay_range_s[0]), float(system_delay_range_s[1])
        if delay_min_s < 0.0 or delay_min_s > delay_max_s:
            raise ValueError(f"system_delay_range_s must be ordered and non-negative, got {system_delay_range_s}")

        capture_period_s = 1.0 / capture_frequency_hz
        delay_range_s = (delay_min_s, delay_max_s)
        history_size = max(1, math.ceil(delay_max_s / capture_period_s) + 1)
        needs_init = (
            self._frames is None
            or self._frames.shape[0] != history_size
            or self._capture_period_s != capture_period_s
            or self._delay_range_s != delay_range_s
        )
        capture_due = self._next_capture_time_s is None or current_time_s + 1.0e-9 >= self._next_capture_time_s
        needs_reset_fill = self._invalid_env_ids is not None
        if not (needs_init or capture_due or needs_reset_fill):
            return self._select(current_time_s)

        frame = capture()
        if needs_init:
            self._capture_period_s = capture_period_s
            self._delay_range_s = delay_range_s
            self._frames = frame.unsqueeze(0).repeat(history_size, *([1] * frame.ndim))
            self._capture_times_s = current_time_s - capture_period_s * torch.arange(
                history_size, device=frame.device, dtype=torch.float64
            )
            self._next_capture_time_s = current_time_s + capture_period_s
            self._delay_s = self._sample_delay(frame.shape[0], frame.device)
            self._invalid_env_ids = None
            return self._select(current_time_s)

        if capture_due:
            # Catch up on skipped capture ticks; the frame is stamped with the latest one.
            periods_elapsed = max(1, int((current_time_s - self._next_capture_time_s + 1.0e-9) // capture_period_s) + 1)
            self._frames = torch.roll(self._frames, shifts=1, dims=0)
            self._frames[0].copy_(frame)
            self._capture_times_s = torch.roll(self._capture_times_s, shifts=1, dims=0)
            self._capture_times_s[0] = self._next_capture_time_s + (periods_elapsed - 1) * capture_period_s
            self._next_capture_time_s += periods_elapsed * capture_period_s

        if needs_reset_fill:
            env_ids = self._invalid_env_ids.to(device=frame.device, dtype=torch.long)
            self._frames[:, env_ids] = frame[env_ids].unsqueeze(0)
            self._delay_s[env_ids] = self._sample_delay(env_ids.numel(), frame.device)
            self._invalid_env_ids = None
        return self._select(current_time_s)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Forget everything (``None`` / slice), or refill the given environments on the next call."""
        if env_ids is None or isinstance(env_ids, slice):
            self._frames = None
            self._capture_times_s = None
            self._next_capture_time_s = None
            self._capture_period_s = None
            self._delay_s = None
            self._delay_range_s = None
            self._invalid_env_ids = None
            return
        env_ids = torch.as_tensor(env_ids)
        if self._frames is None or env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self._frames.device, dtype=torch.long)
        if self._invalid_env_ids is not None:
            env_ids = torch.unique(torch.cat((self._invalid_env_ids, env_ids)))
        self._invalid_env_ids = env_ids

    def _sample_delay(self, count: int, device) -> torch.Tensor:
        delay_min_s, delay_max_s = self._delay_range_s
        return delay_min_s + (delay_max_s - delay_min_s) * torch.rand(count, device=device, dtype=torch.float64)

    def _select(self, current_time_s: float) -> torch.Tensor:
        """Per environment, the newest frame captured no later than ``current_time_s - delay``."""
        target_times_s = current_time_s - self._delay_s
        too_new = self._capture_times_s.unsqueeze(0) > (target_times_s.unsqueeze(1) + 1.0e-9)
        frame_indices = too_new.sum(dim=1).clamp_max(self._frames.shape[0] - 1)
        return self._frames[frame_indices, torch.arange(self._frames.shape[1], device=self._frames.device)]
