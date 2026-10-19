"""The async depth observation against a transcription of mjlab's ``depth_image`` and ``AsyncDepthBuffer``."""

import math
from types import SimpleNamespace

import torch

from _loader import depth

H, W, LEFT_CROP, DEPTH_MIN, DEPTH_MAX = 6, 9, 2, 0.2, 2.0
STEP_DT, CAPTURE_HZ, DELAY_RANGE = 0.02, 30.0, (0.0, 0.020)


# --- mjlab reference (tasks/velocity/mdp/observations.py), logic unchanged --------------------------------


def _ref_depth_image(env, sensor_name="depth_camera", left_crop=0, depth_min_m=0.2, depth_max_m=2.0):
    camera = env.scene[sensor_name]
    depth = camera.data.depth.squeeze(-1)
    if left_crop > 0:
        depth = depth[..., left_crop:].contiguous()
    depth = torch.nan_to_num(depth, nan=depth_max_m, posinf=depth_max_m, neginf=depth_max_m)
    depth = torch.where(depth <= 0.0, depth_max_m, depth)
    depth = depth.clamp(min=depth_min_m, max=depth_max_m)
    return (depth - depth_min_m) / (depth_max_m - depth_min_m)


class _RefAsyncDepthBuffer:
    def __init__(self):
        self._frames = None
        self._capture_times_s = None
        self._next_capture_time_s = None
        self._capture_period_s = None
        self._delay_s = None
        self._delay_range_s = None
        self._invalid_env_ids = None

    def __call__(self, env, sensor_name="depth_camera", capture_frequency_hz=30.0,
                 system_delay_range_s=(0.0, 0.0), left_crop=0, depth_min_m=0.2, depth_max_m=2.0):  # fmt: skip
        step_dt = float(env.step_dt)
        delay_min_s = float(system_delay_range_s[0])
        delay_max_s = float(system_delay_range_s[1])
        current_time_s = int(getattr(env, "common_step_counter", 0)) * step_dt
        capture_period_s = 1.0 / capture_frequency_hz
        delay_range_s = (delay_min_s, delay_max_s)
        history_size = max(1, math.ceil(delay_max_s / capture_period_s) + 1)
        config_changed = self._capture_period_s != capture_period_s or self._delay_range_s != delay_range_s
        needs_init = self._frames is None or self._frames.shape[0] != history_size or config_changed
        capture_due = self._next_capture_time_s is None or current_time_s + 1.0e-9 >= self._next_capture_time_s
        needs_reset_fill = self._invalid_env_ids is not None
        if not (needs_init or capture_due or needs_reset_fill):
            return self._select(current_time_s)
        frame = _ref_depth_image(
            env, sensor_name=sensor_name, left_crop=left_crop, depth_min_m=depth_min_m, depth_max_m=depth_max_m
        ).unsqueeze(1)
        if needs_init:
            self._capture_period_s = capture_period_s
            self._delay_range_s = delay_range_s
            self._frames = frame.unsqueeze(0).repeat(history_size, *([1] * frame.ndim))
            self._capture_times_s = (
                current_time_s - torch.arange(history_size, device=frame.device, dtype=torch.float64) * capture_period_s
            )
            self._next_capture_time_s = current_time_s + capture_period_s
            self._delay_s = self._sample_delay(frame.shape[0], frame.device)
            self._invalid_env_ids = None
            return self._select(current_time_s)
        reset_env_ids = None
        if needs_reset_fill:
            reset_env_ids = self._invalid_env_ids.to(device=frame.device, dtype=torch.long)
        if capture_due:
            periods_elapsed = max(
                1, int((current_time_s - self._next_capture_time_s + 1.0e-9) // capture_period_s) + 1
            )
            capture_time_s = self._next_capture_time_s + (periods_elapsed - 1) * capture_period_s
            self._frames = torch.roll(self._frames, shifts=1, dims=0)
            self._frames[0].copy_(frame)
            self._capture_times_s = torch.roll(self._capture_times_s, shifts=1, dims=0)
            self._capture_times_s[0] = capture_time_s
            self._next_capture_time_s += periods_elapsed * capture_period_s
        if reset_env_ids is not None:
            self._frames[:, reset_env_ids] = frame[reset_env_ids].unsqueeze(0)
            self._delay_s[reset_env_ids] = self._sample_delay(reset_env_ids.numel(), frame.device)
            self._invalid_env_ids = None
        return self._select(current_time_s)

    def _sample_delay(self, count, device):
        delay_min_s, delay_max_s = self._delay_range_s
        return delay_min_s + (delay_max_s - delay_min_s) * torch.rand(count, device=device, dtype=torch.float64)

    def _select(self, current_time_s):
        target_times_s = current_time_s - self._delay_s
        too_new = self._capture_times_s.unsqueeze(0) > (target_times_s.unsqueeze(1) + 1.0e-9)
        frame_indices = too_new.sum(dim=1).clamp_max(self._frames.shape[0] - 1)
        env_indices = torch.arange(self._frames.shape[1], device=self._frames.device)
        return self._frames[frame_indices, env_indices]

    def reset(self, env_ids=None):
        if env_ids is None or isinstance(env_ids, slice):
            self.__init__()
            return
        if self._frames is None or env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=self._frames.device, dtype=torch.long)
        if self._invalid_env_ids is None:
            self._invalid_env_ids = env_ids
        else:
            self._invalid_env_ids = torch.unique(torch.cat((self._invalid_env_ids, env_ids)))


# ---------------------------------------------------------------------------------------------------------


def _raw_frame(generator, num_envs):
    """Planar depth with the special values a camera produces: misses (inf / NaN / 0) and negatives."""
    depth = 3.0 * torch.rand(num_envs, H, W, generator=generator)
    special = torch.rand(num_envs, H, W, generator=generator)
    depth[special < 0.05] = float("inf")
    depth[(special >= 0.05) & (special < 0.08)] = float("nan")
    depth[(special >= 0.08) & (special < 0.10)] = 0.0
    depth[(special >= 0.10) & (special < 0.11)] = -1.0
    return depth


def _ours(buffer, raw, step, calls):
    def capture():
        calls.append(step)
        return depth.normalize_depth(raw, LEFT_CROP, DEPTH_MIN, DEPTH_MAX).unsqueeze(1)

    return buffer(step * STEP_DT, capture, CAPTURE_HZ, DELAY_RANGE)


def test_normalize_depth_matches_mjlab():
    raw = _raw_frame(torch.Generator().manual_seed(0), 7)
    env = SimpleNamespace(scene={"depth_camera": SimpleNamespace(data=SimpleNamespace(depth=raw[..., None]))})
    reference = _ref_depth_image(env, left_crop=LEFT_CROP, depth_min_m=DEPTH_MIN, depth_max_m=DEPTH_MAX)
    assert torch.equal(depth.normalize_depth(raw, LEFT_CROP, DEPTH_MIN, DEPTH_MAX), reference)


def test_async_buffer_matches_mjlab_over_irregular_clocks_and_resets():
    num_envs, g = 12, torch.Generator().manual_seed(1)
    ref, ours, calls = _RefAsyncDepthBuffer(), depth.DelayedFrameBuffer(), []
    step = 0
    for k in range(400):
        step += 3 if k % 37 == 5 else 1  # occasional skipped policy steps exercise the catch-up path
        raw = _raw_frame(g, num_envs)
        if k % 97 == 50:  # full reset
            ref.reset(None)
            ours.reset(None)
        elif k % 7 == 3:  # partial reset, sometimes twice before the next read
            env_ids = torch.randperm(num_envs, generator=g)[: int(torch.randint(1, 5, (1,), generator=g))]
            ref.reset(env_ids)
            ours.reset(env_ids)
            if k % 14 == 3:
                ref.reset(env_ids[:1] + 1 if env_ids[0] + 1 < num_envs else env_ids[:1])
                ours.reset(env_ids[:1] + 1 if env_ids[0] + 1 < num_envs else env_ids[:1])
        env = SimpleNamespace(
            step_dt=STEP_DT,
            common_step_counter=step,
            scene={"depth_camera": SimpleNamespace(data=SimpleNamespace(depth=raw[..., None]))},
        )
        torch.manual_seed(k)  # same delay draws for both
        expected = ref(env, capture_frequency_hz=CAPTURE_HZ, system_delay_range_s=DELAY_RANGE,
                       left_crop=LEFT_CROP, depth_min_m=DEPTH_MIN, depth_max_m=DEPTH_MAX)  # fmt: skip
        torch.manual_seed(k)
        assert torch.equal(_ours(ours, raw, step, calls), expected), f"diverged at iteration {k}"
        assert torch.equal(ours._delay_s, ref._delay_s)
    assert len(calls) < 400  # the camera is not read on every step


def test_delay_selects_newest_frame_old_enough():
    """RNG-independent: with fixed delays, each env reads the newest frame whose capture tick is old enough."""
    num_envs, delays = 3, [0.0, 0.019, 0.034]  # the last exceeds the range, so it clamps to the oldest frame
    buffer, calls = depth.DelayedFrameBuffer(), []
    for step in range(10):
        raw = torch.full((num_envs, H, W), 0.2 + 0.1 * step)  # each frame encodes the step it was read at
        out = _ours(buffer, raw, step, calls)
        if step == 0:
            buffer._delay_s = torch.tensor(delays, dtype=torch.float64)
            continue
        # Frames are stamped with the capture tick (k / 30 s), not the step time; the history holds two
        # frames (ceil(0.020 / (1 / 30)) + 1), initially the first frame twice.
        stamped = [(-1.0 / CAPTURE_HZ, 0)] + [(k / CAPTURE_HZ, read_step) for k, read_step in enumerate(calls)]
        newest_first = stamped[::-1][:2]
        for env_id, delay in enumerate(delays):
            target = step * STEP_DT - delay
            index = min(sum(stamp > target + 1e-9 for stamp, _ in newest_first), 1)
            expected = (0.1 * newest_first[index][1]) / (DEPTH_MAX - DEPTH_MIN)
            assert math.isclose(float(out[env_id, 0, 0, 0]), expected, rel_tol=1e-5, abs_tol=1e-6), (step, env_id)
    # 30 Hz capture on a 50 Hz clock: the camera is read at steps 0, 2, 4, 5, 7, 9 only.
    assert calls == [0, 2, 4, 5, 7, 9]
