"""Ray-caster clearance reads like mjlab's ``TerrainHeightSensor``: clamped to ``[0, reach]``, misses read the reach."""

import math
from types import SimpleNamespace

import torch

from _loader import load

sensor_utils = load("bipedal_locomotion.tasks.locomotion.mdp.sensor_utils", "tasks/locomotion/mdp/sensor_utils.py")


def _clearance(frame_z: float, hit_z: list[float], max_distance: float, start_height: float) -> torch.Tensor:
    """Clearance samples of one ray-caster at ``frame_z`` whose rays start ``start_height`` above it."""
    hits = torch.zeros(1, len(hit_z), 3)
    hits[0, :, 2] = torch.tensor(hit_z)
    sensor = SimpleNamespace(
        cfg=SimpleNamespace(max_distance=max_distance, offset=SimpleNamespace(pos=(0.0, 0.0, start_height))),
        data=SimpleNamespace(pos_w=torch.tensor([[0.0, 0.0, frame_z]]), ray_hits_w=hits),
    )
    return sensor_utils.clearance_samples(SimpleNamespace(scene={"scan": sensor}), "scan")[0, 0]


def test_wheel_scan_reads_zero_above_the_frame_and_the_reach_on_a_miss():
    # Wheel centre 0.127 m up; rays from 1.0 m above it, 2.0 m long: floor, 0.05 m ledge, 0.25 m step, misses.
    out = _clearance(0.127, [0.0, 0.05, 0.25, math.inf, math.nan], max_distance=2.0, start_height=1.0)
    torch.testing.assert_close(out, torch.tensor([0.127, 0.127 - 0.05, 0.0, 1.0, 1.0]))


def test_scan_without_offset_keeps_its_max_distance_as_the_miss_value():
    out = _clearance(0.8, [0.0, 0.3, math.inf], max_distance=10.0, start_height=0.0)
    torch.testing.assert_close(out, torch.tensor([0.8, 0.8 - 0.3, 10.0]))
