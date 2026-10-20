"""Geometry of the mjlab-equivalent sub-terrains (pure numpy / trimesh, no Isaac Sim)."""

from types import SimpleNamespace

import numpy as np
import pytest

from _loader import PKG_DIR, load, stub_package

stub_package("bipedal_locomotion.terrains", PKG_DIR / "terrains")
st = load("bipedal_locomotion.terrains.sub_terrains", "terrains/sub_terrains.py")

SIZE = (8.0, 8.0)


def _tops(meshes):
    return np.array([m.bounds[1, 2] for m in meshes])


class TestHeightfieldMesh:
    def test_rows_map_to_y_and_the_grid_spans_the_whole_patch(self):
        noise = np.zeros((3, 5), dtype=np.int16)
        noise[2, 0] = 10  # last row, first column
        mesh = st.heightfield_mesh(noise, SIZE, 0.005)
        top = mesh.vertices[np.argmax(mesh.vertices[:, 2])]
        assert top == pytest.approx([0.0, 8.0, 0.05])
        assert np.allclose(mesh.bounds, [[0.0, 0.0, 0.0], [8.0, 8.0, 0.05]])

    def test_faces_point_up(self):
        mesh = st.heightfield_mesh(np.random.randint(0, 20, (6, 6)).astype(np.int16), SIZE, 0.005)
        assert (mesh.face_normals[:, 2] > 0).all()


@pytest.mark.parametrize("inverted", [False, True])
def test_pyramid_stairs_step_count_heights_and_origin_follow_mjlab(inverted):
    cfg = SimpleNamespace(
        size=SIZE, step_height_range=(0.01, 0.12), step_width=0.3, border_width=0.5, platform_width=2.0, inverted=inverted
    )
    meshes, origin = st.pyramid_stairs_terrain(1.0, cfg)
    n, h = 8, 0.12  # mjlab: int((8 - 2 * 0.5 - 2.0) / (2 * 0.3)) steps, no "+ 1" as in Isaac Lab
    assert len(meshes) == 4 + 4 * n + 1
    expected = [-(k + 1) * h for k in range(n)] if inverted else [(k + 1) * h for k in range(n)]
    assert np.unique(np.round(_tops(meshes[4:-1]), 6)).tolist() == pytest.approx(sorted(expected))
    assert origin[2] == pytest.approx(-(n + 1) * h if inverted else (n + 1) * h)
    assert _tops(meshes[:4]) == pytest.approx(0.0)  # border flush with the ground


def test_random_stairs_rise_inward_to_the_spawn_platform():
    cfg = SimpleNamespace(size=SIZE, step_width=0.35, step_height_range=(0.03, 0.25), border_width=0.5, platform_width=2.0)
    meshes, origin = st.random_stairs_terrain(0.5, cfg)
    ring_tops = _tops(meshes[4:-1])[::4]
    assert len(ring_tops) == 7 and (np.diff(ring_tops) > 0).all()
    assert _tops(meshes[-1:])[0] == pytest.approx(origin[2]) == pytest.approx(ring_tops[-1])


def test_stepping_stones_are_a_flush_floor_at_difficulty_zero():
    cfg = SimpleNamespace(
        size=SIZE, stone_size_range=(0.5, 0.75), stone_distance_range=(0.0, 0.15), stone_height=0.0,
        stone_height_variation=0.075, stone_size_variation=0.075, displacement_range=0.075,
        border_width=0.5, platform_width=2.0, floor_depth=0.8,
    )  # fmt: skip
    meshes, origin = st.stepping_stones_terrain(0.0, cfg)
    floor, platform, stones = meshes[4], meshes[5], meshes[6:]
    assert _tops(stones) == pytest.approx(0.0)
    assert (platform.bounds[1, :2] - platform.bounds[0, :2] >= 2.0).all()
    assert floor.bounds[1, 2] == pytest.approx(-0.8)
    assert origin[2] == 0.0


def test_tilted_grid_is_flat_at_difficulty_zero_and_skips_the_platform():
    cfg = SimpleNamespace(size=SIZE, grid_width=0.6, tilt_range_deg=15.0, height_range=0.15,
                          border_width=0.5, platform_width=2.0, floor_depth=0.8)  # fmt: skip
    meshes, origin = st.tilted_grid_terrain(0.0, cfg)
    tiles = meshes[6:]
    assert len(tiles) == 11 * 11 - 9
    assert _tops(tiles) == pytest.approx(0.2) and origin[2] == pytest.approx(0.2)


def test_discrete_obstacles_heights_and_cleared_platform():
    cfg = SimpleNamespace(
        size=SIZE, obstacle_width_range=(0.3, 1.5), obstacle_height_range=(0.01, 0.15), num_obstacles=200,
        platform_width=2.0, border_width=0.25, horizontal_scale=0.15, vertical_scale=0.005,
        obstacle_height_mode="choice", square_obstacles=False, origin_z_offset=0.0,
    )  # fmt: skip
    (mesh,), _ = st.discrete_obstacles_terrain(1.0, cfg)
    h = int((0.01 + 1.0 * (0.15 - 0.01)) / 0.005)
    z = mesh.vertices[:, 2]
    assert set(np.round(np.unique(z) / 0.005).astype(int)) <= {-h, -h // 2, 0, h // 2, h}
    center = np.linalg.norm(mesh.vertices[:, :2] - 4.0, ord=np.inf, axis=1) < 0.8
    assert z[center] == pytest.approx(0.0)
