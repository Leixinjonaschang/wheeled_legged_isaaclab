"""Sub-terrain meshes reproducing the mjlab terrains used by the WF-TRON1B task.

Each function follows Isaac Lab's sub-terrain contract ``function(difficulty, cfg) ->
(meshes, origin)`` with the patch spanning ``[0, size[0]] x [0, size[1]]``. Geometry,
difficulty scaling and spawn origins follow ``mjlab.terrains`` (``primitive_terrains.py``
and ``heightfield_terrains.py``); only the random streams differ. Isaac Lab's built-ins
of the same names are not used because they differ in step counts, obstacle sampling,
height-field grid spacing and vertical-wall correction.

Pure numpy / trimesh, so the geometry is testable without Isaac Sim.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from scipy import interpolate


def _lerp(value_range: tuple[float, float], t: float) -> float:
    return value_range[0] + t * (value_range[1] - value_range[0])


def _box(lo, hi, yaw: float = 0.0) -> trimesh.Trimesh:
    """Box spanning the corners ``lo`` and ``hi``, optionally yawed about its center."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    transform = trimesh.transformations.rotation_matrix(yaw, (0.0, 0.0, 1.0))
    transform[:3, 3] = (lo + hi) / 2
    return trimesh.creation.box(np.maximum(hi - lo, 2e-6), transform)


def _border(size, thickness: float, z0: float, z1: float) -> list[trimesh.Trimesh]:
    """Square frame of the given thickness around the patch, spanning ``[z0, z1]``."""
    (sx, sy), t = size, thickness
    return [
        _box((0, sy - t, z0), (sx, sy, z1)),
        _box((0, 0, z0), (sx, t, z1)),
        _box((0, t, z0), (t, sy - t, z1)),
        _box((sx - t, t, z0), (sx, sy - t, z1)),
    ]


def _ring(size, border: float, step: float, k: int, z0: float, z1: float) -> list[trimesh.Trimesh]:
    """The k-th square ring of steps, counted inward from the border."""
    x0 = y0 = border + k * step
    x1, y1 = size[0] - x0, size[1] - y0
    return [
        _box((x0, y1 - step, z0), (x1, y1, z1)),
        _box((x0, y0, z0), (x1, y0 + step, z1)),
        _box((x1 - step, y0 + step, z0), (x1, y1 - step, z1)),
        _box((x0, y0 + step, z0), (x0 + step, y1 - step, z1)),
    ]


def _num_steps(cfg) -> int:
    return int(min((s - 2 * cfg.border_width - cfg.platform_width) / (2 * cfg.step_width) for s in cfg.size))


def _center(cfg) -> tuple[float, float]:
    return cfg.size[0] / 2, cfg.size[1] / 2


# --------------------------------------------------------------------------------------
# Box terrains
# --------------------------------------------------------------------------------------


def pyramid_stairs_terrain(difficulty: float, cfg):
    """Square stairs climbing (or, inverted, descending) to a central platform."""
    h = _lerp(cfg.step_height_range, difficulty)
    n = max(0, _num_steps(cfg))
    meshes = _border(cfg.size, cfg.border_width, -max(h, 0.05), 0.0) if cfg.border_width > 0 else []
    if cfg.inverted:
        total = (n + 1) * h
        for k in range(n):
            meshes += _ring(cfg.size, cfg.border_width, cfg.step_width, k, -total, -(k + 1) * h)
        z0, z1, origin_z = -total - h, -total, -total
    else:
        for k in range(n):
            meshes += _ring(cfg.size, cfg.border_width, cfg.step_width, k, -h, (k + 1) * h)
        z0, z1, origin_z = -h, (n + 1) * h, (n + 1) * h
    edge = cfg.border_width + n * cfg.step_width
    meshes.append(_box((edge, edge, z0), (cfg.size[0] - edge, cfg.size[1] - edge, z1)))
    return meshes, np.array([*_center(cfg), origin_z])


def random_stairs_terrain(difficulty: float, cfg):
    """Square stairs whose step heights are drawn independently."""
    n = _num_steps(cfg)
    meshes = _border(cfg.size, cfg.border_width, -0.1, 0.0) if cfg.border_width > 0 else []
    z = 0.0
    for k in range(n):
        z += np.random.uniform(*cfg.step_height_range) * (0.5 + 0.5 * difficulty)
        meshes += _ring(cfg.size, cfg.border_width, cfg.step_width, k, 0.0, z)
    edge = cfg.border_width + n * cfg.step_width
    meshes.append(_box((edge, edge, 0.0), (cfg.size[0] - edge, cfg.size[1] - edge, z)))
    return meshes, np.array([*_center(cfg), z])


def random_spread_terrain(difficulty: float, cfg):
    """Randomly sized, placed and yawed boxes around an obstacle-free spawn square."""
    sx, sy = cfg.size
    b = cfg.border_width
    meshes = _border(cfg.size, b, -1.0, 0.0) if b > 0 else []
    if cfg.add_floor:
        meshes.append(_box((b, b, -0.1), (sx - b, sy - b, 0.0)))
    else:
        half = cfg.platform_width / 2
        meshes.append(_box((sx / 2 - half, sy / 2 - half, -1.0), (sx / 2 + half, sy / 2 + half, 0.0)))

    lo, hi = sx / 2 - cfg.platform_width / 2, sx / 2 + cfg.platform_width / 2
    for _ in range(int(cfg.num_boxes * (0.5 + 0.5 * difficulty))):
        size_x = np.random.uniform(*cfg.box_width_range)
        size_y = np.random.uniform(*cfg.box_length_range)
        height = np.random.uniform(*cfg.box_height_range) * (0.2 + 0.8 * difficulty)
        px = np.random.uniform(b + size_x / 2, sx - b - size_x / 2)
        py = np.random.uniform(b + size_y / 2, sy - b - size_y / 2)
        if lo - size_x / 2 <= px <= hi + size_x / 2 and lo - size_y / 2 <= py <= hi + size_y / 2:
            continue
        yaw = math.radians(np.random.uniform(*cfg.box_yaw_range))
        meshes.append(_box((px - size_x / 2, py - size_y / 2, 0.0), (px + size_x / 2, py + size_y / 2, height), yaw))
    return meshes, np.array([*_center(cfg), 0.0])


def stepping_stones_terrain(difficulty: float, cfg):
    """Stones on a fixed grid over a pit; they shrink (gaps grow) with difficulty."""
    b = cfg.border_width
    size_var = cfg.stone_size_variation * difficulty
    displacement = cfg.displacement_range * difficulty
    height_var = cfg.stone_height_variation * difficulty
    stone = cfg.stone_size_range[1] - difficulty * (cfg.stone_size_range[1] - cfg.stone_size_range[0])

    # Difficulty-independent grid whose pitch exactly spans the inner region.
    inner = (cfg.size[0] - 2 * b, cfg.size[1] - 2 * b)
    nominal_spacing = cfg.stone_size_range[1] + cfg.stone_distance_range[0]
    num = [max(2, int(np.floor(extent / nominal_spacing)) + 1) for extent in inner]
    pitch = [extent / (n - 1) for extent, n in zip(inner, num)]
    gap = [max(0.0, p - stone) for p in pitch]

    # The platform reaches to one gap before the nearest full stone around it.
    center = _center(cfg)

    def snapped_half(c: float, p: float, g: float, n: int) -> float:
        i_keep = min(n - 1, int(np.ceil((c + cfg.platform_width / 2 + stone / 2 + g - b) / p)))
        return max(cfg.platform_width / 2, b + i_keep * p - stone / 2 - g - c)

    half = [snapped_half(center[i], pitch[i], gap[i], num[i]) for i in range(2)]

    z0, z1 = -cfg.floor_depth, cfg.stone_height
    meshes = _border(cfg.size, b, z0, z1) if b > 0 else []
    meshes.append(_box((0, 0, z0 - 0.1), (cfg.size[0], cfg.size[1], z0)))
    meshes.append(_box((center[0] - half[0], center[1] - half[1], z0), (center[0] + half[0], center[1] + half[1], z1)))
    for i in range(num[0]):
        for j in range(num[1]):
            px = b + i * pitch[0] + np.random.uniform(-displacement, displacement)
            py = b + j * pitch[1] + np.random.uniform(-displacement, displacement)
            size_x = stone + np.random.uniform(-size_var, size_var)
            size_y = stone + np.random.uniform(-size_var, size_var)
            if abs(px - center[0]) <= half[0] and abs(py - center[1]) <= half[1]:
                continue
            x_lo, x_hi = np.clip((px - size_x / 2, px + size_x / 2), b, cfg.size[0] - b)
            y_lo, y_hi = np.clip((py - size_y / 2, py + size_y / 2), b, cfg.size[1] - b)
            if x_hi - x_lo < 0.05 or y_hi - y_lo < 0.05:
                continue
            top = cfg.stone_height + np.random.uniform(-height_var, height_var)
            meshes.append(_box((x_lo, y_lo, z0), (x_hi, y_hi, top)))
    return meshes, np.array([*center, cfg.stone_height])


def _tilted_tile(cx: float, cy: float, width: float, z_bottom: float, h: float, tx: float, ty: float):
    """Square column with vertical sides and the top plane ``h + tx (x - cx) + ty (y - cy)``."""
    corners = [(x, y) for x in (cx - width / 2, cx + width / 2) for y in (cy - width / 2, cy + width / 2)]
    vertices = [(x, y, z_bottom) for x, y in corners] + [(x, y, h + tx * (x - cx) + ty * (y - cy)) for x, y in corners]
    faces = [[4, 6, 7], [4, 7, 5], [0, 1, 3], [0, 3, 2], [0, 2, 6], [0, 6, 4],
             [1, 5, 7], [1, 7, 3], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6]]  # fmt: skip
    return trimesh.Trimesh(vertices, faces, process=False)


def tilted_grid_terrain(difficulty: float, cfg):
    """Grid of columns over a pit, each with a randomly tilted and raised top."""
    max_tilt = math.radians(cfg.tilt_range_deg * difficulty)
    height_range = cfg.height_range * difficulty
    width = cfg.grid_width
    nx = int((cfg.size[0] - 2 * cfg.border_width) / width)
    ny = int((cfg.size[1] - 2 * cfg.border_width) / width)
    margin = (cfg.size[0] - nx * width) / 2

    base_h, z0 = 0.2, -cfg.floor_depth
    meshes = _border(cfg.size, margin, z0, base_h) if margin > 0 else []
    meshes.append(_box((0, 0, z0 - 0.1), (cfg.size[0], cfg.size[1], z0)))
    center, half = _center(cfg), cfg.platform_width / 2
    meshes.append(_box((center[0] - half, center[1] - half, z0), (center[0] + half, center[1] + half, base_h)))

    for i in range(nx):
        cx = margin + (i + 0.5) * width
        for j in range(ny):
            cy = margin + (j + 0.5) * width
            if abs(cx - center[0]) <= half and abs(cy - center[1]) <= half:
                continue
            h = base_h + np.random.uniform(-height_range / 2, height_range / 2)
            tx, ty = np.random.uniform(-max_tilt, max_tilt, size=2)
            meshes.append(_tilted_tile(cx, cy, width, z0, h, tx, ty))
    return meshes, np.array([*center, base_h])


# --------------------------------------------------------------------------------------
# Height-field terrains
# --------------------------------------------------------------------------------------


def heightfield_mesh(noise: np.ndarray, size, vertical_scale: float) -> trimesh.Trimesh:
    """Surface of a MuJoCo heightfield spanning the whole patch.

    MuJoCo maps rows to y and columns to x and stretches the grid over the full size, so
    the vertex spacing is ``size / (n - 1)`` rather than the configured horizontal scale.
    Cells are split along the same diagonal as MuJoCo's ray / collision code.
    """
    nrow, ncol = noise.shape
    xx, yy = np.meshgrid(np.linspace(0.0, size[0], ncol), np.linspace(0.0, size[1], nrow))
    vertices = np.stack((xx.ravel(), yy.ravel(), noise.ravel() * vertical_scale), axis=-1)
    idx = np.arange(nrow * ncol).reshape(nrow, ncol)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate((np.stack((a, b, d), -1).reshape(-1, 3), np.stack((a, d, c), -1).reshape(-1, 3)))
    return trimesh.Trimesh(vertices, faces, process=False)


def _hfield_grid(cfg) -> tuple[int, int, int, int, int]:
    """(width, length, border, inner_width, inner_length) in pixels, as mjlab computes them."""
    border = int(cfg.border_width / cfg.horizontal_scale)
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    return width, length, border, width - 2 * border, length - 2 * border


def _embed(inner: np.ndarray, width: int, length: int, border: int) -> np.ndarray:
    noise = np.zeros((width, length), dtype=np.int16)
    noise[border : border + inner.shape[0], border : border + inner.shape[1]] = inner
    return noise


def _elevation_range(noise: np.ndarray) -> int:
    return max(int(noise.max()) - int(noise.min()), 1)


def pyramid_sloped_terrain(difficulty: float, cfg):
    """Four-sided slope rising (or, inverted, sinking) to a flat central platform."""
    slope = _lerp(cfg.slope_range, difficulty) * (-1.0 if cfg.inverted else 1.0)
    width, length, border, inner_w, inner_l = _hfield_grid(cfg)
    extent = inner_w * cfg.horizontal_scale if border > 0 else cfg.size[0]
    height_max = int(slope * extent / 2 / cfg.vertical_scale)

    cx, cy = int(inner_w / 2), int(inner_l / 2)
    xx = (cx - np.abs(cx - np.arange(inner_w))) / cx
    yy = (cy - np.abs(cy - np.arange(inner_l))) / cy
    hf = height_max * xx[:, None] * yy[None, :]
    platform = int(cfg.platform_width / cfg.horizontal_scale / 2)
    x_pf, y_pf = inner_w // 2 - platform, inner_l // 2 - platform
    z_pf = hf[x_pf, y_pf] if x_pf >= 0 and y_pf >= 0 else 0
    hf = np.clip(hf, min(0, z_pf), max(0, z_pf))

    noise = _embed(np.rint(hf).astype(np.int16), width, length, border)
    origin_z = _elevation_range(noise) * cfg.vertical_scale * (-1.0 if cfg.inverted else 1.0)
    return [heightfield_mesh(noise, cfg.size, cfg.vertical_scale)], np.array([*_center(cfg), origin_z])


def random_uniform_terrain(difficulty: float, cfg):
    """Spline-smoothed uniform height noise; difficulty-independent unless configured."""
    scale = difficulty if cfg.scale_with_difficulty else 1.0
    noise_lo, noise_hi = cfg.noise_range[0] * scale, cfg.noise_range[1] * scale
    width, length, border, inner_w, inner_l = _hfield_grid(cfg)
    inner_size = (inner_w * cfg.horizontal_scale, inner_l * cfg.horizontal_scale) if border > 0 else cfg.size
    downsampled = cfg.downsampled_scale or cfg.horizontal_scale
    coarse_shape = (int(inner_size[0] / downsampled), int(inner_size[1] / downsampled))

    step = int(cfg.noise_step / cfg.vertical_scale)
    heights = np.arange(int(noise_lo / cfg.vertical_scale), int(noise_hi / cfg.vertical_scale) + step, step)
    coarse = np.random.choice(heights, size=coarse_shape)
    spline = interpolate.RectBivariateSpline(
        np.linspace(0, inner_size[0], coarse_shape[0]), np.linspace(0, inner_size[1], coarse_shape[1]), coarse
    )
    fine = spline(np.linspace(0, inner_size[0], inner_w), np.linspace(0, inner_size[1], inner_l))

    noise = _embed(np.rint(fine).astype(np.int16), width, length, border)
    origin = np.array([*_center(cfg), (noise_lo + noise_hi) / 2])
    # MuJoCo puts the lowest sample at z=0 (a no-op whenever there is a border).
    return [heightfield_mesh(noise - noise.min(), cfg.size, cfg.vertical_scale)], origin


def discrete_obstacles_terrain(difficulty: float, cfg):
    """Rectangular bumps and pits on a 4-pixel lattice around a cleared platform."""
    obs_h = int(_lerp(cfg.obstacle_height_range, difficulty) / cfg.vertical_scale)
    width, length, border, inner_w, inner_l = _hfield_grid(cfg)
    widths = np.arange(
        int(cfg.obstacle_width_range[0] / cfg.horizontal_scale), int(cfg.obstacle_width_range[1] / cfg.horizontal_scale) + 1, 4
    )
    if len(widths) == 0:
        widths = np.array([int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)])

    inner = np.zeros((inner_w, inner_l), dtype=np.int16)
    for _ in range(cfg.num_obstacles):
        if cfg.obstacle_height_mode == "choice":
            h = np.random.choice(np.array([-obs_h, -obs_h // 2, obs_h // 2, obs_h]))
        else:
            h = obs_h
        w = np.random.choice(widths)
        obs_len = w if cfg.square_obstacles else np.random.choice(widths)
        x = np.random.choice(np.arange(0, inner_w, 4))
        y = np.random.choice(np.arange(0, inner_l, 4))
        inner[x : x + w, y : y + obs_len] = h

    half_pf = int(cfg.platform_width / cfg.horizontal_scale) // 2
    cx, cy = inner_w // 2, inner_l // 2
    inner[max(cx - half_pf, 0) : cx + half_pf, max(cy - half_pf, 0) : cy + half_pf] = 0

    noise = _embed(inner, width, length, border)
    origin = np.array([*_center(cfg), cfg.origin_z_offset])
    return [heightfield_mesh(noise, cfg.size, cfg.vertical_scale)], origin
