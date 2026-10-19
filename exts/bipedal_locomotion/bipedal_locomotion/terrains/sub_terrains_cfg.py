"""Configurations for the mjlab-equivalent sub-terrains.

Class and field names (and defaults) mirror ``mjlab.terrains`` so that the WF-TRON1B
terrain config reads like its mjlab original. Height-field configs deliberately do not
derive from Isaac Lab's ``HfTerrainBaseCfg``: the generator would overwrite their
per-terrain horizontal / vertical scales with its own.
"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.terrains import MeshPlaneTerrainCfg, SubTerrainBaseCfg
from isaaclab.utils import configclass

from . import sub_terrains

BoxFlatTerrainCfg = MeshPlaneTerrainCfg
"""Flat ground at z=0 (mjlab's 1 m thick box is equivalent from above)."""


@configclass
class BoxPyramidStairsTerrainCfg(SubTerrainBaseCfg):
    function = sub_terrains.pyramid_stairs_terrain
    border_width: float = 0.0
    step_height_range: tuple[float, float] = MISSING
    step_width: float = MISSING
    platform_width: float = 1.0
    inverted: bool = False


@configclass
class BoxInvertedPyramidStairsTerrainCfg(BoxPyramidStairsTerrainCfg):
    inverted: bool = True


@configclass
class BoxRandomStairsTerrainCfg(SubTerrainBaseCfg):
    function = sub_terrains.random_stairs_terrain
    step_width: float = 0.8
    step_height_range: tuple[float, float] = (0.1, 0.3)
    platform_width: float = 1.0
    border_width: float = 0.25


@configclass
class BoxRandomSpreadTerrainCfg(SubTerrainBaseCfg):
    function = sub_terrains.random_spread_terrain
    num_boxes: int = 60
    box_width_range: tuple[float, float] = (0.3, 1.0)
    box_length_range: tuple[float, float] = (0.3, 1.0)
    box_height_range: tuple[float, float] = (0.05, 1.0)
    box_yaw_range: tuple[float, float] = (0.0, 360.0)
    add_floor: bool = True
    platform_width: float = 1.0
    border_width: float = 0.25


@configclass
class BoxSteppingStonesTerrainCfg(SubTerrainBaseCfg):
    function = sub_terrains.stepping_stones_terrain
    stone_size_range: tuple[float, float] = (0.4, 0.8)
    stone_distance_range: tuple[float, float] = (0.2, 0.5)
    stone_height: float = 0.2
    stone_height_variation: float = 0.1
    stone_size_variation: float = 0.1
    floor_depth: float = 2.0
    displacement_range: float = 0.1
    platform_width: float = 1.0
    border_width: float = 0.25


@configclass
class BoxTiltedGridTerrainCfg(SubTerrainBaseCfg):
    function = sub_terrains.tilted_grid_terrain
    grid_width: float = 1.0
    tilt_range_deg: float = 15.0
    height_range: float = 0.1
    platform_width: float = 1.0
    border_width: float = 0.25
    floor_depth: float = 2.0


@configclass
class _HfTerrainCfg(SubTerrainBaseCfg):
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_width: float = 0.0


@configclass
class HfPyramidSlopedTerrainCfg(_HfTerrainCfg):
    function = sub_terrains.pyramid_sloped_terrain
    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = False


@configclass
class HfRandomUniformTerrainCfg(_HfTerrainCfg):
    function = sub_terrains.random_uniform_terrain
    noise_range: tuple[float, float] = MISSING
    noise_step: float = 0.005
    downsampled_scale: float | None = None
    scale_with_difficulty: bool = False


@configclass
class HfDiscreteObstaclesTerrainCfg(_HfTerrainCfg):
    function = sub_terrains.discrete_obstacles_terrain
    obstacle_height_mode: Literal["choice", "fixed"] = "choice"
    obstacle_width_range: tuple[float, float] = MISSING
    obstacle_height_range: tuple[float, float] = MISSING
    num_obstacles: int = MISSING
    platform_width: float = 1.0
    square_obstacles: bool = False
    origin_z_offset: float = 0.0
