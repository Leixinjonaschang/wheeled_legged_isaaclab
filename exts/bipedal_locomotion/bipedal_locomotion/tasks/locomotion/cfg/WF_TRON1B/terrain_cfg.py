"""Generated terrain for the WF-TRON1B rough tasks, ported from mjlab's ``terrain_cfg.py``.

Flat columns are interleaved with the obstacle columns, so a straight-line command along
world-y crosses terrain *transitions* at a fixed difficulty level. Curriculum semantics
are mjlab's (see :mod:`bipedal_locomotion.terrains.curriculum`): one column per entry,
``proportion`` sets how many robots spawn on it.
"""

from bipedal_locomotion import terrains
from isaaclab.terrains import TerrainGeneratorCfg

WF_TRON1B_TERRAINS_CFG = TerrainGeneratorCfg(
    class_type=terrains.ColumnPerTerrainGenerator,
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=50,
    num_cols=20,
    curriculum=True,
    sub_terrains={
        # Alternate flat and non-flat columns; the tilted-grid column is last
        # for focused parkour play.
        "flat__0": terrains.BoxFlatTerrainCfg(proportion=0.01),
        "discrete_obstacles": terrains.HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            obstacle_width_range=(0.3, 1.5),
            obstacle_height_range=(0.01, 0.15),
            num_obstacles=200,
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "flat__1": terrains.BoxFlatTerrainCfg(proportion=0.03),
        "random_rough": terrains.HfRandomUniformTerrainCfg(
            proportion=0.08,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "flat__2": terrains.BoxFlatTerrainCfg(proportion=0.03),
        "hf_pyramid_slope": terrains.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "flat__3": terrains.BoxFlatTerrainCfg(proportion=0.03),
        "hf_pyramid_slope_inv": terrains.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
            horizontal_scale=0.15,
            vertical_scale=0.005,
            inverted=True,
        ),
        "flat__4": terrains.BoxFlatTerrainCfg(proportion=0.03),
        "pyramid_stair_inv": terrains.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.01, 0.12),
            step_width=0.3,
            border_width=0.5,
            platform_width=2.0,
        ),
        "flat__5": terrains.BoxFlatTerrainCfg(proportion=0.03),
        "pyramid_stair": terrains.BoxPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.01, 0.12),
            step_width=0.3,
            border_width=0.5,
            platform_width=2.0,
        ),
        "flat__6": terrains.BoxFlatTerrainCfg(proportion=0.03),
        "random_stairs": terrains.BoxRandomStairsTerrainCfg(
            proportion=0.05,
            step_width=0.35,
            step_height_range=(0.03, 0.25),
            border_width=0.5,
            platform_width=2.0,
        ),
        "flat__7": terrains.BoxFlatTerrainCfg(proportion=0.03),
        "random_spread": terrains.BoxRandomSpreadTerrainCfg(
            proportion=0.07,
            num_boxes=64,
            box_width_range=(0.20, 0.70),
            box_length_range=(0.20, 0.90),
            box_height_range=(0.03, 0.25),
            box_yaw_range=(-45.0, 45.0),
            border_width=0.5,
            platform_width=2.0,
        ),
        "flat__8": terrains.BoxFlatTerrainCfg(proportion=0.03),
        "stepping_stones": terrains.BoxSteppingStonesTerrainCfg(
            proportion=0.07,
            # Level zero is a contiguous, uniform tiled surface. Difficulty then
            # ramps across 50 rows into smaller, separated, irregular stones.
            stone_size_range=(0.50, 0.75),
            stone_distance_range=(0.0, 0.15),
            stone_height=0.0,
            stone_height_variation=0.075,
            stone_size_variation=0.075,
            displacement_range=0.075,
            border_width=0.5,
            platform_width=2.0,
            floor_depth=0.8,
        ),
        "flat__9": terrains.BoxFlatTerrainCfg(proportion=0.03),
        "tilted_grid": terrains.BoxTiltedGridTerrainCfg(
            proportion=0.05,
            grid_width=0.6,
            tilt_range_deg=15.0,
            height_range=0.15,
            border_width=0.5,
            platform_width=2.0,
            floor_depth=0.8,
        ),
    },
)
"""Rough terrain: obstacle columns interleaved with flat ones."""
