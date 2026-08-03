"""Generated terrain for the WF-TRON1B rough task.

Ported from the mjlab task's ``terrain_cfg.py``. The distinguishing feature is that flat
columns are interleaved between the obstacle columns: a straight-line command along
world-y then crosses terrain *transitions* at a fixed difficulty level, which is where a
wheeled biped has to actually decide between rolling and stepping. A block of obstacle
columns alone never exercises that decision.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

_FLAT_PROPORTION = 0.3 / 7

WF_TRON1B_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=5.0,
    num_rows=50,
    num_cols=13,
    horizontal_scale=0.15,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat__0": terrain_gen.MeshPlaneTerrainCfg(proportion=_FLAT_PROPORTION),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_width_range=(0.3, 1.5),
            obstacle_height_range=(0.01, 0.15),
            num_obstacles=200,
            platform_width=2.0,
        ),
        "flat__1": terrain_gen.MeshPlaneTerrainCfg(proportion=_FLAT_PROPORTION),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
        "flat__2": terrain_gen.MeshPlaneTerrainCfg(proportion=_FLAT_PROPORTION),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "flat__3": terrain_gen.MeshPlaneTerrainCfg(proportion=_FLAT_PROPORTION),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "flat__4": terrain_gen.MeshPlaneTerrainCfg(proportion=_FLAT_PROPORTION),
        "pyramid_stair_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.01, 0.12),
            step_width=0.3,
            platform_width=2.0,
            border_width=0.5,
        ),
        "flat__5": terrain_gen.MeshPlaneTerrainCfg(proportion=_FLAT_PROPORTION),
        "pyramid_stair": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.01, 0.12),
            step_width=0.3,
            platform_width=2.0,
            border_width=0.5,
        ),
        "flat__6": terrain_gen.MeshPlaneTerrainCfg(proportion=_FLAT_PROPORTION),
    },
)
"""Rough terrain: obstacle columns interleaved with flat ones."""
