"""Terrains ported from mjlab, and mjlab's terrain-curriculum semantics."""

from .curriculum import ColumnPerTerrainGenerator, ProportionalTerrainImporter, proportional_counts
from .sub_terrains_cfg import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    BoxRandomSpreadTerrainCfg,
    BoxRandomStairsTerrainCfg,
    BoxSteppingStonesTerrainCfg,
    BoxTiltedGridTerrainCfg,
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
)

__all__ = [
    "BoxFlatTerrainCfg",
    "BoxInvertedPyramidStairsTerrainCfg",
    "BoxPyramidStairsTerrainCfg",
    "BoxRandomSpreadTerrainCfg",
    "BoxRandomStairsTerrainCfg",
    "BoxSteppingStonesTerrainCfg",
    "BoxTiltedGridTerrainCfg",
    "ColumnPerTerrainGenerator",
    "HfDiscreteObstaclesTerrainCfg",
    "HfPyramidSlopedTerrainCfg",
    "HfRandomUniformTerrainCfg",
    "ProportionalTerrainImporter",
    "proportional_counts",
]
