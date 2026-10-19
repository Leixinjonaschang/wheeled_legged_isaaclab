"""mjlab's terrain-curriculum semantics on top of Isaac Lab's generator and importer.

Isaac Lab's curriculum allocates *columns* by ``proportion``, spreads environments evenly
over the columns and jitters each row's difficulty. mjlab instead gives every sub-terrain
exactly one column, uses a deterministic difficulty per row, and spreads *environments*
over the columns by ``proportion``. With the interleaved flat columns of the WF-TRON1B
terrain the two schemes would put 50% versus 28% of the robots on flat ground.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.terrains import TerrainGenerator, TerrainImporter


def proportional_counts(num_envs: int, proportions: np.ndarray) -> np.ndarray:
    """Split ``num_envs`` by ``proportions`` (largest remainder, at least one per bucket if possible)."""
    n = len(proportions)
    counts = np.ones(n, dtype=int) if num_envs >= n else np.zeros(n, dtype=int)
    remaining = num_envs - counts.sum()
    if remaining > 0:
        ideal = proportions * remaining
        counts += np.floor(ideal).astype(int)
        leftover = num_envs - counts.sum()
        counts[np.argsort(-(ideal - np.floor(ideal)))[:leftover]] += 1
    return counts


class ColumnPerTerrainGenerator(TerrainGenerator):
    """Curriculum with one column per sub-terrain and difficulty linear in the row index."""

    def __init__(self, cfg, device: str = "cpu"):
        if cfg.curriculum:
            cfg.num_cols = len(cfg.sub_terrains)
        super().__init__(cfg, device)

    def _generate_curriculum_terrains(self):
        lower, upper = self.cfg.difficulty_range
        for col, sub_cfg in enumerate(self.cfg.sub_terrains.values()):
            for row in range(self.cfg.num_rows):
                difficulty = lower + (upper - lower) * row / max(self.cfg.num_rows - 1, 1)
                mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
                self._add_sub_terrain(mesh, origin, row, col, sub_cfg)


class ProportionalTerrainImporter(TerrainImporter):
    """Assigns environments to terrain columns in proportion to each sub-terrain's ``proportion``."""

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        env_origins = super()._compute_env_origins_curriculum(num_envs, origins)
        proportions = np.array([sub.proportion for sub in self.cfg.terrain_generator.sub_terrains.values()])
        if len(proportions) == origins.shape[1]:
            counts = proportional_counts(num_envs, proportions / proportions.sum())
            columns = torch.arange(len(counts), device=self.device)
            self.terrain_types = torch.repeat_interleave(columns, torch.as_tensor(counts, device=self.device))
            env_origins[:] = origins[self.terrain_levels, self.terrain_types]
        return env_origins
