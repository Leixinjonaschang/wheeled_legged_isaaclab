"""Domain-randomisation events that also record what they randomised.

Isaac Lab's stock randomisation writes into PhysX and keeps no record. The
teacher-student tasks need the sampled values back as a privileged observation, so each
event here writes to the simulator *and* to
:class:`~.domain_randomization.DomainRandomizationState`.

Recording, not re-reading, is deliberate: reading friction back out of PhysX returns the
bucketed value actually installed, which for the stock material randomisation is one of
a small pool of shared buckets rather than the per-environment draw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .domain_randomization import get_dr_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "randomize_base_com",
    "randomize_encoder_bias",
    "randomize_terrain",
    "randomize_wheel_friction",
]


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(device=env.device, dtype=torch.long)


_SHAPE_COUNTS_ATTR = "_wf_tron1b_num_shapes_per_body"


def _num_shapes_per_body(asset) -> list[int]:
    """Collision-shape count for each rigid body, cached on the asset.

    PhysX indexes material properties by collision *shape*, not by body, and a body may
    own several shapes -- on WF-TRON1B each ``knee_*_Link`` carries both its own
    collision geom and the wheel-motor geom. Indexing the material tensor with a body
    index would therefore write to the wrong shape for every body after the first
    multi-shape one, silently randomising the friction of some other link.
    """
    cached = getattr(asset, _SHAPE_COUNTS_ATTR, None)
    if cached is not None:
        return cached
    counts = []
    for link_path in asset.root_physx_view.link_paths[0]:
        link_view = asset._physics_sim_view.create_rigid_body_view(link_path)
        counts.append(link_view.max_shapes)
    setattr(asset, _SHAPE_COUNTS_ATTR, counts)
    return counts


def randomize_wheel_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_range: tuple[float, float],
) -> None:
    """Sample a per-wheel ground friction coefficient and install it.

    Wheel-ground friction is the single most consequential unmodelled parameter for a
    wheeled robot -- it sets how much of a commanded wheel velocity becomes forward
    motion. Unlike ``randomize_rigid_body_material`` this draws an independent value per
    environment rather than assigning from a shared bucket pool, so the value recorded in
    the randomisation state genuinely describes that environment.

    Sets static and dynamic friction to the same value and leaves restitution untouched.
    """
    asset = env.scene[asset_cfg.name]
    env_ids = _resolve_env_ids(env, env_ids)
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))

    low, high = friction_range
    samples = torch.empty(len(env_ids), len(body_ids), device=env.device).uniform_(low, high)

    # [num_envs, num_shapes, 3] holding (static friction, dynamic friction, restitution).
    # The buffer and the indices are CPU-side.
    materials = asset.root_physx_view.get_material_properties()
    shape_counts = _num_shapes_per_body(asset)
    cpu_env_ids = env_ids.cpu()
    cpu_samples = samples.cpu()
    for column, body_id in enumerate(body_ids):
        start = sum(shape_counts[:body_id])
        end = start + shape_counts[body_id]
        materials[cpu_env_ids, start:end, 0] = cpu_samples[:, column : column + 1]
        materials[cpu_env_ids, start:end, 1] = cpu_samples[:, column : column + 1]
    asset.root_physx_view.set_material_properties(materials, cpu_env_ids)

    get_dr_state(env, num_wheels=len(body_ids)).wheel_friction[env_ids] = samples


def randomize_encoder_bias(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    bias_range: tuple[float, float],
) -> None:
    """Sample a constant per-joint position-measurement offset.

    Models joint-encoder calibration error. There is nothing to write into the simulator:
    the bias is applied by the position action term, which subtracts it from the target
    so that the *commanded* angle is wrong by exactly the amount the real encoder would
    be wrong by. The robot's true joint angles stay clean, which is what makes this a
    measurement error rather than a disturbance.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(env.scene[asset_cfg.name].num_joints))

    low, high = bias_range
    samples = torch.empty(len(env_ids), len(joint_ids), device=env.device).uniform_(low, high)
    get_dr_state(env).encoder_bias[env_ids[:, None], torch.tensor(joint_ids, device=env.device)] = samples


def randomize_base_com(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> None:
    """Offset the base centre of mass from its nominal position.

    A shifted COM changes the pitch dynamics the wheels have to stabilise, which is the
    dominant source of sim-to-real mismatch for a wheeled biped carrying an unmodelled
    payload.
    """
    asset = env.scene[asset_cfg.name]
    env_ids = _resolve_env_ids(env, env_ids)
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))
    if len(body_ids) != 1:
        raise ValueError(f"randomize_base_com expects exactly one body, got {len(body_ids)}")
    body_id = body_ids[0]

    deltas = torch.stack(
        [torch.empty(len(env_ids), device=env.device).uniform_(*com_range[axis]) for axis in range(3)],
        dim=1,
    )

    # get_coms() returns [num_envs, num_bodies, 7]: position followed by quaternion.
    coms = asset.root_physx_view.get_coms()
    cpu_env_ids = env_ids.cpu()
    state = get_dr_state(env)
    previous = state.base_com_delta[env_ids]
    # Re-base rather than accumulate: startup events may run more than once across a
    # resume, and the recorded delta must stay an offset from nominal.
    coms[cpu_env_ids, body_id, :3] += (deltas - previous).cpu()
    asset.root_physx_view.set_coms(coms, cpu_env_ids)

    state.base_com_delta[env_ids] = deltas


def randomize_terrain(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Respawn on a randomly chosen sub-terrain patch.

    Used only by the play configs, which disable the terrain-level curriculum: without a
    curriculum every environment would otherwise stay on whichever patch it started on,
    making a play session show a single difficulty.
    """
    terrain = env.scene.terrain
    # `terrain_origins` is None for a plane, and `terrain_levels` / `terrain_types` only
    # exist once the importer has configured curriculum origins.
    if terrain is None or getattr(terrain, "terrain_origins", None) is None:
        return
    if getattr(terrain, "terrain_levels", None) is None or getattr(terrain, "terrain_types", None) is None:
        return

    env_ids = _resolve_env_ids(env, env_ids)
    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    rows = torch.randint(0, num_rows, (len(env_ids),), device=env.device)
    cols = torch.randint(0, num_cols, (len(env_ids),), device=env.device)
    terrain.terrain_levels[env_ids] = rows
    terrain.terrain_types[env_ids] = cols
    env.scene.env_origins[env_ids] = terrain.terrain_origins[rows, cols]
