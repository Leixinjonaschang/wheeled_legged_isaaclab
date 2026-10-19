"""Startup domain randomisation (mjlab's set and order), recording what each event sampled, and the
depth camera's per-reset calibration randomisation.

Isaac Lab's stock randomisation writes into PhysX and keeps no record. Every startup event here also
stores its draw in :class:`~.domain_randomization.DomainRandomizationState`, which the
privileged ``dynamics_context`` observation reads back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import ImplicitActuator
from isaaclab.managers import SceneEntityCfg

from .domain_randomization import get_dr_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "randomize_com_offset",
    "randomize_depth_camera",
    "randomize_encoder_bias",
    "randomize_mass_inertia",
    "randomize_pd_gains",
    "randomize_terrain",
    "randomize_wheel_friction",
]


def _resolve_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(device=env.device, dtype=torch.long)


def _ids(ids: list[int] | slice, count: int) -> torch.Tensor:
    """Resolved ``SceneEntityCfg`` ids as a (CPU) index tensor."""
    return torch.arange(count) if isinstance(ids, slice) else torch.tensor(ids)


_SHAPE_COUNTS_ATTR = "_wf_tron1b_num_shapes_per_body"


def _num_shapes_per_body(asset) -> list[int]:
    """Collision-shape count of each body, cached on the asset.

    PhysX indexes materials by shape, not body, and a body may own several shapes (each knee
    carries its own collider and the wheel-motor one), so body ids cannot index materials.
    """
    counts = getattr(asset, _SHAPE_COUNTS_ATTR, None)
    if counts is None:
        sim_view = asset._physics_sim_view
        counts = [sim_view.create_rigid_body_view(path).max_shapes for path in asset.root_physx_view.link_paths[0]]
        if sum(counts) != asset.root_physx_view.max_shapes:
            raise RuntimeError(f"Shape counts {counts} do not add up to {asset.root_physx_view.max_shapes} shapes.")
        setattr(asset, _SHAPE_COUNTS_ATTR, counts)
    return counts


def randomize_wheel_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_range: tuple[float, float],
    difference_range: tuple[float, float],
) -> None:
    """One friction draw per environment shared by the wheels, plus a small per-wheel difference.

    mjlab's ``wheel_friction`` (abs, shared) and ``wheel_friction_difference`` (add) events in one.
    Writes static = dynamic friction to every shape of each wheel; restitution and combine modes
    are untouched.
    """
    asset = env.scene[asset_cfg.name]
    env_ids = _resolve_env_ids(env, env_ids)
    body_ids = _ids(asset_cfg.body_ids, asset.num_bodies).tolist()
    shared = torch.empty(len(env_ids), 1, device=env.device).uniform_(*friction_range)
    friction = shared + torch.empty(len(env_ids), len(body_ids), device=env.device).uniform_(*difference_range)

    # [num_envs, num_shapes, 3]: static friction, dynamic friction, restitution (CPU).
    materials = asset.root_physx_view.get_material_properties()
    shape_counts = _num_shapes_per_body(asset)
    cpu_env_ids = env_ids.cpu()
    for column, body_id in enumerate(body_ids):
        start = sum(shape_counts[:body_id])
        materials[cpu_env_ids, start : start + shape_counts[body_id], :2] = friction[:, column, None, None].cpu()
    asset.root_physx_view.set_material_properties(materials, cpu_env_ids)
    get_dr_state(env).wheel_friction[env_ids] = friction


def randomize_mass_inertia(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    alpha_range: tuple[float, float],
) -> None:
    """Scale each body's mass and inertia by exp(2 alpha), alpha ~ U(alpha_range), from nominal.

    mjlab's ``pseudo_inertia`` with only its density term, which leaves the COM in place.
    """
    asset = env.scene[asset_cfg.name]
    env_ids = _resolve_env_ids(env, env_ids)
    body_ids = _ids(asset_cfg.body_ids, asset.num_bodies)
    scale = torch.exp(2.0 * torch.empty(len(env_ids), len(body_ids), device=env.device).uniform_(*alpha_range))

    cpu_env_ids, cpu_scale = env_ids.cpu(), scale.cpu()
    rows = cpu_env_ids[:, None]
    masses = asset.root_physx_view.get_masses()
    inertias = asset.root_physx_view.get_inertias()
    masses[rows, body_ids] = asset.data.default_mass[rows, body_ids] * cpu_scale
    inertias[rows, body_ids] = asset.data.default_inertia[rows, body_ids] * cpu_scale[..., None]
    asset.root_physx_view.set_masses(masses, cpu_env_ids)
    asset.root_physx_view.set_inertias(inertias, cpu_env_ids)
    get_dr_state(env).mass_scale[env_ids[:, None], body_ids.to(env.device)] = scale


def randomize_encoder_bias(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    bias_range: tuple[float, float],
) -> None:
    """Per-joint encoder calibration offset, subtracted from the target by the position action term.

    Nothing is written to the simulator: the true joint state stays clean and observations use the
    unbiased joint positions, as in mjlab.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    joint_ids = _ids(asset_cfg.joint_ids, env.scene[asset_cfg.name].num_joints).to(env.device)
    bias = torch.empty(len(env_ids), len(joint_ids), device=env.device).uniform_(*bias_range)
    get_dr_state(env).encoder_bias[env_ids[:, None], joint_ids] = bias


def randomize_pd_gains(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_scale_range: tuple[float, float],
    damping_scale_range: tuple[float, float],
) -> None:
    """Scale servo gains from nominal, per joint: Kp of the position servos, Kd / Kv of all.

    Only the drive gains are scaled, as mjlab scales only the actuator kp / kv (the passive joint
    damping is the drive-independent ``viscous_friction``). Both the PhysX drives and the actuator
    models are updated, since the latter's gains feed ``applied_torque``.
    """
    asset = env.scene[asset_cfg.name]
    env_ids = _resolve_env_ids(env, env_ids)
    joint_ids = _ids(asset_cfg.joint_ids, asset.num_joints).to(env.device)
    rows = env_ids[:, None]
    kp0 = asset.data.default_joint_stiffness[rows, joint_ids]
    kd0 = asset.data.default_joint_damping[rows, joint_ids]
    position_servo = kp0 > 0.0  # velocity servos have no Kp
    kp_scale = torch.where(position_servo, torch.empty_like(kp0).uniform_(*stiffness_scale_range), 1.0)
    kd_scale = torch.empty_like(kd0).uniform_(*damping_scale_range)

    stiffness = asset.data.joint_stiffness[env_ids].clone()
    damping = asset.data.joint_damping[env_ids].clone()
    stiffness[:, joint_ids] = kp0 * kp_scale
    damping[:, joint_ids] = kd0 * kd_scale
    for actuator in asset.actuators.values():
        ids = actuator.joint_indices
        actuator.stiffness[env_ids] = stiffness[:, ids]
        actuator.damping[env_ids] = damping[:, ids]
        if isinstance(actuator, ImplicitActuator):
            asset.write_joint_stiffness_to_sim(stiffness[:, ids], joint_ids=ids, env_ids=env_ids)
            asset.write_joint_damping_to_sim(damping[:, ids], joint_ids=ids, env_ids=env_ids)

    state = get_dr_state(env)
    state.stiffness_scale[rows, joint_ids] = kp_scale
    state.damping_scale[rows, joint_ids] = kd_scale


def randomize_com_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_range: tuple[float, float],
) -> None:
    """Offset each selected body's COM from nominal, every axis U(com_range), in the body frame."""
    asset = env.scene[asset_cfg.name]
    env_ids = _resolve_env_ids(env, env_ids)
    body_ids = _ids(asset_cfg.body_ids, asset.num_bodies)
    offset = torch.empty(len(env_ids), len(body_ids), 3, device=env.device).uniform_(*com_range)

    state = get_dr_state(env)
    if state.default_coms is None:  # captured before any COM is moved
        state.default_coms = asset.root_physx_view.get_coms().clone()
    # [num_envs, num_bodies, 7]: COM position then orientation, in the body frame (CPU).
    coms = asset.root_physx_view.get_coms().clone()
    cpu_env_ids = env_ids.cpu()
    rows = cpu_env_ids[:, None]
    coms[rows, body_ids, :3] = state.default_coms[rows, body_ids, :3] + offset.cpu()
    asset.root_physx_view.set_coms(coms, cpu_env_ids)
    state.com_offset[env_ids[:, None], body_ids.to(env.device)] = offset


def randomize_depth_camera(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    position_range: tuple[float, float],
    pitch_range: tuple[float, float],
    fovy_range_deg: tuple[float, float],
) -> None:
    """mjlab's ``cam_pos`` / ``cam_quat`` (pitch) / ``cam_fovy`` reset events: per-environment calibration error.

    Every axis of the mount position, the pitch about the robot's y axis and the vertical field of view are
    offset from their nominal values by a fresh uniform draw (see ``SelfOccludingRayCasterCamera.perturb_mount``).
    """
    env_ids = _resolve_env_ids(env, env_ids)
    count = len(env_ids)
    env.scene[sensor_cfg.name].perturb_mount(
        env_ids,
        position=torch.empty(count, 3, device=env.device).uniform_(*position_range),
        pitch=torch.empty(count, device=env.device).uniform_(*pitch_range),
        fovy_deg=torch.empty(count, device=env.device).uniform_(*fovy_range_deg),
    )


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
