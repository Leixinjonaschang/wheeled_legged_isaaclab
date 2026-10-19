"""Quaternion and angle helpers in the (w, x, y, z) convention shared by Isaac Lab and MuJoCo."""

import math

import torch


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by ``quat``.

    Args:
        quat: Quaternions, real part first (w, x, y, z), shape (..., 4).
        vec: Vectors, shape (..., 3), broadcastable against ``quat``.

    Returns:
        The rotated vectors, shape (..., 3).

    The (w, x, y, z) ordering matches Isaac Lab's ``ArticulationData`` and MuJoCo.
    USD and PhysX use (x, y, z, w); never pass one of those straight in.
    """
    shape = torch.broadcast_shapes(quat.shape[:-1], vec.shape[:-1])
    quat = quat.expand(*shape, 4)
    vec = vec.expand(*shape, 3)
    w = quat[..., 0:1]
    xyz = quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by the inverse of ``quat`` (i.e. world -> body).

    Args:
        quat: Quaternions, real part first (w, x, y, z), shape (..., 4).
        vec: Vectors, shape (..., 3), broadcastable against ``quat``.

    Returns:
        The rotated vectors, shape (..., 3).
    """
    conjugate = torch.cat([quat[..., 0:1], -quat[..., 1:]], dim=-1)
    return quat_apply(conjugate, vec)


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles into (-pi, pi]."""
    wrapped = torch.remainder(angles, 2.0 * math.pi)
    return torch.where(wrapped > math.pi, wrapped - 2.0 * math.pi, wrapped)
