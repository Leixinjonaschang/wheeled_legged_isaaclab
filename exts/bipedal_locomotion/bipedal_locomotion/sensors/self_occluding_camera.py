"""Ray-caster depth camera that also sees selected links of the robot it is mounted on.

Isaac Lab's ``RayCasterCamera`` casts against one static mesh (the terrain). mjlab's camera, a mujoco_warp
BVH ray caster over the terrain and the robot's visual geoms, also sees the robot itself: a stepping policy
routinely has its wheels in the bottom of the image. This sensor additionally casts the camera rays against
the selected links' visual meshes, in each link's frame, and keeps the nearer hit per pixel.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import trimesh

import isaaclab.utils.math as math_utils
from isaaclab.sensors import RayCasterCamera, RayCasterCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh


class SelfOccludingRayCasterCamera(RayCasterCamera):
    """``RayCasterCamera`` whose ``distance_to_image_plane`` also includes the robot's own links."""

    cfg: SelfOccludingRayCasterCameraCfg

    def perturb_mount(
        self, env_ids: Sequence[int], position: torch.Tensor, pitch: torch.Tensor, fovy_deg: torch.Tensor
    ) -> None:
        """Mount the camera at its configured pose offset by ``position`` (parent frame) and rotated by ``pitch``
        about the parent's y axis, with the configured vertical field of view plus ``fovy_deg``.

        Always relative to the configured mount, so repeated calls do not accumulate.
        """
        zeros = torch.zeros_like(pitch)
        self._offset_pos[env_ids] = self._default_offset_pos[env_ids] + position
        self._offset_quat[env_ids] = math_utils.quat_mul(
            math_utils.quat_from_euler_xyz(zeros, pitch, zeros), self._default_offset_quat[env_ids]
        )
        intrinsics = self._default_intrinsics[env_ids].clone()
        half_height = 0.5 * self.image_shape[0]
        fovy = 2.0 * torch.atan(half_height / intrinsics[:, 1, 1]) + torch.deg2rad(fovy_deg)
        intrinsics[:, 0, 0] = intrinsics[:, 1, 1] = half_height / torch.tan(0.5 * fovy)  # square pixels
        self.set_intrinsic_matrices(intrinsics, env_ids=env_ids)

    def _initialize_impl(self):
        super()._initialize_impl()
        robot_path = self.cfg.prim_path.rsplit("/", 1)[0]
        self._link_views, self._link_meshes = [], []
        for link in self.cfg.self_occlusion_links:
            view = self._physics_sim_view.create_rigid_body_view(f"{robot_path}/{link}".replace(".*", "*"))
            if view.count != self._view.count:
                raise RuntimeError(f"Expected {self._view.count} '{link}' prims, found {view.count}.")
            mesh = trimesh.load(os.path.join(self.cfg.mesh_dir, f"{link}.STL"))
            self._link_views.append(view)
            self._link_meshes.append(
                convert_to_warp_mesh(mesh.vertices.astype("float32"), mesh.faces.astype("int32"), self._device)
            )

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        self._default_offset_pos = self._offset_pos.clone()
        self._default_offset_quat = self._offset_quat.clone()
        self._default_intrinsics = self._data.intrinsic_matrices.clone()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        if not self._link_views:
            return
        quat_w = self._data.quat_w_world[env_ids][:, None].expand(-1, self.num_rays, -1)
        starts_w = math_utils.quat_apply(quat_w, self.ray_starts[env_ids]) + self._data.pos_w[env_ids][:, None]
        directions_w = math_utils.quat_apply(quat_w, self.ray_directions[env_ids])
        distance = torch.full(directions_w.shape[:2], float("inf"), device=self._device)
        for view, mesh in zip(self._link_views, self._link_meshes):
            link_pos, link_quat = view.get_transforms()[env_ids].split([3, 4], dim=-1)
            link_quat = math_utils.convert_quat(link_quat, to="wxyz")[:, None].expand(-1, self.num_rays, -1)
            starts = math_utils.quat_apply_inverse(link_quat, starts_w - link_pos[:, None]).contiguous()
            directions = math_utils.quat_apply_inverse(link_quat, directions_w).contiguous()
            _, link_distance, _, _ = raycast_mesh(starts, directions, mesh, return_distance=True)
            distance = torch.minimum(distance, link_distance)
        # Unit rays along the camera's forward (+x) axis: planar depth is distance times the x component.
        planar = (distance * self.ray_directions[env_ids][..., 0]).view(-1, *self.image_shape, 1)
        output = self._data.output["distance_to_image_plane"]
        output[env_ids] = torch.fmin(output[env_ids], planar)  # fmin: a terrain miss is NaN, not far


@configclass
class SelfOccludingRayCasterCameraCfg(RayCasterCameraCfg):
    """Configuration for :class:`SelfOccludingRayCasterCamera`. ``prim_path`` must be a link of the robot."""

    class_type: type = SelfOccludingRayCasterCamera

    self_occlusion_links: tuple[str, ...] = ()
    """Links of the camera's robot (siblings of the mounting link) whose visual meshes the camera sees."""

    mesh_dir: str = ""
    """Directory holding ``<link>.STL`` for every link in :attr:`self_occlusion_links`, in the link frame."""
