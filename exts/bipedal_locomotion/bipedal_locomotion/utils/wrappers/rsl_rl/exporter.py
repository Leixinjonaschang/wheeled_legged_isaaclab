"""ONNX export metadata: the sim-to-real deployment contract (mjlab ``get_wheeled_legged_metadata``).

The deploy runtime (``wf_tron1b_deploy``) validates these fields, so they are written as mjlab
writes them: same keys, same serialisation, joint-indexed values in MJCF joint order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import onnx
import torch
from rsl_rl.models import DepthRepresentationVelocityActorCritic, RepresentationVelocityActorCritic

from bipedal_locomotion.tasks.locomotion.cfg.WF_TRON1B.constants import MJCF_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def list_to_csv_str(arr: list, *, decimals: int = 3, delimiter: str = ",", sub_delimiter: str = ";") -> str:
    """Fixed-precision CSV; nested sequences (per-dim scales, clip ranges) are joined with ``sub_delimiter``."""
    fmt = f"{{:.{decimals}f}}"

    def scalar(x) -> str:
        return fmt.format(x) if isinstance(x, (int, float)) else str(x)

    return delimiter.join(sub_delimiter.join(map(scalar, x)) if isinstance(x, (list, tuple)) else scalar(x) for x in arr)


def attach_metadata_to_onnx(onnx_path: str, metadata: dict[str, Any]) -> None:
    """Attach string metadata to an ONNX model file, in place."""
    model = onnx.load(onnx_path)
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = list_to_csv_str(value) if isinstance(value, list) else str(value)
        model.metadata_props.append(entry)
    onnx.save(model, onnx_path)


def _plain(value):
    return value.tolist() if isinstance(value, torch.Tensor) else value


def get_wheeled_legged_metadata(env: ManagerBasedRLEnv, run_path: str, policy=None) -> dict[str, Any]:
    """Deployment metadata for a WF-TRON1B policy: mixed leg-position / wheel-velocity actions and the
    proprioceptive observation terms the deployed student consumes."""
    robot = env.scene["robot"]
    joint_ids = [robot.joint_names.index(name) for name in MJCF_JOINT_NAMES]

    action_names, action_target_names, action_scales = [], [], []
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        action_names += [name] * term.action_dim
        action_target_names += list(term._joint_names)
        scale = _plain(term._scale)
        action_scales += scale[0] if isinstance(scale, list) else [float(scale)] * term.action_dim

    obs_names = env.observation_manager.active_terms["proprio_history"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["proprio_history"]
    # Nominal (not randomised) drive gains, as mjlab reads them from the actuators' gain parameter: a
    # velocity servo's (zero-stiffness) gain is its kv.
    stiffness = robot.data.default_joint_stiffness[0, joint_ids]
    damping = robot.data.default_joint_damping[0, joint_ids]
    metadata: dict[str, Any] = {
        "run_path": run_path,
        "joint_names": list(MJCF_JOINT_NAMES),
        "joint_stiffness": torch.where(stiffness > 0.0, stiffness, damping).tolist(),
        "joint_damping": damping.tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0, joint_ids].tolist(),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": obs_names,
        "observation_terms_scale": [1.0 if cfg.scale is None else _plain(cfg.scale) for cfg in obs_cfgs],
        "observation_terms_flatten_history_dim": [cfg.flatten_history_dim for cfg in obs_cfgs],
        "observation_terms_history_length": [cfg.history_length for cfg in obs_cfgs],
        "observation_terms_clip": [[-float("inf"), float("inf")] if cfg.clip is None else list(cfg.clip) for cfg in obs_cfgs],
        "action_names": action_names,
        "action_target_names": action_target_names,
        "action_scale": action_scales,
    }
    if isinstance(policy, RepresentationVelocityActorCritic):
        depth = isinstance(policy, DepthRepresentationVelocityActorCritic)
        history_cfg = env.cfg.observations.proprio_history
        metadata.update(
            {
                "policy_input_names": ["proprio_history", "actor_command"] + (["depth", "hidden_state_in"] if depth else []),
                "policy_output_names": ["actions", "predicted_lin_vel"] + (["hidden_state_out"] if depth else []),
                "student_observation_names": obs_names,
                "command_observation_names": env.observation_manager.active_terms["actor_command"],
                "student_history_length": str(history_cfg.history_length),
                "student_history_flatten_dim": str(history_cfg.flatten_history_dim).lower(),
                "student_history_order": "oldest_to_newest",
            }
        )
    return metadata
