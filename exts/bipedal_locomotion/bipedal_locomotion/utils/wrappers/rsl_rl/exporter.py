"""ONNX export helpers and the sim-to-real deployment metadata contract.

The exported ONNX file carries the joint order, action scales, default joint
positions and observation term names that the deployment runtime must reproduce
exactly. Deployment should read these from the ONNX metadata rather than
hard-coding them; see ``docs/deployment-scale-notes.md`` in the mjlab task
repository for the layout this mirrors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import onnx
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def list_to_csv_str(arr: list, *, decimals: int = 6, delimiter: str = ",") -> str:
    """Convert a list to a CSV string, formatting numbers to a fixed precision."""
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr)


def attach_metadata_to_onnx(onnx_path: str, metadata: dict[str, Any]) -> None:
    """Attach string metadata to an ONNX model file, in place."""
    model = onnx.load(onnx_path)
    for key, value in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = key
        entry.value = list_to_csv_str(value) if isinstance(value, list) else str(value)
        model.metadata_props.append(entry)
    onnx.save(model, onnx_path)


def _action_term_joint_names(action_term) -> list[str]:
    """Resolved joint names for an action term, in the term's own action order."""
    for attr in ("_joint_names", "joint_names"):
        names = getattr(action_term, attr, None)
        if names:
            return list(names)
    raise AttributeError(
        f"Action term {type(action_term).__name__} exposes neither `_joint_names` nor "
        "`joint_names`; the deployment metadata cannot record which joint each action "
        "drives. Update this helper for the installed Isaac Lab version."
    )


def _action_term_scales(action_term) -> list[float]:
    """Per-dimension action scale, broadcast to the term's action dimension."""
    scale = getattr(action_term, "_scale", None)
    if scale is None:
        scale = getattr(action_term, "scale", 1.0)
    if isinstance(scale, torch.Tensor):
        # Stored as [num_envs, action_dim] when resolved per joint.
        flat = scale[0] if scale.ndim > 1 else scale
        return [float(v) for v in flat.detach().cpu().tolist()]
    return [float(scale)] * action_term.action_dim


def _joint_gains(robot) -> tuple[list[float], list[float]]:
    """Per-joint drive stiffness and damping in the articulation's joint order."""
    stiffness = getattr(robot.data, "joint_stiffness", None)
    damping = getattr(robot.data, "joint_damping", None)
    if stiffness is None or damping is None:
        raise AttributeError(
            "ArticulationData exposes neither `joint_stiffness` nor `joint_damping`; the "
            "deployment metadata cannot record the drive gains. Update this helper for the "
            "installed Isaac Lab version."
        )
    return (
        [float(v) for v in stiffness[0].detach().cpu().tolist()],
        [float(v) for v in damping[0].detach().cpu().tolist()],
    )


def get_wheeled_legged_metadata(env: ManagerBasedRLEnv, run_path: str, policy=None) -> dict[str, Any]:
    """Build the deployment metadata for a WF-TRON1B policy.

    Handles *mixed* position / velocity action terms: unlike a uniformly
    position-controlled robot, the action vector here is a concatenation of a
    6-dimensional leg-position term (scale 0.5, offset by the default joint position)
    and a 2-dimensional wheel-velocity term (scale 10.0, no offset). The per-dimension
    ``action_names`` / ``action_target_names`` / ``action_scale`` lists make that split
    explicit to the deployment side.

    The observation term names are emitted for whichever proprioceptive group the
    policy class actually consumes, so a student policy does not advertise the
    teacher's privileged observations.
    """
    robot = env.scene["robot"]

    action_names: list[str] = []
    action_target_names: list[str] = []
    action_scales: list[float] = []
    for action_name in env.action_manager.active_terms:
        term = env.action_manager.get_term(action_name)
        action_names.extend([action_name] * term.action_dim)
        action_target_names.extend(_action_term_joint_names(term))
        action_scales.extend(_action_term_scales(term))

    joint_stiffness, joint_damping = _joint_gains(robot)

    obs_terms = env.observation_manager.active_terms
    proprio_group = "actor" if "actor" in obs_terms else "proprio_history"

    metadata: dict[str, Any] = {
        "run_path": run_path,
        "joint_names": list(robot.joint_names),
        "joint_stiffness": joint_stiffness,
        "joint_damping": joint_damping,
        "default_joint_pos": [float(v) for v in robot.data.default_joint_pos[0].detach().cpu().tolist()],
        "command_names": list(env.command_manager.active_terms),
        "observation_names": list(obs_terms[proprio_group]),
        "action_names": action_names,
        "action_target_names": action_target_names,
        "action_scale": action_scales,
    }

    # Representation policies take history (and optionally a command) as separate ONNX
    # inputs, so record the input names and the history layout. `oldest_to_newest` is a
    # contract with the observation manager's history buffer; `tests/` pins it.
    policy_cls = type(policy).__name__ if policy is not None else ""
    if policy_cls == "RepresentationActorCritic":
        history_cfg = env.cfg.observations.actor_history
        metadata.update(
            {
                "policy_input_names": ["student_history"],
                "student_observation_names": list(obs_terms["actor_history"]),
                "student_history_length": str(history_cfg.history_length),
                "student_history_flatten_dim": str(history_cfg.flatten_history_dim).lower(),
                "student_history_order": "oldest_to_newest",
            }
        )
    elif "RepresentationVelocity" in policy_cls:
        history_cfg = env.cfg.observations.proprio_history
        metadata.update(
            {
                "policy_input_names": ["proprio_history", "actor_command"],
                "policy_output_names": ["actions", "predicted_lin_vel"],
                "student_observation_names": list(obs_terms["proprio_history"]),
                "command_observation_names": list(obs_terms["actor_command"]),
                "student_history_length": str(history_cfg.history_length),
                "student_history_flatten_dim": str(history_cfg.flatten_history_dim).lower(),
                "student_history_order": "oldest_to_newest",
            }
        )
    return metadata
