"""The ONNX deployment metadata matches what mjlab exports and what ``wf_tron1b_deploy`` validates."""

import math
import sys
from types import SimpleNamespace

import pytest
import torch

from _loader import PKG_DIR, _load, _stub_package

sys.path.insert(0, str(PKG_DIR.parents[2] / "rsl_rl"))
from rsl_rl.models import DepthRepresentationVelocityActorCritic, RepresentationVelocityActorCritic  # noqa: E402

_stub_package("bipedal_locomotion.tasks.locomotion.cfg", PKG_DIR / "tasks" / "locomotion" / "cfg")
_stub_package("bipedal_locomotion.tasks.locomotion.cfg.WF_TRON1B", PKG_DIR / "tasks" / "locomotion" / "cfg" / "WF_TRON1B")
_load("bipedal_locomotion.tasks.locomotion.cfg.WF_TRON1B.constants", "tasks/locomotion/cfg/WF_TRON1B/constants.py")
_stub_package("bipedal_locomotion.utils.wrappers", PKG_DIR / "utils" / "wrappers")
_stub_package("bipedal_locomotion.utils.wrappers.rsl_rl", PKG_DIR / "utils" / "wrappers" / "rsl_rl")
exporter = _load("bipedal_locomotion.utils.wrappers.rsl_rl.exporter", "utils/wrappers/rsl_rl/exporter.py")

MJCF_JOINTS = [
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "wheel_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "wheel_R_Joint",
]  # fmt: skip
SIM_JOINTS = [MJCF_JOINTS[i] for i in (0, 4, 1, 5, 2, 6, 3, 7)]  # PhysX's breadth-first order
# What wf_tron1b_deploy/rl-deploy-with-python/mjlab_repts.py expects.
PROPRIO_TERMS = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "wheel_vel", "actions"]
POLICY_ACTION_NAMES = MJCF_JOINTS[:3] + MJCF_JOINTS[4:7] + [MJCF_JOINTS[3], MJCF_JOINTS[7]]


def _env():
    def per_joint(values: dict[str, float]):
        return torch.tensor([[values.get(name, 0.0) for name in SIM_JOINTS]])

    robot = SimpleNamespace(
        joint_names=SIM_JOINTS,
        data=SimpleNamespace(
            default_joint_stiffness=per_joint({n: 40.0 for n in MJCF_JOINTS if "wheel" not in n}),
            default_joint_damping=per_joint({n: 0.5 if "wheel" in n else 1.8 for n in MJCF_JOINTS}),
            default_joint_pos=per_joint({"hip_L_Joint": 0.1, "knee_R_Joint": -0.2}),
        ),
    )
    terms = {
        "leg_pos": SimpleNamespace(action_dim=6, _joint_names=POLICY_ACTION_NAMES[:6], _scale=0.5),
        "wheel_vel": SimpleNamespace(action_dim=2, _joint_names=POLICY_ACTION_NAMES[6:], _scale=10.0),
    }
    scales = {"joint_vel": 0.05, "wheel_vel": 0.05}
    obs_cfgs = [
        SimpleNamespace(
            scale=torch.tensor(scales[n]) if n in scales else None, clip=None, history_length=5, flatten_history_dim=False
        )
        for n in PROPRIO_TERMS
    ]
    return SimpleNamespace(
        scene={"robot": robot},
        action_manager=SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__),
        command_manager=SimpleNamespace(active_terms=["twist"]),
        observation_manager=SimpleNamespace(
            active_terms={"proprio_history": PROPRIO_TERMS, "actor_command": ["command"]},
            _group_obs_term_cfgs={"proprio_history": obs_cfgs},
        ),
        cfg=SimpleNamespace(
            observations=SimpleNamespace(proprio_history=SimpleNamespace(history_length=5, flatten_history_dim=False))
        ),
    )


def test_metadata_matches_mjlab_layout():
    meta = exporter.get_wheeled_legged_metadata(_env(), "local", object.__new__(RepresentationVelocityActorCritic))

    assert meta["joint_names"] == MJCF_JOINTS
    assert meta["joint_stiffness"] == pytest.approx([40.0, 40.0, 40.0, 0.5, 40.0, 40.0, 40.0, 0.5])
    assert meta["joint_damping"] == pytest.approx([1.8, 1.8, 1.8, 0.5, 1.8, 1.8, 1.8, 0.5])
    assert meta["default_joint_pos"] == pytest.approx([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0])
    assert meta["observation_names"] == meta["student_observation_names"] == PROPRIO_TERMS
    assert meta["observation_terms_scale"] == pytest.approx([1.0, 1.0, 1.0, 0.05, 0.05, 1.0])
    assert meta["observation_terms_history_length"] == [5] * 6
    assert meta["observation_terms_flatten_history_dim"] == [False] * 6
    assert meta["observation_terms_clip"] == [[-math.inf, math.inf]] * 6
    assert meta["action_names"] == ["leg_pos"] * 6 + ["wheel_vel"] * 2
    assert meta["action_target_names"] == POLICY_ACTION_NAMES
    assert meta["action_scale"] == [0.5] * 6 + [10.0] * 2
    assert meta["command_observation_names"] == ["command"]
    assert meta["policy_input_names"] == ["proprio_history", "actor_command"]
    assert meta["policy_output_names"] == ["actions", "predicted_lin_vel"]
    assert (meta["student_history_length"], meta["student_history_flatten_dim"]) == ("5", "false")
    assert meta["student_history_order"] == "oldest_to_newest"


def test_depth_policy_declares_recurrent_io():
    meta = exporter.get_wheeled_legged_metadata(_env(), "local", object.__new__(DepthRepresentationVelocityActorCritic))
    assert meta["policy_input_names"] == ["proprio_history", "actor_command", "depth", "hidden_state_in"]
    assert meta["policy_output_names"] == ["actions", "predicted_lin_vel", "hidden_state_out"]


def test_serialisation_matches_mjlab():
    # mjlab.rl.exporter_utils.list_to_csv_str: 3 decimals, nested sequences joined with ";".
    assert exporter.list_to_csv_str([0.05, [-math.inf, math.inf], "a", False]) == "0.050,-inf;inf,a,0.000"
