"""RSL-RL runner configurations for the WF-TRON1B velocity tasks.

Phase 1 ships the plain PPO runner only. The teacher-student representation
runners (``-RepTS``, ``-RepTS-LinVel``, ``-Depth``, ``-Predict``) land with the
observation groups they consume, so that ``obs_groups`` never names a group the
environment does not emit.
"""

from dataclasses import dataclass, field

from bipedal_locomotion.utils.wrappers.rsl_rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def _wf_tron1b_algorithm() -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@dataclass
class WFTron1BPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Plain PPO for the flat and rough velocity tasks.

    Observation normalisation is on for both networks: the concatenated actor group
    mixes quantities on very different scales (projected gravity in [-1, 1], wheel
    velocity in tens of rad/s), and the fixed per-term scales alone do not equalise
    them across the terrain curriculum.
    """

    actor: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        )
    )
    critic: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            hidden_dims=(512, 256, 256, 128),
            activation="elu",
            obs_normalization=True,
        )
    )
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=_wf_tron1b_algorithm)
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)}
    )

    experiment_name: str = "wf_tron1b_velocity"
    save_interval: int = 200
    num_steps_per_env: int = 24
    max_iterations: int = 30_000
    # The leg-position term multiplies by 0.5 and offsets by the default joint position,
    # so an unclipped Gaussian sample can command far outside the joint limit and let the
    # solver, rather than the policy, decide the pose.
    clip_actions: float | None = 2.0
    upload_model: bool = False
