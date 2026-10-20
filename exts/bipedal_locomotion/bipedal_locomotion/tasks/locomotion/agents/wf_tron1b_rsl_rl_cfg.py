"""RSL-RL runner configurations for the WF-TRON1B velocity tasks (port of mjlab ``rl_cfg.py``).

Factories rather than classes, as in mjlab, so that variants can start from a base config and
override fields. The gym registry calls them through their ``"module:function"`` entry points.
"""

from bipedal_locomotion.utils.wrappers.rsl_rl import (
    RslRlDepthRepresentationVelocityModelCfg,
    RslRlDepthRepresentationVelocityPredictorModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlRepresentationVelocityModelCfg,
    RslRlRepresentationVelocityPredictorTeacherStudentPpoAlgorithmCfg,
    RslRlRepresentationVelocityTeacherStudentPpoAlgorithmCfg,
)

_GAUSSIAN = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
_PPO = {
    "value_loss_coef": 1.0,
    "use_clipped_value_loss": True,
    "clip_param": 0.2,
    "entropy_coef": 0.01,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "learning_rate": 1.0e-3,
    "schedule": "adaptive",
    "gamma": 0.99,
    "lam": 0.95,
    "desired_kl": 0.01,
    "max_grad_norm": 1.0,
}


def wf_tron1b_rep_ts_lin_vel_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Velocity representation teacher-student runner (the blind ``-RepTS-LinVel-BlindGP`` task)."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlRepresentationVelocityModelCfg(
            hidden_dims=(512, 256, 128),
            encoder_hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            latent_dim=64,
            normalize_latent=True,
            distribution_cfg=dict(_GAUSSIAN),
        ),
        algorithm=RslRlRepresentationVelocityTeacherStudentPpoAlgorithmCfg(
            **_PPO,
            student_learning_rate=1.0e-3,
            num_student_substeps=1,
            representation_loss_coef=1.0,
            lin_vel_loss_coef=1.0,
        ),
        obs_groups={
            "proprio_history": ("proprio_history",),
            "actor_command": ("actor_command",),
            "lin_vel_target": ("lin_vel_target",),
            "critic": ("critic", "dynamics_context"),
            "privileged_encoder": ("privileged_encoder", "dynamics_context"),
        },
        experiment_name="wf_tron1b_velocity_rep_ts_lin_vel_latent64",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=30_000,
        clip_actions=2.0,
        upload_model=False,
    )


def wf_tron1b_rep_ts_lin_vel_depth_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Depth runner without a latent dynamics predictor; the base of the predictor runner below."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlDepthRepresentationVelocityModelCfg(
            hidden_dims=(512, 256, 256, 128),
            encoder_hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            latent_dim=64,
            normalize_latent=True,
            depth_feature_dim=64,
            depth_gru_hidden_dim=128,
            depth_channels=(16, 32, 32),
            depth_conv_strides=(2, 2, 1),
            distribution_cfg=dict(_GAUSSIAN),
        ),
        algorithm=RslRlRepresentationVelocityTeacherStudentPpoAlgorithmCfg(
            **_PPO,
            student_learning_rate=1.0e-3,
            num_student_substeps=1,
            num_representation_epochs=1,
            num_representation_mini_batches=4,
            representation_chunk_length=24,
            representation_loss_coef=1.0,
            lin_vel_loss_coef=1.0,
        ),
        obs_groups={
            "proprio_history": ("proprio_history",),
            "actor_command": ("actor_command",),
            "lin_vel_target": ("lin_vel_target",),
            "critic": ("critic", "dynamics_context"),
            "privileged_encoder": ("privileged_encoder", "dynamics_context"),
            "depth_encoder": ("depth_camera",),
            "wheel_roughness": ("wheel_roughness",),
        },
        experiment_name="wf_tron1b_velocity_rep_ts_lin_vel_depth_latent64",
        save_interval=200,
        num_steps_per_env=24,
        max_iterations=30_000,
        clip_actions=2.0,
        upload_model=False,
    )


def wf_tron1b_rep_ts_lin_vel_depth_predict_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Depth runner with multi-horizon latent-and-velocity dynamics prediction (the ``-Depth-Predict-OursGP`` task)."""
    cfg = wf_tron1b_rep_ts_lin_vel_depth_runner_cfg()
    cfg.actor = RslRlDepthRepresentationVelocityPredictorModelCfg(
        hidden_dims=(512, 256, 256, 128),
        encoder_hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        latent_dim=64,
        normalize_latent=True,
        depth_feature_dim=64,
        depth_gru_hidden_dim=128,
        depth_channels=(16, 32, 32),
        depth_conv_strides=(2, 2, 1),
        latent_dynamics_hidden_dims=(128, 256, 256, 128),
        latent_dynamics_horizons=(1, 5, 10),
        distribution_cfg=dict(_GAUSSIAN),
    )
    cfg.algorithm = RslRlRepresentationVelocityPredictorTeacherStudentPpoAlgorithmCfg(
        **_PPO,
        predictor_learning_rate=1.0e-3,
        student_learning_rate=1.0e-3,
        num_student_substeps=1,
        num_representation_epochs=1,
        num_representation_mini_batches=4,
        representation_chunk_length=24,
        representation_loss_coef=1.0,
        lin_vel_loss_coef=1.0,
        latent_dynamics_loss_coef=3.0,
        latent_dynamics_velocity_loss_coef=1.0,
        latent_dynamics_use_ema_target=False,
        latent_dynamics_ema_decay=0.99,
        latent_dynamics_horizons=(1, 5, 10),
        latent_dynamics_horizon_weights=(1.0, 0.75, 0.5),
        latent_dynamics_detach_source=False,
        latent_rollout_horizon=5,
        latent_rollout_loss_coef=0.75,
        num_latent_dynamics_epochs=1,
        num_latent_dynamics_mini_batches=4,
    )
    cfg.experiment_name = "wf_tron1b_velocity_rep_ts_lin_vel_depth_predict_latent64"
    return cfg
