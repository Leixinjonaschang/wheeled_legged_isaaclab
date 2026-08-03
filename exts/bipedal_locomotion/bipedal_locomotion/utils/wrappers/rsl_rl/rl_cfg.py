"""Configuration dataclasses for the vendored RSL-RL (``rsl-rl-lib`` 5.3.0).

These are plain dataclasses rather than Isaac Lab ``@configclass`` objects: the
runner consumes ``dataclasses.asdict(cfg)`` and resolves model / algorithm
classes by the ``class_name`` field, so the nesting produced by ``asdict`` *is*
the interface. Wrapping them in ``configclass`` would add nothing and would
diverge from the upstream RSL-RL examples this fork tracks.

Note that ``isaaclab_rl.rsl_rl`` provides similarly named dataclasses. Those
target whichever ``rsl-rl-lib`` the installed Isaac Lab pins and use a
``policy:`` / ``empirical_normalization`` layout; 5.3.0 wants ``actor:`` /
``critic:`` / ``obs_groups:``. Do not mix the two.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RslRlModelCfg:
    """Config for a single neural network model (actor or critic)."""

    hidden_dims: tuple[int, ...] = (128, 128, 128)
    """The hidden dimensions of the network."""
    activation: str = "elu"
    """The activation function."""
    obs_normalization: bool = False
    """Whether to normalize the observations."""
    cnn_cfg: dict[str, Any] | None = None
    """CNN encoder config. When set, ``class_name`` should be ``"CNNModel"``.

    Passed to ``rsl_rl.modules.CNN``. Common keys: output_channels, kernel_size,
    stride, padding, activation, global_pool, max_pool.
    """
    distribution_cfg: dict[str, Any] | None = None
    """Distribution config dict passed to RSL-RL. Example::

        {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}

    ``None`` means deterministic output (use for the critic).
    """
    rnn_type: str | None = None
    """RNN type ("lstm" or "gru"). When set, ``class_name`` should be ``"RNNModel"``."""
    rnn_hidden_dim: int = 256
    """Hidden state dimension for the RNN."""
    rnn_num_layers: int = 1
    """Number of stacked RNN layers."""
    class_name: str = "MLPModel"
    """Model class name resolved by RSL-RL (``MLPModel``, ``CNNModel``, ``RNNModel``),
    or a fully qualified ``"module.path:ClassName"`` for classes outside the default
    RSL-RL namespace."""


@dataclass
class RslRlPpoAlgorithmCfg:
    """Config for the PPO algorithm."""

    num_learning_epochs: int = 5
    """The number of learning epochs per update."""
    num_mini_batches: int = 4
    """The number of mini-batches per update.

    mini batch size = num_envs * num_steps / num_mini_batches
    """
    learning_rate: float = 1e-3
    """The learning rate."""
    schedule: Literal["adaptive", "fixed"] = "adaptive"
    """The learning rate schedule."""
    gamma: float = 0.99
    """The discount factor."""
    lam: float = 0.95
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""
    entropy_coef: float = 0.005
    """The coefficient for the entropy loss."""
    desired_kl: float = 0.01
    """The desired KL divergence between the new and old policies."""
    max_grad_norm: float = 1.0
    """The maximum gradient norm for the policy."""
    value_loss_coef: float = 1.0
    """The coefficient for the value loss."""
    use_clipped_value_loss: bool = True
    """Whether to use clipped value loss."""
    clip_param: float = 0.2
    """The clipping parameter for the policy."""
    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize the advantage per mini-batch. If True, the advantage is
    normalized over the mini-batches only; otherwise over the entire collected
    trajectories."""
    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """The optimizer to use."""
    share_cnn_encoders: bool = False
    """Share CNN encoders between actor and critic."""
    class_name: str = "PPO"
    """Algorithm class name resolved by RSL-RL."""


@dataclass
class RslRlBaseRunnerCfg:
    """Fields shared by every runner configuration."""

    seed: int = 42
    """The seed for the experiment."""
    num_steps_per_env: int = 24
    """The number of steps per environment update."""
    max_iterations: int = 300
    """The maximum number of iterations."""
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)},
    )
    """Maps a model's observation *set* to the environment observation *groups* that
    feed it. Must stay in sync with the groups the observation manager emits."""
    save_interval: int = 50
    """The number of iterations between saves."""
    experiment_name: str = "exp1"
    """Directory name used to group runs under ``logs/rsl_rl/{experiment_name}/``."""
    run_name: str = ""
    """Optional label appended to the timestamped run directory. Also becomes the
    display name for the run in W&B."""
    logger: Literal["wandb", "tensorboard", "neptune"] = "tensorboard"
    """The logger to use."""
    wandb_project: str = "tron1b_rl_lab"
    """The W&B project name."""
    wandb_tags: tuple[str, ...] = ()
    """Tags for the W&B run."""
    resume: bool = False
    """Whether to resume the experiment."""
    load_run: str = ".*"
    """The run directory to load. If a regex, the latest (alphabetical) match is used."""
    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. If a regex, the latest (alphabetical) match is used."""
    clip_actions: float | None = None
    """The clipping range for action values. ``None`` means no clipping."""
    upload_model: bool = False
    """Whether to upload model files (.pt, .onnx) to W&B on save."""


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
    """Configuration for the on-policy runner."""

    class_name: str = "OnPolicyRunner"
    """The runner class name."""
    actor: RslRlModelCfg = field(
        default_factory=lambda: RslRlModelCfg(
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            }
        )
    )
    """The actor configuration."""
    critic: RslRlModelCfg = field(default_factory=RslRlModelCfg)
    """The critic configuration."""
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
    """The algorithm configuration."""
