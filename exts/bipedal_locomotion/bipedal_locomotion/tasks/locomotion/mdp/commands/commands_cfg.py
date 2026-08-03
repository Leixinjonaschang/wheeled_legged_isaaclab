from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .gait_command import GaitCommand  # Import the GaitCommand class
from .wheeled_velocity_command import WheeledUniformVelocityCommand


@configclass
class UniformGaitCommandCfg(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = GaitCommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] = MISSING
        """Range for gait frequencies [Hz]."""
        offsets: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        durations: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""
        swing_height: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""


@configclass
class WheeledUniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the world-frame velocity command generator.

    Unlike Isaac Lab's ``UniformVelocityCommandCfg``, the sampled linear velocity is
    fixed in the world frame once drawn; the robot must turn to face it. See
    :class:`WheeledUniformVelocityCommand`.
    """

    class_type: type = WheeledUniformVelocityCommand

    asset_name: str = MISSING
    """Name of the articulation to command."""

    heading_command: bool = False
    """Sample a target heading and derive the yaw rate from the heading error, rather
    than commanding a yaw rate directly."""
    heading_control_stiffness: float = 1.0
    """Proportional gain from heading error to commanded yaw rate."""

    rel_standing_envs: float = 0.0
    """Fraction of environments commanded to hold still."""
    rel_heading_envs: float = 1.0
    """Fraction of environments whose yaw rate is driven by the heading error."""
    rel_forward_envs: float = 0.0
    """Fraction of environments given an axis-aligned straight-line command: +world-x
    with probability 0.5, +world-y with 0.25, -world-y with 0.25. These are the commands
    a wheeled robot should answer by rolling rather than stepping, and uniform sampling
    almost never produces them."""

    init_velocity_prob: float = 0.0
    """Probability of starting an episode already at the commanded velocity, so the
    policy is not trained almost exclusively on the start-up transient."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity command."""

        lin_vel_x: tuple[float, float] = MISSING
        """Forward velocity range in the target-heading frame (world-x if
        ``heading_command`` is False)."""
        lin_vel_y: tuple[float, float] = MISSING
        """Lateral velocity range in the target-heading frame (world-y if
        ``heading_command`` is False)."""
        ang_vel_z: tuple[float, float] = MISSING
        """Yaw-rate range. Also clamps the heading-derived yaw rate."""
        heading: tuple[float, float] | None = None
        """Target heading range. Required when ``heading_command`` is True."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity command."""