"""Tunables for the WF-TRON1B velocity task, in one place.

Collected here rather than spread through the config classes so that this file can be
diffed against the mjlab task's ``env_cfgs.py`` module constants, which is the reference
these values were ported from.
"""

import math

ROBOT_ENTITY = "robot"
COMMAND_NAME = "twist"

BASE_BODY = "base_Link"

# Explicit, ordered joint lists. Every SceneEntityCfg and action term passes these with
# preserve_order=True: PhysX is free to order the articulation's joints differently from
# the MJCF declaration, and an implicit ordering would silently permute the action vector
# and the deployment metadata without changing any reward curve.
LEG_JOINT_NAMES = (
    "abad_L_Joint",
    "hip_L_Joint",
    "knee_L_Joint",
    "abad_R_Joint",
    "hip_R_Joint",
    "knee_R_Joint",
)
WHEEL_JOINT_NAMES = ("wheel_L_Joint", "wheel_R_Joint")
ALL_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES

WHEEL_BODY_NAMES = ("wheel_L_Link", "wheel_R_Link")
# Bodies that must never touch the ground: the bodies of mjlab's NON_WHEEL_COLLISION_GEOMS.
# Contact is sensed per body here, so the knees' `wheel_motor_[LR]_collision` geoms (in neither
# of mjlab's contact groups) also count as illegal contact.
NON_WHEEL_BODY_NAMES = (
    "base_Link",
    "abad_L_Link",
    "hip_L_Link",
    "knee_L_Link",
    "abad_R_Link",
    "hip_R_Link",
    "knee_R_Link",
)

# Posture and geometry.
BASE_HEIGHT_TARGET = 0.82
BASE_HEIGHT_DEADBAND = 0.04
# Base height is measured against this quantile of the base scan's hits (robust to pits).
BASE_HEIGHT_TERRAIN_QUANTILE = 0.75
POSE_TARGET_JOINT_POS = {
    "abad_L_Joint": 0.0,
    "hip_L_Joint": 0.2,
    "knee_L_Joint": 0.48,
    "abad_R_Joint": 0.0,
    "hip_R_Joint": -0.2,
    "knee_R_Joint": -0.48,
}
POSE_STD_STANDING = {r".*abad.*": 0.05, r".*hip.*": 0.08, r".*knee.*": 0.10}
POSE_STD_WALKING = {r".*abad.*": 0.15, r".*hip.*": 0.25, r".*knee.*": 0.35}
POSE_STD_RUNNING = {r".*abad.*": 0.20, r".*hip.*": 0.35, r".*knee.*": 0.50}
POSE_WALKING_THRESHOLD = 0.05
POSE_RUNNING_THRESHOLD = 0.5

WHEEL_DISTANCE_RANGE = (0.25, 0.55)
WHEEL_RADIUS = 0.127

# Height scans. The base scan drives the roughness gate and the critic's terrain map; the
# per-wheel scans drive swing clearance and the per-wheel roughness observation.
TERRAIN_SCAN_SIZE = (1.0, 1.0)
TERRAIN_SCAN_RESOLUTION = 0.1
TERRAIN_SCAN_GRID_SHAPE = (11, 11)
WHEEL_HEIGHT_SCAN_SIZE = (0.40, 0.40)
WHEEL_HEIGHT_SCAN_RESOLUTION = 0.10
WHEEL_HEIGHT_GRID_SHAPE = (5, 5)
WHEEL_HEIGHT_MAX_DISTANCE = 1.0
TERRAIN_SCAN_MAX_DISTANCE = 10.0

# Terrain force above which a body counts as touching the ground. Set on the contact sensor,
# which maintains current_air_time with it; contact-based terms read it back from the sensor.
CONTACT_FORCE_THRESHOLD = 1.0
# Force above which an illegal-ground or self contact counts (mjlab's default 10 N), shared by
# the penalties and the termination.
ILLEGAL_CONTACT_FORCE_THRESHOLD = 10.0

TERRAIN_SCAN_SENSOR = "terrain_scan"
WHEEL_HEIGHT_SENSORS = ("wheel_height_scan_L", "wheel_height_scan_R")
CONTACT_SENSOR = "contact_forces"

# Roughness gate. Starting the threshold at zero makes every surface count as rough, so
# the legged-motion rewards are live from the first iteration; raising it later narrows
# "rough" to terrain that genuinely needs stepping.
ROUGHNESS_GATE_MIN = 0.0
ROUGHNESS_GATE_MAX = 0.5
ROUGHNESS_GATE_THRESHOLD_INITIAL = 0.0
ROUGHNESS_GATE_THRESHOLD_FINAL = 0.65
ROUGHNESS_GATE_THRESHOLD_RAMP_STEPS = 5_000 * 24

# Fall-over tolerance, relaxed over training so early episodes end while recovery is
# still plausible and later ones allow learning to recover from steeper attitudes.
FELL_OVER_LIMIT_ANGLE_INITIAL = math.radians(65.0)
FELL_OVER_LIMIT_ANGLE_FINAL = math.radians(85.0)
FELL_OVER_LIMIT_ANGLE_RAMP_STEPS = 5_000 * 24

# Global policy step after which world-command tracking failures terminate episodes. The
# RSL-RL runner collects 24 policy steps per learning iteration.
WORLD_COMMAND_TRACKING_ACTIVATION_STEPS = 5_000 * 24

# Action scales. Legs are position-controlled around the default pose; wheels are
# velocity-controlled in absolute terms, hence the much larger scale and no offset.
LEG_ACTION_SCALE = 0.5
WHEEL_ACTION_SCALE = 10.0
ACTION_DELAY_RANGE_S = (0.0, 0.02)
ACTION_DELAY_RESAMPLING_TIME_S = 5.0
ACTION_DELAY_GROUP = "base_actions"

# Domain randomisation, sampled once per environment at startup (mjlab ranges).
WHEEL_FRICTION_RANGE = (0.45, 1.1)  # one draw shared by both wheels ...
WHEEL_FRICTION_DIFFERENCE_RANGE = (-0.04, 0.04)  # ... plus an independent per-wheel offset
MASS_INERTIA_ALPHA_RANGE = (0.5 * math.log(0.8), 0.5 * math.log(1.2))  # mass and inertia scale by exp(2 alpha)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
STIFFNESS_SCALE_RANGE = (0.8, 1.2)
DAMPING_SCALE_RANGE = (0.8, 1.2)
BASE_COM_RANGE = (-0.03, 0.03)  # per axis
LINK_COM_RANGE = (-0.015, 0.015)  # per axis, every non-base body
# MJCF (depth-first) order, which mjlab's privileged DR observation is laid out in.
MJCF_JOINT_NAMES = (
    "abad_L_Joint",
    "hip_L_Joint",
    "knee_L_Joint",
    "wheel_L_Joint",
    "abad_R_Joint",
    "hip_R_Joint",
    "knee_R_Joint",
    "wheel_R_Joint",
)
MJCF_BODY_NAMES = (
    "base_Link",
    "abad_L_Link",
    "hip_L_Link",
    "knee_L_Link",
    "wheel_L_Link",
    "abad_R_Link",
    "hip_R_Link",
    "knee_R_Link",
    "wheel_R_Link",
)

# Observation noise, matching the mjlab task.
NOISE_BASE_ANG_VEL = 0.2
NOISE_PROJECTED_GRAVITY = 0.05
NOISE_JOINT_POS = 0.01
NOISE_JOINT_VEL = 1.5
NOISE_WHEEL_VEL = 0.2

# Observation scales.
SCALE_JOINT_VEL = 0.05
SCALE_WHEEL_VEL = 0.05
SCALE_HEIGHT_SCAN = 0.1

HISTORY_LENGTH = 5

DECIMATION = 4
SIM_DT = 0.005
EPISODE_LENGTH_S = 20.0
NUM_ENVS = 4096
