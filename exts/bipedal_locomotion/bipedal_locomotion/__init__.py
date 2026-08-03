"""LimX bipedal-locomotion Isaac Lab extension.

Importing this package is what registers the gym tasks; see ``tasks/__init__.py``.
"""

# Register Gym environments.
from .tasks import *  # noqa: F401,F403

# Register UI extensions.
from .ui_extension_example import *  # noqa: F401,F403
