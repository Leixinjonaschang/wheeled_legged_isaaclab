"""Environment configuration groups, one sub-package per robot variant.

Sub-packages are imported by their fully qualified path from the environment
cfg modules under ``tasks/locomotion/robots/``. Nothing is star-imported here
on purpose: variants export identically named classes (``ActionsCfg``,
``RewardsCfg``, ...), so re-exporting them into this namespace would make which
one you get depend on import order.
"""
