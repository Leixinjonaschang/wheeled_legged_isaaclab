"""``get_applied_actions`` must report the pre-reset action for environments that just finished.

Mirrors mjlab ``tests/test_action_delay.py::test_done_env_uses_terminal_applied_action_snapshot``.
"""

import sys
from types import SimpleNamespace

import torch

from _loader import PKG_DIR, _load, _stub_package

sys.path.insert(0, str(PKG_DIR.parents[2] / "rsl_rl"))
_stub_package("bipedal_locomotion.utils.wrappers", PKG_DIR / "utils" / "wrappers")
_stub_package("bipedal_locomotion.utils.wrappers.rsl_rl", PKG_DIR / "utils" / "wrappers" / "rsl_rl")
wrapper = _load("bipedal_locomotion.utils.wrappers.rsl_rl.vecenv_wrapper", "utils/wrappers/rsl_rl/vecenv_wrapper.py")


def _action_manager(**terms):
    return SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__)


def test_done_env_uses_terminal_applied_action_snapshot():
    delayed = SimpleNamespace(
        applied_action=torch.tensor([[0.0], [2.0]]),
        terminal_applied_action=torch.tensor([[1.0], [9.0]]),
        raw_actions=torch.zeros(2, 1),
    )
    plain = SimpleNamespace(raw_actions=torch.tensor([[5.0], [6.0]]))
    manager = _action_manager(delayed=delayed, plain=plain)

    applied = wrapper.get_applied_actions(manager, dones=torch.tensor([1, 0]))
    assert torch.equal(applied, torch.tensor([[1.0, 5.0], [2.0, 6.0]]))
    assert torch.equal(wrapper.get_applied_actions(manager), torch.tensor([[0.0, 5.0], [2.0, 6.0]]))
