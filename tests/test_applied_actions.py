"""``get_applied_actions`` must report the pre-reset action for environments that just finished.

Mirrors mjlab ``tests/test_action_delay.py::test_done_env_uses_terminal_applied_action_snapshot``.
"""

from types import SimpleNamespace

import torch

from _loader import load_wrapper

wrapper = load_wrapper("vecenv_wrapper")


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
