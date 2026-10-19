"""Import the Isaac-Sim-free parts of the extension without importing the extension.

``bipedal_locomotion/__init__.py`` imports the task package, which imports Isaac Lab.
The modules exercised by this test suite (``mdp/kernels.py``, ``mdp/domain_randomization.py``,
``mdp/depth.py`` and ``utils/math.py``) deliberately depend on nothing but ``torch``, so they are loaded here
by file path with stub parent packages registered in ``sys.modules``. That keeps the suite
runnable on a machine with no Isaac Sim, which is the point of splitting the maths out in
the first place.
"""

import importlib.util
import sys
import types
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "exts" / "bipedal_locomotion" / "bipedal_locomotion"


def _stub_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load(name: str, relpath: str):
    if name in sys.modules:
        return sys.modules[name]
    file_path = PKG_DIR / relpath
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_stub_package("bipedal_locomotion", PKG_DIR)
_stub_package("bipedal_locomotion.utils", PKG_DIR / "utils")
_stub_package("bipedal_locomotion.tasks", PKG_DIR / "tasks")
_stub_package("bipedal_locomotion.tasks.locomotion", PKG_DIR / "tasks" / "locomotion")
_stub_package("bipedal_locomotion.tasks.locomotion.mdp", PKG_DIR / "tasks" / "locomotion" / "mdp")

math_utils = _load("bipedal_locomotion.utils.math", "utils/math.py")
kernels = _load("bipedal_locomotion.tasks.locomotion.mdp.kernels", "tasks/locomotion/mdp/kernels.py")
domain_randomization = _load(
    "bipedal_locomotion.tasks.locomotion.mdp.domain_randomization", "tasks/locomotion/mdp/domain_randomization.py"
)
depth = _load("bipedal_locomotion.tasks.locomotion.mdp.depth", "tasks/locomotion/mdp/depth.py")
