"""Import the Isaac-Sim-free parts of the extension without importing the extension.

``bipedal_locomotion/__init__.py`` imports the task package, which imports Isaac Lab, so the modules
under test are loaded by file path with stub parent packages registered in ``sys.modules``. The
torch-only maths is loaded here; tests needing other modules call ``load`` (or ``load_wrapper``)
themselves. That keeps the suite runnable on a machine with no Isaac Sim, which is the point of
splitting the maths out in the first place.
"""

import importlib.util
import sys
import types
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1] / "exts" / "bipedal_locomotion" / "bipedal_locomotion"


def stub_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load(name: str, relpath: str):
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


def load_wrapper(module: str):
    """Load ``utils/wrappers/rsl_rl/<module>.py``, with the vendored ``rsl_rl`` importable."""
    sys.path.insert(0, str(PKG_DIR.parents[2] / "rsl_rl"))
    stub_package("bipedal_locomotion.utils.wrappers", PKG_DIR / "utils" / "wrappers")
    stub_package("bipedal_locomotion.utils.wrappers.rsl_rl", PKG_DIR / "utils" / "wrappers" / "rsl_rl")
    return load(f"bipedal_locomotion.utils.wrappers.rsl_rl.{module}", f"utils/wrappers/rsl_rl/{module}.py")


stub_package("bipedal_locomotion", PKG_DIR)
stub_package("bipedal_locomotion.utils", PKG_DIR / "utils")
stub_package("bipedal_locomotion.tasks", PKG_DIR / "tasks")
stub_package("bipedal_locomotion.tasks.locomotion", PKG_DIR / "tasks" / "locomotion")
stub_package("bipedal_locomotion.tasks.locomotion.mdp", PKG_DIR / "tasks" / "locomotion" / "mdp")

math_utils = load("bipedal_locomotion.utils.math", "utils/math.py")
kernels = load("bipedal_locomotion.tasks.locomotion.mdp.kernels", "tasks/locomotion/mdp/kernels.py")
domain_randomization = load(
    "bipedal_locomotion.tasks.locomotion.mdp.domain_randomization", "tasks/locomotion/mdp/domain_randomization.py"
)
depth = load("bipedal_locomotion.tasks.locomotion.mdp.depth", "tasks/locomotion/mdp/depth.py")
