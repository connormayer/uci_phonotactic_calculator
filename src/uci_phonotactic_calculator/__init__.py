# ------------------------------------------------------------------
# Built-in registry modules - these must be imported first for side effects
# These imports ensure all registry entries are available before Config.default()
# or validate_choice() are called from any entry point.
# ------------------------------------------------------------------
# Each import has a leading underscore to indicate it's used for side effects only
# Standard imports
import importlib
import pkgutil

from . import (
    aggregators_builtin as _aggregators_builtin,  # noqa: F401 - aggregate_mode registry
)
from . import (
    boundaries_builtin as _boundaries_builtin,  # noqa: F401 - boundary_scheme registry
)
from . import (
    boundary_modes_builtin as _boundary_modes_builtin,  # noqa: F401 - boundary_mode registry
)
from . import (
    neighbourhood_builtin as _neighbourhood_builtin,  # noqa: F401 - neighbourhood_mode registry
)
from . import (
    smoothing_builtin as _smoothing_builtin,  # noqa: F401 - smoothing_scheme registry
)
from . import weighting as _weighting  # noqa: F401 - weight_mode registry


def import_all_submodules(package_name):
    print(f"[DEBUG] Dynamic plugin import running for {package_name}...")
    package = importlib.import_module(package_name)
    for _, modname, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if not ispkg:
            importlib.import_module(modname)


# Automatically import all plugins and strategies so their registrations run
import_all_submodules("uci_phonotactic_calculator.plugins")
import_all_submodules("uci_phonotactic_calculator.plugins.strategies")
