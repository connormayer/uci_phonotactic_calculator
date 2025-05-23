# ------------------------------------------------------------------
# Built-in registry modules - these must be imported first for side effects
# These imports ensure all registry entries are available before Config.default()
# or validate_choice() are called from any entry point.
# ------------------------------------------------------------------
# Standard imports
import importlib
import pkgutil

# Re-export CLI entry points for package scripts
from .cli.legacy import run  # noqa: F401
from .cli.main import main  # noqa: F401
from .core.config import Config  # noqa: F401
from .core.corpus import Corpus  # noqa: F401

# Import model modules with leading underscores to indicate side-effects only
from .models import aggregators as _aggregators_builtin  # noqa: F401
from .models import boundaries as _boundaries_builtin  # noqa: F401
from .models import boundary_modes as _boundary_modes_builtin  # noqa: F401
from .models import neighbourhood as _neighbourhood_builtin  # noqa: F401
from .models import smoothing as _smoothing_builtin  # noqa: F401
from .models import weighting as _weighting  # noqa: F401

# Define public API
__all__ = ["run", "main", "Config", "Corpus"]

# Define the package version
__version__ = "0.2.2"


def import_all_submodules(package_name):
    """Import all submodules of a package for side effects."""
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
