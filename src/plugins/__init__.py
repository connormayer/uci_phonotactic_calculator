"""src/plugins/__init__.py — Plugin registry, discovery, and decorator"""

import pkgutil
import importlib
import pathlib

# Registry mapping plugin names to their model classes
PluginRegistry: dict[str, type["BaseModel"]] = {}


def register(name: str):
    """
    Decorator to register a plugin class under PluginRegistry[name].
    Raises KeyError if the name is already in use.
    """
    def _decorator(cls):
        if name in PluginRegistry:
            raise KeyError(f"Plugin '{name}' already registered")
        PluginRegistry[name] = cls
        return cls
    return _decorator


# Internal flag to ensure we only do discovery once
_DISCOVERED: bool = False


def _discover() -> None:
    """
    Lazily import all plugin modules in this package (except private ones
    and the 'strategies' subpackage) so that PluginRegistry is populated.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return

    pkg_path = pathlib.Path(__file__).parent
    for finder, mod_name, is_pkg in pkgutil.iter_modules([str(pkg_path)]):
        if mod_name.startswith("_") or mod_name == "strategies":
            continue
        importlib.import_module(f"{__name__}.{mod_name}")

    _DISCOVERED = True


def get_model(name: str) -> type["BaseModel"]:
    """
    Ensure all plugins are discovered, then return the model class
    registered under *name*. Raises KeyError listing available plugins.
    """
    _discover()
    try:
        return PluginRegistry[name]
    except KeyError:
        available = ", ".join(sorted(PluginRegistry)) or "-- none found --"
        raise KeyError(f"No plugin called '{name}'. Available: {available}")


__all__ = ["register", "PluginRegistry", "get_model"]

# End of src/plugins/__init__.py
