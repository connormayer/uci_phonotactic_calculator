"""Light-weight hook so external packages can inject extra CLI flags."""

import argparse
from typing import List, Protocol

from ..core.registries import register


class CLIExtension(Protocol):
    def flags(self) -> List[argparse.Action]: ...


def register_cli_ext(name):
    """Register a CLI extension with the given name."""
    return register("cli_ext", name)


__all__ = ["CLIExtension", "register_cli_ext"]
