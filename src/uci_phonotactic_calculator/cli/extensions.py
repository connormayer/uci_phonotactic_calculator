"""Light-weight hook so external packages can inject extra CLI flags."""

import argparse
from typing import Protocol

from ..core.registries import register


class CLIExtension(Protocol):
    def flags(self) -> list[argparse.Action]: ...


def register_cli_ext(name: str) -> object:
    """Register a CLI extension with the given name."""
    return register("cli_ext", name)


__all__ = ["CLIExtension", "register_cli_ext"]
