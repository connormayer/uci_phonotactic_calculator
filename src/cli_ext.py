"""Light-weight hook so external packages can inject extra CLI flags."""

from typing import Protocol, List
import argparse
from .registries import register

class CLIExtension(Protocol):
    def flags(self) -> List[argparse.Action]: ...

register_cli_ext = lambda name: register('cli_ext', name)

__all__ = ["CLIExtension", "register_cli_ext"]
