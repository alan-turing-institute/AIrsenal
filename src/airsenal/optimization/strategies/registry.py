"""
The name-to-strategy registry, kept separate so strategy modules can import it
without importing each other.
"""

from dataclasses import dataclass

from airsenal.core.registry import Registry
from airsenal.optimization.protocols import TransferStrategy


@dataclass(frozen=True)
class NoOptions:
    """For strategies that enumerate every possibility and so have nothing to tune."""


TRANSFER_STRATEGIES: Registry[TransferStrategy] = Registry("transfer strategy")
