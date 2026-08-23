"""
The name-to-optimizer registry, kept separate so implementation modules can
import it without importing each other.
"""

from airsenal.core.registry import Registry
from airsenal.optimization.protocols import TransferOptimizer

TRANSFER_OPTIMIZERS: Registry[TransferOptimizer] = Registry("transfer optimizer")
