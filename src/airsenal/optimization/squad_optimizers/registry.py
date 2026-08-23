"""
The name-to-optimizer registry, kept separate so implementation modules can
import it without importing each other.
"""

from airsenal.core.registry import Registry
from airsenal.optimization.protocols import SquadOptimizer

SQUAD_OPTIMIZERS: Registry[SquadOptimizer] = Registry("squad optimizer")
