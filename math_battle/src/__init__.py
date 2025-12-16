# Math Battle - A JAX-based RL environment for fantasy duels

from .game_state import GameState, Entity
from .env import MathBattleFuncEnv, EnvParams
from .effect_ops import Program, ProgramBuilder, ValueSpec
from .effect_interpreter import execute_program
from .heroes import create_fighter, create_fire_mage

__all__ = [
    "GameState",
    "Entity",
    "MathBattleFuncEnv",
    "EnvParams",
    "Program",
    "ProgramBuilder",
    "ValueSpec",
    "execute_program",
    "create_fighter",
    "create_fire_mage",
]
