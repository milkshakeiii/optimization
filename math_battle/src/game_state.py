"""Core game state and type definitions for Math Battle.

This module defines the data structures for the game state using JAX-compatible
types with fixed array sizes for JIT compilation.
"""

from typing import NamedTuple
import jax.numpy as jnp
from jax import Array

# Fixed sizes for JAX compilation
MAX_ATTRIBUTES = 16
MAX_ABILITIES = 8
MAX_EFFECTS = 16
MAX_SCRIPT_LEN = 64
MAX_TAGS_PER_ABILITY = 4
MAX_QUEUE = 32

# Attribute indices (canonical mapping)
ATTR_HEALTH = 0
ATTR_MAX_HEALTH = 1
ATTR_MANA = 2
ATTR_MAX_MANA = 3
ATTR_STRENGTH = 4
ATTR_DEFENSE = 5
ATTR_BURN = 6
ATTR_STUN = 7
ATTR_MANA_REGEN = 8
ATTR_RAGE_BONUS = 9

ATTRIBUTE_NAMES = [
    "health", "max_health", "mana", "max_mana", "strength",
    "defense", "burn", "stun", "mana_regen", "rage_bonus",
    "unused_10", "unused_11", "unused_12", "unused_13", "unused_14", "unused_15"
]

# Trigger types
TRIGGER_NONE = 0
TRIGGER_ON_TURN_START = 1
TRIGGER_ON_ACTION_PHASE_START = 2
TRIGGER_ON_TURN_END = 3
TRIGGER_ON_ABILITY_USED = 4
TRIGGER_ON_ATTRIBUTE_CHANGE = 5
TRIGGER_ON_GAME_START = 6

# DSL Opcodes
OP_NOOP = 0
OP_SELF = 1
OP_OPPONENT = 2
OP_CONTEXT = 3
OP_GET = 4
OP_SET = 5
OP_MODIFY = 6
OP_ADD = 7
OP_SUB = 8
OP_MUL = 9
OP_DIV = 10
OP_ABS = 11
OP_MIN = 12
OP_MAX = 13
OP_ROLL = 14
OP_IF = 15
OP_SEQ = 16
OP_EQ = 17
OP_GT = 18
OP_LT = 19
OP_AND = 20
OP_OR = 21
OP_NOT = 22
OP_WIN = 23
OP_LOSE = 24
OP_PASS = 25
OP_PUSH = 26  # Push literal value onto stack
OP_END = 27   # End of script marker

# Context keys
CTX_ABILITY_ID = 0
CTX_ABILITY_COST = 1
CTX_ATTR_DELTA = 2
CTX_ATTR_NEW = 3
CTX_ATTR_OLD = 4
CTX_ATTR_NAME = 5

# Tags (for abilities and triggers)
TAG_NONE = 0
TAG_ATTACK = 1
TAG_SPELL = 2
TAG_FIRE = 3
TAG_PHYSICAL = 4
TAG_HEAL = 5
TAG_BUFF = 6
TAG_DEBUFF = 7


class Entity(NamedTuple):
    """Represents a player or opponent entity.

    Attributes:
        attributes: Fixed-size array of attribute values (float32)
        abilities_script: Scripts for each ability (MAX_ABILITIES x MAX_SCRIPT_LEN)
        abilities_valid: Boolean mask for valid abilities
        abilities_tags: Tags for each ability (MAX_ABILITIES x MAX_TAGS_PER_ABILITY)
        effects_trigger: Trigger type for each passive effect
        effects_script: Scripts for each passive effect
        effects_trigger_param: Parameter for trigger (e.g., attribute index)
    """
    attributes: Array  # (MAX_ATTRIBUTES,)
    abilities_script: Array  # (MAX_ABILITIES, MAX_SCRIPT_LEN) - int32 opcodes
    abilities_valid: Array  # (MAX_ABILITIES,) - bool
    abilities_tags: Array  # (MAX_ABILITIES, MAX_TAGS_PER_ABILITY) - int32
    effects_trigger: Array  # (MAX_EFFECTS,) - int32
    effects_script: Array  # (MAX_EFFECTS, MAX_SCRIPT_LEN) - int32 opcodes
    effects_trigger_param: Array  # (MAX_EFFECTS,) - int32 (e.g., attr index)


class GameState(NamedTuple):
    """Complete game state.

    Attributes:
        player: Player entity (index 0)
        opponent: Opponent entity (index 1)
        active_player: Index of active player (0 or 1)
        turn_count: Number of turns elapsed
        done: Whether the game has ended
        winner: Winner index (-1 if ongoing, 0 or 1 if finished)
        passed: Whether the action phase was skipped (via PASS)
        context: Execution context for triggers
    """
    player: Entity
    opponent: Entity
    active_player: Array  # scalar int32
    turn_count: Array  # scalar int32
    done: Array  # scalar bool
    winner: Array  # scalar int32 (-1 = ongoing)
    passed: Array  # scalar bool
    context: Array  # (8,) float32 - execution context
    queue: Array  # (MAX_QUEUE, 2) int32 - pending triggers
    queue_count: Array  # scalar int32


def create_empty_entity() -> Entity:
    """Create an entity with all zeros/defaults."""
    return Entity(
        attributes=jnp.zeros(MAX_ATTRIBUTES, dtype=jnp.float32),
        abilities_script=jnp.zeros((MAX_ABILITIES, MAX_SCRIPT_LEN), dtype=jnp.int32),
        abilities_valid=jnp.zeros(MAX_ABILITIES, dtype=jnp.bool_),
        abilities_tags=jnp.zeros((MAX_ABILITIES, MAX_TAGS_PER_ABILITY), dtype=jnp.int32),
        effects_trigger=jnp.zeros(MAX_EFFECTS, dtype=jnp.int32),
        effects_script=jnp.zeros((MAX_EFFECTS, MAX_SCRIPT_LEN), dtype=jnp.int32),
        effects_trigger_param=jnp.zeros(MAX_EFFECTS, dtype=jnp.int32),
    )


def create_initial_game_state(player: Entity, opponent: Entity) -> GameState:
    """Create a new game state with the given entities."""
    return GameState(
        player=player,
        opponent=opponent,
        active_player=jnp.array(0, dtype=jnp.int32),
        turn_count=jnp.array(0, dtype=jnp.int32),
        done=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(-1, dtype=jnp.int32),
        passed=jnp.array(False, dtype=jnp.bool_),
        context=jnp.zeros(8, dtype=jnp.float32),
        queue=jnp.zeros((MAX_QUEUE, 2), dtype=jnp.int32),
        queue_count=jnp.array(0, dtype=jnp.int32),
    )


def get_entity(state: GameState, idx: int) -> Entity:
    """Get entity by index (0 = player, 1 = opponent)."""
    # Note: For JAX compatibility, this uses lax.cond in the actual implementation
    return state.player if idx == 0 else state.opponent


def set_entity(state: GameState, idx: int, entity: Entity) -> GameState:
    """Set entity by index."""
    if idx == 0:
        return state._replace(player=entity)
    else:
        return state._replace(opponent=entity)
