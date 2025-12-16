"""Core game state and type definitions for Math Battle.

This module defines the data structures for the game state using JAX-compatible
types with fixed array sizes for JIT compilation.

Uses the Effect Ops system from effect_ops.py for ability and effect programs.
"""

from typing import NamedTuple
import jax.numpy as jnp
from jax import Array

from .effect_ops import (
    MAX_OPS, MAX_VALUE_DEPTH,
    CTX_SIZE,
)

# Fixed sizes for JAX compilation
MAX_ATTRIBUTES = 16
MAX_ABILITIES = 8
MAX_EFFECTS = 16
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

# Tags (for abilities and triggers)
TAG_NONE = 0
TAG_ATTACK = 1
TAG_SPELL = 2
TAG_FIRE = 3
TAG_PHYSICAL = 4
TAG_HEAL = 5
TAG_BUFF = 6
TAG_DEBUFF = 7

# Context keys (re-exported from effect_ops for convenience)
CTX_ABILITY_ID = 0
CTX_ABILITY_COST = 1
CTX_ATTR_DELTA = 2
CTX_ATTR_NEW = 3
CTX_ATTR_OLD = 4
CTX_ATTR_NAME = 5


class Entity(NamedTuple):
    """Represents a player or opponent entity.

    Attributes:
        attributes: Fixed-size array of attribute values (float32)

        Abilities (Effect Ops programs):
        abilities_op_types: (MAX_ABILITIES, MAX_OPS) operation types
        abilities_targets: (MAX_ABILITIES, MAX_OPS) target specs
        abilities_attr_ids: (MAX_ABILITIES, MAX_OPS) attribute IDs
        abilities_value_types: (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH) value types
        abilities_value_param1: (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH) value params
        abilities_value_param2: (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH) value params
        abilities_value_num_nodes: (MAX_ABILITIES, MAX_OPS) node counts
        abilities_value2_types: (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH) 2nd value types
        abilities_value2_param1: (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH) 2nd value params
        abilities_value2_param2: (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH) 2nd value params
        abilities_value2_num_nodes: (MAX_ABILITIES, MAX_OPS) 2nd value node counts
        abilities_if_then_count: (MAX_ABILITIES, MAX_OPS) IF then counts
        abilities_if_else_count: (MAX_ABILITIES, MAX_OPS) IF else counts
        abilities_num_ops: (MAX_ABILITIES,) number of ops per ability
        abilities_valid: (MAX_ABILITIES,) boolean mask for valid abilities
        abilities_tags: (MAX_ABILITIES, MAX_TAGS_PER_ABILITY) tags

        Effects (passive abilities):
        effects_trigger: (MAX_EFFECTS,) trigger type for each effect
        effects_trigger_param: (MAX_EFFECTS,) trigger parameter (e.g., attr ID)
        effects_op_types: (MAX_EFFECTS, MAX_OPS) operation types
        effects_targets: (MAX_EFFECTS, MAX_OPS) target specs
        effects_attr_ids: (MAX_EFFECTS, MAX_OPS) attribute IDs
        effects_value_types: (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH) value types
        effects_value_param1: (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH) value params
        effects_value_param2: (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH) value params
        effects_value_num_nodes: (MAX_EFFECTS, MAX_OPS) node counts
        effects_value2_types: (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH) 2nd value types
        effects_value2_param1: (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH) 2nd value params
        effects_value2_param2: (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH) 2nd value params
        effects_value2_num_nodes: (MAX_EFFECTS, MAX_OPS) 2nd value node counts
        effects_if_then_count: (MAX_EFFECTS, MAX_OPS) IF then counts
        effects_if_else_count: (MAX_EFFECTS, MAX_OPS) IF else counts
        effects_num_ops: (MAX_EFFECTS,) number of ops per effect
    """
    attributes: Array  # (MAX_ATTRIBUTES,)

    # Abilities (programs stored as parallel arrays)
    abilities_op_types: Array        # (MAX_ABILITIES, MAX_OPS)
    abilities_targets: Array         # (MAX_ABILITIES, MAX_OPS)
    abilities_attr_ids: Array        # (MAX_ABILITIES, MAX_OPS)
    abilities_value_types: Array     # (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH)
    abilities_value_param1: Array    # (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH)
    abilities_value_param2: Array    # (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH)
    abilities_value_num_nodes: Array # (MAX_ABILITIES, MAX_OPS)
    abilities_value2_types: Array    # (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH)
    abilities_value2_param1: Array   # (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH)
    abilities_value2_param2: Array   # (MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH)
    abilities_value2_num_nodes: Array # (MAX_ABILITIES, MAX_OPS)
    abilities_if_then_count: Array   # (MAX_ABILITIES, MAX_OPS)
    abilities_if_else_count: Array   # (MAX_ABILITIES, MAX_OPS)
    abilities_num_ops: Array         # (MAX_ABILITIES,)
    abilities_valid: Array           # (MAX_ABILITIES,)
    abilities_tags: Array            # (MAX_ABILITIES, MAX_TAGS_PER_ABILITY)

    # Effects (passive abilities with triggers)
    effects_trigger: Array           # (MAX_EFFECTS,)
    effects_trigger_param: Array     # (MAX_EFFECTS,)
    effects_op_types: Array          # (MAX_EFFECTS, MAX_OPS)
    effects_targets: Array           # (MAX_EFFECTS, MAX_OPS)
    effects_attr_ids: Array          # (MAX_EFFECTS, MAX_OPS)
    effects_value_types: Array       # (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH)
    effects_value_param1: Array      # (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH)
    effects_value_param2: Array      # (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH)
    effects_value_num_nodes: Array   # (MAX_EFFECTS, MAX_OPS)
    effects_value2_types: Array      # (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH)
    effects_value2_param1: Array     # (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH)
    effects_value2_param2: Array     # (MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH)
    effects_value2_num_nodes: Array  # (MAX_EFFECTS, MAX_OPS)
    effects_if_then_count: Array     # (MAX_EFFECTS, MAX_OPS)
    effects_if_else_count: Array     # (MAX_EFFECTS, MAX_OPS)
    effects_num_ops: Array           # (MAX_EFFECTS,)


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
        queue: Pending attribute change triggers
        queue_count: Number of items in queue
    """
    player: Entity
    opponent: Entity
    active_player: Array  # scalar int32
    turn_count: Array     # scalar int32
    done: Array           # scalar bool
    winner: Array         # scalar int32 (-1 = ongoing)
    passed: Array         # scalar bool
    context: Array        # (CTX_SIZE,) float32
    queue: Array          # (MAX_QUEUE, 2) int32 - [target_idx, attr_idx]
    queue_count: Array    # scalar int32


def create_empty_entity() -> Entity:
    """Create an entity with all zeros/defaults."""
    return Entity(
        attributes=jnp.zeros(MAX_ATTRIBUTES, dtype=jnp.float32),

        # Abilities
        abilities_op_types=jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32),
        abilities_targets=jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32),
        abilities_attr_ids=jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32),
        abilities_value_types=jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.int32),
        abilities_value_param1=jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        abilities_value_param2=jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        abilities_value_num_nodes=jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32),
        abilities_value2_types=jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.int32),
        abilities_value2_param1=jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        abilities_value2_param2=jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        abilities_value2_num_nodes=jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32),
        abilities_if_then_count=jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32),
        abilities_if_else_count=jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32),
        abilities_num_ops=jnp.zeros(MAX_ABILITIES, dtype=jnp.int32),
        abilities_valid=jnp.zeros(MAX_ABILITIES, dtype=jnp.bool_),
        abilities_tags=jnp.zeros((MAX_ABILITIES, MAX_TAGS_PER_ABILITY), dtype=jnp.int32),

        # Effects
        effects_trigger=jnp.zeros(MAX_EFFECTS, dtype=jnp.int32),
        effects_trigger_param=jnp.zeros(MAX_EFFECTS, dtype=jnp.int32),
        effects_op_types=jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32),
        effects_targets=jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32),
        effects_attr_ids=jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32),
        effects_value_types=jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.int32),
        effects_value_param1=jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        effects_value_param2=jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        effects_value_num_nodes=jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32),
        effects_value2_types=jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.int32),
        effects_value2_param1=jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        effects_value2_param2=jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        effects_value2_num_nodes=jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32),
        effects_if_then_count=jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32),
        effects_if_else_count=jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32),
        effects_num_ops=jnp.zeros(MAX_EFFECTS, dtype=jnp.int32),
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
        context=jnp.zeros(CTX_SIZE, dtype=jnp.float32),
        queue=jnp.zeros((MAX_QUEUE, 2), dtype=jnp.int32),
        queue_count=jnp.array(0, dtype=jnp.int32),
    )


def get_entity(state: GameState, idx: int) -> Entity:
    """Get entity by index (0 = player, 1 = opponent).

    Note: For JAX compatibility in JIT context, use get_entity_by_idx
    from effect_interpreter instead.
    """
    return state.player if idx == 0 else state.opponent


def set_entity(state: GameState, idx: int, entity: Entity) -> GameState:
    """Set entity by index.

    Note: For JAX compatibility in JIT context, use set_entity_by_idx
    from effect_interpreter instead.
    """
    if idx == 0:
        return state._replace(player=entity)
    else:
        return state._replace(opponent=entity)
