"""Game Engine for Math Battle.

Implements the turn structure and effect triggering system using Effect Ops.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import lax, Array

from .game_state import (
    GameState, Entity, MAX_EFFECTS, MAX_ABILITIES, MAX_ATTRIBUTES, MAX_QUEUE,
    TRIGGER_ON_TURN_START, TRIGGER_ON_ACTION_PHASE_START,
    TRIGGER_ON_TURN_END, TRIGGER_ON_ABILITY_USED, TRIGGER_ON_ATTRIBUTE_CHANGE,
    TRIGGER_ON_GAME_START, ATTR_HEALTH,
    CTX_ABILITY_ID, CTX_ATTR_DELTA, CTX_ATTR_NEW, CTX_ATTR_OLD, CTX_ATTR_NAME,
)
from .effect_ops import Program, MAX_OPS, MAX_VALUE_DEPTH
from .effect_interpreter import (
    execute_program, get_entity_by_idx, set_entity_by_idx
)


def extract_ability_program(entity: Entity, ability_idx: int) -> Program:
    """Extract a Program from an Entity's ability arrays.

    Args:
        entity: The entity containing the ability
        ability_idx: Index of the ability (0 to MAX_ABILITIES-1)

    Returns:
        A Program object that can be passed to execute_program
    """
    return Program(
        op_types=entity.abilities_op_types[ability_idx],
        targets=entity.abilities_targets[ability_idx],
        attr_ids=entity.abilities_attr_ids[ability_idx],
        value_types=entity.abilities_value_types[ability_idx],
        value_param1=entity.abilities_value_param1[ability_idx],
        value_param2=entity.abilities_value_param2[ability_idx],
        value_num_nodes=entity.abilities_value_num_nodes[ability_idx],
        value2_types=entity.abilities_value2_types[ability_idx],
        value2_param1=entity.abilities_value2_param1[ability_idx],
        value2_param2=entity.abilities_value2_param2[ability_idx],
        value2_num_nodes=entity.abilities_value2_num_nodes[ability_idx],
        if_then_count=entity.abilities_if_then_count[ability_idx],
        if_else_count=entity.abilities_if_else_count[ability_idx],
        num_ops=entity.abilities_num_ops[ability_idx],
    )


def extract_effect_program(entity: Entity, effect_idx: int) -> Program:
    """Extract a Program from an Entity's effect arrays.

    Args:
        entity: The entity containing the effect
        effect_idx: Index of the effect (0 to MAX_EFFECTS-1)

    Returns:
        A Program object that can be passed to execute_program
    """
    return Program(
        op_types=entity.effects_op_types[effect_idx],
        targets=entity.effects_targets[effect_idx],
        attr_ids=entity.effects_attr_ids[effect_idx],
        value_types=entity.effects_value_types[effect_idx],
        value_param1=entity.effects_value_param1[effect_idx],
        value_param2=entity.effects_value_param2[effect_idx],
        value_num_nodes=entity.effects_value_num_nodes[effect_idx],
        value2_types=entity.effects_value2_types[effect_idx],
        value2_param1=entity.effects_value2_param1[effect_idx],
        value2_param2=entity.effects_value2_param2[effect_idx],
        value2_num_nodes=entity.effects_value2_num_nodes[effect_idx],
        if_then_count=entity.effects_if_then_count[effect_idx],
        if_else_count=entity.effects_if_else_count[effect_idx],
        num_ops=entity.effects_num_ops[effect_idx],
    )


def trigger_effects(
    state: GameState,
    trigger_type: int,
    rng: Array,
    executor_idx: int = -1,
    trigger_param: int = -1,
) -> Tuple[GameState, Array]:
    """Trigger all effects matching the given trigger type.

    Args:
        state: Current game state
        trigger_type: Type of trigger to fire
        rng: Random key
        executor_idx: Which entity's effects to trigger (-1 for both)
        trigger_param: Parameter to match (e.g., attribute index for ON_ATTRIBUTE_CHANGE)

    Returns:
        Updated state and new rng
    """

    def process_entity(idx, carry):
        s, r = carry

        # Determine if we should process this entity
        should_process = jnp.logical_or(executor_idx < 0, executor_idx == idx)

        def run_entity_effects(inner_args):
            s_in, r_in = inner_args
            entity = get_entity_by_idx(s_in, idx)

            def effect_loop_body(i, loop_carry):
                cs, cr = loop_carry

                def execute_effect(args):
                    curr_s, curr_r = args
                    curr_entity = get_entity_by_idx(curr_s, idx)
                    trigger = curr_entity.effects_trigger[i]
                    param = curr_entity.effects_trigger_param[i]

                    # Check if trigger matches
                    trigger_matches = (trigger == trigger_type)
                    # param matches if trigger_param < 0 or param == trigger_param or param < 0
                    param_matches = jnp.logical_or(
                        trigger_param < 0,
                        jnp.logical_or(param == trigger_param, param < 0)
                    )

                    should_run = jnp.logical_and(trigger_matches, param_matches)

                    def run_program(s_r_tuple):
                        st, rn = s_r_tuple
                        # Extract program and execute
                        program = extract_effect_program(
                            get_entity_by_idx(st, idx), i
                        )
                        st, passed, winner, rn = execute_program(
                            program, st, idx, rn
                        )

                        # Update winner/done
                        has_winner = winner >= 0
                        st = st._replace(
                            done=jnp.logical_or(st.done, has_winner),
                            winner=jnp.where(has_winner, winner, st.winner),
                            passed=jnp.logical_or(st.passed, passed)
                        )
                        return st, rn

                    return lax.cond(should_run, run_program, lambda x: x, (curr_s, curr_r))

                return lax.cond(cs.done, lambda x: x, execute_effect, (cs, cr))

            return lax.fori_loop(0, MAX_EFFECTS, effect_loop_body, (s_in, r_in))

        return lax.cond(should_process, run_entity_effects, lambda x: x, (s, r))

    # Process both entities (0 and 1)
    state, rng = lax.fori_loop(0, 2, process_entity, (state, rng))

    return state, rng


def process_trigger_queue(state: GameState, rng: Array) -> Tuple[GameState, Array]:
    """Process pending triggers in the queue."""

    def cond_fun(args):
        s, r, processed_count = args
        has_items = s.queue_count > 0
        return jnp.logical_and(has_items, processed_count < MAX_QUEUE)

    def body_fun(args):
        s, r, processed_count = args

        # Take one item from end (LIFO)
        idx = s.queue_count - 1
        safe_idx = jnp.clip(idx, 0, MAX_QUEUE - 1)
        target_idx = s.queue[safe_idx, 0]
        attr_idx = s.queue[safe_idx, 1]

        # Decrement count
        s = s._replace(queue_count=idx)

        # Trigger attribute change effects
        s, r = trigger_effects(s, TRIGGER_ON_ATTRIBUTE_CHANGE, r, target_idx, attr_idx)

        return s, r, processed_count + 1

    state, rng, _ = lax.while_loop(cond_fun, body_fun, (state, rng, 0))

    # Ensure queue is clear
    state = state._replace(queue_count=jnp.array(0, dtype=jnp.int32))

    return state, rng


def execute_ability(
    state: GameState,
    ability_idx: int,
    rng: Array,
) -> Tuple[GameState, Array]:
    """Execute an ability for the active player."""

    active_idx = state.active_player
    entity = get_entity_by_idx(state, active_idx)

    # Check if ability is valid
    valid = entity.abilities_valid[ability_idx]

    def run_ability(args):
        s, r = args

        # Set context (use astype for JIT compatibility)
        new_context = s.context.at[CTX_ABILITY_ID].set(jnp.asarray(ability_idx).astype(jnp.float32))
        s = s._replace(context=new_context)

        # Trigger ON_ABILITY_USED
        s, r = trigger_effects(s, TRIGGER_ON_ABILITY_USED, r, executor_idx=-1)

        # Process queue (from ability used effects)
        s, r = process_trigger_queue(s, r)

        def continue_execution(inner_args):
            s_in, r_in = inner_args
            # Get updated entity and extract program
            current_entity = get_entity_by_idx(s_in, active_idx)
            program = extract_ability_program(current_entity, ability_idx)

            s_in, passed, winner, r_in = execute_program(
                program, s_in, active_idx, r_in
            )

            # Update winner
            has_winner = winner >= 0
            s_in = s_in._replace(
                done=jnp.logical_or(s_in.done, has_winner),
                winner=jnp.where(has_winner, winner, s_in.winner)
            )

            # Process queue (from ability script)
            s_in, r_in = process_trigger_queue(s_in, r_in)

            return s_in, r_in

        return lax.cond(s.done, lambda x: x, continue_execution, (s, r))

    return lax.cond(valid, run_ability, lambda x: x, (state, rng))


def run_turn_start_phase(state: GameState, rng: Array) -> Tuple[GameState, Array]:
    """Run the turn start phase."""
    active_idx = state.active_player
    state, rng = trigger_effects(state, TRIGGER_ON_TURN_START, rng, executor_idx=active_idx)
    # Process any attribute change events
    state, rng = process_trigger_queue(state, rng)
    return state, rng


def run_turn_end_phase(state: GameState, rng: Array) -> Tuple[GameState, Array]:
    """Run the turn end phase."""
    active_idx = state.active_player
    state, rng = trigger_effects(state, TRIGGER_ON_TURN_END, rng, executor_idx=active_idx)
    state, rng = process_trigger_queue(state, rng)
    return state, rng


def check_action_phase_start(
    state: GameState, rng: Array
) -> Tuple[GameState, bool, Array]:
    """Check if action phase should be skipped (via PASS effect)."""
    active_idx = state.active_player
    state, rng = trigger_effects(
        state, TRIGGER_ON_ACTION_PHASE_START, rng,
        executor_idx=active_idx
    )
    state, rng = process_trigger_queue(state, rng)

    should_skip = state.passed

    # Reset passed
    state = state._replace(passed=jnp.array(False, dtype=jnp.bool_))

    return state, should_skip, rng


def swap_active_player(state: GameState) -> GameState:
    """Swap the active player."""
    new_active = 1 - state.active_player
    new_turn = state.turn_count + 1
    return state._replace(
        active_player=new_active.astype(jnp.int32),
        turn_count=new_turn.astype(jnp.int32)
    )


def get_action_mask(state: GameState) -> Array:
    """Get mask of valid actions for the active player."""
    active_idx = state.active_player
    entity = get_entity_by_idx(state, active_idx)
    return entity.abilities_valid


def initialize_game(state: GameState, rng: Array) -> Tuple[GameState, Array]:
    """Initialize a new game by triggering ON_GAME_START."""
    state, rng = trigger_effects(state, TRIGGER_ON_GAME_START, rng)
    state, rng = process_trigger_queue(state, rng)
    return state, rng


# Aliases for backward compatibility
get_entity_jax = get_entity_by_idx
set_entity_jax = set_entity_by_idx
