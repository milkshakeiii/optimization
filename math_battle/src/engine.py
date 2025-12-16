"""Game Engine for Math Battle.

Implements the turn structure and effect triggering system.
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
from .dsl import execute_script, get_entity_jax, set_entity_jax


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
        # executor_idx < 0 means all entities
        should_process = jnp.logical_or(executor_idx < 0, executor_idx == idx)
        
        def run_entity_effects(inner_args):
            s_in, r_in = inner_args
            entity = get_entity_jax(s_in, idx)
            
            def effect_loop_body(i, loop_carry):
                cs, cr = loop_carry
                
                # If game is already done, skip execution (effectively no-op)
                # We use lax.cond inside to respect functional purity
                
                def execute_effect(args):
                    curr_s, curr_r = args
                    trigger = entity.effects_trigger[i]
                    param = entity.effects_trigger_param[i]
                    script = entity.effects_script[i]
                    
                    # Check matches
                    trigger_matches = (trigger == trigger_type)
                    # param matches if trigger_param < 0 or param == trigger_param or param < 0
                    # Note: trigger_param is usually passed as int, but if dynamic we handle it.
                    param_matches = jnp.logical_or(
                        trigger_param < 0,
                        jnp.logical_or(param == trigger_param, param < 0)
                    )
                    
                    should_run = jnp.logical_and(trigger_matches, param_matches)
                    
                    def run_script(s_r_tuple):
                        st, rn = s_r_tuple
                        st, _, winner, rn = execute_script(script, st, idx, rn)
                        
                        # Update winner/done
                        has_winner = winner >= 0
                        st = st._replace(
                             done=jnp.logical_or(st.done, has_winner),
                             winner=jnp.where(has_winner, winner, st.winner)
                        )
                        return st, rn

                    return lax.cond(should_run, run_script, lambda x: x, (curr_s, curr_r))

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
        # Continue while queue has items AND we haven't exceeded MAX_QUEUE total processing steps
        # The processed_count limits infinite loops if queue keeps growing
        has_items = s.queue_count > 0
        return jnp.logical_and(has_items, processed_count < MAX_QUEUE)
    
    def body_fun(args):
        s, r, processed_count = args
        
        # Pop first item (FIFO) or simpler: take all current items and reset queue?
        # But effect scripts might add NEW items.
        # Queue is FIFO.
        # But implementing a FIFO queue with shift in JAX is expensive (array copy).
        # Better: use queue as stack (LIFO) or just iterate up to count?
        # If we iterate 0..count, we can't easily handle appended items unless we restart loop.
        
        # Simpler approach:
        # Take ONE item from end (LIFO). 
        # s.queue_count - 1
        
        idx = s.queue_count - 1
        safe_idx = jnp.clip(idx, 0, MAX_QUEUE - 1)
        target_idx = s.queue[safe_idx, 0]
        attr_idx = s.queue[safe_idx, 1]
        
        # Decrement count
        s = s._replace(queue_count=idx)
        
        # Trigger
        s, r = trigger_effects(s, TRIGGER_ON_ATTRIBUTE_CHANGE, r, target_idx, attr_idx)
        
        return s, r, processed_count + 1

    state, rng, _ = lax.while_loop(cond_fun, body_fun, (state, rng, 0))
    
    # Ensure queue is clear (in case we hit limit)
    state = state._replace(queue_count=jnp.array(0, dtype=jnp.int32))
    
    return state, rng


def execute_ability(
    state: GameState,
    ability_idx: int,
    rng: Array,
) -> Tuple[GameState, Array]:
    """Execute an ability for the active player."""
    
    active_idx = state.active_player # Keep as Array/Tracer
    entity = get_entity_jax(state, active_idx)
    
    # Accessing array with ability_idx. If ability_idx is python int, it's fine.
    # If it's a tracer, we need slice/gather.
    # Assuming ability_idx is int or scalar array. 
    # entity.abilities_script is (MAX_ABILITIES, MAX_SCRIPT_LEN).
    # We use jnp.array(ability_idx) just in case.
    
    # Note: simple indexing `script[idx]` works in JAX if idx is tracer.
    script = entity.abilities_script[ability_idx]
    valid = entity.abilities_valid[ability_idx]

    def run_ability(args):
        s, r = args
        
        # Set context
        # We need to update context array functionally
        new_context = s.context.at[CTX_ABILITY_ID].set(ability_idx)
        s = s._replace(context=new_context)
        
        # Trigger ON_ABILITY_USED
        s, r = trigger_effects(s, TRIGGER_ON_ABILITY_USED, r, executor_idx=-1)
        
        # Process queue (e.g. from ability used effects)
        s, r = process_trigger_queue(s, r)
        
        def continue_execution(inner_args):
            s_in, r_in = inner_args
            s_in, _, winner, r_in = execute_script(script, s_in, active_idx, r_in)
            
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
    return trigger_effects(state, TRIGGER_ON_TURN_START, rng)


def run_turn_end_phase(state: GameState, rng: Array) -> Tuple[GameState, Array]:
    """Run the turn end phase."""
    state, rng = trigger_effects(state, TRIGGER_ON_TURN_END, rng)

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
    entity = get_entity_jax(state, active_idx)
    return entity.abilities_valid


def initialize_game(state: GameState, rng: Array) -> Tuple[GameState, Array]:
    """Initialize a new game by triggering ON_GAME_START."""
    return trigger_effects(state, TRIGGER_ON_GAME_START, rng)
