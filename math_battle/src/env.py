"""MathBattleFuncEnv - A JAX-compatible functional environment for Math Battle.

This module provides a pure functional environment interface using the
Gymnasium Experimental Functional API.
"""

from typing import Tuple, Dict, Any, Optional, NamedTuple, Union
import jax
import jax.numpy as jnp
from jax import Array
import gymnasium as gym
from gymnasium import spaces
from gymnasium.experimental.functional import FuncEnv

from .game_state import (
    GameState, Entity, MAX_ATTRIBUTES, MAX_ABILITIES,
    create_initial_game_state, ATTR_HEALTH, ATTR_MAX_HEALTH,
)
from .engine import (
    initialize_game, run_turn_start_phase, run_turn_end_phase,
    check_action_phase_start, execute_ability, swap_active_player,
    get_action_mask,
)
from .dsl import get_entity_jax


class EnvParams(NamedTuple):
    """Static environment parameters."""
    player_template: Entity
    opponent_template: Entity
    max_turns: int = 100
    dense_reward: bool = False


class MathBattleFuncEnv(FuncEnv):
    """A functional environment for the Math Battle game.

    This environment uses the Gymnasium Experimental Functional API:
    - initial(rng, params) -> state
    - transition(state, action, rng, params) -> next_state
    - observation(state, params) -> obs
    - reward(state, action, next_state, params) -> reward
    - terminal(state, params) -> done

    The observation space is (2, MAX_ATTRIBUTES) representing both players' attributes.
    The action space is Discrete(MAX_ABILITIES).
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """Initialize the environment definition (spaces and metadata).
        
        Actual configuration (templates, max_turns) is passed via EnvParams.
        """
        # Define spaces
        self.observation_space = spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(2, MAX_ATTRIBUTES),
            dtype=jnp.float32
        )
        self.action_space = spaces.Discrete(MAX_ABILITIES)
        
        # Metadata
        self.metadata = {"render_modes": [], "render_fps": 30}

    def initial(self, rng: Array, params: EnvParams) -> GameState:
        """Pure functional reset."""
        state = create_initial_game_state(
            params.player_template,
            params.opponent_template,
        )

        # Initialize game (trigger ON_GAME_START)
        rng, init_rng = jax.random.split(rng)
        state, _ = initialize_game(state, init_rng)

        return state

    def transition(
        self, state: GameState, action: int, rng: Array, params: EnvParams
    ) -> GameState:
        """Pure functional state transition."""
        rng, step_rng = jax.random.split(rng)
        new_state, _ = self._execute_turn(state, action, step_rng)
        
        # Check truncation (max turns)
        is_truncated = new_state.turn_count >= params.max_turns
        new_done = jnp.logical_or(new_state.done, is_truncated)
        new_state = new_state._replace(done=new_done)

        return new_state

    def observation(self, state: GameState, params: EnvParams) -> Array:
        """Pure functional observation."""
        player_attrs = state.player.attributes
        opponent_attrs = state.opponent.attributes
        return jnp.stack([player_attrs, opponent_attrs])

    def reward(
        self, 
        state: GameState, 
        action: int, 
        next_state: GameState, 
        params: EnvParams
    ) -> float:
        """Pure functional reward calculation."""
        # Sparse reward: +1 for win, -1 for loss
        # Use jax.lax.cond or jnp.where for branchless logic if inside JIT.
        # Here we return float, but if JIT-ed it should return Array.
        # Assuming this runs in a JIT context, we return jnp arrays.
        
        def _calculate_reward(state, next_state):
             # Winner: -1 ongoing, 0 player 0, 1 player 1
             winner = next_state.winner
             
             win_reward = 1.0
             loss_reward = -1.0
             draw_reward = 0.0
             
             is_p0_win = (winner == 0)
             is_p1_win = (winner == 1)
             
             # Basic reward
             r = jnp.where(is_p0_win, win_reward, 
                           jnp.where(is_p1_win, loss_reward, draw_reward))
             
             # Only apply if done
             r = jnp.where(next_state.done, r, 0.0)
             
             # Dense reward
             if params.dense_reward:
                 prev_p_health = state.player.attributes[ATTR_HEALTH]
                 prev_o_health = state.opponent.attributes[ATTR_HEALTH]
                 new_p_health = next_state.player.attributes[ATTR_HEALTH]
                 new_o_health = next_state.opponent.attributes[ATTR_HEALTH]
                 
                 p_delta = new_p_health - prev_p_health
                 o_delta = new_o_health - prev_o_health
                 
                 dense = (o_delta - p_delta) * 0.01
                 r = r + dense
                 
             return r

        return _calculate_reward(state, next_state)

    def terminal(self, state: GameState, params: EnvParams) -> bool:
        """Pure functional termination check."""
        return state.done

    def _execute_turn(
        self, state: GameState, action: int, rng: Array
    ) -> Tuple[GameState, Array]:
        """Execute a full turn including all phases (internal helper)."""

        # Phase 1: Turn Start
        rng, phase_rng = jax.random.split(rng)
        state, phase_rng = run_turn_start_phase(state, phase_rng)

        # Optimization: Early exit check logic needs to be functional for JIT
        # We'll just run through but updates will be no-ops if done.
        # However, the engine functions should handle done states gracefully (check_done checks).
        
        # Phase 2: Action Phase
        rng, phase_rng = jax.random.split(rng)

        # Check for forced pass
        state, should_skip, phase_rng = check_action_phase_start(state, phase_rng)

        # Conditional execution of ability
        # If not done and not skip: execute ability
        # Logic: execute_ability should probably handle the check or we use lax.cond
        # For simplicity and robust JIT, we rely on engine to check `done`. 
        # But `should_skip` is local.
        
        rng, ability_rng = jax.random.split(rng)
        
        def _do_ability(s, r):
            return execute_ability(s, action, r)
            
        def _skip_ability(s, r):
            return s, r

        # Execute ability if (not should_skip) and (not state.done)
        # engine.execute_ability handles "state.done" check internally? 
        # If not, we should wrap it. Assuming engine functions are safe.
        # But `should_skip` logic:
        
        can_act = jnp.logical_not(jnp.logical_or(should_skip, state.done))
        
        # We need to pass rng to both branches or handle it. 
        # execute_ability returns (state, rng) (actually (state, new_rng) or just state?)
        # Signature: execute_ability(state, action_idx, rng) -> (state, rng)
        
        state, _ = jax.lax.cond(
            can_act,
            lambda args: execute_ability(args[0], action, args[1]),
            lambda args: (args[0], args[1]), # Identity
            (state, ability_rng)
        )

        # Phase 3: Turn End
        rng, phase_rng = jax.random.split(rng)
        state, _ = run_turn_end_phase(state, phase_rng)

        # Swap active player
        state = swap_active_player(state)

        return state, rng


class TwoPlayerEnv:
    """Wrapper for two-player games where both players take actions.
    
    Updated to use MathBattleFuncEnv logic.
    """

    def __init__(self, base_env: MathBattleFuncEnv, params: EnvParams):
        self.env = base_env
        self.params = params

    def reset(self, rng: Array) -> Tuple[GameState, Array, Dict[str, Any]]:
        """Reset and return initial state."""
        state = self.env.initial(rng, self.params)
        obs = self.env.observation(state, self.params)
        return state, obs, {}

    def step_player(
        self,
        state: GameState,
        action: int,
        rng: Array,
    ) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """Take a step for the current active player.

        Returns reward from the perspective of player 0.
        """
        rng, step_rng = jax.random.split(rng)
        next_state = self.env.transition(state, action, step_rng, self.params)
        reward = self.env.reward(state, action, next_state, self.params)
        done = self.env.terminal(next_state, self.params)
        
        info = {
            "action_mask": get_action_mask(next_state),
            "active_player": next_state.active_player,
            "turn_count": next_state.turn_count,
            "winner": next_state.winner,
        }
        
        return next_state, reward, done, info

    def get_observation_for_player(self, state: GameState, player: int) -> Array:
        """Get observation from a player's perspective."""
        # Use lax.cond or select for JIT compatibility
        # But player is usually int. 
        if isinstance(player, int):
             if player == 0:
                return jnp.stack([
                    state.player.attributes,
                    state.opponent.attributes,
                ])
             else:
                return jnp.stack([
                    state.opponent.attributes,
                    state.player.attributes,
                ])
        
        # If player is an array (JIT context)
        return jax.lax.cond(
            player == 0,
            lambda s: jnp.stack([s.player.attributes, s.opponent.attributes]),
            lambda s: jnp.stack([s.opponent.attributes, s.player.attributes]),
            state
        )

    def get_valid_actions(self, state: GameState) -> Array:
        """Get valid actions for current player."""
        return get_action_mask(state)

    def get_active_player(self, state: GameState) -> int:
        """Get index of active player."""
        # Cast to int for python side usage if needed, or keep as array
        return state.active_player

