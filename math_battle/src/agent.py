"""Agents for Math Battle.

This module provides different agent implementations for playing the game.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import jax
import jax.numpy as jnp
from jax import Array

from .game_state import GameState, MAX_ABILITIES, ATTR_HEALTH, ATTR_MAX_HEALTH, ATTR_MANA
from .engine import get_action_mask
from .dsl import get_entity_jax


class Agent(ABC):
    """Abstract base class for agents."""

    @abstractmethod
    def select_action(
        self,
        state: GameState,
        rng: Array,
    ) -> int:
        """Select an action given the current state.

        Args:
            state: Current game state
            rng: Random key

        Returns:
            Action index (ability to use)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get agent name for display."""
        pass


class RandomAgent(Agent):
    """Agent that selects random valid actions."""

    def __init__(self, seed: int = 42):
        # Seed is handled by caller passing rng
        pass

    def select_action(
        self,
        state: GameState,
        rng: Array,
    ) -> int:
        mask = get_action_mask(state)
        # Convert mask to probabilities
        p = mask.astype(jnp.float32)
        p = p / jnp.sum(p)
        
        # Select action
        action = jax.random.choice(rng, MAX_ABILITIES, p=p)
        return int(action)

    def get_name(self) -> str:
        return "Random Agent"


class DummyAgent(Agent):
    """Simple heuristic-based agent.

    Strategy:
    1. If health is low and heal is available, heal
    2. If opponent health is low, use strongest attack
    3. Otherwise, use strongest available attack
    """

    def __init__(self, hero_type: str = "fighter"):
        self.hero_type = hero_type

    def select_action(
        self,
        state: GameState,
        rng: Array,
    ) -> int:
        active_idx = state.active_player
        entity = get_entity_jax(state, active_idx)
        opponent = get_entity_jax(state, 1 - active_idx)

        mask = get_action_mask(state)
        # mask is boolean array of size MAX_ABILITIES
        
        # Get health values
        my_health = entity.attributes[ATTR_HEALTH]
        max_health = entity.attributes[ATTR_MAX_HEALTH]
        opponent_health = opponent.attributes[ATTR_HEALTH]
        my_mana = entity.attributes[ATTR_MANA]

        # Note: logic here uses Python if/else which is fine if not JIT-ed.
        # But if JIT-ed, this must use lax.cond. 
        # Assuming this agent is primarily for evaluation/testing and not inside strict JIT training loop.
        # But to be safe and consistent with "JAX support only", we should try to be JIT compatible or at least use jnp types.
        # However, complex heuristics are hard to write with lax.cond without total restructuring.
        # For now, we will cast values to python types for logic if we assume this runs on CPU host (common for agents),
        # OR we write it using JAX primitives. 
        # Given "remove support for non-JAX environments", we should use JAX arrays.
        
        # We will implement a simplified version that returns a JAX scalar.
        
        # Helper to check validity
        def is_valid(idx):
            return mask[idx]

        action = jnp.array(0, dtype=jnp.int32) # Default
        
        # Find valid action fallback
        # First valid action
        first_valid = jnp.argmax(mask) # Returns index of first True
        action = first_valid

        if self.hero_type == "fire_mage":
            # Fire Mage logic: 0=Staff, 1=Fireball, 2=Heal, 3=Ice Bolt
            
            # 1. Heal
            cond_heal = jnp.logical_and(
                my_health < max_health * 0.4, 
                jnp.logical_and(is_valid(2), my_mana >= 3)
            )
            action = jnp.where(cond_heal, 2, action)
            
            # 2. Kill (Fireball)
            cond_kill = jnp.logical_and(
                opponent_health <= 8,
                jnp.logical_and(is_valid(1), my_mana >= 5)
            )
            action = jnp.where(cond_kill, 1, action)
            
            # 3. Fireball (Burn)
            cond_fire = jnp.logical_and(
                is_valid(1), my_mana >= 5
            )
            # Priority over fallback, but check if we didn't already pick heal/kill?
            # jnp.where is parallel. Order matters if nesting.
            # We structure it as a chain of priority:
            # priority: heal > kill > fireball > icebolt > staff
            
            # Let's reconstruct priority chain
            # Base: first_valid (Staff usually)
            current_best = first_valid
            
            # Ice Bolt
            cond_ice = jnp.logical_and(is_valid(3), my_mana >= 4)
            current_best = jnp.where(cond_ice, 3, current_best)
            
            # Fireball
            current_best = jnp.where(cond_fire, 1, current_best)
            
            # Kill
            current_best = jnp.where(cond_kill, 1, current_best)
            
            # Heal
            current_best = jnp.where(cond_heal, 2, current_best)
            
            action = current_best

        else:  # Fighter
            # Fighter logic: 0=Basic, 1=Power, 2=Defend
            
            # Base: Basic (0) or First valid
            current_best = first_valid
            
            # Defend (2)
            cond_def = is_valid(2)
            current_best = jnp.where(cond_def, 2, current_best)
            
            # Basic (0) - usually always valid, priority over defend?
            # Code said: Defend if nothing else. So Basic > Defend.
            cond_basic = is_valid(0)
            current_best = jnp.where(cond_basic, 0, current_best)
            
            # Power Strike (1) - 70% chance
            # We need rng
            rng, key = jax.random.split(rng)
            chance = jax.random.uniform(key)
            cond_power = jnp.logical_and(is_valid(1), chance < 0.7)
            current_best = jnp.where(cond_power, 1, current_best)
            
            # Kill (Power Strike)
            cond_kill = jnp.logical_and(opponent_health <= 15, is_valid(1))
            current_best = jnp.where(cond_kill, 1, current_best)
            
            action = current_best

        return int(action)

    def get_name(self) -> str:
        return f"Dummy Agent ({self.hero_type})"


class GreedyAgent(Agent):
    """Agent that always picks the highest damage ability."""

    def __init__(self, hero_type: str = "fighter"):
        self.hero_type = hero_type
        # Damage estimates for each hero's abilities
        # Padded to MAX_ABILITIES
        self.damage_estimates = jnp.zeros(MAX_ABILITIES)
        if hero_type == "fighter":
             # 0=Basic(10), 1=Power(13), 2=Defend(0)
             self.damage_estimates = self.damage_estimates.at[0].set(10)
             self.damage_estimates = self.damage_estimates.at[1].set(13)
             self.damage_estimates = self.damage_estimates.at[2].set(0)
        elif hero_type == "fire_mage":
             # 0=Staff(5), 1=Fireball(10), 2=Heal(-5), 3=Ice(7)
             self.damage_estimates = self.damage_estimates.at[0].set(5)
             self.damage_estimates = self.damage_estimates.at[1].set(10)
             self.damage_estimates = self.damage_estimates.at[2].set(-5)
             self.damage_estimates = self.damage_estimates.at[3].set(7)

    def select_action(
        self,
        state: GameState,
        rng: Array,
    ) -> int:
        mask = get_action_mask(state)
        # mask is boolean (MAX_ABILITIES,)
        
        # We want to select index with max damage where mask is True
        # Mask out invalid actions by setting damage to -inf
        masked_damage = jnp.where(mask, self.damage_estimates, -jnp.inf)
        
        best_action = jnp.argmax(masked_damage)
        
        return int(best_action)

    def get_name(self) -> str:
        return f"Greedy Agent ({self.hero_type})"


class HumanAgent(Agent):
    """Placeholder for human player in TUI."""

    def select_action(
        self,
        state: GameState,
        rng: Array,
    ) -> int:
        # This should never be called - TUI handles human input
        raise NotImplementedError("Human actions are handled by TUI")

    def get_name(self) -> str:
        return "Human Player"
