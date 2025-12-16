"""Tests for the JAX-compatible Math Battle environment."""

import jax
import jax.numpy as jnp
import unittest
from src.env import MathBattleFuncEnv, EnvParams, TwoPlayerEnv
from src.heroes import create_fighter, create_fire_mage
from src.agent import RandomAgent
from src.game_state import GameState, ATTR_HEALTH


class TestMathBattleJax(unittest.TestCase):
    def setUp(self):
        self.fighter = create_fighter()
        self.mage = create_fire_mage()
        self.params = EnvParams(
            player_template=self.fighter,
            opponent_template=self.mage,
            max_turns=50
        )
        self.env = MathBattleFuncEnv()
        self.rng = jax.random.PRNGKey(0)

    def test_initialization(self):
        state = self.env.initial(self.rng, self.params)
        # Check initial health
        self.assertEqual(float(state.player.attributes[ATTR_HEALTH]), 100.0)  # Fighter HP
        self.assertEqual(float(state.opponent.attributes[ATTR_HEALTH]), 70.0)  # Mage HP
        # Check done is False
        self.assertFalse(bool(state.done))

    def test_jit_step(self):
        """Test that the environment transition can be JIT compiled."""

        # Define a jitted step function
        @jax.jit
        def step_fn(state, action, rng):
            return self.env.transition(state, action, rng, self.params)

        state = self.env.initial(self.rng, self.params)
        rng, step_rng = jax.random.split(self.rng)

        # Run first step (compilation happens here)
        action = 0  # Basic attack
        next_state = step_fn(state, action, step_rng)

        # Check that something changed
        # Player (Fighter) active. Uses ability 0 (Basic Attack) on Opponent.
        # Fighter Strength=10. Mage HP should go from 70 to 60.
        final_hp = float(next_state.opponent.attributes[ATTR_HEALTH])
        self.assertEqual(final_hp, 60.0)
        self.assertFalse(bool(next_state.done))

    def test_full_game_random(self):
        """Simulate a game with random actions to ensure no crashes."""
        state = self.env.initial(self.rng, self.params)

        @jax.jit
        def play_step(s, r):
            a = 0
            ns = self.env.transition(s, a, r, self.params)
            d = self.env.terminal(ns, self.params)
            return ns, d

        rng = self.rng
        for _ in range(20):
            rng, step_rng = jax.random.split(rng)
            state, done = play_step(state, step_rng)
            if done:
                break

        # Should reach here without error
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
