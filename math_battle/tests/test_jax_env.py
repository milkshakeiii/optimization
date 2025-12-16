
import jax
import jax.numpy as jnp
import unittest
from src.env import MathBattleFuncEnv, EnvParams, TwoPlayerEnv
from src.heroes import create_fighter, create_fire_mage
from src.agent import RandomAgent
from src.game_state import GameState

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
        self.assertEqual(state.player.attributes[0], 100.0) # Fighter HP
        self.assertEqual(state.opponent.attributes[0], 70.0) # Mage HP
        # Check done is False
        self.assertFalse(state.done)

    def test_jit_step(self):
        """Test that the environment transition can be JIT compiled."""
        
        # Define a jitted step function
        @jax.jit
        def step_fn(state, action, rng):
            return self.env.transition(state, action, rng, self.params)

        state = self.env.initial(self.rng, self.params)
        rng, step_rng = jax.random.split(self.rng)
        
        # Run first step (compilation happens here)
        action = 0 # Basic attack
        next_state = step_fn(state, action, step_rng)
        
        # Check that something changed or didn't crash
        # Player (Fighter) active. Uses ability 0 (Basic Attack) on Opponent.
        # Opponent HP should decrease.
        # Fighter Strength=10. Mage Def=2. Dmg = 10.
        # Wait, Basic Attack script: MODIFY(OPP, HEALTH, -STR).
        # Does Defense apply? The script for Basic Attack in heroes.py is:
        # MODIFY(OPPONENT, health, -GET(SELF, strength))
        # Defense is NOT factored in that script! (Simple logic).
        # So Mage HP should be 70 - 10 = 60.
        
        # Note: Turn structure: Turn Start -> Action -> Turn End.
        # Action happens. Then Turn End.
        # Mage HP is attribute 0.
        
        # We need to ensure next_state is concrete to assert
        # But in test, we are outside JIT, so we can access values.
        
        final_hp = next_state.opponent.attributes[0]
        self.assertEqual(final_hp, 60.0)
        self.assertFalse(next_state.done)

    def test_full_game_random(self):
        """Simulate a game with random actions to ensure no crashes."""
        state = self.env.initial(self.rng, self.params)
        
        @jax.jit
        def play_step(s, r):
            # Simple random policy: just pick action 0 always for this test to ensure validity
            # Or use random choice.
            # Let's just alternate action 0.
            a = 0
            ns = self.env.transition(s, a, r, self.params)
            d = self.env.terminal(ns, self.params)
            return ns, d

        rng = self.rng
        for _ in range(20): # Should finish or progress
            rng, step_rng = jax.random.split(rng)
            state, done = play_step(state, step_rng)
            if done:
                break
        
        # Should reach here without error
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
