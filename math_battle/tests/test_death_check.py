
import jax
import jax.numpy as jnp
from src.engine import initialize_game, execute_ability
from src.heroes import create_fighter, create_fire_mage
from src.game_state import create_initial_game_state, ATTR_HEALTH

def test_death_check_trigger():
    # Create two fighters
    p1 = create_fighter()
    p2 = create_fighter()
    
    # Set p2 health to 5 so one hit kills/triggers death check
    # But fighter has 100 HP. Let's set it manually.
    p2_attrs = p2.attributes.at[ATTR_HEALTH].set(5.0)
    p2 = p2._replace(attributes=p2_attrs)
    
    state = create_initial_game_state(p1, p2)
    rng = jax.random.PRNGKey(0)
    
    state, rng = initialize_game(state, rng)
    
    # Player 1 uses Basic Attack (Ability 0) on Player 2
    # Basic attack deals 10 damage (Strength 10)
    # p2 health 5 -> -5. Should trigger death check (< 1) -> Lose.
    
    state, rng = execute_ability(state, 0, rng)
    
    # Check if game is done
    assert state.done, "Game should be done after fatal blow"
    assert state.winner == 0, "Player 1 should win"
    
    print("Test passed!")

if __name__ == "__main__":
    test_death_check_trigger()
