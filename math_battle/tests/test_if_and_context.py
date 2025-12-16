"""Tests for IF branching semantics and attribute change context."""

import jax
import jax.numpy as jnp
import pytest

from src.effect_ops import (
    ProgramBuilder, C, Attr, Ctx,
    TARGET_SELF, TARGET_OPPONENT,
    CTX_ATTR_DELTA, CTX_ATTR_OLD, CTX_ATTR_NEW, CTX_ATTR_ID,
)
from src.game_state import (
    create_initial_game_state, ATTR_HEALTH, ATTR_STRENGTH, ATTR_DEFENSE,
    TRIGGER_ON_ATTRIBUTE_CHANGE,
)
from src.heroes import create_fighter
from src.engine import initialize_game, execute_ability, process_trigger_queue
from src.effect_interpreter import execute_program


def test_if_branch_only_one_executes():
    """Test that only one branch of IF executes, not both.

    Creates a program: if health > 50: set strength=100 else set defense=100
    Then verifies only one attribute was modified.
    """
    # Create fighter with health=100
    p1 = create_fighter()
    p2 = create_fighter()

    # Build a custom ability that tests IF branching
    # if self.health > 50: set self.strength = 100
    # else: set self.defense = 100
    # Then set health = 1 (to verify we continue after IF)
    then_builder = ProgramBuilder()
    then_builder.set_attr(TARGET_SELF, ATTR_STRENGTH, C(100.0))

    else_builder = ProgramBuilder()
    else_builder.set_attr(TARGET_SELF, ATTR_DEFENSE, C(100.0))

    builder = ProgramBuilder()
    builder.if_gt(
        Attr(TARGET_SELF, ATTR_HEALTH),  # health = 100
        C(50.0),  # > 50
        then_builder,
        else_builder
    )
    # This runs after the IF to verify execution continues
    builder.set_attr(TARGET_SELF, ATTR_HEALTH, C(1.0))
    builder.end()
    program = builder.build()

    state = create_initial_game_state(p1, p2)
    rng = jax.random.PRNGKey(42)

    # Execute the program (health=100 > 50, so then-branch runs)
    state, passed, winner, rng = execute_program(program, state, 0, rng)

    # Then-branch should have run (strength = 100)
    assert float(state.player.attributes[ATTR_STRENGTH]) == 100.0, \
        f"Expected strength=100, got {state.player.attributes[ATTR_STRENGTH]}"

    # Else-branch should NOT have run (defense stays at original)
    original_defense = float(p1.attributes[ATTR_DEFENSE])
    assert float(state.player.attributes[ATTR_DEFENSE]) == original_defense, \
        f"Defense should be unchanged at {original_defense}, got {state.player.attributes[ATTR_DEFENSE]}"

    # Code after IF should have run (health = 1)
    assert float(state.player.attributes[ATTR_HEALTH]) == 1.0, \
        f"Expected health=1 after IF block, got {state.player.attributes[ATTR_HEALTH]}"


def test_if_else_branch_executes():
    """Test that else branch executes when condition is false."""
    p1 = create_fighter()
    p2 = create_fighter()

    # Set health low so condition fails
    p1_attrs = p1.attributes.at[ATTR_HEALTH].set(30.0)
    p1 = p1._replace(attributes=p1_attrs)

    # Build: if health > 50: strength=100 else defense=100
    then_builder = ProgramBuilder()
    then_builder.set_attr(TARGET_SELF, ATTR_STRENGTH, C(100.0))

    else_builder = ProgramBuilder()
    else_builder.set_attr(TARGET_SELF, ATTR_DEFENSE, C(100.0))

    builder = ProgramBuilder()
    builder.if_gt(
        Attr(TARGET_SELF, ATTR_HEALTH),  # health = 30
        C(50.0),  # > 50 is FALSE
        then_builder,
        else_builder
    )
    builder.end()
    program = builder.build()

    state = create_initial_game_state(p1, p2)
    rng = jax.random.PRNGKey(42)

    state, passed, winner, rng = execute_program(program, state, 0, rng)

    # Else-branch should have run (defense = 100)
    assert float(state.player.attributes[ATTR_DEFENSE]) == 100.0, \
        f"Expected defense=100, got {state.player.attributes[ATTR_DEFENSE]}"

    # Then-branch should NOT have run (strength unchanged)
    original_strength = float(create_fighter().attributes[ATTR_STRENGTH])
    assert float(state.player.attributes[ATTR_STRENGTH]) == original_strength, \
        f"Strength should be unchanged at {original_strength}"


def test_ctx_attr_delta_on_attribute_change():
    """Test that CTX_ATTR_DELTA is correctly set for ON_ATTRIBUTE_CHANGE triggers.

    Creates an effect that reads CTX_ATTR_DELTA and applies it to another attribute.
    """
    from src.heroes import set_effect

    p1 = create_fighter()
    p2 = create_fighter()

    # Create an effect that triggers on health change and adds delta to defense
    # Effect: ON_ATTRIBUTE_CHANGE(HEALTH): add_attr(self, defense, ctx[DELTA])
    effect_builder = ProgramBuilder()
    effect_builder.add_attr(TARGET_SELF, ATTR_DEFENSE, Ctx(CTX_ATTR_DELTA))
    effect_builder.end()
    effect_program = effect_builder.build()

    # Add this effect to p1
    p1 = set_effect(p1, 0, TRIGGER_ON_ATTRIBUTE_CHANGE, ATTR_HEALTH, effect_program)

    state = create_initial_game_state(p1, p2)
    rng = jax.random.PRNGKey(42)

    # Initialize game
    state, rng = initialize_game(state, rng)

    initial_defense = float(state.player.attributes[ATTR_DEFENSE])
    initial_health = float(state.player.attributes[ATTR_HEALTH])

    # Create a program that reduces health by 15
    damage_builder = ProgramBuilder()
    damage_builder.add_attr(TARGET_SELF, ATTR_HEALTH, C(-15.0))
    damage_builder.end()
    damage_program = damage_builder.build()

    # Execute the damage program
    state, passed, winner, rng = execute_program(damage_program, state, 0, rng)

    # Process the attribute change queue
    state, rng = process_trigger_queue(state, rng)

    # Health should be reduced by 15
    expected_health = initial_health - 15.0
    assert float(state.player.attributes[ATTR_HEALTH]) == expected_health, \
        f"Expected health={expected_health}, got {state.player.attributes[ATTR_HEALTH]}"

    # Defense should have delta (-15) added to it via the trigger
    expected_defense = initial_defense + (-15.0)
    assert float(state.player.attributes[ATTR_DEFENSE]) == expected_defense, \
        f"Expected defense={expected_defense} (delta=-15 applied), got {state.player.attributes[ATTR_DEFENSE]}"


def test_ctx_attr_old_new_values():
    """Test that CTX_ATTR_OLD and CTX_ATTR_NEW are correctly populated."""
    from src.heroes import set_effect

    p1 = create_fighter()
    p2 = create_fighter()

    # Set initial health to 80
    p1_attrs = p1.attributes.at[ATTR_HEALTH].set(80.0)
    p1 = p1._replace(attributes=p1_attrs)

    # Create an effect that stores old value in strength and new value in defense
    # ON_ATTRIBUTE_CHANGE(HEALTH):
    #   set_attr(self, strength, ctx[OLD])
    #   set_attr(self, defense, ctx[NEW])
    effect_builder = ProgramBuilder()
    effect_builder.set_attr(TARGET_SELF, ATTR_STRENGTH, Ctx(CTX_ATTR_OLD))
    effect_builder.set_attr(TARGET_SELF, ATTR_DEFENSE, Ctx(CTX_ATTR_NEW))
    effect_builder.end()
    effect_program = effect_builder.build()

    p1 = set_effect(p1, 0, TRIGGER_ON_ATTRIBUTE_CHANGE, ATTR_HEALTH, effect_program)

    state = create_initial_game_state(p1, p2)
    rng = jax.random.PRNGKey(42)
    state, rng = initialize_game(state, rng)

    # Change health from 80 to 50
    change_builder = ProgramBuilder()
    change_builder.set_attr(TARGET_SELF, ATTR_HEALTH, C(50.0))
    change_builder.end()
    change_program = change_builder.build()

    state, passed, winner, rng = execute_program(change_program, state, 0, rng)
    state, rng = process_trigger_queue(state, rng)

    # Strength should be old value (80)
    assert float(state.player.attributes[ATTR_STRENGTH]) == 80.0, \
        f"Expected strength=80 (old health), got {state.player.attributes[ATTR_STRENGTH]}"

    # Defense should be new value (50)
    assert float(state.player.attributes[ATTR_DEFENSE]) == 50.0, \
        f"Expected defense=50 (new health), got {state.player.attributes[ATTR_DEFENSE]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
