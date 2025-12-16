"""Hero definitions for Math Battle.

This module provides pre-built hero configurations using the Effect Ops system.
Heroes are defined with attributes, abilities, and passive effects.
"""

import jax.numpy as jnp
from jax import Array

from .game_state import (
    Entity, MAX_ATTRIBUTES, MAX_ABILITIES, MAX_EFFECTS, MAX_TAGS_PER_ABILITY,
    ATTR_HEALTH, ATTR_MAX_HEALTH, ATTR_MANA, ATTR_MAX_MANA, ATTR_STRENGTH,
    ATTR_DEFENSE, ATTR_BURN, ATTR_STUN, ATTR_MANA_REGEN, ATTR_RAGE_BONUS,
    TRIGGER_ON_TURN_START, TRIGGER_ON_ACTION_PHASE_START,
    TRIGGER_ON_ATTRIBUTE_CHANGE,
    TAG_ATTACK, TAG_SPELL, TAG_FIRE, TAG_PHYSICAL, TAG_HEAL,
    create_empty_entity,
)
from .effect_ops import (
    Program, ProgramBuilder, MAX_OPS, MAX_VALUE_DEPTH,
    TARGET_SELF, TARGET_OPPONENT,
    C, Attr, Roll, vs_add, vs_neg, vs_min,
    create_const, create_attr,
)


# =============================================================================
# Entity Builder
# =============================================================================

def create_entity(
    attributes: dict,
    abilities: list,
    effects: list,
) -> Entity:
    """Create an Entity from human-readable definitions.

    Args:
        attributes: Dict mapping attribute names to values
        abilities: List of dicts with 'name', 'tags', 'program' (Program) keys
        effects: List of dicts with 'trigger', 'trigger_param', 'program' (Program) keys

    Returns:
        A fully initialized Entity
    """
    # Map attribute names to indices
    attr_name_to_idx = {
        "health": ATTR_HEALTH,
        "max_health": ATTR_MAX_HEALTH,
        "mana": ATTR_MANA,
        "max_mana": ATTR_MAX_MANA,
        "strength": ATTR_STRENGTH,
        "defense": ATTR_DEFENSE,
        "burn": ATTR_BURN,
        "stun": ATTR_STUN,
        "mana_regen": ATTR_MANA_REGEN,
        "rage_bonus": ATTR_RAGE_BONUS,
    }

    # Start from empty entity
    entity = create_empty_entity()

    # Fill attributes
    attr_list = [0.0] * MAX_ATTRIBUTES
    for name, value in attributes.items():
        if name in attr_name_to_idx:
            attr_list[attr_name_to_idx[name]] = float(value)
    attr_array = jnp.array(attr_list, dtype=jnp.float32)
    entity = entity._replace(attributes=attr_array)

    # Fill abilities
    abilities_valid_list = [False] * MAX_ABILITIES
    abilities_tags_list = [[0] * MAX_TAGS_PER_ABILITY for _ in range(MAX_ABILITIES)]

    # Initialize ability program arrays
    ab_op_types = jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32)
    ab_targets = jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32)
    ab_attr_ids = jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32)
    ab_value_types = jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.int32)
    ab_value_param1 = jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32)
    ab_value_param2 = jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32)
    ab_value_num = jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32)
    ab_value2_types = jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.int32)
    ab_value2_param1 = jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32)
    ab_value2_param2 = jnp.zeros((MAX_ABILITIES, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32)
    ab_value2_num = jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32)
    ab_then_count = jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32)
    ab_else_count = jnp.zeros((MAX_ABILITIES, MAX_OPS), dtype=jnp.int32)
    ab_num_ops = jnp.zeros(MAX_ABILITIES, dtype=jnp.int32)

    for i, ability in enumerate(abilities[:MAX_ABILITIES]):
        program = ability.get("program")
        if program is None:
            continue

        abilities_valid_list[i] = True

        # Copy program data
        ab_op_types = ab_op_types.at[i].set(program.op_types)
        ab_targets = ab_targets.at[i].set(program.targets)
        ab_attr_ids = ab_attr_ids.at[i].set(program.attr_ids)
        ab_value_types = ab_value_types.at[i].set(program.value_types)
        ab_value_param1 = ab_value_param1.at[i].set(program.value_param1)
        ab_value_param2 = ab_value_param2.at[i].set(program.value_param2)
        ab_value_num = ab_value_num.at[i].set(program.value_num_nodes)
        ab_value2_types = ab_value2_types.at[i].set(program.value2_types)
        ab_value2_param1 = ab_value2_param1.at[i].set(program.value2_param1)
        ab_value2_param2 = ab_value2_param2.at[i].set(program.value2_param2)
        ab_value2_num = ab_value2_num.at[i].set(program.value2_num_nodes)
        ab_then_count = ab_then_count.at[i].set(program.if_then_count)
        ab_else_count = ab_else_count.at[i].set(program.if_else_count)
        ab_num_ops = ab_num_ops.at[i].set(program.num_ops)

        # Tags
        tags = ability.get("tags", [])
        for j, tag in enumerate(tags[:MAX_TAGS_PER_ABILITY]):
            abilities_tags_list[i][j] = tag

    abilities_valid = jnp.array(abilities_valid_list, dtype=jnp.bool_)
    abilities_tags = jnp.array(abilities_tags_list, dtype=jnp.int32)

    entity = entity._replace(
        abilities_op_types=ab_op_types,
        abilities_targets=ab_targets,
        abilities_attr_ids=ab_attr_ids,
        abilities_value_types=ab_value_types,
        abilities_value_param1=ab_value_param1,
        abilities_value_param2=ab_value_param2,
        abilities_value_num_nodes=ab_value_num,
        abilities_value2_types=ab_value2_types,
        abilities_value2_param1=ab_value2_param1,
        abilities_value2_param2=ab_value2_param2,
        abilities_value2_num_nodes=ab_value2_num,
        abilities_if_then_count=ab_then_count,
        abilities_if_else_count=ab_else_count,
        abilities_num_ops=ab_num_ops,
        abilities_valid=abilities_valid,
        abilities_tags=abilities_tags,
    )

    # Fill effects
    effects_trigger_list = [0] * MAX_EFFECTS
    effects_trigger_param_list = [0] * MAX_EFFECTS

    # Initialize effect program arrays
    ef_op_types = jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32)
    ef_targets = jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32)
    ef_attr_ids = jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32)
    ef_value_types = jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.int32)
    ef_value_param1 = jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32)
    ef_value_param2 = jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32)
    ef_value_num = jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32)
    ef_value2_types = jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.int32)
    ef_value2_param1 = jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32)
    ef_value2_param2 = jnp.zeros((MAX_EFFECTS, MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32)
    ef_value2_num = jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32)
    ef_then_count = jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32)
    ef_else_count = jnp.zeros((MAX_EFFECTS, MAX_OPS), dtype=jnp.int32)
    ef_num_ops = jnp.zeros(MAX_EFFECTS, dtype=jnp.int32)

    for i, effect in enumerate(effects[:MAX_EFFECTS]):
        effects_trigger_list[i] = effect.get("trigger", 0)
        effects_trigger_param_list[i] = effect.get("trigger_param", -1)

        program = effect.get("program")
        if program is None:
            continue

        # Copy program data
        ef_op_types = ef_op_types.at[i].set(program.op_types)
        ef_targets = ef_targets.at[i].set(program.targets)
        ef_attr_ids = ef_attr_ids.at[i].set(program.attr_ids)
        ef_value_types = ef_value_types.at[i].set(program.value_types)
        ef_value_param1 = ef_value_param1.at[i].set(program.value_param1)
        ef_value_param2 = ef_value_param2.at[i].set(program.value_param2)
        ef_value_num = ef_value_num.at[i].set(program.value_num_nodes)
        ef_value2_types = ef_value2_types.at[i].set(program.value2_types)
        ef_value2_param1 = ef_value2_param1.at[i].set(program.value2_param1)
        ef_value2_param2 = ef_value2_param2.at[i].set(program.value2_param2)
        ef_value2_num = ef_value2_num.at[i].set(program.value2_num_nodes)
        ef_then_count = ef_then_count.at[i].set(program.if_then_count)
        ef_else_count = ef_else_count.at[i].set(program.if_else_count)
        ef_num_ops = ef_num_ops.at[i].set(program.num_ops)

    effects_trigger = jnp.array(effects_trigger_list, dtype=jnp.int32)
    effects_trigger_param = jnp.array(effects_trigger_param_list, dtype=jnp.int32)

    entity = entity._replace(
        effects_trigger=effects_trigger,
        effects_trigger_param=effects_trigger_param,
        effects_op_types=ef_op_types,
        effects_targets=ef_targets,
        effects_attr_ids=ef_attr_ids,
        effects_value_types=ef_value_types,
        effects_value_param1=ef_value_param1,
        effects_value_param2=ef_value_param2,
        effects_value_num_nodes=ef_value_num,
        effects_value2_types=ef_value2_types,
        effects_value2_param1=ef_value2_param1,
        effects_value2_param2=ef_value2_param2,
        effects_value2_num_nodes=ef_value2_num,
        effects_if_then_count=ef_then_count,
        effects_if_else_count=ef_else_count,
        effects_num_ops=ef_num_ops,
    )

    return entity


# =============================================================================
# Ability Program Builders
# =============================================================================

def build_basic_attack() -> Program:
    """Basic Attack: Deal damage equal to SELF's strength.

    Effect Ops: DAMAGE(OPPONENT, ATTR(SELF, STRENGTH))
    """
    builder = ProgramBuilder()
    # DAMAGE is ADD_ATTR(target, HEALTH, -amount)
    builder.add_attr(
        TARGET_OPPONENT,
        ATTR_HEALTH,
        vs_neg(Attr(TARGET_SELF, ATTR_STRENGTH))
    )
    builder.end()
    return builder.build()


def build_power_strike() -> Program:
    """Power Strike: Deal strength + roll(6) damage.

    Effect Ops: DAMAGE(OPPONENT, ADD(ATTR(SELF, STRENGTH), ROLL(6)))
    """
    builder = ProgramBuilder()
    damage = vs_add(Attr(TARGET_SELF, ATTR_STRENGTH), Roll(6))
    builder.add_attr(TARGET_OPPONENT, ATTR_HEALTH, vs_neg(damage))
    builder.end()
    return builder.build()


def build_defend() -> Program:
    """Defend: Increase defense by 3.

    Effect Ops: ADD_ATTR(SELF, DEFENSE, 3)
    """
    builder = ProgramBuilder()
    builder.add_attr(TARGET_SELF, ATTR_DEFENSE, C(3.0))
    builder.end()
    return builder.build()


def build_fireball() -> Program:
    """Fireball: Cost 5 mana, deal 8 damage, apply 2 burn.

    Effect Ops:
        ADD_ATTR(SELF, MANA, -5)
        ADD_ATTR(OPPONENT, HEALTH, -8)
        ADD_ATTR(OPPONENT, BURN, 2)
    """
    builder = ProgramBuilder()
    builder.add_attr(TARGET_SELF, ATTR_MANA, C(-5.0))
    builder.add_attr(TARGET_OPPONENT, ATTR_HEALTH, C(-8.0))
    builder.add_attr(TARGET_OPPONENT, ATTR_BURN, C(2.0))
    builder.end()
    return builder.build()


def build_heal() -> Program:
    """Heal: Cost 3 mana, heal 5 HP (up to max).

    Effect Ops:
        ADD_ATTR(SELF, MANA, -3)
        SET_ATTR(SELF, HEALTH, MIN(HEALTH + 5, MAX_HEALTH))
    """
    builder = ProgramBuilder()
    builder.add_attr(TARGET_SELF, ATTR_MANA, C(-3.0))
    # Heal up to max: SET to MIN(current + 5, max)
    new_health = vs_min(
        vs_add(Attr(TARGET_SELF, ATTR_HEALTH), C(5.0)),
        Attr(TARGET_SELF, ATTR_MAX_HEALTH)
    )
    builder.set_attr(TARGET_SELF, ATTR_HEALTH, new_health)
    builder.end()
    return builder.build()


def build_ice_bolt() -> Program:
    """Ice Bolt: Cost 4 mana, deal 6 damage, apply 1 stun.

    Effect Ops:
        ADD_ATTR(SELF, MANA, -4)
        ADD_ATTR(OPPONENT, HEALTH, -6)
        ADD_ATTR(OPPONENT, STUN, 1)
    """
    builder = ProgramBuilder()
    builder.add_attr(TARGET_SELF, ATTR_MANA, C(-4.0))
    builder.add_attr(TARGET_OPPONENT, ATTR_HEALTH, C(-6.0))
    builder.add_attr(TARGET_OPPONENT, ATTR_STUN, C(1.0))
    builder.end()
    return builder.build()


# =============================================================================
# Effect Program Builders
# =============================================================================

def build_mana_regen_effect() -> Program:
    """Mana Regen: ON_TURN_START, add mana_regen to mana.

    Effect Ops: ADD_ATTR(SELF, MANA, ATTR(SELF, MANA_REGEN))
    """
    builder = ProgramBuilder()
    builder.add_attr(TARGET_SELF, ATTR_MANA, Attr(TARGET_SELF, ATTR_MANA_REGEN))
    builder.end()
    return builder.build()


def build_burn_effect() -> Program:
    """Burn Damage: ON_TURN_START, if burn > 0, take burn damage and reduce burn.

    Effect Ops:
        IF_GT(ATTR(SELF, BURN), 0,
            then=[
                ADD_ATTR(SELF, HEALTH, -ATTR(SELF, BURN))
                ADD_ATTR(SELF, BURN, -1)
            ],
            else=[END]
        )
    """
    # Then branch: take damage, reduce burn
    then_builder = ProgramBuilder()
    then_builder.add_attr(TARGET_SELF, ATTR_HEALTH, vs_neg(Attr(TARGET_SELF, ATTR_BURN)))
    then_builder.add_attr(TARGET_SELF, ATTR_BURN, C(-1.0))
    then_builder.end()

    # Else branch: do nothing
    else_builder = ProgramBuilder()
    else_builder.end()

    # Main program
    builder = ProgramBuilder()
    builder.if_gt(
        Attr(TARGET_SELF, ATTR_BURN),
        C(0.0),
        then_builder,
        else_builder
    )
    builder.end()
    return builder.build()


def build_stun_effect() -> Program:
    """Stun: ON_ACTION_PHASE_START, if stun > 0, reduce stun and PASS.

    Effect Ops:
        IF_GT(ATTR(SELF, STUN), 0,
            then=[
                ADD_ATTR(SELF, STUN, -1)
                PASS()
            ],
            else=[END]
        )
    """
    # Then branch: reduce stun, pass
    then_builder = ProgramBuilder()
    then_builder.add_attr(TARGET_SELF, ATTR_STUN, C(-1.0))
    then_builder.pass_turn()

    # Else branch: do nothing
    else_builder = ProgramBuilder()
    else_builder.end()

    # Main program
    builder = ProgramBuilder()
    builder.if_gt(
        Attr(TARGET_SELF, ATTR_STUN),
        C(0.0),
        then_builder,
        else_builder
    )
    builder.end()
    return builder.build()


def build_death_check_effect() -> Program:
    """Death Check: ON_ATTRIBUTE_CHANGE(HEALTH), if health < 1, LOSE.

    Effect Ops:
        IF_LT(ATTR(SELF, HEALTH), 1,
            then=[LOSE(SELF)],
            else=[END]
        )
    """
    # Then branch: lose
    then_builder = ProgramBuilder()
    then_builder.lose(TARGET_SELF)

    # Else branch: do nothing
    else_builder = ProgramBuilder()
    else_builder.end()

    # Main program
    builder = ProgramBuilder()
    builder.if_lt(
        Attr(TARGET_SELF, ATTR_HEALTH),
        C(1.0),
        then_builder,
        else_builder
    )
    builder.end()
    return builder.build()


# =============================================================================
# Predefined Heroes
# =============================================================================

def create_fighter() -> Entity:
    """Create a Fighter hero.

    Fighter: High health, strength-based damage, no mana.
    - Health: 100, Max Health: 100
    - Strength: 10, Defense: 5
    - Abilities:
        1. Basic Attack: Deal strength damage
        2. Power Strike: Deal strength + d6 damage
        3. Defend: +3 defense
    """
    attributes = {
        "health": 100,
        "max_health": 100,
        "mana": 0,
        "max_mana": 0,
        "strength": 10,
        "defense": 5,
        "burn": 0,
        "stun": 0,
    }

    abilities = [
        {
            "name": "Basic Attack",
            "tags": [TAG_ATTACK, TAG_PHYSICAL],
            "program": build_basic_attack(),
        },
        {
            "name": "Power Strike",
            "tags": [TAG_ATTACK, TAG_PHYSICAL],
            "program": build_power_strike(),
        },
        {
            "name": "Defend",
            "tags": [],
            "program": build_defend(),
        },
    ]

    effects = [
        # Death check on health change
        {
            "trigger": TRIGGER_ON_ATTRIBUTE_CHANGE,
            "trigger_param": ATTR_HEALTH,
            "program": build_death_check_effect(),
        },
        # Burn damage on turn start
        {
            "trigger": TRIGGER_ON_TURN_START,
            "trigger_param": -1,
            "program": build_burn_effect(),
        },
        # Stun check on action phase
        {
            "trigger": TRIGGER_ON_ACTION_PHASE_START,
            "trigger_param": -1,
            "program": build_stun_effect(),
        },
    ]

    return create_entity(attributes, abilities, effects)


def create_fire_mage() -> Entity:
    """Create a Fire Mage hero.

    Fire Mage: Lower health, mana-based spellcasting.
    - Health: 70, Max Health: 70
    - Mana: 20, Max Mana: 30
    - Mana Regen: 3
    - Strength: 5
    - Abilities:
        1. Staff Strike: Deal strength damage (no mana cost)
        2. Fireball: 5 mana, 8 damage + 2 burn
        3. Heal: 3 mana, heal 5 HP
        4. Ice Bolt: 4 mana, 6 damage + 1 stun
    """
    attributes = {
        "health": 70,
        "max_health": 70,
        "mana": 20,
        "max_mana": 30,
        "strength": 5,
        "defense": 2,
        "burn": 0,
        "stun": 0,
        "mana_regen": 3,
    }

    abilities = [
        {
            "name": "Staff Strike",
            "tags": [TAG_ATTACK, TAG_PHYSICAL],
            "program": build_basic_attack(),
        },
        {
            "name": "Fireball",
            "tags": [TAG_SPELL, TAG_FIRE],
            "program": build_fireball(),
        },
        {
            "name": "Heal",
            "tags": [TAG_SPELL, TAG_HEAL],
            "program": build_heal(),
        },
        {
            "name": "Ice Bolt",
            "tags": [TAG_SPELL],
            "program": build_ice_bolt(),
        },
    ]

    effects = [
        # Death check
        {
            "trigger": TRIGGER_ON_ATTRIBUTE_CHANGE,
            "trigger_param": ATTR_HEALTH,
            "program": build_death_check_effect(),
        },
        # Mana regen
        {
            "trigger": TRIGGER_ON_TURN_START,
            "trigger_param": -1,
            "program": build_mana_regen_effect(),
        },
        # Burn damage
        {
            "trigger": TRIGGER_ON_TURN_START,
            "trigger_param": -1,
            "program": build_burn_effect(),
        },
        # Stun check
        {
            "trigger": TRIGGER_ON_ACTION_PHASE_START,
            "trigger_param": -1,
            "program": build_stun_effect(),
        },
    ]

    return create_entity(attributes, abilities, effects)


# =============================================================================
# Hero Registry
# =============================================================================

HERO_NAMES = {
    "fighter": create_fighter,
    "fire_mage": create_fire_mage,
}

ABILITY_NAMES = {
    "fighter": ["Basic Attack", "Power Strike", "Defend"],
    "fire_mage": ["Staff Strike", "Fireball", "Heal", "Ice Bolt"],
}


def get_ability_name(hero_type: str, ability_idx: int) -> str:
    """Get human-readable ability name."""
    names = ABILITY_NAMES.get(hero_type, [])
    if ability_idx < len(names):
        return names[ability_idx]
    return f"Ability {ability_idx}"


# =============================================================================
# Entity Modification Helpers
# =============================================================================

def set_effect(
    entity: Entity,
    effect_idx: int,
    trigger_type: int,
    trigger_param: int,
    program: Program,
) -> Entity:
    """Set a single effect on an entity.

    Args:
        entity: The entity to modify
        effect_idx: Index of the effect slot (0 to MAX_EFFECTS-1)
        trigger_type: Trigger type (e.g., TRIGGER_ON_ATTRIBUTE_CHANGE)
        trigger_param: Trigger parameter (e.g., attribute index for attr change)
        program: The effect program to execute

    Returns:
        Modified entity with the effect set
    """
    # Update trigger arrays
    new_trigger = entity.effects_trigger.at[effect_idx].set(trigger_type)
    new_trigger_param = entity.effects_trigger_param.at[effect_idx].set(trigger_param)

    # Update program arrays
    new_op_types = entity.effects_op_types.at[effect_idx].set(program.op_types)
    new_targets = entity.effects_targets.at[effect_idx].set(program.targets)
    new_attr_ids = entity.effects_attr_ids.at[effect_idx].set(program.attr_ids)
    new_value_types = entity.effects_value_types.at[effect_idx].set(program.value_types)
    new_value_param1 = entity.effects_value_param1.at[effect_idx].set(program.value_param1)
    new_value_param2 = entity.effects_value_param2.at[effect_idx].set(program.value_param2)
    new_value_num_nodes = entity.effects_value_num_nodes.at[effect_idx].set(program.value_num_nodes)
    new_value2_types = entity.effects_value2_types.at[effect_idx].set(program.value2_types)
    new_value2_param1 = entity.effects_value2_param1.at[effect_idx].set(program.value2_param1)
    new_value2_param2 = entity.effects_value2_param2.at[effect_idx].set(program.value2_param2)
    new_value2_num_nodes = entity.effects_value2_num_nodes.at[effect_idx].set(program.value2_num_nodes)
    new_then_count = entity.effects_if_then_count.at[effect_idx].set(program.if_then_count)
    new_else_count = entity.effects_if_else_count.at[effect_idx].set(program.if_else_count)
    new_num_ops = entity.effects_num_ops.at[effect_idx].set(program.num_ops)

    return entity._replace(
        effects_trigger=new_trigger,
        effects_trigger_param=new_trigger_param,
        effects_op_types=new_op_types,
        effects_targets=new_targets,
        effects_attr_ids=new_attr_ids,
        effects_value_types=new_value_types,
        effects_value_param1=new_value_param1,
        effects_value_param2=new_value_param2,
        effects_value_num_nodes=new_value_num_nodes,
        effects_value2_types=new_value2_types,
        effects_value2_param1=new_value2_param1,
        effects_value2_param2=new_value2_param2,
        effects_value2_num_nodes=new_value2_num_nodes,
        effects_if_then_count=new_then_count,
        effects_if_else_count=new_else_count,
        effects_num_ops=new_num_ops,
    )
