"""Hero definitions for Math Battle.

This module provides pre-built hero configurations based on the design document.
"""

import jax.numpy as jnp

from .game_state import (
    Entity, MAX_ATTRIBUTES, MAX_ABILITIES, MAX_EFFECTS, MAX_SCRIPT_LEN,
    MAX_TAGS_PER_ABILITY,
    ATTR_HEALTH, ATTR_MAX_HEALTH, ATTR_MANA, ATTR_MAX_MANA, ATTR_STRENGTH,
    ATTR_DEFENSE, ATTR_BURN, ATTR_STUN, ATTR_MANA_REGEN, ATTR_RAGE_BONUS,
    TRIGGER_ON_TURN_START, TRIGGER_ON_ACTION_PHASE_START,
    TRIGGER_ON_ATTRIBUTE_CHANGE,
    TAG_ATTACK, TAG_SPELL, TAG_FIRE, TAG_PHYSICAL, TAG_HEAL,
)
from .dsl import (
    compile_self, compile_opponent, compile_push, compile_get, compile_set,
    compile_modify, compile_add, compile_sub, compile_mul, compile_roll,
    compile_win, compile_lose, compile_pass, compile_end, compile_lt,
    compile_gt, compile_if,
    pad_script, script_to_array,
    OP_SELF, OP_OPPONENT, OP_PUSH, OP_GET, OP_SET, OP_MODIFY, OP_ADD, OP_SUB,
    OP_MUL, OP_ROLL, OP_WIN, OP_LOSE, OP_PASS, OP_END, OP_LT, OP_GT, OP_IF,
    OP_NOOP, OP_MIN,
)


def create_entity(
    attributes: dict,
    abilities: list,
    effects: list,
) -> Entity:
    """Create an Entity from human-readable definitions.

    Args:
        attributes: Dict mapping attribute names to values
        abilities: List of dicts with 'name', 'tags', 'script' keys
        effects: List of dicts with 'trigger', 'trigger_param', 'script' keys
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

    # Fill attributes
    attr_list = [0.0] * MAX_ATTRIBUTES
    for name, value in attributes.items():
        if name in attr_name_to_idx:
            attr_list[attr_name_to_idx[name]] = float(value)
    attr_array = jnp.array(attr_list, dtype=jnp.float32)

    # Fill abilities
    abilities_script_list = [[0] * MAX_SCRIPT_LEN for _ in range(MAX_ABILITIES)]
    abilities_valid_list = [False] * MAX_ABILITIES
    abilities_tags_list = [[0] * MAX_TAGS_PER_ABILITY for _ in range(MAX_ABILITIES)]

    for i, ability in enumerate(abilities[:MAX_ABILITIES]):
        script = ability.get("script", [OP_NOOP, OP_END])
        padded = pad_script(script)
        abilities_script_list[i] = padded
        abilities_valid_list[i] = True

        tags = ability.get("tags", [])
        for j, tag in enumerate(tags[:MAX_TAGS_PER_ABILITY]):
            abilities_tags_list[i][j] = tag
            
    abilities_script = jnp.array(abilities_script_list, dtype=jnp.int32)
    abilities_valid = jnp.array(abilities_valid_list, dtype=jnp.bool_)
    abilities_tags = jnp.array(abilities_tags_list, dtype=jnp.int32)

    # Fill effects
    effects_trigger_list = [0] * MAX_EFFECTS
    effects_script_list = [[0] * MAX_SCRIPT_LEN for _ in range(MAX_EFFECTS)]
    effects_trigger_param_list = [0] * MAX_EFFECTS

    for i, effect in enumerate(effects[:MAX_EFFECTS]):
        effects_trigger_list[i] = effect.get("trigger", 0)
        effects_trigger_param_list[i] = effect.get("trigger_param", -1)
        script = effect.get("script", [OP_NOOP, OP_END])
        padded = pad_script(script)
        effects_script_list[i] = padded
        
    effects_trigger = jnp.array(effects_trigger_list, dtype=jnp.int32)
    effects_script = jnp.array(effects_script_list, dtype=jnp.int32)
    effects_trigger_param = jnp.array(effects_trigger_param_list, dtype=jnp.int32)

    return Entity(
        attributes=attr_array,
        abilities_script=abilities_script,
        abilities_valid=abilities_valid,
        abilities_tags=abilities_tags,
        effects_trigger=effects_trigger,
        effects_script=effects_script,
        effects_trigger_param=effects_trigger_param,
    )


# === SCRIPT BUILDERS ===

def build_basic_attack_script(damage_attr: int = ATTR_STRENGTH) -> list:
    """Build script: Deal damage equal to SELF's strength to OPPONENT.

    Script: MODIFY(OPPONENT, health, -GET(SELF, strength))
    Stack-based: OPPONENT, SELF, GET strength, NEGATE, MODIFY health
    """
    return [
        OP_OPPONENT,           # push opponent idx
        OP_SELF,               # push self idx
        OP_GET, damage_attr,   # get self's strength -> stack: [opp, strength]
        OP_PUSH, -100,         # push -1 (encoded as -100)
        OP_MUL,                # negate -> stack: [opp, -strength]
        OP_MODIFY, ATTR_HEALTH,  # modify opponent health
        OP_END,
    ]


def build_power_strike_script() -> list:
    """Build script: Deal strength + roll(6) damage.

    Script: MODIFY(OPPONENT, health, -(GET(SELF, strength) + ROLL(6)))
    """
    return [
        OP_OPPONENT,             # push opponent idx
        OP_SELF,                 # push self idx
        OP_GET, ATTR_STRENGTH,   # get strength
        OP_PUSH, 600,            # push 6 for dice
        OP_ROLL,                 # roll d6
        OP_ADD,                  # strength + roll
        OP_PUSH, -100,           # push -1
        OP_MUL,                  # negate
        OP_MODIFY, ATTR_HEALTH,  # apply damage
        OP_END,
    ]


def build_defend_script() -> list:
    """Build script: Increase defense by 3 for one turn."""
    return [
        OP_SELF,
        OP_PUSH, 300,            # 3.0
        OP_MODIFY, ATTR_DEFENSE,
        OP_END,
    ]


def build_fireball_script() -> list:
    """Build script: Cost 5 mana, deal 8 damage + 2 burn.

    MODIFY(SELF, mana, -5)
    MODIFY(OPPONENT, health, -8)
    MODIFY(OPPONENT, burn, 2)
    """
    return [
        # Cost: -5 mana
        OP_SELF,
        OP_PUSH, -500,           # -5.0
        OP_MODIFY, ATTR_MANA,

        # Damage: -8 health
        OP_OPPONENT,
        OP_PUSH, -800,           # -8.0
        OP_MODIFY, ATTR_HEALTH,

        # Apply burn: +2 burn stacks
        OP_OPPONENT,
        OP_PUSH, 200,            # 2.0
        OP_MODIFY, ATTR_BURN,

        OP_END,
    ]


def build_heal_script() -> list:
    """Build script: Cost 3 mana, heal 5 health (up to max).

    MODIFY(SELF, mana, -3)
    SET(SELF, health, MIN(GET(SELF, health) + 5, GET(SELF, max_health)))
    """
    return [
        # Cost: -3 mana
        OP_SELF,
        OP_PUSH, -300,
        OP_MODIFY, ATTR_MANA,

        # Heal: +5 health (capped at max)
        OP_SELF,
        OP_SELF,
        OP_GET, ATTR_HEALTH,
        OP_PUSH, 500,            # 5.0
        OP_ADD,                  # health + 5
        OP_SELF,
        OP_GET, ATTR_MAX_HEALTH,
        OP_MIN,                  # min(health+5, max_health)
        OP_SET, ATTR_HEALTH,

        OP_END,
    ]


def build_ice_bolt_script() -> list:
    """Build script: Cost 4 mana, deal 6 damage + 1 stun."""
    return [
        # Cost: -4 mana
        OP_SELF,
        OP_PUSH, -400,
        OP_MODIFY, ATTR_MANA,

        # Damage
        OP_OPPONENT,
        OP_PUSH, -600,
        OP_MODIFY, ATTR_HEALTH,

        # Stun
        OP_OPPONENT,
        OP_PUSH, 100,
        OP_MODIFY, ATTR_STUN,

        OP_END,
    ]


def build_mana_regen_effect() -> list:
    """Effect: ON_TURN_START, regenerate mana.

    MODIFY(SELF, mana, MIN(GET(SELF, mana_regen), GET(SELF, max_mana) - GET(SELF, mana)))
    Simplified: MODIFY(SELF, mana, mana_regen)
    """
    return [
        OP_SELF,
        OP_SELF,
        OP_GET, ATTR_MANA_REGEN,
        OP_MODIFY, ATTR_MANA,
        OP_END,
    ]


def build_burn_effect() -> list:
    """Effect: ON_TURN_START, take burn damage and reduce burn.

    IF(GET(SELF, burn) > 0):
        MODIFY(SELF, health, -GET(SELF, burn))
        MODIFY(SELF, burn, -1)
    """
    return [
        # Check if burn > 0
        OP_SELF,
        OP_GET, ATTR_BURN,
        OP_PUSH, 0,
        OP_GT,                   # burn > 0?
        OP_IF, 3, 15,            # if true, jump +3; else jump +15 (to end)

        # True branch: apply burn damage
        OP_SELF,
        OP_SELF,
        OP_GET, ATTR_BURN,
        OP_PUSH, -100,
        OP_MUL,                  # -burn
        OP_MODIFY, ATTR_HEALTH,

        # Reduce burn by 1
        OP_SELF,
        OP_PUSH, -100,
        OP_MODIFY, ATTR_BURN,

        OP_END,
    ]


def build_stun_effect() -> list:
    """Effect: ON_ACTION_PHASE_START, if stunned, pass and reduce stun.

    IF(GET(SELF, stun) > 0):
        MODIFY(SELF, stun, -1)
        PASS()
    """
    return [
        # Check if stun > 0
        OP_SELF,
        OP_GET, ATTR_STUN,
        OP_PUSH, 0,
        OP_GT,
        OP_IF, 3, 10,            # if true, jump +3; else jump +10

        # True branch: reduce stun and pass
        OP_SELF,
        OP_PUSH, -100,
        OP_MODIFY, ATTR_STUN,
        OP_PASS,

        OP_END,
    ]


def build_death_check_effect() -> list:
    """Effect: ON_ATTRIBUTE_CHANGE(health), if health < 1, lose.

    IF(GET(SELF, health) < 1):
        LOSE(SELF)
    """
    return [
        OP_SELF,
        OP_GET, ATTR_HEALTH,
        OP_PUSH, 100,            # 1.0
        OP_LT,                   # health < 1?
        OP_IF, 3, 5,             # if true, jump +3; else jump +5

        # True branch: lose
        OP_SELF,
        OP_LOSE,

        OP_END,
    ]


# === PREDEFINED HEROES ===

def create_fighter() -> Entity:
    """Create a Fighter hero.

    Fighter: High health, strength-based damage, no mana.
    - Health: 100, Max Health: 100
    - Strength: 10, Defense: 5
    - Abilities:
        1. Basic Attack: Deal strength damage
        2. Power Strike: Deal strength + d6 damage
        3. Defend: +3 defense (temporary)
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
            "script": build_basic_attack_script(),
        },
        {
            "name": "Power Strike",
            "tags": [TAG_ATTACK, TAG_PHYSICAL],
            "script": build_power_strike_script(),
        },
        {
            "name": "Defend",
            "tags": [],
            "script": build_defend_script(),
        },
    ]

    effects = [
        # Death check on health change
        {
            "trigger": TRIGGER_ON_ATTRIBUTE_CHANGE,
            "trigger_param": ATTR_HEALTH,
            "script": build_death_check_effect(),
        },
        # Burn damage on turn start
        {
            "trigger": TRIGGER_ON_TURN_START,
            "trigger_param": -1,
            "script": build_burn_effect(),
        },
        # Stun check on action phase
        {
            "trigger": TRIGGER_ON_ACTION_PHASE_START,
            "trigger_param": -1,
            "script": build_stun_effect(),
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
            "script": build_basic_attack_script(),
        },
        {
            "name": "Fireball",
            "tags": [TAG_SPELL, TAG_FIRE],
            "script": build_fireball_script(),
        },
        {
            "name": "Heal",
            "tags": [TAG_SPELL, TAG_HEAL],
            "script": build_heal_script(),
        },
        {
            "name": "Ice Bolt",
            "tags": [TAG_SPELL],
            "script": build_ice_bolt_script(),
        },
    ]

    effects = [
        # Death check
        {
            "trigger": TRIGGER_ON_ATTRIBUTE_CHANGE,
            "trigger_param": ATTR_HEALTH,
            "script": build_death_check_effect(),
        },
        # Mana regen
        {
            "trigger": TRIGGER_ON_TURN_START,
            "trigger_param": -1,
            "script": build_mana_regen_effect(),
        },
        # Burn damage
        {
            "trigger": TRIGGER_ON_TURN_START,
            "trigger_param": -1,
            "script": build_burn_effect(),
        },
        # Stun check
        {
            "trigger": TRIGGER_ON_ACTION_PHASE_START,
            "trigger_param": -1,
            "script": build_stun_effect(),
        },
    ]

    return create_entity(attributes, abilities, effects)


# Hero name mappings for TUI
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
