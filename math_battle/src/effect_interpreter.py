"""Effect Ops Interpreter for Math Battle.

This module implements a JAX-compatible interpreter for Effect Ops programs.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import Array, lax
import jax.tree_util as tree_util

from .effect_ops import (
    Program, MAX_OPS, MAX_VALUE_DEPTH,
    TARGET_SELF, TARGET_OPPONENT,
    VALUE_CONST, VALUE_ATTR, VALUE_CTX, VALUE_ROLL,
    VALUE_ADD, VALUE_SUB, VALUE_MUL, VALUE_MIN, VALUE_MAX, VALUE_ABS, VALUE_NEG,
    OP_END, OP_PASS, OP_WIN, OP_LOSE, OP_SET_ATTR, OP_ADD_ATTR,
    OP_IF_GT, OP_IF_LT, OP_IF_EQ, OP_NOOP,
    CTX_SIZE,
)
from .game_state import GameState, Entity, MAX_ATTRIBUTES, MAX_QUEUE


def get_entity_by_idx(state: GameState, idx: Array) -> Entity:
    """Get entity by index functionally (0 = player, 1 = opponent)."""
    idx = jnp.asarray(idx)
    return tree_util.tree_map(
        lambda p, o: jnp.where(idx == 0, p, o),
        state.player,
        state.opponent
    )


def set_entity_by_idx(state: GameState, idx: Array, entity: Entity) -> GameState:
    """Set entity by index functionally."""
    idx = jnp.asarray(idx)
    new_player = tree_util.tree_map(
        lambda old, new: jnp.where(idx == 0, new, old),
        state.player,
        entity
    )
    new_opponent = tree_util.tree_map(
        lambda old, new: jnp.where(idx == 1, new, old),
        state.opponent,
        entity
    )
    return state._replace(player=new_player, opponent=new_opponent)


def resolve_target(target: Array, executor_idx: Array) -> Array:
    """Resolve TARGET_SELF/TARGET_OPPONENT to actual entity index."""
    return jnp.where(target == TARGET_SELF, executor_idx, 1 - executor_idx)


def eval_value_spec(
    v_types: Array,
    v_param1: Array,
    v_param2: Array,
    game_state: GameState,
    executor_idx: Array,
    rng: Array,
) -> Tuple[float, Array]:
    """Evaluate a ValueSpec expression tree.

    Uses bottom-up evaluation: we evaluate all nodes from the end backwards,
    storing results in a values array. The root (index 0) contains the final result.
    """
    # Array to store computed values for each node
    values = jnp.zeros(MAX_VALUE_DEPTH, dtype=jnp.float32)

    # Evaluate nodes from end to beginning (bottom-up)
    def eval_node(i, carry):
        vals, r = carry
        # Evaluate node at index (MAX_VALUE_DEPTH - 1 - i)
        node_idx = MAX_VALUE_DEPTH - 1 - i

        ntype = v_types[node_idx]
        p1 = v_param1[node_idx]
        p2 = v_param2[node_idx]

        # CONST
        const_val = p1

        # ATTR
        attr_target = resolve_target(p1.astype(jnp.int32), executor_idx)
        attr_entity = get_entity_by_idx(game_state, attr_target)
        attr_idx = jnp.clip(p2.astype(jnp.int32), 0, MAX_ATTRIBUTES - 1)
        attr_val = attr_entity.attributes[attr_idx]

        # CTX
        ctx_key = jnp.clip(p1.astype(jnp.int32), 0, CTX_SIZE - 1)
        ctx_val = game_state.context[ctx_key]

        # ROLL - use deterministic key based on node index
        r, roll_key = jax.random.split(r)
        sides = jnp.maximum(1, p1.astype(jnp.int32))
        roll_val = jax.random.randint(roll_key, (), 1, sides + 1).astype(jnp.float32)

        # For compound ops, get child values from vals array
        left_idx = jnp.clip(p1.astype(jnp.int32), 0, MAX_VALUE_DEPTH - 1)
        right_idx = jnp.clip(p2.astype(jnp.int32), 0, MAX_VALUE_DEPTH - 1)
        left_val = vals[left_idx]
        right_val = vals[right_idx]

        # Binary ops
        add_val = left_val + right_val
        sub_val = left_val - right_val
        mul_val = left_val * right_val
        min_val = jnp.minimum(left_val, right_val)
        max_val = jnp.maximum(left_val, right_val)

        # Unary ops
        abs_val = jnp.abs(left_val)
        neg_val = -left_val

        # Select based on type
        result = jnp.where(ntype == VALUE_CONST, const_val,
                 jnp.where(ntype == VALUE_ATTR, attr_val,
                 jnp.where(ntype == VALUE_CTX, ctx_val,
                 jnp.where(ntype == VALUE_ROLL, roll_val,
                 jnp.where(ntype == VALUE_ADD, add_val,
                 jnp.where(ntype == VALUE_SUB, sub_val,
                 jnp.where(ntype == VALUE_MUL, mul_val,
                 jnp.where(ntype == VALUE_MIN, min_val,
                 jnp.where(ntype == VALUE_MAX, max_val,
                 jnp.where(ntype == VALUE_ABS, abs_val,
                 jnp.where(ntype == VALUE_NEG, neg_val,
                 0.0)))))))))))

        new_vals = vals.at[node_idx].set(result)
        return new_vals, r

    # Evaluate all nodes
    values, rng = lax.fori_loop(0, MAX_VALUE_DEPTH, eval_node, (values, rng))

    # Root is at index 0
    return values[0], rng


def execute_program(
    program: Program,
    game_state: GameState,
    executor_idx: int,
    rng: Array,
) -> Tuple[GameState, bool, int, Array]:
    """Execute an Effect Ops program."""
    executor_idx = jnp.array(executor_idx, dtype=jnp.int32)

    def process_op(carry, op_idx):
        state, done, passed, winner, curr_rng, ip = carry

        # Only process if ip matches current index and not done
        should_process = jnp.logical_and(ip == op_idx, jnp.logical_not(done))

        # Fetch op data
        safe_idx = jnp.clip(op_idx, 0, MAX_OPS - 1)
        op_type = program.op_types[safe_idx]
        target_raw = program.targets[safe_idx]
        attr_id = program.attr_ids[safe_idx]
        then_count = program.if_then_count[safe_idx]
        else_count = program.if_else_count[safe_idx]

        target = resolve_target(target_raw, executor_idx)

        # Get value specs
        v_types = program.value_types[safe_idx]
        v_param1 = program.value_param1[safe_idx]
        v_param2 = program.value_param2[safe_idx]
        v2_types = program.value2_types[safe_idx]
        v2_param1 = program.value2_param1[safe_idx]
        v2_param2 = program.value2_param2[safe_idx]

        # Evaluate values
        val1, curr_rng = eval_value_spec(v_types, v_param1, v_param2, state, executor_idx, curr_rng)
        val2, curr_rng = eval_value_spec(v2_types, v2_param1, v2_param2, state, executor_idx, curr_rng)

        # OP_END
        new_done_end = jnp.where(
            jnp.logical_and(should_process, op_type == OP_END),
            True, done
        )

        # OP_PASS
        new_done_pass = jnp.where(
            jnp.logical_and(should_process, op_type == OP_PASS),
            True, new_done_end
        )
        new_passed = jnp.where(
            jnp.logical_and(should_process, op_type == OP_PASS),
            True, passed
        )

        # OP_WIN
        new_done_win = jnp.where(
            jnp.logical_and(should_process, op_type == OP_WIN),
            True, new_done_pass
        )
        new_winner_win = jnp.where(
            jnp.logical_and(should_process, op_type == OP_WIN),
            target, winner
        )

        # OP_LOSE
        new_done_lose = jnp.where(
            jnp.logical_and(should_process, op_type == OP_LOSE),
            True, new_done_win
        )
        new_winner_lose = jnp.where(
            jnp.logical_and(should_process, op_type == OP_LOSE),
            1 - target, new_winner_win
        )

        # OP_SET_ATTR
        entity = get_entity_by_idx(state, target)
        safe_attr = jnp.clip(attr_id, 0, MAX_ATTRIBUTES - 1)
        new_attrs_set = entity.attributes.at[safe_attr].set(val1)
        entity_set = entity._replace(attributes=new_attrs_set)
        state_set = set_entity_by_idx(state, target, entity_set)

        should_set = jnp.logical_and(should_process, op_type == OP_SET_ATTR)
        state = tree_util.tree_map(
            lambda old, new: jnp.where(should_set, new, old),
            state, state_set
        )

        # OP_ADD_ATTR
        entity = get_entity_by_idx(state, target)
        current_val = entity.attributes[safe_attr]
        new_val = current_val + val1
        new_attrs_add = entity.attributes.at[safe_attr].set(new_val)
        entity_add = entity._replace(attributes=new_attrs_add)
        state_add = set_entity_by_idx(state, target, entity_add)

        should_add = jnp.logical_and(should_process, op_type == OP_ADD_ATTR)
        state = tree_util.tree_map(
            lambda old, new: jnp.where(should_add, new, old),
            state, state_add
        )

        # Queue attribute change for SET/ADD
        should_queue = jnp.logical_or(should_set, should_add)
        q_idx = state.queue_count
        safe_q = jnp.clip(q_idx, 0, MAX_QUEUE - 1)
        new_queue_entry = jnp.array([target, safe_attr], dtype=jnp.int32)
        new_queue = state.queue.at[safe_q].set(new_queue_entry)
        new_q_count = jnp.where(should_queue, jnp.minimum(q_idx + 1, MAX_QUEUE), q_idx)
        state = state._replace(
            queue=jnp.where(should_queue, new_queue, state.queue),
            queue_count=new_q_count
        )

        # IF ops
        is_if_gt = op_type == OP_IF_GT
        is_if_lt = op_type == OP_IF_LT
        is_if_eq = op_type == OP_IF_EQ
        is_any_if = jnp.logical_or(is_if_gt, jnp.logical_or(is_if_lt, is_if_eq))

        cond_gt = val1 > val2
        cond_lt = val1 < val2
        cond_eq = val1 == val2

        condition = jnp.where(is_if_gt, cond_gt,
                   jnp.where(is_if_lt, cond_lt,
                   jnp.where(is_if_eq, cond_eq, False)))

        if_ip = jnp.where(condition, ip + 1, ip + 1 + then_count)

        next_ip = jnp.where(
            jnp.logical_and(should_process, is_any_if),
            if_ip,
            jnp.where(should_process, ip + 1, ip)
        )

        return (state, new_done_lose, new_passed, new_winner_lose, curr_rng, next_ip), None

    init_carry = (
        game_state,
        jnp.array(False),
        jnp.array(False),
        jnp.array(-1, dtype=jnp.int32),
        rng,
        jnp.array(0, dtype=jnp.int32)
    )

    final_carry, _ = lax.scan(process_op, init_carry, jnp.arange(MAX_OPS))
    final_state, done, passed, winner, final_rng, _ = final_carry

    return final_state, passed, winner, final_rng


execute_script = execute_program
