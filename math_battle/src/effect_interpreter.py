"""Effect Ops Interpreter for Math Battle.

This module implements a JAX-compatible interpreter for Effect Ops programs.
Uses an efficient ip-dispatch loop with lax.while_loop and lax.switch.
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
    OP_IF_GT, OP_IF_LT, OP_IF_EQ, OP_NOOP, OP_JUMP,
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
    num_nodes: Array,
    game_state: GameState,
    executor_idx: Array,
    rng: Array,
) -> Tuple[float, Array]:
    """Evaluate a ValueSpec expression tree.

    Uses bottom-up evaluation respecting num_nodes, and only splits RNG for ROLL nodes.
    """
    values = jnp.zeros(MAX_VALUE_DEPTH, dtype=jnp.float32)

    def eval_node(i, carry):
        vals, r = carry
        # Evaluate from end to beginning (bottom-up)
        node_idx = MAX_VALUE_DEPTH - 1 - i

        # Skip if beyond num_nodes (but still iterate for static shape)
        in_bounds = node_idx < num_nodes

        ntype = v_types[node_idx]
        p1 = v_param1[node_idx]
        p2 = v_param2[node_idx]

        # === Leaf nodes ===
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

        # ROLL - only split RNG if this is actually a ROLL node
        is_roll = jnp.logical_and(ntype == VALUE_ROLL, in_bounds)
        # Always split for consistent shape, but use original key if not rolling
        r_split = jax.random.split(r)
        r_new = jnp.where(is_roll, r_split[0], r)
        roll_key = jnp.where(is_roll, r_split[1], r)
        r = r_new
        sides = jnp.maximum(1, p1.astype(jnp.int32))
        roll_val = jax.random.randint(roll_key, (), 1, sides + 1).astype(jnp.float32)

        # === Compound ops - get child values ===
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

        # Only update if in bounds
        new_vals = jnp.where(in_bounds, vals.at[node_idx].set(result), vals)
        return new_vals, r

    values, rng = lax.fori_loop(0, MAX_VALUE_DEPTH, eval_node, (values, rng))
    return values[0], rng


def execute_program(
    program: Program,
    game_state: GameState,
    executor_idx: int,
    rng: Array,
) -> Tuple[GameState, bool, int, Array]:
    """Execute an Effect Ops program using efficient ip-dispatch loop.

    Returns: (new_state, passed, winner, new_rng)
    """
    executor_idx = jnp.array(executor_idx, dtype=jnp.int32)

    # Carry: (state, done, passed, winner, rng, ip, steps)
    # steps is a safety counter to ensure termination

    def cond_fn(carry):
        state, done, passed, winner, curr_rng, ip, steps = carry
        # Continue while not done, ip < num_ops, and steps < MAX_OPS
        return jnp.logical_and(
            jnp.logical_not(done),
            jnp.logical_and(ip < program.num_ops, steps < MAX_OPS)
        )

    def body_fn(carry):
        state, done, passed, winner, curr_rng, ip, steps = carry

        # Fetch current op
        safe_ip = jnp.clip(ip, 0, MAX_OPS - 1)
        op_type = program.op_types[safe_ip]
        target_raw = program.targets[safe_ip]
        attr_id = program.attr_ids[safe_ip]
        then_count = program.if_then_count[safe_ip]

        target = resolve_target(target_raw, executor_idx)

        # Get value specs
        v_types = program.value_types[safe_ip]
        v_param1 = program.value_param1[safe_ip]
        v_param2 = program.value_param2[safe_ip]
        v_num_nodes = program.value_num_nodes[safe_ip]
        v2_types = program.value2_types[safe_ip]
        v2_param1 = program.value2_param1[safe_ip]
        v2_param2 = program.value2_param2[safe_ip]
        v2_num_nodes = program.value2_num_nodes[safe_ip]

        # Determine which ops need value evaluation
        needs_values = jnp.isin(op_type, jnp.array([
            OP_SET_ATTR, OP_ADD_ATTR, OP_IF_GT, OP_IF_LT, OP_IF_EQ
        ]))

        # Evaluate values only if needed (but always compute for JIT shape consistency)
        val1, curr_rng = lax.cond(
            needs_values,
            lambda args: eval_value_spec(args[0], args[1], args[2], args[3], args[4], args[5], args[6]),
            lambda args: (0.0, args[6]),
            (v_types, v_param1, v_param2, v_num_nodes, state, executor_idx, curr_rng)
        )
        val2, curr_rng = lax.cond(
            needs_values,
            lambda args: eval_value_spec(args[0], args[1], args[2], args[3], args[4], args[5], args[6]),
            lambda args: (0.0, args[6]),
            (v2_types, v2_param1, v2_param2, v2_num_nodes, state, executor_idx, curr_rng)
        )

        # === Handle each op type ===

        # Default next_ip (advance by 1)
        next_ip = ip + 1

        # OP_END: terminate
        done = jnp.where(op_type == OP_END, True, done)

        # OP_PASS: terminate with passed=True
        done = jnp.where(op_type == OP_PASS, True, done)
        passed = jnp.where(op_type == OP_PASS, True, passed)

        # OP_WIN: target wins
        done = jnp.where(op_type == OP_WIN, True, done)
        winner = jnp.where(op_type == OP_WIN, target, winner)

        # OP_LOSE: target loses (opponent wins)
        done = jnp.where(op_type == OP_LOSE, True, done)
        winner = jnp.where(op_type == OP_LOSE, 1 - target, winner)

        # OP_SET_ATTR: set attribute to val1
        entity = get_entity_by_idx(state, target)
        safe_attr = jnp.clip(attr_id, 0, MAX_ATTRIBUTES - 1)
        old_val = entity.attributes[safe_attr]
        new_val_set = val1
        delta_set = new_val_set - old_val

        new_attrs_set = entity.attributes.at[safe_attr].set(new_val_set)
        entity_set = entity._replace(attributes=new_attrs_set)
        state_set = set_entity_by_idx(state, target, entity_set)

        is_set = op_type == OP_SET_ATTR
        state = tree_util.tree_map(
            lambda old, new: jnp.where(is_set, new, old),
            state, state_set
        )

        # OP_ADD_ATTR: add val1 to attribute
        entity = get_entity_by_idx(state, target)
        old_val_add = entity.attributes[safe_attr]
        new_val_add = old_val_add + val1
        delta_add = val1

        new_attrs_add = entity.attributes.at[safe_attr].set(new_val_add)
        entity_add = entity._replace(attributes=new_attrs_add)
        state_add = set_entity_by_idx(state, target, entity_add)

        is_add = op_type == OP_ADD_ATTR
        state = tree_util.tree_map(
            lambda old, new: jnp.where(is_add, new, old),
            state, state_add
        )

        # Queue attribute change for SET/ADD (with old/new/delta)
        is_attr_change = jnp.logical_or(is_set, is_add)
        q_idx = state.queue_count
        safe_q = jnp.clip(q_idx, 0, MAX_QUEUE - 1)
        can_queue = jnp.logical_and(is_attr_change, q_idx < MAX_QUEUE)

        # Compute old/new/delta for queueing
        queue_old = jnp.where(is_set, old_val, old_val_add)
        queue_new = jnp.where(is_set, new_val_set, new_val_add)
        queue_delta = jnp.where(is_set, delta_set, delta_add)

        new_queue_entry = jnp.array([target, safe_attr], dtype=jnp.int32)
        new_queue_values_entry = jnp.array([queue_old, queue_new, queue_delta], dtype=jnp.float32)

        new_queue = state.queue.at[safe_q].set(new_queue_entry)
        new_queue_values = state.queue_values.at[safe_q].set(new_queue_values_entry)
        new_q_count = jnp.where(can_queue, q_idx + 1, q_idx)

        state = state._replace(
            queue=jnp.where(can_queue, new_queue, state.queue),
            queue_values=jnp.where(can_queue, new_queue_values, state.queue_values),
            queue_count=new_q_count
        )

        # OP_IF_*: conditional branching
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

        # IF semantics:
        # - condition true: ip = ip + 1 (execute then branch)
        # - condition false: ip = ip + 1 + then_count (skip then+JUMP, land on else)
        if_ip = jnp.where(condition, ip + 1, ip + 1 + then_count)
        next_ip = jnp.where(is_any_if, if_ip, next_ip)

        # OP_JUMP: unconditional jump (skip_count in attr_id)
        is_jump = op_type == OP_JUMP
        jump_ip = ip + 1 + attr_id  # attr_id stores skip count
        next_ip = jnp.where(is_jump, jump_ip, next_ip)

        # OP_NOOP: just advance
        # (already handled by default next_ip = ip + 1)

        return (state, done, passed, winner, curr_rng, next_ip, steps + 1)

    init_carry = (
        game_state,
        jnp.array(False),
        jnp.array(False),
        jnp.array(-1, dtype=jnp.int32),
        rng,
        jnp.array(0, dtype=jnp.int32),  # ip
        jnp.array(0, dtype=jnp.int32),  # steps
    )

    final_carry = lax.while_loop(cond_fn, body_fn, init_carry)
    final_state, done, passed, winner, final_rng, _, _ = final_carry

    return final_state, passed, winner, final_rng


# Alias for backward compatibility
execute_script = execute_program
