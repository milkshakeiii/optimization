"""DSL Interpreter for Math Battle using a JAX-compatible stack machine.

This module implements the Domain Specific Language (DSL) for scripts
using a functional stack-based virtual machine suitable for JIT compilation.
"""

from typing import Tuple, NamedTuple, Callable, List
import jax
import jax.numpy as jnp
from jax import lax
import jax.tree_util as tree_util
from jax import Array

from .game_state import (
    GameState, Entity, MAX_ATTRIBUTES, MAX_SCRIPT_LEN, MAX_EFFECTS,
    TRIGGER_ON_ATTRIBUTE_CHANGE, MAX_QUEUE,
    OP_NOOP, OP_SELF, OP_OPPONENT, OP_CONTEXT, OP_GET, OP_SET, OP_MODIFY,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_ABS, OP_MIN, OP_MAX, OP_ROLL,
    OP_IF, OP_SEQ, OP_EQ, OP_GT, OP_LT, OP_AND, OP_OR, OP_NOT,
    OP_WIN, OP_LOSE, OP_PASS, OP_PUSH, OP_END,
    ATTR_HEALTH,
)

# Stack machine constants
STACK_SIZE = 32


class VMState(NamedTuple):
    """Immutable state for the virtual machine."""
    stack: Array       # (STACK_SIZE,) float32
    sp: Array          # scalar int32
    ip: Array          # scalar int32
    terminated: Array  # scalar bool (or int32 0/1)
    passed: Array      # scalar bool
    winner: Array      # scalar int32
    rng: Array         # (2,) uint32
    game_state: GameState


def get_entity_func(state: GameState, idx: Array) -> Entity:
    """Get entity by index functionally."""
    # We use tree_map to select between player and opponent for every field
    return tree_util.tree_map(
        lambda p, o: jnp.where(idx == 0, p, o),
        state.player,
        state.opponent
    )


def set_entity_func(state: GameState, idx: Array, new_entity: Entity) -> GameState:
    """Set entity by index functionally."""
    new_player = tree_util.tree_map(
        lambda old, new: jnp.where(idx == 0, new, old),
        state.player,
        new_entity
    )
    new_opponent = tree_util.tree_map(
        lambda old, new: jnp.where(idx == 1, new, old),
        state.opponent,
        new_entity
    )
    return state._replace(player=new_player, opponent=new_opponent)


def push(vm: VMState, value: float) -> VMState:
    """Push value onto stack."""
    # Ensure sp is within bounds (wrap or clamp, though logic should prevent overflow)
    # Using .at[].set for functional update
    safe_sp = jnp.clip(vm.sp, 0, STACK_SIZE - 1)
    new_stack = vm.stack.at[safe_sp].set(value)
    return vm._replace(stack=new_stack, sp=vm.sp + 1)


def pop(vm: VMState) -> Tuple[VMState, float]:
    """Pop value from stack."""
    new_sp = vm.sp - 1
    safe_sp = jnp.clip(new_sp, 0, STACK_SIZE - 1)
    val = vm.stack[safe_sp]
    return vm._replace(sp=new_sp), val


def execute_script(
    script: Array,
    game_state: GameState,
    executor_idx: int,
    rng: Array
) -> Tuple[GameState, bool, int, Array]:
    """Execute a DSL script and return updated state.

    Args:
        script: Array of opcodes (MAX_SCRIPT_LEN,)
        game_state: Current game state
        executor_idx: Index of the entity executing the script (0 or 1)
        rng: Random key

    Returns:
        Tuple of (new_state, passed, winner, new_rng)
    """
    
    # Initial VM State
    init_vm = VMState(
        stack=jnp.zeros(STACK_SIZE, dtype=jnp.float32),
        sp=jnp.array(0, dtype=jnp.int32),
        ip=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False, dtype=bool),
        passed=jnp.array(False, dtype=bool),
        winner=jnp.array(-1, dtype=jnp.int32),
        rng=rng,
        game_state=game_state
    )

    # Core Step Function
    def vm_step(vm: VMState, _):
        # If terminated, return identity
        def _step_logic(vm: VMState):
            # Fetch instruction
            # Use clip to ensure we don't read out of bounds if IP is invalid
            safe_ip = jnp.clip(vm.ip, 0, MAX_SCRIPT_LEN - 1)
            opcode = script[safe_ip]
            
            # Fetch arguments (optimistically)
            arg1_idx = jnp.clip(vm.ip + 1, 0, MAX_SCRIPT_LEN - 1)
            arg2_idx = jnp.clip(vm.ip + 2, 0, MAX_SCRIPT_LEN - 1)
            arg1 = script[arg1_idx]
            arg2 = script[arg2_idx]
            
            # Dispatch
            # We define a list of branch functions. 
            # Each must take (vm, arg1, arg2) and return new_vm.
            
            branches = [
                _op_noop,      # 0
                lambda v, a, b: _op_self(v, executor_idx), # 1
                lambda v, a, b: _op_opponent(v, executor_idx), # 2
                _op_context,   # 3
                _op_get,       # 4
                _op_set,       # 5
                _op_modify,    # 6
                _op_add,       # 7
                _op_sub,       # 8
                _op_mul,       # 9
                _op_div,       # 10
                _op_abs,       # 11
                _op_min,       # 12
                _op_max,       # 13
                _op_roll,      # 14
                _op_if,        # 15
                _op_seq,       # 16
                _op_eq,        # 17
                _op_gt,        # 18
                _op_lt,        # 19
                _op_and,       # 20
                _op_or,        # 21
                _op_not,       # 22
                _op_win,       # 23
                _op_lose,      # 24
                _op_pass,      # 25
                _op_push,      # 26
                _op_end,       # 27
            ]
            
            # Pad branches to safe length if opcode > 27 (shouldn't happen with correct constants)
            # lax.switch requires valid index bounds or it's undefined/clamped? 
            # Best to ensure opcode is within 0-27.
            safe_opcode = jnp.clip(opcode, 0, len(branches) - 1)
            
            return lax.switch(safe_opcode, branches, vm, arg1, arg2)

        new_vm = lax.cond(
            vm.terminated,
            lambda v: v, # Identity if terminated
            _step_logic,
            vm
        )
        return new_vm, None

    # Run loop
    final_vm, _ = lax.scan(vm_step, init_vm, None, length=MAX_SCRIPT_LEN)

    return final_vm.game_state, final_vm.passed, final_vm.winner, final_vm.rng


# --- Opcode Implementations ---
# All must have signature (vm: VMState, arg1: int, arg2: int) -> VMState

def _op_noop(vm: VMState, arg1, arg2) -> VMState:
    return vm._replace(ip=vm.ip + 1)

def _op_end(vm: VMState, arg1, arg2) -> VMState:
    return vm._replace(terminated=jnp.array(True))

def _op_push(vm: VMState, arg1, arg2) -> VMState:
    # arg1 is the value * 100 (int)
    val = arg1 / 100.0
    vm = push(vm, val)
    return vm._replace(ip=vm.ip + 2)

def _op_self(vm: VMState, exec_idx: int) -> VMState:
    vm = push(vm, exec_idx.astype(jnp.float32))
    return vm._replace(ip=vm.ip + 1)

def _op_opponent(vm: VMState, exec_idx: int) -> VMState:
    vm = push(vm, (1 - exec_idx).astype(jnp.float32))
    return vm._replace(ip=vm.ip + 1)

def _op_context(vm: VMState, arg1, arg2) -> VMState:
    # arg1 is index into context
    # Context is size 8
    safe_idx = jnp.clip(arg1, 0, 7)
    val = vm.game_state.context[safe_idx]
    vm = push(vm, val)
    return vm._replace(ip=vm.ip + 2)

def _op_get(vm: VMState, arg1, arg2) -> VMState:
    # arg1 is attr index
    vm, target_idx = pop(vm)
    entity = get_entity_func(vm.game_state, jnp.int32(target_idx))
    safe_attr_idx = jnp.clip(arg1, 0, MAX_ATTRIBUTES - 1)
    val = entity.attributes[safe_attr_idx]
    vm = push(vm, val)
    return vm._replace(ip=vm.ip + 2)


def _op_set(vm: VMState, arg1, arg2) -> VMState:
    # arg1 is attr index
    vm, val = pop(vm)
    vm, target_idx = pop(vm)
    target_idx = jnp.int32(target_idx)
    
    entity = get_entity_func(vm.game_state, target_idx)
    safe_attr_idx = jnp.clip(arg1, 0, MAX_ATTRIBUTES - 1)
    
    new_attrs = entity.attributes.at[safe_attr_idx].set(val)
    new_entity = entity._replace(attributes=new_attrs)
    
    new_gs = set_entity_func(vm.game_state, target_idx, new_entity)
    
    # Add to queue
    q_idx = new_gs.queue_count
    safe_q_idx = jnp.clip(q_idx, 0, MAX_QUEUE - 1)
    new_queue = new_gs.queue.at[safe_q_idx].set(jnp.array([target_idx, safe_attr_idx], dtype=jnp.int32))
    new_gs = new_gs._replace(
        queue=new_queue,
        queue_count=jnp.minimum(q_idx + 1, MAX_QUEUE)
    )
    
    return vm._replace(game_state=new_gs, ip=vm.ip + 2)

def _op_modify(vm: VMState, arg1, arg2) -> VMState:
    # arg1 is attr index
    vm, delta = pop(vm)
    vm, target_idx = pop(vm)
    target_idx = jnp.int32(target_idx)
    
    entity = get_entity_func(vm.game_state, target_idx)
    safe_attr_idx = jnp.clip(arg1, 0, MAX_ATTRIBUTES - 1)
    
    current_val = entity.attributes[safe_attr_idx]
    new_val = current_val + delta
    
    new_attrs = entity.attributes.at[safe_attr_idx].set(new_val)
    new_entity = entity._replace(attributes=new_attrs)
    
    new_gs = set_entity_func(vm.game_state, target_idx, new_entity)
    
    # Add to queue
    q_idx = new_gs.queue_count
    safe_q_idx = jnp.clip(q_idx, 0, MAX_QUEUE - 1)
    new_queue = new_gs.queue.at[safe_q_idx].set(jnp.array([target_idx, safe_attr_idx], dtype=jnp.int32))
    new_gs = new_gs._replace(
        queue=new_queue,
        queue_count=jnp.minimum(q_idx + 1, MAX_QUEUE)
    )
    
    return vm._replace(game_state=new_gs, ip=vm.ip + 2)

def _op_add(vm: VMState, arg1, arg2) -> VMState:
    vm, a = pop(vm)
    vm, b = pop(vm) # Stack is LIFO, so pop order: b then a? Code says: b, a = pop(), pop(). Stack: [..., a, b]. pop() -> b. pop() -> a.
    # But DSL says: "b, a = pop(), pop()". So 'b' is top.
    vm = push(vm, a + b)
    return vm._replace(ip=vm.ip + 1)

def _op_sub(vm: VMState, arg1, arg2) -> VMState:
    vm, b = pop(vm)
    vm, a = pop(vm)
    vm = push(vm, a - b)
    return vm._replace(ip=vm.ip + 1)

def _op_mul(vm: VMState, arg1, arg2) -> VMState:
    vm, b = pop(vm)
    vm, a = pop(vm)
    vm = push(vm, a * b)
    return vm._replace(ip=vm.ip + 1)

def _op_div(vm: VMState, arg1, arg2) -> VMState:
    vm, b = pop(vm)
    vm, a = pop(vm)
    # Avoid div by zero
    res = jnp.where(b != 0, a / b, 0.0)
    vm = push(vm, res)
    return vm._replace(ip=vm.ip + 1)

def _op_abs(vm: VMState, arg1, arg2) -> VMState:
    vm, a = pop(vm)
    vm = push(vm, jnp.abs(a))
    return vm._replace(ip=vm.ip + 1)

def _op_min(vm: VMState, arg1, arg2) -> VMState:
    vm, b = pop(vm)
    vm, a = pop(vm)
    vm = push(vm, jnp.minimum(a, b))
    return vm._replace(ip=vm.ip + 1)

def _op_max(vm: VMState, arg1, arg2) -> VMState:
    vm, b = pop(vm)
    vm, a = pop(vm)
    vm = push(vm, jnp.maximum(a, b))
    return vm._replace(ip=vm.ip + 1)

def _op_roll(vm: VMState, arg1, arg2) -> VMState:
    vm, sides = pop(vm)
    sides_int = jnp.maximum(1, sides.astype(jnp.int32))
    
    # Split RNG
    rng, subkey = jax.random.split(vm.rng)
    roll = jax.random.randint(subkey, (), 1, sides_int + 1)
    
    vm = push(vm, roll.astype(jnp.float32))
    return vm._replace(rng=rng, ip=vm.ip + 1)

def _op_eq(vm: VMState, arg1, arg2) -> VMState:
    vm, b = pop(vm)
    vm, a = pop(vm)
    res = jnp.where(a == b, 1.0, 0.0)
    vm = push(vm, res)
    return vm._replace(ip=vm.ip + 1)

def _op_gt(vm: VMState, arg1, arg2) -> VMState:
    vm, b = pop(vm)
    vm, a = pop(vm)
    res = jnp.where(a > b, 1.0, 0.0)
    vm = push(vm, res)
    return vm._replace(ip=vm.ip + 1)

def _op_lt(vm: VMState, arg1, arg2) -> VMState:
    vm, b = pop(vm)
    vm, a = pop(vm)
    res = jnp.where(a < b, 1.0, 0.0)
    vm = push(vm, res)
    return vm._replace(ip=vm.ip + 1)

def _op_and(vm: VMState, arg1, arg2) -> VMState:
    vm, b = pop(vm)
    vm, a = pop(vm)
    res = jnp.where((a > 0) & (b > 0), 1.0, 0.0)
    vm = push(vm, res)
    return vm._replace(ip=vm.ip + 1)

def _op_or(vm: VMState, arg1, arg2) -> VMState:
    vm, b = pop(vm)
    vm, a = pop(vm)
    res = jnp.where((a > 0) | (b > 0), 1.0, 0.0)
    vm = push(vm, res)
    return vm._replace(ip=vm.ip + 1)

def _op_not(vm: VMState, arg1, arg2) -> VMState:
    vm, a = pop(vm)
    res = jnp.where(a <= 0, 1.0, 0.0)
    vm = push(vm, res)
    return vm._replace(ip=vm.ip + 1)

def _op_if(vm: VMState, arg1, arg2) -> VMState:
    vm, cond = pop(vm)
    # arg1 is offset if true, arg2 is offset if false
    offset = jnp.where(cond > 0, arg1, arg2)
    return vm._replace(ip=vm.ip + offset)

def _op_seq(vm: VMState, arg1, arg2) -> VMState:
    # Just a marker? implementation in python was ip+=1
    return vm._replace(ip=vm.ip + 1)

def _op_win(vm: VMState, arg1, arg2) -> VMState:
    vm, target = pop(vm)
    return vm._replace(
        winner=target.astype(jnp.int32),
        terminated=jnp.array(True)
    )

def _op_lose(vm: VMState, arg1, arg2) -> VMState:
    vm, target = pop(vm)
    # 1 - target
    return vm._replace(
        winner=(1 - target.astype(jnp.int32)),
        terminated=jnp.array(True)
    )

def _op_pass(vm: VMState, arg1, arg2) -> VMState:
    return vm._replace(
        passed=jnp.array(True),
        terminated=jnp.array(True),
        ip=vm.ip + 1
    )


# --- Compilation Helpers (unchanged logic, just ensuring they return lists) ---

def compile_noop() -> list:
    return [OP_NOOP]

def compile_push(value: float) -> list:
    return [OP_PUSH, int(value * 100)]

def compile_self() -> list:
    return [OP_SELF]

def compile_opponent() -> list:
    return [OP_OPPONENT]

def compile_get(attr_idx: int) -> list:
    return [OP_GET, attr_idx]

def compile_set(attr_idx: int) -> list:
    return [OP_SET, attr_idx]

def compile_modify(attr_idx: int) -> list:
    return [OP_MODIFY, attr_idx]

def compile_add() -> list:
    return [OP_ADD]

def compile_sub() -> list:
    return [OP_SUB]

def compile_mul() -> list:
    return [OP_MUL]

def compile_roll() -> list:
    return [OP_ROLL]

def compile_win() -> list:
    return [OP_WIN]

def compile_lose() -> list:
    return [OP_LOSE]

def compile_pass() -> list:
    return [OP_PASS]

def compile_end() -> list:
    return [OP_END]

def compile_lt() -> list:
    return [OP_LT]

def compile_gt() -> list:
    return [OP_GT]

def compile_if(true_offset: int, false_offset: int) -> list:
    return [OP_IF, true_offset, false_offset]

def pad_script(script: list, length: int = MAX_SCRIPT_LEN) -> list:
    result = script + [OP_END] * (length - len(script))
    return result[:length]

def script_to_array(script: list) -> Array:
    padded = pad_script(script)
    return jnp.array(padded, dtype=jnp.int32)

# Shim aliases for compatibility if needed
get_entity_jax = get_entity_func
set_entity_jax = set_entity_func
