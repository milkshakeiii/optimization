"""Effect Ops system for Math Battle.

This module implements the Effect Ops design, replacing the stack-based DSL
with a structured instruction set that is JAX/JIT-friendly.

Key concepts:
- ValueSpec: How to compute a numeric value (CONST, ATTR, CTX, ROLL, arithmetic)
- EffectOp: An operation that mutates state or controls flow
- Programs: Fixed-size arrays of ops with fixed-size ValueSpec arrays
"""

from typing import NamedTuple, List, Tuple, Optional
import jax.numpy as jnp
from jax import Array

# =============================================================================
# Constants
# =============================================================================

MAX_OPS = 16          # Max operations per program
MAX_VALUE_DEPTH = 8   # Max nodes in a ValueSpec expression tree

# Target constants
TARGET_SELF = 0
TARGET_OPPONENT = 1

# ValueSpec type constants
VALUE_CONST = 0
VALUE_ATTR = 1
VALUE_CTX = 2
VALUE_ROLL = 3
VALUE_ADD = 4
VALUE_SUB = 5
VALUE_MUL = 6
VALUE_MIN = 7
VALUE_MAX = 8
VALUE_ABS = 9
VALUE_NEG = 10

# EffectOp type constants
OP_END = 0
OP_PASS = 1
OP_WIN = 2
OP_LOSE = 3
OP_SET_ATTR = 4
OP_ADD_ATTR = 5
OP_IF_GT = 6
OP_IF_LT = 7
OP_IF_EQ = 8
OP_NOOP = 9

# Context keys
CTX_ABILITY_ID = 0
CTX_ABILITY_COST = 1
CTX_ATTR_DELTA = 2
CTX_ATTR_NEW = 3
CTX_ATTR_OLD = 4
CTX_ATTR_ID = 5
CTX_SIZE = 8


# =============================================================================
# ValueSpec Data Structure
# =============================================================================

class ValueSpec(NamedTuple):
    """A specification for computing a numeric value.

    Stored as arrays representing an expression tree. Node 0 is the root.
    For compound ops, param1/param2 store child indices.
    For leaf ops (CONST, ATTR, etc), param1/param2 store the actual values.
    """
    types: Array       # (MAX_VALUE_DEPTH,) int32
    param1: Array      # (MAX_VALUE_DEPTH,) float32
    param2: Array      # (MAX_VALUE_DEPTH,) float32
    num_nodes: Array   # scalar int32


def _make_value_spec(types_list, param1_list, param2_list):
    """Helper to create a ValueSpec from Python lists."""
    n = len(types_list)
    types = [0] * MAX_VALUE_DEPTH
    param1 = [0.0] * MAX_VALUE_DEPTH
    param2 = [0.0] * MAX_VALUE_DEPTH

    for i in range(min(n, MAX_VALUE_DEPTH)):
        types[i] = types_list[i]
        param1[i] = float(param1_list[i])
        param2[i] = float(param2_list[i])

    return ValueSpec(
        types=jnp.array(types, dtype=jnp.int32),
        param1=jnp.array(param1, dtype=jnp.float32),
        param2=jnp.array(param2, dtype=jnp.float32),
        num_nodes=jnp.array(n, dtype=jnp.int32),
    )


def create_const(value: float) -> ValueSpec:
    """Create a constant ValueSpec."""
    return _make_value_spec([VALUE_CONST], [value], [0])


def create_attr(target: int, attr_id: int) -> ValueSpec:
    """Create a ValueSpec that reads an attribute."""
    return _make_value_spec([VALUE_ATTR], [target], [attr_id])


def create_ctx(ctx_key: int) -> ValueSpec:
    """Create a ValueSpec that reads from context."""
    return _make_value_spec([VALUE_CTX], [ctx_key], [0])


def create_roll(sides: int) -> ValueSpec:
    """Create a ValueSpec for random roll(1, sides)."""
    return _make_value_spec([VALUE_ROLL], [sides], [0])


def _merge_trees(root_type, left: ValueSpec, right: ValueSpec) -> ValueSpec:
    """Merge two ValueSpec trees under a new binary root.

    Layout: [root, left_tree..., right_tree...]
    Root's param1 = left child index (1)
    Root's param2 = right child index (1 + left.num_nodes)
    """
    left_n = int(left.num_nodes)
    right_n = int(right.num_nodes)

    types_list = [root_type]
    param1_list = [1]  # left child at index 1
    param2_list = [1 + left_n]  # right child index

    # Copy left tree, adjusting indices
    for i in range(left_n):
        t = int(left.types[i])
        p1 = float(left.param1[i])
        p2 = float(left.param2[i])

        # If this is a compound op, adjust child indices
        if t in (VALUE_ADD, VALUE_SUB, VALUE_MUL, VALUE_MIN, VALUE_MAX):
            p1 = p1 + 1  # left child index shifted by 1
            p2 = p2 + 1  # right child index shifted by 1
        elif t in (VALUE_NEG, VALUE_ABS):
            p1 = p1 + 1  # operand index shifted by 1

        types_list.append(t)
        param1_list.append(p1)
        param2_list.append(p2)

    # Copy right tree, adjusting indices
    right_offset = 1 + left_n
    for i in range(right_n):
        t = int(right.types[i])
        p1 = float(right.param1[i])
        p2 = float(right.param2[i])

        # If this is a compound op, adjust child indices
        if t in (VALUE_ADD, VALUE_SUB, VALUE_MUL, VALUE_MIN, VALUE_MAX):
            p1 = p1 + right_offset
            p2 = p2 + right_offset
        elif t in (VALUE_NEG, VALUE_ABS):
            p1 = p1 + right_offset

        types_list.append(t)
        param1_list.append(p1)
        param2_list.append(p2)

    return _make_value_spec(types_list, param1_list, param2_list)


def _wrap_unary(root_type, operand: ValueSpec) -> ValueSpec:
    """Wrap a ValueSpec tree under a new unary root.

    Layout: [root, operand_tree...]
    Root's param1 = operand index (1)
    """
    op_n = int(operand.num_nodes)

    types_list = [root_type]
    param1_list = [1]  # operand at index 1
    param2_list = [0]

    # Copy operand tree, adjusting indices
    for i in range(op_n):
        t = int(operand.types[i])
        p1 = float(operand.param1[i])
        p2 = float(operand.param2[i])

        # If this is a compound op, adjust child indices
        if t in (VALUE_ADD, VALUE_SUB, VALUE_MUL, VALUE_MIN, VALUE_MAX):
            p1 = p1 + 1
            p2 = p2 + 1
        elif t in (VALUE_NEG, VALUE_ABS):
            p1 = p1 + 1

        types_list.append(t)
        param1_list.append(p1)
        param2_list.append(p2)

    return _make_value_spec(types_list, param1_list, param2_list)


# Convenience constructors for arithmetic
def vs_add(a: ValueSpec, b: ValueSpec) -> ValueSpec:
    return _merge_trees(VALUE_ADD, a, b)

def vs_sub(a: ValueSpec, b: ValueSpec) -> ValueSpec:
    return _merge_trees(VALUE_SUB, a, b)

def vs_mul(a: ValueSpec, b: ValueSpec) -> ValueSpec:
    return _merge_trees(VALUE_MUL, a, b)

def vs_min(a: ValueSpec, b: ValueSpec) -> ValueSpec:
    return _merge_trees(VALUE_MIN, a, b)

def vs_max(a: ValueSpec, b: ValueSpec) -> ValueSpec:
    return _merge_trees(VALUE_MAX, a, b)

def vs_abs(a: ValueSpec) -> ValueSpec:
    return _wrap_unary(VALUE_ABS, a)

def vs_neg(a: ValueSpec) -> ValueSpec:
    return _wrap_unary(VALUE_NEG, a)


# Shorthand constructors
def C(value: float) -> ValueSpec:
    """Shorthand for create_const."""
    return create_const(value)

def Attr(target: int, attr_id: int) -> ValueSpec:
    """Shorthand for create_attr."""
    return create_attr(target, attr_id)

def Ctx(key: int) -> ValueSpec:
    """Shorthand for create_ctx."""
    return create_ctx(key)

def Roll(sides: int) -> ValueSpec:
    """Shorthand for create_roll."""
    return create_roll(sides)


# =============================================================================
# Program Data Structure
# =============================================================================

class Program(NamedTuple):
    """A fixed-size program of effect operations."""
    op_types: Array           # (MAX_OPS,) int32
    targets: Array            # (MAX_OPS,) int32
    attr_ids: Array           # (MAX_OPS,) int32
    value_types: Array        # (MAX_OPS, MAX_VALUE_DEPTH) int32
    value_param1: Array       # (MAX_OPS, MAX_VALUE_DEPTH) float32
    value_param2: Array       # (MAX_OPS, MAX_VALUE_DEPTH) float32
    value_num_nodes: Array    # (MAX_OPS,) int32
    value2_types: Array       # (MAX_OPS, MAX_VALUE_DEPTH) int32
    value2_param1: Array      # (MAX_OPS, MAX_VALUE_DEPTH) float32
    value2_param2: Array      # (MAX_OPS, MAX_VALUE_DEPTH) float32
    value2_num_nodes: Array   # (MAX_OPS,) int32
    if_then_count: Array      # (MAX_OPS,) int32
    if_else_count: Array      # (MAX_OPS,) int32
    num_ops: Array            # scalar int32


def create_empty_program() -> Program:
    """Create an empty program (all END ops)."""
    return Program(
        op_types=jnp.zeros(MAX_OPS, dtype=jnp.int32),
        targets=jnp.zeros(MAX_OPS, dtype=jnp.int32),
        attr_ids=jnp.zeros(MAX_OPS, dtype=jnp.int32),
        value_types=jnp.zeros((MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.int32),
        value_param1=jnp.zeros((MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        value_param2=jnp.zeros((MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        value_num_nodes=jnp.zeros(MAX_OPS, dtype=jnp.int32),
        value2_types=jnp.zeros((MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.int32),
        value2_param1=jnp.zeros((MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        value2_param2=jnp.zeros((MAX_OPS, MAX_VALUE_DEPTH), dtype=jnp.float32),
        value2_num_nodes=jnp.zeros(MAX_OPS, dtype=jnp.int32),
        if_then_count=jnp.zeros(MAX_OPS, dtype=jnp.int32),
        if_else_count=jnp.zeros(MAX_OPS, dtype=jnp.int32),
        num_ops=jnp.array(0, dtype=jnp.int32),
    )


# =============================================================================
# Program Builder
# =============================================================================

class ProgramBuilder:
    """Helper class to build programs from Effect Ops at Python level."""

    def __init__(self):
        self.ops = []

    def end(self) -> 'ProgramBuilder':
        self.ops.append({
            'type': OP_END, 'target': 0, 'attr_id': 0,
            'value': create_const(0), 'value2': create_const(0),
            'then_count': 0, 'else_count': 0,
        })
        return self

    def pass_turn(self) -> 'ProgramBuilder':
        self.ops.append({
            'type': OP_PASS, 'target': 0, 'attr_id': 0,
            'value': create_const(0), 'value2': create_const(0),
            'then_count': 0, 'else_count': 0,
        })
        return self

    def win(self, target: int) -> 'ProgramBuilder':
        self.ops.append({
            'type': OP_WIN, 'target': target, 'attr_id': 0,
            'value': create_const(0), 'value2': create_const(0),
            'then_count': 0, 'else_count': 0,
        })
        return self

    def lose(self, target: int) -> 'ProgramBuilder':
        self.ops.append({
            'type': OP_LOSE, 'target': target, 'attr_id': 0,
            'value': create_const(0), 'value2': create_const(0),
            'then_count': 0, 'else_count': 0,
        })
        return self

    def set_attr(self, target: int, attr_id: int, value: ValueSpec) -> 'ProgramBuilder':
        self.ops.append({
            'type': OP_SET_ATTR, 'target': target, 'attr_id': attr_id,
            'value': value, 'value2': create_const(0),
            'then_count': 0, 'else_count': 0,
        })
        return self

    def add_attr(self, target: int, attr_id: int, delta: ValueSpec) -> 'ProgramBuilder':
        self.ops.append({
            'type': OP_ADD_ATTR, 'target': target, 'attr_id': attr_id,
            'value': delta, 'value2': create_const(0),
            'then_count': 0, 'else_count': 0,
        })
        return self

    def if_gt(self, lhs: ValueSpec, rhs: ValueSpec,
              then_builder: 'ProgramBuilder',
              else_builder: Optional['ProgramBuilder'] = None) -> 'ProgramBuilder':
        then_ops = then_builder.ops if then_builder else []
        else_ops = else_builder.ops if else_builder else []
        self.ops.append({
            'type': OP_IF_GT, 'target': 0, 'attr_id': 0,
            'value': lhs, 'value2': rhs,
            'then_count': len(then_ops), 'else_count': len(else_ops),
            'then_ops': then_ops, 'else_ops': else_ops,
        })
        return self

    def if_lt(self, lhs: ValueSpec, rhs: ValueSpec,
              then_builder: 'ProgramBuilder',
              else_builder: Optional['ProgramBuilder'] = None) -> 'ProgramBuilder':
        then_ops = then_builder.ops if then_builder else []
        else_ops = else_builder.ops if else_builder else []
        self.ops.append({
            'type': OP_IF_LT, 'target': 0, 'attr_id': 0,
            'value': lhs, 'value2': rhs,
            'then_count': len(then_ops), 'else_count': len(else_ops),
            'then_ops': then_ops, 'else_ops': else_ops,
        })
        return self

    def if_eq(self, lhs: ValueSpec, rhs: ValueSpec,
              then_builder: 'ProgramBuilder',
              else_builder: Optional['ProgramBuilder'] = None) -> 'ProgramBuilder':
        then_ops = then_builder.ops if then_builder else []
        else_ops = else_builder.ops if else_builder else []
        self.ops.append({
            'type': OP_IF_EQ, 'target': 0, 'attr_id': 0,
            'value': lhs, 'value2': rhs,
            'then_count': len(then_ops), 'else_count': len(else_ops),
            'then_ops': then_ops, 'else_ops': else_ops,
        })
        return self

    def noop(self) -> 'ProgramBuilder':
        self.ops.append({
            'type': OP_NOOP, 'target': 0, 'attr_id': 0,
            'value': create_const(0), 'value2': create_const(0),
            'then_count': 0, 'else_count': 0,
        })
        return self

    def _flatten_ops(self, ops_list: List[dict]) -> List[dict]:
        """Flatten nested IF ops into a linear list."""
        result = []
        for op in ops_list:
            if op['type'] in (OP_IF_GT, OP_IF_LT, OP_IF_EQ):
                then_ops = op.get('then_ops', [])
                else_ops = op.get('else_ops', [])
                flat_then = self._flatten_ops(then_ops)
                flat_else = self._flatten_ops(else_ops)

                result.append({
                    'type': op['type'], 'target': op['target'], 'attr_id': op['attr_id'],
                    'value': op['value'], 'value2': op['value2'],
                    'then_count': len(flat_then), 'else_count': len(flat_else),
                })
                result.extend(flat_then)
                result.extend(flat_else)
            else:
                result.append({
                    'type': op['type'], 'target': op['target'], 'attr_id': op['attr_id'],
                    'value': op['value'], 'value2': op['value2'],
                    'then_count': 0, 'else_count': 0,
                })
        return result

    def build(self) -> Program:
        """Build the final Program from accumulated ops."""
        flat_ops = self._flatten_ops(self.ops)

        if not flat_ops or flat_ops[-1]['type'] != OP_END:
            flat_ops.append({
                'type': OP_END, 'target': 0, 'attr_id': 0,
                'value': create_const(0), 'value2': create_const(0),
                'then_count': 0, 'else_count': 0,
            })

        num_ops = min(len(flat_ops), MAX_OPS)

        # Build arrays
        op_types = [0] * MAX_OPS
        targets = [0] * MAX_OPS
        attr_ids = [0] * MAX_OPS
        value_types = [[0] * MAX_VALUE_DEPTH for _ in range(MAX_OPS)]
        value_param1 = [[0.0] * MAX_VALUE_DEPTH for _ in range(MAX_OPS)]
        value_param2 = [[0.0] * MAX_VALUE_DEPTH for _ in range(MAX_OPS)]
        value_num_nodes = [0] * MAX_OPS
        value2_types = [[0] * MAX_VALUE_DEPTH for _ in range(MAX_OPS)]
        value2_param1 = [[0.0] * MAX_VALUE_DEPTH for _ in range(MAX_OPS)]
        value2_param2 = [[0.0] * MAX_VALUE_DEPTH for _ in range(MAX_OPS)]
        value2_num_nodes = [0] * MAX_OPS
        if_then_count = [0] * MAX_OPS
        if_else_count = [0] * MAX_OPS

        for i, op in enumerate(flat_ops[:num_ops]):
            op_types[i] = op['type']
            targets[i] = op['target']
            attr_ids[i] = op['attr_id']
            if_then_count[i] = op['then_count']
            if_else_count[i] = op['else_count']

            v1 = op['value']
            for j in range(MAX_VALUE_DEPTH):
                value_types[i][j] = int(v1.types[j])
                value_param1[i][j] = float(v1.param1[j])
                value_param2[i][j] = float(v1.param2[j])
            value_num_nodes[i] = int(v1.num_nodes)

            v2 = op['value2']
            for j in range(MAX_VALUE_DEPTH):
                value2_types[i][j] = int(v2.types[j])
                value2_param1[i][j] = float(v2.param1[j])
                value2_param2[i][j] = float(v2.param2[j])
            value2_num_nodes[i] = int(v2.num_nodes)

        return Program(
            op_types=jnp.array(op_types, dtype=jnp.int32),
            targets=jnp.array(targets, dtype=jnp.int32),
            attr_ids=jnp.array(attr_ids, dtype=jnp.int32),
            value_types=jnp.array(value_types, dtype=jnp.int32),
            value_param1=jnp.array(value_param1, dtype=jnp.float32),
            value_param2=jnp.array(value_param2, dtype=jnp.float32),
            value_num_nodes=jnp.array(value_num_nodes, dtype=jnp.int32),
            value2_types=jnp.array(value2_types, dtype=jnp.int32),
            value2_param1=jnp.array(value2_param1, dtype=jnp.float32),
            value2_param2=jnp.array(value2_param2, dtype=jnp.float32),
            value2_num_nodes=jnp.array(value2_num_nodes, dtype=jnp.int32),
            if_then_count=jnp.array(if_then_count, dtype=jnp.int32),
            if_else_count=jnp.array(if_else_count, dtype=jnp.int32),
            num_ops=jnp.array(num_ops, dtype=jnp.int32),
        )


def empty_value_spec() -> ValueSpec:
    """Create an empty/zero ValueSpec."""
    return create_const(0.0)
