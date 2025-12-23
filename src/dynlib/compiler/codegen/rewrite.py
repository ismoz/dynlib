# src/dynlib/compiler/codegen/rewrite.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List
import ast
import re

from dynlib.dsl.constants import BUILTIN_CONSTS
from dynlib.errors import ModelLoadError

__all__ = [
    "sanitize_expr",
    "compile_scalar_expr",
    "lower_expr_node",
    "lower_expr_with_preamble",
    "NameMaps",
]

_POW = re.compile(r"\^")

def sanitize_expr(expr: str) -> str:
    """Normalize DSL math to Python."""
    expr = expr.strip()
    expr = _POW.sub("**", expr)
    return expr

@dataclass(frozen=True)
class NameMaps:
    # indices for y_vec and params
    state_to_ix: Dict[str, int]
    param_to_ix: Dict[str, int]
    # aux names exist only during evaluation; not assignable in events
    aux_names: Tuple[str, ...]
    # function table: name -> (argnames, expr_str)
    functions: Dict[str, Tuple[Tuple[str, ...], str]]
    # builtin constants (already cast to target dtype if needed)
    constants: Dict[str, float | int]
    # lag map: state_name -> (buffer_len, ring_offset, head_index)
    lag_map: Dict[str, Tuple[int, int, int]] = None

# Map DSL math names → math.<fn> (Numba-friendly)
_MATH_FUNCS = {
    # Builtins
    "abs":   ("builtins", "abs"),
    "min":   ("builtins", "min"),
    "max":   ("builtins", "max"),
    "round": ("builtins", "round"),

    # Exponentials / logs
    "exp":   ("math", "exp"),
    "expm1": ("math", "expm1"),
    "log":   ("math", "log"),
    "log10": ("math", "log10"),
    "log2":  ("math", "log2"),
    "log1p": ("math", "log1p"),
    "sqrt":  ("math", "sqrt"),

    # Trig
    "sin":  ("math", "sin"),
    "cos":  ("math", "cos"),
    "tan":  ("math", "tan"),
    "asin": ("math", "asin"),
    "acos": ("math", "acos"),
    "atan": ("math", "atan"),
    "atan2":("math", "atan2"),

    # Hyperbolic
    "sinh":  ("math", "sinh"),
    "cosh":  ("math", "cosh"),
    "tanh":  ("math", "tanh"),
    "asinh": ("math", "asinh"),
    "acosh": ("math", "acosh"),
    "atanh": ("math", "atanh"),

    # Rounding
    "floor": ("math", "floor"),
    "ceil":  ("math", "ceil"),
    "trunc": ("math", "trunc"),

    # Misc
    "hypot":    ("math", "hypot"),
    "copysign": ("math", "copysign"),

    # Special
    "erf":  ("math", "erf"),
    "erfc": ("math", "erfc"),
}

_SCALAR_MACROS = {
    "sign",
    "heaviside",
    "step",
    "relu",
    "clip",
    "approx",
}

class _NameLowerer(ast.NodeTransformer):
    """Lower states/params to array indexing; inline aux and functions."""

    def __init__(
        self,
        nmap: NameMaps,
        aux_defs: Dict[str, ast.AST],
        fn_defs: Dict[str, Tuple[Tuple[str, ...], ast.AST]],
        constants: Dict[str, float | int] | None = None,
        runtime_arg: str = "runtime_ws",
        hoist_aux: bool = False,
    ):
        super().__init__()
        self.nmap = nmap
        self.aux_defs = aux_defs
        self.fn_defs = fn_defs
        self.constants = constants or BUILTIN_CONSTS
        self.runtime_arg = runtime_arg
        self.hoist_aux = hoist_aux

    def visit_Name(self, node: ast.Name):
        if node.id in self.nmap.state_to_ix:
            ix = self.nmap.state_to_ix[node.id]
            return ast.Subscript(
                value=ast.Name(id="y_vec", ctx=ast.Load()),
                slice=ast.Constant(value=ix),
                ctx=ast.Load(),
            )
        if node.id in self.nmap.param_to_ix:
            ix = self.nmap.param_to_ix[node.id]
            return ast.Subscript(
                value=ast.Name(id="params", ctx=ast.Load()),
                slice=ast.Constant(value=ix),
                ctx=ast.Load(),
            )
        # Keep time symbol 't' as a plain name (it is a formal arg to the emitted functions)
        if node.id == "t":
            return node
        # Inline builtin constants as literals
        if node.id in self.constants:
            return ast.copy_location(ast.Constant(value=self.constants[node.id]), node)
        # Aux: either treat as precomputed local (hoisted) or inline expression
        if node.id in self.aux_defs:
            if self.hoist_aux:
                return ast.copy_location(ast.Name(id=node.id, ctx=ast.Load()), node)
            cloned = self._clone(self.aux_defs[node.id])
            return self.visit(cloned)  # continue lowering the inlined aux
        # Allow math/builtins symbols to pass through (resolved at module scope)
        return node
    
    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            macro_name = node.func.id
            if macro_name in _SCALAR_MACROS:
                return self._expand_scalar_macro(macro_name, node)
            if macro_name in {
                "cross_up",
                "cross_down",
                "cross_either",
                "changed",
                "in_interval",
                "enters_interval",
                "leaves_interval",
                "increasing",
                "decreasing",
            }:
                return self._expand_macro(macro_name, node)
            if macro_name == "range":
                if node.keywords:
                    raise ModelLoadError("range() does not support keyword arguments in the DSL")
                if len(node.args) not in {1, 2, 3}:
                    raise ModelLoadError("range() expects 1, 2, or 3 positional arguments")
                casted_args = [
                    ast.Call(
                        func=ast.Name(id="int", ctx=ast.Load()),
                        args=[self.visit(arg)],
                        keywords=[],
                    )
                    for arg in node.args
                ]
                return ast.copy_location(
                    ast.Call(func=ast.Name(id="range", ctx=ast.Load()), args=casted_args, keywords=[]),
                    node,
                )
        # Check for lag_<name>(k) pattern (with optional arg defaulting to 1)
        if isinstance(node.func, ast.Name) and node.func.id.startswith("lag_"):
            state_name = node.func.id[4:]  # remove "lag_" prefix
            if len(node.args) == 0:
                k = 1
            elif len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                k = int(node.args[0].value)
            else:
                raise ModelLoadError(
                    f"lag_{state_name}() expects no args or a single integer literal"
                )

            if not isinstance(k, int) or k < 1:
                raise ModelLoadError(
                    f"lag_{state_name}({k}) requires a positive integer literal"
                )

            return self._make_lag_access(state_name, k, node)
        
        # Lower function calls: inline user-defined function bodies
        if isinstance(node.func, ast.Name) and node.func.id in self.fn_defs:
            argnames, body_ast = self.fn_defs[node.func.id]
            # Map actual args to formal names
            subs: Dict[str, ast.AST] = {}
            for i, an in enumerate(argnames):
                if i < len(node.args):
                    subs[an] = self.visit(node.args[i])
                else:
                    # no kwargs support in v2 DSL functions
                    subs[an] = ast.Name(id=an, ctx=ast.Load())
            replacer = _ArgReplacer(subs)
            inlined = replacer.visit(ast.copy_location(self._clone(body_ast), node))
            return self.visit(inlined)  # continue lowering inside
        # Lower math/builtins to explicit module attrs (math.fn)
        if isinstance(node.func, ast.Name) and node.func.id in _MATH_FUNCS:
            mod, fn = _MATH_FUNCS[node.func.id]
            if mod == "builtins":
                # keep as plain name (abs/min/max/round) — Numba supports them
                return self.generic_visit(node)
            return ast.copy_location(
                ast.Call(func=ast.Attribute(value=ast.Name(id=mod, ctx=ast.Load()), attr=fn, ctx=ast.Load()),
                         args=[self.visit(a) for a in node.args], keywords=[]),
                node,
            )
        return self.generic_visit(node)
    
    def _make_lag_access(self, state_name: str, k: int, node: ast.AST) -> ast.AST:
        """
        Generate AST for accessing lag buffer:
        
        ss[ss_offset + ((iw0[iw0_index] - k) % depth)]
        
        Where:
          - ss_offset: starting lane in ss for this state's circular buffer
          - iw0_index: slot in iw0 for this state's head pointer
          - depth: max lag depth for this state
          - k: lag amount (1, 2, 3, ...)
        """
        if self.nmap.lag_map is None or state_name not in self.nmap.lag_map:
            raise ModelLoadError(
                f"lag_{state_name}({k}) used, "
                f"but '{state_name}' is not available for lagging. "
                f"This is an internal error - lag detection should have caught this."
            )
        
        buffer_len, ring_offset, head_index = self.nmap.lag_map[state_name]
        max_supported = buffer_len - 1
        
        if k > max_supported:
            raise ModelLoadError(
                f"lag_{state_name}({k}) exceeds detected max depth {max_supported}. "
                f"This is an internal error - lag detection should have caught this."
            )

        lag_head_attr = ast.Attribute(
            value=ast.copy_location(ast.Name(id=self.runtime_arg, ctx=ast.Load()), node),
            attr="lag_head",
            ctx=ast.Load(),
        )
        lag_ring_attr = ast.Attribute(
            value=ast.copy_location(ast.Name(id=self.runtime_arg, ctx=ast.Load()), node),
            attr="lag_ring",
            ctx=ast.Load(),
        )

        head_access = ast.Subscript(
            value=lag_head_attr,
            slice=ast.Constant(value=head_index),
            ctx=ast.Load(),
        )

        head_minus_k = ast.BinOp(
            left=head_access,
            op=ast.Sub(),
            right=ast.Constant(value=k),
        )

        wrap_test = ast.Compare(
            left=self._clone(head_minus_k),
            ops=[ast.Lt()],
            comparators=[ast.Constant(value=0)],
        )
        wrapped_pos = ast.IfExp(
            test=wrap_test,
            body=ast.BinOp(
                left=self._clone(head_minus_k),
                op=ast.Add(),
                right=ast.Constant(value=buffer_len),
            ),
            orelse=head_minus_k,
        )

        index_expr = ast.BinOp(
            left=ast.Constant(value=ring_offset),
            op=ast.Add(),
            right=wrapped_pos,
        )

        lag_access = ast.Subscript(
            value=lag_ring_attr,
            slice=index_expr,
            ctx=ast.Load(),
        )

        return ast.copy_location(lag_access, node)

    def _expand_scalar_macro(self, name: str, node: ast.Call) -> ast.AST:
        if node.keywords:
            raise ModelLoadError(f"{name}() does not support keyword arguments")
        if name == "sign":
            self._expect_arg_len(node, name, 1)
            arg = self.visit(node.args[0])
            gt_zero = ast.Compare(
                left=self._clone(arg),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=0)],
            )
            lt_zero = ast.Compare(
                left=self._clone(arg),
                ops=[ast.Lt()],
                comparators=[ast.Constant(value=0)],
            )
            gt_int = ast.Call(func=ast.Name(id="int", ctx=ast.Load()), args=[gt_zero], keywords=[])
            lt_int = ast.Call(func=ast.Name(id="int", ctx=ast.Load()), args=[lt_zero], keywords=[])
            return ast.copy_location(
                ast.BinOp(left=gt_int, op=ast.Sub(), right=lt_int),
                node,
            )
        if name in {"heaviside", "step"}:
            self._expect_arg_len(node, name, 1)
            arg = self.visit(node.args[0])
            return ast.copy_location(
                ast.Compare(
                    left=arg,
                    ops=[ast.GtE()],
                    comparators=[ast.Constant(value=0)],
                ),
                node,
            )
        if name == "relu":
            self._expect_arg_len(node, name, 1)
            arg = self.visit(node.args[0])
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="max", ctx=ast.Load()),
                    args=[arg, ast.Constant(value=0)],
                    keywords=[],
                ),
                node,
            )
        if name == "clip":
            self._expect_arg_len(node, name, 3)
            value = self.visit(node.args[0])
            lower = self.visit(node.args[1])
            upper = self.visit(node.args[2])
            max_call = ast.Call(
                func=ast.Name(id="max", ctx=ast.Load()),
                args=[value, lower],
                keywords=[],
            )
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="min", ctx=ast.Load()),
                    args=[max_call, upper],
                    keywords=[],
                ),
                node,
            )
        if name == "approx":
            self._expect_arg_len(node, name, 3)
            left = self.visit(node.args[0])
            right = self.visit(node.args[1])
            tol = self.visit(node.args[2])
            diff = ast.BinOp(left=left, op=ast.Sub(), right=right)
            abs_call = ast.Call(
                func=ast.Name(id="abs", ctx=ast.Load()),
                args=[diff],
                keywords=[],
            )
            return ast.copy_location(
                ast.Compare(
                    left=abs_call,
                    ops=[ast.LtE()],
                    comparators=[tol],
                ),
                node,
            )
        raise ModelLoadError(f"Unsupported macro {name}()")

    def _expand_macro(self, name: str, node: ast.Call) -> ast.AST:
        if node.keywords:
            raise ModelLoadError(f"{name}() does not support keyword arguments")
        if name == "cross_up":
            self._expect_arg_len(node, name, 2)
            state = self._state_arg(node, name)
            thresh = self.visit(node.args[1])
            return self._cross_up_expr(state, thresh, node)
        if name == "cross_down":
            self._expect_arg_len(node, name, 2)
            state = self._state_arg(node, name)
            thresh = self.visit(node.args[1])
            return self._cross_down_expr(state, thresh, node)
        if name == "cross_either":
            self._expect_arg_len(node, name, 2)
            state = self._state_arg(node, name)
            thresh = self.visit(node.args[1])
            return ast.BoolOp(
                op=ast.Or(),
                values=[
                    self._cross_up_expr(state, self._clone(thresh), node),
                    self._cross_down_expr(state, self._clone(thresh), node),
                ],
            )
        if name == "changed":
            self._expect_arg_len(node, name, 1)
            state = self._state_arg(node, name)
            return ast.Compare(
                left=self._state_value_node(state),
                ops=[ast.NotEq()],
                comparators=[self._make_lag_access(state, 1, node)],
            )
        if name == "in_interval":
            self._expect_arg_len(node, name, 3)
            value = self.visit(node.args[0])
            lower = self.visit(node.args[1])
            upper = self.visit(node.args[2])
            return self._between_expr(value, lower, upper)
        if name == "enters_interval":
            self._expect_arg_len(node, name, 3)
            state = self._state_arg(node, name)
            lower = self.visit(node.args[1])
            upper = self.visit(node.args[2])
            lag = self._make_lag_access(state, 1, node)
            lag_lt_lower = ast.Compare(
                left=lag,
                ops=[ast.Lt()],
                comparators=[self._clone(lower)],
            )
            lag_gt_upper = ast.Compare(
                left=self._make_lag_access(state, 1, node),
                ops=[ast.Gt()],
                comparators=[self._clone(upper)],
            )
            prev_outside = ast.BoolOp(
                op=ast.Or(),
                values=[lag_lt_lower, lag_gt_upper],
            )
            now_inside = self._between_expr(self._state_value_node(state), lower, upper)
            return ast.BoolOp(op=ast.And(), values=[prev_outside, now_inside])
        if name == "leaves_interval":
            self._expect_arg_len(node, name, 3)
            state = self._state_arg(node, name)
            lower = self.visit(node.args[1])
            upper = self.visit(node.args[2])
            lag_inside = self._between_expr(self._make_lag_access(state, 1, node), lower, upper)
            x_lt_lower = ast.Compare(
                left=self._state_value_node(state),
                ops=[ast.Lt()],
                comparators=[self._clone(lower)],
            )
            x_gt_upper = ast.Compare(
                left=self._state_value_node(state),
                ops=[ast.Gt()],
                comparators=[self._clone(upper)],
            )
            now_outside = ast.BoolOp(op=ast.Or(), values=[x_lt_lower, x_gt_upper])
            return ast.BoolOp(op=ast.And(), values=[lag_inside, now_outside])
        if name == "increasing":
            self._expect_arg_len(node, name, 1)
            state = self._state_arg(node, name)
            return ast.Compare(
                left=self._state_value_node(state),
                ops=[ast.Gt()],
                comparators=[self._make_lag_access(state, 1, node)],
            )
        if name == "decreasing":
            self._expect_arg_len(node, name, 1)
            state = self._state_arg(node, name)
            return ast.Compare(
                left=self._state_value_node(state),
                ops=[ast.Lt()],
                comparators=[self._make_lag_access(state, 1, node)],
            )
        raise ModelLoadError(f"Unsupported macro {name}()")

    def _expect_arg_len(self, node: ast.Call, name: str, expected: int) -> None:
        if len(node.args) != expected:
            raise ModelLoadError(f"{name}() expects {expected} positional arguments")

    def _state_arg(self, node: ast.Call, name: str) -> str:
        if not node.args:
            raise ModelLoadError(f"{name}() requires at least one argument")
        target = node.args[0]
        if not isinstance(target, ast.Name):
            raise ModelLoadError(f"{name}() first argument must be a state identifier (e.g. x)")
        state_name = target.id
        if state_name not in self.nmap.state_to_ix:
            raise ModelLoadError(
                f"{name}() first argument must be a declared state, got '{state_name}'"
            )
        return state_name

    def _state_value_node(self, state_name: str) -> ast.AST:
        return ast.Subscript(
            value=ast.Name(id="y_vec", ctx=ast.Load()),
            slice=ast.Constant(value=self.nmap.state_to_ix[state_name]),
            ctx=ast.Load(),
        )

    def _cross_up_expr(self, state: str, thresh: ast.AST, node: ast.AST) -> ast.AST:
        return ast.BoolOp(
            op=ast.And(),
            values=[
                ast.Compare(
                    left=self._make_lag_access(state, 1, node),
                    ops=[ast.LtE()],
                    comparators=[self._clone(thresh)],
                ),
                ast.Compare(
                    left=self._state_value_node(state),
                    ops=[ast.Gt()],
                    comparators=[self._clone(thresh)],
                ),
            ],
        )

    def _cross_down_expr(self, state: str, thresh: ast.AST, node: ast.AST) -> ast.AST:
        return ast.BoolOp(
            op=ast.And(),
            values=[
                ast.Compare(
                    left=self._make_lag_access(state, 1, node),
                    ops=[ast.GtE()],
                    comparators=[self._clone(thresh)],
                ),
                ast.Compare(
                    left=self._state_value_node(state),
                    ops=[ast.Lt()],
                    comparators=[self._clone(thresh)],
                ),
            ],
        )

    def _between_expr(self, value: ast.AST, lower: ast.AST, upper: ast.AST) -> ast.AST:
        return ast.BoolOp(
            op=ast.And(),
            values=[
                ast.Compare(
                    left=self._clone(lower),
                    ops=[ast.LtE()],
                    comparators=[self._clone(value)],
                ),
                ast.Compare(
                    left=self._clone(value),
                    ops=[ast.LtE()],
                    comparators=[self._clone(upper)],
                ),
            ],
        )

    @staticmethod
    def _clone(node: ast.AST) -> ast.AST:
        return ast.parse(ast.unparse(node), mode="eval").body  # simple structural clone


class _ArgReplacer(ast.NodeTransformer):
    """Replace function-arg names with provided expression ASTs."""
    def __init__(self, subs: Dict[str, ast.AST]):
        self.subs = subs
    def visit_Name(self, node: ast.Name):
        if node.id in self.subs:
            return ast.copy_location(self.subs[node.id], node)
        return node


class _SumGenLowerer(ast.NodeTransformer):
    """
    Lower sum/prod(generator) into a for-loop preamble and a temp variable.
    
    Recognized pattern:
        sum(<elt> for <target> in range(...) [if cond1 [if cond2 ...]])
        prod(<elt> for <target> in range(...) [if cond1 [if cond2 ...]])
    
    Returns a Name node referring to a generated temp (e.g. _sum0) and
    accumulates preamble statements on self.preamble.
    """
    def __init__(self):
        self.preamble: List[ast.stmt] = []
        self._counter = 0

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and node.func.id in {"sum", "prod"}):
            return node
        func_name = node.func.id
        if node.keywords:
            raise ModelLoadError(f"{func_name}(...) with keyword arguments is not supported in the DSL")
        if len(node.args) != 1:
            return node
        gen = node.args[0]
        if not isinstance(gen, ast.GeneratorExp):
            return node
        if len(gen.generators) != 1:
            raise ModelLoadError(f"Only a single generator is supported inside {func_name}(...)")
        comp = gen.generators[0]
        if not isinstance(comp.target, ast.Name):
            raise ModelLoadError(f"{func_name}(generator) requires a simple loop variable (e.g. 'for i in ...')")
        loop_var = comp.target.id
        iter_call = comp.iter
        if not (
            isinstance(iter_call, ast.Call)
            and isinstance(iter_call.func, ast.Name)
            and iter_call.func.id == "range"
        ):
            raise ModelLoadError(f"{func_name}(generator) is only supported for ranges: {func_name}(expr for i in range(...))")

        temp_name = f"_{func_name}{self._counter}"
        self._counter += 1

        init = ast.Assign(
            targets=[ast.Name(id=temp_name, ctx=ast.Store())],
            value=ast.Constant(value=0.0 if func_name == "sum" else 1.0),
        )

        aug_op = ast.Add() if func_name == "sum" else ast.Mult()
        aug = ast.AugAssign(
            target=ast.Name(id=temp_name, ctx=ast.Store()),
            op=aug_op,
            value=gen.elt,
        )

        loop_body: List[ast.stmt]
        if comp.ifs:
            test = comp.ifs[0] if len(comp.ifs) == 1 else ast.BoolOp(op=ast.And(), values=comp.ifs)
            loop_body = [ast.If(test=test, body=[aug], orelse=[])]
        else:
            loop_body = [aug]

        loop = ast.For(
            target=ast.Name(id=loop_var, ctx=ast.Store()),
            iter=iter_call,
            body=loop_body,
            orelse=[],
        )

        self.preamble.extend([init, loop])
        return ast.copy_location(ast.Name(id=temp_name, ctx=ast.Load()), node)

def _parse_expr(expr: str) -> ast.AST:
    return ast.parse(sanitize_expr(expr), mode="eval").body

def lower_expr_node(
    expr: str,
    nmap: NameMaps,
    *,
    aux_defs: Dict[str, str] | None = None,
    fn_defs: Dict[str, Tuple[Tuple[str, ...], str]] | None = None,
    runtime_arg: str = "runtime_ws",
    hoist_aux: bool = False,
) -> ast.AST:
    """Return a lowered AST node for the expression (supports 't', y_vec, p_vec, aux, functions)."""
    aux_defs = aux_defs or {}
    fn_defs = fn_defs or {}
    aux_ast = {k: _parse_expr(v) for k, v in aux_defs.items()}
    fn_ast  = {k: (args, _parse_expr(v)) for k, (args, v) in fn_defs.items()}
    lowered = _NameLowerer(
        nmap,
        aux_ast,
        fn_ast,
        constants=nmap.constants,
        runtime_arg=runtime_arg,
        hoist_aux=hoist_aux,
    ).visit(_parse_expr(expr))
    ast.fix_missing_locations(lowered)
    return lowered

def lower_expr_with_preamble(
    expr: str,
    nmap: NameMaps,
    *,
    aux_defs: Dict[str, str] | None = None,
    fn_defs: Dict[str, Tuple[Tuple[str, ...], str]] | None = None,
    runtime_arg: str = "runtime_ws",
    hoist_aux: bool = False,
) -> Tuple[List[ast.stmt], ast.AST]:
    """
    Lower an expression and return (preamble, expr_node), where preamble contains any
    statements needed to evaluate constructs that are not Numba-friendly as-is
    (currently sum(generator) comprehensions).
    """
    lowered = lower_expr_node(
        expr,
        nmap,
        aux_defs=aux_defs,
        fn_defs=fn_defs,
        runtime_arg=runtime_arg,
        hoist_aux=hoist_aux,
    )
    sg = _SumGenLowerer()
    rewritten = sg.visit(lowered)
    return sg.preamble, rewritten

def compile_scalar_expr(
    expr: str,
    nmap: NameMaps,
    *,
    aux_defs: Dict[str, str] | None = None,
    fn_defs: Dict[str, Tuple[Tuple[str, ...], str]] | None = None,
) -> Callable:
    """
    Return a pure-numeric callable: f(t, y_vec, params, runtime_ws) -> float.
    """
    preamble, lowered = lower_expr_with_preamble(expr, nmap, aux_defs=aux_defs, fn_defs=fn_defs)
    mod = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math", asname=None)]),
            ast.FunctionDef(
                name="_f",
                args=ast.arguments(posonlyargs=[], args=[
                    ast.arg(arg="t"), 
                    ast.arg(arg="y_vec"), 
                    ast.arg(arg="params"),
                    ast.arg(arg="runtime_ws"),
                ], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[*preamble, ast.Return(value=lowered)],
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    ns: Dict[str, object] = {}
    exec(compile(mod, "<dsl-lowered>", "exec"), ns, ns)
    return ns["_f"]
