# src/dynlib/compiler/codegen/rewrite.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List
import ast
import re

__all__ = ["sanitize_expr", "compile_scalar_expr", "lower_expr_node", "NameMaps"]

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

# Map DSL math names → math.<fn> (Numba-friendly)
_MATH_FUNCS = {
    "abs": ("builtins", "abs"),
    "min": ("builtins", "min"),
    "max": ("builtins", "max"),
    "round": ("builtins", "round"),
    "exp": ("math", "exp"),
    "log": ("math", "log"),
    "sqrt": ("math", "sqrt"),
    "sin": ("math", "sin"),
    "cos": ("math", "cos"),
    "tan": ("math", "tan"),
}

class _NameLowerer(ast.NodeTransformer):
    """Lower states/params to array indexing; inline aux and functions."""

    def __init__(self, nmap: NameMaps, aux_defs: Dict[str, ast.AST], fn_defs: Dict[str, Tuple[Tuple[str, ...], ast.AST]]):
        super().__init__()
        self.nmap = nmap
        self.aux_defs = aux_defs
        self.fn_defs = fn_defs

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
        # Inline aux by substituting its expression AST
        if node.id in self.aux_defs:
            cloned = self._clone(self.aux_defs[node.id])
            return self.visit(cloned)  # continue lowering the inlined aux
        # Allow math/builtins symbols to pass through (resolved at module scope)
        return node
    
    def visit_Call(self, node: ast.Call):
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

def _parse_expr(expr: str) -> ast.AST:
    return ast.parse(sanitize_expr(expr), mode="eval").body

def lower_expr_node(expr: str, nmap: NameMaps, *, aux_defs: Dict[str, str] | None = None, fn_defs: Dict[str, Tuple[Tuple[str, ...], str]] | None = None) -> ast.AST:
    """Return a lowered AST node for the expression (supports 't', y_vec, p_vec, aux, functions)."""
    aux_defs = aux_defs or {}
    fn_defs = fn_defs or {}
    aux_ast = {k: _parse_expr(v) for k, v in aux_defs.items()}
    fn_ast  = {k: (args, _parse_expr(v)) for k, (args, v) in fn_defs.items()}
    lowered = _NameLowerer(nmap, aux_ast, fn_ast).visit(_parse_expr(expr))
    ast.fix_missing_locations(lowered)
    return lowered

def compile_scalar_expr(expr: str, nmap: NameMaps, *, aux_defs: Dict[str, str] | None = None, fn_defs: Dict[str, Tuple[Tuple[str, ...], str]] | None = None) -> Callable:
    """
    Return a pure-numeric callable: f(t, y_vec, params) -> float
    """
    lowered = lower_expr_node(expr, nmap, aux_defs=aux_defs, fn_defs=fn_defs)
    mod = ast.Module(
        body=[
            ast.Import(names=[ast.alias(name="math", asname=None)]),
            ast.FunctionDef(
                name="_f",
                args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="t"), ast.arg(arg="y_vec"), ast.arg(arg="params")], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=[ast.Return(value=lowered)],
                decorator_list=[],
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    ns: Dict[str, object] = {}
    exec(compile(mod, "<dsl-lowered>", "exec"), ns, ns)
    return ns["_f"]

