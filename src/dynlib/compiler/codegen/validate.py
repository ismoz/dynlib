# src/dynlib/compiler/codegen/validate.py
"""
Lightweight static checks for stepper implementations.

Guardrails enforced at build time:
    * Forbid writes to runner-owned buffers (records, events, cursors, etc.)
    * Ensure only the documented output arrays are mutated (y_prop, t_prop, dt_next, err_est)
    * Treat runtime_ws as read-only and stepper workspace tuples as immutable

These checks purposely stay heuristic – they catch obvious mistakes without
trying to execute the stepper. They are aimed at providing better error
messages when integrating third-party steppers.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable, List, Set

from dynlib.errors import ModelLoadError

__all__ = ["validate_stepper_function", "report_validation_issues", "StepperValidationError"]


class StepperValidationError(ModelLoadError):
    """Raised when stepper code violates guardrails."""


@dataclass
class ValidationIssue:
    """A single validation warning or error."""

    severity: str  # "error" | "warning"
    message: str
    line: int | None = None


ALLOWED_WRITES = {"y_prop", "t_prop", "dt_next", "err_est"}
IMMUTABLE_ARGS = {"t", "dt", "rhs", "params", "y_curr", "runtime_ws", "ws", "stepper_config"}

FORBIDDEN_WRITES = {
    # Runner-owned scratch/recording buffers
    "y_curr",
    "y_prev",
    "T",
    "Y",
    "STEP",
    "FLAGS",
    "EVT_CODE",
    "EVT_INDEX",
    "EVT_LOG_DATA",
    "evt_log_scratch",
    # Runner cursors / control structures
    "i_start",
    "step_start",
    "cap_rec",
    "cap_evt",
    "user_break_flag",
    "status_out",
    "hint_out",
    "i_out",
    "step_out",
    "t_out",
}
FORBIDDEN_WRITES.update(IMMUTABLE_ARGS)

WORKSPACE_NAMES = {"ws", "stepper_ws"}
RUNTIME_WS_NAME = "runtime_ws"


class StepperASTVisitor(ast.NodeVisitor):
    """AST visitor that enforces the guardrails."""

    def __init__(self) -> None:
        self.issues: List[ValidationIssue] = []
        self.allowed_locals: Set[str] = set()

    def _add_error(self, msg: str, node: ast.AST) -> None:
        self.issues.append(ValidationIssue("error", msg, getattr(node, "lineno", None)))

    def _add_warning(self, msg: str, node: ast.AST) -> None:
        self.issues.append(ValidationIssue("warning", msg, getattr(node, "lineno", None)))

    # --- Assignment tracking -------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_assignment_target(target)
            self.visit(target)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_assignment_target(node.target)
        self.visit(node.target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_assignment_target(node.target)
        if node.annotation:
            self.visit(node.annotation)
        if node.value:
            self.visit(node.value)
        self.visit(node.target)

    def _check_assignment_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.allowed_locals.add(target.id)
            self._check_name_store(target.id, target)
        elif isinstance(target, ast.Subscript):
            base = target.value
            if isinstance(base, ast.Name):
                self._check_name_store(base.id, target, via_subscript=True)
            elif isinstance(base, ast.Attribute):
                self._check_attribute_store(base, target, via_subscript=True)
            self.visit(target.slice)
        elif isinstance(target, ast.Attribute):
            self._check_attribute_store(target, target, via_subscript=False)
        else:
            # Recurse into nested targets (e.g., tuple unpacking)
            for child in ast.iter_child_nodes(target):
                self._check_assignment_target(child)

    def _check_name_store(self, name: str, node: ast.AST, *, via_subscript: bool = False) -> None:
        if name in FORBIDDEN_WRITES:
            self._add_error(f"Stepper must not write to runner-owned variable '{name}'", node)
            return

        if name in ALLOWED_WRITES:
            return

        if name in WORKSPACE_NAMES and not via_subscript:
            self._add_error(
                "Workspace tuples are immutable; mutate the contained arrays (e.g. ws.field[:]) instead of rebinding",
                node,
            )
            return

        if name not in self.allowed_locals:
            self._add_warning(
                f"Assignment to unknown variable '{name}' (ensure this is a local temporary)",
                node,
            )

    def _check_attribute_store(self, attr: ast.Attribute, node: ast.AST, *, via_subscript: bool) -> None:
        owner = attr.value
        if isinstance(owner, ast.Name):
            owner_name = owner.id
            if owner_name == RUNTIME_WS_NAME:
                self._add_error("runtime_ws is read-only; steppers must not write to runtime workspace fields", node)
            elif owner_name in WORKSPACE_NAMES and not via_subscript:
                self._add_error(
                    "Workspace tuples are immutable; assign to slices of the contained arrays instead",
                    node,
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Reset locals per function
        prev_locals = self.allowed_locals
        self.allowed_locals = set()
        for stmt in node.body:
            self.visit(stmt)
        self.allowed_locals = prev_locals


def validate_stepper_function(stepper_fn: Callable, stepper_name: str) -> List[ValidationIssue]:
    """
    Validate a stepper callable against light AST-based guardrails.

    Args:
        stepper_fn: Python function implementing the stepper.
        stepper_name: Name of the stepper (for error messages).

    Returns:
        List of ValidationIssue items (errors and warnings).
    """
    try:
        source = inspect.getsource(stepper_fn)
    except (OSError, TypeError) as exc:  # pragma: no cover - defensive
        raise StepperValidationError(
            f"Cannot inspect source of stepper '{stepper_name}': {exc}"
        ) from exc

    source = textwrap.dedent(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise StepperValidationError(
            f"Cannot parse stepper '{stepper_name}' source: {exc}"
        ) from exc

    visitor = StepperASTVisitor()
    visitor.visit(tree)
    return visitor.issues


def report_validation_issues(issues: List[ValidationIssue], stepper_name: str, *, strict: bool = True) -> None:
    """
    Report validation issues, raising descriptive errors in strict mode.
    """
    if not issues:
        return

    errors = [iss for iss in issues if iss.severity == "error"]
    warnings = [iss for iss in issues if iss.severity == "warning"]

    if errors:
        lines = [f"Stepper '{stepper_name}' validation FAILED:"]
        for issue in errors:
            loc = f" (line {issue.line})" if issue.line else ""
            lines.append(f"  ERROR{loc}: {issue.message}")
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for issue in warnings:
                loc = f" (line {issue.line})" if issue.line else ""
                lines.append(f"  WARN{loc}: {issue.message}")
        raise StepperValidationError("\n".join(lines))

    if warnings and strict:
        lines = [f"Stepper '{stepper_name}' validation warnings (strict mode):"]
        for issue in warnings:
            loc = f" (line {issue.line})" if issue.line else ""
            lines.append(f"  WARN{loc}: {issue.message}")
        raise StepperValidationError("\n".join(lines))
