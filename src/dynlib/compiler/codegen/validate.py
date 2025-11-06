# src/dynlib/compiler/codegen/validate.py
"""
Build-time validation for stepper implementations.

Implements guardrails checks:
- Verify steppers write only allowed outputs (y_prop, t_prop[0], dt_next[0], err_est[0])
- Ensure bank accesses use whole-lane slicing (no strided/partial lanes)
- Detect forbidden writes to record/log buffers or runner-owned arrays
"""
from __future__ import annotations
import ast
import inspect
import textwrap
from typing import Set, List, Callable
from dataclasses import dataclass

from dynlib.errors import ModelLoadError

__all__ = ["validate_stepper_function", "StepperValidationError"]


class StepperValidationError(ModelLoadError):
    """Raised when stepper code violates guardrails."""
    pass


@dataclass
class ValidationIssue:
    """A single validation warning or error."""
    severity: str  # "error" | "warning"
    message: str
    line: int | None = None


# Allowed write targets (per guardrails)
ALLOWED_WRITES = {"y_prop", "t_prop", "dt_next", "err_est"}

# Forbidden targets (runner-owned)
FORBIDDEN_WRITES = {
    "y_curr", "y_prev", "T", "Y", "STEP", "FLAGS",
    "EVT_TIME", "EVT_CODE", "EVT_INDEX", "EVT_LOG_DATA",
    "i_start", "step_start", "cap_rec", "cap_evt",
    "user_break_flag", "status_out", "hint_out",
    "i_out", "step_out", "t_out",
}

# Work banks (read/write allowed, but must use whole-lane slicing)
WORK_BANKS = {"sp", "ss", "sw0", "sw1", "sw2", "sw3", "iw0", "bw0"}

# Read-only inputs
READ_ONLY = {"t", "dt", "y_curr", "rhs", "params"}


class StepperASTVisitor(ast.NodeVisitor):
    """AST visitor to check stepper function for guardrails violations."""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.allowed_locals: Set[str] = set()
    
    def _add_error(self, msg: str, node: ast.AST) -> None:
        line = getattr(node, "lineno", None)
        self.issues.append(ValidationIssue("error", msg, line))
    
    def _add_warning(self, msg: str, node: ast.AST) -> None:
        line = getattr(node, "lineno", None)
        self.issues.append(ValidationIssue("warning", msg, line))
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Check assignment targets."""
        for target in node.targets:
            self._check_assignment_target(target)
        self.generic_visit(node)
    
    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Check augmented assignment (+=, etc.)."""
        self._check_assignment_target(node.target)
        self.generic_visit(node)
    
    def _check_assignment_target(self, target: ast.AST) -> None:
        """Validate an assignment target."""
        if isinstance(target, ast.Name):
            name = target.id
            # Track local variables
            self.allowed_locals.add(name)
            
            # Check if it's a forbidden parameter
            if name in FORBIDDEN_WRITES:
                self._add_error(
                    f"Stepper must not write to runner-owned variable '{name}'",
                    target
                )
        
        elif isinstance(target, ast.Subscript):
            # Check subscript assignment (e.g., y_prop[i] = ...)
            if isinstance(target.value, ast.Name):
                name = target.value.id
                
                # Check forbidden writes
                if name in FORBIDDEN_WRITES:
                    self._add_error(
                        f"Stepper must not write to runner-owned array '{name}'",
                        target
                    )
                
                # Check allowed outputs (must be simple indexed or sliced)
                elif name in ALLOWED_WRITES:
                    # This is OK - writing to allowed outputs
                    pass
                
                # Check work bank writes (must use whole-lane slicing)
                elif name in WORK_BANKS:
                    self._check_bank_slicing(name, target.slice, target)
                
                # Unknown target - warn
                elif name not in self.allowed_locals:
                    self._add_warning(
                        f"Assignment to unknown variable '{name}' (may be OK if local)",
                        target
                    )
    
    def _check_bank_slicing(self, bank_name: str, slice_node: ast.AST, parent: ast.AST) -> None:
        """
        Check that bank slicing uses whole-lane patterns.
        
        Allowed:
            sw0[:n]           # First lane
            sw0[n:2*n]        # Second lane
            sw0[i*n:(i+1)*n]  # i-th lane
        
        Forbidden:
            sw0[::2]          # Strided
            sw0[5]            # Single element (should use lanes)
            sw0[a:b:c]        # Strided slice
        """
        # For now, we check for:
        # 1. No stride (step != None and step != 1)
        # 2. Subscript should be a slice, not a single index (unless it's an integer index in iw0/bw0)
        
        if isinstance(slice_node, ast.Slice):
            # Check for stride
            if slice_node.step is not None:
                if not (isinstance(slice_node.step, ast.Constant) and slice_node.step.value == 1):
                    self._add_error(
                        f"Strided slicing forbidden for bank '{bank_name}' (must use whole lanes)",
                        parent
                    )
        
        elif isinstance(slice_node, ast.Index):
            # ast.Index for older Python versions; check wrapped value
            if hasattr(slice_node, "value"):
                inner = slice_node.value
                if bank_name not in ("iw0", "bw0"):
                    # Float banks should use slices, not single indices
                    self._add_warning(
                        f"Single-element indexing of '{bank_name}' detected; "
                        f"consider whole-lane slicing for clarity",
                        parent
                    )
        
        elif isinstance(slice_node, (ast.Constant, ast.Name, ast.BinOp)):
            # Direct index (e.g., sw0[5] or sw0[i])
            if bank_name not in ("iw0", "bw0"):
                # Float banks should use slices, not single indices
                self._add_warning(
                    f"Single-element indexing of '{bank_name}' detected; "
                    f"consider whole-lane slicing for clarity",
                    parent
                )
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function-local definitions."""
        # Don't descend into nested functions
        # Just process this function's body
        for stmt in node.body:
            self.visit(stmt)


def validate_stepper_function(stepper_fn: Callable, stepper_name: str) -> List[ValidationIssue]:
    """
    Validate a stepper function against guardrails.
    
    Performs static analysis of the stepper function's AST to check:
    - No writes to forbidden runner-owned buffers
    - Only allowed outputs are written (y_prop, t_prop[0], dt_next[0], err_est[0])
    - Bank accesses use whole-lane slicing patterns
    
    Args:
        stepper_fn: The stepper function to validate
        stepper_name: Name of the stepper (for error messages)
    
    Returns:
        List of ValidationIssue (errors and warnings)
    
    Raises:
        StepperValidationError: If source code cannot be inspected
    """
    # Get source code
    try:
        source = inspect.getsource(stepper_fn)
    except (OSError, TypeError) as e:
        raise StepperValidationError(
            f"Cannot inspect source of stepper '{stepper_name}': {e}"
        )
    
    # Dedent to handle nested function definitions
    source = textwrap.dedent(source)
    
    # Parse AST
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise StepperValidationError(
            f"Cannot parse stepper '{stepper_name}' source: {e}"
        )
    
    # Visit and collect issues
    visitor = StepperASTVisitor()
    visitor.visit(tree)
    
    return visitor.issues


def report_validation_issues(
    issues: List[ValidationIssue],
    stepper_name: str,
    strict: bool = True
) -> None:
    """
    Report validation issues, raising an error if strict mode is enabled.
    
    Args:
        issues: List of issues from validation
        stepper_name: Name of the stepper (for error messages)
        strict: If True, raise error on any issue; if False, only raise on errors
    
    Raises:
        StepperValidationError: If issues found and strict mode enabled
    """
    if not issues:
        return
    
    errors = [iss for iss in issues if iss.severity == "error"]
    warnings = [iss for iss in issues if iss.severity == "warning"]
    
    if errors:
        msg_lines = [f"Stepper '{stepper_name}' validation FAILED:"]
        for err in errors:
            loc = f" (line {err.line})" if err.line else ""
            msg_lines.append(f"  ERROR{loc}: {err.message}")
        
        if warnings:
            msg_lines.append("\nWarnings:")
            for warn in warnings:
                loc = f" (line {warn.line})" if warn.line else ""
                msg_lines.append(f"  WARN{loc}: {warn.message}")
        
        raise StepperValidationError("\n".join(msg_lines))
    
    if warnings and strict:
        msg_lines = [f"Stepper '{stepper_name}' validation warnings (strict mode):"]
        for warn in warnings:
            loc = f" (line {warn.line})" if warn.line else ""
            msg_lines.append(f"  WARN{loc}: {warn.message}")
        
        raise StepperValidationError("\n".join(msg_lines))
