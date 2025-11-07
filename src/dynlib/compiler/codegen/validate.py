# src/dynlib/compiler/codegen/validate.py
"""
Build-time validation for stepper implementations.

Guardrails enforced here:
- Verify steppers write only allowed outputs (y_prop, t_prop[0], dt_next[0], err_est[0])
- Ensure bank accesses use whole-lane slicing (no strided/partial lanes)
- Detect forbidden writes to record/log buffers or runner-owned arrays
- Validate StructSpec lane counts/dtypes/persistence flags
- Forbid writing float data to iw0/bw0 and detect obvious persistence misuse
"""
from __future__ import annotations
import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable, Dict, List, Set

from dynlib.errors import ModelLoadError
from dynlib.steppers.base import StructSpec

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
    "EVT_CODE", "EVT_INDEX", "EVT_LOG_DATA",
    "i_start", "step_start", "cap_rec", "cap_evt",
    "user_break_flag", "status_out", "hint_out",
    "i_out", "step_out", "t_out",
}

# Work banks (read/write allowed, but must use whole-lane slicing)
WORK_BANKS = {"sp", "ss", "sw0", "sw1", "sw2", "sw3", "iw0", "bw0"}

# Banks whose persistence must be warned about if read before write
EPHEMERAL_BANKS = {"sp", "sw0", "sw1", "sw2", "sw3"}

# Persistent integer/flag banks
INT_BANKS = {"iw0", "bw0"}

# Names that clearly represent float/model-dtype buffers
FLOATY_SOURCES = {
    "t", "dt", "y_curr", "y_prop", "t_prop", "dt_next", "err_est",
    "params", "rhs", "sp", "ss", "sw0", "sw1", "sw2", "sw3",
}

# StructSpec field groups
LANE_FIELDS = ("sp_size", "ss_size", "sw0_size", "sw1_size", "sw2_size", "sw3_size")
RAW_INT_FIELDS = ("iw0_size", "bw0_size")

# Read-only inputs
READ_ONLY = {"t", "dt", "y_curr", "rhs", "params"}


class StepperASTVisitor(ast.NodeVisitor):
    """AST visitor to check stepper function for guardrails violations."""
    
    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.allowed_locals: Set[str] = set()
        self._bank_written: Dict[str, bool] = {name: False for name in EPHEMERAL_BANKS}
        self._persistence_warned: Set[str] = set()
    
    def _add_error(self, msg: str, node: ast.AST) -> None:
        line = getattr(node, "lineno", None)
        self.issues.append(ValidationIssue("error", msg, line))
    
    def _add_warning(self, msg: str, node: ast.AST) -> None:
        line = getattr(node, "lineno", None)
        self.issues.append(ValidationIssue("warning", msg, line))
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Check assignment targets and inspect RHS for dtype misuse."""
        for target in node.targets:
            self._check_assignment_target(target, node.value)
            self.visit(target)
        self.visit(node.value)
    
    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Check augmented assignment (+=, etc.)."""
        self._check_assignment_target(node.target, node.value)
        self.visit(node.target)
        self.visit(node.value)
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments."""
        self._check_assignment_target(node.target, node.value)
        self.visit(node.target)
        if node.value:
            self.visit(node.value)
        if node.annotation:
            self.visit(node.annotation)
    
    def _check_assignment_target(self, target: ast.AST, value_node: ast.AST | None) -> None:
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
                    if name in EPHEMERAL_BANKS:
                        self._bank_written[name] = True
                    if name in INT_BANKS:
                        self._check_int_bank_assignment(name, value_node, target)
                    else:
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

    def _check_int_bank_assignment(self, bank_name: str, value_node: ast.AST | None, parent: ast.AST) -> None:
        """Ensure iw0/bw0 only store integer/flag data."""
        if value_node is None:
            return
        if self._value_is_floaty(value_node):
            self._add_error(
                f"Bank '{bank_name}' stores persistent integers/flags; "
                f"assigning float/model-dtype data is forbidden",
                parent
            )

    def _value_is_floaty(self, node: ast.AST | None) -> bool:
        """Heuristic: detect expressions that obviously carry float/model-dtype data."""
        if node is None:
            return False
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (float, complex))
        if isinstance(node, ast.Name):
            return node.id in FLOATY_SOURCES
        if isinstance(node, ast.Attribute):
            return self._value_is_floaty(node.value)
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in FLOATY_SOURCES:
                return True
            return self._value_is_floaty(node.value)
        if isinstance(node, ast.BinOp):
            return self._value_is_floaty(node.left) or self._value_is_floaty(node.right)
        if isinstance(node, ast.UnaryOp):
            return self._value_is_floaty(node.operand)
        if isinstance(node, ast.BoolOp):
            return any(self._value_is_floaty(val) for val in node.values)
        if isinstance(node, ast.Compare):
            if self._value_is_floaty(node.left):
                return True
            return any(self._value_is_floaty(comp) for comp in node.comparators)
        if isinstance(node, ast.IfExp):
            return (
                self._value_is_floaty(node.body) or
                self._value_is_floaty(node.orelse)
            )
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return any(self._value_is_floaty(elt) for elt in node.elts)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in {"int", "range"}:
                    return False
                if node.func.id == "float":
                    return True
            return (
                any(self._value_is_floaty(arg) for arg in node.args)
                or any(self._value_is_floaty(kw.value) for kw in node.keywords)
            )
        return False

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Warn if ephemeral banks are read before being written (persistence misuse)."""
        if (
            isinstance(node.value, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.value.id in EPHEMERAL_BANKS
        ):
            bank = node.value.id
            if not self._bank_written.get(bank, False) and bank not in self._persistence_warned:
                self._add_warning(
                    f"Bank '{bank}' read before any writes; "
                    f"persistent data must live in 'ss' or the iw0/bw0 banks.",
                    node
                )
                self._persistence_warned.add(bank)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Track plain-name loads for persistence heuristics."""
        if (
            isinstance(node.ctx, ast.Load)
            and node.id in EPHEMERAL_BANKS
            and not self._bank_written.get(node.id, False)
            and node.id not in self._persistence_warned
        ):
            self._add_warning(
                f"Bank '{node.id}' read before any writes; "
                f"persistent data must move to 'ss' or iw0/bw0.",
                node
            )
            self._persistence_warned.add(node.id)
        # No children to visit for Name
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function-local definitions."""
        # Don't descend into nested functions
        # Just process this function's body
        for stmt in node.body:
            self.visit(stmt)


def validate_stepper_function(
    stepper_fn: Callable,
    stepper_name: str,
    struct_spec: StructSpec | None = None,
) -> List[ValidationIssue]:
    """
    Validate a stepper function against guardrails.
    
    Performs static analysis of the stepper function's AST to check:
    - No writes to forbidden runner-owned buffers
    - Only allowed outputs are written (y_prop, t_prop[0], dt_next[0], err_est[0])
    - Bank accesses use whole-lane slicing patterns
    - StructSpec lane counts and persistence flags are sane
    - iw0/bw0 are not fed float data
    - Ephemeral banks aren't (obviously) relied upon for persistence
    
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
    
    issues: List[ValidationIssue] = []
    if struct_spec is not None:
        issues.extend(_validate_struct_spec(struct_spec))
    
    # Visit and collect issues
    visitor = StepperASTVisitor()
    visitor.visit(tree)
    
    issues.extend(visitor.issues)
    return issues


def _validate_struct_spec(struct: StructSpec) -> List[ValidationIssue]:
    """Validate StructSpec sizes and persistence-related flags."""
    issues: List[ValidationIssue] = []
    
    for field_name in (*LANE_FIELDS, *RAW_INT_FIELDS):
        value = getattr(struct, field_name)
        if not isinstance(value, int):
            issues.append(
                ValidationIssue(
                    "error",
                    f"StructSpec field '{field_name}' must be an int (got {type(value).__name__})",
                    None,
                )
            )
            continue
        if value < 0:
            issues.append(
                ValidationIssue(
                    "error",
                    f"StructSpec field '{field_name}' must be non-negative (got {value})",
                    None,
                )
            )
    
    # Persistence flags imply certain storage
    if (struct.use_history or struct.use_f_history) and struct.ss_size == 0:
        issues.append(
            ValidationIssue(
                "error",
                "StructSpec declares use_history/use_f_history but ss_size==0. "
                "Persistent state must live in 'ss'.",
                None,
            )
        )
    if (struct.use_history or struct.use_f_history) and struct.iw0_size == 0:
        issues.append(
            ValidationIssue(
                "error",
                "StructSpec declares use_history/use_f_history but iw0_size==0. "
                "History rings need at least one iw0 slot for indices.",
                None,
            )
        )
    if struct.dense_output and struct.ss_size == 0:
        issues.append(
            ValidationIssue(
                "warning",
                "StructSpec enables dense_output but allocates no ss lanes; "
                "dense output typically needs persistent coefficients.",
                None,
            )
        )
    
    return issues


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
