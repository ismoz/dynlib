# Known Issues

## Issue 1: Duplicate ODE dtype validation ⚠️ DUPLICATION

**Files:**
- `schema.py` (line 51)
- `astcheck.py` (line 144)

Both files validate that ODE models require floating dtypes:

```python
# schema.py:51
if mtype == "ode" and not _is_float_dtype(dtype):
    raise ModelLoadError("ODE models require a floating dtype...")

# astcheck.py:144
if mtype == "ode":
    if dtype not in {"float32", "float64", "float16", "bfloat16"}:
        raise ModelLoadError("ODE models require a floating dtype...")
```

**Recommendation:**
Remove the validation from `astcheck.py:validate_dtype_rules()` since `schema.py:validate_model_header()` already enforces this during structural validation. This keeps validation layered properly (structural → semantic).

---

## Issue 2: Duplicate equation target validation ⚠️ DUPLICATION

**Files:**
- `schema.py:validate_name_collisions()`
- `astcheck.py:validate_equation_targets()`

Both check for:
- Unknown targets (not in `[states]`)
- Duplicates across RHS and block expressions

**Current Flow:**
- `schema.py:validate_name_collisions()` ← called by parser
- `astcheck.py:validate_equation_targets()` ← standalone, not called by parser

**Recommendation:**
Since `validate_name_collisions()` is already called by `parse_model_v2()`, the `validate_equation_targets()` function in `astcheck.py` is redundant unless you plan to use it separately. Consider either:
1. Removing `validate_equation_targets()` entirely, OR
2. Moving both checks into `astcheck.py` and calling it from the parser.

---

## Issue 3: Unused import in `errors.py` 🐛 BUG

**File:**
- `errors.py` (line 3)

```python
from email.mime import message  # ← UNUSED
```

**Recommendation:**
This import should be removed.