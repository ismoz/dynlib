# Fix Plan for Critical Guardrails Violations

**Date:** November 5, 2025  
**Status:** Draft  
**Priority:** High (blocks advertised features)

---

## Executive Summary

Four critical gaps exist between the guardrails specification and implementation:

1. **[HIGH]** Event logging is completely non-functional
2. **[HIGH]** Semantic validation is not enforced at build time
3. **[MEDIUM]** Block-form equations are parsed but ignored in codegen
4. **[MEDIUM]** Mod verbs incomplete (aux/functions not supported)

All reviews are valid. This plan provides a systematic fix approach.

---

## Issue 1: Runner Never Services Event Log

### Problem
- Event cursor `m` initialized to 0, never incremented (runner.py:71)
- Event buffers `EVT_TIME`, `EVT_CODE`, `EVT_INDEX` never written
- `GROW_EVT` never returned when `m >= cap_evt`
- Violates guardrails lines 24-71: runner must own event recording

### Impact
- Any model with `log = ["x", "aux:E", ...]` in events silently fails
- Event logging feature completely broken despite infrastructure existing

### Root Cause
Runner implementation is incomplete stub; events infrastructure (buffers, results.py) exists but runner never uses it.

### Fix Strategy

#### Phase 1: Define Event Recording Interface
**Files:** `src/dynlib/compiler/codegen/emitter.py`, `src/dynlib/dsl/spec.py`

1. **Extend event compilation** to track which events request logging:
   ```python
   # In emitter.py, events should return metadata:
   # - event_index: int (0-based order in spec)
   # - log_vars: list of (kind, name) for "x", "aux:E", "param:a"
   ```

2. **Event code assignment**:
   - Assign each event a unique code (0-based index in spec.events)
   - Map event names → codes at build time

#### Phase 2: Implement Event Recording in Runner
**Files:** `src/dynlib/compiler/codegen/runner.py`

1. **Add helper function** (non-JIT, called from events):
   ```python
   def _log_event(m, cap_evt, EVT_TIME, EVT_CODE, EVT_INDEX, 
                  t, event_code, event_index):
       """Helper to record a single event. Returns new cursor or -1 if full."""
       if m >= cap_evt:
           return -1  # signal growth needed
       EVT_TIME[m] = t
       EVT_CODE[m] = event_code
       EVT_INDEX[m] = event_index  # 0 for non-indexed events
       return m + 1
   ```

2. **Modify runner main loop**:
   ```python
   # After pre-events:
   if events_pre_logged:  # flag from events_pre return
       m_new = _log_event(m, cap_evt, EVT_TIME, EVT_CODE, EVT_INDEX, 
                          t, event_code, 0)
       if m_new < 0:
           # Return GROW_EVT
           i_out[0] = i
           step_out[0] = step
           t_out[0] = t
           status_out[0] = GROW_EVT
           hint_out[0] = m
           return GROW_EVT
       m = m_new
   
   # Similar after post-events
   ```

3. **Update all exit points** to save `m` in `hint_out[0]`

#### Phase 3: Update Events Functions to Signal Logging
**Files:** `src/dynlib/compiler/codegen/emitter.py`

1. **Change events_pre/post signature** from:
   ```python
   def events_post(t, y_vec, params): ...
   ```
   To:
   ```python
   def events_post(t, y_vec, params, log_out):
       # log_out is int32[MAX_EVENTS] array
       # Set log_out[event_idx] = 1 if that event fired and has log=true
   ```

2. **Emit logging logic** in event bodies:
   ```python
   # For each event with log=true:
   if <cond>:
       <mutations>
       if record:  # from EventSpec.record
           log_out[EVENT_IDX] = 1
   ```

#### Phase 4: Testing
**New test file:** `tests/integration/test_event_logging.py`

1. Use `decay_with_event.toml` (already has `record = true`)
2. Add `log = ["x"]` to the reset event
3. Verify `result.EVT_TIME_view()` has entries when event fires
4. Test `GROW_EVT` return by setting tiny `cap_evt` and many events

### Files to Modify
- `src/dynlib/compiler/codegen/runner.py` (main implementation)
- `src/dynlib/compiler/codegen/emitter.py` (event codegen changes)
- `src/dynlib/runtime/wrapper.py` (may need GROW_EVT handling)
- `tests/integration/test_event_logging.py` (new)

### Estimated Effort
- **Development:** 6-8 hours
- **Testing:** 3-4 hours
- **Risk:** Medium (needs careful JIT compatibility testing)

---

## Issue 2: Required Semantic Validators Not Wired In

### Problem
- `validate_expr_acyclic` and `validate_event_legality` exist but never called
- Build path `load_model_from_uri → parse_model_v2 → build_spec` skips validation
- Cyclic aux/function deps and illegal event mutations go undetected

### Impact
- Invalid models compile and produce wrong results or runtime errors
- Users get poor error messages (deep in codegen or at runtime)

### Root Cause
Validators were implemented but never integrated into the build pipeline.

### Fix Strategy

#### Phase 1: Add Validation Call to Build Path
**File:** `src/dynlib/dsl/spec.py`

Modify `build_spec` to call validators before returning:

```python
def build_spec(normal: Dict[str, Any]) -> ModelSpec:
    # ... existing parsing ...
    
    # VALIDATE BEFORE FINALIZING SPEC
    from .astcheck import (
        validate_expr_acyclic,
        validate_event_legality,
        validate_functions_signature,
    )
    
    validate_expr_acyclic(normal)
    validate_event_legality(normal)
    validate_functions_signature(normal)
    
    # ... build and return ModelSpec ...
```

**Rationale:** `build_spec` is the canonical "validated spec constructor," so validation belongs there.

#### Phase 2: Add Name Uniqueness Check
**File:** `src/dynlib/dsl/schema.py`

The `validate_name_collisions` already exists and is called in `parse_model_v2`. Verify it covers:
- States, params, aux, functions, events (✓ already done)

No changes needed unless missing coverage found.

#### Phase 3: Testing
**File:** `tests/integration/test_semantic_validation.py` (new)

Test that build fails gracefully with clear errors for:
1. Cyclic aux: `aux.a = "b"`, `aux.b = "a"`
2. Cyclic functions: `f1` calls `f2`, `f2` calls `f1`
3. Illegal event mutation: `action.aux_var = "1"` (should fail)
4. Mixed cycles: aux depends on function that depends on aux

### Files to Modify
- `src/dynlib/dsl/spec.py` (add 3 lines of validation calls)
- `tests/integration/test_semantic_validation.py` (new, comprehensive tests)

### Estimated Effort
- **Development:** 1-2 hours (trivial code change)
- **Testing:** 2-3 hours (test invalid models)
- **Risk:** Low (pure validation, no runtime impact)

---

## Issue 3: Block-Form Equations Ignored

### Problem
- Guardrails promise both `[equations.rhs]` and `[equations].expr` forms
- Parser accepts both, stores in `ModelSpec.equations_block`
- Codegen only processes `equations_rhs`, ignores `equations_block`
- Models using block form get zero derivatives

### Impact
- Documented DSL feature doesn't work
- Silent failure (no error, just wrong simulation)

### Root Cause
Feature was specified and parsed but codegen was deferred ("for later" comment).

### Fix Strategy

#### Phase 1: Implement Block Parsing in Emitter
**File:** `src/dynlib/compiler/codegen/emitter.py`

Modify `_compile_rhs`:

```python
def _compile_rhs(spec: ModelSpec, nmap: NameMaps):
    import ast
    body: List[ast.stmt] = []
    
    # Case 1: Per-state RHS form
    if spec.equations_rhs:
        for sname, expr in spec.equations_rhs.items():
            idx = nmap.state_to_ix[sname]
            node = lower_expr_node(expr, nmap, aux_defs=_aux_defs(spec), 
                                   fn_defs=nmap.functions)
            assign = ast.Assign(
                targets=[ast.Subscript(value=ast.Name(id="dy_out", ctx=ast.Load()), 
                                       slice=ast.Constant(value=idx), ctx=ast.Store())],
                value=node,
            )
            body.append(assign)
    
    # Case 2: Block form
    if spec.equations_block:
        # Parse block: each line is "dx = <expr>" or "d(x) = <expr>"
        for line in sanitize_expr(spec.equations_block).splitlines():
            line = line.strip()
            if not line:
                continue
            
            # Parse "dx = expr" or "d(x) = expr"
            lhs, rhs = [p.strip() for p in line.split("=", 1)]
            
            # Extract state name: "dx" -> "x", "d(x)" -> "x"
            if lhs.startswith("d(") and lhs.endswith(")"):
                sname = lhs[2:-1].strip()
            elif lhs.startswith("d"):
                sname = lhs[1:].strip()
            else:
                raise ModelLoadError(
                    f"Block equation LHS must be 'dx' or 'd(x)', got: {lhs}"
                )
            
            if sname not in nmap.state_to_ix:
                raise ModelLoadError(
                    f"Unknown state in block equation: {sname}"
                )
            
            idx = nmap.state_to_ix[sname]
            node = lower_expr_node(rhs, nmap, aux_defs=_aux_defs(spec), 
                                   fn_defs=nmap.functions)
            assign = ast.Assign(
                targets=[ast.Subscript(value=ast.Name(id="dy_out", ctx=ast.Load()), 
                                       slice=ast.Constant(value=idx), ctx=ast.Store())],
                value=node,
            )
            body.append(assign)
    
    # ... rest of function unchanged ...
```

#### Phase 2: Validation - Prevent Duplicate Targets
**File:** `src/dynlib/dsl/spec.py` or `src/dynlib/dsl/astcheck.py`

Add validator to ensure a state isn't defined in both forms:

```python
def validate_no_duplicate_equation_targets(normal: Dict[str, Any]) -> None:
    """Ensure states aren't defined in both rhs and block forms."""
    rhs_targets = set(normal["equations"].get("rhs", {}).keys())
    
    if normal["equations"].get("expr"):
        block_targets = set()
        for line in normal["equations"]["expr"].splitlines():
            # Parse dx = ... to extract state name
            # (similar logic to emitter)
            # Add to block_targets
        
        overlap = rhs_targets & block_targets
        if overlap:
            raise ModelLoadError(
                f"States defined in both [equations.rhs] and [equations].expr: {overlap}"
            )
```

Call this in `build_spec` before finalizing.

#### Phase 3: Testing
**New test file:** `tests/unit/test_equations_block_form.py`

1. **Test block-only model**:
   ```toml
   [equations]
   expr = """
   dx = -a*x
   dy = x - b*y
   """
   ```

2. **Test mixed form** (if allowed):
   ```toml
   [equations.rhs]
   x = "-a*x"
   
   [equations]
   expr = "dy = x - b*y"
   ```

3. **Test invalid**: duplicate targets should fail

4. **Integration test**: Run simulation, verify derivatives computed correctly

### Files to Modify
- `src/dynlib/compiler/codegen/emitter.py` (block parsing logic)
- `src/dynlib/dsl/astcheck.py` (duplicate target validation, new function)
- `src/dynlib/dsl/spec.py` (call new validator)
- `tests/unit/test_equations_block_form.py` (new)
- `tests/integration/test_block_equations_sim.py` (new)

### Estimated Effort
- **Development:** 4-5 hours
- **Testing:** 3-4 hours
- **Risk:** Medium (parser complexity, edge cases)

---

## Issue 4: Incomplete Mod Verbs

### Problem
- Guardrails specify `set|add|replace|remove` for states, params, aux, functions, events
- Current implementation:
  - ✓ Events: add, replace, remove work
  - ✓ States/params: set works
  - ✗ Aux: no verbs implemented
  - ✗ Functions: no verbs implemented

### Impact
- Users cannot modify aux or functions via mods
- Limits composability and reuse

### Root Cause
Partial implementation; events prioritized, aux/functions deferred.

### Fix Strategy

#### Phase 1: Implement Missing Verbs for Aux
**File:** `src/dynlib/compiler/mods.py`

```python
def _apply_set(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    if not payload:
        return
    
    # Existing: states, params
    # ... existing code ...
    
    # NEW: aux
    a = payload.get("aux")
    if isinstance(a, dict):
        for k, v in a.items():
            if k not in normal["aux"]:
                raise ModelLoadError(f"set.aux.{k}: unknown aux variable")
            normal["aux"][k] = v


def _apply_add(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    # Existing: events
    # ... existing code ...
    
    # NEW: aux
    add_aux = payload.get("aux", {})
    if add_aux:
        existing = set(normal.get("aux", {}).keys())
        for k, v in add_aux.items():
            if k in existing:
                raise ModelLoadError(f"add.aux.{k}: aux already exists")
            normal.setdefault("aux", {})[k] = v


def _apply_replace(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    # Existing: events
    # ... existing code ...
    
    # NEW: aux
    repl_aux = payload.get("aux", {})
    if repl_aux:
        for k, v in repl_aux.items():
            if k not in normal.get("aux", {}):
                raise ModelLoadError(f"replace.aux.{k}: aux does not exist")
            normal["aux"][k] = v


def _apply_remove(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    # Existing: events.names
    # ... existing code ...
    
    # NEW: aux
    remove_aux = payload.get("aux", {}).get("names", [])
    if remove_aux:
        for k in remove_aux:
            if k in normal.get("aux", {}):
                del normal["aux"][k]
```

#### Phase 2: Implement Missing Verbs for Functions
**File:** `src/dynlib/compiler/mods.py`

Similar pattern to aux, but normalize function definition:

```python
def _normalize_function(name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Convert TOML function def to internal format."""
    args = body.get("args", [])
    expr = body.get("expr")
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ModelLoadError(f"functions.{name}.args must be list of strings")
    if not isinstance(expr, str):
        raise ModelLoadError(f"functions.{name}.expr must be a string")
    return {"args": list(args), "expr": expr}


def _apply_add(normal: Dict[str, Any], payload: Dict[str, Any]) -> None:
    # ... existing events, aux ...
    
    # NEW: functions
    add_funcs = payload.get("functions", {})
    if add_funcs:
        existing = set(normal.get("functions", {}).keys())
        for fname, fbody in add_funcs.items():
            if fname in existing:
                raise ModelLoadError(f"add.functions.{fname}: function already exists")
            normal.setdefault("functions", {})[fname] = _normalize_function(fname, fbody)

# Similar for replace, remove, set
```

#### Phase 3: Update Verb Order Enforcement
**File:** `src/dynlib/compiler/mods.py`

Current order is: `remove → replace → add → set`

Verify this works for aux/functions (it should; same semantics as events).

#### Phase 4: Testing
**New test file:** `tests/unit/test_mods_aux_functions.py`

1. **Test add aux**:
   ```toml
   [mod.add.aux]
   new_aux = "x + y"
   ```

2. **Test replace aux**:
   ```toml
   [mod.replace.aux]
   existing_aux = "new_expr"
   ```

3. **Test remove aux**:
   ```toml
   [mod.remove.aux]
   names = ["old_aux"]
   ```

4. **Test add/replace/remove functions** (similar)

5. **Test verb order**: remove then add same name should work

6. **Integration test**: Load model, apply mod that adds aux, verify it's used in RHS

### Files to Modify
- `src/dynlib/compiler/mods.py` (extend all 4 verb functions)
- `tests/unit/test_mods_aux_functions.py` (new, comprehensive)
- `tests/integration/test_mods_with_aux.py` (new, end-to-end)

### Estimated Effort
- **Development:** 5-6 hours
- **Testing:** 4-5 hours
- **Risk:** Low-Medium (similar to existing event verbs)

---

## Implementation Timeline

### Phase 1: Low-Hanging Fruit (Week 1)
**Priority:** Fix validation wiring (Issue 2)

- **Day 1-2:** Add validator calls to `build_spec` + comprehensive tests
- **Deliverable:** Semantic validation enforced, clear error messages
- **Risk:** Minimal

### Phase 2: DSL Completeness (Week 2)
**Priority:** Block equations (Issue 3) and mod verbs (Issue 4)

- **Day 3-5:** Implement block-form equation parsing in emitter
- **Day 6-7:** Implement aux/function mod verbs
- **Deliverable:** Full DSL feature parity with guardrails
- **Risk:** Medium (parser edge cases)

### Phase 3: Event Logging (Week 3-4)
**Priority:** Event log servicing (Issue 1)

- **Day 8-12:** Design and implement event recording in runner
- **Day 13-14:** Modify event codegen to signal logging
- **Day 15-16:** Integration testing, JIT compatibility
- **Deliverable:** Event logging fully functional
- **Risk:** High (ABI changes, JIT constraints, growth protocol)

---

## Testing Strategy

### Unit Tests (Isolated)
- Validators reject invalid models gracefully
- Block equation parser handles edge cases
- Mod verbs apply correctly in isolation

### Integration Tests (End-to-End)
- Build → compile → run models using all fixed features
- Event logging produces correct results
- Mods with aux/functions propagate through build

### Regression Tests
- Existing tests must pass (no breaking changes)
- `examples/uri_demo.py` still works

---

## Risks and Mitigations

### Risk 1: JIT Compatibility (Issue 1)
**Concern:** Event logging adds complexity to hot path

**Mitigation:**
- Keep logging logic minimal (single array writes)
- Profile JIT overhead before/after
- Make event logging optional if needed (cap_evt=1 to disable)

### Risk 2: Breaking Changes (All Issues)
**Concern:** Fixes might break existing (working) models

**Mitigation:**
- All fixes are additive (enable missing features) except Issue 2
- Issue 2 only rejects *invalid* models (good breakage)
- Comprehensive regression testing

### Risk 3: Complexity Creep (Issue 3)
**Concern:** Block equation parser could get complicated

**Mitigation:**
- Keep parser simple: only support `dx = expr` and `d(x) = expr`
- Reject complex forms (multi-statement lines, etc.)
- Clear error messages for unsupported syntax

---

## Success Criteria

### Issue 1 (Event Logging)
- [ ] `decay_with_event.toml` with `log = ["x"]` produces non-empty `EVT_TIME_view()`
- [ ] `GROW_EVT` returned when event capacity exceeded
- [ ] No performance regression (< 5% overhead with logging disabled)

### Issue 2 (Validation)
- [ ] Cyclic aux model fails at build with clear error
- [ ] Illegal event mutation fails at build with clear error
- [ ] All existing valid models still build

### Issue 3 (Block Equations)
- [ ] Model with only `[equations].expr` runs correctly
- [ ] Mixed form allowed if no duplicate targets
- [ ] Clear error if duplicate targets

### Issue 4 (Mod Verbs)
- [ ] Can add/replace/remove aux via mods
- [ ] Can add/replace/remove functions via mods
- [ ] Verb order (remove→replace→add→set) enforced correctly

---

## Open Questions

1. **Event logging performance**: Should we batch event writes or keep per-event? (Recommend: per-event, simpler)

2. **Block equations mixed form**: Allow or forbid? (Recommend: allow if no overlap, easier migration)

3. **Mod verb priority**: Should aux/function verbs respect group/exclusive like events? (Recommend: yes, same rules)

4. **Validation strictness**: Make semantic validation opt-out? (Recommend: always on, opt-out is footgun)

---

## Conclusion

All four reviews are **valid and critical**. The fix plan is structured to:

1. **Quick wins first** (Issue 2: validation wiring, ~2 days)
2. **Medium complexity** (Issues 3, 4: DSL completeness, ~1 week)
3. **Complex last** (Issue 1: event logging, ~2 weeks)

Total estimated effort: **3-4 weeks** for one developer, or **1.5-2 weeks** with two developers (parallelizable after Phase 1).

**Recommendation:** Prioritize Issues 2 and 4 first (validation + mod verbs) as they're low-risk and high-value. Then tackle Issue 3 (block equations). Finally, Issue 1 (event logging) as a separate focused sprint.
