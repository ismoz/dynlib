
- Steppers are self-contained. They should contain reject-retry logic.

- Steppers should detect `STEPFAIL` cases. On success they should return `OK` (step accepted).

- Only adaptive steppers need to check `NAN_DETECTED` (NaN/Inf) cases in their step size determination loops. Runners already have the same checks so fixed-step steppers do not need these checks.