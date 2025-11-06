
- Steppers are self-contained. They should contain reject-retry logic.

- Steppers should detect `NAN_DETECTED`, `STEPFAIL` cases. On success they should return `OK` (step accepted).