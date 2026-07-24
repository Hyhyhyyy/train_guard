## Summary

- Describe the problem and the focused change.

## Validation

- [ ] Tests added or updated for observable behavior
- [ ] `ruff check .`
- [ ] `mypy src/train_guard scripts`
- [ ] `pytest --cov=train_guard`
- [ ] Tracked-source privacy scan
- [ ] Single-file regeneration and drift check, when applicable

## Compatibility and privacy

- [ ] Required runtime dependencies remain empty
- [ ] Read-only and telemetry-free boundaries remain intact
- [ ] No credentials, private paths, real data, reports, logs, or training artifacts are included
- [ ] User-facing documentation and changelog are updated when needed
- [ ] `release/train_guard.py` was not hand-edited
