# Roadmap

## 0.6 release candidates: prove first-run reliability

- Validate installation and one monitored synthetic run with 5–10 independent users.
- Add CPU-only live-training and slow-checkpoint benchmark scenarios.
- Add DeepSpeed-style synthetic log fixtures without making DeepSpeed a base dependency.
- Keep Linux, Windows, macOS, Python 3.10–3.14, privacy, package, and single-file gates green.

## 0.7: integration depth

- Stabilize versioned event and run-manifest schemas.
- Add documented adapter contracts and representative distributed-training fixtures.
- Publish benchmark history while keeping synthetic and real-world evidence separate.
- Add opt-in, privacy-preserving notification integrations.

## 1.0 criteria

- Ten independent clean-environment installs and five real training pilots.
- Recovery actions remain opt-in, bounded, persisted, and auditable.
- Versioned schemas have migration guidance and compatibility tests.
- At least two external contributors complete a meaningful issue or pull request.
- No unresolved critical correctness, privacy, or dependency vulnerability.

Release scope and commands are documented in [docs/RELEASE.md](docs/RELEASE.md).
