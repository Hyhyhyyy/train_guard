# Security Policy

## Supported versions

During the 0.6.0 candidate cycle, security fixes target the newest 0.6.0 prerelease and the
latest stable minor when maintainers judge backporting practical. Release candidates are not
recommended for environments that require stable support guarantees.

## Reporting a vulnerability

Do not open a public issue. Use GitHub's private vulnerability reporting for this repository:

https://github.com/Hyhyhyyy/train_guard/security/advisories/new

Include the affected version, impact, minimal reproduction, and any suggested mitigation.
Remove credentials, private paths, real records, model data, and training artifacts from the
report. Maintainers aim to acknowledge a report within seven days and will coordinate
disclosure after a fix is available.

## Scope

Relevant reports include unsafe file writes, privacy-redaction bypasses, command injection,
credential disclosure, release-boundary escapes, and dependency or workflow supply-chain
issues. General support questions and non-security defects should follow [SUPPORT.md](SUPPORT.md).

Train Guard is a diagnostic aid, not a security sandbox or compliance certification.
The loopback Web dashboard is read-only by default. Control mode requires an ephemeral
in-memory authorization value, local origin checks, and a supervised process. It must not be
publicly exposed. Controlled recovery
does not make untrusted training commands or checkpoints safe.
