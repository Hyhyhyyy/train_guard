# Release engineering — 0.6.0rc1

## Artifact boundary

The release publishes four artifact classes:

1. a wheel;
2. an sdist;
3. the generated single-file `train_guard.py`; and
4. `SHA256SUMS` plus a CycloneDX JSON SBOM.

`MANIFEST.in` defines the sdist boundary: runtime package source, project/license documents,
topic docs, example/public release configuration, and release-gate scripts. It excludes
`.github`, `release/`, benchmarks, local build output, and the full test tree. The wheel
contains only installed package metadata and `train_guard`.

The single-file boundary is the fixed allowlist in `configs/release_manifest.json`.
Each allowed path has a SHA256 digest. Generate it only after intentionally regenerating the
candidate, then validate without mutation:

```bash
python scripts/bundle_singlefile.py
python scripts/check_release_manifest.py --generate
python scripts/check_release_manifest.py
python scripts/privacy_scan.py --mode release \
  --allowlist configs/privacy_scan_allowlist.json
```

Review both generated diffs. `release/train_guard.py` is generated and never hand-edited.

## Tag and publish flow

1. Set the source version and all release docs to `0.6.0rc1`.
2. Regenerate the single file and fixed hash manifest.
3. Run full pytest, Ruff, mypy, privacy scans, manifest validation, build, metadata checks,
   isolated wheel smoke, dependency audit, and sdist-boundary tests.
4. Tag exactly `v0.6.0rc1`.
5. The publish workflow repeats all gates and creates a GitHub prerelease with all artifacts.

The `train-guard` project name on PyPI is owned by an unrelated project. The distribution name
is therefore `llm-train-guard`; the import package remains `train_guard` and the CLI remains
`train-guard`. GitHub prerelease artifacts and explicit source installation are the supported
distribution channels until the maintainer registers and configures the distinct PyPI project.

GitHub Actions and third-party actions are pinned. The release job requires only
`contents: write` for the GitHub prerelease; no other workflow job receives write permissions.

## Verification

Consumers should compare downloads with `SHA256SUMS`. Maintainers should install the wheel in
a fresh virtual environment and run both `train-guard --version` and a help command. The SBOM
describes the isolated wheel environment and is not a vulnerability attestation.
