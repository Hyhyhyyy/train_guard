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
5. The publish workflow repeats all gates, publishes through PyPI trusted publishing, and
   creates a GitHub prerelease with all artifacts.

Before tagging, a maintainer must configure a PyPI project named `train-guard`, add a trusted
publisher for this repository's `publish.yml` workflow and `pypi` environment, and configure
that protected GitHub environment. These external settings cannot be verified from the
repository. Missing or mismatched trusted-publisher settings block PyPI publication.

GitHub Actions and third-party actions are pinned. The release job requires OIDC
`id-token: write` for PyPI and `contents: write` for the GitHub prerelease; no other workflow
job receives those write permissions.

## Verification

Consumers should compare downloads with `SHA256SUMS`. Maintainers should install the wheel in
a fresh virtual environment and run both `train-guard --version` and a help command. The SBOM
describes the isolated wheel environment and is not a vulnerability attestation.
