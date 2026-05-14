# Release Process

Navigation: [Index](index.md#recommended-reading-paths) | [README](../README.md) | [Changelog](../CHANGELOG.md) | [Testing](testing.md#exact-commands) | [ONNX](onnx.md#onnx-import-and-export)

This document describes the release hygiene expected for the `0.x` public
preview line.

## Current Release Line

Current version:

```text
0.1.0-alpha.2
```

The version source of truth is the repository-root `VERSION` file. `build.gradle`
reads that file for `project.version`.

## Release Meaning

`0.1.0-alpha.2` means:

- public technical preview;
- APIs are allowed to change before `1.0.0`;
- CPU execution is the correctness baseline;
- ONNX support targets the documented static dense inference subset;
- accelerator support is scoped, capability-gated, and trace-visible;
- performance claims require local benchmark evidence and must not be inferred
  from coverage rows alone.

## Required Files

A public-preview release should include:

- `VERSION`;
- `CHANGELOG.md`;
- `README.md`;
- `LICENSE.md`;
- `.github/workflows/ci.yml`;
- `docs/onnx.md`;
- `docs/onnx-coverage.md`;
- `docs/testing.md`;

## Verification Gates

Portable public-preview gate:

```bash
./gradlew classes
./gradlew test --tests 'onnx.*' --tests SourceTreeHygieneTest
git diff --check
```

Broader local gate when time allows:

```bash
./gradlew test
```

Optional hardware-specific gates:

```bash
./gradlew metalTest
./gradlew cudaTest
```

`metalTest` and `cudaTest` are not portable release blockers unless the release
claim explicitly includes the corresponding native hardware lane. They are still
the correct gates before making backend-specific native execution claims.

## Artifact Hygiene

Do not commit local benchmark or calibration output unless the release explicitly
promotes it as canonical evidence.

Usually exclude:

```text
profiles/platform/<platform-id>/tuning/abc/*-best-profile.json
profiles/platform/<platform-id>/tuning/abc/*-history.jsonl
profiles/platform/<platform-id>/tuning/<local-workload>/
build/
.plan
.planning/
todo/
```

Checked-in fixtures are acceptable only when they are deterministic, documented,
and consumed by tests.

## Tagging

Recommended tag format:

```bash
git tag -a v0.1.0-alpha.2 -m "Synaptik 0.1.0-alpha.2"
git push origin v0.1.0-alpha.2
```

Before tagging, ensure `CHANGELOG.md` names the exact version and date, CI is
green, and the working tree does not include unrelated local profiles.

## License Boundary

`LICENSE.md` currently states that no open-source license has been granted.
Before publishing Synaptik as an externally reusable open-source library, choose
and commit an explicit license such as Apache-2.0, MIT, BSD-3-Clause, or another
license that matches the intended distribution model.
