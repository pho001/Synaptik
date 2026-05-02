---
phase: 46-cross-backend-router-calibration-and-regression-gates
status: verification
created: 2026-05-02
---

# Phase 46 Verification

## Completed Checks

- `./gradlew test --tests CrossBackendRouterEvidenceTest`
- `./gradlew test --tests CrossBackendRouterEvidenceTest --tests BenchmarkSessionTest`
- `./gradlew test --tests CrossBackendRouterEvidenceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests GpuCoverageRegressionGateTest`
- `./gradlew classes`
- `./gradlew metalTest`
- `git diff --check`

## Artifact Hygiene

Local tuning profile artifacts under `profiles/platform/.../tuning/` remain unstaged and are not Phase 46 source evidence.
