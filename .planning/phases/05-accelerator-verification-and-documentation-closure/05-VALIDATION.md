---
phase: 05
slug: accelerator-verification-and-documentation-closure
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
---

# Phase 05 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | JUnit Jupiter 5.11.2 via Gradle |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests BenchmarkSessionTest --tests StandardWorkloadsTest` |
| Full suite command | `./gradlew classes && ./gradlew test --tests BenchmarkSessionTest --tests StandardWorkloadsTest --tests SourceTreeHygieneTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest && ./gradlew metalTest` |
| Estimated runtime | Native Metal runtime dependent; use targeted filters for iteration |

## Sampling Rate

- After every task commit: run the task's targeted Gradle command.
- After every plan wave: run the plan's verification command.
- Before `$gsd-verify-work`: run final closure commands listed above.
- Max feedback latency: one focused Gradle filter per task before wider closure.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | OBS-01 | T-5-01 | Fallback/materialization/report evidence cannot silently disappear | unit/integration | `./gradlew test --tests BenchmarkSessionTest --tests PreparedExecutionBuildTest` | yes | pending |
| 05-01-02 | 01 | 1 | OBS-01 | T-5-02 | Report contract covers planner context and accelerator evidence without measured local output | unit | `./gradlew test --tests BenchmarkSessionTest` | yes | pending |
| 05-02-01 | 02 | 2 | OBS-02 | T-5-03 | Closure workload stresses required operation families and gradients | unit | `./gradlew test --tests StandardWorkloadsTest --tests BenchmarkSessionTest` | yes | pending |
| 05-02-02 | 02 | 2 | OBS-03, OBS-04 | T-5-04 | Metal evidence is real when native shim is available and skipped otherwise | native/integration | `./gradlew metalTest` | yes | pending |
| 05-03-01 | 03 | 3 | DOC-01, DOC-02, DOC-03 | T-5-05 | Docs do not hide fallback or overclaim CUDA/native support | docs grep | `rg -n "device-owned|CPU materialization|fallback diagnostics|benchmark reports" docs/metal-backend.md docs/calibration-autotune.md docs/testing.md docs/troubleshooting.md docs/architecture.md` | yes | pending |
| 05-03-02 | 03 | 3 | DOC-04 | T-5-06 | Local scratch/profile artifacts are ignored or rejected before commit | hygiene | `./gradlew test --tests SourceTreeHygieneTest && ./gradlew verifySourceTreeClean` | yes | pending |

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Native Metal runtime availability | OBS-03, OBS-04 | Local macOS shim and hardware are optional | Run `./gradlew metalTest`; skipped tests are acceptable only when assumptions report missing native shim/library. |

## Validation Sign-Off

- [x] All tasks have automated verification commands or a documented native assumption.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing references.
- [x] No watch-mode flags.
- [x] Feedback latency is bounded through targeted Gradle filters.
- [x] `nyquist_compliant: true` set in frontmatter.

Approval: approved 2026-04-30
