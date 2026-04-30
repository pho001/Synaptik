---
phase: 4
slug: tuning-and-profile-ownership-audit
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
updated: 2026-04-30
---

# Phase 4 - Validation Strategy

> Per-phase validation contract for tuning/profile ownership cleanup.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit 5 via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests CalibrationCandidateOwnershipTest --tests GraphAutotuneCandidateSpaceTest --tests PlatformCalibrationSessionTest --tests ExecutionProfileIoTest` |
| **Full suite command** | `./gradlew test --tests CalibrationCandidateOwnershipTest --tests GraphAutotuneCandidateSpaceTest --tests TuningKnobOwnershipTest --tests PlatformCalibrationSessionTest --tests ExecutionProfileIoTest --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest --tests synaptik.app.TuningCliParsingTest --tests TuningStoreTest --tests BenchmarkSessionTest` |
| **Estimated runtime** | ~30-180 seconds depending on Gradle warmup |

## Sampling Rate

- **After every task commit:** Run the task-specific targeted Gradle command.
- **After every plan wave:** Run the quick command or the wave-specific command named in the plan.
- **Before `$gsd-verify-work`:** Run the full targeted suite and `./gradlew classes`.
- **Max feedback latency:** 180 seconds for targeted feedback on a warm Gradle daemon.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 1 | TUNE-01, TUNE-02 | T-4-01, T-4-02 | Candidate spaces cannot mutate knobs outside their ownership class | unit | `./gradlew test --tests CalibrationCandidateOwnershipTest --tests GraphAutotuneCandidateSpaceTest --tests TuningKnobOwnershipTest` | yes | green |
| 4-01-02 | 01 | 1 | TUNE-01, TUNE-02 | T-4-03 | Ownership matrix documents graph/workload, platform/dtype, and obsolete knobs | docs/unit | `./gradlew test --tests TuningKnobOwnershipTest` | yes | green |
| 4-02-01 | 02 | 2 | TUNE-03 | T-4-04, T-4-05, T-4-06 | Invalid platform profile schema/version and accelerator buffer fields fail loudly | unit | `./gradlew test --tests PlatformCalibrationSessionTest --tests ExecutionProfileIoTest` | yes | green |
| 4-02-02 | 02 | 2 | TUNE-03 | T-4-04, T-4-05 | CLI/calibration profile loading reports strict IO failures instead of silently using fallback | unit | `./gradlew test --tests synaptik.app.TuningCliParsingTest --tests PlatformCalibrationSessionTest` | yes | green |
| 4-03-01 | 03 | 3 | TUNE-01, TUNE-02 | T-4-07, T-4-09 | Accelerator cost summaries consume only audited runtime/profile-derived factors from `RuntimeConfig` | unit | `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest` | yes | green |
| 4-03-02 | 03 | 3 | TUNE-01, TUNE-02 | T-4-08 | CPU safeguards remain intact when profile-derived accelerator cost factors are active | regression | `./gradlew test --tests PreparedExecutionBuildTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest` | yes | green |
| 4-04-01 | 04 | 4 | TUNE-04 | T-4-10 | Benchmark commands remain read-only for profile/best/history/calibration artifacts | unit | `./gradlew test --tests synaptik.app.TuningCliParsingTest --tests TuningStoreTest --tests BenchmarkSessionTest` | yes | green |
| 4-04-02 | 04 | 4 | TUNE-01, TUNE-02, TUNE-03, TUNE-04 | T-4-11, T-4-12 | Docs and final targeted suite prove ownership, strict IO, cost model, and write-boundary behavior | integration/docs | `./gradlew classes && ./gradlew test --tests CalibrationCandidateOwnershipTest --tests GraphAutotuneCandidateSpaceTest --tests TuningKnobOwnershipTest --tests PlatformCalibrationSessionTest --tests ExecutionProfileIoTest --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest --tests synaptik.app.TuningCliParsingTest --tests TuningStoreTest --tests BenchmarkSessionTest` | yes | green |

*Status values: pending, green, red, flaky.*

## Wave 0 Requirements

- [x] Existing JUnit/Gradle test infrastructure is present.
- [x] Existing ownership and graph autotune tests provide extension points.
- [x] Existing profile IO and benchmark tests provide extension points.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Native Metal runtime smoke | TUNE-01, TUNE-02 | Phase 4 changed tuning ownership, profile IO, cost selection, CLI metadata, docs, and Java tests rather than native Metal execution code. | Run `./gradlew metalTest` only if Phase 4 changes are rebased with native Metal execution changes. |

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing test references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 180 seconds for targeted test commands.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** verified 2026-04-30

## Evidence

Validation status is based on the completed Phase 4 summaries, `04-VERIFICATION.md`, `04-SECURITY.md`, and a fresh validation rerun:

- `./gradlew classes` - PASS
- `./gradlew test --tests CalibrationCandidateOwnershipTest --tests GraphAutotuneCandidateSpaceTest --tests TuningKnobOwnershipTest --tests PlatformCalibrationSessionTest --tests ExecutionProfileIoTest --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest --tests synaptik.app.TuningCliParsingTest --tests TuningStoreTest --tests BenchmarkSessionTest` - PASS
- `git status --short` - PASS for staged hygiene: no `profiles/platform/.../tuning/abc/*` or `.planning/tmp/` files were staged; those paths remain only as local unstaged artifacts.

## Validation Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
