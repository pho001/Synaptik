---
phase: 3
slug: materialization-aware-region-planning
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
updated: 2026-04-30
---

# Phase 3 - Validation Strategy

> Per-phase validation contract for materialization-aware region planning.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit 5 via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest` |
| **Full suite command** | `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests PreparedExecutionBuildTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest` |
| **Estimated runtime** | ~30-180 seconds for targeted tests, depending on Gradle warmup |

## Sampling Rate

- **After every task commit:** Run the quick targeted Gradle command for the touched area.
- **After every plan wave:** Run the full targeted Gradle command above.
- **Before `$gsd-verify-work`:** Run the full targeted command and `./gradlew classes`; run `./gradlew metalTest` only when native Metal execution code changed.
- **Max feedback latency:** 180 seconds for targeted feedback on a warm Gradle daemon.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 1 | PLAN-01 | T-3-01 | Cost summaries do not use profile-derived costs and clamp invalid byte/work inputs | unit | `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest` | yes | green |
| 3-01-02 | 01 | 1 | PLAN-02, PLAN-03 | T-3-02 | Planner records selected and top rejected finalists with bounded trace output | unit/compile | `./gradlew test --tests PreparedExecutionBuildTest --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest` | yes | green |
| 3-02-01 | 02 | 2 | PLAN-01, PLAN-03 | T-3-03 | Backend selection rejects costly accelerator regions before execution and reports stable reasons | unit/prepare | `./gradlew test --tests PreparedExecutionBuildTest` | yes | green |
| 3-02-02 | 02 | 2 | PLAN-03 | T-3-04 | Benchmark text/json reports expose compact planning cost summaries without dumping all candidates | unit | `./gradlew test --tests BenchmarkSessionTest` | yes | green |
| 3-03-01 | 03 | 3 | PLAN-04 | T-3-05 | CPU natural regions and CPU fusion/BLAS remain selectable when accelerator static benefit is not clear | regression | `./gradlew test --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest` | yes | green |
| 3-03-02 | 03 | 3 | PLAN-01, PLAN-02, PLAN-03, PLAN-04 | T-3-06 | End-to-end targeted suite compiles and preserves visible fallback diagnostics | integration | `./gradlew classes && ./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests PreparedExecutionBuildTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest` | yes | green |

*Status values: pending, green, red, flaky.*

## Wave 0 Requirements

- [x] Extend `src/test/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModelTest.java` for static cost summary math before or alongside score-model implementation.
- [x] Add planner/backend selection trace assertions in `src/test/java/PreparedExecutionBuildTest.java`.
- [x] Add renderer assertions in `src/test/java/BenchmarkSessionTest.java`.
- [x] Add or extend CPU guard tests in `src/test/java/graph/optimizer/partition/CpuNaturalExecutionRegionPlannerTest.java`, `src/test/java/OptimizerFuseTest.java`, and `src/test/java/BFloat16BlasDispatchTest.java`.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Native Metal runtime smoke | PLAN-02, PLAN-03 | Local native shim availability varies | Run `./gradlew metalTest` only if Phase 3 changes native Metal execution behavior; otherwise record "not applicable, Java planner/trace only" |

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing or changed test references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 180 seconds for targeted test commands.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** verified 2026-04-30

## Evidence

Validation status is based on the completed Phase 3 verification record in `003-VERIFICATION.md`, including the full targeted Gradle suite and Phase 1/2 regression coverage. `./gradlew metalTest` remains manual/not applicable for this phase because Phase 3 changed Java planner, trace, and reporting behavior, not native Metal execution code.

## Validation Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
