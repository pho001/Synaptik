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
| 3-01-01 | 01 | 1 | PLAN-01 | T-3-01 | Compile-time cost summaries use named static presets, clamp invalid byte/work inputs, and reject non-finite scores. Current prepare-time `PROFILE_DERIVED` factors are Phase 4-owned and covered through runtime-config tests. | unit | `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest` | yes | green |
| 3-01-02 | 01 | 1 | PLAN-02, PLAN-03 | T-3-02 | Planner records selected and top rejected finalists with bounded trace output | unit/compile | `./gradlew test --tests PreparedExecutionBuildTest --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest` | yes | green |
| 3-02-01 | 02 | 2 | PLAN-01, PLAN-03 | T-3-03 | Backend selection rejects costly accelerator regions before execution and reports stable reasons | unit/prepare | `./gradlew test --tests PreparedExecutionBuildTest` | yes | green |
| 3-02-02 | 02 | 2 | PLAN-03 | T-3-04 | Benchmark text/json reports expose compact planning cost summaries without dumping all candidates | unit | `./gradlew test --tests BenchmarkSessionTest` | yes | green |
| 3-03-01 | 03 | 3 | PLAN-04 | T-3-05 | CPU natural regions and CPU fusion/BLAS remain selectable when accelerator static benefit is not clear | regression | `./gradlew test --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest` | yes | green |
| 3-03-02 | 03 | 3 | PLAN-02, PLAN-03 | T-3-06, T-3-08 | Architecture docs and roadmap state Phase 3 static planning boundaries and Phase 4 profile/calibration-derived cost ownership. | docs/traceability | `rg -n "Materialization-aware region planning|Phase 3 uses static named presets only|Profile/calibration-derived costs are deferred to Phase 4" docs/architecture.md && rg -n "003-01-PLAN.md|003-02-PLAN.md|003-03-PLAN.md|profile/calibration-derived cost model update deferred from Phase 3" .planning/ROADMAP.md` | yes | green |
| 3-03-03 | 03 | 3 | PLAN-01, PLAN-02, PLAN-03, PLAN-04 | T-3-06 | End-to-end targeted suite compiles and preserves visible fallback diagnostics | integration | `./gradlew classes && ./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests PreparedExecutionBuildTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest` | yes | green |

*Status values: pending, green, red, flaky.*

## Wave 0 Requirements

- [x] Extend `src/test/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModelTest.java` for static cost summary math before or alongside score-model implementation.
- [x] Add planner/backend selection trace assertions in `src/test/java/PreparedExecutionBuildTest.java`.
- [x] Add renderer assertions in `src/test/java/BenchmarkSessionTest.java`.
- [x] Add or extend CPU guard tests in `src/test/java/graph/optimizer/partition/CpuNaturalExecutionRegionPlannerTest.java`, `src/test/java/OptimizerFuseTest.java`, and `src/test/java/BFloat16BlasDispatchTest.java`.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| None | All Phase 3 requirements | Phase 3 changed Java planner, trace, reporting, docs, and CPU guard behavior; all requirement-level behaviors have automated Gradle or source-check coverage. | N/A |

## Environment Notes

- `./gradlew metalTest` remains not applicable for Phase 3 because no native Metal execution code or native ABI files changed in this phase.
- After Phase 4, prepare-time `PROFILE_DERIVED` cost summaries are intentionally runtime-config derived. Phase 3 validation covers the static compile-time model, trace/report boundaries, CPU safeguards, and Phase 4 handoff; Phase 4 validation/security cover profile ownership and runtime-derived cost inputs.

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all missing or changed test references.
- [x] No watch-mode flags.
- [x] Feedback latency target is under 180 seconds for targeted test commands.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** verified 2026-04-30

## Evidence

Validation status is based on the completed Phase 3 verification record in `003-VERIFICATION.md`, including the full targeted Gradle suite and Phase 1/2 regression coverage. The 2026-04-30 re-audit also reran `./gradlew classes` and the Phase 3 targeted Gradle suite on the current tree.

## Validation Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

## Validation Re-Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Requirements audited | 4 |
| Plan tasks audited | 7 |
| Automated test files verified | 6 |
| Validation map corrections | 1 |
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

## Re-Audit Evidence

| Evidence | Result |
|----------|--------|
| `003-01-PLAN.md`, `003-02-PLAN.md`, and `003-03-PLAN.md` contain automated verification for all 7 plan tasks. | PASS |
| `AcceleratorPartitionScoreModelTest` covers static cost score math, rejected materialization cost, static presets, and runtime-derived `PROFILE_DERIVED` factor behavior added by Phase 4. | PASS |
| `PreparedExecutionBuildTest` covers compile/prepare cost summaries, stable backend rejection reasons, CPU fallback availability, and current `PROFILE_DERIVED` summaries. | PASS |
| `BenchmarkSessionTest` covers compact text/json `backendSelectionCost` and bounded `rejectedFinalists` output. | PASS |
| `CpuNaturalExecutionRegionPlannerTest`, `OptimizerFuseTest`, and `BFloat16BlasDispatchTest` cover CPU natural region, CPU fused execution, and BF16 BLAS safeguards. | PASS |
| `docs/architecture.md`, `.planning/ROADMAP.md`, and `.planning/PROJECT.md` preserve the Phase 3 static planning boundary and Phase 4 runtime/profile-derived cost ownership handoff. | PASS |
| Backend accelerator cost packages contain no direct `PlatformRuntimeProfileIO`, `JsonFileBestProfileStore`, `CalibrationArtifactLayout`, or `profiles/platform` path references. | PASS |
| `./gradlew classes` | PASS |
| `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests PreparedExecutionBuildTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest` | PASS |
