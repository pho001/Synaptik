---
phase: 14
slug: coverage-gap-triage-and-hot-path-targets
status: draft
nyquist_compliant: false
wave_0_complete: true
created: 2026-05-01
---

# Phase 14 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter 5.11.2 via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests GpuCoverageGapTriageTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageTriageReportTest` |
| **Full suite command** | `./gradlew classes && ./gradlew test --tests GpuCoverageGapTriageTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageTriageReportTest --tests BenchmarkSuiteSessionTest --tests GpuCoverageSummaryTest` |
| **Estimated runtime** | ~90 seconds focused; native checks are not required for Phase 14 |

## Sampling Rate

- **After every task commit:** Run the focused test class for the touched area.
- **After every plan wave:** Run the quick run command.
- **Before `$gsd-verify-work`:** Run the full suite command and `git status --short`.
- **Max feedback latency:** 120 seconds for focused tests.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 14-01-01 | 01 | 1 | GPUTRIAGE-01 | T-14-01 | Triage reports cannot hide CPU exits behind generic accelerator presence. | unit | `./gradlew test --tests GpuCoverageGapTriageTest` | W0 | pending |
| 14-01-02 | 01 | 1 | GPUTRIAGE-01, GPUTRIAGE-03 | T-14-02 | Ranked gaps preserve raw counts and stable reasons. | unit | `./gradlew test --tests GpuCoverageGapTriageTest --tests GpuCoverageSummaryTest` | W0 | pending |
| 14-02-01 | 02 | 2 | GPUTRIAGE-02 | T-14-03 | Hot-path names are checked against `StandardWorkloads.defaultCatalog()`. | unit | `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSuiteSessionTest` | W0 | pending |
| 14-02-02 | 02 | 2 | GPUTRIAGE-02, GPUTRIAGE-03 | T-14-04 | Target metadata maps workloads to downstream requirement families. | unit | `./gradlew test --tests GpuHotPathCoverageTargetsTest` | W0 | pending |
| 14-03-01 | 03 | 3 | GPUTRIAGE-01, GPUTRIAGE-03 | T-14-05 | Triage report text and JSON have stable contract fields. | unit | `./gradlew test --tests GpuCoverageTriageReportTest` | W0 | pending |
| 14-03-02 | 03 | 3 | GPUTRIAGE-01, GPUTRIAGE-02, GPUTRIAGE-03 | T-14-06 | Checked-in target list records evidence and downstream owner phases. | rg/docs | `rg -n "GPU Coverage Gap Triage|Hot Path Targets|Downstream Phase Targets|profiles/platform" .planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md docs/gpu-coverage-triage.md` | W0 | pending |
| 14-04-01 | 04 | 4 | GPUTRIAGE-01, GPUTRIAGE-02, GPUTRIAGE-03 | T-14-07 | Final docs and summary distinguish canonical evidence from local artifacts. | docs/test | `./gradlew classes && ./gradlew test --tests GpuCoverageGapTriageTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageTriageReportTest --tests BenchmarkSuiteSessionTest --tests GpuCoverageSummaryTest` | W0 | pending |

## Wave 0 Requirements

Existing infrastructure covers all phase requirements:

- `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java`
- `src/main/java/tuning/benchmark/report/GpuCoverageRegressionGate.java`
- `src/main/java/tuning/workload/StandardWorkloads.java`
- `src/test/java/GpuCoverageSummaryTest.java`
- `src/test/java/BenchmarkSuiteSessionTest.java`

## Manual-Only Verifications

All Phase 14 behaviors have automated verification. Native CUDA execution remains optional and capability-gated outside this phase.

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies.
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify.
- [ ] Wave 0 covers all existing references.
- [ ] No watch-mode flags.
- [ ] Feedback latency < 120s for focused tests.
- [ ] `nyquist_compliant: true` set in frontmatter after execution evidence is recorded.

**Approval:** pending
