---
phase: 21
slug: milestone-verification-evidence-closure
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
updated: 2026-05-01
---

# Phase 21 — Validation Strategy

Per-phase validation contract for milestone verification evidence closure.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Gradle + JUnit |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew classes` |
| **Full suite command** | `./gradlew test --tests GpuCoverageGapTriageTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageTriageReportTest --tests BenchmarkSuiteSessionTest --tests GpuCoverageSummaryTest && ./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests graph.optimizer.region.DefaultRegionOptimizerTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests backend.cpu.fused.plan.LoweredFusedOperationBuilderTest --tests SourceTreeHygieneTest && ./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests CompiledGraphTraceTest --tests SourceTreeHygieneTest` |
| **Estimated runtime** | Previously passed during Phase 21 execution; exact runtime environment-dependent |

## Sampling Rate

- **After every task commit:** Run the task's artifact grep or focused Gradle command from `21-01-PLAN.md`.
- **After every plan wave:** Run the three focused Phase 14, Phase 18, and Phase 20 Gradle command groups recorded in `21-VERIFICATION.md`.
- **Before milestone audit:** Confirm `.planning/v1.3-MILESTONE-AUDIT.md` contains no `missing_verification` finding and cites `14-VERIFICATION.md`, `18-VERIFICATION.md`, and `20-VALIDATION.md`.
- **Max feedback latency:** Bounded by focused Gradle command groups; no watch-mode commands are required.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 21-01-01 | 01 | 1 | GPUAUDIT-01 | T-21-01 / T-21-04 | Phase 14 evidence is proven from existing artifacts without production runtime changes. | artifact + regression | `test -f .planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-VERIFICATION.md && rg -n "GPUTRIAGE-01|GPUTRIAGE-02|GPUTRIAGE-03|SATISFIED" .planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-VERIFICATION.md && ./gradlew test --tests GpuCoverageGapTriageTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageTriageReportTest --tests BenchmarkSuiteSessionTest --tests GpuCoverageSummaryTest` | yes | green |
| 21-01-02 | 01 | 1 | GPUAUDIT-02 | T-21-01 / T-21-04 | Phase 18 evidence is proven from existing artifacts without production runtime changes. | artifact + regression | `test -f .planning/phases/18-fused-elementwise-and-epilogue-subregions/18-VERIFICATION.md && rg -n "GPUFUSEX-01|GPUFUSEX-02|GPUFUSEX-03|SATISFIED" .planning/phases/18-fused-elementwise-and-epilogue-subregions/18-VERIFICATION.md && ./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests graph.optimizer.region.DefaultRegionOptimizerTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests backend.cpu.fused.plan.LoweredFusedOperationBuilderTest --tests SourceTreeHygieneTest` | yes | green |
| 21-01-03 | 01 | 1 | GPUAUDIT-03 | T-21-03 | Phase 20 validation metadata is unambiguous and audit-readable. | artifact + regression | `rg -n "status: verified|nyquist_compliant: true|wave_0_complete: true|Execution Evidence|GpuCoverageRegressionGateTest|SourceTreeHygieneTest" .planning/phases/20-coverage-regression-hardening/20-VALIDATION.md && ./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests CompiledGraphTraceTest --tests SourceTreeHygieneTest` | yes | green |
| 21-01-04 | 01 | 1 | GPUAUDIT-03 | T-21-02 / T-21-04 | The refreshed milestone audit has no stale missing verification findings, and profile artifacts remain non-canonical. | artifact hygiene | `! rg -n "missing_verification" .planning/v1.3-MILESTONE-AUDIT.md && rg -n "14-VERIFICATION.md|18-VERIFICATION.md|GPUTRIAGE-01|GPUFUSEX-01|nyquist_compliant" .planning/v1.3-MILESTONE-AUDIT.md && git diff --quiet -- src/main/java src/test/java docs && git diff --cached --name-only -- profiles/platform` | yes | green |

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. Phase 21 is audit evidence closure only and does not require new test files.

## Manual-Only Verifications

All phase behaviors have automated verification.

## Validation Audit 2026-05-01

| Metric | Count |
|--------|-------|
| Requirements audited | 3 |
| Tasks audited | 4 |
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

## Execution Evidence

Phase 21 validation relies on the same focused evidence groups already recorded in `21-VERIFICATION.md`:

```bash
./gradlew classes
./gradlew test --tests GpuCoverageGapTriageTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageTriageReportTest --tests BenchmarkSuiteSessionTest --tests GpuCoverageSummaryTest
./gradlew test --tests backend.accelerator.lowering.GpuCompoundPatternDetectorTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests graph.optimizer.region.DefaultRegionOptimizerTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests backend.cpu.fused.plan.LoweredFusedOperationBuilderTest --tests SourceTreeHygieneTest
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests CompiledGraphTraceTest --tests SourceTreeHygieneTest
gsd-sdk query verify.schema-drift 21
! rg -n "missing_verification" .planning/v1.3-MILESTONE-AUDIT.md
git diff --quiet -- src/main/java src/test/java docs
git diff --cached --name-only -- profiles/platform
```

`profiles/platform/.../tuning/abc/* remained unstaged`.

## Validation Sign-Off

- [x] All tasks have automated verify commands or artifact checks.
- [x] Sampling continuity: no 3 consecutive tasks without automated verify.
- [x] Wave 0 covers all MISSING references; no missing references remain.
- [x] No watch-mode flags.
- [x] Feedback latency is bounded by focused Gradle and grep commands.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-01
