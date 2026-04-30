---
phase: 05
slug: accelerator-verification-and-documentation-closure
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-30
updated: 2026-04-30
---

# Phase 05 - Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | JUnit Jupiter 5.11.2 via Gradle |
| **Config file** | `build.gradle` |
| **Quick run command** | `./gradlew test --tests BenchmarkSessionTest --tests PreparedExecutionBuildTest --tests StandardWorkloadsTest` |
| **Full suite command** | `./gradlew classes && ./gradlew test --tests BenchmarkSessionTest --tests PreparedExecutionBuildTest --tests StandardWorkloadsTest --tests SourceTreeHygieneTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest && ./gradlew metalTest` |
| **Estimated runtime** | Native Metal runtime dependent; use targeted filters for iteration |

## Sampling Rate

- After every task commit: run the task's targeted Gradle command.
- After every plan wave: run the plan's verification command.
- Before `$gsd-verify-work`: run final closure commands listed above.
- Max feedback latency: one focused Gradle filter per task before wider closure.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 05-01-01 | 01 | 1 | OBS-01 | T-5-01 | Fallback/materialization/report evidence cannot silently disappear | unit/integration | `./gradlew test --tests BenchmarkSessionTest --tests PreparedExecutionBuildTest` | yes | green |
| 05-01-02 | 01 | 1 | OBS-01, OBS-02 | T-5-02, T-5-03 | Report contract covers planner context and accelerator evidence without measured local output | unit | `./gradlew test --tests BenchmarkSessionTest` | yes | green |
| 05-02-01 | 02 | 2 | OBS-02 | T-5-04 | Closure workload stresses required operation families and gradients | unit | `./gradlew test --tests StandardWorkloadsTest --tests BenchmarkSessionTest` | yes | green |
| 05-02-02 | 02 | 2 | OBS-03, OBS-04 | T-5-05, T-5-06, T-5-07 | Metal/CUDA evidence is capability-gated and adjacent device-buffer handoff avoids Java array round trips | native/integration | `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest && ./gradlew metalTest` | yes | green |
| 05-03-01 | 03 | 3 | DOC-01, DOC-02, DOC-03 | T-5-08, T-5-09 | Docs do not hide fallback, confuse report/profile ownership, or overclaim CUDA/native support | docs grep | `rg -n "device-owned|acceleratorBufferExecutionPath|acceleratorBufferReasonCode|CPU materialization boundary|storageResidency|nativeDeviceCopyNs|CUDA remains capability-gated|report-contract|Benchmark reports are explain artifacts|selected accelerator candidate|rejectedFinalists|boundaryCount|benchmark commands remain read-only" docs/metal-backend.md docs/calibration-autotune.md docs/testing.md docs/troubleshooting.md docs/architecture.md` | yes | green |
| 05-03-02 | 03 | 3 | DOC-04 | T-5-10 | Local scratch/profile artifacts are ignored or rejected before commit | hygiene | `./gradlew test --tests SourceTreeHygieneTest && ./gradlew verifySourceTreeClean` | yes | green |

*Status values: pending, green, red, flaky.*

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

**Approval:** verified 2026-04-30

## Evidence

Validation status is based on the completed Phase 5 summaries, `05-SECURITY.md`, and a fresh validation rerun:

- `rg -n "renderersExposeAcceleratorEvidenceContract|acceleratorBufferMode|BUFFER_BINDING_AVAILABLE|DEVICE_OWNED|nativeDeviceCopyNs|boundaryCount=2|rejectedFinalists|cpuMaterializationCount=1|sourceResidency|durationNs|prepareTraceSelectedAcceleratorDecisionCarriesPlannerEvidence" src/test/java/BenchmarkSessionTest.java src/test/java/PreparedExecutionBuildTest.java` - PASS
- `rg -n "transformerBlockClosureWorkloadCoversAcceleratorEvidenceFamilies|accelerator_closure_transformer_block|scaledDotProductAttention|gradientLabels|adjacentDeviceOwnedInputUsesBufferBindingWithoutCpuMaterialization|acceleratorBufferExecutionPath|acceleratorBufferPreparedInputUsed|metalNativeDeviceCopyNs|REQUIRED_BUFFER_EXECUTION_UNAVAILABLE" src/test/java/StandardWorkloadsTest.java src/test/java/BenchmarkSessionTest.java src/test/java/backend/metal src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` - PASS
- `rg -n "device-owned|acceleratorBufferExecutionPath|acceleratorBufferReasonCode|CPU materialization boundary|storageResidency|nativeDeviceCopyNs|CUDA remains capability-gated|report-contract|Benchmark reports are explain artifacts|selected accelerator candidate|rejectedFinalists|boundaryCount|benchmark commands remain read-only|BenchmarkSessionTest|PreparedExecutionBuildTest|StandardWorkloadsTest|PreparedMetalExecutableBufferBindingTest|metalTest|SourceTreeHygieneTest" docs/metal-backend.md docs/calibration-autotune.md docs/testing.md docs/troubleshooting.md docs/architecture.md` - PASS
- `rg -n "\\.planning/tmp/|/\\*\\.class|\\*\\*/\\*\\.class|planningTmpScratchIsIgnored|rootGeneratedClassArtifactsAreIgnored|trackedLocalTuningArtifactsStayExplicit" .gitignore src/test/java/SourceTreeHygieneTest.java` - PASS
- `./gradlew classes` - PASS
- `./gradlew test --tests BenchmarkSessionTest --tests PreparedExecutionBuildTest --tests StandardWorkloadsTest --tests SourceTreeHygieneTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - PASS
- `./gradlew metalTest` - PASS
- `./gradlew test --tests SourceTreeHygieneTest` - PASS
- `./gradlew verifySourceTreeClean` - PASS after rerun outside sandbox so Gradle could access the wrapper lock under `~/.gradle`

No generated validation test files were needed; all Phase 5 requirements already had automated coverage.

## Validation Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |
