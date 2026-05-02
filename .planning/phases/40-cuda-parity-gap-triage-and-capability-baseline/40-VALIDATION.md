---
phase: 40
slug: cuda-parity-gap-triage-and-capability-baseline
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-02
---

# Phase 40 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | JUnit Jupiter 5.11.2 via Gradle |
| Config file | `build.gradle` |
| Quick run command | `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.cuda.bridge.CudaFfmBridgeTest` |
| Full suite command | `./gradlew test --tests backend.accelerator.lowering.* --tests backend.cuda.bridge.CudaFfmBridgeTest --tests GpuCoverageTriageReportTest --tests GpuHotPathCoverageTargetsTest --tests SourceTreeHygieneTest && ./gradlew classes` |
| Estimated runtime | 60-180 seconds portable; optional native CUDA depends on toolkit/hardware |

## Sampling Rate

- After every task commit: run the focused command named in that plan.
- After every plan wave: run the wave-specific focused tests plus `git diff --check`.
- Before phase verification: run the full suite command and document optional CUDA native pass/skip.
- Max feedback latency: 180 seconds for portable gates.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 40-01-01 | 40-01 | 1 | CUDAPARITY-01 | T40-01 | Matrix rows cannot mark capability skip as support | unit | `./gradlew test --tests backend.accelerator.lowering.GpuBackendParityReportTest` | W0 | green |
| 40-02-01 | 40-02 | 2 | CUDAPARITY-02 | T40-02 | Capability dimensions remain explicit and stable | unit | `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest --tests backend.cuda.bridge.CudaCapabilityReportTest` | W0 | green |
| 40-03-01 | 40-03 | 3 | CUDAPARITY-03 | T40-03 | Hot-path blockers are classified without hidden fallback | unit | `./gradlew test --tests GpuCoverageTriageReportTest --tests GpuHotPathCoverageTargetsTest` | W0 | green |
| 40-04-01 | 40-04 | 4 | CUDAPARITY-01..03 | T40-04 | Docs and reports do not overclaim CUDA support | docs/integration | `./gradlew test --tests SourceTreeHygieneTest && ./gradlew classes && git diff --check` | W0 | green |

## Wave 0 Requirements

Existing Gradle/JUnit infrastructure covers all Phase 40 requirements. No new test framework is needed.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Optional native CUDA pass/skip interpretation | CUDAPARITY-02 | Local host may lack `nvcc` or CUDA hardware | Run `./gradlew buildCudaGraphShim cudaTest` when available and record pass or capability skip; do not block portable verification on missing CUDA hardware. |

## Validation Sign-Off

- [x] All tasks have automated verification commands.
- [x] Sampling continuity has no three-task gap without automated verification.
- [x] Wave 0 covers existing test infrastructure.
- [x] No watch-mode flags.
- [x] Portable feedback latency target is below 180 seconds.

**Approval:** approved 2026-05-02

## Validation Audit 2026-05-02

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

All `CUDAPARITY-*` requirements have automated verification through focused JUnit/Gradle gates. Optional native CUDA pass/skip evidence is manual-only because CUDA toolkit/hardware availability is host-dependent and not required for portable Phase 40 validation.
