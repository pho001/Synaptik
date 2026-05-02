---
phase: 25
slug: forward-sdpa-semantic-enablement
status: verified
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 25 - Validation Strategy

## Requirement Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| GPUSDPA-01 | COVERED | `GpuTargetSemanticsContractTest`, `AttentionExecutionTest`, and Metal parity tests cover scale, rank, shape, mask/no-mask, dtype, and backward interaction. |
| GPUSDPA-02 | COVERED | Metal direct unmasked SDPA is supported through native primitive-DAG parity; masked Metal SDPA remains `UNSUPPORTED_MASK_SEMANTICS`; CUDA direct SDPA remains `CAPABILITY_MISSING`. |
| GPUSDPA-03 | COVERED | `GpuHotPathCoverageTargetsTest`, `GpuCoverageRegressionGateTest`, `GpuCoverageSummaryTest`, and benchmark renderer coverage expose native evidence, lowered primitive count, fallback/materialization counts, and visible SDPA reason codes. |

## Execution Evidence

| Command | Result |
|---------|--------|
| `./gradlew test --tests GpuTargetSemanticsContractTest --tests AttentionExecutionTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest` | Passed |
| `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` | Passed |
| `./gradlew test --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests CompiledGraphTraceTest --tests BenchmarkSessionTest --tests StandardWorkloadsTest` | Passed |
| `./gradlew metalTest` | Passed |
| `git diff --check` | Passed |
| `./gradlew cudaTest` | Not run locally; `nvcc` unavailable |

## Validation Audit 2026-05-02

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

**Approval:** verified 2026-05-02
