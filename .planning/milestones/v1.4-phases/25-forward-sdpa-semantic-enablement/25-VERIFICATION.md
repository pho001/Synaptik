---
status: passed
phase: 25-forward-sdpa-semantic-enablement
requirements:
  GPUSDPA-01: passed
  GPUSDPA-02: passed
  GPUSDPA-03: passed
must_haves_total: 5
must_haves_passed: 5
human_verification_required: false
created: 2026-05-01
---

# Verification: Phase 25 Forward SDPA Semantic Enablement

## Verdict

Phase 25 passed verification.

Forward SDPA semantics are now test-visible before admission. Metal admits the verified unmasked `FLOAT32` rank-3/rank-4 path through a native primitive MPSGraph DAG. CUDA remains an explicit `CAPABILITY_MISSING` fallback with stable mask, dtype, layout, and shape diagnostics. Coverage gates distinguish Metal native evidence from CUDA capability fallback and fail hidden CPU/tensor-array exits for the supported Metal case.

## Must-Have Verification

| # | Must Have | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Forward SDPA semantics are locked for scale, rank, mask/no-mask, dtype, and backward-pass interaction. | Pass | `GpuTargetSemanticsContractTest`, `AttentionExecutionTest`, `MetalRegionLowererTest`, and `PreparedExecutionBuildTest` cover scale, rank, mask, dtype, and backward safety. |
| 2 | Metal direct SDPA is enabled only for verified MPSGraph primitive-DAG semantics and capability. | Pass | `MetalMpsFfmBridgeTest` verifies explicit-scale rank-3 and default-scale rank-4 parity; `PreparedExecutionBuildTest` verifies the prepared Metal executable path; `AcceleratorSubgraphSignature` includes scalar bits. |
| 3 | CUDA either gains supported SDPA lowering/native execution or a precise required-capability rejection that does not masquerade as support. | Pass | `CudaRegionLowererTest` and `PreparedExecutionBuildTest` verify `CAPABILITY_MISSING` for legal CUDA SDPA and stable mask/dtype/layout rejection reasons. |
| 4 | Transformer attention coverage gates prove backend execution path and CPU exit count. | Pass | `GpuHotPathCoverageTargetsTest`, `GpuCoverageRegressionGateTest`, and `GpuCoverageSummaryTest` verify Metal native evidence requirements, CUDA visible fallback reasons, fallback counts, materialization counts, and target coverage truth. |
| 5 | Numerical tests cover masked and unmasked supported cases. | Pass | `AttentionExecutionTest` covers public CPU masked/unmasked SDPA semantics; Metal native support covers unmasked supported rank-3/rank-4 cases; masked direct GPU SDPA remains explicitly unsupported rather than falsely admitted. |

## Requirement Traceability

| Requirement | Status | Evidence |
|-------------|--------|----------|
| GPUSDPA-01 | Passed | Semantics contract, CPU parity, rank/default-scale tests, mask/no-mask contract tests, backward interaction guards. |
| GPUSDPA-02 | Passed | Metal native primitive-DAG execution for supported unmasked SDPA; Metal masked `UNSUPPORTED_MASK_SEMANTICS`; CUDA `CAPABILITY_MISSING` plus stable illegal-input reasons. |
| GPUSDPA-03 | Passed | Transformer coverage expectations require Metal native buffer evidence and visible CUDA fallback evidence; regression gates fail hidden tensor-array/CPU fallback for supported SDPA. |

## Automated Checks

Passed:

```bash
./gradlew test --tests GpuTargetSemanticsContractTest --tests AttentionExecutionTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageRegressionGateTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests StandardWorkloadsTest
./gradlew metalTest
git diff --check
```

Not run:

```bash
./gradlew cudaTest
```

`nvcc` is unavailable locally, so native CUDA remains hardware/toolchain-gated. Portable CUDA tests verify the stable unsupported behavior.

## Residual Risk

- CUDA direct SDPA is not implemented; this is deliberate and visible as `CAPABILITY_MISSING`.
- Direct masked GPU SDPA remains unsupported for both backends until BOOL-mask semantics are mapped and parity-tested.
- Local tuning/profile artifacts remain dirty and unstaged; they are not Phase 25 evidence.

## Human Verification

No human verification required.
