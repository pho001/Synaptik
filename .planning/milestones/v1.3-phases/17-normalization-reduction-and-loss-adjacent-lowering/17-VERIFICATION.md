---
phase: 17-normalization-reduction-and-loss-adjacent-lowering
status: passed
score: 10/10
verified: 2026-05-01
requirements_verified: [GPUNORM-01, GPUNORM-02, GPUNORM-03]
human_verification_required: false
---

# Phase 17 Verification

Phase 17 achieved the goal: high-impact normalization, reduction, softmax-ish, conv, and loss-adjacent GPU lowering gaps are now represented in the shared Metal/CUDA coverage contract, routed through backend legality diagnostics, and proven by CPU parity plus trace/report evidence where execution remains fallback-owned.

## Requirement Verification

| Requirement | Status | Evidence |
|-------------|--------|----------|
| `GPUNORM-01` | Passed | `GpuLoweringCoverageMatrix` explicitly covers `LAYER_NORM`, `RMS_NORM`, reductions, `SOFTMAX`, `LOG_SOFTMAX`, conv, and loss-adjacent rows for both `GPU_METAL` and `GPU_CUDA`. |
| `GPUNORM-02` | Passed | Matrix notes and backend rejection detail preserve `target=layer_norm_small`, `target=conv2d_resnet_3x3`, and `target=transformer_block_hot_path` in planner, trace, coverage, and benchmark evidence. |
| `GPUNORM-03` | Passed | `PreparedExecutionBuildTest`, `CompiledGraphTraceTest`, `GpuCoverageSummaryTest`, and `BenchmarkSessionTest` cover CPU parity, dtype/layout legality, and visible fallback for numerically sensitive flows. |

## Must-Have Verification

| Item | Status | Notes |
|------|--------|-------|
| Shared Metal/CUDA matrix covers Phase 17 families | Passed | `phaseSeventeenMatrixCoversNormalizationReductionSoftmaxAndLossRows` checks both backends. |
| Supported vs fallback vs unsupported statuses are explicit | Passed | `LOG_SOFTMAX` remains supported; reductions and normalization remain fallback; loss-adjacent and conv blockers remain unsupported or dtype-rejected. |
| Hot-path blockers are tied to measured targets | Passed | Coverage rows and docs preserve `layer_norm_small`, `conv2d_resnet_3x3`, and `transformer_block_hot_path`. |
| Backend legality consumes shared matrix detail | Passed | Metal and CUDA adapters call `GpuLoweringCoverageMatrix.plannerUnsupportedDetail(...)` for unsupported matrix rows. |
| Backend-specific dtype/layout gates remain owned by backends | Passed | Metal dtype gates still precede matrix checks, and CUDA direct non-dense layout rejection remains exact. |
| Softmax-ish supported lowering remains visible | Passed | `LOG_SOFTMAX` manifests render original `LOG_SOFTMAX` and lowered `SOFTMAX` plus `LOG` primitives. |
| Loss-adjacent fallback remains visible | Passed | Index-target cross entropy exposes `UNSUPPORTED_DTYPE`, `family=LOSS_ADJACENT`, and hot-path target evidence. |
| CPU parity remains the correctness oracle | Passed | Accelerator-configured softmax, layer norm, and index-target loss flows are compared against CPU-only execution. |
| Docs do not overstate native reduction or normalization support | Passed | Docs state that native reduction and normalization support is not implied by a fallback row. |
| Source hygiene preserved | Passed | `profiles/platform/.../tuning/abc/* remained unstaged`. |

## Automated Checks

| Command | Result |
|---------|--------|
| `./gradlew classes` | Passed |
| `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests SourceTreeHygieneTest` | Passed |
| `./gradlew test --tests graph.execution.RuntimeMemoryBinderTest --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.cuda.buffer.CudaBufferAllocatorTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests SourceTreeHygieneTest` | Passed regression gate |
| `gsd-sdk query verify.schema-drift 17` | Passed; no schema drift detected |
| `gsd-sdk query verify.codebase-drift 17` | Non-blocking; no action required |
| `git status --short` | Only local tuning profile artifacts were dirty |

## Code Review

`17-REVIEW.md` status is `clean` with 0 findings.

## Residual Risk

Native Metal/CUDA reductions, normalization, conv, and loss-adjacent execution are still intentionally narrow. Phase 17 verifies shared coverage, stable rejection, parity, and reportability; it does not claim native reduction or normalization support.

## Verdict

Passed. Phase 17 is ready for security and Nyquist validation follow-up before milestone audit.
