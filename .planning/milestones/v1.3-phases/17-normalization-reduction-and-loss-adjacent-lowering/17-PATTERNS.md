# Phase 17 Pattern Map: Normalization Reduction And Loss-Adjacent Lowering

**Phase:** 17 - Normalization Reduction And Loss-Adjacent Lowering
**Date:** 2026-05-01
**Status:** Complete

## Pattern Mapping Complete

## Files To Modify Or Extend

| Planned file | Role | Closest existing analog | Notes |
|---|---|---|---|
| `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` | Shared lowering coverage source of truth | Existing Phase 11/12 coverage rows | Add Phase 17 helpers/notes without creating backend-specific duplicate matrices. |
| `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java` | Matrix contract tests | Existing required-family tests | Add tests for Phase 17 op rows and hot-path target evidence. |
| `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` | Metal planner legality diagnostics | Existing matrix-backed rejection path | Preserve existing reason substrings while adding richer family/status detail. |
| `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java` | CUDA planner legality diagnostics | Existing matrix-backed rejection path and non-dense guard | Preserve CUDA layout guard and append shared matrix details. |
| `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` | Metal legality/lowering tests | Existing unsupported reduction/norm/loss tests | Add Phase 17 tests for `SUM`, `MEAN`, `LAYER_NORM`, `RMS_NORM`, and loss-adjacent reasons. |
| `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` | CUDA legality/lowering tests | Existing unsupported reduction/norm/loss tests | Mirror Metal expectations while keeping CUDA non-dense rejection intact. |
| `src/test/java/PreparedExecutionBuildTest.java` | Build/prepare and CPU parity tests | Existing required-mode and accelerator prepare tests | Add numerically sensitive CPU parity plus visible fallback checks. |
| `src/test/java/CompiledGraphTraceTest.java` | Prepare trace evidence | Existing manifest and dtype residency evidence tests | Assert Phase 17 reason strings and selected softmax lowering evidence. |
| `src/test/java/GpuCoverageSummaryTest.java` | Coverage summary | Existing unsupported dtype/materialization counters | Add Phase 17 fallback/rejection reason coverage. |
| `src/test/java/BenchmarkSessionTest.java` | Benchmark report text/JSON rendering | Existing lowering manifest and dtype evidence tests | Assert Phase 17 rows render in text and JSON. |
| `docs/gpu-lowering-coverage.md` | Developer lowering docs | Existing coverage matrix docs | Add Phase 17 normalization/reduction/loss section and commands. |
| `docs/compute-flow.md` | Runtime flow docs | Existing accelerator region and materialization docs | Explain visible fallback for numerically sensitive flows. |
| `docs/development.md` | Verification command docs | Existing focused test sections | Add Phase 17 focused command. |
| `.planning/phases/17-normalization-reduction-and-loss-adjacent-lowering/17-*-SUMMARY.md` | Execution evidence | Phase 16 summaries | Record verification and profile artifact hygiene per wave. |

## Reusable Code Patterns

### Shared Matrix First

Metal and CUDA legality adapters already consult `GpuLoweringCoverageMatrix`. Add shared helper methods there when a detail must be identical across backends.

### Stable Reason Codes Plus Appended Detail

Tests currently assert substrings like `operation SUM is not supported by GPU_METAL lowering`. Preserve those substrings and append detail instead of replacing diagnostics wholesale.

### Portable Synthetic Evidence

Use Java tests and synthetic traces for coverage/report behavior. Native Metal/CUDA execution remains optional and capability-gated.

### CPU Parity For Sensitive Flows

Use CPU output as the correctness oracle for softmax/loss/norm graphs. Accelerator-configured executions may fallback, but the fallback reason must be visible.

## Landmines

- Do not mark reductions or normalization as `SUPPORTED` unless lowering and parity tests prove execution.
- Do not hide loss-adjacent fallback behind generic CPU replay.
- Do not bypass Phase 16 dtype legality for INT32 target tensors.
- Do not remove `LOG_SOFTMAX` lowering as `SOFTMAX` plus `LOG`.
- Do not turn GPU fusion into CPU `Operation.OpType.FUSED`.
- Do not commit `profiles/platform/.../tuning/abc/*`.
