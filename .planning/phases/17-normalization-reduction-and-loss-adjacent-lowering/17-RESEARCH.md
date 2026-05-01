# Phase 17 Research: Normalization Reduction And Loss-Adjacent Lowering

**Phase:** 17 - Normalization Reduction And Loss-Adjacent Lowering
**Date:** 2026-05-01
**Status:** Complete

## RESEARCH COMPLETE

## Planning Question

What needs to be known to plan Phase 17 well?

Phase 17 should close the next measured GPU-region blockers from the Phase 14 target list by expanding the shared Metal/CUDA lowering contract around normalization, reductions, softmax-ish residuals, and loss-adjacent operations. In the current codebase, several Phase 17 families are already present in `GpuLoweringCoverageMatrix`, but many are fallback or unsupported rows. The correct implementation shape is to make those rows more explicit, tie them to hot-path targets, preserve current supported `LOG_SOFTMAX` lowering, and ensure backend legality plus traces/reports expose precise reason codes.

## Current Architecture

### Shared Lowering Matrix

`GpuLoweringCoverageMatrix` is already the shared source of truth for Metal and CUDA. It includes:

- supported matmul/linear and elementwise operations,
- layout/view-adjacent operations,
- `SOFTMAX` and `LOG_SOFTMAX`,
- fallback rows for `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX`,
- fallback rows for `LAYER_NORM` and `RMS_NORM`,
- unsupported rows for loss-adjacent operations,
- unsupported `FUSED` rows that keep CPU fused ASM out of GPU lowering.

Phase 17 does not need to invent a parallel matrix. It should harden the existing one so `GPUNORM-01` can be tested directly.

### Backend Legality

Metal legality is centralized in `MetalPartitionSupport.plannerUnsupportedReason(...)`. CUDA legality is centralized in `CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(...)`. Both consult `GpuLoweringCoverageMatrix`, then add backend-specific gates such as Metal dtype support and CUDA direct non-dense input rejection.

This is the right place to keep semantic symmetry while preserving backend-specific capability ownership.

### Existing Softmax-ish Lowering

`AcceleratorSubgraphLowerer` already lowers `LOG_SOFTMAX` to two accelerator DAG primitives: `SOFTMAX` followed by `LOG`. Tests already assert that Metal and CUDA plans contain those primitives. Phase 17 should protect that support while making loss-adjacent operations that often sit near softmax explicit rejections when unsupported.

### Current Reduction And Normalization State

Forward reductions and normalization are currently not legal accelerator compute nodes. The current reason rows are stable enough for Phase 11/12, but Phase 17 needs richer target evidence:

- `SUM`, `MEAN`, `REDUCE_MIN`, `REDUCE_MAX` are reduction blockers.
- `LAYER_NORM` and `RMS_NORM` are normalization blockers for `layer_norm_small`.
- `CONV2D` remains unsupported for `conv2d_resnet_3x3`; Phase 17 should document this as a high-impact stable rejection if implementation is not added.

### Loss-Adjacent State

`NLL_LOSS`, `CROSS_ENTROPY_LOSS`, `CROSS_ENTROPY_LOSS_INDICES`, and `CROSS_ENTROPY_LOSS_INDICES_GRAD` are unsupported. Index-target variants depend on `INT32` target tensors, so `UNSUPPORTED_DTYPE` is the correct stable reason unless and until accelerator target-index handling is implemented.

## Gaps To Close

### Gap 1: Phase 17 Coverage Is Not Testable As A Unit

The matrix covers many families, but tests still frame the contract as Phase 11 coverage. Phase 17 needs focused tests for `GPUNORM-01` rows and the Phase 14 target workload relationship.

Recommended shape:

- Add helper accessors such as `entriesForFamily(backend, family)` if useful.
- Add tests that assert exact Phase 17 op rows for Metal and CUDA.
- Add notes or docs that reference `layer_norm_small`, `conv2d_resnet_3x3`, and `transformer_block_hot_path`.

### Gap 2: Backend Rejection Strings Are Stable But Not Rich Enough

Current Metal/CUDA rejection strings include reason code and operation name. Phase 17 should preserve that compatibility while appending structured family/status/note detail so traces and reports can diagnose why a hot path exited.

Recommended shape:

- Keep existing substrings such as `operation SUM is not supported by GPU_METAL lowering`.
- Append stable detail such as `family=REDUCTION`, `status=fallback`, and the matrix note.
- Keep `REDUCTION_ADJACENT: ` prefix for reductions and normalization where relevant.

### Gap 3: Softmax Support And Loss Rejection Need Joint Evidence

Transformer-like regions often combine `LOG_SOFTMAX` and loss-adjacent boundaries. Phase 17 should prove that `LOG_SOFTMAX` remains lowerable while `CROSS_ENTROPY_LOSS_INDICES` remains visibly rejected with `UNSUPPORTED_DTYPE`.

Recommended shape:

- Add tests around `linear(...).logSoftmax(1)` selected for Metal/CUDA.
- Add required-mode or prepare trace tests for `crossEntropyLossFromIndices(...)`.
- Add coverage summary and benchmark report assertions for `SOFTMAX`, `LOG`, `CROSS_ENTROPY_LOSS_INDICES`, and `UNSUPPORTED_DTYPE`.

### Gap 4: Numeric Parity Must Stay Explicit

Even when a path is rejected and falls back to CPU, numerically sensitive flows must still have CPU parity evidence at the graph output. This phase should compare CPU baseline to accelerator-configured execution for selected softmax/loss/norm test graphs and assert that any fallback is visible.

## Recommended Plan Shape

1. Phase 17 coverage matrix and target contract.
2. Backend legality and stable rejection detail for reduction/normalization/conv blockers.
3. Softmax-ish and loss-adjacent trace/report parity evidence.
4. Docs, validation, and hygiene closure.

## Validation Architecture

### Automated Sampling

- After matrix work, run `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest`.
- After backend legality work, run `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest`.
- After trace/report work, run `./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest`.
- Before phase verification, run `./gradlew classes` and the focused Phase 17 test slice.

### Nyquist Targets

- Every plan must include automated verification.
- At least one test proves Phase 17 matrix coverage for Metal and CUDA.
- At least one test proves reduction or normalization rejection is precise, stable, and tied to the shared matrix.
- At least one test proves `LOG_SOFTMAX` remains supported while loss-adjacent rejection remains visible.
- At least one test compares a numerically sensitive flow against CPU output.
- `git status --short` must show local `profiles/platform/.../tuning/abc/*` artifacts remain unstaged.

## Research Risks

- Risk: claiming reduction or normalization GPU support without execution coverage. Mitigation: require lowering, legality, trace, and parity tests before marking a row `SUPPORTED`.
- Risk: generic CPU fallback hides a high-impact gap. Mitigation: all unsupported cases need stable `GpuLoweringUnsupportedReason` and trace/report evidence.
- Risk: backend-specific shortcuts fork the architecture. Mitigation: keep semantic status in `GpuLoweringCoverageMatrix`; backend adapters only add capability/layout details.
- Risk: loss-adjacent parity hides INT32 target dtype rejection. Mitigation: tests assert `UNSUPPORTED_DTYPE` for index-target losses.
- Risk: local benchmark artifacts pollute the commit. Mitigation: final hygiene requires `git status --short` and explicit unstaged profile artifact note.
