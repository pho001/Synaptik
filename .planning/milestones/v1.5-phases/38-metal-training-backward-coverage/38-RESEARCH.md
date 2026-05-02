# Phase 38 Research: Metal Training Backward Coverage

**Phase:** 38 Metal Training Backward Coverage
**Requirements:** METALTRAIN-01, METALTRAIN-02, METALTRAIN-03
**Status:** Ready for planning
**Researched:** 2026-05-02

## Current Evidence

- Training execution is compiled as one closure and split into forward/backward prepared steps in `PreparedExecution`.
- `GpuLoweringCoverageMatrix` already has supported rows for backward-adjacent `SOFTMAX_GRAD`, `LOG_SOFTMAX_GRAD`, `REDUCE_MIN_GRAD`, `REDUCE_MAX_GRAD`, `MIN_GRAD`, `MAX_GRAD`, plus Metal `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`.
- Phase 37 proved dense Metal loss forward can remain GPU-owned during `FORWARD_BACKWARD`, while index-target loss gradients remain explicit CPU boundaries with `UNSUPPORTED_INDEX_SEMANTICS`.
- Phase 36 proved forward index support must not be counted as scatter/index-gradient support; `GATHER_GRAD`, `TAKE_ALONG_AXIS_GRAD`, and `SCATTER_ADD` remain `UNSUPPORTED_DUPLICATE_INDEX`.
- `ExecutionState` and traces already distinguish CPU materialization reasons such as `GRADIENT_PUBLICATION` and `CPU_CONSUMER`.

## Planning Implications

- The phase should not claim full training support. It should classify every v1.5-supported forward family into one of: verified Metal backward execution, explicit CPU boundary, or unsupported rejection with stable reason.
- Positive work should focus on backward-adjacent rows already in the matrix and representative training traces that prove no hidden internal CPU materialization before gradient publication.
- Unsupported conv/pool backward, index gradients, index-target loss gradients, and unsupported dtype/layout variants must remain visible blockers.
- Coverage reports need training-specific gates so true gradient publication does not look like a regression, while avoidable internal `CPU_CONSUMER` materialization still fails.

## Files Likely Touched

- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java`
- `src/main/java/backend/metal/lowering/MetalPartitionSupport.java`
- `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java`
- `src/main/java/backend/metal/prepare/` and `src/main/java/backend/metal/exec/`
- `src/main/java/tuning/benchmark/report/`
- `src/main/java/tuning/workload/StandardWorkloads.java`
- `src/test/java/PreparedExecutionBuildTest.java`
- `src/test/java/CompiledGraphTraceTest.java`
- `src/test/java/GpuCoverageRegressionGateTest.java`
- `src/test/java/GpuHotPathCoverageTargetsTest.java`
- `docs/gpu-lowering-coverage.md`
- `docs/metal-backend.md`

## Verification Direction

Use CPU as oracle, native Metal tests where bridge support is available, and portable tests for matrix/traces/report gates. Keep CUDA capability-gated or explicit fallback; v1.5 remains Metal-first.
