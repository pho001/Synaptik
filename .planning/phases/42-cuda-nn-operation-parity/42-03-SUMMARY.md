---
phase: 42
plan: 42-03
type: summary
status: complete
requirements:
  - CUDANN-03
---

# 42-03 Summary: CUDA Dense Loss Parity

## Completed

- CUDA dense `NLL_LOSS` and dense `CROSS_ENTROPY_LOSS` now validate dense `FLOAT32` scores/log-probabilities, dense targets, rank 1..4, valid class axis, matching target shape, and scalar mean output shape `[1]`.
- Legal dense CUDA loss candidates remain `DAG_PRIMITIVE_UNSUPPORTED` because no CUDA dense-loss primitive or lowered native execution path is implemented.
- Index-target loss remains separate as `UNSUPPORTED_INDEX_SEMANTICS`.

## Evidence

- `src/main/java/backend/cuda/lowering/CudaNnSemantics.java`
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`
- `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java`

