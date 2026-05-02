---
phase: 43
plan: 43-02
type: summary
status: complete
requirements:
  - CUDATRAIN-02
---

# 43-02 Summary: CUDA Scatter And Index-Gradient Semantics

## Completed

- Added `CudaIndexWriteSemantics` for CUDA `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD`.
- CUDA index-write/gradient candidates now validate dense `FLOAT32` values/output, dense static `INT32` indices, rank/axis/shape, and static bounds before final duplicate-index rejection.
- Legal CUDA index-write/gradient candidates remain `UNSUPPORTED_DUPLICATE_INDEX`; no native CUDA scatter/index-gradient execution is claimed.

## Evidence

- `src/main/java/backend/cuda/lowering/CudaIndexWriteSemantics.java`
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java`
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`

