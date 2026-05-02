---
phase: 42
plan: 42-01
type: summary
status: complete
requirements:
  - CUDANN-01
---

# 42-01 Summary: CUDA SDPA Forward Parity

## Completed

- Added `CudaNnSemantics` as the CUDA NN semantic gate before generic coverage-matrix rejection.
- CUDA forward SDPA now validates dense `FLOAT32` rank-3/4 query/key/value/output tensors before final capability rejection.
- Masked SDPA now validates public `BOOL` dense mask inputs and exact broadcasted score-shape compatibility.
- SDPA blocker details classify `UNMASKED`, `EXTERNAL_BOOL_MASK`, `CAUSAL_BOOL_MASK`, and `EXTERNAL_AND_CAUSAL_BOOL_MASK`.
- Legal CUDA SDPA candidates remain `CAPABILITY_MISSING`; no native CUDA SDPA support is claimed.

## Evidence

- `src/main/java/backend/cuda/lowering/CudaNnSemantics.java`
- `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java`
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`

