---
phase: 42
plan: 42-02
type: summary
status: complete
requirements:
  - CUDANN-02
---

# 42-02 Summary: CUDA Conv/Pool Forward Parity

## Completed

- CUDA `CONV2D` and `CONV2D_GEMM` now validate dense `FLOAT32` NCHW/OIHW inputs, optional bias shape, output shape, group/channel contract, and dilation before final rejection.
- CUDA `MAX_POOL2D` and `AVG_POOL2D` now validate dense `FLOAT32` NCHW rank-4 input/output, kernel/stride/padding output shape, and average-pool divisor scope.
- Legal CUDA conv/pool forward candidates remain `CAPABILITY_MISSING`; no cuDNN or custom CUDA route is claimed.
- Grouped/depthwise conv, dilated conv, invalid layout/dtype/shape, and `AVG_POOL2D countIncludePad=true` have explicit blockers.

## Evidence

- `src/main/java/backend/cuda/lowering/CudaNnSemantics.java`
- `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`
- `docs/cuda-backend.md`
- `docs/gpu-lowering-coverage.md`

