# Summary 33-01: Layout Router Contract And Reason Vocabulary

**Phase:** 33 GPU Layout Router And Strided Materialization
**Status:** Completed
**Date:** 2026-05-02

## Delivered

- Extended the backend-neutral layout transform route vocabulary with `BROADCAST_GPU_MATERIALIZATION` and `STRIDED_NATIVE_COMPUTE`.
- Added stable reason codes for broadcast materialization availability/rejection and strided-native compute availability/rejection.
- Updated `AcceleratorLayoutTransformPlanner` to keep existing metadata-only and dense materialization behavior while distinguishing:
  - zero-stride broadcast repair to dense targets,
  - broadcast repair rejected for non-dense targets,
  - direct strided native compute rejection for non-layout operations.
- Updated `DeviceLayoutViewPropagator` and Metal layout materialization dispatch so GPU materialization routes are handled generically instead of hard-coding only dense materialization.
- Added/updated tests proving metadata views, dense repair, broadcast repair, strided rejection, backend mismatch, missing binding, unsupported layout, and CUDA unsupported behavior.

## Verification

```bash
./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutTransformPlannerTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest
./gradlew test --tests PreparedExecutionBuildTest
```

## Notes

- This wave intentionally defines router semantics and reason vocabulary. Broader native dtype support and additional Metal materialization kernels remain in 33-02.
- CUDA remains capability-gated; shared route vocabulary does not imply CUDA support.
