---
phase: 19-multi-op-gpu-region-execution
plan: "01"
status: complete
---

# 19-01 Summary

## Multi-op lowered-region contract

Implemented Phase 19 contract coverage for multi-op lowered regions:

- `GPUMULTI-01`: shared lowerer tests now prove one selected GPU region can carry multiple original operations and multiple lowered primitives, including layout, elementwise, and softmax/log primitives.
- `GPUMULTI-03`: Metal and CUDA lowerer tests now assert one backend-owned graph unit through `METAL_GRAPH_REGION` and `CUDA_GRAPH_REGION` for supported multi-op candidates.
- Unsupported internal primitives can be shortened before execution with `GpuLoweringUnsupportedReason.DAG_CANDIDATE_SHORTENED`, preserving original candidate ids, accepted ids, and rejection evidence.

Verification run:

```bash
./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest
```

Result: passed.

## Deviations from Plan

None - plan executed exactly as written.

## Hygiene

profiles/platform/.../tuning/abc/* remained unstaged.
