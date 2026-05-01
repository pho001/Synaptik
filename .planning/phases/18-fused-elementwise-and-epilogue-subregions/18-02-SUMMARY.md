# 18-02 Summary: Elementwise Chain Subregions

## Elementwise chain subregions

Implemented region-internal GPU elementwise subchain fusion for mixed accelerator regions. `GenericGpuRegionOptimizationPolicy` now keeps the full selected GPU partition intact while emitting `FUSED_ELEMENTWISE` execution units for maximal dependent elementwise subchains inside that region.

The shared accelerator lowerer now records `GpuFusionSubpatternSummary` metadata for `ELEMENTWISE_CHAIN` spans even when the full selected DAG also contains non-elementwise primitives such as matmul. Metal and CUDA plans expose the same manifest contract, including original operation node ids and lowered primitive ids/counts.

## Requirement Notes

- `GPUFUSEX-01`: Supported elementwise chains inside a GPU region are represented as fused subpatterns without requiring intermediate CPU materialization for interior values.
- `GPUFUSEX-03`: GPU elementwise fusion is derived from normal graph operations and lowered DAG metadata. It does not reuse or depend on CPU `Operation.OpType.FUSED`.

## Verification

Passed:

```bash
./gradlew test --tests graph.optimizer.region.DefaultRegionOptimizerTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest
```

Acceptance grep checks for the required test names, `FUSED_ELEMENTWISE`, `ELEMENTWISE_CHAIN`, `loweredPrimitiveCount`, and `CPU_CONSUMER` also passed.

## Hygiene

`profiles/platform/.../tuning/abc/*` remained unstaged.
