# 18-03 Summary: Matmul And Linear Epilogue Subregions

## Matmul and linear epilogue subregions

Implemented explicit GPU region-internal epilogue units for supported matmul/linear plus bias plus activation spans. The optimizer now emits `SPECIALIZED_PRIMITIVE` units with `gpu-epilogue-subregion:` trace metadata when an epilogue is terminal inside the selected region, while longer elementwise chains remain handled by Wave 2 elementwise subchain fusion.

The shared lowering manifest already carried `LINEAR_BIAS_ACTIVATION`; this wave added tests and gates proving that Metal and CUDA plans expose the epilogue subpattern with original node span, primitive count, and `epilogue` detail. Illegal non-dense epilogue bias inputs now report backend-specific `UNSUPPORTED_LAYOUT` detail without widening general Metal layout rejection behavior.

## Requirement Notes

- `GPUFUSEX-02`: Supported matmul/linear epilogues are represented as region-internal GPU subpatterns when backend legality and dtype/layout gates allow them.
- `GPUFUSEX-03`: Epilogue fusion uses accelerator DAG/post-op metadata and does not reuse CPU fused ASM, CPU vector dispatch, or `Operation.OpType.FUSED`.

## Verification

Passed:

```bash
./gradlew test --tests graph.optimizer.region.DefaultRegionOptimizerTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest
```

Acceptance grep checks for the epilogue test names, `LINEAR_BIAS_ACTIVATION`, `SPECIALIZED_PRIMITIVE`, `gpu-epilogue-subregion:`, `CPU_CONSUMER`, and `epilogue` also passed.

## Hygiene

`profiles/platform/.../tuning/abc/*` remained unstaged.
