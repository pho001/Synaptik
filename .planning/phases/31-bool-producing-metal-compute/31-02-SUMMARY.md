# 31-02 Summary: Native Compare Logical And BOOL Reduction Execution

## Completed

- Added accelerator DAG ABI nodes for `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT`, `REDUCE_ALL`, and `REDUCE_ANY`.
- Extended shared lowering so BOOL logical and reduction ops carry BOOL output dtype metadata and reduction axis/keepDims scalar metadata.
- Implemented native MPSGraph shim branches for compare, logical, and BOOL reduction nodes.
- Widened Metal buffer output validation, allocation, and CPU materialization for one-byte BOOL outputs.
- Updated Metal capability and planner legality so supported BOOL ops admit only legal dense dtype/layout contracts.
- Promoted Metal BOOL compare/logical/reduction rows to supported while leaving CUDA BOOL-producing rows unsupported.
- Updated target coverage truth so Metal BOOL target rows count as native executable.

## Verification

- `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests PreparedExecutionBuildTest`
- `./gradlew test --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests CompiledGraphTraceTest --tests GpuTargetSemanticsContractTest`
- `./gradlew test --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalAcceleratorBufferBinderTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest`
- `./gradlew metalTest`
- `git diff --check`

## Outcome

`METALBOOL-01` and `METALBOOL-03` now have native Metal execution evidence for compare, logical, and BOOL reduction outputs. Supported BOOL results can be represented as device-owned one-byte buffers; Phase 31-03 will harden mask-chain residency through `WHERE` and coverage gates.
