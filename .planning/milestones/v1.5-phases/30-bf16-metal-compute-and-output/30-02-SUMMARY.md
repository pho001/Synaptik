# 30-02 Summary: BF16 Primitive Lowering And Native Execution

## Completed

- Scoped Metal BF16 compute/output capability to explicit operation families instead of a blanket dtype allowlist.
- Allowed BF16 external data inputs only for BF16 consumers whose operation family is BF16-legal.
- Updated Metal planner legality so unsupported BF16 operation families still reject with `UNSUPPORTED_DTYPE` details.
- Extended normalization lowering so BF16 LayerNorm/RMSNorm DAG internals preserve BF16 node output dtype metadata.
- Updated native MPSGraph scalar constants for scalar BF16 primitives and cast final graph outputs to the executable output dtype.
- Added native Metal BF16 buffer execution coverage for BF16 `RELU` and BF16 `MATMUL`.
- Updated prepared-execution and planner tests for the new scoped BF16 Metal legality.

## Verification

- `./gradlew classes`
- `./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest`
- `./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests PreparedExecutionBuildTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest`
- `./gradlew test --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests CompiledGraphTraceTest`
- `./gradlew metalTest`
- `git diff --check`

## Outcome

`METALBF16-01` now has executable native evidence for high-value BF16 primitives and planner legality no longer treats all BF16 Metal compute as impossible. Unsupported BF16 families remain operation-scoped rejections; broader parity/tolerance and coverage gates remain Phase 30-03 work.
