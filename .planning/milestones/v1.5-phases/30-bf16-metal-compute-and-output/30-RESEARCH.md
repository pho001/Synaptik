# Phase 30 Research: BF16 Metal Compute And Output

## Summary

BF16 Metal support is not a simple allowlist change. The native shim can likely compile against `MPSDataTypeBFloat16`, but the Java/native DAG ABI currently assumes FLOAT32 for operation outputs and final executable outputs. Phase 30 must therefore widen transport metadata first, then add BF16 buffer/materialization support, then admit operation families.

## Local Implementation Findings

### Java dtype truth

Phase 29 added `MetalDTypeCapabilityDecision`, `MetalDTypeRole`, and `MetalDTypeReasonCode`. BF16 is currently storage-representable but unsupported for native compute/output.

Phase 30 should update these decisions only after the corresponding transport and execution paths are implemented:

- `externalInputDecision(BFLOAT16)` should become supported when BF16 raw input bindings are implemented.
- `computeDecision(BFLOAT16)` should become supported only for operations admitted by the Metal planner and native shim.
- `outputDecision(BFLOAT16)` should become supported when Metal can write BF16 outputs back to device/CPU storage.
- `operationDecision(op, BFLOAT16)` should be scoped by operation family.

### DAG/native ABI gap

`AcceleratorDagInput` carries external input dtype, but `AcceleratorDagNode` does not carry output dtype. Native compile receives output rank/dims but not output dtype. The native shim currently stores `outputDTypesBoxed addObject:@(1)` for every output.

This must change before BF16 output can be true:

- add output dtype metadata to the DAG node or a parallel bridge array;
- include output dtype in `AcceleratorSubgraphSignature` cache keys;
- pass node output dtype codes into the native compile ABI or a new v3 compile symbol;
- store final output dtype codes in `SynaptikAppleMpsExecutableBox`;
- validate buffer binding dtype against executable output dtype.

### Native BF16 availability

The local SDK exposes `MPSDataTypeBFloat16` in `MPSCoreTypes.h`. The native shim should still treat BF16 as capability-gated because older SDK/runtime combinations may not support it. Tests should build the shim locally and skip or reject cleanly when runtime support is absent.

### Buffer and materialization path

`MetalBufferAllocator` currently:

- uploads FLOAT32 inputs from `float[]`;
- uploads BOOL predicate inputs from `byte[]`;
- allocates only FLOAT32 output bindings;
- reads only FLOAT32 bindings into `float[]`;
- scatters only dense FLOAT32 logical readback into strided destinations.

BF16 needs raw `short[]` upload/readback and equivalent dense logical scatter. It also needs output binding byte lengths based on BF16 element size.

### Operation scope

The first supported BF16 families should be:

- elementwise arithmetic/activation chains that preserve dtype;
- matmul/linear and linear+bias+activation;
- reductions and softmax/log-softmax only after tolerance policy is locked;
- LayerNorm/RMSNorm only after gamma/beta BF16 and accumulation/tolerance behavior is locked.

Masked SDPA, conv/pool, loss/indexing, and backward completeness remain later phase concerns unless explicitly required by this phase's tests.

## Risks

- MPSGraph may internally promote BF16 or fail selected ops even when placeholder dtype is BF16. Parity tests and capability-gated rejections must be precise.
- If executable cache signatures omit output dtype, BF16 and FLOAT32 compiled DAGs can collide.
- BF16 raw storage read/write must not accidentally reinterpret bytes as IEEE float16.
- Reports must distinguish native BF16 from conversion-based execution if a conversion path is later added.

## Verification Plan

Recommended focused tests:

```bash
./gradlew test --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests PreparedExecutionBuildTest --tests GpuCoverageSummaryTest
./gradlew metalTest
```

---

*Research completed inline because GSD planner agents are unavailable in this runtime.*
