# Summary 33-02: Metal GPU-Side Materialization Primitives

**Phase:** 33 GPU Layout Router And Strided Materialization
**Status:** Completed
**Date:** 2026-05-02

## Delivered

- Proved scoped FLOAT32 broadcast zero-stride layout repair through Metal GPU-side materialization.
- Added native Metal parity coverage for:
  - permuted/strided FLOAT32 view -> dense buffer,
  - zero-stride broadcast FLOAT32 view -> dense buffer,
  - non-zero-offset FLOAT32 view -> dense buffer.
- Tightened `MetalBufferBinding` availability so borrowed logical-view handles validate against physical layout span, while dense output bindings continue to validate against logical byte length.
- Kept non-FLOAT32 Metal layout materialization explicitly rejected with `NATIVE_LAYOUT_DTYPE_UNSUPPORTED` evidence instead of implying BF16/BOOL/INT32 materialization support.
- Updated Java materializer tests to prove broadcast materialization dispatches through the bridge and unsupported dtypes reject before native dispatch.

## Verification

```bash
./gradlew test --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.metal.buffer.MetalDeviceLayoutMaterializerTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest
./gradlew test --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.metal.buffer.MetalDeviceLayoutMaterializerTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest
./gradlew metalTest
```

## Notes

- BF16/BOOL/INT32 layout materialization remains out of scope for this wave and is intentionally rejected. This keeps dtype residency separate from native layout materialization support.
- The native shim did not need a new kernel for FLOAT32 broadcast repair; the existing stride-based materialization kernel already supports zero strides once binding availability validates physical span correctly.
