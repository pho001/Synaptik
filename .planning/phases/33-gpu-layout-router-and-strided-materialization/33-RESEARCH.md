# Phase 33 Research: GPU Layout Router And Strided Materialization

## Current Implementation

- `AcceleratorBufferLayoutClassifier` classifies logical layouts into `DENSE_CONTIGUOUS`, `ZERO_OFFSET_VIEW`, `NON_ZERO_OFFSET_VIEW`, `PERMUTED_OR_STRIDED_VIEW`, `BROADCAST_ZERO_STRIDE_VIEW`, and `UNSUPPORTED`.
- `AcceleratorLayoutTransformPlanner` currently returns three execution kinds: metadata-only view, dense GPU materialization, or unsupported.
- Metadata-only view propagation is implemented in `DeviceLayoutViewPropagator` by attaching a view binding to the same device handle.
- Dense GPU materialization is implemented via `DeviceLayoutMaterializer`. Metal registers `MetalDeviceLayoutMaterializer` when the bridge supports layout materialization.
- The native Apple shim has `synaptik_apple_mps_layout_contiguous_f32_buffer`, a custom Metal kernel that gathers from source `shape/strides/storageOffset` into a dense FLOAT32 output.
- Coverage summaries already expose `gpuLayoutMaterializationCount` and `gpuLayoutMaterializationBytes`; hard hot-path targets do not yet specifically fail layout-repair CPU exits.

## Gaps To Close

1. Router vocabulary is too coarse.
   - Broadcast materialization and selected strided-native compute need distinct route classes and stable reasons.
   - Today `BROADCAST_ZERO_STRIDE_VIEW` generally reaches `OUTPUT_LAYOUT_UNSUPPORTED` or input rejection instead of a repair route.

2. Metal dense materialization is too narrow.
   - Current native kernel and Java materializer are FLOAT32-only.
   - Phase 33 should at minimum make dtype legality explicit for BF16/BOOL/INT32 roles so v1.5 phases do not misread layout support.

3. Region legality still has operation-owned dense requirements.
   - Several Metal lowering families reject non-dense inputs directly.
   - The desired path is: route layout repair to a dense Metal binding, then let the consumer see a dense-compatible binding without CPU materialization.

4. REQUIRE mode and traces need stronger layout evidence.
   - Missing layout support should fail as a layout router decision, not silently enter tensor-array or CPU fallback.
   - Trace/report detail should distinguish metadata view, dense materialization, broadcast materialization, direct strided compute, and unsupported layout.

5. Coverage needs representative gates.
   - Existing layout tests prove behavior locally, but benchmark gates should fail unexpected `CPU_CONSUMER` materialization on non-contiguous/view workloads.
   - At least one workload should exercise `permute -> contiguous -> supported compute`; another should exercise `expand/broadcast -> contiguous` or stable broadcast rejection.

## Recommended Implementation Shape

1. Expand the backend-neutral layout route contract.
   - Add route kinds for `BROADCAST_GPU_MATERIALIZATION` and `STRIDED_NATIVE_COMPUTE` if directly useful.
   - Add reason codes that make broadcast-vs-strided-vs-unsupported decisions machine-checkable.
   - Keep existing metadata/dense decisions backwards compatible.

2. Extend Metal materialization carefully.
   - Keep the existing FLOAT32 kernel as the first executable path.
   - Add scoped BF16 and BOOL support only with native parity. Otherwise return stable dtype rejection reasons.
   - For broadcast zero-stride, either adapt the same gather kernel by allowing zero strides or add a separate route with explicit rank/dtype gates.

3. Integrate router output with region execution.
   - Ensure prepared execution can reuse router-produced dense bindings as inputs to the next Metal step.
   - Ensure `MetalPartitionSupport` rejects non-dense consumers only after considering legal router repair.
   - Preserve runtime validation of residency/currentness and concrete layout.

4. Harden coverage.
   - Add hot-path target(s) with `METALLAYOUT` requirement family.
   - Require native Metal evidence, GPU layout materialization evidence, zero unexpected CPU materialization, and zero tensor-array fallback for supported cases.
   - Keep unsupported broadcast/strided cases visible until implemented.

## Risks

- Broadcast zero-stride materialization can read the same source element many times; bounds and physical span validation must allow zero strides without permitting invalid storage access.
- BF16/BOOL layout materialization can look like dtype compute support if reporting is not role-specific.
- Route repair can accidentally mask costly materialization. Coverage should record materialization count/bytes and not only execution success.
- Changing dense-layout assumptions may shorten or split GPU regions differently. Region length and lowered primitive evidence should stay in gates.
