# Phase 9: Native Layout ABI v2 - Research

**Date:** 2026-04-30
**Status:** Ready for planning

## RESEARCH COMPLETE

## Phase Boundary

Phase 9 should establish a versioned, backend-neutral layout ABI v2 contract for Metal and CUDA native buffer execution. It should not implement GPU-side layout transforms or broader operation lowering. Later phases can consume this contract to keep non-contiguous/view values device-resident.

## Implementation Findings

### Shared Layout Metadata

The current `AcceleratorBufferLayout` already carries dtype, shape, strides, storage offset, logical element count, logical byte length, and layout class. Phase 9 should add a descriptor layer rather than replacing this record.

Recommended shared records under `backend.accelerator.buffer`:

- `AcceleratorLayoutAbiV2Descriptor`
- `AcceleratorLayoutAbiV2Status`
- `AcceleratorLayoutAbiV2StatusCode`
- `AcceleratorLayoutAbiV2Support`

The descriptor should be constructible from `DeviceBufferBinding` so Metal and CUDA retain backend-owned native handles while common code can inspect backend id, access mode, native handle identity, rank, shape, strides, storage offset, logical bytes, and physical byte span.

### Physical Span

Physical byte span must be separate from logical byte length. The span should be computed from shape, strides, storage offset, and bytes per element:

```text
maxElementOffset = storageOffset + sum((shape[i] - 1) * stride[i])
physicalElements = maxElementOffset + 1
physicalByteSpan = physicalElements * bytesPerElement
```

Zero-stride broadcast views should remain representable: their physical span can be one element plus storage offset. Negative strides remain unsupported unless a backend explicitly advertises support later. Overflow should be detected with checked arithmetic and reported through an ABI-specific status code.

### Capability Surface

CUDA already has `CudaBridgeCapabilities` and `CudaBridgeCapabilityCode`; extend those rather than adding a parallel CUDA-only mechanism. Metal currently has static dtype helpers in `MetalMpsCapabilities` but no layered bridge capability record. Phase 9 should add a Metal bridge capability record near `backend.metal.bridge` so bridge availability, buffer binding support, and layout ABI v2 support are distinguishable.

Capability fields should include:

- native library available
- runtime/context available where the backend can know it
- graph execution ABI available
- v1 buffer execution supported
- layout ABI v2 supported
- layout ABI v2 version
- stable capability code
- reason

### Optional Native Symbols

The v2 symbols should be optional. Missing v2 symbols must not make the bridge unavailable and must not break existing dense buffer execution.

Recommended native symbols:

- `synaptik_apple_mps_layout_abi_version`
- `synaptik_apple_mps_validate_layout_abi_v2`
- `synaptik_cuda_graph_layout_abi_version`
- `synaptik_cuda_graph_validate_layout_abi_v2`

The validate function can be metadata-only in this phase. A parallel-array FFM contract is easier to test than Java struct layouts:

- binding count
- rank per binding
- dtype code per binding
- storage offset per binding
- logical element count per binding
- logical byte length per binding
- physical byte span per binding
- access mode per binding
- layout class per binding
- flattened shape values with per-binding offsets
- flattened stride values with per-binding offsets
- native handle pointer per binding

Status codes should be stable integers documented in Java tests and native comments.

### Fallback And Required Mode

Existing reason codes already distinguish generic buffer ABI absence and layout unsupported cases. Phase 9 should add ABI-v2-specific codes so reports can distinguish:

- missing layout ABI v2 symbols
- layout ABI v2 version mismatch
- unsupported layout metadata
- unsupported layout rank
- unsupported layout dtype
- physical span overflow

`AUTO` should visibly reject to the existing fallback path. `REQUIRE` should fail before tensor-array or CPU execution can satisfy the operation silently.

### Portable Test Strategy

Most coverage should be Java-first:

- descriptor construction from synthetic Metal/CUDA `DeviceBufferBinding` implementations
- defensive copies for shape/stride arrays
- physical span and overflow tests
- capability record tests without native hardware
- bridge no-throw capability probes when symbols are absent
- binder/prepared-executable tests for AUTO fallback and REQUIRE failure

Native checks should stay optional:

- `./gradlew metalTest` when the Metal shim is available
- `./gradlew buildCudaGraphShim cudaTest` when CUDA tooling is available

## Validation Architecture

Validation should be fast and portable first, native-capability-gated second.

Recommended focused commands:

- `./gradlew classes`
- `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.accelerator.buffer.AcceleratorLayoutAbiV2DescriptorTest`
- `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.cuda.bridge.CudaFfmBridgeTest`
- `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest`

Optional native commands:

- `./gradlew metalTest`
- `./gradlew buildCudaGraphShim cudaTest`

## Planning Implications

Split the phase into three plans:

1. Shared metadata descriptor and physical span contract.
2. Metal/CUDA capability handshake and optional native symbols.
3. ABI-specific fallback reason codes, prepared-executable required-mode behavior, tests, and docs.

This sequence lets execution keep existing dense paths working while each later plan consumes the stable records and status codes from the earlier plans.

