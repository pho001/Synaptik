# Phase 09 - Pattern Map

## Existing Patterns To Reuse

| New or Modified Area | Closest Existing Analog | Reuse Guidance |
|----------------------|-------------------------|----------------|
| `AcceleratorLayoutAbiV2Descriptor` | `AcceleratorBufferLayout` and `DeviceBufferBinding` | Keep the common record immutable, validate constructor arguments, defensive-copy arrays, and expose backend id/access/native identity without backend-specific handles. |
| Physical span calculation | `AcceleratorBufferLayout.byteLength` and `AcceleratorBufferLayoutClassifier.describe` | Use checked arithmetic and fail early on invalid metadata. Preserve logical byte length as dtype * logical elements. |
| ABI-specific status codes | `AcceleratorBufferReasonCode` and `CudaBridgeCapabilityCode` | Stable enum names are part of trace/report contracts. Add new names rather than reusing generic fallback reasons. |
| Metal bridge capability reporting | `CudaBridgeCapabilities` | Model layered state: native library, runtime/context, graph ABI, buffer ABI, layout ABI v2, code, reason. |
| Optional FFM symbols | `MetalMpsFfmBridge.optionalHandle` and `CudaFfmBridge.optionalHandle` | Missing optional symbols should produce unavailable capability detail, not bridge failure. |
| Native stub versioning | `synaptik_cuda_graph_available` / `synaptik_apple_mps_available` pattern | Keep new symbols additive and do not change existing compile/execute/buffer signatures. |
| Portable bridge tests | `CudaFfmBridgeTest.capabilitiesReportNativeAndBufferStateWithoutThrowing` | Assert no-throw capability behavior without requiring local hardware. Native tests stay guarded by explicit system properties or Gradle native test tasks. |
| REQUIRED-mode failures | `PreparedMetalExecutableBufferBindingTest` and `PreparedCudaExecutableBufferPolicyTest` | Existing tests assert required buffer failures contain stable reason codes; extend with layout ABI v2 reason codes. |

## Files Most Likely To Change

- `src/main/java/backend/accelerator/buffer/`
- `src/main/java/backend/memory/DeviceBufferBinding.java`
- `src/main/java/backend/metal/bridge/`
- `src/main/java/backend/cuda/bridge/`
- `src/main/java/backend/metal/buffer/`
- `src/main/java/backend/cuda/buffer/`
- `src/main/native/apple/synaptik_apple_mps_stub.m`
- `src/main/native/cuda/synaptik_cuda_graph_stub.cu`
- `src/test/java/backend/accelerator/buffer/`
- `src/test/java/backend/metal/bridge/`
- `src/test/java/backend/cuda/bridge/`
- `src/test/java/backend/metal/exec/`
- `src/test/java/backend/cuda/exec/`
- `docs/metal-backend.md`
- `docs/development.md`
- `docs/native-bridges-and-blas.md`

## Risks

- Do not expose `MemorySegment`, Metal object types, or CUDA pointers in shared records.
- Do not make layout ABI v2 symbols required for existing dense buffer execution.
- Do not claim native non-contiguous execution in Phase 9; only metadata, capability, validation, and explicit fallback are in scope.
- Do not stage local benchmark/profile files under `profiles/platform/.../tuning/abc/*`.

