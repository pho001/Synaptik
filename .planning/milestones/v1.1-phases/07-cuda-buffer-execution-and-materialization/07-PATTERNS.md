# Phase 07 Pattern Map

## Closest Analogs

| New/Changed Area | Closest Existing Analog | Pattern To Reuse |
|------------------|-------------------------|------------------|
| `backend.cuda.buffer.CudaBufferAllocator` | `backend.metal.buffer.MetalBufferAllocator` | Run-scoped native allocation, read-back, destroy, layout validation, materialization result timing |
| `backend.cuda.buffer.CudaDeviceToCpuMaterializer` | `backend.metal.buffer.MetalDeviceToCpuMaterializer` | Strict support checks before `ExecutionState` marks CPU storage current |
| `backend.cuda.buffer.CudaBufferBinding` | `backend.metal.buffer.MetalBufferBinding` | Backend-specific `DeviceBufferBinding` carrying logical layout and native identity |
| `backend.cuda.buffer.CudaBufferResource` | `backend.metal.buffer.MetalBufferResource` | Idempotent execution-resource cleanup wrapper |
| `CudaFfmBridge` buffer ABI | `MetalMpsFfmBridge` optional buffer symbols | Advertise buffer support only when every required symbol is available |
| `PreparedCudaExecutable` buffer path | `PreparedMetalExecutable` buffer path | Decide, resolve bindings, execute buffers, attach device-owned outputs, fallback visibly |
| CUDA materialization tests | `ExecutionStateResidencyTest` and `PreparedMetalExecutableBufferBindingTest` | Prove `requireCpuReadable` invokes materializer and records traces |

## Concrete File Targets

- `src/main/native/cuda/synaptik_cuda_graph_stub.cu`
- `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`
- `src/main/java/backend/cuda/bridge/CudaGraphBridge.java`
- `src/main/java/backend/cuda/bridge/CudaBridgeCapabilities.java`
- `src/main/java/backend/cuda/buffer/CudaBufferAccess.java`
- `src/main/java/backend/cuda/buffer/CudaBufferHandle.java`
- `src/main/java/backend/cuda/buffer/CudaBufferBinding.java`
- `src/main/java/backend/cuda/buffer/CudaBufferResource.java`
- `src/main/java/backend/cuda/buffer/CudaBufferAllocator.java`
- `src/main/java/backend/cuda/buffer/CudaDeviceToCpuMaterializer.java`
- `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java`
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java`
- `src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java`
- `src/test/java/backend/cuda/buffer/CudaBufferAllocatorTest.java`
- `src/test/java/backend/cuda/buffer/CudaDeviceToCpuMaterializerTest.java`
- `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java`
- `docs/architecture.md`
- `docs/development.md`
- `docs/configuration.md`

## Implementation Notes

- Keep CUDA-specific native handles under `backend.cuda.buffer`; do not extend public `Tensor`.
- Treat CUDA output buffers as `StorageResidency.DEVICE_OWNED`; CPU arrays become current only after materializer read-back.
- Reuse `AcceleratorBufferDecision` reason codes before adding new ones. Existing codes cover the Phase 7 cases: `BUFFER_ALLOCATOR_UNAVAILABLE`, `INPUT_BINDING_UNAVAILABLE`, `INPUT_NOT_CPU_CURRENT`, `NATIVE_BUFFER_EXECUTION_FAILED`, and `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE`.
- Portable tests should use fake bridge/allocator implementations and real `ExecutionState`/`ExecutionContext` where possible.
