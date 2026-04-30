# Phase 7: CUDA Buffer Execution And Materialization - Research

## RESEARCH COMPLETE

## Scope

Phase 7 should turn the Phase 6 CUDA buffer-policy seam into a narrow real device-buffer execution path. The key proof is not broad CUDA coverage; it is that CUDA can allocate native buffers, run at least one dense `FLOAT32` operation using buffer handles, publish device-owned outputs into `ExecutionState`, materialize them through the standard CPU-readable path, and reuse a CUDA-owned binding across adjacent CUDA work.

## Existing Facts

- `CudaFfmBridge` already discovers optional buffer symbols for create/destroy/execute-buffer, but Java support is disabled by `CUDA_BUFFER_EXECUTION_ENABLED = false`.
- `CudaFfmBridge` does not yet look up a CUDA read-buffer symbol, expose a buffer allocator, or execute a compiled CUDA graph with `DeviceBufferBinding` handles.
- `CudaAcceleratorBufferBinder` already accepts dense `FLOAT32` metadata and returns stable reason codes for unsupported dtype/layout or missing native ABI.
- `PreparedCudaExecutable` currently converts a policy-accepted `BUFFER_BINDING` decision back to `TENSOR_ARRAY` with reason `BACKEND_BUFFER_NOT_IMPLEMENTED`.
- `ExecutionState` already provides the right runtime hooks: `reserveDeviceBufferBinding`, `attachDeviceBufferBinding`, `deviceBufferBindingForNodeId`, `writableDeviceBufferBindingForNodeId`, `registerDeviceToCpuMaterializer`, `registerResource`, and `requireCpuReadable`.
- Metal's `MetalAcceleratorBufferBinder`, `MetalBufferAllocator`, `MetalDeviceToCpuMaterializer`, and `PreparedMetalExecutable` are the closest working analogs, but Metal's host-shared storage mode should not be copied blindly for CUDA. CUDA outputs should be treated as `DEVICE_OWNED` until materialized.
- The current CUDA native stub stores only node/output counts and returns nonzero from tensor-array execution. Phase 7 needs the native executable to retain enough DAG metadata to run a minimal dense `FLOAT32` operation through buffer handles.

## Recommended Implementation Shape

### Native CUDA Buffer ABI

Add CUDA-owned buffer functions to the native shim and Java bridge:

- `synaptik_cuda_graph_create_buffer(void* context, const void* initialData, int byteLength)`
- `synaptik_cuda_graph_read_buffer(void* context, void* buffer, void* destination, int byteLength)`
- `synaptik_cuda_graph_destroy_buffer(void* buffer)`
- `synaptik_cuda_graph_execute_partition_f32_buffers(void* context, void* executable, void** inputBuffers, int inputCount, void** outputBuffers, int outputCount)`

Use `cudaMalloc`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, and `cudaFree` in the native shim. Store byte length in the buffer handle so native execution can reject undersized buffers.

For the representative operation, start with dense `FLOAT32` elementwise `RELU` or `ADD`. `RELU` is the simplest single-input proof; `ADD` exercises multiple external inputs. The implementation can support both if the evaluator abstraction is already present, but planning should not require broad DAG coverage.

### Java CUDA Buffer Classes

Introduce CUDA-specific classes under `backend.cuda.buffer`:

- `CudaBufferAccess`
- `CudaBufferHandle`
- `CudaBufferBinding implements DeviceBufferBinding`
- `CudaBufferResource implements ExecutionResource`
- `CudaBufferAllocator`
- `CudaDeviceToCpuMaterializer`

`CudaBufferAllocator` should create dense `FLOAT32` input/output bindings, read buffers into CPU `float[]` storage, and destroy owned handles. `CudaDeviceToCpuMaterializer` should mirror Metal support checks: matching dtype, shape, strides, storage offset, logical element count, layout class, and active CUDA binding.

### Bridge And Prepared Execution

Extend `CudaGraphBridge` with CUDA-specific buffer hooks rather than polluting shared accelerator contracts:

- `CudaBufferAllocator createBufferAllocator(CudaBridgeContext context)`
- `void executeBuffers(CudaBridgeContext context, CudaBridgeExecutable executable, List<CudaBufferBinding> inputs, List<CudaBufferBinding> outputs)`

`CudaFfmBridge.supportsBufferBindings()` should require Java support plus create/read/destroy/execute buffer symbols. `PreparedCudaExecutable` should evaluate `CudaAcceleratorBufferBinder`, resolve input/output bindings, execute via `executeBuffers`, then attach output bindings as `StorageResidency.DEVICE_OWNED`.

Fallback rules:

- REQUIRED mode throws before tensor-array or CPU fallback when buffer execution is unavailable.
- AUTO/default mode can fall back to tensor-array or CPU when native buffer execution is unavailable or fails, while publishing a stable `AcceleratorBufferDecision`.
- Accepted buffer execution should not call the legacy tensor-array `CudaGraphBridge.execute(...)`.

### Materialization And Handoff

Use `ExecutionState.requireCpuReadable(...)` for graph-output and CPU-consumer materialization. Tests should assert that `CpuMaterializationTrace` records backend `GPU_CUDA`, residency `DEVICE_OWNED`, reason `GRAPH_OUTPUT` or `CPU_CONSUMER`, logical bytes, and success.

For adjacent handoff, `CudaAcceleratorBufferBinder` and `CudaBufferAllocator` should first check for an existing `CudaBufferBinding` with matching backend, layout, dtype, and readable access. If compatible, reuse it. Otherwise upload CPU storage if current, or reject with `INPUT_BINDING_UNAVAILABLE` / `INPUT_NOT_CPU_CURRENT`.

## Risks And Mitigations

| Risk | Mitigation |
|------|------------|
| CUDA buffer ABI is partially present and over-advertised | Require create, read, destroy, and execute-buffer symbols before `supportsBufferBindings()` returns true. |
| Java marks CPU storage current without copying native data | Materialization only succeeds through `CudaDeviceToCpuMaterializer.materialize(...)`; `ExecutionState` owns the CPU-current transition. |
| Native failures become silent CPU replay | Convert failures to `NATIVE_BUFFER_EXECUTION_FAILED` and keep REQUIRED mode throwing. |
| Adjacent handoff reuses incompatible handles | Validate backend id, `CudaBufferBinding`, layout, dtype, and access mode before reuse. |
| Portable tests require CUDA | Use fake bridges/allocators for Java behavior and gate native `cudaTest` with assumptions/tasks. |

## Validation Architecture

Portable gate:

- `./gradlew classes`
- `./gradlew test --tests backend.cuda.buffer.CudaBufferAllocatorTest`
- `./gradlew test --tests backend.cuda.buffer.CudaDeviceToCpuMaterializerTest`
- `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest`
- `./gradlew test --tests backend.cuda.bridge.CudaFfmBridgeTest`
- `./gradlew test --tests SourceTreeHygieneTest`

Optional native gate:

- `./gradlew buildCudaGraphShim cudaTest`

Native CUDA tests must skip when `nvcc`, CUDA runtime, or hardware is unavailable.

## Planning Recommendation

Use three plans:

1. Add native/Java CUDA buffer ABI, allocator, binding, materializer, capability reporting, and unit tests.
2. Wire `PreparedCudaExecutable` to execute accepted buffer decisions, attach CUDA-owned outputs, and materialize through `ExecutionState`.
3. Prove adjacent CUDA handoff, update docs, run portable/native verification, and close Phase 7 tracking.
