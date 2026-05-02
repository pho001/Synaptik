# CUDA Backend

## Purpose And Current Status

The CUDA backend executes selected accelerator DAG regions through the optional native CUDA graph shim when the local native library, CUDA runtime, graph ABI, and buffer ABI are available. It is not a public eager GPU tensor API.

CUDA parity means matching the Metal support-or-rejection evidence standard, not copying Metal support rows.

Current CUDA native execution is intentionally conservative. Dense `FLOAT32` graph and buffer-binding paths are the proven native baseline. Several operation families have shared DAG support rows, but a row is not counted as broad CUDA parity until semantic contract, lowering, legality, native execution, CPU parity, trace/report evidence, and gates all exist.

Public Tensor remains logical; CUDA residency belongs in ExecutionState and DeviceBufferBinding.

## DType Roles

Phase 41 adds role-specific CUDA dtype truth. `dtype residency is not native dtype compute`.

- `FLOAT32` is the proven CUDA native compute/input/output dtype for the dense buffer path.
- `INT32` is represented only as an `INDEX_INPUT` / residency role for index legality evidence. It is not generic CUDA INT32 arithmetic or output support.
- `BOOL` is represented only as a `PREDICATE_INPUT` / residency role. CUDA BOOL-producing compute remains unsupported.
- `BFLOAT16` is residency-only evidence on CUDA until native BF16 execution is implemented.
- `FLOAT64` remains unsupported for CUDA native roles.

CUDA buffer binder diagnostics include backend, role, dtype, and reason code, for example `backend=GPU_CUDA role=COMPUTE_OUTPUT dtype=INT32 code=RESIDENCY_ONLY_NOT_COMPUTE`.

## Current Native Bridge

Main source files:

- `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`
- `src/main/java/backend/cuda/bridge/CudaGraphBridge.java`
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java`
- `src/main/java/backend/cuda/buffer/CudaAcceleratorBufferBinder.java`
- `src/main/native/cuda/synaptik_cuda_graph_stub.cu`

Native lookup uses `-Dsynaptik.cuda.graph.lib=<path>`, then `SYNAPTIK_CUDA_GRAPH_LIB`, then the library name `synaptik_cuda_graph`.

## Capability Dimensions

Phase 40 exposes CUDA capability evidence through `CudaCapabilityReport`:

- `NATIVE_LIBRARY`
- `CUDA_RUNTIME`
- `CONTEXT`
- `GRAPH_EXECUTION_ABI`
- `BUFFER_BINDING_ABI`
- `LAYOUT_ABI_V2`
- `DTYPE_ROLE`
- `DAG_PRIMITIVE`
- `VENDOR_LIBRARY_ROUTE`
- `HARDWARE_DEVICE`
- `TOOLCHAIN`

Capability skip is evidence, not support.

`capabilitySkipCountsAsSupport()` is always false. Optional native CUDA pass/skip output can explain why a local run did not execute CUDA, but it cannot promote a matrix row to `SUPPORTED`.

CUDA vendor-library routing for cuBLAS/cuDNN is not integrated until a dedicated route proves it.

## Supported Rows Versus Parity Gaps

`GpuBackendParityReporter.cudaAgainstMetal()` derives CUDA-vs-Metal parity rows from `GpuLoweringCoverageMatrix`. It reports:

- rows supported by both Metal and CUDA;
- Metal-supported rows that still need CUDA evidence;
- CUDA unsupported/fallback rows grouped by required evidence.

Current CUDA parity gaps include operation families such as dense loss, SDPA, conv/pool, forward gather/take, BOOL-producing compute, and selected training/index-gradient rows. These rows must remain explicit fallback or unsupported evidence until their own CUDA implementation proves correctness.

## Hot Path Blockers

`CudaHotPathBlockerPolicy` classifies hot-path targets into:

- `V16_BLOCKER`
- `REQUIRES_NATIVE_EVIDENCE`
- `ACCEPTED_CAPABILITY_GAP`
- `FUTURE_SCOPE`

The v1.6 blocker set includes CUDA exits for transformer/SDPA, conv/pool, dense loss, gather/take, and BOOL compare/where targets. Scatter/index-gradient and training targets require native execution and parity evidence before support promotion.

## Layout And Index Scope

CUDA layout handling is explicit:

- metadata-only layout views can keep CUDA residency when a later consumer can be made legal;
- dense GPU materialization is scoped to dense `FLOAT32` target layouts and native layout-materialization capability;
- broadcast/zero-stride materialization rejects with `CUDA_LAYOUT_BROADCAST_UNSUPPORTED` until a CUDA native materializer exists;
- arbitrary strided native compute rejects with `CUDA_STRIDED_COMPUTE_UNSUPPORTED`.

Forward `GATHER` and `TAKE_ALONG_AXIS` now validate the scoped CUDA contract before final rejection: dense `FLOAT32` value/output, dense static `INT32` indices, rank/axis/shape compatibility, and in-bounds indices. Legal candidates still end in `CAPABILITY_MISSING` because CUDA native forward gather/take execution has not been implemented. Invalid candidates reject earlier with `UNSUPPORTED_DTYPE`, `UNSUPPORTED_LAYOUT`, `UNSUPPORTED_RANK_OR_SHAPE`, or `UNSUPPORTED_BOUNDS_CHECK`.

## Fallback And Report Interpretation

CUDA fallback interpretation starts with:

- `acceleratorBufferReasonCode`
- `cudaFallbackReason`
- `cudaExecutionPath`
- `cudaSupportsBufferBindings`
- `storageResidency`
- `storageCpuCurrent`
- `storageDeviceCurrent`

Native buffer execution is separate from tensor-array bridge execution. Tensor-array execution and CPU fallback must remain visible in traces and benchmark reports.

## Verification Commands

Portable CUDA parity baseline checks:

```bash
./gradlew test --tests backend.accelerator.lowering.GpuBackendParityReportTest
./gradlew test --tests backend.cuda.bridge.CudaCapabilityReportTest --tests backend.cuda.bridge.CudaFfmBridgeTest
./gradlew test --tests backend.cuda.CudaDTypeRolePolicyTest --tests backend.cuda.buffer.CudaDeviceLayoutMaterializerTest --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests GpuCoverageTriageReportTest --tests GpuHotPathCoverageTargetsTest
./gradlew classes
```

Optional native CUDA check when toolkit and hardware are available:

```bash
./gradlew buildCudaGraphShim cudaTest
```
