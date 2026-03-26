# Backend (src/Backend)

## Purpose

The backend layer executes graph operations on a selected compute target while keeping graph orchestration (`Tensor`, `CompiledGraph`) separate from device-specific kernel code.

Current backend targets:

- `CPU` (implemented)
- `GPU_CUDA` (scaffolding)
- `GPU_OPENCL` (scaffolding)

## Main Components

- Dispatcher:
  - [src/Backend/ComputeEngine.java](../Backend/ComputeEngine.java)
  - [src/Backend/ComputeBackend.java](../Backend/ComputeBackend.java)
- Per-device backends:
  - [src/Backend/CPUBackend.java](../Backend/CPUBackend.java)
  - [src/Backend/CudaBackend.java](../Backend/CudaBackend.java)
  - [src/Backend/OpenClBackend.java](../Backend/OpenClBackend.java)
- Kernel registries:
  - [src/Backend/registry/CpuKernelRegistry.java](../Backend/registry/CpuKernelRegistry.java)
  - [src/Backend/registry/CudaKernelRegistry.java](../Backend/registry/CudaKernelRegistry.java)
  - [src/Backend/registry/OpenClKernelRegistry.java](../Backend/registry/OpenClKernelRegistry.java)
- Kernel interfaces/impls:
  - [src/Backend/kernels/cpu/](../Backend/kernels/cpu)
  - [src/Backend/kernels/cuda/](../Backend/kernels/cuda)
  - [src/Backend/kernels/opencl/](../Backend/kernels/opencl)

## Execution Flow

1. `CompiledGraph.execute()` iterates compiled nodes.
2. For each executable node, it calls `ComputeEngine.compute(tensor, tensor.getResolvedBackend())`.
3. `ComputeEngine` routes to `CPUBackend`, `CudaBackend`, or `OpenClBackend`.
4. Backend resolves kernel for `opType` (usually via registry).
5. Kernel `forward(...)` performs the operation.

Related files:

- [src/Graph/CompiledGraph.java](../Graph/CompiledGraph.java)
- [src/Tensor/Tensor.java](../Tensor/Tensor.java)

## Compile-Time Resolution and Runtime Overhead

During graph compilation, backend and CPU execution metadata are pre-resolved per node:

- `Tensor.setResolvedBackend(...)`
- `Tensor.setResolvedCpuKernel(...)`
- `Tensor.setResolvedCpuExecutionPlan(...)`
- `Tensor.setResolvedBroadcastPlan(...)`
- `Tensor.setResolvedCpuConfigEpoch(...)`

`CPUBackend.execute(...)` first uses pre-resolved kernel from node cache and only falls back to `CpuKernelRegistry.resolve(...)` if cache is missing. This keeps hot-path dispatch overhead low.

Execution plan currently precomputes:

- target dtype
- non-contiguous remap preparation (`TensorRemap.RemapPlan`)
- broadcast stride plan (`ResolvedBroadcastPlan`) for binary broadcast kernels
- dispatch hints (mode + chunk sizing) used by `CpuExecutionConfig`

Plan staleness is guarded by `ComputeEngine` CPU-config epoch, so runtime rebuild happens only when kernel config changes.

Debug toggles:

- `-Dcg.cpu.disableResolveExecutionHints=true` disables compile-time resolve in `CompiledGraph`.
- `-Dcg.cpu.disablePreResolvedExecutionPlan=true` disables execution-plan reuse in `CPUBackend`.

## CPU Backend Details

`CPUBackend` executes through `CpuKernel` implementations and a runtime `CpuExecutionConfig`.

Dispatch modes:

- `SCALAR`
- `VECTOR`
- `PARALLEL`
- `PARALLEL_VECTOR`

Element-wise mode selection is threshold-based in [src/Backend/kernels/cpu/CpuExecutionConfig.java](../Backend/kernels/cpu/CpuExecutionConfig.java):

- `vectorMinSize`
- `parallelMinSize`
- `contiguousMaterializeThreshold` (non-contiguous input routing threshold)

Reduction (`SUM`) mode selection uses the same mode set and threshold logic via `modeForReduction(...)`.

Parallel chunking knobs:

- `parallelism` (`0` = use available processors)
- `chunksPerWorker`
- `minChunkSize`

Parallel execution uses [src/Backend/kernels/cpu/CpuThreadPool.java](../Backend/kernels/cpu/CpuThreadPool.java) with per-parallelism `ForkJoinPool` reuse.

Vector dispatch policy can be controlled per operation group via `CpuKernelConfig`:

- `vectorPolicyCheap`
- `vectorPolicyTranscendental`
- `vectorPolicyReduction`

Each policy supports `AUTO | FORCE_ON | FORCE_OFF`.

Non-contiguous input handling is hybrid:

- `size < contiguousMaterializeThreshold`:
  - use strided fallback path in [src/Backend/kernels/cpu/CpuStridedElementWise.java](../Backend/kernels/cpu/CpuStridedElementWise.java)
- `size >= contiguousMaterializeThreshold`:
  - materialize non-contiguous input to temporary contiguous tensor and run regular fast kernel path

`CONTIGUOUS` op is excluded from preprocessing and handled directly by `CpuContiguousKernel`.

`SUM` is also excluded from generic preprocessing and handled by its own reduction pipeline:

- [src/Backend/kernels/cpu/CpuSumKernel.java](../Backend/kernels/cpu/CpuSumKernel.java)
- [src/Backend/kernels/cpu/reduction/SumExecutor.java](../Backend/kernels/cpu/reduction/SumExecutor.java)
- [src/Backend/kernels/cpu/reduction/SumLoops.java](../Backend/kernels/cpu/reduction/SumLoops.java)

The reduction pipeline supports:

- `sumAll` and `sum(axis)`
- contiguous fast paths (including vectorized last-dimension reduction)
- strided non-contiguous fallback
- threshold-based materialization to contiguous temporary tensors
- numerical stability mode via `SumAccuracyMode` (`FAST`, `KAHAN`, `NEUMAIER`)

## Backend Tuning Configuration

CPU kernel dispatch parameters are represented by:

- [src/Config/backend/CpuKernelConfig.java](../Config/backend/CpuKernelConfig.java)
- [src/Config/backend/SumAccuracyMode.java](../Config/backend/SumAccuracyMode.java)

Cross-backend tuning container:

- [src/Config/backend/KernelTuningConfig.java](../Config/backend/KernelTuningConfig.java)

At runtime, CPU backend config is applied through:

- `ComputeEngine.setCpuKernelConfig(...)`

Approximation policy is applied through:

- `ComputeEngine.setApproxMode(...)`
- enum: [src/Backend/ApproxMode.java](../Backend/ApproxMode.java)
- modes:
  - `OFF` (always exact)
  - `TRAINING_ONLY` (fast approximation only during training execution)
  - `ALWAYS` (fast approximation in both inference and training)

Current policy-enabled operations:

- `exp` (can route to `fastExp`)
- `tanh` (can route to `fastTanh`)

The same policy is honored in:

- standard CPU kernels
- strided fallback path
- fused codegen helper operations

## Experimental BLAS (FFM)

CPU matmul can optionally route to OpenBLAS via Java FFM API.

Runtime properties:

- `cg.cpu.blas.provider=NONE|OPENBLAS_FFM` (default `NONE`)
- `cg.cpu.blas.matmulMinWork=<long>` (default `2000000`)
- `cg.cpu.blas.debug=true|false` (default `false`)
- optional library override:
  - `-Dopenblas.lib=<absolute-path-to-library>`
  - or `OPENBLAS_LIB` environment variable

Notes:

- current BLAS routing is used only for contiguous matmul tensors and large enough workloads
- fallback to Java kernel is automatic on any lookup/call failure
- FFM native access warning can be suppressed by running with:
  - `--enable-native-access=ALL-UNNAMED`

## Profile-Driven Runtime Configuration

Optimizer profiles are used as runtime source of tuning knobs:

- training profile path: `config/optimizer-profile.json`
- hardware-bucket profile path: `config/optimizer-hw-profiles.tsv`
- autotune training winner: `build/optimizer-autotune/best-profile-training.json`
- autotune inference winner: `build/optimizer-autotune/best-profile-inference.json`

[src/Graph/optimizer/OptimizerFactory.java](../Graph/optimizer/OptimizerFactory.java):

- `createRecommendedTrainingOptimizer()` loads profile chain and applies CPU kernel config.
- `createInferencePerformanceOptimizer()` loads profile chain and applies CPU kernel config.

Profile chain priority:

1. HW-bucket profile (`optimizer-hw-profiles.tsv`) for current machine.
2. Architecture preset (`os.arch`, including ARM/aarch64 and x86_64/amd64).
3. Persisted best-profile override (`best-profile-*.json`).
4. Defaults.

This means backend dispatch thresholds/chunking are profile-driven, with robust fallbacks before autotune is run.

Related numerics diagnostics tooling:

- [src/Numerics/README.md](../Numerics/README.md)

## CUDA and OpenCL Status

CUDA and OpenCL backends are intentionally scaffolded.

Current registries map only `NOOP`:

- [src/Backend/registry/CudaKernelRegistry.java](../Backend/registry/CudaKernelRegistry.java)
- [src/Backend/registry/OpenClKernelRegistry.java](../Backend/registry/OpenClKernelRegistry.java)

Missing kernels throw `UnsupportedOperationException` at execution time.

## Adding a New CPU Operation Kernel

1. Implement `CpuKernel` in `src/Backend/kernels/cpu/` (forward path and mode dispatch).
2. Register new op type in `CpuKernelRegistry`.
3. Ensure operation class reports correct `opType`.
4. Add regression tests and benchmark coverage.

## Adding a New Backend

1. Add new enum value to `ComputeBackend`.
2. Implement backend class (similar to `CPUBackend`/`CudaBackend`/`OpenClBackend`).
3. Add kernel interface + registry for that backend.
4. Extend `ComputeEngine.compute(...)` switch.
5. Extend tuning config objects and profile I/O if backend has runtime knobs.
6. Add benchmark candidate support if backend should be autotuned.

## Known Limitations

- CPU is the only fully implemented execution backend.
- CUDA/OpenCL registries are placeholder-level today.
- Backend profile loading currently focuses on optimizer-driven entry points.
