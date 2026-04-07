# Backend (src/main/java/backend)

## Purpose

The backend layer executes graph operations on a selected compute target while keeping graph orchestration (`Tensor`, `CompiledGraph`) separate from device-specific kernel code.

Current backend targets:

- `CPU` (implemented)
- `GPU_CUDA` (scaffolding)
- `GPU_OPENCL` (scaffolding)

## Main Components

- Dispatcher:
  - [src/main/java/backend/ComputeEngine.java](../backend/ComputeEngine.java)
  - [src/main/java/backend/ComputeBackend.java](../backend/ComputeBackend.java)
- Per-device backends:
  - [src/main/java/backend/CPUBackend.java](../backend/CPUBackend.java)
  - [src/main/java/backend/CudaBackend.java](../backend/CudaBackend.java)
  - [src/main/java/backend/OpenClBackend.java](../backend/OpenClBackend.java)
- Kernel registries:
  - [src/main/java/backend/registry/CpuKernelRegistry.java](../backend/registry/CpuKernelRegistry.java)
  - [src/main/java/backend/registry/CudaKernelRegistry.java](../backend/registry/CudaKernelRegistry.java)
  - [src/main/java/backend/registry/OpenClKernelRegistry.java](../backend/registry/OpenClKernelRegistry.java)
- Kernel interfaces/impls:
  - [src/main/java/backend/kernels/cpu/](../backend/kernels/cpu)
  - [src/main/java/backend/kernels/cuda/](../backend/kernels/cuda)
  - [src/main/java/backend/kernels/opencl/](../backend/kernels/opencl)

## Execution Flow

1. `CompiledGraph.prepare(RuntimeConfig)` builds a runtime-specific `PreparedExecution`.
2. `PreparedExecution.execute(...)` iterates prepared forward/backward steps.
3. Each step calls `ComputeEngine.compute(node, metadata, context)`.
4. `ComputeEngine` routes to `CPUBackend`, `CudaBackend`, or `OpenClBackend`.
5. Backend uses prebuilt per-node metadata instead of resolving execution hints from `Tensor`.

Related files:

- [src/main/java/graph/CompiledGraph.java](../graph/CompiledGraph.java)
- [src/main/java/tensor/Tensor.java](../tensor/Tensor.java)

## Prepared Runtime Metadata

`CPUBackend.buildExecutionPlan(...)` precomputes:

- target dtype
- non-contiguous remap preparation (`TensorRemap.RemapPlan`)
- broadcast stride plan (`ResolvedBroadcastPlan`) for binary broadcast kernels
- dispatch hints
- reduction hints
- matmul hints

These are stored in `PreparedExecution`, not on `Tensor`.

## CPU Backend Details

`CPUBackend` executes through `CpuKernel` implementations, `CpuKernelContext`, and prepared per-node plans.

Dispatch modes:

- `SCALAR`
- `VECTOR`
- `PARALLEL`
- `PARALLEL_VECTOR`

Element-wise mode selection is threshold-based in [src/main/java/backend/kernels/cpu/CpuExecutionPlanner.java](../backend/kernels/cpu/CpuExecutionPlanner.java):

- `vectorMinSize`
- `parallelMinSize`
- `contiguousMaterializeThreshold` (non-contiguous input routing threshold)

Reduction (`SUM`) mode selection uses `ResolvedReductionHints`.

Parallel chunking behavior:

- available CPU workers are derived from the runtime environment (`Runtime.getRuntime().availableProcessors()`)
- chunk sizing is derived during `prepare(...)`
- the derivation is driven by advanced CPU scheduler policy values stored in `CpuKernelConfig`
- those values are transparent in execution profiles, but are not part of the default autotune candidate surface
- runtime execution follows prepared chunk sizes instead of recomputing chunk heuristics per call

Parallel execution uses [src/main/java/backend/kernels/cpu/CpuThreadPool.java](../backend/kernels/cpu/CpuThreadPool.java) with per-parallelism `ForkJoinPool` reuse.

Vector dispatch policy can be controlled per operation group via `CpuKernelConfig`:

- `vectorPolicyCheap`
- `vectorPolicyTranscendental`
- `vectorPolicyReduction`

Each policy supports `AUTO | FORCE_ON | FORCE_OFF`.

Non-contiguous input handling is hybrid:

- `size < contiguousMaterializeThreshold`:
  - use strided fallback path in [src/main/java/backend/kernels/cpu/CpuStridedElementWise.java](../backend/kernels/cpu/CpuStridedElementWise.java)
- `size >= contiguousMaterializeThreshold`:
  - materialize non-contiguous input to temporary contiguous tensor and run regular fast kernel path

`CONTIGUOUS` op is excluded from preprocessing and handled directly by `CpuContiguousKernel`.

`SUM` is also excluded from generic preprocessing and handled by its own reduction pipeline:

- [src/main/java/backend/kernels/cpu/CpuSumKernel.java](../backend/kernels/cpu/CpuSumKernel.java)
- [src/main/java/backend/kernels/cpu/reduction/SumExecutor.java](../backend/kernels/cpu/reduction/SumExecutor.java)
- [src/main/java/backend/kernels/cpu/reduction/SumLoops.java](../backend/kernels/cpu/reduction/SumLoops.java)

The reduction pipeline supports:

- `sumAll` and `sum(axis)`
- contiguous fast paths (including vectorized last-dimension reduction)
- strided non-contiguous fallback
- threshold-based materialization to contiguous temporary tensors
- numerical stability mode via `SumAccuracyMode` (`FAST`, `KAHAN`, `NEUMAIER`)

## Backend Tuning Configuration

CPU kernel dispatch parameters are represented by:

- [src/main/java/config/backend/CpuKernelConfig.java](../config/backend/CpuKernelConfig.java)
- [src/main/java/config/backend/SumAccuracyMode.java](../config/backend/SumAccuracyMode.java)

Cross-backend tuning container:

- [src/main/java/config/backend/KernelTuningConfig.java](../config/backend/KernelTuningConfig.java)

At runtime, CPU backend config is carried explicitly in:

- [src/main/java/backend/runtime/RuntimeConfig.java](../backend/runtime/RuntimeConfig.java)
- [src/main/java/backend/runtime/ExecutionContext.java](../backend/runtime/ExecutionContext.java)

Current policy-enabled operations:

- `exp` (can route to `fastExp`)
- `tanh` (can route to `fastTanh`)

The same policy is honored through explicit runtime context in:

- standard CPU kernels
- strided fallback path
- fused codegen helper operations

## Experimental BLAS (FFM)

CPU matmul can optionally route to OpenBLAS via Java FFM API.

Runtime properties:

- `cg.cpu.blas.provider=NONE|OPENBLAS_FFM` (default `NONE`)
- `cg.cpu.blas.matmulMinWork=<long>` (default `2000000`)
- `cg.cpu.blas.debug=true|false` (default `false`)
- `cg.cpu.blas.f32RequireMgeK=true|false` (default `true`)
- `cg.cpu.blas.f32MaxNOverK=<double>` (default `3.0`)
- `cg.cpu.blas.threadPolicy=AUTO|FIXED` (profile/runtime controlled)
- `cg.cpu.blas.threads=<int>` (used when policy is `FIXED`)
- optional library override:
  - `-Dopenblas.lib=<absolute-path-to-library>`
  - or `OPENBLAS_LIB` environment variable

Notes:

- current BLAS routing is used only for contiguous matmul tensors and large enough workloads
- for `FLOAT32`, default heuristic also requires:
  - `m >= k` (`f32RequireMgeK=true`)
  - `n/k <= f32MaxNOverK` (`f32MaxNOverK=3.0`)
  to avoid short-fat / extra-wide regressions
- fallback to Java kernel is automatic on any lookup/call failure
- FFM native access warning can be suppressed by running with:
  - `--enable-native-access=ALL-UNNAMED`

## Compute Mode

CPU execution distinguishes between:

- tensor storage dtype
- resolved compute mode

This matters most for `BFLOAT16`.

Examples:

- `FLOAT64` storage -> `F64`
- `FLOAT32` storage -> `F32`
- `BFLOAT16` storage -> `BF16_F32_COMPUTE`
- `BFLOAT16` storage for eligible GEMM nodes -> `BF16_BLAS`

The compute mode is resolved during prepare and stored in the prepared execution recipe.
Hot execution then follows the recipe instead of re-deciding the path per step.

Important consequence for fused execution:

- `FLOAT64` and `FLOAT32` fused nodes still prepare a generated compiled fused kernel
- `BF16_F32_COMPUTE` fused nodes do not prepare ASM bytecode
- BF16 fused execution runs through the direct Java executor, computing in `float` over `BFLOAT16` storage
- that direct executor now has a vector fast path for contiguous numeric chains built from basic arithmetic / clamp / abs / sqrt style ops
- compare/select/logical and transcendental-heavy chains still fall back to scalar BF16 direct execution

For BLAS-backed BF16 nodes:

- `MATMUL` still materializes its result into `BFLOAT16` because it is itself the public graph result of that node
- `LINEAR` keeps BF16 BLAS output in `float[]` through the internal bias-add phase and only then materializes once to `BFLOAT16`
- `CONV2D_GEMM` already follows the same pattern when scattering GEMM output with bias into the tensor result

So current continuation scope is:

- inside one compound kernel node: yes
- across a subsequent graph node: not yet

There is now one narrow cross-node exception for BF16 inference:

- a `BF16_BLAS` `MATMUL` or `LINEAR` node may publish its result as float continuation instead of immediately materializing to `BFLOAT16`
- this is enabled only when:
  - the graph is inference-only
  - the producer has exactly one consumer
  - that consumer is a supported BF16 consumer shape
- currently supported unary consumers are:
  - `RELU`
  - `ABS`
  - `CLAMP_MIN`
  - `CLAMP_MAX`
  - `SQRT`
  - `EXP`
  - `FAST_EXP`
  - `LOG`
  - `TANH`
  - `FAST_TANH`
  - `SIGMOID`
  - `INV`
- currently supported binary consumers are:
  - `ADD`
  - `SUB`
  - `MUL`
  - `DIV`
  - `MIN`
  - `MAX`
  - only in no-broadcast shape-equal contiguous cases
- currently supported fused consumers are:
  - numeric-only contiguous BF16 fused chains
  - no bool/select/compare/logical semantics
  - all external inputs must use linear access

This is intentionally narrow:

- no branching fan-out
- no training/backward graphs
- no broadcast consumers
- no cross-node continuation for arbitrary tensors

That keeps the storage contract small:

- graph/runtime still stores tensors as `BFLOAT16`
- planner decides whether the node should compute as float-backed direct execution or as BLAS-backed BF16 GEMM
- prepare stores only the executable artifact that the resolved mode actually needs

## Profile-Driven Runtime Configuration

Optimizer profiles are used as runtime source of tuning knobs:

- training profile path: `config/optimizer-profile.json`
- hardware-bucket profile path: `config/optimizer-hw-profiles.tsv`
- autotune training winner: `build/optimizer-autotune/best-profile-training.json`
- autotune inference winner: `build/optimizer-autotune/best-profile-inference.json`

[src/main/java/graph/optimizer/OptimizerFactory.java](../graph/optimizer/OptimizerFactory.java):

- builds compile-time optimizers only

Profile chain priority:

1. HW-bucket profile (`optimizer-hw-profiles.tsv`) for current machine.
2. Architecture preset (`os.arch`, including ARM/aarch64 and x86_64/amd64).
3. Persisted best-profile override (`best-profile-*.json`).
4. Defaults.

This means backend dispatch thresholds/chunking are profile-driven, with robust fallbacks before autotune is run.

Optimizer-side memory planning now also exposes explain/summary hooks through the `graph.optimizer.memory` package.
This is currently primarily an optimizer/planner concern rather than a backend runtime API, but it is relevant when investigating memory reuse behavior and peak-memory tradeoffs in prepared execution flows.

Related numerics diagnostics tooling:

- [src/main/java/numerics/README.md](../numerics/README.md)

## CUDA and OpenCL Status

CUDA and OpenCL backends are intentionally scaffolded.

Current registries map only `NOOP`:

- [src/main/java/backend/registry/CudaKernelRegistry.java](../backend/registry/CudaKernelRegistry.java)
- [src/main/java/backend/registry/OpenClKernelRegistry.java](../backend/registry/OpenClKernelRegistry.java)

Missing kernels throw `UnsupportedOperationException` at execution time.

## Adding a New CPU Operation Kernel

1. Implement `CpuKernel` in `src/main/java/backend/kernels/cpu/` (forward path and mode dispatch).
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
