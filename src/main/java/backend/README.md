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
  - [src/main/java/backend/registry/CpuKernelResolver.java](../backend/registry/CpuKernelResolver.java)
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

- `cheapVectorMinSize`
- `transcendentalVectorMinSize`
- `reductionVectorMinSize`

Each policy supports `AUTO | FORCE_ON | FORCE_OFF`.

Non-contiguous input handling is hybrid:

- `size < contiguousMaterializeThreshold`:
  - use strided fallback path in [src/main/java/backend/kernels/cpu/CpuStridedElementWise.java](../backend/kernels/cpu/CpuStridedElementWise.java)
- `size >= contiguousMaterializeThreshold`:
  - materialize non-contiguous input to temporary contiguous tensor and run regular fast kernel path

`CONTIGUOUS` op is excluded from preprocessing and handled directly by `CpuContiguousKernel`.

`SUM` is also excluded from generic preprocessing and handled by its own reduction pipeline:

- [src/main/java/backend/kernels/cpu/CpuSumKernel.java](../backend/kernels/cpu/CpuSumKernel.java)
- [src/main/java/backend/kernels/cpu/reduction/SumLikeReductionExecutor.java](../backend/kernels/cpu/reduction/SumLikeReductionExecutor.java)
- [src/main/java/backend/kernels/cpu/reduction/SumLoops.java](../backend/kernels/cpu/reduction/SumLoops.java)

CPU family packages now follow one naming rule:

- `Cpu*Kernel`
  - graph/runtime entrypoint resolved by `CpuKernelResolver`
- `*Executor`
  - family orchestration layer
  - validates family-specific execution path and dispatches to the concrete low-level implementation
- `*Loops`
  - tight low-level loop implementation

This keeps orchestration separate from the actual inner compute loops without adding a new global abstraction hierarchy.

The root package `backend.kernels.cpu` now intentionally keeps only:

- shared contracts
- prepared execution metadata
- planner / context / workspace
- shared dispatch hints
- shared runtime utilities

Operation entrypoints now live in their family packages:

- `elementwise/`
- `reduction/`
- `linalg/`
- `nn/`
- `index/`
- `layout/`
- `fused/`

For the pointwise batch, CPU now also uses an explicit split between:

- local element algebra
  - for example numeric binary ops such as `ADD/SUB/MUL/DIV/MIN/MAX`
  - unary ops such as `NEG/ABS/INV/SQRT/EXP/LOG/TANH/SIGMOID/RELU`
  - scalar-parameter unary ops such as `CLAMP_*`, `MUL_SCALAR`, `POW`
  - compare ops
  - logical ops
  - `WHERE`
- shared loop execution
  - scalar loop
  - vector loop where available
  - parallel chunk scheduling
  - broadcast/stride walking

Implementation lives in:

- [src/main/java/backend/kernels/cpu/elementwise/](../backend/kernels/cpu/elementwise)

This means the operation definition is no longer encoded as a hand-written full loop inside each kernel class.
Instead:

- `Cpu*Kernel`
  - selects the family entrypoint
- elementwise op descriptor
  - defines the local per-element algebra
- shared executor
  - owns loop structure and dispatch policy

For simple reductions, CPU now follows the same direction for shared traversal:

- `REDUCE_MIN/MAX`
- `REDUCE_ALL/ANY`

These reductions now share common axis-group traversal logic in:

- [src/main/java/backend/kernels/cpu/reduction/ReductionTraversal.java](../backend/kernels/cpu/reduction/ReductionTraversal.java)

That shared layer owns:

- reduction-axis group walking
- shape/stride to base-offset mapping
- optional reduction parallel chunk scheduling for supported families

while the per-op algebra stays in the leaf reduction implementation.

For sum-like reductions, CPU now shares one family orchestration layer across:

- `SUM`
- `MEAN`

Implementation lives in:

- [src/main/java/backend/kernels/cpu/reduction/SumLikeReduction.java](../backend/kernels/cpu/reduction/SumLikeReduction.java)
- [src/main/java/backend/kernels/cpu/reduction/SumLikeReductionExecutor.java](../backend/kernels/cpu/reduction/SumLikeReductionExecutor.java)

The fold engine remains in `SumLoops`, while `SUM` vs `MEAN` differs only in finalization.

For softmax-like structured reductions, CPU now also shares one family layer across:

- `SOFTMAX`
- `LOG_SOFTMAX`

Implementation lives in:

- [src/main/java/backend/kernels/cpu/reduction/SoftmaxLikeReduction.java](../backend/kernels/cpu/reduction/SoftmaxLikeReduction.java)
- [src/main/java/backend/kernels/cpu/reduction/SoftmaxLikeTraversal.java](../backend/kernels/cpu/reduction/SoftmaxLikeTraversal.java)
- [src/main/java/backend/kernels/cpu/reduction/SoftmaxLikeExecutor.java](../backend/kernels/cpu/reduction/SoftmaxLikeExecutor.java)

This shared layer owns:

- group traversal over the softmax axis
- dtype-specific execution routing
- BF16 continuation materialization / float-publish handling for `LOG_SOFTMAX`

while the algebraic difference between `SOFTMAX` and `LOG_SOFTMAX` stays in the reduction descriptor.

For loss reductions, CPU now also shares one structured family layer across:

- `NLL_LOSS`
- `CROSS_ENTROPY_LOSS`

Implementation lives in:

- [src/main/java/backend/kernels/cpu/reduction/LossReduction.java](../backend/kernels/cpu/reduction/LossReduction.java)
- [src/main/java/backend/kernels/cpu/reduction/LossReductionTraversal.java](../backend/kernels/cpu/reduction/LossReductionTraversal.java)
- [src/main/java/backend/kernels/cpu/reduction/LossReductionExecutor.java](../backend/kernels/cpu/reduction/LossReductionExecutor.java)

This shared layer owns:

- class-axis group traversal
- scalar loss aggregation over groups
- dtype-specific routing
- BF16 continuation handling

while the difference between `NLL_LOSS` and `CROSS_ENTROPY_LOSS` stays in the reduction descriptor algebra.

For linear algebra kernels, CPU now separates:

- `CpuMatMulKernel`
  - thin runtime entrypoint
- `MatMulExecutor`
  - execution orchestration
  - chooses BLAS vs Java backend
  - handles BF16 continuation publish/materialize policy
- `MatMulBlasBackend`
  - OpenBLAS/FFM dispatch
- `MatMulJavaBackend`
  - tiled Java compute backend

Implementation lives in:

- [src/main/java/backend/kernels/cpu/linalg/CpuMatMulKernel.java](../backend/kernels/cpu/linalg/CpuMatMulKernel.java)
- [src/main/java/backend/kernels/cpu/linalg/MatMulExecutor.java](../backend/kernels/cpu/linalg/MatMulExecutor.java)
- [src/main/java/backend/kernels/cpu/linalg/MatMulBlasBackend.java](../backend/kernels/cpu/linalg/MatMulBlasBackend.java)
- [src/main/java/backend/kernels/cpu/linalg/MatMulJavaBackend.java](../backend/kernels/cpu/linalg/MatMulJavaBackend.java)

`LINEAR` now reuses the same matmul executor and adds only the bias epilogue / BF16 continuation policy in:

- [src/main/java/backend/kernels/cpu/linalg/LinearExecutor.java](../backend/kernels/cpu/linalg/LinearExecutor.java)

For neural-network spatial kernels, CPU now follows the same separation:

- `CpuConv2dKernel` / backward kernels
  - thin runtime entrypoints
- `Conv2dExecutor`
  - family orchestration layer
- `Conv2dDirectBackend`
  - direct convolution compute implementation

- `CpuConv2dGemmKernel`
  - thin runtime entrypoint
- `Conv2dGemmExecutor`
  - family orchestration layer
- `Conv2dGemmBackend`
  - im2col + GEMM backend implementation

- `CpuMaxPool2dKernel`, `CpuAvgPool2dKernel`, backward kernels
  - thin runtime entrypoints
- `Pool2dExecutor`
  - family orchestration layer
- `Pool2dDirectBackend`
  - pooling compute implementation

Implementation lives in:

- [src/main/java/backend/kernels/cpu/nn/Conv2dExecutor.java](../backend/kernels/cpu/nn/Conv2dExecutor.java)
- [src/main/java/backend/kernels/cpu/nn/Conv2dDirectBackend.java](../backend/kernels/cpu/nn/Conv2dDirectBackend.java)
- [src/main/java/backend/kernels/cpu/nn/Conv2dGemmExecutor.java](../backend/kernels/cpu/nn/Conv2dGemmExecutor.java)
- [src/main/java/backend/kernels/cpu/nn/Conv2dGemmBackend.java](../backend/kernels/cpu/nn/Conv2dGemmBackend.java)
- [src/main/java/backend/kernels/cpu/nn/Pool2dExecutor.java](../backend/kernels/cpu/nn/Pool2dExecutor.java)
- [src/main/java/backend/kernels/cpu/nn/Pool2dDirectBackend.java](../backend/kernels/cpu/nn/Pool2dDirectBackend.java)

For fused kernels, CPU now separates:

- `CpuFusedKernel`
  - thin runtime entrypoint
- `FusedExecutor`
  - fused dispatch orchestration
  - scalar/vector/parallel scheduling
  - profiler integration
- `PreparedFusedExecutable`
  - prepared backend implementation contract
  - implemented today by ASM-generated and direct fused prepared executables

Implementation lives in:

- [src/main/java/backend/kernels/cpu/fused/CpuFusedKernel.java](../backend/kernels/cpu/fused/CpuFusedKernel.java)
- [src/main/java/backend/kernels/cpu/fused/FusedExecutor.java](../backend/kernels/cpu/fused/FusedExecutor.java)
- [src/main/java/graph/fused/PreparedFusedExecutable.java](../graph/fused/PreparedFusedExecutable.java)

For indexing kernels, CPU now separates:

- `CpuGatherKernel`, `CpuTakeAlongAxisKernel`, `CpuScatterAddKernel`, gradient variants
  - thin runtime entrypoints
- `IndexExecutor`
  - family orchestration layer
- `IndexReadWriteBackend`
  - specialized hot loops for gather / scatter / takeAlongAxis / scatterAdd

Implementation lives in:

- [src/main/java/backend/kernels/cpu/index/IndexExecutor.java](../backend/kernels/cpu/index/IndexExecutor.java)
- [src/main/java/backend/kernels/cpu/index/IndexReadWriteBackend.java](../backend/kernels/cpu/index/IndexReadWriteBackend.java)

For layout kernels, CPU uses one small family executor:

- alias/view-like ops
  - `SELECT`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`
- reshape-like ops
  - alias when possible, materialize when necessary
- `CONTIGUOUS`
  - explicit materialization through `TensorRemap`

Implementation lives in:

- [src/main/java/backend/kernels/cpu/layout/LayoutExecutor.java](../backend/kernels/cpu/layout/LayoutExecutor.java)

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
- `cg.cpu.blas.threads=<0|1|2|4|...>` (`0` means auto, profile/runtime controlled)
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

## Compute Contract

CPU execution distinguishes between:

- tensor storage dtype
- resolved compute dtype
- resolved execution backend
- resolved accumulate dtype where relevant

This matters most for `BFLOAT16`.

Examples:

- `FLOAT64` storage -> compute `F64` -> backend `CPU_*`
- `FLOAT32` storage -> compute `F32` -> backend `CPU_*`
- `BFLOAT16` storage -> compute `F32` -> backend `CPU_FUSED`
- `BFLOAT16` storage -> compute `F32` -> backend `CPU_MATMUL_BLAS`

The resolved compute contract is built during `prepare(...)` and stored in the prepared execution recipe.
Hot execution then follows the recipe instead of re-deciding the path per step.

Important consequence for fused execution:

- `FLOAT64` and `FLOAT32` fused nodes still prepare a generated compiled fused kernel
- `BFLOAT16` fused nodes use `BFLOAT16` storage but currently compute as `F32` in the direct runtime backend
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

- a `CPU_MATMUL_BLAS` BF16 `MATMUL` or `LINEAR` node may publish its result as float continuation instead of immediately materializing to `BFLOAT16`
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
2. Register new op type in `CpuKernelResolver`.
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
