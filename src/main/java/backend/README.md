# Backend

The backend layer performs the actual computation over a prepared graph. It does not build graphs and it does not own optimizer transformations. Its contract is:

- it receives a `Tensor` node
- it receives prepared metadata from `CompiledGraph.prepare(...)`
- it runs the correct kernel family entry point for the selected backend

Today only the CPU backend is fully implemented.

## Reading Guide

This document is mainly for:

- implementing a new CPU kernel
- auditing runtime dispatch flow
- understanding prepared metadata and the compute contract
- tuning backend performance paths

If you are looking for:

- how the graph is built: [../tensor/README.md](../tensor/README.md)
- what an operation descriptor is: [../operations/README.md](../operations/README.md)
- how the compile/prepare lifecycle works: [../graph/README.md](../graph/README.md)
- how backend knobs are calibrated: [../tuning/README.md](../tuning/README.md)

## Main Components

- dispatch facade
  - [ComputeEngine.java](../backend/ComputeEngine.java)
  - [ComputeBackend.java](../backend/ComputeBackend.java)
- concrete backends
  - [CPUBackend.java](../backend/CPUBackend.java)
  - [CudaBackend.java](../backend/CudaBackend.java)
  - [OpenClBackend.java](../backend/OpenClBackend.java)
- CPU kernel resolver
  - [CpuKernelResolver.java](../backend/registry/CpuKernelResolver.java)
- prepared runtime metadata
  - [CompiledNodeExecutionMetadata.java](../graph/execution/CompiledNodeExecutionMetadata.java)
  - [CpuNodeExecutionPlan.java](../backend/kernels/cpu/CpuNodeExecutionPlan.java)
  - [CpuNodeWorkspace.java](../backend/kernels/cpu/CpuNodeWorkspace.java)

## End-To-End Execution Flow

The real runtime flow looks like this:

1. `CompiledGraph.prepare(runtimeConfig)` creates `PreparedExecution`
2. it prepares `CompiledNodeExecutionMetadata` for every runtime node
3. `PreparedExecution.execute(...)` iterates prepared forward/backward steps
4. each step calls `ComputeEngine.compute(node, metadata, context)`
5. `ComputeEngine` switches to the concrete backend
6. `CPUBackend.execute(...)` takes the prepared plan and invokes the corresponding `CpuKernel`

This is an important boundary:

- the backend must not reinvent optimizer policy
- the hot runtime path must not re-decide work that could already be resolved during prepare

## Prepared CPU Metadata

`CPUBackend.buildExecutionPlan(...)` prepares per-node metadata for the CPU path. It typically contains:

- layout plan
  - prepared/materialized inputs
  - broadcast plan
  - `where` broadcast plan
  - strided-path information
- compute contract
  - storage dtype
  - compute dtype
  - accumulate dtype
  - resolved backend kind
- dispatch hints
  - scalar/vector/parallel mode
  - vector width
  - chunk sizes
  - worker count
- reduction hints
  - reduction mode
  - vector width
  - chunking
  - accuracy mode
- matmul hints
  - BLAS vs Java selection
  - tile sizes
  - parallelism
  - selected microkernel
- optional workspace
  - float continuation
  - packed linear weights
  - max-pool argmax buffers

The result is not a generic abstract "execution descriptor language". It is a concrete runtime recipe for one backend node.

## CPU Package Structure

The root `backend.kernels.cpu` package intentionally contains only shared contracts and planner layers:

- planner
- context
- dispatch hints
- dtype helpers
- workspace
- prepared plan records
- thread pool

The actual family entry points are split thematically:

- `elementwise/`
- `reduction/`
- `linalg/`
- `nn/`
- `index/`
- `layout/`
- `fused/`
- `grad/`

That matches how the CPU backend actually looks in the code today.

## CPU Kernel Design Rules

The current design is not "one giant kernel class for everything". The recurring pattern is:

- `Cpu*Kernel`
  - thin runtime entry point
  - implements `CpuKernel`
  - takes prepared metadata and calls the family executor
- `*Executor`
  - family orchestration
  - validation of family-specific invariants
  - dispatch to the concrete low-level path
- `*Loops` or `*Backend`
  - hot inner loops or specialized compute implementation

Not every family has exactly the same trio of names, but the principle is consistent:

- the runtime entry point stays thin
- orchestration stays out of hot loops
- low-level compute stays localized

## Elementwise Families

The elementwise batch is divided into:

- `elementwise/binary`
- `elementwise/unary`
- `elementwise/compare`
- `elementwise/logical`
- `elementwise/where`

Inside `binary` and `unary`, there are dtype-specialized leaf implementations:

- `f64`
- `f32`
- `bf16`

Architecturally, the implementation separates:

- local operation algebra
- shared loop execution
- broadcasting / stride walking
- vector vs scalar dispatch
- parallel chunk scheduling

This means:

- `CpuAddKernel` does not contain the whole runtime orchestration story
- loop structure is shared inside family executors
- the actual per-element algebra lives in narrower leaf implementations

### Non-Contiguous Routing

CPU uses a hybrid strategy:

- small non-contiguous tensors
  - run through the strided path
  - [CpuStridedElementWise.java](../backend/kernels/cpu/CpuStridedElementWise.java)
- larger non-contiguous tensors
  - inputs are materialized into contiguous temporary storage
  - then the regular fast path runs

Boundary knob:

- `cpu.contiguousMaterializeThreshold`

This belongs to runtime family tuning, not optimizer-stage policy.

## Reduction Families

The CPU reduction path is not one homogeneous branch. It has several family executors depending on operation structure.

### Sum-Like

Shared family for:

- `SUM`
- `MEAN`

Main pieces:

- [SumLikeReduction.java](../backend/kernels/cpu/reduction/SumLikeReduction.java)
- [SumLikeReductionExecutor.java](../backend/kernels/cpu/reduction/SumLikeReductionExecutor.java)
- [SumLoops.java](../backend/kernels/cpu/reduction/SumLoops.java)

The difference between `SUM` and `MEAN` is primarily in finalization, not in the traversal engine.

### Generic Axis Reductions

Shared traversal for:

- `REDUCE_MIN`
- `REDUCE_MAX`
- `REDUCE_ALL`
- `REDUCE_ANY`

Main piece:

- [ReductionTraversal.java](../backend/kernels/cpu/reduction/ReductionTraversal.java)

The traversal layer handles:

- output-group to base-storage-offset mapping
- reduction-axis walking
- optional parallel chunking

Per-op algebra then stays in the leaf reduction implementation.

### Softmax-Like

Shared structured family for:

- `SOFTMAX`
- `LOG_SOFTMAX`

and their gradient families:

- `SOFTMAX_GRAD`
- `LOG_SOFTMAX_GRAD`

Main pieces:

- [SoftmaxLikeReduction.java](../backend/kernels/cpu/reduction/SoftmaxLikeReduction.java)
- [SoftmaxLikeTraversal.java](../backend/kernels/cpu/reduction/SoftmaxLikeTraversal.java)
- [SoftmaxLikeExecutor.java](../backend/kernels/cpu/reduction/SoftmaxLikeExecutor.java)

This is not composed through generic elementwise kernels. It is a specialized structured kernel family with its own traversal logic.

### Loss Reductions

Shared family for:

- `NLL_LOSS`
- `CROSS_ENTROPY_LOSS`
- `CROSS_ENTROPY_LOSS_INDICES`
- gradient variants where they make sense

Main pieces:

- [LossReduction.java](../backend/kernels/cpu/reduction/LossReduction.java)
- [LossReductionTraversal.java](../backend/kernels/cpu/reduction/LossReductionTraversal.java)
- [LossReductionExecutor.java](../backend/kernels/cpu/reduction/LossReductionExecutor.java)

## Linear Algebra Families

### MatMul

The matmul path uses this stack:

- [CpuMatMulKernel.java](../backend/kernels/cpu/linalg/CpuMatMulKernel.java)
- [MatMulExecutor.java](../backend/kernels/cpu/linalg/MatMulExecutor.java)
- [MatMulJavaBackend.java](../backend/kernels/cpu/linalg/MatMulJavaBackend.java)
- [MatMulBlasBackend.java](../backend/kernels/cpu/linalg/MatMulBlasBackend.java)

`MatMulExecutor` decides:

- BLAS vs Java backend
- BF16 continuation policy
- batched vs non-batched flow
- workspace usage

`ResolvedMatMulHints` already carries the resolved runtime recipe:

- tiles
- parallelism
- microkernel
- BLAS enablement

### Linear

`LINEAR` is not implemented as a completely separate GEMM system. It reuses the matmul family and adds:

- the bias epilogue
- packed weight workspace
- BF16 continuation policy

Relevant classes:

- [CpuLinearKernel.java](../backend/kernels/cpu/linalg/CpuLinearKernel.java)
- [LinearExecutor.java](../backend/kernels/cpu/linalg/LinearExecutor.java)

## NN Spatial Families

### Conv2d

There are two main forward paths:

- direct convolution
- GEMM-lowered convolution

Main pieces:

- [Conv2dExecutor.java](../backend/kernels/cpu/nn/Conv2dExecutor.java)
- [Conv2dDirectBackend.java](../backend/kernels/cpu/nn/Conv2dDirectBackend.java)
- [Conv2dGemmExecutor.java](../backend/kernels/cpu/nn/Conv2dGemmExecutor.java)
- [Conv2dGemmBackend.java](../backend/kernels/cpu/nn/Conv2dGemmBackend.java)

The choice between `CONV2D` and `CONV2D_GEMM` is not a backend runtime decision. It is a compile-time rewrite/lowering decision in the optimizer.

### Pool2d

Pool family:

- [Pool2dExecutor.java](../backend/kernels/cpu/nn/Pool2dExecutor.java)
- [Pool2dDirectBackend.java](../backend/kernels/cpu/nn/Pool2dDirectBackend.java)

Max-pool backward additionally reuses prepared int workspace for argmax indices.

### Attention

Attention is no longer forced to run as a decomposed `matmul + softmax + where` runtime path if the rewrite finds the pattern. It can be lowered to specialized primitives:

- `SCALED_DOT_PRODUCT_ATTENTION`
- `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`
- `SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS`

The CPU resolver has explicit kernel entry points for them:

- [CpuScaledDotProductAttentionKernel.java](../backend/kernels/cpu/linalg/CpuScaledDotProductAttentionKernel.java)
- [CpuScaledDotProductAttentionBackwardKernel.java](../backend/kernels/cpu/linalg/CpuScaledDotProductAttentionBackwardKernel.java)
- [CpuScaledDotProductAttentionWeightsKernel.java](../backend/kernels/cpu/linalg/CpuScaledDotProductAttentionWeightsKernel.java)

## Index And Layout Families

### Index

The indexing family includes:

- `GATHER`
- `GATHER_GRAD`
- `TAKE_ALONG_AXIS`
- `TAKE_ALONG_AXIS_GRAD`
- `SCATTER_ADD`

Main pieces:

- [IndexExecutor.java](../backend/kernels/cpu/index/IndexExecutor.java)
- [IndexReadWriteBackend.java](../backend/kernels/cpu/index/IndexReadWriteBackend.java)

These are hard barriers for fused compute. They are not just "elementwise access variants".

### Layout

The layout family handles:

- `RESHAPE`
- `EXPAND`
- `SELECT`
- `PERMUTE`
- `EXPAND_DIMS`
- `SQUEEZE`
- `CONTIGUOUS`

Main piece:

- [LayoutExecutor.java](../backend/kernels/cpu/layout/LayoutExecutor.java)

Important contract:

- some layout ops are alias/view-only
- `CONTIGUOUS` is an explicit materialization node
- `RESHAPE` can be either an alias or a materialization depending on layout reality

## Fused Family

The fused runtime stack today looks like this:

- the optimizer creates a `FusedOperation` descriptor
- `CompiledGraph.prepare(...)` turns it into a `PreparedFusedExecutable`
- `CpuFusedKernel` is the thin entry point
- `FusedExecutor` handles runtime scheduling

Relevant classes:

- [CpuFusedKernel.java](../backend/kernels/cpu/fused/CpuFusedKernel.java)
- [FusedExecutor.java](../backend/kernels/cpu/fused/FusedExecutor.java)
- [PreparedFusedExecutable.java](../graph/fused/PreparedFusedExecutable.java)
- [FusedExecutionBackendResolver.java](../graph/fused/FusedExecutionBackendResolver.java)

### Important Current Reality

The CPU fused backend no longer uses the old direct fused backend.

`FusedExecutionBackendResolver` currently:

- tries the ASM fused backend
- throws if the plan is not supported

That means:

- the prepared fused executable is now generated through the ASM path
- `applyRangeVector(...)` still exists as part of the contract
- the default interface implementation still falls back to scalar
- but actual performance now depends on what the ASM backend generates

### What Controls Runtime Fused Dispatch

`FusedExecutor` uses prepared `ResolvedDispatchHints`:

- `SCALAR`
- `VECTOR`
- `PARALLEL`
- `PARALLEL_VECTOR`

and calls:

- `applyRangeScalar(...)`
- `applyRangeVector(...)`

So a hot fused node still runs as:

- one prepared executable
- one output-space loop structure
- without materializing intermediate tensors

## Compute Contract

The CPU runtime distinguishes between:

- storage dtype
- compute dtype
- accumulate dtype
- resolved backend kind

This matters most for `BFLOAT16`.

Examples:

- `FLOAT64` storage -> compute `FLOAT64`
- `FLOAT32` storage -> compute `FLOAT32`
- `BFLOAT16` storage -> compute `FLOAT32`

This contract is resolved during prepare and stored in `ResolvedCpuComputeContract`.

## BF16 Continuations And Workspace

Some kernels can keep intermediate results in `float[]` workspace longer than just until public tensor materialization.

This mainly applies to:

- `MATMUL`
- `LINEAR`
- `CONV2D_GEMM`
- selected reduction/layout combinations with explicit float workspace

It is important to distinguish two cases:

- intra-kernel continuation
  - the intermediate stays in workspace inside one kernel family
- cross-node continuation
  - a producer publishes float continuation for a specific supported consumer

Cross-node continuation is intentionally narrow and explicitly planned.

## Runtime Dispatch Knobs

The CPU planner reads runtime knobs from `CpuKernelConfig` and related config objects.

Typical knob families:

- elementwise dispatch
- fused dispatch
- reduction dispatch
- scheduler chunking
- materialization threshold
- matmul heuristics
- numerics approximation policy

The tuning surface is described in more detail in:

- [../tuning/KNOBS.md](../tuning/KNOBS.md)

## BLAS Path

CPU matmul can optionally switch to OpenBLAS through FFM.

Relevant runtime properties:

- `cg.cpu.blas.provider=NONE|OPENBLAS_FFM`
- `cg.cpu.blas.matmulMinWork=<long>`
- `cg.cpu.blas.f32RequireMgeK=true|false`
- `cg.cpu.blas.f32MaxNOverK=<double>`
- `cg.cpu.blas.threads=<int>`
- `cg.cpu.blas.debug=true|false`
- `openblas.lib=<absolute-path>`

In practice:

- BLAS is used only for suitable contiguous workloads
- the heuristic for `F32` is stricter than for `F64`
- fallback back to the Java backend is automatic

## Trace And Debug Metadata

`PreparedExecution.executeTraced(...)` can return step-level traces. Backend metadata show up there through:

- compute metadata
- layout metadata
- dispatch metadata
- reduction metadata
- matmul metadata
- fused metadata

This is an important tool for:

- verifying that a benchmark really ran on the expected path
- auditing `vectorWidth`, worker count, tiles, and BLAS use
- finding unexpected scalar fallbacks

## Adding A New CPU Kernel

Recommended process:

1. add an operation descriptor or reuse an existing `Operation.OpType`
2. create a `Cpu*Kernel` entry point in the appropriate family
3. if it is a broader family, create or reuse a `*Executor`
4. keep hot loops in the leaf implementation, not in the resolver
5. register the op in [CpuKernelResolver.java](../backend/registry/CpuKernelResolver.java)
6. add prepare-time hints or workspace if the kernel needs them
7. add execution and regression tests
8. if a new runtime knob is introduced, thread it through the tuning surface and the documentation

## Adding A New Backend

1. add a new enum value to `ComputeBackend`
2. implement the concrete backend class
3. create a registry/resolver for the new backend
4. extend `ComputeEngine.compute(...)`
5. add runtime config and tuning persistence if the backend introduces new knobs
6. do not violate the boundary:
   - the graph still builds `Operation` descriptors
   - the backend still only executes the prepared graph

## Current Limitations

- CPU is the only fully implemented backend
- CUDA/OpenCL are scaffolding
- fused execution on CPU is now an ASM-only prepare path
- some performance optimizations are tightly bound to prepared metadata and are not a general tensor runtime contract
