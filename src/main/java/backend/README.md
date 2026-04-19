# Backend Package

The `backend` layer performs concrete computation over a prepared graph.

It does not build semantic graphs.
It does not run optimizer stages.
It does not own autotune or calibration.

Its contract is:

1. receive a prepared node step
2. receive prepared backend metadata
3. execute the correct kernel family for the selected backend

Today the CPU backend is the only complete backend.

## Reading Guide

Use this package when you need to answer questions such as:

- which kernel family executes this primitive?
- where is backend selection resolved?
- which work is decided in `prepare(...)` and which work is still done at runtime?
- how do matmul / conv2d / fused / reduction dispatch hints reach the executor?

Related docs:

- tensor/public graph builders: [../tensor/README.md](../tensor/README.md)
- primitive descriptors: [../operations/README.md](../operations/README.md)
- compile/prepare lifecycle: [../graph/README.md](../graph/README.md)
- tuning and runtime knob ownership: [../tuning/README.md](../tuning/README.md)

## Main Components

- backend dispatch facade:
  - [ComputeEngine.java](../backend/ComputeEngine.java)
  - [ComputeBackend.java](../backend/ComputeBackend.java)
- concrete backends:
  - [CPUBackend.java](../backend/CPUBackend.java)
  - [CudaBackend.java](../backend/CudaBackend.java)
  - [OpenClBackend.java](../backend/OpenClBackend.java)
- backend prepare layer:
  - [prepare/BackendPrepareDispatcher.java](../backend/prepare/BackendPrepareDispatcher.java)
  - [prepare/CpuNodePreparer.java](../backend/prepare/CpuNodePreparer.java)
- CPU kernel resolution:
  - [registry/CpuKernelResolver.java](../backend/registry/CpuKernelResolver.java)
- runtime context:
  - [runtime/ExecutionContext.java](../backend/runtime/ExecutionContext.java)

## Execution Path

The normal execution path is:

1. `PreparedExecution` iterates prepared steps
2. each step calls `ComputeEngine.compute(compiledNode, metadata, context)`
3. `ComputeEngine` switches on backend
4. `CPUBackend.execute(...)` receives:
   - semantic node
   - compiled node snapshot
   - prepared metadata
   - execution context
5. the resolved CPU kernel executes the node

This means the backend consumes prepared metadata rather than rediscovering policy.

## What Prepare Fixes Before Runtime

For the CPU backend, `prepare(...)` already resolves:

- target backend kind
- compute contract:
  - storage dtype
  - compute dtype
  - accumulate dtype
- kernel selection
- elementwise dispatch hints
- reduction hints
- matmul hints
- conv2d hints
- fused executable preparation
- optional node workspace requirements

Runtime execution should therefore be read as:

- execute the recipe
- not rediscover the recipe

## CPU Package Shape

The root `backend.kernels.cpu` package intentionally keeps shared contracts and planning pieces.
The concrete execution code is split by family.

Important subareas:

- elementwise:
  - contiguous and strided loops
  - unary, binary, compare, logical, and `where`
- reduction
- linalg:
  - matmul
  - linear
  - attention
- nn:
  - conv2d
  - pool2d
- fused
- plan

That split mirrors the semantic organization used in `tensor.ops.*` and `operations.*`.

## CPU Planning Boundary

The CPU backend uses planning objects so that expensive policy decisions do not happen inside hot loops.

Examples of planned metadata:

- elementwise dispatch mode
  - scalar / vector / parallel
- vector width
- planned worker count
- chunk sizes
- reduction accuracy mode
- matmul tiles and microkernel
- conv2d GEMM BLAS-vs-Java decision
- attention direct thresholds and delegated matmul hints

The planner layer is therefore the natural home for runtime threshold interpretation.
The executor layer should stay small and execution-focused.

## Prepared CPU Metadata

The prepared CPU plan for one node may contain:

- layout plan
  - broadcast prep
  - strided-path info
  - where-broadcast info
- compute contract
  - storage/compute/accumulate dtypes
- dispatch hints
  - mode
  - vector width
  - workers
  - scalar/vector chunk sizes
- reduction hints
  - reduction mode
  - workers
  - chunk size
  - vector width
  - accuracy mode
- matmul hints
  - BLAS vs Java
  - tile sizes
  - microkernel
  - work estimate
  - planned workers
- conv2d hints
  - lowered GEMM `m/n/k/work`
  - prepared BLAS-vs-Java selection
- fused executable
  - prepared ASM/vector/direct implementation metadata

That object is not a theoretical IR.
It is a concrete execution recipe for one backend node.

## Runtime Context

`ExecutionContext` is the runtime-scoped holder for:

- execution mode
- approximation policy
- prepared metadata index
- execution state
- family-specific runtime caches

This is the current architectural rule:

- runtime state belongs to the run context
- not to the semantic `Tensor`

Examples of execution-scoped state:

- attention forward/backward auxiliary state
- conv trace metadata for traced execution

## Example: Elementwise Execution

Suppose the graph contains:

```text
y = relu(add(x, b))
```

and:

- `x` has shape `[2, 2]`
- `b` has shape `[2]`

After prepare, the CPU plan may already know:

- this is an elementwise node
- output dtype is `FLOAT64`
- inputs are broadcast-compatible
- the best dispatch mode is vector or scalar

At runtime the backend does not ask:

- "is this add fusable?"
- "should we rewrite this into something else?"

That work is already done earlier.
The runtime simply executes the planned loop/kernel path.

## Matmul and BLAS Routing

Matmul execution currently supports:

- Java microkernel paths
- BLAS routing through the configured provider

The important ownership rule is:

- the backend planner decides whether BLAS or Java should run for the prepared node
- the executor performs the chosen path

The public BLAS thread knob is intentionally canonicalized to `0` (`auto`) in current config.
Synaptik does not try to maintain a separate runtime-managed global BLAS thread policy.

## Conv2d Execution

The conv2d family currently has two conceptual execution shapes:

- direct conv kernels
- lowered GEMM kernels

Graph rewrite can lower:

- `conv2d` -> `conv2dGemm`
- `conv2dBackwardInput` -> `conv2dBackwardInputGemm`
- `conv2dBackwardWeight` -> `conv2dBackwardWeightGemm`

Then CPU preparation resolves BLAS-vs-Java policy for the lowered GEMM nodes.

That split is important:

- graph rewrite decides semantic lowering
- backend preparation decides runtime execution path for the lowered primitive

## Fused Backend

Graph-level fusion produces one `FUSED` node.
Runtime still needs to choose how that fused node runs.

Current fused execution ideas include:

- dispatch family classification
- backend resolution for the fused executable
- ASM/vector/scalar execution preparation

The backend documentation should be read together with [../graph/optimizer/FUSE.md](../graph/optimizer/FUSE.md):

- `FUSE` decides that a cluster should become one primitive
- backend preparation decides how to execute that primitive

## Trace Metadata

When execution is traced, the backend can publish rich step metadata such as:

- dispatch mode and vector width
- reduction chunking
- matmul tiles and microkernel
- conv2d GEMM path details
- fused backend information

That metadata is later consumed by benchmark/reporting code in the tuning layer.

## Architectural Boundaries

These backend rules are intentional:

- executors should not rerun optimizer logic
- executors should not reinterpret graph semantics
- runtime caches should not be written back into semantic tensors
- planner decisions should be made during prepare whenever possible
- tuning knobs should influence planners/config, not ad hoc executor branches

If execution code starts deciding graph-shape policy or recomputing compile-time dispatch boundaries, the design is drifting in the wrong direction.
