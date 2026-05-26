# Backend Package

The `backend` layer performs concrete computation over a prepared graph.

It does not build semantic graphs.
It does not run graph optimization or backend planning.
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
- which work is decided in `prepare(...)` and which work is only per-run state during `execute(...)`?
- how do matmul / conv2d / fused / reduction dispatch hints reach the executor?

Related docs:

- tensor/public graph builders: [../tensor/README.md](../tensor/README.md)
- primitive descriptors: [../operations/README.md](../operations/README.md)
- compile/prepare lifecycle: [../graph/README.md](../graph/README.md)
- tuning and runtime knob ownership: [../tuning/README.md](../tuning/README.md)

## Target Package Ownership

The root `backend` package is a facade/API boundary only.
It should contain backend-neutral contracts such as `ComputeEngine`, `ComputeBackend`, and shared execution modes.

Concrete implementation belongs under backend-owned roots:

- `backend.cpu`
- `backend.metal`
- `backend.cuda`
- `backend.opencl`

Shared accelerator-only artifacts belong under `backend.accelerator`.
Generic backend selection belongs under `backend.select`.
Generic lowering contracts belong under `backend.lowering`.
Generic prepare orchestration belongs under `backend.prepare`.
Backend-specific registries and kernels belong under their backend root.

Root-level concrete backend implementations and CPU-specific helper types have been removed.
Do not add root wrappers for backend-specific implementation classes.

## Main Components

- backend dispatch facade:
  - [ComputeEngine.java](../backend/ComputeEngine.java)
  - [ComputeBackend.java](../backend/ComputeBackend.java)
- concrete backends:
  - target: `backend.cpu.CpuBackend`
  - target: `backend.metal.MetalBackend`
  - target: `backend.cuda.CudaBackend`
  - target: `backend.opencl.OpenClBackend`
- backend prepare layer:
  - [prepare/BackendPrepareDispatcher.java](../backend/prepare/BackendPrepareDispatcher.java)
- backend-specific preparers live under backend roots
- CPU kernel resolution:
  - target: `backend.cpu.kernels.CpuKernelRegistry`
- runtime context:
  - [runtime/ExecutionContext.java](../backend/runtime/ExecutionContext.java)

## Execution Path

The normal execution path is:

1. `PreparedExecution` iterates prepared steps
2. each step calls `ComputeEngine.compute(compiledNode, metadata, context)`
3. `ComputeEngine` switches on backend
4. `backend.cpu.CpuBackend.execute(...)` receives:
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

The target CPU owner root is `backend.cpu`.
CPU runtime kernels should live under `backend.cpu.kernels`.
CPU fused planning/codegen/generated executable code remains under `backend.cpu.fused`.

The legacy split CPU kernel root has been removed.
Do not add CPU runtime code outside `backend.cpu.kernels`.

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
- attention direct thresholds and delegated matmul hints

The planner layer is therefore the natural home for runtime-profile threshold interpretation.
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

Semantic `conv2d` can stay as a direct backend operation or lower to canonical
Tensor primitives (`UNFOLD2D`, `MATMUL`, `RESHAPE`, and optional bias add).

That split is important:

- graph rewrite decides semantic lowering
- backend preparation sees only the resulting primitive nodes
- GEMM dispatch is owned by the normal `MATMUL` path

## Fused Backend

The optimizer produces optimized region units.
CPU lowering turns fused elementwise units into backend-owned fused plan artifacts, and preparation chooses how those artifacts run.

Current fused execution ideas include:

- dispatch family classification
- backend resolution for the fused executable
- ASM/vector/scalar execution preparation

The backend documentation should be read together with [../graph/optimizer/FUSE.md](../graph/optimizer/FUSE.md):

- region optimization decides optimized region shape
- backend lowering builds backend-owned execution artifacts
- backend preparation decides how to execute those artifacts

## Trace Metadata

When execution is traced, the backend can publish rich step metadata such as:

- dispatch mode and vector width
- reduction chunking
- matmul tiles and microkernel
- fused backend information

That metadata is later consumed by benchmark/reporting code in the tuning layer.

## Architectural Boundaries

These backend rules are intentional:

- executors should not rerun optimizer logic
- executors should not reinterpret graph semantics
- runtime caches should not be written back into semantic tensors
- planner decisions should be made during prepare whenever possible
- tuning knobs should influence planners/config, not ad hoc executor branches
- project-owned Metal code should use Metal names; Apple appears only when an error message names the external MPS/MPSGraph framework
- old backend owner packages should be deleted after their migration wave, not kept as forwarding wrappers

If execution code starts deciding graph-shape policy or recomputing compile-time dispatch boundaries, the design is drifting in the wrong direction.
