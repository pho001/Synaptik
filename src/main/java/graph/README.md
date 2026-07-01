# Graph Package

The `graph` package owns the immutable compiled graph model and graph compilation workflow.

Its job is not to build graphs and not to execute kernels directly.
Its job is to manage the lifecycle between those two moments:

1. take the public tensor graph built by `tensor`
2. canonicalize and optimize it
3. snapshot compile-time structure
4. hand immutable compile artifacts to prepare orchestration
5. expose compile and prepare through the single `graph.CompiledGraph` lifecycle facade

Backend-specific preparation is not graph work. Each `backend.<name>.prepare` package is
that backend's compiler; it turns backend-neutral plans into immutable executable state.
Runtime then invokes the prepared executable contract directly, without a central
`ComputeEngine` switch.

## Mental Model

You should keep three artifacts separate at all times.

### 1. Public graph: `Tensor`

Built by the public tensor API.

Contains:

- operation descriptors
- input edges
- dtype/shape/storage metadata
- backward lambdas
- published output data and published gradients

Does not contain:

- prepared CPU dispatch hints
- prepared reduction plans
- prepared matmul or conv2d plans
- prepared fused executable objects

### 2. Lifecycle facade: `CompiledGraph`

Built from a semantic graph and a `CompileConfig`. It is the only class in `graph` that
bridges compilation to prepare orchestration and public runtime execution types.

Contains:

- compile-time node ordering
- forward/backward boundary
- graph optimization and planning output
- compiled-node snapshots
- gradient binding metadata
- compile trace

### 3. Runtime-bound artifact: `PreparedExecution`

Built from a compiled graph and a runtime config.

Contains:

- ordered forward steps
- ordered backward steps
- per-node prepared backend metadata
- runtime config
- prepare trace

`PreparedExecution` is the object you want to reuse in hot loops when graph structure stays the same.

## Main Classes

- compile artifact:
  - [CompiledGraph.java](../graph/CompiledGraph.java)
- compile orchestration and graph snapshot support:
  - [compile/GraphCompiler.java](../graph/compile/GraphCompiler.java)
  - [compile/CompileArtifacts.java](../graph/compile/CompileArtifacts.java)
  - [compile/session/CompileSession.java](../graph/compile/session/CompileSession.java)
  - [compile/session/BackwardGraphBuilder.java](../graph/compile/session/BackwardGraphBuilder.java)
  - [compile/session/GradientBindingCollector.java](../graph/compile/session/GradientBindingCollector.java)
  - [compile/session/OptimizerGraphSnapshot.java](../graph/compile/session/OptimizerGraphSnapshot.java)
  - [planning/backend/BackendPlanningService.java](../planning/backend/BackendPlanningService.java)
  - [planning/backend/BackendPlanningJobResolver.java](../planning/backend/BackendPlanningJobResolver.java)
  - [compile/canonical/SemanticForwardCanonicalizer.java](../graph/compile/canonical/SemanticForwardCanonicalizer.java)
- prepare pipeline:
  - [prepare/orchestration/PreparedExecutionBuilder.java](../prepare/orchestration/PreparedExecutionBuilder.java)
  - [runtime/execution/PreparedExecution.java](../runtime/execution/PreparedExecution.java)
  - [runtime/execution/PreparedExecutionStep.java](../runtime/execution/PreparedExecutionStep.java)
  - [runtime/execution/PreparedStepMetadata.java](../runtime/execution/PreparedStepMetadata.java)
  - [runtime/execution/ExecutionState.java](../runtime/execution/ExecutionState.java)
  - [runtime/state/RuntimeTensorStore.java](../runtime/state/RuntimeTensorStore.java)
  - [runtime/execution/RuntimeWorkspaceStore.java](../runtime/execution/RuntimeWorkspaceStore.java)
  - [runtime/state/RuntimeNativeCpuMemoryState.java](../runtime/state/RuntimeNativeCpuMemoryState.java)
  - [runtime/state/RuntimeDeviceMemoryState.java](../runtime/state/RuntimeDeviceMemoryState.java)
  - [runtime/state/RuntimeMaterializationService.java](../runtime/state/RuntimeMaterializationService.java)
  - [runtime/state/RuntimeResourceRegistry.java](../runtime/state/RuntimeResourceRegistry.java)
  - [runtime/residency/RuntimeResidencyStore.java](../runtime/residency/RuntimeResidencyStore.java)
  - [runtime/residency/RuntimeMemoryBinder.java](../runtime/residency/RuntimeMemoryBinder.java)
  - [runtime/publication/ExecutionPublisher.java](../runtime/publication/ExecutionPublisher.java)
- traces:
  - [trace/compile/CompileTrace.java](../trace/compile/CompileTrace.java)
  - [trace/prepare/PrepareTrace.java](../trace/prepare/PrepareTrace.java)
  - [trace/execution/RunTrace.java](../trace/execution/RunTrace.java)
- fused preparation now lives under backend CPU ownership:
  - [../backend/cpu/fused/plan/FusedExecutionPlan.java](../backend/cpu/fused/plan/FusedExecutionPlan.java)
  - [../backend/cpu/fused/exec/FusedExecutablePreparer.java](../backend/cpu/fused/exec/FusedExecutablePreparer.java)
  - [../backend/cpu/fused/exec/PreparedFusedExecutable.java](../backend/cpu/fused/exec/PreparedFusedExecutable.java)

## Compile Pipeline

`CompiledGraph.compile(root, compileConfig, compileMode)` is the public entry point. Internally,
`graph.compile.GraphCompiler` is the compile boundary, `graph.compile.session.CompileSession`
owns the mutable compile run, and `CompileArtifacts` is the immutable handoff to the facade
and prepare layer.

### Step 1: choose the semantic forward output

The compile root is normalized through:

- `rootTensor.forwardOutput()`

This matters because the semantic root may be a publication wrapper rather than the true final forward node.

### Step 2: initialize the working forward graph

If semantic forward canonicalization is enabled in the compile configuration, `CompiledGraph` starts compilation from a canonicalized forward graph snapshot.

This stage can normalize decomposed forward patterns before the main optimizer runs.

### Step 3: decide whether backward should be compiled

Backward compilation depends on:

- the `CompileMode`
- whether the forward graph has trainable leaf tensors

Current logic:

- `INFERENCE_ONLY`
  - never compile backward
- `TRAINING`
  - compile backward only if the graph actually has trainable leaves
- `AUTO`
  - same practical behavior as `TRAINING`, but chosen from graph structure

### Step 4: build backward if needed

If backward is enabled:

1. the actual forward output gets a seed gradient of ones
2. forward nodes are walked in reverse order
3. each node runs its backward builder lambda
4. backward targets are collected
5. a temporary `noop` super-root is created to make the joint graph observable as one closure

That means the optimizer sees one combined graph containing:

- forward nodes
- backward nodes
- the forward/backward boundary

This is deliberate.
It lets graph-level stages such as CSE and fusion reason over the full compile-time graph rather than over two disjoint partial views.

### Step 5: optimize a compile-time snapshot

Compile does not mutate the live semantic graph in place and then optimize that mutable structure directly.

Instead it captures an `OptimizerGraphSnapshot`, optimizes the snapshot graph, and then rebuilds compiled-node metadata from the optimized result.

That boundary is important because:

- compile can still reason over one joint forward/backward graph
- optimizer rewrites stay compile-local
- repeated compile does not implicitly accumulate more optimizer passes over already rewritten public graph nodes

### Step 6: capture compile-time bindings

After optimization, compile captures:

- compiled node list
- mapping from compiled graph tensors to publication tensors
- gradient bindings
- forward boundary index
- optional seed-gradient binding

## Example: Forward-Only Compile

```java
Tensor x = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{2, 2}, null, "x", DataType.FLOAT64);
Tensor y = x.relu().sum();

CompiledGraph compiled = CompiledGraph.compile(
        y,
        CompileConfig.inference(),
        CompileMode.INFERENCE_ONLY
);
```

Result:

- only forward graph is compiled
- no backward graph is built
- `compiled.supportsBackward()` is `false`

## Example: Training Compile

```java
Tensor x = new Tensor(new double[]{1.0, -2.0, 3.0}, new int[]{3}, null, "x", DataType.FLOAT64);
x.setRequiresGrad(true);

Tensor loss = x.mul(x).sum();
CompiledGraph compiled = CompiledGraph.compile(
        loss,
        CompileConfig.training(),
        CompileMode.TRAINING
);
```

Result:

- forward graph is compiled
- backward graph is built because `x` is a trainable leaf
- `compiled.supportsBackward()` is `true`

## Prepare Pipeline

`CompiledGraph.prepare(runtimeConfig)` converts compile-time structure into executable runtime metadata.

What prepare resolves today:

- backend choice per node
- CPU kernel choice
- elementwise dispatch hints
- reduction hints
- matmul hints
- attention hints
- fused executable backend preparation

What prepare does not do:

- change graph semantics
- rerun graph optimization or backend planning
- mutate tensor formulas

`prepare.orchestration.PreparedExecutionBuilder` performs prepare orchestration from compile artifacts and delegates backend-specific node preparation to backend-owned preparers.

## Execution Model

`PreparedExecution.execute(mode)` performs:

1. create per-run `ExecutionState`
2. create per-run `ExecutionContext`
3. run prepared forward steps
4. publish forward result back to the user-visible root tensor
5. if backward is requested:
   - seed compiled root gradient
   - run prepared backward steps
   - publish detached gradient tensors back to publication tensors

Two details matter:

- published gradients are detached copies, not direct aliases of internal runtime buffers
- runtime auxiliary state lives in execution-scoped state, not on the user-visible `Tensor`

## Traces

Trace DTOs live in the top-level `trace` package and do not depend back on their producers.
Graph compilation, prepare orchestration, runtime, and backends snapshot their own diagnostics.
There are three lifecycle trace layers:

- `CompileTrace`
  - total compile duration
  - graph sizes
  - whether backward support was compiled
- `PrepareTrace`
  - total prepare duration and metadata preparation visibility
- `RunTrace`
  - execution-mode duration
  - optional per-step trace list

Per-step trace metadata can include:

- layout information
- dispatch hints
- reduction hints
- matmul hints
- conv2d hints
- fused execution metadata

That is what powers detailed benchmark and hotspot reports in the tuning layer.

## Fused Preparation

Graph-level fusion publishes optimized partition units for lowering.
CPU lowering attaches backend-owned fused plan artifacts to lowered execution units, and preparation resolves how those artifacts should run.

That preparation currently involves:

- build a `FusedExecutionPlan`
- classify dispatch family
- prepare the ASM executable, with interpreted fallback only when policy allows it

The important boundary is:

- partition optimization decides optimized partition shape
- CPU lowering builds backend-owned fused plan artifacts for fused elementwise units
- `prepare(...)` decides how that fused artifact is executed on the current backend
- `execute(...)` runs that already prepared fused executable

## Data Publication Contract

The graph layer still publishes runtime results back into user-visible tensors because the public API is built around `Tensor` as the value anchor.

Current publication behavior:

- forward output is copied back into the user-visible root tensor
- gradients are copied back into publication tensors after backward

This is why compiled/prepared artifacts are reusable, but caller-side public graph mutation after compile/prepare is still caller-owned behavior.

## Important Architectural Boundaries

The current design intentionally enforces these rules:

- compile owns graph semantics and optimizer execution
- prepare owns runtime policy concretization
- executors consume prepared metadata instead of re-planning
- runtime execution state stays execution-scoped
- user-visible tensors remain the publication surface, not the hidden runtime cache layer

If a proposed change causes executors to recompute planner decisions or pushes execution caches back onto user-visible tensors, it is crossing the wrong boundary.
