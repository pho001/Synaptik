# Lifecycle

This document explains the compile, prepare, run, and training lifecycles defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

These lifecycle stages are architecture contracts, not current runnable APIs. The repository currently implements only initial model value types; the [roadmap](../planning/roadmap.md) tracks delivery.

## State across the lifecycle

Synaptik deliberately separates three kinds of state:

- The **compile-time graph** is the immutable `CompiledGraphModel` and its associated `CompileArtifacts`. It contains graph semantics, backend ownership, logical memory requirements, and publication bindings.
- **Prepared execution** is reusable runtime-ready state: prepared partitions, executable units, a physical memory plan, and an execution schedule.
- **Per-run mutable state** is `RunState`, which tracks the inputs and dynamic execution state for a particular invocation.

The compiler does not create physical buffers or executable units. Runtime does not revisit graph transformations or implementation selection.

## Compile lifecycle

```text
Tensor output
  -> GraphCapture
  -> topological sort
  -> producer/use index
  -> canonicalization
  -> shape and data type inference
  -> validation
  -> forward optimization
     - dead-code elimination
     - common subexpression elimination
     - constant folding
     - algebraic simplification
  -> autograd, when required by CompileMode
  -> post-autograd optimization
  -> publication binding
  -> backend intent propagation
  -> capability analysis
  -> partition scoring
  -> ownership decision
  -> maximal same-owner partitioning
  -> logical memory/materialization requirements
  -> CompileArtifacts
```

`CompileArtifacts` packages the immutable graph, planned partitions, logical memory plan, publication plan, and compile diagnostics. Compile does not create prepared schedules, executable units, physical buffers, runtime workspaces, concrete kernel routes, or backend-specific executable graphs.

Partition scoring chooses backend ownership, not concrete implementation. See [Partition Scoring](partition-scoring.md).

## Prepare lifecycle

```text
CompileArtifacts
  -> validate partition coverage
  -> call the owning BackendPartitionPreparer for each PlannedPartition
  -> backend lowering, specialization, fusion, and kernel selection
  -> build PreparedPartition[]
  -> build PreparedMemoryPlan
  -> build PreparedSchedule
  -> validate prepared memory and schedule
  -> PreparedExecution
```

Prepare creates `PreparedPartition`, `PreparedUnit`, `PreparedExecutable`, `PreparedMemoryPlan`, `PreparedSchedule`, and `PreparedExecution`. Shared prepare code owns contracts and validation; concrete backend modules own lowering and executable implementations.

See [Runtime, Prepare, and Backend Boundary](runtime-prepare-backend-boundary.md) for the exact ownership split.

## Run lifecycle

```text
PreparedExecution.run(...)
  -> create or reuse RunState
  -> bind inputs
  -> execute PreparedSchedule
  -> perform prepared residency/materialization work as needed
  -> PreparedExecutable.execute(...)
  -> update residency
  -> publish requested results
  -> RunResult
```

Run executes prepared work. It must not perform graph optimization, autograd construction, compiler passes, backend discovery, backend-specific lowering, or kernel selection.

## State scenario

Suppose one compiled graph is prepared once for CPU and then run twice with different input values. The immutable `CompileArtifacts` and reusable `PreparedExecution` are shared. Each invocation has distinct input bindings and `RunState`. Storing the second run's current buffer residency in the compiled graph would mix per-run mutable state into the immutable recipe; selecting a different CPU route during the second run would repeat a prepare-time decision in the hot path.

## Training lifecycle

Before compilation, an `extensions/nn` module tree supplies forward behavior. Each module declares
its trainable `Parameter` values and persistent `Buffer` values; `train()` or `eval()` propagates
the selected mode to children. This is not optimizer work: for example, a batch-normalization
layer can select its training or inference forward behavior before gradients exist. The generic
Tensor operations used by that forward pass remain owned by `modules/model`.

The compile mode determines how much graph the compiler constructs:

- `FORWARD_ONLY` compiles forward computation only.
- `FORWARD_AND_BACKWARD` expands the forward graph with autograd and compiles a combined forward and backward graph.
- `TRAINING_STEP` represents the training-step direction in which optimizer updates may also become graph operations.

In backward-capable modes, forward and backward may be combined at compile time so post-autograd optimization and planning can see the entire graph. Prepare may still expose separate forward and backward schedules or one training-step schedule.

See [Training Graph](training-graph.md) for the graph model and optimization rationale.

## Optimizer and training-step lifecycle

The initial lifecycle keeps the optimizer as a backend-agnostic step after the prepared forward/backward execution:

```text
compile:
  forward graph
  -> autograd
  -> combined forward + backward graph
  -> optimize
  -> CompileArtifacts

run:
  forward/backward prepared execution
  -> publish gradients
  -> optimizer.step()
```

A later architecture version may represent the optimizer update in the graph, but only after an explicit architecture update where required:

```text
compile:
  forward + backward + optimizer update graph
  -> optimize
  -> partition scoring
  -> CompileArtifacts

prepare:
  backend prepare may fuse optimizer update routes

run:
  training-step schedule
```

In both forms, `extensions/training` owns optimizer algorithms and training orchestration over the
parameters declared by `extensions/nn`. It remains independent of concrete backends. Backend-
specific optimizer execution belongs to backend prepare and kernels.
