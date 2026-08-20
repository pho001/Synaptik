# Lifecycle

This document explains the compile, prepare, run, and training lifecycles defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

These lifecycle stages are architecture contracts, not a claim that the complete public lifecycle is runnable today. The repository implements the Model, Planning, and Compiler portions of compile, staged Prepare contracts through backend finalization, and Runtime prepared-execution and per-run orchestration contracts. It does not yet provide concrete backend execution or the Engine composition needed for an end-to-end runnable lifecycle. The [roadmap](../planning/roadmap.md) records delivery status.

## State across the lifecycle

Synaptik deliberately separates three kinds of state:

- The **compile-time graph** is the immutable `CompiledGraphModel` and its associated `CompileArtifacts`. It contains graph semantics, backend ownership, logical memory requirements, and publication bindings.
- **Prepared execution** is immutable reusable runtime-ready state: prepared partitions,
  executable recipes, memory geometry, a schedule, and any immutable persistent prepared
  resources.
- **Per-run mutable state** is exactly one `RunState` for each active complete logical invocation.
  It covers every backend partition in that heterogeneous run and tracks its logical slots,
  resources, validity, and residency without sharing mutable state with another run.

The compiler does not create physical buffers or executable units. Runtime does not revisit graph transformations or implementation selection.

## Compile lifecycle

```text
forward Tensor outputs
  -> if backward is requested:
     - fail-closed operation/attribute/policy preflight
     - reverse accumulation through ordinary public Tensor operations
     - combined forward + gradient Tensor expression DAG
  -> one phase-aware GraphCapture
  -> topological sort
  -> producer/use index
  -> shape and data type inference
  -> validation
  -> canonicalization
  -> combined-graph optimization
     - dead-code elimination
     - phase-local common subexpression elimination
     - constant folding
     - algebraic simplification
  -> final validation
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

Autograd preflight checks every backward-reachable producer occurrence and its exact attributes
before constructing a derivative expression. The compiler then uses exact Tensor identity only
for temporary contribution accumulation and combines contributions with ordinary `Tensor.add`.
Seeds and derivative constants are storage-free leaves registered explicitly as logical splats.
The one capture call receives the original forward-producer identity set so it can assign
`FORWARD` or `BACKWARD` per node while assigning graph-local IDs only once. Full inference happens
after capture; a failure at that or a later stage may therefore consume temporary Tensor IDs,
which remain opaque and non-reusable.

Partition scoring chooses backend ownership, not concrete implementation. See [Partition Scoring](partition-scoring.md).

## Prepare lifecycle

```text
CompileArtifacts
  -> validate partition coverage
  -> project partition-scoped facts and resolved prepare inputs
  -> backend analysis, lowering, specialization, fusion, and kernel selection
  -> exact shared buffer/workspace declarations
  -> build BackendPartitionAnalysis[]
  -> assign stable slots and build PreparedMemoryPlan
  -> backend finalization against assigned slots
  -> build PreparedPartition[] and PreparedExecutable[]
  -> build PreparedSchedule
  -> validate prepared memory and schedule
  -> PreparedExecution
```

Prepare creates `BackendPartitionAnalysis`, `PreparedPartition`, `PreparedUnit`,
`PreparedExecutable`, `PreparedMemoryPlan`, `PreparedSchedule`, and `PreparedExecution`. Shared
Prepare owns projection, orchestration, exact resource declarations, slot assignment, and
validation. Concrete backends own deterministic analysis/lowering/route choice, retain the
selected plan opaquely, and construct executables only during finalization after slot assignment.

See [Runtime, Prepare, and Backend Boundary](runtime-prepare-backend-boundary.md) for the exact ownership split.

## Run lifecycle

```text
PreparedExecution.run(...)
  -> create one RunState for this complete logical run
  -> bind caller inputs as borrowed representations
  -> create run-owned internal and workspace representations through prepared backend work
  -> cold-bind backend-owned typed invocation objects to checked direct references
  -> execute PreparedSchedule
  -> perform explicit prepared residency/materialization/transfer work as needed
  -> PreparedExecutable.execute(...)
  -> update residency
  -> publish requested results and transfer or lease their ownership
  -> RunResult
  -> release resources still owned by RunState
```

Run executes prepared work. It must not perform graph optimization, autograd construction,
compiler passes, backend discovery, backend-specific lowering, or kernel selection. Runtime owns
logical per-run state and cleanup orchestration; concrete backends own physical representation
classes and their allocation, release, transfer, and access mechanics.

The cold binding step is the only boundary where heterogeneous backend representation types are
checked dynamically. It creates backend-owned typed objects with direct references before the hot
path. Execution therefore needs no map lookup, reflection, string dispatch, graph inspection,
service lookup, or repeated unsafe cast.

## Planned fixed recurrent scan through the lifecycle

The fixed recurrent scan is a current ordinary Model expression whose `INT64[batch]`
valid-length values are modeled as ordinary inputs while every Shape remains fully static. Model
construction is implemented; Compiler adoption, public execution, and backend realization are
not. The complete planned lifecycle reads left to right as follows:

```text
one flat multi-output TensorProducer
  -> one flat CompiledNode
  -> one ordinary capability and ownership decision
  -> one backend analysis and exact resource declaration
  -> one finalized reusable PreparedExecutable
  -> one cold-bound invocation per RunState
  -> one backend-internal bounded recurrent loop
```

Current Model fixes `RNN_TANH`, `GRU_RESET_AFTER`, and `LSTM`, `FORWARD` or `REVERSE`, ordered inputs and
outputs, static descriptor rules, and dense original-time-aligned zero-filled padded results.
Generic capture already preserves the occurrence as one node, but current Compiler inference
rejects its kind as unsupported and current autograd rejects it before derivative Tensor
construction. Later Compiler, Planning, Prepare, Engine, Runtime, and backend work owns the
remaining arrows. Planning will treat it as one ordinary capability query. Shared Prepare will
project its static facts and assign backend-declared resources without receiving a loop body.
Engine will cold-bind the typed logical inputs, including the runtime length Tensor, and Runtime
will invoke the prepared bound action without interpreting recurrence.

The concrete backend validates the complete length vector before mutating any output
representation. It traverses only each row's valid prefix, writes exact positive zero at padded
coordinates, returns explicit final hidden and LSTM cell states, and skips recurrent arithmetic
for invalid coordinates. `REVERSE` traverses `L[b]-1 .. 0`, not the padded suffix. A zero-length
row returns its initial states semantically; a zero-time input requires all lengths to be zero.

One occurrence and one prepared transition remain constant in graph size as `time` grows. A
backend-internal row/time loop does not create nested graph identity. Physical active-row
compaction and backpropagation through time (BPTT) remain later owner-specific decisions. See
[ADR 0012](../design/decisions/0012-fixed-recurrent-scan-without-regions.md).

## State scenario

Suppose one compiled graph is prepared once for CPU and then run concurrently twice with different
input values. The immutable `CompileArtifacts` and `PreparedExecution` are shared. Each invocation
has one distinct `RunState`, borrowed input bindings, and isolated run-owned buffers/workspaces.
Storing either run's residency in the prepared recipe would mix mutable invocation state into
shared state; selecting another CPU route during either run would repeat a prepare-time decision
in the hot path.

## Training lifecycle

Before compilation, an `extensions/nn` module tree supplies forward behavior. Each module declares
its trainable `Parameter` values and persistent `Buffer` values; `train()` or `eval()` propagates
the selected mode to children. This is not optimizer work: for example, a batch-normalization
layer can select its training or inference forward behavior before gradients exist. The generic
Tensor operations used by that forward pass remain owned by `modules/model`.

The compile mode determines how much expression and graph work the compiler constructs:

- `FORWARD_ONLY` compiles forward computation only.
- `FORWARD_AND_BACKWARD` constructs backward Tensor expressions from the original forward
  expression and captures a combined forward and backward graph once.
- `TRAINING_STEP` represents the training-step direction in which optimizer updates may also become graph operations.

In the initial backward-capable lifecycle, `FORWARD_AND_BACKWARD` and `TRAINING_STEP` both build
the combined expression before capture; `TRAINING_STEP` does not yet add optimizer-update graph
work. Combined optimization and planning therefore see the entire immutable graph. Prepare may
still expose separate forward and backward schedules or one training-step schedule.

See [Training Graph](training-graph.md) for the graph model and optimization rationale.

## Optimizer and training-step lifecycle

The initial lifecycle keeps the optimizer as a backend-agnostic step after the prepared forward/backward execution:

```text
compile:
  forward Tensor expression DAG
  -> compiler-owned autograd through ordinary Tensor expressions
  -> one phase-aware capture of forward outputs and gradient roots
  -> infer, validate, optimize, and revalidate the immutable combined graph
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
