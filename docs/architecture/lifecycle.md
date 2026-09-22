# Lifecycle

This document explains the compile, prepare, run, and training lifecycles defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

The ordinary public lifecycle is runnable today through `Engine.standard()`, whose fixed current
composition owns one fresh CPU integration. It implements the compile, staged prepare, run,
publication, and explicit host-materialization path described below. The architecture is broader
than that implementation: Metal, CUDA, mixed-owner schedules, and training orchestration remain
planned. The [roadmap](../planning/roadmap.md) records delivery status.

## State across the lifecycle

Synaptik deliberately separates three kinds of state:

- The **compile-time graph** is the immutable `CompiledGraphModel` and its associated `CompileArtifacts`. It contains graph semantics, backend ownership, logical memory requirements, and publication bindings.
- **Prepared execution** is immutable reusable runtime-ready state: prepared partitions,
  executable recipes, memory geometry, a schedule, and any immutable persistent prepared
  resources. Its logical recipe is immutable, while its explicit close state controls the finite
  lifetime of those resources.
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

Prepare creates `BackendPartitionAnalysis`, `PreparedPartition`, `PreparedExecutable`,
`PreparedMemoryPlan`, `PreparedSchedule`, and `PreparedExecution`. Shared
Prepare owns projection, orchestration, exact resource declarations, slot assignment, and
validation. Concrete backends own deterministic analysis/lowering/route choice, retain the
selected plan opaquely, and construct executables only during finalization after slot assignment.
Finalization may acquire an immutable persistent prepared resource, such as a compiled native
executable, only through the transactional ownership handoff: the backend rolls back before
return, Prepare rolls back after return, and the completed `PreparedExecution` becomes the sole
owner. Runtime's owner and lease lifecycle are current. The finalizer return and Prepare
transaction described here are current through Prepare 0006; current CPU preparation contributes
an empty persistent-resource list.

The resulting Runtime `PreparedExecution` retains the exact assigned memory plan and exact
same-plan schedule. `PreparedSchedule` is an ordered list of step occurrences: optional
representation creation is first, executable and explicit buffer-transfer occurrences follow,
and publications form the dense final suffix. Repeating an executable or transfer occurrence
means repeated work, not another ownership occurrence. No additional aggregate sits between a
schedule occurrence and its recipe.

See [Runtime, Prepare, and Backend Boundary](runtime-prepare-backend-boundary.md) for the exact ownership split.

## Run lifecycle

```text
PreparedExecutionRunner.run(PreparedExecution, callerInputs)
  -> acquire one prepared-execution run lease
  -> create one RunState for this complete logical run
     - bind caller inputs as borrowed representations
     - invoke the optional first representation-creation recipe
  -> cold-bind every executable, transfer, and publication occurrence
  -> traverse the PreparedSchedule.Step occurrences in order
     - execute prepared executable occurrences
     - perform explicit buffer-transfer/materialization occurrences
     - publish the dense final suffix
  -> RunResult
     - owns the complete RunState lease until close
  -> release the prepared-execution run lease before the synchronous call returns
```

The stateless `PreparedExecutionRunner` owns synchronous run orchestration and creates one
isolated `RunState` per call. Run executes prepared work. It must not perform graph optimization,
autograd construction,
compiler passes, backend discovery, backend-specific lowering, or kernel selection. Runtime owns
logical per-run state and cleanup orchestration; concrete backends own physical representation
classes and their allocation, release, transfer, and access mechanics.

The cold binding step is the only boundary where heterogeneous backend representation types are
checked dynamically. Executable and transfer binding create backend-owned typed objects with
direct references; Runtime publication binding retains the selected representation directly.
Execution therefore needs no map lookup, reflection, string dispatch, graph inspection, service
lookup, or repeated unsafe cast.

Each synchronous run first acquires a lease on its prepared execution. Once close begins, no new
lease is admitted. Existing runs finish without close waiting for them; the last lease performs
deferred reverse cleanup when close raced an active run. Thus close is deterministic without an
unbounded wait, and the prepared recipe never moves native compilation onto the run path. This
Runtime lifecycle is current. Engine prepared handles now own exactly one inward Runtime
execution. Their outward synchronization covers delegate retrieval and the inward close
transition, not a complete run; a delegate-first run still arbitrates at Runtime's unique lease
authority. Once outward closure is observable, no later retrieval can admit work. Engine adds
neither another lease nor a waiting close.

## Current public Engine lifecycle

The ordinary reusable path is:

```text
Tensor outputs -> engine.compile(...) -> engine.prepare(...)
caller-owned input Tensors -> engine.run(...) -> leased RunResult
exact publication occurrence -> result.materialize(...) -> detached HostTensorValue
```

`CompiledGraph.inputs()` supplies the authoritative caller-input membership and order. A caller
may pass those Tensors to `run(...)` in any order because Engine matches exact Tensor identities.
The prepared recipe is immutable and reusable; its Engine handle is explicitly closeable, and
every run receives isolated mutable `RunState`.
`RunResult` retains publication leases until it closes, while each `HostTensorValue` is a copied,
immutable value that remains readable after the result, Engine, and caller storage close.

`Engine.compute(...)` is a different lifetime choice: it discovers reachable expression leaves,
then freshly compiles, prepares, runs, materializes, closes the temporary result, and closes the
temporary preparation during every call. `Engine.backward(...)` does the same for one scalar
objective and explicit gradient targets. It
does not install gradient state on Tensor. Use the reusable path when compilation or preparation
should be amortized across runs.

Engine shutdown first waits for admitted operations, then closes retained results in reverse run
publication order, retained preparations in reverse prepare-publication order, and backend
composition. A prepared-handle close does not independently close an already returned result.

`AdvancedEngine.takeOwnership(...)` is the lower-level explicit-composition surface. Its caller
supplies one supported CPU integration and transfers ownership to Engine. Neither ordinary nor
advanced composition currently assembles mixed-owner schedules or discovers backends.

## Planned fixed recurrent scan through the lifecycle

The fixed recurrent scan is a current ordinary Model expression whose `INT64[batch]`
valid-length values are modeled as ordinary inputs while every Shape remains fully static. Model
construction and forward-only Compiler adoption are implemented; public execution and backend
realization are not. The complete lifecycle reads left to right as follows:

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
Generic capture preserves the occurrence as one node, and current Compiler forward-only inference
independently revalidates its fully static descriptors. Both backward-capable modes reject a
complete forward inventory containing recurrence before derivative Tensor allocation. Later
Planning, Prepare, Engine, Runtime, and backend work owns the remaining execution arrows. Planning
will treat the node as one ordinary capability query. Shared Prepare will
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
