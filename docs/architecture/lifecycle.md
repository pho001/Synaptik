# Lifecycle

This document explains the compile, prepare, run, and training lifecycles defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract remains authoritative.

The ordinary public lifecycle is runnable through `Engine.standard()` for fixed CPU ownership and
through `Engine.builder()` for explicit CPU/Metal ownership. Builder preparation composes
single-owner or mixed-owner partitions through deterministic owner-indexed representations and
explicit bidirectional transfer for all seven model data types over exact rank-0..16 static
canonical or positive-stride non-overlapping storage layouts. The public Training module now
layers one optimizer/session owner over that same Engine compile, prepare, run, publication, and
materialization path. CUDA execution remains planned.
The [roadmap](../planning/roadmap.md) records delivery status.

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

NN builder construction precedes compilation: copy supplied raw parameter leaves once, initialize
fresh automatic layers, then build one ordinary prototype expression through the frozen Sequential.
Cold Model.forward applies that same chain to an exact-descriptor expression; it creates no Engine
and does not change cached outputs. Compute/train compile those fixed outputs once per setup and
reuse positional bindings across batches. For example, Conv2d/pool/flatten can execute although
GraphCheckpoint cannot save them. A rejected save touches no file and does not freeze or invalidate
the Model. Definition/graph/load still use the bounded source path below. The
[owned Model contract](contracts/extensions-training.md#owned-tensor-models) owns this distinction.

Before ordinary compilation, integration callers may use the bounded cold source path defined by
the [Compiler lifecycle contract](contracts/compiler-autograd.md#compile-lifecycle):

```text
ordinary forward Tensor expressions
  -> one raw capture + exact input-aligned borrowed leaves
  -> complete static source verification
  -> optional cold reconstruction with positional descriptor-compatible substitutions
  -> ordinary Tensor expressions -> ordinary compile below
```

For example, capture `x.matmul(w.permute(1,0)).add(b)`, then reconstruct it with a fresh input and
the same weight and bias. Reconstruction builds each stored occurrence once through its public
Tensor constructor, so shared operands remain shared and later autograd sees ordinary provenance.
A replacement can itself be a produced expression; its upstream computation remains attached and
is checked by later ordinary compile/autograd, not widened source admission. Position associates
inputs, so exchanging same-descriptor weights cannot be detected.

This source path admits only static FORWARD ADD/MUL/NEG/MATMUL, floating RELU/TANH/SIGMOID and
rank-two PERMUTE `[1,0]`. Activations reconstruct through their ordinary Tensor constructors;
Checkpoint selects v2 only when one occurs, leaving five-kind source bytes at v1.
It neither optimizes nor closes raw layouts. An optimized, layout-closed artifact graph is not a
reconstructible source merely because it has the same operations. The borrowed source sidecar
owns no storage or close lifecycle; it is not retained in graph/artifact state. Only after complete
preflight does reconstruction use a call-local identity table, discarded on return or failure.
No storage read/allocation, backend discovery, preparation, execution or runtime graph is added.
See the [cold source API example](../api/compile-api.md#current-cold-source-capture-and-reconstruction).

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

Before publication and planning, final static-result closure resolves proven materialized logical
results, then propagates exact input-relative views in topological order. Scalar SELECT uses the
existing indexing inference; positive-step coordinate SLICE, target-relative crop, and shape/axis
views use layout inference. For example, a `[2,6]` matrix product followed by a two-column gate
slice retains row stride `6`, including a nonzero column-start offset, rather than acquiring a
contiguous copy. Negative-step, empty, dynamic, and otherwise unproved extractions stay unresolved.
These descriptor facts leave physical storage, lowering, and execution with Prepare and backends;
they do not widen the separate cold source boundary above.

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

The ordinary reusable prepared path is:

```text
Tensor outputs -> engine.compile(...) -> engine.prepare(...) -> prepared owner
exact-ID inputs in any order -> engine.run(prepared, inputs) -> leased RunResult
fresh values in compiled order -> engine.runValues(prepared, values) -> leased RunResult
exact publication occurrence -> result.materialize(...) -> detached HostTensorValue
```

One Engine prepared handle borrows the composition lifetime and owns its exact inward preparation;
Runtime alone owns persistent-resource and run-lease authority. The final `CompiledGraph.inputs()`
list authenticates identity membership for run and positional order for runValues. Fresh values
require fully static exact descriptors, including layout presence and requiresGrad. Neither path
mutates prototypes, recompiles, prepares, queries providers/registries or adds per-element work.
Before any borrow, both snapshot associations once and validate storage. runValues additionally
uses checked canonical static element/byte geometry for absent layouts, or checked referenced spans
for resolved layouts. Unknown/overflow geometry rejects only positional eligibility.

Both calls check Engine admission, prepared nullness/exact owner and exact inward delegate closure
before inputs. This intentionally changes former direct-run precedence: closed preparation wins
null/malformed/dead inputs with `prepared execution is closed` and zero borrows. Runtime still
arbitrates any later close/run race without an outward lock held through execution.

Each run has isolated mutable state and a result retaining caller-input borrows until result close.
A prepared close does not wait for already leased runs or close returned results. Materialization
remains usable while result and Engine are open. Metadata survives close; detached values survive
result/prepared/Engine/input-storage closure. Engine shutdown waits for admitted operations, then
closes results, ordinary/advanced preparations and integrations in their reverse publication or
registration order, with rollback, suppression and failure replay.

`InferenceSession`/`Engine.session` are removed; migrate to explicit prepare/run and recompile.
No new owner/wrapper or lease protocol replaces them.
`Engine.compute(...)` is a different lifetime choice: it discovers reachable expression leaves,
then freshly compiles, prepares, runs, materializes, closes the temporary result, and closes the
temporary preparation during every call. `Engine.backward(...)` does the same for one scalar
objective and explicit gradient targets. Neither convenience installs gradient state on Tensor.

## Public training lifecycle

`TrainingSession.open(engine, parameters, objective, optimizer)` snapshots the live source's
ordered paths and exact Tensor/storage identities, compiles the scalar
objective with those parameters as ordered first-gradient targets once, and calls
`engine.prepare(compiledGraph)` once. It borrows Engine and the live source, owns the resulting
Engine `PreparedExecution`, a private native cotangent seed, primitive optimizer/accumulation staging
buffers, counters, and persistent optimizer slots.

```text
open: snapshot parameters -> Engine.compile once -> Engine.prepare once
run:  bind captured parameters + seed + caller inputs -> Engine.run or Engine.runValues
      -> validate/materialize objective and target-indexed gradients
      -> stage and validate every update -> atomically publish parameters/state/counters
```

Every later `run` uses the same compiled graph and prepared execution. Training neither
calls Runtime/Prepare nor reconstructs publication mapping: `CompiledGraph.inputs()` supplies
input identity/order and each gradient publication's `targetIndex` maps to the snapshotted
parameter position. Only one Training operation is admitted at a time. Close rejects new work,
waits for admitted Training work, closes the prepared execution and private seed arena once, and
never closes the borrowed Engine, source, or parameter storage. Detached `TrainingStep` and
`TrainingState` values outlive all of those owners.

`run` accepts exact external TensorIds in arbitrary order. `runValues` instead takes stored values
in `session.inputs()` order, including targets but excluding captured parameters and the private
seed. Exact full static descriptors must match; expressions are not executed or rebound. Fresh
storage is borrowed through the synchronous call and internal result closure only. Both paths
use the same optimizer transaction and reject nonempty overlapping inputs, even repeated values
that generic Engine positional binding would allow. For example, a fresh batch and target can
replace yesterday's input values without replacing the graph, parameters or optimizer history.

Checkpoint does not add a step to this execution lifecycle. Its model API can save one complete
quiescent NN state through public Engine materialization, inspect the bounded file, then load it
into a fully bound recreated target through NN's strict whole-tree installation. The caller owns
quiescence and versioned configuration/input fingerprints; v1 does not infer source alias identity.
The opaque artifact API provides atomic live visibility, not power-loss durability. Exact training
continuation would additionally
require Training-owned state and the caller's next-input and external-state handoff, coordinated
by a separately gated optional adapter rather than by NN or the model-only project. This is a
planned ownership flow, not a current runnable training resume sequence. See the [checkpoint
boundary decision](../design/decisions/0031-model-checkpoint-extension-boundary.md) and
[artifact envelope decision](../design/decisions/0032-checkpoint-artifact-envelope.md) and
[model schema decision](../design/decisions/0033-model-candidate-schema.md). The
[public model checkpoint decision](../design/decisions/0034-public-model-checkpoint.md)
records the bound-target and alias policy.

The [accepted CP0005B whole-model lifecycle](contracts/extensions-training.md#whole-model-definition-and-composition)
is implemented and reviewed (Complete, 2026-10-15), separately from values-only recovery.
Under caller quiescence, NN description and public
values capture may each read state; no combined snapshot transaction is promised. Dry sizing and
residual checks precede bounded definition encoding/hash; public capture's projected preflight
enforces outer bounds before Engine work or bulk value copies. Load preflights outer framing before
any copy/projection, independently validates configuration/descriptors/alias bytes, reconstructs an
unpublished descriptor-only NN candidate and invokes public values target-load. Only the complete
heap-backed eval model is returned, with no Engine/native owner or autograd. For example, closing
the source Arena and Engine after successful save cannot invalidate a later loaded model's values;
its forward expression still needs an admitted caller Engine route. Empty identity models retain
semantics without promising zero-operation Engine execution. Default load rejects a stored input
digest; explicit verified identity must match presence/bytes. File durability policy is unchanged.

The first update domain is scalar `FLOAT32`/`FLOAT64` objectives and connected, fully static,
dense-contiguous, offset-zero non-view parameters in exact-capacity writable shareable native
host storage. Unsupported Engine preparation/execution—including a Metal or mixed plan outside
current backend coverage—fails before optimizer mutation. Training branches on no backend
identity.

`AdvancedEngine.takeOwnership(...)` is the lower-level CPU-only explicit-composition surface. Its
caller supplies one supported CPU integration and transfers ownership to Engine. Ordinary builder
composition can assemble registered CPU/Metal mixed-owner schedules; neither surface discovers
backends.

## Explicit backend composition

The public composition surface is an `AutoCloseable Engine.Builder`. Its concrete
`takeOwnership(...)` overloads accept opened CPU and Metal integrations. Ownership transfers at
each non-null method entry, so the builder—not the caller—owns rollback even when registration
fails. Successful `build()` moves the complete ordered registry into one Engine; unsuccessful
construction and builder close release accepted integrations in reverse registration order.

```text
backend-owned configuration -> backend opens integration and native state
                            -> Engine.Builder takes ownership
                            -> capture BackendId + capability provider + availability snapshot
                            -> Engine.compile selects one owner identity per operation
                            -> Engine.prepare validates all owners/transfers, then runs one shared
                               Prepare transaction and captures per-occurrence outer adapters
                            -> Engine run/materialize use those captured direct references
                            -> Runtime executes the immutable prepared schedule
```

The capability provider reference and point-in-time availability snapshot are captured once for
each registration and reused for every compile by that Engine. Equal backend IDs are rejected.
Compile-time Planning receives the immutable registration-ordered inputs and records only
`BackendId` ownership; it receives no Engine registry or live integration.

Ordinary cold preparation requires a non-empty plan and resolves every partition owner against the
frozen registry. Before any backend analysis, it rejects a missing owner or unsupported directed
transfer pair. Shared Prepare then analyzes and finalizes every partition, assigns one logical slot
plus one deterministic representation position per participating owner, collects backend physical
creation contributions, and assembles explicit execution, transfer, and publication occurrences.
Current CPU/Metal transfers cover all seven carriers over exact rank-0..16 static canonical or
positive-stride non-overlapping layouts with checked physical spans; there is no retry, owner
substitution, conversion, heap staging, or CPU fallback.

One outward prepared Engine handle keeps direct non-owning adapter references in caller-input and
publication occurrence order while Engine owns the integrations. Run uses each input adapter to
wrap caller host storage, and each result publication uses its captured adapter for backend-owned
host materialization. Neither operation re-queries the registry. Engine shutdown closes retained
results first, then ordinary and advanced prepared handles in reverse
preparation-publication order, then integrations, so no captured reference can outlive its owner.

`prepareTuned(...)` accepts one eligible single-owner CPU plan or exact singleton-NEG Metal plan.
Metal Phase 1 measures its custom-kernel and MPSGraph routes; Phase 2 checks one complete candidate
fixed to the selected route. Mixed ownership, multiple partitions, and MPSGraph-only Metal plans
remain ineligible. A safe-heuristic fallback always prepares through the already-selected adapter;
it never substitutes CPU for Metal or otherwise changes ownership.

Engine shutdown rejects new work, waits for admitted Engine operations, closes retained results in
reverse successful-run publication order, then ordinary and advanced prepared handles in
reverse preparation-publication order, and finally backend integrations in reverse
registration order. Closing a prepared handle rejects later runs without
closing an already returned result. A run that already acquired Runtime's prepared lease may
finish; close does not wait for it, and deferred prepared-resource cleanup occurs when the final
lease releases. The registry map has no post-prepare execution role, and no capability query,
availability check, reflection, or `ServiceLoader` lookup enters cold-bound Runtime execution.
The Engine-handle adapter calls described above occur outside Runtime.

`Engine.standard()` remains the current CPU-only convenience and constructs its fresh CPU
integration through the same ownership path. Metal owns its configuration and native open
operation; Engine accepts the opened integration and does not parse a native library path or
select a Metal device. See
[ADR 0015](../design/decisions/0015-explicit-engine-backend-composition.md) for the alternatives and
consequences.

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

In both forms, `modules/training` owns optimizer algorithms and training orchestration over a
generic live parameter source. NN supplies `Module.trainingParameters()` without moving wrapper
ownership into Training. Live membership validation precedes run/state/restore and every existing
complete-view precommit check; Training separately checks exact storage and finite values. A
pre-session native copy preserves the same source by changing only storage associations.
Training remains independent of concrete backends. Backend-
specific optimizer execution belongs to backend prepare and kernels.

## Numerical semantics across the lifecycle

```text
Model semantics -> Planning occurrence query -> CompileArtifacts -> PrepareContext
                -> qualified backend route -> PreparedExecution -> RunState
```

There is no Engine numerical selection or cold profile transport. Model fixes each operation
family's exact and named-arithmetic-site meaning. Planning queries provider capability for the
complete occurrence; Compiler preserves guarded rewrites and immutable occurrence facts. Prepare
uses qualified backend routes and non-profile compatibility identities, then Runtime executes the
prepared result without tolerance, certificate, route, or capability lookup. For example, a
BFLOAT16 raw move preserves its stored word, while a following named arithmetic primitive may
use its permitted input DAZ. Neither step changes the other's boundary. Metal retains MPSGraph
for eligible FLOAT32-only steps and the fixed custom program for low-containing partitions. Its
current program is schema 20 with a 124-byte header and tuning identity 36; CPU generator schema
is 70, and native ABI 7 retains thirteen exports. Schema 19/identity 29 and CPU schema 68 are
historical, incompatible inputs.
