# Runtime, Prepare, and Backend Boundary

This document explains the boundary defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The root contract remains authoritative.

Runtime currently implements the immutable `BufferSlot` and `WorkspaceSlot` identities,
`PreparedMemoryPlan` final slot geometry, nominal buffer/workspace representation roles,
borrowed/run-owned buffer bindings, immutable prepared representation origins and backend-owned
creator callbacks, package-private cold creation with rollback, structural per-run residency and
explicit buffer-copy validity, the array-backed one-run `RunState` lifecycle, immutable
`PreparedExecutable` and `PreparedBufferTransfer` recipes, per-run `BoundInvocation` and
`BoundBufferTransfer` objects, and the immutable creation-plus-execution-plus-transfer
`PreparedSchedule` contract described below, the dense final publication suffix, direct per-run
publication binding, the whole-`RunState` `RunResult` lease, the immutable two-component
`PreparedExecution` root, explicit executable buffer-access declarations, and the stateless
prepared-execution runner. Prepare
currently implements the analysis-side projection, opaque marker roles, exact resource
declarations, analysis result, preparer collaboration, deterministic complete-set slot assignment,
typed backend finalization input/collaboration, and the minimal prepared-partition association.
Physical allocation and access implementations, public output-value access, public Prepare
orchestration, engine, and production concrete backends remain planned.
The lifecycle flow therefore mixes current foundations with later stages; each focused section
states its implementation status.
[ADR 0011](../design/decisions/0011-per-run-runtime-resource-ownership.md) defines the
resource-ownership and cold-binding architecture.

## Boundary in one flow

```text
CompileArtifacts
  -> shared prepare validation and orchestration
  -> Prepare-owned partition analysis request
  -> concrete backend analysis, lowering, route choice, and exact resource declaration
  -> BackendPartitionAnalysis[]
  -> shared buffer/workspace slot assignment and PreparedMemoryPlan
  -> concrete backend finalization against assigned slots
  -> PreparedPartition[] + PreparedExecutable[]
  -> PreparedSchedule
  -> PreparedExecution
  -> runtime execution with RunState
```

Read the arrows from compile-time facts toward reusable runtime state. Prepare turns an immutable
compile recipe into runtime-ready state, but executable construction is deliberately split around
shared slot assignment. Runtime executes the result without rediscovering, lowering, or selecting
backend work.

## Current Runtime memory geometry

The current `modules/runtime` production surface contains two nominally distinct non-negative
plan-local identities:

- `io.github.pho001.synaptik.runtime.memory.BufferSlot(long value)` for a reusable buffer
  position; and
- `io.github.pho001.synaptik.runtime.memory.WorkspaceSlot(long value)` for a reusable workspace
  position.

Neither record stores a plan reference. Equal numeric values are valid across the two domains,
and another plan may reuse either number without establishing cross-plan identity.

The current `PreparedMemoryPlan` carries final ordered geometry through immutable buffer-entry and
workspace-entry snapshots. Each entry retains one exact slot reference plus exact non-negative
byte size and positive power-of-two byte alignment. Slots are unique within their respective
lists, the two domains remain separate, and empty plans are valid.

These Runtime contracts deliberately retain no compile-time `ValueId`, analysis-local workspace
requirement ID, Prepare requirement, or source-to-slot association. They do not derive assignment,
allocate or own a physical buffer or workspace, define aliasing or lifetime, bind a run, provide
access, transfer data, or execute work. This distinction lets later prepared units refer to final
positions without moving graph objects or physical storage into the Runtime hot-path contract.

## Current analysis foundation

The current `modules/prepare` production surface is the
`io.github.pho001.synaptik.prepare.analysis` package. Its exact six top-level declarations are:

- `BackendAnalysisInputs`, the marker role for one concrete backend's immutable target,
  capability, configuration, and compatible cached-decision inputs;
- `BackendPreparationPlan`, the marker role for that backend's immutable selected lowering,
  route, and private configuration;
- `PrepareContext`, the validated partition projection;
- `PreparationResourceRequirement`, the sealed buffer/workspace declaration family;
- `BackendPartitionAnalysis`, the immutable selected-plan and requirement result; and
- `BackendPartitionPreparer`, the typed backend analysis collaboration.

`PrepareContext` retains the exact planned-partition reference and immutable ordered snapshots of
its nodes, projected values, and logical-memory requirements. Its node IDs must match the
partition exactly and in order. Projected values are unique by `ValueId`, every node input and
output resolves to one projected value, and every projected value has one descriptor-matching
logical-memory requirement. This is intentionally asymmetric: the current contract does not
separately require every projected value to occur in a node input or output.

All projected Shapes must be fully static. An immutable logical-splat constant may be projected
only for a projected graph input, and its exact `ScalarValue` type must match that input's
descriptor. Neither the context nor either marker role exposes `CompileArtifacts` or another
Compiler-owned type.

The sealed resource family has two immutable records. `Buffer` associates one projected
`ValueId` with an exact non-negative byte size and positive power-of-two byte alignment.
`Workspace` uses a non-negative analysis-local requirement ID with the same size and alignment
rules. Buffer IDs and workspace IDs are unique within their respective domains in one
`BackendPartitionAnalysis`; neither identity is a Runtime slot, address, allocation, handle, or
per-run binding.

`BackendPartitionPreparer.analyze` must be deterministic from the complete context and return the
exact context partition reference. Analysis performs no measurement, tuning search, cache
mutation, allocation, executable construction, or slot assignment. No concrete backend currently
implements this collaboration.

## The staged prepare handoff

The first handoff is now represented by `PrepareContext`. Shared Prepare can build this
partition-scoped projection from stable semantic and Planning facts plus fully resolved
prepare-time inputs. It is not `CompileArtifacts` and exposes no Compiler-owned implementation
state. Concrete backends may consume the projected Model and Planning contracts during analysis;
Runtime never receives those graph objects.

Through the current `BackendPartitionPreparer` collaboration, the owning backend deterministically
analyzes and lowers the partition from those explicit inputs. It selects one supported route and
configuration and returns a `BackendPartitionAnalysis`. The result has two parts:

- an opaque backend plan retaining the selected lowering, fusion, route, and private
  configuration; and
- exact backend-neutral declarations for every buffer and workspace resource that shared
  preparation must assign.

The next lifecycle step is current inside a package-private complete-set handoff. Shared Prepare
assigns stable Runtime-owned `BufferSlot` and `WorkspaceSlot` identities, retains exact
requirement-to-slot associations, and constructs one `PreparedMemoryPlan`. The current rule is
conservative: one buffer slot per distinct declared `ValueId`, with maximum declared size and
alignment when that value is repeated, and one distinct workspace slot per workspace declaration.
No aliasing, lifetime, interference, or reuse model is inferred.

Backend finalization is also current as a shared contract. After assignment, the owning backend's
`BackendPartitionFinalizer` receives one typed `BackendPartitionFinalization` containing its exact
analysis, the exact shared plan, and assignments in declaration order. It constructs a current
`PreparedExecutable`; shared Prepare then creates the minimal current `PreparedPartition`
association. Finalization may validate backend-private immutable state and construct ordinary
immutable Java recipe state, but it must not change route choice, add an undeclared need, allocate
physical resources, or acquire a closeable prepared resource under the current contract.

The batch handoff, entry, and result are package-private. Public orchestration that projects
compile artifacts and selects the complete explicitly registered backend set remains planned.

Any dynamic or unresolved Shape currently fails `PrepareContext` construction before backend
analysis. A future fact may remain run-dynamic only when an explicit prepared contract represents
it without changing the selected route, declared resources, or slot assignment. The current
repository has no such run-dynamic fact contract.

## What prepare creates

The complete architecture prepare lifecycle creates:

- `PreparedPartition`, which associates a planned partition and backend identity with its
  prepared work;
- a possible `PreparedUnit`, if the actual finalization or schedule consumer later establishes a
  distinct invariant beyond the executable and its selections;
- `PreparedExecutable`, the current reusable cold-binding recipe for one prepared region;
- `PreparedMemoryPlan`, whose current Runtime contract defines final buffer/workspace slot
  geometry without allocating physical storage;
- `PreparedSchedule`, which orders executable, transfer/materialization, and publication work; and
- `PreparedExecution`, the immutable reusable runtime-ready result, including any immutable
  persistent prepared resources that are not ordinary per-run workspace.

Today, `PreparedMemoryPlan`, `PreparedRepresentationPlan`, `PreparedExecutable`,
`PreparedSchedule`, `PreparedExecution`, and `PreparedPartition` exist among the prepared/runtime
contracts in this list; `BoundInvocation` is
the current per-run result of binding an executable. The current `PreparedExecution` retains one
exact plan and one schedule that reports that same plan reference. It owns no resource and has no
run or close lifecycle. The current schedule retains one exact plan and an immutable ordered
snapshot. It permits one optional first-only representation-creation prefix plus executable and
buffer-transfer occurrences, followed by an optional dense publication-only suffix. Empty,
executable-only, transfer-only, and zero-publication schedules and repeated pre-publication
occurrences remain valid. Distinct publication positions may name the same exact representation.
It has no `PreparedUnit` and invokes no callback, binding, publication, or execution. No current public Prepare orchestration
constructs or consumes it. The analysis-side and finalization-side Prepare contracts described
above are also current. `modules/prepare` owns `PrepareContext`,
`BackendPartitionPreparer`, `BackendPartitionAnalysis`, shared resource declarations, current
assignment and source associations, `PreparedPartition`, and later public orchestration and
validation. Engine-level composition supplies explicitly registered backend implementations and
their input facts. Prepare
does not interpret the backend's opaque route plan.

Shared prepare code does not implement concrete CPU, Metal, or CUDA lowering and does not own backend-specific executable or storage implementations.

## What runtime does

`modules/runtime` owns the contracts and state needed after preparation. It:

- executes `PreparedSchedule`;
- creates exactly one `RunState` for each active complete logical run and binds caller inputs as
  borrowed resources;
- manages runtime slots, resources, and workspaces through prepared contracts;
- performs scheduled residency, transfer, and materialization work;
- invokes current cold-bound `BoundInvocation.execute()` objects from later scheduled work;
- updates residency after execution; and
- publishes outputs and gradients according to the prepared plan and run policy.

Runtime does not optimize the graph, construct autograd, choose backend ownership, discover backends, lower partitions, or select kernels. Its hot path does not operate on `Operation` or `CompiledNode`.

Runtime profiling is passive observation of this execution. Runtime may translate observed facts
into typed trace payloads, but neither profiling nor tracing selects or mutates execution
settings. Runtime performs no model-autotuning search, cache lookup or mutation, or hot-path graph
inspection.

## Per-run resources and cold binding

```text
immutable PreparedExecution
  -> one isolated RunState for the complete heterogeneous run
     -> BufferSlot -> one or more explicitly prepared representations
     -> WorkspaceSlot -> one backend-local scratch representation
  -> cold checked binding -> backend-owned typed direct-reference invocation
  -> hot-path execution -> publication -> cleanup of resources still owned by the run
```

Concurrent invocations reuse the prepared recipe but never a `RunState` or run-owned mutable
resource. Runtime owns logical slot state, ownership transitions, validity/residency, cleanup
orchestration, and failure isolation. Concrete backends own physical representation classes and
perform allocation, release, transfer, and access.

The current Runtime foundation implements the prepared origin description, package-private cold
creation and rollback, structural residency, explicit per-copy buffer validity, carrier, cleanup,
and cold-binding portions of this flow.
`BufferRepresentation` and `WorkspaceRepresentation` are distinct nominal closeable roles with
no physical access API. `BufferRepresentationBinding` marks one exact buffer representation as
borrowed or run-owned. `RunState` retains one exact `PreparedMemoryPlan`, copies supplied list
structure into private arrays in plan encounter order, rejects repeated representation identity,
and exposes direct indexed access. Every supplied workspace is run-owned. Successful construction
transfers cleanup responsibility; failed construction transfers nothing and closes nothing.

Each bound representation is structurally resident until closure. One independent boolean per
buffer copy records logical validity: borrowed caller inputs start valid, created run-owned
buffers start invalid, zero or multiple copies may be valid, and workspaces remain scratch
outside logical validity. Query and mutation are explicit constant-time array operations without
copying, backend work, ownership changes, or implicit coherence.

The package-private cold creation operation validates the complete dense caller-input list before
callbacks, creates buffers in buffer/representation order and then workspaces, and constructs one
state. A partial failure closes successfully created results once in reverse creation order,
preserves the original unchecked failure, suppresses cleanup failures, and never closes borrowed
inputs.

Current cleanup marks the state closed first, skips borrowed buffers, and attempts every owned
representation once in deterministic reverse order. It preserves the first unchecked exception
or error and suppresses later failures. The state is not thread-safe, but separate states may
share the immutable plan while keeping run-owned representations and validity arrays isolated.
The current transfer foundation adds an immutable recipe for two distinct already-created
representation positions of one buffer. Cold binding validates exact plan/state association,
position bounds, and concrete source/destination compatibility once, then produces a backend-
owned bound action retaining direct concrete references. Its final action makes a valid
destination a no-op; otherwise it requires a valid source, invokes backend work once, and marks
only the destination valid after success. Backend failure leaves Runtime validity unchanged.
Materialization is this same explicit transfer when it produces an equivalent already-created
destination representation. It adds no second operation kind, allocation, route search,
invalidation, or hidden coherence.

This implemented foundation still has no concrete allocation or storage-access implementation or
public output-value access.

The current `PreparedExecutable` retains one exact plan reference plus private immutable snapshots
of ordered dense buffer/representation and workspace selections, with aligned read-only,
write-only, or read-write declarations. Empty and repeated selections are valid. `bind` requires
an open `RunState` associated by exact plan reference identity, resolves
buffers before workspaces in selection order, and invokes concrete-backend compatibility hooks
once per selected representation. A normal incompatibility returns `false` so Runtime can issue a
stable indexed failure.

After all compatibility checks pass, the backend constructs a current `BoundInvocation` retaining
the exact run state and direct concrete typed buffer/workspace fields. Binding may allocate fresh
nominal arrays and the invocation object, but it acquires no auxiliary closeable binding resource
and changes no ownership. `BoundInvocation.execute()` performs one state-open guard, then calls
the backend implementation. It rejects execution after the state closes and owns or closes
nothing. One immutable executable may bind concurrently to distinct states; a bound invocation is
not thread-safe and must not race execution with closure.

Caller inputs are borrowed, internal buffers and workspaces are run-owned, and current published
results lease the complete `RunState` to `RunResult`; immutable persistent prepared resources
remain `PreparedExecution`-owned. A workspace is backend-local scratch rather than a transferable
logical value; host staging and device scratch use separate workspace requirements.

The current immutable `PreparedPublication` identifies one already-created buffer representation
with dense Runtime buffer, representation, and result positions against one exact plan. These are
prepared/run coordinates, not compiler graph or Tensor identities. Its cold binding resolves the
selected physical representation once into a per-run `BoundPublication`. Publication requires
the exact selected copy to be valid at that moment, changes only a local one-shot flag, and
performs no lookup, transfer, fallback, conversion, allocation, backend callback, validity change,
or ownership mutation. Publication occurrences form the schedule's dense final suffix. Different
result positions may intentionally alias one exact representation.

After all bound occurrences publish successfully, `RunResult` privately snapshots their direct
representation references and leases cleanup of the complete open state. Empty results are valid.
Partial publication or failed result construction transfers no cleanup responsibility. Closing
the result delegates to the existing state cleanup, so borrowed inputs remain caller-owned and
run-owned resources retain deterministic reverse cleanup. The result exposes count and lifecycle
only; output value, representation, Tensor, and state access remain deliberately absent.

Java representation compatibility is now checked explicitly once at the current cold binding
boundary. The backend creates typed bound objects with direct references, so the hot path needs no map lookup,
reflection, string dispatch, graph inspection, backend discovery, kernel selection, or repeated
unsafe cast. The shared contracts use no raw `Object`, unchecked generic API, public backend type
switch, registry, or service locator.

`PreparedExecutionRunner` creates one isolated state, binds all remaining schedule occurrences
before the first action, and traverses direct bound references in encounter order. It validates
executable reads before invalidating every copy of each output buffer and validates only exact
writes after success. Failure leaves those output copies invalid and closes the state while
preserving the original unchecked failure. The stateless runner supports concurrent isolated
calls, but each call is synchronous and uses one orchestrating thread.

The initial model adds no automatic pooling, reuse, aliasing, hidden coherence/write-back,
distributed sharding, or multi-device scheduling. Transfer/materialization recipes and their
success-only validity transition, prepared publication, result lease, executable-output
invalidation, and schedule traversal are current; public output access remains later work.

## What a concrete backend does

Each concrete backend owns preparation and execution details for the partitions assigned to it. This includes:

- backend-owned analysis and lowering of planned graph regions;
- backend-specific fusion and specialization;
- concrete kernel or executable route selection;
- exact declaration of shared buffer and workspace requirements;
- construction of immutable `PreparedExecutable` implementations during the current finalization
  contract;
- physical buffer and workspace representation implementations plus their
  allocation/release/transfer/access mechanics;
- current backend-owned typed `BoundInvocation` subclasses with direct representation references;
- backend-specific materialization and native bridge integration; and
- backend trace contributions.

Planning chooses an owner such as CPU, Metal, or CUDA. The owning backend then chooses scalar, Vector API, OpenBLAS, MPSGraph, a custom Metal kernel, a CUDA kernel, or another backend-internal route during prepare.

Route selection occurs during analysis, before shared slot assignment. Executable construction
occurs during finalization, after slot assignment. Neither step consumes `CompileArtifacts` or
Compiler internals, and neither is repeated by Runtime.

Each backend also owns typed, version-controlled, tested candidate generators beside the routes
they configure. A generator uses target capabilities, canonical workload facts, and the tuning
budget to return complete valid configurations. The operation family selects the generator; it is
not a cache key for one universal family-wide setting. Hardware and supported JDK Vector API
species constrain CPU vector candidates, so no candidate can promise an arbitrary physical lane
count.

A future narrow prepare/tuning boundary exposes complete backend candidates opaquely to shared
orchestration. Shared code does not interpret route, vector, thread, tile, kernel, or other private
fields. During ordinary preparation, the backend may reuse a compatible entry from an explicit
workload cache or apply safe heuristics. Model autotuning remains optional for correctness.

The model-specific tuning result is an explicit prepared plan or artifact, not hidden global
state. Its persistent plan record and the reusable workload cache are loaded and updated outside
runtime; incompatible or corrupt entries fall back safely. Physical cache formats and prepared-
executable serialization remain deferred.

## Why there is no shared `backend.lowering` module

Lowering is inseparable from a backend's implementation model, fusion opportunities, specialization, storage, and executable construction. A shared lowering module would either encode concrete backend knowledge in a supposedly common layer or reduce lowering to abstractions that do not own the real decision.

Backend-neutral graph canonicalization and decomposition belong in compiler passes. Backend-specific lowering belongs in each concrete backend's prepare implementation.

## Why there is no runtime service locator

Runtime must execute an already-prepared schedule. Looking up a backend or kernel service during execution would move selection and executable construction into the hot path and hide dependencies.

The engine instead registers backends explicitly and supplies their capability and preparation services before runtime execution. `PreparedExecutable` provides the backend-independent runtime call boundary.

## Why reflective backend discovery is not the core mechanism

Classpath scanning, annotation scanning, reflection, or `ServiceLoader` as the default mechanism would make backend availability implicit and mix composition with execution concerns. Engine is the composition root, so backends are registered explicitly when the engine is built.

An optional convenience layer for plugin discovery would require the architecture update specified by the contract. It must not become a runtime hot-path mechanism.

See [Lifecycle](lifecycle.md) for the complete stage flow, [Module Boundaries](module-boundaries.md) for ownership, and [Dependency Rules](dependency-rules.md) for prohibited dependency edges.
See [Performance Evidence and Tuning](performance-evidence-and-tuning.md) for the optimization
workflow boundaries that feed prepare without entering runtime.
