# Runtime, Prepare, and Backend Boundary

This document explains the boundary defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The root contract remains authoritative.

Runtime currently implements the immutable `BufferSlot` and `WorkspaceSlot` identities plus
`PreparedMemoryPlan` final slot geometry described below. Prepare currently implements the
analysis-side projection, opaque marker roles, exact resource declarations, analysis result, and
preparer collaboration. The Prepare-owned assignment that constructs Runtime geometry from those
analyses remains planned, as do backend finalization, physical allocation, per-run binding,
engine, concrete backends, and every execution contract named here. The lifecycle flow therefore
mixes current foundations with later stages; each focused section states its implementation
status.

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

The next lifecycle step after the analysis result remains planned even though its Runtime result
types are current. Shared Prepare will assign stable Runtime-owned `BufferSlot` and
`WorkspaceSlot` identities, retain the exact requirement-to-slot associations, and construct a
`PreparedMemoryPlan`. The initial rule is conservative: one distinct buffer slot per distinct
declared buffer value and one distinct workspace slot per workspace declaration, so no unproved
aliasing or lifetime model is required.

Backend finalization is also planned. After assignment, the same backend will finalize its opaque
plan against those slots and construct the `PreparedExecutable` and `PreparedPartition`.
Finalization may validate or acquire backend-owned executable resources, but it must not change
route choice or add an undeclared shared buffer or workspace need.

Any dynamic or unresolved Shape currently fails `PrepareContext` construction before backend
analysis. A future fact may remain run-dynamic only when an explicit prepared contract represents
it without changing the selected route, declared resources, or slot assignment. The current
repository has no such binding/resource contract.

## What prepare creates

The complete architecture prepare lifecycle creates:

- `PreparedPartition`, which associates a planned partition and backend identity with its prepared units;
- `PreparedUnit`, which connects a prepared executable to its input and output slots;
- `PreparedExecutable`, the hot-path executable contract for one prepared region;
- `PreparedMemoryPlan`, whose current Runtime contract defines final buffer/workspace slot
  geometry without allocating physical storage;
- `PreparedSchedule`, which orders executable, transfer, materialization, and publication work; and
- `PreparedExecution`, the reusable runtime-ready result.

Today, only `PreparedMemoryPlan` exists among the prepare-result contracts in this list; the
analysis-side Prepare contracts described above are also current. `modules/prepare` owns
`PrepareContext`, `BackendPartitionPreparer`,
`BackendPartitionAnalysis`, shared resource declarations, later assignment and source
associations, `PreparedPartition`, orchestration, and validation. Engine-level composition
supplies explicitly registered backend implementations and their input facts. Prepare does not
interpret the backend's opaque route plan.

Shared prepare code does not implement concrete CPU, Metal, or CUDA lowering and does not own backend-specific executable or storage implementations.

## What runtime does

`modules/runtime` owns the contracts and state needed after preparation. It:

- executes `PreparedSchedule`;
- creates or reuses per-run `RunState` and binds inputs;
- manages runtime slots, resources, and workspaces through prepared contracts;
- performs scheduled residency, transfer, and materialization work;
- invokes `PreparedExecutable.execute(...)`;
- updates residency after execution; and
- publishes outputs and gradients according to the prepared plan and run policy.

Runtime does not optimize the graph, construct autograd, choose backend ownership, discover backends, lower partitions, or select kernels. Its hot path does not operate on `Operation` or `CompiledNode`.

Runtime profiling is passive observation of this execution. Runtime may translate observed facts
into typed trace payloads, but neither profiling nor tracing selects or mutates execution
settings. Runtime performs no model-autotuning search, cache lookup or mutation, or hot-path graph
inspection.

## What a concrete backend does

Each concrete backend owns preparation and execution details for the partitions assigned to it. This includes:

- backend-owned analysis and lowering of planned graph regions;
- backend-specific fusion and specialization;
- concrete kernel or executable route selection;
- exact declaration of shared buffer and workspace requirements;
- construction of `PreparedExecutable` implementations;
- backend storage, buffer, and workspace implementations;
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
