# Runtime, Prepare, and Engine contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.

## Scope

This contract owns Runtime, Prepare, and Engine responsibilities; explicit composition and the
service-locator/plugin prohibitions; prepare and run lifecycles; and persistent/per-run resource
ownership and closure. It does not own compile semantics or concrete backend implementation.

## `modules/runtime`

`modules/runtime` owns prepared execution contracts and dynamic runtime state.

Allowed:

- `PreparedExecution`
- `PreparedExecutable`
- `PreparedSchedule`
- `PreparedMemoryPlan`
- `BufferSlot`
- `WorkspaceSlot`
- `RuntimeSlotTable`
- `RunState`
- residency management
- transfer execution
- publication execution
- runtime resources
- prepared execution runner
- passive runtime profiling and observation through typed trace contracts

Forbidden:

- graph optimization
- autograd construction
- compiler passes
- backend discovery
- service lookup for backend implementations
- concrete backend dependencies
- kernel selection
- backend-specific lowering
- `Operation` in hot path
- `CompiledNode` in hot path
- model-autotuning search, tuning-cache lookup or mutation, graph inspection for tuning, or
  selection of execution settings in the hot path

Runtime executes prepared schedules.

One `PreparedExecution` retains exactly one immutable `PreparedMemoryPlan` and one same-plan
`PreparedSchedule`. The schedule is an ordered list of `PreparedSchedule.Step` occurrences: an
optional `RepresentationCreationStep` may occur only first, followed by executable and explicit
buffer-transfer occurrences, with publication occurrences forming the dense final suffix.
Repeated executable or transfer occurrences mean repeated work and create no additional resource
ownership. Runtime uses no additional aggregate between a schedule occurrence and its recipe.

Runtime does not decide how graph partitions should be lowered.

Runtime does not select kernels.

Runtime does not discover backend plugins.

Runtime owns each run's logical slot state, resource-lifecycle orchestration, representation
validity and residency needed by the prepared schedule, failure cleanup, and concurrent-run
isolation. Concrete backends own physical buffer and workspace representation implementations and
the allocation, release, transfer, and access mechanics for those representations. Runtime must
not know concrete host, native, Metal, or CUDA storage classes and must not choose a backend.

A buffer slot may have one or more backend/device representations only when the prepared schedule
requires them. Representation creation and transfer are explicit prepared work, not on-demand
kernel or backend discovery. A workspace slot is per-run backend-local implementation scratch and
normally binds one physical representation for its declared use; host staging and device scratch
are separate workspace requirements when both are needed.

Caller inputs are borrowed for one run. Internal buffers and workspaces are run-owned. Published
outputs transfer or lease ownership to `RunResult`, while immutable persistent prepared resources
remain owned by `PreparedExecution` and are not ordinary workspace. Runtime orchestrates cleanup,
but concrete representations perform physical release. Failure cleanup releases only resources
still owned by the run, never borrowed inputs or already transferred outputs.

A persistent prepared resource implements a narrow Runtime-owned nominal lifecycle contract;
concrete backend code owns its physical state and performs physical release. `PreparedExecution`
snapshots each acquired resource identity exactly once and closes the unique resources in
deterministic reverse-acquisition order. A repeated schedule occurrence or executable reference
does not create another ownership occurrence.

Closing a prepared execution atomically rejects new runs. A synchronous run that already acquired
its prepared-resource lease may finish. Close does not wait for active runs: if any remain,
physical release is deferred to the last lease release, avoiding an unbounded wait or a close/run
deadlock. Otherwise the closing caller releases resources immediately. Cleanup is attempt-all and
idempotent; the thread that performs physical release receives the first unchecked failure or
error, with later distinct failures suppressed in reverse-cleanup encounter order and repeated
references to the same primary throwable skipped to avoid self-suppression.

Before hot-path execution, a cold binding phase validates representation compatibility and creates
backend-owned typed bound invocation objects with direct references. Any heterogeneous Java type
check is explicit, checked, and confined to that boundary. The hot path performs no map lookup,
reflection, string dispatch, graph inspection, backend discovery, kernel selection, or repeated
unsafe cast. Runtime resource contracts must not use raw `Object`, unchecked generic access, a
global registry, a service locator, or a public switch over concrete backend types.

## `modules/prepare`

`modules/prepare` owns shared prepare contracts and validation.

Allowed:

- `PrepareContext`
- `BackendPartitionPreparer`
- `BackendPartitionAnalysis`
- backend-neutral buffer and workspace requirement declarations
- `PreparedPartition`
- shared assignment of stable runtime buffer and workspace slot identities
- partition coverage validation
- prepared memory validation
- prepared schedule validation
- a future narrow orchestration boundary that exposes complete valid preparation candidates
  opaquely to model-autotuning tooling

Forbidden:

- concrete CPU lowering
- concrete Metal lowering
- concrete CUDA lowering
- concrete kernel selection
- backend-specific executable implementation
- backend-specific storage implementation
- interpretation of private backend candidate parameters

Prepare is the bridge between compile artifacts and runtime. It projects the exact stable
semantic and planning facts, resolved prepare-time bindings, target capabilities, configuration,
and compatible cached tuning decisions that backend analysis may consume. That projection must
not expose `CompileArtifacts` or another compiler-owned aggregate to a concrete backend.

For each planned partition, the concrete backend first analyzes and lowers the projected facts,
selects a supported route and configuration, and returns a `BackendPartitionAnalysis`. The
analysis retains the backend's selected lowering and route state opaquely while declaring every
shared buffer and workspace requirement exactly enough for shared preparation to assign stable
runtime slot identities. Shared preparation does not interpret the opaque backend plan or private
route vocabulary.

After shared preparation assigns slots, the same backend finalizes the analysis against those
assignments and constructs the `PreparedExecutable` and `PreparedPartition`. Backend
finalization must not change the selected route or introduce undeclared shared requirements. It
may acquire immutable persistent prepared resources after assignment. A finalizer owns rollback
before it returns; shared Prepare owns returned resources transactionally until a completely
validated `PreparedExecution` accepts ownership. Any intervening failure closes the acquired
resources once in deterministic reverse order. Finalizer results list resources in acquisition
order; Prepare concatenates them in partition-finalization order, rejects repeated exact resource
identities, and never derives ownership from executable or schedule occurrences. Rollback
preserves the preparation failure and suppresses distinct resource-close failures in cleanup
encounter order, skipping self-suppression. Per-run physical allocation and binding remain
runtime/backend concerns after preparation.

Backend analysis is deterministic from its explicit facts, configuration, and compatible cache
inputs. An explicitly enabled later model-autotuning workflow may instead supply a selected
compatible decision before analysis; this lifecycle does not authorize prepare-time measurement
or search. Any unresolved fact needed to choose a route or declare an exact resource requirement
must fail preparation unless an explicit prepared contract represents that fact as run-dynamic
without changing route or slot assignment.

Concrete backend prepare implementations live in concrete backend modules.


## Prepare lifecycle

Prepare lifecycle:

```text
CompileArtifacts
  -> validate partition coverage
  -> project partition-scoped semantic/planning facts and resolved prepare inputs
  -> for each PlannedPartition call BackendPartitionPreparer analysis
  -> backend analysis does lowering/specialization/fusion/kernel selection
     and declares exact shared buffer/workspace requirements
  -> build BackendPartitionAnalysis[]
  -> assign stable buffer/workspace slots and build PreparedMemoryPlan
  -> finalize each backend analysis against its assigned slots
  -> build PreparedPartition[] and PreparedExecutable[]
  -> build PreparedSchedule
  -> validate prepared memory/schedule
  -> PreparedExecution
```

Prepare is where these are created:

- `BackendPartitionAnalysis`
- `PreparedPartition`
- `PreparedExecutable`
- `PreparedMemoryPlan`
- `PreparedSchedule`
- `PreparedExecution`

Concrete backend lowering occurs in concrete backend modules.
Route selection and shared-resource discovery occur during backend analysis. Executable
construction occurs only during backend finalization after shared slot assignment. Shared Prepare
assembles and validates ordered schedule-step occurrences, then constructs the Runtime
`PreparedExecution` with the exact assigned memory plan, exact schedule, and acquisition-ordered
persistent resources. Successful construction transfers unique ownership of those resources to
that Runtime aggregate.


## Run lifecycle

Run lifecycle:

```text
PreparedExecutionRunner.run(PreparedExecution, callerInputs)
  -> acquire one prepared-execution run lease
  -> create exactly one RunState for the complete logical run
     - bind caller inputs as borrowed representations
     - invoke the optional first representation-creation recipe for run-owned buffers/workspaces
  -> cold-bind every executable, transfer, and publication occurrence
     to typed direct-reference actions
  -> traverse PreparedSchedule.Step occurrences in order
     - execute PreparedExecutable occurrences
     - perform explicit prepared buffer-transfer/materialization occurrences
     - publish the dense final suffix
  -> RunResult
     - owns the complete RunState lease and releases resources still owned by the run on close
  -> release the prepared-execution run lease before the synchronous call returns
```

`PreparedExecutionRunner` owns this synchronous orchestration. It is stateless and creates one
isolated `RunState` per call; concurrent calls may share the immutable prepared recipe but not
mutable run state. `PreparedExecution` remains the unique owner of persistent prepared resources.
Executable and transfer binding create backend-owned typed actions; Runtime publication binding
retains the selected representation directly without transferring its ownership.

Run must not perform:

- graph optimization
- autograd construction
- backend discovery
- kernel selection
- backend-specific lowering
- compiler passes

The initial runtime resource model introduces no automatic pooling, reuse, aliasing, distributed
sharding, hidden mutation/coherence protocol, or multi-device scheduling. Multiple physical
representations exist only when explicitly required by a prepared schedule, and immutable
functional value semantics do not imply hidden write-back between them.


## `modules/engine`

`modules/engine` owns public lifecycle orchestration and composition.

Allowed:

- public `Engine` facade
- public `CompiledGraph` facade
- owner-bound public `PreparedExecution` and `RunResult` handles
- compile orchestration
- prepare orchestration
- synchronous run orchestration
- wiring compiler, runtime, prepare, and concrete backends

Forbidden:

- kernel implementations
- backend internals
- graph optimizer passes
- runtime service locator
- reflective plugin discovery as the core backend mechanism

Engine is the composition root.

Current ordinary composition is fixed: every `Engine.standard()` call constructs and owns one
fresh CPU integration. Current advanced composition is also CPU-only:
`AdvancedEngine.takeOwnership(...)` takes cleanup ownership of one explicitly supplied supported
CPU integration. Neither surface currently registers a generic backend inventory, combines
multiple backend owners, or assembles mixed-owner schedules.

Generic explicit backend registration is planned composition-time work. If introduced, it must
remain Engine-owned and explicit; it must not become discovery, a service locator, or runtime
hot-path selection. No current builder or generic registration API is implied.

## Runtime service locator

A runtime service locator is forbidden as a core mechanism.

A runtime service locator means runtime dynamically asks for services or backends during execution, for example:

```java
Backend backend = RuntimeServices.get("metal");
KernelRegistry kernels = RuntimeServices.get(KernelRegistry.class);
```

This is forbidden because runtime must execute already-prepared schedules.

Backend selection and executable construction must happen before runtime hot path execution.

## Reflective backend plugin discovery

Reflective backend plugin discovery is forbidden as the core backend mechanism.

Examples include:

- classpath scanning
- annotation scanning
- automatic backend discovery through reflection
- `ServiceLoader` as the default runtime backend mechanism

Any future generic backend registration must be explicit through Engine composition.

`ServiceLoader` or plugin discovery may be added later as a convenience layer only if this document is updated first.

It must not become a runtime hot-path mechanism.
