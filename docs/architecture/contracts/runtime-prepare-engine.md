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

`modules/engine` owns public lifecycle orchestration and explicit backend composition.

Allowed:

- public `Engine` facade
- public `CompiledGraph` facade
- owner-bound public `PreparedExecution` and `RunResult` handles
- compile orchestration
- prepare orchestration
- synchronous run orchestration
- an Engine-owned composition builder and private backend registry
- wiring compiler, runtime, Prepare, and concrete backends

Forbidden:

- kernel implementations
- backend internals or native-configuration parsing
- graph optimizer passes
- runtime service locator
- reflective plugin discovery as the core backend mechanism
- a process-global or otherwise shared mutable backend registry

Engine is the composition root. Concrete backend modules expose backend-owned integrations and
configuration without depending on Engine; Engine supplies the outward adapters and registry that
compose them.

### Public explicit composition

The selected public declaration shape is:

```java
public final class Engine implements AutoCloseable {
    public static Builder builder();

    public static final class Builder implements AutoCloseable {
        public Builder takeOwnership(CpuBackendIntegration integration);
        public Builder takeOwnership(MetalBackendIntegration integration);
        public Engine build();
        @Override public void close();
    }
}
```

Each `takeOwnership` returns the same builder for chaining. `build()` requires at least one
accepted integration. The builder is single-use: successful build spends it, failed build closes
it terminally, and explicit close is idempotent. Later non-null `takeOwnership` calls still take
and immediately close their argument before rejecting use of a spent or closed builder, so a
caller never regains ownership implicitly.

`Engine.Builder` is an explicit, single-use, `AutoCloseable` construction owner. It has concrete
`takeOwnership(CpuBackendIntegration)` and `takeOwnership(MetalBackendIntegration)` overloads
rather than a backend-implemented Engine interface. A later built-in backend requires another
explicit Engine-owned adapter and overload. This keeps the dependency direction from Engine to
backends and prevents a backend module from depending on Engine.

Because those backend-owned types appear in public Engine signatures, the implementation must
publish the CPU and Metal module dependencies to Engine API consumers. That intentional outward
visibility is limited to explicit construction; it does not permit either backend to depend on
Engine or leak backend types into compile, prepare, run, handle, or result signatures.

For a non-null argument, ownership transfers at entry to `takeOwnership`, before Engine
interrogates the integration. The builder therefore closes the supplied integration if identity,
capability, availability, or duplicate validation fails; the caller must not close or reuse it
after calling the method, whether the method succeeds or fails. A null argument transfers
nothing. After `build()` succeeds, ownership of the complete ordered registry moves atomically to
the returned Engine and closing the spent builder is a no-op. Before successful build, closing the
builder closes accepted integrations in reverse registration order.

Each successful registration captures exactly one `BackendId`, one stable capability-provider
reference, and one immutable `BackendAvailabilitySnapshot`. All three identities must name equal
backend IDs. The snapshot is taken at registration time and is fixed for the lifetime of the
built Engine; compiling never refreshes availability. Capability queries use the retained
provider, which must remain deterministic for its unchanged backend configuration. To observe a
different availability or configuration, construct a new Engine.

The registry preserves registration order and rejects an equal `BackendId` without replacing or
using the previously accepted integration. Equality is `BackendId.equals`, not object identity or
normalized text. The previous entry remains owned and unchanged; the rejected newly transferred
integration is closed. The registry is private Engine construction state, not a public lookup API,
global registry, service locator, or discovery mechanism.

`Engine.standard()` remains the ordinary CPU convenience. Every call opens one fresh default CPU
integration and transfers it through the same builder ownership path. It performs no discovery
and neither enables Metal nor changes the existing CPU lifecycle behavior.

`AdvancedEngine.takeOwnership(CpuBackendIntegration)` remains the current CPU-only lower-level
surface; the builder decision neither removes it nor makes it generic.

`MetalBackendConfiguration` and `MetalBackendIntegration` belong to `backends/metal`.
`MetalBackendIntegration.open(configuration)` validates and snapshots the backend-owned
configuration and acquires native ownership before it can be transferred to Engine. Engine
neither owns a duplicate Metal configuration type nor reads a library path, device selector, or
other Metal-private option.

### Construction failure and closure

A `takeOwnership` failure remains primary and closes the newly transferred candidate; integrations
accepted by earlier calls remain registered and owned by the builder. A `build()` failure remains
primary, terminally closes the builder, and attempts every accepted integration in reverse
registration order. Later distinct close failures are suppressed on the primary in cleanup
encounter order; an identical throwable reference is skipped to avoid self-suppression. Engine
construction publishes no partially built Engine.

An explicit builder or Engine close with no earlier primary failure attempts all owned cleanup in
the required reverse order. Its first unchecked failure or error is primary; later distinct
failures are suppressed in cleanup encounter order, and repeated references to the primary are
skipped. Repeated close is idempotent and replays the retained primary failure.

An Engine owns its registry until Engine close. Shutdown first rejects new operations and waits
for admitted Engine operations, then closes retained results in reverse run-publication order,
retained prepared handles in reverse prepare-publication order, and backend integrations in
reverse registration order. Cleanup is attempt-all, idempotent, and failure-retaining. Closing a
prepared handle closes its one inward Runtime execution but does not close an already-returned
result. Runtime remains the unique prepared-resource and run-lease authority.

### Compile ownership and Prepare routing

Each compile receives the immutable provider and availability snapshots in registration order.
Planning selects a `BackendId` owner before preparation and compile artifacts retain identities,
not live integration objects. Registration order is deterministic compile-time input; it is not
a runtime fallback chain.

During cold preparation, Engine validates the complete planned-partition owner set and routes
each partition by exact `BackendId` to the matching private registry entry. Shared Prepare still
owns projection, staged analysis/finalization, slot assignment, transactional resource transfer,
schedule validation, and construction of the one Runtime `PreparedExecution`. A concrete backend
receives only its partition-scoped Prepare projection and never receives the Engine registry or
Compiler aggregate.

The first explicit-composition implementation is a single-owner vertical slice: a non-empty
compile plan may prepare only when every planned partition has one equal registered owner. Engine
then supplies that owner's complete preparation and schedule contribution. A missing owner,
zero-partition plan, or more than one distinct owner fails before any backend analysis or
preparation-time persistent-resource acquisition. There is no retry, implicit CPU fallback,
recompilation, or owner change.

On successful single-owner preparation, the outward Engine prepared handle retains a direct,
non-owning reference to that owner's private adapter alongside its one inward Runtime execution.
Engine retains integration ownership. `Engine.run(...)` uses the direct adapter to create
backend-owned representations that borrow caller `HostTensorStorage` before invoking Runtime; the
returned Engine result retains the same direct adapter for authenticated backend-owned host
materialization. These references are Engine-handle state, not Runtime recipe state or registry
lookups. Closing results and prepared handles before integrations keeps every direct adapter
reference within its owner lifetime.

Mixed-owner preparation remains fail-closed until a separate contract defines cross-owner
representation creation, transfer direction and ownership, transfer capability and failure,
schedule ordering, shared-slot declarations, and rollback across backend contributions. Merely
registering CPU and Metal does not authorize a mixed-owner schedule.

`Engine.prepareTuned(...)` remains the current bounded CPU-only workflow. After ordinary argument
and owner validation, it may proceed only when the complete non-empty plan has the single
registered CPU owner; a registry that also contains Metal does not prevent that CPU-owned plan
from tuning. A single Metal owner fails with `IllegalStateException` before representative-input
borrowing, candidate generation, trial preparation, or execution. Missing, empty, and mixed owner
sets retain the ordinary preparation rejection. `ALLOW_SAFE_HEURISTIC` may fall back only within
that already-selected CPU entry after a recoverable CPU tuning failure; it never changes owner,
uses Metal, or turns unavailable tuning into ordinary Metal preparation.

The registry map has no post-prepare execution role. Runtime receives direct prepared executable,
transfer, representation, and publication references, while outer Engine run and materialization
use the direct selected adapter captured in their handles. Runtime and its cold-bound hot path
never query the Engine, registry, adapter, provider, availability snapshot, configuration, backend
ID, `ServiceLoader`, reflection, or a global lookup.

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
