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

After shared preparation assigns slots, it assigns every participating backend owner one stable
representation position for each declared logical buffer. The producing owner is first when one
exists; otherwise the first consuming owner is first. Remaining distinct consumer owners follow
logical consumer-partition order, and repeated partitions with the same owner reuse that owner's
position. The exact position is part of each buffer assignment supplied to the finalizer, so a
backend executable selects its own representation directly rather than assuming position zero.
The memory-plan buffer order begins with Compiler bindable-input IDs in exact caller occurrence
order, independent of partition and backend declaration order. Remaining ordinary buffers retain
first-declaration order and producerless resources remain last. The plan-ordered buffer association
retains the complete owner order, and each workspace association retains its exact partition owner.

The same backend then finalizes the analysis against those assignments and constructs the
`PreparedExecutable` and `PreparedPartition`. Backend finalization must not change the selected
route or introduce undeclared shared requirements. It may acquire immutable persistent prepared
resources after assignment. A finalizer owns rollback before it returns; shared Prepare owns
returned resources transactionally until a completely validated `PreparedExecution` accepts
ownership. Any intervening contribution, transfer-recipe, schedule-assembly, validation, or
aggregate-construction failure closes the acquired resources once in deterministic reverse order.
Finalizer results list resources in acquisition order; Prepare concatenates them in partition-
finalization order, rejects repeated exact resource identities, and never derives ownership from
executable, transfer, or schedule occurrences. Rollback preserves the preparation failure and
suppresses distinct resource-close failures in cleanup encounter order, skipping self-suppression.
Per-run physical allocation and binding remain runtime/backend concerns after preparation.

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
construction occurs only during backend finalization after shared slot and representation-position
assignment. Each participating backend contributes exact representation creators for its assigned
buffer positions and workspace creators for its partition-owned workspace positions. Contributions
are physical creation recipes, not an alternative schedule. Shared preparation validates complete,
non-overlapping contribution coverage, assembles and validates ordered schedule-step occurrences,
then constructs the Runtime `PreparedExecution` with the exact assigned memory plan, exact
schedule, and acquisition-ordered persistent resources. Successful construction transfers unique
ownership of those resources to that Runtime aggregate.


## Run lifecycle

Run lifecycle:

```text
PreparedExecutionRunner.run(PreparedExecution, callerInputs)
  -> acquire one prepared-execution run lease
  -> create exactly one RunState for the complete logical run
     - bind each logical caller input once through its prepared source-owner representation
     - invoke the optional first representation-creation recipe for run-owned buffers/workspaces
  -> cold-bind every executable, transfer, and publication occurrence
     to typed direct-reference actions
  -> traverse the immutable PreparedSchedule.Step occurrences in order
     - execute each partition executable in compile-partition order
     - immediately before a destination owner's first consuming partition, perform each required
       explicit source-owner-to-destination-owner buffer transfer
     - publish the dense final suffix
  -> RunResult
     - owns the complete RunState lease and releases resources still owned by the run on close
  -> release the prepared-execution run lease before the synchronous call returns
```

One logical value has at most one representation per participating backend owner. A transfer
copies from the producing owner's representation, or from the first consuming owner's caller-input
representation when the value has no producer, to one distinct consuming owner's representation.
Fan-out creates one immutable transfer recipe per distinct destination owner and schedules it once
before that owner's first consumer. The source remains valid. Runtime marks the destination valid
only after successful physical transfer; failure leaves source and destination validity unchanged
and triggers ordinary reverse run cleanup with the transfer failure primary.

`PreparedExecutionRunner` owns this synchronous orchestration. It is stateless and creates one
isolated `RunState` per call; concurrent and repeated calls may share the immutable prepared recipe
but not mutable state or run-owned representations. `PreparedExecution` remains the unique owner
of persistent prepared resources. Concrete transfer recipes retain exact direction, byte geometry,
and direct backend-owned cold binders. Cold binding validates concrete representations once and
creates an action with direct typed source, destination, and staging references. Executable and
transfer execution, and Runtime publication binding, perform no Engine, registry, backend-ID,
provider, availability, configuration, discovery, reflection, or representation-map lookup.
Publication binding retains the selected representation directly without transferring ownership.

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
- public `InferenceSession` facade over exactly one ordinary prepared handle
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
sessions and standalone prepared handles in their shared reverse preparation-publication order,
and backend integrations in reverse registration order. Cleanup is attempt-all, idempotent, and
failure-retaining. Closing a session or standalone prepared handle closes its one inward Runtime
execution without closing an already returned result. A run that already acquired Runtime's
prepared lease may finish without making close wait; its final lease release performs deferred
prepared-resource cleanup.

### Compile ownership and mixed-owner Prepare routing

Each compile receives the immutable provider and availability snapshots in registration order.
Planning selects a `BackendId` owner before preparation and compile artifacts retain identities,
not live integration objects. Registration order is deterministic compile-time input; it is not a
runtime fallback chain.

During cold preparation, Engine validates the complete non-empty planned-partition owner set and
routes every partition by exact `BackendId` to the matching private registry entry. A missing
owner fails before backend analysis. Engine then supplies one positional preparation per partition
to the existing shared `GraphPreparation` transaction; a concrete backend receives only its
partition-scoped projection and never receives the Engine registry or Compiler aggregate. There is
no retry, implicit CPU fallback, recompilation, owner substitution, or normalized-ID match.

Before analysis, Engine derives every cross-owner edge from `LogicalMemoryRequirement` producer
and consumer partitions and validates the exact ordered source/destination adapter pair, tensor
descriptor, layout, data type, and byte geometry. Unsupported direction, representation,
layout, data type, or size fails closed before backend analysis or persistent-resource acquisition.
The currently supported heterogeneous transfer domain is CPU to Metal and Metal to CPU for
positive rank-1..16 fully static canonical contiguous `FLOAT32` values within Metal's supported
canonical operation domains and with checked element and byte geometry. It performs no conversion.
The CPU run-owned native representation is the reusable host staging storage: CPU to Metal uploads
its exact bytes, while Metal to CPU downloads into it. No per-element object, canonical-byte
materialization, or additional hot-path copy is permitted.

One complete Engine preparation supplies one shared schedule assembler. Each exact owner adapter
contributes only the representation and workspace creators for positions assigned to that owner.
The assembler emits the sole optional representation-creation prefix, then partition executions in
compile order with required transfer steps immediately before each destination owner's first
consumer, and finally the dense publication suffix. Schedule construction uses the stable logical
producer/consumer facts and shared assignments; it does not rescore ownership, inspect backend
routes, or create a second scheduling convention.

On successful preparation, a session-owned hidden prepared handle or standalone outward prepared
handle retains immutable direct adapter lists aligned with logical caller inputs and publication
occurrences. Exactly one source-owner adapter creates each caller-input representation. Each
publication captures the adapter owning its selected representation, including mixed result
lists. Run and materialization use those direct references without querying the registry.
Transfer recipes retain only the direct backend-owned binders and native context references
selected during preparation. Engine shutdown preserves their owner lifetime by closing retained
results first, sessions and standalone prepared handles in their shared reverse
preparation-publication order second, and integrations last.

Preparation is transactional across all owner contributions. A finalizer retains rollback
responsibility until it returns. Shared Prepare then owns all returned persistent resources across
later finalizers, contribution assembly, transfer-recipe creation, schedule validation, Runtime
aggregate construction, and outward-handle publication. Failure closes acquired persistent
resources in reverse partition/acquisition order. Per-run representation creation is separately
transactional in `RunStateCreation`; a later creation, binding, transfer, execution, or publication
failure closes only run-owned resources in reverse acquisition order and never closes borrowed
caller storage.

`Engine.prepareTuned(...)` remains CPU-only. It rejects an empty, missing, Metal-only, or mixed
owner plan before representative-input borrowing, candidate generation, trial preparation, or
execution. A complete CPU-owned plan may tune when Metal is also registered.
`ALLOW_SAFE_HEURISTIC` may fall back only within that already-selected CPU entry after a
recoverable CPU tuning failure; it never changes owner or admits a heterogeneous trial.

The registry map has no post-prepare execution role. Runtime receives immutable schedules with
direct prepared executable, transfer, representation-creation, and publication references. Outer
Engine handles retain direct per-input and per-publication adapters. Runtime's cold-bound action
array and hot traversal never query the Engine, registry, adapter inventory, provider, availability
snapshot, configuration, backend ID, `ServiceLoader`, reflection, or a global lookup.

### Public reusable inference session

The ordinary reusable user-facing declaration is:

```java
public final class Engine implements AutoCloseable {
    public InferenceSession session(CompiledGraph compiledGraph);
}

public final class InferenceSession implements AutoCloseable {
    public CompiledGraph compiledGraph();
    public RunResult run(List<Tensor> inputs);
    public boolean isClosed();
    @Override public void close();
}
```

`Engine.session(compiledGraph)` accepts only a graph compiled by that exact Engine. Under one
Engine lifecycle admission it performs exactly one ordinary preparation and publishes one session
that owns the resulting prepared handle. Construction performs no run and binds no caller storage.
The session borrows the Engine; it does not take ownership of the Engine, integrations, registry,
or graph. The Engine remains the composition owner and final cleanup boundary.

The graph fixes the complete public boundary. `CompiledGraph.inputs()` is the stable input
membership and final Compiler occurrence order. Every session run accepts those exact logical
Tensors once in arbitrary list order, verifies exact Tensor identity and the complete compiled
descriptor, then snapshots each current `HostTensorStorage` association in compiled occurrence
order. Before borrowing, Engine validates storage data type, resolved-layout capacity, liveness,
and current-thread accessibility. The caller continues to own each storage and must keep it usable
and free from conflicting mutation until that run's result closes.

The graph's publication specifications also remain fixed. Every run returns the existing
`RunResult`; its occurrences retain dense compile publication order, with requested forward
occurrences first and explicit gradient-target occurrences second. Aliases remain distinct
occurrence objects. Exact `RunResult.Publication` identity authenticates materialization, and each
successful materialization returns a fresh detached `HostTensorValue` with its existing
lifecycle-independent lifetime.

Session run is exactly the existing prepared execution path. It uses the prepared handle's direct
input and publication adapter arrays and Runtime's stateless runner. It must not compile, prepare,
query capability providers or availability, inspect the Engine registry, select a backend or
kernel, assemble another schedule, or cache a result. It adds no per-run collection, copy, or
resource beyond the existing `Engine.run` contract. Runtime creates one isolated `RunState` and
fresh run-owned resources per call, so repeated and concurrent session runs share only immutable
prepared state.

Session run admission checks the Engine lifecycle first. Under that one Engine admission, it
retrieves and checks the session's exact hidden prepared delegate before inspecting the caller
input list, Tensor metadata, storage, or captured adapters, then passes that exact delegate into
the existing binding/run path. A closed Engine therefore wins before all arguments; under an open
Engine, a closed session fails as `prepared execution is closed` without input inspection or
borrowing. Runtime's lease authority still arbitrates a concurrent close that begins after the
delegate check.

Session close owns no second lease protocol. It closes the exact prepared handle once, atomically
rejects later run admission, and unregisters the session even if cleanup fails. A run that already
acquired Runtime's prepared lease may finish; close does not wait for it, and deferred prepared
resource cleanup occurs when the last Runtime lease releases. A returned result has independent
ownership: session close does not close it, and materialization remains valid while that result and
the borrowed Engine are open.

Engine shutdown rejects new work, waits for admitted Engine operations, closes retained results in
reverse successful-run order, then sessions and standalone prepared handles in their shared reverse
preparation-publication order, and finally backend integrations. Closing a session or standalone
prepared handle rejects later runs without closing an already returned result. A run that already
acquired Runtime's prepared lease may finish; close does not wait, and the final lease release
performs deferred prepared-resource cleanup. Session metadata and immutable graph metadata remain
readable after closure. A detached host value remains readable after result, session, Engine, and
caller-storage closure.

Null arguments, foreign graph ownership, invalid logical input membership, descriptor or storage
incompatibility, closed Engine/session state, and inward prepare/run/materialization failures use
the existing ordinary lifecycle exception categories. Engine closure wins admission precedence;
after session close under an open Engine, a run fails as a closed prepared execution. No new error
hierarchy, compiler, scheduler, result type, backend-composition rule, or tuning surface is added.

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

## Numerical-profile lifecycle

Engine construction captures one graph-wide `NumericalProfile`, defaulting to `STRICT_IEEE`.
Compiler artifacts and every `PrepareContext` projection retain that exact value. Backend
preparation may use it for capability, routes, and compatibility identity; Runtime and Trace do not
receive it and perform no per-run profile lookup.
