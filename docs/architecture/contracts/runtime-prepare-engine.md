# [Runtime](../../glossary.md#term-runtime), [Prepare](../../glossary.md#prepare), and [Engine](../../glossary.md#term-engine) contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.<span data-glossary-navigation="true"> ([normative glossary](../../glossary.md#term-normative); [scope glossary](../../glossary.md#term-scope); [dependency glossary](../../glossary.md#term-dependency))</span>

## [Scope](../../glossary.md#term-scope)

This contract owns [Runtime](../../glossary.md#term-runtime), [Prepare](../../glossary.md#prepare), and [Engine](../../glossary.md#term-engine) responsibilities; explicit composition and the
service-locator/[plugin](../../glossary.md#term-plugin) prohibitions; prepare and [run](../../glossary.md#term-run) lifecycles; and persistent/per-run resource
[ownership](../../glossary.md#term-ownership) and closure. It does not own [compile](../../glossary.md#compile) semantics or [concrete backend](../../glossary.md#concrete-backend) implementation.

## `modules/runtime`

`modules/runtime` owns [prepared execution](../../glossary.md#prepared-execution--preparedexecution) contracts and dynamic [runtime](../../glossary.md#term-runtime) state.

Allowed:

- [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution)
- [`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable)
- [`PreparedSchedule`](../../glossary.md#prepared-schedule--preparedschedule)
- [`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan)
- [`BufferSlot`](../../glossary.md#buffer-slot--bufferslot)
- [`WorkspaceSlot`](../../glossary.md#workspace-slot--workspaceslot)
- `RuntimeSlotTable`
- [`RunState`](../../glossary.md#run-state--runstate)
- [residency](../../glossary.md#residency) management
- transfer execution
- [publication](../../glossary.md#term-publication) execution
- runtime resources
- [prepared execution runner](../../glossary.md#prepared-execution-runner--preparedexecutionrunner)
- passive [runtime profiling](../../glossary.md#runtime-profiling) and observation through typed [trace](../../glossary.md#trace) contracts

Forbidden:

- [graph optimization](../../glossary.md#forward-graph-optimization)
- [autograd](../../glossary.md#autograd) construction
- [compiler](../../glossary.md#term-compiler) passes
- [backend](../../glossary.md#backend) discovery
- service [lookup](../../glossary.md#term-lookup) for backend implementations
- [concrete backend](../../glossary.md#concrete-backend) dependencies
- [kernel](../../glossary.md#kernel) selection
- backend-specific [lowering](../../glossary.md#lowering)
- [`Operation`](../../glossary.md#operation) in [hot path](../../glossary.md#term-hot-path)
- `CompiledNode` in hot path
- model-autotuning search, tuning-cache lookup or mutation, graph inspection for tuning, or
  selection of execution settings in the hot path<span data-glossary-navigation="true"> ([model glossary](../../glossary.md#term-model); [autotuning glossary](../../glossary.md#autotuning--model-autotuning); [cache glossary](../../glossary.md#term-cache))</span>

Runtime executes prepared schedules.

[General scan](../../glossary.md#general-differentiable-scan) adds no Runtime [graph](../../glossary.md#graph)/body interpreter or nested execution [owner](../../glossary.md#term-owner). Its concrete loop
is already prepared; Runtime retains exactly one RunState for the whole logical [run](../../glossary.md#term-run), not one per
step or body. Complete [scan](../../glossary.md#term-cumulative-scan) semantics belong to [the scan owner](recurrent-scan.md#general-differentiable-scan).

One `PreparedExecution` retains exactly one [immutable](../../glossary.md#term-immutable) `PreparedMemoryPlan` and one same-plan
`PreparedSchedule`. The schedule is an ordered list of `PreparedSchedule.Step` occurrences: an
optional `RepresentationCreationStep` may occur only first, followed by executable and explicit
[buffer](../../glossary.md#term-buffer)-transfer occurrences, with publication occurrences forming the dense final suffix.
Repeated executable or transfer occurrences [mean](../../glossary.md#term-mean) repeated work and create no additional resource
[ownership](../../glossary.md#term-ownership). Runtime uses no additional aggregate between a schedule [occurrence](../../glossary.md#term-occurrence) and its recipe.

Runtime does not decide how graph partitions should be lowered.

Runtime does not select [kernels](../../glossary.md#kernel).

Runtime does not discover backend plugins.

Runtime owns each run's logical slot state, resource-[lifecycle](../../glossary.md#lifecycle) orchestration, representation
validity and residency needed by the [prepared schedule](../../glossary.md#prepared-schedule--preparedschedule), failure cleanup, and concurrent-run
isolation. Concrete backends own physical buffer and workspace representation implementations and
the allocation, [release](../../glossary.md#term-release), transfer, and access mechanics for those representations. Runtime must
not know concrete [host](../../glossary.md#term-host), native, [Metal](../../glossary.md#term-metal), or [CUDA](../../glossary.md#term-cuda) storage classes and must not choose a backend.

A [buffer slot](../../glossary.md#buffer-slot--bufferslot) may have one or more backend/[device](../../glossary.md#term-device) representations only when the prepared schedule
requires them. Representation creation and transfer are explicit prepared work, not on-demand
kernel or backend discovery. A [workspace slot](../../glossary.md#workspace-slot--workspaceslot) is per-run backend-local implementation scratch and
normally binds one physical representation for its declared use; host staging and device scratch
are separate workspace requirements when both are needed.

Caller inputs are [borrowed](../../glossary.md#ownership-borrowing-and-lease) for one run. [Mutable](../../glossary.md#term-mutable) internal buffers and workspaces are run-owned.
Published outputs transfer or [lease](../../glossary.md#term-lease) [ownership](../../glossary.md#term-ownership-meanings) to [`RunResult`](../../glossary.md#run-result--runresult). Immutable persistent prepared
resources remain owned by `PreparedExecution` and are not ordinary workspace. A backend may issue
a fresh run-owned read-only representation [binding](../../glossary.md#term-binding) as a child lease into immutable prepared
physical storage. Prepared-owner close rejects new children and relinquishes its owner reference;
an already-issued child keeps that storage resident until its exact [run state](../../glossary.md#run-state--runstate)/result closes.
Runtime orchestrates cleanup, but concrete representations perform physical release. Failure
cleanup [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) only resources still owned by the run, never borrowed inputs or already
transferred outputs.

A persistent [prepared resource](../../glossary.md#prepared-resource--preparedresource) implements a narrow Runtime-owned [nominal](../../glossary.md#term-nominal) lifecycle contract;
concrete backend code owns its physical state and performs physical release. `PreparedExecution`
snapshots each acquired resource [identity](../../glossary.md#term-identity-meanings) exactly once and closes the unique prepared owners in
deterministic reverse-acquisition order. A repeated schedule occurrence or executable reference
does not create another ownership occurrence. Child bindings do not add prepared-resource entries.

Closing a prepared execution atomically rejects new runs. A [synchronous](../../glossary.md#term-synchronous) run that already acquired
its prepared-[resource lease](../../glossary.md#ownership-borrowing-and-lease) may finish. Close does not wait for active runs: if any remain,
physical release is deferred to the last lease release, avoiding an unbounded wait or a close/run
[deadlock](../../glossary.md#term-deadlock). Otherwise the closing caller releases resources immediately. Cleanup is attempt-all and
[idempotent](../../glossary.md#term-idempotent); the thread that performs physical release receives the first unchecked failure or
error, with later distinct failures suppressed in reverse-cleanup encounter order and repeated
references to the same primary throwable skipped to avoid self-suppression.

Before hot-path execution, a cold binding phase validates representation compatibility and creates
backend-owned typed [bound invocation](../../glossary.md#bound-invocation--boundinvocation) objects with direct references. Any [heterogeneous](../../glossary.md#term-heterogeneous) Java type
check is explicit, checked, and confined to that boundary. The hot path performs no map lookup,
reflection, string [dispatch](../../glossary.md#term-dispatch), graph inspection, backend discovery, kernel selection, or repeated
unsafe cast. Runtime resource contracts must not use raw `Object`, unchecked generic access, a
global [registry](../../glossary.md#term-registry), a [service locator](../../glossary.md#term-service-locator), or a public switch over concrete backend types.

### Accepted symbolic-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) resource contract

The user accepted this protocol on 2026-10-24.
[Prepare0013](../../planning/modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result)
implements its declarations, explicit batch-required mode, authenticated [run](../../glossary.md#term-run) geometry and
size/[alignment](../../glossary.md#term-alignment) creator migration for the bounded ordinary [FLOAT32](../../glossary.md#term-float32) [CPU](../../glossary.md#term-cpu) path. Static consumers
retain fixed-size construction/accessors; their creators also receive exact bytes and alignment.
Final independent reviews and MAIN's native-enabled integration gate passed;
MAIN accepted the capability on 2026-10-24.
Prepare0015 implements combined one-B/fixed-T/[features](../../glossary.md#term-software-feature) FLOAT32/[FLOAT64](../../glossary.md#term-float64) length-[Scan](../../glossary.md#term-cumulative-scan) through this
existing protocol, including the separately bounded CPU outer typed/control prerequisite. Owning
[validation](../../glossary.md#term-validation) and independent code/structural review passed; root validation and the sole bounded probe completed;
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
records evidence and limits; final MAIN acceptance remains pending. No declaration/[signature](../../glossary.md#term-signature), creator,
storage/[lease](../../glossary.md#term-lease) or numeric [Runtime](../../glossary.md#term-runtime)
protocol changes.
Runtime remains [Model](../../glossary.md#term-model)/[Compiler](../../glossary.md#term-compiler)-free: no [Shape](../../glossary.md#shape), [dimension](../../glossary.md#dimension)
expression, [Operation](../../glossary.md#operation), [graph](../../glossary.md#graph), route search, generation or [cache](../../glossary.md#term-cache) [lookup](../../glossary.md#term-lookup) enters run [admission](../../glossary.md#term-admission) or
execution. CPU owns the physical meaning of numeric recipes; Runtime evaluates their exact sizes.

The following declaration excerpts use existing packages under `io.github.pho001.synaptik`;
they are not runnable examples. New Runtime types are exactly nested `PreparedMemoryPlan.ByteSize`,
nested `PreparedMemoryPlan.FixedByteSize` and `runtime.run.RunMemoryGeometry`.

```java
// runtime.memory: existing record gains explicit execution mode
public record PreparedMemoryPlan(List<BufferEntry> buffers, List<WorkspaceEntry> workspaces,
        boolean requiresBatchSize) {
    public PreparedMemoryPlan(List<BufferEntry> buffers, List<WorkspaceEntry> workspaces);
    public interface ByteSize {
        OptionalLong fixedByteSize();
        long byteSize(long batchSize);
    }
    public record FixedByteSize(long bytes) implements ByteSize { /* methods above */ }
    public record BufferEntry(BufferSlot slot, ByteSize size, long byteAlignment) {
        public BufferEntry(BufferSlot slot, long byteSize, long byteAlignment);
        public long byteSize();
    }
    public record WorkspaceEntry(WorkspaceSlot slot, ByteSize size, long byteAlignment) {
        public WorkspaceEntry(WorkspaceSlot slot, long byteSize, long byteAlignment);
        public long byteSize();
    }
}
// runtime.run.RunMemoryGeometry: public final, private construction
public static RunMemoryGeometry resolve(PreparedMemoryPlan plan, OptionalLong batchSize);
public PreparedMemoryPlan memoryPlan();
public OptionalLong batchSize();
public long bufferByteSize(int bufferIndex);
public long workspaceByteSize(int workspaceIndex);
// Existing RunState gains constructor/accessor; original constructor remains
public RunState(PreparedMemoryPlan plan, List<List<BufferRepresentationBinding>> buffers,
        List<WorkspaceRepresentation> workspaces, RunMemoryGeometry geometry);
public RunMemoryGeometry geometry();
// Existing runtime.run.PreparedExecutionRunner gains overload; original remains
public RunResult run(PreparedExecution execution, List<BufferRepresentation> callerInputs,
        RunMemoryGeometry geometry);
// Existing package-private runtime.run.RunStateCreation gains overload
static RunState create(PreparedRepresentationPlan plan, List<BufferRepresentation> inputs,
        RunMemoryGeometry geometry);
// Existing runtime.resource.PreparedRepresentationPlan creator SAMs are replaced
BufferRepresentation create(long byteSize, long byteAlignment); // BufferCreator
WorkspaceRepresentation create(long byteSize, long byteAlignment); // WorkspaceCreator
// Existing runtime.execution.PreparedExecutable gains protected state-aware hooks
protected boolean acceptsBufferRepresentation(RunState state, int selectionIndex,
        BufferRepresentation representation);
protected boolean acceptsWorkspaceRepresentation(RunState state, int selectionIndex,
        WorkspaceRepresentation representation);
```

References/results are non-null. Lists [snapshot membership](../../glossary.md#occurrence-identity-and-canonical-form) and reject null [elements](../../glossary.md#term-element) in encounter
order. Null fails with `NullPointerException`; invalid fields/[metadata](../../glossary.md#term-metadata) with `IllegalArgumentException`;
checked [overflow](../../glossary.md#term-overflow) with `ArithmeticException`; invalid dense indices with `IndexOutOfBoundsException`.
Existing [lifecycle](../../glossary.md#lifecycle) precedence/failures remain. [Immutable](../../glossary.md#term-immutable) recipes are storage-free/[thread-safe](../../glossary.md#term-thread-safe);
resolved private size arrays currently belong to their invocation. The accepted future
[Engine geometry memo](#accepted-engine-geometry-memo)<span data-glossary-navigation="true"> ([Engine geometry memo glossary](../../glossary.md#engine-geometry-memo))</span> narrowly permits sharing encapsulated,
never-mutated numeric facts/private backing for an exact plan and B/mode. It does not permit sharing
[mutable](../../glossary.md#term-mutable) invocation size/access/packed/address/[carry](../../glossary.md#term-carry) arrays. This distinction changes no current
fresh `resolve` [API](../../glossary.md#term-application-programming-interface-api) or implementation. Size is nonnegative long bytes; alignment is fixed
power-of-two bytes from 1 through `1L<<62`. `byteSize(B)` validates B>=0 and computes exact checked
bytes. Recipes contain numeric facts only, with no value reads, providers, allocation or discovery.

`fixedByteSize()` is structural: present for FixedByteSize, absent for run-bound recipes even when
their result happens to be [constant](../../glossary.md#term-constant). Old long constructors wrap FixedByteSize. Old `byteSize()`
accessors return its exact bytes (including zero), but throw `IllegalStateException` for run-bound
declarations; no fake zero/-1/maximum size. Constructor/accessor source compatibility is retained,
not record reflection/[binary](../../glossary.md#term-binary-descriptor-and-metadata) compatibility. Creator single-abstract-method (SAM) consumers migrate
explicitly, including retained static/native routes; no unsafe zero-[argument](../../glossary.md#term-argument) dynamic adapter.

Execution mode is independent of byte independence. [Prepare](../../glossary.md#prepare) sets `requiresBatchSize=true` for
qualified symbolic projections, including all-constant-byte `X[B,0]` plans. Old two-list construction
means false and rejects run-bound declarations. `resolve` requires present B iff mode is true and
resolves every entry in plan order. Geometry authenticates the exact plan reference; foreign geometry
rejects. Absent-B runner/creation/state overloads reject true-mode plans before creators or [ownership](../../glossary.md#term-ownership-meanings)
acceptance, even if every byte count is constant. State-aware default hooks may delegate to old
static hooks only for false-mode plans, never by testing byte independence.

Creators allocate exactly the requested usable span with fixed alignment; allocator [padding](../../glossary.md#padding) is not
capacity. Every success returns fresh non-null physical objects even for zero bytes. Initialized
[buffers](../../glossary.md#term-buffer) fill the actual count. Preserve complete caller validation before creation, duplicate-result
rejection, origin/validity rules and reverse [rollback](../../glossary.md#term-rollback). [Backend](../../glossary.md#backend) rollback applies before return;
setup/run owns returned resources afterward. Allocation errors propagate and null results fail.
No mutable B, working numeric array, address, workspace or output mutates shared prepared state.
Only the explicitly bounded future [memo](../../glossary.md#cache-and-memoization) below may replace a metadata [snapshot](../../glossary.md#term-snapshot); prepared meaning,
recipes and persistent [resource ownership](../../glossary.md#term-resource-ownership) remain immutable.

## `modules/prepare`

`modules/prepare` owns shared [prepare](../../glossary.md#prepare) contracts and [validation](../../glossary.md#term-validation).

Allowed:

- [`PrepareContext`](../../glossary.md#prepare-context-preparecontext)
- [`BackendPartitionPreparer`](../../glossary.md#backend-partition-preparer--backendpartitionpreparer)
- [`BackendPartitionAnalysis`](../../glossary.md#backend-partition-analysis-backendpartitionanalysis)
- [backend](../../glossary.md#backend)-neutral [buffer](../../glossary.md#term-buffer) and workspace requirement declarations
- [`PreparedPartition`](../../glossary.md#prepared-partition--preparedpartition)
- shared assignment of stable [runtime](../../glossary.md#term-runtime) buffer and [workspace slot](../../glossary.md#workspace-slot--workspaceslot) [identities](../../glossary.md#term-identity-meanings)
- [partition](../../glossary.md#partition) coverage validation
- prepared memory validation
- [prepared schedule](../../glossary.md#prepared-schedule--preparedschedule) validation
- a future narrow orchestration boundary that exposes complete valid preparation candidates
  opaquely to model-autotuning tooling

Forbidden:

- concrete [CPU](../../glossary.md#term-cpu) [lowering](../../glossary.md#lowering)
- concrete [Metal](../../glossary.md#term-metal) lowering
- concrete [CUDA](../../glossary.md#term-cuda) lowering
- concrete [kernel](../../glossary.md#kernel) selection
- backend-specific executable implementation
- backend-specific storage implementation
- interpretation of private backend candidate [parameters](../../glossary.md#term-parameter)

Prepare is the bridge between [compile artifacts](../../glossary.md#compile-artifacts) and runtime. It projects the exact stable
semantic and [planning](../../glossary.md#term-planning) facts, resolved prepare-time [bindings](../../glossary.md#term-binding), [target](../../glossary.md#term-target) capabilities, configuration,
and compatible cached tuning decisions that backend analysis may consume. That projection must
not expose `CompileArtifacts` or another [compiler](../../glossary.md#term-compiler)-owned aggregate to a [concrete backend](../../glossary.md#concrete-backend).

For each planned partition, the concrete backend first analyzes and lowers the projected facts,
selects a supported route and configuration, and returns a `BackendPartitionAnalysis`. The
analysis retains the backend's selected lowering and route state opaquely while declaring every
shared buffer and workspace requirement exactly enough for shared preparation to assign stable
runtime slot identities. Shared preparation does not interpret the [opaque](../../glossary.md#term-opaque) backend plan or private
route vocabulary.

After shared preparation assigns slots, it assigns every participating backend [owner](../../glossary.md#term-owner) one stable
representation position for each declared logical buffer. The producing owner is first when one
exists; otherwise the first consuming owner is first. Remaining distinct consumer owners follow
logical consumer-partition order, and repeated partitions with the same owner reuse that owner's
position. The exact position is part of each buffer assignment supplied to the finalizer, so a
backend executable selects its own representation directly rather than assuming position zero.
The memory-plan buffer order begins with Compiler bindable-input IDs in exact caller [occurrence](../../glossary.md#term-occurrence)
order, including producerless published caller sources, independent of partition and backend
declaration order. Remaining ordinary buffers retain first-declaration order; producerless
published [constants](../../glossary.md#term-constant) remain last in final [graph](../../glossary.md#graph)-value order. The plan-ordered buffer association
retains the complete owner order, and each workspace association retains its exact partition owner.

The same backend then finalizes the analysis against those assignments and constructs the
[`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable) and `PreparedPartition`. Backend finalization must not change the selected
route or introduce undeclared shared requirements. It may acquire [immutable](../../glossary.md#term-immutable) persistent prepared
resources after assignment. A finalizer owns [rollback](../../glossary.md#term-rollback) before it returns; shared Prepare owns
returned resources transactionally until a completely validated [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) accepts
[ownership](../../glossary.md#term-ownership-meanings). Any intervening contribution, transfer-recipe, schedule-assembly, validation, or
aggregate-construction failure closes the acquired resources once in deterministic reverse order.
Finalizer results list resources in acquisition order; Prepare concatenates them in partition-
finalization order, rejects repeated exact resource identities, and never derives ownership from
executable, transfer, or schedule occurrences. Rollback preserves the preparation failure and
suppresses distinct resource-close failures in cleanup encounter order, skipping self-suppression.
Per-[run](../../glossary.md#term-run) physical allocation and binding remain runtime/backend concerns after preparation.

Backend analysis is deterministic from its explicit facts, configuration, and compatible [cache](../../glossary.md#term-cache)
inputs. An explicitly enabled later [model](../../glossary.md#term-model)-[autotuning](../../glossary.md#autotuning--model-autotuning) workflow may instead supply a selected
compatible decision before analysis; this [lifecycle](../../glossary.md#lifecycle) does not authorize prepare-time measurement
or search. Any unresolved fact needed to choose a route or declare an exact resource requirement
must fail preparation unless an explicit prepared contract represents that fact as run-dynamic
without changing route or slot assignment.

Concrete backend prepare implementations live in concrete backend [modules](../../glossary.md#term-module).

For accepted [general scan](../../glossary.md#general-differentiable-scan), PrepareContext/[PartitionDag](../../glossary.md#partition-dag-partitiondag) must preserve the validated CompiledScanBody
attachment, role/result bindings and exact demanded outer-port indices. Validate static local body
[descriptors](../../glossary.md#term-descriptor) in their own [scope](../../glossary.md#term-scope) separately from outer IDs; numeric ID coincidence must not [alias](../../glossary.md#aliasing-and-arena)
the descriptor domains. Prepare0010 implements this scoped projection and static binding/demand
validation; Prepare0011 extends the same owners to length-aware forward and actual generated
reverse scans. The separate non-[gradient](../../glossary.md#term-gradient) [INT64](../../glossary.md#data-type--datatype) lengths actual has no local formal or gradient
contribution. Declared row boundaries and tied actual positions are checked without re-proving
Model/Compiler body algebra or importing CPU read-[layout](../../glossary.md#layout) predicates. This is not backend execution. One selected
owner analyzes/prepares the complete body once; no child owner, body partitioning or new executable
aggregate is authorized. Ordinary explicit [residual](../../glossary.md#scan-residual-port-and-history-demand) output/input edges, not an opaque saved-
workspace handshake, express saved-value dependencies. The
[scan owner](recurrent-scan.md#canonical-residual-ports-and-demand) owns demand semantics.


Prepare0013 implements qualified ordinary symbolic projection alongside the retained static path.
Existing `prepare.analysis.PreparationResourceRequirement` variants expose these declarations:

```java
record Buffer(ValueId valueId, PreparedMemoryPlan.ByteSize size, long byteAlignment)
        implements PreparationResourceRequirement {
    public Buffer(ValueId valueId, long byteSize, long byteAlignment);
    public long byteSize();
}
record Workspace(long requirementId, PreparedMemoryPlan.ByteSize size, long byteAlignment)
        implements PreparationResourceRequirement {
    public Workspace(long requirementId, long byteSize, long byteAlignment);
    public long byteSize();
}
```

The same null/size/[alignment](../../glossary.md#term-alignment)/static-accessor rules as the Runtime resource contract apply;
workspace IDs remain nonnegative and analysis-local. `size()` retains the complete declaration.
Shared Prepare transports/compares structural declarations, alignment and coverage without
evaluating backend geometry. Qualified symbolic contexts set the explicit plan mode; fixed static
contexts retain their existing guards. Slot identities, requirement membership, representation
positions, scoped IDs, demand, routes and code remain fixed before finalization. All resources
must be declared as exact fixed bytes or complete immutable run-size recipes, never discovered
during binding. No new Prepare production type, backend [dependency](../../glossary.md#term-dependency) or geometry interpreter is added.

The accepted [symbolic Scan contract](recurrent-scan.md#accepted-symbolic-batch-scan)<span data-glossary-navigation="true"> ([symbolic Scan glossary](../../glossary.md#symbolic-scan))</span>
requires a bounded extension of the ordinary [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) projection guards, not blanket
symbolic/[rank](../../glossary.md#term-rank) [admission](../../glossary.md#term-admission). Validate all declared outer and local templates, every actual/formal/
result binding, [canonical](../../glossary.md#term-canonical-form) siblings and complete demand in their own scopes before analyzers.
Preserve named-[axis](../../glossary.md#term-axis) obligations and qualified rank-three outer [AD](../../glossary.md#autograd) closure without importing CPU
read predicates or re-proving the body's semantic algebra. Set `requiresBatchSize` even for
constant-byte symbolic plans. Shared Prepare transports exact fixed/run-size declarations and
stable coverage/slots; it does not evaluate CPU recipes, choose B-specific [topology](../../glossary.md#term-topology) or infer
[Scan](../../glossary.md#term-cumulative-scan) capability. Prepare0014 implements this symbolic projection and the separately qualified
[homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) CPU/[Engine](../../glossary.md#term-engine) execution [slice](../../glossary.md#slice); its
[task](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md) tracks final [qualification](../../glossary.md#term-qualification).
Static Scan projection remains unchanged.

Prepare0015 implements length projection with static or one named B, fixed T/[features](../../glossary.md#term-feature) and homogeneous
FLOAT32/[FLOAT64](../../glossary.md#term-float64) locals. The separate non-gradient INT64 `[B]`
actual has no formal. Apply the same complete pre-analyzer outer/local declaration, positional
binding, latent-port, role and demand validation to forward and actual generated mixed-role reverse;
do not substitute numeric size coincidence for named-axis [identity](../../glossary.md#term-identity-meanings). Static canonical guards and
unresolved symbolic templates remain distinct. This capability extends support through these
existing owners, not the [handoff](../../glossary.md#term-handoff) or semantic row algebra.

### Producerless published values

The reviewed complete-[graph](../../glossary.md#graph) source-[publication](../../glossary.md#term-publication) cutover renames the existing
`ProducerlessPublishedConstantResource` [owner](../../glossary.md#term-owner) to [`ProducerlessPublishedValueResource`](../../glossary.md#producerless-published-value-resource--producerlesspublishedvalueresource).
Its five [canonical](../../glossary.md#term-canonical-form) components are `GraphValue value`, `LogicalMemoryRequirement logicalRequirement`,
`BackendId owner`, `PreparedMemoryPlan.ByteSize size`, and `long byteAlignment`. Migrate all
callers, docs and inventory entries; remove the old name/hook without an [alias](../../glossary.md#aliasing-and-arena) or compatibility
overload. The existing three-/four-[argument](../../glossary.md#term-argument) [GraphPreparation](../../glossary.md#graph-preparation--graphpreparation) forms retain their roles.

The migrated default assembler geometry hook is
`producerlessPublishedValue(GraphValue, LogicalMemoryRequirement, Optional<ScalarValue>)`.
The [scalar](../../glossary.md#term-scalar) is present exactly for an authenticated [constant](../../glossary.md#term-constant) source, empty exactly for a caller.
It preserves existing constant [dtype](../../glossary.md#data-type--datatype)/canonical-[layout](../../glossary.md#layout) policy; initialization still belongs to
the existing per-[run](../../glossary.md#term-run) constant initializer. Caller sources still use the existing [borrowed](../../glossary.md#ownership-borrowing-and-lease)
CallerInput path. The contribution reads no [payload](../../glossary.md#artifact-envelope-and-atomic-publication), allocates/acquires nothing and creates no
[operation](../../glossary.md#operation), [partition](../../glossary.md#partition), representation or executable.

The audit-driven [Metal](../../glossary.md#term-metal) completion extends this same protocol to Metal-only registrations,
including constant cotangents and source-only callers. [Engine](../../glossary.md#term-engine)'s source-owner selection and
composite assembler must agree on a registered supporting [backend](../../glossary.md#backend) without requiring a [CPU](../../glossary.md#term-cpu)
adapter. Preserve existing CPU behavior and deterministic registration/[ownership](../../glossary.md#term-ownership-meanings) policy; the
chosen backend supplies exact physical geometry and the existing initialization/import/export
path. Do not synthesize a [Tensor](../../glossary.md#tensor) operation or partition, install a CPU fallback, introduce a
second initializer, or weaken source [authentication](../../glossary.md#term-authentication) and pre-finalization [validation](../../glossary.md#term-validation).

One shared discovery/validation path requires an actual graph input that is published without
[producer](../../glossary.md#term-producer) or consumer, classified disjointly by the existing constant/caller plans. Preserve exact
graph-value/logical-requirement references, canonical complete coverage and duplicate/foreign/
extra/missing/overlap rejection before finalization. Reuse/factor the existing fixed-or-sole-named-B
vocabulary and common-[symbol](../../glossary.md#term-symbol) proof for complete source values; do not copy another [guard](../../glossary.md#term-guard).
Constants retain static/resolved/nongradient/type policy. Static caller [views](../../glossary.md#term-view-meanings) retain exact
resolved origin/[strides](../../glossary.md#term-stride); unresolved static caller layouts remain unsupported. Named templates
retain their existing unresolved declaration and canonical zero-offset actual-input contract.

Index ordinary declarations and producerless contributions before slot assignment. Preseed
both ordinary and source-only callers in exact [Compiler](../../glossary.md#term-compiler) bindable [occurrence](../../glossary.md#term-occurrence) order, then assign
actual partition requirements unchanged. Source-only callers receive no partition-finalizer
requirement. Append source-only constants in final graph-value order. Retain the supplied
ByteSize recipe, including structural symbolic zero-byte cases; no [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) B, fake fixed size,
maximum geometry or payload-derived declaration is allowed.

Determine complete physical [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) mode from source/complete-graph named facts as well as
partition contexts. A genuinely fixed CPU unit inside that complete batch-required plan keeps
its static [lowering](../../glossary.md#lowering)/[artifact](../../glossary.md#term-artifact). Existing CPU state-aware cold hooks authenticate fixed selected
declarations, exact plan-authenticated resolved geometry and retained unit facts, plus unchanged
dtype/span/[alignment](../../glossary.md#term-alignment)/writability/[lifetime](../../glossary.md#term-lifetime) guards. Nonfixed selections require their genuine
dynamic route or reject. [Runtime](../../glossary.md#term-runtime)'s default static guards remain unchanged; no new flag, field,
per-B [topology](../../glossary.md#term-topology), relowering or generated [ABI](../../glossary.md#term-application-binary-interface-abi) follows.

Static geometry reuses [Model](../../glossary.md#term-model) [referenced span](../../glossary.md#referenced-element-span), including empty-span semantics; named geometry
reuses existing CPU access/[binding](../../glossary.md#term-binding) proofs and checked actual-B sizing. Raw source storage and
publication do not invoke arithmetic or unlock CPU low arithmetic. Existing ownership, [rollback](../../glossary.md#term-rollback),
creators, publication adapters and nonempty-partition policies remain unchanged. In particular,
this cutover does not add no-[node](../../glossary.md#node) Engine/[Prepare](../../glossary.md#prepare) support or a new standalone CPU assembly policy.

### [General Scan](../../glossary.md#general-differentiable-scan) activity projection

Implemented coordinated Class C projection, accepted with Compiler0017/CPU0081 after independent
authority/interface and exact-source implementation reviews and actual complete-preparation/
[Engine](../../glossary.md#term-engine) consumers. The [Scan owner](recurrent-scan.md#general-ordinary-unit-scan-cutover) owns
semantics; [Planning](foundational-modules.md#general-scan-activity-transport)<span data-glossary-navigation="true"> ([Planning glossary](../../glossary.md#term-planning))</span> owns the neutral
[immutable](../../glossary.md#term-immutable) [carrier](../../glossary.md#term-storage-carrier). Shared [Prepare](../../glossary.md#prepare) does not interpret [derivative](../../glossary.md#term-derivative) mathematics or [CPU](../../glossary.md#term-cpu) pack [layouts](../../glossary.md#layout).

[PrepareContext](../../glossary.md#prepare-context-preparecontext) adds sixth [canonical](../../glossary.md#term-canonical-form) component `Map<NodeId, ScanActivityPlan> scanActivityPlans`
after backendInputs. [GraphPreparation](../../glossary.md#graph-preparation--graphpreparation) projects only containing [Scan](../../glossary.md#term-cumulative-scan) entries from component eight
of CompileArtifacts. Context copies/subsets preserve/filter exact associations; intentional
ordinary/synthetic construction has an empty map, never a silent drop of annotated [metadata](../../glossary.md#term-metadata).
Validate [partition](../../glossary.md#partition) membership, exact local-body [graph](../../glossary.md#graph) [identity](../../glossary.md#term-identity-meanings), complete typed declarations,
positional boundaries, roots/scopes/merge/[accumulator](../../glossary.md#accumulator-type) applicability and complete demand BEFORE
[backend](../../glossary.md#backend) analyzers. No first-[carry](../../glossary.md#term-carry) [dtype](../../glossary.md#data-type--datatype), nonempty/positive-ROW or obsolete [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix)-history
shortcut replaces those checks. Independent ordinary and symbolic named-[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) guards remain.
Retain the Scan [owner](../../glossary.md#term-owner)'s local materialized-boundary meaning: do not infer incoming absence from
numeric values, [producer](../../glossary.md#term-producer) [provenance](../../glossary.md#provenance) or shared length storage. No cross-partition activity
communication field or additional [resource owner](../../glossary.md#ownership-borrowing-and-lease) is introduced.
Before backend analysis, use the scoped four-[argument](../../glossary.md#term-argument) [Model](../../glossary.md#term-model) ordinary [guard](../../glossary.md#term-guard) with the exact sole
declared optional [symbol](../../glossary.md#term-symbol), then check all declared canonical siblings. Do not choose a [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) B,
ignore extra/foreign symbols, copy symbolic family rules or impose length ROW locality on
all-valid whole-[Tensor](../../glossary.md#tensor) reshape. Ordinary three-argument [inference](../../glossary.md#term-inference) may defer these equalities.

For the qualified CPU continuation, the single backend owner may construct an internal context
from the EXACT contained graph, derive its local [DAG](../../glossary.md#term-directed-acyclic-graph-dag)/logical requirements and reuse its ordinary
physical [decomposition](../../glossary.md#term-decomposition). This explicitly supersedes the older body-partitioning prohibition only
for backend-internal physical units. No local IDs [alias](../../glossary.md#aliasing-and-arena) outer requirements/[constants](../../glossary.md#term-constant), no semantic
child owner or new executable aggregate is created, and no [Compiler](../../glossary.md#term-compiler) aggregate crosses to CPU.
Zero-[node](../../glossary.md#node) passthrough bodies require no forged partition node. Ordinary explicit [residual](../../glossary.md#scan-residual-port-and-history-demand) edges
remain the saved-value [handoff](../../glossary.md#term-handoff); compiled activity is neither persisted source nor an [opaque](../../glossary.md#term-opaque) tape.

Legitimate inferred computed body values may retain unresolved logical layouts. The backend's
local ordinary projection completes only those computed [descriptors](../../glossary.md#term-descriptor) through the shared Model
ordinary-layout closure, updating matching logical-memory descriptors before ordinary analysis.
The captured graph, exact nodes/IDs/order/phases and activity graph association do not change.
Actual and formal layout guards stay strict; no unresolved ingress becomes canonical by default.
Both capability/decomposition and helper resource preparation consume the same projection path.
Shared Prepare still validates semantic declarations; it gains no CPU layout algorithm,
additional context field or physical allocation responsibility.

Body cuts, helper scratch and numeric control use existing [buffer](../../glossary.md#term-buffer)/workspace declarations and
[RunState](../../glossary.md#run-state--runstate) [ownership](../../glossary.md#term-ownership-meanings). Fixed or checked symbolic [run](../../glossary.md#term-run)-size recipes declare all storage before slots;
no hidden [Arena](../../glossary.md#term-arena) or [runtime](../../glossary.md#term-runtime)-discovered route/[topology](../../glossary.md#term-topology). Invocation-owned [mutable](../../glossary.md#term-mutable) packs/cursors
never enter reusable prepared state. Runtime/Engine [lifecycle](../../glossary.md#lifecycle), [lease](../../glossary.md#term-lease)/cleanup and execution
protocols remain unchanged.

## [Prepare](../../glossary.md#prepare) [lifecycle](../../glossary.md#lifecycle)

The implemented [CPU0074 reservation exception](#cpu0074-run-owned-synchronous-reservation)
adds only [backend](../../glossary.md#backend)-declared selected-region staging/partials and a [run](../../glossary.md#term-run)-bound [CPU](../../glossary.md#term-cpu) context; the
shared analysis/assignment/finalization [handoff](../../glossary.md#term-handoff) is unchanged.

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

- [`BackendPartitionAnalysis`](../../glossary.md#backend-partition-analysis-backendpartitionanalysis)
- [`PreparedPartition`](../../glossary.md#prepared-partition--preparedpartition)
- [`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable)
- [`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan)
- [`PreparedSchedule`](../../glossary.md#prepared-schedule--preparedschedule)
- [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution)

[Concrete backend](../../glossary.md#concrete-backend) [lowering](../../glossary.md#lowering) occurs in concrete backend [modules](../../glossary.md#term-module).
Route selection and shared-resource discovery occur during backend analysis. Executable
construction occurs only during backend finalization after shared slot and representation-position
assignment. Each participating backend contributes exact representation creators for its assigned
[buffer](../../glossary.md#term-buffer) positions and workspace creators for its [partition](../../glossary.md#partition)-owned workspace positions. Contributions
are physical creation recipes, not an alternative schedule. For the implemented bounded symbolic-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)
path, exactness means fixed bytes or a complete run-size declaration with fixed [alignment](../../glossary.md#term-alignment);
Prepare validates mode/declarations/coverage without evaluating numeric recipes. Current static
analysis/finalization remains supported. Shared preparation validates complete,
non-overlapping contribution coverage, assembles and validates ordered schedule-step occurrences,
then constructs the [Runtime](../../glossary.md#term-runtime) `PreparedExecution` with the exact assigned memory plan, exact
schedule, and acquisition-ordered persistent resources. Successful construction transfers unique
[ownership](../../glossary.md#term-ownership-meanings) of those resources to that Runtime aggregate.

General-[scan](../../glossary.md#term-cumulative-scan) analysis declares checked exact output, staging, [carry](../../glossary.md#term-carry)-bank, body-temporary and live
history bytes before slot assignment, with zero history bytes/stores for undemanded [residual](../../glossary.md#scan-residual-port-and-history-demand) ports
including reverse residuals. Revalidate demand against the projected roles and logical requirements.
Only finalization after exact slot/representation assignment may emit the selected CPU time loop
and typed body code. It may not discover new shared storage requirements during emission. Concrete
lowering stays in the backend; local [descriptors](../../glossary.md#term-descriptor) never become outer logical slot IDs by coincidence.

Prepare0011 implements static length-mode projection through this lifecycle. It preserves ordered
ROW(0), [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1) and SHARED roles, exact scoped bodies and the extra [INT64](../../glossary.md#data-type--datatype) actual without
a body formal. All partition contexts, including unused local declarations and checked logical
geometry, validate before the first analyzer; complete-[graph](../../glossary.md#graph) demand includes every outer consumer.
Concrete analysis owns exact mixed-boundary buffer/workspace geometry and declares it before slots.
Shared assignment, finalization, [rollback](../../glossary.md#term-rollback) and Runtime ownership/[RunState](../../glossary.md#run-state--runstate) isolation are unchanged.
No length value is read, and [metadata](../../glossary.md#term-metadata)/resource-handoff tests allocate no physical history or run
a body. CPU0042 separately implements fixed-geometry [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) forward and actual reverse:
backend analysis declares mixed-width resources before slots, and the existing scan invocation
checks values after [producers](../../glossary.md#term-producer) before generated skipping. No shared Prepare or Runtime handoff changes.
The [scan owner](recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span> retains its semantic [scope](../../glossary.md#term-scope).

Prepare0015's implemented combined path uses this same sequence with exact fixed/run-size
declarations before slots, explicit `requiresBatchSize` even for [constant](../../glossary.md#term-constant)-byte `[B,0]`, and
[dtype](../../glossary.md#data-type--datatype)-width-correct lengths, outputs, banks, cuts, temporaries and demanded histories. Declared
zero-byte workspaces retain slots/creators. Shared Prepare still transports declarations without
evaluating CPU recipes or reading lengths.


## [Run](../../glossary.md#term-run) [lifecycle](../../glossary.md#lifecycle)

Only the implemented [CPU0074 reservation exception](#cpu0074-run-owned-synchronous-reservation)
qualifies the no-per-step-allocation statement below for existing [synchronous](../../glossary.md#term-synchronous) scheduler [metadata](../../glossary.md#term-metadata).
It adds no [Runtime](../../glossary.md#term-runtime) protocol, public async contract, extra [lease](../../glossary.md#term-lease) [owner](../../glossary.md#term-owner) or [RunState](../../glossary.md#run-state--runstate).

[General scan](../../glossary.md#general-differentiable-scan)'s [mutable](../../glossary.md#term-mutable) [carry](../../glossary.md#term-carry), changing addresses and geometry arrays, temporaries and histories
remain run-owned under this lifecycle, never mutable fields of shared prepared state. Existing
[ownership](../../glossary.md#term-ownership-meanings), overlap [validation](../../glossary.md#term-validation), [rollback](../../glossary.md#term-rollback), retry and concurrent-run isolation apply; no runtime tape,
per-step allocation or extra RunState is introduced. Runtime never inspects body [graph](../../glossary.md#graph) metadata.

For the implemented bounded symbolic-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) path, [Engine](../../glossary.md#term-engine) resolves semantic and checked numeric/size geometry
before [borrowing](../../glossary.md#term-borrowing); Runtime still acquires its unique prepared lease before inspecting supplied
inputs/geometry. Actual-byte validation and every mutable packed/access/size array are run-owned.
Under the accepted future [memo exception](#accepted-engine-geometry-memo), [immutable](../../glossary.md#term-immutable) exact-plan
numeric facts may be shared, not those working arrays or any resource. Runtime performs no [memo](../../glossary.md#cache-and-memoization)
[lookup](../../glossary.md#term-lookup)/[publication](../../glossary.md#term-publication); the exception adds no lease, execution lock or late outward-close rejection.
[Shape](../../glossary.md#shape)/byte [admission](../../glossary.md#term-admission) never reads lengths: length VALUE checks remain after their [producers](../../glossary.md#term-producer) and
before that [Scan](../../glossary.md#term-cumulative-scan)'s first writes, including all B lengths and T=0/B=0. CPU0042 implements those
checks inside the existing [CPU](../../glossary.md#term-cpu) scan invocation for fixed T/B/[features](../../glossary.md#term-feature) and [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64)
forward/actual reverse. All-valid [symbolic Scan](../../glossary.md#symbolic-scan) retains its separate [qualification](../../glossary.md#term-qualification). Prepare0015's
implemented one-B length-Scan path follows the same sequence: every call performs fresh
complete [Compiler](../../glossary.md#term-compiler) [binding](../../glossary.md#term-binding)/private obligations, current input-backed publication/storage admission
and checked numeric sizing before borrow, then run-aware typed span/access/[alignment](../../glossary.md#term-alignment)/overlap and
all actual B length checks after producers before the first Scan-local write. B is explicit even
with zero features; no length-based shapes or cached admission are allowed. Current combined
execution uses the existing CPU invocation. Runtime receives no new graph metadata, protocol or
execution owner.

Ordinary Engine calls admit the Engine, validate the prepared handle's non-nullness and exact
owner, retrieve/check its exact inward delegate, then inspect inputs. Closed Engine wins all
[arguments](../../glossary.md#term-argument); under an open Engine, closed preparation reports `prepared execution is closed` before
null, malformed or dead inputs and without borrowing. Runtime alone arbitrates a later close/run
[race](../../glossary.md#term-race) through its lease below; no outward lock is held through execution.

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

Successful return is the completion barrier for that invocation: every schedule action and native
[operation](../../glossary.md#operation) has completed, and the published destinations are usable. A failed invocation returns no
[`RunResult`](../../glossary.md#run-result--runresult). The returned result is a lease over completed run-state resources and publication
[lifetime](../../glossary.md#term-lifetime), not a future, event, fence, or in-flight command handle. It may remain open after the run
call; its later close [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) run-owned resources independently of the already-released prepared-
execution lease. [Materialization](../../glossary.md#materialization) and transfers retain their own synchronous call boundaries.

Closing a RunResult releases publication-representation references even when RunState cleanup
fails. RunState releases [borrowed](../../glossary.md#ownership-borrowing-and-lease)/owned binding and workspace reference cells after their required
cleanup handling, without closing borrowed representations or changing reverse cleanup, unique
ownership, closed-access precedence or post-close plan/count/geometry inspection. Array [dimensions](../../glossary.md#dimension)
remain available; these Runtime objects remain non-[thread-safe](../../glossary.md#term-thread-safe) and their existing repeated-close
behavior is unchanged. This releases direct owner references, not externally retained bound objects
or required retained cleanup throwables.

The Runtime surface has no [asynchronous](../../glossary.md#term-asynchronous) run, [cancellation](../../glossary.md#term-cancellation), timeout, polling, [callback](../../glossary.md#term-callback), or explicit
completion-event contract. Concurrent calls create isolated state but make no promise of [device](../../glossary.md#term-device)
overlap, submission/completion order, fairness, queue behavior, or [throughput](../../glossary.md#term-throughput). Schedule order is
defined within one invocation only.

One logical value has at most one representation per participating [backend](../../glossary.md#backend) owner. A transfer
copies from the producing owner's representation, or from the first consuming owner's caller-input
representation when the value has no [producer](../../glossary.md#term-producer), to one distinct consuming owner's representation.
[Fan-out](../../glossary.md#term-fan-out) creates one immutable transfer recipe per distinct destination owner and schedules it once
before that owner's first consumer. The source remains valid. Runtime marks the destination valid
only after successful physical transfer; failure leaves source and destination validity unchanged
and triggers ordinary reverse run cleanup with the transfer failure primary.

[`PreparedExecutionRunner`](../../glossary.md#prepared-execution-runner--preparedexecutionrunner) owns this synchronous orchestration. It is stateless and creates one
isolated `RunState` per call; concurrent and repeated calls may share immutable prepared physical
state only through fresh backend-authenticated read-only child bindings. They never share mutable
state, writable internal/output [buffers](../../glossary.md#term-buffer), or workspaces. [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) remains the unique
owner of each prepared-resource [identity](../../glossary.md#term-identity-meanings); a run owns only its issued child binding. Concrete
transfer recipes retain exact direction, byte geometry, and direct backend-owned cold binders.
Cold binding validates concrete representations once and creates an action with direct typed
source, destination, and staging references. Executable and transfer execution, and Runtime
publication binding, perform no Engine, [registry](../../glossary.md#term-registry), backend-ID, provider, availability,
configuration, discovery, reflection, or representation-map lookup.

Run must not perform:

- [graph optimization](../../glossary.md#forward-graph-optimization)
- [autograd](../../glossary.md#autograd) construction
- backend discovery
- [kernel](../../glossary.md#kernel) selection
- backend-specific [lowering](../../glossary.md#lowering)
- compiler passes

The initial runtime resource [model](../../glossary.md#term-model) introduces no automatic [pooling](../../glossary.md#term-pooling), reuse, [aliasing](../../glossary.md#term-aliasing), distributed
sharding, hidden mutation/coherence protocol, or multi-device scheduling. Multiple physical
representations exist only when explicitly required by a [prepared schedule](../../glossary.md#prepared-schedule--preparedschedule), and immutable
functional value semantics do not imply hidden write-back between them.


### CPU0074 [run](../../glossary.md#term-run)-owned [synchronous](../../glossary.md#term-synchronous) reservation

Implemented, independently reviewed and MAIN accepted, 2026-10-30. CPU0074 is Complete;
the [task Result](../../planning/backends/cpu/tasks/0074-row-fusion-simd-parallel.md#result)
records bounded [validation](../../glossary.md#term-validation) and performance limitations. The [CPU mechanism owner](backend-execution.md#cpu0074-selected-row-region-execution-exception)
defines exact internal seams and eligibility; the [Scan owner](recurrent-scan.md#cpu0074-selected-terminal-grouping-exception)
defines arithmetic/activity. This subsection owns only [Prepare](../../glossary.md#prepare)/Run resource and budget [lifetime](../../glossary.md#term-lifetime).

[Backend](../../glossary.md#backend) analysis predeclares every staging/partial byte and fixed frame/[layout](../../glossary.md#layout) fact before
shared slot assignment. Finalization binds assigned storage and explicit generated helpers,
never discovers requirements. One [RunState](../../glossary.md#run-state--runstate) owns fresh StepDispatch, operand/frame arrays,
prebound range calls and all [mutable](../../glossary.md#term-mutable) scratch. Prepared state retains [immutable](../../glossary.md#term-immutable) recipes/handles
and [borrowed](../../glossary.md#ownership-borrowing-and-lease) [CPU](../../glossary.md#term-cpu) group/budget references only. Ordinary [Runtime](../../glossary.md#term-runtime) cold [binding](../../glossary.md#term-binding) still precedes
schedule execution; [producer](../../glossary.md#term-producer)-sensitive value [preflight](../../glossary.md#term-preflight) still follows producers before writes.

For a selected parallel [Scan](../../glossary.md#term-cumulative-scan), cold run binding computes actual participant count N from the
fixed prepared numeric [partition](../../glossary.md#partition) rule. Acquire max(1,N) permits from the exact coordinated
finalizer budget before the generated entry. Caller setup/merge/[carry](../../glossary.md#term-carry) uses this same reservation.
At synchronous [dispatch](../../glossary.md#term-dispatch), lend it exclusively to the prebound worker submission: validate exact
budget [identity](../../glossary.md#term-identity-meanings), unreleased state and sufficient demand before enqueue; acquire/[release](../../glossary.md#term-release) no
additional permits there. The caller performs no concurrent math while waiting. N<2 runs inline
under one permit and submits nothing. A parallel selection requires the coordinated group/budget
pair; legacy uncoordinated ordinary calls keep their existing behavior.

The reservation is bound privately to this invocation for one synchronous call and must not be
shared with another RunState or concurrently borrowed. Only after all started workers quiesce
may it be unbound/released or may the caller consume partials, change carry/time or reuse scratch.
Normal merge/carry runs under the still-live reservation. Existing worker failure ordering,
unclaimed-work [cancellation](../../glossary.md#term-cancellation), interruption/status restoration, racing close and synchronous join
apply unchanged; workers cannot submit nested work. Reject wrong-budget, released or insufficient
reservations before publishing work. Success/failure cleanup must release the outer reservation
exactly once after [quiescence](../../glossary.md#term-quiescence). This adds neither a second execution [owner](../../glossary.md#term-owner) nor a shared run lock.

The emitted outer Scan loop may invoke only the generation-selected typed StepDispatch seam.
Run-bound calls/arrays are allocated once and reused across T. Existing calls.clone/Job/failure/
[latch](../../glossary.md#control-flow-graph)/queue [metadata](../../glossary.md#term-metadata) per synchronous dispatch is the narrow per-step allocation exception, not
body/[tensor](../../glossary.md#tensor)/[element](../../glossary.md#term-element) allocation, [graph](../../glossary.md#graph) interpretation, per-op [lookup](../../glossary.md#term-lookup) or geometry repacking.
Job reuse is optional, not required. A new pool, generic phase/cohort framework, Runtime service
and async protocol are not authorized. CPU0074 does not implement Job reuse or depend on it.
Whole-run completion/[publication](../../glossary.md#term-publication) remains synchronous.

Worker segments, workspace and any staged [transpose](../../glossary.md#term-transpose) must be accessible to every participating
worker and pass existing typed spans/[alignment](../../glossary.md#term-alignment)/[alias](../../glossary.md#aliasing-and-arena) checks before writes. Stage invariants only
inside the required activity gate; changing cuts/SHARED histories use their exact active-time
binding and cannot survive beyond proved lifetimes. Join precedes consumers and scratch reuse.
No B×[parameter](../../glossary.md#term-parameter) bank, maximum-B allocation or T-fold graph/code/storage follows. Unselected [scalar](../../glossary.md#term-scalar)
paths retain existing [admission](../../glossary.md#term-admission), resources and [lifecycle](../../glossary.md#lifecycle). MAIN acceptance changes no production default.

## `modules/engine`

`modules/engine` owns public [lifecycle](../../glossary.md#lifecycle) orchestration and explicit [backend composition](../../glossary.md#backend-composition).

Allowed:

- public [`Engine`](../../glossary.md#term-engine) [facade](../../glossary.md#term-facade)
- public `CompiledGraph` facade
- [owner](../../glossary.md#term-owner)-bound public [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) and [`RunResult`](../../glossary.md#run-result--runresult) handles
- positional [fresh-value binding](../../glossary.md#fresh-value-binding--enginerunvalues) through [`Engine.runValues`](../../glossary.md#fresh-value-binding--enginerunvalues) on the existing prepared handle
- [compile](../../glossary.md#compile) orchestration
- [prepare](../../glossary.md#prepare) orchestration
- [synchronous](../../glossary.md#term-synchronous) [run](../../glossary.md#term-run) orchestration
- an Engine-owned composition builder and private [backend](../../glossary.md#backend) [registry](../../glossary.md#term-registry)
- wiring [compiler](../../glossary.md#term-compiler), [runtime](../../glossary.md#term-runtime), Prepare, and concrete backends

Forbidden:

- [kernel](../../glossary.md#kernel) implementations
- backend internals or native-configuration parsing
- [graph](../../glossary.md#graph) [optimizer](../../glossary.md#term-optimizer) passes
- runtime [service locator](../../glossary.md#term-service-locator)
- reflective [plugin](../../glossary.md#term-plugin) discovery as the core backend mechanism
- a process-global or otherwise shared [mutable](../../glossary.md#term-mutable) backend registry

Engine is the [composition root](../../glossary.md#term-composition-root). [Concrete backend](../../glossary.md#concrete-backend) [modules](../../glossary.md#term-module) expose backend-owned integrations and
configuration without depending on Engine; Engine supplies the outward adapters and registry that
compose them.

### Public explicit composition

The selected public declaration [shape](../../glossary.md#shape) is:

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
it terminally, and explicit close is [idempotent](../../glossary.md#term-idempotent). Later non-null `takeOwnership` calls still take
and immediately close their [argument](../../glossary.md#term-argument) before rejecting use of a spent or closed builder, so a
caller never regains [ownership](../../glossary.md#term-ownership-meanings) implicitly.

`Engine.Builder` is an explicit, single-use, `AutoCloseable` construction [owner](../../glossary.md#term-owner). It has concrete
`takeOwnership(CpuBackendIntegration)` and `takeOwnership(MetalBackendIntegration)` overloads
rather than a [backend](../../glossary.md#backend)-implemented [Engine](../../glossary.md#term-engine) interface. A later built-in backend requires another
explicit Engine-owned adapter and overload. This keeps the [dependency direction](../../glossary.md#java-platform-and-build-tools) from Engine to
backends and prevents a backend [module](../../glossary.md#term-module) from depending on Engine.

Because those backend-owned types appear in public Engine [signatures](../../glossary.md#term-signature), the implementation must
publish the [CPU](../../glossary.md#term-cpu) and [Metal](../../glossary.md#term-metal) module dependencies to Engine [API](../../glossary.md#term-application-programming-interface-api) consumers. That intentional outward
visibility is limited to explicit construction; it does not permit either backend to depend on
Engine or leak backend types into [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare), [run](../../glossary.md#term-run), handle, or result signatures.

Current supported Engine [distribution](../../glossary.md#term-distribution) composes CPU and Metal, not the unused [CUDA](../../glossary.md#term-cuda) marker.
Keep CUDA's source/project registration for future development, without a required Engine
[dependency](../../glossary.md#term-dependency) or supported [publication](../../glossary.md#term-publication). Export existing inward contracts named by supported public
signatures; keep implementation-only collaborators, including Tuning runners, private.

For a non-null argument, ownership transfers at entry to `takeOwnership`, before Engine
interrogates the integration. The builder therefore closes the supplied integration if [identity](../../glossary.md#term-identity-meanings),
capability, availability, or duplicate [validation](../../glossary.md#term-validation) fails; the caller must not close or reuse it
after calling the method, whether the method succeeds or fails. A null argument transfers
nothing. After `build()` succeeds, ownership of the complete ordered [registry](../../glossary.md#term-registry) moves atomically to
the returned Engine and closing the spent builder is a no-op. Before successful build, closing the
builder closes accepted integrations in reverse registration order.

Each successful registration [captures](../../glossary.md#term-capture-meanings) exactly one [`BackendId`](../../glossary.md#backend-identity--backendid), one stable capability-provider
reference, and one [immutable](../../glossary.md#term-immutable) [`BackendAvailabilitySnapshot`](../../glossary.md#backend-availability-snapshot--backendavailabilitysnapshot). All three identities must name equal
backend IDs. The [snapshot](../../glossary.md#term-snapshot) is taken at registration time and is fixed for the [lifetime](../../glossary.md#term-lifetime) of the
built Engine; compiling never refreshes availability. Capability queries use the retained
provider, which must remain deterministic for its unchanged backend configuration. To observe a
different availability or configuration, construct a new Engine.

The registry preserves registration order and rejects an equal `BackendId` without replacing or
using the previously accepted integration. Equality is `BackendId.equals`, not object [identity](../../glossary.md#term-reference-identity) or
normalized text. The previous entry remains owned and unchanged; the rejected newly transferred
integration is closed. The registry is private Engine construction state, not a public [lookup](../../glossary.md#term-lookup) API,
global registry, [service locator](../../glossary.md#term-service-locator), or discovery mechanism.

For Metal, the equal-`BackendId` rule means one Engine can own at most one
`MetalBackendIntegration`. That integration has one system-default-[device context](../../glossary.md#device-submission-synchronization-and-execution-context) and queue; its
`BackendDeviceId(metal, "default")` snapshot entry is an abstract eligibility slot, not a retained
physical-[device](../../glossary.md#term-device) selection. [Planning](../../glossary.md#term-planning) retains the owning `BackendId`. Resources stay authenticated
to the integration's exact context. Separate Engines may open separate contexts, but they are not
coordinated multi-device execution and may resolve to the same default device.

`Engine.standard()` remains the ordinary CPU convenience. Every call opens one fresh default CPU
integration and transfers it through the same builder ownership path. It performs no discovery
and neither enables Metal nor changes the existing CPU [lifecycle](../../glossary.md#lifecycle) behavior.

`AdvancedEngine.takeOwnership(CpuBackendIntegration)` remains the current CPU-only lower-level
surface; the builder decision neither removes it nor makes it generic.

`MetalBackendConfiguration` and `MetalBackendIntegration` belong to `modules/backend/metal`.
`MetalBackendIntegration.open(configuration)` validates and snapshots the backend-owned
configuration and acquires native ownership before it can be transferred to Engine. Engine
neither owns a duplicate Metal configuration type nor reads a library path, device selector, or
other Metal-private option.

### Construction failure and closure

A `takeOwnership` failure remains primary and closes the newly transferred candidate; integrations
accepted by earlier calls remain registered and owned by the builder. A `build()` failure remains
primary, terminally closes the builder, and attempts every accepted integration in reverse
registration order. Later distinct close failures are suppressed on the primary in cleanup
encounter order; an identical throwable reference is skipped to avoid self-suppression. [Engine](../../glossary.md#term-engine)
construction publishes no partially built Engine.

An explicit builder or Engine close with no earlier primary failure attempts all owned cleanup in
the required reverse order. Its first unchecked failure or error is primary; later distinct
failures are suppressed in cleanup encounter order, and repeated references to the primary are
skipped. Repeated close is [idempotent](../../glossary.md#term-idempotent) and replays the retained primary failure.

An Engine owns its [registry](../../glossary.md#term-registry) until Engine close. Shutdown first rejects new operations and waits
for admitted Engine operations, then closes retained results in reverse [run](../../glossary.md#term-run)-[publication](../../glossary.md#term-publication) order,
ordinary and advanced prepared handles in reverse preparation-publication order,
and [backend](../../glossary.md#backend) integrations in reverse registration order. Cleanup is attempt-all, idempotent, and
failure-retaining. Closing a prepared handle closes its one inward [Runtime](../../glossary.md#term-runtime)
execution without closing an already returned result. A run that already acquired Runtime's
prepared [lease](../../glossary.md#term-lease) may finish without making close wait; its final lease [release](../../glossary.md#term-release) performs deferred
prepared-resource cleanup.

#### [Reentrant](../../glossary.md#memory-management-atomic-counters-and-complexity) [Engine](../../glossary.md#term-engine) close and [lease](../../glossary.md#term-lease) [admission](../../glossary.md#term-admission)

The [Engine0023 correction](../../planning/modules/engine/tasks/0023-reentrant-close-and-lease-admission.md)
is implemented in the existing [owners](../../glossary.md#term-owner) and source-qualified. A [synchronous](../../glossary.md#term-synchronous) close by a thread holding
admission to that exact Engine must throw IllegalStateException before changing [lifecycle](../../glossary.md#lifecycle) or waiting, even if another close is
already pending. The thread performing Engine cleanup may reenter close idempotently; other
close callers retain completion waiting, interruption restoration and retained failure replay.
Callbacks must not synchronously wait for another thread to close the Engine whose work they serve.

The existing Engine lifecycle [owner](../../glossary.md#term-owner) tracks exact [operation](../../glossary.md#operation) admission and every terminal [handoff](../../glossary.md#term-handoff)/
[rollback](../../glossary.md#term-rollback) [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) it once. No second [Runtime](../../glossary.md#term-runtime) lease, owner, executor, [callback](../../glossary.md#term-callback) protocol or lock is
introduced. Tracking must not add per-operation [token](../../glossary.md#term-token) allocation or boxing. Existing representative
session threading/cleanup and reverse [publication](../../glossary.md#term-publication)/resource/integration [ownership](../../glossary.md#term-ownership-meanings) remain intact.

[PreparedExecution](../../glossary.md#prepared-execution--preparedexecution).RunLease and [Metal](../../glossary.md#term-metal) context ChildLease allocation precede their existing count
increments. Failure to create a token therefore leaves no phantom [lifetime](../../glossary.md#term-lifetime) that could indefinitely
defer persistent/context [release](../../glossary.md#term-release). Runtime prepared close remains non-waiting and its last admitted
[run](../../glossary.md#term-run) remains the unique deferred cleanup trigger. Native [ABI](../../glossary.md#term-application-binary-interface-abi), source, [kernels](../../glossary.md#kernel) and [identities](../../glossary.md#term-identity-meanings) do not change.


### [Compile](../../glossary.md#compile) [ownership](../../glossary.md#term-ownership-meanings) and mixed-[owner](../../glossary.md#term-owner) [Prepare](../../glossary.md#prepare) routing

Each compile receives the [immutable](../../glossary.md#term-immutable) provider and availability snapshots in registration order.
[Planning](../../glossary.md#term-planning) selects a [`BackendId`](../../glossary.md#backend-identity--backendid) owner before preparation and [compile artifacts](../../glossary.md#compile-artifacts) retain [identities](../../glossary.md#term-identity-meanings),
not live integration objects. Registration order is deterministic compile-time input; it is not a
[runtime](../../glossary.md#term-runtime) fallback chain.

During cold preparation, [Engine](../../glossary.md#term-engine) validates the complete non-empty planned-[partition](../../glossary.md#partition) owner set and
routes every partition by exact `BackendId` to the matching private [registry](../../glossary.md#term-registry) entry. A missing
owner fails before [backend](../../glossary.md#backend) analysis. Engine then supplies one positional preparation per partition
to the existing shared [`GraphPreparation`](../../glossary.md#graph-preparation--graphpreparation) [transaction](../../glossary.md#term-transaction); a [concrete backend](../../glossary.md#concrete-backend) receives only its
partition-scoped projection and never receives the Engine registry or [Compiler](../../glossary.md#term-compiler) aggregate. There is
no retry, implicit [CPU](../../glossary.md#term-cpu) fallback, recompilation, owner substitution, or normalized-ID match.

Before analysis, Engine derives every cross-owner edge from `LogicalMemoryRequirement` [producer](../../glossary.md#term-producer)
and consumer partitions and validates the exact ordered source/destination adapter pair, [tensor
descriptor](../../glossary.md#tensor-descriptor), [layout](../../glossary.md#layout), [data type](../../glossary.md#data-type--datatype), and byte geometry. Unsupported direction, representation, layout,
data type, or size fails closed before backend analysis or persistent-resource acquisition. The
currently supported [heterogeneous](../../glossary.md#term-heterogeneous) transfer domain is CPU to [Metal](../../glossary.md#term-metal) and Metal to CPU for all seven
carriers over [rank](../../glossary.md#term-rank)-0..16 fully static [canonical](../../glossary.md#term-canonical-form) or positive-[stride](../../glossary.md#term-stride) non-overlapping layouts with
checked physical spans. It performs no conversion. The CPU [run](../../glossary.md#term-run)-owned native representation is the
reusable [host](../../glossary.md#term-host) staging storage: CPU to Metal uploads its exact physical bytes, while Metal to CPU
downloads through cold-bound staging for [BOOL](../../glossary.md#data-type--datatype) [validation](../../glossary.md#term-validation) and [commits](../../glossary.md#commit-release-distribution-log-and-seed-meanings) only after all logical bytes
pass. No per-[element](../../glossary.md#term-element) object, canonical-byte [materialization](../../glossary.md#materialization), or additional non-BOOL hot-path copy
is permitted.

One complete Engine preparation supplies one shared schedule assembler. Each exact owner adapter
contributes only the representation and workspace creators for positions assigned to that owner.
The assembler emits the sole optional representation-creation prefix, then partition executions in
compile order with required transfer steps immediately before each destination owner's first
consumer, and finally the dense [publication](../../glossary.md#term-publication) suffix. Schedule construction uses the stable logical
producer/consumer facts and shared assignments; it does not rescore ownership, inspect backend
routes, or create a second scheduling convention.

On successful preparation, exactly one outward prepared handle is registered and retains immutable direct adapter lists aligned with logical caller inputs and publication
occurrences. Exactly one source-owner adapter creates each caller-input representation. Each
publication [captures](../../glossary.md#term-capture-meanings) the adapter owning its selected representation, including mixed result
lists. Run and materialization use those direct references without querying the registry.
Transfer recipes retain only the direct backend-owned binders and native context references
selected during preparation. Engine shutdown preserves their owner [lifetime](../../glossary.md#term-lifetime) by closing retained
results first, ordinary and advanced prepared handles in reverse
preparation-publication order second, and integrations last.

Preparation is transactional across all owner contributions. A finalizer retains [rollback](../../glossary.md#term-rollback)
responsibility until it returns. Shared Prepare then owns all returned persistent resources across
later finalizers, contribution assembly, transfer-recipe creation, schedule validation, Runtime
aggregate construction, and outward-handle publication. Failure closes acquired persistent owners
in reverse partition/acquisition order. Per-run representation creation is separately
transactional in `RunStateCreation`; a later creation, [binding](../../glossary.md#term-binding), transfer, execution, or publication
failure closes only run-owned resources in reverse acquisition order and never closes [borrowed](../../glossary.md#ownership-borrowing-and-lease)
caller storage. Closing a read-only prepared child [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) its [lease](../../glossary.md#term-lease), not another run's child or
the still-open prepared owner.

`Engine.prepareTuned(...)` admits one eligible single-owner CPU plan or exact [singleton](../../glossary.md#term-singleton)-NEG Metal
plan. It rejects empty, missing, mixed-owner, and multiple-partition plans before representative
trial execution; Metal additionally requires the complete custom-[kernel](../../glossary.md#kernel)/[MPSGraph](../../glossary.md#term-mpsgraph) local [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters), so
MPSGraph-only plans are not tuned. `ALLOW_SAFE_HEURISTIC` may fall back only through the exact
already-selected adapter after a recoverable tuning failure; it never changes owner or admits a
heterogeneous trial.

The registry map has no post-prepare execution role. Runtime receives immutable schedules with
direct [prepared executable](../../glossary.md#prepared-executable--preparedexecutable), transfer, representation-creation, and publication references. Outer
Engine handles retain direct per-input and per-publication adapters. Runtime's cold-bound action
array and hot traversal never query the Engine, registry, adapter inventory, provider, availability
[snapshot](../../glossary.md#term-snapshot), configuration, backend ID, `ServiceLoader`, reflection, or a global [lookup](../../glossary.md#term-lookup).

<a id="public-reusable-inference-session"></a>

### Public reusable [prepared execution](../../glossary.md#prepared-execution--preparedexecution)

The ordinary reusable user-facing declaration is:

```java
public final class Engine implements AutoCloseable {
    public PreparedExecution prepare(CompiledGraph compiledGraph);
    public RunResult run(PreparedExecution prepared, List<Tensor> inputs);
    public RunResult runValues(PreparedExecution prepared, List<Tensor> values);
}

public final class PreparedExecution implements AutoCloseable {
    public CompiledGraph compiledGraph();
    public boolean isClosed();
    @Override public void close();
}
```

`Engine.prepare` accepts only a [graph](../../glossary.md#graph) compiled by that exact [Engine](../../glossary.md#term-engine). One Engine [admission](../../glossary.md#term-admission) performs
one preparation and publishes exactly one outward prepared [owner](../../glossary.md#term-owner). Inward preparation, wrapper
construction and [publication](../../glossary.md#term-publication) are transactional: failure rolls back acquired resources, retaining
the primary failure and suppressing distinct cleanup failures. Construction runs nothing and binds
no caller storage. The handle borrows the Engine's composition [lifetime](../../glossary.md#term-lifetime).

`CompiledGraph.inputs()` fixes membership and final [Compiler](../../glossary.md#term-compiler) input [occurrence](../../glossary.md#term-occurrence) order. `Engine.run`
requires each exact compiled [TensorId](../../glossary.md#tensorid) once in arbitrary list order and the complete exact [descriptor](../../glossary.md#term-descriptor).
Its existing open-valid-handle diagnostics and resolved-[layout](../../glossary.md#layout) capacity checks remain unchanged.
For static templates, [`Engine.runValues`](../../glossary.md#fresh-value-binding--enginerunvalues) requires exactly one value in each compiled input position, without TensorId
matching; the same fresh [Tensor](../../glossary.md#tensor) may fill distinct matching slots. Full descriptors must match
exactly, including [dtype](../../glossary.md#data-type--datatype), fully static [shape](../../glossary.md#shape), layout presence/geometry and requiresGrad.
[64,784] cannot bind [32,784]; no implicit [specialization](../../glossary.md#term-specialization) occurs.

Prepare0013 implements an explicitly qualified ordinary [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) [CPU](../../glossary.md#term-cpu) branch of the
same public [API](../../glossary.md#term-application-programming-interface-api): one named B, fixed [features](../../glossary.md#term-feature)/[parameters](../../glossary.md#term-parameter), [rank](../../glossary.md#term-rank)-at-most-two boundaries plus qualified
[AD](../../glossary.md#autograd) [views](../../glossary.md#term-view-meanings), generated [scalar](../../glossary.md#term-scalar) and single-range execution. The
[bounded symbolic Scan contract](recurrent-scan.md#accepted-symbolic-batch-scan)<span data-glossary-navigation="true"> ([symbolic Scan glossary](../../glossary.md#symbolic-scan))</span> is implemented by
Prepare0014 for homogeneous FLOAT32 forward and actual first-order reverse, fixed T/features/state
widths, generated scalar and single-range execution. Its task tracks final [qualification](../../glossary.md#term-qualification).
Prepare0015 implements bounded combined length-[Scan](../../glossary.md#term-cumulative-scan) for one named B, fixed T/features, homogeneous
FLOAT32/[FLOAT64](../../glossary.md#term-float64) forward and actual Compiler reverse, including bounded outer typed/control/ROW(1)
closure. Owning [validation](../../glossary.md#term-validation) and independent code/structural review passed; root validation and the sole bounded
probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending. This uses the
same [prepare](../../glossary.md#prepare)/[run](../../glossary.md#term-run)/runValues, occurrence indexing and dtype-generic publication owners, with no new
public API, Engine [memo](../../glossary.md#cache-and-memoization) or second [lease](../../glossary.md#term-lease). It is not all-valid symbolic FLOAT64 qualification.
[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[TrainingSession](../../glossary.md#training-session--trainingsession), symbolic cold/copy/wire, native/mixed, other symbolic CPU dtype/family extensions,
variable T/features, higher-order, [vector](../../glossary.md#term-vector)/parallel and tuning remain separately gated.
Compiler-owned `BatchBindingPlan` derives consistent B>=0 from actual
fully static descriptors in final input order, including unused positions. Fixed [axes](../../glossary.md#term-axis), parameters,
[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), dtype and requiresGrad match exactly. Dynamic caller layouts must be resolved zero-offset
dense with actual [canonical](../../glossary.md#term-canonical-form) [strides](../../glossary.md#term-stride); same-address dense read views may qualify, but absent layout
is not silently canonical. Proved internal [transpose](../../glossary.md#term-transpose)/expand AD views use separate CPU mappings.
Only B/[constants](../../glossary.md#term-constant) determine qualified expressions; output-only/anonymous symbols reject at prepare.
`run` keeps exact-ID/arbitrary-order semantics; changing B uses runValues without [prototype](../../glossary.md#term-prototype) mutation.

Dynamic order is Engine/owner/closed-prepared gates, [snapshot](../../glossary.md#term-snapshot)/count/null, semantic and numeric/size
[preflight](../../glossary.md#term-preflight), snapshot all storage associations once and validate all, borrow, [Runtime](../../glossary.md#term-runtime) lease/create/
cold bind, execute, then complete result. Before borrow/allocation check fixed axes/type/flags,
counts/strides, byte products, referenced spans/offset sums and aggregate bounds including [alignment](../../glossary.md#term-alignment)
[padding](../../glossary.md#padding) and representation multiplicity. First CPU extents/counts/strides/offsets/referenced-[element](../../glossary.md#term-element)
spans fit Integer.MAX_VALUE, bytes/aggregate bounds fit Long.MAX_VALUE; these are representability
limits, not B_max. Empty count cannot hide overflowing trailing geometry or unsupported unused
declarations. Storage must be matching, present, live, thread-accessible and sufficient. Physical
address/alignment and writable overlap checks follow allocation, before execution.

Implemented inward Engine declaration changes are confined to existing non-public owners (excerpts):

```java
// Existing package-scoped EngineBackendComposition
BufferRepresentation borrow(HostTensorStorage storage, long referencedByteSize);
// Existing package-private RunResult constructor
RunResult(AdvancedRunResult owner, List<CompiledGraph.PublicationSpec> specifications,
        List<TensorDescriptor> actualDescriptors);
// Existing private Publication constructor
Publication(RunResult result, int index, CompiledGraph.PublicationSpec specification,
        TensorDescriptor actualDescriptor);
public TensorDescriptor descriptor(); // existing RunResult.Publication accessor
```

The resource-contract null/invalid/[overflow](../../glossary.md#term-overflow) rules apply. [Borrowing](../../glossary.md#term-borrowing) wraps exactly the referenced
prefix, never excess capacity or caller [ownership](../../glossary.md#term-ownership-meanings); invalid prefixes reject, failure closes wrappers
only. Static borrowing/fast spans stay. Engine preindexes publication-to-value-list positions and
resolves only requested descriptors, with exact occurrence count. [Gradient](../../glossary.md#term-gradient) [metadata](../../glossary.md#term-metadata) comes from
generated gradient values, not [target](../../glossary.md#term-target) flags. Preserve tensorId/role/derivativeOrder/targetIndex,
dense order and distinct authenticated [aliases](../../glossary.md#aliasing-and-arena). [Materialization](../../glossary.md#materialization) uses actual shape/layout/bytes and
returns [detached](../../glossary.md#detached-model-results) [HostTensorValue](../../glossary.md#host-tensor-value--hosttensorvalue). Materialized outputs are canonical non-view; qualified view
publications retain actual geometry. Failure publishes nothing, closes inward result then reverse
borrows with distinct suppression, and never closes caller storage. Existing leases, close
precedence, retry and different-B run/result isolation remain; no second owner or lease is added.

Before any borrow, each call snapshots every selected storage association once and validates all
values and storage for presence, dtype, [liveness](../../glossary.md#term-liveness), current-thread accessibility and capacity.
For runValues, resolved layouts use checked referenced element/byte spans; absent layouts use
checked canonical static geometry. Unknown or overflowing geometry rejects. This additional
absent-layout validation is scoped to runValues, not run. Static positional geometry may be
precomputed during preparation. Caller storage stays owned by the caller and must remain usable
and free from conflicting mutation until the returned result closes.

Both ordinary run methods check Engine admission first, then prepared nullness/exact ownership,
then the exact inward delegate's outward close gate, then inputs. This deliberately changes direct
run's former precedence: closed preparation now wins null/malformed/dead inputs with
`prepared execution is closed` and no borrow. Null or foreign prepared handles also win input-list
errors. Runtime's unique nonwaiting run lease still arbitrates a subsequent close. The outward
gate is not held through execution and adds no second lease protocol.

Both paths reuse prepared metadata, direct input/publication adapters and Runtime's stateless
runner. They perform no prototype mutation, synthetic Tensor construction, recapture, [compilation](../../glossary.md#compiler-engine-and-execution-stage-names),
preparation, provider/[registry](../../glossary.md#term-registry) [lookup](../../glossary.md#term-lookup), [backend](../../glossary.md#backend)/[kernel](../../glossary.md#kernel) selection or extra per-element work. The
positional path reuses ordinary per-run storage/borrow collections, not another [binding](../../glossary.md#term-binding) map.
Concurrent calls create isolated [mutable](../../glossary.md#term-mutable) state and results, with no cross-call input retention;
there is no [device](../../glossary.md#term-device)-overlap, [throughput](../../glossary.md#term-throughput), submission-order or fairness promise.

Every run returns existing [`RunResult`](../../glossary.md#run-result--runresult) with dense forward-then-gradient publication order.
Aliases remain distinct occurrence objects; exact publication [identity](../../glossary.md#term-identity-meanings) authenticates explicit
materialization. Each successful materialization returns a fresh detached `HostTensorValue`.

Prepared close closes the exact inward Runtime owner once and unregisters even on cleanup failure.
Repeated/concurrent close retains failure replay and interruption restoration. It rejects new runs,
but does not wait for already leased runs or close returned results. A leased run may finish and
its last lease [release](../../glossary.md#term-release) performs deferred prepared-resource cleanup. Engine shutdown rejects new
operations, waits for admitted work, then closes results in reverse run-publication order,
ordinary/advanced preparations in reverse preparation-publication order, and integrations in reverse
registration order, with attempt-all cleanup and suppression. Result materialization remains valid
after prepared close while result and Engine are open. Graph/prepared metadata remains readable
after close; detached values outlive result, preparation, Engine and caller storage.

Advanced and ordinary run publication includes wrapper construction and registration in one
[rollback](../../glossary.md#term-rollback) boundary. Failure closes the completed inward result or its constructed outward owner,
never both, and [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) the admitted Engine [operation](../../glossary.md#operation) exactly once even if rollback fails.
Rollback runs outside lifecycleLock. The first outward result close transfers Engine-owned borrow
wrappers into its cleanup [scope](../../glossary.md#term-scope) and drops the retained field reference, preserving reverse wrapper
cleanup, serialized materialization, concurrent-close waiting, interruption restoration and exact
outward failure replay. Runtime repeated close retains its separate existing non-replay behavior.
[Engine0022](../../planning/modules/engine/tasks/0022-result-publication-and-closed-payload-cleanup.md#result)
implements these guarantees after independent review and MAIN's before/after validation, without
adding an owner, lease, lock, factory or fault flag.

[`InferenceSession`](../../glossary.md#former-inference-session--inferencesession) and `Engine.session` are removed without a replacement wrapper or compatibility
shim. External callers must migrate to prepare/run and recompile. Training retains its separate
stateful `TrainingSession`; no compiler, scheduler, result type, composition rule, tuning surface or
error hierarchy is added.

### Accepted [Engine geometry memo](../../glossary.md#engine-geometry-memo)

Accepted future direction on 2026-10-25, unimplemented; Model0041's authority delivery is Complete,
MAIN accepted 2026-10-25 after independent APPROVE. This is the sole [metadata](../../glossary.md#term-metadata)-only
exception to prepared immutability and invocation-private numeric facts above. It applies only
at the exact [Engine](../../glossary.md#term-engine) [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) entry for a separately qualified symbolic-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) plan,
including the separately qualified [Scan](../../glossary.md#term-cumulative-scan) vertical. It does not change current static behavior or add a [cache](../../glossary.md#term-cache)
to [Runtime](../../glossary.md#term-runtime). Prepared [graph](../../glossary.md#graph) meaning, routes, code, recipes, slots, representation positions and
persistent resource [identities](../../glossary.md#term-identity-meanings) never change.

Retain at most one replaceable complete [immutable](../../glossary.md#term-immutable) entry, empty initially. Its key authenticates
the exact compiled/[binding](../../glossary.md#term-binding) plan, inward preparation and memory-plan references, fixed numeric
recipes/[publication](../../glossary.md#term-publication) mapping, explicit B and execution mode, including [constant](../../glossary.md#term-constant)-byte `[B,0]`.
Equal B/bytes/[shapes](../../glossary.md#shape) or structurally equal foreign plans cannot authenticate a hit. Retain only
covered pure Engine B-dependent numeric validity, exact sizes/[strides](../../glossary.md#term-stride)/offsets, aligned aggregate
bounds and template-derived non-input publication metadata. No [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings), supplied
[descriptor](../../glossary.md#term-descriptor), `Binding`, input/storage association, physical address, [lease](../../glossary.md#term-lease), [mutable](../../glossary.md#term-mutable) working array,
resource, result or failure may enter the [memo](../../glossary.md#cache-and-memoization). One graph-sized [snapshot](../../glossary.md#term-snapshot) is allowed, not B-sized
[payload](../../glossary.md#artifact-envelope-and-atomic-publication), map, pool, global/ThreadLocal state or per-B executable/full-preparation cache. Replaced
snapshots survive only while referenced by live calls/results; concurrent candidate retention is
bounded by active admissions, never an accumulating history.

Every call preserves Engine/[owner](../../glossary.md#term-owner)/closed-prepared precedence and exact inward-delegate [admission](../../glossary.md#term-admission),
then snapshots inputs, checks counts/nulls and establishes exact-ID or positional order. It must
perform the complete fresh [Compiler](../../glossary.md#term-compiler) bind and every private typed obligation, including unused and
repeated input occurrences. Validate every B [occurrence](../../glossary.md#term-occurrence), fixed [axis](../../glossary.md#term-axis), [rank](../../glossary.md#term-rank), [dtype](../../glossary.md#data-type--datatype), requiresGrad,
[layout](../../glossary.md#layout) presence, strides, offsets and [view](../../glossary.md#term-view-meanings) policy before considering a hit. Equality of the first
input's B is not admission. A hit avoids only covered Engine pure numeric/size [evaluation](../../glossary.md#term-evaluation) and
template-only non-input descriptor reconstruction; it never skips Compiler work or unrelated checks.

Preparation preindexes complete numeric checks for all outer and scoped local domains and latent
ports, including unused/[scalar](../../glossary.md#term-scalar)-backed values; no per-[run](../../glossary.md#term-run) graph walk/[inference](../../glossary.md#term-inference) or synthetic Tensor
construction is allowed. For the initially qualified [CPU](../../glossary.md#term-cpu) domain, counts/extents/[element](../../glossary.md#term-element)
strides/offsets/referenced spans fit CPU int. Checked bytes, aligned offsets and
representation-multiplied aggregate bounds fit long. Other [backends](../../glossary.md#backend) require separately qualified
representability bounds; no additional [backend capability](../../glossary.md#backend-capability) is granted here. Empty
leading domains cannot hide overflowing trailing geometry. Numeric recipes may use checked
piecewise [alignment](../../glossary.md#term-alignment)/empty-scratch cases, never select resources/[topology](../../glossary.md#term-topology)/routes for a particular B.

Each call snapshots current storage associations once, revalidates presence/type/[liveness](../../glossary.md#term-liveness)/current-
thread accessibility/capacity and borrows exact current prefixes. Physical type/access/alignment
and writable-overlap checks still precede execution. No hit authenticates storage or addresses.
Input-backed publications use the fresh binding's current supplied descriptor, including admitted
dense-read `isView` differences at the same B. Cache only template-derived non-input facts and
preserve publication order, tensorId/role/derivativeOrder/targetIndex, [aliases](../../glossary.md#aliasing-and-arena) and generated [gradient](../../glossary.md#term-gradient)
flags. A fresh `Binding` is not retained across calls.

Encapsulated immutable exact-plan numeric geometry and never-mutated private backing may be shared.
Mutable size/access/packed/address/[carry](../../glossary.md#term-carry) arrays are run-owned: copy reusable facts into fresh
isolated working arrays when required. Runtime receives only authenticated numeric geometry,
never the memo, symbols or semantic metadata. `RunMemoryGeometry.resolve(plan, OptionalLong)`
remains a fresh resolver, not a hit [API](../../glossary.md#term-application-programming-interface-api); its existing test that shares geometry across fresh
states is evidence of immutability, not authority for this memo. Every invocation still creates
one fresh [RunState](../../glossary.md#run-state--runstate), fresh physical resources even at zero bytes, fresh writable outputs and a
distinct result under existing [borrowing](../../glossary.md#term-borrowing)/cleanup rules.

Publish only a complete immutable candidate, atomically, after pre-borrow descriptor/numeric/
storage admission. Readers take a local immutable snapshot; mixed-B races may compute duplicate
candidates and either valid replacement may win, while each call keeps its own selected facts.
Failed admission publishes nothing and preserves a prior good entry. Later allocation/bind/
execution failure may retain valid metadata only, never resources/results/failures; reverse
cleanup is unchanged. Retry repeats all current input/storage checks and creates fresh resources.

Close clears the slot and prevents late republishing. [Coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) [atomic](../../glossary.md#resource-transaction-and-rollback) replacement versus close
through the existing [lifecycle](../../glossary.md#lifecycle), with a bounded mechanism chosen in a later exact scoped brief.
After existing inward-delegate admission, concurrent close suppresses/clears publication; it must
not add a new late outward-close rejection. Admitted calls retain local immutable facts and
Runtime's existing sole run lease arbitrates the later close/run [race](../../glossary.md#term-race) and error precedence.
No execution lock, second lease or new owner/protocol is permitted. [Lookup](../../glossary.md#term-lookup)/publication/copying
stays at cold Engine entry, never generated time/row/contributor loops or Runtime execution.

No new production type/public [signature](../../glossary.md#term-signature), Compiler refactoring or changed Runtime [handoff](../../glossary.md#term-handoff) is
selected. Exact fields/methods, [atomic publication](../../glossary.md#term-atomic-publication)/close mechanism and any necessary protocol
change require later owner-specific approval; if a new named type is necessary, stop for exact
package/name/purpose/signature and user approval. Future tests must witness one preparation and
64→64→40→40→64 miss/hit/miss/hit/miss using bounded test-local counters for avoided Engine work,
not skipped binder obligations. Cover invalid same-B metadata/storage, current input-backed views,
B=0/1 and `[B,0]`, [overflow](../../glossary.md#term-overflow), foreign preparations, failure/retry, close without late publication,
concurrent mixed-B snapshots/results, fresh resources/working arrays and generated loops with no
memo access. No test is newly run by this authority-only change and no speed claim follows.

## [Runtime](../../glossary.md#term-runtime) [service locator](../../glossary.md#term-service-locator)

A runtime service locator is forbidden as a core mechanism.

A runtime service locator means runtime dynamically asks for services or [backends](../../glossary.md#backend) during execution, for example:

```java
Backend backend = RuntimeServices.get("metal");
KernelRegistry kernels = RuntimeServices.get(KernelRegistry.class);
```

This is forbidden because runtime must execute already-prepared schedules.

Backend selection and executable construction must happen before runtime [hot path](../../glossary.md#term-hot-path) execution.

## Reflective [backend](../../glossary.md#backend) [plugin](../../glossary.md#term-plugin) discovery

Reflective backend plugin discovery is forbidden as the core backend mechanism.

Examples include:

- [classpath](../../glossary.md#term-classpath) scanning
- annotation scanning
- automatic backend discovery through reflection
- `ServiceLoader` as the default [runtime](../../glossary.md#term-runtime) backend mechanism

Any future generic backend registration must be explicit through [Engine](../../glossary.md#term-engine) composition.

`ServiceLoader` or plugin discovery may be added later as a convenience layer only if this document is updated first.

It must not become a runtime hot-path mechanism.

## Profile-free [Engine](../../glossary.md#term-engine) and [Prepare](../../glossary.md#prepare) [lifecycle](../../glossary.md#lifecycle)

<a id="numerical-profile-lifecycle"></a>

Engine construction offers no
numerical-profile selection or replacement mode. [Compiler](../../glossary.md#term-compiler) [artifacts](../../glossary.md#term-artifact) and [`PrepareContext`](../../glossary.md#prepare-context-preparecontext)
projections [carry](../../glossary.md#term-carry) no profile field; [backend](../../glossary.md#backend) preparation uses complete [occurrence](../../glossary.md#term-occurrence) facts, actual
provider capability, qualified routes, and versioned non-profile compatibility [identity](../../glossary.md#term-identity-meanings).
[Runtime](../../glossary.md#term-runtime) and [Trace](../../glossary.md#trace) remain numerical-policy-free and perform no per-[run](../../glossary.md#term-run) [tolerance](../../glossary.md#term-tolerance), [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions),
capability, or route [lookup](../../glossary.md#term-lookup). Removing the selector does not change Engine [ownership](../../glossary.md#term-ownership-meanings), Prepare
transactions, resource lifetimes, transfers, tuning, or [synchronous](../../glossary.md#term-synchronous) run/[publication](../../glossary.md#term-publication) behavior.
