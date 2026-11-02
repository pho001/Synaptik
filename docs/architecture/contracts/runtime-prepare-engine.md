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

General scan adds no Runtime graph/body interpreter or nested execution owner. Its concrete loop
is already prepared; Runtime retains exactly one RunState for the whole logical run, not one per
step or body. Complete scan semantics belong to [the scan owner](recurrent-scan.md#general-differentiable-scan).

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

Caller inputs are borrowed for one run. Mutable internal buffers and workspaces are run-owned.
Published outputs transfer or lease ownership to `RunResult`. Immutable persistent prepared
resources remain owned by `PreparedExecution` and are not ordinary workspace. A backend may issue
a fresh run-owned read-only representation binding as a child lease into immutable prepared
physical storage. Prepared-owner close rejects new children and relinquishes its owner reference;
an already-issued child keeps that storage resident until its exact run state/result closes.
Runtime orchestrates cleanup, but concrete representations perform physical release. Failure
cleanup releases only resources still owned by the run, never borrowed inputs or already
transferred outputs.

A persistent prepared resource implements a narrow Runtime-owned nominal lifecycle contract;
concrete backend code owns its physical state and performs physical release. `PreparedExecution`
snapshots each acquired resource identity exactly once and closes the unique prepared owners in
deterministic reverse-acquisition order. A repeated schedule occurrence or executable reference
does not create another ownership occurrence. Child bindings do not add prepared-resource entries.

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

### Accepted symbolic-batch resource contract

The user accepted this protocol on 2026-10-24.
[Prepare0013](../../planning/modules/prepare/tasks/0013-ordinary-dynamic-batch-cpu-vertical.md#result)
implements its declarations, explicit batch-required mode, authenticated run geometry and
size/alignment creator migration for the bounded ordinary FLOAT32 CPU path. Static consumers
retain fixed-size construction/accessors; their creators also receive exact bytes and alignment.
Final independent reviews and MAIN's native-enabled integration gate passed;
MAIN accepted the capability on 2026-10-24.
Prepare0015 implements combined one-B/fixed-T/features FLOAT32/FLOAT64 length-Scan through this
existing protocol, including the separately bounded CPU outer typed/control prerequisite. Owning
validation and independent code/structural review passed; root validation and the sole bounded probe completed;
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result)
records evidence and limits; final MAIN acceptance remains pending. No declaration/signature, creator,
storage/lease or numeric Runtime
protocol changes.
Runtime remains Model/Compiler-free: no Shape, dimension
expression, Operation, graph, route search, generation or cache lookup enters run admission or
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

References/results are non-null. Lists snapshot membership and reject null elements in encounter
order. Null fails with `NullPointerException`; invalid fields/metadata with `IllegalArgumentException`;
checked overflow with `ArithmeticException`; invalid dense indices with `IndexOutOfBoundsException`.
Existing lifecycle precedence/failures remain. Immutable recipes are storage-free/thread-safe;
resolved private size arrays currently belong to their invocation. The accepted future
[Engine geometry memo](#accepted-engine-geometry-memo) narrowly permits sharing encapsulated,
never-mutated numeric facts/private backing for an exact plan and B/mode. It does not permit sharing
mutable invocation size/access/packed/address/carry arrays. This distinction changes no current
fresh `resolve` API or implementation. Size is nonnegative long bytes; alignment is fixed
power-of-two bytes from 1 through `1L<<62`. `byteSize(B)` validates B>=0 and computes exact checked
bytes. Recipes contain numeric facts only, with no value reads, providers, allocation or discovery.

`fixedByteSize()` is structural: present for FixedByteSize, absent for run-bound recipes even when
their result happens to be constant. Old long constructors wrap FixedByteSize. Old `byteSize()`
accessors return its exact bytes (including zero), but throw `IllegalStateException` for run-bound
declarations; no fake zero/-1/maximum size. Constructor/accessor source compatibility is retained,
not record reflection/binary compatibility. Creator single-abstract-method (SAM) consumers migrate
explicitly, including retained static/native routes; no unsafe zero-argument dynamic adapter.

Execution mode is independent of byte independence. Prepare sets `requiresBatchSize=true` for
qualified symbolic projections, including all-constant-byte `X[B,0]` plans. Old two-list construction
means false and rejects run-bound declarations. `resolve` requires present B iff mode is true and
resolves every entry in plan order. Geometry authenticates the exact plan reference; foreign geometry
rejects. Absent-B runner/creation/state overloads reject true-mode plans before creators or ownership
acceptance, even if every byte count is constant. State-aware default hooks may delegate to old
static hooks only for false-mode plans, never by testing byte independence.

Creators allocate exactly the requested usable span with fixed alignment; allocator padding is not
capacity. Every success returns fresh non-null physical objects even for zero bytes. Initialized
buffers fill the actual count. Preserve complete caller validation before creation, duplicate-result
rejection, origin/validity rules and reverse rollback. Backend rollback applies before return;
setup/run owns returned resources afterward. Allocation errors propagate and null results fail.
No mutable B, working numeric array, address, workspace or output mutates shared prepared state.
Only the explicitly bounded future memo below may replace a metadata snapshot; prepared meaning,
recipes and persistent resource ownership remain immutable.

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
order, including producerless published caller sources, independent of partition and backend
declaration order. Remaining ordinary buffers retain first-declaration order; producerless
published constants remain last in final graph-value order. The plan-ordered buffer association
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

For accepted general scan, PrepareContext/PartitionDag must preserve the validated CompiledScanBody
attachment, role/result bindings and exact demanded outer-port indices. Validate static local body
descriptors in their own scope separately from outer IDs; numeric ID coincidence must not alias
the descriptor domains. Prepare0010 implements this scoped projection and static binding/demand
validation; Prepare0011 extends the same owners to length-aware forward and actual generated
reverse scans. The separate non-gradient INT64 lengths actual has no local formal or gradient
contribution. Declared row boundaries and tied actual positions are checked without re-proving
Model/Compiler body algebra or importing CPU read-layout predicates. This is not backend execution. One selected
owner analyzes/prepares the complete body once; no child owner, body partitioning or new executable
aggregate is authorized. Ordinary explicit residual output/input edges, not an opaque saved-
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

The same null/size/alignment/static-accessor rules as the Runtime resource contract apply;
workspace IDs remain nonnegative and analysis-local. `size()` retains the complete declaration.
Shared Prepare transports/compares structural declarations, alignment and coverage without
evaluating backend geometry. Qualified symbolic contexts set the explicit plan mode; fixed static
contexts retain their existing guards. Slot identities, requirement membership, representation
positions, scoped IDs, demand, routes and code remain fixed before finalization. All resources
must be declared as exact fixed bytes or complete immutable run-size recipes, never discovered
during binding. No new Prepare production type, backend dependency or geometry interpreter is added.

The accepted [symbolic Scan contract](recurrent-scan.md#accepted-symbolic-batch-scan)
requires a bounded extension of the ordinary rank-two projection guards, not blanket
symbolic/rank admission. Validate all declared outer and local templates, every actual/formal/
result binding, canonical siblings and complete demand in their own scopes before analyzers.
Preserve named-axis obligations and qualified rank-three outer AD closure without importing CPU
read predicates or re-proving the body's semantic algebra. Set `requiresBatchSize` even for
constant-byte symbolic plans. Shared Prepare transports exact fixed/run-size declarations and
stable coverage/slots; it does not evaluate CPU recipes, choose B-specific topology or infer
Scan capability. Prepare0014 implements this symbolic projection and the separately qualified
homogeneous FLOAT32 CPU/Engine execution slice; its
[task](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md) tracks final qualification.
Static Scan projection remains unchanged.

Prepare0015 implements length projection with static or one named B, fixed T/features and homogeneous
FLOAT32/FLOAT64 locals. The separate non-gradient INT64 `[B]`
actual has no formal. Apply the same complete pre-analyzer outer/local declaration, positional
binding, latent-port, role and demand validation to forward and actual generated mixed-role reverse;
do not substitute numeric size coincidence for named-axis identity. Static canonical guards and
unresolved symbolic templates remain distinct. This capability extends support through these
existing owners, not the handoff or semantic row algebra.

### Producerless published values

The reviewed complete-graph source-publication cutover renames the existing
`ProducerlessPublishedConstantResource` owner to `ProducerlessPublishedValueResource`.
Its five canonical components are `GraphValue value`, `LogicalMemoryRequirement logicalRequirement`,
`BackendId owner`, `PreparedMemoryPlan.ByteSize size`, and `long byteAlignment`. Migrate all
callers, docs and inventory entries; remove the old name/hook without an alias or compatibility
overload. The existing three-/four-argument GraphPreparation forms retain their roles.

The migrated default assembler geometry hook is
`producerlessPublishedValue(GraphValue, LogicalMemoryRequirement, Optional<ScalarValue>)`.
The scalar is present exactly for an authenticated constant source, empty exactly for a caller.
It preserves existing constant dtype/canonical-layout policy; initialization still belongs to
the existing per-run constant initializer. Caller sources still use the existing borrowed
CallerInput path. The contribution reads no payload, allocates/acquires nothing and creates no
operation, partition, representation or executable.

The audit-driven Metal completion extends this same protocol to Metal-only registrations,
including constant cotangents and source-only callers. Engine's source-owner selection and
composite assembler must agree on a registered supporting backend without requiring a CPU
adapter. Preserve existing CPU behavior and deterministic registration/ownership policy; the
chosen backend supplies exact physical geometry and the existing initialization/import/export
path. Do not synthesize a Tensor operation or partition, install a CPU fallback, introduce a
second initializer, or weaken source authentication and pre-finalization validation.

One shared discovery/validation path requires an actual graph input that is published without
producer or consumer, classified disjointly by the existing constant/caller plans. Preserve exact
graph-value/logical-requirement references, canonical complete coverage and duplicate/foreign/
extra/missing/overlap rejection before finalization. Reuse/factor the existing fixed-or-sole-named-B
vocabulary and common-symbol proof for complete source values; do not copy another guard.
Constants retain static/resolved/nongradient/type policy. Static caller views retain exact
resolved origin/strides; unresolved static caller layouts remain unsupported. Named templates
retain their existing unresolved declaration and canonical zero-offset actual-input contract.

Index ordinary declarations and producerless contributions before slot assignment. Preseed
both ordinary and source-only callers in exact Compiler bindable occurrence order, then assign
actual partition requirements unchanged. Source-only callers receive no partition-finalizer
requirement. Append source-only constants in final graph-value order. Retain the supplied
ByteSize recipe, including structural symbolic zero-byte cases; no sample B, fake fixed size,
maximum geometry or payload-derived declaration is allowed.

Determine complete physical batch mode from source/complete-graph named facts as well as
partition contexts. A genuinely fixed CPU unit inside that complete batch-required plan keeps
its static lowering/artifact. Existing CPU state-aware cold hooks authenticate fixed selected
declarations, exact plan-authenticated resolved geometry and retained unit facts, plus unchanged
dtype/span/alignment/writability/lifetime guards. Nonfixed selections require their genuine
dynamic route or reject. Runtime's default static guards remain unchanged; no new flag, field,
per-B topology, relowering or generated ABI follows.

Static geometry reuses Model referenced span, including empty-span semantics; named geometry
reuses existing CPU access/binding proofs and checked actual-B sizing. Raw source storage and
publication do not invoke arithmetic or unlock CPU low arithmetic. Existing ownership, rollback,
creators, publication adapters and nonempty-partition policies remain unchanged. In particular,
this cutover does not add no-node Engine/Prepare support or a new standalone CPU assembly policy.

### General Scan activity projection

Implemented coordinated Class C projection, accepted with Compiler0017/CPU0081 after independent
authority/interface and exact-source implementation reviews and actual complete-preparation/
Engine consumers. The [Scan owner](recurrent-scan.md#general-ordinary-unit-scan-cutover) owns
semantics; [Planning](foundational-modules.md#general-scan-activity-transport) owns the neutral
immutable carrier. Shared Prepare does not interpret derivative mathematics or CPU pack layouts.

PrepareContext adds sixth canonical component `Map<NodeId, ScanActivityPlan> scanActivityPlans`
after backendInputs. GraphPreparation projects only containing Scan entries from component eight
of CompileArtifacts. Context copies/subsets preserve/filter exact associations; intentional
ordinary/synthetic construction has an empty map, never a silent drop of annotated metadata.
Validate partition membership, exact local-body graph identity, complete typed declarations,
positional boundaries, roots/scopes/merge/accumulator applicability and complete demand BEFORE
backend analyzers. No first-carry dtype, nonempty/positive-ROW or obsolete rank-two-history
shortcut replaces those checks. Independent ordinary and symbolic named-coordinate guards remain.
Retain the Scan owner's local materialized-boundary meaning: do not infer incoming absence from
numeric values, producer provenance or shared length storage. No cross-partition activity
communication field or additional resource owner is introduced.
Before backend analysis, use the scoped four-argument Model ordinary guard with the exact sole
declared optional symbol, then check all declared canonical siblings. Do not choose a sample B,
ignore extra/foreign symbols, copy symbolic family rules or impose length ROW locality on
all-valid whole-Tensor reshape. Ordinary three-argument inference may defer these equalities.

For the qualified CPU continuation, the single backend owner may construct an internal context
from the EXACT contained graph, derive its local DAG/logical requirements and reuse its ordinary
physical decomposition. This explicitly supersedes the older body-partitioning prohibition only
for backend-internal physical units. No local IDs alias outer requirements/constants, no semantic
child owner or new executable aggregate is created, and no Compiler aggregate crosses to CPU.
Zero-node passthrough bodies require no forged partition node. Ordinary explicit residual edges
remain the saved-value handoff; compiled activity is neither persisted source nor an opaque tape.

Legitimate inferred computed body values may retain unresolved logical layouts. The backend's
local ordinary projection completes only those computed descriptors through the shared Model
ordinary-layout closure, updating matching logical-memory descriptors before ordinary analysis.
The captured graph, exact nodes/IDs/order/phases and activity graph association do not change.
Actual and formal layout guards stay strict; no unresolved ingress becomes canonical by default.
Both capability/decomposition and helper resource preparation consume the same projection path.
Shared Prepare still validates semantic declarations; it gains no CPU layout algorithm,
additional context field or physical allocation responsibility.

Body cuts, helper scratch and numeric control use existing buffer/workspace declarations and
RunState ownership. Fixed or checked symbolic run-size recipes declare all storage before slots;
no hidden Arena or runtime-discovered route/topology. Invocation-owned mutable packs/cursors
never enter reusable prepared state. Runtime/Engine lifecycle, lease/cleanup and execution
protocols remain unchanged.

## Prepare lifecycle

The implemented [CPU0074 reservation exception](#cpu0074-run-owned-synchronous-reservation)
adds only backend-declared selected-region staging/partials and a run-bound CPU context; the
shared analysis/assignment/finalization handoff is unchanged.

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
are physical creation recipes, not an alternative schedule. For the implemented bounded symbolic-batch
path, exactness means fixed bytes or a complete run-size declaration with fixed alignment;
Prepare validates mode/declarations/coverage without evaluating numeric recipes. Current static
analysis/finalization remains supported. Shared preparation validates complete,
non-overlapping contribution coverage, assembles and validates ordered schedule-step occurrences,
then constructs the Runtime `PreparedExecution` with the exact assigned memory plan, exact
schedule, and acquisition-ordered persistent resources. Successful construction transfers unique
ownership of those resources to that Runtime aggregate.

General-scan analysis declares checked exact output, staging, carry-bank, body-temporary and live
history bytes before slot assignment, with zero history bytes/stores for undemanded residual ports
including reverse residuals. Revalidate demand against the projected roles and logical requirements.
Only finalization after exact slot/representation assignment may emit the selected CPU time loop
and typed body code. It may not discover new shared storage requirements during emission. Concrete
lowering stays in the backend; local descriptors never become outer logical slot IDs by coincidence.

Prepare0011 implements static length-mode projection through this lifecycle. It preserves ordered
ROW(0), rank-two ROW(1) and SHARED roles, exact scoped bodies and the extra INT64 actual without
a body formal. All partition contexts, including unused local declarations and checked logical
geometry, validate before the first analyzer; complete-graph demand includes every outer consumer.
Concrete analysis owns exact mixed-boundary buffer/workspace geometry and declares it before slots.
Shared assignment, finalization, rollback and Runtime ownership/RunState isolation are unchanged.
No length value is read, and metadata/resource-handoff tests allocate no physical history or run
a body. CPU0042 separately implements fixed-geometry FLOAT32/FLOAT64 forward and actual reverse:
backend analysis declares mixed-width resources before slots, and the existing scan invocation
checks values after producers before generated skipping. No shared Prepare or Runtime handoff changes.
The [scan owner](recurrent-scan.md#runtime-preflight-and-implementation-gates) retains its semantic scope.

Prepare0015's implemented combined path uses this same sequence with exact fixed/run-size
declarations before slots, explicit `requiresBatchSize` even for constant-byte `[B,0]`, and
dtype-width-correct lengths, outputs, banks, cuts, temporaries and demanded histories. Declared
zero-byte workspaces retain slots/creators. Shared Prepare still transports declarations without
evaluating CPU recipes or reading lengths.


## Run lifecycle

Only the implemented [CPU0074 reservation exception](#cpu0074-run-owned-synchronous-reservation)
qualifies the no-per-step-allocation statement below for existing synchronous scheduler metadata.
It adds no Runtime protocol, public async contract, extra lease owner or RunState.

General scan's mutable carry, changing addresses and geometry arrays, temporaries and histories
remain run-owned under this lifecycle, never mutable fields of shared prepared state. Existing
ownership, overlap validation, rollback, retry and concurrent-run isolation apply; no runtime tape,
per-step allocation or extra RunState is introduced. Runtime never inspects body graph metadata.

For the implemented bounded symbolic-batch path, Engine resolves semantic and checked numeric/size geometry
before borrowing; Runtime still acquires its unique prepared lease before inspecting supplied
inputs/geometry. Actual-byte validation and every mutable packed/access/size array are run-owned.
Under the accepted future [memo exception](#accepted-engine-geometry-memo), immutable exact-plan
numeric facts may be shared, not those working arrays or any resource. Runtime performs no memo
lookup/publication; the exception adds no lease, execution lock or late outward-close rejection.
Shape/byte admission never reads lengths: length VALUE checks remain after their producers and
before that Scan's first writes, including all B lengths and T=0/B=0. CPU0042 implements those
checks inside the existing CPU scan invocation for fixed T/B/features and FLOAT32/FLOAT64
forward/actual reverse. All-valid symbolic Scan retains its separate qualification. Prepare0015's
implemented one-B length-Scan path follows the same sequence: every call performs fresh
complete Compiler binding/private obligations, current input-backed publication/storage admission
and checked numeric sizing before borrow, then run-aware typed span/access/alignment/overlap and
all actual B length checks after producers before the first Scan-local write. B is explicit even
with zero features; no length-based shapes or cached admission are allowed. Current combined
execution uses the existing CPU invocation. Runtime receives no new graph metadata, protocol or
execution owner.

Ordinary Engine calls admit the Engine, validate the prepared handle's non-nullness and exact
owner, retrieve/check its exact inward delegate, then inspect inputs. Closed Engine wins all
arguments; under an open Engine, closed preparation reports `prepared execution is closed` before
null, malformed or dead inputs and without borrowing. Runtime alone arbitrates a later close/run
race through its lease below; no outward lock is held through execution.

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
operation has completed, and the published destinations are usable. A failed invocation returns no
`RunResult`. The returned result is a lease over completed run-state resources and publication
lifetime, not a future, event, fence, or in-flight command handle. It may remain open after the run
call; its later close releases run-owned resources independently of the already-released prepared-
execution lease. Materialization and transfers retain their own synchronous call boundaries.

Closing a RunResult releases publication-representation references even when RunState cleanup
fails. RunState releases borrowed/owned binding and workspace reference cells after their required
cleanup handling, without closing borrowed representations or changing reverse cleanup, unique
ownership, closed-access precedence or post-close plan/count/geometry inspection. Array dimensions
remain available; these Runtime objects remain non-thread-safe and their existing repeated-close
behavior is unchanged. This releases direct owner references, not externally retained bound objects
or required retained cleanup throwables.

The Runtime surface has no asynchronous run, cancellation, timeout, polling, callback, or explicit
completion-event contract. Concurrent calls create isolated state but make no promise of device
overlap, submission/completion order, fairness, queue behavior, or throughput. Schedule order is
defined within one invocation only.

One logical value has at most one representation per participating backend owner. A transfer
copies from the producing owner's representation, or from the first consuming owner's caller-input
representation when the value has no producer, to one distinct consuming owner's representation.
Fan-out creates one immutable transfer recipe per distinct destination owner and schedules it once
before that owner's first consumer. The source remains valid. Runtime marks the destination valid
only after successful physical transfer; failure leaves source and destination validity unchanged
and triggers ordinary reverse run cleanup with the transfer failure primary.

`PreparedExecutionRunner` owns this synchronous orchestration. It is stateless and creates one
isolated `RunState` per call; concurrent and repeated calls may share immutable prepared physical
state only through fresh backend-authenticated read-only child bindings. They never share mutable
state, writable internal/output buffers, or workspaces. `PreparedExecution` remains the unique
owner of each prepared-resource identity; a run owns only its issued child binding. Concrete
transfer recipes retain exact direction, byte geometry, and direct backend-owned cold binders.
Cold binding validates concrete representations once and creates an action with direct typed
source, destination, and staging references. Executable and transfer execution, and Runtime
publication binding, perform no Engine, registry, backend-ID, provider, availability,
configuration, discovery, reflection, or representation-map lookup.

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


### CPU0074 run-owned synchronous reservation

Implemented, independently reviewed and MAIN accepted, 2026-10-30. CPU0074 is Complete;
the [task Result](../../planning/backends/cpu/tasks/0074-row-fusion-simd-parallel.md#result)
records bounded validation and performance limitations. The [CPU mechanism owner](backend-execution.md#cpu0074-selected-row-region-execution-exception)
defines exact internal seams and eligibility; the [Scan owner](recurrent-scan.md#cpu0074-selected-terminal-grouping-exception)
defines arithmetic/activity. This subsection owns only Prepare/Run resource and budget lifetime.

Backend analysis predeclares every staging/partial byte and fixed frame/layout fact before
shared slot assignment. Finalization binds assigned storage and explicit generated helpers,
never discovers requirements. One RunState owns fresh StepDispatch, operand/frame arrays,
prebound range calls and all mutable scratch. Prepared state retains immutable recipes/handles
and borrowed CPU group/budget references only. Ordinary Runtime cold binding still precedes
schedule execution; producer-sensitive value preflight still follows producers before writes.

For a selected parallel Scan, cold run binding computes actual participant count N from the
fixed prepared numeric partition rule. Acquire max(1,N) permits from the exact coordinated
finalizer budget before the generated entry. Caller setup/merge/carry uses this same reservation.
At synchronous dispatch, lend it exclusively to the prebound worker submission: validate exact
budget identity, unreleased state and sufficient demand before enqueue; acquire/release no
additional permits there. The caller performs no concurrent math while waiting. N<2 runs inline
under one permit and submits nothing. A parallel selection requires the coordinated group/budget
pair; legacy uncoordinated ordinary calls keep their existing behavior.

The reservation is bound privately to this invocation for one synchronous call and must not be
shared with another RunState or concurrently borrowed. Only after all started workers quiesce
may it be unbound/released or may the caller consume partials, change carry/time or reuse scratch.
Normal merge/carry runs under the still-live reservation. Existing worker failure ordering,
unclaimed-work cancellation, interruption/status restoration, racing close and synchronous join
apply unchanged; workers cannot submit nested work. Reject wrong-budget, released or insufficient
reservations before publishing work. Success/failure cleanup must release the outer reservation
exactly once after quiescence. This adds neither a second execution owner nor a shared run lock.

The emitted outer Scan loop may invoke only the generation-selected typed StepDispatch seam.
Run-bound calls/arrays are allocated once and reused across T. Existing calls.clone/Job/failure/
latch/queue metadata per synchronous dispatch is the narrow per-step allocation exception, not
body/tensor/element allocation, graph interpretation, per-op lookup or geometry repacking.
Job reuse is optional, not required. A new pool, generic phase/cohort framework, Runtime service
and async protocol are not authorized. CPU0074 does not implement Job reuse or depend on it.
Whole-run completion/publication remains synchronous.

Worker segments, workspace and any staged transpose must be accessible to every participating
worker and pass existing typed spans/alignment/alias checks before writes. Stage invariants only
inside the required activity gate; changing cuts/SHARED histories use their exact active-time
binding and cannot survive beyond proved lifetimes. Join precedes consumers and scratch reuse.
No B×parameter bank, maximum-B allocation or T-fold graph/code/storage follows. Unselected scalar
paths retain existing admission, resources and lifecycle. MAIN acceptance changes no production default.

## `modules/engine`

`modules/engine` owns public lifecycle orchestration and explicit backend composition.

Allowed:

- public `Engine` facade
- public `CompiledGraph` facade
- owner-bound public `PreparedExecution` and `RunResult` handles
- positional fresh-value binding through `Engine.runValues` on the existing prepared handle
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

Current supported Engine distribution composes CPU and Metal, not the unused CUDA marker.
Keep CUDA's source/project registration for future development, without a required Engine
dependency or supported publication. Export existing inward contracts named by supported public
signatures; keep implementation-only collaborators, including Tuning runners, private.

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

For Metal, the equal-`BackendId` rule means one Engine can own at most one
`MetalBackendIntegration`. That integration has one system-default-device context and queue; its
`BackendDeviceId(metal, "default")` snapshot entry is an abstract eligibility slot, not a retained
physical-device selection. Planning retains the owning `BackendId`. Resources stay authenticated
to the integration's exact context. Separate Engines may open separate contexts, but they are not
coordinated multi-device execution and may resolve to the same default device.

`Engine.standard()` remains the ordinary CPU convenience. Every call opens one fresh default CPU
integration and transfers it through the same builder ownership path. It performs no discovery
and neither enables Metal nor changes the existing CPU lifecycle behavior.

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
encounter order; an identical throwable reference is skipped to avoid self-suppression. Engine
construction publishes no partially built Engine.

An explicit builder or Engine close with no earlier primary failure attempts all owned cleanup in
the required reverse order. Its first unchecked failure or error is primary; later distinct
failures are suppressed in cleanup encounter order, and repeated references to the primary are
skipped. Repeated close is idempotent and replays the retained primary failure.

An Engine owns its registry until Engine close. Shutdown first rejects new operations and waits
for admitted Engine operations, then closes retained results in reverse run-publication order,
ordinary and advanced prepared handles in reverse preparation-publication order,
and backend integrations in reverse registration order. Cleanup is attempt-all, idempotent, and
failure-retaining. Closing a prepared handle closes its one inward Runtime
execution without closing an already returned result. A run that already acquired Runtime's
prepared lease may finish without making close wait; its final lease release performs deferred
prepared-resource cleanup.

#### Reentrant Engine close and lease admission

The [Engine0023 correction](../../planning/modules/engine/tasks/0023-reentrant-close-and-lease-admission.md)
is implemented in the existing owners and source-qualified. A synchronous close by a thread holding
admission to that exact Engine must throw IllegalStateException before changing lifecycle or waiting, even if another close is
already pending. The thread performing Engine cleanup may reenter close idempotently; other
close callers retain completion waiting, interruption restoration and retained failure replay.
Callbacks must not synchronously wait for another thread to close the Engine whose work they serve.

The existing Engine lifecycle owner tracks exact operation admission and every terminal handoff/
rollback releases it once. No second Runtime lease, owner, executor, callback protocol or lock is
introduced. Tracking must not add per-operation token allocation or boxing. Existing representative
session threading/cleanup and reverse publication/resource/integration ownership remain intact.

PreparedExecution.RunLease and Metal context ChildLease allocation precede their existing count
increments. Failure to create a token therefore leaves no phantom lifetime that could indefinitely
defer persistent/context release. Runtime prepared close remains non-waiting and its last admitted
run remains the unique deferred cleanup trigger. Native ABI, source, kernels and identities do not change.


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
descriptor, layout, data type, and byte geometry. Unsupported direction, representation, layout,
data type, or size fails closed before backend analysis or persistent-resource acquisition. The
currently supported heterogeneous transfer domain is CPU to Metal and Metal to CPU for all seven
carriers over rank-0..16 fully static canonical or positive-stride non-overlapping layouts with
checked physical spans. It performs no conversion. The CPU run-owned native representation is the
reusable host staging storage: CPU to Metal uploads its exact physical bytes, while Metal to CPU
downloads through cold-bound staging for BOOL validation and commits only after all logical bytes
pass. No per-element object, canonical-byte materialization, or additional non-BOOL hot-path copy
is permitted.

One complete Engine preparation supplies one shared schedule assembler. Each exact owner adapter
contributes only the representation and workspace creators for positions assigned to that owner.
The assembler emits the sole optional representation-creation prefix, then partition executions in
compile order with required transfer steps immediately before each destination owner's first
consumer, and finally the dense publication suffix. Schedule construction uses the stable logical
producer/consumer facts and shared assignments; it does not rescore ownership, inspect backend
routes, or create a second scheduling convention.

On successful preparation, exactly one outward prepared handle is registered and retains immutable direct adapter lists aligned with logical caller inputs and publication
occurrences. Exactly one source-owner adapter creates each caller-input representation. Each
publication captures the adapter owning its selected representation, including mixed result
lists. Run and materialization use those direct references without querying the registry.
Transfer recipes retain only the direct backend-owned binders and native context references
selected during preparation. Engine shutdown preserves their owner lifetime by closing retained
results first, ordinary and advanced prepared handles in reverse
preparation-publication order second, and integrations last.

Preparation is transactional across all owner contributions. A finalizer retains rollback
responsibility until it returns. Shared Prepare then owns all returned persistent resources across
later finalizers, contribution assembly, transfer-recipe creation, schedule validation, Runtime
aggregate construction, and outward-handle publication. Failure closes acquired persistent owners
in reverse partition/acquisition order. Per-run representation creation is separately
transactional in `RunStateCreation`; a later creation, binding, transfer, execution, or publication
failure closes only run-owned resources in reverse acquisition order and never closes borrowed
caller storage. Closing a read-only prepared child releases its lease, not another run's child or
the still-open prepared owner.

`Engine.prepareTuned(...)` admits one eligible single-owner CPU plan or exact singleton-NEG Metal
plan. It rejects empty, missing, mixed-owner, and multiple-partition plans before representative
trial execution; Metal additionally requires the complete custom-kernel/MPSGraph local batch, so
MPSGraph-only plans are not tuned. `ALLOW_SAFE_HEURISTIC` may fall back only through the exact
already-selected adapter after a recoverable tuning failure; it never changes owner or admits a
heterogeneous trial.

The registry map has no post-prepare execution role. Runtime receives immutable schedules with
direct prepared executable, transfer, representation-creation, and publication references. Outer
Engine handles retain direct per-input and per-publication adapters. Runtime's cold-bound action
array and hot traversal never query the Engine, registry, adapter inventory, provider, availability
snapshot, configuration, backend ID, `ServiceLoader`, reflection, or a global lookup.

<a id="public-reusable-inference-session"></a>

### Public reusable prepared execution

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

`Engine.prepare` accepts only a graph compiled by that exact Engine. One Engine admission performs
one preparation and publishes exactly one outward prepared owner. Inward preparation, wrapper
construction and publication are transactional: failure rolls back acquired resources, retaining
the primary failure and suppressing distinct cleanup failures. Construction runs nothing and binds
no caller storage. The handle borrows the Engine's composition lifetime.

`CompiledGraph.inputs()` fixes membership and final Compiler input occurrence order. `Engine.run`
requires each exact compiled TensorId once in arbitrary list order and the complete exact descriptor.
Its existing open-valid-handle diagnostics and resolved-layout capacity checks remain unchanged.
For static templates, `Engine.runValues` requires exactly one value in each compiled input position, without TensorId
matching; the same fresh Tensor may fill distinct matching slots. Full descriptors must match
exactly, including dtype, fully static shape, layout presence/geometry and requiresGrad.
[64,784] cannot bind [32,784]; no implicit specialization occurs.

Prepare0013 implements an explicitly qualified ordinary homogeneous FLOAT32 CPU branch of the
same public API: one named B, fixed features/parameters, rank-at-most-two boundaries plus qualified
AD views, generated scalar and single-range execution. The
[bounded symbolic Scan contract](recurrent-scan.md#accepted-symbolic-batch-scan) is implemented by
Prepare0014 for homogeneous FLOAT32 forward and actual first-order reverse, fixed T/features/state
widths, generated scalar and single-range execution. Its task tracks final qualification.
Prepare0015 implements bounded combined length-Scan for one named B, fixed T/features, homogeneous
FLOAT32/FLOAT64 forward and actual Compiler reverse, including bounded outer typed/control/ROW(1)
closure. Owning validation and independent code/structural review passed; root validation and the sole bounded
probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending. This uses the
same prepare/run/runValues, occurrence indexing and dtype-generic publication owners, with no new
public API, Engine memo or second lease. It is not all-valid symbolic FLOAT64 qualification.
NN/TrainingSession, symbolic cold/copy/wire, native/mixed, other symbolic CPU dtype/family extensions,
variable T/features, higher-order, vector/parallel and tuning remain separately gated.
Compiler-owned `BatchBindingPlan` derives consistent B>=0 from actual
fully static descriptors in final input order, including unused positions. Fixed axes, parameters,
seeds, dtype and requiresGrad match exactly. Dynamic caller layouts must be resolved zero-offset
dense with actual canonical strides; same-address dense read views may qualify, but absent layout
is not silently canonical. Proved internal transpose/expand AD views use separate CPU mappings.
Only B/constants determine qualified expressions; output-only/anonymous symbols reject at prepare.
`run` keeps exact-ID/arbitrary-order semantics; changing B uses runValues without prototype mutation.

Dynamic order is Engine/owner/closed-prepared gates, snapshot/count/null, semantic and numeric/size
preflight, snapshot all storage associations once and validate all, borrow, Runtime lease/create/
cold bind, execute, then complete result. Before borrow/allocation check fixed axes/type/flags,
counts/strides, byte products, referenced spans/offset sums and aggregate bounds including alignment
padding and representation multiplicity. First CPU extents/counts/strides/offsets/referenced-element
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

The resource-contract null/invalid/overflow rules apply. Borrowing wraps exactly the referenced
prefix, never excess capacity or caller ownership; invalid prefixes reject, failure closes wrappers
only. Static borrowing/fast spans stay. Engine preindexes publication-to-value-list positions and
resolves only requested descriptors, with exact occurrence count. Gradient metadata comes from
generated gradient values, not target flags. Preserve tensorId/role/derivativeOrder/targetIndex,
dense order and distinct authenticated aliases. Materialization uses actual shape/layout/bytes and
returns detached HostTensorValue. Materialized outputs are canonical non-view; qualified view
publications retain actual geometry. Failure publishes nothing, closes inward result then reverse
borrows with distinct suppression, and never closes caller storage. Existing leases, close
precedence, retry and different-B run/result isolation remain; no second owner or lease is added.

Before any borrow, each call snapshots every selected storage association once and validates all
values and storage for presence, dtype, liveness, current-thread accessibility and capacity.
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
runner. They perform no prototype mutation, synthetic Tensor construction, recapture, compilation,
preparation, provider/registry lookup, backend/kernel selection or extra per-element work. The
positional path reuses ordinary per-run storage/borrow collections, not another binding map.
Concurrent calls create isolated mutable state and results, with no cross-call input retention;
there is no device-overlap, throughput, submission-order or fairness promise.

Every run returns existing `RunResult` with dense forward-then-gradient publication order.
Aliases remain distinct occurrence objects; exact publication identity authenticates explicit
materialization. Each successful materialization returns a fresh detached `HostTensorValue`.

Prepared close closes the exact inward Runtime owner once and unregisters even on cleanup failure.
Repeated/concurrent close retains failure replay and interruption restoration. It rejects new runs,
but does not wait for already leased runs or close returned results. A leased run may finish and
its last lease release performs deferred prepared-resource cleanup. Engine shutdown rejects new
operations, waits for admitted work, then closes results in reverse run-publication order,
ordinary/advanced preparations in reverse preparation-publication order, and integrations in reverse
registration order, with attempt-all cleanup and suppression. Result materialization remains valid
after prepared close while result and Engine are open. Graph/prepared metadata remains readable
after close; detached values outlive result, preparation, Engine and caller storage.

Advanced and ordinary run publication includes wrapper construction and registration in one
rollback boundary. Failure closes the completed inward result or its constructed outward owner,
never both, and releases the admitted Engine operation exactly once even if rollback fails.
Rollback runs outside lifecycleLock. The first outward result close transfers Engine-owned borrow
wrappers into its cleanup scope and drops the retained field reference, preserving reverse wrapper
cleanup, serialized materialization, concurrent-close waiting, interruption restoration and exact
outward failure replay. Runtime repeated close retains its separate existing non-replay behavior.
[Engine0022](../../planning/modules/engine/tasks/0022-result-publication-and-closed-payload-cleanup.md#result)
implements these guarantees after independent review and MAIN's before/after validation, without
adding an owner, lease, lock, factory or fault flag.

`InferenceSession` and `Engine.session` are removed without a replacement wrapper or compatibility
shim. External callers must migrate to prepare/run and recompile. Training retains its separate
stateful `TrainingSession`; no compiler, scheduler, result type, composition rule, tuning surface or
error hierarchy is added.

### Accepted Engine geometry memo

Accepted future direction on 2026-10-25, unimplemented; Model0041's authority delivery is Complete,
MAIN accepted 2026-10-25 after independent APPROVE. This is the sole metadata-only
exception to prepared immutability and invocation-private numeric facts above. It applies only
at the exact Engine `PreparedExecution` entry for a separately qualified symbolic-batch plan,
including the separately qualified Scan vertical. It does not change current static behavior or add a cache
to Runtime. Prepared graph meaning, routes, code, recipes, slots, representation positions and
persistent resource identities never change.

Retain at most one replaceable complete immutable entry, empty initially. Its key authenticates
the exact compiled/binding plan, inward preparation and memory-plan references, fixed numeric
recipes/publication mapping, explicit B and execution mode, including constant-byte `[B,0]`.
Equal B/bytes/shapes or structurally equal foreign plans cannot authenticate a hit. Retain only
covered pure Engine B-dependent numeric validity, exact sizes/strides/offsets, aligned aggregate
bounds and template-derived non-input publication metadata. No Tensor identity, supplied
descriptor, `Binding`, input/storage association, physical address, lease, mutable working array,
resource, result or failure may enter the memo. One graph-sized snapshot is allowed, not B-sized
payload, map, pool, global/ThreadLocal state or per-B executable/full-preparation cache. Replaced
snapshots survive only while referenced by live calls/results; concurrent candidate retention is
bounded by active admissions, never an accumulating history.

Every call preserves Engine/owner/closed-prepared precedence and exact inward-delegate admission,
then snapshots inputs, checks counts/nulls and establishes exact-ID or positional order. It must
perform the complete fresh Compiler bind and every private typed obligation, including unused and
repeated input occurrences. Validate every B occurrence, fixed axis, rank, dtype, requiresGrad,
layout presence, strides, offsets and view policy before considering a hit. Equality of the first
input's B is not admission. A hit avoids only covered Engine pure numeric/size evaluation and
template-only non-input descriptor reconstruction; it never skips Compiler work or unrelated checks.

Preparation preindexes complete numeric checks for all outer and scoped local domains and latent
ports, including unused/scalar-backed values; no per-run graph walk/inference or synthetic Tensor
construction is allowed. For the initially qualified CPU domain, counts/extents/element
strides/offsets/referenced spans fit CPU int. Checked bytes, aligned offsets and
representation-multiplied aggregate bounds fit long. Other backends require separately qualified
representability bounds; no additional backend capability is granted here. Empty
leading domains cannot hide overflowing trailing geometry. Numeric recipes may use checked
piecewise alignment/empty-scratch cases, never select resources/topology/routes for a particular B.

Each call snapshots current storage associations once, revalidates presence/type/liveness/current-
thread accessibility/capacity and borrows exact current prefixes. Physical type/access/alignment
and writable-overlap checks still precede execution. No hit authenticates storage or addresses.
Input-backed publications use the fresh binding's current supplied descriptor, including admitted
dense-read `isView` differences at the same B. Cache only template-derived non-input facts and
preserve publication order, tensorId/role/derivativeOrder/targetIndex, aliases and generated gradient
flags. A fresh `Binding` is not retained across calls.

Encapsulated immutable exact-plan numeric geometry and never-mutated private backing may be shared.
Mutable size/access/packed/address/carry arrays are run-owned: copy reusable facts into fresh
isolated working arrays when required. Runtime receives only authenticated numeric geometry,
never the memo, symbols or semantic metadata. `RunMemoryGeometry.resolve(plan, OptionalLong)`
remains a fresh resolver, not a hit API; its existing test that shares geometry across fresh
states is evidence of immutability, not authority for this memo. Every invocation still creates
one fresh RunState, fresh physical resources even at zero bytes, fresh writable outputs and a
distinct result under existing borrowing/cleanup rules.

Publish only a complete immutable candidate, atomically, after pre-borrow descriptor/numeric/
storage admission. Readers take a local immutable snapshot; mixed-B races may compute duplicate
candidates and either valid replacement may win, while each call keeps its own selected facts.
Failed admission publishes nothing and preserves a prior good entry. Later allocation/bind/
execution failure may retain valid metadata only, never resources/results/failures; reverse
cleanup is unchanged. Retry repeats all current input/storage checks and creates fresh resources.

Close clears the slot and prevents late republishing. Coordinate atomic replacement versus close
through the existing lifecycle, with a bounded mechanism chosen in a later exact scoped brief.
After existing inward-delegate admission, concurrent close suppresses/clears publication; it must
not add a new late outward-close rejection. Admitted calls retain local immutable facts and
Runtime's existing sole run lease arbitrates the later close/run race and error precedence.
No execution lock, second lease or new owner/protocol is permitted. Lookup/publication/copying
stays at cold Engine entry, never generated time/row/contributor loops or Runtime execution.

No new production type/public signature, Compiler refactoring or changed Runtime handoff is
selected. Exact fields/methods, atomic publication/close mechanism and any necessary protocol
change require later owner-specific approval; if a new named type is necessary, stop for exact
package/name/purpose/signature and user approval. Future tests must witness one preparation and
64→64→40→40→64 miss/hit/miss/hit/miss using bounded test-local counters for avoided Engine work,
not skipped binder obligations. Cover invalid same-B metadata/storage, current input-backed views,
B=0/1 and `[B,0]`, overflow, foreign preparations, failure/retry, close without late publication,
concurrent mixed-B snapshots/results, fresh resources/working arrays and generated loops with no
memo access. No test is newly run by this authority-only change and no speed claim follows.

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

## Profile-free Engine and Prepare lifecycle

<a id="numerical-profile-lifecycle"></a>

Engine construction offers no
numerical-profile selection or replacement mode. Compiler artifacts and `PrepareContext`
projections carry no profile field; backend preparation uses complete occurrence facts, actual
provider capability, qualified routes, and versioned non-profile compatibility identity.
Runtime and Trace remain numerical-policy-free and perform no per-run tolerance, certificate,
capability, or route lookup. Removing the selector does not change Engine ownership, Prepare
transactions, resource lifetimes, transfers, tuning, or synchronous run/publication behavior.
