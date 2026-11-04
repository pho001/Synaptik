# Analyze and finalize an owned partition

## Outcome and current scope

This guide shows a concrete backend contributor how to implement both current backend stages of
Synaptik's staged preparation handoff. Analysis accepts one validated static partition or the
qualified ordinary or all-valid Scan one-B projection and returns an opaque backend plan plus exact shared buffer
and workspace declarations.
After shared Prepare assigns Runtime slots, finalization consumes those exact assignments and
constructs an immutable executable recipe.

A **partition** is a group of graph operations assigned to one backend owner. **Analysis**
decides how that owner will realize the work and declares its resource needs; **assignment**
gives those declarations stable positions in a shared memory plan; **finalization** builds the
selected executable against those positions. Declaring a 64-byte workspace is not allocating
64 bytes, and a slot number is not a memory address.

Prepare also exposes a separate cold transport for a complete backend-owned tuning candidate
batch and an optional backend-owned selected decision. The transport keeps both values typed and
ties them to one exact planned partition, but it is not part of `PrepareContext`, analysis,
assignment, or finalization and performs no tuning work.

The current `tools/tuning` module consumes these handoffs through cold caller-supplied
collaborations. It performs cache-first deduplication, bounded warmup and sampling, deterministic
median selection, and atomic workload-cache publication without adding methods to Prepare's
opaque roles or interpreting backend fields. The CPU backend supplies the current adapter and
candidate collaborations. `Engine.prepareTuned(...)` composes them into a bounded two-phase
CPU-only workflow from an explicit `ModelAutotuningRequest` carrying Config-owned policy and
representative inputs: Phase 1 selects local workload decisions, and Phase 2 compares complete
prepared CPU plans.

Public `GraphPreparation.prepare(...)` coordinates package-internal assignment and validates one
complete schedule supplied by an explicit assembler. CPU and Metal integrations currently supply
production partition finalization, physical representation contributions, and backend-local
complete assembly. Ordinary Engine composes those contributions across single-owner or mixed
CPU/Metal plans; fixed CPU `Engine.standard()` and the advanced surface remain CPU-only. Runtime
creates every assigned physical representation per run and executes direct executable and transfer
references from the prepared schedule. Generic plugin registration remains planned. The example
therefore demonstrates the shared analysis and finalization contracts with illustrative backend
types rather than either production implementation.

## Prerequisites

- Read the authoritative [architecture contract](../../ARCHITECTURE.md), especially the Prepare,
  concrete-backend, dependency, and lifecycle sections.
- Read [ADR 0010](../design/decisions/0010-staged-backend-preparation.md) for why backend analysis
  and finalization are separated by shared slot assignment.
- Understand the immutable Model graph values and Planning partition and logical-memory recipes
  supplied in `PrepareContext`.
- Use Java 26 and the repository Gradle build. No native toolchain is needed for the contract-only
  example below.

## Terms

- A [`PrepareContext`](../glossary.md#prepare-context-preparecontext) is the immutable,
  partition-scoped projection that shared Prepare validates before calling a backend.
- A [`PartitionDag`](../glossary.md#partition-dag-partitiondag) is the context's immutable local
  topology. It preserves node and port occurrences without adding backend policy.
- A [backend partition analysis](../glossary.md#backend-partition-analysis-backendpartitionanalysis)
  is the result of route selection and exact resource declaration, before any slot exists.
- A [preparation resource requirement](../glossary.md#preparation-resource-requirement) declares
  exact bytes and alignment for either a graph-value buffer or analysis-local workspace.
- A [preparation resource assignment](../glossary.md#preparation-resource-assignment) retains one
  exact requirement, its assigned Runtime slot, and its dense prepared-plan index.
- A [backend partition finalization](../glossary.md#backend-partition-finalization) supplies one
  typed analysis and its complete assignments to the owning backend.
- An [opaque backend analysis role](../glossary.md#opaque-backend-analysis-roles) keeps
  backend-specific input and selected-plan fields typed without making shared Prepare interpret
  them.
- An [opaque backend tuning handoff](../glossary.md#opaque-backend-tuning-handoff) retains one
  exact partition, one typed candidate batch, and an optional typed decision without exposing
  backend fields.

“Opaque” means that shared preparation can carry a typed backend value without reading its
private fields. A **buffer** represents a logical graph value; a **workspace** is temporary
working storage. **Alignment** requires an address to be a multiple of a specified byte count.
These distinctions explain why buffer identity, workspace identity and assigned position are
kept separate even when their numeric IDs happen to match.

## Ownership and lifecycle

Read this flow from backend-neutral compile facts toward reusable runtime state:

```text
capability reporting and Planning ownership
  -> PartitionDag + PrepareContext for one PlannedPartition  current
  -> BackendPartitionPreparer.analyze                        current
  -> BackendPartitionAnalysis + exact requirements           current
  -> shared BufferSlot/WorkspaceSlot assignment              current, package-internal
  -> same backend finalizes the opaque plan                  current implementation
  -> PreparedPartition + PreparedExecutable                  current implementation
  -> persistent prepared-resource transaction               current implementation
  -> explicit assembler + Prepare schedule validation        current shared contract
  -> Runtime creates per-run physical representations        current shared/backend contract
  -> Runtime executes the prepared schedule                  current shared runner contract
```

Planning chooses an owner such as CPU. Shared Prepare validates and projects that owner's exact
partition facts. The concrete backend then owns deterministic lowering, fusion, route selection,
and private configuration. Shared Prepare sees only the typed opaque plan and backend-neutral
resource declarations. Runtime owns the stable slot and memory-plan types; shared Prepare owns the
source assignment and finalizer handoff. Concrete per-run population and executable work remain
backend responsibilities.

This division keeps `CompileArtifacts` and other Compiler-owned types out of the concrete backend
surface. It also keeps route selection, measurement, cache mutation, allocation, and graph
inspection out of Runtime.

The tuning handoff is a side transport around this lifecycle, not another arrow in it. A concrete
backend produces the complete immutable batch and later validates any supplied decision against a
fresh batch. Shared Prepare retains exact references only. It does not enumerate the batch,
match the decision, serialize either value, or insert them into Runtime.

An owning composition may pass the handoff to `WorkloadTuning` together with a
`BackendWorkloadTuning<C, D, K>` implementation and a `ColdCandidateMeasurement<C, K>` action.
The collaboration supplies backend-owned compatibility and candidate identities, complete
candidate enumeration, decision construction and codec operations; the measurement action runs
one complete candidate on equivalent representative inputs. Tuning returns a new handoff for each
original occurrence with the selected decision present. The concrete backend must still validate
that decision against a freshly generated compatible batch before it can affect analysis.

## General-scan projection and resource handoff

A scan **formal** is a named input role inside the repeated body; an **actual** is the outer
value bound to that role. A **carry** passes state from one step to the next. A **residual**
is a body value made available to save for a consumer, and a **history** stores demanded
per-step values. Declaring a residual port does not require saving every temporary.

`GraphPreparation` now preserves each scan's exact compiled body and ordered role bindings inside
its selected partition. A residual port names a body value that may be saved as history; demand
means an outer publication or consumer uses a port. Before any analyzer runs, projection checks
demand against final graph outputs and all outer input uses, including other partitions.
`PrepareContext` checks every local descriptor
for static shape or the occurrence's single named batch dimension B and validates formal/actual and outer-role descriptors, even for unused formals,
sibling results and zero time steps. A directly constructed context checks locally visible uses
and supplied logical obligations; it cannot verify unseen external edges. Local numeric IDs may
coincide with outer IDs without identifying the same values. Model/Compiler retain ordinary body
inference and automatic differentiation (AD); backends still qualify execution.

All-valid symbolic Scan projection retains fixed time/features and unresolved symbolic
formal/result layouts; static formal/results remain canonical. The bounded rank-three outer
sequence/history and AD projection is not arbitrary-rank backend support. `requiresBatchSize`
is explicit even when every byte recipe has zero slope, as with `[B,0]`. Analysis declares exact
fixed or run-size requirements before slots; assignment fixes identities and representation
positions without evaluating CPU recipes. The CPU binds actual B later, not a sample B or a
new resource topology. [Prepare0014](../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md)
tracks the implemented FLOAT32 execution slice and its final qualification separately.

Length-aware projection uses those same owners. T is the static time count and B the batch extent.
The final non-gradient INT64 `[B]` lengths actual remains an outer input, never a body formal or
gradient contributor. Prepare validates T/B,
homogeneous FLOAT32/FLOAT64 locals and declared ROW(0), rank-two ROW(1) or SHARED boundaries,
including tied actual positions and empty domains. SHARED has no inferred batch axis, even when
its extent equals B. Actual layouts, including lengths, are retained; CPU read-layout predicates
do not belong in shared validation. Generated reverse scans may mix these roles and append
SHARED parameter accumulators without inheriting Compiler's restricted original-source AD admission.

The remaining pipeline is unchanged: analysis declares outer-value buffers and analysis-local
scratch workspaces; shared Prepare assigns slots; finalization receives those exact declarations
and assignments. Shared buffers use maximum declared size/alignment across participating owners;
each workspace must match its declaration exactly. A body-local ID is not an outer buffer ID,
and numeric coincidence supplies no local-storage provenance.

For example, consider a scalar FLOAT32 body with three time steps, a published final carry and a
residual consumed by a different backend owner's partition. If these are its only outer uses,
only those ports are demanded. A test backend can declare the residual's `3 × 4 = 12` bytes at both
partitions and its scratch separately; both residual declarations receive one shared buffer slot
with owner-specific representation positions. Missing producer or consumer declarations reject
before finalization.
At zero steps a zero-byte residual declaration is legal, but descriptor and demand checks still
run. These are metadata/resource-handoff guarantees, not physical history or numerical execution
evidence.

For a length-mode FLOAT64 fixture with T=3 and batch, feature and hidden extents B=F=H=2,
lengths requires `2 × 8 = 16` bytes.
A demanded SHARED transposed weight history `[T,F,H]` requires `3 × 2 × 2 × 8 = 96` bytes,
not a 192-byte `[T,B,F,H]` bank. A test-only backend can declare two banks for one `[B,H]` row
carry and one `[F,H]` shared accumulator: `2 × (4 + 4) × 8 = 128` workspace bytes. This is the
fixture's geometry, not a CPU policy. It also declares staging/temporaries before assignment;
dead histories have no declaration. Finalizers receive exact declarations, slots and owner
representation positions; missing cross-partition declarations fail before finalization.
All fixture creators throw if invoked, and no executable runs. Real CPU length-mode lowering,
runtime length validation and skipping remain closed; qualified all-valid CPU support is unchanged.
Shared Prepare cannot detect hidden allocations inside an opaque backend executable. See
[general-scan resource and execution boundaries](../architecture/general-differentiable-scan.md#capability-resources-and-generated-code).

## Current-contract example

### Inputs and initial state

Assume shared Prepare has constructed a `PrepareContext<CpuInputs>` for one CPU-owned operation:

- the operation has one FLOAT32 input and one FLOAT32 output;
- both values have fully static Shape `[2, 3]`;
- each direct CPU representation therefore needs `2 × 3 × 4 = 24` bytes;
- the projected lists contain the exact partition nodes, unique values, and one matching logical
  memory requirement per projected value;
- the context has no logical-splat constants; and
- `CpuInputs(alignment=4)` is an immutable backend input selected for this target.

The following is current Java API, but the sample backend is illustrative rather than a
production CPU implementation:

```java
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.List;
import java.util.Objects;

record CpuInputs(long alignment) implements BackendAnalysisInputs {}

record CpuPlan(String route, long alignment) implements BackendPreparationPlan {}

final class CpuPreparer
        implements BackendPartitionPreparer<CpuInputs, CpuPlan> {
    @Override
    public BackendPartitionAnalysis<CpuPlan> analyze(
            PrepareContext<CpuInputs> context) {
        Objects.requireNonNull(context, "context");
        PartitionDag dag = context.partitionDag();
        if (dag.nodes().size() != 1
                || context.values().size() != 2
                || context.backendInputs().alignment() != 4) {
            throw new IllegalArgumentException("unsupported example context");
        }

        ValueId inputId = dag.externalInputs().getFirst().valueId();
        ValueId outputId = dag.localSinks().getFirst().outputs().getFirst();
        CpuPlan plan = new CpuPlan("vector", 4);

        return new BackendPartitionAnalysis<>(
                context.partition(),
                plan,
                List.of(
                        new PreparationResourceRequirement.Buffer(inputId, 24, 4),
                        new PreparationResourceRequirement.Buffer(outputId, 24, 4),
                        new PreparationResourceRequirement.Workspace(0, 64, 16)));
    }
}
```

### Meaningful steps

- The generic marker roles keep `CpuInputs` paired with `CpuPlan`; the implementation needs no
  cast, raw `Object`, or string-keyed parameter map.
- `partitionDag()` is the sole retained topology source. This one-node example reads its external
  input occurrence and local sink directly; a multi-node backend can also query exact producer,
  consumer, and edge ports without reconstructing adjacency.
- The preparer rejects any context outside this example's supported shape before returning a
  partial result. A production backend would validate its complete supported operation,
  descriptor, capability, and configuration contract.
- The plan records the selected `"vector"` route opaquely. Shared Prepare can retain it for later
  finalization but cannot inspect its fields.
- The two buffer declarations use graph `ValueId` values only to associate requirements with
  projected logical values. Those IDs are not Runtime slots.
- Workspace ID `0` is unique only within this analysis result. Its 64-byte size and 16-byte
  alignment are exact declarations, not an allocation.

### Result and interpretation

Calling the preparer twice with the same complete context produces equal analysis values in this
example. Each result retains the exact `context.partition()` reference, the selected `CpuPlan`,
and declarations in the supplied order:

```text
input buffer:   ValueId(input),  24 bytes,  4-byte alignment
output buffer:  ValueId(output), 24 bytes,  4-byte alignment
workspace:      local ID 0,      64 bytes, 16-byte alignment
```

The result proves that one supported deterministic analysis is available and tells shared
preparation exactly which resources a later slot plan must cover. It does not assign a slot,
allocate memory, create an executable, perform a measurement, or execute the operation.

### Optional opaque candidate transport

The following current API shape illustrates the separate transport. The records are illustrative
backend-owned values; their fields are invisible through the method-free shared roles.

```java
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import java.util.List;
import java.util.Optional;

record CpuCandidates(List<String> routes) implements BackendTuningCandidateBatch {
    CpuCandidates {
        routes = List.copyOf(routes);
    }
}

record CpuDecision(String route) implements BackendTuningDecision {}

CpuCandidates candidates = new CpuCandidates(List.of("scalar", "vector"));
CpuDecision decision = new CpuDecision("vector");
var handoff = new BackendPartitionTuningHandoff<>(
        context.partition(), candidates, Optional.of(decision));
```

`handoff.partition()` is the exact `context.partition()` reference,
`handoff.candidateBatch()` is the exact `candidates` reference, and the optional contains the
exact `decision` reference. The record snapshots no route list itself; the backend-owned batch is
responsible for its own immutability. This result proves only type-safe partition association and
transport. It does not prove that `"vector"` is present, valid, compatible, measured, or selected
for execution. The owning backend must validate those meanings against a freshly generated
complete batch.

### Failure variation

A `Buffer` or `Workspace` rejects a negative byte size and an alignment that is not a positive
power of two. `BackendPartitionAnalysis` also rejects a repeated buffer `ValueId` or repeated
workspace requirement ID. A backend that cannot realize the complete context should throw
`IllegalArgumentException` rather than return an incomplete plan or defer route fallback to
Runtime.

### Current assignment-to-finalization continuation

After every analysis in the complete partition set passes shared validation, the package-internal
Prepare handoff creates one shared memory plan and one ordered assignment per declaration. A
concrete backend implements only the public finalizer collaboration; it neither calls nor replaces
the package-internal assignment operation.

This illustrative executable has no resource selections because the example stops at immutable
recipe construction. A production executable would derive its dense selections from the supplied
assignments and implement checked representation binding through Runtime's protected hooks.

```java
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizationResult;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizer;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.PreparedPartition;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.List;

final class CpuExecutable extends PreparedExecutable {
    CpuExecutable(PreparedMemoryPlan memoryPlan) {
        super(memoryPlan, List.of(), List.of());
    }

    @Override
    protected boolean acceptsBufferRepresentation(
            int selectionIndex, BufferRepresentation representation) {
        return false;
    }

    @Override
    protected boolean acceptsWorkspaceRepresentation(
            int selectionIndex, WorkspaceRepresentation representation) {
        return false;
    }

    @Override
    protected BoundInvocation bindCompatible(
            RunState runState,
            BufferRepresentation[] buffers,
            WorkspaceRepresentation[] workspaces) {
        throw new UnsupportedOperationException("the example does not execute");
    }
}

final class CpuFinalizer implements BackendPartitionFinalizer<CpuPlan> {
    private final BackendId backendId;

    CpuFinalizer(BackendId backendId) {
        this.backendId = backendId;
    }

    @Override
    public BackendId backendId() {
        return backendId;
    }

    @Override
    public BackendPartitionFinalizationResult finalizePartition(
            BackendPartitionFinalization<CpuPlan> finalization) {
        if (!finalization.analysis().plan().route().equals("vector")) {
            throw new IllegalArgumentException("unexpected example route");
        }
        return new BackendPartitionFinalizationResult(
                new CpuExecutable(finalization.memoryPlan()));
    }
}
```

For the analysis returned above, the shared handoff constructs equivalent current values in this
order. This direct construction illustrates the public value contracts; production shared Prepare
performs it only after validating the complete ordered partition set.

```java
var inputRequirement =
        (PreparationResourceRequirement.Buffer) analysis.requirements().get(0);
var outputRequirement =
        (PreparationResourceRequirement.Buffer) analysis.requirements().get(1);
var workspaceRequirement =
        (PreparationResourceRequirement.Workspace) analysis.requirements().get(2);

BufferSlot inputSlot = new BufferSlot(0);
BufferSlot outputSlot = new BufferSlot(1);
WorkspaceSlot workspaceSlot = new WorkspaceSlot(0);
PreparedMemoryPlan memoryPlan =
        new PreparedMemoryPlan(
                List.of(
                        new PreparedMemoryPlan.BufferEntry(inputSlot, 24, 4),
                        new PreparedMemoryPlan.BufferEntry(outputSlot, 24, 4)),
                List.of(new PreparedMemoryPlan.WorkspaceEntry(workspaceSlot, 64, 16)));

BackendPartitionFinalization<CpuPlan> finalization =
        new BackendPartitionFinalization<>(
                analysis,
                memoryPlan,
                List.of(
                        new PreparationResourceAssignment.Buffer(
                                inputRequirement, inputSlot, 0, 0),
                        new PreparationResourceAssignment.Buffer(
                                outputRequirement, outputSlot, 1, 0),
                        new PreparationResourceAssignment.Workspace(
                                workspaceRequirement, workspaceSlot, 0)));

BackendId cpu = context.partition().owner();
BackendPartitionFinalizationResult finalized =
        new CpuFinalizer(cpu).finalizePartition(finalization);
PreparedExecutable executable = finalized.executable();
PreparedPartition prepared =
        new PreparedPartition(context.partition(), executable);
```

`prepared.partition()` is the exact analyzed partition, and
`prepared.executable().memoryPlan()` is the exact `memoryPlan` object. The assignment list follows
analysis declaration order and retains the exact requirement, slot, and owner-specific
representation position. This single-owner example uses position zero for both buffers. This
proves the current typed finalization handoff. `finalized.resources()` is empty because this
illustrative
finalizer acquired no persistent resource. Public orchestration performs equivalent assignment
and finalization internally, then owns every returned resource transactionally while it assembles
and validates the schedule. The snippet does not prove per-run physical allocation, binding, or
execution.

### Current complete-graph orchestration

The public orchestration input keeps each backend's analysis and finalization generic types
paired. One preparation value is supplied for each compile partition, in compile-partition order:

```java
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import java.util.List;

PartitionPreparation<CpuInputs, CpuPlan> cpuPreparation =
        new PartitionPreparation<>(cpuInputs, new CpuPreparer(), new CpuFinalizer(cpu));

// Application composition code supplies a complete immutable schedule recipe.
PreparedScheduleAssembler assembler = context -> assembleExampleSchedule(context);

PreparedExecution execution =
        GraphPreparation.prepare(artifacts, List.of(cpuPreparation), assembler);
```

`artifacts`, `cpuInputs`, and `assembleExampleSchedule` are deliberately illustrative inputs.
Production Engine obtains artifacts through `GraphCompilationPort`, asks CPU only for artifact-free
`PartitionPreparation` and schedule-assembler collaborations, and invokes this same shared
operation. Prepare first constructs all `PrepareContext` values, then analyzes and finalizes every
partition in order, calls the assembler once with a complete immutable
`PreparedScheduleContext`, validates the returned recipe, and returns the exact
`PreparedExecution`. The schedule context exposes stable projected facts rather than the Compiler
aggregate.

The assembler must describe every bindable input with exactly one `CallerInput` in compiler input
order. A compile-time logical splat instead needs at least one `InitializedBuffer` and no caller
input. The backend creator materializes that constant into a fresh run-owned representation;
Runtime records only that the representation starts valid. Prepare also validates executable
coverage and order, transfer and executable representation coordinates, and ordered forward then
gradient publications. It invokes no creator and performs no physical work.

## Context and requirement invariants

`PartitionDag` snapshots membership while retaining the exact node references. Node IDs must
match `PlannedPartition.nodeIds()` exactly and in order. Producer and consumer occurrences retain
their zero-based node and output/input port positions. Repeated inputs remain separate consumer
and edge entries, and every output of a multi-output node remains separately addressable. An
external input has no local producer. A local sink has no output consumed locally, regardless of
publication or cross-partition use.

These are topology-only facts. They do not describe graph regions, cross-partition ownership,
transfer, publication, materialization, memory policy, fusion legality, route selection,
scheduling, or execution. `PrepareContext` stores the DAG as its only node/topology state and
snapshots its other lists and map before analysis. Its canonical record shape has five
components; the six-argument partition-and-node-list constructor preserves source calls only and
does not promise binary compatibility with the former record shape.

Projected values are unique by `ValueId`; every node input and output must resolve to one of
them. Every projected value has exactly one descriptor-matching `LogicalMemoryRequirement`, but
the contract does not add a separate rule that every projected value must appear in a node input
or output.

Projected Shapes are fully static except for the ordinary rank-at-most-two single named-B
projection; expressions, unknown axes and general Scan dynamic geometry remain unsupported here.
`PrepareContext.requiresBatchSize()` records symbolic context even when every byte recipe is
constant, as in `X[B,0].sum()`. A projected logical-splat constant is
permitted only for a projected graph input, and its `ScalarValue` data type must match the input
descriptor exactly. The constant map is immutable metadata; analysis does not materialize its
logical repetitions.

Resource declarations retain `PreparedMemoryPlan.ByteSize`: fixed bytes or an immutable run-size
recipe. Existing long constructors wrap `FixedByteSize`; `byteSize()` rejects a run-bound
declaration rather than guessing. Prepare never evaluates a recipe. Fixed buffer aggregation
retains maximum size/alignment; run-bound buffer declarations must agree structurally and in
alignment, and workspaces agree exactly. Runtime creators now take actual
`(byteSize, byteAlignment)`; migrate static SAM consumers without a zero-argument adapter.
Resource sizes are non-negative byte counts, so zero is valid. Alignments are positive powers of
two measured in bytes. Buffer and workspace identities occupy separate domains. Current initial
assignment gives each workspace declaration its own stable Runtime slot; reuse and lifetime
interference are not represented by the current API.

## Failures, ownership, and concurrency

An owner must release a resource; a borrower may use it only within the owner's permitted
lifetime. A **lease** keeps an admitted use valid until that use finishes. **Rollback** here
releases resources acquired during a failed preparation or setup; it is not a promise to undo
arithmetic already performed by earlier operations in a run.

- Reject an unsupported partition owner, operation, descriptor, capability, configuration, or
  cached decision before producing an analysis result.
- Treat backend input and plan implementations as immutable values. Shared Prepare retains their
  exact references and does not copy or synchronize backend-private state.
- Treat tuning-batch and decision implementations as immutable values. Their shared marker roles
  expose no method, and the handoff rejects a null partition, batch, or optional container while
  retaining exact references and acquiring no ownership.
- Make `analyze` deterministic from its complete context. Do not measure candidates, mutate a
  cache, allocate physical resources, compile native executables, or retain per-run bindings.
- Do not return a `PreparedExecutable`, slot, address, storage handle, or resource lifetime from
  analysis.
- Finalization may acquire immutable persistent prepared resources, such as a compiled native
  executable, only after shared slot assignment. Before a complete finalizer result returns, the
  backend owns rollback. After return, shared Prepare owns the unique resource identities until a
  successfully constructed `PreparedExecution` accepts ownership; any intervening failure closes
  them in reverse acquisition order.
- The current CPU finalizer returns an empty persistent-resource list. CPU buffer and workspace
  representations are separate per-run physical resources created by CPU-owned schedule recipes,
  retained by one Runtime `RunState`, and closed when its Runtime `RunResult` lease closes or
  failed setup or execution rolls the state back.

The contracts are immutable and can be shared safely when the backend's marker-role
implementations are themselves immutable. One prepared execution may be reused by concurrent
runs; each run has an isolated `RunState` and separately owned mutable physical representations.
Persistent resources remain immutable and are released by the prepared execution's explicit,
idempotent close lifecycle after admitted runs release their leases.

## Registration, diagnostics, and validation

Current ordinary composition is explicit: `Engine.standard()` owns one fresh CPU integration,
while `Engine.builder()` can own opened CPU and/or Metal integrations. Builder preparation
preflights all owners/transfers, supplies every partition's exact preparation to shared Prepare,
collects owner-indexed physical contributions, and composes one execution/transfer/publication
schedule. Advanced CPU-only composition remains explicit through
`AdvancedEngine.takeOwnership(...)`. No surface discovers plugins; do not add runtime service
lookup, reflection, or `ServiceLoader` as a substitute.

Prepare/trace payloads and emitters are also planned. A backend may design typed diagnostic facts
for its analysis, but it must not leak business logic into trace data-transfer objects or use a
string map as the primary trace model.

For a concrete implementation, add:

- focused unit tests for supported analysis, failure order, immutable results, exact partition
  identity, deterministic ordering, and duplicate rejection;
- architecture validation when dependencies change;
- backend-conformance tests for every concrete backend capability claimed through the shared
  lifecycle;
- integration tests for fixed or explicit Engine composition and end-to-end execution; and
- native resource and platform validation when finalization begins owning native resources.

For the current shared contract, run:

```bash
./gradlew :modules:prepare:test
./gradlew :modules:prepare:javadoc
```

## Limitations and related documentation

The preparation lifecycle supports the bounded ordinary named-B handoff, not general dynamic
binding or workspace reuse. Compiler owns semantic admission, CPU numeric recipes and Runtime
per-run state without Shapes. It does
support exact shared-resource declarations, backend-owned per-run representation recipes, and
persistent-resource transfer, but the concrete backend owns the physical mechanics and Runtime
owns the per-run state. Fixed CPU `Engine.standard()`, explicit CPU/Metal builder composition
including bounded mixed-owner schedules, and advanced CPU composition are current; generic plugin
registration and executable persistence are not. Mixed CPU/Metal values have deterministic
owner-indexed representations and exact all-seven-carrier, rank-0..16 static canonical or positive-
stride non-overlapping storage-layout transfer recipes with checked physical spans. The current CPU
complete-plan tuning producer is session-scoped, so Phase 2 does not persist or reuse a complete
prepared plan across sessions.

The separate tuning tool provides bounded local measurement, selection, reusable workload-cache
persistence, and rich evidence around the opaque handoff. Current Engine composition supplies the
CPU collaborations for a bounded two-phase workflow. Prepare itself still performs no
measurement, selection, cache access, persistence, model aggregation, or tuning-aware graph
preparation. Slot assignment, finalization input/result, prepared partition, explicit schedule
assembly and validation, prepared-execution construction, and shared runner contracts are
current. Compatible cached decisions may be explicit immutable backend inputs, but analysis
neither loads nor mutates a cache.

See the [Runtime/Prepare/Backend boundary](../architecture/runtime-prepare-backend-boundary.md),
[Planning ownership and partition scoring](../architecture/partition-scoring.md), and
[kernel routes](kernel-routes.md).

## Numerical-semantics admission

A preparer receives complete occurrence facts in `PrepareContext`, not a graph-wide numerical
selector. Model alone defines each family/dtype formula, exact stored and discrete boundaries, and
named arithmetic-site permissions. Preparation rejects unsupported occurrences before route
analysis; each retained route must qualify against Model semantics, with finite tolerance used
only in tests. CPU keeps its qualified generated and OpenBLAS routes and non-profile identities.
Metal admits exact unary, affine, canonicalization, indexing, BOOL-domain, Task-0059 raw
movement, Task-0060 replacement/fold/aggregate, Task-0063 ordering/top-K/numeric arg-extrema,
promoted integral MATMUL, Task-0064 maximum-pooling, and Task-0065 raw INITIAL_STATE rows. Metal
also admits the established arithmetic/reduction/scan rows,
same-type canonical positive-rank FLOAT32 MSE, every positive-static FLOAT32 MATMUL geometry,
no-gradient mixed BFLOAT16/FLOAT32 MATMUL, Task-0064 convolution/average-pooling, Task-0065 FLOAT32
dropout, and the exact Task-0069 rank-one FLOAT32 L1_NORM, SCATTER_ADD, and singleton VARIANCE
domains. The preparer authenticates a local MATMUL affine operand as the exact identity-prefix,
last-two-axis transpose of its canonical source; keeps canonical BOOL values available to admitted
consumers; retains exact storage layouts for SELECT/positive-step SLICE; and declares only the
authenticated compact materialized set for a shared custom-program route. It lowers MSE to one
typed MPSGraph node whose native recipe is fixed subtraction, self-multiplication, and optional full
SUM or MEAN. It lowers Task-0060 scatter preflight, signed slice placement, single-writer non-overlap
folds, exact modular/logical reductions, every Task-0063 row, and selected Task-0066/Task-0069
occurrences to their fixed custom programs without atomics or fallback. Every partition containing
BFLOAT16 or FLOAT16 values uses that custom-only route, including homogeneous no-gradient
RESHAPE, simple PERMUTE, materializing CONTIGUOUS, SLICE, CONCAT, and TILE. A bounded canonical
no-gradient BFLOAT16 or FLOAT16 EXP uses its own typed custom step. A separate FLOAT32 EXP may use
an internal MPSGraph boundary step on an independent FLOAT32 branch or after an explicit
low-to-FLOAT32 cast. Bounded canonical no-gradient FLOAT32 SIGMOID likewise uses a sign-guarded
composed MPSGraph step on either FLOAT32 path; separately admitted low-valued SIGMOID uses
its typed custom step in the same fixed custom partition. The
preparer may replace only a qualified private MATMUL/Conv2d suffix with one typed anchor
instruction. It preserves typed
ingress, target and internal physical byte geometry, window/index obligations, unsigned-32-bit
geometry and the schema-twenty/version-thirty-six route identity without
widening capability. Every other identity fails closed.

Metal's one closed prepared-route identity owns the existing candidate wires `1..3` and the
MPSGraph/custom-kernel family. Candidate serialization delegates to it. A returned plan retains one
final route before finalization; finalization and Runtime cannot reselect it. The separate
package-private 115-kind structural catalog is descriptive cold metadata, not admission or route
approval. Its package-private forcing seam is test-only, runs normal capability and fresh-handoff
authentication first, and accepts only a route already present in the exact regenerated candidate
batch.
