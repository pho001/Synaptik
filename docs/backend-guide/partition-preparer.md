# Analyze and finalize an owned [partition](../glossary.md#partition)

## Outcome and current [scope](../glossary.md#term-scope)

This guide shows a [concrete backend](../glossary.md#concrete-backend) contributor how to implement both current [backend](../glossary.md#backend) stages of
Synaptik's staged preparation [handoff](../glossary.md#term-handoff). Analysis accepts one validated static [partition](../glossary.md#partition) or the
qualified ordinary or all-valid [Scan](../glossary.md#term-cumulative-scan) one-B projection and returns an [opaque](../glossary.md#term-opaque) backend plan plus exact shared [buffer](../glossary.md#term-buffer)
and workspace declarations.
After shared [Prepare](../glossary.md#prepare) assigns [Runtime](../glossary.md#term-runtime) slots, finalization consumes those exact assignments and
constructs an [immutable](../glossary.md#term-immutable) executable recipe.

A **partition** is a group of [graph](../glossary.md#graph) operations assigned to one backend [owner](../glossary.md#term-owner). **Analysis**
decides how that owner will realize the work and declares its resource needs; **assignment**
gives those declarations stable positions in a shared memory plan; **finalization** builds the
selected executable against those positions. Declaring a 64-byte workspace is not allocating
64 bytes, and a slot number is not a memory address.

Prepare also exposes a separate cold transport for a complete backend-owned [tuning candidate](../glossary.md#candidate-batch)
[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) and an optional backend-owned selected decision. The transport keeps both values typed and
ties them to one exact planned partition, but it is not part of [`PrepareContext`](../glossary.md#prepare-context-preparecontext), analysis,
assignment, or finalization and performs no tuning work.

The current `tools/tuning` [module](../glossary.md#term-module) consumes these handoffs through cold caller-supplied
collaborations. It performs [cache](../glossary.md#term-cache)-first deduplication, bounded [warmup](../glossary.md#term-warmup) and [sampling](../glossary.md#term-sampling-machine-learning-and-hyperparameters), deterministic
[median](../glossary.md#term-median) selection, and [atomic](../glossary.md#resource-transaction-and-rollback) workload-cache [publication](../glossary.md#term-publication) without adding methods to Prepare's
opaque roles or interpreting backend fields. The [CPU](../glossary.md#term-cpu) backend supplies the current adapter and
candidate collaborations. `Engine.prepareTuned(...)` composes them into a bounded two-phase
CPU-only workflow from an explicit `ModelAutotuningRequest` carrying Config-owned policy and
representative inputs: Phase 1 selects local workload decisions, and Phase 2 compares complete
prepared CPU plans.

Public `GraphPreparation.prepare(...)` [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix) package-internal assignment and validates one
complete schedule supplied by an explicit assembler. CPU and [Metal](../glossary.md#term-metal) integrations currently supply
production partition finalization, physical representation contributions, and backend-local
complete assembly. Ordinary [Engine](../glossary.md#term-engine) composes those contributions across single-owner or mixed
CPU/Metal plans; fixed CPU `Engine.standard()` and the advanced surface remain CPU-only. Runtime
creates every assigned physical representation per [run](../glossary.md#term-run) and executes direct executable and transfer
references from the [prepared schedule](../glossary.md#prepared-schedule--preparedschedule). Generic [plugin](../glossary.md#term-plugin) registration remains planned. The example
therefore demonstrates the shared analysis and finalization contracts with illustrative backend
types rather than either production implementation.

## Prerequisites

- Read the authoritative [architecture contract](../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([architecture contract glossary](../glossary.md#architecture-contract))</span>, especially the Prepare,
  concrete-backend, dependency, and lifecycle sections.<span data-glossary-navigation="true"> ([Prepare glossary](../glossary.md#prepare); [backend glossary](../glossary.md#backend); [dependency glossary](../glossary.md#term-dependency); [lifecycle glossary](../glossary.md#lifecycle))</span>
- Read [ADR 0010](../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span> for why backend analysis
  and finalization are separated by shared slot assignment.
- Understand the immutable Model graph values and Planning partition and logical-memory recipes
  supplied in <span data-glossary-navigation="true"> ([immutable glossary](../glossary.md#term-immutable); [Model glossary](../glossary.md#term-model); [graph glossary](../glossary.md#graph); [Planning glossary](../glossary.md#term-planning); [partition glossary](../glossary.md#partition))</span>[`PrepareContext`](../glossary.md#prepare-context-preparecontext).
- Use Java 26 and the repository Gradle build. No native toolchain is needed for the contract-only
  example below.<span data-glossary-navigation="true"> ([Gradle glossary](../glossary.md#term-gradle); [toolchain glossary](../glossary.md#term-toolchain))</span>

## Terms

- A [`PrepareContext`](../glossary.md#prepare-context-preparecontext) is the immutable,
  partition-scoped projection that shared Prepare validates before calling a backend.<span data-glossary-navigation="true"> ([immutable glossary](../glossary.md#term-immutable); [Prepare glossary](../glossary.md#prepare))</span>
- A [`PartitionDag`](../glossary.md#partition-dag-partitiondag) is the context's immutable local
  topology. It preserves node and port occurrences without adding backend policy.<span data-glossary-navigation="true"> ([topology glossary](../glossary.md#term-topology); [node glossary](../glossary.md#node))</span>
- A [backend partition analysis](../glossary.md#backend-partition-analysis-backendpartitionanalysis)
  is the result of route selection and exact resource declaration, before any slot exists.
- A [preparation resource requirement](../glossary.md#preparation-resource-requirement) declares
  exact bytes and alignment for either a graph-value buffer or analysis-local workspace.<span data-glossary-navigation="true"> ([graph glossary](../glossary.md#graph))</span>
- A [preparation resource assignment](../glossary.md#preparation-resource-assignment) retains one
  exact requirement, its assigned Runtime slot, and its dense prepared-plan index.<span data-glossary-navigation="true"> ([Runtime glossary](../glossary.md#term-runtime))</span>
- A [backend partition finalization](../glossary.md#backend-partition-finalization) supplies one
  typed analysis and its complete assignments to the owning backend.
- An [opaque backend analysis role](../glossary.md#opaque-backend-analysis-roles) keeps
  backend-specific input and selected-plan fields typed without making shared Prepare interpret
  them.
- An [opaque backend tuning handoff](../glossary.md#opaque-backend-tuning-handoff) retains one
  exact partition, one typed candidate batch, and an optional typed decision without exposing
  backend fields.<span data-glossary-navigation="true"> ([candidate batch glossary](../glossary.md#candidate-batch))</span>

“Opaque” means that shared preparation can [carry](../glossary.md#term-carry) a typed backend value without reading its
private fields. A **[buffer](../glossary.md#term-buffer)** represents a logical [graph value](../glossary.md#graph-value); a **workspace** is temporary
working storage. **[Alignment](../glossary.md#term-alignment)** requires an address to be a multiple of a specified byte count.
These distinctions explain why buffer [identity](../glossary.md#term-identity-meanings), workspace identity and assigned position are
kept separate even when their numeric IDs happen to match.

## [Ownership](../glossary.md#term-ownership-meanings) and [lifecycle](../glossary.md#lifecycle)

Read this flow from [backend](../glossary.md#backend)-neutral [compile](../glossary.md#compile) facts toward reusable [runtime](../glossary.md#term-runtime) state:

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

[Planning](../glossary.md#term-planning) chooses an [owner](../glossary.md#term-owner) such as [CPU](../glossary.md#term-cpu). Shared [Prepare](../glossary.md#prepare) validates and projects that owner's exact
[partition](../glossary.md#partition) facts. The [concrete backend](../glossary.md#concrete-backend) then owns deterministic [lowering](../glossary.md#lowering), [fusion](../glossary.md#term-fusion), route selection,
and private configuration. Shared Prepare sees only the typed [opaque](../glossary.md#term-opaque) plan and backend-neutral
resource declarations. Runtime owns the stable slot and memory-plan types; shared Prepare owns the
source assignment and finalizer [handoff](../glossary.md#term-handoff). Concrete per-[run](../glossary.md#term-run) population and executable work remain
backend responsibilities.

This division keeps `CompileArtifacts` and other [Compiler](../glossary.md#term-compiler)-owned types out of the concrete backend
surface. It also keeps route selection, measurement, [cache](../glossary.md#term-cache) mutation, allocation, and [graph](../glossary.md#graph)
inspection out of Runtime.

The tuning handoff is a side transport around this lifecycle, not another arrow in it. A concrete
backend produces the complete [immutable](../glossary.md#term-immutable) [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) and later validates any supplied decision against a
fresh batch. Shared Prepare retains exact references only. It does not enumerate the batch,
match the decision, serialize either value, or insert them into Runtime.

An owning composition may pass the handoff to `WorkloadTuning` together with a
`BackendWorkloadTuning<C, D, K>` implementation and a `ColdCandidateMeasurement<C, K>` action.
The collaboration supplies backend-owned compatibility and candidate [identities](../glossary.md#term-identity-meanings), complete
candidate enumeration, decision construction and [codec](../glossary.md#term-codec) operations; the measurement action runs
one complete candidate on equivalent representative inputs. Tuning returns a new handoff for each
original [occurrence](../glossary.md#term-occurrence) with the selected decision present. The concrete backend must still validate
that decision against a freshly generated compatible batch before it can affect analysis.

## General-[scan](../glossary.md#term-cumulative-scan) projection and resource [handoff](../glossary.md#term-handoff)

A scan **formal** is a named input role inside the repeated body; an **actual** is the outer
value bound to that role. A **[carry](../glossary.md#term-carry)** passes state from one step to the next. A **[residual](../glossary.md#scan-residual-port-and-history-demand)**
is a body value made available to save for a consumer, and a **history** stores demanded
per-step values. Declaring a [residual port](../glossary.md#scan-residual-port-and-history-demand) does not require saving every temporary.

[`GraphPreparation`](../glossary.md#graph-preparation--graphpreparation) now preserves each scan's exact compiled body and ordered role [bindings](../glossary.md#term-binding) inside
its selected [partition](../glossary.md#partition). A residual port names a body value that may be saved as history; demand
means an outer [publication](../glossary.md#term-publication) or consumer uses a port. Before any analyzer runs, projection checks
demand against final [graph](../glossary.md#graph) outputs and all outer input uses, including other partitions.
[`PrepareContext`](../glossary.md#prepare-context-preparecontext) checks every local [descriptor](../glossary.md#term-descriptor)
for static [shape](../glossary.md#shape) or the [occurrence](../glossary.md#term-occurrence)'s single named [batch dimension](../glossary.md#batch-dimension) B and validates formal/actual and outer-role descriptors, even for unused formals,
sibling results and zero time steps. A directly constructed context checks locally visible uses
and supplied logical obligations; it cannot verify unseen external edges. Local numeric IDs may
coincide with outer IDs without identifying the same values. [Model](../glossary.md#term-model)/[Compiler](../glossary.md#term-compiler) retain ordinary body
[inference](../glossary.md#term-inference) and [automatic differentiation](../glossary.md#autograd) ([AD](../glossary.md#autograd)); [backends](../glossary.md#backend) still qualify execution.

All-valid [symbolic Scan](../glossary.md#symbolic-scan) projection retains fixed time/[features](../glossary.md#term-feature) and unresolved symbolic
formal/result [layouts](../glossary.md#layout); static formal/results remain [canonical](../glossary.md#term-canonical-form). The bounded [rank](../glossary.md#term-rank)-three outer
sequence/history and AD projection is not arbitrary-rank backend support. `requiresBatchSize`
is explicit even when every byte recipe has zero slope, as with `[B,0]`. Analysis declares exact
fixed or [run](../glossary.md#term-run)-size requirements before slots; assignment fixes [identities](../glossary.md#term-identity-meanings) and representation
positions without evaluating [CPU](../glossary.md#term-cpu) recipes. The CPU binds actual B later, not a [sample](../glossary.md#term-sample-machine-learning-and-hyperparameters) B or a
new resource [topology](../glossary.md#term-topology). [Prepare0014](../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md)
tracks the implemented [FLOAT32](../glossary.md#term-float32) execution [slice](../glossary.md#slice) and its final [qualification](../glossary.md#term-qualification) separately.

Length-aware projection uses those same [owners](../glossary.md#term-owner). T is the static time count and B the [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) [extent](../glossary.md#term-extent).
The final non-[gradient](../glossary.md#term-gradient) [INT64](../glossary.md#data-type--datatype) `[B]` lengths actual remains an outer input, never a body formal or
gradient contributor. [Prepare](../glossary.md#prepare) validates T/B,
[homogeneous](../glossary.md#term-homogeneous) FLOAT32/[FLOAT64](../glossary.md#term-float64) locals and declared ROW(0), [rank-two](../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1) or SHARED boundaries,
including tied actual positions and empty domains. SHARED has no inferred batch [axis](../glossary.md#term-axis), even when
its extent equals B. Actual layouts, including lengths, are retained; CPU read-layout predicates
do not belong in shared [validation](../glossary.md#term-validation). Generated reverse scans may mix these roles and append
SHARED [parameter](../glossary.md#term-parameter) accumulators without inheriting Compiler's restricted original-source AD [admission](../glossary.md#term-admission).

The remaining pipeline is unchanged: analysis declares outer-value [buffers](../glossary.md#term-buffer) and analysis-local
scratch workspaces; shared Prepare assigns slots; finalization receives those exact declarations
and assignments. Shared buffers use maximum declared size/[alignment](../glossary.md#term-alignment) across participating owners;
each workspace must match its declaration exactly. A body-local ID is not an outer buffer ID,
and numeric coincidence supplies no local-storage [provenance](../glossary.md#provenance).

For example, consider a [scalar](../glossary.md#term-scalar) FLOAT32 body with three time steps, a published final carry and a
residual consumed by a different backend [owner](../glossary.md#term-owner)'s partition. If these are its only outer uses,
only those ports are demanded. A test backend can declare the residual's `3 × 4 = 12` bytes at both
partitions and its scratch separately; both residual declarations receive one shared [buffer slot](../glossary.md#buffer-slot--bufferslot)
with owner-specific representation positions. Missing [producer](../glossary.md#term-producer) or consumer declarations reject
before finalization.
At zero steps a zero-byte residual declaration is legal, but descriptor and demand checks still
run. These are [metadata](../glossary.md#term-metadata)/resource-handoff guarantees, not physical history or numerical execution
evidence.

For a length-mode FLOAT64 [fixture](../glossary.md#term-fixture) with T=3 and batch, feature and hidden extents B=F=H=2,
lengths requires `2 × 8 = 16` bytes.
A demanded SHARED transposed [weight](../glossary.md#term-weight) history `[T,F,H]` requires `3 × 2 × 2 × 8 = 96` bytes,
not a 192-byte `[T,B,F,H]` bank. A test-only backend can declare two banks for one `[B,H]` row
carry and one `[F,H]` shared [accumulator](../glossary.md#accumulator-type): `2 × (4 + 4) × 8 = 128` workspace bytes. This is the
fixture's geometry, not a CPU policy. It also declares staging/temporaries before assignment;
dead histories have no declaration. Finalizers receive exact declarations, slots and owner
representation positions; missing cross-partition declarations fail before finalization.
All fixture creators throw if invoked, and no executable runs. Real CPU length-mode [lowering](../glossary.md#lowering),
[runtime](../glossary.md#term-runtime) length validation and skipping remain closed; qualified all-valid CPU support is unchanged.
Shared Prepare cannot detect hidden allocations inside an [opaque](../glossary.md#term-opaque) backend executable. See
[general-scan resource and execution boundaries](../architecture/general-differentiable-scan.md#capability-resources-and-generated-code).

## Current-contract example

### Inputs and initial state

Assume shared [Prepare](../glossary.md#prepare) has constructed a `PrepareContext<CpuInputs>` for one [CPU](../glossary.md#term-cpu)-owned [operation](../glossary.md#operation):

- the operation has one [FLOAT32](../glossary.md#term-float32) input and one FLOAT32 output;
- both values have fully static [Shape](../glossary.md#shape) `[2, 3]`;
- each direct CPU representation therefore needs `2 × 3 × 4 = 24` bytes;
- the projected lists contain the exact partition nodes, unique values, and one matching logical
  memory requirement per projected value;<span data-glossary-navigation="true"> ([partition glossary](../glossary.md#partition); [nodes glossary](../glossary.md#node))</span>
- the context has no logical-splat [constants](../glossary.md#term-constant); and
- `CpuInputs(alignment=4)` is an [immutable](../glossary.md#term-immutable) [backend](../glossary.md#backend) input selected for this [target](../glossary.md#term-target).

The following is current Java [API](../glossary.md#term-application-programming-interface-api), but the [sample](../glossary.md#term-sample-machine-learning-and-hyperparameters) backend is illustrative rather than a
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
  cast, raw `Object`, or string-keyed [parameter](../glossary.md#term-parameter) map.
- `partitionDag()` is the sole retained topology source. This one-node example reads its external
  input occurrence and local sink directly; a multi-node backend can also query exact producer,
  consumer, and edge ports without reconstructing adjacency.<span data-glossary-navigation="true"> ([topology glossary](../glossary.md#term-topology); [node glossary](../glossary.md#node); [occurrence glossary](../glossary.md#term-occurrence); [backend glossary](../glossary.md#backend); [producer glossary](../glossary.md#term-producer))</span>
- The preparer rejects any context outside this example's supported shape before returning a
  partial result. A production backend would validate its complete supported operation,
  descriptor, capability, and configuration contract.<span data-glossary-navigation="true"> ([shape glossary](../glossary.md#shape); [operation glossary](../glossary.md#operation); [descriptor glossary](../glossary.md#term-descriptor))</span>
- The plan records the selected `"vector"` route opaquely. Shared Prepare can retain it for later
  finalization but cannot inspect its fields.<span data-glossary-navigation="true"> ([Prepare glossary](../glossary.md#prepare))</span>
- The two [buffer](../glossary.md#term-buffer) declarations use [graph](../glossary.md#graph) [`ValueId`](../glossary.md#logical-valueid-versus-physical-memory-slot) values only to associate requirements with
  projected logical values. Those IDs are not Runtime slots.<span data-glossary-navigation="true"> ([Runtime glossary](../glossary.md#term-runtime))</span>
- Workspace ID `0` is unique only within this analysis result. Its 64-byte size and 16-byte
  alignment are exact declarations, not an allocation.<span data-glossary-navigation="true"> ([byte alignment glossary](../glossary.md#term-byte-alignment))</span>

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
allocate memory, create an executable, perform a measurement, or execute the [operation](../glossary.md#operation).

### Optional [opaque](../glossary.md#term-opaque) candidate transport

The following current [API](../glossary.md#term-application-programming-interface-api) [shape](../glossary.md#shape) illustrates the separate transport. The records are illustrative
[backend](../glossary.md#backend)-owned values; their fields are invisible through the method-free shared roles.

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
exact `decision` reference. The record snapshots no route list itself; the backend-owned [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) is
responsible for its own immutability. This result proves only type-safe [partition](../glossary.md#partition) association and
transport. It does not prove that `"vector"` is present, valid, compatible, measured, or selected
for execution. The owning backend must validate those meanings against a freshly generated
complete batch.

### Failure variation

A [`Buffer`](../glossary.md#term-buffer) or `Workspace` rejects a negative byte size and an [alignment](../glossary.md#term-alignment) that is not a positive
power of two. [`BackendPartitionAnalysis`](../glossary.md#backend-partition-analysis-backendpartitionanalysis) also rejects a repeated buffer [`ValueId`](../glossary.md#logical-valueid-versus-physical-memory-slot) or repeated
workspace requirement ID. A [backend](../glossary.md#backend) that cannot realize the complete context should throw
`IllegalArgumentException` rather than return an incomplete plan or defer route fallback to
[Runtime](../glossary.md#term-runtime).

### Current assignment-to-finalization continuation

After every analysis in the complete [partition](../glossary.md#partition) set passes shared [validation](../glossary.md#term-validation), the package-internal
[Prepare](../glossary.md#prepare) [handoff](../glossary.md#term-handoff) creates one shared memory plan and one ordered assignment per declaration. A
[concrete backend](../glossary.md#concrete-backend) implements only the public finalizer collaboration; it neither calls nor replaces
the package-internal assignment [operation](../glossary.md#operation).

This illustrative executable has no resource selections because the example stops at [immutable](../glossary.md#term-immutable)
recipe construction. A production executable would derive its dense selections from the supplied
assignments and implement checked representation [binding](../glossary.md#term-binding) through [Runtime](../glossary.md#term-runtime)'s protected hooks.

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
analysis declaration order and retains the exact requirement, slot, and [owner](../glossary.md#term-owner)-specific
representation position. This single-owner example uses position zero for both [buffers](../glossary.md#term-buffer). This
proves the current typed finalization handoff. `finalized.resources()` is empty because this
illustrative
finalizer acquired no persistent resource. Public orchestration performs equivalent assignment
and finalization internally, then owns every returned resource transactionally while it assembles
and validates the schedule. The snippet does not prove per-[run](../glossary.md#term-run) physical allocation, binding, or
execution.

### Current complete-[graph](../glossary.md#graph) orchestration

The public orchestration input keeps each [backend](../glossary.md#backend)'s analysis and finalization generic types
paired. One preparation value is supplied for each [compile](../glossary.md#compile) [partition](../glossary.md#partition), in compile-partition order:

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

[`artifacts`](../glossary.md#term-artifact), `cpuInputs`, and `assembleExampleSchedule` are deliberately illustrative inputs.
Production [Engine](../glossary.md#term-engine) obtains artifacts through `GraphCompilationPort`, asks [CPU](../glossary.md#term-cpu) only for artifact-free
`PartitionPreparation` and schedule-assembler collaborations, and invokes this same shared
[operation](../glossary.md#operation). [Prepare](../glossary.md#prepare) first constructs all [`PrepareContext`](../glossary.md#prepare-context-preparecontext) values, then analyzes and finalizes every
partition in order, calls the assembler once with a complete [immutable](../glossary.md#term-immutable)
[`PreparedScheduleContext`](../glossary.md#prepared-schedule-context--preparedschedulecontext), validates the returned recipe, and returns the exact
[`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution). The schedule context exposes stable projected facts rather than the [Compiler](../glossary.md#term-compiler)
aggregate.

The assembler must describe every [bindable input](../glossary.md#bindable-input) with exactly one `CallerInput` in compiler input
order. A compile-time [logical splat](../glossary.md#logical-splat) instead needs at least one `InitializedBuffer` and no caller
input. The backend creator materializes that [constant](../glossary.md#term-constant) into a fresh [run](../glossary.md#term-run)-owned representation;
[Runtime](../glossary.md#term-runtime) records only that the representation starts valid. Prepare also validates executable
coverage and order, transfer and executable representation [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix), and ordered forward then
[gradient](../glossary.md#term-gradient) publications. It invokes no creator and performs no physical work.

## Context and requirement invariants

[`PartitionDag`](../glossary.md#partition-dag-partitiondag) snapshots membership while retaining the exact [node](../glossary.md#node) references. Node IDs must
match `PlannedPartition.nodeIds()` exactly and in order. [Producer](../glossary.md#term-producer) and consumer occurrences retain
their zero-based node and output/[input port](../glossary.md#capture-graph-ports-and-reconstruction) positions. Repeated inputs remain separate consumer
and edge entries, and every output of a multi-output node remains separately addressable. An
external input has no local producer. A local sink has no output consumed locally, regardless of
[publication](../glossary.md#term-publication) or cross-[partition](../glossary.md#partition) use.

These are [topology](../glossary.md#term-topology)-only facts. They do not describe [graph](../glossary.md#graph) regions, cross-partition [ownership](../glossary.md#term-ownership-meanings),
transfer, publication, [materialization](../glossary.md#materialization), memory policy, [fusion](../glossary.md#term-fusion) legality, route selection,
scheduling, or execution. [`PrepareContext`](../glossary.md#prepare-context-preparecontext) stores the [DAG](../glossary.md#term-directed-acyclic-graph-dag) as its only node/topology state and
snapshots its other lists and map before analysis. Its [canonical](../glossary.md#term-canonical-form) record shape has five
components; the six-[argument](../glossary.md#term-argument) partition-and-node-list constructor preserves source calls only and
does not promise [binary](../glossary.md#term-binary-descriptor-and-metadata) compatibility with the former record shape.

Projected values are unique by [`ValueId`](../glossary.md#logical-valueid-versus-physical-memory-slot); every node input and output must resolve to one of
them. Every projected value has exactly one [descriptor](../glossary.md#term-descriptor)-matching `LogicalMemoryRequirement`, but
the contract does not add a separate rule that every projected value must appear in a node input
or output.

Projected [Shapes](../glossary.md#shape) are fully static except for the ordinary [rank](../glossary.md#term-rank)-at-most-two single named-B
projection; expressions, unknown [axes](../glossary.md#term-axis) and [general Scan](../glossary.md#general-differentiable-scan) dynamic geometry remain unsupported here.
`PrepareContext.requiresBatchSize()` records symbolic context even when every byte recipe is
[constant](../glossary.md#term-constant), as in `X[B,0].sum()`. A projected logical-splat constant is
permitted only for a projected graph input, and its [`ScalarValue`](../glossary.md#typed-scalar-value--scalarvalue) [data type](../glossary.md#data-type--datatype) must match the input
descriptor exactly. The constant map is [immutable](../glossary.md#term-immutable) [metadata](../glossary.md#term-metadata); analysis does not materialize its
logical repetitions.

Resource declarations retain `PreparedMemoryPlan.ByteSize`: fixed bytes or an immutable [run](../glossary.md#term-run)-size
recipe. Existing long constructors wrap `FixedByteSize`; `byteSize()` rejects a run-bound
declaration rather than guessing. [Prepare](../glossary.md#prepare) never evaluates a recipe. Fixed [buffer](../glossary.md#term-buffer) aggregation
retains maximum size/[alignment](../glossary.md#term-alignment); run-bound buffer declarations must agree structurally and in
alignment, and workspaces agree exactly. [Runtime](../glossary.md#term-runtime) creators now take actual
`(byteSize, byteAlignment)`; migrate static SAM consumers without a zero-argument adapter.
Resource sizes are non-negative byte counts, so zero is valid. Alignments are positive powers of
two measured in bytes. Buffer and workspace [identities](../glossary.md#term-identity-meanings) occupy separate domains. Current initial
assignment gives each workspace declaration its own stable Runtime slot; reuse and [lifetime](../glossary.md#term-lifetime)
interference are not represented by the current [API](../glossary.md#term-application-programming-interface-api).

## Failures, [ownership](../glossary.md#term-ownership-meanings), and [concurrency](../glossary.md#term-concurrency)

An [owner](../glossary.md#term-owner) must [release](../glossary.md#term-release) a resource; a borrower may use it only within the owner's permitted
[lifetime](../glossary.md#term-lifetime). A **[lease](../glossary.md#term-lease)** keeps an admitted use valid until that use finishes. **[Rollback](../glossary.md#term-rollback)** here
[releases](../glossary.md#commit-release-distribution-log-and-seed-meanings) resources acquired during a failed preparation or setup; it is not a promise to undo
arithmetic already performed by earlier operations in a [run](../glossary.md#term-run).

- Reject an unsupported partition owner, operation, descriptor, capability, configuration, or
  cached decision before producing an analysis result.<span data-glossary-navigation="true"> ([partition glossary](../glossary.md#partition); [operation glossary](../glossary.md#operation); [descriptor glossary](../glossary.md#term-descriptor))</span>
- Treat backend input and plan implementations as immutable values. Shared Prepare retains their
  exact references and does not copy or synchronize backend-private state.<span data-glossary-navigation="true"> ([Prepare glossary](../glossary.md#prepare))</span>
- Treat tuning-batch and decision implementations as immutable values. Their shared marker roles
  expose no method, and the handoff rejects a null partition, batch, or optional container while
  retaining exact references and acquiring no ownership.<span data-glossary-navigation="true"> ([batch glossary](../glossary.md#term-batch-machine-learning-and-hyperparameters); [handoff glossary](../glossary.md#term-handoff))</span>
- Make `analyze` deterministic from its complete context. Do not measure candidates, mutate a
  cache, allocate physical resources, compile native executables, or retain per-run bindings.<span data-glossary-navigation="true"> ([cache glossary](../glossary.md#term-cache); [compile glossary](../glossary.md#compile); [bindings glossary](../glossary.md#term-binding))</span>
- Do not return a [`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable), slot, address, storage handle, or resource lifetime from
  analysis.<span data-glossary-navigation="true"> ([resource lifetime glossary](../glossary.md#ownership-borrowing-and-lease))</span>
- Finalization may acquire immutable persistent prepared resources, such as a compiled native
  executable, only after shared slot assignment. Before a complete finalizer result returns, the
  backend owns rollback. After return, shared Prepare owns the unique resource identities until a
  successfully constructed <span data-glossary-navigation="true"> ([identities glossary](../glossary.md#term-identity-meanings))</span>[`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution) accepts ownership; any intervening failure closes
  them in reverse acquisition order.
- The current CPU finalizer returns an empty persistent-resource list. CPU buffer and workspace
  representations are separate per-run physical resources created by CPU-owned schedule recipes,
  retained by one Runtime <span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu); [buffer glossary](../glossary.md#term-buffer))</span>[`RunState`](../glossary.md#run-state--runstate), and closed when its [Runtime](../glossary.md#term-runtime) [`RunResult`](../glossary.md#run-result--runresult) lease closes or
  failed setup or execution rolls the state back.

The contracts are [immutable](../glossary.md#term-immutable) and can be shared safely when the [backend](../glossary.md#backend)'s marker-role
implementations are themselves immutable. One [prepared execution](../glossary.md#prepared-execution--preparedexecution) may be reused by concurrent
runs; each run has an isolated `RunState` and separately owned [mutable](../glossary.md#term-mutable) physical representations.
Persistent resources remain immutable and are released by the prepared execution's explicit,
[idempotent](../glossary.md#term-idempotent) [close lifecycle](../glossary.md#ownership-borrowing-and-lease) after admitted runs release their leases.

## Registration, diagnostics, and [validation](../glossary.md#term-validation)

Current ordinary composition is explicit: `Engine.standard()` owns one fresh [CPU](../glossary.md#term-cpu) integration,
while `Engine.builder()` can own opened CPU and/or [Metal](../glossary.md#term-metal) integrations. Builder preparation
preflights all [owners](../glossary.md#term-owner)/transfers, supplies every [partition](../glossary.md#partition)'s exact preparation to shared [Prepare](../glossary.md#prepare),
collects [owner](../glossary.md#term-owner)-indexed physical contributions, and composes one execution/transfer/[publication](../glossary.md#term-publication)
schedule. Advanced CPU-only composition remains explicit through
`AdvancedEngine.takeOwnership(...)`. No surface discovers plugins; do not add [runtime](../glossary.md#term-runtime) service
[lookup](../glossary.md#term-lookup), reflection, or `ServiceLoader` as a substitute.

Prepare/[trace](../glossary.md#trace) payloads and emitters are also planned. A [backend](../glossary.md#backend) may design typed diagnostic facts
for its analysis, but it must not leak business logic into trace data-transfer objects or use a
string map as the primary trace [model](../glossary.md#term-model).

For a concrete implementation, add:

- focused unit tests for supported analysis, failure order, immutable results, exact partition
  identity, deterministic ordering, and duplicate rejection;<span data-glossary-navigation="true"> ([immutable glossary](../glossary.md#term-immutable); [identity glossary](../glossary.md#term-identity-meanings))</span>
- architecture validation when dependencies change;
- backend-conformance tests for every concrete backend capability claimed through the shared
  lifecycle;<span data-glossary-navigation="true"> ([conformance glossary](../glossary.md#term-conformance); [concrete backend glossary](../glossary.md#concrete-backend); [lifecycle glossary](../glossary.md#lifecycle))</span>
- integration tests for fixed or explicit [Engine](../glossary.md#term-engine) composition and end-to-end execution; and
- native resource and platform validation when finalization begins owning native resources.

For the current shared contract, [run](../glossary.md#term-run):

```bash
./gradlew :modules:prepare:test
./gradlew :modules:prepare:javadoc
```

## Limitations and related documentation

The preparation [lifecycle](../glossary.md#lifecycle) supports the bounded ordinary named-B [handoff](../glossary.md#term-handoff), not general dynamic
[binding](../glossary.md#term-binding) or workspace reuse. [Compiler](../glossary.md#term-compiler) owns semantic [admission](../glossary.md#term-admission), [CPU](../glossary.md#term-cpu) numeric recipes and [Runtime](../glossary.md#term-runtime)
per-[run state](../glossary.md#run-state--runstate) without [Shapes](../glossary.md#shape). It does
support exact shared-resource declarations, [backend](../glossary.md#backend)-owned per-[run](../glossary.md#term-run) representation recipes, and
persistent-resource transfer, but the [concrete backend](../glossary.md#concrete-backend) owns the physical mechanics and Runtime
owns the per-run state. Fixed CPU `Engine.standard()`, explicit CPU/[Metal](../glossary.md#term-metal) builder composition
including bounded mixed-[owner](../glossary.md#term-owner) schedules, and advanced CPU composition are current; generic [plugin](../glossary.md#term-plugin)
registration and executable [persistence](../glossary.md#term-persistence) are not. Mixed CPU/Metal values have deterministic
owner-indexed representations and exact all-seven-[carrier](../glossary.md#term-storage-carrier), [rank](../glossary.md#term-rank)-0..16 static [canonical](../glossary.md#term-canonical-form) or positive-
[stride](../glossary.md#term-stride) non-overlapping storage-[layout](../glossary.md#layout) transfer recipes with checked physical spans. The current CPU
complete-plan tuning [producer](../glossary.md#term-producer) is session-scoped, so Phase 2 does not persist or reuse a complete
prepared plan across sessions.

The separate tuning tool provides bounded local measurement, selection, reusable workload-[cache](../glossary.md#term-cache)
persistence, and rich evidence around the [opaque](../glossary.md#term-opaque) handoff. Current [Engine](../glossary.md#term-engine) composition supplies the
CPU collaborations for a bounded two-phase workflow. [Prepare](../glossary.md#prepare) itself still performs no
measurement, selection, cache access, persistence, [model](../glossary.md#term-model) aggregation, or tuning-aware [graph
preparation](../glossary.md#graph-preparation--graphpreparation). Slot assignment, finalization input/result, [prepared partition](../glossary.md#prepared-partition--preparedpartition), explicit schedule
assembly and [validation](../glossary.md#term-validation), prepared-execution construction, and shared runner contracts are
current. Compatible cached decisions may be explicit [immutable](../glossary.md#term-immutable) backend inputs, but analysis
neither loads nor mutates a cache.

See the [Runtime/Prepare/Backend boundary](../architecture/runtime-prepare-backend-boundary.md),
[Planning ownership and partition scoring](../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Planning glossary](../glossary.md#term-planning); [ownership glossary](../glossary.md#term-ownership-meanings); [partition glossary](../glossary.md#partition))</span>, and
[kernel routes](kernel-routes.md)<span data-glossary-navigation="true"> ([kernel glossary](../glossary.md#kernel))</span>.

## Numerical-semantics [admission](../glossary.md#term-admission)

A preparer receives complete [occurrence](../glossary.md#term-occurrence) facts in [`PrepareContext`](../glossary.md#prepare-context-preparecontext), not a [graph](../glossary.md#graph)-wide numerical
selector. [Model](../glossary.md#term-model) alone defines each family/[dtype](../glossary.md#data-type--datatype) formula, exact stored and discrete boundaries, and
named arithmetic-site permissions. Preparation rejects unsupported occurrences before route
analysis; each retained route must qualify against Model semantics, with [finite](../glossary.md#term-finite) [tolerance](../glossary.md#term-tolerance) used
only in tests. [CPU](../glossary.md#term-cpu) keeps its qualified generated and [OpenBLAS](../glossary.md#term-openblas) routes and non-profile [identities](../glossary.md#term-identity-meanings).
[Metal](../glossary.md#term-metal) admits exact [unary](../glossary.md#term-unary), affine, [canonicalization](../glossary.md#occurrence-identity-and-canonical-form), indexing, [BOOL](../glossary.md#data-type--datatype)-domain, Task-0059 raw
movement, Task-0060 replacement/fold/aggregate, Task-0063 ordering/top-K/numeric arg-[extrema](../glossary.md#saturation-pole-and-extremum),
promoted integral [MATMUL](../glossary.md#matrix-multiplication--matmul), Task-0064 maximum-[pooling](../glossary.md#term-pooling), and Task-0065 raw INITIAL_STATE rows. Metal
also admits the established arithmetic/reduction/[scan](../glossary.md#term-cumulative-scan) rows,
same-type [canonical](../glossary.md#term-canonical-form) positive-[rank](../glossary.md#term-rank) [FLOAT32](../glossary.md#term-float32) [MSE](../glossary.md#mean-squared-error--mse), every positive-static FLOAT32 MATMUL geometry,
no-[gradient](../glossary.md#term-gradient) mixed [BFLOAT16](../glossary.md#term-bfloat16)/FLOAT32 MATMUL, Task-0064 [convolution](../glossary.md#term-channels-first-convolution-layers)/average-pooling, Task-0065 FLOAT32
[dropout](../glossary.md#dropout--inverted-dropout), and the exact Task-0069 [rank-one](../glossary.md#axis-rank-scalar-vector-and-matrix) FLOAT32 L1_NORM, SCATTER_ADD, and [singleton](../glossary.md#term-singleton) [VARIANCE](../glossary.md#term-variance)
domains. The preparer authenticates a local MATMUL affine operand as the exact [identity](../glossary.md#term-identity-meanings)-prefix,
last-two-[axis](../glossary.md#term-axis) [transpose](../glossary.md#term-transpose) of its canonical source; keeps canonical BOOL values available to admitted
consumers; retains exact storage [layouts](../glossary.md#layout) for SELECT/positive-step [SLICE](../glossary.md#slice); and declares only the
authenticated compact materialized set for a shared custom-program route. It lowers MSE to one
typed [MPSGraph](../glossary.md#term-mpsgraph) [node](../glossary.md#node) whose native recipe is fixed subtraction, self-multiplication, and optional full
SUM or [MEAN](../glossary.md#term-mean). It lowers Task-0060 scatter [preflight](../glossary.md#term-preflight), signed slice placement, single-writer non-overlap
folds, exact modular/logical reductions, every Task-0063 row, and selected Task-0066/Task-0069
occurrences to their fixed custom programs without atomics or fallback. Every [partition](../glossary.md#partition) containing
BFLOAT16 or [FLOAT16](../glossary.md#term-float16) values uses that custom-only route, including [homogeneous](../glossary.md#term-homogeneous) no-gradient
RESHAPE, simple PERMUTE, materializing [CONTIGUOUS](../glossary.md#term-contiguous), SLICE, CONCAT, and TILE. A bounded canonical
no-gradient BFLOAT16 or FLOAT16 [EXP](../glossary.md#term-expx) uses its own typed custom step. A separate FLOAT32 EXP may use
an internal MPSGraph boundary step on an independent FLOAT32 branch or after an explicit
low-to-FLOAT32 cast. Bounded canonical no-gradient FLOAT32 [SIGMOID](../glossary.md#term-sigmoid) likewise uses a sign-guarded
composed MPSGraph step on either FLOAT32 path; separately admitted low-valued SIGMOID uses
its typed custom step in the same fixed custom partition. The
preparer may replace only a qualified private MATMUL/[Conv2d](../glossary.md#convolution--conv2d) suffix with one typed anchor
instruction. It preserves typed
ingress, [target](../glossary.md#term-target) and internal physical byte geometry, window/index obligations, unsigned-32-bit
geometry and the [schema](../glossary.md#term-schema)-twenty/version-thirty-six route identity without
[widening](../glossary.md#term-widening) capability. Every other identity fails closed.

Metal's one closed prepared-route identity owns the existing candidate wires `1..3` and the
MPSGraph/custom-[kernel](../glossary.md#kernel) family. Candidate [serialization](../glossary.md#term-serialization) delegates to it. A returned plan retains one
final route before finalization; finalization and [Runtime](../glossary.md#term-runtime) cannot reselect it. The separate
package-private 115-kind structural catalog is descriptive cold [metadata](../glossary.md#term-metadata), not admission or route
approval. Its package-private forcing seam is test-only, runs normal capability and fresh-[handoff](../glossary.md#term-handoff)
[authentication](../glossary.md#term-authentication) first, and accepts only a route already present in the exact regenerated [candidate
batch](../glossary.md#candidate-batch).
