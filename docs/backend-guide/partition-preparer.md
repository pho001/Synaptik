# Analyze and finalize an owned partition

## Outcome and current scope

This guide shows a concrete backend contributor how to implement both current backend stages of
Synaptik's staged preparation handoff. Analysis accepts one validated, fully static planned
partition and returns an opaque backend plan plus exact shared buffer and workspace declarations.
After shared Prepare assigns Runtime slots, finalization consumes those exact assignments and
constructs an immutable executable recipe.

Public `GraphPreparation.prepare(...)` now coordinates the package-internal batch assignment and
validates one complete schedule supplied by an explicit assembler. Physical allocation, backend
registration, production schedule assembly, and end-to-end Engine execution remain planned. The
example therefore demonstrates current analysis and finalization contracts with illustrative
backend types, not a production backend or runnable engine.

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

## Ownership and lifecycle

Read this flow from backend-neutral compile facts toward reusable runtime state:

```text
capability reporting and Planning ownership
  -> PrepareContext for one PlannedPartition                 current
  -> BackendPartitionPreparer.analyze                        current
  -> BackendPartitionAnalysis + exact requirements           current
  -> shared BufferSlot/WorkspaceSlot assignment              current, package-internal
  -> same backend finalizes the opaque plan                  current contract
  -> PreparedPartition + PreparedExecutable                  current contracts
  -> explicit assembler + Prepare schedule validation        current shared contract
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
        if (context.nodes().size() != 1
                || context.values().size() != 2
                || context.backendInputs().alignment() != 4) {
            throw new IllegalArgumentException("unsupported example context");
        }

        ValueId inputId = context.nodes().getFirst().inputs().getFirst();
        ValueId outputId = context.nodes().getFirst().outputs().getFirst();
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
    public PreparedExecutable finalizePartition(
            BackendPartitionFinalization<CpuPlan> finalization) {
        if (!finalization.analysis().plan().route().equals("vector")) {
            throw new IllegalArgumentException("unexpected example route");
        }
        return new CpuExecutable(finalization.memoryPlan());
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
                                inputRequirement, inputSlot, 0),
                        new PreparationResourceAssignment.Buffer(
                                outputRequirement, outputSlot, 1),
                        new PreparationResourceAssignment.Workspace(
                                workspaceRequirement, workspaceSlot, 0)));

BackendId cpu = context.partition().owner();
PreparedExecutable executable =
        new CpuFinalizer(cpu).finalizePartition(finalization);
PreparedPartition prepared =
        new PreparedPartition(context.partition(), executable);
```

`prepared.partition()` is the exact analyzed partition, and
`prepared.executable().memoryPlan()` is the exact `memoryPlan` object. The assignment list follows
analysis declaration order and retains the exact requirement and slot references. This proves the
current typed finalization handoff. Public orchestration performs equivalent assignment and
finalization internally, then validates a separately assembled schedule. The snippet does not
prove physical allocation, binding, execution, or cleanup of a persistent prepared resource.

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

`artifacts`, `cpuInputs`, and `assembleExampleSchedule` are deliberately illustrative inputs; no
public Compiler entry, production CPU backend, or Engine facade currently supplies them. The
result is nevertheless the current API shape: Prepare first constructs all `PrepareContext`
values, then analyzes and finalizes every partition in order, calls the assembler once with a
complete immutable `PreparedScheduleContext`, validates the returned recipe, and returns the
exact `PreparedExecution`.

The assembler must describe every bindable input with exactly one `CallerInput` in compiler input
order. A compile-time logical splat instead needs at least one `InitializedBuffer` and no caller
input. The backend creator materializes that constant into a fresh run-owned representation;
Runtime records only that the representation starts valid. Prepare also validates executable
coverage and order, transfer and executable representation coordinates, and ordered forward then
gradient publications. It invokes no creator and performs no physical work.

## Context and requirement invariants

`PrepareContext` snapshots its lists and map before analysis. Node IDs must match
`PlannedPartition.nodeIds()` exactly and in order. Projected values are unique by `ValueId`;
every node input and output must resolve to one of them. Every projected value has exactly one
descriptor-matching `LogicalMemoryRequirement`, but the contract does not add a separate rule
that every projected value must appear in a node input or output.

Every projected descriptor must have a fully static Shape. A projected logical-splat constant is
permitted only for a projected graph input, and its `ScalarValue` data type must match the input
descriptor exactly. The constant map is immutable metadata; analysis does not materialize its
logical repetitions.

Resource sizes are non-negative byte counts, so zero is valid. Alignments are positive powers of
two measured in bytes. Buffer and workspace identities occupy separate domains. Current initial
assignment gives each workspace declaration its own stable Runtime slot; reuse and lifetime
interference are not represented by the current API.

## Failures, ownership, and concurrency

- Reject an unsupported partition owner, operation, descriptor, capability, configuration, or
  cached decision before producing an analysis result.
- Treat backend input and plan implementations as immutable values. Shared Prepare retains their
  exact references and does not copy or synchronize backend-private state.
- Make `analyze` deterministic from its complete context. Do not measure candidates, mutate a
  cache, allocate physical resources, compile native executables, or retain per-run bindings.
- Do not return a `PreparedExecutable`, slot, address, storage handle, or resource lifetime from
  analysis.
- Current finalization constructs immutable Java recipe state only. Native-handle acquisition and
  cleanup require a later finalized prepared-resource lifecycle; the current analysis and
  finalization values own none.

The contracts are immutable and can be shared safely when the backend's marker-role
implementations are themselves immutable. They do not add synchronization or define concurrent
access to future native resources.

## Registration, diagnostics, and validation

Engine composition and explicit backend registration are not implemented yet. Do not add runtime
service lookup, reflection, or `ServiceLoader` as a substitute. A future engine will supply the
explicitly registered concrete preparations and schedule assembler before Runtime execution.

Prepare/trace payloads and emitters are also planned. A backend may design typed diagnostic facts
for its analysis, but it must not leak business logic into trace data-transfer objects or use a
string map as the primary trace model.

For a concrete implementation, add:

- focused unit tests for supported analysis, failure order, immutable results, exact partition
  identity, deterministic ordering, and duplicate rejection;
- architecture validation when dependencies change;
- backend-conformance tests once a concrete backend can prepare and execute its claimed
  capabilities;
- integration tests once Engine composition and end-to-end execution exist; and
- native resource and platform validation when finalization begins owning native resources.

For the current shared contract, run:

```bash
./gradlew :modules:prepare:test
./gradlew :modules:prepare:javadoc
```

## Limitations and related documentation

The current API has no dynamic-dimension binding, workspace reuse, physical resource, production
concrete backend implementation, Engine composition, or model-autotuning workflow. Slot
assignment, finalization input/collaboration, prepared partition, explicit schedule assembly and
validation, prepared-execution construction, and shared runner contracts are current. Compatible cached decisions may be explicit
immutable backend inputs, but analysis neither loads nor mutates a cache.

See the [Runtime/Prepare/Backend boundary](../architecture/runtime-prepare-backend-boundary.md),
[Planning ownership and partition scoring](../architecture/partition-scoring.md), and
[kernel routes](kernel-routes.md).
