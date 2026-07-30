# Analyze an owned partition

## Outcome and current scope

This guide shows a concrete backend contributor how to implement the current analysis stage of
Synaptik's staged preparation handoff. The implemented
`io.github.pho001.synaptik.prepare.analysis` package accepts one validated, fully static planned
partition and returns an opaque backend plan plus exact shared buffer and workspace declarations.

The current API stops there. Runtime slot assignment, backend finalization, prepared executables,
physical allocation, registration, and execution remain planned. The example therefore
demonstrates a complete current analysis call, not a runnable backend or end-to-end engine.

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
  -> shared BufferSlot/WorkspaceSlot assignment              planned
  -> same backend finalizes the opaque plan                  planned
  -> PreparedPartition + PreparedExecutable                  planned
  -> Runtime executes the prepared schedule                  planned
```

Planning chooses an owner such as CPU. Shared Prepare validates and projects that owner's exact
partition facts. The concrete backend then owns deterministic lowering, fusion, route selection,
and private configuration. Shared Prepare sees only the typed opaque plan and backend-neutral
resource declarations. Runtime will later own stable slots and per-run binding.

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
two measured in bytes. Buffer and workspace identities occupy separate domains. A future initial
assignment will give each workspace declaration its own stable Runtime slot; reuse and lifetime
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
- Native-handle acquisition and cleanup belong to the later finalized prepared-resource
  lifecycle. The current analysis result owns none.

The contracts are immutable and can be shared safely when the backend's marker-role
implementations are themselves immutable. They do not add synchronization or define concurrent
access to future native resources.

## Registration, diagnostics, and validation

Engine composition and explicit backend registration are not implemented yet. Do not add runtime
service lookup, reflection, or `ServiceLoader` as a substitute. A future engine will supply the
explicitly registered concrete preparer before Runtime execution.

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

The current API has no dynamic-dimension binding, slot assignment, workspace reuse, physical
resource, finalization, executable, schedule, Runtime behavior, concrete backend implementation,
or model-autotuning workflow. Compatible cached decisions may be explicit immutable backend
inputs, but analysis neither loads nor mutates a cache.

See the [Runtime/Prepare/Backend boundary](../architecture/runtime-prepare-backend-boundary.md),
[Planning ownership and partition scoring](../architecture/partition-scoring.md), and
[kernel routes](kernel-routes.md).
