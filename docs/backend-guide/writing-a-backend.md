# Integrate a concrete backend

## Outcome and supported scope

This guide maps current backend extension contracts into the complete planned lifecycle. Prepare
analysis, slot assignment, backend finalization, immutable physical-representation creation
callbacks, Runtime cold creation and binding, explicit per-copy validity, and creation/executable
scheduling are current. Production physical implementations, transfer/materialization/publication
steps, schedule consumption, Engine composition, and every production concrete backend remain
planned. The guide therefore separates compilable extension patterns from conceptual integration
steps.

## Prerequisites

Read the authoritative [`ARCHITECTURE.md`](../../ARCHITECTURE.md), [module boundaries](../architecture/module-boundaries.md), [dependency rules](../architecture/dependency-rules.md), and the concrete backend's master plan. A backend module may depend on shared contracts but must not depend on engine.

## Integration lifecycle

```text
capability -> compile ownership -> backend prepare -> executable -> runtime
```

1. Implement declarative capability reporting for semantic graph facts.
2. Implement a partition preparer that accepts only partitions owned by this backend.
3. Lower, specialize, fuse, and select routes inside the backend.
4. During current finalization, construct an immutable `PreparedExecutable` subclass against the
   assigned slots.
5. Supply immutable thread-safe buffer/workspace creators in one current
   `PreparedRepresentationPlan`; each call returns a fresh physical result for one run.
6. Place that plan in the optional first-only `PreparedSchedule.RepresentationCreationStep`, then
   place finalized executables in `ExecutionStep` occurrences. Schedule construction validates
   exact memory-plan identity and preserves occurrence order without invoking callbacks.
7. At run setup, current package-private Runtime orchestration validates all caller inputs, invokes
   creators before binding, and rolls back partial creation. Cold binding then validates concrete
   representation types and constructs one `BoundInvocation` with direct typed fields.
8. Emit typed backend trace contributions when the producer contract exists.
9. Expose a backend component that later Engine composition can register explicitly.

## Current representation-creation pattern

### Goal and inputs

Describe one borrowed caller input, one backend-created internal buffer, and one backend-created
workspace against the exact prepared memory plan supplied by shared finalization. The creator
objects are immutable and stateless; each call returns a fresh concrete result.

```java
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import java.util.List;

final class CpuBuffer implements BufferRepresentation {
    @Override public void close() { /* release concrete storage */ }
}

final class CpuWorkspace implements WorkspaceRepresentation {
    @Override public void close() { /* release concrete scratch */ }
}

final class CpuBufferCreator implements PreparedRepresentationPlan.BufferCreator {
    @Override public CpuBuffer create() {
        return new CpuBuffer();
    }
}

final class CpuWorkspaceCreator implements PreparedRepresentationPlan.WorkspaceCreator {
    @Override public CpuWorkspace create() {
        return new CpuWorkspace();
    }
}

PreparedRepresentationPlan representationPlan =
        new PreparedRepresentationPlan(
                plan,
                List.of(
                        List.of(
                                new PreparedRepresentationPlan.CallerInput(),
                                new PreparedRepresentationPlan.CreatedBuffer(
                                        new CpuBufferCreator()))),
                List.of(new CpuWorkspaceCreator()));
```

### Result and interpretation

The immutable plan contains only dense origins and exact typed callback references. Constructing
it allocates no physical buffer or workspace. During cold run setup, Runtime validates the
complete caller-input list before invoking `CpuBufferCreator`, then invokes workspace creators.
Every successful result is run-owned only after complete state construction. A partial failure
closes successfully created results in reverse creation order and preserves the original
unchecked failure, with cleanup failures suppressed.

This pattern has no registry, service lookup, reflection, raw object payload, or backend switch.
Allocation occurs inside explicitly prepared backend callbacks during cold setup, never inside
`PreparedExecutable.bind` or `BoundInvocation.execute`. A concrete backend must test callback
freshness across positions and concurrent runs, immutability/thread safety of retained creators,
null/duplicate rejection, and physical cleanup.

## Current cold-binding pattern

The following focused skeleton uses only current Runtime contracts. It is a local extension
pattern, not a complete backend: a current Prepare finalizer supplies the exact plan and dense
selections, while current cold setup creates the described representations package-privately and
later Runtime work will consume the schedule.

```java
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.List;

final class CpuInvocation extends BoundInvocation {
    private final CpuBuffer input;
    private final CpuWorkspace scratch;

    CpuInvocation(RunState state, CpuBuffer input, CpuWorkspace scratch) {
        super(state);
        this.input = input;
        this.scratch = scratch;
    }

    @Override protected void executeBound() {
        // Call the already-selected backend route through input and scratch directly.
    }
}

final class CpuExecutable extends PreparedExecutable {
    CpuExecutable(PreparedMemoryPlan plan) {
        super(
                plan,
                List.of(new BufferSelection(0, 0)),
                List.of(new WorkspaceSelection(0)));
    }

    @Override protected boolean acceptsBufferRepresentation(
            int index, BufferRepresentation representation) {
        return index == 0 && representation instanceof CpuBuffer;
    }

    @Override protected boolean acceptsWorkspaceRepresentation(
            int index, WorkspaceRepresentation representation) {
        return index == 0 && representation instanceof CpuWorkspace;
    }

    @Override protected BoundInvocation bindCompatible(
            RunState state,
            BufferRepresentation[] buffers,
            WorkspaceRepresentation[] workspaces) {
        return new CpuInvocation(
                state, (CpuBuffer) buffers[0], (CpuWorkspace) workspaces[0]);
    }
}
```

The two `instanceof` checks run once for each selected resource during cold binding. Runtime
passes fresh nominal arrays to `bindCompatible` only after all checks succeed. The invocation
stores the concrete references in final typed fields; it does not retain those arrays or perform
slot lookup or compatibility casting inside `executeBound()`.

Selections use dense `PreparedMemoryPlan` and `RunState` positions, not slot numeric components.
They may repeat when the same representation fills multiple operand roles. The executable must
retain the exact plan reference and be immutable and thread-safe so it can bind concurrently to
distinct run states. Each invocation retains one exact state, is not thread-safe, and rejects
execution after that state closes.

Cold binding may allocate the invocation and temporary arrays, but the current contract forbids
acquiring any auxiliary closeable or native binding resource. Binding changes no ownership and
has no partial-failure cleanup protocol. A future contract must add an explicit lifecycle before
a backend may acquire such a resource while binding.

## Current executable scheduling pattern

### Goal and inputs

Make the representation plan reachable as the first occurrence, then order two occurrences of
one finalized `CpuExecutable`. This example assumes `plan` is the exact shared plan supplied by
current Prepare finalization; it does not construct a runner or invoke a creator.

```java
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.List;

CpuExecutable executable = new CpuExecutable(plan);
PreparedSchedule.ExecutionStep occurrence =
        new PreparedSchedule.ExecutionStep(executable);
PreparedSchedule.RepresentationCreationStep creation =
        new PreparedSchedule.RepresentationCreationStep(representationPlan);
PreparedSchedule schedule =
        new PreparedSchedule(plan, List.of(creation, occurrence, occurrence));
```

### Result and interpretation

`schedule.steps()` retains the exact creation prefix followed by the same executable occurrence
twice in deterministic order, and every occurrence reports the exact `plan` reference. Repetition
means execute the prepared region twice when a future runner consumes the schedule; it does not
duplicate executable, representation, or cleanup ownership. Empty and executable-only schedules
remain valid for compatibility; a later Prepare validator may require creation for runnable work.

Schedule construction only validates and snapshots the recipe. It does not bind or execute the
`CpuExecutable`, invoke the creator callbacks, create a `RunState`, allocate or close resources,
or imply transfer,
materialization, publication, or result behavior. A backend must not encode those deferred
semantics in either current step; later Runtime-owned variants require their own stable transfer
or delivery contracts.

## Conceptual registration

```java
// Conceptual API; engine and backend factories are not implemented.
SynaptikEngine engine = SynaptikEngine.builder()
        .addBackend(cpuBackend())
        .addBackend(metalBackend())
        .build();
```

Each `addBackend` call makes composition visible before compilation and preparation. Runtime must not use classpath scanning, `ServiceLoader`, or a service locator to discover the same components during execution.

## Resources, concurrency, and failures

A backend implements physical buffer and workspace representations plus their allocation,
release, transfer, and access mechanics. Its retained creator callbacks and prepared executable
recipes are immutable and reusable. Each active complete logical execution has one isolated
`RunState`; caller inputs are borrowed and initially valid, created internal buffers are run-owned
and initially invalid, workspaces are run-owned scratch without logical validity, and published
outputs later transfer or lease ownership to a result. Each callback must return a fresh result
for every position and concurrent run.

Before the hot path, current backend extension code performs explicit checked compatibility and
creates typed bound invocation objects with direct representation references. It must not rely on
raw `Object`, unchecked generic access, reflection, string dispatch, a registry, or repeated
hot-path casts. Runtime orchestrates cleanup, while the backend representation performs physical
release. Failure cleanup must never close borrowed inputs or outputs already transferred from the
run.

Capability rejection occurs before ownership. Lowering and finalization failure occur during
prepare. Representation creation and execution failure occur during run setup or execution and
must not trigger hidden cross-backend fallback.

## Validation

Current backend tests should prove creator immutability and freshness, reverse partial-failure
cleanup, validity initialization, exact plan identity, selection order, explicit checked
compatibility, direct typed fields, exact run association, post-close rejection, isolated
concurrent binding, and unchanged exception propagation. A production backend will additionally
require architecture dependency tests, applicable backend-conformance tests, end-to-end
integration tests, native cleanup checks, and benchmarks for performance claims. Passing a
benchmark never substitutes for correctness tests.

## Related documentation

- [Capability provider](capability-provider.md)
- [Partition preparer](partition-preparer.md)
- [Kernel routes](kernel-routes.md)
- [Runtime/prepare/backend boundary](../architecture/runtime-prepare-backend-boundary.md)
