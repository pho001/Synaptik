# Integrate a concrete backend

## Outcome and supported scope

This guide maps current backend extension contracts into the complete planned lifecycle. Prepare
analysis and Runtime cold binding are current, but slot assignment, backend finalization,
scheduling, Engine composition, and every production concrete backend remain planned. The guide
therefore separates compilable extension patterns from conceptual integration steps.

## Prerequisites

Read the authoritative [`ARCHITECTURE.md`](../../ARCHITECTURE.md), [module boundaries](../architecture/module-boundaries.md), [dependency rules](../architecture/dependency-rules.md), and the concrete backend's master plan. A backend module may depend on shared contracts but must not depend on engine.

## Integration lifecycle

```text
capability -> compile ownership -> backend prepare -> executable -> runtime
```

1. Implement declarative capability reporting for semantic graph facts.
2. Implement a partition preparer that accepts only partitions owned by this backend.
3. Lower, specialize, fuse, and select routes inside the backend.
4. During later finalization, construct an immutable `PreparedExecutable` subclass against the
   assigned slots.
5. At run setup, use current cold binding to validate concrete representation types and construct
   one current `BoundInvocation` with direct typed fields.
6. Emit typed backend trace contributions when the producer contract exists.
7. Expose a backend component that later Engine composition can register explicitly.

## Current cold-binding pattern

The following focused skeleton uses only current Runtime contracts. It is a local extension
pattern, not a complete backend: a later Prepare finalizer must supply the exact plan and dense
selections, and later Runtime work must create representations and schedule the invocation.

```java
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.List;

final class CpuBuffer implements BufferRepresentation {
    @Override public void close() { /* release concrete storage */ }
}

final class CpuWorkspace implements WorkspaceRepresentation {
    @Override public void close() { /* release concrete scratch */ }
}

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
release, transfer, and access mechanics. Prepared executable recipes and immutable persistent
prepared resources are reusable. Each active complete logical execution has one isolated
`RunState`; caller inputs are borrowed, internal buffers and workspaces are run-owned, and
published outputs transfer or lease ownership to the later result.

Before the hot path, current backend extension code performs explicit checked compatibility and
creates typed bound invocation objects with direct representation references. It must not rely on
raw `Object`, unchecked generic access, reflection, string dispatch, a registry, or repeated
hot-path casts. Runtime orchestrates cleanup, while the backend representation performs physical
release. Failure cleanup must never close borrowed inputs or outputs already transferred from the
run.

Capability rejection occurs before ownership. Lowering or resource-creation failure occurs during prepare. Execution failure occurs during run and must not trigger hidden cross-backend fallback.

## Validation

Current executable unit tests should prove exact plan identity, selection order, explicit checked
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
