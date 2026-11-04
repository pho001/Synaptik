# Integrate a [concrete backend](../glossary.md#concrete-backend)

## Outcome and supported [scope](../glossary.md#term-scope)

This guide maps current [backend](../glossary.md#backend) extension contracts into the complete planned [lifecycle](../glossary.md#lifecycle). [Prepare](../glossary.md#prepare)
analysis, slot assignment, backend finalization, [immutable](../glossary.md#term-immutable) physical-representation creation
callbacks, [Runtime](../glossary.md#term-runtime) cold creation and [binding](../glossary.md#term-binding), explicit per-copy validity, and creation/executable
scheduling plus the prepared/bound [buffer](../glossary.md#term-buffer)-transfer and Runtime-only [publication](../glossary.md#term-publication)/result contracts
are current. Runtime's persistent prepared-[resource owner](../glossary.md#ownership-borrowing-and-lease) and non-waiting close/[run](../glossary.md#term-run) [lease](../glossary.md#term-lease),
Prepare's transactional finalizer-result [handoff](../glossary.md#term-handoff), and [Engine](../glossary.md#term-engine) prepared-handle closure are current.
The guide therefore separates compilable extension patterns from conceptual
integration steps.

A **backend** owns a supported group of calculations and the physical resources used to perform
them. A **[kernel](../glossary.md#kernel)** is an executable computation routine, while a **route** is a backend-local
implementation choice. Integrating a backend therefore means supplying [ownership](../glossary.md#term-ownership-meanings), preparation,
storage and cleanup together, not merely making one numerical function callable.

### Name the integration callers

The application opens a CPU/Metal integration and transfers it to Engine's builder. Engine
supplies that integration's collaborators to shared Prepare. Prepare calls backend analysis
with descriptions, assigns requirements, then calls backend finalization with the exact slots.
Runtime later calls creators and binds executables against **one invocation's physical
representations**. There is no call from a generated element loop back to Engine to discover
a service.

Resource ownership follows successful handoffs, not Java visibility. The finalizer owns
acquisitions until successful return; Prepare owns rollback until it constructs the Runtime
prepared owner. Runtime owns each run's created buffers but borrows its input representations.
The application (or NN Model) retains the underlying input-storage duty until result closure.
See [the resource transaction](../architecture/runtime-prepare-backend-boundary.md#the-staged-prepare-handoff)
and [which API copies versus borrows](../explanation/ownership.md#which-call-borrows-or-copies).
Examples below are focused illustrative contracts, not complete backend implementations.
Their initial empty-resource CPU checkpoint does not describe every later native CPU route;
the [CPU preparation atlas](../explanation/packages/cpu.md#read-the-pipeline-before-the-packages)
separates generated/direct Java work from native persistent resources.


## Prerequisites

Read the authoritative [`ARCHITECTURE.md`](../../ARCHITECTURE.md), [module boundaries](../architecture/module-boundaries.md), [dependency rules](../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([dependency glossary](../glossary.md#term-dependency))</span>, and the [concrete backend](../glossary.md#concrete-backend)'s master plan. A [backend](../glossary.md#backend) [module](../glossary.md#term-module) may depend on shared contracts but must not depend on [engine](../glossary.md#term-engine).

## Integration [lifecycle](../glossary.md#lifecycle)

Read the flow as a sequence of decisions, not five names for the same object. Capability reports
what can be owned; [compilation](../glossary.md#compiler-engine-and-execution-stage-names) selects the [owner](../glossary.md#term-owner); preparation fixes implementation and resources;
[Runtime](../glossary.md#term-runtime) invokes that fixed recipe on one [run](../glossary.md#term-run)'s data. **[Lowering](../glossary.md#lowering)** translates a computation into
a more concrete form, **[specialization](../glossary.md#term-specialization)** fixes useful known facts, and **[fusion](../glossary.md#term-fusion)** combines eligible
calculations without removing required observable results or [rounding](../glossary.md#term-rounding) boundaries.

```text
capability -> compile ownership -> backend prepare -> executable -> runtime
```

1. Implement declarative capability reporting for semantic [graph](../glossary.md#graph) facts.
2. Implement a [partition](../glossary.md#partition) preparer that accepts only partitions owned by this [backend](../glossary.md#backend).
3. Lower, specialize, fuse, and select routes inside the backend.
4. During current finalization, construct an [immutable](../glossary.md#term-immutable) [`PreparedExecutable`](../glossary.md#prepared-executable--preparedexecutable) subclass against the
   assigned slots and return it in `BackendPartitionFinalizationResult` with persistent resources
   in physical acquisition order. Roll back finalizer-local acquisition before any failed return.
5. Supply immutable [thread-safe](../glossary.md#term-thread-safe) [buffer](../glossary.md#term-buffer)/workspace creators in one current
   [`PreparedRepresentationPlan`](../glossary.md#prepared-representation-plan--preparedrepresentationplan); each call returns a fresh physical result for one run.
6. Place that plan in the optional first-only `PreparedSchedule.RepresentationCreationStep`, then
   place finalized executables and explicit transfers in `ExecutionStep` and
   `BufferTransferStep` occurrences. Schedule construction validates exact memory-plan identity
   and preserves occurrence order without invoking callbacks.<span data-glossary-navigation="true"> ([identity glossary](../glossary.md#term-identity-meanings); [occurrence glossary](../glossary.md#term-occurrence))</span>
7. At run setup, current package-private Runtime orchestration validates all caller inputs, invokes
   creators before binding, and rolls back partial creation. Cold binding then validates concrete
   representation types and constructs one <span data-glossary-navigation="true"> ([binding glossary](../glossary.md#term-binding))</span>[`BoundInvocation`](../glossary.md#bound-invocation--boundinvocation) with direct typed fields.
8. Cold-bind each current [`PreparedBufferTransfer`](../glossary.md#prepared-buffer-transfer--preparedbuffertransfer) into a [`BoundBufferTransfer`](../glossary.md#bound-buffer-transfer--boundbuffertransfer) with direct typed
   source and destination fields. Runtime owns its validity transition; the backend owns physical
   copy mechanics.
9. End the schedule with dense `PublicationStep` occurrences naming already-created result
   representations. Runtime validates current validity and performs no backend callback.<span data-glossary-navigation="true"> ([callback glossary](../glossary.md#term-callback))</span>
10. Emit typed backend [trace](../glossary.md#trace) contributions when the [producer](../glossary.md#term-producer) contract exists.
11. Expose a backend component that later [Engine](../glossary.md#term-engine) composition can register explicitly.

## Current persistent prepared-resource [ownership](../glossary.md#term-ownership)

### Persistent-resource goal and inputs

Here **persistent** means reused by runs of one [prepared execution](../glossary.md#prepared-execution--preparedexecution), not serialized to a file.
A native **handle** is an identifier used to refer to a resource outside Java; possessing its
numeric value does not establish [ownership](../glossary.md#term-ownership-meanings) or extend its [lifetime](../glossary.md#term-lifetime).

Represent reusable [backend](../glossary.md#backend)-native state, such as a compiled executable, without exposing its
physical handle through [Runtime](../glossary.md#term-runtime). The current Runtime contract is intentionally limited to
unchecked cleanup:

The following conceptual excerpt uses the current contracts and assumes the backend has already
constructed the exact-plan `executable` and acquired `nativeHandle`:

```java
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizationResult;
import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import java.util.List;

final class CompiledBackendExecutable implements PreparedResource {
    private final long nativeHandle;

    CompiledBackendExecutable(long nativeHandle) {
        this.nativeHandle = nativeHandle;
    }

    @Override public void close() {
        // Release nativeHandle through this backend's checked native boundary.
    }
}

CompiledBackendExecutable compiled = new CompiledBackendExecutable(nativeHandle);
// Successful return transfers this resource; the finalizer must not independently close it later.
BackendPartitionFinalizationResult result =
        new BackendPartitionFinalizationResult(executable, List.of(compiled));
```

The snippet assumes finalization has already constructed `executable`, which retains the exact
assigned memory plan. Its other input is the resource list in physical acquisition order. This is
the current finalizer return shape. Shared [Prepare](../glossary.md#prepare) collects successful
results in [partition](../glossary.md#partition) order, validates exact resource [identity](../glossary.md#term-identity-meanings) uniqueness, assembles and validates
the schedule, and finally constructs `PreparedExecution(plan, schedule, resources)`. Current [CPU](../glossary.md#term-cpu)
preparation uses the one-[argument](../glossary.md#term-argument) result constructor and therefore contributes an empty list.

### Persistent-resource result and interpretation

Successful return transfers `compiled` from the finalizer to shared [Prepare](../glossary.md#prepare). Successful
[`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution) construction then transfers it once to [Runtime](../glossary.md#term-runtime); any intervening failure
reverse-closes all accepted [identities](../glossary.md#term-identity-meanings). Runtime retains no native handle, storage accessor,
resource count, [lookup](../glossary.md#term-lookup) key, or [concrete backend](../glossary.md#concrete-backend) type. Two executions with equal plan and schedule
references remain distinct [owners](../glossary.md#term-owner) because `PreparedExecution` uses `Object` [identity](../glossary.md#term-reference-identity) rather than
the former record's [structural equality](../glossary.md#term-structural-equality).

Each [synchronous](../glossary.md#term-synchronous) runner call [leases](../glossary.md#term-lease) the [owner](../glossary.md#term-owner) before inspecting caller inputs or creating [run
state](../glossary.md#run-state--runstate). `execution.close()` rejects later leases and never waits for admitted work. With no active
lease, that caller performs reverse-order attempt-all cleanup; otherwise the last lease does so.
[Backend](../glossary.md#backend) `close()` callbacks execute outside Runtime's [lifecycle](../glossary.md#lifecycle) monitor. The first unchecked
failure is primary, later distinct failures are suppressed in cleanup order, and repeated close
does not replay a failure.

This contract is for [immutable](../glossary.md#term-immutable) state reused across runs. Per-[run](../glossary.md#term-run) [buffers](../glossary.md#term-buffer) and workspaces still
belong to one [`RunState`](../glossary.md#run-state--runstate); backend-global caches, schedule occurrences, and executable references
must not become alternative owners. A finalizer owns local [rollback](../glossary.md#term-rollback) before successful return;
shared Prepare owns later rollback and the single successful Runtime transfer. [Engine](../glossary.md#term-engine) closes
inward owners through ordinary and advanced prepared handles: published handles close explicitly
or during Engine shutdown, and temporary one-shot or tuning preparations close after their
results. [Metal](../glossary.md#term-metal) currently exercises this contract with typed custom-pipeline and [MPSGraph](../glossary.md#term-mpsgraph)
resources, and explicit `Engine.builder()` composition exposes both its complete backend-local
route and its contribution to supported mixed [CPU](../glossary.md#term-cpu)/Metal schedules.

## Current representation-creation pattern

### Representation-creation goal and inputs

Describe one [borrowed](../glossary.md#ownership-borrowing-and-lease) caller input, one [backend](../glossary.md#backend)-created internal [buffer](../glossary.md#term-buffer), and one backend-created
workspace against the exact [prepared memory plan](../glossary.md#prepared-memory-plan--preparedmemoryplan) supplied by shared finalization. The creator
objects are [immutable](../glossary.md#term-immutable) and stateless; each call returns a fresh concrete result.

```java
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import java.util.List;

final class CpuBuffer implements BufferRepresentation {
    void copyFrom(CpuBuffer source) { /* copy concrete storage */ }
    @Override public void close() { /* release concrete storage */ }
}

final class CpuWorkspace implements WorkspaceRepresentation {
    @Override public void close() { /* release concrete scratch */ }
}

final class CpuBufferCreator implements PreparedRepresentationPlan.BufferCreator {
    @Override public CpuBuffer create() {
        // Runtime invokes this per run, so mutable storage cannot be shared by preparations.
        return new CpuBuffer();
    }
}

final class CpuWorkspaceCreator implements PreparedRepresentationPlan.WorkspaceCreator {
    @Override public CpuWorkspace create() {
        // Scratch belongs to the new RunState, not to this reusable creator object.
        return new CpuWorkspace();
    }
}

PreparedRepresentationPlan representationPlan =
        // This is a recipe only: constructing it invokes none of the creators.
        new PreparedRepresentationPlan(
                plan,
                List.of(
                        List.of(
                                new PreparedRepresentationPlan.CallerInput(),
                                new PreparedRepresentationPlan.CreatedBuffer(
                                        new CpuBufferCreator()))),
                List.of(new CpuWorkspaceCreator()));
```

### Representation-creation result and interpretation

The [immutable](../glossary.md#term-immutable) plan contains only dense origins and exact typed [callback](../glossary.md#term-callback) references. Constructing
it allocates no physical [buffer](../glossary.md#term-buffer) or workspace. During cold [run](../glossary.md#term-run) setup, [Runtime](../glossary.md#term-runtime) validates the
complete caller-input list before invoking `CpuBufferCreator`, then invokes workspace creators.
Every successful result is run-owned only after complete state construction. A partial failure
closes successfully created results in reverse creation order and preserves the original
unchecked failure, with cleanup failures suppressed.

This pattern has no [registry](../glossary.md#term-registry), service [lookup](../glossary.md#term-lookup), reflection, raw object [payload](../glossary.md#artifact-envelope-and-atomic-publication), or [backend](../glossary.md#backend) switch.
Allocation occurs inside explicitly prepared backend callbacks during cold setup, never inside
`PreparedExecutable.bind` or `BoundInvocation.execute`. A [concrete backend](../glossary.md#concrete-backend) must test callback
freshness across positions and concurrent runs, immutability/thread safety of retained creators,
null/duplicate rejection, and physical cleanup.

## Current cold-[binding](../glossary.md#term-binding) pattern

“Cold” work performs setup and compatibility checks before repeated computation. **Binding**
connects the prepared recipe to one [run](../glossary.md#term-run)'s actual representations and produces direct typed
references for the **[hot path](../glossary.md#term-hot-path)**, the execution code whose repeated overhead matters. Cold binding
is not another opportunity to choose an [owner](../glossary.md#term-owner) or silently change a selected route.

The following focused skeleton uses only current [Runtime](../glossary.md#term-runtime) contracts. It is a local extension
pattern, not a complete [backend](../glossary.md#backend): a current [Prepare](../glossary.md#prepare) finalizer supplies the exact plan and dense
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
                List.of(new WorkspaceSelection(0)),
                List.of(BufferAccess.READ_ONLY));
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
                // Cold checked binding captures direct resources for the later hot action.
                state, (CpuBuffer) buffers[0], (CpuWorkspace) workspaces[0]);
    }
}
```

The two `instanceof` checks run once for each selected resource during cold binding. Runtime
passes fresh [nominal](../glossary.md#term-nominal) arrays to `bindCompatible` only after all checks succeed. The invocation
stores the concrete references in final typed fields; it does not retain those arrays or perform
slot [lookup](../glossary.md#term-lookup) or compatibility casting inside `executeBound()`.

Selections use dense [`PreparedMemoryPlan`](../glossary.md#prepared-memory-plan--preparedmemoryplan) and [`RunState`](../glossary.md#run-state--runstate) positions, not slot numeric components.
They may repeat when the same representation fills multiple operand roles. The executable must
retain the exact plan reference and be [immutable](../glossary.md#term-immutable) and [thread-safe](../glossary.md#term-thread-safe) so it can bind concurrently to
distinct run states. Each invocation retains one exact state, is not thread-safe, and rejects
execution after that state closes.

Every [buffer](../glossary.md#term-buffer) selection must declare its actual access. `READ_ONLY` requires an existing valid
copy, `WRITE_ONLY` declares a produced copy without reading its old value, and `READ_WRITE` does
both. The runner trusts these immutable declarations: it validates reads, invalidates every copy
of each output buffer before the backend call, and validates only exact declared writes after
success. Backend authors must not omit or understate writes; Runtime does not inspect storage or
infer access from [graph](../glossary.md#graph) facts or physical types.

Cold binding may allocate the invocation and temporary arrays, but the current contract forbids
acquiring any auxiliary closeable or native binding resource. Binding changes no [ownership](../glossary.md#term-ownership-meanings) and
has no partial-failure cleanup protocol. A future contract must add an explicit [lifecycle](../glossary.md#lifecycle) before
a backend may acquire such a resource while binding.

## Current [buffer](../glossary.md#term-buffer)-transfer pattern

[Materialization](../glossary.md#materialization) uses the same explicit buffer-transfer contract when the destination is an
equivalent already-created representation required by later work. A [backend](../glossary.md#backend) adds no separate
materialization recipe, allocation path, route search, or coherence layer.

```java
import io.github.pho001.synaptik.runtime.execution.BoundBufferTransfer;
import io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer;

final class CpuBufferTransfer extends BoundBufferTransfer {
    private final CpuBuffer source;
    private final CpuBuffer destination;

    CpuBufferTransfer(RunState state, CpuBuffer source, CpuBuffer destination) {
        super(state, 0, 0, 1);
        this.source = source;
        this.destination = destination;
    }

    @Override protected void executeTransfer() {
        // Copy through the already-selected route using these direct concrete fields.
        destination.copyFrom(source);
    }
}

final class CpuPreparedBufferTransfer extends PreparedBufferTransfer {
    CpuPreparedBufferTransfer(PreparedMemoryPlan plan) {
        super(plan, 0, 0, 1);
    }

    @Override protected boolean acceptsSourceBufferRepresentation(
            BufferRepresentation representation) {
        return representation instanceof CpuBuffer;
    }

    @Override protected boolean acceptsDestinationBufferRepresentation(
            BufferRepresentation representation) {
        return representation instanceof CpuBuffer;
    }

    @Override protected BoundBufferTransfer bindCompatible(
            RunState state,
            BufferRepresentation source,
            BufferRepresentation destination) {
        return new CpuBufferTransfer(
                // The destination already exists; binding must not allocate or choose a route.
                state, (CpuBuffer) source, (CpuBuffer) destination);
    }
}
```

The two compatibility checks and casts are cold. The bound action retains concrete source and
destination fields and performs no [Runtime](../glossary.md#term-runtime) [lookup](../glossary.md#term-lookup). `execute()` makes a valid destination a no-op
without requiring the source valid. For an invalid destination, Runtime requires the source valid,
calls `executeTransfer()` once, and marks only the destination valid after success. If
`copyFrom` throws, the exact failure propagates and Runtime validity remains unchanged; the
backend must treat partially written destination bytes as invalid. A backend test should cover
all three cases: no-op, successful one-call copy, and failure with unchanged validity.

The current runner binds every `BufferTransferStep` [occurrence](../glossary.md#term-occurrence) before its hot loop. The transfer
recipe itself does not allocate a
destination or choose a route.

## Current executable and [publication](../glossary.md#term-publication) scheduling pattern

### Scheduling goal and inputs

Make the representation plan reachable as the first [occurrence](../glossary.md#term-occurrence), then order one prepared transfer,
two occurrences of one finalized `CpuExecutable`, and one final [publication](../glossary.md#term-publication). This example assumes
`plan` is the exact shared plan supplied by current [Prepare](../glossary.md#prepare) finalization; it does not construct a
runner action or invoke a creator.

```java
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import java.util.List;

CpuExecutable executable = new CpuExecutable(plan);
PreparedSchedule.ExecutionStep occurrence =
        new PreparedSchedule.ExecutionStep(executable);
PreparedSchedule.RepresentationCreationStep creation =
        new PreparedSchedule.RepresentationCreationStep(representationPlan);
PreparedSchedule.BufferTransferStep transfer =
        new PreparedSchedule.BufferTransferStep(new CpuPreparedBufferTransfer(plan));
PreparedSchedule.PublicationStep publication =
        new PreparedSchedule.PublicationStep(
                new PreparedPublication(plan, 0, 1, 0));
PreparedSchedule schedule =
        // Repetition means two executions, not two cleanup owners for the same executable.
        new PreparedSchedule(
                plan, List.of(creation, occurrence, occurrence, transfer, publication));
```

### Scheduling result and interpretation

`schedule.steps()` retains the exact creation prefix, the same executable [occurrence](../glossary.md#term-occurrence) twice, the
transfer, and the [publication](../glossary.md#term-publication) suffix in deterministic order. Every occurrence reports the exact
`plan` reference. Repetition means execute the prepared region twice when the runner consumes
the schedule; it does not duplicate executable, representation, or cleanup [ownership](../glossary.md#term-ownership-meanings). Result zero
names [buffer](../glossary.md#term-buffer) zero's representation one, which the explicit transfer made valid before the suffix.
Empty and executable-only schedules remain valid for compatibility; a later [Prepare](../glossary.md#prepare) validator may
require creation and complete result reachability for runnable work.

Schedule construction only validates and snapshots the recipe. It does not bind or execute the
`CpuExecutable` or transfer, invoke creator callbacks, create a [`RunState`](../glossary.md#run-state--runstate), allocate or close
resources, perform [materialization](../glossary.md#materialization), publish, or create a result. Publication remains a separate
runner-invoked [Runtime](../glossary.md#term-runtime) action rather than [backend](../glossary.md#backend) behavior hidden in an executable or transfer.

## Current [publication](../glossary.md#term-publication) boundary for [backend](../glossary.md#backend) authors

**Publication** makes an already-computed representation available as a result; **[materialization](../glossary.md#materialization)**
creates the required physical or [detached](../glossary.md#detached-model-results) form. Publishing does not itself copy bytes to the [host](../glossary.md#term-host).
A representation's **validity** says whether that particular copy currently contains the required
logical value, not whether its memory was merely allocated.

[`PreparedPublication`](../glossary.md#prepared-publication--preparedpublication) and [`BoundPublication`](../glossary.md#bound-publication--boundpublication) are final [Runtime](../glossary.md#term-runtime) types, not backend extension
points. A [prepared publication](../glossary.md#prepared-publication--preparedpublication) uses dense [buffer](../glossary.md#term-buffer), representation, and result positions. Its cold
[binding](../glossary.md#term-binding) resolves the selected representation once, and its `publish()` action requires that exact
copy to be valid while the state is open. It changes only a one-shot flag and never calls the
representation or backend.

A backend's responsibility is therefore upstream of publication: its executable and transfer
actions must perform their prepared physical work correctly, and the runner applies the
explicit Runtime validity transitions. If the desired result exists in another representation,
the schedule must contain a prepared transfer before its publication suffix. Publication never
discovers a source, selects a route, converts or copies bytes, retries, or falls back.

After every dense ordered [occurrence](../glossary.md#term-occurrence) publishes, current Runtime [`RunResult`](../glossary.md#run-result--runresult) privately retains the
direct references and [leases](../glossary.md#term-lease) the complete [`RunState`](../glossary.md#run-state--runstate). The outward [Engine](../glossary.md#term-engine) result exposes [metadata](../glossary.md#term-metadata)
and explicit detached bounded host materialization, never a physical representation or live
backend value. Closing the result delegates cleanup orchestration to the state, which still skips
[borrowed](../glossary.md#ownership-borrowing-and-lease) inputs and calls backend-owned cleanup only for [run](../glossary.md#term-run)-owned representations. Intentional
result [aliases](../glossary.md#aliasing-and-arena) do not duplicate cleanup because state binding [identity](../glossary.md#term-identity-meanings) remains unique.

## Current explicit built-in registration

```java
try (Engine.Builder builder = Engine.builder()) {
    // Transfer cleanup duty before compilation; no Runtime discovery is involved.
    builder.takeOwnership(CpuBackendIntegration.open());
    builder.takeOwnership(MetalBackendIntegration.open(metalConfiguration));
    try (Engine engine = builder.build()) {
        // compile, prepare, and run a single-owner or supported mixed-owner plan
    }
}
```

Each `takeOwnership` call makes a supported built-in integration visible before [compilation](../glossary.md#compiler-engine-and-execution-stage-names) and
preparation. Shared [Prepare](../glossary.md#prepare) assigns deterministic [owner](../glossary.md#term-owner)-indexed representations; each [backend](../glossary.md#backend)
contributes its exact physical creators, while [Engine](../glossary.md#term-engine) composes explicit transfer occurrences.
[Runtime](../glossary.md#term-runtime) does not use [classpath](../glossary.md#term-classpath) scanning, `ServiceLoader`, or a [service locator](../glossary.md#term-service-locator) to discover those
components during execution. A generic backend [plugin](../glossary.md#term-plugin) registration [SPI](../glossary.md#term-service-provider-interface-spi) and general cross-backend
conversion system remain planned.

## Resources, [concurrency](../glossary.md#term-concurrency), and failures

A [backend](../glossary.md#backend) implements physical [buffer](../glossary.md#term-buffer) and workspace representations plus their allocation,
[release](../glossary.md#term-release), transfer, and access mechanics. Its retained creator callbacks and [prepared executable](../glossary.md#prepared-executable--preparedexecutable)
recipes are [immutable](../glossary.md#term-immutable) and reusable. Each active complete logical execution has one isolated
[`RunState`](../glossary.md#run-state--runstate); caller inputs are [borrowed](../glossary.md#ownership-borrowing-and-lease) and initially valid, created internal buffers are [run](../glossary.md#term-run)-owned
and initially invalid, workspaces are run-owned scratch without logical validity, and a completed
current result [leases](../glossary.md#term-lease) cleanup of the whole state. Each [callback](../glossary.md#term-callback) must return a fresh result
for every position and concurrent run.

Before the [hot path](../glossary.md#term-hot-path), current backend extension code performs explicit checked compatibility and
creates typed [bound invocation](../glossary.md#bound-invocation--boundinvocation) objects with direct representation references. It must not rely on
raw `Object`, unchecked generic access, reflection, string [dispatch](../glossary.md#term-dispatch), a [registry](../glossary.md#term-registry), or repeated
hot-path casts. [Runtime](../glossary.md#term-runtime) orchestrates cleanup, while the backend representation performs physical
release. Failure cleanup must never close borrowed inputs or outputs already transferred from the
run. Current [publication](../glossary.md#term-publication) transfers no individual representation [ownership](../glossary.md#term-ownership-meanings); borrowed published
representations remain caller-owned for the complete result lease.

Capability rejection occurs before ownership. [Lowering](../glossary.md#lowering) and finalization failure occur during
[prepare](../glossary.md#prepare). Representation creation and execution failure occur during run setup or execution and
must not trigger hidden cross-backend fallback.

## [Validation](../glossary.md#term-validation)

Current [backend](../glossary.md#backend) tests should prove creator immutability and freshness, reverse partial-failure
cleanup, validity initialization, exact plan [identity](../glossary.md#term-identity-meanings), selection order, explicit checked
compatibility, direct typed fields, exact [run](../glossary.md#term-run) association, post-close rejection, isolated
concurrent [binding](../glossary.md#term-binding), unchanged exception propagation, and correct validity before any scheduled
[publication](../glossary.md#term-publication). [Runtime](../glossary.md#term-runtime) tests, rather than backend hooks, own publication one-shot, suffix, [alias](../glossary.md#aliasing-and-arena),
partial-result, and whole-state [lease](../glossary.md#term-lease) behavior. A production backend will additionally
require architecture [dependency](../glossary.md#term-dependency) tests, applicable backend-[conformance](../glossary.md#term-conformance) tests, end-to-end
integration tests, native cleanup checks, and benchmarks for performance claims. Passing a
benchmark never substitutes for correctness tests.

## Related documentation

- [Capability provider](capability-provider.md)
- [Partition preparer](partition-preparer.md)<span data-glossary-navigation="true"> ([Partition glossary](../glossary.md#partition))</span>
- [Kernel routes](kernel-routes.md)<span data-glossary-navigation="true"> ([Kernel glossary](../glossary.md#kernel))</span>
- [Runtime/prepare/backend boundary](../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../glossary.md#term-runtime); [prepare glossary](../glossary.md#prepare); [backend glossary](../glossary.md#backend))</span>
