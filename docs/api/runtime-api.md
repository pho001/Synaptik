# Runtime API

## Purpose and implementation status

This reference explains the implemented Runtime foundations for prepared-memory geometry, one
run's physical-representation lifecycle, checked binding of reusable backend recipes, ordered
executable scheduling, and the current Prepare-owned handoff that constructs that geometry and
finalizes backend recipes. The current public surface contains:

- `runtime.memory`: `BufferSlot`, `WorkspaceSlot`, and `PreparedMemoryPlan` with its nested
  `BufferEntry` and `WorkspaceEntry` records;
- `runtime.resource`: the nominal `BufferRepresentation` and `WorkspaceRepresentation` lifecycle
  roles implemented by concrete backends; and
- `runtime.run`: `RunResourceOwnership`, `BufferRepresentationBinding`, and `RunState`;
- `runtime.execution`: `PreparedExecutable`, its nested `BufferSelection` and
  `WorkspaceSelection` records, and `BoundInvocation`; and
- `runtime.schedule`: `PreparedSchedule`, its sealed nested `Step` contract, and the sole current
  `ExecutionStep` variant.

The geometry, per-run representation carrier, cold-bound invocation contracts, executable-only
schedule recipe, Prepare-owned resource assignments, typed backend finalization, and
`PreparedPartition` association are current. Physical allocation and storage access, full
validity/residency, transfer/materialization/publication step variants, publication/results,
public Prepare orchestration, schedule consumption, and a runnable Runtime facade remain planned.
No production concrete backend currently implements the finalization or execution contracts.

## Mental model

```text
compile facts             prepare handoff               Runtime state
ValueId / requirement -> source-to-slot assignment -> PreparedMemoryPlan + RunState
current                  current                      current
                                                              |
                                                              v
                  PreparedSchedule -> cold bind -> BoundInvocation -> execute
                  current              current       current            current contract
```

Read the flow from logical source facts toward invocation state. Prepare exposes exact buffer and
workspace requirements through `BackendPartitionAnalysis`, then its current package-internal
handoff retains source-to-slot associations and constructs the Runtime geometry. Current
`RunState` accepts already-created representations in that geometry's encounter order. A current
`PreparedExecutable` selects those representations by dense position, checks backend
compatibility during cold binding, and creates a per-run `BoundInvocation`. None of these
contracts allocates or transfers physical storage or supplies a runner. The current schedule
orders executable occurrences only; it does not bind or execute them.

## Current slot identities

`BufferSlot` and `WorkspaceSlot` are public, deeply immutable records with one `long value`
component. Each accepts every value from zero through `Long.MAX_VALUE`; no sentinel is reserved.
A negative value fails with `IllegalArgumentException` and message
`value must be non-negative`.

Each component is opaque outside its owning plan context. Neither record stores an owner
reference, so ordinary record equality and hashing compare only the numeric component within the
same nominal record type. A buffer slot and workspace slot with the same number are distinct, and
two plans may reuse a number without referring to the same conceptual slot. Diagnostic record
text is not a serialization format.

Creating a slot does not allocate, acquire, retain, release, or identify physical storage. A
buffer slot is not a `ValueId`; a workspace slot is not an analysis-local workspace requirement
ID. Neither is an address, storage handle, allocation, device, residency fact, or resource.

## Current prepared-memory plan

`PreparedMemoryPlan` is the immutable final byte geometry for one prepared plan. Its `buffers`
and `workspaces` components are ordered immutable snapshots. Each entry retains the exact slot
reference and records:

- `byteSize`, an exact non-negative byte count; zero is valid; and
- `byteAlignment`, an exact positive power of two from `1` through `1L << 62`.

Buffer slots must be unique within `buffers`, and workspace slots must be unique within
`workspaces`. The two uniqueness domains are separate. Either list may be empty, including both
lists together.

The plan retains neither the caller's list containers nor any Prepare requirement or
source-to-slot association. It does not sort, renumber, merge, derive, allocate, bind, own, or
release storage.

## Focused valid-plan example

### Goal and inputs

Describe one 24-byte reusable buffer position and one 64-byte workspace position. The supplied
order is already the final deterministic plan order.

```java
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import java.util.List;

BufferSlot input = new BufferSlot(0L);
WorkspaceSlot scratch = new WorkspaceSlot(0L);

PreparedMemoryPlan plan =
        new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(input, 24L, 4L)),
                List.of(new PreparedMemoryPlan.WorkspaceEntry(scratch, 64L, 16L)));
```

### Result and interpretation

`plan.buffers().getFirst()` retains the exact `input` reference with 24-byte size and 4-byte
alignment. `plan.workspaces().getFirst()` retains the exact `scratch` reference with 64-byte size
and 16-byte alignment. Both lists are immutable and preserve their supplied order.

This result describes geometry only. It does not say which graph value produced the buffer
requirement, which backend analysis requested the workspace, whether the two positions alias
physical storage, or how a run accesses either position.

## Current per-run representation foundation

`BufferRepresentation` and `WorkspaceRepresentation` are deliberately distinct nominal
`AutoCloseable` roles. Each exposes only `close()` without a checked exception. A concrete backend
implements the physical storage and cleanup mechanics; the shared Runtime API exposes no storage,
backend, device, transfer, validity, or residency accessor.

`BufferRepresentationBinding` retains one exact buffer representation and one ownership value:

- `BORROWED` keeps cleanup responsibility with the caller for the complete run; and
- `RUN_OWNED` transfers cleanup responsibility only after `RunState` construction succeeds.

Every workspace supplied to a successfully constructed state is run-owned. Construction failure
transfers no ownership and closes nothing.

One `RunState` covers one complete logical run, including all backend partitions. It retains the
exact `PreparedMemoryPlan`, snapshots only the supplied list structure into private arrays, and
retains every exact binding and representation reference. Buffer and workspace indices are dense
zero-based positions in `plan.buffers()` and `plan.workspaces()` encounter order; they are not the
numeric values inside `BufferSlot` or `WorkspaceSlot`. A buffer position has one or more ordered
representations, while a workspace position has exactly one representation.

The carrier does not say which buffer representation is valid or resident and provides no
coherence, transfer, or backend-compatibility behavior. The same representation object cannot
occur twice anywhere in one state, including across buffer and workspace domains.

## Focused run-state example

### Goal and inputs

Bind one borrowed caller buffer, one run-owned internal buffer, and one run-owned workspace to the
single buffer and workspace positions from the preceding plan. The counters make cleanup
observable without claiming a physical storage implementation.

```java
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

AtomicInteger borrowedCloses = new AtomicInteger();
AtomicInteger ownedCloses = new AtomicInteger();
AtomicInteger workspaceCloses = new AtomicInteger();

BufferRepresentation borrowed = borrowedCloses::incrementAndGet;
BufferRepresentation owned = ownedCloses::incrementAndGet;
WorkspaceRepresentation workspace = workspaceCloses::incrementAndGet;

RunState state =
        new RunState(
                plan,
                List.of(
                        List.of(
                                new BufferRepresentationBinding(
                                        borrowed, RunResourceOwnership.BORROWED),
                                new BufferRepresentationBinding(
                                        owned, RunResourceOwnership.RUN_OWNED))),
                List.of(workspace));

BufferRepresentationBinding first = state.bufferRepresentation(0, 0);
state.close();
state.close();
```

### Result and interpretation

Before closure, `first` is the exact borrowed binding supplied at construction. The state reports
one buffer position, two ordered buffer representations, and one workspace position. After both
`close()` calls, `borrowedCloses.get()` is `0`, while `ownedCloses.get()` and
`workspaceCloses.get()` are each `1`. This demonstrates ownership-sensitive, idempotent cleanup;
it does not allocate storage, select a backend, establish representation validity, or run a
prepared executable.

## Run-state lifecycle and failures

`RunState.close()` marks the state closed before physical cleanup. It closes workspaces from last
position to first, then run-owned buffer representations from last buffer position to first and
last representation to first. Borrowed buffers are skipped. Every owned representation is
attempted once. The first `RuntimeException` or `Error` is rethrown after all attempts, with later
failures attached in cleanup encounter order as suppressed exceptions. Repeated closure performs
no cleanup and does not rethrow an earlier failure.

Representation access after closure fails first with
`IllegalStateException("run state is closed")`. The retained plan, slot counts, per-buffer
representation counts, and `isClosed()` remain inspectable. `RunState` is not thread-safe;
callers must not race access and closure on one instance. Concurrent runs may share the immutable
plan but require distinct run-owned representations. Borrowed representations may be shared only
when the caller guarantees their lifetime, safe access, and external synchronization.

Construction validates top-level nulls, plan-sized counts, each ordered buffer position, and then
each ordered workspace position. Null failures identify the argument or indexed position. Count,
empty-list, and duplicate-identity failures are `IllegalArgumentException`; every duplicate exact
identity reports `representation is already bound to this run`. Invalid access indices use
`IndexOutOfBoundsException` and identify `bufferIndex`, `representationIndex`, or
`workspaceIndex` with the rejected value.

## Failures

Entry construction checks slot nullness, then byte size, then byte alignment. A null slot fails
with `NullPointerException("slot")`; a negative size fails with
`IllegalArgumentException("byteSize must be non-negative")`; and an invalid alignment fails with
`IllegalArgumentException("byteAlignment must be a positive power of two")`.

Plan construction checks both top-level lists before scanning entries. It then validates and
snapshots buffers before workspaces. A null entry reports its supplied zero-based list position.
The first later duplicate reports that position and the diagnostic slot text. These failures
reject ambiguous geometry; they do not perform assignment, deduplication, or recovery.

## Current prepared executable and bound invocation

`PreparedExecutable` is an immutable reusable recipe for one prepared computation region. Its
constructor retains the exact `PreparedMemoryPlan` reference and snapshots ordered selections
into private arrays:

- `BufferSelection(bufferIndex, representationIndex)` addresses a dense position in
  `memoryPlan().buffers()` and then a dense representation position in that buffer's `RunState`
  bindings; and
- `WorkspaceSelection(workspaceIndex)` addresses a dense position in
  `memoryPlan().workspaces()`.

These indices are list positions, not `BufferSlot.value()` or `WorkspaceSlot.value()` values.
Selections may be empty or repeated. Repetition represents repeated operand roles and does not
duplicate resource ownership in the state.

`bind(runState)` is the cold boundary. It first requires an open state whose `memoryPlan()` is the
exact same object as the executable's plan; an equal plan constructed separately is rejected. It
resolves buffers in selection order, then workspaces, and calls the backend's checked
compatibility hook exactly once for each resolved selection. Only after all checks pass does it
call `bindCompatible` with fresh nominal arrays in the same order.

The backend uses explicit checked tests such as `instanceof`, then constructs a
`BoundInvocation` with direct concrete typed fields. The returned invocation must retain the exact
supplied `RunState`. Binding may allocate the temporary Java Virtual Machine (JVM) arrays and
invocation object, but it acquires no auxiliary closeable or native binding resource, changes no
ownership, and performs no cleanup on failure.

`BoundInvocation.execute()` checks that its retained state is still open, then calls the
backend's `executeBound()` method. It performs no slot lookup, compatibility cast, graph work,
backend discovery, route/configuration search, allocation, transfer, residency decision,
publication, tuning, or tracing. Sequential calls while open are permitted. One invocation is not
thread-safe and must not race execution with state closure; after closure it fails with
`IllegalStateException("run state is closed")` before backend work.

One immutable executable may bind concurrently to distinct run states. Each resulting invocation
belongs to exactly one state and owns neither that state nor its buffer/workspace representations.
Concrete executable subclasses must therefore be immutable and thread-safe, while invocation
subclasses keep their per-run direct references isolated.

## Focused cold-binding example

### Goal and inputs

Bind one backend-specific buffer and workspace from a matching open state, execute once through
direct typed fields, then demonstrate the post-close guard. This local fake backend illustrates
the extension contract; it is not a production backend or a Prepare finalizer.

```java
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.List;

final class ExampleBuffer implements BufferRepresentation {
    int calls;
    @Override public void close() {}
}

final class ExampleWorkspace implements WorkspaceRepresentation {
    int calls;
    @Override public void close() {}
}

final class ExampleInvocation extends BoundInvocation {
    private final ExampleBuffer buffer;
    private final ExampleWorkspace workspace;

    ExampleInvocation(RunState state, ExampleBuffer buffer, ExampleWorkspace workspace) {
        super(state);
        this.buffer = buffer;
        this.workspace = workspace;
    }

    @Override protected void executeBound() {
        buffer.calls++;
        workspace.calls++;
    }
}

final class ExampleExecutable extends PreparedExecutable {
    ExampleExecutable(PreparedMemoryPlan plan) {
        super(
                plan,
                List.of(new BufferSelection(0, 0)),
                List.of(new WorkspaceSelection(0)));
    }

    @Override protected boolean acceptsBufferRepresentation(
            int index, BufferRepresentation representation) {
        return index == 0 && representation instanceof ExampleBuffer;
    }

    @Override protected boolean acceptsWorkspaceRepresentation(
            int index, WorkspaceRepresentation representation) {
        return index == 0 && representation instanceof ExampleWorkspace;
    }

    @Override protected BoundInvocation bindCompatible(
            RunState state,
            BufferRepresentation[] buffers,
            WorkspaceRepresentation[] workspaces) {
        return new ExampleInvocation(
                state, (ExampleBuffer) buffers[0], (ExampleWorkspace) workspaces[0]);
    }
}
```

Given the one-buffer/one-workspace `plan` from the earlier geometry example:

```java
ExampleBuffer buffer = new ExampleBuffer();
ExampleWorkspace workspace = new ExampleWorkspace();
RunState state =
        new RunState(
                plan,
                List.of(
                        List.of(
                                new BufferRepresentationBinding(
                                        buffer, RunResourceOwnership.RUN_OWNED))),
                List.of(workspace));

BoundInvocation invocation = new ExampleExecutable(plan).bind(state);
invocation.execute();
state.close();
```

### Result and interpretation

Both `calls` fields are `1`. Compatibility was checked during `bind`, and execution used the two
direct concrete fields. Calling `invocation.execute()` after `state.close()` fails before either
counter changes. This proves the current binding and lifecycle guard only; it does not allocate
storage, finalize a backend analysis, schedule work, transfer values, or publish a result.

## Current prepared schedule

`PreparedSchedule` is an immutable reusable recipe that associates an ordered list of work
occurrences with one exact `PreparedMemoryPlan`. Its sealed nested `Step` interface exposes only
that plan association. The sole current variant is
`ExecutionStep(PreparedExecutable)`, whose `memoryPlan()` result is exactly
`executable().memoryPlan()`.

Construction validates each occurrence in supplied order and requires reference identity with
the schedule plan; a structurally equal but separately constructed plan is not the same prepared
association. Only after the complete scan succeeds does the schedule use `List.copyOf` to retain
an immutable ordered snapshot. It keeps exact step and executable references. Empty schedules,
repeated step references, and repeated executable references are valid because each list position
is one explicit occurrence and does not create another ownership relationship.

### Focused schedule example

#### Goal and inputs

Order two occurrences of the current `ExampleExecutable` recipe from the cold-binding example.
Both occurrences use the same exact plan; no run state or physical representation is needed to
construct the schedule.

```java
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.List;

ExampleExecutable executable = new ExampleExecutable(plan);
PreparedSchedule.ExecutionStep occurrence =
        new PreparedSchedule.ExecutionStep(executable);
ArrayList<PreparedSchedule.Step> supplied =
        new ArrayList<>(List.of(occurrence, occurrence));

PreparedSchedule schedule = new PreparedSchedule(plan, supplied);
supplied.clear();
```

#### Result and interpretation

`schedule.memoryPlan()` is the exact `plan` object, and `schedule.steps()` still contains the
same `occurrence` reference twice in supplied order after the mutable source list is cleared. The
returned list is immutable. This proves deterministic executable scheduling and snapshot
isolation only. Construction does not bind or execute the executable, allocate or close a
resource, create a `RunState`, or select transfer, materialization, or publication behavior.

A null executable fails with `NullPointerException("executable")`. A null schedule plan, list,
or element reports `memoryPlan`, `steps`, or `steps[index]`, respectively. The first step whose
plan is not the exact schedule plan fails with
`IllegalArgumentException("steps[index] memory plan does not match schedule memory plan")`.
Transfer, materialization, and publication variants wait for later Runtime-owned residency and
result contracts; they are not hidden inside `ExecutionStep`.

## Current Prepare assignment and backend finalization

The public root of `io.github.pho001.synaptik.prepare` now contains four contracts that bridge
analysis declarations to current Runtime recipes:

- `PreparationResourceAssignment.Buffer` and `.Workspace` retain one exact analysis requirement,
  its exact assigned slot, and the dense index of that slot's entry in `PreparedMemoryPlan`;
- `BackendPartitionFinalization<P>` retains one exact typed analysis, the exact shared memory
  plan, and an immutable assignment list in analysis-requirement order;
- `BackendPartitionFinalizer<P>` is implemented by the owning concrete backend to construct one
  immutable `PreparedExecutable` recipe after assignment; and
- `PreparedPartition` retains the exact `PlannedPartition` and finalized executable references.

The package-internal complete-set handoff validates expected partition coverage, exact projected
source references, and backend ownership before assignment. It traverses partitions and
requirements in stored order. A first-seen buffer `ValueId` receives the next dense buffer slot;
later declarations of that value share the exact slot and its plan entry uses the maximum
declared size and alignment. Every workspace declaration receives a fresh dense workspace slot
with unchanged geometry. No lifetime, interference, aliasing, or reuse model is inferred.

Every typed finalization is constructed before any backend is invoked. Finalizers then run once
in partition order, and each returned executable must be non-null and retain the exact shared
plan object. Finalization may construct immutable Java recipe state only under the current
contract. It does not allocate physical resources, acquire a closeable prepared resource, create
a `RunState`, bind or execute an invocation, or create a schedule.

The complete-set operation and its batch result remain package-private. There is no public
Prepare orchestration facade, production backend finalizer, or end-to-end consumer yet.

## Planned prepared aggregates

`PreparedExecution` will contain or reference prepared partitions, executable recipes, prepared
memory, the current schedule recipe, and any immutable persistent prepared resources. Preparation
creates it once for a selected set of explicitly registered backends; multiple runs may reuse it
concurrently without sharing mutable invocation state.

The current `PreparedPartition`, `PreparedExecutable`, `PreparedSchedule`, and `BoundInvocation`
contracts will be consumed by those later aggregates. The current schedule consumer establishes
no distinct `PreparedUnit` invariant: list position is the occurrence, and the exact executable
already supplies the work recipe and memory-plan association.

## Planned execution and publication contract

```java
// Conceptual API; not currently runnable.
RunResult result = execution.run(inputs, RunOptions.defaults());
```

- `inputs` will bind invocation values to prepared input bindings.
- `RunOptions` will hold declarative run and publication choices, not live services.
- Exactly one current `RunState` will be populated and consumed by the future runner for the
  complete heterogeneous logical run.
- `RunResult` will expose results published by the prepared publication plan and run policy.

Current ownership distinguishes borrowed inputs from run-owned internal resources. Future
publication will transfer or lease selected outputs to `RunResult`, while immutable persistent
prepared resources stay with `PreparedExecution`. Current cold checked binding creates
backend-owned typed invocation objects with direct references before execution. Exact result,
transfer, residency, non-executable schedule variants, and runner behavior remain for focused
Runtime tasks.

## Boundary and failure model

Run may fail because an input binding is missing or incompatible, a prepared resource cannot be used, transfer or execution fails, or publication fails. It must not recover by discovering another backend and lowering again. Unsupported work must be resolved during compile ownership or fail during prepare.

## Related contracts

- [Runtime, prepare, and backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Preparing execution](../user-guide/preparing-execution.md)
- [Running models](../user-guide/running-models.md)
- [Glossary](../glossary.md)
