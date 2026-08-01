# Runtime API

## Purpose and implementation status

This reference explains the implemented Runtime foundations for prepared-memory geometry,
reusable representation creation descriptions, one run's physical-representation lifecycle and
validity, checked binding of backend recipes, ordered scheduling, prepared publication, the
completed-result lease, the immutable
prepared-execution root, and the current Prepare-owned handoff that constructs geometry and
finalizes backend recipes. The current public surface contains:

- `runtime.memory`: `BufferSlot`, `WorkspaceSlot`, and `PreparedMemoryPlan` with its nested
  `BufferEntry` and `WorkspaceEntry` records;
- `runtime.resource`: the nominal `BufferRepresentation` and `WorkspaceRepresentation` lifecycle
  roles implemented by concrete backends plus `PreparedRepresentationPlan` and its nested
  preparation and creator contracts;
- `runtime.run`: `RunResourceOwnership`, `BufferRepresentationBinding`, `RunState`,
  `PreparedPublication`, `BoundPublication`, `RunResult`, and stateless
  `PreparedExecutionRunner`;
- `runtime.execution`: `PreparedExecution`, `PreparedExecutable`, its nested `BufferSelection`,
  `WorkspaceSelection`, and `BufferAccess` contracts, `BoundInvocation`,
  `PreparedBufferTransfer`, and `BoundBufferTransfer`; and
- `runtime.schedule`: `PreparedSchedule`, its sealed nested `Step` contract,
  `RepresentationCreationStep`, `ExecutionStep`, `BufferTransferStep`, and `PublicationStep`.

The geometry, reusable creation description, package-private all-or-cleaned setup, structural
residency, explicit per-buffer-copy validity, cold-bound invocation and transfer contracts,
creation, execution, transfer, and dense publication-suffix schedule recipes, the whole-state
result lease, two-component prepared-execution aggregate,
Prepare-owned resource assignments, typed backend finalization, and `PreparedPartition`
association are current. Concrete physical allocation and storage access, public result-value
access and public Prepare orchestration remain planned. No production concrete backend currently
implements the creation,
finalization, or execution contracts.

## Mental model

```text
compile facts             prepare handoff               Runtime state
ValueId / requirement -> source-to-slot assignment -> PreparedMemoryPlan
current                  current                      current
                                                              |
                         PreparedRepresentationPlan ----------+
                         current reusable creation description |
                                                              v
                         cold setup -> RunState + validity
                         current internal  current per run
     PreparedMemoryPlan + PreparedSchedule -> PreparedExecution
     current              current            current reusable root

        PreparedExecutionRunner -> create/bind all -> ordered direct traversal
        current public runner     cold setup          current synchronous run

        BufferTransferStep -> cold bind -> BoundBufferTransfer -> transfer + validity
        current recipe        current       current per run      current contract

        PublicationStep -> cold bind -> BoundPublication -> publish -> RunResult
        current recipe      current       current per run     current    current lease
```

Read the flow from logical source facts toward invocation state. Prepare exposes exact buffer and
workspace requirements through `BackendPartitionAnalysis`, then its current package-internal
handoff retains source-to-slot associations and constructs the Runtime geometry. Current
`PreparedRepresentationPlan` describes borrowed inputs and concrete-backend creators in that
geometry's encounter order. Package-private cold setup validates all caller inputs, creates
run-owned buffers and workspaces, and constructs one `RunState`. A current `PreparedExecutable`
selects those resident representations by dense position, checks backend
compatibility during cold binding, and creates a per-run `BoundInvocation`. A current
`PreparedBufferTransfer` selects two distinct already-created representations of one buffer,
checks their concrete compatibility during cold binding, and creates a per-run
`BoundBufferTransfer` that orchestrates the explicit validity transition around backend-owned
physical transfer work. The shared contracts still implement no concrete allocation or storage
access. The current schedule can retain one first-only creation prefix
followed by executable or transfer occurrences and then a dense publication-only suffix. It does
not invoke or execute any step. Current publication names an already-created valid copy and
leases the complete state to a result, but deliberately exposes no output value. The runner
composes these contracts without backend discovery or graph interpretation.

## Current prepared execution

`PreparedExecution` is the immutable reusable Runtime root for the prepared state that currently
exists. Its two components, in order, are one exact `PreparedMemoryPlan` and one exact
`PreparedSchedule`. Construction requires both references to be non-null and requires
`schedule.memoryPlan() == memoryPlan`; a structurally equal plan created separately is not the
same prepared context.

The record retains and returns both supplied references exactly. It is safe for concurrent
readers because its current components are immutable, but every later active invocation must use
its own mutable `RunState`. The aggregate does not implement `AutoCloseable`, acquire or own a
resource, create run state, consume the schedule, bind an invocation, execute work, allocate a
representation, or publish a result. Construction and component access are constant-time.

```java
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.List;

PreparedSchedule schedule = new PreparedSchedule(plan, List.of());
PreparedExecution execution = new PreparedExecution(plan, schedule);
```

Here `execution.memoryPlan()` is exactly `plan`, and `execution.schedule()` is exactly
`schedule`. The example creates reusable recipe state only; no run begins and no resource changes
ownership. A null component reports `memoryPlan` or `schedule` in component order. A schedule for
a different plan reference fails with
`IllegalArgumentException("schedule memory plan does not match prepared execution memory plan")`.

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
implements physical allocation, storage access, transfer, and cleanup mechanics; the shared
Runtime API never exposes concrete storage, backend, or device access.

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

Every bound representation is structurally resident until the state closes: the exact physical
object exists and remains bound to that run. Each buffer representation also has one independent
validity bit. A borrowed buffer starts valid because it is a caller input containing the logical
slot value; a run-owned buffer starts invalid because fresh storage does not yet contain that
value. Zero, one, or multiple copies may be valid. Workspaces are resident run-owned scratch and
have no logical validity. The same representation object cannot occur twice anywhere in one
state, including across buffer and workspace domains.

`isBufferRepresentationValid(bufferIndex, representationIndex)` reads one bit in constant time.
`setBufferRepresentationValid(bufferIndex, representationIndex, valid)` writes exactly one bit in
constant time. The setter performs no storage access, copy, backend call, ownership transition,
implicit invalidation, or coherence. A later transfer or execution runner must change validity
explicitly only after its physical action succeeds.

## Focused creation and validity example

### Goal and inputs

Describe one borrowed caller buffer, one backend-created internal buffer, and one backend-created
workspace for the single buffer and workspace positions from the preceding plan. Then construct
the public per-run state, observe its initial validity, mark the internal copy valid explicitly,
and close the state. The local representation classes make cleanup observable without claiming a
production backend.

```java
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

final class CreationBuffer implements BufferRepresentation {
    private final AtomicInteger closes;
    CreationBuffer(AtomicInteger closes) { this.closes = closes; }
    @Override public void close() { closes.incrementAndGet(); }
}

final class CreationWorkspace implements WorkspaceRepresentation {
    private final AtomicInteger closes;
    CreationWorkspace(AtomicInteger closes) { this.closes = closes; }
    @Override public void close() { closes.incrementAndGet(); }
}

AtomicInteger borrowedCloses = new AtomicInteger();
AtomicInteger ownedCloses = new AtomicInteger();
AtomicInteger workspaceCloses = new AtomicInteger();

BufferRepresentation borrowed = new CreationBuffer(borrowedCloses);
PreparedRepresentationPlan.BufferCreator bufferCreator =
        () -> new CreationBuffer(ownedCloses);
PreparedRepresentationPlan.WorkspaceCreator workspaceCreator =
        () -> new CreationWorkspace(workspaceCloses);

PreparedRepresentationPlan representationPlan =
        new PreparedRepresentationPlan(
                plan,
                List.of(
                        List.of(
                                new PreparedRepresentationPlan.CallerInput(),
                                new PreparedRepresentationPlan.CreatedBuffer(bufferCreator))),
                List.of(workspaceCreator));

BufferRepresentation owned = bufferCreator.create();
WorkspaceRepresentation workspace = workspaceCreator.create();

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

boolean borrowedInitiallyValid = state.isBufferRepresentationValid(0, 0);
boolean ownedInitiallyValid = state.isBufferRepresentationValid(0, 1);
state.setBufferRepresentationValid(0, 1, true);
boolean ownedAfterProduction = state.isBufferRepresentationValid(0, 1);
state.close();
state.close();
```

### Result and interpretation

`borrowedInitiallyValid` is `true`, `ownedInitiallyValid` is `false`, and
`ownedAfterProduction` is `true`. The state reports one buffer position, two structurally
resident buffer representations, and one resident workspace. Setting the internal bit performs
no physical work; the example treats a successful production action as already completed before
that line. After both `close()` calls, `borrowedCloses.get()` is `0`, while
`ownedCloses.get()` and `workspaceCloses.get()` are each `1`.

The package-private cold setup used by `PreparedExecutionRunner` invokes the same immutable creator
references and construct this state with rollback. Because that operation is intentionally not a
public facade, this current public example stages the successful callback results directly before
calling the public constructor. It demonstrates prepared origins, initial validity, explicit
mutation, workspace exclusion, and ownership-sensitive idempotent cleanup. It does not implement
a runner, physical copy, kernel, or automatic validity transition.

## Current prepared representation creation

`PreparedRepresentationPlan` retains the exact `PreparedMemoryPlan`, immutable snapshots of both
buffer-list levels, and an immutable workspace-creator list. Each buffer preparation is either a
zero-component `CallerInput` occurrence or `CreatedBuffer(BufferCreator)`. Dense caller-input
encounter order is buffer position first and representation position second. Each workspace
position has one `WorkspaceCreator`. Callback implementations must be immutable and thread-safe,
and each successful call must return a fresh non-null representation for that run.

The package-private `RunStateCreation` operation validates the complete caller count, non-null
elements, and caller identity uniqueness before invoking any callback. It then creates buffers in
dense buffer/representation order and workspaces in workspace order. A successful result becomes
run-owned only when complete `RunState` construction succeeds. If a creator, result validation,
or state construction reports a `RuntimeException` or `Error`, setup closes every successfully
created result once in reverse creation order, preserves the original failure unchanged, and adds
cleanup failures as suppressed exceptions in cleanup encounter order. It never closes borrowed
inputs or closes a duplicate callback result as a second owned object.

Plan construction invokes no callback. A null or duplicate callback result is rejected during
cold setup. No creation callback performs transfer, materialization, execution, publication, or
validity mutation as part of the shared contract.

## Run-state lifecycle and failures

`RunState.close()` marks the state closed before physical cleanup. It closes workspaces from last
position to first, then run-owned buffer representations from last buffer position to first and
last representation to first. Borrowed buffers are skipped. Every owned representation is
attempted once. The first `RuntimeException` or `Error` is rethrown after all attempts, with later
failures attached in cleanup encounter order as suppressed exceptions. Repeated closure performs
no cleanup and does not rethrow an earlier failure.

Representation and validity access or mutation after closure fails first with
`IllegalStateException("run state is closed")`. The retained plan, slot counts, per-buffer
representation counts, and `isClosed()` remain inspectable. `RunState` is not thread-safe;
callers must not race access, validity mutation, and closure on one instance. Concurrent runs may
share the immutable plan but have independent validity arrays and require distinct run-owned
representations. Borrowed representations may be shared only when the caller guarantees their
lifetime, safe access, and external synchronization.

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

## Current prepared buffer transfer

`PreparedBufferTransfer` is an immutable reusable recipe for copying one logical buffer value
between two distinct, already-created representation positions of the same prepared buffer.
Materialization is this same explicit transfer when it produces the equivalent value in an
already-created representation required by later work. It is not another recipe, allocation
operation, route search, or coherence policy.

`bind(runState)` is the cold boundary. It requires the exact open state, validates both positions,
and lets the concrete backend check source compatibility before destination compatibility. The
backend then constructs a `BoundBufferTransfer` subclass with direct typed source and destination
fields. Binding performs no transfer and changes no ownership or validity.

`BoundBufferTransfer.execute()` first makes an already-valid destination a no-op, even if the
selected source is invalid. Otherwise it requires a valid source, calls the backend hook exactly
once, and marks only the destination valid after successful return. A backend
`RuntimeException` or `Error` propagates unchanged and leaves every Runtime validity bit
unchanged. Physical destination contents may be partial after failure, but Runtime still
classifies that copy as invalid.

### Focused direct-reference transfer example

#### Goal and inputs

This current extension pattern binds two concrete buffers directly. It demonstrates success, the
resulting destination-valid no-op, and backend failure. A later runner will bind schedule
occurrences conceptually; this example calls the current recipe and action directly because no
runner exists yet.

```java
import io.github.pho001.synaptik.runtime.execution.BoundBufferTransfer;
import io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;

final class TransferBuffer implements BufferRepresentation {
    int reads;
    int writes;
    @Override public void close() {}
}

final class TransferAction extends BoundBufferTransfer {
    private final TransferBuffer source;
    private final TransferBuffer destination;
    private final boolean fail;

    TransferAction(
            RunState state,
            TransferBuffer source,
            TransferBuffer destination,
            boolean fail) {
        super(state, 0, 0, 1);
        this.source = source;
        this.destination = destination;
        this.fail = fail;
    }

    @Override protected void executeTransfer() {
        source.reads++;
        destination.writes++;
        if (fail) {
            throw new IllegalStateException("physical copy failed");
        }
    }
}

final class TransferRecipe extends PreparedBufferTransfer {
    private final boolean fail;

    TransferRecipe(PreparedMemoryPlan plan, boolean fail) {
        super(plan, 0, 0, 1);
        this.fail = fail;
    }

    @Override protected boolean acceptsSourceBufferRepresentation(
            BufferRepresentation representation) {
        return representation instanceof TransferBuffer;
    }

    @Override protected boolean acceptsDestinationBufferRepresentation(
            BufferRepresentation representation) {
        return representation instanceof TransferBuffer;
    }

    @Override protected BoundBufferTransfer bindCompatible(
            RunState state,
            BufferRepresentation source,
            BufferRepresentation destination) {
        return new TransferAction(
                state, (TransferBuffer) source, (TransferBuffer) destination, fail);
    }
}
```

Given the earlier one-buffer/one-workspace `plan`, create two isolated states. Each starts with a
valid borrowed source and invalid run-owned destination and receives its own workspace because
the plan declares one workspace position.

```java
TransferBuffer source = new TransferBuffer();
TransferBuffer destination = new TransferBuffer();
RunState transferState =
        new RunState(
                plan,
                List.of(
                        List.of(
                                new BufferRepresentationBinding(
                                        source, RunResourceOwnership.BORROWED),
                                new BufferRepresentationBinding(
                                        destination, RunResourceOwnership.RUN_OWNED))),
                List.of(new ExampleWorkspace()));

BoundBufferTransfer transfer = new TransferRecipe(plan, false).bind(transferState);
transfer.execute();
transfer.execute();

TransferBuffer failingSource = new TransferBuffer();
TransferBuffer failingDestination = new TransferBuffer();
RunState failingState =
        new RunState(
                plan,
                List.of(
                        List.of(
                                new BufferRepresentationBinding(
                                        failingSource, RunResourceOwnership.BORROWED),
                                new BufferRepresentationBinding(
                                        failingDestination, RunResourceOwnership.RUN_OWNED))),
                List.of(new ExampleWorkspace()));
BoundBufferTransfer failing = new TransferRecipe(plan, true).bind(failingState);
try {
    failing.execute();
} catch (IllegalStateException expected) {
    // The exact backend failure propagates; Runtime leaves destination validity false.
}
```

#### Result and interpretation

After the two successful calls, `source.reads` and `destination.writes` are both `1` and both
copies in `transferState` are valid. The second call observed the valid destination and skipped
the backend hook. In `failingState`, the source remains valid and the destination remains invalid;
both physical counters are `1` because one backend attempt occurred. The action uses only its two
direct fields for physical work. This proves the current transfer and validity contract, not
byte-copy correctness, schedule traversal, executable-output invalidation, publication, result
ownership, or a public run lifecycle.

## Current prepared schedule

`PreparedSchedule` is an immutable reusable recipe that associates an ordered list of work
occurrences with one exact `PreparedMemoryPlan`. Its sealed nested `Step` interface exposes only
that plan association. `RepresentationCreationStep(PreparedRepresentationPlan)` retains the sole
optional cold-setup prefix, `ExecutionStep(PreparedExecutable)` retains one executable occurrence,
`BufferTransferStep(PreparedBufferTransfer)` retains one explicit transfer/materialization
occurrence, and `PublicationStep(PreparedPublication)` retains one final ordered result
occurrence. Each derives `memoryPlan()` from its exact component.

Construction validates each occurrence in supplied order and requires reference identity with
the schedule plan; a structurally equal but separately constructed plan is not the same prepared
association. Only after the complete scan succeeds does the schedule use `List.copyOf` to retain
an immutable ordered snapshot. It keeps exact step and recipe references. Empty schedules,
repeated executable or transfer step references, and repeated recipe references are valid because
each list position is one explicit occurrence and does not create another ownership relationship. A
creation step may occur zero or one time and, when present, must be index zero. A later Prepare
validator may require it for a runnable result; compatibility schedules may still be empty or
executable-only. Publication steps, when present, form the final suffix. Their result indices must
be `0..N-1` in encounter order, and `publicationCount()` reports `N`. Distinct result positions
may intentionally name the same buffer and representation coordinate.

### Focused schedule example

#### Goal and inputs

Retain the representation plan from the creation example as the first occurrence, followed by one
transfer occurrence and two occurrences of the current `ExampleExecutable` recipe from the cold-
binding example. All occurrences use the same exact plan; constructing the schedule invokes no
callback and needs no run state or physical representation.

```java
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.List;

ExampleExecutable executable = new ExampleExecutable(plan);
PreparedSchedule.ExecutionStep occurrence =
        new PreparedSchedule.ExecutionStep(executable);
PreparedSchedule.RepresentationCreationStep creation =
        new PreparedSchedule.RepresentationCreationStep(representationPlan);
PreparedSchedule.BufferTransferStep transferOccurrence =
        new PreparedSchedule.BufferTransferStep(new TransferRecipe(plan, false));
ArrayList<PreparedSchedule.Step> supplied =
        new ArrayList<>(List.of(creation, transferOccurrence, occurrence, occurrence));

PreparedSchedule schedule = new PreparedSchedule(plan, supplied);
supplied.clear();
```

#### Result and interpretation

`schedule.memoryPlan()` is the exact `plan` object. `schedule.steps().getFirst()` is the exact
`creation` reference, `schedule.steps().get(1)` is the exact `transferOccurrence`, and the list
still contains the same `occurrence` reference twice after the mutable source list is cleared. The
returned list is immutable. This proves creation-plan reachability, deterministic recipe order,
and snapshot isolation only. Construction does not invoke a creator, bind or execute either
recipe, allocate or close a resource, create a `RunState`, perform a transfer, or select
publication behavior.

A null executable fails with `NullPointerException("executable")`. A null schedule plan, list,
or element reports `memoryPlan`, `steps`, or `steps[index]`, respectively. The first step whose
plan is not the exact schedule plan fails with
`IllegalArgumentException("steps[index] memory plan does not match schedule memory plan")`.
A creation step after index zero fails with
`IllegalArgumentException("steps[index] representation creation must be the first schedule occurrence")`.
Current transfer and materialization use `BufferTransferStep`; no second materialization variant
is hidden in the schedule. A publication step with a reordered or gapped result index fails, as
does any non-publication occurrence after publication begins. Schedule construction itself does
not bind, publish, inspect validity, or transfer ownership.

## Current prepared publication and result lease

`PreparedPublication` is an immutable, reusable recipe for one ordered result position. It uses
only dense Runtime coordinates: a buffer position in one exact `PreparedMemoryPlan`, a
representation position within that buffer in a matching `RunState`, and a result position. These
coordinates are not compiler `TensorId` or `ValueId` identities. Prepare will later translate
compiler logical publication roles into these Runtime coordinates.

`bind(runState)` is the sole representation lookup. It requires an open state associated with the
same exact plan reference, validates the representation position, and returns a new
`BoundPublication` retaining the selected physical representation directly. Binding changes no
validity or ownership and invokes no backend work.

`BoundPublication.publish()` requires the state to remain open and the selected copy to be valid
at that exact moment. It then changes only the bound occurrence's one-shot flag. A repeated
publication, a closed state, or an invalid selected copy fails. Publication never searches for a
different valid copy, performs an implicit transfer or materialization, calls a backend, or
changes `RunState` validity. When another representation is needed, an explicit prepared buffer
transfer must occur before the publication suffix.

`RunResult` accepts a complete dense result-ordered list of successfully published occurrences
for one exact open state. It privately snapshots their direct representation references,
including intentional aliases, and semantically takes responsibility for closing the complete
`RunState`. It exposes only `resultCount()`, `isClosed()`, and idempotent `close()`; it exposes no
representation, storage, Tensor, value, or state accessor. An empty list is valid. Constructor
failure and partial publication transfer no cleanup responsibility, so the runner remains
responsible for closing the state. Borrowed inputs remain caller-owned throughout the result
lease, while state-owned resources retain the existing deterministic cleanup behavior.

The bound publication and result are not thread-safe. They must not race publication, validity
mutation, execution, transfer, result construction, or closure. Immutable prepared publication
recipes may bind concurrently to distinct states, producing isolated flags and leases.

### Focused publication example

#### Goal and inputs

Publish two ordered results that intentionally alias one already-valid borrowed representation,
then close the complete state through the result. This current example calls the publication
objects directly to isolate publication and result leasing from runner traversal.

```java
import io.github.pho001.synaptik.runtime.run.BoundPublication;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.run.RunResult;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

BufferRepresentation published = new CreationBuffer(new AtomicInteger());
RunState publicationState =
        new RunState(
                plan,
                List.of(
                        List.of(
                                new BufferRepresentationBinding(
                                        published, RunResourceOwnership.BORROWED))),
                List.of(new CreationWorkspace(new AtomicInteger())));

PreparedPublication first = new PreparedPublication(plan, 0, 0, 0);
PreparedPublication alias = new PreparedPublication(plan, 0, 0, 1);
PreparedSchedule schedule =
        new PreparedSchedule(
                plan,
                List.of(
                        new PreparedSchedule.PublicationStep(first),
                        new PreparedSchedule.PublicationStep(alias)));

BoundPublication boundFirst = first.bind(publicationState);
BoundPublication boundAlias = alias.bind(publicationState);
boundFirst.publish();
boundAlias.publish();

try (RunResult result = new RunResult(
        publicationState, List.of(boundFirst, boundAlias))) {
    assert schedule.publicationCount() == 2;
    assert result.resultCount() == 2;
}
```

#### Result and interpretation

Both result positions retain the same exact representation privately, but the result count is
two because aliases preserve ordered result multiplicity. Publication reads no storage and
performs no copy. Closing the result closes the leased state; because this example's selected
representation is borrowed, its physical cleanup remains the caller's responsibility.

If the selected copy were invalid, `publish()` would fail with
`IllegalStateException("published buffer representation is invalid")`. It would not choose
another copy or create a partial result. The owner of the still-open state would then close it.

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

## Current aggregate and run orchestration

The current `PreparedExecution` contains only the exact memory plan and exact same-plan schedule.
Executable recipes are already reachable through schedule occurrences, while `PreparedPartition`
remains a Prepare-owned association and does not cross into this Runtime aggregate. There is no
distinct `PreparedUnit`: list position is the occurrence, and the exact executable supplies the
work recipe and memory-plan association.

Public Prepare orchestration will later construct and validate this aggregate. A future need for
immutable persistent prepared resources must define its own ownership and partial-construction
failure lifecycle; the current record does not anticipate it with an empty close contract.

## Current prepared runner

```java
PreparedExecutionRunner runner = new PreparedExecutionRunner();
RunResult result = runner.run(execution, callerInputs);
```

- `callerInputs` supplies dense borrowed representations in creation-plan encounter order.
- Exactly one isolated `RunState` is created and consumed for the complete heterogeneous run.
- Every executable, transfer, and publication occurrence cold-binds before the first action.
- Traversal uses direct bound references and precomputed primitive executable coordinates.
- The current `RunResult` leases the whole run state but exposes no values; a later public
  Engine-facing result API must define value access separately.

Current ownership distinguishes borrowed inputs from run-owned internal resources, and current
per-copy validity is explicit within `RunState`. Current publication leases the complete state to
`RunResult`, while immutable persistent prepared resources stay with `PreparedExecution`.
Current cold checked binding creates backend-owned typed invocation and
transfer objects with direct references. One exact prepared buffer transfer and its success-only
destination-valid transition are current. Prepared publication, its suffix ordering, one-shot
validity check, alias preservation, empty result, and whole-state lease are also current. Transfer
route selection and public output access remain later work. The runner validates declared reads,
invalidates every copy of each declared output buffer before backend work, validates exact writes
only after success, and closes the state after any post-creation failure while preserving the
original unchecked failure. It performs no Trace emission because no current run payload exists.

## Boundary and failure model

Run may fail because an input binding is missing or incompatible, a prepared resource cannot be used, transfer or execution fails, or publication fails. It must not recover by discovering another backend and lowering again. Unsupported work must be resolved during compile ownership or fail during prepare.

## Related contracts

- [Runtime, prepare, and backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Preparing execution](../user-guide/preparing-execution.md)
- [Running models](../user-guide/running-models.md)
- [Glossary](../glossary.md)
