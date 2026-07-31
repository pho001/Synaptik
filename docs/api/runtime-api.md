# Runtime API

## Purpose and implementation status

This reference explains the implemented Runtime foundations for prepared-memory geometry and one
run's physical-representation lifecycle. The current public surface contains:

- `runtime.memory`: `BufferSlot`, `WorkspaceSlot`, and `PreparedMemoryPlan` with its nested
  `BufferEntry` and `WorkspaceEntry` records;
- `runtime.resource`: the nominal `BufferRepresentation` and `WorkspaceRepresentation` lifecycle
  roles implemented by concrete backends; and
- `runtime.run`: `RunResourceOwnership`, `BufferRepresentationBinding`, and `RunState`.

The geometry carrier and per-run representation carrier are current. Prepare-owned slot
assignment, physical allocation and storage access, compatibility-aware cold binding, full
validity/residency, transfers, schedules, prepared executables, publication/results, and a
runnable Runtime facade remain planned.

## Mental model

```text
compile facts             prepare handoff               Runtime geometry        current run carrier
ValueId / requirement -> source-to-slot assignment -> prepared slot order -> RunState arrays
current                  planned                      current               current
                                                                                |
                                                                                v
                                                     allocation / cold binding / execute
                                                                  planned
```

Read the flow from logical source facts toward invocation state. Prepare currently exposes exact
buffer and workspace requirements through `BackendPartitionAnalysis`. A later Prepare contract
will retain source-to-slot associations and construct the Runtime geometry. Current `RunState`
then accepts already-created representations in that geometry's encounter order. It neither
derives the source mapping nor allocates, inspects, transfers, or executes physical storage.

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

## Planned prepared contracts

Prepare will translate ordered analysis requirements into stable slots, retain exact
requirement-to-slot associations for backend finalization, and construct `PreparedMemoryPlan`.
The initial planned assignment is conservative: one distinct buffer slot per distinct declared
buffer value and one distinct workspace slot per workspace declaration. Reuse requires a later
proved lifetime/interference model.

`PreparedExecution` will contain or reference prepared partitions, executable units, prepared
memory, a schedule, and any immutable persistent prepared resources. Preparation creates it once
for a selected set of explicitly registered backends; multiple runs may reuse it concurrently
without sharing mutable invocation state.

`PreparedExecutable` will compute only its prepared region. Its hot-path contract will not receive `Operation` or `CompiledNode`, and runtime will not ask it to rediscover a backend or select a kernel.

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
prepared resources stay with `PreparedExecution`. A future cold checked binding phase will create
backend-owned typed invocation objects with direct references before execution. Exact result,
transfer, residency, and runner behavior remain for focused Runtime tasks.

## Boundary and failure model

Run may fail because an input binding is missing or incompatible, a prepared resource cannot be used, transfer or execution fails, or publication fails. It must not recover by discovering another backend and lowering again. Unsupported work must be resolved during compile ownership or fail during prepare.

## Related contracts

- [Runtime, prepare, and backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Preparing execution](../user-guide/preparing-execution.md)
- [Running models](../user-guide/running-models.md)
- [Glossary](../glossary.md)
