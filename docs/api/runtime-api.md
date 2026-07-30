# Runtime API

## Purpose and implementation status

This reference explains the current Runtime-owned slot identities and final prepared-memory
geometry. The implemented `io.github.pho001.synaptik.runtime.memory` surface contains
`BufferSlot`, `WorkspaceSlot`, and `PreparedMemoryPlan` with its nested `BufferEntry` and
`WorkspaceEntry` records.

The final geometry carrier is current, but the Prepare-owned assignment that constructs it from
backend analyses is not. Physical allocation, per-run binding and access, schedules, prepared
executables, and runnable Runtime APIs also remain planned.

## Mental model

```text
compile facts             prepare handoff                  Runtime geometry           run
ValueId / requirement -> source-to-slot assignment -> BufferSlot / WorkspaceSlot -> bound storage
current                  planned                         current                    planned
```

Read the flow from logical source facts toward invocation state. Prepare currently exposes exact
buffer and workspace requirements through `BackendPartitionAnalysis`. A later Prepare contract
will retain the source-to-slot associations and construct the current Runtime geometry. The
Runtime records neither derive that mapping nor bind storage.

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
memory, and a schedule. Preparation creates it once for a selected set of explicitly registered
backends; multiple runs may reuse it.

`PreparedExecutable` will compute only its prepared region. Its hot-path contract will not receive `Operation` or `CompiledNode`, and runtime will not ask it to rediscover a backend or select a kernel.

## Planned run contract

```java
// Conceptual API; not currently runnable.
RunResult result = execution.run(inputs, RunOptions.defaults());
```

- `inputs` will bind invocation values to prepared input bindings.
- `RunOptions` will hold declarative run and publication choices, not live services.
- `RunState` will own per-run mutable slots, resources, and residency facts.
- `RunResult` will expose results published by the prepared publication plan and run policy.

Exact collection types, nullability, concurrency guarantees, ownership of returned values, exception types, and resource-lifetime methods remain open until focused runtime tasks define and test them.

## Boundary and failure model

Run may fail because an input binding is missing or incompatible, a prepared resource cannot be used, transfer or execution fails, or publication fails. It must not recover by discovering another backend and lowering again. Unsupported work must be resolved during compile ownership or fail during prepare.

## Related contracts

- [Runtime, prepare, and backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Preparing execution](../user-guide/preparing-execution.md)
- [Running models](../user-guide/running-models.md)
- [Glossary](../glossary.md)
