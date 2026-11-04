# Slots and prepared-memory geometry

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

<a id="current-slot-identities"></a>
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

<a id="current-prepared-memory-plan"></a>
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

<a id="focused-valid-plan-example"></a>
## Focused valid-plan example

<a id="valid-plan-goal-and-inputs"></a>
### Valid-plan goal and inputs

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

<a id="valid-plan-result-and-interpretation"></a>
### Valid-plan result and interpretation

`plan.buffers().getFirst()` retains the exact `input` reference with 24-byte size and 4-byte
alignment. `plan.workspaces().getFirst()` retains the exact `scratch` reference with 64-byte size
and 16-byte alignment. Both lists are immutable and preserve their supplied order.

This result describes geometry only. It does not say which graph value produced the buffer
requirement, which backend analysis requested the workspace, whether the two positions alias
physical storage, or how a run accesses either position.


---

[Previous topic](prepared-execution.md) · [Next topic](representations.md)
