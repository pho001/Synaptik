# Slots and prepared-memory geometry

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

A **[buffer slot](../../glossary.md#buffer-slot--bufferslot)** names a planned place for a logical value, while a **[workspace slot](../../glossary.md#workspace-slot--workspaceslot)** names
scratch storage used while computing. Neither slot is a physical memory address. **Byte geometry**
specifies storage size and [alignment](../../glossary.md#term-alignment); a dense index is merely an entry's zero-based list position,
not the [opaque](../../glossary.md#term-opaque) number stored inside the slot identifier.

<a id="current-slot-identities"></a>
## Current slot [identities](../../glossary.md#term-identity-meanings)

[`BufferSlot`](../../glossary.md#buffer-slot--bufferslot) and [`WorkspaceSlot`](../../glossary.md#workspace-slot--workspaceslot) are public, deeply [immutable](../../glossary.md#term-immutable) records with one `long value`
component. Each accepts every value from zero through `Long.MAX_VALUE`; no sentinel is reserved.
A negative value fails with `IllegalArgumentException` and message
`value must be non-negative`.

Each component is [opaque](../../glossary.md#term-opaque) outside its owning plan context. Neither record stores an [owner](../../glossary.md#term-owner)
reference, so ordinary record equality and hashing compare only the numeric component within the
same [nominal](../../glossary.md#term-nominal) record type. A [buffer slot](../../glossary.md#buffer-slot--bufferslot) and [workspace slot](../../glossary.md#workspace-slot--workspaceslot) with the same number are distinct, and
two plans may reuse a number without referring to the same conceptual slot. Diagnostic record
text is not a [serialization](../../glossary.md#term-serialization) format.

Creating a slot does not allocate, acquire, retain, [release](../../glossary.md#term-release), or identify physical storage. A
buffer slot is not a [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot); a workspace slot is not an analysis-local workspace requirement
ID. Neither is an address, storage handle, allocation, [device](../../glossary.md#term-device), [residency](../../glossary.md#residency) fact, or resource.

<a id="current-prepared-memory-plan"></a>
## Current prepared-memory plan

[`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan) is the [immutable](../../glossary.md#term-immutable) final byte geometry for one prepared plan. Its [`buffers`](../../glossary.md#term-buffer)
and `workspaces` components are ordered immutable snapshots. Each entry retains the exact slot
reference and records:

- `byteSize`, an exact non-negative byte count; zero is valid; and
- `byteAlignment`, an exact positive power of two from `1` through `1L << 62`.

Buffer slots must be unique within `buffers`, and workspace slots must be unique within
`workspaces`. The two uniqueness domains are separate. Either list may be empty, including both
lists together.

The plan retains neither the caller's list containers nor any [Prepare](../../glossary.md#prepare) requirement or
source-to-slot association. It does not sort, renumber, merge, derive, allocate, bind, own, or
[release](../../glossary.md#term-release) storage.

<a id="focused-valid-plan-example"></a>
## Focused valid-plan example

<a id="valid-plan-goal-and-inputs"></a>
### Valid-plan goal and inputs

Describe one 24-byte reusable [buffer](../../glossary.md#term-buffer) position and one 64-byte workspace position. The supplied
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

`plan.buffers().getFirst()` retains the exact `input` reference with 24-byte size and 4-[byte
alignment](../../glossary.md#term-byte-alignment). `plan.workspaces().getFirst()` retains the exact `scratch` reference with 64-byte size
and 16-byte alignment. Both lists are [immutable](../../glossary.md#term-immutable) and preserve their supplied order.

This result describes geometry only. It does not say which [graph value](../../glossary.md#graph-value) produced the [buffer](../../glossary.md#term-buffer)
requirement, which [backend](../../glossary.md#backend) analysis requested the workspace, whether the two positions [alias](../../glossary.md#aliasing-and-arena)
physical storage, or how a [run](../../glossary.md#term-run) accesses either position.


---

[Previous topic](prepared-execution.md) · [Next topic](representations.md)
