# Mutable borrowed-storage association

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**Reader model.** A storage **association** is a Tensor's current reference to a borrowed memory
wrapper. Replacing that reference is not copying values, changing the descriptor, or freeing the
previous memory. Multiple tensors can refer to the same wrapper, so changing one association does
not change the others. **Atomic** replacement means another association access sees a complete
old or new reference; it does not make all reads and writes to the underlying bytes atomic or
thread-safe. **Liveness** asks whether the memory scope is still open at the instant checked.

The matching `MemorySegment.ofArray(...)` overload creates a writable heap segment with an
automatic scope. That scope keeps the primitive array reachable and permits access from any
thread. The attached `MemorySegmentStorage` remains a borrowed, non-closing wrapper; no arena,
external owner, close operation, or deterministic reclamation is introduced. The object chain is
garbage-collected when the tensor, storage, segment, and array are no longer reachable.

Each successful factory construction receives a non-negative `TensorId`. The value is unique among
all allocations made by that factory in the current JVM, including concurrent calls. Callers treat
the numeric value as opaque. Completion order, adjacency, gaplessness, cross-process uniqueness,
persistence, and uniqueness relative to manually constructed `TensorId` values are not promised.

Null factory arguments are rejected before allocation and consume no identifier.
Label normalization and storage compatibility remain `Tensor` constructor responsibilities.
A blank label or invalid storage therefore consumes an identifier before construction fails.
Consumed identifiers are never reused.
The final candidate is `Long.MAX_VALUE`; once it is claimed, all later factory allocations fail
permanently with `IllegalStateException` and message `tensor identifier space exhausted`.

Only the host-storage reference can change. `replaceHostStorage(storage)` validates the proposed
storage before atomically replacing that reference and returns the exact previous reference, or an
empty optional when there was none. `clearHostStorage()` atomically removes and returns the exact
previous reference. `hostStorage()`, replacement, and clearing are synchronized with respect to
one another, so each returned optional is a snapshot of reference state. The synchronization does
not make raw memory access thread-safe, keep an arena alive, or make a confined segment accessible
from another thread.

Storage compatibility has three checks in a fixed order:

1. The storage data type must exactly match the descriptor data type.
2. When the descriptor has a resolved layout, capacity must be at least the layout's complete
   referenced element span. This includes offset, strided, broadcast, scalar, and zero-sized
   geometry. A static or dynamic unresolved layout skips this capacity check because no physical
   geometry is known.
3. The storage must be alive at attachment time.

Read-only storage is valid because `Tensor` performs no memory writes. Storage remains borrowed:
the tensor itself never allocates, copies, accesses, or closes it. For caller-supplied arena-backed
storage, the caller may close the owning scope after attachment; the tensor then continues to
return the exact associated storage, whose own `isAlive()` reports the point-in-time state. For
factory-created heap storage, the automatic segment scope remains alive and keeps the backing
array reachable without an external owner. One storage object may be associated with multiple
tensors; changing one tensor's association does not change another's.

`Tensor` inherits ordinary object equality and hashing. Equal `TensorId` values do not make two
tensor objects equal; the factory's ID guarantee does not change object equality. Its text form is
stable metadata-only diagnostic output containing the ID, descriptor, and normalized label. It omits
storage presence and identity, memory addresses and contents, liveness, graph state, and runtime
state, and it is not a serialization format.

Provenance is fixed at construction and needs no synchronization. `TensorProvenance` retains one exact `TensorProducer` and a checked zero-based output index. Its operation, immutable ordered input snapshot and selected descriptor are derived from that producer. Inputs retain exact Tensor identities, including repeats and zero-input occurrences, without retaining the caller's mutable list. Record equality compares producer object identity and output index: equal operation values and inputs do not merge separately constructed occurrences. This is not graph validation or automatic common-subexpression elimination. See [the source contract](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProvenance.java).

Every public `TensorFactory` and `TensorRandoms` creation method returns a provenance-free leaf. The
package-private derived-construction seam used by the implemented binary arithmetic, comparison,
boolean logical, conditional-selection, cast, unary, and scalar expression helpers attaches one
already-validated provenance value, allocates exactly one ID through the existing allocator, and
creates no storage. The
factory seam itself does not inspect the operation or inputs, infer a descriptor, traverse an
expression, capture a graph, or perform semantic validation.


---

[Previous topic](import.md) · [Next topic](arithmetic.md)
