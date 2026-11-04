# [Host](../../glossary.md#term-host)-visible storage and lifetimes

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**[Reader model](../../glossary.md#term-reader-model).** **Host** means memory visible to the Java-side program, as opposed to memory
managed only by a [device](../../glossary.md#term-device) route. A storage wrapper describes an existing region; it does not own
the region's [lifetime](../../glossary.md#term-lifetime). **[Borrowed](../../glossary.md#ownership-borrowing-and-lease)** means the caller must keep that region usable for as long as
it is needed. An **[arena](../../glossary.md#term-arena)** is a Java memory-lifetime [owner](../../glossary.md#term-owner); closing its [scope](../../glossary.md#term-scope) can invalidate a
segment even when a [Tensor](../../glossary.md#tensor) still refers to the wrapper. **Capacity** counts physically available
[elements](../../glossary.md#term-element), not necessarily the number of logical tensor positions. Repeated [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) and gaps
can make those counts differ; [layouts](layouts.md)<span data-glossary-navigation="true"> ([layouts glossary](../../glossary.md#layout))</span> explains that mapping.

<a id="host-visible-storage"></a>
## [Host](../../glossary.md#term-host)-visible storage

The public storage contracts live in `io.github.pho001.synaptik.model.storage`.
`HostTensorStorage` is a [sealed](../../glossary.md#term-sealed) raw-storage boundary with exactly one permitted implementation,
the final `MemorySegmentStorage` class. The wrapper exposes six facts:

| Fact | Meaning |
|---|---|
| `dataType()` | The exact non-null logical [element](../../glossary.md#term-element) type supplied at construction. |
| `elementCapacity()` | A non-negative physical capacity in complete elements, not a [tensor](../../glossary.md#tensor)'s logical element count. |
| `byteSize()` | The checked product of capacity and `DataType.byteWidth()`, in bytes. |
| `segment()` | The exact supplied [`MemorySegment`](../../glossary.md#term-memorysegment) reference, including after its [scope](../../glossary.md#term-scope) closes. |
| `isReadOnly()` | The supplied segment's [JDK](../../glossary.md#term-java-development-kit-jdk) read-only state. |
| `isAlive()` | A point-in-time [snapshot](../../glossary.md#term-snapshot) of the supplied segment scope's [liveness](../../glossary.md#term-liveness). |

Construction requires exact equality between `segment.byteSize()` and the checked byte-size
product. Zero capacity therefore requires a live zero-byte segment. Extra bytes are not treated as
implicit capacity; a caller that wants a subrange supplies an explicit exact-size segment [slice](../../glossary.md#slice).
[Heap](../../glossary.md#term-heap), native, mapped, global, confined, shared, read-only, writable, and sliced segments are all
accepted when they meet the same size and initial-liveness rules.

The wrapper is [borrowed](../../glossary.md#ownership-borrowing-and-lease) and non-owning. It retains the exact segment but does not allocate memory,
own or close an [arena](../../glossary.md#term-arena), extend a scope's [lifetime](../../glossary.md#term-lifetime), or implement `AutoCloseable`. For an arena-backed
segment, the caller must keep scoped memory alive and obey the JDK's thread-access rules. A
factory-created primitive-array segment instead has an automatic scope that stays alive, keeps its
heap base reachable, and is accessible from any thread. `isAlive()` can become stale immediately
for caller-controlled scopes, and a false result does not hide the segment: `segment()` still
returns the same dead reference, so JDK memory access reports the closed-scope failure.

Writable raw memory can be changed through JDK APIs, but the wrapper supplies no typed element
access, conversion, copying, [synchronization](../../glossary.md#term-synchronization), or mutation-version tracking. It chooses no
`ValueLayout`, [byte order](../../glossary.md#text-encoding-and-file-boundaries), or [alignment](../../glossary.md#term-alignment). An exact unaligned slice can therefore be wrapped, but
that acceptance is not a promise that a future typed or [backend](../../glossary.md#backend) path can access it without an
unaligned [layout](../../glossary.md#layout) or [materialization](../../glossary.md#materialization).

`MemorySegmentStorage` uses ordinary object [identity](../../glossary.md#term-reference-identity). Two wrappers around the same segment and
[metadata](../../glossary.md#term-metadata) are still unequal. The wrapper itself does not validate physical capacity against
[`Shape`](../../glossary.md#shape), `TensorDescriptor`, `LayoutDescriptor`, logical element count, offset, or [referenced
span](../../glossary.md#referenced-element-span). The current `Tensor` association performs the compatibility checks described above without
changing the wrapper's raw-storage contract. `TensorFactory` can attach an existing caller-supplied
borrowed storage object or allocate one exact-span automatic-scope heap segment and pass the wrapper
through that same [validation](../../glossary.md#term-validation) path. Import and population policy, deterministic native-resource
[ownership](../../glossary.md#term-ownership), [runtime](../../glossary.md#term-runtime) [residency](../../glossary.md#residency), prepared memory, [device](../../glossary.md#term-device) [buffers](../../glossary.md#term-buffer), and backend storage remain separate
planned work.

<a id="borrowed-lifetime-example"></a>
### [Borrowed](../../glossary.md#ownership-borrowing-and-lease)-[lifetime](../../glossary.md#term-lifetime) example

<a id="goal-and-inputs-40"></a>
#### Goal and inputs

Wrap a caller-owned 16-byte confined segment as four physical [`FLOAT32`](../../glossary.md#term-float32) [elements](../../glossary.md#term-element), then show what
the wrapper reports after the caller closes the [arena](../../glossary.md#term-arena). The allocation supplies the example input;
it is not performed by `MemorySegmentStorage`.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

public final class HostStorageExample {
    public static void main(String[] args) {
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(16, 1);
        MemorySegmentStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 4, segment);

        System.out.println(storage.elementCapacity());
        System.out.println(storage.byteSize());
        System.out.println(storage.segment() == segment);
        System.out.println(storage.isAlive());

        arena.close();
        System.out.println(storage.isAlive());
        System.out.println(storage.segment() == segment);
    }
}
```

<a id="meaningful-lines-11"></a>
#### Meaningful lines

- `arena.allocate(16, 1)` creates the caller-owned input region. [Alignment](../../glossary.md#term-alignment) `1` is sufficient for
  this raw wrapper because it performs no typed access.
- Capacity `4` with [`FLOAT32`](../../glossary.md#term-float32) requires exactly `4 × 4 = 16` bytes, matching the supplied segment.
- The [identity](../../glossary.md#term-identity-meanings) comparison demonstrates exact segment retention rather than a copy or [slice](../../glossary.md#slice).
- Closing [`arena`](../../glossary.md#term-arena) ends the [scope](../../glossary.md#term-scope) because the caller, not the wrapper, owns the [lifetime](../../glossary.md#term-lifetime).

<a id="result-and-interpretation-40"></a>
#### Result and interpretation

The program prints:

```text
4
16
true
true
false
true
```

The result proves exact byte sizing, [borrowed](../../glossary.md#ownership-borrowing-and-lease) [lifetime](../../glossary.md#term-lifetime), point-in-time [liveness](../../glossary.md#term-liveness), and segment [identity](../../glossary.md#term-identity-meanings).
It does not construct a [`Tensor`](../../glossary.md#tensor), validate [layout](../../glossary.md#layout) compatibility, select [byte order](../../glossary.md#text-encoding-and-file-boundaries) or typed access,
allocate through the wrapper, or create [runtime](../../glossary.md#term-runtime)/[backend](../../glossary.md#backend) storage.

<a id="failures-and-useful-variations-36"></a>
#### Failures and useful variations

- A negative capacity fails with `IllegalArgumentException` before multiplication or segment-size
  inspection.
- Capacity multiplication [overflow](../../glossary.md#term-overflow) fails with `ArithmeticException` before size comparison.
- An undersized, oversized, or non-divisible segment fails with `IllegalArgumentException` because
  its byte size is not exact.
- A segment whose [scope](../../glossary.md#term-scope) is already dead fails construction with `IllegalStateException` after the
  size check. Closing the scope after successful construction instead makes `isAlive()` false;
  later JDK memory access through the returned segment fails under JDK scope rules.<span data-glossary-navigation="true"> ([JDK glossary](../../glossary.md#term-java-development-kit-jdk))</span>


---

[Previous topic](sort-topk.md) · [Next topic](identifiers.md)
