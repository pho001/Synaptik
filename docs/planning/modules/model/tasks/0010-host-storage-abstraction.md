# Task 0010: [Host Storage](../../../../glossary.md#host-storage) Abstraction

**Editorial reading key (later context, not a status update).**

The application or a factory supplies an existing MemorySegment to MemorySegmentStorage; the wrapper retains that exact region with type and capacity metadata. It borrows, rather than copies, the bytes. For arena-backed memory, the component that owns the Arena must keep it open while the data is used and close it according to that scope's rules. The wrapper and Tensor are not universal closeable resource owners. Capacity counts physical elements; a later Tensor/layout boundary determines whether those elements cover the described coordinates.

Focused current reading: [Current borrowed host-storage contract](../../../../reference/tensor/host-storage.md#host-visible-storage); [Arena lifetime example](../../../../reference/tensor/host-storage.md#borrowed-lifetime-example). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Define the smallest [model](../../../../glossary.md#term-model)-level contract for [host](../../../../glossary.md#term-host)-visible [tensor](../../../../glossary.md#tensor) storage on Java 26. Add one
[sealed](../../../../glossary.md#term-sealed) `HostTensorStorage` abstraction and one final, [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) `MemorySegmentStorage` wrapper that
bind a logical [`DataType`](../../../../glossary.md#data-type--datatype) and physical [element](../../../../glossary.md#term-element) capacity to an exact-size [JDK](../../../../glossary.md#term-java-development-kit-jdk) [`MemorySegment`](../../../../glossary.md#term-memorysegment).
Expose raw segment access, mutability, and [liveness](../../../../glossary.md#term-liveness) without allocating memory, owning an [arena](../../../../glossary.md#term-arena),
adding typed element access, validating tensor geometry, or representing [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) storage.

## [Scope](../../../../glossary.md#term-scope)

- Add public [sealed](../../../../glossary.md#term-sealed) interface `HostTensorStorage` in the existing `model.storage` package.
- Add public final class `MemorySegmentStorage` as its only permitted implementation.
- Expose exactly the data type, physical element capacity, byte size, underlying segment,
  read-only state, and current scope liveness.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [element glossary](../../../../glossary.md#term-element); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- Validate non-null inputs, non-negative capacity, checked element-to-byte multiplication, exact
  segment byte size, and initial liveness in a deterministic order with exact messages.
- Accept exact-size heap, native, mapped, global-scope, confined-scope, shared-scope, read-only,
  writable, and sliced segments without imposing allocation policy.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- Define [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [ownership](../../../../glossary.md#term-ownership-meanings) and [JDK](../../../../glossary.md#term-java-development-kit-jdk)-scope [lifetime](../../../../glossary.md#term-lifetime) behavior explicitly.
- Preserve [identity](../../../../glossary.md#term-identity-meanings) semantics for [mutable](../../../../glossary.md#term-mutable)/resource-bearing storage wrappers.
- Add focused reflection, [validation](../../../../glossary.md#term-validation), ownership, [lifecycle](../../../../glossary.md#lifecycle), segment-kind, and boundary tests.
- During implementation, update the Tensor API, glossary, task evidence, model master plan, and
  implementation roadmap through the required separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- public [`Tensor`](../../../../glossary.md#tensor), `TensorFactory`, tensor provenance, tensor mutation, storage replacement, or
  gradient/publication state<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication))</span>
- scalar or indexed reads and writes, typed accessors, typed bulk import/export, conversions,
  boolean normalization, BFLOAT16 conversion, copies, fills, or mutation/version tracking<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [boolean glossary](../../../../glossary.md#data-type--datatype); [normalization glossary](../../../../glossary.md#term-normalization); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- array-specific storage classes, one storage class per data type, mutable backing-array access,
  or Java-array allocation<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- allocation of [heap](../../../../glossary.md#term-heap), native, mapped, [arena](../../../../glossary.md#term-arena)-owned, pooled, prepared, [backend](../../../../glossary.md#backend), or [runtime](../../../../glossary.md#term-runtime) memory
- owning, retaining, closing, or hiding an `Arena`; `AutoCloseable`; cleanup callbacks; reference
  counting; leases; pooling; or resource registries<span data-glossary-navigation="true"> ([leases glossary](../../../../glossary.md#term-lease); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- [`Shape`](../../../../glossary.md#shape), `TensorDescriptor`, `LayoutDescriptor`, logical element count, layout offset/span,
  view/alias validation, or tensor-to-storage capacity validation<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [validation glossary](../../../../glossary.md#term-validation))</span>
- byte-order selection, typed `ValueLayout` selection, ABI alignment, backend alignment, address
  exposure, padding, or reinterpretation<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [alignment glossary](../../../../glossary.md#term-alignment); [padding glossary](../../../../glossary.md#padding))</span>
- device buffers, runtime residency, memory slots, workspaces, transfers, prepared memory,
  backend-native storage, native backend registries, kernel routes, or execution state<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [buffers glossary](../../../../glossary.md#term-buffer); [residency glossary](../../../../glossary.md#residency); [kernel glossary](../../../../glossary.md#kernel))</span>
- factories, builders, convenience overloads, storage adapters, additional public storage types,
  package changes, dependencies, preview/incubator features, or Gradle changes<span data-glossary-navigation="true"> ([features glossary](../../../../glossary.md#term-software-feature); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- task 0011 implementation or a detailed task-0011 specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially `modules/model` [ownership](../../../../glossary.md#term-ownership-meanings) of
  `HostTensorStorage` and `MemorySegmentStorage` and its prohibition on [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) storage
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, especially [host storage](../../../../glossary.md#host-storage)
- [Model master plan](../master-plan.md), especially the `model.storage` package boundary
- [Task 0001](0001-data-type-model.md), which defines logical [element](../../../../glossary.md#term-element) byte width
- [Task 0003](0003-layout-descriptor-model.md), which separates [layout](../../../../glossary.md#layout) geometry from storage
- [Task 0007](0007-tensor-descriptor-model.md), which deliberately owns no storage
- [Task 0009](0009-compiled-graph-model.md), which keeps storage out of [compile](../../../../glossary.md#compile)-time [graph](../../../../glossary.md#graph) state
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> and [glossary](../../../../glossary.md), which
  currently describe host storage as planned

## Legacy evidence and rejected coupling

The read-only `legacy/pre-rewrite` branch was inspected for capability evidence through
`tensor/storage/TensorStorage.java`, `TensorStorageAccess.java`, `NativeTensorStorage.java`,
`AbstractNativeTensorStorage.java`, `NativeMemoryAllocation.java`, representative [heap](../../../../glossary.md#term-heap) and native
typed storage classes, `TensorStorageDataTypeTest`, `NativeCpuStorageTest`, and
`docs/cpu-storage-rewrite-plan.md`.

Useful evidence retained by this task is limited to the need for an explicit logical [data type](../../../../glossary.md#data-type--datatype),
physical capacity, byte sizing, [host](../../../../glossary.md#term-host)-memory access, mutability/[lifetime](../../../../glossary.md#term-lifetime) awareness, zero-sized
storage, and support for heap and native host representations. The new contract uses `long`
capacity and stable Java 26 [`MemorySegment`](../../../../glossary.md#term-memorysegment) rather than the legacy `int` size and typed-class
hierarchy.

The following legacy design is rejected:

- one heap and native class per data type;
- a [dtype](../../../../glossary.md#data-type--datatype)-switching access utility with [scalar](../../../../glossary.md#term-scalar) conversion and array downcasts;
- [mutable](../../../../glossary.md#term-mutable)-array exposure and storage version counters in the foundational wrapper;
- allocation handles coupled to [runtime](../../../../glossary.md#term-runtime) `ExecutionResource`;
- [ownership](../../../../glossary.md#term-ownership-meanings) flags, close behavior, [pooling](../../../../glossary.md#term-pooling), retention, or prepared-execution allocation;
- runtime/native registries, materializers, [backend](../../../../glossary.md#backend) routing, [kernel](../../../../glossary.md#kernel) policy, or [residency](../../../../glossary.md#residency); and
- package names, source structure, [validation](../../../../glossary.md#term-validation) accidents, and backend/runtime coupling.

The historical [CPU](../../../../glossary.md#term-cpu) storage rewrite plan is evidence that array/native routing, [materialization](../../../../glossary.md#materialization),
and hot-loop [specialization](../../../../glossary.md#term-specialization) belong to the CPU backend and runtime [lifecycle](../../../../glossary.md#lifecycle). It is not a [model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api)
design and is not copied.

## Architecture constraints

- Both production types live in `io.github.pho001.synaptik.model.storage`.
- `model.storage` may depend only on `model.datatype` and the JDK. No other model package or
  project module is imported.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [module glossary](../../../../glossary.md#term-module))</span>
- `HostTensorStorage` represents physical host-visible capacity. It is not tensor logical shape,
  layout geometry, a graph value, a device buffer, runtime residency, or a prepared memory slot.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [tensor glossary](../../../../glossary.md#tensor); [shape glossary](../../../../glossary.md#shape); [graph value glossary](../../../../glossary.md#graph-value); [buffer glossary](../../../../glossary.md#term-buffer); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [memory slot glossary](../../../../glossary.md#memory-slot))</span>
- The interface is [sealed](../../../../glossary.md#term-sealed) and permits exactly `MemorySegmentStorage`. Java array storage is
  represented without dtype-specific classes by wrapping <span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype))</span>`MemorySegment.ofArray(...)` results.
  The one implementation also accepts native, mapped, global, scoped, and sliced segments.
- The interface is closed so callers cannot provide implementations that evade the capacity,
  ownership, and lifetime contract. Adding a genuinely different host-storage representation is
  a later explicit API decision, not an implementation shortcut in this task.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `MemorySegmentStorage` is a final class, not a record. It wraps mutable memory and a scoped
  resource handle, so equality, hashing, and synchronization identity use ordinary object
  identity. Two wrappers are never equal merely because they contain the same data type,
  capacity, or segment reference.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [synchronization glossary](../../../../glossary.md#term-synchronization); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- The wrapper is borrowed and non-owning. It retains the supplied segment reference but does not
  own, retain, or close the segment scope or its <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [scope glossary](../../../../glossary.md#term-scope))</span>[`Arena`](../../../../glossary.md#term-arena), and it does not implement
  `AutoCloseable`.
- The caller that supplies a scoped segment remains responsible for keeping its scope alive for
  every memory access and closing the owning arena at the correct lifecycle boundary. Global and
  heap segment lifetimes remain governed by the JDK.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [heap glossary](../../../../glossary.md#term-heap))</span>
- Construction rejects a segment whose scope is already not alive. If the scope later becomes not
  alive, `isAlive()` returns false. [Metadata](../../../../glossary.md#term-metadata) remains descriptive, and `segment()` still returns
  the exact supplied segment reference; JDK memory-access operations on that segment enforce the
  closed-scope failure. The wrapper does not replace JDK lifetime or wrong-thread exceptions.
- Read-only state is the supplied segment's JDK state. Writable raw segment access permits callers
  to mutate bytes subject to the segment's scope and thread-access rules. A read-only segment is
  accepted and reports read-only. This task adds no write API, copy, conversion, normalization,
  version counter, synchronization, or thread-safety promise.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- The supplied segment's accessibility policy is preserved. Confined segments remain accessible
  only to their owner thread; shared segments retain shared access; the wrapper performs no thread
  transfer, synchronization, or accessibility conversion.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- [Element](../../../../glossary.md#term-element) capacity is a non-negative `long` physical capacity. Byte size is exactly
  `Math.multiplyExact(elementCapacity, dataType.byteWidth())`. There is no Java-array-size or
  `Integer.MAX_VALUE` limit in the [model](../../../../glossary.md#term-model) contract.
- The supplied segment must have exactly the calculated byte size. Extra capacity is rejected;
  callers that want a smaller view must supply an explicit exact-size `asSlice(...)` segment.
  Exact sizing makes every accepted non-empty segment size divisible by element byte width and
  prevents an unmodeled trailing-byte region.
- Zero element capacity is valid for every data type and requires a zero-byte segment. A zero-byte
  slice or the global <span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>`MemorySegment.NULL` value may therefore be wrapped when alive.
- For width greater than one, the greatest arithmetically representable capacity is
  `Long.MAX_VALUE / byteWidth`; any larger value fails checked multiplication before segment-size
  comparison. Width-one [`BOOL`](../../../../glossary.md#data-type--datatype) can represent `Long.MAX_VALUE` in the arithmetic contract when an
  exact-size segment can exist. No implementation is required to allocate such a segment.
- No alignment check is performed. In particular, an exact-size slice at an address or heap
  offset that is not naturally aligned for the logical data type is accepted because this task
  performs no typed access. `DataType.byteWidth()` does not prescribe [ABI](../../../../glossary.md#term-application-binary-interface-abi), native, [device](../../../../glossary.md#term-device), or
  `ValueLayout` [alignment](../../../../glossary.md#term-alignment).
- No byte order is selected or implied. The wrapper exposes raw bytes plus logical element width;
  later typed access or copy contracts must choose and document <span data-glossary-navigation="true"> ([byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>`ValueLayout` and byte order. A
  backend must not infer its native ABI from this model wrapper.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- No [validation](../../../../glossary.md#term-validation) relates capacity to `Shape.knownElementCount()`,
  `LayoutDescriptor.referencedElementSpan()`, [layout](../../../../glossary.md#layout) offset, [view](../../../../glossary.md#term-view-meanings) metadata, or
  `TensorDescriptor`. Task 0011 or task 0012 owns tensor/storage association and factory
  validation after its exact contract is planned.
- If implementation requires allocation, typed access, another public type, non-borrowed
  ownership, descriptor/layout coupling, runtime/backend state, or an architecture/dependency
  change, stop and report the conflict.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype` — owns [`DataType`](../../../../glossary.md#data-type--datatype) and its logical byte width.
- `io.github.pho001.synaptik.model.storage` — planned owner of host-visible storage contracts; the
  package does not yet contain production types.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [host glossary](../../../../glossary.md#term-host))</span>

Packages added or changed:

- No package boundary is added. The planned `model.storage` package receives its first two public
  types.

Type placement:

- `io.github.pho001.synaptik.model.storage.HostTensorStorage` — closed public model boundary for
  host-visible physical storage facts.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- `io.github.pho001.synaptik.model.storage.MemorySegmentStorage` — validated borrowed wrapper for
  all supported JDK <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) kinds.

Test placement:

- `io.github.pho001.synaptik.model.storage.HostTensorStorageTest` — verifies the exact sealed
  interface shape, permitted implementation, and cross-layer API exclusions.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [shape glossary](../../../../glossary.md#shape); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `io.github.pho001.synaptik.model.storage.MemorySegmentStorageTest` — verifies construction,
  sizing, segment kinds, identity semantics, ownership, mutability, liveness, and deterministic
  failures.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [liveness glossary](../../../../glossary.md#term-liveness))</span>

## Required contracts

### `HostTensorStorage`

Implement exactly this public interface [shape](../../../../glossary.md#shape):

```java
public sealed interface HostTensorStorage permits MemorySegmentStorage {
    DataType dataType();
    long elementCapacity();
    long byteSize();
    MemorySegment segment();
    boolean isReadOnly();
    boolean isAlive();
}
```

The methods [mean](../../../../glossary.md#term-mean):

- `dataType()` returns the exact non-null logical [element](../../../../glossary.md#term-element) type supplied by the wrapper.
- `elementCapacity()` returns the non-negative number of complete physical elements that fit in
  the exact byte region. This is capacity, not a tensor's logical element count.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>
- `byteSize()` returns the non-negative exact product of capacity and logical byte width and is
  also exactly `segment().byteSize()`.
- `segment()` returns the exact supplied non-null [`MemorySegment`](../../../../glossary.md#term-memorysegment) reference, including after its
  scope becomes not alive. It does not create a slice, copy, read-only view, reinterpretation, or
  new lifetime.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [view glossary](../../../../glossary.md#term-view-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- `isReadOnly()` returns the supplied segment's read-only property. It does not mean immutable
  storage when false or confer permission to mutate through a read-only segment.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `isAlive()` returns a point-in-time [snapshot](../../../../glossary.md#term-snapshot) of `segment().scope().isAlive()`. It does not
  guarantee that the scope remains alive for a later access and does not replace the JDK's
  accessibility checks.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>

Do not extend `AutoCloseable`, `Iterable`, a [buffer](../../../../glossary.md#term-buffer) [API](../../../../glossary.md#term-application-programming-interface-api), or another project contract. Do not add
default methods, typed methods, [aliases](../../../../glossary.md#aliasing-and-arena) such as `size()` or `count()`, [scope](../../../../glossary.md#term-scope)/[arena](../../../../glossary.md#term-arena) accessors,
[ownership](../../../../glossary.md#term-ownership-meanings) flags, version methods, address methods, factories, or nested types.

### `MemorySegmentStorage`

Implement a public final class with exactly one public constructor:

```java
public MemorySegmentStorage(
        DataType dataType,
        long elementCapacity,
        MemorySegment segment)
```

The class implements `HostTensorStorage`, is not a record, does not implement `AutoCloseable`, and
does not override `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), or `toString`. Ordinary object [identity](../../../../glossary.md#term-reference-identity) is therefore the
only equality contract; the class's inherited text is diagnostic object identity and not a stable
[serialization](../../../../glossary.md#term-serialization) format.

The constructor retains the exact validated [`DataType`](../../../../glossary.md#data-type--datatype) and [`MemorySegment`](../../../../glossary.md#term-memorysegment) references, the
capacity, and the checked byte size. It performs no allocation, copying, slicing, conversion,
fill, memory access, address [lookup](../../../../glossary.md#term-lookup), [alignment](../../../../glossary.md#term-alignment) lookup, or [arena](../../../../glossary.md#term-arena) [operation](../../../../glossary.md#operation).

#### [Validation](../../../../glossary.md#term-validation) order and exact failures

Apply validation in this order:

1. Reject null `dataType` with `NullPointerException` and exact message `dataType`.
2. Reject null `segment` with `NullPointerException` and exact message `segment`.
3. Reject negative `elementCapacity` with `IllegalArgumentException` and exact message
   `elementCapacity must be non-negative: <value>`.
4. Calculate `byteSize` with checked `long` multiplication. If it overflows, throw a new
   `ArithmeticException` with exact message
   `element byte size overflows long: elementCapacity=<value>, byteWidth=<width>`.
5. Compare `segment.byteSize()` with the calculated value. If unequal, throw
   `IllegalArgumentException` with exact message
   `segment byte size must equal required byte size: required=<required>, actual=<actual>`.
   This one failure covers undersized, oversized, and non-divisible physical regions.
6. If `segment.scope().isAlive()` is false, throw `IllegalStateException` with exact message
   `segment scope is not alive`.

When multiple inputs are invalid, the earlier check wins. Do not catch and translate unrelated
[JDK](../../../../glossary.md#term-java-development-kit-jdk) failures. After successful construction, the six interface methods return stored or directly
delegated facts without adding validation. In particular, `segment()` does not call an
`ensureOpen()` helper and `isAlive()` does not throw when the [scope](../../../../glossary.md#term-scope) is closed.

## Capacity, segment, and [lifecycle](../../../../glossary.md#lifecycle) examples

| Construction | Result |
|---|---|
| [`FLOAT32`](../../../../glossary.md#term-float32), capacity `3`, `MemorySegment.ofArray(new float[3])` | Valid; 12 bytes |
| [`INT64`](../../../../glossary.md#data-type--datatype), capacity `2`, exact native 16-byte segment | Valid; 16 bytes |
| [`BFLOAT16`](../../../../glossary.md#term-bfloat16), capacity `2`, `MemorySegment.ofArray(new short[2])` | Valid; 4 bytes |
| [`BOOL`](../../../../glossary.md#data-type--datatype), capacity `0`, `MemorySegment.NULL` | Valid global zero-byte storage |
| `FLOAT32`, capacity `1`, five-byte segment | Invalid exact-size mismatch; also not divisible by width |
| `FLOAT32`, capacity `1`, eight-byte segment | Invalid; extra bytes are not implicit capacity |
| `FLOAT32`, capacity `1`, `eightBytes.asSlice(1, 4)` | Valid despite unaligned [slice](../../../../glossary.md#slice) origin |
| any [data type](../../../../glossary.md#data-type--datatype), negative capacity | Invalid before multiplication or segment inspection |
| [`FLOAT64`](../../../../glossary.md#term-float64), capacity `Long.MAX_VALUE`, any segment | Invalid checked multiplication |
| exact-size segment from an already closed [arena](../../../../glossary.md#term-arena) | Invalid initial [liveness](../../../../glossary.md#term-liveness) |
| valid confined segment whose arena later closes | Wrapper remains; `isAlive()` becomes false and [JDK](../../../../glossary.md#term-java-development-kit-jdk) access fails |

The examples specify only wrapper construction. They do not promise that later typed access can
use every accepted unaligned segment without an explicit unaligned [layout](../../../../glossary.md#layout) or copy.

## Affected files

Expected new production files:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/storage/HostTensorStorage.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/storage/MemorySegmentStorage.java`

Expected new test files:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/storage/HostTensorStorageTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/storage/MemorySegmentStorageTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) files during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

No focused architecture document is expected to change. The architecture already assigns these
types and this package responsibility to `modules/model`. The capability baseline is planning
input and already includes both names; it must not be edited as task evidence.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- two new production Java files;
- two new focused test Java files; and
- the five documentation and [planning](../../../../glossary.md#term-planning) files listed above.

Do not modify existing Java source or tests, [Gradle](../../../../glossary.md#term-gradle) files, `AGENTS.md`, `ARCHITECTURE.md`, focused
architecture documentation, architecture tests, `capabilities.md`, another [module](../../../../glossary.md#term-module), unrelated
documentation, or a task-0011 specification. If another file, public type, method, [dependency](../../../../glossary.md#term-dependency), or
architecture clarification is required, stop and report it instead of expanding this task.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Every public type, constructor, and method must have meaningful detailed Javadoc. Interface
  method documentation must not rely on method names alone, and implementation overrides must
  document their concrete behavior rather than use only `{@inheritDoc}`.
- `HostTensorStorage` Javadoc must define host-visible physical capacity, explain its sealed
  boundary, and distinguish it from <span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [sealed glossary](../../../../glossary.md#term-sealed))</span>[`Tensor`](../../../../glossary.md#tensor), descriptors/layout, graph values, runtime
  residency, prepared memory, device buffers, and backend storage.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [device glossary](../../../../glossary.md#term-device); [buffers glossary](../../../../glossary.md#term-buffer); [backend glossary](../../../../glossary.md#backend))</span>
- Interface method Javadocs must document units, non-nullness, physical capacity versus logical
  count, exact segment identity, raw-memory mutability, liveness races, JDK scope/thread rules,
  and return values.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [liveness glossary](../../../../glossary.md#term-liveness); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [scope glossary](../../../../glossary.md#term-scope))</span>
- `MemorySegmentStorage` Javadoc must explain borrowed/non-owning lifetime, exact-size validation,
  accepted segment kinds, no allocation/close behavior, no typed-access/alignment/byte-order
  policy, identity equality, and closed-scope behavior.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [lifetime glossary](../../../../glossary.md#term-lifetime); [validation glossary](../../../../glossary.md#term-validation); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Constructor Javadoc must document all three inputs, exact reference retention, validation order,
  every exact failure condition, caller ownership, scope/thread obligations, zero capacity,
  checked maximum, exact sizing, and the fact that accepted slices are not alignment promises.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Concrete accessors must document stored/delegated results, units, identity, nullability,
  read-only semantics, and point-in-time liveness. <span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability))</span>`segment()` must state that it can return a
  segment whose scope is no longer alive and that JDK memory access then enforces failure.
- Javadoc must not imply array ownership, close behavior, synchronization, mutation tracking,
  typed access, tensor/layout compatibility, backend support, ABI alignment, byte order, runtime
  residency, or executable use.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>
- The documentation-focused pass must review existing `DataType.byteWidth()`,
  `LayoutDescriptor`, and `TensorDescriptor` Javadocs. It must record why they remain accurate
  without edits or stop if a required correction falls outside scope.

## Acceptance criteria

- Exactly the two required public production types are added under `model.storage`; no helper,
  nested public type, extra implementation, or package is introduced.
- `HostTensorStorage` is a [sealed](../../../../glossary.md#term-sealed) interface that permits exactly `MemorySegmentStorage` and
  declares exactly the six specified abstract methods with the specified return types.
- `MemorySegmentStorage` is final, is not a record, implements only `HostTensorStorage`, has exactly
  one public constructor with the specified parameter order/types, and is not <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>`AutoCloseable`.
- The class adds no public method beyond the six interface implementations and adds no factory,
  overload, typed access, scope/arena accessor, address, ownership, close, version, copy, fill, or
  conversion API.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Reflection and `javap` confirm the exact interface/class [API](../../../../glossary.md#term-application-programming-interface-api) and that `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests), and
  `toString` are inherited rather than overridden.
- Nulls, negative capacity, checked overflow, exact byte-size mismatch, and initial dead scope fail
  in the specified order with the exact exception type and message.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- Every [`DataType`](../../../../glossary.md#data-type--datatype) calculates byte size through its current `byteWidth()` metadata. Focused tests
  cover all six data types, including width-one <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>[`BOOL`](../../../../glossary.md#data-type--datatype) and raw-bit-width [`BFLOAT16`](../../../../glossary.md#term-bfloat16) capacity.
- Capacity zero succeeds only with an exact zero-byte live segment. Positive capacity with a
  zero-byte segment and zero capacity with a positive-byte segment fail exact sizing.
- Undersized, oversized, and non-divisible segments fail exact sizing. Overflow is detected before
  segment-size comparison. No `int` limit or hidden allocation is added.
- Heap array segments for all applicable primitive carriers, native confined and shared segments,
  a mapped segment, a global zero-byte segment, a read-only segment, a writable segment, and exact
  sliced segments construct successfully using stable Java 26 APIs.<span data-glossary-navigation="true"> ([Heap glossary](../../../../glossary.md#term-heap))</span>
- An unaligned exact-size [slice](../../../../glossary.md#slice) constructs successfully, proving the task does not silently treat
  `DataType` width as [alignment](../../../../glossary.md#term-alignment). Tests do not perform typed access through that slice.
- The exact supplied segment reference is returned. The wrapper does not copy, slice,
  reinterpret, convert, allocate, or close it.
- Read-only state matches the supplied segment. Raw writable segment mutation remains possible
  through JDK APIs, while the wrapper adds no mutation or version behavior.
- A live [arena](../../../../glossary.md#term-arena)-backed storage reports alive before caller closure and not alive afterward.
  `segment()` still returns the exact dead segment; a representative JDK memory access then fails
  with the JDK closed-scope exception. The wrapper itself exposes no close method.
- Confined and shared segment accessibility remains a JDK property. Tests prove construction does
  not convert scopes or segment identity and do not promise cross-thread access for confined
  segments.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Distinct wrappers around the same exact segment and metadata remain unequal by identity, and a
  wrapper equals only itself. No structural/resource equality is introduced.
- No test or API validates [`Shape`](../../../../glossary.md#shape), `TensorDescriptor`, `LayoutDescriptor`, logical element count,
  layout span/offset, view aliases, or tensor compatibility.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [tensor glossary](../../../../glossary.md#tensor))</span>
- Production imports are limited to `model.datatype.DataType`, `java.lang.foreign.MemorySegment`,
  and minimal `java.util` [validation](../../../../glossary.md#term-validation) support. No other [model](../../../../glossary.md#term-model) package or project [module](../../../../glossary.md#term-module) appears.
- Stable Java 26 [FFM](../../../../glossary.md#term-ffm) APIs [compile](../../../../glossary.md#compile) with the existing [toolchain](../../../../glossary.md#term-toolchain) and [release](../../../../glossary.md#term-release) configuration. No
  `--enable-preview`, [incubator module](../../../../glossary.md#term-incubator-module), native-access flag, [dependency](../../../../glossary.md#term-dependency), or build change is added.
- All Javadoc requirements are satisfied, and generated model Javadoc includes the sealed
  hierarchy, ownership/lifetime, exact sizing, identity, mutability, liveness, and boundary rules.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [lifetime glossary](../../../../glossary.md#term-lifetime); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- A separate clean-context documentation-focused agent independently inspects the final source,
  tests, generated Javadoc, test evidence, and diff; finalizes Javadoc, Tensor API, glossary, task
  evidence/status, master-plan status, and roadmap status in this same overall change.
- `docs/api/tensor-api.md` moves host storage from planned to current and documents only the raw
  wrapper contract. It must not claim public Tensor integration, typed access, allocation, array
  factories, layout validation, or runtime/backend storage.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- `docs/glossary.md` marks host storage implemented and distinguishes borrowed model host memory
  from logical layout, public Tensor state, device storage, prepared memory, and residency. No new
  glossary term is required for <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [device glossary](../../../../glossary.md#term-device); [residency glossary](../../../../glossary.md#residency))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment); it is a [JDK](../../../../glossary.md#term-java-development-kit-jdk) API named in the [host](../../../../glossary.md#term-host)-storage entry.
- Existing focused architecture documents and `capabilities.md` remain unchanged because no
  architecture or baseline decision changes.
- Task, master-plan row/current status/decisions/notes, and roadmap frontier/table have matching
  final status. After task 0010 is complete, task 0011 may be named the next `Draft` frontier, but
  no detailed task-0011 specification is created.
- No existing Java/test file, Gradle file, architecture file, architecture test, capability
  baseline, other module, or unrelated documentation is changed.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) after implementation and again after the separate documentation-focused pass:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.storage.HostTensorStorageTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.storage.MemorySegmentStorageTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Manually verify:

- the final diff contains only the two new production files, two new test files, and five allowed
  documentation/planning files;<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- reflection reports one sealed interface, exactly one permitted final implementation, exact
  methods/constructor, no record, no <span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed))</span>`AutoCloseable`, and no extra public [API](../../../../glossary.md#term-application-programming-interface-api);
- `javap -p` and `javap -p -c` confirm stored state, checked multiplication, exact validation
  order/messages, exact segment retention, direct accessors, no allocation/slice/close behavior,
  and inherited object equality/text methods;<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- focused tests cover all six data types, zero/maximum/overflow boundaries, exact versus extra or
  insufficient bytes, non-divisible bytes, all required segment kinds, read-only/writable state,
  unaligned slices, dead-at-construction and closed-after-construction behavior, and wrapper
  identity semantics;<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- mapped-segment tests use only stable JDK file-mapping APIs, close the test arena/channel, and
  clean up the temporary file; they do not add a production allocation policy;<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [arena glossary](../../../../glossary.md#term-arena); [channel glossary](../../../../glossary.md#term-channel))</span>
- production imports contain no shape, layout, tensor, graph, compiler, planning, runtime,
  prepare, backend, storage registry, allocation, or execution type;<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [registry glossary](../../../../glossary.md#term-registry))</span>
- no source or build configuration contains `--enable-preview`, `jdk.incubator`, or a new
  dependency/native-access option;<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- the wrapper never invokes `Arena.of*`, `Arena.allocate`, `MemorySegment.ofArray`, file mapping,
  `asSlice`, `reinterpret`, `address`, typed `get`/`set`, copy/fill, or close in production;
- generated Javadoc documents every public member and the exact ownership, lifetime, sizing,
  mutability, identity, alignment, byte-order, and cross-layer boundaries;<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- [`DataType`](../../../../glossary.md#data-type--datatype), `LayoutDescriptor`, and `TensorDescriptor` Javadocs are reviewed with a reasoned
  no-change result unless an out-of-scope inconsistency requires stopping;<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Tensor API and glossary current/planned language agrees with the implementation without
  claiming task-0011 or task-0012 behavior;
- local Markdown links and anchors in all five changed documentation/planning files resolve,
  fences are balanced, terminology agrees with the glossary, and changed files have no trailing
  whitespace;
- the documentation-focused context follows
  `docs/developer-guide/documentation-rules.md`, applies General style plus API and Javadoc style
  to Java/API work and Planning style to planning updates, and records its identity, inspected
  source/tests/diff, commands, results, limitations, Javadoc review, and glossary impact;
- task 0010 status matches the master plan and roadmap, task 0011 remains `Draft`, and no
  task-0011 specification exists; and
- package direction remains `model.storage -> model.datatype` with no forbidden [module](../../../../glossary.md#term-module) edge.

## Dependencies

- Task 0001 is complete and provides the `DataType.byteWidth()` contract.
- Task 0003A is complete and establishes the current `model.datatype` package.
- Tasks 0003, 0007, and 0009 are complete and provide the documented geometry, descriptor, and
  graph boundaries that storage must not absorb.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [graph glossary](../../../../glossary.md#graph))</span>
- The repository Java baseline is 26; [`MemorySegment`](../../../../glossary.md#term-memorysegment) and [`Arena`](../../../../glossary.md#term-arena) are stable APIs and require no
  preview or incubator opt-in.

## Follow-up tasks

- Task 0011 remains the next ordered task. It will define the public [`Tensor`](../../../../glossary.md#tensor) skeleton and decide
  the exact association and compatibility validation between <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`TensorDescriptor` and
  `HostTensorStorage`.
- Task 0012 will define tensor factories and own allocation/import decisions, typed carriers,
  logical count matching, and any zero-initialization behavior.
- Later focused model tasks may define typed scalar or bulk access and mutation/version semantics
  only when required by the public Tensor contract.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Runtime, prepare, and concrete backend tasks own physical execution slots, device/native backend
  storage, residency, transfers, allocation policy, pooling, and kernel routes.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [concrete backend glossary](../../../../glossary.md#concrete-backend); [device glossary](../../../../glossary.md#term-device); [backend glossary](../../../../glossary.md#backend); [residency glossary](../../../../glossary.md#residency); [pooling glossary](../../../../glossary.md#term-pooling); [kernel glossary](../../../../glossary.md#kernel))</span>

Do not create a detailed task-0011 specification as part of task 0010.

## Architecture impact

Expected impact: None.

The architecture already names `HostTensorStorage` and `MemorySegmentStorage`, assigns [host storage](../../../../glossary.md#host-storage)
to `modules/model`, and excludes [device](../../../../glossary.md#term-device) [residency](../../../../glossary.md#residency) and [backend](../../../../glossary.md#backend)-specific storage. This task chooses
the smallest Java 26 representation inside that boundary and introduces only the already planned
`model.storage -> model.datatype` package edge. It changes no [module](../../../../glossary.md#term-module) boundary, [dependency](../../../../glossary.md#term-dependency) rule,
[lifecycle](../../../../glossary.md#lifecycle) stage, [runtime](../../../../glossary.md#term-runtime)/[backend ownership](../../../../glossary.md#backend-ownership), or public [Tensor](../../../../glossary.md#tensor) behavior. Therefore
`ARCHITECTURE.md`, focused architecture documentation, ADRs, and architecture tests require no
update. If implementation reveals otherwise, stop and report the conflicting rule and required
decision before editing architecture files.

## Implementation prompt

Use this prompt in a separate agentic task/thread with a clean context:

```text
You are a clean-context implementation agent working in the Synaptik repository.

Read first and in full:
- AGENTS.md
- ARCHITECTURE.md
- docs/architecture/current-architecture-plan.md
- docs/architecture/module-boundaries.md
- docs/architecture/dependency-rules.md
- docs/architecture/lifecycle.md
- docs/developer-guide/documentation-rules.md
- docs/developer-guide/documentation/README.md
- docs/developer-guide/documentation/general-style.md
- docs/developer-guide/documentation/api-and-javadoc-style.md
- docs/developer-guide/documentation/planning-style.md
- docs/developer-guide/documentation/example-format.md when an API example changes
- docs/planning/planning-guide.md
- docs/planning/roadmap.md
- docs/planning/modules/model/capabilities.md
- docs/planning/modules/model/master-plan.md
- docs/planning/modules/model/tasks/0001-data-type-model.md
- docs/planning/modules/model/tasks/0003-layout-descriptor-model.md
- docs/planning/modules/model/tasks/0007-tensor-descriptor-model.md
- docs/planning/modules/model/tasks/0010-host-storage-abstraction.md
- docs/api/tensor-api.md
- docs/glossary.md
- current DataType, LayoutDescriptor, TensorDescriptor production/tests and model package tree
- root and model Gradle configuration only to confirm the Java 26 baseline

Implement task 0010 exactly as specified. Create only HostTensorStorage.java,
MemorySegmentStorage.java, HostTensorStorageTest.java, and MemorySegmentStorageTest.java for code
and tests.

HostTensorStorage must be a sealed interface permitting exactly MemorySegmentStorage and declaring
exactly dataType(), elementCapacity(), byteSize(), segment(), isReadOnly(), and isAlive().
MemorySegmentStorage must be a final non-record identity wrapper with exactly the constructor
(DataType, long, MemorySegment), the six implementations, no AutoCloseable, no object-method
overrides, and no extra public API. Follow the task's exact validation order, exception types, and
messages. Retain and return the exact supplied segment. Use checked capacity-to-byte
multiplication and exact byte-size equality. Reject an initially dead scope. Borrow the segment;
do not own or close its arena. After caller closure, report not alive and continue returning the
exact dead segment so JDK access rules enforce failure.

Accept exact-size heap, native, mapped, global, confined/shared, read-only/writable, and sliced
segments. Do not validate alignment or choose byte order. Do not allocate, copy, slice,
reinterpret, access addresses, perform typed reads/writes, add mutation/version behavior, expose
scope/arena ownership, or validate tensor/descriptor/layout compatibility. Do not add dtype-
specific classes, factories, overloads, helpers, another public concept, dependencies, preview or
incubator features, or build changes.

Do not implement Tensor, TensorFactory, task 0011, allocation/import/export, typed scalar/bulk
access, storage replacement, tensor mutation, runtime residency, device/backend storage, prepared
memory, registries, materializers, or kernel routes. Do not modify existing Java/tests, Gradle,
AGENTS.md, ARCHITECTURE.md, focused architecture docs, architecture tests, capabilities.md,
another module, or unrelated docs. Stop and report if implementation needs anything beyond the
affected-file and maximum-scope lists or if architecture uncertainty appears.

Add every Javadoc contract required by the task. Run both focused tests, all model tests, model
Javadoc, full repository tests, git diff checks, and every reflection, javap, import, FFM, scope,
segment-kind, identity, documentation, and status check in the task.

After code implementation and initial validation, hand the actual diff to a separate
documentation-focused agent or thread with a clean context in the same overall change. The
handoff must include this task specification, implementation/test diff, host-storage API and
behavior, architecture constraints, exact ownership/lifetime/sizing/alignment/byte-order
boundaries, expected Tensor API and glossary updates, Javadoc requirements, reasoned no-change
expectations for DataType/LayoutDescriptor/TensorDescriptor and focused architecture, and every
validation command.

That documentation agent must independently read AGENTS.md, ARCHITECTURE.md, the documentation
rules and selected General/API/Javadoc/Planning profiles, this task, final source/tests, generated
Javadoc, Tensor API, glossary, model master plan, roadmap, DataType, LayoutDescriptor, and
TensorDescriptor. It must inspect the actual diff and test evidence rather than rely on the
handoff summary. It must finalize all new Javadocs, move host storage from planned to current in
the Tensor API and glossary, review terminology/examples/links/anchors/fences/whitespace, record
why existing component Javadocs and focused architecture remain unchanged, and synchronize only
the allowed planning files.

At the end, update only this task file, the model master plan, and the roadmap for planning status.
Record local decisions, known limitations, every exact validation result including the
documentation-agent identity and results, implementation notes, and the canonical completion
summary. Do not mark task 0010 Complete until implementation, tests, Javadoc, documentation pass,
scope review, and status synchronization all pass. Task 0011 remains Draft. Do not create a
task-0011 specification and do not commit or push.
```

## Local decisions

- A [sealed](../../../../glossary.md#term-sealed) interface with one final implementation is intentional. [`MemorySegment`](../../../../glossary.md#term-memorysegment) already
  represents heap arrays, native allocations, mapped files, global values, scoped values, and
  slices, so another storage implementation or per-data-type hierarchy is not needed for the
  current contract.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- The wrapper is a final identity class rather than a record because mutable memory and resource
  lifetime are identity-bearing. Structural equality could incorrectly equate separate wrappers
  whose mutation/lifetime observations must remain distinct.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [mutable glossary](../../../../glossary.md#term-mutable); [resource lifetime glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Structural equality glossary](../../../../glossary.md#term-structural-equality); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Capacity is explicit rather than inferred so the contract names physical elements and proves
  the byte relation through checked multiplication. Exact byte-size equality rejects hidden
  trailing bytes and makes explicit slices the only way to wrap a subrange.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>
- The wrapper is [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) and non-owning. [Arena](../../../../glossary.md#term-arena) [ownership](../../../../glossary.md#term-ownership-meanings) cannot be safely inferred from a
  `MemorySegment`, and hiding or creating an arena would introduce allocation/lifecycle policy
  that belongs to callers, factories, runtime, or backends.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [runtime glossary](../../../../glossary.md#term-runtime); [backends glossary](../../../../glossary.md#backend))</span>
- The exact segment remains observable after its [scope](../../../../glossary.md#term-scope) closes. `isAlive()` reports the current
  state, while stable JDK memory-access rules remain the authority for closed-scope and
  wrong-thread failures.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
- Read-only state is descriptive and raw segment access is the only access in this task. Typed
  operations, conversions, versioning, alignment requirements, and byte order remain deferred
  until an owning public API demonstrates the need.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Storage capacity is independent of tensor geometry. Task 0010 cannot prove a shape's logical
  element count or a layout's referenced span without importing responsibilities assigned to
  later Tensor/factory work.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [referenced span glossary](../../../../glossary.md#referenced-element-span))</span>

## Known limitations

- The only implementation is `MemorySegmentStorage`; adding a non-segment representation requires
  a later deliberate change to the sealed API.<span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The wrapper does not allocate memory or extend its lifetime. Callers can invalidate scoped
  storage by closing the owning arena.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [arena glossary](../../../../glossary.md#term-arena))</span>
- Raw writable segments can be mutated without version tracking. Public mutation and cache
  invalidation semantics are not defined yet.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- No typed [element](../../../../glossary.md#term-element), bulk copy, conversion, [boolean](../../../../glossary.md#data-type--datatype) [normalization](../../../../glossary.md#term-normalization), or [BFLOAT16](../../../../glossary.md#term-bfloat16) access is provided.
- Accepted unaligned slices may require unaligned layouts or materialization in a later typed or
  backend path. This task promises only raw byte-region validity.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Byte order](../../../../glossary.md#text-encoding-and-file-boundaries) is unspecified. The logical [`DataType`](../../../../glossary.md#data-type--datatype) supplies width, not a [host](../../../../glossary.md#term-host)/[backend](../../../../glossary.md#backend) [ABI](../../../../glossary.md#term-application-binary-interface-abi).
- Capacity is not validated against shape, layout, descriptor, Tensor, Java array allocation
  limits, backend limits, or runtime memory plans.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean [planning](../../../../glossary.md#term-planning) context `/root/plan_model_0010` read the complete agent instructions,
  architecture contract, current architecture index, module boundaries, dependency rules,
  lifecycle explanation, documentation workflow, documentation profile index, General and
  Planning styles, planning guide, roadmap, capability baseline, model master plan, tasks
  0001/0003/0007/0009, Tensor API, glossary, current model package tree, and the complete current<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor))</span>
  [`DataType`](../../../../glossary.md#data-type--datatype), `LayoutDescriptor`, and `TensorDescriptor` production/test contracts while defining
  and reviewing this task.
- Root and model Gradle configuration review confirmed the Java toolchain and release are both 26,
  the model module adds no configuration, and no preview or incubator option is enabled.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [release glossary](../../../../glossary.md#term-release); [model module glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions))</span>
  `java -version` reported OpenJDK `26.0.1`; `javap java.lang.foreign.MemorySegment` and
  `javap java.lang.foreign.MemorySegment$Scope` confirmed stable `byteSize()`, `scope()`,
  `isReadOnly()`, `asSlice(...)`, primitive-array factories, mapping-related segment properties,
  and `Scope.isAlive()` without preview flags.
- Read-only legacy inspection used `git ls-tree` and `git show legacy/pre-rewrite:<path>` for the
  storage interfaces, access utility, native abstraction/allocation handle, representative
  FLOAT32/INT64/BOOL heap and native classes, data-type and native-storage tests, and historical
  CPU storage rewrite plan. The branch was not checked out or modified, and no source, package
  structure, dtype-specific hierarchy, runtime resource coupling, allocation policy, or backend
  routing was copied.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [heap glossary](../../../../glossary.md#term-heap); [CPU glossary](../../../../glossary.md#term-cpu); [dtype glossary](../../../../glossary.md#data-type--datatype); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- `git status --short --untracked-files=all` and `git diff --name-only` confirmed exactly three
  planning paths changed: this new task, the model master plan, and the roadmap. No Java, test,
  Gradle, agent instruction, architecture, API, glossary, capability-baseline, other-module, or
  unrelated documentation file changed.
- A targeted Ruby path-and-heading check resolved all 58 local Markdown links and anchors in the
  three changed files. No changed link uses a heading fragment, so anchor validation is
  vacuously complete beyond the checked path targets.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- Fence inspection reported balanced backtick fences: eight markers in this task and two each in
  the master plan and roadmap. `rg -n '[[:blank:]]+$'` found no trailing whitespace in any changed
  file.
- `git diff --check` passed for tracked changes. `git diff --no-index --check /dev/null
  docs/planning/modules/model/tasks/0010-host-storage-abstraction.md` emitted no whitespace
  diagnostic; exit status `1` was expected because the complete new file differs from `/dev/null`.
- Status review confirmed task 0010 is linked and [`Ready`](../../../../glossary.md#term-ready) in this task, the master-plan row/current
  status/decisions/notes, and the roadmap frontier/table. Task 0011 remains `Draft`, task order and
  dependencies are unchanged, and no task-0011 specification exists.
- Gradle tests and Javadoc were not run for this planning-only change because no Java, test,
  dependency, build, API, or glossary file changed. The implementation task requires both focused
  storage tests, all model tests, model Javadoc, and the complete repository test lifecycle before
  and after the separate documentation-focused pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The implementation context added exactly the two production and two focused test files specified
  by this task. Independent source, reflection-test, and bytecode review confirmed one sealed
  interface with exactly six methods and one permitted final non-record implementation; the class
  has four final fields, one public constructor, the same six public methods, no <span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [sealed glossary](../../../../glossary.md#term-sealed))</span>`AutoCloseable`,
  and no `Object` method override.
- `MemorySegmentStorage` performs the specified checks in order: named null checks, negative
  capacity, `Math.multiplyExact`, exact segment-size equality, and initial scope liveness. It
  retains the exact inputs, stores the checked byte size, delegates read-only/liveness queries,
  and performs no allocation, slicing, typed access, close, or other FFM operation.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [liveness glossary](../../../../glossary.md#term-liveness); [FFM glossary](../../../../glossary.md#term-ffm); [operation glossary](../../../../glossary.md#operation))</span>
- Clean documentation-focused context `/root/implement_model_0010/review_model_0010_docs` applied General style plus API
  and Javadoc style to Java/API work, Planning style to planning status and evidence, and Example
  format to the new API example. It read the complete architecture contract and focused
  architecture documents, documentation workflow/profiles, planning guide/roadmap, capability
  baseline, model master plan, tasks 0001/0003/0007/0010, Tensor API, glossary, final storage
  source/tests, relevant component sources, generated Javadoc, working-tree status, and the actual
  complete diff/new files rather than relying on the implementation handoff.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- The documentation pass changed only Javadoc in the two new production files among Java sources.
  It finalized physical capacity versus logical count, exact segment identity, raw mutability,
  borrowed arena/scope/thread obligations, point-in-time liveness, dead-scope behavior, exact
  sizing and zero/overflow boundaries, deterministic validation failures, supported segment
  kinds, object identity, and the absence of allocation/close, typed access, alignment, byte
  order, tensor/layout association, and runtime/backend policy. Implementation signatures and
  behavior and both focused tests remained unchanged.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([arena glossary](../../../../glossary.md#term-arena); [overflow glossary](../../../../glossary.md#term-overflow); [alignment glossary](../../../../glossary.md#term-alignment); [byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [layout glossary](../../../../glossary.md#layout); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Existing component Javadocs were reviewed without modification. `DataType.byteWidth()` already
  defines a positive logical byte width and explicitly excludes backend alignment and padding;<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
  `LayoutDescriptor` already defines resolved element geometry, no retained source shape, and no
  storage ownership/materialization/device policy; and <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [shape glossary](../../../../glossary.md#shape); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [materialization glossary](../../../../glossary.md#materialization); [device glossary](../../../../glossary.md#term-device))</span>`TensorDescriptor` already states that it
  owns no storage and separates compiler, planning, prepare, runtime, and backend responsibilities.
  The raw storage wrapper composes those boundaries without changing any component contract.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [prepare glossary](../../../../glossary.md#prepare))</span>
- The Tensor API now presents host storage as an implemented raw model contract, documents its six
  facts, checked exact sizing, borrowed lifecycle, segment kinds, identity semantics, raw
  mutability, liveness race, and exclusions, and removes it from the planned-contract list. It
  does not claim public Tensor integration, factories/allocation, typed access, layout
  compatibility, task-0011/task-0012 behavior, or runtime/backend storage.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [race glossary](../../../../glossary.md#term-race))</span>
- The [API](../../../../glossary.md#term-application-programming-interface-api)'s complete [borrowed](../../../../glossary.md#ownership-borrowing-and-lease)-[lifetime](../../../../glossary.md#term-lifetime) example was compiled with
  `javac -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-host-storage-example /tmp/HostStorageExample.java` and [run](../../../../glossary.md#term-run) with `java -cp
  modules/model/build/classes/java/main:/tmp/synaptik-host-storage-example HostStorageExample`.
  It printed `4`, `16`, `true`, `true`, `false`, and `true` on separate lines, confirming physical
  capacity, byte size, exact segment identity, and caller-controlled liveness without claiming
  typed access or Tensor association.
- The glossary implementation-status convention and existing [host](../../../../glossary.md#term-host)-storage entry now mark
  `HostTensorStorage` and `MemorySegmentStorage` implemented and distinguish borrowed raw host
  memory from logical layout, public Tensor state, device/backend storage, prepared memory, and
  residency. No separate <span data-glossary-navigation="true"> ([residency glossary](../../../../glossary.md#residency))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) term or unrelated terminology was added.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.storage.HostTensorStorageTest` — passed after the documentation
  revision; XML reported 2 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.storage.MemorySegmentStorageTest` — passed after the
  documentation revision; XML reported 10 tests with zero failures, errors, or skips.
- `./gradlew :modules:model:test` — passed after the documentation revision. Independent XML
  aggregation reported 141 tests with zero failures, errors, or skips, including the 12 focused
  storage tests.
- `./gradlew :modules:model:javadoc` — passed after the Javadoc revision. Generated output contains
  both storage types, the sealed hierarchy, constructor, all six concrete and interface methods,
  exact sizing/failures, identity, ownership, mutability, lifetime, alignment/byte-order, and
  cross-layer boundaries in the package summary and class/member indexes.
- `./gradlew test` — passed for the complete repository after the documentation revision with 36
  actionable tasks and no failures.
- `javap -classpath modules/model/build/classes/java/main -p` for both storage types and
  `javap -classpath modules/model/build/classes/java/main -p -c` for `MemorySegmentStorage`
  confirmed the exact API, four fields, constructor validation order, checked multiplication,
  exact input retention, direct accessors/delegation, and inherited equality/hash/text methods.<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash))</span>
- Production imports are exactly `model.datatype.DataType`, `MemorySegment`, and `Objects` where
  needed. Targeted production-source review found no arena creation/allocation, array factory,
  mapping, slice, reinterpretation, address lookup, typed get/set, copy/fill, or close call. Source
  and build configuration review found no preview, incubator, native-access, dependency, Gradle, or
  release change.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Focused test review confirmed all six data types, zero and arithmetic capacity boundaries,
  overflow-before-size behavior, undersized/oversized/non-divisible regions, heap/native/mapped/
  global/confined/shared/read-only/writable/sliced/unaligned segments, exact segment/scope
  identity, raw mutation, initial and later closure, and wrapper identity. The mapped test closes
  its channel/arena and deletes its temporary file. Tests introduce no tensor/layout association.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel))</span>
- A targeted Ruby check resolved all 120 local Markdown links and anchors in the Tensor API,
  glossary, this task, model master plan, and roadmap. Fence counts were balanced; trailing-
  whitespace checks and no-index whitespace checks for new files found no diagnostics.
- Focused architecture documents, `ARCHITECTURE.md`, `capabilities.md`, ADRs, and architecture tests
  remain unchanged. The architecture already assigns both storage types to `modules/model` and
  excludes device/runtime/backend storage; the focused documents already describe that boundary.
  The capability baseline remains the broader multi-task target, while this task implements its
  smallest raw wrapper without changing baseline scope. No architecture or dependency decision
  changed, so no architecture documentation, ADR, or architecture-test update is warranted.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Final scope/status review confirmed exactly nine changed or new repository files: two production
  files, two focused tests, the Tensor API, glossary, this task, model master plan, and roadmap.
  Task 0010 is `Complete` in all three planning [views](../../../../glossary.md#term-view-meanings); task 0011 remains the next `Draft` frontier,
  and no task-0011 specification exists.
- `git diff --check` passed after documentation and planning synchronization. No-index checks also
  found no whitespace errors in the five untracked files.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Implementation notes

- Added the exact six-method [sealed](../../../../glossary.md#term-sealed) `HostTensorStorage` boundary and its sole final
  `MemorySegmentStorage` [identity](../../../../glossary.md#term-identity-meanings) implementation in `model.storage`.
- Added deterministic checked construction, exact-size borrowed segment retention, raw
  read-only/liveness reporting, and no allocation, ownership, typed access, or geometry coupling.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [liveness glossary](../../../../glossary.md#term-liveness); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Added 12 focused tests for API shape, every data type and required segment kind, validation order
  and messages, arithmetic boundaries, raw mutability, lifecycle, identity, and exclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype); [validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Finalized both new Javadocs, moved host storage from planned to current in the Tensor API and
  glossary, and synchronized task/master-plan/roadmap status after the clean documentation pass.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [Tensor glossary](../../../../glossary.md#tensor))</span>

## Completion summary

- Completed changes: Implemented and documented exact-size borrowed Java 26 memory-segment host
  storage with physical capacity, raw mutability, and point-in-time liveness facts.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [host storage glossary](../../../../glossary.md#host-storage); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- Files changed or created: Two production types, two focused test classes, the Tensor API,
  glossary, this task specification, model master plan, and implementation roadmap.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: Both focused suites (2 and 10 tests), all 141 model tests, model Javadoc,
  full repository tests, compiled API example, reflection/bytecode/import/FFM/build review,
  link/anchor/fence/whitespace/status/scope checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [bytecode glossary](../../../../glossary.md#term-bytecode); [FFM glossary](../../../../glossary.md#term-ffm); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: Clean context `/root/implement_model_0010/review_model_0010_docs` independently reviewed
  the implementation, tests, generated Javadoc, component contracts, API, glossary, architecture
  boundaries, actual diff, and synchronized planning evidence.
- Documentation impact: Host storage is now a current raw model contract. Public Tensor
  association, factories/allocation, typed access, layout compatibility, runtime residency,
  prepared memory, and backend/device storage remain separate planned responsibilities.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [backend glossary](../../../../glossary.md#backend); [device glossary](../../../../glossary.md#term-device))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: Both new production Javadocs are complete. Existing [`DataType`](../../../../glossary.md#data-type--datatype),
  `LayoutDescriptor`, and `TensorDescriptor` Javadocs remain accurate for the reasons recorded in
  validation evidence and required no out-of-scope edits.
- Glossary impact: The existing host-storage definition and implementation-status convention now
  reflect the implemented borrowed wrapper; no new <span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) term or unrelated change was
  needed.
- Unresolved issues: None.
- Follow-up required: None. Task 0011 is the next [planning](../../../../glossary.md#term-planning) frontier and remains `Draft` without a
  detailed task specification.

Status: Complete
