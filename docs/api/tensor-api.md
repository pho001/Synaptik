# Tensor API

## Purpose and mental model

This reference documents the public model contracts that are implemented today. Despite the page title, the mutable `Tensor`, concrete operation families, and tensor-factory contracts are still planned. Raw host-visible storage is implemented independently of those future contracts. The authoritative module boundary remains [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

The current types describe logical values without allocating or executing a tensor:

```text
DataType + Shape + optional LayoutDescriptor + requiresGrad = TensorDescriptor
DataType + physical capacity + exact MemorySegment           = MemorySegmentStorage
TensorId / NodeId / ValueId                                  = distinct identity domains
Operation                                                     = OperationKind + OperationAttrs
ValueId + TensorDescriptor                                    = GraphValue
NodeId + Operation + ordered input/output ValueIds            = CompiledNode
values + nodes + boundaries + node phases                     = CompiledGraphModel
TensorId + ValueId                                             = PublicationBinding
```

An implemented `TensorDescriptor` keeps the logical element type, shape, explicit layout state,
and gradient eligibility together as one immutable value. It is still only a description: the
mutable `Tensor`, graph-wide validation, storage association, inference, and execution remain separate
contracts.

The implemented `HostTensorStorage` boundary describes a raw host-memory region. Its one
implementation, `MemorySegmentStorage`, borrows an exact JDK memory segment and records physical
capacity without allocating memory or relating that capacity to a tensor descriptor or layout.

An `OperationKind` says which computation is meant, while `OperationAttrs` carries its typed semantic parameters. The implemented `Operation` record keeps those two values together. It does not attach them to a graph, infer a result, or execute them.

An implemented `GraphValue` describes logical data. An implemented `CompiledNode` describes one
place where operation semantics consume and produce that data. The implemented
`CompiledGraphModel` contains those elements, declares ordered graph boundaries, classifies every
node with a `GraphPhase`, and validates structural closure. It is compile-time model state, not a
compiler, compile artifact, prepared program, or runtime schedule. The standalone implemented
`PublicationBinding` associates public tensor identity with graph-local value identity for a later
compiler-owned publication plan.

## Data types

The public data type contracts live in `io.github.pho001.synaptik.model.datatype`. The initial `DataType` model contains six backend-independent element types:

| Data type | Category | Bit width | Byte width | Differentiable |
|---|---|---:|---:|---|
| `FLOAT64` | `FLOATING` | 64 | 8 | yes |
| `FLOAT32` | `FLOATING` | 32 | 4 | yes |
| `BFLOAT16` | `FLOATING` | 16 | 2 | yes |
| `INT32` | `INTEGRAL` | 32 | 4 | no |
| `INT64` | `INTEGRAL` | 64 | 8 | no |
| `BOOL` | `BOOLEAN` | 8 | 1 | no |

`FLOAT32` is the default floating data type. Data type metadata describes logical model semantics and does not claim that a particular backend supports the type or uses the same physical allocation alignment.

### Floating promotion

Implicit promotion is defined only within the floating category:

```text
BFLOAT16 < FLOAT32 < FLOAT64
```

`DataTypePromotion.promoteFloating(left, right)` returns the widest input precision. Integral, boolean, null, and cross-category inputs are rejected. Converting between categories requires a future explicit cast operation rather than an implicit promotion rule.

For example:

```java
DataType result = DataTypePromotion.promoteFloating(
        DataType.BFLOAT16, DataType.FLOAT64);
```

The first input has 16 logical bits and the second has 64, so the widest precision is `FLOAT64`. This result is a model-level type decision; it does not prove that a backend supports the eventual operation.

### BFLOAT16 representation

`BFloat16Bits` converts scalar values between Java `float` and raw BFLOAT16 bits held in a `short`. Conversion to BFLOAT16 uses round-to-nearest with ties to even, preserves signed zero and infinities, and canonicalizes NaN to `0x7FC0`.

The utility describes the value format only. It does not allocate storage, expose device formats, or report backend capabilities.

## Shapes and dimensions

The public shape contracts live in `io.github.pho001.synaptik.model.shape`. `Shape` is an immutable ordered collection of `Dimension` values. A dimension is either:

- `StaticDimension`, with a known non-negative `long` size; or
- `DynamicDimension`, with a canonical non-blank symbolic name.

Dynamic dimensions are explicit values and never use negative numeric sentinels. Shape values defensively isolate caller-owned arrays and expose immutable dimension lists.

### Scalars and empty tensors

`Shape.scalar()` is the canonical rank-0 scalar shape. It has no axes and its known element count is one.

Static size zero is valid. For example, `Shape.of(2, 0, 4)` represents an empty tensor and has a known element count of zero. Static dimensions use `long`; later storage implementations may impose narrower allocation limits without changing the logical shape.

### Static and dynamic element counts

`Shape.knownElementCount()` returns a present checked `long` value only when every dimension is static. Dynamic shapes return an empty optional. Non-zero multiplication overflow raises `ArithmeticException`; a fully static shape containing a zero dimension has count zero even when multiplying other dimensions would overflow.

`Shape.toLongArray()` copies the ordered static sizes and rejects dynamic shapes. Positive and negative axes are normalized through the shape, while every axis is invalid for a rank-0 scalar.

### Broadcasting

`ShapeBroadcast.broadcast(left, right)` applies right-aligned broadcasting. Equal dimensions are preserved and static size `1` expands to the opposing dimension. This includes scalar broadcasting and zero-sized dimensions such as `[0, 3]` with `[1, 3]`.

Equal dynamic symbols are compatible, and a singleton may expand to a dynamic dimension. Different symbols or a dynamic dimension paired with a non-singleton static size are rejected because local model code cannot prove their compatibility. Graph-wide symbolic constraints belong to future compiler shape inference.

Shape broadcasting does not calculate strides, layouts, storage, materialization, or backend execution information.

### Shape example

```java
Shape left = Shape.of(2, 1, 4);
Shape right = Shape.of(3, 4);
Shape result = ShapeBroadcast.broadcast(left, right);
long count = result.knownElementCount().orElseThrow();
```

Broadcasting aligns axes from the right. The pairs are `4` with `4`, `1` with `3`, and the unmatched leading `2` with an implicit singleton. The result is `Shape[2, 3, 4]`, and its element count is `2 × 3 × 4 = 24`. No storage is allocated.

## Resolved layouts

The public layout contracts live in `io.github.pho001.synaptik.model.layout`. `LayoutDescriptor` is an immutable description of resolved logical element geometry for a fully static shape. The descriptor records rank, non-negative `long` element strides, a non-negative storage offset measured in elements, explicit view/alias metadata, and the checked element span needed to contain every referenced index. It does not retain the `Shape` used to construct it.

Resolved layouts have four geometric kinds:

- `DENSE_CONTIGUOUS` uses canonical row-major strides and offset zero;
- `DENSE_WITH_OFFSET` uses canonical row-major strides and a non-zero element offset;
- `STRIDED` uses non-canonical strides without non-singleton broadcast repetition; and
- `BROADCAST_ZERO_STRIDE` repeats at least one static dimension larger than one through stride zero.

`LayoutDescriptor.contiguous(shape)` derives canonical row-major strides. `LayoutDescriptor.of(shape, strides, storageOffset, view)` validates and classifies explicit geometry. Caller-owned stride arrays are defensively copied, and `strides()` returns a new copy. Individual strides support positive and negative axis lookup.

Layout classification is independent of the explicit view flag except that broadcast zero-stride repetition must be marked as a view. A raw zero stride on a singleton or empty dimension is not itself broadcast geometry. Canonical empty layouts are classified as dense even when their canonical stride sequence contains zero.

The referenced element span is zero for any shape containing a zero-sized dimension. A rank-0 scalar references one element at its offset. All stride and span arithmetic is checked for `long` overflow.

Dynamic shapes do not yet have numeric layout descriptors because their concrete strides and span are unresolved. Symbolic layout resolution belongs to later compiler and preparation contracts. The model descriptor exposes geometry only: it does not own storage, byte addresses, device state, backend information, or a decision about whether materialization is required.

### Layout example

```java
Shape shape = Shape.of(2, 3);
LayoutDescriptor layout = LayoutDescriptor.of(
        shape, new long[] {0, 1}, 0, true);
```

The logical shape has six positions, but axis 0 has stride zero: both rows reuse the same three storage elements. The referenced span is therefore `3`, computed as `(2 - 1) × 0 + (3 - 1) × 1 + 1`. The descriptor is a `BROADCAST_ZERO_STRIDE` view. This calculation describes aliasing geometry; it does not authorize in-place writes or decide whether a backend must materialize a copy.

## Tensor descriptors

The public `TensorDescriptor` contract lives in
`io.github.pho001.synaptik.model.tensor`. It is an immutable record with four components:

| Component | Meaning |
|---|---|
| `dataType` | The non-null logical type of every element. |
| `shape` | The non-null ordered logical dimensions. |
| `layout` | A non-null `Optional`: present for resolved numeric geometry, empty for unresolved geometry. |
| `requiresGrad` | Whether model-level gradient eligibility is requested. This may be true only for a differentiable data type. |

Layout resolution and shape resolution are separate facts. A dynamic shape must use
`Optional.empty()` because its numeric strides and span are not known. A fully static shape may
also use `Optional.empty()` when layout has not been resolved yet; the descriptor does not invent a
contiguous default.

When a layout is present, construction rebuilds it against the paired shape through
`LayoutDescriptor.of(...)` and compares the complete value. The check covers rank, strides,
element offset, view flag, derived layout kind, and referenced element span. It proves that the
public geometry is compatible with the paired shape. It cannot prove which shape originally
created the layout because `LayoutDescriptor` deliberately does not retain source-shape identity.

`TensorDescriptor` uses record equality and hashing across all four components. Its `layout`
component follows `Optional` value semantics: inspect presence or compare values, and do not rely
on the optional container's identity. A present optional contains the exact immutable
`LayoutDescriptor` object supplied to the constructor. The generated text form is useful for
diagnostics but is not a serialization format.

### Complete descriptor example

#### Goal and inputs

Create three descriptors that distinguish a fully static unresolved value, the same static value
with resolved row-major geometry, and a dynamic unresolved value. The static shape is `[2, 3]`, the
dynamic shape is `[batch, 3]`, and every value uses `FLOAT32`.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Optional;

public final class TensorDescriptorExample {
    public static void main(String[] args) {
        Shape staticShape = Shape.of(2, 3);
        TensorDescriptor staticUnresolved = new TensorDescriptor(
                DataType.FLOAT32, staticShape, Optional.empty(), false);

        LayoutDescriptor rowMajor = LayoutDescriptor.contiguous(staticShape);
        TensorDescriptor resolved = new TensorDescriptor(
                DataType.FLOAT32, staticShape, Optional.of(rowMajor), true);

        Shape dynamicShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));
        TensorDescriptor dynamicUnresolved = new TensorDescriptor(
                DataType.FLOAT32, dynamicShape, Optional.empty(), false);

        System.out.println(staticUnresolved.layout().isEmpty());
        System.out.println(resolved.layout().orElseThrow().kind());
        System.out.println(resolved.layout().orElseThrow().referencedElementSpan());
        System.out.println(dynamicUnresolved.shape());
    }
}
```

#### Meaningful lines

- `staticUnresolved` shows that known sizes do not force a layout decision.
- `LayoutDescriptor.contiguous(staticShape)` resolves row-major strides `[3, 1]`, offset `0`, and
  referenced span `6`; `resolved` pairs that geometry with the same shape.
- `requiresGrad` is true only on `resolved`. `FLOAT32` permits that request because it is
  differentiable.
- `dynamicUnresolved` keeps the symbolic `batch` dimension and therefore has no numeric layout.

#### Result and interpretation

The program prints:

```text
true
DENSE_CONTIGUOUS
6
Shape[batch, 3]
```

These results show the two layout states and the concrete geometry of the resolved `[2, 3]`
example. They do not allocate tensor storage, create a public `Tensor` or graph node, infer the
dynamic `batch` size, choose materialization, select a backend, or execute an operation.

#### Failures and useful variations

- Supplying `Optional.of(rowMajor)` with `dynamicShape` fails with `IllegalArgumentException`
  because numeric geometry cannot be validated for an unresolved dimension.
- Requesting gradients for `INT32`, `INT64`, or `BOOL` fails with `IllegalArgumentException`.
- A present layout whose rank or reconstructed geometry differs from the paired shape fails with
  `IllegalArgumentException`; checked reconstruction overflow remains an `ArithmeticException`.
- Passing `null` for the data type, shape, or optional itself fails with `NullPointerException`.

## Host-visible storage

The public storage contracts live in `io.github.pho001.synaptik.model.storage`.
`HostTensorStorage` is a sealed raw-storage boundary with exactly one permitted implementation,
the final `MemorySegmentStorage` class. The wrapper exposes six facts:

| Fact | Meaning |
|---|---|
| `dataType()` | The exact non-null logical element type supplied at construction. |
| `elementCapacity()` | A non-negative physical capacity in complete elements, not a tensor's logical element count. |
| `byteSize()` | The checked product of capacity and `DataType.byteWidth()`, in bytes. |
| `segment()` | The exact supplied `MemorySegment` reference, including after its scope closes. |
| `isReadOnly()` | The supplied segment's JDK read-only state. |
| `isAlive()` | A point-in-time snapshot of the supplied segment scope's liveness. |

Construction requires exact equality between `segment.byteSize()` and the checked byte-size
product. Zero capacity therefore requires a live zero-byte segment. Extra bytes are not treated as
implicit capacity; a caller that wants a subrange supplies an explicit exact-size segment slice.
Heap, native, mapped, global, confined, shared, read-only, writable, and sliced segments are all
accepted when they meet the same size and initial-liveness rules.

The wrapper is borrowed and non-owning. It retains the exact segment but does not allocate memory,
own or close an arena, extend a scope's lifetime, or implement `AutoCloseable`. The caller must keep
scoped memory alive and obey the JDK's thread-access rules. `isAlive()` can become stale immediately,
and a false result does not hide the segment: `segment()` still returns the same dead reference, so
JDK memory access reports the closed-scope failure.

Writable raw memory can be changed through JDK APIs, but the wrapper supplies no typed element
access, conversion, copying, synchronization, or mutation-version tracking. It chooses no
`ValueLayout`, byte order, or alignment. An exact unaligned slice can therefore be wrapped, but
that acceptance is not a promise that a future typed or backend path can access it without an
unaligned layout or materialization.

`MemorySegmentStorage` uses ordinary object identity. Two wrappers around the same segment and
metadata are still unequal. Physical capacity is not validated against `Shape`,
`TensorDescriptor`, `LayoutDescriptor`, logical element count, offset, or referenced span. Public
`Tensor` association, factories, allocation/import policy, runtime residency, prepared memory,
device buffers, and backend storage remain separate planned work.

### Borrowed-lifetime example

#### Goal and inputs

Wrap a caller-owned 16-byte confined segment as four physical `FLOAT32` elements, then show what
the wrapper reports after the caller closes the arena. The allocation supplies the example input;
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

#### Meaningful lines

- `arena.allocate(16, 1)` creates the caller-owned input region. Alignment `1` is sufficient for
  this raw wrapper because it performs no typed access.
- Capacity `4` with `FLOAT32` requires exactly `4 × 4 = 16` bytes, matching the supplied segment.
- The identity comparison demonstrates exact segment retention rather than a copy or slice.
- Closing `arena` ends the scope because the caller, not the wrapper, owns the lifetime.

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

The result proves exact byte sizing, borrowed lifetime, point-in-time liveness, and segment identity.
It does not construct a `Tensor`, validate layout compatibility, select byte order or typed access,
allocate through the wrapper, or create runtime/backend storage.

#### Failures and useful variations

- A negative capacity fails with `IllegalArgumentException` before multiplication or segment-size
  inspection.
- Capacity multiplication overflow fails with `ArithmeticException` before size comparison.
- An undersized, oversized, or non-divisible segment fails with `IllegalArgumentException` because
  its byte size is not exact.
- A segment whose scope is already dead fails construction with `IllegalStateException` after the
  size check. Closing the scope after successful construction instead makes `isAlive()` false;
  later JDK memory access through the returned segment fails under JDK scope rules.

## Typed identifiers

The model separates public tensor identity from graph-local computation and data identity:

- `io.github.pho001.synaptik.model.tensor.TensorId` identifies public mutable tensor state;
- `io.github.pho001.synaptik.model.graph.NodeId` identifies a computation occurrence within an owning graph; and
- `io.github.pho001.synaptik.model.graph.ValueId` identifies an input, intermediate, or output logical value within an owning graph.

Each identifier is an immutable record over one non-negative `long` value. Zero is valid, negative sentinels are rejected, and different identifier types cannot be interchanged. The records do not generate or guarantee uniqueness; later tensor factories, graph builders, and compiler sessions own allocation within their lifecycle.

Graph-local numeric identifiers may be reused by different graph containers. `NodeId` identifies a computation, whereas `ValueId` identifies data flowing between computations. A value can exist without a producing node, one node can produce multiple values, and one value can have multiple consumers.

An implemented `PublicationBinding` associates a public `TensorId` with a `ValueId`. The binding is
standalone and cannot prove that the value belongs to a particular graph; a later compiler-owned
`PublicationPlan` will provide that context. A planned `Tensor` does not store graph-local IDs
because the same tensor may participate in multiple separately compiled graphs. `OperationId` is
not currently defined: operation semantics occur through graph nodes, and no independent
operation-identity lifecycle has been established.

## Operation semantic foundation

The public operation-foundation contracts live in `io.github.pho001.synaptik.model.operation`. They separate four concepts that are easy to confuse:

| Concept | Current role |
|---|---|
| `OperationKind` | An implemented open interface that identifies which backend-independent computation an operation describes. |
| `OperationAttrs` | An implemented marker interface for the immutable, typed semantic parameters of that computation. |
| `NoOperationAttrs.INSTANCE` | The implemented canonical attribute value for a kind that has no semantic parameters. |
| `Operation` | An implemented immutable descriptor that stores one kind and one `OperationAttrs` value. |

`OperationKind` declares only `name()`. The result must be a stable, non-null, non-blank diagnostic name, but it is not a serialization token or a global lookup key. Enum-based kind families are the expected common implementation because an enum already supplies a stable `name()`, equality, and hashing. Two enum values from different kind families remain different typed values even if both names contain the same text.

`OperationAttrs` deliberately declares no methods. A family-specific implementation defines its own typed fields and must be immutable, defensively isolate mutable constructor inputs, and provide structural equality and hashing. `NoOperationAttrs.INSTANCE` makes the absence of parameters an explicit non-null value rather than `null` or an empty map. These are implementation contracts, not runtime validators: the interfaces themselves cannot prevent a custom implementation from returning an invalid name or retaining mutable state.

### Test-local conceptual example

The following types are examples local to a test or explanation. They are not production operation kinds or promises about a future operation-family API.

```java
private enum SampleKind implements OperationKind {
    SAMPLE
}

private record SampleAttrs(int axis, boolean keepDimensions)
        implements OperationAttrs {}

OperationKind kind = SampleKind.SAMPLE;
OperationAttrs attrs = new SampleAttrs(2, true);
OperationAttrs emptyAttrs = NoOperationAttrs.INSTANCE;
Operation operation = new Operation(kind, attrs);
Operation equalOperation = new Operation(
        SampleKind.SAMPLE, new SampleAttrs(2, true));
```

The `SampleKind` declaration shows how an enum inherits `Enum.name()` and therefore returns the stable diagnostic text `"SAMPLE"`. The `SampleAttrs` record shows the intended structural value style: the two fields are typed semantic parameters, and two records with the same `axis` and `keepDimensions` values compare equal. Assigning `NoOperationAttrs.INSTANCE` shows the single canonical value used when a kind has no parameters.

`new Operation(kind, attrs)` stores those exact non-null objects without copying or normalization. The second construction uses equal component values, so `operation.equals(equalOperation)` is `true` and their hash codes are equal. The accessors return the original objects: `operation.kind() == kind` and `operation.attrs() == attrs` are both `true`. At the end of the example, `kind.name()` is `"SAMPLE"`, `attrs` contains `axis = 2` and `keepDimensions = true`, and `emptyAttrs` is the singleton `NoOperationAttrs.INSTANCE` available for a parameterless sample kind.

The example demonstrates the implemented descriptor's construction, ownership, and record value semantics. It does not establish that `SampleAttrs` is compatible with `SampleKind`: the generic descriptor checks only that neither component is null. It also does not create a production operation family or graph node, infer a result shape or data type, report backend support, select a kernel, or execute computation. Concrete kinds, family attributes, compiler behavior, and executable support remain later work in their owning layers.

## Graph values and compiled nodes

The public graph-model contracts live in `io.github.pho001.synaptik.model.graph`. They keep data,
computation, graph boundaries, node phase, and publication association separate:

| Contract | Stored state | Meaning |
|---|---|---|
| `GraphValue` | `ValueId id`, `TensorDescriptor descriptor` | One described logical input, intermediate result, or output. |
| `CompiledNode` | `NodeId id`, `Operation operation`, ordered `inputs`, ordered `outputs` | One occurrence of computation and the value positions it consumes and produces. |
| `GraphPhase` | Exactly `FORWARD` or `BACKWARD` | A node's compile-time classification, not a compile mode or runtime schedule. |
| `CompiledGraphModel` | Ordered `values`, topological `nodes`, ordered `inputs` and `outputs`, exact `nodePhases` | One immutable, structurally closed compile-time graph. |
| `PublicationBinding` | `TensorId tensorId`, `ValueId valueId` | A standalone association for a later compiler-owned publication plan. |

Both identity types are local to an owning graph context. Equal numeric IDs in another graph do
not establish a relationship, and a `NodeId` is never interchangeable with a `ValueId`.

`GraphValue` stores no producer or consumer. This allows an input value to use the same record as a
produced value. `CompiledGraphModel` derives producer relationships while validating construction,
but it stores no producer, consumer, node, value, or phase index.

`CompiledNode` copies its input and output lists with `List.copyOf`, preserving encounter order and
preventing later caller mutation from changing the node. Input positions may be empty or repeat a
value ID. Output positions must contain at least one value and must be unique within that one node.
The lists have value semantics: callers rely on their contents and order, not container identity.

The records use generated structural equality and hashing. For `CompiledNode`, list order and
repeated input positions are part of equality. Generated text is useful for diagnostics but is not
a serialization format, graph-validation result, or execution-dispatch key.

`CompiledGraphModel` snapshots all four lists with `List.copyOf`; list order is part of the record
value. It snapshots `nodePhases` with `Map.copyOf`; the mapping is structural, but iteration order
is not an API contract. Construction requires unique value and node IDs, resolvable boundaries and
node references, producer-free inputs, exactly one producer for every non-input value, topological
node order, at least one graph output, and exactly one phase for every node. Repeated node inputs,
zero-input nodes, unused graph inputs, and a zero-node pass-through graph are valid.

`GraphPhase` contains only `FORWARD` and `BACKWARD`, in that declaration order. It does not encode
an optimizer phase, compile mode, or runtime schedule. `PublicationBinding` remains separate from
`CompiledGraphModel` and carries no `Tensor`, gradient role, publication policy or target, storage,
backend, or execution state.

### Complete test-local graph-model example

#### Goal and inputs

Connect two scalar graph values through one sample operation occurrence, validate them as a
compiled graph model, and create a separate publication binding. The input has
`ValueId(0)`, the output has `ValueId(1)`, the node has `NodeId(0)`, and both values use an
unresolved-layout `FLOAT32` scalar descriptor. `SampleKind` is test-local and is not a production
operation family.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.PublicationBinding;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public final class GraphElementExample {
    private enum SampleKind implements OperationKind {
        IDENTITY
    }

    public static void main(String[] args) {
        TensorDescriptor scalar = new TensorDescriptor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), false);
        GraphValue input = new GraphValue(new ValueId(0), scalar);
        GraphValue output = new GraphValue(new ValueId(1), scalar);

        Operation identity = new Operation(
                SampleKind.IDENTITY, NoOperationAttrs.INSTANCE);
        CompiledNode node = new CompiledNode(
                new NodeId(0),
                identity,
                List.of(input.id()),
                List.of(output.id()));

        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(input, output),
                List.of(node),
                List.of(input.id()),
                List.of(output.id()),
                Map.of(node.id(), GraphPhase.FORWARD));
        PublicationBinding publication = new PublicationBinding(
                new TensorId(9), output.id());

        System.out.println(node.operation().kind().name());
        System.out.println(graph.inputs().stream().map(ValueId::value).toList());
        System.out.println(graph.outputs().stream().map(ValueId::value).toList());
        System.out.println(graph.nodePhases().get(node.id()));
        System.out.println(publication.tensorId());
    }
}
```

#### Meaningful lines

- The one `TensorDescriptor` describes the logical scalar facts. Reusing that immutable value does
  not allocate shared storage or claim that input and output are the same graph value.
- The two `GraphValue` constructions give the data positions distinct `ValueId` values. Neither
  construction names a producer.
- `Operation` describes the sample identity semantics. `CompiledNode` gives that operation one
  occurrence and connects its ordered input position to value `0` and output position to value `1`.
- Passing `input.id()` and `output.id()` uses typed data identities. The node snapshots both lists;
  it does not store either `GraphValue` object or infer their descriptors.
- `CompiledGraphModel` snapshots the graph elements and boundaries, verifies that node `0` is in
  topological order, and requires its one `FORWARD` phase entry.
- `PublicationBinding` associates tensor identity `9` with output value `1`. It remains a separate
  object because the graph model does not own publication policy or execution.

#### Result and interpretation

The program prints:

```text
IDENTITY
[0]
[1]
FORWARD
TensorId[value=9]
```

The result shows the operation kind, validated graph boundaries, compile-time node phase, and
public tensor identity selected by the standalone binding. Construction proves structural graph
closure. It does not validate operation arity or descriptor compatibility, prove that the binding
belongs to this graph, create a compiler-owned `PublicationPlan` or `CompileArtifacts`, choose
storage or a backend, prepare execution, or run the operation.

#### Failures and useful variations

- A null `GraphValue` component fails with `NullPointerException` and message `id` or
  `descriptor`.
- A null node component fails with `NullPointerException` and its component name. A null list
  element reports its zero-based position, such as `inputs[1]` or `outputs[2]`.
- An empty output list fails with `IllegalArgumentException` and message
  `outputs must not be empty`. A repeated output reports its later position and duplicate ID.
- `List.of()` is a valid input list, and repeated input IDs remain in their original positions.
  Mutating either list returned by the node fails with `UnsupportedOperationException`.
- An empty graph-output list is invalid. A zero-node graph is valid when its listed value is both
  an input and an output and its phase map is empty.
- A node cannot read its own or a later node's output. Every non-input value needs exactly one
  producer, and the phase map must cover exactly the listed nodes.

`GraphValue` and `CompiledNode` validation is deliberately local. `CompiledGraphModel` adds
graph-wide existence, ID uniqueness, producer, boundary, topology, and phase-coverage validation.
Operation arity, result count, descriptor agreement, compiler transformations, and execution
remain outside the model container.

## Planned contracts

The following contracts appear in the architecture and planning documents but are not implemented:

- mutable public `Tensor` state and `TensorFactory`;
- concrete operation kinds and family-specific attribute values;
- tensor provenance;
- compiler entry points and transformations, compiler-owned `PublicationPlan` and
  `CompileArtifacts`, and the engine `CompiledGraph` facade; and
- planning, prepare, runtime, publication execution, and backend execution.

`OperationKind`, `OperationAttrs`, `NoOperationAttrs`, `Operation`, `GraphValue`, and
`CompiledNode`, `GraphPhase`, `CompiledGraphModel`, and `PublicationBinding` are current Java API
contracts. No production concrete kind or family-specific attribute type exists yet. The graph
records can therefore compose and structurally validate test-local semantics, but they do not
provide a compiler entry point or executable support.

## Failures and ownership summary

- Null inputs are rejected where documented by each current method's Javadoc.
- Negative sizes, offsets, strides, and identifiers are rejected.
- Checked size, stride, and span arithmetic throws `ArithmeticException` on overflow.
- Current value objects are immutable and defensively copy caller-owned arrays where applicable.
- Operation-kind implementations must return a non-null, non-blank name, and operation-attribute implementations must preserve immutable value semantics; the marker interfaces do not enforce those obligations at runtime.
- `Operation` rejects a null kind or attributes value, retains both valid references unchanged, and does not validate family compatibility.
- `TensorDescriptor` rejects null components, resolved layouts incompatible with their paired
  shape, and gradient requests for non-differentiable data types. Layout reconstruction can report
  checked arithmetic overflow.
- `MemorySegmentStorage` rejects null inputs, negative capacity, checked byte-size overflow, an
  inexact segment byte size, and an initially dead scope in that order. It borrows the exact segment
  and owns no allocation or lifetime.
- `GraphValue` rejects null identity or descriptor components and stores no producer relationship.
- `CompiledNode` snapshots ordered lists, accepts empty or repeated inputs, and requires non-empty
  within-node-unique outputs. It does not perform graph-wide or operation-family validation.
- `CompiledGraphModel` snapshots all collections and validates graph-wide structural closure,
  topological order, declared boundaries, producer rules, and exact node-phase coverage. It
  accepts zero-input nodes, repeated node inputs, unused inputs, and zero-node pass-through graphs.
- `PublicationBinding` rejects null identities and remains a standalone association without
  owning-graph validation, policy, storage, backend, or runtime behavior.
- None of the current types owns device storage, runtime residency, or backend selection.

See generated Javadoc for the exact member-level exception and nullability contract.
