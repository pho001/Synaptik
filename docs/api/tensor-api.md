# Tensor API

## Purpose and mental model

This reference documents the public model contracts that are implemented today. The mutable
`Tensor` now connects stable logical metadata to an optional borrowed host-storage association,
and `TensorFactory` provides the public construction boundary for completed descriptors, copied
primitive-array import, independent dense constants including full-value and rectangular identity
tensors, deterministic population, and explicit-source normal, continuous-uniform, bounded
integral, and Bernoulli random population. The current concrete expression surface contains seven
floating tensor-to-tensor binary arithmetic methods, six floating tensor-to-tensor comparison
methods, three BOOL-only logical methods, fifteen floating unary elementwise methods, and five
floating scalar arithmetic and clamp methods, plus one static conditional-selection method and one
explicit cast method. Fifteen floating aggregate methods add full, one-axis, and retained-axis
`sum`, `mean`, `prod`, reduction `min`, and reduction `max` expression construction. Six BOOL
aggregate methods add the same three forms for `all` and `any`, and three axis-only `argMax`
methods add fixed INT64 index results. Two one-axis `cumSum` methods add shape-preserving numeric
scan expressions, and one-axis `softmax` and `logSoftmax` add shape-preserving floating
normalization expressions. The parameterless `contiguous` method adds a shape-preserving request
for canonical dense row-major result geometry. Two `reshape` overloads add ordered-element-
preserving coordinate changes from either raw `long...` dimensions or an exact normalized
`Shape`. Two `expand` overloads add directional right-aligned singleton and leading-axis
repetition with locally derived zero-stride view geometry when possible. `permute(int...)` adds
arbitrary complete axis reordering, and `transpose()` adds its rank-two `[1, 0]` convenience.
`expandDims(int)` inserts one singleton axis, while `squeeze(int)` removes one selected statically
known singleton axis. Typed access, other
expression families, gradient objects and publication behavior,
native/runtime/backend allocation, compiler integration, runtime residency, and backend execution
remain planned. The authoritative module boundary remains [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

The current semantic vocabulary also includes `ContiguousKind.CONTIGUOUS`, a parameterless
request for logically equivalent canonical dense row-major, zero-offset result geometry. Public
`Tensor.contiguous()` now constructs that request without allocating or copying storage.
`ShapeTransformKind.RESHAPE` and `ShapeTransformKind.EXPAND` are also current semantic identities.
Both pair with `TargetShapeAttrs`, which stores an exact normalized model `Shape`.
`Tensor.reshape(long...)` normalizes raw dimensions and performs locally provable count
validation, while `Tensor.reshape(Shape)` retains an exact normalized target. The matching
`Tensor.expand` overloads retain literal or exact target Shapes, validate directional
right-aligned compatibility, and derive logical view strides when geometry is resolved. Compiler,
materialization, gradient, and execution behavior remains planned.

`AxisTransformKind.PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE` are current semantic identities.
`PermutationAttrs` stores a complete normalized output-to-input axis permutation, while
`AxisTransformAttrs` stores one normalized non-negative insertion or removal position. Public
`Tensor.permute` now owns input-rank checks, one-time negative-axis normalization, result
Shape/layout derivation, and provenance; rank-two `Tensor.transpose()` uses the same `PERMUTE`
meaning with `[1, 0]`. Public `Tensor.expandDims` and `Tensor.squeeze` now own their distinct
insertion-position and existing-axis normalization, result Shape/layout derivation, static
singleton proof for removal, and provenance.

The current types describe logical values, construct storage-free arithmetic, comparison, boolean
logical, conditional-selection, unary, scalar, and value- or index-producing aggregate
expressions, and provide one bounded host-allocation path without executing a tensor:

```text
DataType + Shape + optional LayoutDescriptor + requiresGrad = TensorDescriptor
DataType + physical capacity + exact MemorySegment           = MemorySegmentStorage
TensorId + TensorDescriptor + label + optional provenance + optional host storage = Tensor
Operation + ordered input Tensor references                  = TensorProvenance
TensorFactory + completed descriptor + optional metadata     = public Tensor construction
TensorFactory + resolved descriptor                          = exact-span heap storage + Tensor
TensorFactory + dense descriptor + matching flat array       = copied populated heap Tensor
TensorFactory + rectangular nested primitive array           = inferred dense descriptor + copied Tensor
TensorFactory + scalar/shape/type or Tensor template          = independent dense constant Tensor
TensorFactory + static shape + exact primitive value          = independent dense full-value Tensor
TensorFactory + rows + columns + DataType                     = independent dense identity matrix
TensorFactory + integral bounds/step                          = copied dense INT32 or INT64 range Tensor
TensorFactory + static shape + typed flat source              = strict/cyclic copied dense prefix Tensor
TensorFactory + static shape/type + caller RandomGenerator    = copied dense normal/uniform Tensor
TensorFactory + static shape + primitive bounds + source      = copied dense INT32/INT64 random Tensor
TensorFactory + static shape + probability + source           = copied dense BOOL random Tensor
left Tensor + binary kind + right Tensor                       = fresh descriptor + provenance Tensor
left Tensor + comparison kind + right Tensor                   = fresh BOOL descriptor + provenance Tensor
left BOOL Tensor + AND/OR + right BOOL Tensor                   = fresh broadcast BOOL expression Tensor
BOOL Tensor + NOT                                               = fresh shape-preserving BOOL expression Tensor
BOOL condition + true/false floating branches                  = fresh broadcast selection Tensor
input Tensor + unary kind                                      = fresh descriptor + provenance Tensor
input Tensor + scalar kind + exact double attributes           = fresh descriptor + provenance Tensor
input Tensor + target DataType                                 = fresh explicit cast Tensor
floating Tensor + numeric aggregate kind + full/axis attributes = fresh reduced-shape Tensor
BOOL Tensor + ALL/ANY + full/axis attributes                    = fresh reduced-shape BOOL Tensor
numeric Tensor + ARG_MAX + axis/tie attributes                  = fresh reduced-shape INT64 Tensor
numeric Tensor + CUM_SUM + axis/mode attributes                 = fresh shape-preserving Tensor
floating Tensor + SOFTMAX/LOG_SOFTMAX + axis attributes        = fresh shape-preserving Tensor
Tensor + CONTIGUOUS                                = fresh static-resolved or dynamic-unresolved Tensor
Tensor + raw/exact target Shape + RESHAPE          = fresh conditional-view reshape Tensor
Tensor + raw/exact target Shape + EXPAND           = fresh conditional-view expand Tensor
Tensor + complete output-to-input axes + PERMUTE   = fresh conditional-view permute Tensor
Tensor + insertion axis + EXPAND_DIMS              = fresh conditional-view rank-expanded Tensor
Tensor + static singleton axis + SQUEEZE           = fresh conditional-view rank-reduced Tensor
TensorId / NodeId / ValueId                                  = distinct identity domains
Operation                                                     = OperationKind + OperationAttrs
BinaryArithmeticKind                                          = seven parameterless binary arithmetic semantics
BinaryComparisonKind                                          = six parameterless ordered comparison semantics
BooleanLogicalKind                                            = three parameterless boolean logical semantics
WhereSelectionKind                                            = one parameterless ternary conditional-selection semantic
CastKind + CastAttrs                                          = explicit cast identity + target DataType
AggregateReductionKind + reduction attributes                = full/axis aggregate semantics + arg-max tie policy
SUM/MEAN + MaskedReductionAttrs                              = masked axis semantics + ordered mask-axis mapping
CumulativeSumKind + CumulativeSumAttrs                       = shape-preserving cumulative-sum scan semantics
SoftmaxKind + SoftmaxAttrs                                   = shape-preserving probability normalization semantics
ContiguousKind                                               = parameterless canonical dense row-major geometry request
ShapeTransformKind + TargetShapeAttrs                        = reshape/expand meaning + normalized target Shape
AxisTransformKind + PermutationAttrs / AxisTransformAttrs    = axis reorder/insertion/removal semantics
UnaryElementwiseKind                                          = fifteen parameterless unary elementwise semantics
ScalarElementwiseKind                                         = five parameterized one-input scalar semantics
ScalarValueAttrs / ClampRangeAttrs                            = exact scalar parameters or ordered clamp bounds
ValueId + TensorDescriptor                                    = GraphValue
NodeId + Operation + ordered input/output ValueIds            = CompiledNode
values + nodes + boundaries + node phases                     = CompiledGraphModel
TensorId + ValueId                                             = PublicationBinding
```

An implemented `TensorDescriptor` keeps the logical element type, shape, explicit layout state,
and gradient eligibility together as one immutable value. It is still only a description. The
implemented `Tensor` retains that descriptor, a stable `TensorId`, and an optional label while
allowing only its borrowed host-storage association to change. A tensor may also retain immutable
optional `TensorProvenance`: the backend-independent operation that produced it and an ordered
immutable snapshot of the exact input-tensor references. Provenance is origin metadata for later
compiler capture, not graph membership or an intermediate-representation (IR) node. The
implemented `TensorFactory`
creates tensors from completed descriptors and assigns identity unique among factory allocations
within the current Java virtual machine (JVM). For a descriptor with resolved layout geometry, its
allocation overloads also create one matching primitive array whose length is the layout's
referenced element span and attach the resulting heap-segment storage. Its flat-array overloads
copy one matching primitive carrier into a resolved dense-contiguous tensor without retaining the
caller's array. Its nested-array method validates a rank-two-or-greater rectangular primitive-array
graph, infers the exact carrier type and fully static shape, flattens it in row-major order, and
delegates destination creation to that flat-import boundary. Its constant methods synthesize
canonical dense descriptors for rank-zero scalars or fully static caller/template shapes, then
reuse default-zero allocation or exact-carrier flat import. Its full-value methods fill one exact
typed carrier from a primitive scalar, and its identity method creates rank-two rectangular
matrices with typed one on the main diagonal and typed zero elsewhere.
Its deterministic population methods also create non-empty exclusive-end `INT32` and `INT64`
ranges or caller-shaped strict and cyclic prefixes from the same six exact primitive carriers.
Those methods synthesize only canonical dense descriptors, build one complete temporary carrier,
and delegate final storage population and identity assignment to flat import.
Normal, continuous-uniform, bounded integral, and Bernoulli random creation similarly build one
exact carrier, but consume a transient caller-owned `RandomGenerator` instead of retaining,
selecting, or seeding a source. Primitive `int` or `long` bounds infer `INT32` or `INT64`,
respectively; Bernoulli output is always BOOL; and all integral and boolean results disable
gradients. The methods remain explicit distribution-specific factory operations beside one
package-private helper because these cohesive sampling paths do not justify a public random
package, source abstraction, seed API, or distribution enum.

The implemented `HostTensorStorage` boundary describes a raw host-memory region. Its one
implementation, `MemorySegmentStorage`, borrows an exact JDK memory segment and records physical
capacity without allocating memory or relating that capacity to a tensor descriptor or layout.
`TensorFactory`, rather than the storage wrapper, supplies the implemented allocation policy for
automatic-scope primitive-array heap segments.

An `OperationKind` says which computation is meant, while `OperationAttrs` carries its typed
semantic parameters. The implemented `Operation` record keeps those two values together.
`BinaryArithmeticKind` names seven parameterless tensor-to-tensor arithmetic meanings, and
`BinaryComparisonKind` names six parameterless ordered tensor-to-tensor comparison meanings.
`BooleanLogicalKind` names parameterless boolean conjunction, disjunction, and negation meanings.
`WhereSelectionKind` names one parameterless elementwise conditional-selection meaning with
ordered condition, true-branch, and false-branch roles.
`CastKind` names one parameterized elementwise data-type conversion meaning, and `CastAttrs`
carries its exact non-null target `DataType`. The public `Tensor.cast` method composes that pair
with the receiver's source descriptor to create a fresh storage-free expression for every current
source/target combination, including a same-type request.
`AggregateReductionKind` names numeric sum, mean, product, minimum, and maximum reductions,
boolean all and any reductions, and index-producing arg-max. `AxisReductionAttrs` carries one
already normalized non-negative axis plus a retained-dimension choice for ordinary single-axis
forms. Ordinary full forms instead use `NoOperationAttrs.INSTANCE`, so no negative axis sentinel
means "all axes." `ArgMaxAttrs` carries the normalized axis, retained-dimension choice, and an
explicit `ArgMaxTiePolicy`. `MaskedReductionAttrs` carries a normalized reduction axis plus an
immutable ordered mapping from mask dimensions to input axes for masked, axis-removing `SUM` and
`MEAN`. These reduction semantics are implemented, and public Tensor reduction
methods currently cover `sum`, `mean`, `prod`, reduction `min`, reduction `max`, boolean `all`,
boolean `any`, and numeric `argMax`. Ordinary full forms produce a canonical rank-zero scalar;
all axis forms normalize the caller axis, then remove it or retain it with extent one. `argMax`
has no full form, accepts floating and integral inputs, and produces fixed INT64 results.
Masked `sum(axis, mask)` and `mean(axis, mask)` resolve their ordered Shape mapping locally,
remove the selected axis, and record exact `[input, mask]` provenance.
Numerical or truth evaluation, ordinary empty-domain behavior, gradients, and execution also
remain planned.
`CumulativeSumKind.CUM_SUM` and `CumulativeSumAttrs` provide the distinct shape-preserving scan
semantics. Public `Tensor.cumSum(axis)` selects inclusive forward traversal, while
`cumSum(axis, exclusive, reverse)` retains either caller-selected mode. Both accept floating and
integral input, normalize one axis, retain the exact input Shape, data type, and gradient
eligibility, leave layout unresolved, and record exact one-input provenance. They do not inspect
or accumulate values.
`SoftmaxKind.SOFTMAX`, `SoftmaxKind.LOG_SOFTMAX`, and `SoftmaxAttrs` provide distinct
shape-preserving probability and log-probability normalization semantics. The attributes carry
one already normalized non-negative axis. Public `Tensor.softmax` and `Tensor.logSoftmax`
accept floating input, normalize a positive or negative caller axis, preserve the exact Shape,
data type, and gradient-eligibility metadata in an unresolved descriptor, and record exact
one-input provenance. Construction calculates no probability or logarithm and defines no
numerical algorithm or gradient rule.
`ContiguousKind.CONTIGUOUS` provides the parameterless semantic request that one logical input
retain its values, Shape, DataType, and row-major element order while the result targets canonical
dense row-major geometry with logical storage offset zero. The kind is distinct from
`LayoutKind.DENSE_CONTIGUOUS`, which classifies geometry that has already been resolved for a
static Shape. Public `Tensor.contiguous()` constructs a fresh expression with exact Shape, data
type, and gradient eligibility. Fully static Shapes receive newly resolved canonical geometry;
dynamic Shapes remain unresolved. Construction does not inspect input layout or decide aliasing,
copying, or materialization.
`ShapeTransformKind.RESHAPE` and `ShapeTransformKind.EXPAND` provide distinct one-input
target-shape meanings. `RESHAPE` preserves the ordered logical element sequence while changing
the coordinates through which it is interpreted. `EXPAND` logically repeats compatible
singleton dimensions or adds leading dimensions. Both use `TargetShapeAttrs`, whose exact
non-null `Shape` is already normalized model semantics. The attributes therefore contain neither
raw public request syntax nor a numeric `-1` inference sentinel. Public `Tensor.reshape` owns
raw-request normalization, locally provable element-count validation, conditional result-layout
derivation, and one-input provenance. Public `Tensor.expand` owns directional right-aligned
compatibility, leading-axis and input-singleton expansion, and conditional zero-stride view
derivation. The semantic values themselves still do not perform that work. Neither current
expression family defines gradients, compiler behavior, materialization policy, backend behavior,
or execution.
Public `Tensor.permute` owns complete-permutation validation, raw negative-axis normalization,
exact Dimension-reference reordering, conditional resolved-stride reordering, and one-input
provenance. `Tensor.transpose()` supplies normalized axes `[1, 0]` after requiring rank two. These
methods construct logical model metadata only; compiler capture and canonicalization, planning and
prepare-time materialization, backend lowering, storage aliasing, gradients, and execution remain
outside the current Tensor contract.
`UnaryElementwiseKind` names fifteen parameterless unary arithmetic, transcendental, activation,
and explicit fast-approximation meanings. `ScalarElementwiseKind` names five parameterized
one-input meanings, with exact Java `double` parameters carried by `ScalarValueAttrs` or
`ClampRangeAttrs`. The public `Tensor.add`, `sub`, `mul`, `div`, `min`,
`max`, and tensor-valued `pow` methods use the binary kinds to construct storage-free expression
tensors. The public `Tensor.greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`,
and `notEqualTo` methods use the comparison kinds to construct storage-free `BOOL` expressions
from ordered floating inputs. The public `Tensor.abs`, `neg`, `inv`, `log`, `exp`, `erf`, `sqrt`,
`floor`, `ceil`, `sign`, `relu`, `sigmoid`, `tanh`, `fastExp`, and `fastTanh` methods use the unary
kinds to create storage-free expressions from one floating input. The public scalar overloads
`Tensor.mul(double)`, `pow(double)`, `clamp(double, double)`, `clampMin(double)`, and
`clampMax(double)` use the scalar kinds and typed attributes to create one-input storage-free
expressions while retaining every caller-supplied binary64 parameter bit. The public
`Tensor.logicalAnd`, `logicalOr`, and `logicalNot` methods use the logical kinds to construct
storage-free `BOOL` expressions from exact `BOOL` inputs. AND and OR derive a right-aligned
broadcast shape and preserve receiver/argument order; NOT retains the exact input `Shape`
reference without broadcasting. Every logical result has unresolved layout and false gradient
eligibility. Static `Tensor.where(condition, ifTrue, ifFalse)` requires an exact BOOL condition,
promotes the two floating branches, broadcasts the branches first and the condition with their
common shape second, and creates a fresh result whose gradient eligibility is the branch-only OR.
Its provenance preserves the exact ordered condition, true branch, and false branch references.
`Tensor.sum`, `mean`, `prod`, reduction `min`, and reduction `max` construct full or single-axis
floating aggregate expressions. They preserve the input type and gradient-eligibility request,
derive the reduced shape, leave layout unresolved, and record the receiver as their sole
provenance input. Product and extrema preserve eligibility metadata without claiming that a
gradient rule exists.
`Tensor.all` and `Tensor.any` construct full or single-axis boolean aggregate expressions from exact BOOL
input. They use the same shape and provenance rules, produce exact BOOL with false gradient
eligibility, and do not inspect truth values or define empty-domain identities. Aggregate ALL and
ANY remain typed separately from elementwise AND and OR.
`Tensor.argMax` constructs single-axis index-producing expressions from floating or integral
input. Its two convenience forms explicitly use `FIRST_INDEX`, while the complete form retains an
explicit first- or last-index policy. Every result is exact INT64 with false gradient eligibility;
construction does not compare values, select an index, or define empty-axis behavior.
`Tensor.cumSum` constructs single-axis cumulative-addition expressions from floating or integral
input. Its short form is inclusive and forward; its complete form retains explicit exclusive and
reverse choices. Every result preserves the exact input Shape, type, and gradient eligibility,
leaves layout unresolved, and records exactly the receiver as provenance without accumulating
values.
`Tensor.softmax` and `Tensor.logSoftmax` construct single-axis normalization expressions from
floating input. Both preserve the exact input Shape, type, and gradient eligibility, leave layout
unresolved, and record exactly the receiver as provenance. They retain distinct first-class
SOFTMAX or LOG_SOFTMAX semantics without calculating values or selecting a decomposition.
`Tensor.contiguous()` accepts every current data type and preserves the exact input Shape, type, and
gradient eligibility. A fully static result receives a newly constructed canonical dense
row-major layout; a dynamic result remains unresolved. Each call records `CONTIGUOUS` with
`NoOperationAttrs.INSTANCE` and exactly the receiver as provenance, without reading input layout,
storage, or values or performing a copy.
`TensorProvenance` retains the exact operation and ordered public-tensor inputs. All current
expression families derive result descriptors, but none assigns graph identity, captures
a graph, calculates values, defines gradient rules, or executes computation.

Comparison expression construction validates floating compatibility and local broadcasting, then
derives an unresolved non-differentiable `BOOL` descriptor and exact ordered provenance. It does
not define numerical comparison behavior for NaN, infinity, signed zero, tolerance, or mixed
floating precision. Compiler capture, gradients, backend lowering, and execution also remain
planned for their owning layers.

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

`DataTypePromotion.promoteFloating(left, right)` returns the widest input precision. Integral,
boolean, null, and cross-category inputs are rejected. The implemented `Tensor.cast` method can
represent an explicit target data type, but it constructs expression metadata rather than
converting values; promotion never inserts a cast implicitly.

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

## Public Tensor state

The public `Tensor` contract lives in `io.github.pho001.synaptik.model.tensor`. The current final
class is mutable API state, not an intermediate-representation node. It has five state components:

| Component | Stability and ownership |
|---|---|
| `id()` | Returns the exact immutable `TensorId` reference retained for the tensor's lifetime. |
| `descriptor()` | Returns the exact immutable `TensorDescriptor` reference retained for the tensor's lifetime. |
| `label()` | Returns an immutable optional diagnostic value; present text is stripped and must remain non-blank. |
| `provenance()` | Returns immutable optional expression-origin metadata; a present value is the exact retained `TensorProvenance`. |
| `hostStorage()` | Returns a synchronized snapshot of the optional borrowed `HostTensorStorage` association. |

Construction remains package-private so `Tensor` keeps one validation path. Code outside
`model.tensor` creates tensors through the public static `TensorFactory`: `create(descriptor)`
creates an unlabeled storage-free tensor, while the complete overload also accepts optional label
text and optional existing borrowed host storage. Both overloads retain the exact descriptor, and
the complete overload retains the exact compatible storage object. The factory does not construct
descriptors or choose or resolve layouts; these two creation overloads do not import values.

For a descriptor whose layout is resolved, `allocate(descriptor)` and
`allocate(descriptor, label)` allocate exactly `layout.referencedElementSpan()` elements. The
factory maps `FLOAT64` to `double[]`, `FLOAT32` to `float[]`, `BFLOAT16` to raw `short[]`, `INT32`
to `int[]`, `INT64` to `long[]`, and `BOOL` to raw `byte[]`. The new array begins with the JVM's
default all-zero raw representation. A span above `Integer.MAX_VALUE` is rejected before array
allocation, and an unresolved layout is rejected rather than inferred. These overloads do not
provide typed access, copy or conversion, boolean normalization, value import, fill operations,
native allocation, or descriptor synthesis. The separate constant methods deliberately synthesize
only canonical dense descriptors and reuse this allocation path for zeros.

Flat typed import is the current value-population boundary. `fromFlatArray(descriptor, label,
source)` has exactly six overloads and maps carriers without conversion:

| Source carrier | Required data type | Imported meaning |
|---|---|---|
| `double[]` | `FLOAT64` | IEEE 754 binary64 values copied bit-for-bit |
| `float[]` | `FLOAT32` | IEEE 754 binary32 values copied bit-for-bit |
| `short[]` | `BFLOAT16` | Raw BFLOAT16 bit patterns copied unchanged |
| `int[]` | `INT32` | Signed 32-bit integer values |
| `long[]` | `INT64` | Signed 64-bit integer values |
| `byte[]` | `BOOL` | Zero becomes `0`; every non-zero value becomes canonical `1` |

The descriptor must have the exact mapped data type and a resolved `DENSE_CONTIGUOUS` layout.
Its known logical element count must equal the Java array length. The factory treats source order
as logical row-major order, allocates a new matching heap array through the existing allocation
path, and completely copies or normalizes the values before returning the tensor. The source
array is neither retained nor mutated, so changing it after return cannot change the tensor.
Rank-0 scalars require one source element, and empty dense tensors require an empty source array.

Data-type, unresolved-layout, non-dense-layout, and length failures happen before destination or
identifier allocation. A blank label reaches the existing allocation and Tensor-construction
path, so destination storage and an identifier are allocated before the label fails; no copy is
performed. Identifier exhaustion is also observed after destination allocation and before copy.
Unexpected copy failures occur after identifier allocation and are not rolled back. Offset-dense,
strided, broadcast, and unresolved descriptors remain unsupported because importing independent
row-major values into those geometries requires a separate scatter or view policy.

Nested typed import is the descriptor-inference boundary. The single
`fromNestedArray(source, label, requiresGrad)` method accepts `Object` because Java gives each
primitive-array rank a distinct runtime class and no finite overload family can express arbitrary
rank. The runtime class must prove rank two or greater and an ultimate carrier of `double`,
`float`, `short`, `int`, `long`, or `byte`; those carriers infer `FLOAT64`, `FLOAT32`, `BFLOAT16`,
`INT32`, `INT64`, and `BOOL`, respectively. Boxed arrays, generic object arrays, rank-one arrays,
and every other primitive carrier are rejected rather than converted or defaulted.

The complete reachable structure is validated in depth-first row-major order before the
intermediate flat carrier, destination, or tensor ID is allocated. Every corresponding axis must
have the same length, and every required subarray must be non-null. Paths in diagnostics use
zero-based bracket notation: `[]` names the root and `[1][2]` names its second child's third child.
An empty final primitive axis is observable and valid, so `new int[2][0]` infers shape `[2, 0]`.
An empty earlier axis is rejected because the remaining extents cannot be recovered from the
runtime object graph.

After validation, nested import allocates a fresh matching primitive array and copies leaf arrays
into it in logical row-major order. Numeric values and raw BFLOAT16 shorts are unchanged. BOOL
bytes remain raw in that intermediate array and are normalized once by flat import while copying
to destination storage. The inferred `TensorDescriptor` has the exact data type, fully static
shape, canonical dense-contiguous layout, and requested `requiresGrad` flag; descriptor validation
remains authoritative for gradient eligibility. No source level is retained or mutated, and later
caller mutation cannot change the tensor. Callers must not mutate the source concurrently with
import because inspection and flattening do not provide an atomic deep snapshot.

Structural, carrier, checked-count, Java-array-limit, and descriptor-eligibility failures consume
no ID and allocate no destination. Descriptor eligibility is checked after the intermediate array
is allocated. A blank label reaches flat import, so destination storage is allocated and one ID is
consumed before Tensor validation fails. Identifier exhaustion is likewise observed after
destination allocation and before the final flat-to-storage copy. Neither failure exposes or
retains the intermediate carrier.

Constant creation is the factory's other descriptor-synthesis boundary. Its six scalar methods
infer an exact data type from the declared primitive input and create canonical rank-0 dense
tensors:

| Method input | Result data type | Stored representation |
|---|---|---|
| `double` | `FLOAT64` | Exact binary64 bits |
| `float` | `FLOAT32` | Exact binary32 bits |
| `scalarBFloat16(float)` | `BFLOAT16` | `BFloat16Bits.fromFloat(value)` |
| `int` | `INT32` | Exact signed 32-bit value |
| `long` | `INT64` | Exact signed 64-bit value |
| `boolean` | `BOOL` | Canonical byte `0` or `1` |

The BFLOAT16 method is named explicitly because it rounds a semantic binary32 input to BFLOAT16
using round-to-nearest with ties to even; it is not the FLOAT32 overload and does not accept raw
`short` bits. Every scalar descriptor uses `Shape.scalar()`, a resolved canonical
dense-contiguous layout, exactly one logical element, and the caller's explicit `requiresGrad`
request.

`zeros(shape, dataType, label, requiresGrad)` and
`ones(shape, dataType, label, requiresGrad)` accept every current data type. The shape must be
fully static, but it may be rank zero or contain a zero-sized dimension. Both methods synthesize a
new canonical dense descriptor. Zeros allocate the destination once and rely on JVM primitive
array initialization, so no source array or element fill is performed. Ones fill one exact typed
source carrier with `1.0d`, `1.0f`, BFLOAT16 bits `0x3F80`, `1`, `1L`, or BOOL byte `1`, then copy
through the matching flat-import overload. An empty shape therefore creates empty storage rather
than one physical element.

`zerosLike(template, label, requiresGrad)` and
`onesLike(template, label, requiresGrad)` read only the template descriptor's shape and data type.
They do not inspect or preserve the template layout, label, storage, storage liveness, identity,
gradient request, or other state. Static unresolved, offset, strided, broadcast, and dense
templates are all accepted; the result always has a newly synthesized canonical dense layout and
uses the explicit result label and gradient request. Each successful call creates a new Tensor,
descriptor, layout, storage wrapper, backing array, and ID. The template object and any temporary
source carrier are not retained.

Public null checks happen before descriptor or storage work. Constant descriptor validation then
requires a fully static shape, calculates its checked logical element count, applies the Java
array limit, constructs dense layout geometry, and validates gradient eligibility in that order.
Those failures allocate no destination and consume no ID. Source-carrier allocation for scalars
and ones occurs only after descriptor validation. A blank label fails later, after destination and
ID allocation, and consumes that ID; scalar and one creation have also allocated their source
carrier. Identifier exhaustion is observed after destination allocation, with the same source
carrier distinction. A JVM allocation failure that occurs before ID allocation consumes no ID.

Full-value creation extends the same dense constant path without adding a generic conversion or
mutable fill operation. The six type-safe entries are:

| Factory value | Result data type | Stored value |
|---|---|---|
| `double` | `FLOAT64` | Exact binary64 value, including signed-zero and NaN payload bits |
| `float` | `FLOAT32` | Exact binary32 value, including signed-zero and NaN payload bits |
| `fullBFloat16(..., float, ...)` | `BFLOAT16` | One `BFloat16Bits.fromFloat(value)` conversion repeated as raw bits |
| `int` | `INT32` | Exact signed 32-bit value |
| `long` | `INT64` | Exact signed 64-bit value |
| `boolean` | `BOOL` | Canonical false byte `0` or true byte `1` |

Every `full` method requires a fully static caller shape, but rank-zero scalar and zero-element
shapes are valid. It constructs one canonical dense descriptor, fills one exact primitive source
carrier, and delegates once to matching flat import. The source is copied and not retained. The
result has independent descriptor, layout, destination storage, Tensor identity, and empty
provenance. The API deliberately has no boxed value, caller-selected conversion, raw BFLOAT16
scalar, `fullLike`, default-label overload, or post-construction mutation.

`identityMatrix(rows, columns, dataType, label, requiresGrad)` creates a canonical dense rank-two
matrix for any of the six current data types. Both dimensions are non-negative `long` values, so
square, wide, tall, zero-row, and zero-column matrices are valid when their checked logical count
fits a Java array. The method writes typed one only at coordinates `(i, i)` for
`0 <= i < min(rows, columns)` and leaves every other position at typed zero. Its exact values are
`1.0d`/`0.0d`, `1.0f`/`0.0f`, converted BFLOAT16 one bits/raw zero bits, `1`/`0`, `1L`/`0L`, or
canonical BOOL `1`/`0`. `eye(...)` is only an unchanged-argument call to `identityMatrix(...)`.
Equal arguments therefore produce equal descriptors and values, while separate successful calls
still return different Tensor, ID, descriptor, layout, storage, and backing-array objects.

Full-value methods check shape then label nullity before static-shape, checked-count, Java array
limit, dense-layout, and gradient validation. Identity checks data type then label nullity,
negative rows then negative columns, and then uses the same checked descriptor path. These early
failures allocate no carrier and consume no ID. Each successful path creates one source carrier,
then flat import creates one destination and allocates the ID. A blank label therefore fails after
both arrays and the ID exist and consumes that ID. Exhaustion occurs after both arrays exist, and
an unexpected flat-copy failure occurs after ID allocation; identifiers are never rolled back.

### Complete constant-creation example

#### Goal and inputs

Create a BFLOAT16 scalar from a halfway binary32 value, create a `2 × 2` INT32 one tensor, and
create an independent zero tensor using only the one's shape and data type. The scalar input
`1.00390625f` is halfway between adjacent BFLOAT16 values and rounds to the even representation
`0x3F80` for `1.0`.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class ConstantCreationExample {
    public static void main(String[] args) {
        Tensor scalar = TensorFactory.scalarBFloat16(
                1.00390625f, Optional.of("scale"), true);
        Tensor ones = TensorFactory.ones(
                Shape.of(2, 2), DataType.INT32, Optional.of("ones"), false);
        Tensor zeros = TensorFactory.zerosLike(
                ones, Optional.of("zeros"), false);

        short[] scalarData = (short[]) scalar.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();
        int[] oneData = (int[]) ones.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.printf("%04X%n", scalarData[0] & 0xFFFF);
        System.out.println(Arrays.toString(oneData));
        System.out.println(zeros.descriptor().shape());
        System.out.println(zeros.descriptor().layout().orElseThrow().kind());
        System.out.println(ones.hostStorage().orElseThrow()
                != zeros.hostStorage().orElseThrow());
    }
}
```

#### Meaningful lines

- `scalarBFloat16` makes the rounding request explicit and creates a differentiable rank-0
  descriptor before copying the converted raw bits into independent storage.
- `ones` selects the `int[]` carrier from `INT32` and fills its four logical positions with exact
  integer one.
- `zerosLike` reuses only `ones`' immutable shape and data type. Its supplied label and gradient
  request are independent, and it creates a new dense descriptor and storage.
- Heap-base inspection exposes the already implemented raw storage only so the values are visible
  in this example; it is not a typed Tensor access or export API.

#### Result and interpretation

The program prints:

```text
3F80
[1, 1, 1, 1]
Shape[2, 2]
DENSE_CONTIGUOUS
true
```

The result demonstrates BFLOAT16 ties-to-even rounding, exact INT32 one population, dense
like-shaped creation, and independent storage. It does not provide general fill values, numeric
conversion, random generation, typed access/export, view preservation, or execution; deterministic
range and prefix creation are separate factory methods described below.

#### Failures and useful variations

- Requesting gradients for `INT32`, `INT64`, or `BOOL` fails before source or destination
  allocation.
- A dynamic shape or logical element count above `Integer.MAX_VALUE` fails before destination or
  ID allocation.
- A zero-sized static shape is valid and produces an empty backing array for either zeros or ones.

### Complete full-value and rectangular-identity example

#### Goal and inputs

Create a nontrivial `2 × 3` INT64 tensor whose six logical positions all contain `-7`, and create
a wide `2 × 4` FLOAT32 identity matrix. Both results use explicit labels and gradient intent. The
identity is differentiable because FLOAT32 is differentiable; the INT64 full tensor must disable
gradients.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class FullIdentityExample {
    public static void main(String[] args) {
        Tensor full = TensorFactory.full(
                Shape.of(2, 3), -7L, Optional.of("weights"), false);
        Tensor identity = TensorFactory.identityMatrix(
                2, 4, DataType.FLOAT32, Optional.of("projection"), true);

        long[] fullValues = (long[]) full.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();
        float[] identityValues = (float[]) identity.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(Arrays.toString(fullValues));
        System.out.println(Arrays.toString(identityValues));
        System.out.println(identity.descriptor().shape());
        System.out.println(identity.descriptor().layout().orElseThrow().kind());
        System.out.println(full.provenance().isEmpty() && identity.provenance().isEmpty());
    }
}
```

#### Meaningful lines and intermediate values

- `full(Shape.of(2, 3), -7L, ...)` selects INT64 from the Java `long` argument. Row-major
  positions `0` through `5` all become `-7`, which can be read as two logical rows
  `[-7, -7, -7]` and `[-7, -7, -7]`.
- `identityMatrix(2, 4, FLOAT32, ...)` creates a wide matrix. Its diagonal length is
  `min(2, 4) = 2`, so row-major indices `0 × 4 + 0 = 0` and `1 × 4 + 1 = 5` receive `1.0f`.
  The other six positions retain `0.0f`.
- `heapBase()` inspection is used only to make the current raw values observable. It is not a
  typed Tensor access or export API.
- The descriptor queries verify that identity creation retained the rectangular shape and
  synthesized canonical dense-contiguous geometry. The final expression verifies that both
  public initializers produced leaves without provenance.

#### Result and interpretation

The program prints:

```text
[-7, -7, -7, -7, -7, -7]
[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
Shape[2, 4]
DENSE_CONTIGUOUS
true
```

The first line is the complete copied INT64 carrier in row-major order. The second line represents
the rectangular matrix rows `[1, 0, 0, 0]` and `[0, 1, 0, 0]`. The result proves exact full-value
population, main-diagonal placement, dense metadata, and provenance-free leaf construction. It
does not add post-construction fill, typed Tensor export, an identity operation, graph capture, or
backend execution.

#### Failures and useful variations

- Replacing `Shape.of(2, 3)` with a dynamic shape fails before source, destination, or ID
  allocation. A static zero-element shape is valid and creates empty storage.
- A negative row is rejected before a negative column; zero rows or zero columns create a valid
  empty rank-two identity matrix.
- Requesting gradients for an INT32, INT64, or BOOL full or identity tensor fails before
  allocation. `eye(...)` has the same validation and values as `identityMatrix(...)`, but a
  separate call still returns a fresh Tensor and ID.

Deterministic range creation has exactly two overloads. `range(int, int, int, label)` creates an
`INT32` result, while `range(long, long, long, label)` creates `INT64`. Both produce eager,
rank-one, non-differentiable tensors with inclusive start and exclusive end. A positive step must
advance from a smaller start toward a larger end; a negative step must advance from a larger start
toward a smaller end. Step zero, equal bounds, and the wrong direction are rejected rather than
producing an empty tensor or correcting the step. Exact count arithmetic prevents primitive
overflow, and the final emitted value may be less than one step from the exclusive bound. Counts
above `Integer.MAX_VALUE` are rejected before allocation.

Strict and cyclic flat-prefix creation each have six overloads for `double[]`, `float[]`, raw
BFLOAT16 `short[]`, `int[]`, `long[]`, and BOOL `byte[]`. The carrier-to-data-type mapping is the
same exact mapping as flat import; no conversion, promotion, or caller-selected data type is
available. The caller supplies a fully static shape, optional label, and explicit gradient intent.
The result always has a newly synthesized canonical dense-contiguous descriptor and independent
heap storage.

A strict prefix source must contain at least the shape's logical element count. The factory copies
exactly that many leading values into a fresh carrier and ignores any tail. A cyclic prefix copies
`source[i % source.length]` at output position `i`; when the source is longer than the output, this
is exactly its requested prefix. A non-empty cyclic result requires a non-empty source. Both modes
accept an empty source for a zero-element result because no value is needed, and neither mode
retains or mutates the caller's array. Numeric values and raw BFLOAT16 bits remain unchanged. BOOL
bytes are normalized downstream by flat import, where zero becomes `0` and every non-zero value
becomes `1`.

Prefix null checks run in shape, label, source order. Dynamic shape, checked-count overflow,
Java-array limit, strict source sufficiency or cyclic source availability, dense descriptor
construction, and gradient eligibility are validated before the result carrier, destination, or
ID is allocated. Range label and argument validation likewise precede allocation. Successful
paths build one complete temporary carrier and delegate once to matching flat import. A blank
label therefore fails after the carrier, destination, and ID have been allocated but before the
flat copy; it consumes the ID. Identifier exhaustion is also observed after both arrays exist and
before copying. Neither failure rolls back an identifier. Concurrent source mutation is outside
the prefix snapshot contract.

### Complete deterministic-population example

#### Goal and inputs

Create the INT32 range `[1, 4, 7]`, copy the first four values of a five-value strict source into
shape `[2, 2]`, and repeat a two-byte logical cycle across shape `[2, 3]`. The strict tail value
`99` is deliberately outside the requested result, while cyclic non-zero byte `-3` represents
logical true.

```java
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class DeterministicPopulationExample {
    public static void main(String[] args) {
        Tensor range = TensorFactory.range(
                1, 8, 3, Optional.of("indices"));

        int[] strictSource = {10, 20, 30, 40, 99};
        Tensor strict = TensorFactory.fromStrictFlatPrefix(
                Shape.of(2, 2), Optional.empty(), false, strictSource);

        byte[] cycle = {0, -3};
        Tensor cyclic = TensorFactory.fromCyclicFlatPrefix(
                Shape.of(2, 3), Optional.of("mask"), false, cycle);

        int[] rangeData = (int[]) range.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();
        int[] strictData = (int[]) strict.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();
        byte[] cyclicData = (byte[]) cyclic.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        strictSource[0] = -1;
        cycle[0] = 1;

        System.out.println(range.descriptor().dataType());
        System.out.println(Arrays.toString(rangeData));
        System.out.println(strict.descriptor().shape());
        System.out.println(Arrays.toString(strictData));
        System.out.println(Arrays.toString(cyclicData));
    }
}
```

#### Meaningful lines

- The `int` range overload fixes the result data type as `INT32`. Ceiling division of the distance
  by step gives three values, and the exclusive bound `8` is not emitted.
- The strict call obtains four logical positions from shape `[2, 2]`, copies source values
  `[10, 20, 30, 40]`, and ignores `99`.
- The cyclic call repeats raw bytes as `[0, -3, 0, -3, 0, -3]`; delegated BOOL import then stores
  canonical `[0, 1, 0, 1, 0, 1]`.
- Mutating both sources after construction demonstrates that neither returned tensor retains the
  caller's array. Heap-base inspection only makes existing raw storage observable; it is not a
  typed Tensor access or export API.

#### Result and interpretation

The program prints:

```text
INT32
[1, 4, 7]
Shape[2, 2]
[10, 20, 30, 40]
[0, 1, 0, 1, 0, 1]
```

The result demonstrates range typing and exclusive-end semantics, strict tail handling, cyclic
repetition, BOOL normalization, canonical dense shapes, and copied ownership. It does not provide
an empty or floating range, implicit conversion, general fill/repeat/tile operations, view
population, typed access/export, provenance, or execution. Normal random creation is the separate
explicit-source factory method described next.

#### Failures and useful variations

- `range(0, 3, -1, label)` fails because the negative step cannot advance toward the larger end.
- A strict source with only three values fails for shape `[2, 2]`; a longer source remains valid.
- An empty cyclic source is valid for shape `[0, 3]` and invalid for every non-empty shape.
- `requiresGrad` must be false for `INT32`, `INT64`, and `BOOL`; floating prefix carriers may
  request gradients.

Normal random creation has exactly one public method:
`randomNormal(shape, dataType, mean, standardDeviation, randomGenerator, label, requiresGrad)`.
It accepts only fully static Java-array-sized shapes and `FLOAT64`, `FLOAT32`, or `BFLOAT16`.
Both parameters must be finite, and the standard deviation must be numerically non-negative;
either signed zero is valid. The result has a newly synthesized canonical dense-contiguous
descriptor, independent writable heap storage, the explicit label and gradient intent, and a new
factory ID.

For every logical row-major element, the method calls the exact supplied
`RandomGenerator.nextGaussian()` once, multiplies the returned binary64 value by the standard
deviation, and then adds the binary64 mean. It does not use fused multiply-add. `FLOAT64` stores
the result directly, `FLOAT32` narrows it once to binary32, and `BFLOAT16` first narrows to
binary32 and then uses `BFloat16Bits.fromFloat`. A scalar consumes one call; an empty shape
consumes none. Generated non-finite values are stored according to the requested conversion and
are not rejected after sampling.

The generator is transient caller-owned state. The factory does not select an algorithm or
default source, store a seed, retain or replace the generator, synchronize access, or reset,
split, or close it. Equivalent output therefore requires equivalent generator implementation and
initial state, identical arguments, and no interfering source use. There is no sequence promise
across algorithms, providers, Java versions, seed expansion, concurrent access, or different
initial states.

### Complete normal-random example

#### Goal and inputs

Create a `2 × 2` FLOAT32 tensor from the scripted Gaussian sequence `[-1, 0, 1, 0.5]`, mean `1`,
and standard deviation `2`. A scripted source makes the transformation and exact call count
observable without introducing a production seed or generator-selection API.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;
import java.util.random.RandomGenerator;

public final class NormalRandomExample {
    private static final class ScriptedGenerator implements RandomGenerator {
        private final double[] values = {-1.0, 0.0, 1.0, 0.5};
        private int calls;

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong is not used");
        }

        @Override
        public double nextGaussian() {
            return values[calls++];
        }
    }

    public static void main(String[] args) {
        ScriptedGenerator source = new ScriptedGenerator();
        Tensor tensor = TensorFactory.randomNormal(
                Shape.of(2, 2),
                DataType.FLOAT32,
                1.0,
                2.0,
                source,
                Optional.of("samples"),
                true);

        float[] data = (float[]) tensor.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(Arrays.toString(data));
        System.out.println(source.calls);
        System.out.println(tensor.descriptor().layout().orElseThrow().kind());
    }
}
```

#### Meaningful lines

- The source implements only the methods needed by `RandomGenerator` and overrides
  `nextGaussian()` with four known values. TensorFactory uses that exact object and does not call
  `nextLong()` or select another generator.
- Each output applies `1 + gaussian × 2` in binary64 before narrowing to the requested FLOAT32
  carrier. The four transformed values are `-1`, `1`, `3`, and `2` in row-major order.
- Heap-base inspection makes the already implemented raw storage observable for the example; it
  is not a typed Tensor access or export API.

#### Result and interpretation

The program prints:

```text
[-1.0, 1.0, 3.0, 2.0]
4
DENSE_CONTIGUOUS
```

The result proves transformation order, one source call per element, FLOAT32 conversion, and
canonical dense layout for this scripted input. It does not promise statistical properties of a
custom generator, cross-generator reproducibility, a Synaptik seed API, continuous-uniform,
integral, or Bernoulli behavior, typed export, or runtime/backend random execution.

#### Failures and useful variations

- A dynamic shape, non-floating data type, non-finite mean, negative or non-finite deviation, or
  logical count above `Integer.MAX_VALUE` fails before sampling or ID allocation.
- If `nextGaussian()` throws, preceding calls remain consumed, but no destination or ID exists.
- A blank present label is detected by delegated Tensor validation after all samples, destination
  allocation, and ID allocation; the ID is consumed and no state is rolled back.
- Two equivalent caller-created generators can produce equivalent tensors when their initial
  states and all method arguments match and neither source has interfering use.

Continuous-uniform random creation also has exactly one public method:
`randomUniform(shape, dataType, lowerBoundInclusive, upperBoundExclusive, randomGenerator, label,
requiresGrad)`. It accepts only fully static Java-array-sized shapes and `FLOAT64`, `FLOAT32`, or
`BFLOAT16`. Both binary64 bounds must be finite, and the lower bound must be strictly less than the
upper bound. The result has a newly synthesized canonical dense-contiguous descriptor, independent
writable heap storage, the explicit label and gradient intent, and a new factory ID.

For every logical row-major element, the method calls the exact supplied
`RandomGenerator.nextDouble(lowerBoundInclusive, upperBoundExclusive)` once. A conforming source
returns a binary64 value in the half-open interval `[lowerBoundInclusive, upperBoundExclusive)`.
`FLOAT64` stores that value directly, `FLOAT32` narrows it once to binary32, and `BFLOAT16` first
narrows to binary32 and then uses `BFloat16Bits.fromFloat`. The half-open promise applies before
narrowing: a stored FLOAT32 or BFLOAT16 value may equal the corresponding narrowed upper bound or
may round to a lower representable value. The factory does not clamp, resample, or post-validate a
custom non-conforming source result. A scalar consumes one bounded call; an empty shape consumes
none.

The source has the same ownership and reproducibility boundary as normal creation. The caller
creates, configures, seeds, owns, and advances it, and must provide exclusive or otherwise safe
access. The factory does not select an algorithm or default source, store a seed, retain or replace
the generator, synchronize access, or reset, split, or close it. Equivalent output requires an
equivalent generator implementation and initial state, identical arguments, and no interfering
source use. No sequence is promised across algorithms, providers, Java versions, seed expansion,
concurrent use, or different initial states.

### Complete continuous-uniform example

#### Goal and inputs

Create a `2 × 2` FLOAT32 tensor from four known binary64 values in `[-1, 1)`. The final source
value is `Math.nextDown(1.0)`, which is strictly below the binary64 upper bound but rounds to
binary32 `1.0f`. A scripted source makes the exact bounded-call arguments, call count, and
narrowing caveat observable without adding a production seed or source-selection API.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;
import java.util.random.RandomGenerator;

public final class UniformRandomExample {
    private static final class ScriptedGenerator implements RandomGenerator {
        private final double[] values = {-1.0, 0.0, 0.5, Math.nextDown(1.0)};
        private int calls;

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong is not used");
        }

        @Override
        public double nextDouble() {
            throw new AssertionError("unbounded nextDouble is not used");
        }

        @Override
        public double nextDouble(double origin, double bound) {
            if (origin != -1.0 || bound != 1.0) {
                throw new AssertionError("unexpected bounds");
            }
            return values[calls++];
        }
    }

    public static void main(String[] args) {
        ScriptedGenerator source = new ScriptedGenerator();
        Tensor tensor = TensorFactory.randomUniform(
                Shape.of(2, 2),
                DataType.FLOAT32,
                -1.0,
                1.0,
                source,
                Optional.of("samples"),
                true);

        float[] data = (float[]) tensor.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(Arrays.toString(data));
        System.out.println(source.calls);
        System.out.println(tensor.descriptor().layout().orElseThrow().kind());
    }
}
```

#### Meaningful lines

- The source overrides only the bounded `nextDouble(origin, bound)` path used by the factory and
  rejects both `nextLong()` and unbounded `nextDouble()`. It also verifies the unchanged `-1.0`
  and `1.0` arguments on every call.
- The first three binary64 values remain `-1.0f`, `0.0f`, and `0.5f`. The final conforming value is
  below binary64 `1.0`, but one binary32 narrowing rounds it to stored `1.0f`.
- Heap-base inspection makes the already implemented raw storage observable for the example; it
  is not a typed Tensor access or export API.

#### Result and interpretation

The program prints:

```text
[-1.0, 0.0, 0.5, 1.0]
4
DENSE_CONTIGUOUS
```

The result proves the exact bounded-call path, one call per row-major element, FLOAT32 narrowing,
the narrowed-upper-bound caveat, and canonical dense layout for this scripted input. It does not
promise statistical quality of a custom generator, a closed binary64 interval, cross-generator
reproducibility, a Synaptik seed API, integral or Bernoulli sampling, typed export, or
runtime/backend random execution.

#### Failures and useful variations

- A dynamic shape, non-floating data type, non-finite bound, unordered bounds, or logical count
  above `Integer.MAX_VALUE` fails before sampling or ID allocation.
- If bounded `nextDouble(lower, upper)` throws, preceding calls remain consumed, but no destination
  or ID exists.
- A blank present label is detected by delegated Tensor validation after all samples, destination
  allocation, and ID allocation; the ID is consumed and no state is rolled back.
- Two equivalent caller-created generators can produce equivalent tensors when their initial
  states and all method arguments match and neither source has interfering use.

Bounded integral random creation has exactly two public overloads named `randomInt`. Primitive
`int` bounds produce `INT32`, while primitive `long` bounds produce `INT64`. Both take a fully
static Java-array-sized shape, inclusive origin, exclusive bound, caller-owned `RandomGenerator`,
and optional label. The origin must be strictly less than the bound, including for an empty
result. The synthesized descriptor is canonical dense-contiguous and always has
`requiresGrad == false`; there is no data-type or gradient parameter because the bound carrier
already selects a non-differentiable result type.

Each logical row-major element is the direct result of exactly one matching bounded call:
`nextInt(origin, bound)` for `INT32` or `nextLong(origin, bound)` for `INT64`. The factory uses no
unbounded draw, modulo reduction, floating arithmetic, narrowing, widening, or alternate carrier.
It fills one exact `int[]` or `long[]` and delegates once to matching flat import. A conforming
source supplies values in the half-open interval `[origin, bound)` without project-owned modulo
bias; custom non-conforming values are copied without post-validation. A scalar consumes one
bounded call and a valid empty result consumes none.

The source has the same ownership and bounded reproducibility contract as the floating random
methods. The caller configures, seeds, owns, advances, and provides safe access to it. The factory
does not substitute, retain, synchronize, reset, split, or close the source. Equivalent output
requires equivalent generator implementation and initial state, identical arguments, and no
interfering source use. The one-call guarantee concerns bounded method invocations rather than the
generator's internal random-bit consumption.

The bounds use the result carrier itself. `Integer.MAX_VALUE` or `Long.MAX_VALUE` can therefore be
an exclusive bound but cannot be emitted, and the mathematical exclusive bound one greater than
the carrier maximum cannot be expressed. No unbounded or full-domain convenience is supplied.
Negative and mixed-sign intervals are supported when strictly ordered.

### Complete bounded-integral example

#### Goal and inputs

Create a `2 × 2` INT32 tensor from the scripted values `[-3, -1, 0, 4]` over `[-3, 5)`. The
scripted source verifies that every call receives the exact bounds and that neither an unbounded
draw nor a production seed/source-selection API is involved.

```java
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;
import java.util.random.RandomGenerator;

public final class IntegralRandomExample {
    private static final class ScriptedGenerator implements RandomGenerator {
        private final int[] values = {-3, -1, 0, 4};
        private int calls;

        @Override
        public long nextLong() {
            throw new AssertionError("unbounded nextLong is not used");
        }

        @Override
        public int nextInt(int origin, int bound) {
            if (origin != -3 || bound != 5) {
                throw new AssertionError("unexpected bounds");
            }
            return values[calls++];
        }
    }

    public static void main(String[] args) {
        ScriptedGenerator source = new ScriptedGenerator();
        Tensor tensor = TensorFactory.randomInt(
                Shape.of(2, 2), -3, 5, source, Optional.of("indices"));

        int[] data = (int[]) tensor.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(Arrays.toString(data));
        System.out.println(tensor.descriptor().dataType());
        System.out.println(tensor.descriptor().requiresGrad());
        System.out.println(source.calls);
    }
}
```

#### Meaningful lines

- Primitive `int` bounds select `INT32`; the call has no `DataType` or gradient argument.
- The source rejects unbounded `nextLong()` and validates `-3` and `5` on every bounded
  `nextInt(origin, bound)` invocation. The factory stores each returned value directly.
- Heap-base inspection makes the implemented raw storage observable for the example; it is not a
  typed Tensor access or export API.

#### Result and interpretation

The program prints:

```text
[-3, -1, 0, 4]
INT32
false
4
```

The result proves inferred INT32 type, false gradient intent, exact bounded-call order and count,
direct carrier storage, and half-open boundary values for this scripted input. The primitive
`long` overload follows the same contract with `nextLong(origin, bound)` and an `INT64` result. The
example does not promise statistical quality for a custom generator, full-domain sampling, typed
export, a random graph Operation, or runtime/backend generation.

#### Failures and useful variations

- A dynamic shape or logical count above `Integer.MAX_VALUE` fails before sampling or ID
  allocation. Checked element-count overflow remains an `ArithmeticException`.
- Equal or reversed bounds fail even for an empty result, before source-carrier allocation.
- If the bounded source method throws, earlier calls remain consumed, but no destination or ID
  exists.
- A blank label fails after all bounded calls, destination allocation, and ID allocation; the ID
  is consumed and no state is rolled back.

Bernoulli random creation has exactly one public method:
`randomBernoulli(shape, probability, randomGenerator, label)`. It creates a canonical dense BOOL
tensor with gradients disabled. The shape must be fully static, its checked logical count must fit
a Java array, and the binary64 probability must be finite and in the closed interval `[0, 1]`.
Positive and negative zero are accepted as zero. No data-type or gradient argument is present
because Bernoulli output is logical non-differentiable leaf data; numeric truthiness and output
conversion are outside this method.

For each logical row-major element, the factory calls the exact supplied source's unbounded
`nextDouble()` method once and stores canonical byte `1` exactly when `draw < probability`;
otherwise it stores byte `0`. The strict comparison means that a draw equal to the probability is
false. A custom non-conforming draw is compared directly without post-validation. Calls are not
skipped at probability zero or one: conforming draws still produce all false or all true values,
respectively, while source advancement stays independent of endpoint optimization. A scalar
consumes one call and an empty result consumes none.

The source follows the same caller-ownership and bounded-reproducibility contract as the other
random methods. The factory does not substitute, retain, seed, synchronize, reset, split, or close
it. Equivalent output requires equivalent generator implementation and initial state, identical
arguments, and no interfering use. One complete canonical `byte[]` is sampled before one BOOL
flat import; there is no partial Tensor, direct storage construction, or direct ID allocation in
the sampling helper.

### Complete Bernoulli-random example

#### Goal and inputs

Create a `2 × 2` BOOL tensor with probability `0.5` from the scripted draws `0.0`, `0.25`, `0.5`,
and `Math.nextDown(1.0)`. The draw equal to the probability demonstrates strict comparison, while
the scripted source makes the unbounded call path and exact call count observable.

```java
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;
import java.util.random.RandomGenerator;

public final class BernoulliRandomExample {
    private static final class ScriptedGenerator implements RandomGenerator {
        private final double[] values = {0.0, 0.25, 0.5, Math.nextDown(1.0)};
        private int calls;

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong is not used");
        }

        @Override
        public double nextDouble() {
            return values[calls++];
        }
    }

    public static void main(String[] args) {
        ScriptedGenerator source = new ScriptedGenerator();
        Tensor tensor = TensorFactory.randomBernoulli(
                Shape.of(2, 2), 0.5, source, Optional.of("mask"));

        byte[] data = (byte[]) tensor.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(Arrays.toString(data));
        System.out.println(tensor.descriptor().dataType());
        System.out.println(tensor.descriptor().requiresGrad());
        System.out.println(source.calls);
    }
}
```

#### Meaningful lines

- The source overrides unbounded `nextDouble()` and rejects `nextLong()`. The factory uses the
  exact supplied object and makes one unbounded call for each output position.
- The first two draws are strictly below `0.5` and become canonical true bytes. The equal and
  greater draws become canonical false bytes.
- The public method fixes BOOL type and false gradient intent, so the call has no `DataType` or
  `requiresGrad` argument.
- Heap-base inspection makes implemented raw storage visible for the example; it is not a typed
  Tensor access or export API.

#### Result and interpretation

The program prints:

```text
[1, 1, 0, 0]
BOOL
false
4
```

The result proves strict comparison, canonical BOOL storage, fixed descriptor facts, and one
unbounded call per row-major element for this scripted input. It does not establish statistical
independence for an arbitrary custom generator, provide numeric Bernoulli output, promise
cross-generator reproducibility, or create a random graph Operation or runtime/backend generator.

#### Failures and useful variations

- A dynamic shape, a logical count above `Integer.MAX_VALUE`, or a non-finite or out-of-range
  probability fails before sampling or ID allocation. Probability is validated for empty output.
- Probability `0.0` or `-0.0` still consumes one call per element and stores all false for a
  conforming source; probability `1.0` consumes the same calls and stores all true.
- If `nextDouble()` throws, earlier calls remain consumed, but no destination or ID exists.
- A blank label fails after all calls, destination allocation, and ID allocation; the ID is
  consumed and no state is rolled back.

### Complete flat-import example

#### Goal and inputs

Import three logical BOOL values from bytes `{0, -2, 3}`, verify canonical storage, and show that
later source mutation does not affect the tensor.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class FlatImportExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(3);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BOOL,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        byte[] source = {0, -2, 3};

        Tensor tensor = TensorFactory.fromFlatArray(
                descriptor, Optional.of("mask"), source);
        byte[] imported = (byte[]) tensor.hostStorage()
                .orElseThrow()
                .segment()
                .heapBase()
                .orElseThrow();

        source[0] = 9;
        System.out.println(tensor.label().orElseThrow());
        System.out.println(Arrays.toString(imported));
    }
}
```

#### Meaningful lines

- `LayoutDescriptor.contiguous(shape)` supplies the required resolved dense-contiguous geometry.
- The `byte[]` carrier matches `BOOL`; import normalizes both non-zero values to canonical byte
  `1` while copying into a distinct factory-allocated array.
- Mutating `source` after import demonstrates that the returned tensor does not retain that array.
  The raw heap-base inspection uses the already exposed storage segment only to make this example's
  result observable; it is not a typed Tensor access API.

#### Result and interpretation

The program prints:

```text
mask
[0, 1, 1]
```

The result proves BOOL normalization, copied ownership, label retention, and row-major order for a
dense flat import. It does not demonstrate nested shape inference or provide public typed access
or export, numeric conversion, view scattering, native allocation, runtime residency, or backend
execution.

#### Failures and useful variations

- Using this `byte[]` with a non-`BOOL` descriptor fails before destination or ID allocation.
- An offset, strided, broadcast, or unresolved layout is rejected before source-length validation.
- A source of length two or four fails because the shape has exactly three logical elements.
- `short[]` imports raw BFLOAT16 bits; it does not convert Java `float` values.

### Complete nested-import example

#### Goal and inputs

Import the rectangular `float[][]` value `{{1, 2}, {3, 4}}`, infer its descriptor, preserve the
gradient request, and verify that later source mutation does not affect the tensor. The source has
declared rank two, exact `float` leaves, and two elements along each axis.

```java
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class NestedImportExample {
    public static void main(String[] args) {
        float[] firstRow = {1.0f, 2.0f};
        float[][] source = {firstRow, {3.0f, 4.0f}};

        Tensor tensor = TensorFactory.fromNestedArray(
                source, Optional.of("weights"), true);
        float[] imported = (float[]) tensor.hostStorage()
                .orElseThrow()
                .segment()
                .heapBase()
                .orElseThrow();

        firstRow[0] = 99.0f;
        System.out.println(tensor.descriptor().dataType());
        System.out.println(tensor.descriptor().shape());
        System.out.println(tensor.descriptor().layout().orElseThrow().kind());
        System.out.println(tensor.descriptor().requiresGrad());
        System.out.println(tensor.label().orElseThrow());
        System.out.println(Arrays.toString(imported));
    }
}
```

#### Meaningful lines

- Passing the nested primitive array as `Object` does not erase its runtime metadata: the factory
  reads declared rank two and ultimate carrier `float`, which infer `FLOAT32` and shape `[2, 2]`.
- `true` requests gradient eligibility on the inferred descriptor. `FLOAT32` permits that request;
  an integral or BOOL carrier would reject it before destination or ID allocation.
- The factory flattens the two rows in encounter order and delegates the resulting `float[]` to
  flat import, which creates the independent destination array.
- Mutating `firstRow` after return demonstrates source independence. The heap-base inspection is
  only a way to observe this example's raw result; it is not a public typed Tensor access API.

#### Result and interpretation

The program prints:

```text
FLOAT32
Shape[2, 2]
DENSE_CONTIGUOUS
true
weights
[1.0, 2.0, 3.0, 4.0]
```

The result proves exact carrier inference, fully static dense shape and layout inference,
row-major copying, label retention, gradient eligibility, and source independence. It does not
promise a snapshot under concurrent source mutation, numeric conversion, dynamic-shape inference,
view population, typed export, native allocation, or execution.

#### Failures and useful variations

- `new int[2][0]` is valid and infers shape `[2, 0]`; `new int[0][2]` is rejected because axis 1 is
  not observable from the empty root.
- `new float[][] {{1}, {2, 3}}` is rejected as ragged at axis `1`, path `[1]`.
- `new float[][] {null, {1}}` is rejected for a null subarray at path `[0]`.
- `Integer[][]`, `boolean[][]`, and rank-one `float[]` values are outside this API.

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

Provenance is fixed at construction and needs no synchronization. `TensorProvenance` contains one
exact immutable `Operation` reference and an ordered immutable `List<Tensor>` snapshot. The list
preserves exact input-object identities, including repeated references and an empty zero-input
case, but does not retain the caller's mutable list container. It performs no operation-arity,
descriptor, cycle, or graph validation. Record equality compares the operation value and ordered
input objects using their ordinary equality; it is not producer-occurrence identity and must not
be used as automatic common-subexpression elimination.

Every public `TensorFactory` creation and population method returns a provenance-free leaf. The
package-private derived-construction seam used by the implemented binary arithmetic, comparison,
boolean logical, conditional-selection, cast, unary, and scalar expression helpers attaches one
already-validated provenance value, allocates exactly one ID through the existing allocator, and
creates no storage. The
factory seam itself does not inspect the operation or inputs, infer a descriptor, traverse an
expression, capture a graph, or perform semantic validation.

### Binary arithmetic expressions

The seven current expression methods build model semantics; they do not calculate element values:

| Method | Ordered meaning of `left.method(right)` |
|---|---|
| `add` | left plus the right addend |
| `sub` | left minus the right subtrahend |
| `mul` | left multiplied by the right factor |
| `div` | left divided by the right denominator |
| `min` | minimum of the left and right operands |
| `max` | maximum of the left and right operands |
| `pow` | left base raised to the right exponent |

The receiver is always the ordered left input and the argument is always the ordered right input.
This order remains in provenance even for operations whose mathematical result is commonly
commutative.

Each method accepts only `BFLOAT16`, `FLOAT32`, and `FLOAT64`. It promotes the two types through
`BFLOAT16 < FLOAT32 < FLOAT64`, applies the existing right-aligned local broadcast rule, and
creates a fresh `TensorDescriptor`. The result layout is unresolved even when every shape
dimension is static because expression construction has not chosen storage geometry. Result
`requiresGrad` is the logical OR of the input requests; that flag records gradient eligibility
only and does not install a gradient rule.

The fresh result has a new factory identity, no label, and no host storage. Its provenance stores
one `Operation` with the matching `BinaryArithmeticKind` and `NoOperationAttrs.INSTANCE`, followed
by the exact ordered input references `[left, right]`. Repeated calls create distinct results, and
even identities such as adding a stored zero or multiplying by a stored one are not simplified at
this boundary. Compiler-owned graph capture, canonicalization, common-subexpression elimination,
autograd, and backend execution remain later responsibilities.

#### Complete binary-expression example

##### Goal and inputs

Build a division expression from a `BFLOAT16` left tensor of shape `[2, 1]` and a `FLOAT32` right
tensor of shape `[1, 3]`. The left descriptor does not request gradients; the right descriptor
does. Both inputs are storage-free leaves, so the example demonstrates expression metadata rather
than numerical division.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class BinaryExpressionExample {
    public static void main(String[] args) {
        Tensor left = TensorFactory.create(new TensorDescriptor(
                DataType.BFLOAT16, Shape.of(2, 1), Optional.empty(), false));
        Tensor right = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(1, 3), Optional.empty(), true));

        Tensor result = left.div(right);
        TensorProvenance provenance = result.provenance().orElseThrow();

        System.out.println("type=" + result.descriptor().dataType());
        System.out.println("shape=" + result.descriptor().shape());
        System.out.println("layoutUnresolved=" + result.descriptor().layout().isEmpty());
        System.out.println("requiresGrad=" + result.descriptor().requiresGrad());
        System.out.println("unlabeled=" + result.label().isEmpty());
        System.out.println("storageFree=" + result.hostStorage().isEmpty());
        System.out.println("kind=" + provenance.operation().kind());
        System.out.println("parameterless="
                + (provenance.operation().attrs() == NoOperationAttrs.INSTANCE));
        System.out.println("orderedInputs="
                + (provenance.inputs().get(0) == left
                && provenance.inputs().get(1) == right));
        System.out.println("fresh=" + (result != left && result != right));
    }
}
```

##### Meaningful lines and intermediate results

- `BFLOAT16` and `FLOAT32` promote to `FLOAT32`.
- Right alignment compares shape axes as `1` with `3` and `2` with `1`; each singleton expands,
  producing shape `[2, 3]`.
- `left.div(right)` preserves the non-commutative roles: `left` is the numerator and `right` is
  the denominator. The operation kind is `DIV`, and the parameterless attributes are the canonical
  singleton.
- The input gradient requests `false` and `true` combine to `true`. This result is eligibility
  metadata rather than a generated backward computation.

##### Result and interpretation

The program prints:

```text
type=FLOAT32
shape=Shape[2, 3]
layoutUnresolved=true
requiresGrad=true
unlabeled=true
storageFree=true
kind=DIV
parameterless=true
orderedInputs=true
fresh=true
```

The output proves local type promotion, broadcasting, descriptor derivation, fresh identity, and
ordered provenance. It does not prove a numerical quotient, division-by-zero behavior, a gradient
formula, graph capture, backend support, or execution because none of those behaviors is part of
the current expression-construction API.

##### Failures and useful variations

- A null right operand fails with `NullPointerException` and message `right`.
- An `INT32`, `INT64`, or `BOOL` operand fails through floating promotion; no implicit cast is
  inserted.
- Incompatible static dimensions, different dynamic symbols, and a dynamic dimension paired with
  a non-singleton static size fail through local broadcasting. Equal dynamic symbols and singleton
  expansion remain valid.
- Validation failures happen before result identity allocation. Exhausting the factory's tensor-ID
  space instead fails after the local descriptor, operation, and provenance values are built.

### Binary comparison expressions

The six current comparison methods build ordered elementwise relation semantics; they do not read
or compare element values:

| Method | Ordered meaning of `left.method(right)` |
|---|---|
| `greaterThan` | Left value is strictly greater than the right value. |
| `greaterOrEqual` | Left value is greater than or equal to the right value. |
| `lessThan` | Left value is strictly less than the right value. |
| `lessOrEqual` | Left value is less than or equal to the right value. |
| `equalTo` | Left and right values compare equal. |
| `notEqualTo` | Left and right values compare unequal. |

The receiver remains the ordered left input and the argument remains the ordered right input for
every method. Equality and inequality retain this order in provenance even though their relation
is symmetric.

Each method accepts every ordered pair of `BFLOAT16`, `FLOAT32`, and `FLOAT64` inputs. The existing
floating-promotion hierarchy validates their common comparison domain, but the promoted type is
not the output type. Right-aligned local broadcasting derives the result shape. The result data
type is always `BOOL`, its layout is unresolved, and `requiresGrad` is always false even when one
or both inputs request gradients.

Every valid call returns a fresh Tensor with a new factory identity, no label, and no host storage.
Its provenance contains one `Operation` with the exact matching `BinaryComparisonKind` and
`NoOperationAttrs.INSTANCE`, followed by exact ordered input references `[left, right]`. Repeated,
self, and symmetric calls are not interned, reordered, or canonicalized. Numerical comparison
policy, compiler capture, training-graph treatment, gradient rules, and backend execution remain
later responsibilities.

#### Complete comparison-expression example

##### Goal and inputs

Build a less-than-or-equal expression from a `BFLOAT16` left tensor of shape `[2, 1]` and a
`FLOAT64` right tensor of shape `[1, 3]`. The left input requests gradients, but the logical result
is a non-differentiable `BOOL` tensor. Both inputs are storage-free leaves, so the example observes
expression metadata rather than numerical comparison results.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class ComparisonExpressionExample {
    public static void main(String[] args) {
        Tensor left = TensorFactory.create(new TensorDescriptor(
                DataType.BFLOAT16, Shape.of(2, 1), Optional.empty(), true));
        Tensor right = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT64, Shape.of(1, 3), Optional.empty(), false));

        Tensor result = left.lessOrEqual(right);
        TensorProvenance provenance = result.provenance().orElseThrow();

        System.out.println("type=" + result.descriptor().dataType());
        System.out.println("shape=" + result.descriptor().shape());
        System.out.println("layoutUnresolved=" + result.descriptor().layout().isEmpty());
        System.out.println("requiresGrad=" + result.descriptor().requiresGrad());
        System.out.println("unlabeled=" + result.label().isEmpty());
        System.out.println("storageFree=" + result.hostStorage().isEmpty());
        System.out.println("kind=" + provenance.operation().kind());
        System.out.println("parameterless="
                + (provenance.operation().attrs() == NoOperationAttrs.INSTANCE));
        System.out.println("orderedInputs="
                + (provenance.inputs().get(0) == left
                && provenance.inputs().get(1) == right));
        System.out.println("fresh=" + (result != left && result != right));
    }
}
```

##### Meaningful lines and intermediate results

- `BFLOAT16` and `FLOAT64` establish `FLOAT64` as the common comparison domain, but the result
  remains `BOOL`.
- Right alignment compares shape axes as `1` with `3` and `2` with `1`; singleton expansion
  produces shape `[2, 3]`.
- `left.lessOrEqual(right)` preserves the ordered relation: `left` is compared to `right`, the
  operation kind is `LESS_OR_EQUAL`, and the attributes are the canonical parameterless singleton.
- The true input gradient request does not propagate. A comparison result has false gradient
  eligibility and this construction adds no gradient rule.

##### Result and interpretation

The program prints:

```text
type=BOOL
shape=Shape[2, 3]
layoutUnresolved=true
requiresGrad=false
unlabeled=true
storageFree=true
kind=LESS_OR_EQUAL
parameterless=true
orderedInputs=true
fresh=true
```

The output proves floating-domain validation, broadcasting, fixed `BOOL` descriptor facts, fresh
identity, and ordered provenance. It does not prove any elementwise truth value, NaN or signed-zero
policy, tolerance behavior, gradient handling by a future compiler, backend support, or execution.

##### Failures and useful variations

- A null right operand fails with `NullPointerException` and message `right`.
- An `INT32`, `INT64`, or `BOOL` operand fails through floating validation; no implicit cast is
  inserted.
- Incompatible static dimensions, different dynamic symbols, and a dynamic dimension paired with
  a non-singleton static size fail through local broadcasting. Equal dynamic symbols and singleton
  expansion remain valid.
- Validation failures happen before result identity allocation. Exhausting the factory's tensor-ID
  space instead fails after the local descriptor, operation, and provenance values are built.

### Boolean logical expressions

The three current logical methods build elementwise truth semantics from exact `BOOL` tensors;
they do not read or interpret stored truth bytes:

| Method | Ordered meaning and shape rule |
|---|---|
| `logicalAnd(right)` | Conjunction of ordered receiver and argument values; right-aligned broadcasting derives the result shape. |
| `logicalOr(right)` | Disjunction of ordered receiver and argument values; right-aligned broadcasting derives the result shape. |
| `logicalNot()` | Negation of the receiver values; the result retains the exact input `Shape` reference without broadcasting. |

AND and OR require both operands to have exactly `DataType.BOOL`; NOT requires the receiver to
have exactly `DataType.BOOL`. No floating or integral value is accepted as truth, and construction
does not insert a cast. Binary broadcasting uses the same local proof rules as other binary
expressions: equal dimensions and singleton expansion are accepted, while incompatible or locally
unprovable symbolic pairs fail. NOT performs no shape algebra and does not copy a resolved input
layout.

Every valid call returns a fresh Tensor with a new factory identity, fixed `BOOL` data type,
unresolved layout, false gradient eligibility, no label, and no host storage. Its provenance
contains the exact `BooleanLogicalKind` with `NoOperationAttrs.INSTANCE`. AND and OR preserve exact
ordered inputs `[receiver, right]`, including self-use; NOT records exactly `[receiver]`. Repeated
calls, commutative operand reversals, and double negation remain distinct expressions.

Construction is eager only for descriptor, operation, provenance, and identity metadata. It does
not short-circuit, inspect input storage, compute truth values, simplify self-use or double
negation, reorder commutative inputs, propagate gradient eligibility, create a gradient rule,
capture a graph, or execute a backend operation.

#### Failures and useful variations

- A null AND or OR argument fails with `NullPointerException` and message `right`.
- A non-BOOL receiver or argument fails with `IllegalArgumentException`; binary checks the left
  input before the right input, and unary reports the receiver as `input`.
- Incompatible static shapes and locally unprovable symbolic pairs fail through local
  broadcasting. Equal symbols and singleton-to-symbol expansion are valid.
- Comparison results already have `BOOL` descriptors and can feed all three logical methods
  directly. This chaining constructs metadata only and does not calculate either comparison or
  logical truth values.
- Validation failures occur before result identity allocation. Identifier exhaustion occurs only
  after the fixed descriptor, operation, and provenance values have been built.

The current class stores no graph-local `NodeId` or `ValueId`, gradient object or trainable role,
publication state, typed element access, device buffer, runtime residency, prepared state, or
backend support. Provenance does not add any of those roles; they remain separate planned
contracts in their owning layers.

### Conditional-selection expression

The static `Tensor.where(condition, ifTrue, ifFalse)` method builds one elementwise conditional
selection expression. It constructs metadata and does not read a condition value or choose a
branch:

```text
branchShape = broadcast(ifTrue.shape, ifFalse.shape)
resultShape = broadcast(condition.shape, branchShape)
```

The condition must have exactly `DataType.BOOL`. The true and false branches must each be
`BFLOAT16`, `FLOAT32`, or `FLOAT64`; the shared floating-promotion hierarchy derives the result
data type from the branches only. The first local broadcast proves branch compatibility. The
second proves that the condition can address their common result shape. Equal dimensions and
static singleton expansion are accepted, including locally provable dynamic cases; incompatible
or locally unprovable symbolic pairs fail.

Every valid call returns a fresh Tensor with the promoted branch data type, final three-way
broadcast shape, unresolved layout, no label, and no host storage. Result `requiresGrad` is
`ifTrue.requiresGrad || ifFalse.requiresGrad`; the non-differentiable BOOL condition does not
contribute. Provenance contains `WhereSelectionKind.WHERE`, `NoOperationAttrs.INSTANCE`, and exact
ordered inputs `[condition, ifTrue, ifFalse]`. Repeated branch references remain repeated, and
repeated calls receive fresh identities.

Construction is eager only for descriptor, operation, provenance, and identity metadata. It does
not inspect input storage, evaluate either branch, define eager or lazy evaluation order, copy or
convert values, simplify equal branches, create gradient-routing rules, capture a graph, map ONNX,
or execute backend code. Conditional `where` is also distinct from scalar-index `select`, which
belongs to the later indexing family.

#### Complete conditional-selection example

##### Goal and inputs

Build a selection expression from a BOOL condition of shape `[2, 1]`, a gradient-eligible
`BFLOAT16` true branch of shape `[1, 3]`, and a `FLOAT64` false branch of shape `[2, 1]`. All three
inputs are storage-free leaves, so the example observes expression metadata rather than selected
values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class WhereExpressionExample {
    public static void main(String[] args) {
        Tensor condition = TensorFactory.create(new TensorDescriptor(
                DataType.BOOL, Shape.of(2, 1), Optional.empty(), false));
        Tensor ifTrue = TensorFactory.create(new TensorDescriptor(
                DataType.BFLOAT16, Shape.of(1, 3), Optional.empty(), true));
        Tensor ifFalse = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT64, Shape.of(2, 1), Optional.empty(), false));

        Tensor result = Tensor.where(condition, ifTrue, ifFalse);
        TensorProvenance provenance = result.provenance().orElseThrow();

        System.out.println("type=" + result.descriptor().dataType());
        System.out.println("shape=" + result.descriptor().shape());
        System.out.println("layoutUnresolved=" + result.descriptor().layout().isEmpty());
        System.out.println("requiresGrad=" + result.descriptor().requiresGrad());
        System.out.println("unlabeled=" + result.label().isEmpty());
        System.out.println("storageFree=" + result.hostStorage().isEmpty());
        System.out.println("kind=" + provenance.operation().kind());
        System.out.println("parameterless="
                + (provenance.operation().attrs() == NoOperationAttrs.INSTANCE));
        System.out.println("orderedInputs="
                + (provenance.inputs().get(0) == condition
                && provenance.inputs().get(1) == ifTrue
                && provenance.inputs().get(2) == ifFalse));
        System.out.println("fresh="
                + (result != condition && result != ifTrue && result != ifFalse));
    }
}
```

##### Meaningful lines and intermediate results

- `BFLOAT16` and `FLOAT64` branches promote to `FLOAT64`; the BOOL condition does not participate
  in promotion.
- Broadcasting the branch shapes `[1, 3]` and `[2, 1]` first produces `[2, 3]`. Broadcasting the
  condition shape `[2, 1]` with that branch shape then retains `[2, 3]`.
- `Tensor.where` records exact ordered condition, true-branch, and false-branch references. The
  operation kind is `WHERE`, and the attributes are the canonical parameterless singleton.
- Only the true branch requests gradients, so the result's eligibility flag is true. This is
  descriptor metadata, not a generated backward computation or routing rule.

##### Result and interpretation

The program prints:

```text
type=FLOAT64
shape=Shape[2, 3]
layoutUnresolved=true
requiresGrad=true
unlabeled=true
storageFree=true
kind=WHERE
parameterless=true
orderedInputs=true
fresh=true
```

The output proves local branch promotion, ordered two-stage broadcasting, descriptor derivation,
fresh identity, and exact three-input provenance. It does not prove which branch value is chosen,
gradient routing, graph capture, ONNX mapping, backend support, or execution.

##### Failures and useful variations

- A null argument fails in condition, true-branch, then false-branch order with the parameter name
  as the `NullPointerException` message.
- A non-BOOL condition fails before branch promotion. An integral or BOOL branch fails through
  floating promotion; no implicit cast is inserted.
- Incompatible branches fail before the condition shape is combined. A condition incompatible
  with the common branch shape fails in the second broadcast.
- Scalar, zero-sized, rank-mismatched, singleton-expanded, and locally provable symbolic shapes
  are valid. Different dynamic symbols and a dynamic dimension paired with a non-singleton static
  size remain locally unprovable.
- Validation failures happen before result identity allocation. Exhausting the factory's tensor-ID
  space instead fails after the local descriptor, operation, and provenance values are built.

### Cast expressions

`Tensor.cast(targetDataType)` records an explicit request to convert each logical input value to a
target data type. The method accepts every ordered pair formed from the six current `DataType`
values, so all 36 source/target combinations are representable. Representability means the model
can preserve the request; it does not define the numerical result or promise that a backend can
execute every pair.

Cast changes the result data type but not its logical dimensions. The result descriptor therefore
retains the input descriptor's exact immutable `Shape` reference. Its layout is always unresolved,
including when the input has resolved geometry and when source and target types are equal. A cast
may change element width, and expression construction does not decide whether storage can be
reused, copied, or materialized. The result has no label and no host storage.

Result gradient eligibility is true exactly when the input already requests gradients and both
the source and target data types are floating. This flag is descriptor metadata only. It neither
creates a backward rule nor promises that a backend supports differentiation through the cast.

Every call returns a fresh Tensor with a new factory identity. A same-type call is still an
explicit `CAST` expression rather than an early return of the receiver. Repeated calls remain
distinct, and a chain retains only its immediately preceding Tensor at each provenance link.
Compiler optimization later owns any legal redundant-cast or cast-chain simplification.

Provenance contains one `Operation` with `CastKind.CAST`, one fresh
`CastAttrs(targetDataType)`, and the exact one-input list `[input]`. The source type is read from
the input descriptor and is not duplicated in `CastAttrs`. Construction does not read or mutate
input storage, convert a value, attach output storage, define rounding or overflow behavior,
capture a graph, or execute work.

#### Complete cast-expression example

##### Goal and inputs

Build a same-type `FLOAT32` cast from a tensor with resolved dense layout, then cast that explicit
result to `INT64`. The input requests gradients. The example observes descriptor, identity,
ownership, and provenance boundaries; it does not observe converted element values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class CastExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        LayoutDescriptor inputLayout = LayoutDescriptor.contiguous(shape);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(inputLayout), true));

        Tensor sameType = input.cast(DataType.FLOAT32);
        Tensor integral = sameType.cast(DataType.INT64);
        TensorProvenance firstOrigin = sameType.provenance().orElseThrow();
        TensorProvenance secondOrigin = integral.provenance().orElseThrow();
        CastAttrs firstAttrs = (CastAttrs) firstOrigin.operation().attrs();
        CastAttrs secondAttrs = (CastAttrs) secondOrigin.operation().attrs();

        System.out.println("sameTypeFresh=" + (sameType != input));
        System.out.println("sameShape=" + (integral.descriptor().shape() == shape));
        System.out.println("layoutUnresolved=" + integral.descriptor().layout().isEmpty());
        System.out.println("sameTypeRequiresGrad="
                + sameType.descriptor().requiresGrad());
        System.out.println("integralRequiresGrad="
                + integral.descriptor().requiresGrad());
        System.out.println("unlabeled=" + integral.label().isEmpty());
        System.out.println("storageFree=" + integral.hostStorage().isEmpty());
        System.out.println("firstKind=" + firstOrigin.operation().kind());
        System.out.println("firstTarget=" + firstAttrs.targetDataType());
        System.out.println("firstInput=" + (firstOrigin.inputs().getFirst() == input));
        System.out.println("secondTarget=" + secondAttrs.targetDataType());
        System.out.println("immediateInput="
                + (secondOrigin.inputs().getFirst() == sameType));
    }
}
```

##### Meaningful lines and intermediate results

- The input has a resolved dense layout, but `input.cast(DataType.FLOAT32)` creates a fresh
  same-type result whose layout is unresolved. The method records the request instead of returning
  `input` or claiming physical storage reuse.
- The same-type floating cast retains the true gradient request. Casting that result to `INT64`
  sets eligibility to false because the target is not floating.
- Both results retain the exact `shape` object. The first provenance link points to `input`; the
  second points to `sameType`, so the chain remains explicit and local.
- Each operation uses `CAST`; its `CastAttrs` stores only the exact target data type.

##### Result and interpretation

The program prints:

```text
sameTypeFresh=true
sameShape=true
layoutUnresolved=true
sameTypeRequiresGrad=true
integralRequiresGrad=false
unlabeled=true
storageFree=true
firstKind=CAST
firstTarget=FLOAT32
firstInput=true
secondTarget=INT64
immediateInput=true
```

The output proves fresh same-type identity, exact shape-reference retention, unresolved result
layout, the gradient-eligibility boundary, absent result label/storage, typed target attributes,
and immediate-input provenance. It does not prove how any `FLOAT32` value becomes `INT64`, whether
rounding or saturation occurs, whether a compiler removes the first cast, or whether a backend can
execute either request.

##### Failures and useful variations

- A null target fails with `NullPointerException` and message `targetDataType` before result
  identity allocation.
- Scalar, zero-sized, fully static, and dynamic shapes are accepted without shape reconstruction.
- Integral-to-floating, BOOL-to-floating, integral-to-integral, and BOOL-related casts are valid
  expressions but always have false gradient eligibility because the source or target is
  non-floating.
- Exhausting the factory's tensor-ID space fails after the local descriptor, attributes,
  operation, and provenance values are built.

### Unary elementwise expressions

The fifteen current zero-argument methods create one-input elementwise semantics without reading
or calculating element values:

| Method | Elementwise meaning |
|---|---|
| `abs` | Absolute magnitude. |
| `neg` | Additive inverse. |
| `inv` | Multiplicative reciprocal. |
| `log` | Natural logarithm. |
| `exp` | Strict natural exponential request. |
| `erf` | Gaussian error function. |
| `sqrt` | Principal square root. |
| `floor` | Greatest integer-valued result not greater than the input. |
| `ceil` | Least integer-valued result not less than the input. |
| `sign` | Numeric negative, zero, or positive classification. |
| `relu` | Rectified linear unit. |
| `sigmoid` | Logistic sigmoid. |
| `tanh` | Strict hyperbolic tangent request. |
| `fastExp` | Explicit approximate natural exponential request. |
| `fastTanh` | Explicit approximate hyperbolic tangent request. |

Each method accepts only `BFLOAT16`, `FLOAT32`, or `FLOAT64`. The result retains the exact input
data type and immutable `Shape` reference; no promotion or shape algebra is needed for one input.
Its layout is unresolved even when the input layout is resolved, because expression construction
does not select storage geometry or a materialization route. The input `requiresGrad` value is
preserved for every kind, including `floor`, `ceil`, and `sign`. That flag remains eligibility
metadata and does not assert that a derivative or backward rule exists.

Every valid call returns a fresh Tensor with a new factory identity, no label, and no host storage.
Its provenance contains one `Operation` with the exact matching `UnaryElementwiseKind` and
`NoOperationAttrs.INSTANCE`, followed by exactly the receiver reference. A chain retains its
immediately preceding result as the next input. Calls are never interned or simplified at this
boundary, and `log`, `sqrt`, and `inv` do not inspect values to enforce mathematical domains.
Compiler optimization, autograd, numerical edge behavior, and backend execution remain later
responsibilities.

`fastExp` and `fastTanh` construct kinds distinct from `exp` and `tanh`. The “fast” names express
approximation intent only; this API does not choose an algorithm, promise an error bound, or claim
backend availability.

#### Complete unary-expression example

##### Goal and inputs

Build a fast-exponential expression from a storage-free `FLOAT32` tensor of shape `[2, 3]` that
requests gradient eligibility. The example observes expression metadata, not exponential values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class UnaryExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));

        Tensor result = input.fastExp();
        TensorProvenance provenance = result.provenance().orElseThrow();

        System.out.println("type=" + result.descriptor().dataType());
        System.out.println("sameShape=" + (result.descriptor().shape() == shape));
        System.out.println("layoutUnresolved=" + result.descriptor().layout().isEmpty());
        System.out.println("requiresGrad=" + result.descriptor().requiresGrad());
        System.out.println("unlabeled=" + result.label().isEmpty());
        System.out.println("storageFree=" + result.hostStorage().isEmpty());
        System.out.println("kind=" + provenance.operation().kind());
        System.out.println("parameterless="
                + (provenance.operation().attrs() == NoOperationAttrs.INSTANCE));
        System.out.println("exactInput=" + (provenance.inputs().getFirst() == input));
        System.out.println("fresh=" + (result != input));
    }
}
```

##### Meaningful lines and intermediate results

- `TensorFactory.create` makes a provenance-free leaf whose exact immutable shape object is
  `shape` and whose descriptor requests gradients.
- `input.fastExp()` creates a new semantic expression without reading storage or calculating an
  exponential. It retains `FLOAT32`, the exact `shape` reference, and the true eligibility flag.
- The operation kind is `FAST_EXP`, not `EXP`, and its complete parameter value is the canonical
  no-attributes singleton. Provenance retains `input` as its sole exact input reference.

##### Result and interpretation

The program prints:

```text
type=FLOAT32
sameShape=true
layoutUnresolved=true
requiresGrad=true
unlabeled=true
storageFree=true
kind=FAST_EXP
parameterless=true
exactInput=true
fresh=true
```

The output proves exact type and shape retention, unresolved layout, gradient-eligibility
propagation, fresh identity, and one-input provenance. It does not prove a numerical exponential,
an approximation algorithm or accuracy bound, a gradient rule, graph capture, backend support, or
execution.

##### Failures and useful variations

- Calling any unary elementwise method on an `INT32`, `INT64`, or `BOOL` tensor fails with
  `IllegalArgumentException` before result identity allocation; no implicit cast is inserted.
- Scalar, zero-sized, ordinary static, and dynamic shapes are accepted. A resolved input layout is
  deliberately not copied to the result.
- Replacing `fastExp()` with `exp()` creates the distinct strict `EXP` semantic request. Chaining
  another unary call records the first result, not the original leaf, as its exact input.
- Exhausting the factory's tensor-ID space fails after the local descriptor, operation, and
  provenance values are built.

### Scalar arithmetic and clamp expressions

The five current scalar methods create parameterized one-input semantics without converting a
parameter to the input type or calculating element values:

| Method | Elementwise meaning | Operation attributes |
|---|---|---|
| `mul(double)` | Multiply the input by the scalar. | `ScalarValueAttrs` |
| `pow(double)` | Raise the input to the scalar exponent. | `ScalarValueAttrs` |
| `clamp(double, double)` | Constrain the input to inclusive lower and upper bounds. | `ClampRangeAttrs` |
| `clampMin(double)` | Apply an inclusive scalar lower bound. | `ScalarValueAttrs` |
| `clampMax(double)` | Apply an inclusive scalar upper bound. | `ScalarValueAttrs` |

Each method accepts only `BFLOAT16`, `FLOAT32`, or `FLOAT64`. The result retains the exact input
data type and immutable `Shape` reference, preserves `requiresGrad`, and leaves layout unresolved.
The scalar or bounds remain Java `double` values in the operation attributes with their exact
binary64 bits, independent of input type. There is no implicit `BFLOAT16` or `FLOAT32` conversion.

Every valid call returns a fresh Tensor with a new factory identity, no label, and no host storage.
Its provenance contains the exact matching `ScalarElementwiseKind`, typed attributes, and exactly
the receiver reference. `clamp(minValue, maxValue)` is one first-class `CLAMP` operation; it is not
expanded into `CLAMP_MIN` followed by `CLAMP_MAX`. Repeated, nested, identity-like, and
special-value calls are not interned, folded, or simplified at this boundary.

For `clamp`, floating-input eligibility is checked before range ordering. `ClampRangeAttrs` then
rejects only primitive `minValue > maxValue`; equal bounds, either ordering of signed zeros,
ordered infinities, and a NaN endpoint are representable. Numerical special-value behavior,
scalar conversion for execution, optimization, gradients, graph capture, and backend execution
remain later responsibilities.

#### Complete scalar-expression example

##### Goal and inputs

Build a scalar multiplication followed by one range-clamp expression from a storage-free
`FLOAT32` tensor of shape `[2, 3]`. The example uses negative zero to demonstrate exact binary64
attribute retention and observes expression metadata rather than numerical values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class ScalarExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));

        Tensor scaled = input.mul(-0.0);
        Tensor result = scaled.clamp(-0.0, 1.0);
        TensorProvenance scaledFrom = scaled.provenance().orElseThrow();
        TensorProvenance clampedFrom = result.provenance().orElseThrow();
        ScalarValueAttrs multiplier =
                (ScalarValueAttrs) scaledFrom.operation().attrs();
        ClampRangeAttrs range =
                (ClampRangeAttrs) clampedFrom.operation().attrs();

        System.out.println("type=" + result.descriptor().dataType());
        System.out.println("sameShape=" + (result.descriptor().shape() == shape));
        System.out.println("layoutUnresolved=" + result.descriptor().layout().isEmpty());
        System.out.println("requiresGrad=" + result.descriptor().requiresGrad());
        System.out.println("unlabeled=" + result.label().isEmpty());
        System.out.println("storageFree=" + result.hostStorage().isEmpty());
        System.out.println("multiplierKind=" + scaledFrom.operation().kind());
        System.out.println("multiplierBits="
                + Long.toHexString(Double.doubleToRawLongBits(multiplier.value())));
        System.out.println("clampKind=" + clampedFrom.operation().kind());
        System.out.println("minBits="
                + Long.toHexString(Double.doubleToRawLongBits(range.minValue())));
        System.out.println("maxBits="
                + Long.toHexString(Double.doubleToRawLongBits(range.maxValue())));
        System.out.println("exactInput=" + (clampedFrom.inputs().getFirst() == scaled));
        System.out.println("oneClampInput=" + (clampedFrom.inputs().size() == 1));
        System.out.println("fresh=" + (result != scaled && scaled != input));
    }
}
```

##### Meaningful lines and intermediate results

- `input.mul(-0.0)` records scalar `MUL` and retains the negative-zero bit pattern in one
  `ScalarValueAttrs`; it does not convert the multiplier to `FLOAT32`.
- `scaled.clamp(-0.0, 1.0)` records exactly one `CLAMP` with one `ClampRangeAttrs`. Its sole input
  is the exact `scaled` object, so the chain preserves immediate expression provenance.
- Both calls preserve `FLOAT32`, the exact `shape` reference, and true gradient eligibility while
  creating fresh unresolved, unlabeled, storage-free results.

##### Result and interpretation

The program prints:

```text
type=FLOAT32
sameShape=true
layoutUnresolved=true
requiresGrad=true
unlabeled=true
storageFree=true
multiplierKind=MUL
multiplierBits=8000000000000000
clampKind=CLAMP
minBits=8000000000000000
maxBits=3ff0000000000000
exactInput=true
oneClampInput=true
fresh=true
```

The output proves exact attribute-bit retention, first-class range-clamp identity, descriptor
retention, fresh identity, and one-input provenance. It does not prove multiplication or clamp
values, parameter conversion for a backend, gradient rules, graph capture, backend support, or
execution.

##### Failures and useful variations

- Calling any scalar method on an `INT32`, `INT64`, or `BOOL` tensor fails with
  `IllegalArgumentException` before result identity allocation; no implicit cast is inserted.
- `clamp(2.0, 1.0)` fails with `IllegalArgumentException` and message
  `minValue must be less than or equal to maxValue`. On a non-floating input, the data-type failure
  occurs before that range failure.
- Signed zeros, infinities, and NaN payload bits remain unchanged in attributes. A NaN clamp
  endpoint is accepted because primitive `>` is false when either operand is NaN.
- Scalar, zero-sized, ordinary static, and dynamic shapes are accepted. Repeated calls and values
  such as multiplier `1.0` still create fresh expressions without canonicalization.

### Numeric aggregate expressions

The fifteen current numeric aggregate methods build sum, arithmetic-mean, product, minimum, and
maximum expression metadata without reading or combining element values:

| Family | Full form | Axis removed | Axis optionally retained |
|---|---|---|---|
| Sum | `sum()` | `sum(axis)` | `sum(axis, keepDimensions)` |
| Arithmetic mean | `mean()` | `mean(axis)` | `mean(axis, keepDimensions)` |
| Product | `prod()` | `prod(axis)` | `prod(axis, keepDimensions)` |
| Minimum | `min()` | `min(axis)` | `min(axis, keepDimensions)` |
| Maximum | `max()` | `max(axis)` | `max(axis, keepDimensions)` |

Every method accepts only `BFLOAT16`, `FLOAT32`, or `FLOAT64` and preserves that exact input data
type. A full form reduces every input axis to the canonical rank-zero `Shape.scalar()` and records
`NoOperationAttrs.INSTANCE`. An axis form accepts a positive or negative axis, normalizes it
against the input `Shape`, and records `AxisReductionAttrs(normalizedAxis, keepDimensions)`. A
false `keepDimensions` removes the selected axis; true replaces only that axis with a new static
extent of one. Every unaffected static or dynamic `Dimension` object is retained by exact
reference. Removing the only axis of a rank-one tensor also produces the canonical scalar shape.

Every valid call returns a fresh Tensor with unchanged gradient eligibility, unresolved layout,
no label, and no host storage. Its provenance contains the matching `SUM`, `MEAN`, `PROD`, `MIN`,
or `MAX` aggregate operation and exactly the receiver reference. The eligibility flag remains a
request in model metadata; preserving it for product or an extrema reduction does not install or
promise a gradient rule or a policy for distributing gradients across tied extrema.

The aggregate `min()` and `max()` families are distinct from binary elementwise `min(Tensor)` and
`max(Tensor)`. An aggregate form has one provenance input, uses `AggregateReductionKind.MIN` or
`AggregateReductionKind.MAX`, and reduces every axis or one selected axis. A binary form has two
ordered provenance inputs, uses `BinaryArithmeticKind.MIN` or `BinaryArithmeticKind.MAX`, and
broadcasts corresponding elements without reducing rank. Java overload signatures select the
family; equal enum constant names do not make the typed operation kinds equal.

Zero-sized and dynamic dimensions are valid structural inputs. Expression construction does not
inspect element counts or define an empty-domain result, a mean denominator, accumulation
precision or order, overflow, extrema comparison rules, NaN or signed-zero handling, numerical
values, compiler capture, gradients, extrema-tie behavior, backend support, or execution.

#### Complete numeric-aggregate example

##### Goal and inputs

Build full, axis-removing, and retained-axis expressions from a storage-free `FLOAT32` tensor of
shape `[batch, 3, 4]`, where `batch` is a symbolic dynamic dimension. The example observes
result-shape and provenance metadata rather than aggregate values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class NumericAggregateExpressionExample {
    public static void main(String[] args) {
        DynamicDimension batch = new DynamicDimension("batch");
        Shape inputShape = Shape.ofDimensions(
                batch, new StaticDimension(3), new StaticDimension(4));
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, inputShape, Optional.empty(), true));

        Tensor full = input.sum();
        Tensor removed = input.mean(-1);
        Tensor retained = input.prod(1, true);
        TensorProvenance fullFrom = full.provenance().orElseThrow();
        TensorProvenance removedFrom = removed.provenance().orElseThrow();
        TensorProvenance retainedFrom = retained.provenance().orElseThrow();

        System.out.println("full=" + full.descriptor().shape());
        System.out.println("fullKind=" + fullFrom.operation().kind());
        System.out.println("fullParameterless="
                + (fullFrom.operation().attrs() == NoOperationAttrs.INSTANCE));
        System.out.println("removed=" + removed.descriptor().shape());
        System.out.println("removedAttrs=" + removedFrom.operation().attrs());
        System.out.println("retained=" + retained.descriptor().shape());
        System.out.println("retainedAttrs=" + retainedFrom.operation().attrs());
        System.out.println("sameBatch="
                + (removed.descriptor().shape().dimensions().getFirst() == batch
                && retained.descriptor().shape().dimensions().getFirst() == batch));
        System.out.println("metadata="
                + (retained.descriptor().dataType() == DataType.FLOAT32
                && retained.descriptor().requiresGrad()
                && retained.descriptor().layout().isEmpty()
                && retained.label().isEmpty()
                && retained.hostStorage().isEmpty()));
        System.out.println("exactInputs="
                + (fullFrom.inputs().getFirst() == input
                && removedFrom.inputs().getFirst() == input
                && retainedFrom.inputs().getFirst() == input));
    }
}
```

##### Meaningful lines and intermediate results

- `input.sum()` selects every axis. Its complete parameter value is the canonical no-attributes
  singleton, and its result shape is the canonical scalar.
- `input.mean(-1)` normalizes `-1` to axis `2` and removes the final extent `4`, producing
  `[batch, 3]` with `AxisReductionAttrs[axis=2, keepDimensions=false]`.
- `input.prod(1, true)` retains normalized axis `1` with extent one, producing
  `[batch, 1, 4]` with `AxisReductionAttrs[axis=1, keepDimensions=true]`.
- Both axis results retain the exact `batch` object. All results preserve `FLOAT32` and the true
  gradient request while remaining unresolved, unlabeled, storage-free expressions.

##### Result and interpretation

The program prints:

```text
full=Shape[]
fullKind=SUM
fullParameterless=true
removed=Shape[batch, 3]
removedAttrs=AxisReductionAttrs[axis=2, keepDimensions=false]
retained=Shape[batch, 1, 4]
retainedAttrs=AxisReductionAttrs[axis=1, keepDimensions=true]
sameBatch=true
metadata=true
exactInputs=true
```

The output proves the local axis normalization, structural result-shape derivation, attribute
choice, metadata retention, and one-input provenance shared by all five numeric aggregate
families. It does not prove a sum, mean, product, extrema comparison, gradient formula, graph
capture, backend implementation, or execution.

##### Failures and useful variations

- Calling any numeric aggregate method on `INT32`, `INT64`, or `BOOL` fails with
  `IllegalArgumentException`; no cast is inserted. For an axis form, this type check precedes axis
  validation.
- An axis outside `[-rank, rank - 1]` fails with `IndexOutOfBoundsException`. Every axis is invalid
  for a scalar input, while a scalar full reduction is valid and still returns a fresh scalar.
- A rank-one axis removal returns `Shape.scalar()`. A zero extent or dynamic dimension does not
  trigger numerical inspection at this boundary.
- Repeated and nested calls always create fresh expression tensors; this API performs no
  canonicalization or common-subexpression elimination.

### Masked sum and mean expressions

The two masked aggregate overloads build one axis-removing expression from a floating input and
an exact BOOL mask:

| Public form | Operation kind | Result axis |
|---|---|---|
| `sum(axis, mask)` | `AggregateReductionKind.SUM` | Removed |
| `mean(axis, mask)` | `AggregateReductionKind.MEAN` | Removed |

Mask alignment is an ordered injection rather than ordinary right-aligned broadcasting. Each mask
dimension maps to one distinct, increasing input-axis position. Equal immutable dimensions are
compatible, and a static mask singleton may align to any input dimension; input axes omitted by
the mapping broadcast implicitly. Among valid mappings, construction first prefers one containing
the normalized reduction axis, then the smallest total positional displacement, then the
lexicographically smallest input-axis list. A scalar mask therefore maps to the empty list.

For input `[batch, time, features]`, mask `[batch, time]`, and reduction axis `1`, the selected
mapping is `[0, 1]`: mask axis `0` maps to input axis `0` (`batch`), mask axis `1` maps to input
axis `1` (`time`), and omitted input axis `2` (`features`) is an implicit broadcast dimension. The
result removes `time` and has shape `[batch, features]`.

Every valid call returns a fresh Tensor with the input's exact floating data type and gradient
eligibility, unresolved layout, no label, and no host storage. Its operation carries
`MaskedReductionAttrs(normalizedAxis, mapping)`, and provenance retains exact ordered references
`[input, mask]`. The mask's gradient eligibility does not affect the result.

The semantic meaning excludes false mask positions. Masked sum produces zero when no value is
selected. Masked mean divides each output by its selected true-count and also produces zero when
that count is zero. Expression construction records those rules but does not align storage,
materialize a mask, inspect values, count positions, aggregate, divide, create a gradient rule,
capture a graph, or execute work.

#### Complete masked-reduction example

##### Goal and inputs

Build masked sum and mean metadata for a storage-free `FLOAT32` input shaped
`[batch, time, 4]` and a BOOL mask shaped `[batch, time]`. The example observes mapping, result
shape, and provenance; it does not calculate aggregate values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class MaskedReductionExpressionExample {
    public static void main(String[] args) {
        DynamicDimension batch = new DynamicDimension("batch");
        DynamicDimension time = new DynamicDimension("time");
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(batch, time, new StaticDimension(4)),
                Optional.empty(),
                true));
        Tensor mask = TensorFactory.create(new TensorDescriptor(
                DataType.BOOL,
                Shape.ofDimensions(batch, time),
                Optional.empty(),
                false));

        Tensor sum = input.sum(1, mask);
        Tensor mean = input.mean(-2, mask);
        MaskedReductionAttrs attrs =
                (MaskedReductionAttrs) sum.provenance().orElseThrow().operation().attrs();

        System.out.println("sumShape=" + sum.descriptor().shape());
        System.out.println("meanShape=" + mean.descriptor().shape());
        System.out.println("axis=" + attrs.axis());
        System.out.println("mapping=" + attrs.maskInputAxes());
        System.out.println("orderedInputs="
                + (sum.provenance().orElseThrow().inputs().get(0) == input
                && sum.provenance().orElseThrow().inputs().get(1) == mask));
        System.out.println("metadata="
                + (sum.descriptor().dataType() == DataType.FLOAT32
                && sum.descriptor().requiresGrad()
                && sum.descriptor().layout().isEmpty()
                && sum.label().isEmpty()
                && sum.hostStorage().isEmpty()));
    }
}
```

##### Meaningful lines and intermediate results

- Both `1` and `-2` normalize to input axis `1`, the `time` axis.
- The equal `batch` and `time` dimensions map to input axes `[0, 1]`. Input axis `2`, extent
  `4`, is omitted and broadcasts implicitly.
- Removing input axis `1` produces `[batch, 4]`. The exact `batch` dimension object is retained.
- Provenance stores the input first and mask second. Only the input supplies result type and
  gradient eligibility.

##### Result and interpretation

The program prints:

```text
sumShape=Shape[batch, 4]
meanShape=Shape[batch, 4]
axis=1
mapping=[0, 1]
orderedInputs=true
metadata=true
```

The output proves local axis normalization, deterministic ordered mapping, axis removal, result
metadata, and provenance order. It does not prove mask materialization, selected values, a sum or
mean result, gradient support, compiler capture, backend support, or execution.

##### Failures and useful variations

- A null mask fails with `NullPointerException`. Integral or BOOL input and every non-BOOL mask
  fail with `IllegalArgumentException`; type checks occur before axis validation.
- An invalid axis fails with `IndexOutOfBoundsException`, including every axis for a scalar input.
  Axis validation occurs before mask-rank and alignment validation.
- Mask rank greater than input rank, or dimensions without a locally provable ordered mapping,
  fail with `IllegalArgumentException`. No cast, symbolic constraint, or reshape is inserted.
- Scalar masks, equal-rank masks, zero extents, equal dynamic symbols, and mask-side singleton
  dimensions are supported structurally. Repeated valid calls remain fresh expressions.

### Boolean aggregate expressions

The six current boolean aggregate methods build conjunction (`all`) or disjunction (`any`)
reduction metadata from one exact BOOL input:

| Family | Full form | Axis removed | Axis optionally retained |
|---|---|---|---|
| Boolean conjunction | `all()` | `all(axis)` | `all(axis, keepDimensions)` |
| Boolean disjunction | `any()` | `any(axis)` | `any(axis, keepDimensions)` |

Every full form selects all input axes, records `NoOperationAttrs.INSTANCE`, and returns a fresh
canonical rank-zero BOOL scalar. An axis form accepts a positive or negative axis and normalizes it
against the input `Shape`. A false `keepDimensions` removes the selected axis; true replaces only
that axis with a new static extent of one. Removing the sole axis of a rank-one input also returns
the canonical scalar shape. Every unaffected static or dynamic `Dimension` object is retained by
exact reference.

Every result has exact `DataType.BOOL`, false gradient eligibility, unresolved layout, no label,
and no host storage. Provenance records `AggregateReductionKind.ALL` or
`AggregateReductionKind.ANY` with exactly the receiver as its one input. This is different from
elementwise `logicalAnd(Tensor)` and `logicalOr(Tensor)`: those methods record
`BooleanLogicalKind.AND` or `BooleanLogicalKind.OR`, retain two ordered inputs, and broadcast
corresponding positions instead of reducing a domain. Equal words such as “and” and “all” describe
related truth concepts, but their typed operation identities and provenance arities are distinct.

Expression construction accepts scalar, ordinary static, zero-extent, and dynamic shapes without
reading truth bytes or element counts. It does not short-circuit, evaluate conjunction or
disjunction, decide whether an empty conjunction is true or an empty disjunction is false, define
gradient behavior, capture or optimize a graph, report backend support, or execute work.

Failure behavior is local and deterministic:

- `all` and `any` accept only exact BOOL input. Every floating or integral input fails with
  `IllegalArgumentException`; no numeric truthiness or implicit cast is used. Type validation
  occurs before axis validation and result identity allocation.
- An axis outside `[-rank, rank - 1]` fails with `IndexOutOfBoundsException`. Every axis is invalid
  for a scalar input, while a full scalar reduction is valid and returns a fresh scalar.
- Repeated and nested calls always create fresh expression tensors; no truth simplification,
  canonicalization, or common-subexpression elimination occurs here.

### Arg-max expressions

The three current `argMax` methods build index-producing aggregate metadata along exactly one
input axis:

| Public form | Retained dimension | Tie policy |
|---|---|---|
| `argMax(axis)` | Removed | Explicitly supplies `FIRST_INDEX` |
| `argMax(axis, keepDimensions)` | Caller choice | Explicitly supplies `FIRST_INDEX` |
| `argMax(axis, keepDimensions, tiePolicy)` | Caller choice | Exact non-null caller policy |

The two convenience forms choose the smallest logical index among equal maxima by passing
`ArgMaxTiePolicy.FIRST_INDEX` to the shared construction boundary. This is a public-overload
convenience, not an implicit default in `ArgMaxAttrs`: the semantic attribute record continues to
require an explicit non-null policy. The complete overload also accepts `LAST_INDEX`, which
requests the largest logical index among equal maxima.

Every form accepts `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, or `INT64` input and rejects BOOL;
numeric inputs are neither promoted nor converted. The positive or negative caller axis is
normalized exactly once against the input `Shape`. A false `keepDimensions` removes the selected
axis; true replaces only that axis with a new static extent of one. Removing the sole axis of a
rank-one tensor produces the canonical scalar shape. Every unaffected static or dynamic
`Dimension` object is retained by exact reference.

Every result is a fresh `DataType.INT64` Tensor with false gradient eligibility, unresolved
layout, no label, and no host storage. Provenance records
`AggregateReductionKind.ARG_MAX`, one `ArgMaxAttrs(normalizedAxis, keepDimensions, tiePolicy)`,
and exactly the receiver as its sole input. Repeated or nested calls are not simplified or
canonicalized.

Construction accepts dynamic dimensions and zero extents structurally, including a zero selected
axis, without claiming that a maximum index exists. It does not read storage, compare values,
select an index, or define ordering for NaN, signed zero, infinity, or equality. Empty-axis
behavior, gradients, graph capture, compiler canonicalization, backend support, and execution
remain deferred to their owning layers.

Failure behavior is deterministic: a null explicit policy fails before input-type or axis
validation; BOOL input fails before axis validation; and an axis outside
`[-rank, rank - 1]` fails with `IndexOutOfBoundsException`, including every axis for a scalar
input. These local failures occur before result identity allocation.

### Cumulative-sum expressions

The two current `cumSum` methods build a shape-preserving cumulative-addition scan along exactly
one input axis:

| Public form | Inclusion | Direction |
|---|---|---|
| `cumSum(axis)` | Inclusive | Forward |
| `cumSum(axis, exclusive, reverse)` | Selected by `exclusive` | Selected by `reverse` |

An axis identifies the dimension whose positions contribute to each cumulative prefix. Forward
traversal visits that axis from lower to higher indices; reverse traversal visits it from higher
to lower indices without reversing the returned dimension or output positions. Inclusive mode
includes the current value. Exclusive mode omits it, so the first position visited in either
direction has an empty prefix whose additive identity is zero.

For logical input `[1, 2, 3]`, the four modes are:

| Call mode | Semantic result | Position-by-position interpretation |
|---|---|---|
| `cumSum(axis)` or `(false, false)` | `[1, 3, 6]` | Forward prefixes are `1`, `1 + 2`, and `1 + 2 + 3`. |
| `(true, false)` | `[0, 1, 3]` | Excluding each current value leaves the empty prefix, `1`, and `1 + 2`. |
| `(false, true)` | `[6, 5, 3]` | Reverse-inclusive contributions are `1 + 2 + 3`, `2 + 3`, and `3`. |
| `(true, true)` | `[5, 3, 0]` | Reverse-exclusive contributions are `2 + 3`, `3`, and the empty reverse prefix. |

These values explain the requested operation. Current expression construction computes none of
them: it reads no element or host storage and allocates no result storage.

Both methods accept `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, and `INT64`; `BOOL` is rejected
without truthiness or conversion. A positive or negative caller axis is normalized exactly once
against the input `Shape`. Every valid result is a fresh unlabeled, storage-free Tensor whose
descriptor retains the exact input Shape reference, data type, and `requiresGrad` eligibility,
but uses unresolved layout even when the input layout is resolved. Integral inputs necessarily
remain ineligible for gradients under the descriptor contract.

Provenance records `CumulativeSumKind.CUM_SUM`, one
`CumulativeSumAttrs(normalizedAxis, exclusive, reverse)`, and exact ordered inputs `[input]`.
Repeated equivalent requests remain distinct expressions with fresh identities. Construction
does not define accumulation precision, overflow, empty-axis value behavior, a gradient rule,
compiler capture or canonicalization, backend lowering, or execution.

Failure behavior is local and deterministic:

- `BOOL` fails with `IllegalArgumentException` and message
  `input must have a numeric data type, but was BOOL` before axis validation.
- An axis outside `[-rank, rank - 1]` fails with `IndexOutOfBoundsException`; every scalar axis is
  invalid. These validation failures consume no Tensor identity.
- Exhausted factory identity space fails only after descriptor, operation, and provenance
  metadata have been constructed.

### Softmax expressions

The two current softmax methods build shape-preserving normalization metadata along exactly one
input axis:

| Public form | Operation kind | Ideal slice meaning |
|---|---|---|
| `softmax(axis)` | `SoftmaxKind.SOFTMAX` | Positive normalized probabilities whose slice total is one |
| `logSoftmax(axis)` | `SoftmaxKind.LOG_SOFTMAX` | Natural logarithms of the corresponding softmax probabilities |

A normalization slice fixes every coordinate except the selected axis. For a rank-two tensor and
axis `1`, each row is therefore normalized independently. The input must be `FLOAT64`, `FLOAT32`,
or `BFLOAT16`; integral and BOOL inputs are rejected before axis validation. A positive or
negative caller axis is normalized exactly once against the input Shape.

Every valid result is a fresh unlabeled, storage-free Tensor. Its descriptor retains the exact
input Shape reference, data type, and `requiresGrad` value, but its layout is unresolved even when
the input layout is resolved. Provenance records the exact requested kind, one
`SoftmaxAttrs(normalizedAxis)`, and exact ordered inputs `[input]`. Construction does not inspect
input values or storage, calculate exponentials, logarithms, sums, or probabilities, select a
finite-precision algorithm, define a gradient rule, capture a graph, decompose the operation, or
execute backend work.

For one ideal slice `[1, 2, 3]`, the requested mathematical values are:

| Method | Approximate result | Relationship |
|---|---|---|
| `softmax(axis)` | `[0.09003057, 0.24472847, 0.66524096]` | The probabilities sum to approximately one. |
| `logSoftmax(axis)` | `[-2.40760596, -1.40760596, -0.40760596]` | Exponentiating each value recovers the corresponding softmax probability. |

These numbers explain the expression's meaning; the current API constructs only metadata for the
request.

#### Complete expression-construction example

##### Goal and inputs

Construct softmax and log-softmax expressions for a storage-free `FLOAT32` tensor with Shape
`[2, 3]`. Axis `-1` and axis `1` both select the final dimension, so each logical row is one
normalization slice.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class SoftmaxExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));

        Tensor probabilities = input.softmax(-1);
        Tensor logProbabilities = input.logSoftmax(1);
        SoftmaxAttrs probabilitiesAttrs = (SoftmaxAttrs) probabilities
                .provenance().orElseThrow().operation().attrs();

        System.out.println(probabilities.descriptor().shape() == shape);
        System.out.println(probabilities.descriptor().dataType());
        System.out.println(probabilities.descriptor().requiresGrad());
        System.out.println(probabilities.descriptor().layout().isEmpty());
        System.out.println(probabilities.provenance().orElseThrow().operation().kind());
        System.out.println(logProbabilities.provenance().orElseThrow().operation().kind());
        System.out.println(probabilitiesAttrs.axis());
        System.out.println(probabilities.provenance().orElseThrow().inputs().getFirst() == input);
    }
}
```

##### Meaningful lines and intermediate results

- `input.softmax(-1)` normalizes `-1` to axis `1` and records `SOFTMAX`.
- `input.logSoftmax(1)` stores the same normalized axis but records the distinct
  `LOG_SOFTMAX` kind.
- Both descriptors retain the exact `shape`, `FLOAT32`, and the true gradient-eligibility
  metadata while leaving layout unresolved.
- Each provenance value contains exactly `input`; neither expression receives label or storage.

##### Result and interpretation

The program prints:

```text
true
FLOAT32
true
true
SOFTMAX
LOG_SOFTMAX
1
true
```

The result proves axis normalization, kind selection, descriptor retention, unresolved layout,
and one-input provenance. It does not prove the ideal numerical values above, gradient support,
compiler capture or decomposition, backend support, or execution.

##### Failures and useful variations

- `INT32`, `INT64`, and `BOOL` input fails with `IllegalArgumentException`; no conversion or
  promotion is inserted. This type check occurs before axis validation.
- An axis outside `[-rank, rank - 1]` fails with `IndexOutOfBoundsException`. Every axis is invalid
  for a scalar input.
- Dynamic and zero extents are accepted structurally when the selected axis exists. Repeated
  equivalent requests still return fresh expression identities.

### Contiguous expressions

The current parameterless `Tensor.contiguous()` method requests canonical dense row-major result
geometry for one logical input. It accepts all six data types and preserves the exact input Shape
reference, data type, and `requiresGrad` value. Every valid call creates a fresh unlabeled,
storage-free Tensor with `CONTIGUOUS`, `NoOperationAttrs.INSTANCE`, and exact ordered provenance
`[input]`.

For a fully static Shape, construction creates a new resolved `LayoutDescriptor` with canonical
row-major strides, logical storage offset zero, `isView() == false`, and the checked referenced
element span. A rank-zero scalar therefore has no strides and span one; any zero-extent Shape has
span zero. For a Shape containing a dynamic dimension, layout remains unresolved because numeric
strides and span cannot yet be calculated.

Construction does not inspect the input layout, label, provenance, storage, liveness, or values.
An unresolved, already dense, offset-contiguous, strided, or broadcast input therefore follows the
same Shape-based result rule. Repeated requests and a request on an already-contiguous expression
remain distinct until a later compiler proves a legal canonicalization. Resolved result geometry
describes the requested logical representation; it does not prove allocation, copying, distinct
physical storage, runtime residency, backend support, or execution.

#### Complete expression-construction example

##### Goal and inputs

Construct contiguous expressions from a static strided `FLOAT32` tensor and a dynamic `INT64`
tensor. Neither input needs host storage because this boundary creates expression metadata only.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class ContiguousExpressionExample {
    public static void main(String[] args) {
        Shape staticShape = Shape.of(2, 3);
        LayoutDescriptor strided = LayoutDescriptor.of(
                staticShape, new long[] {1, 2}, 4, true);
        Tensor staticInput = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, staticShape, Optional.of(strided), true));

        Shape dynamicShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));
        Tensor dynamicInput = TensorFactory.create(new TensorDescriptor(
                DataType.INT64, dynamicShape, Optional.empty(), false));

        Tensor staticResult = staticInput.contiguous();
        Tensor dynamicResult = dynamicInput.contiguous();
        LayoutDescriptor resultLayout = staticResult.descriptor().layout().orElseThrow();

        System.out.println(staticResult.descriptor().shape() == staticShape);
        System.out.println(Arrays.toString(resultLayout.strides()));
        System.out.println(resultLayout.storageOffset());
        System.out.println(resultLayout.isView());
        System.out.println(resultLayout.referencedElementSpan());
        System.out.println(dynamicResult.descriptor().layout().isEmpty());
        System.out.println(staticResult.provenance().orElseThrow().operation().kind());
        System.out.println(staticResult.provenance().orElseThrow().inputs().getFirst()
                == staticInput);
        System.out.println(staticResult.hostStorage().isEmpty());
    }
}
```

##### Meaningful lines and intermediate results

- The input uses non-canonical strides `[1, 2]`, offset `4`, and view metadata, but
  `staticInput.contiguous()` derives result geometry only from `staticShape`.
- The static result retains the exact Shape, `FLOAT32`, and true gradient-eligibility metadata,
  but receives newly constructed canonical strides `[3, 1]`, offset zero, non-view metadata, and
  span six.
- The dynamic result retains its exact Shape and `INT64` metadata while leaving layout unresolved.
- Provenance records exactly `staticInput`; neither result receives label or storage.

##### Result and interpretation

The program prints:

```text
true
[3, 1]
0
false
6
true
CONTIGUOUS
true
true
```

The result demonstrates static and dynamic descriptor derivation plus one-input provenance. It
does not demonstrate a copy, materialization, compiler canonicalization, backend route, or
execution.

##### Failures and useful variations

- Canonical stride or span overflow for a fully static Shape throws `ArithmeticException` before
  a Tensor identity is consumed. Dynamic Shapes do not perform that numeric layout calculation.
- Exhausted Tensor identifier space throws `IllegalStateException` only after local immutable
  descriptor, operation, and provenance metadata has been constructed.
- Scalar and zero-extent Shapes are valid, as are all current data types. Repeated or nested
  requests always return fresh expression identities.

### Reshape expressions

The current `Tensor.reshape(long...)` and `Tensor.reshape(Shape)` methods create fresh model
expressions that preserve the input's ordered logical element sequence under new coordinates.
Both overloads accept all six current data types, retain the exact input data type and
`requiresGrad` value, and return an unlabeled, storage-free Tensor. Provenance contains exactly
`Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(targetShape))` and ordered input
`[input]`. Same-shape, repeated, and nested requests remain explicit fresh expressions.

The raw `long...` overload treats its array as caller-owned input: it neither mutates nor retains
the array. An empty request means the canonical rank-zero scalar Shape. Every ordinary extent must
be non-negative, so zero extents are valid. At most one exact `-1` may request inference; every
value below `-1` is invalid. The sentinel is normalized before `Shape` and `TargetShapeAttrs`
construction and therefore never appears in stored semantic state.

Inference requires a fully known input element count and a non-zero checked product of all other
requested extents. The input count must be exactly divisible by that product. A zero input count
can therefore infer zero when the other requested product is non-zero, such as reshaping `[0, 3]`
with `[-1, 2]` to `[0, 2]`. A request such as `[0, -1]` is rejected because infinitely many
inferred extents satisfy a zero product. Once a zero requested extent is present, that ambiguity is
reported without performing irrelevant multiplication that could overflow elsewhere in the raw
request.

The exact-Shape overload accepts a non-null already normalized scalar, zero-extent, static, mixed
dynamic, or fully dynamic `Shape` and retains that exact immutable reference in both the result
descriptor and `TargetShapeAttrs`. For either overload, known input and target element counts must
match. If either Shape has a dynamic dimension, the equality constraint is accepted and deferred
to later compiler validation; this method neither binds symbols nor assumes that the counts match.

Result layout is resolved only when two facts are locally available: the input descriptor has
resolved contiguous geometry, including offset-contiguous geometry, and the target Shape is fully
static. That branch creates a new view-marked layout with canonical target row-major strides and
the exact input element offset. Offset zero classifies as `DENSE_CONTIGUOUS`; non-zero offset
classifies as `DENSE_WITH_OFFSET`. Unresolved, strided, or broadcast input geometry, or a dynamic
target, produces unresolved result layout. Reshape does not insert `contiguous()`, access or attach
storage, copy values, or decide an executable alias-versus-materialization route. Even resolved
view metadata is only a logical model fact and does not promise zero-copy execution.

#### Complete expression-construction example

##### Goal and inputs

Normalize one inferred raw request against a static offset-contiguous input, then retain one exact
dynamic target Shape. Both calls observe expression metadata only; no storage is needed.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class ReshapeExpressionExample {
    public static void main(String[] args) {
        Shape inputShape = Shape.of(2, 3);
        LayoutDescriptor inputLayout = LayoutDescriptor.of(
                inputShape, new long[] {3, 1}, 5, true);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, inputShape, Optional.of(inputLayout), true));

        long[] request = {3, -1};
        Tensor staticResult = input.reshape(request);
        request[0] = 99;

        Shape dynamicTarget = Shape.ofDimensions(
                new DynamicDimension("items"), new StaticDimension(2));
        Tensor dynamicResult = input.reshape(dynamicTarget);
        LayoutDescriptor resultLayout = staticResult.descriptor().layout().orElseThrow();

        System.out.println(staticResult.descriptor().shape());
        System.out.println(Arrays.toString(resultLayout.strides()));
        System.out.println(resultLayout.storageOffset());
        System.out.println(resultLayout.isView());
        System.out.println(staticResult.provenance().orElseThrow().operation().kind());
        System.out.println(staticResult.provenance().orElseThrow().inputs().getFirst() == input);
        System.out.println(dynamicResult.descriptor().shape() == dynamicTarget);
        System.out.println(dynamicResult.descriptor().layout().isEmpty());
        System.out.println(staticResult.hostStorage().isEmpty());
    }
}
```

##### Meaningful lines and intermediate results

- Input Shape `[2, 3]` has six elements. In raw request `[3, -1]`, the known product is three, so
  the inferred extent is `6 / 3 = 2` and the normalized target is `[3, 2]`.
- Mutating `request` after the call cannot change the result because raw array ownership remains
  with the caller and the normalized Shape owns its immutable dimensions.
- The input layout is contiguous at element offset five. The static result therefore receives new
  canonical target strides `[2, 1]`, retains offset five, and is explicitly marked as a view.
- The exact dynamic target is accepted because its element count is not locally known. Its
  reference is retained, while layout remains unresolved.
- Both results retain `FLOAT32` and true gradient eligibility, record exact one-input `RESHAPE`
  provenance, and have no label or host storage.

##### Result and interpretation

The program prints:

```text
Shape[3, 2]
[2, 1]
5
true
RESHAPE
true
true
true
true
```

The result demonstrates raw inference, defensive ownership, exact dynamic-Shape retention,
conditional same-offset view geometry, and provenance. It does not demonstrate value movement,
storage aliasing, compiler constraint solving or canonicalization, gradients, materialization,
backend lowering, or execution.

##### Failures and useful variations

- A null raw array or exact Shape fails with `NullPointerException` naming `requestedShape` or
  `targetShape`, respectively.
- A raw extent below `-1`, a second `-1`, dynamic-input inference, zero-product inference,
  non-divisible inference, or unequal known counts fails with `IllegalArgumentException`.
- Checked element-count, non-zero requested-product, canonical-stride, and referenced-span
  overflow propagates as `ArithmeticException` before a Tensor identity is consumed.
- Exhausted Tensor identifier space fails with `IllegalStateException` only at final derived
  construction after local validation and metadata construction.
- `reshape()` requests a scalar and succeeds only when the input has one known element or when
  compatibility is dynamic. `reshape(-1, 2)` on a zero-element input infers `[0, 2]`, while
  `reshape(0, -1)` is ambiguous and fails.

### Expand expressions

The current `Tensor.expand(long...)` and `Tensor.expand(Shape)` methods create fresh model
expressions that logically repeat compatible input positions into an exact target Shape. Both
overloads accept every current data type, retain the exact input data type and `requiresGrad`
value, and return an unlabeled, storage-free Tensor. Provenance contains exactly
`Operation(ShapeTransformKind.EXPAND, new TargetShapeAttrs(targetShape))` and ordered input
`[input]`. Identity-like, repeated, and nested calls remain explicit fresh expressions.

Expansion is directional rather than symmetric broadcasting. Input and target axes align from
the right. Each aligned input dimension must equal its target dimension structurally, or the input
dimension must be the static singleton `1`. Extra leading target axes are valid because the
missing input axes behave as logical singletons. The target cannot remove axes or shrink an input
dimension. Equal dynamic symbols and a static input singleton expanding to a dynamic target are
locally provable; unequal symbols and a dynamic input paired with another dimension are rejected
without binding symbols or recording hidden constraints.

The raw `long...` overload passes the caller-owned dimensions through `Shape.of(long...)`; the
array is not retained or mutated. Every value is a literal non-negative extent, including zero,
and an empty request denotes scalar Shape. Unlike reshape, numeric `-1` has no inference meaning
and is rejected. The `Shape` overload instead retains the exact immutable target reference,
including scalar, zero-extent, mixed dynamic, and fully dynamic Shapes when compatibility is
provable.

For a fully static target and any resolved input layout, expand publishes a new logical view
layout. It preserves the input element offset and the exact aligned stride for unchanged axes,
including an existing zero or non-canonical stride. New leading axes and changed input-singleton
axes receive stride zero. Dynamic targets or unresolved input layouts leave result layout
unresolved. The derived view metadata does not attach or alias host storage, repeat values, choose
materialization, or promise zero-copy execution.

#### Complete expression-construction example

##### Goal and inputs

Expand an offset-contiguous `[1, 3]` input to `[2, 4, 3]`, then retain an exact dynamic target for
a separate singleton input. The example observes descriptors and provenance only.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class ExpandExpressionExample {
    public static void main(String[] args) {
        Shape inputShape = Shape.of(1, 3);
        LayoutDescriptor inputLayout = LayoutDescriptor.of(
                inputShape, new long[] {3, 1}, 5, true);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, inputShape, Optional.of(inputLayout), true));

        long[] request = {2, 4, 3};
        Tensor staticResult = input.expand(request);
        request[0] = 99;
        LayoutDescriptor resultLayout = staticResult.descriptor().layout().orElseThrow();

        Tensor singleton = TensorFactory.create(new TensorDescriptor(
                DataType.INT64, Shape.of(1), Optional.empty(), false));
        Shape dynamicTarget = Shape.ofDimensions(new DynamicDimension("items"));
        Tensor dynamicResult = singleton.expand(dynamicTarget);

        System.out.println(staticResult.descriptor().shape());
        System.out.println(Arrays.toString(resultLayout.strides()));
        System.out.println(resultLayout.storageOffset());
        System.out.println(resultLayout.kind());
        System.out.println(resultLayout.referencedElementSpan());
        System.out.println(resultLayout.isView());
        System.out.println(staticResult.provenance().orElseThrow().operation().kind());
        System.out.println(staticResult.provenance().orElseThrow().inputs().getFirst() == input);
        System.out.println(dynamicResult.descriptor().shape() == dynamicTarget);
        System.out.println(dynamicResult.descriptor().layout().isEmpty());
        System.out.println(staticResult.hostStorage().isEmpty());
    }
}
```

##### Meaningful lines and intermediate results

- Right alignment maps input `[1, 3]` to the final two target axes in `[2, 4, 3]`. The new leading
  extent `2` and expanded singleton extent `4` receive zero strides; the unchanged extent `3`
  preserves stride one. The result strides are therefore `[0, 0, 1]`.
- The input offset five is preserved. The greatest referenced element index is seven, so the
  referenced element span is eight even though the result has 24 logical positions.
- Mutating the raw request cannot change the normalized target Shape.
- Static singleton `[1]` can expand to dynamic `[items]` locally. The exact target reference is
  retained, while layout remains unresolved because the target extent is not numeric.
- Both results record one-input `EXPAND` provenance and receive no label or host storage.

##### Result and interpretation

The program prints:

```text
Shape[2, 4, 3]
[0, 0, 1]
5
BROADCAST_ZERO_STRIDE
8
true
EXPAND
true
true
true
true
```

The result demonstrates directional compatibility, defensive raw-array ownership, exact dynamic-
Shape retention, preserved and zero strides, same-offset view geometry, and provenance. It does
not demonstrate value repetition, storage aliasing, gradients, compiler capture or constraints,
materialization, backend lowering, or execution.

##### Failures and useful variations

- A null raw array or exact Shape fails with `NullPointerException` naming `requestedShape` or
  `targetShape`, respectively.
- A negative raw extent, lower-rank target, attempted shrink, unequal aligned dimensions, unequal
  dynamic symbols, or another unprovable dynamic pair fails with `IllegalArgumentException`.
- Resolved layout arithmetic overflow propagates as `ArithmeticException` before a Tensor identity
  is consumed.
- Exhausted Tensor identifier space fails with `IllegalStateException` only at final derived
  construction after local validation and metadata construction.
- Scalar input may expand to any target Shape. Static or dynamic identity-like requests remain
  fresh expressions rather than returning the input.

### Permute and transpose expressions

The current `Tensor.permute(int...)` method creates a fresh model expression for one complete axis
reordering. The requested array uses output-to-input order: entry `i` identifies the input axis
placed at output axis `i`. Its length must equal input rank, and its normalized entries must contain
every input axis exactly once. Each negative entry adds the rank once, so rank-three axes
`[1, -1, 0]` normalize to `[1, 2, 0]`. The method copies the array after count validation and never
retains or mutates caller-owned storage. Empty axes form the valid rank-zero scalar permutation.

The result Shape contains the exact immutable input `Dimension` references in reordered positions.
If input layout is resolved, the result receives one new view-marked `LayoutDescriptor` with the
same element offset and exact strides in the same reordered positions. Every resolved layout kind
is accepted, and the descriptor derives the result kind and referenced span from the new geometry.
If input layout is unresolved, including dynamic Shape cases, result layout remains unresolved.

`Tensor.transpose()` requires rank two and delegates to the same construction with normalized axes
`[1, 0]`. It records `AxisTransformKind.PERMUTE` and `PermutationAttrs([1, 0])`; there is no separate
transpose kind. Both methods preserve exact data type and gradient eligibility, return a fresh
unlabeled Tensor without host storage, and record exactly the receiver as their provenance input.
Identity, inverse, repeated, and nested requests remain explicit expressions.

Resolved view layout is logical metadata. It does not attach or inspect storage, prove a physical
alias, promise zero-copy execution, or choose whether a backend must materialize a copy. Compiler
capture and permutation canonicalization, planning and prepare-time materialization, backend
lowering, gradients, runtime residency, and execution remain planned in their owning layers.

#### Complete expression-construction example

##### Goal and inputs

Permute a resolved rank-three Tensor with one negative raw axis, then construct a rank-two
transpose and inspect their normalized semantics. The example observes descriptors and provenance
only.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class PermuteExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3, 4);
        LayoutDescriptor layout = LayoutDescriptor.of(
                shape, new long[] {12, 4, 1}, 5, true);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(layout), true));

        int[] axes = {1, -1, 0};
        Tensor result = input.permute(axes);
        axes[0] = 0;
        LayoutDescriptor resultLayout = result.descriptor().layout().orElseThrow();
        PermutationAttrs attrs = (PermutationAttrs) result.provenance().orElseThrow()
                .operation().attrs();

        Tensor matrix = TensorFactory.create(new TensorDescriptor(
                DataType.INT64, Shape.of(2, 3), Optional.empty(), false));
        Tensor transposed = matrix.transpose();
        PermutationAttrs transposeAttrs = (PermutationAttrs) transposed.provenance()
                .orElseThrow().operation().attrs();

        System.out.println(result.descriptor().shape());
        System.out.println(Arrays.toString(resultLayout.strides()));
        System.out.println(resultLayout.storageOffset());
        System.out.println(resultLayout.kind());
        System.out.println(resultLayout.referencedElementSpan());
        System.out.println(attrs.axes());
        System.out.println(result.provenance().orElseThrow().inputs().getFirst() == input);
        System.out.println(result.hostStorage().isEmpty());
        System.out.println(transposed.descriptor().shape());
        System.out.println(transposeAttrs.axes());
        System.out.println(transposed.descriptor().layout().isEmpty());
    }
}
```

##### Meaningful lines and intermediate results

- Rank-three raw axes `[1, -1, 0]` normalize once to `[1, 2, 0]`, so input Shape `[2, 3, 4]`
  becomes `[3, 4, 2]`. Mutating the caller array later cannot change stored attributes.
- Input strides `[12, 4, 1]` follow the same output-to-input mapping and become `[4, 1, 12]`.
  Offset five remains unchanged. The greatest referenced index is 28, so the referenced element
  span is 29.
- The resolved result is a logical strided view descriptor. The method does not attach the input's
  storage or claim that execution can consume the view without a copy.
- Transpose swaps `[2, 3]` to `[3, 2]`, records normalized `[1, 0]`, and keeps layout unresolved
  because its input layout was unresolved.

##### Result and interpretation

The program prints:

```text
Shape[3, 4, 2]
[4, 1, 12]
5
STRIDED
29
[1, 2, 0]
true
true
Shape[3, 2]
[1, 0]
true
```

The output demonstrates defensive raw-axis ownership, normalization, Shape/stride reordering,
offset preservation, exact provenance, and rank-two transpose semantics. It does not demonstrate
value movement, physical storage aliasing, gradients, compiler capture or canonicalization,
materialization, backend lowering, or execution.

##### Failures and useful variations

- A null axis array fails with `NullPointerException` naming `requestedAxes`.
- An axis count different from rank, a value outside the rank after one negative normalization, or
  the first duplicated normalized axis fails with `IllegalArgumentException`. These failures occur
  before Tensor identity allocation.
- `transpose()` on any rank other than two fails with `IllegalStateException` reporting the rank.
- Resolved layout classification or referenced-span overflow propagates as `ArithmeticException`
  before identity allocation.
- Exhausted Tensor identifier space fails with `IllegalStateException` only at final derived
  construction after local immutable metadata has been built.
- `scalar.permute()` is valid. Identity permutations still return fresh expressions rather than
  the receiver.

### Expand-dimensions and squeeze expressions

`Tensor.expandDims(int)` inserts one new static dimension of extent one. For input rank `r`, its
caller-facing insertion position is normalized against the result rank `r + 1`: valid raw values
are `[-r - 1, r]`, and a negative value adds `r + 1` once. Rank-two positions `-3` and `0`
therefore insert at the start, while `-1` and `2` insert at the end. Both `-1` and `0` are valid
for a scalar and produce rank one.

`Tensor.squeeze(int)` instead selects an existing input axis through the ordinary Shape range
`[-rank, rank - 1]`. The selected dimension must be a `StaticDimension` whose extent is exactly
one. A zero extent, another static extent, or a dynamic dimension is rejected; expression
construction neither guesses a future dynamic binding nor records a singleton constraint.
Squeezing the only axis of a rank-one Tensor produces the canonical scalar Shape, while every
axis is invalid for an already scalar Tensor.

Both operations preserve the exact data type, gradient eligibility, and immutable references and
order of every unaffected Dimension. Every valid call creates a fresh unlabeled, storage-free
Tensor with one-input provenance. `expandDims` records `AxisTransformKind.EXPAND_DIMS` and an
`AxisTransformAttrs` containing the normalized output insertion position; `squeeze` records
`AxisTransformKind.SQUEEZE` and the normalized input removal position. Repeated, nested, and
inverse-like requests remain explicit and are not canonicalized at this boundary.

When input layout is unresolved, result layout stays unresolved. For any resolved input layout,
`expandDims` constructs a new same-offset view descriptor by inserting one stride while preserving
all existing strides. The inserted stride is one at the end; otherwise it is the checked product
of the following input stride and extent. `squeeze` constructs a new same-offset view descriptor
by omitting only the selected stride. These descriptors are logical geometry: neither operation
attaches storage, proves a physical alias, promises zero-copy execution, or chooses
materialization.

#### Complete rank-editing example

##### Goal and inputs

Insert a trailing singleton into an unresolved-layout Tensor of Shape `[2, 1, 3]`, and separately
remove its middle singleton. The example observes result Shapes, normalized attributes, and the
storage-free model boundary; it does not edit stored values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class RankEditingExpressionExample {
    public static void main(String[] args) {
        Shape inputShape = Shape.of(2, 1, 3);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, inputShape, Optional.empty(), true));

        Tensor expanded = input.expandDims(-1);
        Tensor squeezed = input.squeeze(-2);
        AxisTransformAttrs expandedAttrs = (AxisTransformAttrs) expanded
                .provenance().orElseThrow().operation().attrs();
        AxisTransformAttrs squeezedAttrs = (AxisTransformAttrs) squeezed
                .provenance().orElseThrow().operation().attrs();

        System.out.println("input=" + input.descriptor().shape());
        System.out.println("expanded=" + expanded.descriptor().shape());
        System.out.println("expandedAxis=" + expandedAttrs.axis());
        System.out.println("squeezed=" + squeezed.descriptor().shape());
        System.out.println("squeezedAxis=" + squeezedAttrs.axis());
        System.out.println("kinds="
                + expanded.provenance().orElseThrow().operation().kind() + ","
                + squeezed.provenance().orElseThrow().operation().kind());
        System.out.println("unresolved="
                + (expanded.descriptor().layout().isEmpty()
                && squeezed.descriptor().layout().isEmpty()));
        System.out.println("storageFree="
                + (expanded.hostStorage().isEmpty()
                && squeezed.hostStorage().isEmpty()));
    }
}
```

##### Meaningful lines and intermediate results

- For rank three, insertion axis `-1` adds the result rank `4` and normalizes to output position
  `3`, producing `[2, 1, 3, 1]` and `AxisTransformAttrs[axis=3]`.
- Existing-axis value `-2` adds input rank `3` and normalizes to input position `1`. That
  dimension is statically one, so removal produces `[2, 3]` and
  `AxisTransformAttrs[axis=1]`.
- Both results preserve `FLOAT32` and true gradient eligibility, retain exact one-input
  provenance, and remain unresolved because the input layout is unresolved.
- Neither result receives a label or host storage. The calls describe rank edits without reading,
  moving, or attaching values.

##### Result and interpretation

The program prints:

```text
input=Shape[2, 1, 3]
expanded=Shape[2, 1, 3, 1]
expandedAxis=3
squeezed=Shape[2, 3]
squeezedAxis=1
kinds=EXPAND_DIMS,SQUEEZE
unresolved=true
storageFree=true
```

The output proves the two different normalization domains, exact rank changes, typed operation
identity, normalized attributes, unresolved layout propagation, and absent result storage. It
does not prove value movement, physical aliasing, compiler canonicalization, gradient rules,
materialization, backend support, or execution.

##### Failures and useful variations

- An expand-dimensions insertion outside `[-rank - 1, rank]` fails with
  `IndexOutOfBoundsException`. An existing squeeze axis outside `[-rank, rank - 1]`, including
  every axis for a scalar, fails through Shape normalization. Both failures occur before result
  identity allocation.
- Squeezing a zero, non-one, or dynamic dimension fails with `IllegalArgumentException` before
  identity allocation because the singleton fact is not statically proven.
- For resolved geometry, insertion-stride multiplication, layout classification, or referenced-
  span overflow fails with `ArithmeticException` before identity allocation.
- A resolved canonical, offset, strided, or broadcast input receives new same-offset view
  geometry. Repeated edits and an expand/squeeze pair still return fresh expression identities.

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
own or close an arena, extend a scope's lifetime, or implement `AutoCloseable`. For an arena-backed
segment, the caller must keep scoped memory alive and obey the JDK's thread-access rules. A
factory-created primitive-array segment instead has an automatic scope that stays alive, keeps its
heap base reachable, and is accessible from any thread. `isAlive()` can become stale immediately
for caller-controlled scopes, and a false result does not hide the segment: `segment()` still
returns the same dead reference, so JDK memory access reports the closed-scope failure.

Writable raw memory can be changed through JDK APIs, but the wrapper supplies no typed element
access, conversion, copying, synchronization, or mutation-version tracking. It chooses no
`ValueLayout`, byte order, or alignment. An exact unaligned slice can therefore be wrapped, but
that acceptance is not a promise that a future typed or backend path can access it without an
unaligned layout or materialization.

`MemorySegmentStorage` uses ordinary object identity. Two wrappers around the same segment and
metadata are still unequal. The wrapper itself does not validate physical capacity against
`Shape`, `TensorDescriptor`, `LayoutDescriptor`, logical element count, offset, or referenced
span. The current `Tensor` association performs the compatibility checks described above without
changing the wrapper's raw-storage contract. `TensorFactory` can attach an existing caller-supplied
borrowed storage object or allocate one exact-span automatic-scope heap segment and pass the wrapper
through that same validation path. Import and population policy, deterministic native-resource
ownership, runtime residency, prepared memory, device buffers, and backend storage remain separate
planned work.

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

Each identifier is an immutable record over one non-negative `long` value. Zero is valid, negative
sentinels are rejected, and different identifier types cannot be interchanged. The records
themselves do not generate or guarantee uniqueness. `TensorFactory` now allocates `TensorId`
values unique among its own calls in one JVM; manually constructed values remain outside that
guarantee. Future graph builders and compiler sessions own graph-local allocation within their
lifecycles.

Graph-local numeric identifiers may be reused by different graph containers. `NodeId` identifies a computation, whereas `ValueId` identifies data flowing between computations. A value can exist without a producing node, one node can produce multiple values, and one value can have multiple consumers.

An implemented `PublicationBinding` associates a public `TensorId` with a `ValueId`. The binding is
standalone and cannot prove that the value belongs to a particular graph; a later compiler-owned
`PublicationPlan` will provide that context. The implemented `Tensor` stores no graph-local IDs
because the same tensor may participate in multiple separately compiled graphs. `OperationId` is
not currently defined: operation semantics occur through graph nodes, and no independent
operation-identity lifecycle has been established.

## Operation semantic foundation

The public operation-foundation contracts live in `io.github.pho001.synaptik.model.operation`.
They separate semantic identity, typed parameters, explicit parameter absence, and immutable
composition while the table also lists the current production families:

| Concept | Current role |
|---|---|
| `OperationKind` | An implemented open interface that identifies which backend-independent computation an operation describes. |
| `OperationAttrs` | An implemented marker interface for the immutable, typed semantic parameters of that computation. |
| `NoOperationAttrs.INSTANCE` | The implemented canonical attribute value for a kind that has no semantic parameters. |
| `Operation` | An implemented immutable descriptor that stores one kind and one `OperationAttrs` value. |
| `BinaryArithmeticKind` | The implemented production enum for seven parameterless tensor-to-tensor arithmetic meanings. |
| `BinaryComparisonKind` | The implemented production enum for six parameterless ordered tensor-to-tensor comparison meanings. |
| `BooleanLogicalKind` | The implemented production enum for parameterless elementwise boolean conjunction, disjunction, and negation meanings. |
| `WhereSelectionKind` | The implemented production enum for one parameterless ternary conditional-selection meaning. |
| `CastKind` | The implemented production enum for one parameterized elementwise data-type conversion meaning. |
| `CastAttrs` | The implemented immutable value for one exact non-null target `DataType`. |
| `AggregateReductionKind` | The implemented production enum for seven ordinary aggregate meanings plus axis-only arg-max. |
| `AxisReductionAttrs` | The implemented immutable normalized-axis and retained-dimension value for ordinary single-axis reductions. |
| `ArgMaxAttrs` | The implemented immutable normalized-axis, retained-dimension, and explicit tie-policy value for arg-max. |
| `ArgMaxTiePolicy` | The implemented `FIRST_INDEX` or `LAST_INDEX` choice for equal maxima. |
| `MaskedReductionAttrs` | The implemented immutable normalized reduction axis and ordered mask-dimension-to-input-axis mapping for masked, axis-removing sum and mean semantics. |
| `CumulativeSumKind` | The implemented production enum whose sole `CUM_SUM` value identifies cumulative addition along one axis. |
| `CumulativeSumAttrs` | The implemented immutable normalized axis, inclusive/exclusive choice, and forward/reverse traversal choice for cumulative sum. |
| `SoftmaxKind` | The implemented production enum for one-axis probability and log-probability normalization meanings. |
| `SoftmaxAttrs` | The implemented immutable normalized-axis value shared by softmax and log-softmax. |
| `ContiguousKind` | The implemented production enum whose sole `CONTIGUOUS` value requests logically equivalent canonical dense row-major, zero-offset result geometry. |
| `ShapeTransformKind` | The implemented production enum for ordered-element-preserving `RESHAPE` and singleton/leading-axis-repeating `EXPAND` meanings. |
| `TargetShapeAttrs` | The implemented immutable normalized target-`Shape` value shared by reshape and expand. |
| `AxisTransformKind` | The implemented production enum for complete axis permutation, singleton-axis insertion, and selected singleton-axis removal meanings. |
| `PermutationAttrs` | The implemented immutable complete normalized output-to-input axis permutation. |
| `AxisTransformAttrs` | The implemented immutable normalized non-negative position shared by singleton-axis insertion and removal. |
| `UnaryElementwiseKind` | The implemented production enum for fifteen parameterless unary elementwise meanings. |
| `ScalarElementwiseKind` | The implemented production enum for five parameterized one-input scalar elementwise meanings. |
| `ScalarValueAttrs` | The implemented immutable value for one exact Java `double` scalar parameter. |
| `ClampRangeAttrs` | The implemented immutable value for exact ordered inclusive clamp bounds. |

`OperationKind` declares only `name()`. The result must be a stable, non-null, non-blank diagnostic name, but it is not a serialization token or a global lookup key. Enum-based kind families are the expected common implementation because an enum already supplies a stable `name()`, equality, and hashing. Two enum values from different kind families remain different typed values even if both names contain the same text.

`OperationAttrs` deliberately declares no methods. A family-specific implementation defines its own typed fields and must be immutable, defensively isolate mutable constructor inputs, and provide structural equality and hashing. `NoOperationAttrs.INSTANCE` makes the absence of parameters an explicit non-null value rather than `null` or an empty map. These are implementation contracts, not runtime validators: the interfaces themselves cannot prevent a custom implementation from returning an invalid name or retaining mutable state.

### Binary arithmetic semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind` implements
`OperationKind` with exactly these constants in declaration order:

| Kind | Ordered elementwise meaning |
|---|---|
| `ADD` | Left value plus right value. |
| `SUB` | Left value minus right value. |
| `MUL` | Left value multiplied by right value. |
| `DIV` | Left value divided by right value. |
| `MIN` | Mathematical minimum of the left and right values. |
| `MAX` | Mathematical maximum of the left and right values. |
| `POW` | Left value as the base raised to the right value as the exponent. |

The enum values are semantic identities only. All seven have no intrinsic parameters and compose
explicitly with the existing generic descriptor:

```java
Operation addition = new Operation(
        BinaryArithmeticKind.ADD,
        NoOperationAttrs.INSTANCE);
```

The result is an immutable operation description whose kind is exactly
`BinaryArithmeticKind.ADD` and whose attributes value is exactly the canonical singleton. It is
not a Tensor result, provenance value, graph occurrence, executable operation, or backend-support
claim. No family factory or implicit attributes assignment is provided.

Broadcast geometry is derived from operand shapes rather than stored on the enum or in operation
attributes. The current public Tensor methods own local floating eligibility, promotion,
broadcasting, result-descriptor construction, and ordered provenance. Numerical edge behavior,
gradient rules, compiler capture, execution, and backend availability remain deferred to their
owning contracts. The inherited enum names and text are stable diagnostics, not serialization
tokens, registry keys, or string-dispatch contracts. Equality remains typed: an `ADD` value
declared by another kind family is not equal to `BinaryArithmeticKind.ADD` even though both
diagnostics may contain `ADD`.

### Binary comparison semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind` implements
`OperationKind` with exactly these constants in declaration order:

| Kind | Ordered elementwise meaning |
|---|---|
| `GREATER_THAN` | The left value is strictly greater than the right value. |
| `GREATER_OR_EQUAL` | The left value is greater than or equal to the right value. |
| `LESS_THAN` | The left value is strictly less than the right value. |
| `LESS_OR_EQUAL` | The left value is less than or equal to the right value. |
| `EQUAL` | The left and right values compare equal. |
| `NOT_EQUAL` | The left and right values compare unequal. |

All six kinds are parameterless semantic identities. They compose explicitly with the generic
operation descriptor:

```java
Operation comparison = new Operation(
        BinaryComparisonKind.GREATER_THAN,
        NoOperationAttrs.INSTANCE);
```

The resulting operation retains the exact `GREATER_THAN` kind and canonical no-attributes value.
It describes ordered comparison meaning only; it stores no operands or broadcast plan and is not
a Tensor result, provenance value, graph occurrence, or executable operation. The generic
`Operation` descriptor does not enforce the documented parameterless pairing.

The current public comparison methods separately own floating-pair validation, right-aligned local
broadcasting, fixed `BOOL` result derivation with false `requiresGrad`, and ordered provenance
construction. NaN, infinity, signed-zero, tolerance, and mixed-precision numerical behavior,
gradient policy beyond the descriptor flag, compiler capture, execution, and backend availability
remain planned. Inherited enum names and text are diagnostic rather than serialization or dispatch
keys. Typed identity keeps an equally named kind from another family unequal to
`BinaryComparisonKind`.

### Boolean logical semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind` implements
`OperationKind` with exactly these constants in declaration order:

| Kind | Elementwise boolean meaning | Logical input roles |
|---|---|---|
| `AND` | True exactly when both input values are true. | Ordered left and right inputs. |
| `OR` | True exactly when at least one input value is true. | Ordered left and right inputs. |
| `NOT` | True exactly when the input value is false. | One input. |

All three kinds are parameterless semantic identities. Their explicit generic composition is:

```java
Operation conjunction = new Operation(
        BooleanLogicalKind.AND,
        NoOperationAttrs.INSTANCE);
```

The resulting operation retains the exact `AND` kind and canonical no-attributes value. The
generic `Operation` descriptor does not validate that `AND` and `OR` have two logical inputs or
that `NOT` has one; these roles are family context rather than stored arity metadata. The enum
stores no input references and creates no Tensor, provenance, or result descriptor.

This vocabulary defines conjunction, disjunction, and negation truth meaning only. The current
`Tensor.logicalAnd`, `logicalOr`, and `logicalNot` methods separately own exact BOOL eligibility,
binary broadcasting, unary shape preservation, fixed BOOL result construction, and provenance.
BOOL storage encoding, numeric truthiness, gradients, compiler capture, execution, and backend
availability remain outside both semantic identity and current expression construction. Inherited
enum names and text are diagnostic rather than serialization or dispatch keys, and an equally
named kind from another family remains a different typed value.

### Where selection semantic kind

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind` implements
`OperationKind` with exactly one constant:

| Kind | Elementwise meaning | Logical input roles |
|---|---|---|
| `WHERE` | Choose the true-branch value when the corresponding condition is true; otherwise choose the false-branch value. | Ordered condition, true branch, and false branch. |

`WHERE` has no intrinsic parameters. Its explicit generic composition is:

```java
Operation selection = new Operation(
        WhereSelectionKind.WHERE,
        NoOperationAttrs.INSTANCE);
```

The resulting operation retains the exact `WHERE` kind and canonical no-attributes value. The
three logical roles are ternary family context rather than stored arity metadata: the enum stores
no inputs, and generic `Operation` validates neither the input count nor family-specific
kind-to-attributes compatibility.

This semantic identity describes elementwise conditional choice, not scalar-index `select`,
gather, take, or scatter. The later indexing family owns those distinct capabilities. The enum
does not itself define condition and branch eligibility, branch promotion, three-way broadcasting,
result descriptors, provenance, evaluation order, gradients, compiler capture, ONNX mapping,
execution, or backend availability. The current static `Tensor.where` expression method separately
owns local BOOL/floating validation, branch promotion, ordered pairwise broadcasting, descriptor
construction, and exact three-input provenance. Its inherited name and text are diagnostic rather
than serialization or dispatch keys, and an equally named kind from another family remains a
different typed value.

### Cast semantic kind and attributes

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind` implements `OperationKind`
with exactly one constant:

| Kind | Elementwise meaning | Required attributes |
|---|---|---|
| `CAST` | Convert each value from one logical input to a requested target data type. | `CastAttrs` |

`CastAttrs` has exactly one component, `targetDataType`. It accepts and retains every current
`DataType`: `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL`. The source data type is
not duplicated in the attributes because it belongs to the later input Tensor or graph-value
descriptor. The explicit generic composition is:

```java
Operation cast = new Operation(
        CastKind.CAST,
        new CastAttrs(DataType.FLOAT64));
```

The resulting descriptor retains the exact kind and attributes values. Generic `Operation`
checks only that both components are non-null; it does not enforce the documented
`CAST`/`CastAttrs` pairing. A null target fails during `CastAttrs` construction with
`NullPointerException` and message `targetDataType`.

One logical input and elementwise conversion are family context rather than stored arity or input
state. The current `Tensor.cast(DataType)` method separately owns source-descriptor inspection,
exact shape retention, unresolved result layout, floating-only gradient eligibility, fresh
same-type identity, and one-input provenance. Neither the semantic pair nor expression
construction defines numerical conversion rules, gradient rules, compiler capture, execution, or
backend availability. Accepting every target type is a representability contract, not a promise
that every backend implements every conversion. Enum and record text remain diagnostic rather than
serialization or dispatch contracts.

### Aggregate reduction semantic kinds and attributes

The public enum
`io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind` implements
`OperationKind` with exactly these constants in declaration order:

| Kind | Requested aggregate meaning | Supported semantic forms |
|---|---|---|
| `SUM` | Add values in the selected reduction domain. | Ordinary full/one-axis forms or a masked, axis-removing form. |
| `MEAN` | Compute the arithmetic mean in the selected reduction domain. | Ordinary full/one-axis forms or a masked, axis-removing form. |
| `PROD` | Multiply values in the selected reduction domain. | Full or one normalized axis. |
| `MIN` | Select the minimum value in the selected reduction domain. | Full or one normalized axis. |
| `MAX` | Select the maximum value in the selected reduction domain. | Full or one normalized axis. |
| `ALL` | Compute boolean conjunction in the selected reduction domain. | Full or one normalized axis. |
| `ANY` | Compute boolean disjunction in the selected reduction domain. | Full or one normalized axis. |
| `ARG_MAX` | Select a logical index of a maximum value along one axis. | One normalized axis with an explicit tie policy. |

A full ordinary reduction combines every input axis and pairs the kind with
`NoOperationAttrs.INSTANCE`. A single-axis ordinary reduction pairs the same kind with
`AxisReductionAttrs(axis, keepDimensions)`. This distinction represents "all axes" through the
absence of axis parameters rather than a negative numeric sentinel:

```java
Operation fullSum = new Operation(
        AggregateReductionKind.SUM,
        NoOperationAttrs.INSTANCE);

Operation axisSum = new Operation(
        AggregateReductionKind.SUM,
        new AxisReductionAttrs(1, true));
```

The axis stored in `AxisReductionAttrs` is already normalized: it is a non-negative index, not a
caller-facing negative axis. Construction accepts any non-negative `int` because these attributes
do not retain a `Shape` or know an input rank. A negative value fails with
`IllegalArgumentException` and a message containing the rejected value. The current `sum`, `mean`,
`prod`, reduction `min`, reduction `max`, boolean `all`, and boolean `any` Tensor expressions
and the current `argMax` expressions normalize a caller axis through the input shape before
creating attributes.

When `keepDimensions` is false, the selected axis is removed from the eventual result. When it is
true, the axis remains with extent one. The record stores that request but does not construct an
output shape. Full reductions in this contract have no `keepDimensions` parameter, and
`ARG_MAX` has no full form.

Arg-max pairs only with `ArgMaxAttrs(axis, keepDimensions, tiePolicy)`. `FIRST_INDEX` requests the
smallest logical index among equal maxima along the selected axis; `LAST_INDEX` requests the
largest. A logical index is an axis position rather than a storage offset. The policy must be
supplied explicitly and is never defaulted by the semantic value:

```java
Operation argMax = new Operation(
        AggregateReductionKind.ARG_MAX,
        new ArgMaxAttrs(1, false, ArgMaxTiePolicy.FIRST_INDEX));
```

`ArgMaxAttrs` rejects a negative axis before checking the tie policy and rejects a null policy
with `NullPointerException("tiePolicy")`. Both attribute records use generated record value
semantics, while their text remains diagnostic rather than serialization or dispatch syntax.
Generic `Operation` still checks only that kind and attributes are non-null; it does not enforce
these documented family pairings.

A masked sum or mean pairs its existing kind with
`MaskedReductionAttrs(axis, maskInputAxes)`. The axis is already normalized, non-negative, and
removed from the eventual result. Element `maskInputAxes[i]` gives the input axis aligned with
mask dimension `i`. For example, mapping `[0, 1]` aligns a mask shaped like `[batch, time]` with
the first two axes of input `[batch, time, features]`; the omitted features axis is an implicit
broadcast dimension. Mapping `[0, 2]` instead aligns the same two mask dimensions with the first
and third input axes. An empty mapping represents a scalar mask.

The mapping must contain non-null, non-negative, strictly increasing input-axis positions. This
preserves mask-dimension order, prevents duplicate target axes, and permits an immutable snapshot
without storing Shapes or an expanded mask. `Integer.MAX_VALUE` is structurally valid because the
attribute cannot know the eventual input rank. Public `Tensor.sum(axis, mask)` and
`Tensor.mean(axis, mask)` separately resolve and validate input-axis bounds and dimension
compatibility before constructing the attributes.

The public provenance order is `[input, mask]`. False mask positions exclude their aligned input
values. When no values are selected, a masked sum produces zero. A masked mean divides by the true
selected-count for each output and produces zero when that count is zero. These are semantic
requirements for later execution; constructing `MaskedReductionAttrs` does not inspect a mask,
count values, divide, derive an output descriptor, or execute computation. Generic `Operation`
does not enforce that only `SUM` and `MEAN` use this attributes type.

The public Tensor surface now covers every current aggregate semantic kind. `Tensor.sum`, `mean`,
`prod`, reduction `min`, and reduction `max` provide floating full and single-axis expression
construction. Public `Tensor.all` and `Tensor.any` provide the corresponding exact-BOOL forms with fixed
false gradient eligibility. Public `Tensor.argMax` provides axis-only floating/integral input
construction with fixed INT64, false-gradient results and explicit tie semantics. Every current
family derives shapes locally and records exact one-input provenance. Aggregate extrema remain
typed separately from equally named binary elementwise kinds, while aggregate ALL/ANY remain typed
separately from elementwise AND/OR. Neither the semantic values nor current expression
construction defines empty-domain, numerical, or truth-evaluation policy, extrema comparison or
tie behavior, gradients, compiler capture, backend availability, or execution.

### Cumulative-sum semantic kind and attributes

The public enum
`io.github.pho001.synaptik.model.operation.scan.CumulativeSumKind` implements `OperationKind`
with exactly one constant, `CUM_SUM`. It identifies cumulative addition along one logical input
axis. Unlike an aggregate reduction, a cumulative-sum scan preserves one output position for
every input position and therefore preserves the logical shape. The normalized axis and scan mode
are carried by `CumulativeSumAttrs(axis, exclusive, reverse)` rather than stored in the kind.

The semantic pairing is explicit:

```java
CumulativeSumAttrs attrs = new CumulativeSumAttrs(0, true, false);
Operation cumulativeSum = new Operation(CumulativeSumKind.CUM_SUM, attrs);
```

The result is an immutable semantic descriptor for an exclusive forward cumulative sum on
normalized axis zero. Generic `Operation` retains the exact kind and attributes references, but
does not enforce their family pairing, input count, axis bounds for a particular shape, or result
facts. `CumulativeSumAttrs` accepts every non-negative `int`, including `Integer.MAX_VALUE`, and
rejects a negative axis with `IllegalArgumentException` and message
`axis must be non-negative: <axis>`. The current Shape-aware `Tensor.cumSum` methods normalize and
validate a caller-facing axis before constructing these attributes.

For logical input `[1, 2, 3]`, the complete mode table is:

| `exclusive` | `reverse` | Semantic output | Interpretation |
|---|---|---|---|
| `false` | `false` | `[1, 3, 6]` | Forward traversal includes the current value. |
| `true` | `false` | `[0, 1, 3]` | Forward traversal excludes the current value, so the lowest-index output is additive zero. |
| `false` | `true` | `[6, 5, 3]` | Reverse traversal includes the current value and accumulates from higher indices. |
| `true` | `true` | `[5, 3, 0]` | Reverse traversal excludes the current value, so the highest-index output is additive zero. |

The table describes requested mathematics; constructing either the semantic pair or a public
Tensor expression does not execute these additions. Reverse changes traversal direction, not
output order: all four outputs retain the same logical index order as the input. Exclusive mode
excludes only the current value, and its zero is the additive identity at the first traversed
position rather than a stored Tensor or attribute.

The public methods separately own numeric input validation, caller-axis normalization, exact
Shape/type/eligibility retention in an unresolved descriptor, fresh identity, and exact one-input
provenance. Accumulation policy, empty-axis value behavior, numerical edge cases, gradients,
compiler behavior, storage, backend support, and execution remain with later owning contracts.

### Softmax semantic kinds and attributes

The public enum
`io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind` implements `OperationKind`
with exactly `SOFTMAX` and `LOG_SOFTMAX`, in that order. Both meanings have one logical input and
preserve every logical position. `SoftmaxAttrs(axis)` carries the one already normalized,
non-negative axis shared by both kinds.

Fixing every logical coordinate except the selected axis identifies one normalization slice. For
example, in a rank-two input with axis `1`, every row is a separate slice: positions within a row
vary along axis `1`, while the row coordinate remains fixed. Each output position depends on the
complete slice, so these meanings differ from both aggregate reduction, which contracts an axis,
and cumulative sum, which depends on an ordered prefix.

The semantic pairings are explicit:

```java
SoftmaxAttrs attrs = new SoftmaxAttrs(1);
Operation probabilities = new Operation(SoftmaxKind.SOFTMAX, attrs);
Operation logProbabilities = new Operation(SoftmaxKind.LOG_SOFTMAX, attrs);
```

For slice values `x_i`, ideal SOFTMAX at position `i` is
`exp(x_i) / sum_j(exp(x_j))`. The outputs are positive probabilities whose slice total is one.
Ideal LOG_SOFTMAX is the natural logarithm of the corresponding SOFTMAX probability, equivalently
`x_i - log(sum_j(exp(x_j)))`. Thus exponentiating an ideal LOG_SOFTMAX output recovers the
corresponding ideal SOFTMAX output.

For the slice `[1, 2, 3]`, the ideal meanings are approximately:

| Kind | Semantic output | Relationship |
|---|---|---|
| `SOFTMAX` | `[0.09003057, 0.24472847, 0.66524096]` | The three probabilities sum to approximately one. |
| `LOG_SOFTMAX` | `[-2.40760596, -1.40760596, -0.40760596]` | Exponentiating each value yields the corresponding SOFTMAX probability. |

These values explain ideal mathematics, not a finite-precision evaluation algorithm.
`SoftmaxAttrs` accepts every non-negative `int`, including `Integer.MAX_VALUE`, because it stores
no Shape and cannot prove that an axis exists for an eventual input. A negative axis fails with
`IllegalArgumentException` and exact message `axis must be non-negative: <axis>`. Generic
`Operation` retains either kind and the exact attributes reference but does not enforce family
compatibility, one-input arity, rank bounds, or result facts.

The semantic values themselves store no Tensor, Shape, data-type rule, result descriptor,
provenance, numerical policy, gradient, compiler decomposition, storage, backend support, or
executable behavior. Public floating `Tensor.softmax` and `Tensor.logSoftmax` now add Shape-aware
caller-axis normalization, exact shape/type/eligibility retention in unresolved descriptors, and
fresh one-input provenance without changing that semantic boundary.

### Contiguous semantic kind

The public enum
`io.github.pho001.synaptik.model.operation.layout.ContiguousKind` implements `OperationKind` with
exactly one constant, `CONTIGUOUS`. It describes one logical input and requests a result with the
same logical values, Shape, DataType, and row-major element order in canonical dense row-major
geometry with logical storage offset zero. Input count and these preserved facts are semantic
context; the enum stores no input, arity, descriptor, layout geometry, or result state.

The kind has no intrinsic parameters and composes explicitly with the generic descriptor:

```java
Operation contiguous = new Operation(
        ContiguousKind.CONTIGUOUS,
        NoOperationAttrs.INSTANCE);
```

The resulting `Operation` retains the exact kind and canonical no-attributes singleton. Generic
composition checks only that both references are non-null; it does not enforce this family pairing
or validate one-input arity.

`ContiguousKind.CONTIGUOUS` is a computation request.
`LayoutKind.DENSE_CONTIGUOUS` instead classifies resolved strides and zero offset for a static
Shape. The semantic kind therefore neither depends on nor replaces the layout classification. It
does not itself inspect whether an input is already contiguous, construct a `LayoutDescriptor`,
return an alias, allocate or copy storage, derive a materialization requirement, select lowering,
or execute work. Public `Tensor.contiguous()` now supplies the Shape-aware expression boundary:
fully static results receive new canonical geometry, while dynamic results remain unresolved.
Compiler canonicalization, materialization policy, planning, prepare, backend, runtime, training,
and execution behavior remain with their owning later contracts.

Inherited enum names are diagnostic typed vocabulary, not serialization, registry, dispatch,
kernel, or reflection identifiers. An equally named constant from another kind family remains a
different typed value.

### Reshape and expand semantic kinds and target-shape attributes

The public enum
`io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind` implements `OperationKind`
with exactly `RESHAPE` and `EXPAND`, in that order. Both meanings have one logical input and one
exact result `Shape`, carried by `TargetShapeAttrs(targetShape)` rather than stored on the enum.

| Kind | Semantic meaning | Deferred input-dependent decision |
|---|---|---|
| `RESHAPE` | Preserve the ordered logical element sequence while interpreting it through `targetShape` coordinates. | Graph-wide dynamic count constraints and executable alias or copy choice. |
| `EXPAND` | Logically repeat compatible singleton dimensions or add repeated leading dimensions to produce `targetShape`. | Rank and singleton compatibility, symbolic constraints, and zero-stride or materialized geometry. |

The pairings are explicit:

```java
TargetShapeAttrs attrs = new TargetShapeAttrs(Shape.of(3, 2));
Operation reshape = new Operation(ShapeTransformKind.RESHAPE, attrs);
Operation expand = new Operation(ShapeTransformKind.EXPAND, attrs);
```

Both operations retain the exact `attrs` reference. Generic `Operation` checks only that its two
components are non-null; it does not enforce these family pairings or one-input context.
`TargetShapeAttrs` rejects a null target with `NullPointerException("targetShape")`, retains every
non-null immutable `Shape` reference unchanged, and accepts scalar, zero-extent, fully static,
mixed static/dynamic, and fully dynamic Shapes without inspecting an input.

The stored `Shape` is normalized semantic state. Its static dimensions are non-negative and its
dynamic dimensions are explicit symbols, so a raw numeric `-1` reshape-inference sentinel cannot
occur in the attributes. The current raw public reshape boundary normalizes that caller-facing
syntax before constructing this value.

Conceptually, reshaping logical sequence `[a, b, c, d, e, f]` from shape `[2, 3]` to `[3, 2]`
preserves that sequence and changes only its coordinate grouping to `[[a, b], [c, d], [e, f]]`.
Expanding logical row `[a, b, c]` from `[1, 3]` to `[2, 3]` instead repeats it as
`[[a, b, c], [a, b, c]]`. These examples explain the semantic distinction; the implemented enum
and attributes perform neither transformation, compatibility validation, layout construction, nor
value execution. Current `Tensor.reshape` overloads separately perform raw request normalization,
locally provable count validation, result descriptor/layout derivation, and provenance
construction as described under [Reshape expressions](#reshape-expressions). Current
`Tensor.expand` overloads separately validate directional right-aligned compatibility and derive
same-offset zero-stride view geometry when numeric layout facts are resolved, as described under
[Expand expressions](#expand-expressions).

Dynamic constraints, gradients, compiler and planning behavior, materialization, backend support,
and execution remain planned.

### Axis-transform semantic kinds and attributes

The public enum
`io.github.pho001.synaptik.model.operation.layout.AxisTransformKind` implements `OperationKind`
with exactly `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`, in that order. These are one-input semantic
meanings; the enum stores no input, rank, Shape, layout, or result state.

| Kind | Semantic meaning | Required attributes |
|---|---|---|
| `PERMUTE` | Reorder every axis through one complete output-to-input mapping. | `PermutationAttrs` |
| `EXPAND_DIMS` | Insert one extent-one axis at a normalized output position. | `AxisTransformAttrs` |
| `SQUEEZE` | Remove one selected extent-one input axis at a normalized input position. | `AxisTransformAttrs` |

#### Output-to-input order example

##### Goal and inputs

Describe a permutation that swaps the conceptual `rows` and `columns` axes while retaining the
`channels` axis, then show all three exact kind/attributes pairings. The input-axis labels are
`[rows, columns, channels]`, and the normalized permutation is `[1, 0, 2]`.

```java
PermutationAttrs permutation = new PermutationAttrs(List.of(1, 0, 2));
AxisTransformAttrs position = new AxisTransformAttrs(1);

Operation permute = new Operation(AxisTransformKind.PERMUTE, permutation);
Operation expandDims = new Operation(AxisTransformKind.EXPAND_DIMS, position);
Operation squeeze = new Operation(AxisTransformKind.SQUEEZE, position);
```

##### Result and interpretation

`PermutationAttrs.axes()` uses output-to-input order: element `i` names the normalized input axis
occupying output position `i`. The mapping therefore yields output-axis labels
`[columns, rows, channels]`. The three `Operation` values retain their exact supplied attributes.
This example demonstrates semantic coordinate order and composition only; it does not construct a
Tensor, calculate a result Shape or layout, move values, or execute an operation.

##### Failure variation

The permutation must contain every integer in `[0, axes.size())` exactly once. An empty list is the
valid identity permutation for a rank-zero scalar. Construction validates the caller list in
ascending index order and stores an immutable snapshot, so later caller mutation cannot change
the attributes and the returned list cannot be mutated. The record proves completeness against
its own list size, not against a future input Tensor rank.

`AxisTransformAttrs` accepts every non-negative `int`. With `EXPAND_DIMS`, its axis is the output
position where the new singleton is inserted. With `SQUEEZE`, it is the input position selected
for removal. The record has no input rank or dimension, so it cannot prove the insertion/removal
bound or that a squeezed dimension has extent one. The current Shape-aware public expression
boundary owns raw negative-axis normalization and those input-dependent checks.

A parameterless rank-two `transpose()` method is now implemented as a convenience over
`PERMUTE + PermutationAttrs(List.of(1, 0))`; `TRANSPOSE` is not a fourth semantic kind. Generic
`Operation` retains the supplied kind and attributes but does not enforce the documented family
pairings. Public `Tensor.permute` and `transpose` now construct result descriptors, logical view
layout when input geometry is resolved, and exact one-input provenance. Public `expandDims` and
`squeeze` now construct the corresponding rank-edited result descriptors, conditional logical
view layouts, and exact one-input provenance. Gradients, compiler behavior, materialization,
backend support, and execution remain planned.

### Unary elementwise semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind` implements
`OperationKind` with exactly these constants in declaration order:

| Kind | Elementwise meaning |
|---|---|
| `ABS` | Absolute magnitude of the input value. |
| `NEG` | Additive inverse of the input value. |
| `INV` | Multiplicative reciprocal of the input value. |
| `LOG` | Natural logarithm of the input value. |
| `EXP` | Natural exponential of the input value. |
| `ERF` | Gaussian error function of the input value. |
| `SQRT` | Principal square root of the input value. |
| `FLOOR` | Greatest integer-valued result not greater than the input value. |
| `CEIL` | Least integer-valued result not less than the input value. |
| `SIGN` | Negative, zero, or positive sign classification represented numerically. |
| `RELU` | Rectified linear unit of the input value. |
| `SIGMOID` | Logistic sigmoid of the input value. |
| `TANH` | Hyperbolic tangent of the input value. |
| `FAST_EXP` | Explicitly approximate natural exponential request. |
| `FAST_TANH` | Explicitly approximate hyperbolic tangent request. |

All fifteen kinds have one logical input and no intrinsic parameters. One-input arity is family
context rather than stored enum metadata, and the canonical operation composition remains
explicit:

```java
Operation exponential = new Operation(
        UnaryElementwiseKind.EXP,
        NoOperationAttrs.INSTANCE);
```

The enum does not store the input, infer a result descriptor, or create provenance. The current
public unary Tensor methods own those expression-construction rules. `FAST_EXP` and
`FAST_TANH` are distinct approximate semantic requests rather than aliases or backend flags for
`EXP` and `TANH`; their algorithms, accuracy, special-value behavior, differentiation, execution,
and backend availability remain undefined here. Inherited enum names are diagnostic text, not
serialization or dispatch keys, and equally named kinds from another family remain different
typed values.

### Scalar arithmetic and clamp semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind` implements
`OperationKind` with exactly these constants in declaration order:

| Kind | Elementwise meaning | Required attributes |
|---|---|---|
| `MUL` | Input value multiplied by a scalar multiplier. | `ScalarValueAttrs` |
| `POW` | Input value as the base raised to a scalar exponent. | `ScalarValueAttrs` |
| `CLAMP` | Input value constrained to inclusive lower and upper scalar bounds. | `ClampRangeAttrs` |
| `CLAMP_MIN` | Input value constrained to be no lower than a scalar minimum. | `ScalarValueAttrs` |
| `CLAMP_MAX` | Input value constrained to be no greater than a scalar maximum. | `ScalarValueAttrs` |

The scalar parameter or bounds are operation attributes, not additional Tensor inputs. The
generic `Operation` descriptor accepts any non-null kind and attributes value, so it does not
enforce this table. A consumer that understands the scalar family must enforce the pairing when
that layer needs compatibility validation.

`ScalarValueAttrs` accepts every Java `double` and retains its exact binary64 value without
validation, conversion, normalization, or defaulting. `ClampRangeAttrs` retains both bounds
unchanged and rejects only `minValue > maxValue` under the primitive Java comparison. Therefore it
accepts equal bounds, either ordering of signed zeros, ordered infinities, and any range containing
a NaN endpoint. An inverted range fails with `IllegalArgumentException` and the message
`minValue must be less than or equal to maxValue`.

The following example creates semantic descriptors directly; it does not call the matching public
Tensor expression methods or execute an operation:

```java
Operation scaled = new Operation(
        ScalarElementwiseKind.MUL,
        new ScalarValueAttrs(-0.0));

Operation clipped = new Operation(
        ScalarElementwiseKind.CLAMP,
        new ClampRangeAttrs(0.0, 1.0));
```

Both attribute records use standard generated record equality for `double` components. Positive
and negative zero are unequal components. All NaN components compare equal even when the accessors
retain different raw NaN payload bits. Exact retention can be observed with
`Double.doubleToRawLongBits`; record equality does not imply raw-bit equality.

One-input arity is family context rather than enum metadata. The kinds and attributes alone do not
infer result descriptors, define scalar conversion, establish numerical or special-value execution
behavior, create provenance, define gradients, report backend support, select kernels, or execute
computation. Their enum names and generated record text are diagnostics, not serialization or
dispatch contracts. The current public scalar Tensor methods separately own floating-input
validation, descriptor derivation, exact attribute composition, and one-input provenance.

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

The example demonstrates the implemented descriptor's construction, ownership, and record value
semantics for an attributes-bearing test value. It does not establish that `SampleAttrs` is
compatible with `SampleKind`: the generic descriptor checks only that neither component is null.
It also does not create a graph node, infer a result shape or data type, report backend support,
select a kernel, or execute computation. Production `BinaryArithmeticKind`,
`BinaryComparisonKind`, `BooleanLogicalKind`, `WhereSelectionKind`, and `UnaryElementwiseKind`
provide parameterless families, and
`ScalarElementwiseKind` provides a parameterized family with `ScalarValueAttrs` and
`ClampRangeAttrs`; `CastKind` provides the parameterized cast family with `CastAttrs`. Additional
kind families, compiler behavior, and executable support remain later work in their owning layers.
Binary arithmetic, binary comparison, unary elementwise, and
scalar elementwise semantics have current public Tensor expression methods, as do boolean logical
and conditional-selection semantics. Cast semantics also have current public Tensor expression
construction.

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

- random Operations and typed tensor access or export;
- native, mapped, runtime, and backend allocation with deterministic resource ownership;
- expression families beyond binary arithmetic, binary comparison, boolean logical, conditional
  selection, cast, unary elementwise, scalar elementwise, the current value- and
  index-producing aggregate operations, cumulative-sum scans, softmax normalization, and
  contiguous, reshape, expand, permute, expand-dimensions, and squeeze requests, plus gradient and
  trainable state and publication behavior;
- operation-kind families beyond the current binary arithmetic, binary comparison, boolean
  logical, unary elementwise, scalar elementwise, conditional selection, cast, aggregate
  reduction, cumulative-sum scan, softmax normalization, contiguous-request, reshape/expand, and
  axis-transform semantics, plus family-specific attribute values beyond those documented above;
- compiler entry points and transformations, compiler-owned `PublicationPlan` and
  `CompileArtifacts`, and the engine `CompiledGraph` facade; and
- planning, prepare, runtime, publication execution, and backend execution.

`OperationKind`, `OperationAttrs`, `NoOperationAttrs`, `Operation`, `BinaryArithmeticKind`,
`BinaryComparisonKind`, `BooleanLogicalKind`, `WhereSelectionKind`, `UnaryElementwiseKind`,
`ScalarElementwiseKind`, `ScalarValueAttrs`, `ClampRangeAttrs`, `CastKind`, `CastAttrs`,
`AggregateReductionKind`, `AxisReductionAttrs`, `ArgMaxTiePolicy`, `ArgMaxAttrs`,
`MaskedReductionAttrs`, `CumulativeSumKind`, `CumulativeSumAttrs`, `SoftmaxKind`, `SoftmaxAttrs`,
`ContiguousKind`, `ShapeTransformKind`, `TargetShapeAttrs`, `AxisTransformKind`,
`PermutationAttrs`, `AxisTransformAttrs`, `GraphValue`, `CompiledNode`,
`GraphPhase`, `CompiledGraphModel`, and
`PublicationBinding` are current Java API contracts. Binary arithmetic, binary comparison,
boolean logical, unary
elementwise, scalar elementwise, conditional selection, and cast semantics are the current
production concrete kind families with matching public Tensor expression construction. Aggregate
reduction is also a current production semantic family; `sum`, `mean`, `prod`, reduction `min`,
reduction `max`, boolean `all`, boolean `any`, and axis-only `argMax` have matching public Tensor
expression methods, including the masked `sum(axis, mask)` and `mean(axis, mask)` forms. The graph
records can compose these, current cumulative-sum expressions, or test-local semantics, but they
do not provide a compiler entry point or executable support. Softmax and log-softmax semantic
values and their public Tensor expression construction are current. The contiguous semantic value
and public Tensor expression construction are also current, including static resolved and dynamic
unresolved result-layout rules. Reshape and expand semantic values and both public Tensor
expression families are current, with their distinct count-preserving and directional-
broadcasting validation and layout rules.
Axis permutation, singleton-axis insertion, and selected singleton-axis removal semantic values
are current. Public permutation, rank-two transpose, expand-dimensions, and squeeze expression
construction is also current.

## Failures and ownership summary

- Null inputs are rejected where documented by each current method's Javadoc.
- Negative sizes, offsets, strides, and identifiers are rejected.
- Checked size, stride, and span arithmetic throws `ArithmeticException` on overflow.
- `Tensor.contiguous()` accepts every data type, preserves exact Shape, type, and gradient
  eligibility, and performs checked canonical layout construction only for fully static Shapes.
  Overflow consumes no Tensor identity; identifier exhaustion occurs after local immutable
  expression metadata construction. Valid results remain unlabeled and storage-free.
- `Tensor.reshape(long...)` accepts non-negative extents plus at most one inferable `-1`, treats an
  empty request as scalar Shape, copies request semantics without retaining the caller array, and
  rejects unavailable, ambiguous, non-divisible, overflowing, or known-mismatched requests before
  identity allocation.
- `Tensor.reshape(Shape)` retains the exact non-null normalized target reference. Both overloads
  defer equality when either count is dynamic and resolve same-offset canonical view geometry only
  for resolved contiguous input plus a fully static target; every other layout remains unresolved.
  Valid results retain exact type and gradient eligibility, use exact RESHAPE/target-shape
  semantics and `[input]` provenance, and remain fresh, unlabeled, and storage-free.
- `Tensor.expand(long...)` accepts only literal non-negative extents, treats an empty request as
  scalar Shape, and neither retains nor mutates the caller array. `Tensor.expand(Shape)` retains
  the exact non-null target reference. Both require target rank at least input rank and accept a
  right-aligned pair only when dimensions are equal or the input dimension is a static singleton;
  leading target axes are valid, and unprovable dynamic pairs are rejected without binding.
- A static expand target and any resolved input layout produce a fresh same-offset view layout
  that preserves unchanged aligned strides and uses zero strides for leading and expanded-
  singleton axes. Dynamic targets or unresolved input layouts remain unresolved. Every result
  retains exact type and gradient eligibility, records exact EXPAND/target-shape semantics and
  `[input]` provenance, and remains fresh, unlabeled, and storage-free.
- `Tensor.permute(int...)` requires exactly one raw axis per input rank, copies the request after
  count validation, normalizes each negative entry once, and rejects an out-of-range or duplicate
  normalized axis before identity allocation. Empty axes are valid exactly for scalar input.
- Every resolved permute input layout produces one new same-offset view layout with exact reordered
  strides; unresolved input remains unresolved. `Tensor.transpose()` requires rank two and uses
  normalized `[1, 0]`. Both methods retain exact type and eligibility, record PERMUTE attributes
  and `[input]` provenance, and remain fresh, unlabeled, storage-free logical metadata without an
  alias, copy, materialization, or execution guarantee.
- `Tensor.expandDims(int)` normalizes an insertion position against rank plus one, inserts exactly
  one new static singleton while preserving exact unaffected Dimension references, and records
  the normalized output position. For resolved input geometry it inserts one checked following-
  axis stride, or one at the end, in a new same-offset view descriptor; unresolved input remains
  unresolved.
- `Tensor.squeeze(int)` uses existing Shape-axis normalization, requires the selected Dimension to
  be statically known as one, and removes exactly that Dimension and stride while preserving exact
  unaffected references and offset. Dynamic, zero, and non-one dimensions are rejected without a
  symbolic constraint. Both rank-edit operations preserve exact type/eligibility, record matching
  `AxisTransformAttrs` and `[input]` provenance, and remain fresh, unlabeled, storage-free logical
  metadata without an alias, copy, canonicalization, gradient, materialization, or execution
  guarantee.
- Current value objects are immutable and defensively copy caller-owned arrays where applicable.
- Operation-kind implementations must return a non-null, non-blank name, and operation-attribute implementations must preserve immutable value semantics; the marker interfaces do not enforce those obligations at runtime.
- `Operation` rejects a null kind or attributes value, retains both valid references unchanged, and does not validate family compatibility.
- `TargetShapeAttrs` rejects a null target with `NullPointerException("targetShape")`, retains the
  exact non-null immutable Shape reference, and performs no input-dependent compatibility check.
- `PermutationAttrs` validates a non-null list and indexed elements in encounter order, rejects a
  negative, out-of-range, or first duplicate axis with the documented indexed message, and only
  then stores one immutable snapshot. An empty axes list is the rank-zero identity. The value
  validates a complete permutation of its own list size but no eventual input rank.
- `AxisTransformAttrs` rejects a negative normalized position with `IllegalArgumentException` and
  exact message `axis must be non-negative: <axis>`. It retains every non-negative value without
  proving an insertion bound, removal bound, or singleton input dimension.
- `CastAttrs` rejects a null target data type with `NullPointerException("targetDataType")`,
  retains every valid `DataType` reference unchanged, and performs no source compatibility,
  conversion, or backend-capability validation.
- `AxisReductionAttrs` rejects a negative normalized axis with `IllegalArgumentException` and
  retains the exact non-negative axis and dimension choice. `ArgMaxAttrs` performs that axis check
  before rejecting a null `ArgMaxTiePolicy` with `NullPointerException("tiePolicy")`. Ordinary
  full reductions use `NoOperationAttrs.INSTANCE`; no negative all-axis sentinel exists. These
  semantic values do not by themselves validate rank, construct Tensor results, or define
  numerical or backend behavior. Public `sum`, `mean`, `prod`, reduction `min`, reduction `max`,
  boolean `all`, boolean `any`, and `argMax` perform the local rank and result construction
  described above. `argMax` additionally accepts only floating or integral input, requires an
  explicit policy at its helper boundary, and fixes the result to non-differentiable INT64 without
  defining comparison or empty-axis behavior.
- `MaskedReductionAttrs` rejects a negative normalized reduction axis before inspecting its
  mapping. It then rejects a null mapping, a null or negative mapped axis in index order, and a
  repeated or descending position before copying the validated list. The stored mapping is an
  immutable snapshot; empty and `Integer.MAX_VALUE` positions are structurally valid. The value
  records masked SUM/MEAN semantics only and performs no Shape resolution, Tensor construction,
  provenance construction, value selection, counting, or execution. Public masked `sum` and
  `mean` separately require floating input and an exact BOOL mask, normalize one axis, resolve a
  locally provable ordered mapping, remove the axis, and construct exact `[input, mask]`
  provenance. Null, type, axis, rank, and alignment failures occur before result identity
  allocation. Construction performs no storage alignment, numerical work, gradient rule,
  compiler behavior, backend behavior, or execution.
- `CumulativeSumAttrs` rejects a negative normalized axis with `IllegalArgumentException` and the
  exact message `axis must be non-negative: <axis>`. It retains every non-negative axis and both
  mode flags unchanged. The value itself neither validates an input rank nor constructs or
  executes a Tensor result. Public `Tensor.cumSum` separately accepts floating or integral input,
  rejects BOOL before axis validation, normalizes one positive or negative axis, retains the exact
  Shape/type/eligibility metadata in an unresolved descriptor, and records exact one-input
  provenance. Each valid call is fresh and performs no accumulation, gradient work, compiler
  capture, backend behavior, or execution.
- `SoftmaxAttrs` rejects a negative normalized axis with `IllegalArgumentException` and the exact
  message `axis must be non-negative: <axis>`. It retains every non-negative axis unchanged but
  does not itself validate rank, construct a Tensor or provenance, evaluate normalization, define
  a gradient or numerical algorithm, report backend support, or execute work. Public
  `Tensor.softmax` and `Tensor.logSoftmax` separately require floating input, normalize one caller
  axis, retain exact Shape/type/eligibility metadata in an unresolved descriptor, and create fresh
  exact-kind one-input provenance without numerical, gradient, compiler, backend, or execution
  behavior.
- `Tensor.cast` requires a non-null target and accepts all 36 current source/target pairs. Each
  successful call returns a fresh unlabeled storage-free expression, including for a same-type
  request, with the exact input Shape reference, unresolved layout, typed target attributes, and
  exact one-input provenance. Gradient eligibility survives only an already-eligible
  floating-to-floating cast. Construction neither reads values/storage nor defines conversion,
  canonicalization, gradient-rule, compiler, or backend behavior.
- `TensorDescriptor` rejects null components, resolved layouts incompatible with their paired
  shape, and gradient requests for non-differentiable data types. Layout reconstruction can report
  checked arithmetic overflow.
- `Tensor` retains exact stable ID and descriptor references, a normalized label, and immutable
  optional provenance, and uses object identity for equality and hashing. A present provenance
  retains the exact `TensorProvenance` reference for the tensor's lifetime without synchronization.
  It borrows optional host storage, validates data type, resolved referenced span, and
  attachment-time liveness in that order, and changes only the storage reference through
  synchronized snapshot, replacement, and clearing methods. Failed replacement preserves the
  previous reference, later storage death remains observable, and neither transition changes
  provenance or metadata-only diagnostics.
- `TensorProvenance` rejects a null operation, input list, or indexed input element; snapshots the
  input list with `List.copyOf`; preserves order, empty inputs, repeated references, and exact
  Tensor identities; and retains the exact Operation reference. Its record value methods do not
  establish producer-occurrence identity, graph membership, or semantic compatibility.
- `Tensor.add`, `sub`, `mul`, `div`, `min`, `max`, and tensor-valued `pow` require a non-null right
  operand, promote only floating data types, and require locally provable right-aligned
  broadcasting. Each successful call returns a fresh unlabeled storage-free Tensor with unresolved
  layout, gradient eligibility equal to input OR, and exact matching parameterless operation plus
  ordered `[left, right]` provenance. Null, type, and shape failures precede ID allocation;
  identity exhaustion follows local descriptor and provenance construction. These methods do not
  calculate values, simplify expressions, capture graphs, or define gradient or backend behavior.
- `Tensor.greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`, and `notEqualTo`
  require a non-null right operand, accept only floating input pairs, and require locally provable
  right-aligned broadcasting. Each successful call returns a fresh unlabeled storage-free `BOOL`
  Tensor with unresolved layout, false gradient eligibility, the exact parameterless comparison
  operation, and ordered `[left, right]` provenance. Null, type, and shape failures precede ID
  allocation; identity exhaustion follows local descriptor and provenance construction. These
  methods do not inspect values, define numerical comparison or tolerance behavior, create
  gradient rules, capture graphs, or provide backend execution.
- `Tensor.logicalAnd` and `logicalOr` require a non-null right operand and exact `BOOL` data type
  on both ordered inputs, then derive one locally provable right-aligned broadcast shape.
  `Tensor.logicalNot` requires an exact `BOOL` receiver and retains its exact input `Shape`
  reference without broadcasting. Every successful call returns a fresh unlabeled storage-free
  `BOOL` Tensor with unresolved layout, false gradient eligibility, the exact parameterless
  logical operation, and ordered binary or one-input provenance. Null, kind, type, and shape
  failures precede ID allocation; identity exhaustion follows fixed descriptor and provenance
  construction. These methods do not inspect truth bytes, short-circuit, simplify or reorder
  expressions, insert casts, create gradient rules, capture graphs, or provide backend execution.
- Static `Tensor.where` requires non-null condition, true-branch, and false-branch inputs in that
  order, exact BOOL condition type, floating branches, and two locally provable broadcasts in
  branch-first then condition-second order. Each successful call returns a fresh unlabeled,
  storage-free Tensor with promoted branch type, final three-way shape, unresolved layout,
  branch-only gradient eligibility OR, exact `WHERE` operation, and ordered
  `[condition, ifTrue, ifFalse]` provenance. Validation failures precede ID allocation; identity
  exhaustion follows local descriptor and provenance construction. The method does not inspect
  values, choose or evaluate a branch, define gradient routing, capture a graph, or provide ONNX
  or backend execution.
- `Tensor.abs`, `neg`, `inv`, `log`, `exp`, `erf`, `sqrt`, `floor`, `ceil`, `sign`, `relu`,
  `sigmoid`, `tanh`, `fastExp`, and `fastTanh` accept only floating receiver data types and retain
  the exact data type, shape reference, and gradient-eligibility flag. Each successful call returns
  a fresh unlabeled storage-free Tensor with unresolved layout and exact matching parameterless
  operation plus one-input provenance. Type failures precede ID allocation; identity exhaustion
  follows local descriptor and provenance construction. The methods do not inspect mathematical
  domains, simplify chains, define strict or fast numerical accuracy, capture graphs, or provide
  gradient or backend behavior.
- `Tensor.mul(double)`, `pow(double)`, `clamp(double, double)`, `clampMin(double)`, and
  `clampMax(double)` accept only floating receiver data types, retain exact caller binary64
  attributes, and preserve the input type, shape reference, and gradient eligibility. Each
  successful call returns a fresh unlabeled storage-free Tensor with unresolved layout and exact
  matching one-input provenance. `clamp` validates the input type before rejecting only a strictly
  inverted range and remains one `CLAMP` operation. Type and range failures precede ID allocation;
  the methods do not convert parameters, simplify expressions, define numerical behavior, capture
  graphs, or provide gradient or backend behavior.
- `TensorFactory` rejects null argument containers before ID allocation, delegates label and
  storage semantics to `Tensor` after allocation, and assigns IDs unique among its allocations in
  one JVM. Delegated semantic failures consume IDs; exhaustion after `Long.MAX_VALUE` is permanent.
  Its allocation overloads require resolved layout, reject referenced span above
  `Integer.MAX_VALUE`, create exact-span type-matched primitive-array heap storage, and delegate
  only after allocation and wrapping. Preallocation and JVM allocation failures consume no ID;
  blank-label failure and identifier exhaustion occur after heap allocation. Automatic scope keeps
  factory-created arrays reachable without an arena, close operation, or external owner.
- Its six flat-array overloads require exact carrier/data-type matching, resolved
  `DENSE_CONTIGUOUS` layout, and source length equal to logical element count. Numeric carriers and
  raw BFLOAT16 bits are copied unchanged; BOOL bytes are normalized to zero or one. The source is
  not retained. Import validation failures consume no ID, while blank-label failure and exhaustion
  occur after destination allocation and before population.
- Its nested-array method requires runtime rank at least two, one of the same six ultimate
  primitive carriers, a completely rectangular non-null structure, and no empty non-final axis.
  It infers an exact fully static dense descriptor, flattens into a fresh matching carrier, and
  delegates once to flat import. Numeric and raw BFLOAT16 values remain unchanged; BOOL
  normalization remains in flat import. Structural and descriptor failures consume no ID; blank
  label and exhaustion retain the flat-import destination-allocation and ID side effects. Source
  ownership remains with the caller, and concurrent mutation has no deep-snapshot guarantee.
- Its scalar methods infer exact data type from primitive signatures and create rank-0 dense
  values; BFLOAT16 alone performs an explicit binary32-to-BFLOAT16 conversion. Zeros and ones
  require fully static shapes, synthesize dense descriptors, and support scalar and empty shapes.
  Like methods copy only shape and data type from the template. Static/count/layout/gradient
  validation happens before source or destination allocation. Blank-label failure and exhaustion
  happen after destination allocation; scalar and one creation have also allocated their source
  carrier, while zero creation has no source carrier. Every successful result has independent
  descriptor, layout, storage, backing array, Tensor object, and factory ID.
- Its six type-safe full-value methods require a fully static shape and infer the exact data type
  from a primitive value; only `fullBFloat16` converts a binary32 semantic input. Each method fills
  one exact carrier and delegates once to flat import, preserving raw floating signed-zero and NaN
  values and canonical BOOL bytes. `identityMatrix` supports all six data types and non-negative
  rectangular dimensions, writes typed one only on the main diagonal of one default-zero carrier,
  and delegates once; `eye` is a pure canonical delegation. Early shape, dimension, count, layout,
  and gradient failures consume no ID. Blank labels, exhaustion, and unexpected copy failures have
  the same late source/destination/ID effects described by the factory contract.
- Its `int` and `long` range overloads create eager non-empty, inclusive-start, exclusive-end
  `INT32` and `INT64` tensors with gradients disabled. Positive or negative non-zero steps must
  advance toward the end. Exact count arithmetic and a no-post-final-addition fill loop preserve
  primitive-boundary behavior, while counts above `Integer.MAX_VALUE` fail before allocation.
  One exact carrier is delegated to flat import and is not retained.
- Its strict and cyclic prefix overloads infer exact data type from the same six primitive
  carriers, require a fully static shape, synthesize canonical dense descriptors, and preserve
  explicit label and gradient intent. Strict mode copies the requested prefix and ignores source
  tail; cyclic mode repeats by modulo and requires a non-empty source only for non-empty output.
  Numeric and raw BFLOAT16 values remain unchanged, while BOOL is normalized by flat import. All
  sources are copied and never retained. Prevalidation consumes no ID; blank-label failure and
  exhaustion occur after carrier and destination allocation and before the flat copy.
- Its one normal-random method requires a caller-owned `RandomGenerator`, a fully static
  Java-array-sized shape, a floating data type, finite mean, and finite numerically non-negative
  standard deviation. It consumes one `nextGaussian()` call per row-major element, transforms in
  binary64 with ordinary multiplication then addition, converts to the exact FLOAT64, FLOAT32, or
  BFLOAT16 carrier, and delegates once to flat import. The source is never retained, substituted,
  synchronized, seeded, reset, split, or closed. Prevalidation and source-carrier allocation
  failures consume no calls or ID; a source exception preserves prior source advancement; blank
  label and exhaustion occur after all calls and destination allocation under the delegated
  flat-import side effects.
- Its one continuous-uniform method requires a caller-owned `RandomGenerator`, a fully static
  Java-array-sized shape, a floating data type, finite binary64 bounds, and a lower bound strictly
  less than the upper bound. It consumes one bounded `nextDouble(lower, upper)` call per row-major
  element, stores the returned binary64 value directly or narrows it to FLOAT32/BFLOAT16, and
  delegates once to flat import. A conforming source's binary64 result is half-open; narrowing may
  equal the corresponding narrowed upper bound. The source is never retained, substituted,
  synchronized, seeded, reset, split, or closed. Prevalidation and source-carrier allocation
  failures consume no calls or ID; a source exception preserves prior advancement; blank label and
  exhaustion occur after all calls and destination allocation under the delegated flat-import
  side effects.
- Its two bounded-integral `randomInt` overloads infer `INT32` from primitive `int` bounds and
  `INT64` from primitive `long` bounds. They require a fully static Java-array-sized shape and a
  strict half-open interval, always disable gradients, and consume one matching bounded
  `nextInt(origin, bound)` or `nextLong(origin, bound)` call per row-major element. Each direct
  result is stored without modulo, unbounded sampling, floating arithmetic, or conversion in one
  exact carrier and delegated once to flat import. The same-carrier bound cannot express an
  exclusive value above the carrier maximum, so no full-domain form is provided. Prevalidation
  and source-carrier allocation failures consume no calls or ID; a source exception preserves
  prior calls; blank-label failure and exhaustion occur after all calls and destination
  allocation without rollback.
- Its one Bernoulli-random method requires a caller-owned `RandomGenerator`, a fully static
  Java-array-sized shape, and a finite probability in `[0, 1]`. It always creates a BOOL result
  with gradients disabled. Every row-major element consumes one unbounded `nextDouble()` call,
  including at probability zero and one, and stores canonical byte one exactly when the draw is
  strictly less than the probability. A custom source result is not post-validated. One complete
  `byte[]` is delegated once to BOOL flat import. Prevalidation and source-carrier allocation
  failures consume no calls or ID; a source exception preserves prior calls; blank-label failure
  and exhaustion occur after all calls and destination allocation without rollback.
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
- None of the current types owns device storage, runtime residency, or backend selection. The
  current `Tensor` also owns no storage lifetime, graph-local identity, compiler graph capture, or
  execution state; its provenance is model origin metadata only.

See generated Javadoc for the exact member-level exception and nullability contract.
